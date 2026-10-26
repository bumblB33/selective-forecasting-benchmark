#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.evaluation import compute_metrics, compute_statistical_tests, compute_win_loss
from core.paths import PROJECT_ROOT
from core.preprocessing import add_calendar_features, add_fourier_features, create_train_test_split
from daily import config
from daily.data_loader import (
    build_actuals,
    build_baseline_variants,
    determine_anchor_date,
    drop_outlier_days,
    load_daily_actual_vs_forecast,
)
from daily.models import run_mlforecast, run_statsforecast
from daily.preprocessing import fill_missing_dates
from reporting.constants import MODEL_COLORS

ABLATION_DIR = PROJECT_ROOT / "data" / "metrics" / "ablation"

MSTL_YEARLY_MIN_WEEKS = 105

warnings.filterwarnings("ignore")

_log_dir = PROJECT_ROOT / "logs"
_log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    handlers=[
        logging.FileHandler(_log_dir / "ablation.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI arguments for the training-length ablation study."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override path to daily_actual_vs_forecast.csv "
        "(default: data/input/daily_actual_vs_forecast.csv)",
    )
    parser.add_argument(
        "--test-windows",
        type=int,
        default=3,
        help="Number of rolling one-week test windows to average across (default: 3)",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[26, 39, 52, 65, 78, 91, config.TRAIN_HISTORY_WEEKS],
        help=(
            "Training length checkpoints in weeks. "
            f"Default: quarterly from 26 to TRAIN_HISTORY_WEEKS ({config.TRAIN_HISTORY_WEEKS})"
        ),
    )
    return parser.parse_args()


def assess_location_history(
    actual_vs_forecast_df: pd.DataFrame,
    checkpoints: list[int],
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[str], dict[int, int]]:
    """Bucket locations by available history and return the longest-checkpoint pool.

    The fixed pool (locations meeting the longest checkpoint) keeps location
    composition constant across comparisons. History is measured as the span
    between first and last observed date.
    """
    pre_test = actual_vs_forecast_df[
        (actual_vs_forecast_df[config.COL_DAY_START_TIME] < test_start)
        & actual_vs_forecast_df[config.COL_ACTUAL_VALUE].notna()
    ]

    location_ranges = (
        pre_test.groupby(config.COL_ENTITY_UID)[config.COL_DAY_START_TIME]
        .agg(first_date="min", last_date="max")
        .reset_index()
        .rename(columns={config.COL_ENTITY_UID: "unique_id"})
    )
    location_ranges["available_weeks"] = (
        (location_ranges["last_date"] - location_ranges["first_date"]).dt.days / 7
    ).round(1)

    max_checkpoint = max(checkpoints)
    pool_sizes: dict[int, int] = {}
    for weeks in sorted(checkpoints):
        col = f"qualifies_{weeks}w"
        location_ranges[col] = location_ranges["available_weeks"] >= weeks
        pool_sizes[weeks] = int(location_ranges[col].sum())

    fixed_pool_ids = location_ranges.loc[
        location_ranges["available_weeks"] >= max_checkpoint, "unique_id"
    ].tolist()

    total = len(location_ranges)
    logger.info("Location history pre-check (reference date: %s):", test_start.date())
    logger.info("  Total locations in dataset: %d", total)
    for weeks in sorted(checkpoints, reverse=True):
        n = pool_sizes[weeks]
        pct = round(100 * n / total) if total else 0
        logger.info("  >= %3d weeks: %4d locations  (%d%%)", weeks, n, pct)
    logger.info(
        "  Fixed pool (>= %d weeks): %d locations; used as primary analysis pool",
        max_checkpoint,
        len(fixed_pool_ids),
    )

    if len(fixed_pool_ids) < 30:
        logger.warning(
            "Fixed pool has only %d locations. Wilcoxon tests require >= 10; "
            "estimates at the longest checkpoint may be unreliable. "
            "Consider reducing the maximum checkpoint.",
            len(fixed_pool_ids),
        )

    return location_ranges, fixed_pool_ids, pool_sizes


def determine_test_windows(
    actual_vs_forecast_df: pd.DataFrame,
    n_windows: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Rolling 1-week test windows ending at the daily-pipeline anchor date."""
    base_anchor = determine_anchor_date(actual_vs_forecast_df)

    windows = []
    for i in range(n_windows):
        anchor = base_anchor - pd.Timedelta(weeks=i)
        test_end = anchor
        test_start = (
            anchor - pd.Timedelta(weeks=config.TEST_PERIOD_WEEKS) + pd.Timedelta(days=1)
        )
        windows.append((test_start, test_end))
        logger.info(
            "Test window %d: %s -> %s", i + 1, test_start.date(), test_end.date()
        )

    return windows


def _evaluate_pool(
    actuals: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    pool_ids: list[str] | None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Filter predictions/actuals to ``pool_ids`` and compute the metric triple.

    Filtering at evaluation time (not training time) lets every model train
    on the broadest available data and only narrows the reporting view.
    """
    if pool_ids is not None:
        id_set = set(pool_ids)
        actuals = actuals[actuals[config.NIXTLA_UNIQUE_ID].isin(id_set)]
        predictions = {
            name: df[df[config.NIXTLA_UNIQUE_ID].isin(id_set)]
            for name, df in predictions.items()
        }

    all_location_metrics = {
        name: compute_metrics(
            actuals,
            pred_df,
            exclude_zero_actuals=config.EXCLUDE_ZERO_ACTUAL_FROM_EVAL,
        )
        for name, pred_df in predictions.items()
    }
    win_loss_df = compute_win_loss(
        all_location_metrics,
        baseline_model=config.BASELINE_MODEL,
    )
    stat_tests_df = compute_statistical_tests(
        all_location_metrics,
        baseline_model=config.BASELINE_MODEL,
    )
    return all_location_metrics, win_loss_df, stat_tests_df


def _pool_result_rows(
    training_weeks: int,
    window_idx: int,
    pool_type: str,
    all_location_metrics: dict[str, pd.DataFrame],
    win_loss_df: pd.DataFrame,
    stat_tests_df: pd.DataFrame,
) -> list[dict]:
    """Per-model result rows for one (training-length, window, pool) cell."""
    win_loss_idx = (
        win_loss_df.set_index("model")
        if not win_loss_df.empty and "model" in win_loss_df.columns
        else pd.DataFrame()
    )
    stat_idx = (
        stat_tests_df.set_index("model")
        if not stat_tests_df.empty and "model" in stat_tests_df.columns
        else pd.DataFrame()
    )

    rows = []
    for model_name, location_metrics in all_location_metrics.items():
        mae_mean = (
            location_metrics["MAE"].mean()
            if "MAE" in location_metrics.columns
            else None
        )
        mae_median = (
            location_metrics["MAE"].median()
            if "MAE" in location_metrics.columns
            else None
        )
        wape_mean = (
            location_metrics["WAPE"].mean()
            if "WAPE" in location_metrics.columns
            else None
        )

        win_rate = p_value = significant = None
        if model_name != config.BASELINE_MODEL:
            if not win_loss_idx.empty and model_name in win_loss_idx.index:
                win_rate = float(win_loss_idx.loc[model_name, "win_rate"])
            if not stat_idx.empty and model_name in stat_idx.index:
                p_value = float(stat_idx.loc[model_name, "p_value"])
                significant = bool(stat_idx.loc[model_name, "significant"])

        rows.append(
            {
                "training_weeks": training_weeks,
                "test_window": window_idx + 1,
                "pool_type": pool_type,
                "model": model_name,
                "n_locations": len(location_metrics),
                "MAE_mean": mae_mean,
                "MAE_median": mae_median,
                "WAPE_mean": wape_mean,
                "win_rate": win_rate,
                "p_value": p_value,
                "significant": significant,
                "mstl_yearly_capable": training_weeks >= MSTL_YEARLY_MIN_WEEKS,
            }
        )
    return rows


def run_ablation(
    actual_vs_forecast_df: pd.DataFrame,
    nixtla_full: pd.DataFrame,
    checkpoints: list[int],
    test_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    fixed_pool_ids: list[str],
) -> pd.DataFrame:
    """Sweep every (training-length, test-window, pool) combination.

    Models train on the natural pool (production reality) and metrics are
    reported on both the natural and the fixed pool. Outlier handling and
    Fourier coefficients are rebuilt per slice to prevent leakage from
    longer histories.
    """
    all_rows: list[dict] = []
    checkpoints_sorted = sorted(checkpoints)

    for window_idx, (test_start, test_end) in enumerate(test_windows):
        logger.info(
            "Test window %d/%d: %s -> %s",
            window_idx + 1,
            len(test_windows),
            test_start.date(),
            test_end.date(),
        )

        window_actual_vs_forecast = actual_vs_forecast_df[
            (actual_vs_forecast_df[config.COL_DAY_START_TIME] >= test_start)
            & (actual_vs_forecast_df[config.COL_DAY_START_TIME] <= test_end)
        ]
        actuals = build_actuals(window_actual_vs_forecast)
        baseline_variants = build_baseline_variants(window_actual_vs_forecast)

        for training_weeks in checkpoints_sorted:
            if training_weeks < MSTL_YEARLY_MIN_WEEKS:
                logger.warning(
                    "  training_weeks=%d is below the MSTL yearly-seasonality floor "
                    "(%d weeks / ~730 days). MSTL_D will fall back to SeasonalNaive "
                    "for the yearly component at affected locations.",
                    training_weeks,
                    MSTL_YEARLY_MIN_WEEKS,
                )

            training_df, _ = create_train_test_split(
                nixtla_full,
                test_start,
                test_end,
                max_train_history_weeks=training_weeks,
            )
            training_df = drop_outlier_days(training_df, actual_vs_forecast_df)
            training_df, future_fourier_df, fourier_columns = add_fourier_features(
                training_df
            )

            logger.info(
                "  Training set: %d rows, %d locations",
                len(training_df),
                training_df[config.NIXTLA_UNIQUE_ID].nunique(),
            )

            predictions: dict[str, pd.DataFrame] = dict(baseline_variants)
            predictions.update(run_statsforecast(training_df))
            predictions.update(
                run_mlforecast(
                    training_df,
                    future_exog_df=future_fourier_df,
                    fourier_columns=fourier_columns,
                )
            )

            for pool_type, pool_ids in [("natural", None), ("fixed", fixed_pool_ids)]:
                loc_metrics, win_loss_df, stat_tests_df = _evaluate_pool(
                    actuals,
                    predictions,
                    pool_ids,
                )
                all_rows.extend(
                    _pool_result_rows(
                        training_weeks,
                        window_idx,
                        pool_type,
                        loc_metrics,
                        win_loss_df,
                        stat_tests_df,
                    )
                )

    return pd.DataFrame(all_rows)


def summarize_ablation(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Average ablation rows across test windows; std-dev surfaces window sensitivity."""
    group_cols = ["training_weeks", "pool_type", "model", "mstl_yearly_capable"]
    summary = (
        detail_df.groupby(group_cols, dropna=False)
        .agg(
            n_locations=("n_locations", "mean"),
            MAE_mean=("MAE_mean", "mean"),
            MAE_mean_std=("MAE_mean", "std"),
            MAE_median=("MAE_median", "mean"),
            WAPE_mean=("WAPE_mean", "mean"),
            win_rate=("win_rate", "mean"),
            win_rate_std=("win_rate", "std"),
            p_value=("p_value", "mean"),
            significant_windows=("significant", "sum"),
        )
        .reset_index()
    )

    return summary.sort_values(["pool_type", "model", "training_weeks"]).reset_index(
        drop=True
    )


def _build_chart(
    summary_df: pd.DataFrame,
    pool_sizes: dict[int, int],
    output_path: Path,
) -> None:
    """Render the two-panel MAE-and-win-rate-vs-training-history chart (fixed pool)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = summary_df[summary_df["pool_type"] == "fixed"].copy()
    checkpoints = sorted(fixed["training_weeks"].unique())

    fig, (ax_mae, ax_win) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.35},
    )
    fig.suptitle(
        "Forecast accuracy vs training history length\n"
        "(fixed location pool; location set held constant)",
        fontsize=12,
    )

    non_baseline = [m for m in fixed["model"].unique() if m != config.BASELINE_MODEL]

    baseline_rows = fixed[fixed["model"] == config.BASELINE_MODEL]
    if not baseline_rows.empty:
        baseline_mae = baseline_rows.groupby("training_weeks")["MAE_mean"].mean().mean()
        ax_mae.axhline(
            baseline_mae,
            color=MODEL_COLORS.get(config.BASELINE_MODEL, "#bf360c"),
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline Production (baseline, MAE=${baseline_mae:,.0f})",
            zorder=1,
        )

    for model_name in non_baseline:
        rows = fixed[fixed["model"] == model_name].sort_values("training_weeks")
        if rows["MAE_mean"].isna().all():
            continue
        color = MODEL_COLORS.get(model_name, "#95a5a6")
        ax_mae.plot(
            rows["training_weeks"],
            rows["MAE_mean"],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=model_name,
            zorder=2,
        )

    ax_mae.axvline(
        MSTL_YEARLY_MIN_WEEKS,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label=f"MSTL yearly-seasonality floor ({MSTL_YEARLY_MIN_WEEKS} wks)",
        zorder=0,
    )
    ax_mae.set_ylabel("Mean MAE across locations ($)")
    ax_mae.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax_mae.grid(True, alpha=0.3)

    for model_name in non_baseline:
        rows = fixed[fixed["model"] == model_name].sort_values("training_weeks")
        if rows["win_rate"].isna().all():
            continue
        color = MODEL_COLORS.get(model_name, "#95a5a6")
        ax_win.plot(
            rows["training_weeks"],
            rows["win_rate"],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=model_name,
            zorder=2,
        )

    ax_win.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        label="50% break-even",
    )
    ax_win.axvline(
        MSTL_YEARLY_MIN_WEEKS, color="grey", linestyle=":", linewidth=1.2, zorder=0
    )
    ax_win.set_ylim(0, 1)
    ax_win.set_ylabel("Win rate vs Baseline Production")
    ax_win.set_xlabel("Training history (weeks)")
    ax_win.set_xticks(checkpoints)
    ax_win.legend(fontsize=6.5, loc="lower right", ncol=2)
    ax_win.grid(True, alpha=0.3)

    ax_win2 = ax_win.twiny()
    ax_win2.set_xlim(ax_win.get_xlim())
    ax_win2.set_xticks(checkpoints)
    ax_win2.set_xticklabels(
        [f"n={pool_sizes.get(w, '?')}" for w in checkpoints],
        fontsize=6.5,
    )
    ax_win2.set_xlabel("Fixed pool size at each checkpoint", fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", output_path)


def save_results(
    history_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    pool_sizes: dict[int, int],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)

    history_path = output_dir / f"ablation_history_check_{timestamp}.csv"
    detail_path = output_dir / f"ablation_detail_{timestamp}.csv"
    summary_path = output_dir / f"ablation_summary_{timestamp}.csv"
    chart_path = output_dir / f"ablation_chart_{timestamp}.png"

    history_df.to_csv(history_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    _build_chart(summary_df, pool_sizes, chart_path)

    logger.info("Ablation outputs written to %s", output_dir)
    logger.info("  %s", history_path.name)
    logger.info("  %s", detail_path.name)
    logger.info("  %s", summary_path.name)
    logger.info("  %s", chart_path.name)


def main() -> None:
    """Ablation entry point. Date-filling and calendar features run once on the
    full dataset; outlier drops and Fourier features re-run per slice to
    prevent leakage from longer histories.
    """
    args = parse_args()
    checkpoints = sorted(set(args.checkpoints))

    logger.info(
        "Training length ablation study; checkpoints: %s weeks, test windows: %d",
        checkpoints,
        args.test_windows,
    )

    actual_vs_forecast_df = load_daily_actual_vs_forecast(args.data_path)

    test_windows = determine_test_windows(
        actual_vs_forecast_df, n_windows=args.test_windows
    )

    reference_test_start = test_windows[0][0]
    history_df, fixed_pool_ids, pool_sizes = assess_location_history(
        actual_vs_forecast_df,
        checkpoints,
        reference_test_start,
    )

    if not fixed_pool_ids:
        logger.error(
            "Fixed pool is empty; no locations have >= %d weeks of history. "
            "Reduce the maximum checkpoint or pull more data from Snowflake.",
            max(checkpoints),
        )
        return

    nixtla_full = actual_vs_forecast_df[
        actual_vs_forecast_df[config.COL_ACTUAL_VALUE].notna()
    ][
        [config.COL_ENTITY_UID, config.COL_DAY_START_TIME, config.COL_ACTUAL_VALUE]
    ].rename(
        columns={
            config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
            config.COL_DAY_START_TIME: config.NIXTLA_DS,
            config.COL_ACTUAL_VALUE: config.NIXTLA_Y,
        }
    )
    nixtla_full = fill_missing_dates(nixtla_full)
    nixtla_full = add_calendar_features(nixtla_full)
    logger.info(
        "Full Nixtla dataset prepared: %d rows, %d locations",
        len(nixtla_full),
        nixtla_full[config.NIXTLA_UNIQUE_ID].nunique(),
    )

    detail_df = run_ablation(
        actual_vs_forecast_df,
        nixtla_full,
        checkpoints,
        test_windows,
        fixed_pool_ids,
    )
    summary_df = summarize_ablation(detail_df)
    save_results(history_df, detail_df, summary_df, pool_sizes, ABLATION_DIR)


if __name__ == "__main__":
    main()
