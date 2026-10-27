#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.evaluation import (
    aggregate_metrics,
    build_volume_tiers,
    combine_aggregate_metrics,
    compute_cv_metrics,
    compute_metrics,
    compute_statistical_tests,
    compute_win_loss,
    save_metrics,
)
from core.paths import PROJECT_ROOT
from core.preprocessing import add_calendar_features, add_fourier_features, create_train_test_split
from reporting.report import ReportContext, build_report

from . import config
from .data_loader import (
    build_actuals,
    build_actuals_strict,
    build_baseline_variants,
    determine_anchor_date,
    drop_outlier_days,
    load_daily_actual_vs_forecast,
    summarize_actual_vs_forecast_data,
)
from .models import (
    run_mlforecast,
    run_mlforecast_cv,
    run_statsforecast,
    run_statsforecast_cv,
)
from .preprocessing import fill_missing_dates

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI flags for the daily benchmark entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-sf", action="store_true", help="Skip StatsForecast models")
    parser.add_argument("--no-ml", action="store_true", help="Skip MLForecast models")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument(
        "--rolling-windows",
        type=int,
        default=1,
        metavar="N",
        help="Number of consecutive 1-week evaluation windows walking backward " "from the anchor date",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override path to daily_actual_vs_forecast.csv",
    )
    parser.add_argument(
        "--metric",
        choices=list(config.METRIC_CONFIGS.keys()),
        default=config.DEFAULT_METRIC,
        help="Forecast metric to benchmark (default: net_sales). "
        "gross_sales runs Nixtla models only; no production-baseline comparison is performed.",
    )
    args = parser.parse_args()
    if args.rolling_windows < 1:
        parser.error("--rolling-windows must be >= 1")
    return args


def _run_single_window(
    actual_vs_forecast_df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    run_sf: bool,
    run_ml: bool,
    run_cv: bool,
    metric_config: config.MetricConfig,
    window_label: str = "",
) -> dict:
    """Run one rolling evaluation window end-to-end and return reporting artifacts.

    ``metric_config`` selects the actuals column and toggles baseline-comparison
    steps. Outlier-flagged training days are dropped, mirroring production.
    """
    logger.info("Window %s: %s to %s", window_label, test_start.date(), test_end.date())

    actuals_column = metric_config.actuals_column
    nixtla_daily = actual_vs_forecast_df[actual_vs_forecast_df[actuals_column].notna()][
        [config.COL_ENTITY_UID, config.COL_DAY_START_TIME, actuals_column]
    ].rename(
        columns={
            config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
            config.COL_DAY_START_TIME: config.NIXTLA_DS,
            actuals_column: config.NIXTLA_Y,
        }
    )

    nixtla_daily = fill_missing_dates(nixtla_daily)
    nixtla_daily = add_calendar_features(nixtla_daily)

    training_df, _ = create_train_test_split(
        nixtla_daily,
        test_start,
        test_end,
        max_train_history_weeks=config.TRAIN_HISTORY_WEEKS,
    )
    training_df = drop_outlier_days(training_df, actual_vs_forecast_df)
    training_df, future_fourier_df, fourier_columns = add_fourier_features(
        training_df,
        weekly_k=config.FOURIER_WEEKLY_K,
        yearly_k=config.FOURIER_YEARLY_K,
        forecast_horizon=config.FORECAST_HORIZON,
        frequency=config.FREQUENCY,
    )

    logger.info(
        "Training data: %d rows, %d locations",
        len(training_df),
        training_df[config.NIXTLA_UNIQUE_ID].nunique(),
    )

    window_actual_vs_forecast = actual_vs_forecast_df[
        (actual_vs_forecast_df[config.COL_DAY_START_TIME] >= test_start)
        & (actual_vs_forecast_df[config.COL_DAY_START_TIME] <= test_end)
    ]

    predictions: dict[str, pd.DataFrame] = (
        build_baseline_variants(window_actual_vs_forecast) if metric_config.has_baseline else {}
    )
    if not metric_config.has_baseline:
        logger.info(
            "%s benchmark: no production baseline available; " "win/loss and statistical tests will be skipped.",
            metric_config.display_label,
        )

    actuals = build_actuals(window_actual_vs_forecast, target_column=actuals_column)
    actuals_strict = build_actuals_strict(window_actual_vs_forecast, target_column=actuals_column)

    if run_sf:
        sf_predictions = run_statsforecast(training_df)
        predictions.update(sf_predictions)

    if run_ml:
        ml_predictions = run_mlforecast(
            training_df,
            future_exog_df=future_fourier_df,
            fourier_columns=fourier_columns,
        )
        predictions.update(ml_predictions)

    cv_metrics_df = pd.DataFrame()
    if run_cv:
        sf_cv = run_statsforecast_cv(training_df) if run_sf else pd.DataFrame()
        ml_cv = run_mlforecast_cv(training_df, fourier_columns=fourier_columns) if run_ml else pd.DataFrame()
        if not sf_cv.empty or not ml_cv.empty:
            combined_cv = pd.concat([df for df in [sf_cv, ml_cv] if not df.empty], ignore_index=True)
            cv_metrics_df = compute_cv_metrics(
                combined_cv,
                exclude_zero_actuals=config.EXCLUDE_ZERO_ACTUAL_FROM_EVAL,
            )
            logger.info("CV metrics:\n%s", cv_metrics_df.to_string(index=False))

    strict_model_names = {
        config.BASELINE_PRODUCTION_STRICT,
        config.BASELINE_ADJUSTED_DRAFT_STRICT,
    }
    all_location_metrics: dict[str, pd.DataFrame] = {}
    aggregate_metrics_dict: dict[str, pd.DataFrame] = {}
    for model_name, pred_df in predictions.items():
        eval_actuals = actuals_strict if model_name in strict_model_names else actuals
        location_metrics = compute_metrics(
            eval_actuals,
            pred_df,
            exclude_zero_actuals=config.EXCLUDE_ZERO_ACTUAL_FROM_EVAL,
        )
        all_location_metrics[model_name] = location_metrics
        aggregate_metrics_dict[model_name] = aggregate_metrics(location_metrics)

    if metric_config.has_baseline:
        win_loss_df = compute_win_loss(
            all_location_metrics,
            baseline_model=config.BASELINE_MODEL,
        )
        statistical_tests_df = compute_statistical_tests(
            all_location_metrics,
            baseline_model=config.BASELINE_MODEL,
        )
        logger.info("Win/loss summary:\n%s", win_loss_df.to_string(index=False))
    else:
        win_loss_df = pd.DataFrame()
        statistical_tests_df = pd.DataFrame()

    volume_tiers = build_volume_tiers(actuals)

    return {
        "predictions": predictions,
        "actuals": actuals,
        "actuals_strict": actuals_strict,
        "all_location_metrics": all_location_metrics,
        "aggregate_metrics_dict": aggregate_metrics_dict,
        "win_loss_df": win_loss_df,
        "statistical_tests_df": statistical_tests_df,
        "cv_metrics_df": cv_metrics_df,
        "volume_tiers": volume_tiers,
        "test_start": test_start,
        "test_end": test_end,
    }


def main() -> None:
    """Daily benchmark entry point: run, persist metrics, and emit reports."""
    Path(PROJECT_ROOT, "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "daily_benchmark.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    args = parse_args()
    metric_config = config.METRIC_CONFIGS[args.metric]
    logger.info(
        "Daily benchmark; metric: %s",
        metric_config.display_label,
    )

    actual_vs_forecast_df = load_daily_actual_vs_forecast(args.data_path)
    location_ids = actual_vs_forecast_df[config.COL_ENTITY_UID].unique().tolist()
    summary = summarize_actual_vs_forecast_data(actual_vs_forecast_df, location_ids)
    logger.info("Data summary: %s", summary)

    base_anchor = determine_anchor_date(actual_vs_forecast_df)

    anchors = [base_anchor - pd.Timedelta(weeks=window_index) for window_index in range(args.rolling_windows)]

    all_results = []
    for window_index, anchor in enumerate(anchors):
        test_end = anchor
        test_start = anchor - pd.Timedelta(weeks=config.TEST_PERIOD_WEEKS) + pd.Timedelta(days=1)
        window_label = f"w{window_index + 1}" if args.rolling_windows > 1 else ""

        result = _run_single_window(
            actual_vs_forecast_df=actual_vs_forecast_df,
            test_start=test_start,
            test_end=test_end,
            run_sf=not args.no_sf,
            run_ml=not args.no_ml,
            run_cv=not args.no_cv,
            metric_config=metric_config,
            window_label=window_label,
        )

        metric_prefix = f"{metric_config.metric_name}_" if metric_config.metric_name != "net_sales" else ""
        window_prefix = f"{window_label}_" if window_label else ""
        file_prefix = f"daily_{metric_prefix}{window_prefix}"
        save_metrics(
            all_location_metrics=result["all_location_metrics"],
            aggregate_metrics_df=combine_aggregate_metrics(result["aggregate_metrics_dict"]),
            win_loss_df=result["win_loss_df"],
            statistical_tests_df=result["statistical_tests_df"],
            cv_metrics_df=result["cv_metrics_df"],
            output_dir=PROJECT_ROOT / config.METRICS_DIR,
            file_prefix=file_prefix,
            actuals=result["actuals"],
            actuals_strict=result["actuals_strict"],
            predictions_dict=result["predictions"],
        )

        timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
        report_filename = f"daily_{metric_prefix}benchmark_{window_prefix}{timestamp}.html"
        baseline_notes = (
            [
                f"<strong>{metric_config.display_label} benchmark:</strong> "
                "No production baseline is available for this metric. "
                "Win/loss and statistical tests are omitted. Only Nixtla model results are shown."
            ]
            if not metric_config.has_baseline
            else []
        )
        report_context = ReportContext(
            interval="daily",
            n_locations=result["actuals"][config.NIXTLA_UNIQUE_ID].nunique(),
            test_start=result["test_start"],
            test_end=result["test_end"],
            baseline_model=config.BASELINE_MODEL if metric_config.has_baseline else None,
            train_history_weeks=config.TRAIN_HISTORY_WEEKS,
            cv_num_windows=config.CV_NUM_WINDOWS if not args.no_cv else None,
            n_train_rows=None,
            location_metadata=result["volume_tiers"],
            metric_label=metric_config.display_label,
            extra_notes=[
                "Missing dates in training data are filled with <code>y=0.0</code>. "
                "This assumes absent dates represent days when the location was closed "
                "or not operating. "
                "These zero-sales days are included in Nixtla model training but are excluded from "
                "evaluation per <code>EXCLUDE_ZERO_ACTUAL_FROM_EVAL=True</code>.",
                *baseline_notes,
            ],
        )
        build_report(
            context=report_context,
            all_location_metrics=result["all_location_metrics"],
            aggregate_metrics_dict=result["aggregate_metrics_dict"],
            win_loss_df=result["win_loss_df"],
            statistical_tests_df=result["statistical_tests_df"],
            cv_metrics_df=result["cv_metrics_df"],
            actuals=result["actuals"],
            predictions_dict=result["predictions"],
            output_path=PROJECT_ROOT / config.REPORTS_DIR / report_filename,
        )

        all_results.append(result)

    if len(all_results) > 1:
        _log_rolling_aggregate(all_results, anchors)

    logger.info("Daily benchmark complete")


def _log_rolling_aggregate(results: list[dict], anchors: list[pd.Timestamp]) -> None:
    """Log mean and std of MAE across rolling windows as a compact stability view."""
    summary_rows = []
    for window_index, result in enumerate(results):
        for model_name, agg_df in result["aggregate_metrics_dict"].items():
            mae_row = agg_df[agg_df["metric"] == "MAE"]
            if not mae_row.empty:
                summary_rows.append(
                    {
                        "model": model_name,
                        "window": window_index + 1,
                        "MAE_mean": mae_row["mean"].iloc[0],
                    }
                )
    rolling_summary_input = pd.DataFrame(summary_rows)
    summary = rolling_summary_input.groupby("model")["MAE_mean"].agg(["mean", "std"]).reset_index().sort_values("mean")
    summary["MAE (mean +/- std)"] = summary.apply(lambda row: f"${row['mean']:,.2f} +/- ${row['std']:,.2f}", axis=1)
    logger.info(
        "Rolling-window aggregate (%d windows):\n%s",
        len(results),
        summary[["model", "MAE (mean +/- std)"]].to_string(index=False),
    )


if __name__ == "__main__":
    main()
