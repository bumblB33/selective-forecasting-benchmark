from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_NIXTLA_UNIQUE_ID = "unique_id"
_NIXTLA_DS = "ds"
_NIXTLA_Y = "y"
_METRICS = ["MAE", "WAPE", "MAPE", "RMSE"]
_TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M"
# A model must beat the baseline on more than this fraction of locations,
# alongside a negative median MAE difference, before it is labelled 'better'.
# Prevents outlier locations from flipping the median verdict.
_WIN_RATE_GUARD = 0.5


def compute_metrics(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    exclude_zero_actuals: bool = True,
) -> pd.DataFrame:
    """Per-location MAE/WAPE/MAPE/RMSE on the inner join of actuals and predictions."""
    aligned_rows = actuals.merge(
        predictions,
        on=[_NIXTLA_UNIQUE_ID, _NIXTLA_DS],
        suffixes=("_actual", "_pred"),
    )

    if exclude_zero_actuals:
        aligned_rows = aligned_rows[aligned_rows[f"{_NIXTLA_Y}_actual"] > 0]

    if aligned_rows.empty:
        return pd.DataFrame(columns=[_NIXTLA_UNIQUE_ID] + _METRICS)

    aligned_rows["_abs_error"] = (
        aligned_rows[f"{_NIXTLA_Y}_actual"] - aligned_rows[f"{_NIXTLA_Y}_pred"]
    ).abs()
    aligned_rows["_sq_error"] = (
        aligned_rows[f"{_NIXTLA_Y}_actual"] - aligned_rows[f"{_NIXTLA_Y}_pred"]
    ) ** 2

    actual_values = aligned_rows[f"{_NIXTLA_Y}_actual"]
    aligned_rows["_pct_error"] = np.where(
        actual_values > 0, aligned_rows["_abs_error"] / actual_values, np.nan
    )

    def compute_location_metrics(location_rows):
        actual_sum = location_rows[f"{_NIXTLA_Y}_actual"].sum()
        wape = (
            location_rows["_abs_error"].sum() / actual_sum if actual_sum > 0 else np.nan
        )
        return pd.Series(
            {
                "MAE": location_rows["_abs_error"].mean(),
                "WAPE": wape,
                "MAPE": location_rows["_pct_error"].mean(),
                "RMSE": np.sqrt(location_rows["_sq_error"].mean()),
                "n_rows": len(location_rows),
            }
        )

    location_metrics = (
        aligned_rows.groupby(_NIXTLA_UNIQUE_ID)
        .apply(compute_location_metrics)
        .reset_index()
    )
    return location_metrics


def aggregate_metrics(location_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, std, and IQR across locations for each metric."""
    metric_columns = [m for m in _METRICS if m in location_metrics.columns]
    summary_rows = []
    for metric in metric_columns:
        summary_rows.append(
            {
                "metric": metric,
                "mean": location_metrics[metric].mean(),
                "median": location_metrics[metric].median(),
                "std": location_metrics[metric].std(),
                "p25": location_metrics[metric].quantile(0.25),
                "p75": location_metrics[metric].quantile(0.75),
            }
        )
    return pd.DataFrame(summary_rows)


def compute_cv_metrics(
    cv_df: pd.DataFrame,
    exclude_zero_actuals: bool = True,
) -> pd.DataFrame:
    """Globally-aggregated metrics across all CV windows, sorted by MAE.

    CV WAPE is a single global ratio (sum of abs errors / sum of positive actuals);
    hold-out WAPE in ``aggregate_metrics`` is per-location then averaged. The two
    are not directly comparable. Output column names omit a ``CV_`` prefix to
    preserve historical CSV column parity.
    """
    if cv_df.empty:
        return pd.DataFrame()

    if exclude_zero_actuals:
        cv_df = cv_df[cv_df[_NIXTLA_Y] > 0]

    if cv_df.empty:
        return pd.DataFrame()

    reserved_columns = {
        _NIXTLA_UNIQUE_ID,
        _NIXTLA_DS,
        "cutoff",
        _NIXTLA_Y,
    }
    model_columns = [c for c in cv_df.columns if c not in reserved_columns]

    actuals = cv_df[_NIXTLA_Y]
    summary_rows = []
    for model_name in model_columns:
        predictions = cv_df[model_name]
        valid_mask = actuals > 0
        abs_errors = (actuals - predictions).abs()

        positive_actuals_sum = actuals[actuals > 0].sum()
        wape = (
            abs_errors.sum() / positive_actuals_sum
            if positive_actuals_sum > 0
            else np.nan
        )

        summary_rows.append(
            {
                "model": model_name,
                "MAE": abs_errors.mean(),
                "WAPE": wape,
                "MAPE": (abs_errors[valid_mask] / actuals[valid_mask]).mean(),
                "RMSE": np.sqrt(((actuals - predictions) ** 2).mean()),
                "n_windows": cv_df["cutoff"].nunique(),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("MAE").reset_index(drop=True)


def compute_win_loss(
    all_location_metrics: dict[str, pd.DataFrame],
    baseline_model: str,
) -> pd.DataFrame:
    """Per-location win/loss/tie counts against the baseline (lower MAE wins)."""
    if baseline_model not in all_location_metrics:
        logger.warning("Baseline model '%s' not found in metrics dict", baseline_model)
        return pd.DataFrame()

    baseline_mae = all_location_metrics[baseline_model].set_index(_NIXTLA_UNIQUE_ID)[
        "MAE"
    ]

    comparison_rows = []
    for model_name, location_metrics in all_location_metrics.items():
        if model_name == baseline_model:
            continue
        model_mae = location_metrics.set_index(_NIXTLA_UNIQUE_ID)["MAE"]
        common_locations = baseline_mae.index.intersection(model_mae.index)
        if len(common_locations) == 0:
            continue
        comparison = model_mae[common_locations] - baseline_mae[common_locations]
        wins = int((comparison < 0).sum())
        losses = int((comparison > 0).sum())
        ties = int((comparison.abs() < 1e-9).sum())
        comparison_rows.append(
            {
                "model": model_name,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "n_locations": len(common_locations),
                "win_rate": wins / len(common_locations)
                if len(common_locations) > 0
                else 0.0,
            }
        )

    if not comparison_rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(comparison_rows)
        .sort_values("win_rate", ascending=False)
        .reset_index(drop=True)
    )


def compute_statistical_tests(
    all_location_metrics: dict[str, pd.DataFrame],
    baseline_model: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Paired Wilcoxon on per-location MAE differences vs. baseline.

    ``direction`` is ``'better'`` only when the median difference is negative
    **and** the win rate exceeds 50 percent — see ``_WIN_RATE_GUARD``.
    """
    if baseline_model not in all_location_metrics:
        return pd.DataFrame()

    baseline_mae = all_location_metrics[baseline_model].set_index(_NIXTLA_UNIQUE_ID)[
        "MAE"
    ]

    test_rows = []
    for model_name, location_metrics in all_location_metrics.items():
        if model_name == baseline_model:
            continue
        model_mae = location_metrics.set_index(_NIXTLA_UNIQUE_ID)["MAE"]
        common_locations = baseline_mae.index.intersection(model_mae.index)
        if len(common_locations) < 10:
            continue
        differences = model_mae[common_locations] - baseline_mae[common_locations]
        differences_clean = differences.dropna()

        if len(differences_clean) == 0:
            continue

        n_unique = len(differences_clean.unique())
        n_ties = len(differences_clean) - n_unique
        if n_ties > len(differences_clean) * 0.1:
            logger.warning(
                "Model %s has %d ties in %d differences (%.1f%%) - "
                "Wilcoxon test may be less accurate",
                model_name,
                n_ties,
                len(differences_clean),
                100 * n_ties / len(differences_clean),
            )

        try:
            statistic, p_value = stats.wilcoxon(differences_clean, method="auto")
        except ValueError:
            continue

        wins = int((differences_clean < 0).sum())
        win_rate = wins / len(differences_clean) if len(differences_clean) > 0 else 0.0
        better = differences_clean.median() < 0 and win_rate > _WIN_RATE_GUARD

        test_rows.append(
            {
                "model": model_name,
                "statistic": round(statistic, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < (1 - confidence),
                "direction": "better" if better else "worse",
                "win_rate": round(win_rate, 4),
                "n_locations": len(common_locations),
            }
        )

    if not test_rows:
        return pd.DataFrame()
    return pd.DataFrame(test_rows).sort_values("p_value").reset_index(drop=True)


def build_volume_tiers(actuals: pd.DataFrame) -> pd.DataFrame:
    """Quartile each location into Q1..Q4 by mean observed sales."""
    mean_location_sales = actuals.groupby(_NIXTLA_UNIQUE_ID)[_NIXTLA_Y].mean()
    tier_labels = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    try:
        volume_tiers = pd.qcut(mean_location_sales, q=4, labels=tier_labels)
    except ValueError:
        volume_tiers = pd.qcut(
            mean_location_sales.rank(method="first"), q=4, labels=tier_labels
        )
    return pd.DataFrame(
        {
            _NIXTLA_UNIQUE_ID: mean_location_sales.index,
            "volume_tier": volume_tiers.values,
        }
    )


def save_metrics(
    all_location_metrics: dict[str, pd.DataFrame],
    aggregate_metrics_df: pd.DataFrame,
    win_loss_df: pd.DataFrame,
    statistical_tests_df: pd.DataFrame,
    cv_metrics_df: pd.DataFrame,
    output_dir: Path = Path("data/metrics"),
    file_prefix: str = "daily_",
    actuals: pd.DataFrame | None = None,
    actuals_strict: pd.DataFrame | None = None,
    predictions_dict: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Persist evaluation tables and (optionally) raw series for report regeneration.

    ``actuals_strict`` is saved separately because strict baseline variants
    evaluate on a broader row set (including zero-projection rows).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)

    combined_location_metrics = pd.concat(
        [
            df.assign(model=model_name)
            for model_name, df in all_location_metrics.items()
        ],
        ignore_index=True,
    )
    combined_location_metrics.to_csv(
        output_dir / f"{file_prefix}location_metrics_{timestamp}.csv", index=False
    )
    aggregate_metrics_df.to_csv(
        output_dir / f"{file_prefix}aggregate_metrics_{timestamp}.csv", index=False
    )
    win_loss_df.to_csv(
        output_dir / f"{file_prefix}win_loss_{timestamp}.csv", index=False
    )
    statistical_tests_df.to_csv(
        output_dir / f"{file_prefix}statistical_tests_{timestamp}.csv", index=False
    )
    if not cv_metrics_df.empty:
        cv_metrics_df.to_csv(
            output_dir / f"{file_prefix}cv_metrics_{timestamp}.csv", index=False
        )
    if actuals is not None:
        actuals.to_csv(
            output_dir / f"{file_prefix}actuals_{timestamp}.csv", index=False
        )
    if actuals_strict is not None:
        actuals_strict.to_csv(
            output_dir / f"{file_prefix}actuals_strict_{timestamp}.csv", index=False
        )
    if predictions_dict is not None:
        pd.concat(
            [
                df.assign(model=model_name)
                for model_name, df in predictions_dict.items()
            ],
            ignore_index=True,
        ).to_csv(output_dir / f"{file_prefix}predictions_{timestamp}.csv", index=False)
    logger.info(
        "Metrics saved to %s (prefix: %s, timestamp: %s)",
        output_dir,
        file_prefix,
        timestamp,
    )


def combine_aggregate_metrics(
    aggregate_metrics_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Flatten per-model aggregate metrics into one DataFrame with a ``model`` column."""
    return (
        pd.concat(
            aggregate_metrics_dict.values(),
            keys=aggregate_metrics_dict.keys(),
        )
        .reset_index(level=0)
        .rename(columns={"level_0": "model"})
    )
