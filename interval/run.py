#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.paths import PROJECT_ROOT
from reporting.report import ReportContext, build_report

from . import config
from .data_loader import (
    build_actuals,
    build_actuals_strict,
    build_baseline_variants,
    build_vsd_hours_lookup,
    determine_test_window,
    fill_fallback_intervals,
    load_interval_actual_vs_forecast,
    summarize_actual_vs_forecast_data,
    to_nixtla_training,
)
from .evaluation import (
    aggregate_metrics,
    build_volume_tiers,
    combine_aggregate_metrics,
    compute_cv_metrics,
    compute_metrics,
    compute_statistical_tests,
    compute_win_loss,
    save_metrics,
)
from .models import (
    run_mlforecast,
    run_mlforecast_cv,
    run_statsforecast,
    run_statsforecast_cv,
)
from .data_loader import drop_outlier_days
from .preprocessing import (
    add_calendar_features,
    add_fourier_features,
    create_train_test_split,
    fill_missing_intervals,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI flags for the interval benchmark entry point."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-sf", action="store_true", help="Skip StatsForecast models")
    parser.add_argument("--no-ml", action="store_true", help="Skip MLForecast models")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override path to interval_actual_vs_forecast.csv",
    )
    parser.add_argument(
        "--metric",
        choices=list(config.METRIC_CONFIGS.keys()),
        default=config.DEFAULT_METRIC,
        help="Forecast metric to benchmark (default: net_sales). "
        "gross_sales runs Nixtla models only; no production-baseline comparison is performed.",
    )
    return parser.parse_args()


def _select_sample_locations(actuals: pd.DataFrame, limit: int = 3) -> list[str]:
    """Pick the top-volume locations for sample forecast charts."""
    return actuals[config.NIXTLA_UNIQUE_ID].value_counts().head(limit).index.tolist()


def main() -> None:
    """Interval benchmark entry point.

    Test window is anchored on the last interval with a baseline projection,
    then a date-based train/test split runs (matching the daily pipeline).
    """
    Path(PROJECT_ROOT, "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "interval_benchmark.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    args = parse_args()
    metric_config = config.METRIC_CONFIGS[args.metric]
    logger.info(
        "Interval benchmark; metric: %s",
        metric_config.display_label,
    )

    actual_vs_forecast_df = load_interval_actual_vs_forecast(args.data_path)
    location_ids = actual_vs_forecast_df[config.COL_ENTITY_UID].unique().tolist()
    summary = summarize_actual_vs_forecast_data(actual_vs_forecast_df, location_ids)
    logger.info("Data summary: %s", summary)

    test_start, test_end = determine_test_window(actual_vs_forecast_df)
    vsd_hours = build_vsd_hours_lookup(actual_vs_forecast_df)
    if not vsd_hours:
        logger.warning(
            "BUSINESS_START_HOUR_UTC column absent from input. "
            "Business-day-aware clipping and fallback interval imputation will use UTC midnight (hour 0) for all locations. "
            "Re-run the Snowflake extract with the updated SQL to enable business-day-aware behavior."
        )

    nixtla_full = to_nixtla_training(actual_vs_forecast_df, target_column=metric_config.actuals_column)
    nixtla_full = fill_missing_intervals(nixtla_full)
    nixtla_full = add_calendar_features(nixtla_full)

    nixtla_training, _ = create_train_test_split(
        nixtla_full,
        test_start,
        test_end,
        max_train_history_weeks=config.TRAIN_HISTORY_WEEKS,
    )
    # Run fallback fill after the train/test split so test-period distributions
    # do not influence training-period imputation weights.
    nixtla_training = fill_fallback_intervals(actual_vs_forecast_df, nixtla_training, vsd_hours)
    nixtla_training = drop_outlier_days(nixtla_training, actual_vs_forecast_df, vsd_hours=vsd_hours)
    nixtla_training, fourier_columns = add_fourier_features(nixtla_training)

    test_actual_vs_forecast = actual_vs_forecast_df[
        (actual_vs_forecast_df[config.COL_INTERVAL_START_TS] >= test_start)
        & (actual_vs_forecast_df[config.COL_INTERVAL_START_TS] <= test_end)
    ]
    predictions: dict[str, pd.DataFrame] = (
        build_baseline_variants(test_actual_vs_forecast) if metric_config.has_baseline else {}
    )
    if not metric_config.has_baseline:
        logger.info(
            "%s benchmark: no production baseline available; win/loss and statistical tests will be skipped.",
            metric_config.display_label,
        )

    actuals = build_actuals(test_actual_vs_forecast, target_column=metric_config.actuals_column)
    actuals_strict = build_actuals_strict(test_actual_vs_forecast, target_column=metric_config.actuals_column)

    if not args.no_sf:
        predictions.update(run_statsforecast(nixtla_training))

    if not args.no_ml:
        predictions.update(
            run_mlforecast(
                nixtla_training,
                fourier_columns=fourier_columns,
            )
        )

    cv_metrics_df = pd.DataFrame()
    if not args.no_cv:
        sf_cv = run_statsforecast_cv(nixtla_training) if not args.no_sf else pd.DataFrame()
        ml_cv = (
            run_mlforecast_cv(nixtla_training, fourier_columns=fourier_columns) if not args.no_ml else pd.DataFrame()
        )
        if not sf_cv.empty or not ml_cv.empty:
            combined_cv = pd.concat([df for df in [sf_cv, ml_cv] if not df.empty], ignore_index=True)
            cv_metrics_df = compute_cv_metrics(
                combined_cv,
                exclude_zero_actuals=config.EXCLUDE_ZERO_ACTUAL_FROM_EVAL,
            )

    strict_model_names = {
        config.Baseline_PRODUCTION_STRICT,
        config.Baseline_ADJUSTED_DRAFT_STRICT,
    }
    all_location_metrics: dict[str, pd.DataFrame] = {}
    aggregate_metrics_dict: dict[str, pd.DataFrame] = {}
    for model_name, prediction_df in predictions.items():
        eval_actuals = actuals_strict if model_name in strict_model_names else actuals
        location_metrics = compute_metrics(
            eval_actuals,
            prediction_df,
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

    metric_prefix = f"{metric_config.metric_name}_" if metric_config.metric_name != "net_sales" else ""
    save_metrics(
        all_location_metrics=all_location_metrics,
        aggregate_metrics_df=combine_aggregate_metrics(aggregate_metrics_dict),
        win_loss_df=win_loss_df,
        statistical_tests_df=statistical_tests_df,
        cv_metrics_df=cv_metrics_df,
        output_dir=PROJECT_ROOT / config.METRICS_DIR,
        file_prefix=f"interval_{metric_prefix}",
        actuals=actuals,
        actuals_strict=actuals_strict,
        predictions_dict=predictions,
    )

    timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
    report_filename = f"interval_{metric_prefix}benchmark_{timestamp}.html"

    sample_locations = _select_sample_locations(actuals)

    baseline_notes = (
        [
            f"<strong>{metric_config.display_label} benchmark:</strong> No production baseline is available for this metric. "
            "Win/loss and statistical tests are omitted. Only Nixtla model results are shown."
        ]
        if not metric_config.has_baseline
        else []
    )
    report_context = ReportContext(
        interval="15min",
        n_locations=actuals[config.NIXTLA_UNIQUE_ID].nunique(),
        test_start=test_start,
        test_end=test_end,
        baseline_model=config.BASELINE_MODEL if metric_config.has_baseline else None,
        train_history_weeks=config.TRAIN_HISTORY_WEEKS,
        cv_num_windows=config.CV_NUM_WINDOWS if not args.no_cv else None,
        n_train_rows=len(nixtla_training),
        location_metadata=volume_tiers,
        metric_label=metric_config.display_label,
        extra_notes=baseline_notes,
    )

    build_report(
        context=report_context,
        all_location_metrics=all_location_metrics,
        aggregate_metrics_dict=aggregate_metrics_dict,
        win_loss_df=win_loss_df,
        statistical_tests_df=statistical_tests_df,
        cv_metrics_df=cv_metrics_df,
        actuals=actuals,
        predictions_dict=predictions,
        sample_location_ids=sample_locations,
        output_path=PROJECT_ROOT / config.REPORTS_DIR / report_filename,
    )


if __name__ == "__main__":
    main()
