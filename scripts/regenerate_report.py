#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from core.paths import PROJECT_ROOT
from core.evaluation import build_volume_tiers
from daily import config as daily_config
from interval import config as interval_config
from reporting.report import ReportContext, build_report

from .utils import find_latest_timestamp

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI flags for rebuilding a report from persisted benchmark artifacts."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interval",
        required=True,
        choices=["daily", "15min"],
        help="Granularity of the run to regenerate",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="File prefix override (e.g. 'daily_w2_'). "
        "Defaults to the standard prefix for the interval.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help="Exact timestamp string to use (e.g. '2024_01_15_14_30'). "
        "Skips auto-detection when provided.",
    )
    return parser.parse_args()


def _require_file(path: Path) -> Path:
    """Exit with a clear error if ``path`` does not exist."""
    if not path.exists():
        logger.error("Required file not found: %s", path)
        sys.exit(1)
    return path


def _read_table(path: Path) -> pd.DataFrame:
    """Read a metrics CSV, returning an empty frame for a table the run left empty."""
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    """Rebuild an HTML report from persisted CSVs, skipping model retraining."""
    args = parse_args()

    if args.interval == "daily":
        config = daily_config
        default_prefix = "daily_"
    else:
        config = interval_config
        default_prefix = "interval_"

    prefix = args.prefix if args.prefix else default_prefix
    metrics_dir = PROJECT_ROOT / config.METRICS_DIR
    reports_dir = PROJECT_ROOT / config.REPORTS_DIR

    if args.timestamp:
        timestamp = args.timestamp
    else:
        timestamp = find_latest_timestamp(metrics_dir, prefix)
        if timestamp is None:
            logger.error(
                "No run files found in %s with prefix '%s'. Run the benchmark first.",
                metrics_dir,
                prefix,
            )
            sys.exit(1)

    logger.info("Regenerating report for prefix='%s' timestamp='%s'", prefix, timestamp)

    actuals_path = _require_file(metrics_dir / f"{prefix}actuals_{timestamp}.csv")
    predictions_path = _require_file(
        metrics_dir / f"{prefix}predictions_{timestamp}.csv"
    )
    location_metrics_path = _require_file(
        metrics_dir / f"{prefix}location_metrics_{timestamp}.csv"
    )
    aggregate_metrics_path = _require_file(
        metrics_dir / f"{prefix}aggregate_metrics_{timestamp}.csv"
    )
    win_loss_path = _require_file(metrics_dir / f"{prefix}win_loss_{timestamp}.csv")
    statistical_tests_path = _require_file(
        metrics_dir / f"{prefix}statistical_tests_{timestamp}.csv"
    )

    cv_metrics_path = metrics_dir / f"{prefix}cv_metrics_{timestamp}.csv"
    cv_metrics_df = (
        pd.read_csv(cv_metrics_path) if cv_metrics_path.exists() else pd.DataFrame()
    )

    # actuals_strict is persisted but not consumed here — strict-variant metrics
    # are already captured in all_location_metrics.
    actuals = pd.read_csv(actuals_path, parse_dates=[config.NIXTLA_DS])
    predictions_combined = pd.read_csv(predictions_path, parse_dates=[config.NIXTLA_DS])
    location_metrics_combined = pd.read_csv(location_metrics_path)
    aggregate_metrics_combined = pd.read_csv(aggregate_metrics_path)
    win_loss_df = _read_table(win_loss_path)
    if win_loss_df.empty:
        logger.error(
            "%s is empty. Runs without a production baseline, such as gross_sales, "
            "cannot be regenerated.",
            win_loss_path.name,
        )
        sys.exit(1)
    statistical_tests_df = _read_table(statistical_tests_path)

    all_location_metrics: dict[str, pd.DataFrame] = {
        model_name: group.drop(columns=["model"]).reset_index(drop=True)
        for model_name, group in location_metrics_combined.groupby("model")
    }

    aggregate_metrics_dict: dict[str, pd.DataFrame] = {
        model_name: group.drop(columns=["model"]).reset_index(drop=True)
        for model_name, group in aggregate_metrics_combined.groupby("model")
    }

    predictions_dict: dict[str, pd.DataFrame] = {
        model_name: group.drop(columns=["model"]).reset_index(drop=True)
        for model_name, group in predictions_combined.groupby("model")
    }

    n_locations = actuals[config.NIXTLA_UNIQUE_ID].nunique()
    test_start = actuals[config.NIXTLA_DS].min()
    test_end = actuals[config.NIXTLA_DS].max()
    cv_num_windows = (
        int(cv_metrics_df["n_windows"].iloc[0]) if not cv_metrics_df.empty else None
    )
    volume_tiers = build_volume_tiers(actuals)

    report_context = ReportContext(
        interval=args.interval,
        n_locations=n_locations,
        test_start=test_start,
        test_end=test_end,
        baseline_model=config.BASELINE_MODEL,
        train_history_weeks=config.TRAIN_HISTORY_WEEKS,
        cv_num_windows=cv_num_windows,
        n_train_rows=None,
        location_metadata=volume_tiers,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"{prefix}{timestamp}.html"
    output_path = reports_dir / report_filename

    build_report(
        context=report_context,
        all_location_metrics=all_location_metrics,
        aggregate_metrics_dict=aggregate_metrics_dict,
        win_loss_df=win_loss_df,
        statistical_tests_df=statistical_tests_df,
        cv_metrics_df=cv_metrics_df,
        actuals=actuals,
        predictions_dict=predictions_dict,
        output_path=output_path,
    )

    logger.info("Report written to %s", output_path)


if __name__ == "__main__":
    main()
