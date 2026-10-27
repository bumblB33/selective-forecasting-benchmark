#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.paths import PROJECT_ROOT

from .utils import find_latest_timestamp

METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
GOLDEN_DIR = METRICS_DIR / "golden"
TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-4
METRIC_FILE_TYPES = [
    "location_metrics",
    "aggregate_metrics",
    "win_loss",
    "statistical_tests",
    "cv_metrics",
]
NON_NUMERIC_COLUMNS = frozenset({"model", "unique_id", "ds", "direction", "metric", "cutoff"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def _load_metric_files(metrics_dir: Path, prefix: str, timestamp: str) -> dict[str, pd.DataFrame]:
    """Load every metric CSV matching the prefix and timestamp."""
    result: dict[str, pd.DataFrame] = {}
    for file_type in METRIC_FILE_TYPES:
        path = metrics_dir / f"{prefix}{file_type}_{timestamp}.csv"
        if path.exists():
            result[file_type] = pd.read_csv(path)
        else:
            logger.warning("File not found (skipping): %s", path)
    return result


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns eligible for tolerance comparison."""
    return [col for col in df.columns if col not in NON_NUMERIC_COLUMNS and pd.api.types.is_numeric_dtype(df[col])]


def _stable_sort_key_columns(df: pd.DataFrame) -> list[str]:
    """Sort keys that produce a canonical row order, avoiding false-positive
    failures from non-deterministic row ordering between runs.
    """
    return [col for col in ("model", "unique_id", "metric", "cutoff", "direction") if col in df.columns]


def _compare_dataframes(name: str, df_a: pd.DataFrame, df_b: pd.DataFrame) -> int:
    """Compare two DataFrames numerically, returning the count of failing columns."""
    sort_keys = _stable_sort_key_columns(df_a)
    if sort_keys:
        df_a = df_a.sort_values(sort_keys).reset_index(drop=True)
        df_b = df_b.sort_values(sort_keys).reset_index(drop=True)

    failures = 0

    if set(df_a.columns) != set(df_b.columns):
        logger.error(
            "[%s] Column mismatch. A: %s  B: %s",
            name,
            sorted(df_a.columns.tolist()),
            sorted(df_b.columns.tolist()),
        )
        return 1

    if len(df_a) != len(df_b):
        logger.error(
            "[%s] Row count mismatch. A: %d rows  B: %d rows",
            name,
            len(df_a),
            len(df_b),
        )
        return 1

    for col in _numeric_columns(df_a):
        a_vals = df_a[col].to_numpy(dtype=float, na_value=np.nan)
        b_vals = df_b[col].to_numpy(dtype=float, na_value=np.nan)

        nan_mismatch = np.isnan(a_vals) != np.isnan(b_vals)
        if nan_mismatch.any():
            logger.error(
                "[%s] Column '%s': NaN presence differs in %d rows",
                name,
                col,
                int(nan_mismatch.sum()),
            )
            failures += 1
            continue

        mask = ~np.isnan(a_vals)
        if mask.any():
            abs_diff = np.abs(a_vals[mask] - b_vals[mask])
            rel_diff = abs_diff / (np.abs(b_vals[mask]) + 1e-10)
            max_diff = float(np.max(abs_diff))
            max_rel_diff = float(np.max(rel_diff))
            exceeded = (abs_diff > TOLERANCE) & (rel_diff > RELATIVE_TOLERANCE)
            if exceeded.any():
                all_exceeded = (np.abs(a_vals - b_vals) > TOLERANCE) & (
                    (np.abs(a_vals - b_vals) / (np.abs(b_vals) + 1e-10)) > RELATIVE_TOLERANCE
                )
                bad_idx = int(np.where(all_exceeded)[0][0])
                n_bad = int(exceeded.sum())
                logger.error(
                    "[%s] Column '%s': max_diff=%.2e (rel=%.2e) > tolerances "
                    "atol=%.0e rtol=%.0e in %d rows "
                    "(first bad row index: %d)",
                    name,
                    col,
                    max_diff,
                    max_rel_diff,
                    TOLERANCE,
                    RELATIVE_TOLERANCE,
                    n_bad,
                    bad_idx,
                )
                failures += 1

    if failures == 0:
        logger.info(
            "[%s] OK (all numeric columns within atol=%.0e rtol=%.0e)",
            name,
            TOLERANCE,
            RELATIVE_TOLERANCE,
        )

    return failures


def _compare_metric_sets(
    label_a: str,
    label_b: str,
    set_a: dict[str, pd.DataFrame],
    set_b: dict[str, pd.DataFrame],
) -> int:
    """Pair-wise compare two metric-file sets; returns total failure count."""
    total_failures = 0
    all_types = sorted(set(set_a) | set(set_b))
    for file_type in all_types:
        if file_type not in set_a:
            logger.warning("[%s] Missing in %s; skipping", file_type, label_a)
            continue
        if file_type not in set_b:
            logger.warning("[%s] Missing in %s; skipping", file_type, label_b)
            continue
        total_failures += _compare_dataframes(file_type, set_a[file_type], set_b[file_type])
    return total_failures


def _generate(metrics_dir: Path, prefix: str, timestamp: str) -> None:
    """Snapshot the latest metric CSVs into the golden directory."""
    golden_prefix_dir = GOLDEN_DIR / prefix.rstrip("_")
    golden_prefix_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for file_type in METRIC_FILE_TYPES:
        src = metrics_dir / f"{prefix}{file_type}_{timestamp}.csv"
        if not src.exists():
            logger.warning("Skipping missing file: %s", src)
            continue
        dst = golden_prefix_dir / f"{file_type}.csv"
        shutil.copy2(src, dst)
        copied += 1

    logger.info(
        "Golden snapshot saved (%d files) for prefix='%s' timestamp='%s'",
        copied,
        prefix,
        timestamp,
    )


def _check(metrics_dir: Path, prefix: str, timestamp: str) -> None:
    """Diff the latest run against the golden snapshot; exits non-zero on drift."""
    golden_prefix_dir = GOLDEN_DIR / prefix.rstrip("_")
    if not golden_prefix_dir.exists():
        logger.error("No golden snapshot found at %s. Run 'generate' first.", golden_prefix_dir)
        sys.exit(2)

    golden: dict[str, pd.DataFrame] = {}
    for file_type in METRIC_FILE_TYPES:
        path = golden_prefix_dir / f"{file_type}.csv"
        if path.exists():
            golden[file_type] = pd.read_csv(path)
        else:
            logger.warning("Golden file missing (skipping): %s", path)

    current = _load_metric_files(metrics_dir, prefix, timestamp)

    failures = _compare_metric_sets("golden", f"run:{timestamp}", golden, current)
    if failures:
        logger.error(
            "PARITY CHECK FAILED: %d column(s) drifted beyond tolerance %.0e",
            failures,
            TOLERANCE,
        )
        sys.exit(1)
    logger.info("PARITY CHECK PASSED: run '%s' matches golden snapshot", timestamp)


def _smoke_test(metrics_dir: Path, prefix: str, timestamp_a: str, timestamp_b: str) -> None:
    """Diff two timestamped runs directly; exits non-zero on drift."""
    set_a = _load_metric_files(metrics_dir, prefix, timestamp_a)
    set_b = _load_metric_files(metrics_dir, prefix, timestamp_b)

    failures = _compare_metric_sets(f"run:{timestamp_a}", f"run:{timestamp_b}", set_a, set_b)
    if failures:
        logger.error(
            "SMOKE TEST FAILED: %d column(s) drifted between %s and %s",
            failures,
            timestamp_a,
            timestamp_b,
        )
        sys.exit(1)
    logger.info("SMOKE TEST PASSED: runs '%s' and '%s' are equivalent", timestamp_a, timestamp_b)


def parse_args() -> argparse.Namespace:
    """CLI arguments for ``generate``, ``check``, and ``smoke-test`` modes."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--interval",
        required=True,
        choices=["daily", "15min"],
        help="Granularity of the benchmark run",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("generate", help="Save golden snapshot from the latest run")
    subparsers.add_parser("check", help="Compare latest run against golden snapshot")

    smoke = subparsers.add_parser("smoke-test", help="Compare two timestamped runs")
    smoke.add_argument("--timestamp-a", required=True, metavar="YYYY_MM_DD_HH_MM")
    smoke.add_argument("--timestamp-b", required=True, metavar="YYYY_MM_DD_HH_MM")

    return parser.parse_args()


def main() -> None:
    """Resolve paths and dispatch to the selected parity-check mode."""
    args = parse_args()

    prefix = "daily_" if args.interval == "daily" else "interval_"
    metrics_dir = METRICS_DIR

    if args.mode in ("generate", "check"):
        timestamp = find_latest_timestamp(metrics_dir, prefix)
        if timestamp is None:
            logger.error(
                "No run files found in %s with prefix '%s'. Run the benchmark first.",
                metrics_dir,
                prefix,
            )
            sys.exit(1)
        logger.info("Using timestamp: %s", timestamp)

    if args.mode == "generate":
        _generate(metrics_dir, prefix, timestamp)
    elif args.mode == "check":
        _check(metrics_dir, prefix, timestamp)
    else:
        _smoke_test(metrics_dir, prefix, args.timestamp_a, args.timestamp_b)


if __name__ == "__main__":
    main()
