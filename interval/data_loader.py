from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_EXPECTED_INTERVAL_ACTUAL_VS_FORECAST_COLUMNS = {
    config.COL_ENTITY_UID,
    config.COL_CUSTOMER_ID,
    config.COL_SITE_ID,
    config.COL_INTERVAL_ID_TICKS,
    config.COL_INTERVAL_START_TS,
    config.COL_NET_SALES,
    config.COL_BASELINE_PROJECTION,
    config.COL_USER_ADJUSTMENT,
    config.COL_PUBLISHED_USER_ADJUSTMENT,
    config.COL_IS_CLOSED,
}

# Optional columns; absent extracts degrade to calendar-day behavior.
_OPTIONAL_INTERVAL_ACTUAL_VS_FORECAST_COLUMNS = {
    config.COL_BUSINESS_START_HOUR_UTC,
    config.COL_DAILY_NET_SALES,
    config.COL_IS_OUTLIER,
    config.COL_OUTLIER_EXCLUSION_REASON,
}


def load_interval_actual_vs_forecast(path: str | Path | None = None) -> pd.DataFrame:
    """Load and schema-validate the interval extract.

    Missing sales are coerced to zero; duplicate (location, timestamp) pairs
    are dropped so later reindex operations stay stable.
    """
    file_path = Path(path) if path is not None else config.INPUT_FILE

    interval_df = pd.read_csv(
        file_path,
        dtype={
            config.COL_ENTITY_UID: str,
            config.COL_CUSTOMER_ID: "Int64",
            config.COL_SITE_ID: str,
            config.COL_INTERVAL_ID_TICKS: "Int64",
            config.COL_NET_SALES: float,
            config.COL_BASELINE_PROJECTION: float,
            config.COL_USER_ADJUSTMENT: float,
            config.COL_PUBLISHED_USER_ADJUSTMENT: float,
            config.COL_IS_CLOSED: "boolean",
            config.COL_BUSINESS_START_HOUR_UTC: "Int64",
            config.COL_DAILY_NET_SALES: float,
            config.COL_IS_OUTLIER: "boolean",
            config.COL_OUTLIER_EXCLUSION_REASON: "Int64",
        },
        parse_dates=[config.COL_INTERVAL_START_TS],
    )

    missing_columns = _EXPECTED_INTERVAL_ACTUAL_VS_FORECAST_COLUMNS - set(interval_df.columns)
    if missing_columns:
        raise ValueError(
            "interval_actual_vs_forecast.csv is missing expected columns: "
            f"{sorted(missing_columns)}\n"
            "Re-run snowflake/interval_avf.sql and save the output to "
            "data/input/interval_actual_vs_forecast.csv"
        )

    absent_optional = _OPTIONAL_INTERVAL_ACTUAL_VS_FORECAST_COLUMNS - set(interval_df.columns)
    if absent_optional:
        outlier_columns_absent = {
            config.COL_IS_OUTLIER,
            config.COL_OUTLIER_EXCLUSION_REASON,
        } & absent_optional
        other_columns_absent = absent_optional - outlier_columns_absent

        if outlier_columns_absent:
            logger.warning(
                "interval_actual_vs_forecast.csv is missing outlier columns %s. Re-run "
                "snowflake/interval_avf.sql to enable production-matching "
                "outlier exclusion. Without IS_OUTLIER the benchmark trains on days that "
                "the production system excludes, making results overly optimistic.",
                sorted(outlier_columns_absent),
            )
        if other_columns_absent:
            logger.warning(
                "interval_actual_vs_forecast.csv is missing optional columns %s. Re-run "
                "snowflake/interval_avf.sql to enable business-day-aware "
                "fallback interval fill. Falling back to calendar-day behavior.",
                sorted(other_columns_absent),
            )

    interval_df[config.COL_ENTITY_UID] = interval_df[config.COL_ENTITY_UID].str.strip()
    interval_df[config.COL_NET_SALES] = interval_df[config.COL_NET_SALES].fillna(0.0)
    if config.COL_GROSS_SALES in interval_df.columns:
        interval_df[config.COL_GROSS_SALES] = interval_df[config.COL_GROSS_SALES].fillna(0.0)

    initial_row_count = len(interval_df)
    interval_df = interval_df.drop_duplicates(
        subset=[config.COL_ENTITY_UID, config.COL_INTERVAL_START_TS],
        keep="first",
    ).reset_index(drop=True)
    duplicates_removed = initial_row_count - len(interval_df)
    if duplicates_removed > 0:
        logger.warning(
            "Removed %d duplicate (location, timestamp) rows from source CSV",
            duplicates_removed,
        )

    location_count = interval_df[config.COL_ENTITY_UID].nunique()
    test_row_count = interval_df[config.COL_BASELINE_PROJECTION].notna().sum()
    logger.info(
        "Loaded %d rows, %d locations, %d test-period rows",
        len(interval_df),
        location_count,
        test_row_count,
    )
    return interval_df


def determine_test_window(
    interval_df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """One-week test window ending at the last projected interval in the extract."""
    projected_rows = interval_df[interval_df[config.COL_BASELINE_PROJECTION].notna()]
    if projected_rows.empty:
        raise ValueError(
            "No baseline projections found in interval_actual_vs_forecast.csv; "
            "cannot determine test window. "
            "Re-run snowflake/interval_avf.sql and save the output to "
            "data/input/interval_actual_vs_forecast.csv"
        )
    test_end = pd.Timestamp(projected_rows[config.COL_INTERVAL_START_TS].max())
    test_start = test_end - pd.Timedelta(weeks=config.TEST_PERIOD_WEEKS) + pd.Timedelta(minutes=15)
    logger.info(
        "Test window: %s to %s (%d weeks)",
        test_start,
        test_end,
        config.TEST_PERIOD_WEEKS,
    )
    return test_start, test_end


def build_baseline_variants(test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Baseline forecast variants for the test period.

    Standard variants drop zero-projection intervals; strict variants retain
    them so the cost of zero-forecast-vs-positive-actual is reflected in
    accuracy. Adjusted variants layer manager or published adjustments on top.
    """
    test_df = test_df[test_df[config.COL_BASELINE_PROJECTION].notna()].copy()
    adjusted_draft_column = "_adjusted_draft"
    adjusted_published_column = "_adjusted_published"

    def _to_nixtla(subset: pd.DataFrame, value_column: str) -> pd.DataFrame:
        return (
            subset[[config.COL_ENTITY_UID, config.COL_INTERVAL_START_TS, value_column]]
            .rename(
                columns={
                    config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
                    config.COL_INTERVAL_START_TS: config.NIXTLA_DS,
                    value_column: config.NIXTLA_Y,
                }
            )
            .reset_index(drop=True)
        )

    test_df[adjusted_draft_column] = test_df[config.COL_BASELINE_PROJECTION] + test_df[
        config.COL_USER_ADJUSTMENT
    ].fillna(0.0)
    test_df[adjusted_published_column] = test_df[config.COL_BASELINE_PROJECTION] + test_df[
        config.COL_PUBLISHED_USER_ADJUSTMENT
    ].fillna(0.0)

    if config.EXCLUDE_ZERO_PROJECTION_FROM_EVAL:
        baseline_filter = test_df[config.COL_BASELINE_PROJECTION] > 0
    else:
        baseline_filter = pd.Series([True] * len(test_df), index=test_df.index)

    return {
        config.BASELINE_PRODUCTION: _to_nixtla(test_df[baseline_filter], config.COL_BASELINE_PROJECTION),
        config.BASELINE_ADJUSTED_DRAFT: _to_nixtla(test_df[baseline_filter], adjusted_draft_column),
        config.BASELINE_ADJUSTED_PUBLISHED: _to_nixtla(test_df[baseline_filter], adjusted_published_column),
        config.BASELINE_PRODUCTION_STRICT: _to_nixtla(test_df, config.COL_BASELINE_PROJECTION),
        config.BASELINE_ADJUSTED_DRAFT_STRICT: _to_nixtla(test_df, adjusted_draft_column),
    }


def build_actuals(
    test_df: pd.DataFrame,
    target_column: str = config.COL_NET_SALES,
) -> pd.DataFrame:
    """Actuals filtered to match ``build_baseline_variants`` row coverage.

    ``target_column`` selects net vs. gross sales.
    """
    subset = test_df
    if config.EXCLUDE_ZERO_PROJECTION_FROM_EVAL:
        subset = test_df[
            test_df[config.COL_BASELINE_PROJECTION].notna() & (test_df[config.COL_BASELINE_PROJECTION] > 0)
        ]

    return (
        subset[
            [
                config.COL_ENTITY_UID,
                config.COL_INTERVAL_START_TS,
                target_column,
            ]
        ]
        .rename(
            columns={
                config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
                config.COL_INTERVAL_START_TS: config.NIXTLA_DS,
                target_column: config.NIXTLA_Y,
            }
        )
        .reset_index(drop=True)
    )


def build_actuals_strict(
    test_df: pd.DataFrame,
    target_column: str = config.COL_NET_SALES,
) -> pd.DataFrame:
    """Actuals for the strict baseline variants — every interval with a non-null projection.

    The strict variants include zero-projection intervals in their predictions;
    evaluating against filtered actuals would silently drop those rows on merge
    and collapse strict and standard into identical results.
    """
    subset = test_df[test_df[config.COL_BASELINE_PROJECTION].notna()]
    return (
        subset[
            [
                config.COL_ENTITY_UID,
                config.COL_INTERVAL_START_TS,
                target_column,
            ]
        ]
        .rename(
            columns={
                config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
                config.COL_INTERVAL_START_TS: config.NIXTLA_DS,
                target_column: config.NIXTLA_Y,
            }
        )
        .reset_index(drop=True)
    )


def summarize_actual_vs_forecast_data(interval_df: pd.DataFrame, location_ids: list[str]) -> dict:
    """Coverage summary (counts, date range, median intervals/location) for logging."""
    location_subset = interval_df[interval_df[config.COL_ENTITY_UID].isin(location_ids)]
    intervals_per_location = location_subset.groupby(config.COL_ENTITY_UID)[config.COL_INTERVAL_START_TS].nunique()
    return {
        "n_locations": len(location_ids),
        "n_rows": len(location_subset),
        "date_min": location_subset[config.COL_INTERVAL_START_TS].min(),
        "date_max": location_subset[config.COL_INTERVAL_START_TS].max(),
        "intervals_per_location_median": int(intervals_per_location.median()),
    }


def build_vsd_hours_lookup(actual_vs_forecast_df: pd.DataFrame) -> dict[str, int]:
    """Per-location business-day start hour (UTC), using the modal observed value.

    Returns an empty dict when ``BUSINESS_START_HOUR_UTC`` is absent — callers
    fall back to calendar-midnight boundaries. Inconsistent values within a
    location (DST or config changes) are reduced to the mode, with a 50/50
    split raising rather than guessing.
    """
    if config.COL_BUSINESS_START_HOUR_UTC not in actual_vs_forecast_df.columns:
        return {}

    grouped = actual_vs_forecast_df.groupby(config.COL_ENTITY_UID)[config.COL_BUSINESS_START_HOUR_UTC]

    modes_per_location = {uid: hours.dropna().mode() for uid, hours in grouped}

    distinct_counts = grouped.nunique()
    mixed_locations = distinct_counts[distinct_counts > 1].index.tolist()
    if mixed_locations:
        tied_locations = [uid for uid in mixed_locations if len(modes_per_location[uid]) > 1]
        if tied_locations:
            raise ValueError(
                f"{len(tied_locations)} location(s) have a 50/50 split in BUSINESS_START_HOUR_UTC "
                f"(likely DST transition or timezone config change mid-dataset). "
                f"Cannot determine a single representative hour. Manual intervention required. "
                f"Affected locations: {tied_locations[:5]}"
            )
        logger.warning(
            "%d location(s) have inconsistent BUSINESS_START_HOUR_UTC values "
            "(DST or timezone config change). Using the most common hour for each. "
            "First 5: %s",
            len(mixed_locations),
            mixed_locations[:5],
        )

    return {uid: int(modes.iloc[0]) for uid, modes in modes_per_location.items() if len(modes)}


def _business_day_series(timestamps: pd.Series, vsd_utc: pd.Series) -> pd.Series:
    """Map UTC timestamps to their owning business day.

    Intervals whose UTC hour falls before the location's business-day start
    hour roll back to the previous calendar day, mirroring the production
    pipeline. ``vsd_utc == 0`` is a no-op, so missing-hour defaults and true
    midnight starts behave identically.
    """
    dates = timestamps.dt.normalize()
    before_vsd = (vsd_utc > 0) & (timestamps.dt.hour < vsd_utc)
    return dates.where(~before_vsd, dates - pd.Timedelta(days=1))


def fill_fallback_intervals(
    actual_vs_forecast_df: pd.DataFrame,
    nixtla_df: pd.DataFrame,
    vsd_hours: dict[str, int],
) -> pd.DataFrame:
    """Distribute daily totals across intervals when interval rows are missing.

    Mirrors production fallback: business days with a positive daily total
    but no interval rows are spread using per-day-of-week historical shares,
    preventing understated training signal. No-op without DAILY_NET_SALES.
    """
    if config.COL_DAILY_NET_SALES not in actual_vs_forecast_df.columns:
        return nixtla_df

    uid_col = config.COL_ENTITY_UID
    ts_col = config.COL_INTERVAL_START_TS

    actual_vs_forecast_work = actual_vs_forecast_df.copy()
    actual_vs_forecast_work["_vsd"] = actual_vs_forecast_work[uid_col].map(vsd_hours).fillna(0).astype(int)
    actual_vs_forecast_work["_biz_day"] = _business_day_series(
        actual_vs_forecast_work[ts_col], actual_vs_forecast_work["_vsd"]
    )

    open_mask = ~actual_vs_forecast_work[config.COL_IS_CLOSED].fillna(False).eq(True)
    days_with_data: set[tuple] = set(
        zip(
            actual_vs_forecast_work.loc[open_mask, uid_col],
            actual_vs_forecast_work.loc[open_mask, "_biz_day"],
        )
    )

    daily_totals = (
        actual_vs_forecast_work.groupby([uid_col, "_biz_day"])[config.COL_DAILY_NET_SALES]
        .first()
        .reset_index()
        .rename(columns={uid_col: config.NIXTLA_UNIQUE_ID})
    )

    fallback_df = daily_totals[
        ~daily_totals.apply(
            lambda r: (r[config.NIXTLA_UNIQUE_ID], r["_biz_day"]) in days_with_data,
            axis=1,
        )
        & (daily_totals[config.COL_DAILY_NET_SALES].fillna(0) > 0)
    ].copy()
    fallback_df = fallback_df.rename(columns={config.COL_DAILY_NET_SALES: "_daily_total"})

    if fallback_df.empty:
        return nixtla_df

    logger.info(
        "Applying fallback interval fill for %d location-business-days",
        len(fallback_df),
    )

    nixtla_work = nixtla_df.copy()
    nixtla_work["_vsd"] = nixtla_work[config.NIXTLA_UNIQUE_ID].map(vsd_hours).fillna(0).astype(int)
    nixtla_work["_biz_day"] = _business_day_series(nixtla_work[config.NIXTLA_DS], nixtla_work["_vsd"])
    timestamps = nixtla_work[config.NIXTLA_DS].dt
    nixtla_work["_slot"] = timestamps.hour * 4 + timestamps.minute // 15
    nixtla_work["_dow"] = pd.to_datetime(nixtla_work["_biz_day"]).dt.dayofweek

    nixtla_work = nixtla_work.merge(
        fallback_df[[config.NIXTLA_UNIQUE_ID, "_biz_day", "_daily_total"]],
        on=[config.NIXTLA_UNIQUE_ID, "_biz_day"],
        how="left",
    )
    fallback_mask = nixtla_work["_daily_total"].notna()

    nixtla_work["_day_total"] = nixtla_work.groupby([config.NIXTLA_UNIQUE_ID, "_biz_day"])[config.NIXTLA_Y].transform(
        "sum"
    )

    share_base = nixtla_work[~fallback_mask & (nixtla_work["_day_total"] > 0)].copy()
    share_base["_share"] = share_base[config.NIXTLA_Y] / share_base["_day_total"]
    avg_shares = share_base.groupby([config.NIXTLA_UNIQUE_ID, "_dow", "_slot"])["_share"].mean().to_dict()

    fallback_rows = nixtla_work[fallback_mask].copy()

    fallback_rows["_share"] = [
        avg_shares.get((uid, dow, slot), 1.0 / config.INTERVALS_PER_DAY)
        for uid, dow, slot in zip(
            fallback_rows[config.NIXTLA_UNIQUE_ID],
            fallback_rows["_dow"],
            fallback_rows["_slot"],
        )
    ]

    share_sums = fallback_rows.groupby([config.NIXTLA_UNIQUE_ID, "_biz_day"])["_share"].transform("sum").replace(0, 1.0)
    fallback_rows[config.NIXTLA_Y] = fallback_rows["_daily_total"] * fallback_rows["_share"] / share_sums

    nixtla_work.loc[fallback_mask, config.NIXTLA_Y] = fallback_rows[config.NIXTLA_Y].values

    return nixtla_work.drop(
        columns=["_vsd", "_biz_day", "_slot", "_dow", "_day_total", "_daily_total"],
        errors="ignore",
    )


def to_nixtla_training(
    training_df: pd.DataFrame,
    target_column: str = config.COL_NET_SALES,
) -> pd.DataFrame:
    """Reshape the raw extract into the Nixtla ``[unique_id, ds, y]`` schema.

    Drops closed intervals (the production baseline excludes them too) and
    defensively de-duplicates in case the same interval appears with
    conflicting IS_CLOSED values.
    """
    open_intervals = training_df[~training_df[config.COL_IS_CLOSED].fillna(False).eq(True)]
    nixtla_df = (
        open_intervals[
            [
                config.COL_ENTITY_UID,
                config.COL_INTERVAL_START_TS,
                target_column,
            ]
        ]
        .rename(
            columns={
                config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
                config.COL_INTERVAL_START_TS: config.NIXTLA_DS,
                target_column: config.NIXTLA_Y,
            }
        )
        .reset_index(drop=True)
    )

    initial_count = len(nixtla_df)
    nixtla_df = nixtla_df.drop_duplicates(
        subset=[config.NIXTLA_UNIQUE_ID, config.NIXTLA_DS],
        keep="first",
    ).reset_index(drop=True)
    if len(nixtla_df) < initial_count:
        logger.warning(
            "Removed %d duplicate (unique_id, ds) pairs during Nixtla conversion",
            initial_count - len(nixtla_df),
        )

    return nixtla_df


def drop_outlier_days(
    nixtla_df: pd.DataFrame,
    actual_vs_forecast_df: pd.DataFrame,
    vsd_hours: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Drop every 15-min interval whose owning business day is flagged outlier.

    The outlier flag is day-level; this re-derives each interval's business
    day with the same boundary logic as ``_business_day_series`` and removes
    flagged ``(unique_id, business_day)`` pairs. Without IS_OUTLIER the
    function warns and returns input unchanged — the benchmark will be
    overly optimistic until the SQL extract is re-run.

    ``vsd_hours`` is the lookup from ``build_vsd_hours_lookup``; an empty or
    None value falls back to calendar-day boundaries.
    """
    if config.COL_IS_OUTLIER not in actual_vs_forecast_df.columns:
        logger.warning(
            "IS_OUTLIER column is absent from the source frame; outlier days will NOT be "
            "dropped. Re-run snowflake/interval_avf.sql to fix this."
        )
        return nixtla_df

    outlier_mask = actual_vs_forecast_df[config.COL_IS_OUTLIER].fillna(False).eq(True)
    if not outlier_mask.any():
        return nixtla_df

    business_start_hours = vsd_hours or {}

    outlier_timestamps = actual_vs_forecast_df.loc[outlier_mask, config.COL_INTERVAL_START_TS]
    outlier_uids = actual_vs_forecast_df.loc[outlier_mask, config.COL_ENTITY_UID]
    outlier_business_days: set[tuple] = set(
        zip(
            outlier_uids,
            _business_day_series(
                outlier_timestamps,
                outlier_uids.map(business_start_hours).fillna(0).astype(int),
            ),
        )
    )

    interval_business_days = _business_day_series(
        nixtla_df[config.NIXTLA_DS],
        nixtla_df[config.NIXTLA_UNIQUE_ID].map(business_start_hours).fillna(0).astype(int),
    )

    keep_mask = ~pd.Series(
        list(
            zip(
                nixtla_df[config.NIXTLA_UNIQUE_ID],
                interval_business_days,
            )
        ),
        index=nixtla_df.index,
    ).isin(outlier_business_days)

    dropped_interval_count = (~keep_mask).sum()
    logger.info(
        "Dropped %d intervals across %d outlier business days flagged by the production system",
        dropped_interval_count,
        len(outlier_business_days),
    )
    return nixtla_df[keep_mask].reset_index(drop=True)
