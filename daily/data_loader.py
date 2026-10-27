from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)

_EXPECTED_DAILY_ACTUAL_VS_FORECAST_COLUMNS = {
    config.COL_ENTITY_UID,
    config.COL_CUSTOMER_ID,
    config.COL_SITE_ID,
    config.COL_DAY_START_TIME,
    config.COL_PROJECTION,
    config.COL_USER_ADJUSTMENT,
    config.COL_PUBLISHED_USER_ADJUSTMENT,
    config.COL_ACTUAL_VALUE,
    config.COL_IS_GENERATED_BY_MANAGER,
    config.COL_IS_SYSTEMATIC_GENERATED,
    config.COL_IS_OUTLIER,
    config.COL_OUTLIER_EXCLUSION_REASON,
}

# Optional columns let single-metric legacy extracts keep loading; gross-sales
# benchmarking fails loudly if its column is missing.
_OPTIONAL_DAILY_ACTUAL_VS_FORECAST_COLUMNS = {
    config.COL_GROSS_SALES_ACTUAL,
}


def load_daily_actual_vs_forecast(path: str | Path | None = None) -> pd.DataFrame:
    """Load and schema-validate the daily extract.

    Rows are preserved for downstream filtering; only exact duplicate
    (location, day) pairs are dropped so later reindex operations stay stable.
    """
    file_path = Path(path) if path is not None else config.INPUT_FILE

    actual_vs_forecast_df = pd.read_csv(
        file_path,
        dtype={
            config.COL_ENTITY_UID: str,
            config.COL_CUSTOMER_ID: "Int64",
            config.COL_SITE_ID: str,
            config.COL_PROJECTION: float,
            config.COL_USER_ADJUSTMENT: float,
            config.COL_PUBLISHED_USER_ADJUSTMENT: float,
            config.COL_ACTUAL_VALUE: float,
            config.COL_IS_GENERATED_BY_MANAGER: "boolean",
            config.COL_IS_SYSTEMATIC_GENERATED: "boolean",
            config.COL_IS_OUTLIER: "boolean",
            config.COL_OUTLIER_EXCLUSION_REASON: "Int64",
        },
        parse_dates=[config.COL_DAY_START_TIME],
    )

    missing_columns = _EXPECTED_DAILY_ACTUAL_VS_FORECAST_COLUMNS - set(
        actual_vs_forecast_df.columns
    )
    if missing_columns:
        raise ValueError(
            f"daily_actual_vs_forecast.csv is missing expected columns: {sorted(missing_columns)}\n"
            "Re-run snowflake/daily_avf.sql and save the output to "
            "data/input/daily_actual_vs_forecast.csv"
        )

    absent_optional = _OPTIONAL_DAILY_ACTUAL_VS_FORECAST_COLUMNS - set(
        actual_vs_forecast_df.columns
    )
    if absent_optional:
        logger.warning(
            "daily_actual_vs_forecast.csv is missing optional columns %s. "
            "gross_sales benchmarking will fail if --metric gross_sales is requested. "
            "Re-run snowflake/daily_avf.sql to add GROSS_SALES_ACTUAL.",
            sorted(absent_optional),
        )

    actual_vs_forecast_df[config.COL_ENTITY_UID] = actual_vs_forecast_df[
        config.COL_ENTITY_UID
    ].str.strip()
    actual_vs_forecast_df[config.COL_DAY_START_TIME] = actual_vs_forecast_df[
        config.COL_DAY_START_TIME
    ].dt.normalize()

    initial_row_count = len(actual_vs_forecast_df)
    actual_vs_forecast_df = actual_vs_forecast_df.drop_duplicates(
        subset=[config.COL_ENTITY_UID, config.COL_DAY_START_TIME],
        keep="first",
    ).reset_index(drop=True)
    duplicates_removed = initial_row_count - len(actual_vs_forecast_df)
    if duplicates_removed > 0:
        logger.warning(
            "Removed %d duplicate (location, date) rows from source CSV",
            duplicates_removed,
        )

    logger.info(
        "Loaded %d rows, %d locations",
        len(actual_vs_forecast_df),
        actual_vs_forecast_df[config.COL_ENTITY_UID].nunique(),
    )
    return actual_vs_forecast_df


def build_baseline_variants(
    actual_vs_forecast_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Construct the baseline projection variants from the raw extract.

    Default variants drop zero-projection rows; strict variants retain them
    so the accuracy impact of those failure cases can be measured separately.
    """
    is_manager_generated = actual_vs_forecast_df[config.COL_IS_GENERATED_BY_MANAGER]
    adjusted_draft_column = "_adjusted_draft"
    adjusted_published_column = "_adjusted_published"

    def _to_nixtla(subset: pd.DataFrame, value_column: str) -> pd.DataFrame:
        return (
            subset[[config.COL_ENTITY_UID, config.COL_DAY_START_TIME, value_column]]
            .rename(
                columns={
                    config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
                    config.COL_DAY_START_TIME: config.NIXTLA_DS,
                    value_column: config.NIXTLA_Y,
                }
            )
            .reset_index(drop=True)
        )

    actual_vs_forecast_df = actual_vs_forecast_df.copy()
    actual_vs_forecast_df[adjusted_draft_column] = (
        actual_vs_forecast_df[config.COL_PROJECTION]
        + actual_vs_forecast_df[config.COL_USER_ADJUSTMENT]
    )
    actual_vs_forecast_df[adjusted_published_column] = (
        actual_vs_forecast_df[config.COL_PROJECTION]
        + actual_vs_forecast_df[config.COL_PUBLISHED_USER_ADJUSTMENT]
    )

    if config.EXCLUDE_ZERO_PROJECTION_FROM_EVAL:
        baseline_filter = actual_vs_forecast_df[config.COL_PROJECTION] > 0
    else:
        baseline_filter = pd.Series(
            [True] * len(actual_vs_forecast_df), index=actual_vs_forecast_df.index
        )

    baselines = {
        config.BASELINE_PRODUCTION: _to_nixtla(
            actual_vs_forecast_df[baseline_filter], config.COL_PROJECTION
        ),
        config.BASELINE_ADJUSTED_DRAFT: _to_nixtla(
            actual_vs_forecast_df[baseline_filter], adjusted_draft_column
        ),
        config.BASELINE_ADJUSTED_PUBLISHED: _to_nixtla(
            actual_vs_forecast_df[baseline_filter], adjusted_published_column
        ),
        config.BASELINE_MANAGER_ONLY_DRAFT: _to_nixtla(
            actual_vs_forecast_df[baseline_filter & is_manager_generated],
            adjusted_draft_column,
        ),
        config.BASELINE_MANAGER_ONLY_PUBLISHED: _to_nixtla(
            actual_vs_forecast_df[baseline_filter & is_manager_generated],
            adjusted_published_column,
        ),
        config.BASELINE_PRODUCTION_STRICT: _to_nixtla(
            actual_vs_forecast_df, config.COL_PROJECTION
        ),
        config.BASELINE_ADJUSTED_DRAFT_STRICT: _to_nixtla(
            actual_vs_forecast_df, adjusted_draft_column
        ),
    }

    return baselines


def build_actuals(
    actual_vs_forecast_df: pd.DataFrame,
    target_column: str = config.COL_ACTUAL_VALUE,
) -> pd.DataFrame:
    """Actuals frame filtered to match ``build_baseline_variants`` row coverage.

    Filtering is by baseline projection, not by model variant, so all variants
    share one ground-truth frame. ``target_column`` selects net vs. gross sales.
    """
    subset = actual_vs_forecast_df
    if config.EXCLUDE_ZERO_PROJECTION_FROM_EVAL:
        subset = actual_vs_forecast_df[actual_vs_forecast_df[config.COL_PROJECTION] > 0]

    return (
        subset[
            [
                config.COL_ENTITY_UID,
                config.COL_DAY_START_TIME,
                target_column,
            ]
        ]
        .rename(
            columns={
                config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
                config.COL_DAY_START_TIME: config.NIXTLA_DS,
                target_column: config.NIXTLA_Y,
            }
        )
        .reset_index(drop=True)
    )


def build_actuals_strict(
    actual_vs_forecast_df: pd.DataFrame,
    target_column: str = config.COL_ACTUAL_VALUE,
) -> pd.DataFrame:
    """Actuals frame without projection filtering, for the strict baseline variants.

    The strict variants include zero-projection rows in their predictions;
    evaluating them against the filtered actuals would silently drop those
    rows on merge and collapse strict vs. standard into identical results.
    """
    return (
        actual_vs_forecast_df[
            [
                config.COL_ENTITY_UID,
                config.COL_DAY_START_TIME,
                target_column,
            ]
        ]
        .rename(
            columns={
                config.COL_ENTITY_UID: config.NIXTLA_UNIQUE_ID,
                config.COL_DAY_START_TIME: config.NIXTLA_DS,
                target_column: config.NIXTLA_Y,
            }
        )
        .reset_index(drop=True)
    )


def drop_outlier_days(nixtla_df: pd.DataFrame, actual_vs_forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Drop flagged outlier days entirely, mirroring the production system.

    Outlier days are removed rather than zeroed: a zero-sales row would read
    as a valid closed-day observation, which is a different signal than a day
    the model should not see. NULL ``IS_OUTLIER`` is treated as False.
    """
    outlier_mask = actual_vs_forecast_df[config.COL_IS_OUTLIER].fillna(False).eq(True)
    if not outlier_mask.any():
        return nixtla_df

    outlier_keys = set(
        zip(
            actual_vs_forecast_df.loc[outlier_mask, config.COL_ENTITY_UID],
            actual_vs_forecast_df.loc[outlier_mask, config.COL_DAY_START_TIME],
        )
    )

    keep_mask = ~pd.Series(
        list(
            zip(
                nixtla_df[config.NIXTLA_UNIQUE_ID],
                nixtla_df[config.NIXTLA_DS],
            )
        ),
        index=nixtla_df.index,
    ).isin(outlier_keys)

    dropped_count = (~keep_mask).sum()
    logger.info(
        "Dropped %d outlier days (%d unique location-days flagged by the production system)",
        dropped_count,
        len(outlier_keys),
    )
    return nixtla_df[keep_mask].reset_index(drop=True)


def determine_anchor_date(actual_vs_forecast_df: pd.DataFrame) -> pd.Timestamp:
    """Most recent day with both a positive projection and a non-null actual.

    Shared by the daily pipeline and the training-length ablation to keep
    test-window construction consistent.
    """
    has_positive_projection = actual_vs_forecast_df[config.COL_PROJECTION] > 0
    projected_actual_rows = actual_vs_forecast_df[
        has_positive_projection & actual_vs_forecast_df[config.COL_ACTUAL_VALUE].notna()
    ]
    if projected_actual_rows.empty:
        raise ValueError(
            "No rows with positive PROJECTION and non-null ACTUAL_VALUE found in "
            "daily_actual_vs_forecast.csv; cannot determine anchor date. Re-run "
            "snowflake/daily_avf.sql and save the output to "
            "data/input/daily_actual_vs_forecast.csv, ensuring the extract includes "
            "test-period projections and actuals."
        )
    return pd.Timestamp(projected_actual_rows[config.COL_DAY_START_TIME].max())


def summarize_actual_vs_forecast_data(
    actual_vs_forecast_df: pd.DataFrame, location_ids: list[str]
) -> dict:
    """Coverage summary (counts, date range, median days/location) for logging."""
    location_subset = actual_vs_forecast_df[
        actual_vs_forecast_df[config.COL_ENTITY_UID].isin(location_ids)
    ]
    days_per_location = location_subset.groupby(config.COL_ENTITY_UID)[
        config.COL_DAY_START_TIME
    ].nunique()
    return {
        "n_locations": len(location_ids),
        "n_rows": len(location_subset),
        "date_min": location_subset[config.COL_DAY_START_TIME].min(),
        "date_max": location_subset[config.COL_DAY_START_TIME].max(),
        "days_per_location_median": int(days_per_location.median()),
    }
