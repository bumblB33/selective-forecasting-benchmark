from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
except ImportError:
    holidays_lib = None

logger = logging.getLogger(__name__)

_NIXTLA_UNIQUE_ID = "unique_id"
_NIXTLA_DS = "ds"
_NIXTLA_Y = "y"


def fill_missing_timesteps(
    nixtla_df: pd.DataFrame,
    freq: str,
    deduplicate: bool = False,
) -> pd.DataFrame:
    """Materialize missing timestamps as zero-sales rows so models see contiguous series.

    Zero-fill (rather than NaN) signals closed or non-operating periods without
    creating gaps. Set ``deduplicate=True`` when the source may contain duplicate
    timestamps for the same location.
    """
    completed_series_parts = []
    for unique_id, location_history in nixtla_df.groupby(_NIXTLA_UNIQUE_ID):
        if deduplicate:
            location_history = location_history.drop_duplicates(
                subset=[_NIXTLA_DS], keep="first"
            )
        full_range = pd.date_range(
            start=location_history[_NIXTLA_DS].min(),
            end=location_history[_NIXTLA_DS].max(),
            freq=freq,
        )
        if len(full_range) == len(location_history):
            completed_series_parts.append(location_history)
            continue
        reindexed = (
            location_history.set_index(_NIXTLA_DS)
            .reindex(full_range)
            .reset_index()
            .rename(columns={"index": _NIXTLA_DS})
        )
        reindexed[_NIXTLA_UNIQUE_ID] = unique_id
        reindexed[_NIXTLA_Y] = reindexed[_NIXTLA_Y].fillna(0.0)
        completed_series_parts.append(reindexed)
    return pd.concat(completed_series_parts, ignore_index=True)


def create_train_test_split(
    nixtla_df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    max_train_history_weeks: int = 105,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test with a bounded training history to keep cost predictable."""
    training_window_start = test_start - pd.Timedelta(weeks=max_train_history_weeks)

    training_df = nixtla_df[
        (nixtla_df[_NIXTLA_DS] >= training_window_start)
        & (nixtla_df[_NIXTLA_DS] < test_start)
    ].copy()

    test_df = nixtla_df[
        (nixtla_df[_NIXTLA_DS] >= test_start) & (nixtla_df[_NIXTLA_DS] <= test_end)
    ].copy()

    logger.info(
        "Train: %d rows (%d locations) | Test: %d rows (%d locations)",
        len(training_df),
        training_df[_NIXTLA_UNIQUE_ID].nunique(),
        len(test_df),
        test_df[_NIXTLA_UNIQUE_ID].nunique(),
    )
    return training_df, test_df


def clip_outliers(
    nixtla_df: pd.DataFrame,
    percentile: float = 99.0,
    caps: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Percentile-based outlier handling for ad hoc use; the main pipeline drops
    flagged days at load time instead.

    Zeroes negatives, then zeroes (does not clip) days exceeding each location's
    cap. Pass ``caps`` from training to apply without leakage. Daily-granularity
    only; the interval pipeline has its own implementation that caps on daily totals.
    """
    result = nixtla_df.copy()
    result[_NIXTLA_Y] = result[_NIXTLA_Y].clip(lower=0.0)

    if caps is None:
        caps = (
            result.groupby(_NIXTLA_UNIQUE_ID)[_NIXTLA_Y]
            .quantile(percentile / 100.0)
            .to_dict()
        )

    def _zero_outlier_days(group: pd.DataFrame) -> pd.DataFrame:
        cap = caps.get(group[_NIXTLA_UNIQUE_ID].iloc[0])
        if cap is not None and cap > 0:
            group = group.copy()
            group.loc[group[_NIXTLA_Y] > cap, _NIXTLA_Y] = 0.0
        return group

    result = result.groupby(_NIXTLA_UNIQUE_ID, group_keys=False).apply(
        _zero_outlier_days
    )
    return result.reset_index(drop=True), caps


def add_calendar_features(nixtla_df: pd.DataFrame) -> pd.DataFrame:
    """Add weekend, month boundary, and US holiday features for the daily pipeline."""
    result = nixtla_df.copy()
    dates = pd.DatetimeIndex(result[_NIXTLA_DS])

    result["is_weekend"] = dates.dayofweek.isin([5, 6]).astype(int)
    result["is_month_start"] = dates.is_month_start.astype(int)
    result["is_month_end"] = dates.is_month_end.astype(int)

    if holidays_lib is not None:
        us_holidays = holidays_lib.US(years=dates.year.unique().tolist())
        holiday_dates = set(us_holidays.keys())
        result["is_holiday"] = dates.normalize().isin(holiday_dates).astype(int)

        holiday_names = {date: name for date, name in us_holidays.items()}
        unique_names = sorted(set(holiday_names.values()))
        name_to_code = {name: idx + 1 for idx, name in enumerate(unique_names)}
        result["holiday_name_encoded"] = (
            dates.normalize()
            .map(lambda d: name_to_code.get(holiday_names.get(d.date(), ""), 0))
            .astype(int)
        )

        holiday_timestamps = pd.DatetimeIndex(
            sorted(pd.Timestamp(d) for d in holiday_dates)
        )
        result["days_to_nearest_holiday"] = dates.normalize().map(
            lambda d: int(
                min(abs((d - h).days) for h in holiday_timestamps)
                if len(holiday_timestamps) > 0
                else 0
            )
        )
        result["is_holiday_week"] = (result["days_to_nearest_holiday"] <= 3).astype(int)
    else:
        logger.warning("holidays package not installed; holiday features set to 0")
        for feature in [
            "is_holiday",
            "holiday_name_encoded",
            "days_to_nearest_holiday",
            "is_holiday_week",
        ]:
            result[feature] = 0

    return result


def add_fourier_features(
    training_df: pd.DataFrame,
    weekly_k: int = 3,
    yearly_k: int = 3,
    forecast_horizon: int = 7,
    frequency: str = "D",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Weekly and annual Fourier terms for training and future horizon, derived from
    the same deterministic formula to prevent leakage. Daily pipeline only.

    Requires all series in ``training_df`` to share the same ``max(ds)`` — fill gaps first.
    """
    per_series_last_dates = training_df.groupby(_NIXTLA_UNIQUE_ID)[_NIXTLA_DS].max()
    assert per_series_last_dates.nunique() == 1, (
        f"add_fourier_features() requires all series to share the same max(ds). "
        f"Found {per_series_last_dates.nunique()} distinct end dates. "
        f"Call fill_missing_dates() (daily pipeline) or fill_missing_intervals() "
        f"(interval pipeline) before this function to ensure uniform endpoints."
    )

    def _fourier_terms(
        dates: pd.DatetimeIndex, period: float, n_harmonics: int, prefix: str
    ) -> pd.DataFrame:
        # Weekly cycle is anchored to Monday via day-of-week; yearly uses
        # fractional day-of-year. Any per-location phase offset is small relative
        # to the 7-day period and absorbed by the LightGBM lag features.
        seasonal_position = (
            (dates.dayofyear + dates.hour / 24.0)
            if period > 7
            else dates.dayofweek.astype(float)
        )
        term_columns = {}
        for harmonic_number in range(1, n_harmonics + 1):
            term_columns[f"{prefix}_sin_{harmonic_number}"] = np.sin(
                2 * np.pi * harmonic_number * seasonal_position / period
            )
            term_columns[f"{prefix}_cos_{harmonic_number}"] = np.cos(
                2 * np.pi * harmonic_number * seasonal_position / period
            )
        return pd.DataFrame(term_columns, index=dates)

    training_dates = pd.DatetimeIndex(training_df[_NIXTLA_DS])

    last_train_date = training_df[_NIXTLA_DS].max()
    future_dates = pd.date_range(
        start=last_train_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=frequency,
    )

    weekly_train = _fourier_terms(
        training_dates, period=7.0, n_harmonics=weekly_k, prefix="fourier_weekly"
    )
    yearly_train = _fourier_terms(
        training_dates, period=365.25, n_harmonics=yearly_k, prefix="fourier_yearly"
    )
    weekly_future = _fourier_terms(
        future_dates, period=7.0, n_harmonics=weekly_k, prefix="fourier_weekly"
    )
    yearly_future = _fourier_terms(
        future_dates, period=365.25, n_harmonics=yearly_k, prefix="fourier_yearly"
    )

    fourier_columns = list(weekly_train.columns) + list(yearly_train.columns)

    result = training_df.copy().reset_index(drop=True)
    fourier_train_df = pd.concat([weekly_train, yearly_train], axis=1).reset_index(
        drop=True
    )
    for column in fourier_columns:
        result[column] = fourier_train_df[column].values

    unique_ids = training_df[_NIXTLA_UNIQUE_ID].unique()
    future_combined = pd.concat([weekly_future, yearly_future], axis=1)
    future_base = pd.DataFrame({_NIXTLA_DS: future_dates.values})
    for column in fourier_columns:
        future_base[column] = future_combined[column].values
    future_fourier_df = pd.concat(
        [future_base.assign(**{_NIXTLA_UNIQUE_ID: uid}) for uid in unique_ids],
        ignore_index=True,
    )[[_NIXTLA_UNIQUE_ID, _NIXTLA_DS] + fourier_columns]

    logger.info(
        "Added %d Fourier features (%d weekly, %d yearly harmonics)",
        len(fourier_columns),
        weekly_k,
        yearly_k,
    )
    return result, future_fourier_df, fourier_columns
