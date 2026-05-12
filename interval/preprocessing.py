from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
except ImportError:
    holidays_lib = None

from core.preprocessing import create_train_test_split, fill_missing_timesteps  # noqa: F401
from . import config

logger = logging.getLogger(__name__)


def clip_outliers(
    nixtla_df: pd.DataFrame,
    percentile: float = config.CLIP_OUTLIER_PERCENTILE,
    caps: dict[str, float] | None = None,
    vsd_hours: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Percentile-based outlier handling for ad hoc use; the main pipeline drops
    flagged days at load time instead.

    Zeros every interval in any business day whose total exceeds the per-location
    cap; ``vsd_hours`` keeps the boundaries aligned with business days. Pass
    ``caps`` from training to apply without leakage.
    """
    result = nixtla_df.copy()
    result[config.NIXTLA_Y] = result[config.NIXTLA_Y].clip(lower=0.0)

    if vsd_hours:
        uid_vsd = result[config.NIXTLA_UNIQUE_ID].map(vsd_hours).fillna(0).astype(int)
        cal_dates = result[config.NIXTLA_DS].dt.normalize()
        before_vsd = (uid_vsd > 0) & (result[config.NIXTLA_DS].dt.hour < uid_vsd)
        biz_day_col = cal_dates.where(~before_vsd, cal_dates - pd.Timedelta(days=1))
    else:
        biz_day_col = result[config.NIXTLA_DS].dt.normalize()

    if caps is None:
        daily_totals = result.groupby([config.NIXTLA_UNIQUE_ID, biz_day_col])[
            config.NIXTLA_Y
        ].transform("sum")
        caps = (
            result.assign(_daily_total=daily_totals)
            .groupby(config.NIXTLA_UNIQUE_ID)["_daily_total"]
            .quantile(percentile / 100.0)
            .to_dict()
        )

    result["_biz_day"] = biz_day_col

    def _zero_outlier_day_intervals(group: pd.DataFrame) -> pd.DataFrame:
        cap = caps.get(group[config.NIXTLA_UNIQUE_ID].iloc[0])
        if cap is None or cap <= 0:
            return group
        group = group.copy()
        day_totals = group.groupby("_biz_day")[config.NIXTLA_Y].transform("sum")
        group.loc[day_totals > cap, config.NIXTLA_Y] = 0.0
        return group

    result = result.groupby(config.NIXTLA_UNIQUE_ID, group_keys=False).apply(
        _zero_outlier_day_intervals
    )
    return result.drop(columns=["_biz_day"]).reset_index(drop=True), caps


def fill_missing_intervals(nixtla_df: pd.DataFrame) -> pd.DataFrame:
    """15-minute zero-fill with dedup, for extracts that may have duplicate rows."""
    return fill_missing_timesteps(nixtla_df, freq=config.FREQUENCY, deduplicate=True)


def add_calendar_features(nixtla_df: pd.DataFrame) -> pd.DataFrame:
    """Add the interval calendar features declared in ``FUTURE_EXOGENOUS_FEATURES``."""
    result = nixtla_df.copy()
    dates = pd.DatetimeIndex(result[config.NIXTLA_DS])

    result["is_weekend"] = dates.dayofweek.isin([5, 6]).astype(int)
    result["is_month_start"] = dates.is_month_start.astype(int)
    result["is_month_end"] = dates.is_month_end.astype(int)

    if holidays_lib is not None:
        us_holidays = holidays_lib.US(years=dates.year.unique().tolist())
        holiday_dates = set(us_holidays.keys())
        result["is_holiday"] = dates.normalize().isin(holiday_dates).astype(int)

        holiday_timestamps = pd.DatetimeIndex(
            sorted(pd.Timestamp(d) for d in holiday_dates)
        )
        result["days_to_nearest_holiday"] = dates.normalize().map(
            lambda d: (
                int(min(abs((d - h).days) for h in holiday_timestamps))
                if len(holiday_timestamps) > 0
                else 0
            )
        )
        result["is_holiday_week"] = (result["days_to_nearest_holiday"] <= 3).astype(int)
    else:
        logger.warning("holidays package not installed; holiday features set to 0")
        for feature in ["is_holiday", "days_to_nearest_holiday", "is_holiday_week"]:
            result[feature] = 0

    return result


def fourier_terms(
    dates: pd.DatetimeIndex,
    period_intervals: int,
    n_harmonics: int,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Fourier terms for the daily (96-slot) or weekly (672-slot) cycle."""
    if period_intervals == config.SEASON_LENGTH_DAILY:
        seasonal_position = (dates.hour * 4 + dates.minute // 15).astype(float)
    else:
        seasonal_position = (
            dates.dayofweek * config.INTERVALS_PER_DAY
            + dates.hour * 4
            + dates.minute // 15
        ).astype(float)
    term_columns = {}
    for harmonic_number in range(1, n_harmonics + 1):
        term_columns[f"{prefix}_sin_{harmonic_number}"] = np.sin(
            2 * np.pi * harmonic_number * seasonal_position / period_intervals
        )
        term_columns[f"{prefix}_cos_{harmonic_number}"] = np.cos(
            2 * np.pi * harmonic_number * seasonal_position / period_intervals
        )
    return term_columns


def add_fourier_features(
    training_df: pd.DataFrame,
    daily_k: int = config.FOURIER_DAILY_K,
    weekly_k: int = config.FOURIER_WEEKLY_K,
) -> tuple[pd.DataFrame, list[str]]:
    """Add daily and weekly Fourier features to training.

    Future-period terms are recomputed at predict time to avoid misalignment
    when series have different end timestamps.
    """
    training_dates = pd.DatetimeIndex(training_df[config.NIXTLA_DS])

    daily_train = fourier_terms(
        training_dates, config.SEASON_LENGTH_DAILY, daily_k, "fourier_daily"
    )
    weekly_train = fourier_terms(
        training_dates, config.SEASON_LENGTH_WEEKLY, weekly_k, "fourier_weekly"
    )

    fourier_columns = list(daily_train.keys()) + list(weekly_train.keys())

    result = training_df.copy().reset_index(drop=True)
    for column, values in {**daily_train, **weekly_train}.items():
        result[column] = values

    return result, fourier_columns
