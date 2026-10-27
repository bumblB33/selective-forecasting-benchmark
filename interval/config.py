from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "input" / "interval_actual_vs_forecast.csv"
METRICS_DIR = DATA_DIR / "metrics"
REPORTS_DIR = DATA_DIR / "reports"

COL_ENTITY_UID = "ENTITY_UID"
COL_CUSTOMER_ID = "CUSTOMER_ID"
COL_SITE_ID = "SITE_ID"
COL_INTERVAL_ID_TICKS = "INTERVAL_ID_TICKS"
COL_INTERVAL_START_TS = "INTERVAL_START_TS"
COL_NET_SALES = "NET_SALES"
COL_GROSS_SALES = "GROSS_SALES"
COL_BASELINE_PROJECTION = "BASELINE_PROJECTION"
COL_USER_ADJUSTMENT = "USER_ADJUSTMENT"
COL_PUBLISHED_USER_ADJUSTMENT = "PUBLISHED_USER_ADJUSTMENT"
COL_IS_CLOSED = "IS_CLOSED"
COL_BUSINESS_START_HOUR_UTC = "BUSINESS_START_HOUR_UTC"
COL_DAILY_NET_SALES = "DAILY_NET_SALES"
COL_IS_OUTLIER = "IS_OUTLIER"
COL_OUTLIER_EXCLUSION_REASON = "OUTLIER_EXCLUSION_REASON"

NIXTLA_UNIQUE_ID = "unique_id"
NIXTLA_DS = "ds"
NIXTLA_Y = "y"

FREQUENCY = "15min"
INTERVALS_PER_DAY = 96
INTERVALS_PER_WEEK = INTERVALS_PER_DAY * 7

SEASON_LENGTH_DAILY = INTERVALS_PER_DAY
SEASON_LENGTH_WEEKLY = INTERVALS_PER_WEEK

FORECAST_HORIZON = INTERVALS_PER_WEEK
TRAIN_HISTORY_WEEKS = 26
TEST_PERIOD_WEEKS = 1
RANDOM_SEED = 42

EXCLUDE_ZERO_ACTUAL_FROM_EVAL = True
EXCLUDE_ZERO_PROJECTION_FROM_EVAL = True
CV_NUM_WINDOWS = 3
CV_STEP_SIZE = FORECAST_HORIZON

# holiday_name_encoded is omitted at 15-minute granularity: is_holiday and
# is_holiday_week already carry the timing signal, and the extra binary
# features competed with MLForecast's hour and day-of-week features without
# measurable accuracy gain.
FUTURE_EXOGENOUS_FEATURES: list[str] = [
    "is_weekend",
    "is_holiday",
    "is_month_start",
    "is_month_end",
    "days_to_nearest_holiday",
    "is_holiday_week",
]

FOURIER_DAILY_K = 3
FOURIER_WEEKLY_K = 2

BASELINE_PRODUCTION = "Baseline_Production"
BASELINE_ADJUSTED_DRAFT = "Baseline_Adjusted_Draft"
BASELINE_ADJUSTED_PUBLISHED = "Baseline_Adjusted_Published"

BASELINE_PRODUCTION_STRICT = "Baseline_Production_Strict"
BASELINE_ADJUSTED_DRAFT_STRICT = "Baseline_Adjusted_Draft_Strict"

BASELINE_MODEL = BASELINE_PRODUCTION

# AutoETS is omitted at 15-minute granularity: state-space estimation at a
# season length of 96 is too slow and rarely converges. MSTL + AutoARIMA
# covers multi-seasonality at acceptable cost.
STATSFORECAST_MODELS: dict[str, dict] = {
    "SeasonalNaive": {"season_length": SEASON_LENGTH_DAILY},
    "MSTL": {
        "season_length": [SEASON_LENGTH_DAILY, SEASON_LENGTH_WEEKLY],
        "trend_forecaster": "AutoARIMA",
    },
}


def _fifteen_minute_slot_index(dates):
    """Quarter-hour slot index (0..95). Accepts both DatetimeIndex and Series."""
    if hasattr(dates, "hour") and hasattr(dates, "minute"):
        return dates.hour * 4 + dates.minute // 15
    return dates.dt.hour * 4 + dates.dt.minute // 15


MLFORECAST_MODELS: dict[str, dict] = {
    "LightGBM_Fast": {
        "lags": [1, SEASON_LENGTH_DAILY, SEASON_LENGTH_WEEKLY],
        "date_features": ["hour", "dayofweek", _fifteen_minute_slot_index],
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_threads": -1,
    },
    "LightGBM_Standard": {
        "lags": [1, SEASON_LENGTH_DAILY, SEASON_LENGTH_WEEKLY],
        "date_features": [
            "hour",
            "dayofweek",
            "day",
            "month",
            _fifteen_minute_slot_index,
        ],
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_threads": -1,
    },
    "LightGBM_Accuracy": {
        "lags": [1, SEASON_LENGTH_DAILY, SEASON_LENGTH_WEEKLY],
        "date_features": [
            "hour",
            "dayofweek",
            "day",
            "month",
            _fifteen_minute_slot_index,
        ],
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 10,
        "num_leaves": 80,
        "num_threads": -1,
    },
}

TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M"


@dataclass(frozen=True)
class MetricConfig:
    """Per-metric wiring for the interval benchmark pipeline.

    ``has_baseline=False`` skips the baseline-comparison steps entirely.
    """

    metric_name: str
    metric_id: int
    actuals_column: str
    has_baseline: bool
    display_label: str


# Interval baseline projection in the extract always represents net sales,
# so gross sales has no comparable baseline.
METRIC_CONFIGS: dict[str, MetricConfig] = {
    "net_sales": MetricConfig(
        metric_name="net_sales",
        metric_id=1,
        actuals_column=COL_NET_SALES,
        has_baseline=True,
        display_label="Net Sales",
    ),
    "gross_sales": MetricConfig(
        metric_name="gross_sales",
        metric_id=3,
        actuals_column=COL_GROSS_SALES,
        has_baseline=False,
        display_label="Gross Sales",
    ),
}

DEFAULT_METRIC = "net_sales"
