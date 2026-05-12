from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "input" / "daily_actual_vs_forecast.csv"
METRICS_DIR = DATA_DIR / "metrics"
REPORTS_DIR = DATA_DIR / "reports"

COL_GROSS_SALES_ACTUAL = "GROSS_SALES_ACTUAL"

COL_ENTITY_UID = "ENTITY_UID"
COL_CUSTOMER_ID = "CUSTOMER_ID"
COL_SITE_ID = "SITE_ID"
COL_DAY_START_TIME = "DAY_START_TIME"
COL_PROJECTION = "PROJECTION"
COL_USER_ADJUSTMENT = "USER_ADJUSTMENT"
COL_PUBLISHED_USER_ADJUSTMENT = "PUBLISHED_USER_ADJUSTMENT"
COL_ACTUAL_VALUE = "ACTUAL_VALUE"
COL_IS_GENERATED_BY_MANAGER = "IS_GENERATED_BY_MANAGER"
COL_IS_SYSTEMATIC_GENERATED = "IS_SYSTEMATIC_GENERATED"
COL_IS_OUTLIER = "IS_OUTLIER"
COL_OUTLIER_EXCLUSION_REASON = "OUTLIER_EXCLUSION_REASON"

NIXTLA_UNIQUE_ID = "unique_id"
NIXTLA_DS = "ds"
NIXTLA_Y = "y"

FREQUENCY = "D"
FORECAST_HORIZON = 7
SEASON_LENGTH_WEEKLY = 7
SEASON_LENGTH_YEARLY = 365
TRAIN_HISTORY_WEEKS = 105
TEST_PERIOD_WEEKS = 1
RANDOM_SEED = 42
TARGET_LOCATION_COUNT = 300
MIN_DAYS_WITH_ACTUALS = 90

CLIP_OUTLIER_PERCENTILE = 99.0
EXCLUDE_ZERO_ACTUAL_FROM_EVAL = True
EXCLUDE_ZERO_PROJECTION_FROM_EVAL = True
CV_NUM_WINDOWS = 3
CV_STEP_SIZE = FORECAST_HORIZON

FUTURE_EXOGENOUS_FEATURES: list[str] = [
    "is_weekend",
    "is_holiday",
    "holiday_name_encoded",
    "is_month_start",
    "is_month_end",
    "days_to_nearest_holiday",
    "is_holiday_week",
]

FOURIER_WEEKLY_K = 3
FOURIER_YEARLY_K = 3

Baseline_PRODUCTION = "Baseline_Production"
Baseline_ADJUSTED_DRAFT = "Baseline_Adjusted_Draft"
Baseline_ADJUSTED_PUBLISHED = "Baseline_Adjusted_Published"
Baseline_MANAGER_ONLY_DRAFT = "Baseline_ManagerOnly_Draft"
Baseline_MANAGER_ONLY_PUBLISHED = "Baseline_ManagerOnly_Published"

Baseline_PRODUCTION_STRICT = "Baseline_Production_Strict"
Baseline_ADJUSTED_DRAFT_STRICT = "Baseline_Adjusted_Draft_Strict"

BASELINE_MODEL = Baseline_PRODUCTION

STATSFORECAST_MODELS: dict[str, dict] = {
    "SeasonalNaive_D": {"season_length": SEASON_LENGTH_WEEKLY},
    "MSTL_D": {
        "season_length": [SEASON_LENGTH_WEEKLY, SEASON_LENGTH_YEARLY],
        "trend_forecaster": "AutoARIMA",
    },
    "AutoETS_D": {"season_length": SEASON_LENGTH_WEEKLY, "model": "ZZZ"},
    "AutoARIMA_D": {"season_length": SEASON_LENGTH_WEEKLY},
    "DOT_D": {"season_length": SEASON_LENGTH_WEEKLY},
}

MLFORECAST_MODELS: dict[str, dict] = {
    "LightGBM_D_Fast": {
        "lags": [1, 7, 14, 21, 28],
        "date_features": ["dayofweek", "day", "month"],
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_threads": -1,
    },
    "LightGBM_D_Standard": {
        "lags": [1, 7, 14, 21, 28],
        "date_features": ["dayofweek", "day", "month", "quarter"],
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_threads": -1,
    },
    "LightGBM_D_Accuracy": {
        "lags": [1, 7, 14, 21, 28, 56],
        "date_features": ["dayofweek", "day", "month", "quarter"],
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 10,
        "num_leaves": 80,
        "num_threads": -1,
    },
}

METRICS: list[str] = ["MAE", "WAPE", "MAPE", "RMSE"]
STATISTICAL_TEST_CONFIDENCE = 0.95

# Capacity-planning estimate, not enforced. MSTL with an AutoARIMA trend
# can be slow when the selected ARIMA order is complex.
ESTIMATED_MAX_TRAIN_TIME_SECONDS = 600

TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M"


@dataclass(frozen=True)
class MetricConfig:
    """Per-metric wiring for the daily benchmark pipeline.

    ``has_baseline=False`` skips the baseline-comparison steps entirely.
    """

    metric_name: str
    metric_id: int
    actuals_column: str
    has_baseline: bool
    display_label: str


METRIC_CONFIGS: dict[str, MetricConfig] = {
    "net_sales": MetricConfig(
        metric_name="net_sales",
        metric_id=1,
        actuals_column=COL_ACTUAL_VALUE,
        has_baseline=True,
        display_label="Net Sales",
    ),
    "gross_sales": MetricConfig(
        metric_name="gross_sales",
        metric_id=3,
        actuals_column=COL_GROSS_SALES_ACTUAL,
        has_baseline=False,
        display_label="Gross Sales",
    ),
}

DEFAULT_METRIC = "net_sales"
