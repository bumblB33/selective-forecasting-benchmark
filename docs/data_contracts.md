# Data Contracts

This document describes the internal function signatures and DataFrame schemas
used across the two benchmark pipelines. Both pipelines share identical
conventions; differences are noted explicitly.

---

## Table of Contents

- [Nixtla DataFrame Schema](#nixtla-dataframe-schema)
- [Data Loader](#data-loader)
- [Preprocessing](#preprocessing)
- [Models](#models)
- [Evaluation](#evaluation)
- [Reporting](#reporting)

---

## Nixtla DataFrame Schema

All DataFrames that flow between modules use the canonical Nixtla format:

| Column | Type | Description |
|---|---|---|
| `unique_id` | str | Composite location identifier: `{CUSTOMER_ID}_{SITE_ID}` |
| `ds` | datetime64[ns] | Timestamp of the observation |
| `y` | float64 | Target value (net sales in dollars) |

Column name constants are defined in each pipeline's `config.py`:

```python
NIXTLA_UNIQUE_ID = "unique_id"
NIXTLA_DS        = "ds"
NIXTLA_Y         = "y"
```

## Data Loader

### `load_daily_actual_vs_forecast(path=None) -> pd.DataFrame`
### `load_interval_actual_vs_forecast(path=None) -> pd.DataFrame`

Loads the Snowflake CSV export and applies type coercion. Returns a wide
DataFrame with all source columns preserved. Raises `ValueError` if required
columns are absent.

- Daily: uses `ACTUAL_VALUE` as the target column. It does not fill missing values on load.
- Interval: uses `NET_SALES` as the target column. It fills missing `NET_SALES` and `GROSS_SALES` values with `0.0`.

### `determine_test_window(interval_df) -> tuple[pd.Timestamp, pd.Timestamp]`

Interval pipeline only. The window ends at the last interval with a non-null
`BASELINE_PROJECTION` and spans `TEST_PERIOD_WEEKS`. Returns `(test_start, test_end)`
timestamps that are used by `create_train_test_split`.

### `create_train_test_split(nixtla_df, test_start, test_end, max_train_history_weeks) -> tuple[pd.DataFrame, pd.DataFrame]`

Available in both pipelines. It splits on explicit timestamp boundaries:
rows before `test_start` form the training data (capped at `max_train_history_weeks`),
and rows from `test_start` through `test_end` form the test set. In the daily pipeline,
`test_start` is derived from an anchor date computed in `run.py`; in the interval
pipeline, it comes from `determine_test_window`.

### `build_baseline_variants(actual_vs_forecast_df) -> dict[str, pd.DataFrame]` (daily)
### `build_baseline_variants(test_df) -> dict[str, pd.DataFrame]` (interval)

Returns a dict of `{model_name: nixtla_df}` for all baseline variants.
Each value is a Nixtla-format DataFrame (`unique_id`, `ds`, `y`).

- Daily: accepts the full ACTUAL_VS_FORECAST DataFrame (`actual_vs_forecast_df`), meaning all rows, not just the test split.
- Interval: accepts the test-split DataFrame (`test_df`).

### `build_actuals(test_df) -> pd.DataFrame`

Returns Nixtla-format actuals from the test set.

- Daily: reads `COL_ACTUAL_VALUE`.
- Interval: reads `COL_NET_SALES`.

## Preprocessing

All preprocessing functions accept and return Nixtla-format DataFrames,
or augmented versions with additional feature columns.

### `fill_missing_dates(df) -> pd.DataFrame`
### `fill_missing_intervals(df) -> pd.DataFrame`

Fills gaps in the time series with `y = 0.0` rows so that downstream models
receive a regular grid.

### `add_calendar_features(df) -> pd.DataFrame`

Adds the following columns (all of which match `config.FUTURE_EXOGENOUS_FEATURES`):
`is_weekend`, `is_holiday`, `is_month_start`, `is_month_end`,
`days_to_nearest_holiday`, and `is_holiday_week`.

The daily pipeline also adds `holiday_name_encoded`; the interval pipeline
does not.

### `add_fourier_features(df) -> ...`

The return signatures differ by pipeline:

- Daily: returns `(train_df, future_df, fourier_columns)`, which are the training rows, future-horizon rows, and the column name list. Fourier column prefixes are `fourier_weekly_*` and `fourier_yearly_*`.
- Interval: returns `(result, fourier_columns)`. No separate `future_df` is returned. Fourier column prefixes are `fourier_daily_*` and `fourier_weekly_*`.

Training and future Fourier values are computed from the same formula, so
there is no leakage between training and forecasting.

## Models

### `run_statsforecast(training_df) -> dict[str, pd.DataFrame]`

Fits all StatsForecast models defined in `config.STATSFORECAST_MODELS` and
returns `{model_name: nixtla_df}`.

### `run_mlforecast(training_df, future_fourier_df, fourier_columns) -> dict[str, pd.DataFrame]`

Fits all MLForecast/LightGBM models defined in `config.MLFORECAST_MODELS`.
`future_fourier_df` provides exogenous features for the forecast horizon.

### `run_statsforecast_cv(training_df) -> pd.DataFrame`
### `run_mlforecast_cv(training_df, fourier_columns) -> pd.DataFrame`

Cross-validation variants. Each returns a long-format DataFrame with columns:
`unique_id`, `ds`, `cutoff`, `y`, and `{model_name}`.

## Evaluation

### `compute_metrics(actuals, predictions) -> pd.DataFrame`

Returns per-location metrics: `unique_id`, `MAE`, `WAPE`, `MAPE`, `RMSE`, and `n_rows`.

### `aggregate_metrics(location_metrics) -> pd.DataFrame`

Returns a DataFrame with columns `metric`, `mean`, `median`, `std`, `p25`, and `p75` (and potentially more) across all locations.

### `compute_win_loss(all_location_metrics) -> pd.DataFrame`

Compares each model to the baseline (`config.BASELINE_MODEL`) by per-location
MAE. Returns columns `model`, `wins`, `losses`, `ties`, `n_locations`, and `win_rate`.

### `compute_statistical_tests(all_location_metrics) -> pd.DataFrame`

Runs a Wilcoxon signed-rank test for each model versus the baseline. Returns
`model`, `statistic`, `p_value`, `significant` (a boolean, true when p < 0.05),
`direction` (either `"better"` or `"worse"`), and `n_locations`. Requires at
least 10 shared locations between the candidate and the baseline; models with
fewer are omitted.

### `save_metrics(all_location_metrics, aggregate_metrics_df, win_loss_df, ..., output_dir, file_prefix) -> None`

Persists all metric DataFrames and the raw actuals and predictions to timestamped CSVs.

### `build_volume_tiers(actuals) -> pd.DataFrame`

Assigns Q1 through Q4 volume tiers based on mean per-location sales. Returns a
DataFrame with columns `unique_id` and `volume_tier`. Lives in `core/evaluation.py`
and is shared by both pipelines.

## Reporting

### `build_report(context, all_location_metrics, aggregate_metrics_dict, win_loss_df, statistical_tests_df, cv_metrics_df, actuals, predictions_dict, sample_location_ids, output_path) -> str`

Assembles and returns the complete HTML report. If `output_path` is provided,
writes the file to disk.

### `ReportContext` (dataclass)

| Field | Type | Description |
|---|---|---|
| `interval` | str | `"daily"` or `"15min"` |
| `n_locations` | int | Number of evaluated locations |
| `test_start` | pd.Timestamp | Start of evaluation window |
| `test_end` | pd.Timestamp | End of evaluation window |
| `baseline_model` | str | Name of the baseline model |
| `train_history_weeks` | int or None | Weeks of training history used |
| `cv_num_windows` | int or None | Number of CV windows (None if CV was skipped) |
| `n_train_rows` | int or None | Total training rows after preprocessing |
| `location_metadata` | pd.DataFrame or None | `unique_id` and `volume_tier` |
| `extra_notes` | list[str] | Optional free-text notes appended to the report |

### `kpi_card(label, value, subtitle="") -> str`

Renders a single KPI HTML card with ARIA `role="region"` and `aria-label` attributes.

### `load_css() -> str`

Returns a `<style>` block. Loads `reporting/report.css` from disk when
available; falls back to the inline `_CSS` string.

---
