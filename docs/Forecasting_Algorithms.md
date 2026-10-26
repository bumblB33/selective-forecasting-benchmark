# Forecasting Algorithms

This document provides an overview of all models evaluated in the benchmark, their
configuration, and their expected behavior on multi-location sales data.

---

## Table of Contents

1. [Production Forecasting System (Baseline)](#production-forecasting-system-baseline)
2. [Benchmark Models (StatsForecast)](#statsforecast-models)
3. [Benchmark Models (MLForecast and LightGBM)](#mlforecast-models-lightgbm)
4. [Baseline Variants in the Benchmark](#baseline-baselines)
5. [Preprocessing Details](#preprocessing-details)
6. [Cross-Validation](#cross-validation)
7. [Relationship Between Benchmark and Production](#relationship-between-benchmark-and-production)

---

## Production Forecasting System (Baseline)

The production system that generates the forecasts used as the benchmark baseline is a
proprietary service that is not included in this repository. Its outputs are captured in
the `PROJECTION` column of the Snowflake extract and become the `Baseline_Production`
comparison series in all benchmark runs.

### What the Baseline System Does

The production system generates per-interval (15-minute granularity) sales projections
for each location. It uses algorithmic models that combine year-over-year patterns with
rolling averages over recent weeks. It stores historical data as precomputed 15-minute
time series, and it applies configurable per-location model selection at forecast time.

Key behaviors that affect the benchmark:

- Outlier exclusion: days flagged by the system's rule engine are excluded from its
  training data. The benchmark replicates this by reading `IS_OUTLIER` from the SQL
  extract and dropping the same days before training.
- Zero-projection rows: the system may produce a $0 projection for days it treats
  as closures or holidays. The default `Baseline_Production` comparison excludes these
  rows from the accuracy calculation. The `Baseline_Production_Strict` variant includes
  them, which results in higher reported error. This reflects a wider evaluation scope,
  not a weaker forecast.
- Manager adjustments: `Baseline_Adjusted_Draft` and `Baseline_Adjusted_Published`
  add manager-authored overrides on top of the algorithmic baseline.

Source: an internal proprietary system. The source is not included in this repository.

---

## StatsForecast Models

StatsForecast models are fitted independently per location using
`statsforecast.StatsForecast`. All models use the Nixtla format
(`unique_id`, `ds`, `y`) with no exogenous regressors.

### Daily Pipeline (`season_length = 7`)

| Model ID | Class | Description |
|---|---|---|
| `SeasonalNaive_D` | `SeasonalNaive` | Repeats the same weekday value from the prior week. A strong baseline for stable weekly patterns. |
| `MSTL_D` | `MSTL` | Multi-seasonal trend decomposition with weekly (7) and yearly (365) seasonality. Uses `AutoARIMA` as the trend forecaster. |
| `AutoETS_D` | `AutoETS` | Selects the best ETS error, trend, and seasonality combination via the Akaike Information Criterion. |
| `AutoARIMA_D` | `AutoARIMA` | Automatically selects ARIMA orders with weekly seasonality, using a stepwise search. |
| `DOT_D` | `DynamicOptimizedTheta` | Ensemble of two Theta lines with optimized damping. Balances short- and long-term trends. |

### Interval Pipeline (`season_length_daily = 96`, `season_length_weekly = 672`)

| Model ID | Class | Description |
|---|---|---|
| `SeasonalNaive` | `SeasonalNaive` | Repeats the corresponding 15-minute slot from the prior day (`season_length=96`). |
| `MSTL` | `MSTL` | Multi-seasonal model with both daily (96) and weekly (672) seasonality components. |

---

## MLForecast Models (LightGBM)

LightGBM models are fitted using `mlforecast.MLForecast` with a shared
feature-engineering pipeline. Three variants vary in tree depth and
learning rate, trading accuracy against latency.

### Feature Set

- Lag features: values at lags [1, 7, 14, 21, 28] for the daily pipeline (the Accuracy variant also adds lag 56), or [1, 96, 672] for the interval pipeline.
- Lag transforms (daily): `ExpandingMean` at lag 1; `SeasonalRollingMean(season_length=7, window_size=8)` at lag 7, which is the average of the same weekday across the prior 8 weeks; `RollingMean(window_size=4)` at lag 28, for `LightGBM_D_Accuracy` only. There is no standard-deviation transform.
- Calendar features: `day_of_week`, `month`, `is_weekend`, `is_holiday`, plus any additional exogenous features defined in `FUTURE_EXOGENOUS_FEATURES` in each `config.py`.
- Fourier features: sine and cosine pairs that capture weekly (period 7) and yearly (period 365.25) cycles for the daily pipeline; daily (period 96 intervals) and weekly (period 672 intervals) cycles for the interval pipeline.

### Daily Pipeline

| Model ID | `max_depth` | `num_leaves` | `learning_rate` | `n_estimators` | Trade-off |
|---|---|---|---|---|---|
| `LightGBM_D_Fast` | 6 | default | 0.05 | 200 | Lowest latency |
| `LightGBM_D_Standard` | 8 | default | 0.05 | 400 | Balanced |
| `LightGBM_D_Accuracy` | 10 | 80 | 0.03 | 500 | Highest accuracy |

### Interval Pipeline

| Model ID | `max_depth` | `num_leaves` | `learning_rate` | `n_estimators` | Trade-off |
|---|---|---|---|---|---|
| `LightGBM_Fast` | 6 | default | 0.05 | 200 | Lowest latency |
| `LightGBM_Standard` | 8 | default | 0.05 | 400 | Balanced |
| `LightGBM_Accuracy` | 10 | 80 | 0.03 | 300 | Highest accuracy |

---

## Baseline Baselines

Baseline variants are derived directly from the Snowflake ACTUAL_VS_FORECAST data and require
no training. They represent the production system's forecasts at evaluation time.

| Variant | Source | Zero-projection rows |
|---|---|---|
| `Baseline_Production` | Raw `Baseline_PROJECTION` (or `PROJECTION`) column | Excluded |
| `Baseline_Adjusted_Draft` | Projection plus `USER_ADJUSTMENT` (manager draft) | Excluded |
| `Baseline_Adjusted_Published` | Projection plus `PUBLISHED_USER_ADJUSTMENT` | Excluded |
| `Baseline_ManagerOnly_Draft` | Draft-adjusted on `IS_GENERATED_BY_MANAGER=TRUE` rows | Excluded |
| `Baseline_ManagerOnly_Published` | Published-adjusted on `IS_GENERATED_BY_MANAGER=TRUE` rows | Excluded |
| `Baseline_Production_Strict` | Raw `Baseline_PROJECTION` column | Included |
| `Baseline_Adjusted_Draft_Strict` | Projection plus draft adjustment | Included |

The `ManagerOnly` variants are only available in the daily pipeline.
The "strict" variants include zero-projection rows, which penalize the baseline for forecasting zero when actuals are positive.

The `PROJECTION` column in ACTUAL_VS_FORECAST data is the output of whichever algorithmic
model was configured for that location. See
[Production Forecasting System (Baseline)](#production-forecasting-system-baseline) for
an overview of how the production system selects and applies models.

---

## Preprocessing Details

### Outlier Day Exclusion

Days flagged `IS_OUTLIER` in the extract are dropped from the training data by
`drop_outlier_days`, which mirrors the baseline's day-exclusion behavior. The rows are
removed, not zeroed, because a zero row would read as a closed day. A NULL flag is
treated as not an outlier. The daily pipeline drops the flagged `(location, day)` rows.
The interval pipeline drops every 15-minute interval in a flagged business day, using
the same business-day boundaries as the fallback interval fill. No statistical
threshold is applied.

### Missing Interval Filling

Any gaps in the time grid are filled with `y = 0.0` before model fitting.
For the interval pipeline, `fill_missing_intervals` uses the frequency
`15min`; for the daily pipeline, `fill_missing_dates` uses frequency `D`.

The production system handles missing interval data analogously, distributing daily
totals across 15-minute slots using a sales-share distribution derived from recent
historical records for each location.

### Fourier Features

Fourier features encode cyclical patterns as sine and cosine pairs. The number of
harmonics per cycle is set by `FOURIER_WEEKLY_K` and `FOURIER_YEARLY_K` in
`daily/config.py`, and by `FOURIER_DAILY_K` and `FOURIER_WEEKLY_K` in
`interval/config.py`. There is no single `FOURIER_ORDER` variable. Training Fourier
values are computed deterministically; future values are computed from the same formula
without using evaluation data.

---

## Cross-Validation

Both pipelines use expanding-window cross-validation via the native
`.cross_validation()` method on both `StatsForecast` and `MLForecast` objects.

The default configuration is 3 windows, each covering one forecast horizon (7 days or 672 intervals).
Results are aggregated to per-model MAE and WAPE across all windows and locations.

---

## Relationship Between Benchmark and Production

The following table maps benchmark models to their closest production equivalents.

| Benchmark model | Closest production model | Notes |
|---|---|---|
| `Baseline_Production` | Year-over-year and rolling-average models | A direct read of the `PROJECTION` column. |
| `SeasonalNaive_D` | Prior same-weekday copy model | Both copy the prior same-weekday value. |
| `AutoETS_D` and `AutoARIMA_D` | No direct equivalent | Production has no statistical model fitting. |
| `LightGBM_D_*` | No direct equivalent | Production has no machine-learning tree-based model. |
| `MSTL_D` | No direct equivalent | Production has no STL decomposition. |

Key structural differences between the benchmark and the production system:

- Granularity of training: the benchmark trains on daily aggregates (`daily/`) or on
  raw 15-minute intervals (`interval/`). The production system always operates at
  15-minute granularity internally; daily totals are derived by summing intervals.
- Outlier handling: the benchmark drops training days the production system
  flagged as `IS_OUTLIER`, mirroring production.
- Manager adjustments: the benchmark evaluates `USER_ADJUSTMENT` and
  `PUBLISHED_USER_ADJUSTMENT` as separate baseline variants (`Baseline_Adjusted_*`).
  The production system stores these as manager-authored edits on top of the algorithmic
  model output.
- The `IS_GENERATED_BY_MANAGER` flag: in ACTUAL_VS_FORECAST data, rows where
  `IS_GENERATED_BY_MANAGER=TRUE` indicate that the forecast was overridden by a manager
  and the projection reflects a user-entered value rather than an algorithm output.
  The `Baseline_ManagerOnly_*` benchmark variants isolate these rows.
- Model selection per location: the production system assigns a model per location
  and supports per-date rule overrides. The benchmark evaluates all models against all
  locations uniformly and compares them using win-and-loss counts.
