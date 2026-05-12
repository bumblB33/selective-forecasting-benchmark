# Nixtla Integration

This document describes how the project uses the Nixtla open-source ecosystem
([StatsForecast](https://github.com/Nixtla/statsforecast),
[MLForecast](https://github.com/Nixtla/mlforecast)).

## DataFrame Convention

All data is passed to the Nixtla libraries in the required three-column format:

| Column | Meaning |
|---|---|
| `unique_id` | Location identifier (for example, `12345_ABCD`) |
| `ds` | Observation timestamp (`datetime64[ns]`) |
| `y` | Sales value (`float64`, in dollars) |

The composite identifier format is `{CUSTOMER_ID}_{UPPER(SITE_ID)}`,
which matches the SQL definition in both Snowflake extraction queries.

## StatsForecast Usage

```python
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, MSTL, AutoETS, AutoARIMA

sf = StatsForecast(
    models=[SeasonalNaive(season_length=7), MSTL(season_length=[7, 365]), ...],
    freq="D",          # "D" for daily, "15min" for interval
    n_jobs=-1,
    fallback_model=SeasonalNaive(season_length=7),
)
predictions = sf.forecast(df=training_df, h=7)  # horizon in periods
```

The output DataFrame has columns `unique_id`, `ds`, and one column per model.
The `run_statsforecast()` function in `models.py` splits this into `{model_name: df}` entries.

Both pipelines set `fallback_model` to a `SeasonalNaive` instance, so that if
a primary model fails for a particular series (for example, a convergence failure),
forecasts are still produced rather than the call raising an error.

### Cross-Validation

```python
cv_df = sf.cross_validation(
    df=training_df,
    h=7,
    n_windows=3,
    step_size=7,
)
```

The output includes a `cutoff` column that identifies the fold boundary.

## MLForecast Usage

```python
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean, SeasonalRollingMean
from lightgbm import LGBMRegressor

fcst = MLForecast(
    models={"LightGBM_D_Fast": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, ...)},
    freq="D",
    lags=[1, 7, 14, 21, 28],
    lag_transforms={
        1: [ExpandingMean()],
        7: [SeasonalRollingMean(season_length=7, window_size=8)],
    },
    date_features=["dayofweek", "day", "month"],
)
fcst.fit(training_df, static_features=[])
predictions = fcst.predict(h=7, X_df=x_df)
```

Fourier features are passed as exogenous variables in `X_df`. The DataFrame is
built in both `interval/models.py` and `daily/models.py` via
`_build_aligned_mlforecast_xdf()`. The future exogenous DataFrame must
cover exactly `h` steps per location, with columns that match those in the
training data.

### Cross-Validation

MLForecast cross-validation uses the native `.cross_validation()` method in
both pipelines. The MLForecast CV API does not accept an `X_df` parameter
(verified in mlforecast version 1.0.2), so CV folds for LightGBM models are
evaluated without exogenous features (calendar and Fourier). As a result, the
CV metrics may understate the accuracy of models that benefit from exogenous
features during the main forecast run.

## Frequency Strings

| Pipeline | Frequency | Notes |
|---|---|---|
| Daily | `"D"` | Calendar days; gaps are filled before fitting. |
| Interval | `"15min"` | 96 slots per day, 672 slots per week. |

## Season Lengths

| Pipeline | Primary season | Secondary season |
|---|---|---|
| Daily (MSTL) | 7 (weekly) | 365 (yearly) |
| Daily (all other models) | 7 (weekly) | not applicable |
| Interval (MSTL) | 96 (daily) | 672 (weekly) |
| Interval (SeasonalNaive) | 96 (daily) | not applicable |

## Dependency Versions

Versions are pinned in `pyproject.toml`. The installed versions used during development were:

```
statsforecast >= 1.7   (tested: 2.0.3)
mlforecast    >= 0.13  (tested: 1.0.2)
lightgbm      >= 4.0   (tested: 4.6.0)
```

StatsForecast and MLForecast imports are deferred inside model functions
(for example, `from statsforecast import ...` inside `run_statsforecast`).
This keeps top-level import time fast, which matters especially for
`regenerate_report.py` and `parity_check.py`, neither of which needs the
model libraries at import time.

## Configuration and Tuning

This section documents every tunable parameter exposed by StatsForecast and
MLForecast that is relevant to this project. Parameters are grouped by
component. Each entry lists the library default, the value this project uses
(where applicable), and guidance on when and why to change the setting.

### StatsForecast Constructor

| Parameter | Type | Default | Project Value | Guidance |
|---|---|---|---|---|
| `models` | `list` | required | See `STATSFORECAST_MODELS` in config | List of model instances to fit on every series. |
| `freq` | `str` | required | `"D"` or `"15min"` | Must match the actual data frequency. |
| `n_jobs` | `int` | `1` | `-1` | A value of `-1` uses all CPU cores. Set to `1` for easier debugging. |
| `fallback_model` | model or `None` | `None` | `SeasonalNaive(...)` | If a primary model fails (for example, a convergence failure), this model produces forecasts instead. Setting it to `SeasonalNaive` guarantees complete output. |
| `verbose` | `bool` | `False` | `False` | Enables progress printing. Useful during interactive exploration, but noisy in continuous integration. |

### StatsForecast `.forecast()` and `.predict()`

| Parameter | Type | Default | Guidance |
|---|---|---|---|
| `h` | `int` | required | Forecast horizon in periods: 7 for daily, 672 for interval. |
| `df` | `DataFrame` | required | Must contain `unique_id`, `ds`, and `y`. |
| `X_df` | `DataFrame` or `None` | `None` | Future exogenous variables. Columns must match training exogenous names. Not used by either pipeline's StatsForecast path (both call `sf.forecast()` without this argument). |
| `level` | `list[float]` or `None` | `None` | Confidence levels for prediction intervals, for example `[80, 95]`. The output includes additional `ModelName-lo-{level}` and `ModelName-hi-{level}` columns. See [Prediction Intervals](#prediction-intervals). |
| `prediction_intervals` | `ConformalIntervals` or `None` | `None` | An alternative to setting `prediction_intervals` on each model instance. Pass `ConformalIntervals(n_windows=..., h=...)` here to enable conformal intervals for all models at call time. Requires `level` to also be set. |
| `fitted` | `bool` | `False` | Store in-sample fitted values. This increases memory; enable it when diagnosing residuals. |

### StatsForecast `.cross_validation()`

| Parameter | Type | Default | Guidance |
|---|---|---|---|
| `h` | `int` | required | Forecast horizon per fold. |
| `n_windows` | `int` | `1` | Number of evaluation windows. The project uses `CV_NUM_WINDOWS = 3`. |
| `step_size` | `int` | `1` | Steps between fold cutoffs. The project sets this equal to `h` for non-overlapping folds. |
| `refit` | `bool` | `True` | Re-train models at each cutoff. Disable for faster CV, at the cost of stale parameters. |
| `test_size` | `int` or `None` | `None` | An alternative to `n_windows`. Specifies the total test set size. In practice, mutually exclusive with `n_windows`. |
| `input_size` | `int` or `None` | `None` | Limit training history per fold. `None` uses all available history (an expanding window). |

### Model: SeasonalNaive

Repeats the last observed seasonal cycle. No parameters are optimized.

| Parameter | Default | Project Value | Guidance |
|---|---|---|---|
| `season_length` | required | `7` (daily), `96` (interval) | Must equal the dominant cycle length in periods. |
| `prediction_intervals` | `None` | `None` | Pass `ConformalIntervals(...)` to enable intervals. |

Minimum data: at least `season_length` observations per series.

### Model: MSTL

Decomposes the series into multiple seasonal components using STL, then
forecasts the trend residual with a pluggable sub-model.

| Parameter | Default | Project Value | Guidance |
|---|---|---|---|
| `season_length` | required | `[7, 365]` daily; `[96, 672]` interval | Pass a list for multi-seasonal decomposition. Each value must divide cleanly into the data cadence. |
| `trend_forecaster` | `AutoETS()` instance | `AutoARIMA()` instance | Any StatsForecast model instance. The library default is an `AutoETS()` instance (not the string `"AutoETS"`); this project overrides with `AutoARIMA()` for the trend component. Using `AutoARIMA` tends to capture trend drift better for sales data. |
| `prediction_intervals` | `None` | `None` | Pass `ConformalIntervals(...)` to enable intervals. |

Minimum data: at least `max(season_length)` observations, plus enough
for the trend forecaster to fit. In practice, twice the longest season is a
safe minimum.

### Model: AutoETS

Automatic exponential smoothing. It selects the best Error, Trend, and Seasonal
combination from up to 30 candidate models using an information criterion.

| Parameter | Default | Project Value | Guidance |
|---|---|---|---|
| `season_length` | `1` | `7` (daily only) | Set to `1` to disable the seasonal component entirely. |
| `model` | `"ZZZ"` | `"ZZZ"` | A three-character string controlling component selection. Each position is Error, Trend, or Seasonal; each character is `Z` (auto), `N` (none), `A` (additive), or `M` (multiplicative). `"ZZZ"` lets the algorithm choose freely. Fix a character to constrain the search; for example, `"AAN"` forces additive error, additive trend, and no seasonal component. |
| `damped` | `None` | `None` | `None` means consider both damped and undamped trend. `True` means only damped variants. `False` means only undamped. Damped trends prevent explosive long-horizon forecasts; force `True` if the horizon greatly exceeds the training length. |
| `prediction_intervals` | `None` | `None` | ETS provides native analytical intervals; it also supports `ConformalIntervals`. |

Minimum data: at least `2 * season_length` for stable seasonal estimation.
AutoETS is omitted from the interval pipeline because state-space estimation
with `season_length=96` is prohibitively slow and rarely converges.

### Model: AutoARIMA

A stepwise search over ARIMA(p, d, q)(P, D, Q)[m] orders, selecting the best
model by an information criterion.

| Parameter | Default | Project Value | Guidance |
|---|---|---|---|
| `season_length` | `1` | `7` (daily) | The seasonal period *m*. |
| `d` | `None` | `None` | Non-seasonal differencing order. `None` means it is determined by a unit-root test. Override to `0` if the series is known to be stationary. |
| `D` | `None` | `None` | Seasonal differencing order. `None` means automatic. |
| `max_p` and `max_q` | `5` | `5` | Upper bounds for non-seasonal AR and MA orders. Increasing widens the search at the cost of compute time. |
| `max_P` and `max_Q` | `2` | `2` | Upper bounds for seasonal AR and MA orders. |
| `max_order` | `5` | `5` | Maximum value of `p + q + P + Q`. Constrains total model complexity. |
| `max_d` and `max_D` | `2` and `1` | `2` and `1` | Maximum differencing orders. |
| `stepwise` | `True` | `True` | Stepwise search is dramatically faster than exhaustive search. Set to `False` for exhaustive search on small datasets, or when stepwise misses a good model. |
| `approximation` | `False` | `False` | Approximates likelihood for faster fitting. Useful for very long series, but sacrifices some accuracy. |
| `ic` | `"aicc"` | `"aicc"` | Information criterion for model selection. Options: `"aicc"`, `"aic"`, `"bic"`. AICc is recommended for small to moderate sample sizes. |
| `test` | `"kpss"` | `"kpss"` | Unit-root test for choosing `d`. Alternative: `"adf"`. |
| `seasonal_test` | `"seas"` | `"seas"` | Test for seasonal differencing. |
| `allowdrift` | `True` | `True` | Allow a drift (constant) term in differenced models. |
| `allowmean` | `True` | `True` | Allow a mean term in undifferenced models. |
| `blambda` | `None` | `None` | Box-Cox lambda. `None` means no transform. Set to `0` for a log transform, or a positive float for a power transform. |
| `trace` | `False` | `False` | Print the search path. Useful for diagnosing why a particular order was chosen. |
| `prediction_intervals` | `None` | `None` | ARIMA provides native asymptotic intervals; it also supports `ConformalIntervals`. |

Minimum data: `2 * season_length` is recommended. Short series may have
differencing orders forced to `0`, which results in white-noise-level forecasts.

Performance note: AutoARIMA can be slow on large datasets (more than 500 locations).
Use `--no-sf` for quick data-loading checks.

### Model: DynamicOptimizedTheta (DOT)

Optimizes a theta-line decomposition with automatic seasonal adjustment.

| Parameter | Default | Project Value | Guidance |
|---|---|---|---|
| `season_length` | `1` | `7` (daily only) | Set to match the dominant cycle. |
| `decomposition_type` | `"multiplicative"` | library default | Either `"multiplicative"` or `"additive"`. The project does not override this parameter; the library default of `"multiplicative"` applies. Use `"additive"` when the seasonal amplitude does not scale with the level (rare for sales data). |
| `prediction_intervals` | `None` | `None` | Supports `ConformalIntervals`. |

Minimum data: at least `2 * season_length`. DOT is configured in the daily
pipeline only (as `DOT_D` in `daily/config.py`).

### MLForecast Constructor

| Parameter | Type | Default | Project Value | Guidance |
|---|---|---|---|---|
| `models` | regressor, list, or dict | required | `{name: LGBMRegressor(...)}` | A single scikit-learn-compatible regressor, a list of regressors, or a `{name: regressor}` dict. The project uses a dict so each model column in the output is named. |
| `freq` | `str` | required | `"D"` or `"15min"` | Must match the data frequency. |
| `lags` | `list[int]` | `None` | Varies per tier | Lag features to create. For example, `[1, 7, 14, 21, 28]`. Longer lags capture longer memory but increase feature dimensionality. |
| `lag_transforms` | `dict` | `None` | See below | A dict keyed by lag number, mapping to a list of transform instances. |
| `date_features` | `list` | `None` | `["dayofweek", "month", ...]` | String names or callables. Strings map to pandas datetime properties. |
| `target_transforms` | `list` | `None` | `None` (unused) | Pre-processing transforms applied to the target before feature engineering. See [Target Transforms](#target-transforms). |
| `num_threads` | `int` | `1` | `-1` | Parallel threads for feature computation. A value of `-1` uses all cores. |

### MLForecast `.fit()`

| Parameter | Type | Default | Guidance |
|---|---|---|---|
| `df` | `DataFrame` | required | Must contain `unique_id`, `ds`, and `y`, plus any exogenous columns. |
| `static_features` | `list[str]` or `None` | `None` | Columns to treat as static (constant per series). The project passes `[]` to explicitly declare no static features. |
| `prediction_intervals` | `PredictionIntervals` or `None` | `None` | Enable interval estimation during fit. See [Prediction Intervals](#prediction-intervals). |
| `keep_last_n` | `int` or `None` | `None` | Retain only the last *n* observations per series for feature computation. Reduces memory for very long histories. |
| `max_horizon` | `int` or `None` | `None` | Train separate models per horizon step. Can improve accuracy at the cost of multiplied training time. |
| `dropna` | `bool` | `True` | Drop rows with NaN lag features (the initial lags). |
| `weight_col` | `str` or `None` | `None` | Column used as sample weights during model fitting. |

### MLForecast `.predict()`

| Parameter | Type | Default | Guidance |
|---|---|---|---|
| `h` | `int` | required | Forecast horizon in periods. |
| `X_df` | `DataFrame` or `None` | `None` | Future exogenous features. Must contain `unique_id`, `ds`, and exogenous columns for exactly `h` future steps per series. |
| `level` | `list[float]` or `None` | `None` | Confidence levels for prediction intervals. Requires `prediction_intervals` to have been set in `.fit()`. |
| `before_predict_callback` | callable or `None` | `None` | Called before each prediction step. Receives the feature DataFrame for inspection or mutation. |
| `new_df` | `DataFrame` or `None` | `None` | Provide new data at prediction time (for example, for online updates) without re-fitting. |

### MLForecast `.cross_validation()`

| Parameter | Type | Default | Guidance |
|---|---|---|---|
| `n_windows` | `int` | required | Number of evaluation windows. The project uses `CV_NUM_WINDOWS = 3`. |
| `h` | `int` | required | Forecast horizon per fold. |
| `step_size` | `int` or `None` | `None` (defaults to `h`) | Steps between fold cutoffs. `None` defaults to `h`. |
| `refit` | `bool` | `True` | Re-train at each cutoff. |
| `input_size` | `int` or `None` | `None` | Limit training window size per fold. |
| `static_features` | `list[str]` or `None` | `None` | Same as `.fit()`. |
| `fitted` | `bool` | `False` | Store in-sample fitted values for each fold. Access via `cross_validation_fitted_values()`. |

Important: MLForecast's `.cross_validation()` does not accept an
`X_df` parameter (verified in version 1.0.31). Exogenous features included as
columns in the training DataFrame are available, but future exogenous
variables cannot be injected per fold. This is why CV metrics for LightGBM
may diverge from the main forecast accuracy.

### Lag Transforms

The project uses the class-based lag transform API from
`mlforecast.lag_transforms`. Each transform is instantiated and placed in the
`lag_transforms` dict, keyed by the lag number it operates on.

Rolling transforms (a sliding window over recent values):

| Class | Parameters | Description |
|---|---|---|
| `RollingMean` | `window_size`, `min_samples=None` | Rolling average. |
| `RollingStd` | `window_size`, `min_samples=None` | Rolling standard deviation. |
| `RollingMin` | `window_size`, `min_samples=None` | Rolling minimum. |
| `RollingMax` | `window_size`, `min_samples=None` | Rolling maximum. |
| `RollingQuantile` | `p`, `window_size`, `min_samples=None` | Rolling quantile at probability `p`. |

Seasonal rolling transforms (a window over same-season lags):

| Class | Parameters | Description |
|---|---|---|
| `SeasonalRollingMean` | `season_length`, `window_size`, `min_samples=None` | Mean across the last `window_size` occurrences of the same seasonal slot. |
| `SeasonalRollingStd` | `season_length`, `window_size`, `min_samples=None` | Standard deviation across seasonal lags. |
| `SeasonalRollingMin` | `season_length`, `window_size`, `min_samples=None` | Minimum across seasonal lags. |
| `SeasonalRollingMax` | `season_length`, `window_size`, `min_samples=None` | Maximum across seasonal lags. |
| `SeasonalRollingQuantile` | `p`, `season_length`, `window_size`, `min_samples=None` | Quantile across seasonal lags. |

Expanding transforms (cumulative over all history):

| Class | Parameters | Description |
|---|---|---|
| `ExpandingMean` | none | Cumulative mean. |
| `ExpandingStd` | none | Cumulative standard deviation. |
| `ExpandingMin` | none | Cumulative minimum. |
| `ExpandingMax` | none | Cumulative maximum. |
| `ExpandingQuantile` | `p` | Cumulative quantile. |

Other transforms:

| Class | Parameters | Description |
|---|---|---|
| `ExponentiallyWeightedMean` | `alpha` | Exponentially weighted mean with decay factor `alpha` (between 0 and 1). A lower alpha produces more smoothing. |
| `Offset` | `tfm`, `n` | Shifts the series by `n` positions before applying `tfm`. |
| `Combine` | `tfm1`, `tfm2`, `operator` | Combines two transforms element-wise (for example, `operator.truediv` for a ratio). |

The following transforms accept an optional `global_=True` to compute the
statistic across all series aggregated by timestamp, or `groupby=["col"]`
to compute by group. The two options are mutually exclusive. The classes that
expose them are: `RollingQuantile`, all `SeasonalRolling*` variants, all
`Expanding*` variants, and `ExponentiallyWeightedMean`. The basic
`RollingMean`, `RollingStd`, `RollingMin`, and `RollingMax` do not expose
these parameters.

### Target Transforms

These are available but not currently used in this project. They are applied
via the `target_transforms` parameter on the `MLForecast` constructor.
Transforms are applied in list order before feature engineering, and
automatically inverted after prediction.

| Class | Import | Parameters | Description |
|---|---|---|---|
| `Differences` | `mlforecast.target_transforms` | `differences: list[int]` | Sequential differencing. `[1]` is the first difference; `[1, 7]` is the first difference plus a weekly seasonal difference. The series must be longer than the sum of all difference values. |
| `LocalStandardScaler` | `mlforecast.target_transforms` | none | Per-series z-score normalization. Useful when series have different scales. |
| `LocalMinMaxScaler` | `mlforecast.target_transforms` | none | Per-series scaling into [0, 1]. |
| `LocalBoxCox` | `mlforecast.target_transforms` | none | Per-series Box-Cox transformation. Stabilizes variance for heteroscedastic series. |
| `GlobalSklearnTransformer` | `mlforecast.target_transforms` | `transformer` | Wraps any scikit-learn transformer applied globally across all series (for example, `PowerTransformer()`). |

Example (not used in the project, shown for reference):

```python
from mlforecast.target_transforms import Differences, LocalStandardScaler

MLForecast(
    models={"lgb": LGBMRegressor()},
    freq="D",
    lags=[1, 7],
    target_transforms=[Differences([1]), LocalStandardScaler()],
)
```

### Prediction Intervals

Neither pipeline currently generates prediction intervals. Both libraries
support them, and they can be enabled without structural changes.

StatsForecast: pass `level` to `.forecast()` or `.predict()`.

```python
# Option A: native model intervals (available for AutoARIMA and AutoETS)
predictions = sf.forecast(df=train, h=7, level=[80, 95])

# Option B: conformal intervals on the model instance (model-agnostic, requires calibration data)
from statsforecast.utils import ConformalIntervals
sf = StatsForecast(
    models=[AutoARIMA(
        season_length=7,
        prediction_intervals=ConformalIntervals(n_windows=3, h=7),
    )],
    freq="D", n_jobs=-1,
)
predictions = sf.forecast(df=train, h=7, level=[80, 95])

# Option C: conformal intervals at forecast time (applies to all models)
predictions = sf.forecast(
    df=train, h=7, level=[80, 95],
    prediction_intervals=ConformalIntervals(n_windows=3, h=7),
)
```

`ConformalIntervals` parameters:

- `n_windows` (default `2`): number of CV windows used for calibration. Must be at least 2.
- `h` (default `1`): forecast horizon. Should match the `h` passed to `.forecast()`.
- `method` (default `"conformal_distribution"`): interval estimation method.

Constraint: `n_windows` multiplied by `h` must be less than the series length.

MLForecast: pass `prediction_intervals` during `.fit()`, then pass `level` to `.predict()`.

```python
from mlforecast.utils import PredictionIntervals

fcst.fit(
    train,
    prediction_intervals=PredictionIntervals(n_windows=3, h=7),
)
predictions = fcst.predict(h=7, level=[80, 95])
```

`PredictionIntervals` accepts `n_windows`, `h`, and `method` with the same defaults as `ConformalIntervals` (`n_windows=2`, `h=1`, `method="conformal_distribution"`).

Output columns follow the pattern `ModelName-lo-{level}` and `ModelName-hi-{level}`.

### Minimum Data Requirements

Series that are too short cause hard errors during fitting or cross-validation.
The requirements vary by model and by operation.

Per-model minimums for forecasting:

| Model | Minimum observations per series | Rationale |
|---|---|---|
| SeasonalNaive | `season_length` | Needs at least one full cycle to repeat. |
| MSTL | `max(season_length)` | Needs the longest seasonal cycle for decomposition. Twice `max(season_length)` is safer to ensure a stable STL fit. |
| AutoETS | `2 * season_length` | Needs enough data to estimate seasonal states. |
| AutoARIMA | `2 * season_length` | Needs enough data for differencing and seasonal pattern detection. Very short series may have differencing forced to `0`. |
| DOT | `2 * season_length` | Needs enough data for seasonal decomposition. |
| LightGBM (MLForecast) | `max(lags) + 1` | Must have enough rows to compute all lag features after NaN rows are dropped. |

Cross-validation minimums:

StatsForecast requires each series to have at least `h + (n_windows - 1) * step_size`
observations. The interval pipeline applies a stricter threshold in
`_filter_short_series_for_cv()`:

```
min_length = (CV_NUM_WINDOWS + 2) * FORECAST_HORIZON
```

With the current interval settings (`CV_NUM_WINDOWS=3`, `FORECAST_HORIZON=672`),
this evaluates to 3,360 intervals, or approximately 35 days of 15-minute data. The extra
`+2` headroom ensures the earliest CV fold's training window covers at least
two full weekly cycles, which is the practical minimum for MSTL to produce
stable decompositions. Series below this threshold are silently excluded
from CV but are still used for the main forecast.

## Known Limitations

- AutoARIMA speed: can be slow on large datasets (more than 500 locations). Use
  `--no-sf` combined with `--no-ml` for quick data-loading checks.
- MSTL multi-seasonal: two seasonal periods requires `statsforecast >= 1.7`.
  Earlier versions only accepted a single `season_length`.
- MLForecast CV without exogenous features: MLForecast's `.cross_validation()`
  does not accept an `X_df` parameter (verified in version 1.0.2), so CV folds
  for LightGBM models are evaluated without future exogenous features
  (calendar and Fourier). Exogenous columns present in the training DataFrame
  are available as features, but future values cannot be injected per fold.
  As a result, CV metrics may understate the accuracy of models that benefit
  from exogenous features during the main forecast run.
- AutoETS on sub-daily data: ETS state-space estimation with
  `season_length=96` is prohibitively slow and rarely converges. The interval
  pipeline omits AutoETS; MSTL combined with AutoARIMA provides a principled
  multi-seasonal alternative.
