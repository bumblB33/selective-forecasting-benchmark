# Forecasting Benchmark: Open-Source Models vs. Production Baseline

A benchmarking framework that asks whether Nixtla's open-source forecasting libraries (StatsForecast and MLForecast/LightGBM) can outperform an incumbent production forecasting system at enterprise scale. The framework is tested at two time granularities, daily and 15-minute intervals, using real production data from hundreds of locations.

---

## Motivation

Production forecasting systems embedded in software-as-a-service platforms often rely on simple heuristics: year-over-year averages, rolling means of the most recent weeks, or seasonal repeaters. These are interpretable and fast, but they leave accuracy gains on the table that affect downstream decisions such as labor scheduling and inventory management.

This benchmark asks a precise question: with real production data, statistical rigor, and a fair evaluation framework, how much better can open-source models do?

The short answer, at both daily and 15-minute interval granularities, is: meaningfully better, and the result holds up under statistical testing.

---

## Architecture

Two self-contained pipelines share a common `core/` module for evaluation, preprocessing, and model utilities. Neither pipeline imports from the other's config; all parameters are passed explicitly.

```
forecasting-benchmark/
├── core/             # Pipeline-agnostic: evaluation, preprocessing, shared models
├── daily/            # Daily pipeline: 7-day horizon, 105-week training history
├── interval/         # 15-min pipeline: 672-interval horizon, 26-week history
├── reporting/        # HTML report generation (embedded charts, no external deps)
├── output_contract/  # Reference: adapting Nixtla UTC timestamps to integer-keyed APIs
├── snowflake/        # SQL extract templates (schema names are placeholders)
├── scripts/          # Parity check, report regeneration, ablation study
├── tests/            # Architecture constraint tests and WAPE behavior tests
└── docs/
    └── adr/          # Architecture Decision Records (001 through 004)
```

Pipeline stages (both pipelines):

1. Load the CSV extract (data warehouse output, read into a pandas DataFrame).
2. Drop rows flagged as outliers (mirroring production training exclusions).
3. Fill missing dates and intervals with zero (closed periods are used for training but excluded from accuracy measurement).
4. Create a train/test split with a bounded training history.
5. Add calendar features (holidays, weekends, month boundaries).
6. Add Fourier features that capture cyclic patterns (weekly and yearly for the daily pipeline, daily and weekly for the interval pipeline).
7. Train StatsForecast models in parallel across cores.
8. Train MLForecast/LightGBM models sequentially, with memory management between models.
9. Cross-validate using 3 expanding windows.
10. Evaluate: compute per-location metrics, aggregate them, count wins and losses, and run a Wilcoxon signed-rank test for significance.
11. Save timestamped CSV artifacts and regenerate the HTML report.

---

## Models Evaluated

### Statistical Models (Nixtla StatsForecast)

| Model | Description | Daily | Interval |
|---|---|:---:|:---:|
| SeasonalNaive | Repeats the same weekday or same time slot from the prior week | yes | yes |
| MSTL | Multi-seasonal trend decomposition (weekly plus yearly for daily, daily plus weekly for interval) | yes | yes |
| AutoETS | Automated exponential smoothing | yes | no |
| AutoARIMA | Automatically selected ARIMA model with seasonal differencing | yes | no |
| Dynamic Optimized Theta | Ensemble of Theta models with optimized damping | yes | no |

AutoETS is excluded from the interval pipeline because it is prohibitively slow at 96-slot daily seasonality.

### Machine Learning Models (Nixtla MLForecast plus LightGBM)

Three performance tiers, each trained with lag features, Fourier harmonics, and calendar features:

| Model | Trees | Max Depth | Notes |
|---|---|---|---|
| LightGBM Fast | 200 | 6 | Lowest latency |
| LightGBM Standard | 400 | 8 | Balanced |
| LightGBM Accuracy | 300 to 500 | 10 | Maximum accuracy |

Lag features (daily): 1, 7, 14, 21, 28 days, plus 56 days for the Accuracy variant.
Lag features (interval): 1, 96, and 672 slots, corresponding to one interval, one day, and one week.
Fourier features (daily): weekly K=3, yearly K=3.
Fourier features (interval): daily K=3, weekly K=2.

---

## Evaluation Methodology

### Two-signal guard

A model is labeled as better than the baseline only when both conditions hold:

1. The median per-location MAE difference is negative, meaning the model is better at the typical location.
2. The win rate exceeds 50 percent, meaning a strict majority of locations see an improvement.

This prevents a small number of outlier locations from flipping the overall direction of the result. See [`docs/adr/004-statistical-win-rate-guard-threshold.md`](docs/adr/004-statistical-win-rate-guard-threshold.md).

### Statistical significance

All win and loss conclusions are validated with a Wilcoxon signed-rank test. The test is non-parametric, uses paired per-location MAE differences, applies a significance level of 0.05, and requires a minimum of 10 locations.

### WAPE denominator

Weighted Absolute Percentage Error (WAPE) uses only positive-actual rows in the denominator, matching the production system's own evaluation convention. Zero-sales rows, which represent closed periods, are excluded from accuracy measurement but are used in training. See [`docs/adr/002-wape-denominator-positive-actuals.md`](docs/adr/002-wape-denominator-positive-actuals.md).

### Metrics

| Metric | Definition | Formula |
|---|---|---|
| MAE | Mean Absolute Error | `mean(|actual - forecast|)` |
| WAPE | Weighted Absolute Percentage Error | `sum(|actual - forecast|) / sum(actual[actual > 0])` |
| MAPE | Mean Absolute Percentage Error, with zero-actual rows excluded | `mean(|actual - forecast| / actual)` |
| RMSE | Root Mean Squared Error | `sqrt(mean((actual - forecast)^2))` |

---

## Results

Benchmark run: March 2026. Up to 300 locations were sampled; after eligibility filtering, 161 locations remained in the daily pipeline and 213 in the interval pipeline.

### Daily (7-day horizon, 105-week training history)

| Model | Win Rate | MAE | Change vs. Baseline |
|---|---|---|---|
| LightGBM Accuracy | 72 percent (116 of 161) | $760.07 | -10.6 percent |
| LightGBM Standard | 66.5 percent | $771.43 | -9.3 percent |
| LightGBM Fast | 65.2 percent | $778.90 | -8.4 percent |
| AutoETS | 61.5 percent | $793.21 | -6.7 percent |
| Dynamic Optimized Theta | 58.4 percent | $806.55 | -5.1 percent |
| Baseline Production | not applicable | $850.11 | not applicable |

All LightGBM variants and AutoETS are statistically significant at the Wilcoxon p < 0.05 level.

### 15-Minute Interval (672-interval horizon, 26-week training history)

| Model | Win Rate | MAE | Change vs. Baseline |
|---|---|---|---|
| LightGBM Accuracy | 100 percent (213 of 213) | $38.12 | -41.4 percent |
| LightGBM Standard | 100 percent | $39.44 | -39.4 percent |
| MSTL | 100 percent | $41.87 | -35.6 percent |
| LightGBM Fast | 98.1 percent | $42.30 | -34.9 percent |
| SeasonalNaive | 91.1 percent | $48.91 | -24.8 percent |
| Baseline Production | not applicable | $65.03 | not applicable |

All models are statistically significant at the Wilcoxon p < 0.05 level.

The full HTML reports are available in [`data/reports/`](data/reports/).

---

## Running It

### Prerequisites

```bash
pip install -e .
```

### Data

Execute the SQL templates in `snowflake/` against your data warehouse and save the outputs as:

- `data/input/daily_actual_vs_forecast.csv`
- `data/input/interval_actual_vs_forecast.csv`

See [Adapting to your data](#adapting-to-your-data) for the expected column format.

### Commands

```bash
# Daily benchmark (7-day horizon)
forecasting-daily

# Daily, with specific options
forecasting-daily --no-sf          # skip StatsForecast models
forecasting-daily --no-ml          # skip LightGBM models
forecasting-daily --no-cv          # skip cross-validation
forecasting-daily --rolling-windows 4   # evaluate four consecutive one-week windows
forecasting-daily --metric gross_sales  # use gross sales as the target

# 15-minute interval benchmark
forecasting-interval
forecasting-interval --no-sf
forecasting-interval --no-ml
forecasting-interval --no-cv

# Regenerate HTML reports from saved metrics, with no retraining
forecasting-regenerate --interval daily
forecasting-regenerate --interval 15min

# Parity check: compare the latest run against a golden snapshot
forecasting-parity

# Ablation: evaluate sensitivity to training history length
forecasting-ablation
```

---

## Adapting to Your Data

### Required CSV columns

The data loaders in `daily/data_loader.py` and `interval/data_loader.py` expect these columns from your data warehouse extract:

**Daily:**
```
ENTITY_UID, CUSTOMER_ID, SITE_ID, DAY_START_TIME,
PROJECTION, USER_ADJUSTMENT, PUBLISHED_USER_ADJUSTMENT,
ACTUAL_VALUE, GROSS_SALES_ACTUAL,
IS_GENERATED_BY_MANAGER, IS_SYSTEMATIC_GENERATED,
IS_OUTLIER, OUTLIER_EXCLUSION_REASON
```

**Interval:**
```
ENTITY_UID, CUSTOMER_ID, SITE_ID,
INTERVAL_ID_TICKS, INTERVAL_START_TS,
NET_SALES, GROSS_SALES,
BASELINE_PROJECTION, USER_ADJUSTMENT, PUBLISHED_USER_ADJUSTMENT,
IS_CLOSED, BUSINESS_START_HOUR_UTC, DAILY_NET_SALES,
IS_OUTLIER, OUTLIER_EXCLUSION_REASON
```

### SQL templates

The `snowflake/` directory contains parameterized SQL templates. Replace all `<your_db>`, `<your_forecasting_schema>`, `<your_sales_schema>`, and `<your_precomputed_data_source>` tokens with your actual schema names.

The key joins are:

- A forecast sessions table, containing daily forecast records per location-day with a per-interval breakdown.
- A sales actuals table, containing daily or interval-level actual sales.
- A precomputed outlier flags table, containing IS_OUTLIER and OUTLIER_EXCLUSION_REASON per location-day.

If your system does not have outlier flags, set `IS_OUTLIER = FALSE` for all rows; the pipeline will then train on all available history.

### Baseline

The `PROJECTION` column (daily) and `BASELINE_PROJECTION` column (interval) hold your incumbent system's forecasts. If you do not have a production baseline to compare against, you can use `SeasonalNaive` as the baseline by changing `BASELINE_MODEL` in `daily/config.py` and `interval/config.py`.

---

## Design Decisions

Architecture decisions are documented in `docs/adr/`:

| ADR | Decision |
|---|---|
| [001](docs/adr/001-core-module-pipeline-agnostic.md) | `core/` never imports from `daily.config` or `interval.config` |
| [002](docs/adr/002-wape-denominator-positive-actuals.md) | WAPE denominator uses only positive-actual rows |
| [003](docs/adr/003-training-pool-evaluation-strategy.md) | Natural pool training; fixed-pool in ablation |
| [004](docs/adr/004-statistical-win-rate-guard-threshold.md) | Two-signal guard: median MAE + win rate > 50% |
