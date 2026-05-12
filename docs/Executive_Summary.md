# Forecasting Benchmark: Project Overview

This project evaluates whether open-source forecasting models from the Nixtla ecosystem
can predict multi-location business sales more accurately than an incumbent production
forecasting system. Models were evaluated at both daily and 15-minute granularities,
using real production data from a software-as-a-service enterprise resource planning
platform that serves hundreds of locations.

---

## Glossary

| Term | Definition |
|------|------------|
| ACTUAL_VS_FORECAST | Actual versus forecast: data comparing predicted sales to what actually occurred. |
| MAE | Mean Absolute Error: average dollar difference between forecast and actual sales. Lower is better. |
| WAPE | Weighted Absolute Percentage Error: forecast error as a percentage of total sales. It accounts for varying sales volumes across locations. |
| MAPE | Mean Absolute Percentage Error: average percentage error across all time periods. |
| RMSE | Root Mean Squared Error: error metric that emphasizes larger forecast errors more heavily than MAE. |
| StatsForecast | Nixtla library that provides traditional statistical forecasting methods. |
| MLForecast | Nixtla library that uses machine learning, specifically LightGBM, for forecasting. |
| LightGBM | Machine learning algorithm that learns patterns from historical data using gradient-boosted decision trees. |
| Baseline | The incumbent production forecast, used as the comparison point. |
| Win rate | The percentage of locations where a new model outperforms the baseline. |
| Composite UID | A unique identifier that combines tenant and location identifiers. |

---

## Table of contents

- [Purpose](#purpose)
- [Benchmark results](#benchmark-results)
- [Architecture](#architecture)
- [Data pipeline](#data-pipeline)
- [Benchmark modes](#benchmark-modes)
- [Models evaluated](#models-evaluated)
- [Evaluation methodology](#evaluation-methodology)
- [Output artifacts](#output-artifacts)
- [Next steps](#next-steps)

---

## Purpose

The incumbent forecasting system generates sales projections for each location using time-series algorithms. This project tests whether newer open-source forecasting models can predict sales more accurately.

Why it matters: more accurate forecasts lead to better labor scheduling, reduced waste, and improved profit margins across all locations.

This is a research and evaluation project, not a production system. The benchmark pipeline:

1. Extracts sample production data, including actual sales and forecasts, from the production data warehouse.
2. Trains and tests alternative forecasting models alongside the incumbent baseline.
3. Generates detailed accuracy reports showing which models perform best and where.

The results inform whether any new model is ready to replace or enhance the incumbent system, and at what time granularity (daily versus 15-minute intervals).

---

## Benchmark results

The March 2026 benchmark sampled up to 300 locations from production data. After eligibility filtering, 161 locations were evaluated at daily granularity and 213 locations at 15-minute intervals. Both benchmarks show that alternative forecasting models significantly outperform the incumbent baseline.

Key findings:

- Daily forecasting: LightGBM Accuracy achieves a 72 percent win rate with a 10.6 percent error reduction.
- Interval forecasting: LightGBM Accuracy achieves a 100 percent win rate with a 41.4 percent error reduction.

### Daily benchmark, run on 2026-03-27

| Dimension | Value |
|---|---|
| Locations evaluated | 161 |
| Test period | 2026-03-04 to 2026-03-10 (one week) |
| Training history | 105 weeks (approximately 2 years) |
| Primary baseline | Baseline Production (MAE: $850.11 per day per location) |

Best-performing model: LightGBM Accuracy (Daily)

| Metric | Value |
|---|---|
| Win rate vs. Baseline | 72 percent (116 of 161 locations) |
| MAE | $760.07 per day per location |
| MAE improvement | -10.6 percent vs. Baseline Production |
| WAPE | 12.8 percent |
| Statistically significant | Yes (Wilcoxon signed-rank, p < 0.05) |

Summary: five of eight alternative models outperform Baseline Production at most locations tested. LightGBM Accuracy, LightGBM Standard (66.5 percent win rate), and LightGBM Fast (65.2 percent) all show statistically significant improvements. AutoETS and Dynamic Optimized Theta also outperform the baseline at the majority of locations.

Business impact: deploying this model could reduce forecast error by approximately $4.7 million annually across 161 locations ($90.04 per day per location, multiplied by 325 operating days). This projection assumes uniform improvement across locations; actual business value depends on how forecast accuracy translates into scheduling efficiency and other operational decisions.

---

### Interval benchmark, run on 2026-03-26

| Dimension | Value |
|---|---|
| Locations in dataset | 292 |
| Nixtla models evaluated | 289 locations |
| Baseline evaluated | 213 locations (those with baseline projections) |
| Win/loss comparison basis | 213 locations (the Nixtla and Baseline intersection) |
| Test period | Most recent one week ending 2026-03-26 (672 intervals) |
| Training history | 26 weeks (approximately 6 months) |
| Primary baseline | Baseline Production (MAE: $65.03 per 15-minute interval per location) |

Best-performing model: LightGBM Accuracy (Interval)

| Metric | Value |
|---|---|
| Win rate vs. Baseline | 100 percent (213 of 213 locations) |
| MAE | $38.12 per 15-minute interval per location |
| MAE improvement | -41.4 percent vs. Baseline Production |
| WAPE | 40.8 percent |
| Statistically significant | Yes (Wilcoxon signed-rank, p < 0.05) |

Summary: all five Nixtla models outperform Baseline Production at every location tested. LightGBM Accuracy, LightGBM Standard, and MSTL each achieve a 100 percent win rate (213 of 213). The improvement over the baseline is consistent across all model types; even the simplest model (SeasonalNaive) wins at 91 percent of locations. The higher WAPE values reflect the inherent difficulty of predicting sales at 15-minute intervals. The relative improvement over the baseline still represents substantial gains.

Business impact: deploying this model could reduce forecast error by approximately $52.1 million annually across 213 locations ($26.91 MAE improvement, multiplied by 96 intervals per day, 213 locations, and 325 days). This projection assumes uniform improvement across locations; actual business value depends on how forecast accuracy translates into scheduling efficiency and other operational decisions.

---

## Project Structure

The project is organized as a Python benchmark suite. Each time granularity (daily and 15-minute intervals) has its own self-contained package. A shared reporting module generates HTML reports from both pipelines.

---

## Data Pipeline

The benchmark uses production data from the data warehouse, including:

- Historical sales actuals from 161 to 292 locations.
- Baseline production forecasts, used as the comparison point.
- For daily granularity, the most recent 3 years of calendar-day totals.
- For 15-minute interval granularity, the most recent 27 weeks of intra-day sales.

Data quality filters ensure:

- A minimum of 90 days of history with forecasts for daily, or 28 days for interval.
- At least 50 percent data coverage for interval locations, so locations with major gaps are excluded.
- Random sampling of up to 300 qualifying locations per benchmark run.

---

## Benchmark Approach

The benchmark evaluates forecast accuracy at two time granularities.

Daily forecasting:

- Training: 105 weeks (approximately 2 years) of historical daily sales.
- Testing: the most recent one week of daily totals.
- Models compared: 8 alternative models against the Baseline Production baseline.
- Additional baselines: manager-adjusted forecasts and strict variants that include zero-forecast periods.

15-minute interval forecasting:

- Training: 26 weeks (approximately 6 months) of 15-minute sales data.
- Testing: the most recent one week (672 intervals, which is 7 days times 96 intervals per day).
- Models compared: 5 alternative models against the Baseline Production baseline.
- Additional baselines: user-adjusted forecasts and strict variants.

---

## Models Evaluated

The benchmark compares Baseline Production against two families of open-source forecasting models.

### Statistical Models (StatsForecast)

These are traditional time-series algorithms that learn patterns from historical sales only.

Daily models:

- MSTL: multi-seasonal decomposition that captures weekly and yearly patterns.
- AutoETS: automated exponential smoothing.
- AutoARIMA: automated ARIMA model selection.
- Dynamic Optimized Theta: an ensemble forecasting method.
- SeasonalNaive: a simple baseline that repeats the prior week's same-day value.

Interval models:

- MSTL: multi-seasonal decomposition that captures daily and weekly patterns.
- SeasonalNaive: repeats the same interval's value from the prior day.

Note: these models require at least 2 years of history for daily, or 6 months for interval, to train reliably. AutoETS is excluded from interval forecasting because of computational cost at 15-minute granularity.

### Machine Learning Models (LightGBM)

LightGBM is a gradient-boosted decision tree method that learns from historical sales patterns plus calendar features (weekends, holidays, month boundaries, and seasonal cycles).

Three performance tiers:

- LightGBM Fast: optimized for speed, with 200 decision trees.
- LightGBM Standard: balanced performance, with 400 trees and deeper analysis.
- LightGBM Accuracy: maximum precision, with 300 to 500 trees and the deepest analysis.

All LightGBM models incorporate:

- Recent sales trends (one to eight weeks of prior values).
- Same-weekday patterns (comparing Mondays to Mondays, and so on).
- Holiday and calendar effects.
- Seasonal cycles (weekly patterns for daily forecasting, daily and weekly for intervals).

---

## Evaluation Methodology

### Data Preparation

Before training, the pipeline standardizes data across all models:

1. Missing period handling: gaps are filled with zero sales, treating them as closure days. These rows are used for training but are excluded from accuracy measurement.

2. Outlier protection: extreme sales values, above the 99th percentile per location, are capped to prevent distortion. Caps are calculated from training data only; test data never influences training.

3. Calendar awareness: models receive features for weekends, federal holidays, and month boundaries so they can recognize calendar patterns.

### Metrics

The benchmark calculates accuracy metrics for each location individually, then summarizes them across all locations using mean, median, and percentile statistics.

| Metric | Formula | Notes |
|---|---|---|
| MAE | `mean(|actual - forecast|)` | Primary metric. Average dollar error per period per location. |
| WAPE | `sum(|actual - forecast|) / sum(actual)` | Volume-weighted; more robust than MAPE when sales vary widely across locations. |
| MAPE | `mean(|actual - forecast| / actual)` | Zero-actual rows are excluded from the denominator. |
| RMSE | `sqrt(mean((actual - forecast)^2))` | Penalizes large individual errors heavily. |

Zero-value handling (the default settings for both pipelines):

- Time periods with zero actual sales are excluded from accuracy calculations.
- Default baseline comparisons exclude periods where the baseline produced a zero forecast. The strict baseline variants include these zero-forecast periods, which results in higher reported error. This reflects a broader evaluation scope, not worse forecasts.

### Win/loss comparison

Each alternative model is compared against Baseline Production on a location-by-location basis. A model wins at a location if its average forecast error (MAE) is lower than the baseline's error. Win rates show the percentage of locations where the new model outperforms the current system.

### Statistical significance

The benchmark uses Wilcoxon signed-rank tests to determine whether performance differences are statistically meaningful or could have occurred by chance. Tests require at least 10 locations and use a 95 percent confidence level. Results marked as statistically significant indicate reliable improvements.

### Volume tiering

Locations are grouped into quartiles (Q1 through Q4) based on average sales volume during the test period. This allows the HTML reports to compare model performance across high-volume versus low-volume locations.

### Cross-validation

Both pipelines test model consistency using time-series cross-validation: training and testing on three non-overlapping historical windows. This verifies that accuracy improvements are stable over time, rather than artifacts of a single test period. Cross-validation can be disabled with the `--no-cv` flag.

Important limitation: the MLForecast cross-validation implementation (version 1.0.31) cannot incorporate calendar and seasonal pattern features. Cross-validation metrics for LightGBM models may therefore underestimate their actual production accuracy.

---

## Benchmark Outputs

Each benchmark run generates:

Metrics files:

- Per-location accuracy scores (MAE, WAPE, MAPE, RMSE).
- Aggregate statistics (mean, median, and percentiles).
- Win/loss results (which model won at each location).
- Statistical significance tests (Wilcoxon signed-rank).
- Cross-validation metrics (consistency across multiple time windows).

HTML report: a self-contained visual report with embedded charts. It shows verdict recommendations, accuracy comparisons, win/loss visualizations, and per-location forecast quality. Reports can be regenerated from saved metrics without retraining models.

---

## Next steps

The March 2026 benchmark confirms that Nixtla models significantly outperform Baseline Production at both daily and 15-minute intervals, with statistical validation.

Next steps:

1. Full production validation: test the models on the complete production dataset (all locations, not just the 300-location sample) to confirm that accuracy improvements hold across all location types and regions.

2. Deployment roadmap: define how to integrate the best-performing models into production, including infrastructure requirements, model update schedules, performance monitoring, and phased rollout strategy.

3. Training data sensitivity analysis (a minimum-history study): many customers will not have 2 years of data available at pilot time. This experiment identifies the shortest training history at which MSTL and AutoETS still outperform Baseline Production, and how much their accuracy degrades as history shortens.

   Method: hold the test window fixed and re-run the benchmark at progressively shorter training lengths (for example, 26, 52, 78, and 104 weeks). For each length, record the win rate and MAE improvement vs. `Baseline_Production` for MSTL and AutoETS. The minimum viable length is where the win rate drops below a defined threshold (for example, no longer statistically significant, or win rate falls below 60 percent).

   Implementation: the `create_train_test_split` function in `daily/preprocessing.py` already accepts `max_train_history_weeks` as a parameter. Adding a `--train-history-weeks` CLI flag to `daily/run.py` is all that is needed to run this experiment. Results will vary by location: some locations may beat the baseline at 52 weeks while others require 78. The output should therefore report per-length win rates rather than a single global threshold.
