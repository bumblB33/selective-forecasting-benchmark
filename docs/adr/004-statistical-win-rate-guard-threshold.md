# ADR 004: Two-Signal Guard: Win-Rate Threshold for the Statistical Direction Label

Status: Accepted
Date: 2026-03-28

## Context

`compute_statistical_tests()` in `core/evaluation.py` labels each model's direction as
`'better'` or `'worse'` relative to the baseline. The Wilcoxon signed-rank test produces
a p-value indicating whether per-location MAE differences are statistically significant,
but the test is non-directional: a significant result tells us the distribution is
shifted, not which way.

The median of per-location MAE differences provides direction. However, using the median
alone is insufficient: a small number of extreme-outlier locations can flip the median's
sign while the model is actually worse for the majority of locations. This is a known
failure mode for skewed distributions of per-location errors, which are common in this
project's sales data (a handful of high-volume locations dominate the totals).

A two-signal guard was introduced to prevent this: a model is labeled `'better'` only
when the median per-location MAE difference is negative and the win rate exceeds a
minimum threshold.

Audit finding 1.4 (the methodology audit of 2026-03-28) identified that the 0.5 threshold
was hardcoded without documentation as a deliberate policy choice.

## Decision

The win-rate guard threshold is formalized as `_WIN_RATE_GUARD = 0.5` in
`core/evaluation.py`. It is intentionally not surfaced in pipeline `config.py` files
because it is a statistical methodology choice rather than a pipeline tuning parameter.
Keeping it in `evaluation.py` co-locates it with the Wilcoxon logic it guards.

The 0.5 threshold means a model must outperform the baseline at a strict majority of
locations. A model that wins at 49.9 percent of locations is not reliably better, even if its
median difference is slightly negative; the result could be driven by a few large
wins and many small losses.

## Consequences

- If the threshold is changed, it must be changed in `core/evaluation.py` directly.
  There is no config knob for callers.
- A model that improves MAE at 49 percent of locations with a highly significant Wilcoxon
  p-value will be labeled `'worse'`. This is intentional: the goal of this benchmark
  is to identify models that are reliably better across the majority of the population,
  not just in aggregate.
- The threshold of 0.5 is conservative relative to some alternatives (for example, 0.45
  to allow near-majority wins). If empirical analysis shows that a lower threshold better
  captures genuine improvements, this ADR should be updated and the constant raised or
  lowered accordingly.
