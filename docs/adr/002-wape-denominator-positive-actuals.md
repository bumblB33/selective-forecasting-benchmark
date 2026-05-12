# ADR 002: WAPE Denominator Uses Positive Actuals Only

Status: Accepted
Date: 2026-03-27

## Context

WAPE (Weighted Absolute Percentage Error) can be computed with different denominators:

- Standard WAPE: the denominator is the sum of all actuals, including zeros.
- Positive-actuals WAPE: the denominator is the sum of positive actuals only.

For multi-location sales data, zero-actual rows represent closed days or non-operating intervals.
Including them in the denominator inflates it without contributing meaningful signal: a
zero-actual row cannot be wrong in a directional sense, because the location simply was not
operating.

## Decision

All WAPE computations in this codebase (in both `compute_metrics()` and `compute_cv_metrics()`)
use the positive-actuals denominator:

```
WAPE = sum(|actual - prediction|) / sum(actual[actual > 0])
```

This matches the production baseline's internal evaluation logic and maintains historical parity with previously
reported benchmark numbers.

The `EXCLUDE_ZERO_ACTUAL_FROM_EVAL` config flag controls whether zero-actual rows are excluded
from the numerator as well. When set to `True` (the default), both the numerator and the
denominator use positive-actual rows only. This makes WAPE equivalent to a weighted MAE
relative to positive sales.

The 10-location minimum for Wilcoxon tests is hardcoded in `core/evaluation.py`, not in config.

## Consequences

- WAPE values are not directly comparable with standard-formula WAPE published in academic literature.
  Any external comparison must note this denominator convention.
- Per-location WAPE in `compute_metrics()` and global CV WAPE in `compute_cv_metrics()` both
  use the same denominator formula, but they differ in aggregation order (per-location first
  versus global first). These are structurally different quantities and must not be compared directly.
- The asymmetric numerator behavior (which includes all absolute errors when `exclude_zero_actuals=False`,
  but for which the denominator always uses positive actuals) is intentional for the CV path and is documented
  in the `compute_cv_metrics()` docstring.
