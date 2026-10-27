# ADR 001: Core Module Must Remain Pipeline-Agnostic

Status: Accepted
Date: 2026-03-27

## Context

The codebase is structured into two benchmark pipelines (`daily/` and `interval/`) that share
mathematical logic, including metric computation, preprocessing, and model utilities. Without a clear boundary,
shared logic would drift into pipeline-specific modules. Changes to common formulas would then require
parallel edits, and the two pipelines would silently diverge over time.

## Decision

All shared logic lives in `core/`. The `core/` module must not import from `daily.config`,
`interval.config`, or any other pipeline-specific module. Pipeline-specific callers pass
configuration values as function arguments; `core/` never reads configuration directly.

Pipeline modules (`daily/` and `interval/`) may import from `core/` freely. `core/` may not import
from any pipeline module. Pipeline modules import `core/` functions directly; they do not re-export them.

The constraint is enforced by `tests/test_architecture_constraints.py::TestCoreModulePipelineAgnostic`.

## Consequences

- Adding a new shared function requires placing it in `core/` and updating callers to pass configuration values.
- Pipeline-specific behavior is encoded at the call site, not inside the shared function.
- `core/preprocessing.py::add_fourier_features()` is daily-only. Its default `frequency="D"` and
  `forecast_horizon=7` reflect daily pipeline conventions. The `create_train_test_split()` default of
  `max_train_history_weeks=105` is the daily value; interval callers pass 26.
