# ADR 003: Training Pool Evaluation Strategy

Status: Accepted
Date: 2026-03-27

## Context

The daily benchmark and the training-length ablation study (`scripts/configure_training_length.py`) both
need to decide which set of locations to use when training models versus when evaluating them.

Two strategies are available:

- Natural pool: train on all locations in the dataset, and evaluate on the same full set. The pool size
  shrinks naturally at longer training checkpoints because fewer locations have enough history.
- Fixed pool: hold the location set constant across all training-length checkpoints by restricting
  evaluation to only those locations that qualify at the longest checkpoint. This eliminates composition
  effects when comparing across checkpoints, but it may underrepresent newer locations.

## Decision

The ablation study trains on the natural pool (all available locations at a given checkpoint) and
evaluates on both the natural pool and a fixed pool derived from the longest checkpoint. The main
benchmark (`daily/run.py`) trains and evaluates on the natural pool only.

Training on the natural pool matches production reality: the baseline's global models train on all customers
regardless of which subset is later reported. Restricting training to the fixed pool would understate
the benefit of the additional data available from shorter checkpoints.

The fixed pool is retained in ablation output as a secondary view because it allows clean
apples-to-apples comparisons across training lengths without confounding pool-composition changes.

## Consequences

- `run_ablation()` in `scripts/configure_training_length.py` iterates over both `("natural", None)` and
  `("fixed", fixed_pool_ids)` evaluation pools.
- The `pool_type` column in ablation CSVs distinguishes the two views.
- Wilcoxon tests in the ablation study may be less reliable for fixed-pool runs at the longest checkpoint
  when fewer than 30 locations qualify. A warning is emitted when this occurs.
- The main benchmark does not expose a `pool_type` concept; it always uses the natural pool.
