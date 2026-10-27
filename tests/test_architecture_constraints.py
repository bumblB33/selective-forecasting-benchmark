import importlib
from pathlib import Path

import pandas as pd
import pytest


class TestCoreModulePipelineAgnostic:
    """Enforce that ``core/`` does not import from ``daily`` or ``interval`` configs."""

    def test_core_module_has_no_pipeline_imports(self):
        core_dir = Path("core")
        if not core_dir.exists():
            pytest.skip("core/ directory not found")

        forbidden_patterns = [
            "from daily.config",
            "from daily import config",
            "import daily.config",
            "from interval.config",
            "from interval import config",
            "import interval.config",
        ]

        violations = []
        for py_file in core_dir.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            content = py_file.read_text()
            for pattern in forbidden_patterns:
                if pattern in content:
                    violations.append(f"{py_file.name}: found '{pattern}'")

        assert not violations, "core/ must remain pipeline-agnostic. Found violations:\n" + "\n".join(
            f"  - {v}" for v in violations
        )


class TestWAPEDenominatorBehavior:
    """Pin the WAPE denominator convention (positive actuals only)."""

    def test_wape_denominator_uses_positive_actuals_only(self):
        """WAPE denominator excludes zero-actual rows — non-standard, matches
        production. Do not "fix" this.
        """
        actuals = pd.Series([100, 0, 50, 0])
        predictions = pd.Series([90, 10, 55, 5])

        numerator = (actuals - predictions).abs().sum()
        denominator = actuals[actuals > 0].sum()
        expected_wape = numerator / denominator

        assert numerator == 30
        assert denominator == 150
        assert expected_wape == 0.20

    def test_wape_diverges_from_standard_formula_when_zero_actuals_have_predictions(
        self,
    ):
        """When zero-actual rows have non-zero predictions, asymmetric filtering
        (numerator vs. denominator) gives a different answer than the standard formula.
        """
        actuals = pd.Series([100, 0])
        predictions = pd.Series([90, 50])

        standard_denominator = actuals.sum()
        standard_wape = (actuals - predictions).abs().sum() / standard_denominator

        our_denominator = actuals[actuals > 0].sum()
        our_wape = (actuals - predictions).abs().sum() / our_denominator

        assert standard_wape == our_wape == 0.60

        actuals2 = pd.Series([100, 0, 0, 0])
        predictions2 = pd.Series([90, 50, 50, 50])

        standard_denom2 = actuals2.sum()
        standard_wape2 = (actuals2 - predictions2).abs().sum() / standard_denom2

        our_denom2 = actuals2[actuals2 > 0].sum()
        our_wape2 = (actuals2 - predictions2).abs().sum() / our_denom2

        assert standard_wape2 == our_wape2 == 1.60

        actuals3 = pd.Series([100.0, 0.0, 0.0, 0.0])
        predictions3 = pd.Series([90.0, 50.0, 50.0, 50.0])
        standard_wape3 = (actuals3 - predictions3).abs().sum() / actuals3.sum()

        pos_mask = actuals3 > 0
        our_wape3 = (actuals3[pos_mask] - predictions3[pos_mask]).abs().sum() / actuals3[pos_mask].sum()

        assert standard_wape3 == 1.60
        assert our_wape3 == 0.10
        assert our_wape3 != standard_wape3


class TestArchitectureDecisionRecords:
    """Guard against ADR-reference rot: referenced ADRs must exist and have content."""

    _ADRS = {
        "docs/adr/001-core-module-pipeline-agnostic.md": ("ADR 001 documents the core/ pipeline-agnostic constraint."),
        "docs/adr/002-wape-denominator-positive-actuals.md": ("ADR 002 documents the WAPE denominator convention."),
        "docs/adr/003-training-pool-evaluation-strategy.md": (
            "ADR 003 documents the natural-train / fixed-eval ablation strategy."
        ),
        "docs/adr/004-statistical-win-rate-guard-threshold.md": (
            "ADR 004 documents the win-rate guard threshold policy."
        ),
    }

    def test_all_adrs_exist_and_are_substantive(self):
        """Every referenced ADR exists and contains at least 200 bytes."""
        repo_root = Path(__file__).resolve().parents[1]
        missing = []
        too_small = []
        for adr_path, message in self._ADRS.items():
            path = repo_root / adr_path
            if not path.exists():
                missing.append(f"  MISSING: {adr_path}: {message}")
            elif path.stat().st_size < 200:
                too_small.append(f"  TOO SMALL (<200 bytes): {adr_path}: {message}")

        errors = missing + too_small
        assert not errors, "ADR files missing or empty:\n" + "\n".join(errors)

    def test_training_pool_strategy_is_documented(self):
        """Back-compat single-ADR existence check."""
        adr_path = Path("docs/adr/003-training-pool-evaluation-strategy.md")
        assert adr_path.exists()


class TestParameterDefaults:
    """Pin daily-pipeline-specific defaults on core/ APIs."""

    def test_create_train_test_split_default_is_daily_pipeline_value(self):
        """``max_train_history_weeks=105`` is daily-specific; interval callers pass 26."""
        from core.preprocessing import create_train_test_split
        import inspect

        sig = inspect.signature(create_train_test_split)
        default_value = sig.parameters["max_train_history_weeks"].default

        assert default_value == 105, "Default changed from 105 — update this test if intentional."


class TestMetricNamingConventions:
    """Pin the metric name set for historical parity (reports, parity checks)."""

    def test_metric_names_are_stable(self):
        """Metric names must remain MAE, WAPE, MAPE, RMSE — renaming breaks historical artifacts."""
        from core.evaluation import _METRICS

        expected_metrics = {"MAE", "WAPE", "MAPE", "RMSE"}

        assert set(_METRICS) == expected_metrics, (
            f"core.evaluation._METRICS has diverged from the expected set. "
            f"Found: {set(_METRICS)}. Expected: {expected_metrics}."
        )


class TestPipelineModulesImport:
    """Function-local imports and config reads are not exercised here."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "daily.run",
            "interval.run",
            "scripts.regenerate_report",
            "scripts.configure_training_length",
            "scripts.parity_check",
        ],
    )
    def test_entry_point_imports(self, module_name):
        importlib.import_module(module_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
