from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager

import pandas as pd

logger = logging.getLogger(__name__)


def _cleanup_memory() -> None:
    """Force GC after large model objects are dereferenced."""
    gc.collect()


@contextmanager
def _heartbeat(label: str, interval_seconds: float = 30.0):
    """Log a heartbeat every ``interval_seconds`` so long training runs don't look stalled."""
    started_at = time.time()
    stop_event = threading.Event()

    def _tick():
        while not stop_event.wait(interval_seconds):
            elapsed = time.time() - started_at
            logger.info("  %s still running... %.0fs elapsed", label, elapsed)

    thread = threading.Thread(target=_tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=2)


def derive_available_exog(
    training_df: pd.DataFrame,
    all_exog_columns: list[str],
    mlforecast_models_config: dict,
) -> list[str]:
    """Exog columns present in ``training_df`` excluding those MLForecast generates
    internally via ``date_features`` — preventing duplicate feature columns.

    The exclusion set is read directly from the model config so changes to
    ``date_features`` propagate without separate edits at call sites.
    """
    mlforecast_date_features = {
        feat
        for model_cfg in mlforecast_models_config.values()
        for feat in model_cfg.get("date_features", [])
        if isinstance(feat, str)
    }
    return [col for col in all_exog_columns if col in training_df.columns and col not in mlforecast_date_features]
