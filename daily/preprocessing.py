from __future__ import annotations

import logging

import pandas as pd

from core.preprocessing import (
    add_calendar_features,
    add_fourier_features,
    clip_outliers,
    create_train_test_split,
    fill_missing_timesteps,
)

__all__ = [
    "add_calendar_features",
    "add_fourier_features",
    "clip_outliers",
    "create_train_test_split",
    "fill_missing_dates",
]
from . import config

logger = logging.getLogger(__name__)


def fill_missing_dates(nixtla_df: pd.DataFrame) -> pd.DataFrame:
    """Daily-frequency zero-fill for absent dates. Source is deduplicated upstream."""
    return fill_missing_timesteps(nixtla_df, freq=config.FREQUENCY, deduplicate=False)
