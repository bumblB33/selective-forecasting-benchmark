from __future__ import annotations

import pandas as pd

from core.preprocessing import fill_missing_timesteps

from . import config


def fill_missing_dates(nixtla_df: pd.DataFrame) -> pd.DataFrame:
    """Daily-frequency zero-fill for absent dates. Source is deduplicated upstream."""
    return fill_missing_timesteps(nixtla_df, freq=config.FREQUENCY, deduplicate=False)
