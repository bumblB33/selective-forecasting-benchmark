from __future__ import annotations

import re
from pathlib import Path


def find_latest_timestamp(metrics_dir: Path, prefix: str) -> str | None:
    """Return the most recent timestamp string found in location_metrics files."""
    timestamps = []
    for path in metrics_dir.glob(f"{prefix}location_metrics_*.csv"):
        match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})\.csv$", path.name)
        if match:
            timestamps.append(match.group(1))
    return max(timestamps) if timestamps else None
