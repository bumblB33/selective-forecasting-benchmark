from __future__ import annotations

import datetime

import pandas as pd


# Reference epoch for the downstream tick representation. Override with your
# own system's epoch. The example uses .NET DateTime ticks (0001-01-01 UTC,
# 100-nanosecond resolution).
_DOWNSTREAM_EPOCH = datetime.datetime(1, 1, 1, tzinfo=datetime.timezone.utc)
_UNIX_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

_TICKS_PER_SECOND: int = 10_000_000
_TICKS_AT_UNIX_EPOCH: int = (
    int((_UNIX_EPOCH - _DOWNSTREAM_EPOCH).total_seconds()) * _TICKS_PER_SECOND
)

_TICKS_PER_15_MINUTES: int = 15 * 60 * _TICKS_PER_SECOND
_TICKS_PER_HOUR: int = 60 * 60 * _TICKS_PER_SECOND


def nixtla_ds_to_slot_id(utc_timestamp: pd.Timestamp) -> int:
    """Convert a UTC Nixtla ``ds`` to a 15-minute integer tick key.

    Reference for adapting Nixtla UTC datetimes to a downstream tick format
    with a non-Unix epoch. Floors to the slot start so sub-15-minute precision
    round-trips stably through ``slot_id_to_nixtla_ds``.
    """
    if utc_timestamp.tzinfo is None:
        raise ValueError(
            f"Timestamp {utc_timestamp!r} is timezone-naive. "
            "All Nixtla 'ds' values must be UTC-aware before conversion. "
            "Use ts.tz_localize('UTC') if you are certain the value is already UTC."
        )
    utc_normalized = utc_timestamp.tz_convert("UTC")
    unix_seconds = int(utc_normalized.timestamp())
    total_ticks = _TICKS_AT_UNIX_EPOCH + unix_seconds * _TICKS_PER_SECOND
    return (total_ticks // _TICKS_PER_15_MINUTES) * _TICKS_PER_15_MINUTES


def nixtla_ds_to_period_id(utc_timestamp: pd.Timestamp) -> int:
    """Hourly variant of ``nixtla_ds_to_slot_id`` — floors to the hour boundary."""
    if utc_timestamp.tzinfo is None:
        raise ValueError(
            f"Timestamp {utc_timestamp!r} is timezone-naive. Provide a UTC-aware timestamp."
        )
    utc_normalized = utc_timestamp.tz_convert("UTC")
    unix_seconds = int(utc_normalized.timestamp())
    total_ticks = _TICKS_AT_UNIX_EPOCH + unix_seconds * _TICKS_PER_SECOND
    return (total_ticks // _TICKS_PER_HOUR) * _TICKS_PER_HOUR


def slot_id_to_nixtla_ds(slot_id: int) -> pd.Timestamp:
    """Inverse of ``nixtla_ds_to_slot_id``; returns a UTC-aware timestamp."""
    unix_ticks = slot_id - _TICKS_AT_UNIX_EPOCH
    unix_seconds = unix_ticks // _TICKS_PER_SECOND
    return pd.Timestamp(unix_seconds, unit="s", tz="UTC")


def verify_round_trip(utc_timestamp: pd.Timestamp) -> None:
    """Assert timestamp → slot_id → timestamp is lossless.

    Failure means the input is not aligned to the 15-minute boundary —
    confirm the Nixtla pipeline uses ``freq='15min'`` end-to-end.
    """
    slot_id = nixtla_ds_to_slot_id(utc_timestamp)
    reconstructed = slot_id_to_nixtla_ds(slot_id)
    if reconstructed != utc_timestamp:
        raise AssertionError(
            f"Round-trip failed for {utc_timestamp!r}: "
            f"slot_id={slot_id}, reconstructed={reconstructed!r}. "
            "The timestamp likely has sub-15-minute precision."
        )


def translate_predictions_to_downstream_keys(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add downstream ``slot_id`` (15-min) and ``period_id`` (hourly) columns.

    Does not translate ``unique_id`` — that requires a separate entity map.
    """
    required_columns = {"unique_id", "ds", "y"}
    missing = required_columns - set(predictions_df.columns)
    if missing:
        raise ValueError(
            f"predictions_df is missing required Nixtla columns: {sorted(missing)}. "
            "Expected columns: unique_id, ds, y."
        )
    result = predictions_df.copy()
    result["slot_id"] = result["ds"].apply(nixtla_ds_to_slot_id)
    result["period_id"] = result["ds"].apply(nixtla_ds_to_period_id)
    return result
