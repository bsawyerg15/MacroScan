"""Pull the ingested data into the transform stage.

Reads today's values from the b_ingest store, one long frame per frequency
(periods of different frequencies can't share a column dtype). Percent series are
stored as the source publishes them (5.0 for 5%) and turned into decimals (0.05) here.
Annual-only series listed in UPSAMPLED are also given rows at a higher frequency, so
recipes at that frequency can use them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macroscan.a_core.a2_schema import PANDAS_FREQ
from macroscan.a_core.a3_indicators import Indicator
from macroscan.b_ingest.b3_store import DEFAULT_ROOT, Store
from macroscan.c_transform.c2_transformations import upsample_loglinear

PERCENT_UNIT = "%"

# Annual-only indicators -> the frequency they are also loaded at, interpolated log-linearly.
UPSAMPLED: dict[Indicator, str] = {Indicator.PGDP: "Q"}


def load_ingested(data_dir: Path = DEFAULT_ROOT, frequencies: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Latest ingested data as {frequency: long frame}, skipping frequencies with nothing stored."""
    store = Store(data_dir)
    wanted = list(frequencies or PANDAS_FREQ)
    # Annual data is read whenever an upsampled indicator's target frequency is wanted.
    reads = wanted + ["A"] if "A" not in wanted and set(UPSAMPLED.values()) & set(wanted) else wanted
    frames = {f: to_decimal_percents(store.read_all_latest(f)) for f in reads}
    frames = upsample(frames)
    return {f: df for f, df in frames.items() if f in wanted and len(df)}


def upsample(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """`frames` plus each UPSAMPLED indicator's annual rows at its target frequency."""
    out = dict(frames)
    annual = frames.get("A")
    for indicator, frequency in UPSAMPLED.items():
        rows = annual[annual["indicator"] == indicator] if annual is not None and len(annual) else None
        if rows is None or rows.empty:
            continue
        target = out.get(frequency)
        up = upsample_loglinear(rows, frequency)
        out[frequency] = up if target is None or target.empty else pd.concat([target, up[target.columns]], ignore_index=True)
    return out


def is_percent_unit(unit: pd.Series) -> pd.Series:
    """Whether each unit is a source percent code: OECD's PA (% per annum) or PT_* (% of something)."""
    return (unit == "PA") | unit.str.startswith("PT_", na=False)


def to_decimal_percents(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with percent values divided by 100 and their unit set to '%'."""
    pct = is_percent_unit(df["unit"])
    return df.assign(value=df["value"].where(~pct, df["value"] / 100), unit=df["unit"].where(~pct, PERCENT_UNIT))


__all__ = ["PERCENT_UNIT", "UPSAMPLED", "load_ingested", "upsample", "is_percent_unit", "to_decimal_percents"]
