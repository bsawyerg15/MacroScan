"""Pull the ingested data into the transform stage.

Reads today's values from the b_ingest store, one long frame per frequency
(periods of different frequencies can't share a column dtype). Percent series are
stored as the source publishes them (5.0 for 5%) and turned into decimals (0.05) here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macroscan.a_core.a2_schema import PANDAS_FREQ
from macroscan.b_ingest.b3_store import DEFAULT_ROOT, Store

PERCENT_UNIT = "%"


def load_ingested(data_dir: Path = DEFAULT_ROOT, frequencies: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Latest ingested data as {frequency: long frame}, skipping frequencies with nothing stored."""
    store = Store(data_dir)
    frames = {f: store.read_all_latest(f) for f in frequencies or PANDAS_FREQ}
    return {f: to_decimal_percents(df) for f, df in frames.items() if len(df)}


def is_percent_unit(unit: pd.Series) -> pd.Series:
    """Whether each unit is a source percent code: OECD's PA (% per annum) or PT_* (% of something)."""
    return (unit == "PA") | unit.str.startswith("PT_", na=False)


def to_decimal_percents(df: pd.DataFrame) -> pd.DataFrame:
    """`df` with percent values divided by 100 and their unit set to '%'."""
    pct = is_percent_unit(df["unit"])
    return df.assign(value=df["value"].where(~pct, df["value"] / 100), unit=df["unit"].where(~pct, PERCENT_UNIT))


__all__ = ["PERCENT_UNIT", "load_ingested", "is_percent_unit", "to_decimal_percents"]
