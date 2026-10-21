"""Check a source's fetch() output matches the canonical schema before storing it."""

from __future__ import annotations

import pandas as pd

from macroscan.a_core.a2_schema import FETCH_COLUMNS, PANDAS_FREQ


class SchemaError(ValueError):
    """Raised when a fetched frame doesn't match the canonical schema."""


def validate_frame(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """Check a fetched frame against the canonical schema; return it with columns in order.

    Requires one indicator, one supported frequency, a matching Period dtype, float
    values, and unique (country, period) rows.
    """
    missing = [c for c in FETCH_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{indicator}: missing columns {missing}")
    df = df[FETCH_COLUMNS]
    if df.empty:
        return df

    if set(df["indicator"]) != {indicator}:
        raise SchemaError(f"{indicator}: frame contains other indicators {sorted(set(df['indicator']))}")
    freqs = df["frequency"].unique().tolist()
    if len(freqs) != 1 or freqs[0] not in PANDAS_FREQ:
        raise SchemaError(f"{indicator}: expected a single frequency in {sorted(PANDAS_FREQ)}, got {freqs}")
    if not isinstance(df["period"].dtype, pd.PeriodDtype):
        raise SchemaError(f"{indicator}: period must be a pandas Period column, got {df['period'].dtype}")
    if df["period"].dtype != pd.PeriodDtype(PANDAS_FREQ[freqs[0]]):
        raise SchemaError(f"{indicator}: period dtype {df['period'].dtype} does not match frequency {freqs[0]}")
    if not pd.api.types.is_float_dtype(df["value"]):
        raise SchemaError(f"{indicator}: value must be float, got {df['value'].dtype}")
    if df["country"].isna().any() or df["period"].isna().any():
        raise SchemaError(f"{indicator}: country and period must not be null")
    dupes = df.duplicated(["country", "period"])
    if dupes.any():
        sample = df.loc[dupes, ["country", "period"]].head(3).astype(str).values.tolist()
        raise SchemaError(f"{indicator}: duplicate (country, period) rows, e.g. {sample}")
    return df
