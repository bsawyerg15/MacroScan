"""Check a source's fetch() output matches the canonical schema before storing it."""

from __future__ import annotations

import pandas as pd

from macroscan.a_core.a1_config import SeriesSpec
from macroscan.a_core.a2_schema import FETCH_COLUMNS, PANDAS_FREQ


class SchemaError(ValueError):
    """Raised when a fetched frame doesn't match the canonical schema."""


def validate_frame(df: pd.DataFrame, spec: SeriesSpec) -> pd.DataFrame:
    """Check a fetched frame against the canonical schema; return it with columns in order.

    Requires the spec's source, indicator and frequency on every row, a matching Period
    dtype, float values, and unique (country, period) rows.
    """
    label = spec.label
    missing = [c for c in FETCH_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"{label}: missing columns {missing}")
    df = df[FETCH_COLUMNS]
    if df.empty:
        return df

    for col in ("source", "indicator", "frequency"):
        found = set(df[col])
        if found != {getattr(spec, col)}:
            raise SchemaError(f"{label}: expected {col} {getattr(spec, col)!r}, frame has {sorted(found)}")
    if df["period"].dtype != pd.PeriodDtype(PANDAS_FREQ[spec.frequency]):
        raise SchemaError(f"{label}: period dtype {df['period'].dtype} does not match frequency {spec.frequency}")
    if not pd.api.types.is_float_dtype(df["value"]):
        raise SchemaError(f"{label}: value must be float, got {df['value'].dtype}")
    if df["country"].isna().any() or df["period"].isna().any():
        raise SchemaError(f"{label}: country and period must not be null")
    dupes = df.duplicated(["country", "period"])
    if dupes.any():
        sample = df.loc[dupes, ["country", "period"]].head(3).astype(str).values.tolist()
        raise SchemaError(f"{label}: duplicate (country, period) rows, e.g. {sample}")
    return df
