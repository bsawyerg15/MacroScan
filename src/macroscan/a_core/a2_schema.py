"""The canonical long-format table every source produces, plus period helpers.

One row per (indicator, frequency, country, period). `indicator` is the
conceptual indicator name from 00_indicators.csv; `frequency` is a separate dimension. `period` is a pandas Period column;
there is deliberately no datetime column. Convert with `.dt.to_timestamp()` only
at the chart boundary.
"""

from __future__ import annotations

import pandas as pd

# Columns a source's fetch() must return, in order.
FETCH_COLUMNS = [
    "source",
    "indicator",
    "country",
    "period",
    "frequency",
    "value",
    "unit",
    "status",  # optional flag (provisional, estimated...); null when a source has none
    "series_key",
]
# Stored rows additionally record when each value was first seen.
STORED_COLUMNS = [*FETCH_COLUMNS, "pulled_at"]

# Our frequency codes (SDMX style) -> pandas period aliases.
PANDAS_FREQ = {"A": "Y", "Q": "Q", "M": "M"}


def parse_periods(values: pd.Series | list[str], frequency: str) -> pd.Series:
    """Parse period strings ('2024', '2024-Q1', '2024Q1', '2024-03') into a Period Series."""
    if frequency not in PANDAS_FREQ:
        raise ValueError(f"unsupported frequency {frequency!r}; expected one of {sorted(PANDAS_FREQ)}")
    text = pd.Series(values, dtype="string").str.replace("-Q", "Q", regex=False)
    return pd.Series(pd.PeriodIndex(text, freq=PANDAS_FREQ[frequency]), index=values.index if isinstance(values, pd.Series) else None)


def to_wide(df: pd.DataFrame, columns: str = "country") -> pd.DataFrame:
    """Pivot a long frame for one indicator at one frequency into a PeriodIndex x country matrix."""
    if df["frequency"].nunique() > 1:
        raise ValueError("to_wide expects one frequency; filter on frequency first")
    if df["indicator"].nunique() > 1 and columns != "indicator":
        raise ValueError("to_wide expects one indicator; filter first or pass columns='indicator'")
    wide = df.pivot_table(index="period", columns=columns, values="value", aggfunc="first")
    wide.columns.name = columns
    return wide.sort_index()
