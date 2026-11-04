"""Building blocks for derived indicators: pure functions over canonical long frames.

Each takes one indicator's rows at one frequency (any number of countries) and returns rows
in the same shape; `c3_derived` composes them into recipes and stamps the metadata.
Windowed operations take a PeriodInterval, so one recipe means the same span at every frequency.
Binary operations align on (country, period); periods missing from either side drop out,
and inputs of different frequencies raise ValueError.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from macroscan.a_core.a2_schema import PERIODS_PER_YEAR, PeriodInterval, to_wide

KEYS = ["country", "period"]


def yoy_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Y/Y % change per country; periods with no year-earlier value drop out."""
    year_ago = df[[*KEYS, "value"]].assign(period=df["period"] + PERIODS_PER_YEAR[df["frequency"].iloc[0]])
    return _combine(df, year_ago, lambda now, then: (now / then - 1) * 100)


def ma(df: pd.DataFrame, window: PeriodInterval) -> pd.DataFrame:
    """Trailing mean over `window` per country; windows spanning a missing period drop out.

    Yields no rows at a frequency the window doesn't divide into (e.g. 3 months on annual data).
    """
    periods = window.periods_at(df["frequency"].iloc[0])
    if periods is None:
        return df.iloc[0:0]
    wide = to_wide(df)
    # Fill in missing periods as NaN so rolling counts periods, not rows, and never spans a gap.
    full = pd.period_range(wide.index.min(), wide.index.max(), freq=wide.index.freq, name="period")
    means = wide.reindex(full).rolling(periods).mean().stack().rename("value").reset_index()
    return _combine(df, means, lambda _, mean: mean)


def ma_change_pct_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """% change (as a decimal) of the latest 1-month mean vs the mean of the 18 months before it.

    Computed in logs, scaled by `annualization_factor`, then turned back into a % change.

    Yields no rows at a frequency the windows don't divide into.
    """
    front = PeriodInterval.months(1)
    back = PeriodInterval.months(18)
    frequency = df["frequency"].iloc[0]
    front_periods, back_periods = front.periods_at(frequency), back.periods_at(frequency)
    if front_periods is None or back_periods is None:
        return df.iloc[0:0]
    # Move each back-end mean onto the period whose front-end window starts right after it.
    back_means = ma(df, back)
    back_means = back_means.assign(period=back_means["period"] + front_periods)
    # TODO: evaluate annualizing, e.g. 12 / ((front + back months) / 2) to scale by the gap between window midpoints.
    annualization_factor = 1
    return _combine(
        ma(df, front), back_means, lambda now, then: np.exp((np.log(now) - np.log(then)) * annualization_factor) - 1
    )


def add(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a + b per (country, period)."""
    return _combine(a, b, lambda x, y: x + y)


def sub(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a - b per (country, period)."""
    return _combine(a, b, lambda x, y: x - y)


def ratio_pct(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """a as a % of b per (country, period)."""
    return _combine(a, b, lambda x, y: x / y * 100)


def _combine(a: pd.DataFrame, b: pd.DataFrame, op: Callable[[pd.Series, pd.Series], pd.Series]) -> pd.DataFrame:
    """a's rows where b has the same (country, period), with value = op(a, b); NaN results drop out."""
    # Periods of different frequencies never compare equal, so a mismatch would silently match nothing.
    if a["period"].dtype != b["period"].dtype:
        raise ValueError(f"cannot combine {a['period'].dtype} with {b['period'].dtype}; convert to one frequency first")
    merged = a.merge(b[[*KEYS, "value"]], on=KEYS, suffixes=("", "_b"))
    out = merged.assign(value=op(merged["value"], merged["value_b"])).drop(columns="value_b")
    return out.dropna(subset=["value"]).reset_index(drop=True)


__all__ = ["yoy_pct", "ma", "ma_change_pct_yoy", "add", "sub", "ratio_pct"]
