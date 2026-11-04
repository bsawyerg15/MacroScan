"""Derive new indicators from ingested ones, as configured in 03_transformations.csv.

`apply_transformations` is the single entry point: it appends one derived indicator per
TransformSpec, running the matching function on each (country) series of the old indicator.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from macroscan.a_core.a1_config import TransformSpec
from macroscan.a_core.a2_schema import PERIODS_PER_YEAR, Transformation

DERIVED_SOURCE = "derived"


def yoy_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Y/Y % change of one country's series at one frequency; periods with no year-earlier value drop out."""
    value = df.set_index("period")["value"]
    year_ago = value.copy()
    year_ago.index = year_ago.index + PERIODS_PER_YEAR[df["frequency"].iloc[0]]  # align t-n to t; gaps give NaN
    change = ((value / year_ago.reindex(value.index) - 1) * 100).dropna()
    return df.set_index("period").loc[change.index].assign(value=change).reset_index()


# Each function takes one country's series (one indicator, one frequency) and returns the transformed rows.
TRANSFORMS: dict[Transformation, Callable[[pd.DataFrame], pd.DataFrame]] = {Transformation.YOY_PCT: yoy_pct}
UNITS = {Transformation.YOY_PCT: "%"}


def apply_transformations(data: dict[str, pd.DataFrame], specs: list[TransformSpec]) -> dict[str, pd.DataFrame]:
    """`data` ({frequency: long frame}) plus each spec's derived indicator; inputs are not modified.

    Specs apply in order, so a spec may build on an earlier one's output. An old indicator
    with no data yields nothing.
    """
    out = dict(data)
    for spec in specs:
        func = TRANSFORMS[spec.transformation]
        for frequency, df in list(out.items()):
            old = df[df["indicator"] == spec.old_indicator]
            if old.empty:
                continue
            derived = pd.concat([func(s) for _, s in old.groupby("country", sort=False)])
            derived = derived.assign(
                source=DERIVED_SOURCE,
                indicator=spec.new_indicator,
                unit=UNITS[spec.transformation],
                status=None,
                series_key=f"{spec.transformation.value}:" + derived["series_key"],
            )
            if "pulled_at" in derived:
                derived = derived.drop(columns="pulled_at")
            out[frequency] = pd.concat([df, derived[[c for c in df.columns if c in derived]]], ignore_index=True)
    return out


__all__ = ["apply_transformations", "yoy_pct", "TRANSFORMS"]
