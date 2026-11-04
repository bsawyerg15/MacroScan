"""Derived indicators, defined in code: one recipe per derived Indicator.

A recipe is a function named after the Indicator it produces, whose parameter names are the
(lower-cased) Indicators it is computed from; `@derived` registers it in DERIVED on import.
`apply_derivations` runs every recipe in dependency order, so recipes can build on each other,
and `lineage` gives an indicator's dependency tree for display. Windowed building blocks take
a PeriodInterval, e.g. `ma(unemployment, PeriodInterval.months(12))`, so a recipe means the same span at every frequency.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Callable

import pandas as pd

from macroscan.a_core.a2_schema import PeriodInterval
from macroscan.a_core.a3_indicators import Indicator
from macroscan.c_transform.c2_transformations import add, ma_change_pct_yoy, ratio_pct, yoy_pct

DERIVED_SOURCE = "derived"


@dataclass(frozen=True)
class Derivation:
    """How `indicator` is computed: `func` called with one long frame per input, by lower-cased name."""
    indicator: Indicator
    inputs: tuple[Indicator, ...]
    func: Callable[..., pd.DataFrame]
    unit: str | None = None  # None keeps the unit of the first input


DERIVED: dict[Indicator, Derivation] = {}


def derivation(func: Callable[..., pd.DataFrame], unit: str | None = None) -> Derivation:
    """A Derivation read off `func`: its name and parameter names must be lower-cased Indicators."""
    indicator = Indicator.parse(func.__name__.upper())
    inputs = tuple(Indicator.parse(p.upper()) for p in inspect.signature(func).parameters)
    if indicator in inputs:
        raise ValueError(f"{indicator!r} cannot be derived from itself")
    return Derivation(indicator, inputs, func, unit)


def derived(*, unit: str | None = None) -> Callable[[Callable[..., pd.DataFrame]], Callable[..., pd.DataFrame]]:
    """Decorator: register the function in DERIVED as the recipe for the Indicator it is named after."""
    def register(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        d = derivation(func, unit)
        if d.indicator in DERIVED:
            raise ValueError(f"{d.indicator!r} has two recipes")
        DERIVED[d.indicator] = d
        return func
    return register


def ordered(derivations: dict[Indicator, Derivation] = DERIVED) -> list[Derivation]:
    """The derivations, each after any it builds on; raises graphlib.CycleError on a cycle."""
    graph = {i: [x for x in d.inputs if x in derivations] for i, d in derivations.items()}
    return [derivations[i] for i in TopologicalSorter(graph).static_order()]


def lineage(indicator: Indicator, derivations: dict[Indicator, Derivation] = DERIVED) -> dict[Indicator, dict]:
    """`indicator`'s inputs as a nested {input: its inputs} tree; non-derived inputs are leaves ({})."""
    d = derivations.get(indicator)
    return {} if d is None else {i: lineage(i, derivations) for i in d.inputs}


def apply_derivations(
    data: dict[str, pd.DataFrame], derivations: dict[Indicator, Derivation] = DERIVED
) -> dict[str, pd.DataFrame]:
    """`data` ({frequency: long frame}) plus each derived indicator; inputs are not modified.

    A recipe runs at every frequency where all its inputs have data, and yields nothing elsewhere.
    """
    out = dict(data)
    for d in ordered(derivations):
        for frequency, df in list(out.items()):
            frames = {i: df[df["indicator"] == i] for i in d.inputs}
            if any(f.empty for f in frames.values()):
                continue
            rows = d.func(**{i.lower(): f for i, f in frames.items()})
            rows = rows.drop(columns="pulled_at", errors="ignore").assign(
                source=DERIVED_SOURCE,
                indicator=d.indicator,
                status=None,
                series_key=f"{DERIVED_SOURCE}:{d.indicator}",
                **({"unit": d.unit} if d.unit else {}),
            )
            out[frequency] = pd.concat([df, rows[[c for c in df.columns if c in rows]]], ignore_index=True)
    return out


# --- Recipes ---------------------------------------------------------------------------------


@derived(unit="%")
def real_growth(real_gdp):
    return yoy_pct(real_gdp)


# Latest quarter vs the 18 months before it: REAL_GDP is quarterly, so the front window can't be shorter than 3 months.
@derived(unit="%")
def real_growth_ma_change(real_gdp):
    return ma_change_pct_yoy(real_gdp, PeriodInterval.months(3), PeriodInterval.months(18))


@derived(unit="%")
def nominal_growth(nominal_gdp):
    return yoy_pct(nominal_gdp)


@derived()
def bus_debt_level(bus_loans, bus_debt_securities):
    return add(bus_loans, bus_debt_securities)


# NOMINAL_GDP is pulled at annual levels (TRANSFORMATION=LA), so a quarter's debt stock divides by it directly.
@derived(unit="%")
def hh_debt_level_gdp(hh_debt_level, nominal_gdp):
    return ratio_pct(hh_debt_level, nominal_gdp)


@derived(unit="%")
def bus_debt_level_gdp(bus_debt_level, nominal_gdp):
    return ratio_pct(bus_debt_level, nominal_gdp)


@derived(unit="%")
def govt_debt_level_gdp(govt_debt_level, nominal_gdp):
    return ratio_pct(govt_debt_level, nominal_gdp)


__all__ = ["DERIVED", "Derivation", "apply_derivations", "derivation", "derived", "lineage", "ordered"]
