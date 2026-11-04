from graphlib import CycleError

import pandas as pd
import pytest

from macroscan.a_core.a2_schema import FETCH_COLUMNS, PeriodInterval, parse_periods
from macroscan.a_core.a3_indicators import Indicator
from macroscan.c_transform.c2_transformations import ma, yoy_pct
from macroscan.c_transform.c3_derived import (
    DERIVED,
    DERIVED_SOURCE,
    apply_derivations,
    derivation,
    lineage,
    ordered,
)


def frame(frequency, periods, values, indicator=Indicator.REAL_GDP, country="USA"):
    """One country's series in the canonical long format."""
    return pd.DataFrame(
        {
            "source": "oecd",
            "indicator": indicator,
            "country": country,
            "period": parse_periods(pd.Series(periods), frequency),
            "frequency": frequency,
            "value": values,
            "unit": "XDC",
            "status": "A",
            "series_key": "x",
        }
    )[FETCH_COLUMNS]


def test_derivation_reads_name_and_inputs_from_signature():
    """The function name is the derived Indicator; parameter names are its inputs, in order."""
    def bus_debt_level_gdp(bus_debt_level, nominal_gdp):
        ...

    d = derivation(bus_debt_level_gdp, unit="%")
    assert d.indicator is Indicator.BUS_DEBT_LEVEL_GDP
    assert d.inputs == (Indicator.BUS_DEBT_LEVEL, Indicator.NOMINAL_GDP)


def test_derivation_rejects_unknown_or_self_referencing_names():
    """Names that are not Indicators, or a recipe that needs its own output, fail at registration."""
    def not_an_indicator(real_gdp):
        ...

    def real_growth(real_gdpp):
        ...

    def pgdp(pgdp):
        ...

    for func in (not_an_indicator, real_growth, pgdp):
        with pytest.raises(ValueError):
            derivation(func)


def test_registry_is_ordered_and_acyclic():
    """Every registered recipe comes after the recipes it builds on."""
    seen = set()
    for d in ordered():
        assert not (set(d.inputs) & set(DERIVED)) - seen
        seen.add(d.indicator)
    assert seen == set(DERIVED)


def test_ordered_rejects_cycles():
    """Two recipes that need each other raise CycleError."""
    def pgdp(pgdp_growth):
        ...

    def pgdp_growth(pgdp):
        ...

    with pytest.raises(CycleError):
        ordered({d.indicator: d for d in (derivation(pgdp), derivation(pgdp_growth))})


def test_lineage_is_a_nested_tree():
    """A chained recipe's lineage reaches down to the ingested indicators."""
    assert lineage(Indicator.BUS_DEBT_LEVEL_GDP) == {
        Indicator.BUS_DEBT_LEVEL: {Indicator.BUS_LOANS: {}, Indicator.BUS_DEBT_SECURITIES: {}},
        Indicator.NOMINAL_GDP: {},
    }
    assert lineage(Indicator.REAL_GDP) == {}


def test_apply_derivations_chains_recipes():
    """A recipe's output feeds the next; derived rows are stamped and the input is untouched."""
    q = pd.concat(
        [
            frame("Q", ["2025Q1"], [30], indicator=Indicator.BUS_LOANS),
            frame("Q", ["2025Q1"], [10], indicator=Indicator.BUS_DEBT_SECURITIES),
            frame("Q", ["2025Q1"], [200], indicator=Indicator.NOMINAL_GDP),
        ],
        ignore_index=True,
    )
    data = {"Q": q}
    out = apply_derivations(data)["Q"].set_index("indicator")
    assert len(data["Q"]) == 3
    assert out.at[Indicator.BUS_DEBT_LEVEL, "value"] == pytest.approx(40)
    assert out.at[Indicator.BUS_DEBT_LEVEL, "unit"] == "XDC"
    assert out.at[Indicator.BUS_DEBT_LEVEL_GDP, "value"] == pytest.approx(20.0)
    assert out.at[Indicator.BUS_DEBT_LEVEL_GDP, "unit"] == "%"
    assert out.at[Indicator.BUS_DEBT_LEVEL_GDP, "source"] == DERIVED_SOURCE
    assert out.at[Indicator.BUS_DEBT_LEVEL_GDP, "series_key"] == f"{DERIVED_SOURCE}:BUS_DEBT_LEVEL_GDP"
    assert Indicator.HH_DEBT_LEVEL_GDP not in out.index  # HH_DEBT_LEVEL is missing


def test_apply_derivations_runs_per_frequency():
    """A recipe runs at each frequency holding its inputs; others are left as they were."""
    def real_growth(real_gdp):
        return yoy_pct(real_gdp)

    recipes = {Indicator.REAL_GROWTH: derivation(real_growth, unit="%")}
    data = {
        "Q": frame("Q", ["2024Q1", "2025Q1"], [100, 110]),
        "M": frame("M", ["2025-01"], [1.0], indicator=Indicator.UNEMPLOYMENT),
    }
    out = apply_derivations(data, recipes)
    assert out["M"].equals(data["M"])
    derived = out["Q"][out["Q"]["indicator"] == Indicator.REAL_GROWTH]
    assert derived["value"].tolist() == pytest.approx([10.0])
    assert out["Q"]["period"].dtype == pd.PeriodDtype("Q")


def test_apply_derivations_with_ma_skips_frequencies_the_window_does_not_fit():
    """A 3-month MA recipe yields rows on monthly data and nothing, without error, on annual data."""
    def real_growth(real_gdp):
        return ma(real_gdp, PeriodInterval.months(3))

    recipes = {Indicator.REAL_GROWTH: derivation(real_growth)}
    data = {
        "M": frame("M", ["2025-01", "2025-02", "2025-03"], [1.0, 2.0, 3.0]),
        "A": frame("A", ["2024", "2025"], [100, 110]),
    }
    out = apply_derivations(data, recipes)
    derived = out["M"][out["M"]["indicator"] == Indicator.REAL_GROWTH]
    assert derived["value"].tolist() == pytest.approx([2.0])
    assert not (out["A"]["indicator"] == Indicator.REAL_GROWTH).any()
