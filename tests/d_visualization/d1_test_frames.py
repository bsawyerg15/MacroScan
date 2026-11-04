from datetime import datetime, timezone

import numpy as np
import pandas as pd

from macroscan.a_core.a1_config import TableRow
from macroscan.a_core.a2_schema import FETCH_COLUMNS, parse_periods
from macroscan.a_core.a3_indicators import Indicator
from macroscan.b_ingest.b3_store import Store
from macroscan.c_transform.c1_load import load_ingested
from macroscan.d_visualization.d1_frames import (
    COLLAPSED,
    EXPANDED,
    INDENT,
    INPUT,
    LEAF,
    base_label,
    chart_series,
    header_labels,
    is_percent_row,
    latest_table,
    level_of,
    outline,
    unindent,
    visible,
)

T1 = datetime(2026, 1, 15, tzinfo=timezone.utc)


def frame(indicator, frequency, rows):
    """rows: (country, period, value)"""
    df = pd.DataFrame(rows, columns=["country", "period", "value"])
    return pd.DataFrame(
        {
            "source": "oecd",
            "indicator": indicator,
            "country": df["country"],
            "period": parse_periods(df["period"], frequency),
            "frequency": frequency,
            "value": df["value"].astype(float),
            "unit": "XDC",
            "status": "A",
            "series_key": "x",
        }
    )[FETCH_COLUMNS]


def ingested(tmp_path):
    """Quarterly GDP for USA and DEU, monthly unemployment for USA only."""
    store = Store(tmp_path)
    gdp = frame(Indicator.REAL_GDP, "Q", [("USA", "2025Q2", 2.0), ("USA", "2025Q1", 1.0), ("DEU", "2025Q1", 5.0)])
    une = frame(Indicator.UNEMPLOYMENT, "M", [("USA", "2025-01", 4.0), ("USA", "2025-02", 4.1)])
    store.apply(gdp, "oecd", Indicator.REAL_GDP, "Q", pulled_at=T1)
    store.apply(une, "oecd", Indicator.UNEMPLOYMENT, "M", pulled_at=T1)
    return load_ingested(tmp_path)


def test_latest_table(tmp_path):
    """Rows are indicators, columns are countries in config order, cells the last value."""
    table = latest_table(ingested(tmp_path), ["DEU", "USA", "JPN"])
    assert list(table.columns) == ["DEU", "USA", "JPN"]
    assert list(table.index) == ["Real GDP (Q)", "Unemployment (%) (M)"]
    assert table.at["Real GDP (Q)", "USA"] == 2.0  # 2025Q2 beats 2025Q1
    assert table.at["Unemployment (%) (M)", "USA"] == 4.1
    assert np.isnan(table.at["Unemployment (%) (M)", "DEU"]) and table["JPN"].isna().all()


LAYOUT = [
    TableRow(0, header="Output"),
    TableRow(1, indicator=Indicator.UNEMPLOYMENT),
    TableRow(1, indicator=Indicator.REAL_GDP),
    TableRow(0, header="Assets"),
    TableRow(1, indicator=Indicator.GOLD),
]


def test_latest_table_layout(tmp_path):
    """Rows follow the layout, indented by level; headers and data-less indicators are empty rows."""
    table = latest_table(ingested(tmp_path), ["USA", "DEU"], LAYOUT)
    assert list(table.index) == [
        "Output",
        INDENT + "Unemployment (%) (M)",
        INDENT + "Real GDP (Q)",
        "Assets",
        INDENT + "Gold",
    ]
    assert list(table.columns) == ["USA", "DEU"]
    assert table.at[INDENT + "Real GDP (Q)", "DEU"] == 5.0
    assert table.loc[["Output", "Assets", INDENT + "Gold"]].isna().all().all()
    assert header_labels(LAYOUT) == ["Output", "Assets"]


def test_latest_table_layout_without_data():
    """With no data at all, the layout still gives every row, all empty."""
    table = latest_table({}, ["USA"], LAYOUT)
    assert len(table) == len(LAYOUT) and table.isna().all().all()


def test_unindent():
    """Indented labels round-trip to the row labels chart_series uses."""
    assert unindent(INDENT * 2 + "Real GDP (Q)") == "Real GDP (Q)"


def test_chart_series(tmp_path):
    """One named column per pick on a shared timestamp index."""
    chart = chart_series(ingested(tmp_path), [("Real GDP (Q)", "USA"), ("Unemployment (%) (M)", "USA")])
    assert list(chart.columns) == ["USA · Real GDP (Q)", "USA · Unemployment (%) (M)"]
    assert isinstance(chart.index, pd.DatetimeIndex) and chart.index.is_monotonic_increasing
    assert chart["USA · Real GDP (Q)"].dropna().tolist() == [1.0, 2.0]
    assert chart_series({}, []).empty


def test_is_percent_row():
    """Rows whose indicator label has a '%' are percents, including '% GDP' ratios."""
    assert is_percent_row("Unemployment (%) (M)") and is_percent_row("HH Debt Level % GDP (Q)")
    assert not is_percent_row("Real GDP (Q)")


def outlined():
    """A table shaped like the app's: headers, a nested header, and an indicator with a child."""
    labels = ["Output", INDENT + "Real Growth (%) (Q)", INDENT * 2 + "PGDP Growth (%) (A)", INDENT + "Unemployment (%) (M)",
              "Debt", INDENT + "Levels", INDENT * 2 + "HH", INDENT + "Flows", INDENT * 2 + "HH", "Assets", INDENT + "Gold"]
    return pd.DataFrame({"USA": range(len(labels))}, index=pd.Index(labels, name="indicator"), dtype=float)


def test_outline():
    """Paths follow ancestry, so a repeated label under two parents stays distinct; parents have deeper rows next."""
    rows = outline(outlined())
    assert rows["level"].tolist() == [0, 1, 2, 1, 0, 1, 2, 1, 2, 0, 1]
    assert rows["path"].iloc[2] == "Output › Real Growth (%) (Q) › PGDP Growth (%) (A)"
    assert rows["path"].iloc[6] == "Debt › Levels › HH" and rows["path"].iloc[8] == "Debt › Flows › HH"
    assert rows["has_children"].tolist() == [True, True, False, False, True, True, False, True, False, True, False]
    assert level_of(INDENT * 2 + "x") == 2


def shown_labels(collapsed):
    return [unindent(label) for label in visible(outlined(), collapsed).index]


def test_visible():
    """A collapsed row stays but its whole subtree goes; siblings and other sections stay."""
    assert len(shown_labels(set())) == 11
    assert shown_labels({"Debt"}) == ["Output", "Real Growth (%) (Q)", "PGDP Growth (%) (A)", "Unemployment (%) (M)", "Debt", "Assets", "Gold"]
    assert shown_labels({"Debt › Levels"}) == [
        "Output", "Real Growth (%) (Q)", "PGDP Growth (%) (A)", "Unemployment (%) (M)", "Debt", "Levels", "Flows", "HH", "Assets", "Gold"
    ]
    assert "PGDP Growth (%) (A)" not in shown_labels({"Output › Real Growth (%) (Q)"})
    assert shown_labels({"Output", "Debt", "Assets"}) == ["Output", "Debt", "Assets"]


def test_visible_display_labels():
    """Parents get ▾ (or ▸ when collapsed, even though their children are hidden); leaves get a gap."""
    display = visible(outlined(), {"Debt"})["display"].tolist()
    assert display[0] == EXPANDED + "Output"
    assert display[1] == INDENT + EXPANDED + "Real Growth (%) (Q)"
    assert display[2] == INDENT * 2 + LEAF + "PGDP Growth (%) (A)"
    assert display[4] == COLLAPSED + "Debt"


def test_latest_table_provenance():
    """Each layout indicator is followed by its inputs, recursively, valued at the parent cell's period."""
    q = pd.concat([
        frame(Indicator.REAL_GDP, "Q", [("USA", "2025Q1", 1.0), ("USA", "2025Q2", 2.0), ("DEU", "2025Q1", 5.0)]),
        frame(Indicator.REAL_GROWTH, "Q", [("USA", "2025Q1", 0.1)]),  # latest is Q1, though GDP has Q2
        frame(Indicator.NOMINAL_GDP, "Q", [("USA", "2025Q1", 9.0)]),
    ], ignore_index=True)
    inputs = {Indicator.REAL_GROWTH: (Indicator.REAL_GDP,), Indicator.REAL_GDP: (Indicator.NOMINAL_GDP,)}
    layout = [TableRow(0, header="Output"), TableRow(1, indicator=Indicator.REAL_GROWTH)]
    table = latest_table({"Q": q}, ["USA", "DEU"], layout, inputs)
    assert list(table.index) == [
        "Output",
        INDENT + "Real Growth (%) (Q)",
        INDENT * 2 + INPUT + "Real GDP (Q)",
        INDENT * 3 + INPUT + "Nominal GDP (Q)",
    ]
    assert table.iloc[2]["USA"] == 1.0  # the 2025Q1 value the growth cell used, not the latest 2.0
    assert np.isnan(table.iloc[2]["DEU"])  # no growth cell for DEU, so no input period
    assert table.iloc[3]["USA"] == 9.0
    assert base_label(table.index[2]) == "Real GDP (Q)"


def provenance_table():
    """Real Growth with an input (which has its own input) and a configured child that has an input."""
    labels = ["Output", INDENT + "Real Growth (%) (Q)", INDENT * 2 + INPUT + "Real GDP (Q)",
              INDENT * 3 + INPUT + "Nominal GDP (Q)", INDENT * 2 + "PGDP Growth (%) (Q)",
              INDENT * 3 + INPUT + "Real Potential GDP (Q)", INDENT + "Unemployment (%) (M)"]
    return pd.DataFrame({"USA": range(len(labels))}, index=pd.Index(labels, name="indicator"), dtype=float)


def test_outline_inputs():
    """Rows whose first child is an input have inputs; input rows are flagged."""
    rows = outline(provenance_table())
    assert rows["is_input"].tolist() == [False, False, True, True, False, True, False]
    assert rows["has_inputs"].tolist() == [False, True, True, False, True, False, False]


def test_visible_folds_only_inputs():
    """Folding a row with inputs hides just its inputs; configured children and their own inputs stay as set."""
    base = "Output › Real Growth (%) (Q)"
    names = lambda collapsed: [base_label(label) for label in visible(provenance_table(), collapsed).index]
    assert names({base}) == ["Output", "Real Growth (%) (Q)", "PGDP Growth (%) (Q)", "Real Potential GDP (Q)", "Unemployment (%) (M)"]
    assert names({base, base + " › PGDP Growth (%) (Q)"}) == ["Output", "Real Growth (%) (Q)", "PGDP Growth (%) (Q)", "Unemployment (%) (M)"]
    assert names({base + " › " + INPUT + "Real GDP (Q)"}) == [
        "Output", "Real Growth (%) (Q)", "Real GDP (Q)", "PGDP Growth (%) (Q)", "Real Potential GDP (Q)", "Unemployment (%) (M)"
    ]
    assert names({"Output"}) == ["Output"]
