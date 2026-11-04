from datetime import datetime, timezone

import numpy as np
import pandas as pd

from macroscan.a_core.a1_config import TableRow
from macroscan.a_core.a2_schema import FETCH_COLUMNS, parse_periods
from macroscan.a_core.a3_indicators import Indicator
from macroscan.b_ingest.b3_store import Store
from macroscan.c_transform.c1_load import load_ingested
from macroscan.d_visualization.d1_frames import (
    INDENT,
    chart_series,
    header_labels,
    is_percent_row,
    latest_table,
    unindent,
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
