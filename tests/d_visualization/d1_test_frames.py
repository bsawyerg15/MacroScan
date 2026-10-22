from datetime import datetime, timezone

import numpy as np
import pandas as pd

from macroscan.a_core.a2_schema import FETCH_COLUMNS, parse_periods
from macroscan.b_ingest.b3_store import Store
from macroscan.c_transform.c1_load import load_ingested
from macroscan.d_visualization.d1_frames import chart_series, latest_table

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
    gdp = frame("Real GDP", "Q", [("USA", "2025Q2", 2.0), ("USA", "2025Q1", 1.0), ("DEU", "2025Q1", 5.0)])
    une = frame("Unemployment", "M", [("USA", "2025-01", 4.0), ("USA", "2025-02", 4.1)])
    store.apply(gdp, "oecd", "Real GDP", "Q", pulled_at=T1)
    store.apply(une, "oecd", "Unemployment", "M", pulled_at=T1)
    return load_ingested(tmp_path)


def test_latest_table(tmp_path):
    """Rows are indicators, columns are countries in config order, cells the last value."""
    table = latest_table(ingested(tmp_path), ["DEU", "USA", "JPN"])
    assert list(table.columns) == ["DEU", "USA", "JPN"]
    assert list(table.index) == ["Real GDP (Q)", "Unemployment (M)"]
    assert table.at["Real GDP (Q)", "USA"] == 2.0  # 2025Q2 beats 2025Q1
    assert table.at["Unemployment (M)", "USA"] == 4.1
    assert np.isnan(table.at["Unemployment (M)", "DEU"]) and table["JPN"].isna().all()


def test_chart_series(tmp_path):
    """One named column per pick on a shared timestamp index."""
    chart = chart_series(ingested(tmp_path), [("Real GDP (Q)", "USA"), ("Unemployment (M)", "USA")])
    assert list(chart.columns) == ["USA · Real GDP (Q)", "USA · Unemployment (M)"]
    assert isinstance(chart.index, pd.DatetimeIndex) and chart.index.is_monotonic_increasing
    assert chart["USA · Real GDP (Q)"].dropna().tolist() == [1.0, 2.0]
    assert chart_series({}, []).empty
