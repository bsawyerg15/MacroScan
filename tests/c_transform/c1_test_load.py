from datetime import datetime, timezone

import pandas as pd

from macroscan.a_core.a2_schema import FETCH_COLUMNS, parse_periods
from macroscan.b_ingest.b3_store import Store
from macroscan.c_transform.c1_load import load_ingested

T1 = datetime(2026, 1, 15, tzinfo=timezone.utc)


def frame(indicator, frequency, periods):
    """One USA row per period with value 1.0."""
    return pd.DataFrame(
        {
            "source": "oecd",
            "indicator": indicator,
            "country": "USA",
            "period": parse_periods(pd.Series(periods), frequency),
            "frequency": frequency,
            "value": 1.0,
            "unit": "XDC",
            "status": "A",
            "series_key": "x",
        }
    )[FETCH_COLUMNS]


def test_load_ingested_groups_by_frequency(tmp_path):
    """Every stored series comes back, grouped by frequency with that frequency's Period dtype."""
    store = Store(tmp_path)
    store.apply(frame("Real GDP", "Q", ["2025Q1", "2025Q2"]), "oecd", "Real GDP", "Q", pulled_at=T1)
    store.apply(frame("Unemployment", "M", ["2025-01"]), "oecd", "Unemployment", "M", pulled_at=T1)
    store.apply(frame("CPI Inflation", "M", ["2025-01"]), "oecd", "CPI Inflation", "M", pulled_at=T1)

    data = load_ingested(tmp_path)
    assert set(data) == {"Q", "M"}  # no annual data stored, so no "A" key
    assert len(data["Q"]) == 2 and set(data["M"]["indicator"]) == {"Unemployment", "CPI Inflation"}
    assert data["M"]["period"].dtype == pd.PeriodDtype("M")
    assert set(load_ingested(tmp_path, ["Q"])) == {"Q"}
