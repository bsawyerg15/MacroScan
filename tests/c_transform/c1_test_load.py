from datetime import datetime, timezone

import pandas as pd
import pytest

from macroscan.a_core.a2_schema import FETCH_COLUMNS, parse_periods
from macroscan.a_core.a3_indicators import Indicator
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
    store.apply(frame(Indicator.REAL_GDP, "Q", ["2025Q1", "2025Q2"]), "oecd", Indicator.REAL_GDP, "Q", pulled_at=T1)
    store.apply(frame(Indicator.UNEMPLOYMENT, "M", ["2025-01"]), "oecd", Indicator.UNEMPLOYMENT, "M", pulled_at=T1)
    store.apply(frame(Indicator.CPI_INFLATION, "M", ["2025-01"]), "oecd", Indicator.CPI_INFLATION, "M", pulled_at=T1)

    data = load_ingested(tmp_path)
    assert set(data) == {"Q", "M"}  # no annual data stored, so no "A" key
    assert len(data["Q"]) == 2 and set(data["M"]["indicator"]) == {Indicator.UNEMPLOYMENT, Indicator.CPI_INFLATION}
    assert data["M"]["period"].dtype == pd.PeriodDtype("M")
    assert set(load_ingested(tmp_path, ["Q"])) == {"Q"}


def test_load_ingested_turns_percents_into_decimals(tmp_path):
    """Percent-unit series (PA, PT_*) come back divided by 100 with unit '%'; others are untouched."""
    store = Store(tmp_path)
    for indicator, unit in [(Indicator.CPI_INFLATION, "PA"), (Indicator.UNEMPLOYMENT, "PT_LF_SUB"), (Indicator.CPI, "IX")]:
        store.apply(frame(indicator, "M", ["2025-01"]).assign(unit=unit), "oecd", indicator, "M", pulled_at=T1)

    out = load_ingested(tmp_path)["M"].set_index("indicator")
    assert out.loc[[Indicator.CPI_INFLATION, Indicator.UNEMPLOYMENT], "value"].tolist() == [0.01, 0.01]
    assert out.loc[[Indicator.CPI_INFLATION, Indicator.UNEMPLOYMENT], "unit"].tolist() == ["%", "%"]
    assert out.at[Indicator.CPI, "value"] == 1.0 and out.at[Indicator.CPI, "unit"] == "IX"


def test_load_ingested_upsamples_annual_pgdp_to_quarterly(tmp_path):
    """Annual PGDP also comes back at Q (log-linear between year ends), even when only Q is asked for."""
    store = Store(tmp_path)
    pgdp = frame(Indicator.PGDP, "A", ["2024", "2025"]).assign(value=[100.0, 104.0])
    store.apply(pgdp, "oecd", Indicator.PGDP, "A", pulled_at=T1)

    data = load_ingested(tmp_path)
    q = data["Q"][data["Q"]["indicator"] == Indicator.PGDP]
    assert q["period"].astype(str).tolist() == ["2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]
    assert q["value"].iloc[-1] == 104.0 and q["value"].iloc[2] == pytest.approx(100 * 1.04 ** 0.5)
    assert len(data["A"]) == 2
    assert set(load_ingested(tmp_path, ["Q"])) == {"Q"}
