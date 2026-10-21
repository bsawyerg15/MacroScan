from datetime import datetime, timezone

import numpy as np
import pandas as pd

from macroscan.a_core.a2_schema import FETCH_COLUMNS, parse_periods
from macroscan.b_ingest.b3_store import Store

T1 = datetime(2026, 1, 15, tzinfo=timezone.utc)
T2 = datetime(2026, 2, 15, tzinfo=timezone.utc)


def frame(rows):
    """rows: (country, period, value, status)"""
    df = pd.DataFrame(rows, columns=["country", "period", "value", "status"])
    return pd.DataFrame(
        {
            "source": "oecd",
            "indicator": "gdp",
            "country": df["country"],
            "period": parse_periods(df["period"], "Q"),
            "frequency": "Q",
            "value": df["value"].astype(float),
            "unit": "XDC",
            "status": df["status"],
            "series_key": "Q." + df["country"],
        }
    )[FETCH_COLUMNS]


A = frame([("USA", "2025Q1", 100.0, "A"), ("USA", "2025Q2", 101.0, "P"), ("DEU", "2025Q1", 50.0, "A"),
           ("DEU", "2025Q2", 51.0, "A")])
B = frame([("USA", "2025Q1", 100.5, "A"),   # revised value
           ("USA", "2025Q2", 101.0, "A"),   # status P -> A
           ("USA", "2025Q3", 102.0, "P"),   # new period
           ("DEU", "2025Q1", 50.0, "A")])   # DEU 2025Q2 dropped -> withdrawn


def test_first_ingest_then_changes(tmp_path):
    """Revisions, status changes, new periods and drops each append exactly one row."""
    store = Store(tmp_path)
    s1 = store.apply(A, "oecd", "gdp", ["USA", "DEU"], pulled_at=T1)
    assert (s1.new, s1.revised, s1.withdrawn) == (4, 0, 0)

    s2 = store.apply(B, "oecd", "gdp", ["USA", "DEU"], pulled_at=T2)
    assert (s2.new, s2.revised, s2.withdrawn) == (1, 2, 1)
    assert len(store.load_log("oecd", "gdp")) == 8  # 4 originals + exactly 4 appended

    pd.testing.assert_frame_equal(_core(store.read_as_of("gdp", T1)), _core(A))
    pd.testing.assert_frame_equal(_core(store.read_latest("gdp")), _core(B))


def test_unchanged_pull_writes_nothing(tmp_path):
    """Re-ingesting identical data (within float tolerance) leaves the file untouched."""
    store = Store(tmp_path)
    store.apply(A, "oecd", "gdp", pulled_at=T1)
    mtime = store.path("oecd", "gdp").stat().st_mtime_ns
    s = store.apply(A.assign(value=A["value"] + 1e-13), "oecd", "gdp", pulled_at=T2)
    assert s.appended == 0
    assert store.path("oecd", "gdp").stat().st_mtime_ns == mtime


def test_missing_country_is_reported_not_withdrawn(tmp_path):
    """A country absent from a pull is reported, and its data kept."""
    store = Store(tmp_path)
    store.apply(A, "oecd", "gdp", ["USA", "DEU"], pulled_at=T1)
    s = store.apply(A[A["country"] == "USA"], "oecd", "gdp", ["USA", "DEU"], pulled_at=T2)
    assert s.withdrawn == 0 and s.countries_missing == ["DEU"]
    assert set(store.read_latest("gdp")["country"]) == {"USA", "DEU"}


def test_withdrawn_value_can_return(tmp_path):
    """A withdrawn observation that reappears counts as new."""
    store = Store(tmp_path)
    store.apply(A, "oecd", "gdp", pulled_at=T1)
    store.apply(B, "oecd", "gdp", pulled_at=T2)
    s = store.apply(A, "oecd", "gdp", pulled_at=datetime(2026, 3, 1, tzinfo=timezone.utc))
    assert s.new == 1  # DEU 2025Q2 is back
    history = store.read_history("gdp", country="DEU", period="2025Q2")
    assert history["value"].tolist()[:1] == [51.0] and np.isnan(history["value"].iloc[1])


def _core(df):
    """Key columns sorted, for comparing frames."""
    return df[["country", "period", "value", "status"]].sort_values(["country", "period"]).reset_index(drop=True)
