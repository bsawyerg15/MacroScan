from pathlib import Path

import pytest

from macroscan.a_core.a1_config import (
    TABLE_FILE,
    ConfigError,
    load_all_series,
    load_table_indicators,
    parse_filters,
    slug,
)
from macroscan.a_core.a3_indicators import Indicator

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


def test_parse_filters():
    """Filters parse into dim -> codes, uppercasing dims and ignoring blanks."""
    assert parse_filters("FREQ=Q; sex=M+F ;") == {"FREQ": ("Q",), "SEX": ("M", "F")}
    assert parse_filters("") == {}


@pytest.mark.parametrize("bad", ["FREQ", "=Q", "FREQ=", "FREQ=Q;FREQ=A"])
def test_parse_filters_rejects(bad):
    """Malformed or repeated filters raise ConfigError."""
    with pytest.raises(ConfigError):
        parse_filters(bad)


def write_config(tmp_path, series_csv: str):
    """Write one OECD series CSV into tmp_path."""
    (tmp_path / "02_oecd_indicators.csv").write_text(series_csv)


def test_load_config(tmp_path):
    """Series load, skipping commented and blank rows."""
    write_config(tmp_path, "indicator,frequency,name,dataflow,filters,start\n#REAL_GDP,A,GDP,X:Y,,\n\nREAL_GDP,q,GDP,A:DSD@DF,SECTOR=S1,1990\n")
    [spec] = load_all_series(tmp_path)
    assert (spec.source, spec.indicator, spec.frequency, spec.filters, spec.start, spec.end) == (
        "oecd", Indicator.REAL_GDP, "Q", {"SECTOR": ("S1",)}, "1990", None
    )
    assert spec.indicator is Indicator.REAL_GDP
    assert spec.label == "oecd/REAL_GDP/Q"


def test_same_indicator_at_two_frequencies(tmp_path):
    """Frequency is its own dimension: one indicator can be configured at several."""
    write_config(tmp_path, "indicator,frequency,dataflow\nREAL_GDP,Q,A:B\nREAL_GDP,A,A:C\n")
    assert [(s.indicator, s.frequency) for s in load_all_series(tmp_path)] == [
        (Indicator.REAL_GDP, "Q"), (Indicator.REAL_GDP, "A")
    ]


def test_slug():
    """Slugs are only for file names."""
    assert slug("REAL_GDP") == "real_gdp"
    assert slug("CPI Inflation (y/y)") == "cpi_inflation_y_y"


@pytest.mark.parametrize(
    "row, match",
    [
        ("Real GDP,Q,A:B,,", "unknown indicator 'Real GDP'"),
        ("REAL_GDP,W,A:B,,", "frequency 'W'"),
        ("REAL_GDP,Q,A:B,FREQ=Q,", "not as FREQ in filters"),
        ("REAL_GDP,Q,A:B,,2020-03", "start '2020-03' is not a valid period for frequency Q"),
        ("REAL_GDP,Q,,,", "no dataflow"),
    ],
)
def test_bad_series(tmp_path, row, match):
    """Series rows must reference a defined indicator, a known frequency and no FREQ filter."""
    write_config(tmp_path, f"indicator,frequency,dataflow,filters,start\n{row}\n")
    with pytest.raises(ConfigError, match=match):
        load_all_series(tmp_path)


def test_duplicate_series(tmp_path):
    """The same source/indicator/frequency can't be configured twice."""
    write_config(tmp_path, "indicator,frequency,dataflow\nREAL_GDP,Q,A:B\nREAL_GDP,Q,A:C\n")
    with pytest.raises(ConfigError, match="oecd/REAL_GDP/Q"):
        load_all_series(tmp_path)


def test_load_table_indicators(tmp_path):
    """The listed indicators come back in file order; unknown or repeated ones raise ConfigError."""
    path = tmp_path / "t.csv"
    path.write_text("indicator\nCPI_INFLATION\nREAL_GDP\n")
    assert load_table_indicators(path) == [Indicator.CPI_INFLATION, Indicator.REAL_GDP]
    for bad in ["Real GDP", "REAL_GDP\nREAL_GDP", ""]:
        path.write_text("indicator\n" + bad + "\n")
        with pytest.raises(ConfigError):
            load_table_indicators(path)


def test_repo_config_is_consistent():
    """The real config/ loads: every indicator it names is defined in Indicator."""
    load_all_series(REPO_CONFIG)
    load_table_indicators(REPO_CONFIG / TABLE_FILE)
