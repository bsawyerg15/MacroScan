import pytest

from macroscan.a_core.a1_config import (
    ConfigError,
    load_all_series,
    load_countries,
    load_indicators,
    parse_filters,
    slug,
)


def test_parse_filters():
    """Filters parse into dim -> codes, uppercasing dims and ignoring blanks."""
    assert parse_filters("FREQ=Q; sex=M+F ;") == {"FREQ": ("Q",), "SEX": ("M", "F")}
    assert parse_filters("") == {}


@pytest.mark.parametrize("bad", ["FREQ", "=Q", "FREQ=", "FREQ=Q;FREQ=A"])
def test_parse_filters_rejects(bad):
    """Malformed or repeated filters raise ConfigError."""
    with pytest.raises(ConfigError):
        parse_filters(bad)


REGISTRY = "indicator,description\nReal GDP,Output\nCPI Inflation,Prices\n"


def write_config(tmp_path, series_csv: str, registry: str = REGISTRY):
    """Write an indicator registry and one OECD series CSV into tmp_path."""
    (tmp_path / "00_indicators.csv").write_text(registry)
    (tmp_path / "02_oecd_indicators.csv").write_text(series_csv)


def test_load_config(tmp_path):
    """Countries and series load, skipping commented and blank rows."""
    (tmp_path / "01_countries.csv").write_text("code,name\nusa,United States\n#FRA,France\n\nDEU,Germany\n")
    write_config(tmp_path, "indicator,frequency,name,dataflow,filters,start\nReal GDP,q,GDP,A:DSD@DF,SECTOR=S1,1990\n")
    assert [c.code for c in load_countries(tmp_path / "01_countries.csv")] == ["USA", "DEU"]
    assert load_indicators(tmp_path / "00_indicators.csv") == {"Real GDP": "Output", "CPI Inflation": "Prices"}
    [spec] = load_all_series(tmp_path)
    assert (spec.source, spec.indicator, spec.frequency, spec.filters, spec.start, spec.end) == (
        "oecd", "Real GDP", "Q", {"SECTOR": ("S1",)}, "1990", None
    )
    assert spec.label == "oecd/Real GDP/Q"


def test_same_indicator_at_two_frequencies(tmp_path):
    """Frequency is its own dimension: one indicator can be configured at several."""
    write_config(tmp_path, "indicator,frequency,dataflow\nReal GDP,Q,A:B\nReal GDP,A,A:C\n")
    assert [(s.indicator, s.frequency) for s in load_all_series(tmp_path)] == [("Real GDP", "Q"), ("Real GDP", "A")]


def test_slug():
    """Slugs are only for file names."""
    assert slug("Real GDP") == "real_gdp"
    assert slug("CPI Inflation (y/y)") == "cpi_inflation_y_y"


def test_bad_country_rejected(tmp_path):
    """Non-ISO3 countries are rejected."""
    (tmp_path / "01_countries.csv").write_text("code\nUS\n")
    with pytest.raises(ConfigError, match="ISO3"):
        load_countries(tmp_path / "01_countries.csv")


@pytest.mark.parametrize(
    "registry, match",
    [
        ("indicator\nReal GDP\nreal gdp\n", "duplicate indicator"),
        ("indicator\nReal GDP\nReal-GDP\n", "too similar"),
        ("indicator\n", "no indicators"),
    ],
)
def test_bad_registry(tmp_path, registry, match):
    """Indicator names must be distinct, also once slugged for file names."""
    write_config(tmp_path, "indicator,frequency,dataflow\n", registry)
    with pytest.raises(ConfigError, match=match):
        load_all_series(tmp_path)


@pytest.mark.parametrize(
    "row, match",
    [
        ("GDP,Q,A:B,,", "unknown indicator 'GDP'"),
        ("Real GDP,W,A:B,,", "frequency 'W'"),
        ("Real GDP,Q,A:B,FREQ=Q,", "not as FREQ in filters"),
        ("Real GDP,Q,A:B,,2020-03", "start '2020-03' is not a valid period for frequency Q"),
        ("Real GDP,Q,,,", "no dataflow"),
    ],
)
def test_bad_series(tmp_path, row, match):
    """Series rows must reference a registered indicator, a known frequency and no FREQ filter."""
    write_config(tmp_path, f"indicator,frequency,dataflow,filters,start\n{row}\n")
    with pytest.raises(ConfigError, match=match):
        load_all_series(tmp_path)


def test_duplicate_series(tmp_path):
    """The same source/indicator/frequency can't be configured twice."""
    write_config(tmp_path, "indicator,frequency,dataflow\nReal GDP,Q,A:B\nReal GDP,Q,A:C\n")
    with pytest.raises(ConfigError, match="oecd/Real GDP/Q"):
        load_all_series(tmp_path)
