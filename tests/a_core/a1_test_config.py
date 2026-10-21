import pytest

from macroscan.a_core.a1_config import ConfigError, load_all_indicators, load_countries, parse_filters


def test_parse_filters():
    """Filters parse into dim -> codes, uppercasing dims and ignoring blanks."""
    assert parse_filters("FREQ=Q; sex=M+F ;") == {"FREQ": ("Q",), "SEX": ("M", "F")}
    assert parse_filters("") == {}


@pytest.mark.parametrize("bad", ["FREQ", "=Q", "FREQ=", "FREQ=Q;FREQ=A"])
def test_parse_filters_rejects(bad):
    """Malformed or repeated filters raise ConfigError."""
    with pytest.raises(ConfigError):
        parse_filters(bad)


def test_load_config(tmp_path):
    """Countries and indicators load, skipping commented and blank rows."""
    (tmp_path / "01_countries.csv").write_text("code,name\nusa,United States\n#FRA,France\n\nDEU,Germany\n")
    (tmp_path / "02_oecd_indicators.csv").write_text(
        "id,name,dataflow,filters,start\ngdp,GDP,A:DSD@DF,FREQ=Q,1990\n"
    )
    assert [c.code for c in load_countries(tmp_path / "01_countries.csv")] == ["USA", "DEU"]
    [spec] = load_all_indicators(tmp_path)
    assert (spec.source, spec.id, spec.filters, spec.start, spec.end) == ("oecd", "gdp", {"FREQ": ("Q",)}, "1990", None)


def test_bad_country_and_duplicate_ids(tmp_path):
    """Non-ISO3 countries and duplicate indicator ids are rejected."""
    (tmp_path / "01_countries.csv").write_text("code\nUS\n")
    with pytest.raises(ConfigError, match="ISO3"):
        load_countries(tmp_path / "01_countries.csv")
    (tmp_path / "02_oecd_indicators.csv").write_text("id,dataflow\ngdp,A:B\ngdp,A:C\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_all_indicators(tmp_path)
