import pandas as pd
import pytest
import responses

from macroscan.a_core.a1_config import IndicatorSpec
from macroscan.b_ingest.b1_sources.oecd import BASE_URL, Dataflow, OECDClient, OECDSource, build_key, normalize, parse_structure
from macroscan.b_ingest.b2_validate import validate_frame

FLOW = "OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA"
GDP_FILTERS = {
    "FREQ": ("Q",), "ADJUSTMENT": ("Y",), "SECTOR": ("S1",), "TRANSACTION": ("B1GQ",),
    "UNIT_MEASURE": ("XDC",), "PRICE_BASE": ("LR",), "TRANSFORMATION": ("LA",),
}
SPEC = IndicatorSpec("gdp_real_q", "GDP", "oecd", FLOW, GDP_FILTERS, start="2025-Q1")


def test_dataflow_parse():
    """Dataflow strings parse with and without an explicit version."""
    assert Dataflow.parse(FLOW) == Dataflow("OECD.SDD.NAD", "DSD_NAMAIN1@DF_QNA", "latest")
    assert Dataflow.parse(FLOW + "@1.1").version == "1.1"
    with pytest.raises(ValueError):
        Dataflow.parse("DSD_NAMAIN1@DF_QNA")


def test_parse_structure(qna_structure_payload):
    """Dimensions come out in key order with codelists and available codes."""
    s = parse_structure(qna_structure_payload, Dataflow.parse(FLOW))
    assert [d.id for d in s.dimensions][:3] == ["FREQ", "ADJUSTMENT", "REF_AREA"]
    assert s.dimension("REF_AREA").codes["DEU"] == "Germany"
    assert "Q" in s.dimension("FREQ").available


def test_build_key(qna_structure_payload):
    """Filters and countries land in the right key positions; others are blank."""
    s = parse_structure(qna_structure_payload, Dataflow.parse(FLOW))
    assert build_key(s, GDP_FILTERS, ["USA", "DEU"]) == "Q.Y.USA+DEU.S1..B1GQ....XDC.LR.LA."


def test_build_key_errors(qna_structure_payload):
    """Unknown dimensions and REF_AREA in filters give clear errors."""
    s = parse_structure(qna_structure_payload, Dataflow.parse(FLOW))
    with pytest.raises(ValueError, match="TRANSACTON.*Valid dimensions: FREQ"):
        build_key(s, {"TRANSACTON": ("B1GQ",)}, ["USA"])
    with pytest.raises(ValueError, match="REF_AREA comes from"):
        build_key(s, {"REF_AREA": ("USA",)}, ["USA"])


def test_normalize(qna_structure_payload, qna_gdp_csv):
    """OECD CSV becomes a valid canonical frame with codes, periods and units."""
    s = parse_structure(qna_structure_payload, Dataflow.parse(FLOW))
    df = validate_frame(normalize(qna_gdp_csv, SPEC, s), SPEC.id)
    assert set(df["country"]) == {"USA", "DEU"}
    assert df["period"].dtype == pd.PeriodDtype("Q")
    assert df["period"].min() == pd.Period("2025Q1")
    assert df["unit"].iloc[0] == "XDC (millions)"
    assert df["series_key"].iloc[0] == "Q.Y.DEU.S1.S1.B1GQ._Z._Z._Z.XDC.LR.LA.T0102"
    assert df["value"].dtype == "float64" and df["value"].notna().all()


def test_normalize_rejects_several_series(qna_structure_payload, qna_gdp_csv):
    """Under-specified filters name the dimensions that still vary."""
    s = parse_structure(qna_structure_payload, Dataflow.parse(FLOW))
    extra = qna_gdp_csv.splitlines()[1].replace("B1GQ: Gross domestic product", "P3: Final consumption expenditure")
    with pytest.raises(ValueError, match="Pin these dimensions.*TRANSACTION"):
        normalize(qna_gdp_csv + extra + "\n", SPEC, s)


@responses.activate
def test_fetch_end_to_end(tmp_path, qna_structure_payload, qna_gdp_csv):
    """fetch() retries a 429, returns data, and caches the structure on disk."""
    responses.get(f"{BASE_URL}/dataflow/OECD.SDD.NAD/DSD_NAMAIN1@DF_QNA/latest", json=qna_structure_payload)
    responses.get(f"{BASE_URL}/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,/Q.Y.USA+DEU.S1..B1GQ....XDC.LR.LA.", status=429,
                  headers={"Retry-After": "0"})
    responses.get(f"{BASE_URL}/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,/Q.Y.USA+DEU.S1..B1GQ....XDC.LR.LA.", body=qna_gdp_csv)
    source = OECDSource(tmp_path, client=OECDClient(min_interval=0, backoff=0))
    df = source.fetch(SPEC, ["USA", "DEU"])
    assert len(df) == 12
    assert "startPeriod=2025-Q1" in source.last_request_url
    # Structure is cached on disk: a second source makes no structure request.
    OECDSource(tmp_path, client=OECDClient(min_interval=0)).structure(FLOW)
    assert sum("/dataflow/" in c.request.url for c in responses.calls) == 1


@responses.activate
def test_fetch_no_data(tmp_path, qna_structure_payload):
    """A 404 NoRecordsFound yields an empty frame, not an error."""
    responses.get(f"{BASE_URL}/dataflow/OECD.SDD.NAD/DSD_NAMAIN1@DF_QNA/latest", json=qna_structure_payload)
    responses.get(f"{BASE_URL}/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA,/Q.Y.USA.S1..B1GQ....XDC.LR.LA.", status=404,
                  body="NoRecordsFound")
    df = OECDSource(tmp_path, client=OECDClient(min_interval=0)).fetch(SPEC, ["USA"])
    assert df.empty
