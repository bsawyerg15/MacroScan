import pandas as pd
import pytest

from macroscan.a_core.a1_config import SeriesSpec
from macroscan.a_core.a2_schema import FETCH_COLUMNS, parse_periods
from macroscan.b_ingest.b2_validate import SchemaError, validate_frame

SPEC = SeriesSpec("oecd", "Real GDP", "Q", "A:B")


def frame(**overrides):
    """A valid two-row frame for SPEC, with columns overridden."""
    df = pd.DataFrame(
        {
            "source": "oecd",
            "indicator": "Real GDP",
            "country": ["USA", "DEU"],
            "period": parse_periods(["2025Q1", "2025Q1"], "Q"),
            "frequency": "Q",
            "value": [1.0, 2.0],
            "unit": "XDC",
            "status": None,
            "series_key": "k",
        }
    )
    return df.assign(**overrides)[FETCH_COLUMNS]


def test_valid_frame_passes():
    """A frame matching the spec comes back unchanged."""
    assert len(validate_frame(frame(), SPEC)) == 2


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"indicator": "CPI Inflation"}, "expected indicator 'Real GDP'"),
        ({"frequency": "M"}, "expected frequency 'Q'"),
        ({"source": "imf"}, "expected source 'oecd'"),
        ({"period": parse_periods(["2025-01", "2025-01"], "M")}, "does not match frequency Q"),
        ({"value": ["1", "2"]}, "value must be float"),
        ({"country": ["USA", "USA"]}, "duplicate"),
    ],
)
def test_mismatches_rejected(overrides, match):
    """Rows must carry the spec's source, indicator and frequency, and be well-formed."""
    with pytest.raises(SchemaError, match=match):
        validate_frame(frame(**overrides), SPEC)
