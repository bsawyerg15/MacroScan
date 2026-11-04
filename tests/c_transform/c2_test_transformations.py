import pandas as pd
import pytest

from macroscan.a_core.a1_config import TransformSpec
from macroscan.a_core.a2_schema import FETCH_COLUMNS, Transformation, parse_periods
from macroscan.a_core.a3_indicators import Indicator
from macroscan.c_transform.c2_transformations import apply_transformations, yoy_pct

SPEC = TransformSpec(Indicator.REAL_GROWTH, Indicator.REAL_GDP, Transformation.YOY_PCT)


def frame(frequency, periods, values, indicator=Indicator.REAL_GDP, country="USA"):
    """One country's series in the canonical long format."""
    return pd.DataFrame(
        {
            "source": "oecd",
            "indicator": indicator,
            "country": country,
            "period": parse_periods(pd.Series(periods), frequency),
            "frequency": frequency,
            "value": values,
            "unit": "XDC",
            "status": "A",
            "series_key": "x",
        }
    )[FETCH_COLUMNS]


def test_yoy_pct_quarterly():
    """Each quarter is compared with the same quarter a year earlier; the first year has no value."""
    df = frame("Q", ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2"], [100, 100, 100, 100, 104, 90])
    out = yoy_pct(df)
    assert list(out["period"].astype(str)) == ["2025Q1", "2025Q2"]
    assert out["value"].tolist() == pytest.approx([4.0, -10.0])


def test_yoy_pct_monthly_gap_is_not_misaligned():
    """A missing month yields no value for the month after a year, not a wrong lag."""
    periods = [str(p) for p in pd.period_range("2024-01", "2025-02", freq="M") if str(p) != "2024-02"]
    out = yoy_pct(frame("M", periods, [100.0] * len(periods)))
    assert "2025-02" not in set(out["period"].astype(str))
    assert "2025-01" in set(out["period"].astype(str))


def test_apply_transformations_appends_derived_indicator():
    """The derived indicator joins the same frequency frame per country; the input is untouched."""
    q = pd.concat(
        [
            frame("Q", ["2024Q1", "2025Q1"], [100, 110]),
            frame("Q", ["2024Q1", "2025Q1"], [50, 40], country="DEU"),
            frame("Q", ["2025Q1"], [7], indicator="Other"),
        ],
        ignore_index=True,
    )
    data = {"Q": q}
    out = apply_transformations(data, [SPEC])
    assert len(data["Q"]) == 5
    derived = out["Q"][out["Q"]["indicator"] == Indicator.REAL_GROWTH].set_index("country")
    assert derived["value"].to_dict() == pytest.approx({"USA": 10.0, "DEU": -20.0})
    assert set(derived["unit"]) == {"%"}
    assert out["Q"]["period"].dtype == pd.PeriodDtype("Q")


def test_apply_transformations_skips_missing_old_indicator():
    """Frequencies without the old indicator are left as they were."""
    data = {"M": frame("M", ["2025-01"], [1.0], indicator=Indicator.UNEMPLOYMENT)}
    assert apply_transformations(data, [SPEC])["M"].equals(data["M"])
