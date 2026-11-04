import numpy as np
import pandas as pd
import pytest

from macroscan.a_core.a2_schema import FETCH_COLUMNS, PeriodInterval, parse_periods
from macroscan.a_core.a3_indicators import Indicator
from macroscan.c_transform.c2_transformations import add, ma, ma_change_pct_yoy, ratio_pct, sub, yoy_pct


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
    assert out["period"].dtype == pd.PeriodDtype("Q")


def test_yoy_pct_monthly_gap_is_not_misaligned():
    """A missing month yields no value for the month after a year, not a wrong lag."""
    periods = [str(p) for p in pd.period_range("2024-01", "2025-02", freq="M") if str(p) != "2024-02"]
    out = yoy_pct(frame("M", periods, [100.0] * len(periods)))
    assert "2025-02" not in set(out["period"].astype(str))
    assert "2025-01" in set(out["period"].astype(str))


def test_yoy_pct_keeps_countries_apart():
    """Several countries in one frame are each compared with their own year-earlier value."""
    df = pd.concat([frame("A", ["2024", "2025"], [100, 110]), frame("A", ["2024", "2025"], [50, 40], country="DEU")])
    assert yoy_pct(df).set_index("country")["value"].to_dict() == pytest.approx({"USA": 10.0, "DEU": -20.0})


def test_binary_ops_align_on_country_and_period():
    """add/sub/ratio_pct pair rows by (country, period); unmatched rows drop out, a's metadata is kept."""
    a = pd.concat([frame("Q", ["2025Q1", "2025Q2"], [30, 40]), frame("Q", ["2025Q1"], [5], country="DEU")])
    b = frame("Q", ["2025Q2", "2025Q3"], [200, 300], indicator=Indicator.NOMINAL_GDP)
    for op, expected in [(add, 240), (sub, -160), (ratio_pct, 20.0)]:
        out = op(a, b)
        assert list(zip(out["country"], out["period"].astype(str))) == [("USA", "2025Q2")]
        assert out["value"].tolist() == pytest.approx([expected])
        assert out["indicator"].tolist() == [Indicator.REAL_GDP]


def test_binary_ops_reject_mixed_frequencies():
    """Quarterly and annual periods never align, so combining them raises instead of returning nothing."""
    q = frame("Q", ["2025Q1"], [30])
    a = frame("A", ["2025"], [200], indicator=Indicator.NOMINAL_GDP)
    for op in (add, sub, ratio_pct):
        with pytest.raises(ValueError, match="one frequency"):
            op(q, a)


def test_ma_monthly():
    """Each month averages itself and the two before it; the first two months have no full window."""
    out = ma(frame("M", ["2025-01", "2025-02", "2025-03", "2025-04"], [1.0, 2.0, 6.0, 7.0]), PeriodInterval.months(3))
    assert list(out["period"].astype(str)) == ["2025-03", "2025-04"]
    assert out["value"].tolist() == pytest.approx([3.0, 5.0])
    assert out["period"].dtype == pd.PeriodDtype("M")


def test_ma_window_is_a_span_of_time():
    """A one-year window on quarterly data averages four quarters."""
    out = ma(frame("Q", ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"], [1, 2, 3, 4, 9]), PeriodInterval.years(1))
    assert out["value"].tolist() == pytest.approx([2.5, 4.5])


def test_ma_gap_drops_windows_that_span_it():
    """A missing month removes every window that would include it, rather than widening the window."""
    out = ma(frame("M", ["2025-01", "2025-02", "2025-04", "2025-05", "2025-06"], [1.0] * 5), PeriodInterval.months(2))
    assert list(out["period"].astype(str)) == ["2025-02", "2025-05", "2025-06"]


def test_ma_keeps_countries_apart():
    """Several countries in one frame are each averaged over their own values."""
    df = pd.concat([frame("A", ["2024", "2025"], [100, 110]), frame("A", ["2024", "2025"], [50, 40], country="DEU")])
    out = ma(df, PeriodInterval.years(2))
    assert out.set_index("country")["value"].to_dict() == pytest.approx({"USA": 105.0, "DEU": 45.0})


def test_ma_window_that_does_not_fit_the_frequency_yields_nothing():
    """Three months is no whole number of years, so annual data gets no rows rather than an error."""
    out = ma(frame("A", ["2024", "2025"], [1, 2]), PeriodInterval.months(3))
    assert out.empty
    assert list(out.columns) == FETCH_COLUMNS


def test_ma_change_pct_yoy_monthly():
    """The latest month vs the mean of the 18 before it, as a decimal % change."""
    periods = [str(p) for p in pd.period_range("2024-01", "2025-08", freq="M")]
    values = [100.0] * 18 + [110.0, 121.0]
    out = ma_change_pct_yoy(frame("M", periods, values))
    assert list(out["period"].astype(str)) == ["2025-07", "2025-08"]
    back_for_aug = (100.0 * 17 + 110.0) / 18
    expected = [0.1, 121.0 / back_for_aug - 1]
    assert out["value"].tolist() == pytest.approx(expected)


def test_ma_change_pct_yoy_window_that_does_not_fit_the_frequency_yields_nothing():
    """One month is no whole number of quarters, so quarterly data gets no rows."""
    out = ma_change_pct_yoy(frame("Q", ["2024Q1", "2024Q2"], [1, 2]))
    assert out.empty
