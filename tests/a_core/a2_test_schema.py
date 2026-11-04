import pytest

from macroscan.a_core.a2_schema import PeriodInterval


def test_period_interval_counts_periods_per_frequency():
    """An interval is the same span of time at every frequency it divides into."""
    for interval in (PeriodInterval.months(12), PeriodInterval.quarters(4), PeriodInterval.years(1)):
        assert {f: interval.periods_at(f) for f in "MQA"} == {"M": 12, "Q": 4, "A": 1}


def test_period_interval_that_does_not_divide_a_frequency_has_no_count():
    """3 months is one quarter but no whole number of years."""
    assert PeriodInterval.months(3).periods_at("Q") == 1
    assert PeriodInterval.months(3).periods_at("A") is None


def test_period_interval_must_be_positive():
    with pytest.raises(ValueError):
        PeriodInterval.months(0)
