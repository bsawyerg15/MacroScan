import re

import pytest

from macroscan.a_core.a1_config import slug
from macroscan.a_core.a3_indicators import Indicator


def test_members_are_their_names():
    """A member's value is its name; the label is for display and every member has a description."""
    assert Indicator.REAL_GDP == "REAL_GDP" and f"{Indicator.REAL_GDP}" == "REAL_GDP"
    assert Indicator.REAL_GDP.label == "Real GDP"
    assert all(i.value == i.name and i.description for i in Indicator)


def test_labels_and_slugs_are_distinct():
    """Labels stay distinct for display, and member names once slugged for file names."""
    labels = [i.label for i in Indicator]
    slugs = [slug(i) for i in Indicator]
    assert len(set(labels)) == len(labels)
    assert all(slugs) and len(set(slugs)) == len(slugs)


def test_parse():
    """Lookup is by exact member name; labels and other spellings name the problem."""
    assert Indicator.parse("CPI_INFLATION") is Indicator.CPI_INFLATION
    for bad in ["CPI Inflation (%)", "real_gdp", "Real GDP"]:
        with pytest.raises(ValueError, match=re.escape(f"unknown indicator {bad!r}")):
            Indicator.parse(bad)
