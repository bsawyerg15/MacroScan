import pytest

from macroscan.a_core.a4_countries import Country


def test_members_are_their_codes():
    """A member's value is its ISO3 code; the label is for display."""
    assert Country.USA == "USA" and f"{Country.USA}" == "USA" and Country.USA.label == "United States"
    assert all(c.value == c.name and len(c) == 3 and c.isalpha() and c.isupper() and c.label for c in Country)


def test_parse():
    """Lookup is by exact ISO3 code."""
    assert Country.parse("DEU") is Country.DEU
    for bad in ["deu", "Germany", "DE"]:
        with pytest.raises(ValueError, match=f"unknown country '{bad}'"):
            Country.parse(bad)
