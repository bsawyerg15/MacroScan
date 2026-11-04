"""The country registry: every country MacroScan knows, keyed by ISO3 code, with a display label.

Code uses `Country.USA`; config CSVs and stored data use "USA". 01_countries.csv picks which
of these to ingest and show, in order. The label ("United States") is only for display.
"""

from __future__ import annotations

from enum import StrEnum, auto


class Country(StrEnum):
    """A country; the value is its ISO3 code (the member name), as used in config and stored data."""

    label: str

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        """auto() gives the member name, so USA's value is 'USA'."""
        return name

    def __new__(cls, value: str, label: str) -> Country:
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    def __repr__(self) -> str:
        """'USA' (quoted), so `{country!r}` in messages reads like a plain string."""
        return repr(self.value)

    @classmethod
    def parse(cls, text: str) -> Country:
        """Look up a country by its exact ISO3 code, e.g. 'USA'."""
        try:
            return cls(text)
        except ValueError:
            raise ValueError(
                f"unknown country {text!r}; use an ISO3 code defined in Country (a4_countries.py), e.g. 'USA'"
            ) from None

    USA = auto(), "United States"
    DEU = auto(), "Germany"
    JPN = auto(), "Japan"
    GBR = auto(), "United Kingdom"
    CAN = auto(), "Canada"
    AUS = auto(), "Australia"
