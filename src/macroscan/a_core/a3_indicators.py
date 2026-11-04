"""The indicator registry: every conceptual indicator, with a display label and a description.

An indicator's key is its member name: code uses `Indicator.REAL_GDP`, and config CSVs,
stored data and file names use "REAL_GDP". The label ("Real GDP") is only for display.
"""

from __future__ import annotations

from enum import StrEnum, auto


class Indicator(StrEnum):
    """A conceptual indicator; the value is the member name, as used in config and stored data."""

    label: str
    description: str

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list) -> str:
        """auto() gives the member name, so REAL_GDP's value is 'REAL_GDP'."""
        return name

    def __new__(cls, value: str, label: str, description: str) -> Indicator:
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        member.description = description
        return member

    def __repr__(self) -> str:
        """'REAL_GDP' (quoted), so `{indicator!r}` in messages reads like a plain string."""
        return repr(self.value)

    @classmethod
    def parse(cls, text: str) -> Indicator:
        """Look up an indicator by its exact member name, e.g. 'REAL_GDP'."""
        try:
            return cls(text)
        except ValueError:
            raise ValueError(
                f"unknown indicator {text!r}; use a member name of Indicator (a3_indicators.py), e.g. 'REAL_GDP'"
            ) from None

    REAL_GDP = auto(), "Real GDP", "Output growth: real (inflation-adjusted) gross domestic product"
    REAL_GROWTH = auto(), "Real Growth (%)", "Y/Y % change of real GDP"
    REAL_GROWTH_MA_CHANGE = auto(), "Real Growth MA Change (%)", "% change of real GDP: latest quarter vs the mean of the 18 months before it"
    PGDP = auto(), "Real Potential GDP", "Real potential gross domestic product"
    PGDP_GROWTH = auto(), "PGDP Growth (%)", "Y/Y % change of real potential GDP"
    NOMINAL_GDP = auto(), "Nominal GDP", "Gross domestic product at current prices"
    NOMINAL_GROWTH = auto(), "Nominal Growth (%)", "Y/Y % change of nominal GDP"
    UNEMPLOYMENT = auto(), "Unemployment (%)", "Unemployment rate"
    SLACK = auto(), "Slack", "Labor market slack: unemployment rate minus its estimated natural rate (NAIRU)"
    CPI = auto(), "CPI", "Consumer price index level"
    CPI_INFLATION = auto(), "CPI Inflation (%)", "Consumer price inflation"
    PPI = auto(), "PPI", "Producer price index level"
    PPI_INFLATION = auto(), "PPI Inflation (%)", "Y/Y % change of the producer price index"
    HH_DEBT_LEVEL = auto(), "HH Debt Level", "Outstanding household debt"
    HH_DEBT_LEVEL_GDP = auto(), "HH Debt Level % GDP", "Outstanding household debt as a % of nominal GDP"
    BUS_DEBT_LEVEL = auto(), "Bus. Debt Level", "Outstanding non-financial business debt"
    BUS_LOANS = auto(), "Bus. Loans", "Outstanding loans of non-financial corporations"
    BUS_DEBT_SECURITIES = auto(), "Bus. Debt Securities", "Outstanding debt securities issued by non-financial corporations"
    BUS_DEBT_LEVEL_GDP = auto(), "Bus. Debt Level % GDP", "Outstanding non-financial business debt as a % of nominal GDP"
    GOVT_DEBT_LEVEL = auto(), "Govt. Debt Level", "Outstanding general government debt"
    GOVT_DEBT_LEVEL_GDP = auto(), "Govt. Debt Level % GDP", "Outstanding general government debt as a % of nominal GDP"
    HH_DEBT_FLOW = auto(), "HH Debt Flow", "Net new household borrowing: loans, financial-account transactions"
    HH_DEBT_FLOW_GDP = auto(), "HH Debt Flow % GDP", "Net new household borrowing as a % of nominal GDP"
    BUS_DEBT_FLOW = auto(), "Bus. Debt Flow", "Net new non-financial business borrowing: loans plus debt securities"
    BUS_LOANS_FLOW = auto(), "Bus. Loans Flow", "Net new loans to non-financial corporations, financial-account transactions"
    BUS_DEBT_SECURITIES_FLOW = auto(), "Bus. Debt Securities Flow", "Net new debt securities issued by non-financial corporations, financial-account transactions"
    BUS_DEBT_FLOW_GDP = auto(), "Bus. Debt Flow % GDP", "Net new business borrowing as a % of nominal GDP"
    GOVT_DEBT_FLOW = auto(), "Govt. Debt Flow", "Net new general government borrowing: loans plus debt securities"
    GOVT_LOANS_FLOW = auto(), "Govt. Loans Flow", "Net new loans to general government, financial-account transactions"
    GOVT_DEBT_SECURITIES_FLOW = auto(), "Govt. Debt Securities Flow", "Net new debt securities issued by general government, financial-account transactions"
    GOVT_DEBT_FLOW_GDP = auto(), "Govt. Debt Flow % GDP", "Net new government borrowing as a % of nominal GDP"
    EQUITIES = auto(), "Equities", "Broad domestic equity market index"
    SHORT_RATES = auto(), "Short Rates", "Short-term interest rate (3-month interbank or policy rate)"
    LONG_RATES = auto(), "Long Rates", "Long-term interest rate (10-year government bond yield)"
    FX_TWI = auto(), "FX vs TWI", "Trade-weighted exchange rate index of the domestic currency"
    OIL = auto(), "Oil", "Crude oil price (Brent) in USD per barrel"
    GOLD = auto(), "Gold", "Gold price in USD per troy ounce"
