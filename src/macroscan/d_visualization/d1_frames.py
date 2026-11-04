"""Shape the ingested data for the app: a latest-value table and chartable series.

Pure pandas, no Streamlit, so it can be tested. Periods are converted to
timestamps only here, at the chart boundary.
"""

from __future__ import annotations

import pandas as pd

from macroscan.a_core.a3_indicators import Indicator


def row_label(indicator: str, frequency: str) -> str:
    """Table row for an (indicator, frequency), by its label: 'REAL_GDP' -> 'Real GDP (Q)'."""
    return f"{Indicator(indicator).label} ({frequency})"


def series_label(row: str, country: str) -> str:
    """Chart line name for one table cell: 'USA · Real GDP (Q)'."""
    return f"{country} · {row}"


def is_percent_row(row: str) -> bool:
    """Whether a table row holds percents (stored as decimals): its indicator's label has a '%'."""
    return "%" in row


def latest_table(data: dict[str, pd.DataFrame], countries: list[str], indicators: list[str] | None = None) -> pd.DataFrame:
    """Last value per indicator (rows) and country (columns, in `countries` order).

    `indicators` limits the rows to those indicators; None shows all.
    """
    frames = [_labelled(df if indicators is None else df[df["indicator"].isin(indicators)]) for df in data.values()]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=countries, dtype=float)
    long = pd.concat([f.sort_values("period").groupby(["row", "country"]).tail(1) for f in frames])
    wide = long.pivot(index="row", columns="country", values="value").reindex(columns=countries)
    wide.index.name, wide.columns.name = "indicator", None
    return wide.sort_index()


def chart_series(data: dict[str, pd.DataFrame], picks: list[tuple[str, str]]) -> pd.DataFrame:
    """One column per (row label, country) pick, indexed by timestamp; frequencies are outer-joined."""
    # Filter each frequency separately: concatenating them would lose the Period dtype.
    frames = [_labelled(df) for df in data.values()]
    columns = {}
    for row, country in picks:
        matches = (f[(f["row"] == row) & (f["country"] == country)] for f in frames)
        s = next((m for m in matches if len(m)), None)
        if s is None:
            continue
        columns[series_label(row, country)] = pd.Series(
            s["value"].to_numpy(), index=pd.DatetimeIndex(s["period"].dt.to_timestamp())
        )
    chart = pd.DataFrame(columns).sort_index() if columns else pd.DataFrame(index=pd.DatetimeIndex([]))
    chart.index.name = "date"
    return chart


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    """A long frame with its table row label added."""
    return df.assign(row=[row_label(i, f) for i, f in zip(df["indicator"], df["frequency"])])


__all__ = ["row_label", "series_label", "is_percent_row", "latest_table", "chart_series"]
