"""Shape the ingested data for the app: a latest-value table and chartable series.

Pure pandas, no Streamlit, so it can be tested. Periods are converted to
timestamps only here, at the chart boundary.
"""

from __future__ import annotations

import pandas as pd


def row_label(indicator: str, frequency: str) -> str:
    """Table row for an (indicator, frequency): 'Real GDP (Q)'."""
    return f"{indicator} ({frequency})"


def series_label(row: str, country: str) -> str:
    """Chart line name for one table cell: 'USA · Real GDP (Q)'."""
    return f"{country} · {row}"


def latest_table(data: dict[str, pd.DataFrame], countries: list[str]) -> pd.DataFrame:
    """Last value per indicator (rows) and country (columns, in `countries` order)."""
    frames = [_labelled(df) for df in data.values()]
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


__all__ = ["row_label", "series_label", "latest_table", "chart_series"]
