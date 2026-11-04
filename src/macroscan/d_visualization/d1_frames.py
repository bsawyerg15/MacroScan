"""Shape the ingested data for the app: a latest-value table and chartable series.

Pure pandas, no Streamlit, so it can be tested. Periods are converted to
timestamps only here, at the chart boundary.
"""

from __future__ import annotations

import pandas as pd

from macroscan.a_core.a1_config import TableRow
from macroscan.a_core.a3_indicators import Indicator

# Non-breaking spaces per outline level: the table shows them, where plain leading spaces may be trimmed.
INDENT = "\u00a0" * 4


def row_label(indicator: str, frequency: str) -> str:
    """Table row for an (indicator, frequency), by its label: 'REAL_GDP' -> 'Real GDP (Q)'."""
    return f"{Indicator(indicator).label} ({frequency})"


def series_label(row: str, country: str) -> str:
    """Chart line name for one table cell: 'USA · Real GDP (Q)'."""
    return f"{country} · {row}"


def is_percent_row(row: str) -> bool:
    """Whether a table row holds percents (stored as decimals): its indicator's label has a '%'."""
    return "%" in row


def indent(label: str, level: int) -> str:
    """A table label shown `level` steps into the outline."""
    return INDENT * level + label


def unindent(label: str) -> str:
    """A table label without its outline indent, as `row_label` gives it."""
    return label.lstrip("\u00a0")


def header_labels(layout: list[TableRow]) -> list[str]:
    """The table labels of the layout's header rows, as `latest_table` shows them."""
    return [indent(r.header, r.level) for r in layout if r.is_header]


def latest_table(data: dict[str, pd.DataFrame], countries: list[str], layout: list[TableRow] | None = None) -> pd.DataFrame:
    """Last value per indicator (rows) and country (columns, in `countries` order).

    With no `layout`, every indicator with data is a row, sorted by label. With one, rows follow
    it: a header is an all-empty row; an indicator gives a row per frequency it has data at, or
    one empty row (labelled by the indicator alone) if it has none. Labels are indented by level.
    """
    frames = [_labelled(df) for df in data.values()]
    frames = [f for f in frames if len(f)]
    if frames:
        long = pd.concat([f.sort_values("period").groupby(["row", "country"]).tail(1) for f in frames])
        wide = long.pivot(index="row", columns="country", values="value").reindex(columns=countries)
        indicator_of = long.drop_duplicates("row").set_index("row")["indicator"]
    else:
        wide, indicator_of = pd.DataFrame(columns=countries, dtype=float), pd.Series(dtype=object)
    wide.columns.name = None
    if layout is None:
        wide.index.name = "indicator"
        return wide.sort_index()

    blank = pd.Series(float("nan"), index=countries)
    rows: dict[str, pd.Series] = {}
    for r in layout:
        if r.is_header:
            rows[indent(r.header, r.level)] = blank
            continue
        labels = sorted(indicator_of.index[indicator_of == r.indicator])
        for label in labels:
            rows[indent(label, r.level)] = wide.loc[label]
        if not labels:
            rows[indent(r.indicator.label, r.level)] = blank
    table = pd.DataFrame(rows).T.reindex(columns=countries).astype(float)
    table.index.name = "indicator"
    return table


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


__all__ = ["INDENT", "row_label", "series_label", "is_percent_row", "indent", "unindent", "header_labels", "latest_table", "chart_series"]
