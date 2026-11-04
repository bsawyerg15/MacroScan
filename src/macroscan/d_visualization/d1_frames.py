"""Shape the ingested data for the app: a latest-value table and chartable series.

Pure pandas, no Streamlit, so it can be tested. Periods are converted to
timestamps only here, at the chart boundary.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from macroscan.a_core.a1_config import TableRow
from macroscan.a_core.a3_indicators import Indicator

# Non-breaking spaces per outline level: the table shows them, where plain leading spaces may be trimmed.
INDENT = "\u00a0" * 4
INPUT = "↳ "  # marks a provenance row: an input its parent row is computed from


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
    """A table label without its outline indent (an INPUT mark stays)."""
    return label.lstrip("\u00a0")


def base_label(label: str) -> str:
    """A table label without indent or INPUT mark, as `row_label` gives it and `chart_series` takes it."""
    return unindent(label).removeprefix(INPUT)


def header_labels(layout: list[TableRow]) -> list[str]:
    """The table labels of the layout's header rows, as `latest_table` shows them."""
    return [indent(r.header, r.level) for r in layout if r.is_header]


def latest_table(
    data: dict[str, pd.DataFrame],
    countries: list[str],
    layout: list[TableRow] | None = None,
    inputs: Mapping[Indicator, tuple[Indicator, ...]] | None = None,
) -> pd.DataFrame:
    """Last value per indicator (rows) and country (columns, in `countries` order).

    With no `layout`, every indicator with data is a row, sorted by label. With one, rows follow
    it: a header is an all-empty row; an indicator gives a row per frequency it has data at, or
    one empty row (labelled by the indicator alone) if it has none. Labels are indented by level.

    `inputs` maps a derived indicator to the indicators it is computed from (c3_derived's
    recipes). With it, each layout indicator row is followed by its provenance: one INPUT-marked
    row per input, a level deeper and recursively down to ingested series, holding each input's
    value at the period of the parent cell, so a cell's inputs are the ones it was computed from.
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
    # (row label, country, period) -> value, per frequency: one Period dtype per lookup.
    by_period = {f["frequency"].iloc[0]: f.set_index(["row", "country", "period"])["value"] for f in frames}
    rows: list[tuple[str, pd.Series]] = []  # a list, not a dict: an input can repeat under several parents

    def add_inputs(indicator: Indicator, frequency: str, level: int, periods: pd.Series) -> None:
        """Rows for `indicator`'s inputs at `frequency`, valued at `periods` (country -> Period)."""
        for i in (inputs or {}).get(indicator, ()):
            label = row_label(i, frequency)
            values = by_period.get(frequency, pd.Series(dtype=float))
            cells = pd.Series(
                [values.get((label, c, p), float("nan")) if pd.notna(p) else float("nan") for c, p in periods.items()],
                index=periods.index,
                dtype=float,
            )
            rows.append((indent(INPUT + label, level), cells))
            add_inputs(i, frequency, level + 1, periods)

    for r in layout:
        if r.is_header:
            rows.append((indent(r.header, r.level), blank))
            continue
        labels = sorted(indicator_of.index[indicator_of == r.indicator])
        for label in labels:
            rows.append((indent(label, r.level), wide.loc[label]))
            mine = long[long["row"] == label]
            periods = pd.Series(mine["period"].to_numpy(), index=mine["country"]).reindex(countries)
            add_inputs(r.indicator, mine["frequency"].iloc[0], r.level + 1, periods)
        if not labels:
            rows.append((indent(r.indicator.label, r.level), blank))
    table = pd.DataFrame([v.reindex(countries) for _, v in rows], index=[label for label, _ in rows], dtype=float)
    table.columns = countries
    table.index.name = "indicator"
    return table


PATH_SEP = " › "
EXPANDED, COLLAPSED, LEAF = "▾ ", "▸ ", "\u00a0\u00a0\u00a0"  # LEAF pads to roughly a marker's width


def level_of(label: str) -> int:
    """A table label's outline depth: how many INDENTs it starts with."""
    return (len(label) - len(unindent(label))) // len(INDENT)


def outline(table: pd.DataFrame) -> pd.DataFrame:
    """Per table row (same index, same order): `level`, `path`, `has_children`, `is_input` and `has_inputs`.

    The path is the row's unindented label after its ancestors', joined by PATH_SEP, so a row
    is identified by where it sits: the same label under two parents gives two paths.
    """
    levels = [level_of(label) for label in table.index]
    paths, ancestors = [], []
    for label, level in zip(table.index, levels):
        ancestors = ancestors[:level] + [unindent(label)]
        paths.append(PATH_SEP.join(ancestors))
    has_children = [nxt > lvl for lvl, nxt in zip(levels, levels[1:] + [-1])]
    is_input = [unindent(label).startswith(INPUT) for label in table.index]
    # Inputs come first under their parent, so a row has inputs when its first child is one.
    has_inputs = [kids and nxt for kids, nxt in zip(has_children, is_input[1:] + [False])]
    return pd.DataFrame(
        {"level": levels, "path": paths, "has_children": has_children, "is_input": is_input, "has_inputs": has_inputs},
        index=table.index,
    )


def _shown(rows: pd.DataFrame, collapsed: set[str]) -> list[bool]:
    """Per outline row, whether it shows: not under a collapsed row (collapsed rows themselves show).

    A collapsed row with inputs folds only its inputs, so children configured in the layout stay
    in view; a collapsed row without inputs folds everything under it.
    """
    keep = []
    folds: list[tuple[int, bool]] = []  # shown collapsed rows still hiding rows: (level, folds only inputs)
    for r in rows.itertuples():
        # A fold ends at the next row as shallow as it, or, for an inputs-only fold, at the first
        # non-input row (inputs come first and their own children are inputs too).
        while folds and (r.level <= folds[-1][0] or (folds[-1][1] and not r.is_input)):
            folds.pop()
        if folds:
            keep.append(False)
            continue
        if r.path in collapsed:
            folds.append((r.level, r.has_inputs))
        keep.append(True)
    return keep


def visible(table: pd.DataFrame, collapsed: set[str]) -> pd.DataFrame:
    """The table without the rows under any collapsed path, plus their outline columns.

    Returns the shown rows' values with the `outline` columns and `display` (the label
    to show: indent, a ▾/▸ marker on parents or a gap on leaves, then the label) appended.
    Outline facts come from the full table, so a collapsed parent keeps its ▸.
    """
    rows = outline(table)
    rows["display"] = [
        INDENT * r.level + (LEAF if not r.has_children else COLLAPSED if r.path in collapsed else EXPANDED) + unindent(label)
        for label, r in zip(table.index, rows.itertuples())
    ]
    # Join by position: labels can repeat under different parents.
    return table.assign(**{c: rows[c].to_numpy() for c in rows.columns})[_shown(rows, collapsed)]


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


__all__ = ["INDENT", "INPUT", "base_label", "row_label", "series_label", "is_percent_row", "indent", "unindent", "header_labels", "PATH_SEP", "level_of", "outline", "visible", "latest_table", "chart_series"]
