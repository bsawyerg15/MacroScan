"""Streamlit app: latest values by indicator and country; click labels to fold rows, cells to chart them.

Run from the repo root: `streamlit run src/macroscan/d_visualization/d2_app.py`
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from macroscan.a_core.a1_config import (
    DEFAULT_CONFIG_DIR,
    TABLE_FILE,
    TableRow,
    load_table_layout,
)
from macroscan.a_core.a4_countries import Country
from macroscan.b_ingest.b3_store import DEFAULT_ROOT
from macroscan.c_transform.c1_load import load_ingested
from macroscan.c_transform.c3_derived import DERIVED, apply_derivations
from macroscan.d_visualization.d1_frames import (
    base_label,
    chart_series,
    header_labels,
    is_percent_row,
    latest_table,
    outline,
    row_shades,
    series_label,
    visible,
)

# A series keeps its color while charted; a removed series frees its color for the next one added.
PALETTE = ["#0066ff", "#d62728", "#2ca02c", "#000000", "#ff7f0e", "#7f7f7f", "#9467bd", "#17becf", "#bcbd22", "#8c564b"]  # blue, red, green, black, orange, gray, then purple, teal, olive, brown
TABLE_KEY = "latest"  # plus a counter, bumped to clear the table's selection after a fold
TABLE_GEN_KEY = "table_gen"
COLLAPSED_KEY = "collapsed"  # outline paths of the folded rows
LABEL_COLUMN = "Indicator"
PICKS_KEY = "picks"
COLORS_KEY = "colors"  # charted label -> palette color
HEADER_STYLE = "background-color: rgba(128, 128, 128, 0.15); font-weight: bold"  # section header rows
INPUT_STYLE = "color: rgba(128, 128, 128, 0.9); font-style: italic"  # provenance rows


@st.cache_data
def _data() -> dict[str, pd.DataFrame]:
    """Ingested data plus the derived indicators, built once per session."""
    return apply_derivations(load_ingested(DEFAULT_ROOT))


def _countries() -> list[str]:
    """Country codes in definition order."""
    return list(Country)


@st.cache_data
def _table_layout() -> list[TableRow]:
    """The table's outline of section headers and indicators, from config."""
    return load_table_layout(DEFAULT_CONFIG_DIR / TABLE_FILE)


def _color(label: str, picks: list[str]) -> str:
    """The color for a charted series: its existing one, else the first palette color nobody is using."""
    colors = st.session_state.setdefault(COLORS_KEY, {})
    for gone in set(colors) - set(picks):
        del colors[gone]
    if label not in colors:
        free = [c for c in PALETTE if c not in colors.values()]
        colors[label] = free[0] if free else PALETTE[len(colors) % len(PALETTE)]
    return colors[label]


def main() -> None:
    """Render the table and the chart of picked cells."""
    st.set_page_config(page_title="MacroScan", layout="wide")
    st.title("MacroScan")
    data = _data()
    layout = _table_layout()
    table = latest_table(data, _countries(), layout, inputs={i: d.inputs for i, d in DERIVED.items()})
    if table.isna().all().all():
        st.info("No ingested data yet; run `macroscan ingest` first.")
        return

    # Every cell's chart label -> (row, country), so picks can live in a multiselect.
    # Labels drop the outline indent and input mark so they match chart_series' row labels; header rows have no cells.
    # Iterate rows by position: an input's label repeats under each parent it feeds.
    cells = {
        series_label(base_label(r), c): (base_label(r), c)
        for r, values in zip(table.index, table.to_numpy())
        for c, v in zip(table.columns, values)
        if pd.notna(v)
    }

    st.session_state.setdefault(PICKS_KEY, [])
    # Provenance starts folded: each row with inputs is collapsed, which hides only its inputs.
    st.session_state.setdefault(COLLAPSED_KEY, set(outline(table).query("has_inputs")["path"]))
    st.session_state.setdefault(TABLE_GEN_KEY, 0)
    collapsed: set[str] = st.session_state[COLLAPSED_KEY]
    shown = visible(table, collapsed)
    table_key = f"{TABLE_KEY}-{st.session_state[TABLE_GEN_KEY]}"

    def refold(paths: set[str]) -> None:
        """Set the folded rows, and start a fresh table widget so the clicked cell isn't left selected."""
        st.session_state[COLLAPSED_KEY] = paths
        st.session_state[TABLE_GEN_KEY] += 1

    def on_click() -> None:
        """A click on a parent's label folds or unfolds it; a click on a value adds it to the chart."""
        for pos, column in st.session_state[table_key].selection.cells:
            row = shown.iloc[pos]
            if column == LABEL_COLUMN:
                if row["has_children"]:
                    refold(collapsed ^ {row["path"]})
                continue
            label = series_label(base_label(shown.index[pos]), column)
            if label in cells and label not in st.session_state[PICKS_KEY]:
                st.session_state[PICKS_KEY] = [*st.session_state[PICKS_KEY], label]

    # The multiselect's state is set before the script runs, so the chart can be drawn above it.
    picks = st.session_state[PICKS_KEY]
    if picks:
        chart = chart_series(data, [cells[p] for p in picks])
        st.line_chart(chart, y=list(chart.columns), color=[_color(label, picks) for label in chart.columns])
    else:
        st.caption("Select cells to chart them.")
    st.multiselect("Charted series", options=list(cells), key=PICKS_KEY)

    parents = set(outline(table).query("has_children")["path"])
    caption, expand, fold = st.columns([6, 1, 1], vertical_alignment="bottom")
    caption.caption(
        "Latest value. Click a ▸/▾ label to expand or collapse it; ↳ rows are the inputs a value was "
        "computed from, at the same period. Click a value to chart it."
    )
    expand.button("Expand all", on_click=refold, args=(set(),), width="stretch")
    fold.button("Collapse all", on_click=refold, args=(parents,), width="stretch")

    # Labels go in a pinned column rather than the index, so clicking one is an ordinary cell selection.
    countries = list(table.columns)
    view = shown[countries].reset_index(drop=True)
    view.insert(0, LABEL_COLUMN, shown["display"].to_numpy())
    # Number formats differ by row, which column_config can't express, so format with a Styler.
    is_header = shown.index.isin(header_labels(layout))
    row_style = [HEADER_STYLE if h else INPUT_STYLE if i else "" for h, i in zip(is_header, shown["is_input"])]
    percent = [i for i, r in enumerate(shown.index) if is_percent_row(r)]
    other = [i for i in range(len(shown)) if i not in percent]
    styled = (
        view.style.format("{:.2%}", subset=pd.IndexSlice[percent, countries], na_rep="")
        .format("{:,.2f}", subset=pd.IndexSlice[other, countries], na_rep="")
        .apply(row_shades, axis=1, subset=countries)  # each row shaded between its own min and max
        .apply(lambda row: [row_style[row.name]] * len(row), axis=1)
    )
    st.dataframe(
        styled,
        hide_index=True,
        column_config={LABEL_COLUMN: st.column_config.TextColumn(LABEL_COLUMN, pinned=True)},
        on_select=on_click,
        selection_mode="single-cell",
        key=table_key,
    )


main()
