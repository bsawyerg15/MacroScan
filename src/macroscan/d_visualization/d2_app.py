"""Streamlit app: latest values by indicator and country; click cells to chart them.

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
from macroscan.c_transform.c3_derived import apply_derivations
from macroscan.d_visualization.d1_frames import (
    chart_series,
    header_labels,
    is_percent_row,
    latest_table,
    series_label,
    unindent,
)

# A series keeps its color while charted; a removed series frees its color for the next one added.
PALETTE = ["#0066ff", "#d62728", "#2ca02c", "#000000", "#ff7f0e", "#7f7f7f", "#9467bd", "#17becf", "#bcbd22", "#8c564b"]  # blue, red, green, black, orange, gray, then purple, teal, olive, brown
TABLE_KEY = "latest"
PICKS_KEY = "picks"
COLORS_KEY = "colors"  # charted label -> palette color
HEADER_STYLE = "background-color: rgba(128, 128, 128, 0.15); font-weight: bold"  # section header rows


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
    table = latest_table(data, _countries(), layout)
    if table.isna().all().all():
        st.info("No ingested data yet; run `macroscan ingest` first.")
        return

    # Every cell's chart label -> (row, country), so picks can live in a multiselect.
    # Labels drop the outline indent so they match chart_series' row labels; header rows have no cells.
    cells = {
        series_label(unindent(r), c): (unindent(r), c)
        for r in table.index
        for c in table.columns
        if pd.notna(table.at[r, c])
    }

    def add_clicked_cell() -> None:
        """Add the clicked cell to the picks (clicks only add; remove in the multiselect)."""
        for pos, country in st.session_state[TABLE_KEY].selection.cells:
            label = series_label(unindent(table.index[pos]), country)
            if label in cells and label not in st.session_state[PICKS_KEY]:
                st.session_state[PICKS_KEY] = [*st.session_state[PICKS_KEY], label]

    st.session_state.setdefault(PICKS_KEY, [])
    # The multiselect's state is set before the script runs, so the chart can be drawn above it.
    picks = st.session_state[PICKS_KEY]
    if picks:
        chart = chart_series(data, [cells[p] for p in picks])
        st.line_chart(chart, y=list(chart.columns), color=[_color(label, picks) for label in chart.columns])
    else:
        st.caption("Select cells to chart them.")
    st.multiselect("Charted series", options=list(cells), key=PICKS_KEY)

    st.caption("Latest value. Click a cell to add it to the chart.")
    # Number formats differ by row, which column_config can't express, so format with a Styler.
    percent_rows = [r for r in table.index if is_percent_row(r)]
    other_rows = [r for r in table.index if r not in percent_rows]
    headers = header_labels(layout)
    styled = (
        table.style.format("{:.2%}", subset=pd.IndexSlice[percent_rows, :], na_rep="")
        .format("{:,.2f}", subset=pd.IndexSlice[other_rows, :], na_rep="")
        .apply(lambda row: [HEADER_STYLE if row.name in headers else ""] * len(row), axis=1)
        .map_index(lambda label: HEADER_STYLE if label in headers else "", axis=0)
    )
    st.dataframe(
        styled,
        on_select=add_clicked_cell,
        selection_mode="single-cell",
        key=TABLE_KEY,
    )


main()
