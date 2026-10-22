"""Streamlit app: latest values by indicator and country; click cells to chart them.

Run from the repo root: `streamlit run src/macroscan/d_visualization/d2_app.py`
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from macroscan.a_core.a1_config import COUNTRIES_FILE, DEFAULT_CONFIG_DIR, load_countries
from macroscan.b_ingest.b3_store import DEFAULT_ROOT
from macroscan.c_transform.c1_load import load_ingested
from macroscan.d_visualization.d1_frames import chart_series, latest_table, series_label

# Colors go to lines by pick order, not alphabetically, so adding a series never recolors earlier ones.
PALETTE = ["#0066ff", "#d62728", "#2ca02c", "#000000", "#ff7f0e", "#7f7f7f", "#9467bd", "#17becf", "#bcbd22", "#8c564b"]  # blue, red, green, black, orange, gray, then purple, teal, olive, brown
TABLE_KEY = "latest"
PICKS_KEY = "picks"


@st.cache_data
def _data() -> dict[str, pd.DataFrame]:
    """Ingested data, read once per session."""
    return load_ingested(DEFAULT_ROOT)


@st.cache_data
def _countries() -> list[str]:
    """Country codes in config order."""
    return [c.code for c in load_countries(DEFAULT_CONFIG_DIR / COUNTRIES_FILE)]


def main() -> None:
    """Render the table and the chart of picked cells."""
    st.set_page_config(page_title="MacroScan", layout="wide")
    st.title("MacroScan")
    data = _data()
    table = latest_table(data, _countries())
    if table.empty:
        st.info("No ingested data yet; run `macroscan ingest` first.")
        return

    # Every cell's chart label -> (row, country), so picks can live in a multiselect.
    cells = {series_label(r, c): (r, c) for r in table.index for c in table.columns if pd.notna(table.at[r, c])}

    def add_clicked_cell() -> None:
        """Add the clicked cell to the picks (clicks only add; remove in the multiselect)."""
        for pos, country in st.session_state[TABLE_KEY].selection.cells:
            label = series_label(table.index[pos], country)
            if label in cells and label not in st.session_state[PICKS_KEY]:
                st.session_state[PICKS_KEY] = [*st.session_state[PICKS_KEY], label]

    st.session_state.setdefault(PICKS_KEY, [])
    # The multiselect's state is set before the script runs, so the chart can be drawn above it.
    picks = st.session_state[PICKS_KEY]
    if picks:
        chart = chart_series(data, [cells[p] for p in picks])
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(chart.columns))]
        st.line_chart(chart, y=list(chart.columns), color=colors)
    else:
        st.caption("Select cells to chart them.")
    st.multiselect("Charted series", options=list(cells), key=PICKS_KEY)

    st.caption("Latest value. Click a cell to add it to the chart.")
    st.dataframe(
        table,
        column_config={c: st.column_config.NumberColumn(format="localized") for c in table.columns},
        on_select=add_clicked_cell,
        selection_mode="single-cell",
        key=TABLE_KEY,
    )


main()
