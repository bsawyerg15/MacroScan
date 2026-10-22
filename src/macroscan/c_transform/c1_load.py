"""Pull the ingested data into the transform stage.

Reads today's values from the b_ingest store, one long frame per frequency
(periods of different frequencies can't share a column dtype).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macroscan.a_core.a2_schema import PANDAS_FREQ
from macroscan.b_ingest.b3_store import DEFAULT_ROOT, Store


def load_ingested(data_dir: Path = DEFAULT_ROOT, frequencies: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Latest ingested data as {frequency: long frame}, skipping frequencies with nothing stored."""
    store = Store(data_dir)
    frames = {f: store.read_all_latest(f) for f in frequencies or PANDAS_FREQ}
    return {f: df for f, df in frames.items() if len(df)}


__all__ = ["load_ingested"]
