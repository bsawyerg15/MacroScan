"""The interface every data source implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from macroscan.a_core.a1_config import SeriesSpec


class Source(ABC):
    """A data provider. fetch() returns a frame with a2_schema.FETCH_COLUMNS."""

    name: str

    def __init__(self, cache_dir: Path, refresh_structures: bool = False):
        """Set up the source with a directory for cached metadata."""
        self.cache_dir = cache_dir
        self.refresh_structures = refresh_structures
        self.last_request_url: str | None = None

    @abstractmethod
    def fetch(self, spec: SeriesSpec, countries: list[str]) -> pd.DataFrame:
        """Fetch one series (indicator at a frequency) for the given countries as a canonical frame."""


SOURCES: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    """Class decorator that adds a Source to SOURCES under its `name`."""
    SOURCES[cls.name] = cls
    return cls
