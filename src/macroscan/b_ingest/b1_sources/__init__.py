"""Data sources. Importing this package registers every source in SOURCES."""

from macroscan.b_ingest.b1_sources import oecd  # noqa: F401  (registers itself)
from macroscan.b_ingest.b1_sources.base import SOURCES, Source

__all__ = ["SOURCES", "Source"]
