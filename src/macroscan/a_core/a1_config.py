"""Load the CSV config files: each source's series and the app table.

Countries are not configured here: every `a4_countries.Country` is ingested and shown.

An indicator is a concept (REAL_GDP), defined in `a3_indicators.Indicator`; CSVs name it by
member name. Each source's NN_<source>_indicators.csv maps an (indicator, frequency) to how
that source serves it. Derived indicators are defined in code, in `c3_derived`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from macroscan.a_core.a2_schema import PANDAS_FREQ
from macroscan.a_core.a3_indicators import Indicator

DEFAULT_CONFIG_DIR = Path("config")
TABLE_FILE = "04_table.csv"  # not *_indicators.csv, which would be read as a source
SERIES_SUFFIX = "_indicators.csv"  # e.g. 02_oecd_indicators.csv -> source "oecd"

# Accepted start/end formats per frequency; a bare year works for any frequency.
_PERIOD_FORMATS = {
    "A": re.compile(r"\d{4}"),
    "Q": re.compile(r"\d{4}(-?Q[1-4])?"),
    "M": re.compile(r"\d{4}(-(0[1-9]|1[0-2]))?"),
}


class ConfigError(ValueError):
    """Raised when a config CSV is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class SeriesSpec:
    """One source's series for an (indicator, frequency): where to fetch it and how to filter it.

    `filters` maps dimension id -> allowed codes (frequency is not among them); `start`/`end`
    bound the periods fetched.
    """
    source: str
    indicator: Indicator
    frequency: str
    dataflow: str
    filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    name: str = ""

    @property
    def label(self) -> str:
        """'oecd/REAL_GDP/Q', for logs and error messages."""
        return f"{self.source}/{self.indicator}/{self.frequency}"


def slug(indicator: str) -> str:
    """Filesystem-safe form of an indicator name: 'REAL_GDP' -> 'real_gdp'. Not an identifier."""
    return re.sub(r"[^a-z0-9]+", "_", indicator.lower()).strip("_")


def parse_filters(text: str) -> dict[str, tuple[str, ...]]:
    """Parse 'FREQ=Q;SEX=M+F' into {'FREQ': ('Q',), 'SEX': ('M', 'F')}."""
    filters: dict[str, tuple[str, ...]] = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ConfigError(f"filter {part!r} is not in DIM=value form")
        dim, _, values = part.partition("=")
        dim = dim.strip().upper()
        codes = tuple(v.strip() for v in values.split("+") if v.strip())
        if not dim or not codes:
            raise ConfigError(f"filter {part!r} needs both a dimension and a value")
        if dim in filters:
            raise ConfigError(f"dimension {dim} appears twice in filters {text!r}")
        filters[dim] = codes
    return filters


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read a CSV into stripped dicts, skipping blank and '#'-commented rows."""
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ConfigError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        rows = [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]
    # Skip blank lines and rows commented out with '#'.
    return [r for r in rows if any(r.values()) and not next(iter(r.values())).startswith("#")]


def _indicator(text: str, where: str) -> Indicator:
    """The Indicator named `text`, or a ConfigError pointing at `where`."""
    try:
        return Indicator.parse(text)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from None


@dataclass(frozen=True)
class TableRow:
    """One row of the app table's outline: a section header (no data) or an indicator, at a depth."""
    level: int
    header: str | None = None
    indicator: Indicator | None = None

    @property
    def is_header(self) -> bool:
        return self.indicator is None


def load_table_layout(path: Path) -> list[TableRow]:
    """Load the app table's outline, in file order.

    Each row sets `level` (depth, 0 = top) and exactly one of `header` (a section title with
    no data) or `indicator` (a defined Indicator). A row may be at most one level deeper than
    the row before it, and the first row is at level 0.
    """
    layout: list[TableRow] = []
    for n, row in enumerate(_read_rows(path, {"level", "header", "indicator"}), start=2):
        where = f"{path} row {n}"
        if not row["level"].isdigit():
            raise ConfigError(f"{where}: level {row['level']!r} must be a whole number >= 0")
        level = int(row["level"])
        if level > (layout[-1].level + 1 if layout else 0):
            raise ConfigError(f"{where}: level {level} skips a level (previous row is at {layout[-1].level if layout else 'none'})")
        if bool(row["header"]) == bool(row["indicator"]):
            raise ConfigError(f"{where}: set exactly one of header or indicator")
        if row["header"]:
            layout.append(TableRow(level, header=row["header"]))
            continue
        name = _indicator(row["indicator"], where)
        if any(r.indicator == name for r in layout):
            raise ConfigError(f"{where}: duplicate indicator {name!r}")
        layout.append(TableRow(level, indicator=name))
    if not any(not r.is_header for r in layout):
        raise ConfigError(f"{path}: no table indicators configured")
    return layout


def load_series(path: Path, source: str) -> list[SeriesSpec]:
    """Load and validate one source's series CSV into SeriesSpecs."""
    specs: list[SeriesSpec] = []
    for n, row in enumerate(_read_rows(path, {"indicator", "frequency", "dataflow"}), start=2):
        where = f"{path} row {n}"
        indicator, frequency = _indicator(row["indicator"], where), row["frequency"].upper()
        if frequency not in PANDAS_FREQ:
            raise ConfigError(f"{where}: frequency {frequency!r} must be one of {', '.join(PANDAS_FREQ)}")
        if not row["dataflow"]:
            raise ConfigError(f"{where}: no dataflow")
        try:
            filters = parse_filters(row.get("filters", ""))
        except ConfigError as e:
            raise ConfigError(f"{where}: {e}") from None
        if "FREQ" in filters:
            raise ConfigError(f"{where}: set frequency in the frequency column, not as FREQ in filters")
        for bound in ("start", "end"):
            value = row.get(bound) or ""
            if value and not _PERIOD_FORMATS[frequency].fullmatch(value):
                raise ConfigError(f"{where}: {bound} {value!r} is not a valid period for frequency {frequency}")
        specs.append(
            SeriesSpec(
                source=source,
                indicator=indicator,
                frequency=frequency,
                dataflow=row["dataflow"],
                filters=filters,
                start=row.get("start") or None,
                end=row.get("end") or None,
                name=row.get("name", ""),
            )
        )
    return specs


def series_files(config_dir: Path) -> dict[str, Path]:
    """Map source name -> its series CSV, e.g. {'oecd': config/02_oecd_indicators.csv}."""
    files: dict[str, Path] = {}
    for path in sorted(config_dir.glob(f"*{SERIES_SUFFIX}")):
        stem = path.name[: -len(SERIES_SUFFIX)]
        source = stem.split("_", 1)[1] if stem[:1].isdigit() and "_" in stem else stem
        files[source] = path
    return files


def load_all_series(config_dir: Path = DEFAULT_CONFIG_DIR) -> list[SeriesSpec]:
    """Load every source's series CSV; each (source, indicator, frequency) may appear once."""
    specs = [s for source, path in series_files(config_dir).items() for s in load_series(path, source)]
    labels = [s.label for s in specs]
    dupes = {label for label in labels if labels.count(label) > 1}
    if dupes:
        raise ConfigError(f"each source/indicator/frequency may be configured once: {', '.join(sorted(dupes))}")
    return specs
