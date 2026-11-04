"""Load the CSV config files: countries, each source's series, transformations and the app table.

Countries are defined in `a4_countries.Country`; 01_countries.csv picks which to use by ISO3 code.

An indicator is a concept (REAL_GDP), defined in `a3_indicators.Indicator`; CSVs name it by
member name. Each source's NN_<source>_indicators.csv maps an (indicator, frequency) to how
that source serves it.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from macroscan.a_core.a2_schema import PANDAS_FREQ, Transformation
from macroscan.a_core.a3_indicators import Indicator
from macroscan.a_core.a4_countries import Country

DEFAULT_CONFIG_DIR = Path("config")
COUNTRIES_FILE = "01_countries.csv"
TRANSFORMATIONS_FILE = "03_transformations.csv"
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


@dataclass(frozen=True)
class TransformSpec:
    """Derive `new_indicator` by applying `transformation` to each series of `old_indicator`."""
    new_indicator: Indicator
    old_indicator: Indicator
    transformation: Transformation


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


def load_countries(path: Path) -> list[Country]:
    """Load the countries to ingest and show, in file order; each must be a defined Country."""
    countries: list[Country] = []
    for n, row in enumerate(_read_rows(path, {"code"}), start=2):
        try:
            country = Country.parse(row["code"])
        except ValueError as e:
            raise ConfigError(f"{path} row {n}: {e}") from None
        if country in countries:
            raise ConfigError(f"{path} row {n}: duplicate country {country}")
        countries.append(country)
    if not countries:
        raise ConfigError(f"{path}: no countries configured")
    return countries


def _indicator(text: str, where: str) -> Indicator:
    """The Indicator named `text`, or a ConfigError pointing at `where`."""
    try:
        return Indicator.parse(text)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from None


def load_transformations(path: Path) -> list[TransformSpec]:
    """Load and validate the transformations CSV; both indicators must be defined Indicators."""
    specs: list[TransformSpec] = []
    seen: set[str] = set()
    for n, row in enumerate(_read_rows(path, {"new_indicator", "old_indicator", "transformation"}), start=2):
        where = f"{path} row {n}"
        new, old = _indicator(row["new_indicator"], where), _indicator(row["old_indicator"], where)
        if new == old:
            raise ConfigError(f"{where}: {new!r} cannot be derived from itself")
        if new in seen:
            raise ConfigError(f"{where}: duplicate new indicator {new!r}")
        seen.add(new)
        try:
            transformation = Transformation.parse(row["transformation"])
        except ValueError as e:
            raise ConfigError(f"{where}: {e}") from None
        specs.append(TransformSpec(new_indicator=new, old_indicator=old, transformation=transformation))
    return specs


def load_table_indicators(path: Path) -> list[Indicator]:
    """Load the indicators the app's table shows, in file order; each must be a defined Indicator."""
    shown: list[Indicator] = []
    for n, row in enumerate(_read_rows(path, {"indicator"}), start=2):
        name = _indicator(row["indicator"], f"{path} row {n}")
        if name in shown:
            raise ConfigError(f"{path} row {n}: duplicate indicator {name!r}")
        shown.append(name)
    if not shown:
        raise ConfigError(f"{path}: no table indicators configured")
    return shown


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
