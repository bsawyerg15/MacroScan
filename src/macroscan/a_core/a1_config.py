"""Load the CSV config files: the indicator registry, countries, and each source's series.

An indicator is a concept ("Real GDP"), named in 00_indicators.csv. Each source's
NN_<source>_indicators.csv maps an (indicator, frequency) to how that source serves it.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from macroscan.a_core.a2_schema import PANDAS_FREQ

DEFAULT_CONFIG_DIR = Path("config")
INDICATORS_FILE = "00_indicators.csv"
COUNTRIES_FILE = "01_countries.csv"
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
class Country:
    """A country to ingest, identified by its ISO3 code."""
    code: str  # ISO3
    name: str


@dataclass(frozen=True)
class SeriesSpec:
    """One source's series for an (indicator, frequency): where to fetch it and how to filter it.

    `indicator` is a conceptual indicator from 00_indicators.csv; `filters` maps dimension
    id -> allowed codes (frequency is not among them); `start`/`end` bound the periods fetched.
    """
    source: str
    indicator: str
    frequency: str
    dataflow: str
    filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    name: str = ""

    @property
    def label(self) -> str:
        """'oecd/Real GDP/Q', for logs and error messages."""
        return f"{self.source}/{self.indicator}/{self.frequency}"


def slug(indicator: str) -> str:
    """Filesystem-safe form of an indicator name: 'Real GDP' -> 'real_gdp'. Not an identifier."""
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
    """Load and validate the countries list (ISO3 codes, no duplicates)."""
    countries: list[Country] = []
    seen: set[str] = set()
    for row in _read_rows(path, {"code"}):
        code = row["code"].upper()
        if len(code) != 3 or not code.isalpha():
            raise ConfigError(f"{path}: country code {code!r} is not ISO3")
        if code in seen:
            raise ConfigError(f"{path}: duplicate country {code}")
        seen.add(code)
        countries.append(Country(code=code, name=row.get("name", "")))
    if not countries:
        raise ConfigError(f"{path}: no countries configured")
    return countries


def load_indicators(path: Path) -> dict[str, str]:
    """Load the indicator registry: indicator name -> description."""
    indicators: dict[str, str] = {}
    slugs: dict[str, str] = {}
    for row in _read_rows(path, {"indicator"}):
        name = row["indicator"]
        if not slug(name):
            raise ConfigError(f"{path}: indicator {name!r} needs at least one letter or digit")
        clash = slugs.get(slug(name))
        if clash is not None:
            what = "duplicate indicator" if clash.lower() == name.lower() else "indicators too similar"
            raise ConfigError(f"{path}: {what}: {clash!r} and {name!r}")
        slugs[slug(name)] = name
        indicators[name] = row.get("description", "")
    if not indicators:
        raise ConfigError(f"{path}: no indicators configured")
    return indicators


def load_series(path: Path, source: str, indicators: dict[str, str]) -> list[SeriesSpec]:
    """Load and validate one source's series CSV into SeriesSpecs."""
    specs: list[SeriesSpec] = []
    for n, row in enumerate(_read_rows(path, {"indicator", "frequency", "dataflow"}), start=2):
        where = f"{path} row {n}"
        indicator, frequency = row["indicator"], row["frequency"].upper()
        if indicator not in indicators:
            raise ConfigError(f"{where}: unknown indicator {indicator!r}; add it to {INDICATORS_FILE}")
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
        if path.name == INDICATORS_FILE:
            continue
        stem = path.name[: -len(SERIES_SUFFIX)]
        source = stem.split("_", 1)[1] if stem[:1].isdigit() and "_" in stem else stem
        files[source] = path
    return files


def load_all_series(config_dir: Path = DEFAULT_CONFIG_DIR) -> list[SeriesSpec]:
    """Load every source's series CSV; each (source, indicator, frequency) may appear once."""
    indicators = load_indicators(config_dir / INDICATORS_FILE)
    specs = [s for source, path in series_files(config_dir).items() for s in load_series(path, source, indicators)]
    labels = [s.label for s in specs]
    dupes = {label for label in labels if labels.count(label) > 1}
    if dupes:
        raise ConfigError(f"each source/indicator/frequency may be configured once: {', '.join(sorted(dupes))}")
    return specs
