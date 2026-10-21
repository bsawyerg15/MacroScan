"""Load the CSV config files: which countries and which indicators to ingest."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_DIR = Path("config")
COUNTRIES_FILE = "01_countries.csv"
INDICATORS_SUFFIX = "_indicators.csv"  # e.g. 02_oecd_indicators.csv -> source "oecd"


class ConfigError(ValueError):
    """Raised when a config CSV is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Country:
    """A country to ingest, identified by its ISO3 code."""
    code: str  # ISO3
    name: str


@dataclass(frozen=True)
class IndicatorSpec:
    """One configured indicator: where to fetch it from and how to filter it.

    `filters` maps dimension id -> allowed codes; `start`/`end` bound the periods fetched.
    """
    id: str
    name: str
    source: str
    dataflow: str
    filters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    start: str | None = None
    end: str | None = None


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


def load_indicators(path: Path, source: str) -> list[IndicatorSpec]:
    """Load and validate one source's indicators CSV into IndicatorSpecs."""
    specs: list[IndicatorSpec] = []
    seen: set[str] = set()
    for row in _read_rows(path, {"id", "dataflow"}):
        spec_id = row["id"]
        if not spec_id.replace("_", "").isalnum():
            raise ConfigError(f"{path}: indicator id {spec_id!r} must be letters, digits and _")
        if spec_id in seen:
            raise ConfigError(f"{path}: duplicate indicator id {spec_id}")
        if not row["dataflow"]:
            raise ConfigError(f"{path}: indicator {spec_id} has no dataflow")
        seen.add(spec_id)
        try:
            filters = parse_filters(row.get("filters", ""))
        except ConfigError as e:
            raise ConfigError(f"{path}: indicator {spec_id}: {e}") from None
        specs.append(
            IndicatorSpec(
                id=spec_id,
                name=row.get("name") or spec_id,
                source=source,
                dataflow=row["dataflow"],
                filters=filters,
                start=row.get("start") or None,
                end=row.get("end") or None,
            )
        )
    return specs


def indicator_files(config_dir: Path) -> dict[str, Path]:
    """Map source name -> its indicators CSV, e.g. {'oecd': config/02_oecd_indicators.csv}."""
    files: dict[str, Path] = {}
    for path in sorted(config_dir.glob(f"*{INDICATORS_SUFFIX}")):
        stem = path.name[: -len(INDICATORS_SUFFIX)]
        source = stem.split("_", 1)[1] if stem[:1].isdigit() and "_" in stem else stem
        files[source] = path
    return files


def load_all_indicators(config_dir: Path = DEFAULT_CONFIG_DIR) -> list[IndicatorSpec]:
    """Load every *_indicators.csv in config_dir; ids must be unique across sources."""
    specs = [s for source, path in indicator_files(config_dir).items() for s in load_indicators(path, source)]
    ids = [s.id for s in specs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ConfigError(f"indicator ids must be unique across sources: {', '.join(sorted(dupes))}")
    return specs
