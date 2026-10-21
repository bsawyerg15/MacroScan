"""OECD SDMX REST API (https://sdmx.oecd.org/public/rest).

Indicators are configured as named dimension filters (FREQ=Q;TRANSACTION=B1GQ).
The dataflow's structure is fetched once and cached so filters can be placed in
the positional SDMX key, e.g. Q.Y.USA+DEU.S1..B1GQ.......
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import requests

from macroscan.a_core.a1_config import IndicatorSpec
from macroscan.a_core.a2_schema import FETCH_COLUMNS, PANDAS_FREQ, parse_periods
from macroscan.b_ingest.b1_sources.base import Source, register

log = logging.getLogger(__name__)

BASE_URL = "https://sdmx.oecd.org/public/rest"
STRUCTURE_ACCEPT = "application/vnd.sdmx.structure+json; version=2; charset=utf-8"
DATA_ACCEPT = "application/vnd.sdmx.data+csv; charset=utf-8; labels=both"
COUNTRY_DIM = "REF_AREA"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class OECDError(RuntimeError):
    """Raised for OECD API failures and unexpected responses."""


# --- Structure ---------------------------------------------------------------


@dataclass(frozen=True)
class Dataflow:
    """Reference to an OECD dataflow: agency, dataflow id and version."""
    agency: str
    id: str
    version: str = "latest"

    @classmethod
    def parse(cls, text: str) -> Dataflow:
        """'OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA[@1.1]' -> Dataflow."""
        agency, sep, rest = text.strip().partition(":")
        if not sep or not agency or not rest:
            raise ValueError(f"dataflow {text!r} must look like AGENCY:DATAFLOW_ID[@version]")
        parts = rest.split("@")
        # Dataflow ids contain one '@' themselves (DSD_X@DF_Y); a third part is the version.
        if len(parts) == 3:
            return cls(agency, f"{parts[0]}@{parts[1]}", parts[2])
        return cls(agency, rest)

    @property
    def label(self) -> str:
        """'AGENCY:DATAFLOW_ID', the form used in config and cache keys."""
        return f"{self.agency}:{self.id}"


@dataclass
class Dimension:
    """One dimension of a dataflow's key, with its codelist."""
    id: str
    position: int
    name: str
    codes: dict[str, str]  # code -> label
    available: list[str] | None = None  # codes that actually have data, if published


@dataclass
class Structure:
    """The parts of a dataflow's structure needed to build keys and describe it."""
    dataflow: str
    name: str
    dimensions: list[Dimension] = field(default_factory=list)

    def dimension(self, dim_id: str) -> Dimension:
        """Look up a dimension by id; raises KeyError if absent."""
        for d in self.dimensions:
            if d.id == dim_id:
                return d
        raise KeyError(dim_id)

    def to_json(self) -> str:
        """Serialize for the on-disk structure cache."""
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> Structure:
        """Load a structure previously written by to_json()."""
        raw = json.loads(text)
        return cls(raw["dataflow"], raw["name"], [Dimension(**d) for d in raw["dimensions"]])


_URN_REF = re.compile(r"=([^:]+):([^(]+)\(([^)]+)\)(?:\.(.+))?$")


def _urn_ref(urn: str) -> tuple[str, str, str, str | None]:
    """'...Codelist=OECD:CL_AREA(1.5)' -> ('OECD', 'CL_AREA', '1.5', None)."""
    m = _URN_REF.search(urn)
    if not m:
        raise OECDError(f"unrecognised URN {urn!r}")
    return m.group(1), m.group(2), m.group(3), m.group(4)


def parse_structure(payload: dict, dataflow: Dataflow) -> Structure:
    """Reduce an SDMX-JSON structure message to dimensions + codelists."""
    data = payload["data"]
    flow = next((f for f in data.get("dataflows", []) if f["id"] == dataflow.id), None)
    if flow is None:
        raise OECDError(f"structure response does not contain dataflow {dataflow.label}")
    dsd = data["dataStructures"][0]
    codelists = {(c["agencyID"], c["id"], c["version"]): c for c in data.get("codelists", [])}
    concepts = {
        (s["agencyID"], s["id"], s["version"], c["id"]): c.get("name", c["id"])
        for s in data.get("conceptSchemes", [])
        for c in s.get("concepts", [])
    }
    available: dict[str, list[str]] = {}
    for constraint in data.get("dataConstraints", []):
        if constraint.get("role") != "Actual":
            continue
        for region in constraint.get("cubeRegions", []):
            for kv in region.get("keyValues", []):
                if "values" in kv:
                    available[kv["id"]] = [v["value"] if isinstance(v, dict) else v for v in kv["values"]]

    dims = []
    for raw in dsd["dataStructureComponents"]["dimensionList"]["dimensions"]:
        enum = raw.get("localRepresentation", {}).get("enumeration")
        codes: dict[str, str] = {}
        if enum:
            cl = codelists.get(_urn_ref(enum)[:3])
            if cl:
                codes = {c["id"]: c.get("name", c["id"]) for c in cl.get("codes", [])}
        concept = raw.get("conceptIdentity")
        name = concepts.get(_urn_ref(concept)[:3] + (raw["id"],), raw["id"]) if concept else raw["id"]
        dims.append(Dimension(raw["id"], raw["position"], name, codes, available.get(raw["id"])))
    dims.sort(key=lambda d: d.position)
    return Structure(dataflow.label, flow.get("name", dataflow.id), dims)


def build_key(structure: Structure, filters: dict[str, tuple[str, ...]], countries: list[str]) -> str:
    """Place named filters into the positional SDMX key; unlisted dims are wildcards."""
    dim_ids = [d.id for d in structure.dimensions]
    unknown = [k for k in filters if k not in dim_ids]
    if unknown:
        raise ValueError(
            f"{structure.dataflow}: unknown dimension(s) {', '.join(unknown)}. "
            f"Valid dimensions: {', '.join(dim_ids)}"
        )
    if COUNTRY_DIM in filters:
        raise ValueError(f"{COUNTRY_DIM} comes from 01_countries.csv; remove it from filters")
    if COUNTRY_DIM not in dim_ids:
        raise ValueError(f"{structure.dataflow} has no {COUNTRY_DIM} dimension")

    parts = []
    for dim in structure.dimensions:
        values = tuple(countries) if dim.id == COUNTRY_DIM else filters.get(dim.id, ())
        if dim.id != COUNTRY_DIM:
            known = set(dim.available or dim.codes)
            bad = [v for v in values if known and v not in known]
            if bad:
                log.warning("%s: %s=%s has no data in this dataflow", structure.dataflow, dim.id, "+".join(bad))
        parts.append("+".join(values))
    return ".".join(parts)


# --- HTTP --------------------------------------------------------------------


class OECDClient:
    """requests wrapper with backoff on 429/5xx and a minimum gap between calls."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = 90,
        retries: int = 4,
        backoff: float = 5,
        min_interval: float = 1.5,
        session: requests.Session | None = None,
    ):
        """Configure base URL, timeout, retry count, backoff (seconds) and minimum gap between calls."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "macroscan/0.1")
        self._last_call = 0.0

    def get(self, path: str, accept: str, params: dict | None = None) -> requests.Response:
        """GET a path, sleeping between calls and retrying 429/5xx and network errors.

        Returns the final response (callers check the status); raises OECDError if the
        network keeps failing.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self.session.get(url, params=params, headers={"Accept": accept}, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == self.retries:
                    raise OECDError(f"GET {url} failed: {e}") from e
                delay = self.backoff * 2**attempt
                log.warning("%s; retrying in %.0fs", e.__class__.__name__, delay)
                time.sleep(delay)
                continue
            if resp.status_code not in RETRY_STATUSES or attempt == self.retries:
                return resp
            delay = _retry_after(resp) or self.backoff * 2**attempt
            log.warning("HTTP %s from OECD; retrying in %.0fs", resp.status_code, delay)
            time.sleep(delay)
        raise AssertionError("unreachable")


def _retry_after(resp: requests.Response) -> float | None:
    """Seconds from a Retry-After header, or None if absent or not a number."""
    try:
        return float(resp.headers.get("Retry-After", ""))
    except ValueError:
        return None


# --- Source ------------------------------------------------------------------


@register
class OECDSource(Source):
    """Source for OECD indicators configured as named dimension filters."""
    name = "oecd"

    def __init__(self, cache_dir: Path, refresh_structures: bool = False, client: OECDClient | None = None):
        """Create the source; `client` can be injected for tests."""
        super().__init__(cache_dir, refresh_structures)
        self.client = client or OECDClient()
        self._structures: dict[str, Structure] = {}

    def structure(self, dataflow: str | Dataflow) -> Structure:
        """Return a dataflow's structure, from memory, the disk cache, or the API."""
        flow = Dataflow.parse(dataflow) if isinstance(dataflow, str) else dataflow
        if flow.label in self._structures:
            return self._structures[flow.label]
        path = self.cache_dir / "oecd_structures" / f"{flow.agency}_{flow.id.replace('@', '_')}.json"
        if path.exists() and not self.refresh_structures:
            structure = Structure.from_json(path.read_text())
        else:
            resp = self.client.get(
                f"dataflow/{flow.agency}/{flow.id}/{flow.version}",
                STRUCTURE_ACCEPT,
                params={"references": "all"},
            )
            if resp.status_code == 404:
                raise OECDError(f"dataflow {flow.label} not found")
            resp.raise_for_status()
            structure = parse_structure(resp.json(), flow)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(structure.to_json())
        self._structures[flow.label] = structure
        return structure

    def fetch(self, spec: IndicatorSpec, countries: list[str]) -> pd.DataFrame:
        """Fetch one indicator for the given countries as a canonical frame.

        Returns an empty frame (with a warning) when OECD has no data for the key.
        """
        flow = Dataflow.parse(spec.dataflow)
        structure = self.structure(flow)
        key = build_key(structure, spec.filters, countries)
        params = {"dimensionAtObservation": "AllDimensions"}
        if spec.start:
            params["startPeriod"] = spec.start
        if spec.end:
            params["endPeriod"] = spec.end
        # The data endpoint takes an empty version for "latest"; the structure endpoint wants the word.
        version = "" if flow.version == "latest" else flow.version
        resp = self.client.get(f"data/{flow.agency},{flow.id},{version}/{key}", DATA_ACCEPT, params)
        self.last_request_url = resp.url
        if resp.status_code == 404 or (resp.ok and not resp.text.strip()):
            log.warning("%s: no data returned for key %s", spec.id, key)
            return normalize("", spec, structure)
        if not resp.ok:
            raise OECDError(f"HTTP {resp.status_code} from {resp.url}: {resp.text[:300]}")
        return normalize(resp.text, spec, structure)


def _code(value: str) -> str:
    """'USA: United States' -> 'USA' (labels=both CSV)."""
    return value.split(": ", 1)[0] if isinstance(value, str) else value


def normalize(csv_text: str, spec: IndicatorSpec, structure: Structure) -> pd.DataFrame:
    """Turn OECD SDMX-CSV (labels=both) into the canonical fetch frame."""
    if not csv_text.strip():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in FETCH_COLUMNS})
    raw = pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)
    raw.columns = [_code(c) for c in raw.columns]
    dim_ids = [d.id for d in structure.dimensions]
    for col in [*dim_ids, "OBS_STATUS", "UNIT_MULT"]:
        if col in raw:
            raw[col] = raw[col].map(_code)

    freqs = raw["FREQ"].unique().tolist() if "FREQ" in raw else []
    if len(freqs) != 1:
        raise ValueError(f"{spec.id}: expected one frequency, got {freqs}; add FREQ=... to filters")
    freq = freqs[0]
    if freq not in PANDAS_FREQ:
        raise ValueError(f"{spec.id}: frequency {freq!r} not supported (use one of {sorted(PANDAS_FREQ)})")

    _check_one_series_per_country(raw, spec, dim_ids)

    unit = raw["UNIT_MEASURE"] if "UNIT_MEASURE" in raw else pd.Series("", index=raw.index)
    if "UNIT_MULT" in raw:
        # Values are published scaled (e.g. millions); keep them as-is and say so in the unit.
        mult_labels = raw["UNIT_MULT"].map(_unit_mult_label)
        unit = unit.where(raw["UNIT_MULT"].isin(["", "0"]), unit + " (" + mult_labels + ")")

    out = pd.DataFrame(
        {
            "source": "oecd",
            "indicator": spec.id,
            "country": raw[COUNTRY_DIM],
            "period": parse_periods(raw["TIME_PERIOD"], freq),
            "frequency": freq,
            "value": pd.to_numeric(raw["OBS_VALUE"].replace("", None), errors="coerce").astype("float64"),
            "unit": unit,
            "status": raw["OBS_STATUS"].replace("", None) if "OBS_STATUS" in raw else None,
            "series_key": raw[[d for d in dim_ids if d in raw]].agg(".".join, axis=1),
        }
    )
    return out.sort_values(["country", "period"]).reset_index(drop=True)[FETCH_COLUMNS]


def _unit_mult_label(code: str) -> str:
    """Label for UNIT_MULT codes, e.g. '6' -> 'millions'."""
    names = {"3": "thousands", "6": "millions", "9": "billions", "12": "trillions"}
    return names.get(code, f"x1e{code}")


def _check_one_series_per_country(raw: pd.DataFrame, spec: IndicatorSpec, dim_ids: list[str]) -> None:
    """Filters must narrow the dataflow to a single series per country."""
    series_dims = [d for d in dim_ids if d != COUNTRY_DIM and d in raw]
    per_country = raw.groupby(COUNTRY_DIM)[series_dims].nunique()
    varying = [d for d in series_dims if (per_country[d] > 1).any()]
    if varying:
        detail = "; ".join(f"{d} in {sorted(raw[d].unique())[:8]}" for d in varying)
        raise ValueError(
            f"{spec.id}: filters match several series per country. "
            f"Pin these dimensions in filters: {detail}"
        )


def describe(structure: Structure, dim: str | None = None, limit: int = 25) -> str:
    """Human-readable listing of a dataflow's dimensions and their codes."""
    lines = [f"{structure.dataflow} — {structure.name}", ""]
    dims = [structure.dimension(dim)] if dim else structure.dimensions
    for d in dims:
        codes = d.available if d.available is not None else list(d.codes)
        note = " (filled from 01_countries.csv)" if d.id == COUNTRY_DIM else ""
        lines.append(f"{d.position:>2}. {d.id} — {d.name}: {len(codes)} codes with data{note}")
        shown = codes if dim else codes[:limit]
        for c in shown:
            lines.append(f"      {c:<12} {d.codes.get(c, '')}")
        if len(shown) < len(codes):
            lines.append(f"      … {len(codes) - len(shown)} more (macroscan describe {structure.dataflow} --dim {d.id})")
    return "\n".join(lines)
