"""Append-only change log of observations, one Parquet file per (source, indicator, frequency).

Each ingest appends only the rows that are new or changed since the latest
stored value (plus tombstones, value=NaN, for observations that disappeared).
Any past state can be rebuilt with read_as_of(); read_latest() is today's data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from macroscan.a_core.a1_config import slug
from macroscan.a_core.a2_schema import FETCH_COLUMNS, STORED_COLUMNS, parse_periods

DEFAULT_ROOT = Path("data/b_ingest")
KEY = ["country", "period"]
# Fields whose change counts as a revision.
TRACKED = ["value", "status", "unit"]


@dataclass
class ChangeSummary:
    """What one ingest of a series changed."""
    new: int = 0
    revised: int = 0
    withdrawn: int = 0
    rows_in_pull: int = 0
    countries_missing: list[str] = field(default_factory=list)

    @property
    def appended(self) -> int:
        """Total rows appended to the log."""
        return self.new + self.revised + self.withdrawn


class Store:
    """Reads and writes the per-series change logs and the manifest."""
    def __init__(self, root: Path = DEFAULT_ROOT):
        """Use `root` (default data/b_ingest) as the storage directory."""
        self.root = Path(root)

    # --- paths & raw io ---------------------------------------------------

    def path(self, source: str, indicator: str, frequency: str) -> Path:
        """Parquet path for one series' change log, e.g. oecd/real_gdp_Q.parquet."""
        return self.root / source / f"{slug(indicator)}_{frequency}.parquet"

    def _find(self, indicator: str, frequency: str, source: str | None) -> Path:
        """Path for an (indicator, frequency), searching all sources when `source` is None."""
        if source:
            return self.path(source, indicator, frequency)
        matches = sorted(self.root.glob(f"*/{slug(indicator)}_{frequency}.parquet"))
        if not matches:
            raise FileNotFoundError(f"no stored data for {indicator!r} at frequency {frequency} under {self.root}")
        if len(matches) > 1:
            sources = ", ".join(m.parent.name for m in matches)
            raise ValueError(f"{indicator!r} at frequency {frequency} is stored by several sources ({sources}); pass source=")
        return matches[0]

    def load_log(self, source: str, indicator: str, frequency: str) -> pd.DataFrame:
        """The raw change log for a series (empty frame if none stored)."""
        return _read(self.path(source, indicator, frequency))

    # --- write ------------------------------------------------------------

    def apply(
        self,
        pulled: pd.DataFrame,
        source: str,
        indicator: str,
        frequency: str,
        requested_countries: list[str] | None = None,
        pulled_at: datetime | None = None,
    ) -> ChangeSummary:
        """Diff a full pull against the stored log and append what changed."""
        pulled_at = pd.Timestamp(pulled_at or datetime.now(timezone.utc))
        if pulled_at.tzinfo is None:
            pulled_at = pulled_at.tz_localize("UTC")
        pulled = pulled[pulled["value"].notna()]  # a blank value is not an observation
        summary = ChangeSummary(rows_in_pull=len(pulled))
        present = set(pulled["country"]) if len(pulled) else set()
        if requested_countries:
            summary.countries_missing = sorted(set(requested_countries) - present)
        if pulled.empty:
            return summary

        path = self.path(source, indicator, frequency)
        log_df = _read(path)
        latest = _latest(log_df)
        latest = latest[latest["value"].notna()]  # tombstoned keys count as absent
        if latest.empty:
            latest = pulled.iloc[0:0]

        merged = pulled.merge(latest[[*KEY, *TRACKED]], on=KEY, how="left", suffixes=("", "_old"), indicator=True)
        is_new = merged["_merge"] == "left_only"
        changed = pd.Series(False, index=merged.index)
        for col in TRACKED:
            changed |= ~_same(merged[col], merged[f"{col}_old"])
        is_revised = ~is_new & changed
        additions = merged.loc[is_new | is_revised, FETCH_COLUMNS]

        # Observations that vanished from a country we did get data for. A country
        # missing entirely is more likely an outage, so it is reported, not tombstoned.
        gone = latest.merge(pulled[KEY], on=KEY, how="left", indicator=True)
        gone = gone[(gone["_merge"] == "left_only") & gone["country"].isin(present)]
        tombstones = gone[FETCH_COLUMNS].assign(value=np.nan, status=None)

        summary.new, summary.revised, summary.withdrawn = int(is_new.sum()), int(is_revised.sum()), len(tombstones)
        if summary.appended:
            frames = [f for f in (additions, tombstones) if len(f)]
            appended = pd.concat(frames, ignore_index=True).assign(pulled_at=pulled_at)
            combined = pd.concat([log_df, appended], ignore_index=True) if len(log_df) else appended
            _write(path, combined[STORED_COLUMNS])
        self._update_manifest(_manifest_key(source, indicator, frequency), summary, pulled_at, path)
        return summary

    def record_request(self, source: str, indicator: str, frequency: str, url: str | None) -> None:
        """Remember the request URL used for a series' last pull."""
        manifest = self.manifest()
        manifest.setdefault(_manifest_key(source, indicator, frequency), {})["request_url"] = url
        self._write_manifest(manifest)

    # --- read -------------------------------------------------------------

    def read_history(
        self,
        indicator: str,
        frequency: str,
        source: str | None = None,
        country: str | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        """Every stored version of every observation (the revision trail)."""
        df = _read(self._find(indicator, frequency, source))
        if country:
            df = df[df["country"] == country]
        if period:
            df = df[df["period"].astype(str) == str(pd.Period(period, freq=df["period"].dtype.freq))]
        return df.sort_values([*KEY, "pulled_at"]).reset_index(drop=True)

    def read_as_of(self, indicator: str, frequency: str, when, source: str | None = None) -> pd.DataFrame:
        """The data as it looked at `when` (tz-naive is treated as UTC)."""
        when = pd.Timestamp(when)
        if when.tzinfo is None:
            when = when.tz_localize("UTC")
        df = _read(self._find(indicator, frequency, source))
        return _current(df[df["pulled_at"] <= when])

    def read_latest(self, indicator: str, frequency: str, source: str | None = None) -> pd.DataFrame:
        """Today's data: the latest value per (country, period), withdrawn rows dropped."""
        return _current(_read(self._find(indicator, frequency, source)))

    def read_all_latest(self, frequency: str, source: str | None = None) -> pd.DataFrame:
        """Latest data for every stored indicator at one frequency, so periods share a dtype."""
        paths = sorted(self.root.glob(f"{source or '*'}/*_{frequency}.parquet"))
        frames = [_current(_read(p)) for p in paths]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else _empty()

    # --- manifest ---------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        """Location of manifest.json."""
        return self.root / "manifest.json"

    def manifest(self) -> dict:
        """The manifest as a dict (empty if not written yet)."""
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    def _write_manifest(self, manifest: dict) -> None:
        """Write the manifest dict to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _update_manifest(self, key: str, summary: ChangeSummary, pulled_at, path: Path) -> None:
        """Record pull/change times, the run summary and stored row count."""
        manifest = self.manifest()
        entry = manifest.setdefault(key, {})
        entry["last_pulled_at"] = pulled_at.isoformat()
        if summary.appended:
            entry["last_changed_at"] = pulled_at.isoformat()
        entry["last_run"] = {**asdict(summary)}
        entry["stored_rows"] = int(pd.read_parquet(path, columns=["value"]).shape[0]) if path.exists() else 0
        self._write_manifest(manifest)


# --- helpers ------------------------------------------------------------------


def _manifest_key(source: str, indicator: str, frequency: str) -> str:
    """Manifest entry key for a series, e.g. 'oecd/real_gdp/Q'."""
    return f"{source}/{slug(indicator)}/{frequency}"


def _empty() -> pd.DataFrame:
    """An empty frame with the stored columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in STORED_COLUMNS})


def _write(path: Path, df: pd.DataFrame) -> None:
    """Write a change log atomically, storing periods as strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.assign(period=df["period"].astype(str))
    tmp = path.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic: a crash mid-write never corrupts the log


def _read(path: Path) -> pd.DataFrame:
    """Read a change log, restoring the Period dtype (empty frame if missing).

    Each file holds a single frequency, taken from its name (real_gdp_Q.parquet).
    """
    if not path.exists():
        return _empty()
    df = pd.read_parquet(path)
    df["period"] = parse_periods(df["period"], path.stem.rsplit("_", 1)[1])
    return df


def _current(log_df: pd.DataFrame) -> pd.DataFrame:
    """Latest value per (country, period) with withdrawn observations dropped."""
    latest = _latest(log_df)
    return latest[latest["value"].notna()].reset_index(drop=True)


def _latest(log_df: pd.DataFrame) -> pd.DataFrame:
    """Most recent stored row per (country, period)."""
    if log_df.empty:
        return log_df
    ordered = log_df.sort_values("pulled_at", kind="stable")
    return ordered.groupby(KEY, sort=True, observed=True).tail(1).sort_values(KEY).reset_index(drop=True)


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    """Elementwise equality treating null==null and near-equal floats as the same."""
    both_null = a.isna() & b.isna()
    if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
        close = pd.Series(np.isclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12), index=a.index)
        return both_null | close
    return both_null | (a.astype(object) == b.astype(object)).fillna(False).astype(bool)


__all__ = ["Store", "ChangeSummary", "DEFAULT_ROOT"]
