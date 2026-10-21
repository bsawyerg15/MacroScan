"""Command line entry point: `macroscan ingest` and `macroscan describe`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from macroscan.a_core.a1_config import COUNTRIES_FILE, DEFAULT_CONFIG_DIR, load_all_indicators, load_countries
from macroscan.b_ingest.b1_sources import SOURCES
from macroscan.b_ingest.b1_sources.oecd import OECDSource, describe
from macroscan.b_ingest.b2_validate import validate_frame
from macroscan.b_ingest.b3_store import DEFAULT_ROOT, Store

log = logging.getLogger("macroscan")


def ingest(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    data_dir: Path = DEFAULT_ROOT,
    only: list[str] | None = None,
    refresh_structures: bool = False,
) -> int:
    """Fetch every configured indicator and append changes. Returns the number of failures."""
    countries = [c.code for c in load_countries(config_dir / COUNTRIES_FILE)]
    specs = load_all_indicators(config_dir)
    if only:
        unknown = set(only) - {s.id for s in specs}
        if unknown:
            raise SystemExit(f"unknown indicator id(s): {', '.join(sorted(unknown))}")
        specs = [s for s in specs if s.id in only]

    store = Store(data_dir)
    sources = {}
    failures = 0
    print(f"{'indicator':<16} {'rows':>7} {'new':>7} {'revised':>8} {'withdrawn':>10}  missing countries")
    for spec in specs:
        if spec.source not in SOURCES:
            log.error("%s: unknown source %r (known: %s)", spec.id, spec.source, ", ".join(SOURCES))
            failures += 1
            continue
        source = sources.get(spec.source) or SOURCES[spec.source](data_dir / "cache", refresh_structures)
        sources[spec.source] = source
        try:
            frame = validate_frame(source.fetch(spec, countries), spec.id)
            summary = store.apply(frame, spec.source, spec.id, requested_countries=countries)
            store.record_request(spec.source, spec.id, source.last_request_url)
        except Exception as e:  # one bad indicator must not stop the run
            msg = str(e)
            log.error("%s", msg if msg.startswith(f"{spec.id}:") else f"{spec.id}: {msg}")
            failures += 1
            continue
        print(
            f"{spec.id:<16} {summary.rows_in_pull:>7} {summary.new:>7} {summary.revised:>8} "
            f"{summary.withdrawn:>10}  {', '.join(summary.countries_missing) or '-'}"
        )
    if failures:
        print(f"\n{failures} indicator(s) failed; their stored data was left unchanged.")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run `ingest` or `describe`; returns the exit code."""
    parser = argparse.ArgumentParser(prog="macroscan")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch configured indicators and store what changed")
    p_ingest.add_argument("--only", nargs="+", metavar="ID", help="ingest only these indicator ids")
    p_ingest.add_argument("--refresh-structures", action="store_true", help="re-download cached dataflow structures")

    p_desc = sub.add_parser("describe", help="list an OECD dataflow's dimensions and codes")
    p_desc.add_argument("dataflow", help="e.g. OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA")
    p_desc.add_argument("--dim", help="show every code for one dimension")
    p_desc.add_argument("--refresh-structures", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if args.command == "ingest":
        return 1 if ingest(args.config_dir, args.data_dir, args.only, args.refresh_structures) else 0
    source = OECDSource(args.data_dir / "cache", args.refresh_structures)
    try:
        print(describe(source.structure(args.dataflow), args.dim))
    except KeyError:
        raise SystemExit(f"no dimension {args.dim!r}; run without --dim to list them") from None
    return 0


if __name__ == "__main__":
    sys.exit(main())
