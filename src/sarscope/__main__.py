"""``sarscope`` command line.

sarscope target CHEMBL5145              check an ID resolves to what you think
sarscope fetch  CHEMBL5145              download and summarise the raw records
sarscope run    CHEMBL5145 --out DIR    the full analysis and report
sarscope run    --input my.csv --out DIR
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sarscope import __version__
from sarscope.params import CurationParams, LandscapeParams, ModelParams, RunParams
from sarscope.sources.chembl import ChemblClient, ChemblError

#: Fields summarised by ``sarscope fetch``: each is a curation decision.
FETCH_SUMMARY_FIELDS: tuple[str, ...] = (
    "standard_relation",
    "standard_units",
    "assay_type",
    "assay_variant_mutation",
    "potential_duplicate",
    "data_validity_comment",
    "bao_label",
)


def default_cache_dir() -> Path:
    env = os.environ.get("SARSCOPE_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "sarscope"


def _client(args: argparse.Namespace) -> ChemblClient:
    return ChemblClient(cache_dir=None if args.no_cache else args.cache_dir)


def _describe_target(client: ChemblClient, target_id: str) -> dict[str, Any]:
    target = client.target(target_id)
    print(
        f"{target['target_chembl_id']}  {target['pref_name']}  "
        f"({target['organism']}, {target['target_type']})  [{client.release}]"
    )
    return target


def cmd_target(args: argparse.Namespace) -> int:
    with _client(args) as client:
        _describe_target(client, args.target)
        n = client.count_activities(args.target, args.types)
        print(f"{n} {'/'.join(args.types)} records")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    with _client(args) as client:
        _describe_target(client, args.target)
        records = client.activities(args.target, args.types)
    molecules = {r.get("molecule_chembl_id") for r in records}
    print(f"{len(records)} records, {len(molecules)} molecules\n")
    for name in FETCH_SUMMARY_FIELDS:
        counts = collections.Counter(r.get(name) for r in records)
        cells = ", ".join(f"{value}: {n}" for value, n in counts.most_common(6))
        more = f" (+{len(counts) - 6} more)" if len(counts) > 6 else ""
        print(f"  {name:<24} {cells}{more}")
    return 0


def build_params(args: argparse.Namespace) -> RunParams:
    variant = None if args.variant in (None, "none", "wild-type") else args.variant
    curation = CurationParams(
        standard_types=tuple(args.types),
        relations=("=", "<", ">", "<=", ">=") if args.keep_censored else ("=",),
        assay_types=tuple(args.assay_types),
        variant=variant,
        max_document_year=args.max_year,
    )
    landscape = LandscapeParams(fingerprints=tuple(args.fingerprints))
    defaults = ModelParams()
    model = ModelParams(
        algorithms=defaults.algorithms if args.algorithms is None else tuple(args.algorithms),
        split=args.split,
        leakage_audit=not args.no_leakage_audit,
        seed=args.seed,
    )
    return RunParams(curation=curation, landscape=landscape, model=model)


def cmd_run(args: argparse.Namespace) -> int:
    if (args.target is None) == (args.input is None):
        print("error: give either a target ID or --input FILE, not both", file=sys.stderr)
        return 2
    params = build_params(args)
    from sarscope import pipeline, report

    read_kwargs = {
        "id_col": args.input_id_col,
        "smiles_col": args.input_smiles_col,
        "value_col": args.input_value_col,
        "unit": args.input_unit,
    }
    try:
        if args.input is not None:
            results = pipeline.run_table(args.input, params, **read_kwargs)
        else:
            with _client(args) as client:
                _describe_target(client, args.target)
                results = pipeline.run_chembl(args.target, params, client)
        path = report.write_report(results, args.out)
    except NotImplementedError:
        print(
            "the analysis layer is not implemented yet (see `make test-science`)", file=sys.stderr
        )
        return 3
    print(f"report written to {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sarscope", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"sarscope {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--types", nargs="+", default=["IC50"], help="ChEMBL standard_type(s)")
    common.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    common.add_argument("--no-cache", action="store_true")

    t = sub.add_parser("target", parents=[common], help="resolve a target ID and count records")
    t.add_argument("target")
    t.set_defaults(func=cmd_target)

    f = sub.add_parser("fetch", parents=[common], help="download and summarise raw records")
    f.add_argument("target")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", parents=[common], help="full analysis and report")
    r.add_argument("target", nargs="?")
    r.add_argument("--input", type=Path, help="your own CSV/TSV instead of ChEMBL")
    r.add_argument("--input-id-col", default="molecule_id", help="with --input")
    r.add_argument("--input-smiles-col", default="smiles", help="with --input")
    r.add_argument("--input-value-col", default="pactivity", help="with --input")
    r.add_argument(
        "--input-unit",
        default="p",
        help='unit of --input-value-col: "p" for a -log10(M) value, or nM/uM/mM/M/pM',
    )
    r.add_argument("--out", type=Path, required=True)
    r.add_argument(
        "--variant", help='mutant to keep, e.g. V600E; "any" pools all (default: wild-type)'
    )
    r.add_argument("--keep-censored", action="store_true", help="keep >, < relations")
    r.add_argument("--assay-types", nargs="+", default=["B"])
    r.add_argument("--max-year", type=int, help="only documents published up to this year")
    r.add_argument("--fingerprints", nargs="+", default=list(LandscapeParams().fingerprints))
    r.add_argument("--algorithms", nargs="+", help='names from analysis.model, or "all"')
    r.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    r.add_argument("--no-leakage-audit", action="store_true")
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=cmd_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (ChemblError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
