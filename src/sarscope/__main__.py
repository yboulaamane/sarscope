"""``sarscope`` command line.

sarscope target CHEMBL5145              check an ID resolves to what you think
sarscope fetch  CHEMBL5145              download and summarise the raw records
sarscope run    CHEMBL5145 --out DIR    the full analysis and report
sarscope run    --input my.csv --out DIR
sarscope predict --model report/ --input new.csv
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
from sarscope.sources.chembl import ChemblClient, ChemblError, normalise_target_id
from sarscope.sources.pubchem import PubChemError

#: Fields summarised by ``sarscope fetch``: each is a curation decision.
FETCH_SUMMARY_FIELDS: tuple[str, ...] = (
    "standard_relation",
    "standard_units",
    "assay_type",
    "assay_variant_mutation",
    "potential_duplicate",
    "data_validity_comment",
    "bao_label",
    "src_id",
)
POOLED_ACTIVITY_TYPES: tuple[str, ...] = ("IC50", "Ki", "Kd", "EC50")


def _types(args: argparse.Namespace) -> tuple[str, ...]:
    return POOLED_ACTIVITY_TYPES if args.pool_types else tuple(args.types)


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
        types = _types(args)
        n = client.count_activities(args.target, types)
        print(f"{n} {'/'.join(types)} records")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    with _client(args) as client:
        _describe_target(client, args.target)
        records = client.activities(args.target, _types(args))
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
        standard_types=_types(args),
        source_ids=tuple(args.source_ids) if args.source_ids else None,
        relations=("=", "<", ">", "<=", ">=") if args.keep_censored else ("=",),
        assay_types=tuple(args.assay_types),
        variant=variant,
        max_document_year=args.max_year,
    )
    landscape = LandscapeParams(fingerprints=tuple(args.fingerprints))
    defaults = ModelParams()
    model = ModelParams(
        algorithms=defaults.algorithms if args.algorithms is None else tuple(args.algorithms),
        regression_algorithms=(
            defaults.regression_algorithms
            if args.regression_algorithms is None
            else tuple(args.regression_algorithms)
        ),
        split=args.split,
        time_cutoff=args.time_cutoff,
        source_test_id=args.source_test_id,
        cv_folds=args.cv_folds,
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
        "year_col": args.input_year_col,
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


def cmd_predict(args: argparse.Namespace) -> int:
    from sarscope.predict import MODEL_FILENAME, load_bundle, predict_table

    bundle = load_bundle(args.model)
    predictions = predict_table(
        bundle,
        args.input,
        id_col=args.input_id_col,
        smiles_col=args.input_smiles_col,
    )
    if args.out is not None:
        out = args.out
    else:
        model_dir = args.model if args.model.is_dir() else args.model.parent
        out = model_dir / "predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    print(
        f"{len(predictions)} predictions written to {out} "
        f"({int(predictions['in_applicability_domain'].sum())} in domain; "
        f"model {bundle.algorithm}, artifact {MODEL_FILENAME})"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from sarscope.compare import run_compare, write_comparison

    variant = None if args.variant in (None, "none", "wild-type") else args.variant
    params = RunParams(
        curation=CurationParams(
            standard_types=_types(args),
            relations=("=", "<", ">", "<=", ">=") if args.keep_censored else ("=",),
            assay_types=tuple(args.assay_types),
            variant=variant,
            max_document_year=args.max_year,
        )
    )
    with _client(args) as client:
        result = run_compare(args.target_a, args.target_b, params, client)
    out = args.out or Path(
        f"sarscope_compare_{normalise_target_id(args.target_a)}_"
        f"{normalise_target_id(args.target_b)}"
    )
    path = write_comparison(result, out)
    print(
        f"comparison written to {path}: {len(result.compounds)} shared compounds, "
        f"{len(result.scaffolds)} shared scaffolds"
    )
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    from sarscope.screen import (
        read_screen_snapshot,
        run_screen,
        run_screen_snapshot,
        write_screen,
    )
    from sarscope.sources.pubchem import PubChemClient

    if args.snapshot is not None:
        aid, rows, smiles, meta = read_screen_snapshot(args.snapshot)
        if args.aid != aid:
            raise ValueError("requested AID differs from the PubChem snapshot")
        result = run_screen_snapshot(
            aid, rows, smiles, meta, cv_folds=args.cv_folds, seed=args.seed
        )
    else:
        with PubChemClient() as client:
            result = run_screen(args.aid, client, cv_folds=args.cv_folds, seed=args.seed)
    path = write_screen(result, args.out)
    print(f"qualitative PubChem AID {args.aid} screen written to {path}")
    return 0


def cmd_benchmark_freeze(args: argparse.Namespace) -> int:
    from sarscope.benchmark import freeze_benchmark

    with _client(args) as client:
        path = freeze_benchmark(args.out, client)
    print(f"frozen multi-family benchmark manifest written to {path}")
    return 0


def cmd_benchmark_run(args: argparse.Namespace) -> int:
    from sarscope.benchmark import run_benchmark

    path = run_benchmark(args.frozen, args.out, validation=args.validation)
    print(f"multi-family benchmark results written to {path}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sarscope", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"sarscope {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--types", nargs="+", default=["IC50"], help="ChEMBL standard_type(s)")
    common.add_argument(
        "--pool-types",
        action="store_true",
        help="pool IC50, Ki, Kd and EC50 after explicit -log10(molar) conversion",
    )
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
    r.add_argument("--input-year-col", help="document-year column, required by --split time")
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
    r.add_argument(
        "--source-ids",
        nargs="+",
        type=int,
        help="keep ChEMBL activity origins, e.g. 1 literature, 7 PubChem, 37 BindingDB",
    )
    r.add_argument("--max-year", type=int, help="only documents published up to this year")
    r.add_argument("--fingerprints", nargs="+", default=list(LandscapeParams().fingerprints))
    r.add_argument("--algorithms", nargs="+", help='names from analysis.model, or "all"')
    r.add_argument(
        "--regression-algorithms",
        nargs="+",
        help='names from analysis.regression, or "all"',
    )
    r.add_argument("--split", choices=["scaffold", "random", "time", "source"], default="scaffold")
    r.add_argument(
        "--time-cutoff",
        type=int,
        default=2019,
        help="with --split time: train through this year and test on later compounds",
    )
    r.add_argument(
        "--source-test-id", type=int, help="with --split source: hold out a ChEMBL src_id"
    )
    r.add_argument("--cv-folds", type=int, default=10)
    r.add_argument("--no-leakage-audit", action="store_true")
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(func=cmd_run)

    q = sub.add_parser(
        "predict",
        help="predict pActivity for new compounds with a saved report model",
    )
    q.add_argument("--model", type=Path, required=True, help="report folder or model.joblib")
    q.add_argument("--input", type=Path, required=True, help="CSV/TSV with IDs and SMILES")
    q.add_argument("--out", type=Path, help="output CSV (default: REPORT/predictions.csv)")
    q.add_argument("--input-id-col", default="molecule_id")
    q.add_argument("--input-smiles-col", default="smiles")
    q.set_defaults(func=cmd_predict)

    c = sub.add_parser(
        "compare",
        parents=[common],
        help="compare compound and scaffold selectivity between two ChEMBL targets",
    )
    c.add_argument("target_a")
    c.add_argument("target_b")
    c.add_argument("--out", type=Path, help="output folder (a target-based name by default)")
    c.add_argument("--variant", help='mutant to keep; "any" pools variants')
    c.add_argument("--keep-censored", action="store_true")
    c.add_argument("--assay-types", nargs="+", default=["B"])
    c.add_argument("--max-year", type=int)
    c.set_defaults(func=cmd_compare)

    screen = sub.add_parser("screen", help="classify qualitative outcomes for one PubChem AID")
    screen.add_argument("aid", type=int, help="PubChem BioAssay ID")
    screen.add_argument("--out", type=Path, required=True)
    screen.add_argument("--cv-folds", type=int, default=3)
    screen.add_argument("--seed", type=int, default=42)
    screen.add_argument("--snapshot", type=Path, help="rerun offline from source_snapshot.json")
    screen.set_defaults(func=cmd_screen)

    freeze = sub.add_parser(
        "benchmark-freeze", parents=[common], help="freeze three ChEMBL target-family snapshots"
    )
    freeze.add_argument("--out", type=Path, required=True)
    freeze.set_defaults(func=cmd_benchmark_freeze)

    bench = sub.add_parser("benchmark-run", help="run the predeclared frozen benchmark offline")
    bench.add_argument("--frozen", type=Path, required=True)
    bench.add_argument("--out", type=Path, required=True)
    bench.add_argument(
        "--validation",
        choices=["scaffold", "origin"],
        default="scaffold",
        help="origin holds out BindingDB-origin-unique compounds (ChEMBL src_id 37)",
    )
    bench.set_defaults(func=cmd_benchmark_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (ChemblError, PubChemError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
