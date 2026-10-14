"""Offline audit of the current workbench against the frozen paper case studies.

This is implementation verification, not independent literature adjudication,
assay-noise calibration, user review or prospective validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase

from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.analysis.sar_map import build_sar_map
from sarscope.analysis.sar_stress import stress_test_sar
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.curate import curate_chembl
from sarscope.params import CurationParams, RunParams
from sarscope.publication import direct_position_pairs, load_cases


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(paper: Path, out: Path) -> Path:
    if out.exists():
        raise ValueError("Use a new output directory; prior evidence is preserved")
    manifest, cohorts = load_cases(paper / "cases")
    out.mkdir(parents=True)
    params = RunParams(curation=CurationParams(source_ids=(1,), variant=None))
    summaries, claims, suggestions, cycles = [], [], [], []
    inputs = {"cases/manifest.json": checksum(paper / "cases/manifest.json")}
    for case, records in cohorts:
        case_id = case["case_id"]
        curated = curate_chembl(records, params)
        table = add_scaffolds(curated.table)
        counts = table.murcko.dropna().value_counts()
        scaffold = sorted(counts.index, key=lambda item: (-counts[item], item))[0]
        series = decompose_scaffold(table, scaffold, min_substituent_count=1)
        if series is None:
            raise ValueError(f"Could not decompose {case_id}")
        result = build_sar_map(series, curated.evidence)
        scoped = curated.evidence[
            curated.evidence.molecule_id.isin(result.series.members.molecule_id)
        ]
        reference = direct_position_pairs(result.series, scoped)
        fields = ["position", "id_from", "id_to"]
        actual = result.pairs.sort_values(fields).reset_index(drop=True)
        reference = reference.sort_values(fields).reset_index(drop=True)
        if not actual[fields].equals(reference[fields]) or not np.allclose(
            actual.delta_pactivity,
            reference.delta_pactivity,
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(f"Exhaustive enumeration mismatch: {case_id}")
        frozen_path = paper / "scientific-results" / f"{case_id}-selected-pairs.csv"
        frozen = pd.read_csv(frozen_path).sort_values(fields).reset_index(drop=True)
        if not actual[fields].equals(frozen[fields]) or not np.allclose(
            actual.delta_pactivity,
            frozen.delta_pactivity,
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError(f"Scientific manuscript pair mismatch: {case_id}")
        inputs[str(frozen_path.relative_to(paper))] = checksum(frozen_path)
        inputs[f"cases/{case['file']}"] = checksum(paper / "cases" / case["file"])
        supported = result.transformations[
            result.transformations.evidence_strength.isin(["strong", "moderate"])
        ]
        case_directory = out / case_id
        case_directory.mkdir()
        for number, (_, claim) in enumerate(supported.iterrows(), start=1):
            identifier = f"{case_id}-{number:02d}"
            stress = stress_test_sar(result, claim, scoped)
            baseline = stress.checks.iloc[0]
            omissions = stress.checks[stress.checks.test != "Baseline"]
            valid = omissions[omissions.pairs > 0]
            compound = valid[valid.test == "Leave one compound out"]
            claims.append(
                {
                    "case": case_id,
                    "claim_id": identifier,
                    **stress.claim,
                    "verdict": stress.verdict,
                    "baseline_delta": baseline.median_delta,
                    "disjoint_pairs": baseline.disjoint_pairs,
                    "compound_omissions": len(compound),
                    "compound_sign_retained": bool(
                        (np.sign(compound.median_delta) == np.sign(baseline.median_delta)).all()
                    ),
                    "compound_directional_criterion_retained": bool(
                        compound.retains_direction.all()
                    ),
                    "all_evaluable_omissions": len(valid),
                    "same_directional_criterion": int(valid.retains_direction.sum()),
                    "empty_publication_omissions": int(
                        (
                            (omissions.test == "Leave one publication out") & (omissions.pairs == 0)
                        ).sum()
                    ),
                    "empty_assay_omissions": int(
                        ((omissions.test == "Leave one assay out") & (omissions.pairs == 0)).sum()
                    ),
                    "cycles": len(stress.cycles),
                    "suggestions": len(stress.suggestions),
                    "partial": any(
                        stress.settings[name]
                        for name in (
                            "omissions_truncated",
                            "cycle_search_truncated",
                            "pair_limit_reached",
                        )
                    ),
                }
            )
            for name, frame in [
                ("checks", stress.checks),
                ("thresholds", stress.thresholds),
                ("contexts", stress.contexts),
                ("cycles", stress.cycles),
                ("suggestions", stress.suggestions),
            ]:
                frame.to_csv(case_directory / f"{identifier}-{name}.csv", index=False)
            suggestions.extend(
                {"case": case_id, "claim_id": identifier, **row}
                for row in stress.suggestions.to_dict("records")
            )
            cycles.extend(
                {"case": case_id, "claim_id": identifier, **row}
                for row in stress.cycles.to_dict("records")
            )
        summaries.append(
            {
                "case": case_id,
                "selected_compounds": result.series.n_molecules,
                "excluded_core_compounds": result.excluded_core_molecules,
                "pairs": len(result.pairs),
                "transformations": len(result.transformations),
                "supported_transformations": len(supported),
                "exhaustive_reference_agreement": True,
                "scientific_manuscript_agreement": True,
            }
        )
    pd.DataFrame(summaries).to_csv(out / "case-audit.csv", index=False)
    pd.DataFrame(claims).to_csv(out / "claim-audit.csv", index=False)
    pd.DataFrame(suggestions).to_csv(out / "suggestions.csv", index=False)
    pd.DataFrame(cycles).to_csv(out / "cycles.csv", index=False)
    root = paper.parent
    source_paths = [
        Path(__file__),
        *(
            root / path
            for path in [
                "src/sarscope/analysis/rgroups.py",
                "src/sarscope/analysis/sar_map.py",
                "src/sarscope/analysis/sar_stress.py",
                "src/sarscope/curate.py",
            ]
        ),
    ]
    provenance = {
        "purpose": "Local implementation verification, not independent scientific validation",
        "protocol": manifest["protocol"],
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "input_sha256": inputs,
        "source_sha256": {str(path.relative_to(root)): checksum(path) for path in source_paths},
        "output_sha256": {
            str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*.csv"))
        },
        "limitations": [
            "Retrospective, selected three-family case panel",
            "One assay and publication per case; removing either leaves no evidence",
            "No original-table adjudication, human review or external reproduction",
            "No temporal SAR transfer evaluation, model retraining or docking",
        ],
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.paper.resolve(), args.out.resolve()))
