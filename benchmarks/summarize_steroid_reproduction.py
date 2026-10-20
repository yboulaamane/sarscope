"""Retain compact, attributed evidence without redistributing source structures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sarscope import provenance


def summarize(source: Path, out: Path) -> None:
    manifest = json.loads((source / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        if provenance.file_sha256(source / name) != expected:
            raise ValueError("Frozen evidence changed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    files = {"protocol.json", "source_accounting.csv", "membership.csv", "summary.csv"}
    historical = {
        "assessment.json",
        "predictions.csv",
        "component_search.csv",
        "input_audit.csv",
    }
    for name in manifest:
        path = Path(name)
        if (
            (len(path.parts) == 1 and path.name in files)
            or (path.parts[0] == "historical_repeat" and path.name in historical)
            or (
                path.parts[0] in {"prealigned", "rigid_mcs", "constrained"}
                and path.suffix in {".json", ".csv"}
            )
        ):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
    shutil.copy2(source / "manifest.sha256.json", out / "raw_manifest.sha256.json")
    scope = {
        "citation": "Haga et al. (2025), Pharmaceuticals 18:440, doi:10.3390/ph18030440",
        "license": "Derived numerical evidence; upstream repository CC BY-SA 4.0; paper CC BY 4.0",
        "raw_sdfs_or_algorithm_source_redistributed": False,
        "historical_repetition_not_new_dataset": True,
        "native_comparison": "All six frozen cases retained, including four alignment failures",
        "constrained_failure_diagnostic": (
            "Separate same-settings probe: EmbedMultipleConfs returned [-1], actual conformer "
            "IDs []; _generate then raised Bad Conformer Id at align3d.py:169. Failed embedding "
            "needs clearer handling; no coordinate, threshold or production code changed."
        ),
        "other_candidates_inspected": [
            {
                "doi": "10.3390/molecules26175249",
                "supplement": "PDF only in public ZIP",
                "blocker": "Selected MOE-aligned coordinates and SRD/FFD state unavailable",
            },
            {
                "doi": "10.3390/ijms26209970",
                "supplement": "PDF only in public ZIP",
                "blocker": "Three selected alignment ensembles unavailable",
            },
            {
                "dataset": "Primary Sutherland GPB",
                "records": 66,
                "train": 44,
                "test": 22,
                "blocker": "Three source structures have unresolved stereochemistry; no guessing",
            },
        ],
        "no_prediction_qualification_or_universal_alignment_claim": True,
    }
    (out / "scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    compact = {
        str(path.relative_to(out)): provenance.file_sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    (out / "manifest.sha256.json").write_text(json.dumps(compact, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.out)
