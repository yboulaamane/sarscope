"""Freeze a Gaussian candidate after sampling gates, then evaluate it once.

Freezing is an auditable protocol/code identity check, NOT certification.
No fresh labels are downloaded by freeze. Evaluation refuses source/config
drift and delegates to the real native campaign without copying CV/model code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from sarscope import provenance
from sarscope.field_campaign import _json, run_campaign
from sarscope.params import FieldQsarParams, ModelParams


def workflow_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    names = [
        "params.py",
        "field_qsar.py",
        "field_campaign.py",
        "field_qualification.py",
        "field_sensitivity.py",
        "analysis/fields.py",
        "analysis/align3d.py",
        "analysis/field_filter.py",
        "analysis/regression.py",
        "analysis/contours.py",
        "analysis/tripos_reference.py",
    ]
    return {name: provenance.file_sha256(root / name) for name in names}


def freeze_candidate(sensitivity: Path, protocol: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("A freeze is immutable; use a new file and retain the old record.")
    manifest = json.loads((sensitivity / "manifest.sha256.json").read_text())
    for name, digest in manifest.items():
        path = (sensitivity / name).resolve()
        if not path.is_relative_to(sensitivity.resolve()) or provenance.file_sha256(path) != digest:
            raise ValueError("Sensitivity evidence hash/path differs.")
    table = pd.read_csv(sensitivity / "summary.csv")
    spec = json.loads(protocol.read_text())
    qualification = spec.get("qualification", {})
    profile = qualification.get("profile", "candidate_block_1A")
    candidate = table[table.profile.eq(profile)]
    required = set(qualification.get("training_cohorts", ["AChE", "CCR5", "THR"]))
    if not required or candidate.empty:
        raise ValueError("The declared sensitivity profile/cohorts must be nonempty.")
    if set(candidate.dataset) != required or len(candidate) != 10 * len(required):
        raise ValueError(
            "Expected all ten candidate diagnostics in every declared training cohort."
        )
    expected_settings = as_plain(FieldQsarParams(**spec["fields"]))
    expected_model = as_plain(ModelParams(**spec["model"]))
    expected_variants = {
        "baseline",
        "mmff94",
        "phase_x",
        "phase_y",
        "phase_z",
        "phase_xyz",
        "pose_1101",
        "pose_1102",
        "pose_1103",
        "common_translation",
    }
    for name in required:
        subset = candidate[candidate.dataset.eq(name)]
        if set(subset.variant) != expected_variants or not subset.variant.is_unique:
            raise ValueError("Missing or duplicate candidate diagnostic variants.")
        gates = subset[subset.kind.isin(["baseline", "grid_phase", "numerical_control"])]
        record = json.loads((sensitivity / name / profile / "provenance.json").read_text())
        if (
            len(gates) != 6
            or not gates.completed.all()
            or not gates.engineering_stability_flag.all()
            or record["held_out_rows_used"] != 0
            or record["held_out_labels_used"]
        ):
            raise ValueError("Training-only sampling/numerical gates did not pass.")
        actual_settings = record["settings"].copy()
        for key in ("permutations", "permutation_timeout"):
            actual_settings.pop(key, None)
        compared_settings = expected_settings.copy()
        for key in ("permutations", "permutation_timeout"):
            compared_settings.pop(key, None)
        if actual_settings != compared_settings or record["model"] != expected_model:
            raise ValueError("Candidate settings/model differ from tested sensitivity profile.")
        if qualification and record.get("workflow_sha256") != workflow_hashes():
            raise ValueError("Sensitivity evidence was produced by different workflow sources.")
    fields = spec["fields"]
    if (
        fields["method"] != "comsia"
        or fields["spacing"] != 1.0
        or fields["field_scaling"] != "block_std"
        or fields["charge_model"] != "gasteiger"
        or fields["alignment"] != "prealigned"
    ):
        raise ValueError("Candidate differs from the prespecified reviewed-SDF Gaussian profile.")
    record = {
        "candidate_id": spec["candidate_id"],
        "protocol_sha256": provenance.file_sha256(protocol),
        "sensitivity_manifest_sha256": provenance.file_sha256(sensitivity / "manifest.sha256.json"),
        "workflow_sha256": workflow_hashes(),
        "sampling_gates_passed": True,
        "pose_warning_count": int(
            (~candidate[candidate.kind.eq("relative_pose")].engineering_stability_flag).sum()
        ),
        "package_or_model_certified": False,
        "fresh_labels_downloaded_for_freeze": False,
        "interpretation": "Immutable candidate, not validation or a winning-model claim",
        "versions": provenance.package_versions(),
        "profile": profile,
        "scope": qualification.get("scope", "Original five-field Gaussian candidate"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    _json(out, record)
    return record


def claim_evaluation(freeze: Path, protocol: Path, out: Path) -> dict[str, Any]:
    """Atomically consume this freeze before fetching test labels; failures stay visible.

    Reproduction is possible using an explicitly new freeze/record, but cannot
    silently reuse this qualification attempt after seeing its test scores.
    A failed attempt is not automatically retried with a new output folder.
    """
    record = verify_freeze(freeze, protocol)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty evaluation directory; preserve earlier evidence.")
    receipt = freeze.with_suffix(".evaluation-started.json")
    with receipt.open("x") as handle:
        json.dump(
            {
                "freeze_sha256": provenance.file_sha256(freeze),
                "candidate_id": record["candidate_id"],
                "output": str(out.resolve()),
                "meaning": "Attempt claimed before test download; not a successful validation",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return record


def as_plain(params: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(params)))


def verify_freeze(freeze: Path, protocol: Path) -> dict[str, Any]:
    record = json.loads(freeze.read_text())
    if record["protocol_sha256"] != provenance.file_sha256(protocol):
        raise ValueError("Protocol differs from the frozen candidate; no evaluation started.")
    if record["workflow_sha256"] != workflow_hashes():
        raise ValueError("Workflow source differs from freeze; no evaluation started.")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.sensitivity is None:
            parser.error("freeze requires --sensitivity")
        print(json.dumps(freeze_candidate(args.sensitivity, args.protocol, args.freeze), indent=2))
        return
    if args.out is None:
        parser.error("evaluate requires --out")
    record = claim_evaluation(args.freeze, args.protocol, args.out)
    protocol = json.loads(args.protocol.read_text())
    payloads: dict[str, dict[str, bytes]] = {}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for source in protocol["datasets"]:
            payloads[source["name"]] = {}
            for split in ("train", "test"):
                response = client.get(protocol["data_base_url"] + f"{source['name']}_{split}.sdf")
                response.raise_for_status()
                payloads[source["name"]][split] = response.content
    result = run_campaign(protocol, payloads, args.out)
    # Additional immutable-stage link does not alter the campaign's existing manifest.
    _json(
        args.out.with_suffix(".freeze-link.json"),
        {
            "freeze_sha256": provenance.file_sha256(args.freeze),
            "candidate_id": record["candidate_id"],
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
