"""Training-only comparison of the two field screens on the three cohorts.

Answers one question: does per-field screening keep both blocks populated, and
what does that do to training-only fit statistics? It fetches only the
``*_train.sdf`` files named in the sensitivity protocol, verifies their
SHA-256, and reuses the real nested regression engine through ``training_fit``.

**This is descriptive, not a predictive claim.** Out-of-fold q2 and RMSE here
are training-set statistics from one fixed configuration. They cannot establish
that either screen predicts better, and they must not be used to select a
screen for a later predictive claim: a screen chosen on these numbers would be
tuned on the same data that reported them. A predictive comparison needs a
frozen cohort that no screening decision has seen. ATA is consumed and is not
eligible.

No held-out SDF is fetched or parsed. Nothing under an existing evidence
directory is modified.

    python benchmarks/field_qsar/screen_comparison_probe.py \
        --protocol benchmarks/field_qsar/public-se-sensitivity-20261016/protocol.json \
        --out runs/screen-comparison
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from sarscope.field_sensitivity import load_training, training_fit
from sarscope.params import FieldQsarParams, ModelParams

SCREENS = ("global_raw", "per_field")


def retained_by_field(fitted: dict[str, Any], params: FieldQsarParams) -> dict[str, int]:
    """Kept columns per block, read from the fitted filter's support mask."""
    filt = fitted.get("filter")
    names = params.field_subset or ("steric", "electrostatic")
    if filt is None or not hasattr(filt, "support_"):
        return dict.fromkeys(names, -1)
    support = np.asarray(filt.support_)
    width = len(support) // len(names)
    return {
        name: int(support[index * width : (index + 1) * width].sum())
        for index, name in enumerate(names)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new empty directory; preserve earlier evidence.")
    args.out.mkdir(parents=True, exist_ok=True)

    profile = protocol["profiles"][0]
    base = {**protocol["fields"], **profile["settings"]}
    model = ModelParams(**protocol["model"])

    rows = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for source in protocol["datasets"]:
            name = source["name"]
            response = client.get(protocol["data_base_url"] + name + "_train.sdf")
            response.raise_for_status()
            dataset = load_training(response.content, source)
            for screen in SCREENS:
                params = FieldQsarParams(**{**base, "field_screen": screen})
                print(f"{name} / {screen} …", flush=True)
                fitted = training_fit(dataset, params, model)
                kept = retained_by_field(fitted, params)
                rows.append(
                    {
                        "cohort": name,
                        "n_training": len(dataset.table),
                        "screen": screen,
                        "training_oof_q2": fitted["q2"],
                        "training_oof_rmse": fitted["rmse"],
                        "components": fitted["components"],
                        "retained_total": fitted["retained_columns"],
                        **{f"retained_{field}": count for field, count in kept.items()},
                        "empty_fields": sum(1 for count in kept.values() if count == 0),
                    }
                )

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "screen-comparison.csv", index=False)
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "scope": (
                    "Training-only descriptive comparison of field screens. Not a "
                    "predictive claim and not a basis for selecting a screen."
                ),
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "protocol": str(args.protocol),
                "profile": profile["name"],
                "screens": list(SCREENS),
                "held_out_data_used": False,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            indent=2,
        )
        + "\n"
    )
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
