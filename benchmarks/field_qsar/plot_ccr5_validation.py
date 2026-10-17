"""Plot frozen validation evidence without fitting, changing or selecting a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(evidence: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError("Keep the previous figure; choose a new output path.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, name, label in zip(
        axes[0],
        ["primary", "historical-63-12"],
        ["Default random split · 19 test", "Historical split · 12 test"],
        strict=True,
    ):
        data = pd.read_csv(evidence / name / "predictions.csv").query("split == 'held_out'")
        stats = pd.read_csv(evidence / name / "statistics.csv").set_index("algorithm")
        lo = min(data.pactivity.min(), data.prediction.min(), data.baseline_prediction.min()) - 0.2
        hi = max(data.pactivity.max(), data.prediction.max(), data.baseline_prediction.max()) + 0.2
        ax.plot(
            [lo, hi],
            [lo, hi],
            color="#71717a",
            linestyle="--",
            linewidth=1,
            label="Perfect prediction",
        )
        ax.scatter(
            data.pactivity, data.prediction, color="#c2410c", s=38, label="3D field PLS", zorder=3
        )
        ax.scatter(
            data.pactivity,
            data.baseline_prediction,
            color="#0f766e",
            marker="^",
            s=38,
            label="2D RF",
            zorder=3,
        )
        ax.set(
            xlabel="Observed deposited log activity",
            ylabel="Predicted log activity",
            xlim=(lo, hi),
            ylim=(lo, hi),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{label}\n3D R² {stats.loc['field_pls', 'test_r2']:.3f}; "
            f"RF R² {stats.loc['random_forest', 'test_r2']:.3f}"
        )
        ax.legend(fontsize=8, loc="upper left")
    data = pd.read_csv(evidence / "additional-splits.csv")
    values = [pd.read_csv(evidence / "primary/statistics.csv").iloc[0].test_r2, *data.test_r2]
    labels = ["42\ndefault", *data.seed.astype(str)]
    ax = axes[1, 0]
    ax.bar(labels, values, color=["#c2410c", *(["#a16207"] * len(data))])
    ax.axhline(0.6, color="#71717a", linestyle="--", label="Existing R² > 0.6 check")
    ax.set(
        xlabel="Fixed random split seed",
        ylabel="Held-out predictive R²",
        ylim=(0, 1),
        title="Additional splits: all outcomes retained",
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    stability = pd.read_csv(evidence / "stability.csv")
    selected = stability[~stability.kind.isin(["baseline", "numerical_control"])]
    x = np.arange(len(selected))
    ax.barh(
        x,
        selected.oof_change_max,
        color=["#0f766e" if v else "#b91c1c" for v in selected.engineering_stability_flag],
    )
    ax.set_yticks(x, selected.variant.str.replace("_", " "))
    ax.axvline(1.0, color="#71717a", linestyle="--", label="Maximum-change limit")
    ax.set(
        xlabel="Maximum change in training OOF prediction (log units)",
        title="Pose uncertainty remains material",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CCR5 retrospective validation · not independently validated", fontsize=15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.evidence, args.out)


if __name__ == "__main__":
    main()
