"""Raw-coordinate coefficient × training-SD maps, never binding-site claims.

The Gaussian fields use a positive-sum convention, including atom-centred
donor/acceptor indicators. LJ steric is an energy, not a volume: a positive
coefficient is NOT unconditionally 'bulk favoured'. The output spells this
out to avoid reversing medicinal-chemistry interpretations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sarscope.analysis.fields import FieldResult, GridSpec
from sarscope.params import FieldQsarParams

COLORS = {
    "steric": ("green", "yellow"),
    "electrostatic": ("blue", "red"),
    "hydrophobic": ("orange", "white"),
    "hbond_donor": ("cyan", "purple"),
    "hbond_acceptor": ("magenta", "salmon"),
}


@dataclass
class ContourResult:
    maps: dict[str, NDArray[np.float64]]
    summary: pd.DataFrame
    levels: dict[str, tuple[float | None, float | None]]


def contribution_maps(
    fields: FieldResult, fitted_filter: Any, model: Any, params: FieldQsarParams
) -> ContourResult:
    """Map coefficients back to the full grid, zeroing unmodelled locations.

    PLS was fit without autoscaling on the raw fields. Multiplying by raw
    training SD therefore produces the stated standardised contribution.
    Percentages are shares of absolute model contributions, not percentages
    of binding energy or independent explained variance.
    """
    coefficient = np.zeros(len(fields.labels))
    coefficient[fitted_filter.support_] = (
        np.asarray(model.coef_).reshape(-1) / fitted_filter.column_scale_[fitted_filter.support_]
    )
    values = coefficient * fitted_filter.training_sd_
    if not np.isfinite(values).all():
        raise ValueError("Non-finite model contributions cannot be drawn.")
    names = list(dict.fromkeys(label.rsplit("_", 1)[0] for label in fields.labels))
    maps = {
        name: values[i * fields.grid.n_points : (i + 1) * fields.grid.n_points]
        for i, name in enumerate(names)
    }
    total = float(np.abs(values).sum())
    rows, levels = [], {}
    for name, grid_values in maps.items():
        positive, negative = grid_values[grid_values > 0], -grid_values[grid_values < 0]
        pos = float(np.percentile(positive, params.contour_percentile)) if len(positive) else None
        neg = -float(np.percentile(negative, params.contour_percentile)) if len(negative) else None
        levels[name] = pos, neg
        positive_meaning = {
            "steric": "More local atomic volume is associated with higher pActivity",
            "electrostatic": "More positive local charge is associated with higher pActivity",
            "hydrophobic": "More local Crippen logP associates with higher pActivity",
            "hbond_donor": "More local ligand donor character is associated with higher pActivity",
            "hbond_acceptor": "More local ligand acceptor character associates with higher potency",
        }[name]
        if name == "steric" and params.method == "comfa":
            positive_meaning = (
                "Higher MMFF buffered 14-7 probe energy is associated with higher pActivity; "
                if params.field_definition == "mmff94_mif"
                else "Higher LJ probe energy is associated with higher pActivity; "
            ) + "this is not an unconditional instruction to add bulk"
        rows.append(
            {
                "field": name,
                "absolute_contribution_percent": 100 * float(np.abs(grid_values).sum()) / total
                if total
                else 0.0,
                "retained_columns": int(
                    np.count_nonzero(
                        fitted_filter.support_.reshape(len(names), -1)[names.index(name)]
                    )
                ),
                "positive_level": pos,
                "negative_level": neg,
                "positive_interpretation": positive_meaning,
                "negative_interpretation": "Opposite association to the positive contour",
                "positive_color": COLORS[name][0],
                "negative_color": COLORS[name][1],
            }
        )
    return ContourResult(maps, pd.DataFrame(rows), levels)


def open_dx(grid: GridSpec, values: NDArray[Any]) -> str:
    """OpenDX regular grid: Cartesian Å coordinates; last (z) index fastest."""
    data = np.asarray(values, dtype=float).reshape(-1)
    if len(data) != grid.n_points or not np.isfinite(data).all():
        raise ValueError("DX needs one finite value at every grid point.")
    counts = " ".join(map(str, grid.shape))
    lines = [
        "# SARscope coefficient x training-SD; unmodelled locations are zero",
        f"object 1 class gridpositions counts {counts}",
        "origin " + " ".join(f"{x:.12g}" for x in grid.origin),
        f"delta {grid.spacing:.12g} 0 0",
        f"delta 0 {grid.spacing:.12g} 0",
        f"delta 0 0 {grid.spacing:.12g}",
        f"object 2 class gridconnections counts {counts}",
        f"object 3 class array type double rank 0 items {len(data)} data follows",
    ]
    lines.extend(" ".join(f"{x:.12g}" for x in data[i : i + 3]) for i in range(0, len(data), 3))
    lines.extend(
        [
            'attribute "dep" string "positions"',
            'object "field" class field',
            'component "positions" value 1',
            'component "connections" value 2',
            'component "data" value 3',
        ]
    )
    return "\n".join(lines) + "\n"


def pymol_script(contours: ContourResult) -> str:
    lines = [
        "# Associations, not observed protein interactions or binding energies.",
        "load reference.sdf, reference",
        "show sticks, reference",
        "bg_color grey30",
    ]
    for name, (positive, negative) in contours.levels.items():
        lines.append(f"load {name}.dx, {name}_grid")
        for sign, level, color in zip(
            ("positive", "negative"), (positive, negative), COLORS[name], strict=True
        ):
            if level is not None:
                lines.extend(
                    [
                        f"isosurface {name}_{sign}, {name}_grid, {level:.12g}",
                        f"color {color}, {name}_{sign}",
                        f"set transparency, 0.45, {name}_{sign}",
                    ]
                )
    lines.append("zoom reference")
    return "\n".join(lines) + "\n"
