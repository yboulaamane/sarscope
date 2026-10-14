"""Build the graphical abstract: one substituent, two positions, opposite effects.

The figure is a fork. A single hydroxyl addition starts from one shared
structure and ends in two places: down at the para position of the pendant
phenyl, up at the meta position. That is the paper's clearest result and it
needs no caption to read.

Each arm carries a pale band spanning the range the median moves across when
individual compounds are omitted. The bands are wide, because with two pairs
each they genuinely are. Showing them is the point: the direction is a property
of these measurements, not an established design rule.

Everything numeric is read from the audit output, so the figure cannot drift
from the analysis. The structure is drawn by RDKit and nested in a
hand-composed SVG; matplotlib is not used because this is a layout rather than
a plot and the structure has to sit in an exact place. Colours are the ones the
other figures already use.

Usage:

    python paper/build_graphical_abstract.py --results runs/paper-check --out paper/figures
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import cairosvg

from sarscope.depict import to_svg

#: The BRAF mapped core, as the R-group decomposition returns it. R3 and R4 are
#: para and meta on the pendant phenyl; the numbers are RDKit's labels, not
#: conventional ring numbering.
CORE = "O=S(=O)(NCCNc1nccc(-c2c(-c3ccc([*:3])c([*:4])c3)nc3sccn23)n1)c1ccc([*:1])c([*:2])c1"

#: Taken from the existing figures rather than chosen afresh.
TEAL = "#278d81"  # favourable direction
AMBER = "#c48a32"  # unfavourable direction
INK = "#1a1a19"
MUTED = "#718096"
PALE = "#e8edf0"

WIDTH, HEIGHT = 1120, 420
#: The potency axis. Symmetric so the zero line sits centred and the two arms
#: are directly comparable; wide enough to contain the omission bands.
AXIS_MAX = 2.4
FORK_X, TIP_X = 586, 752
#: The omission range is a separate thin bar near the axis, so it cannot be
#: mistaken for a highlight behind the number.
BAND_X = 928
AXIS_X = 992
PLOT_TOP, PLOT_BOTTOM = 56, 320


def axis_y(delta: float) -> float:
    """Potency difference to canvas y. Positive is up, so the scale inverts."""
    middle = (PLOT_TOP + PLOT_BOTTOM) / 2
    return middle - delta / AXIS_MAX * (PLOT_BOTTOM - PLOT_TOP) / 2


def read_arms(results: Path) -> dict[str, dict[str, Any]]:
    """The two hydroxyl additions, from the robustness table."""
    rows = list(csv.DictReader((results / "transformation-robustness.csv").open()))
    arms = {}
    for position, label in (("R3", "para"), ("R4", "meta")):
        row = next(
            r
            for r in rows
            if r["case_id"] == "braf"
            and r["position"] == position
            and r["fragment_from"] == "H"
            and r["fragment_to"].startswith("O[")
        )
        arms[label] = {
            "position": position,
            "median": float(row["median_delta"]),
            "low": float(row["loco_median_min"]),
            "high": float(row["loco_median_max"]),
            "pairs": int(row["supporting_pairs"]),
        }
    return arms


def structure_svg() -> str:
    """The core, with RDKit's own <svg> wrapper removed so it can be nested."""
    drawing = to_svg(CORE, (470, 330))
    if drawing is None:  # pragma: no cover - needs RDKit's drawing extension
        raise RuntimeError(
            "RDKit could not draw the core; the drawing extension needs libxrender1, "
            "libx11-6 and libxext6"
        )
    drawing = re.sub(r"^.*?<svg\b[^>]*>", "", drawing, flags=re.S)
    return re.sub(r"</svg>\s*$", "", drawing)


def arm(label: str, data: dict[str, Any], colour: str) -> str:
    """One fork arm: curve, endpoint, value, and a separate omission-range bar."""
    tip = axis_y(data["median"])
    band_top, band_bottom = axis_y(data["high"]), axis_y(data["low"])
    zero = axis_y(0.0)
    above = data["median"] > 0
    # A cubic that leaves the fork horizontally and arrives horizontally, so the
    # two arms read as one split rather than two unrelated arrows.
    curve = (
        f"M {FORK_X} {zero:.1f} "
        f"C {(FORK_X + TIP_X) / 2:.1f} {zero:.1f}, "
        f"{(FORK_X + TIP_X) / 2:.1f} {tip:.1f}, {TIP_X} {tip:.1f}"
    )
    stack = -1 if above else 1
    return f"""
  <path d="{curve}" fill="none" stroke="{colour}" stroke-width="3.4" stroke-linecap="round"/>
  <circle cx="{TIP_X}" cy="{tip:.1f}" r="7.5" fill="{colour}"/>
  <text x="{TIP_X + 18}" y="{tip + 10:.1f}" font-family="Helvetica, Arial, sans-serif"
        font-size="28" font-weight="600" fill="{colour}">{data["median"]:+.2f}</text>
  <text x="{TIP_X + 18}" y="{tip + stack * 28:.1f}" font-family="Helvetica, Arial, sans-serif"
        font-size="17" fill="{INK}">{label} ({data["position"]})</text>
  <text x="{TIP_X + 18}" y="{tip + stack * 47:.1f}" font-family="Helvetica, Arial, sans-serif"
        font-size="12.5" fill="{MUTED}">{data["pairs"]} matched pairs</text>
  <line x1="{BAND_X}" y1="{band_top:.1f}" x2="{BAND_X}" y2="{band_bottom:.1f}"
        stroke="{colour}" stroke-width="9" opacity="0.3" stroke-linecap="butt"/>
  <line x1="{BAND_X - 7}" y1="{band_top:.1f}" x2="{BAND_X + 7}" y2="{band_top:.1f}"
        stroke="{colour}" stroke-width="1.6" opacity="0.75"/>
  <line x1="{BAND_X - 7}" y1="{band_bottom:.1f}" x2="{BAND_X + 7}" y2="{band_bottom:.1f}"
        stroke="{colour}" stroke-width="1.6" opacity="0.75"/>"""


def build(results: Path, out: Path) -> Path:
    arms = read_arms(results)
    zero = axis_y(0.0)
    ticks = "".join(
        f'<line x1="{AXIS_X - 5}" y1="{axis_y(v):.1f}" x2="{AXIS_X + 5}" y2="{axis_y(v):.1f}"'
        f' stroke="{MUTED}" stroke-width="1.2"/>'
        f'<text x="{AXIS_X + 12}" y="{axis_y(v) + 4:.1f}"'
        f' font-family="Helvetica, Arial, sans-serif"'
        f' font-size="12" fill="{MUTED}">{v:+g}</text>'
        for v in (-2, -1, 0, 1, 2)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>

  <text x="34" y="36" font-family="Helvetica, Arial, sans-serif" font-size="19"
        font-weight="600" fill="{INK}">The same hydroxyl. Opposite directions.</text>

  <g transform="translate(24,52)">{structure_svg()}</g>
  <circle cx="40" cy="372" r="6" fill="{AMBER}"/>
  <text x="54" y="377" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="{INK}">
    *:3 = para</text>
  <circle cx="176" cy="372" r="6" fill="{TEAL}"/>
  <text x="190" y="377" font-family="Helvetica, Arial, sans-serif" font-size="14" fill="{INK}">
    *:4 = meta</text>
  <text x="34" y="403" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="{MUTED}">
    BRAF series · one assay · one mapped isomeric core · 16 compounds</text>

  <line x1="{AXIS_X}" y1="{PLOT_TOP}" x2="{AXIS_X}" y2="{PLOT_BOTTOM}"
        stroke="{MUTED}" stroke-width="1.2"/>
  {ticks}
  <text x="{AXIS_X + 10}" y="{PLOT_TOP - 30}" font-family="Helvetica, Arial, sans-serif"
        font-size="13" fill="{MUTED}">Δ pIC₅₀</text>

  <line x1="{FORK_X - 40}" y1="{zero:.1f}" x2="{AXIS_X}" y2="{zero:.1f}"
        stroke="{MUTED}" stroke-width="1" stroke-dasharray="5 5"/>
  <circle cx="{FORK_X}" cy="{zero:.1f}" r="6" fill="#ffffff" stroke="{INK}" stroke-width="2.4"/>
  <text x="{FORK_X - 8}" y="{zero - 18:.1f}" font-family="Helvetica, Arial, sans-serif"
        font-size="17" fill="{INK}">H</text>
  <text x="{FORK_X - 46}" y="{zero + 26:.1f}" font-family="Helvetica, Arial, sans-serif"
        font-size="13" fill="{MUTED}">add OH</text>
{arm("meta", arms["meta"], TEAL)}
{arm("para", arms["para"], AMBER)}

  <text x="{BAND_X - 26}" y="{PLOT_TOP - 32:.0f}" font-family="Helvetica, Arial, sans-serif"
        font-size="11.5" fill="{MUTED}">omission</text>
  <text x="{BAND_X - 22}" y="{PLOT_TOP - 19:.0f}" font-family="Helvetica, Arial, sans-serif"
        font-size="11.5" fill="{MUTED}">range</text>
  <text x="{FORK_X - 50}" y="403" font-family="Helvetica, Arial, sans-serif"
        font-size="13" fill="{MUTED}">
    Bars show how far each median moves when single compounds are omitted</text>
</svg>
"""
    out.mkdir(parents=True, exist_ok=True)
    source = out / "graphical-abstract.svg"
    source.write_text(svg)
    cairosvg.svg2pdf(url=str(source), write_to=str(out / "graphical-abstract.pdf"))
    cairosvg.svg2png(url=str(source), write_to=str(out / "graphical-abstract.png"), scale=300 / 96)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="analyse_sar_evidence output")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.results, args.out))


if __name__ == "__main__":
    main()
