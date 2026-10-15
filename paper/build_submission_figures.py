"""Convert the manuscript figures to the formats a journal asks for.

The committed sources are SVG. Journals want vector PDF, and a 300 dpi raster
as a fallback. This script records which source is which numbered figure, which
is otherwise knowledge that exists only in whoever assembled the submission.

    python paper/build_submission_figures.py --out ~/sarscope-submission/figures
"""

from __future__ import annotations

import argparse
import pathlib

import cairosvg

#: Figure number -> source, in the order the manuscript introduces them.
FIGURES = {
    "Figure1": "figures/sar-stress-workflow.svg",
    "Figure2": "results/braf/sar-map.svg",
    "Figure3": "scientific-results/position-effects.svg",
    "Figure4": "scientific-results/drd2-chlorine-structures.svg",
    "FigureS1": "benchmark-audit/benchmark.svg",
    "graphical-abstract": "figures/graphical-abstract.svg",
}

#: CSS pixels are 96 per inch, so this scale gives a true 300 dpi raster.
DPI_SCALE = 300 / 96


def build(paper: pathlib.Path, out: pathlib.Path) -> list[pathlib.Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, relative in FIGURES.items():
        source = paper / relative
        if not source.is_file():
            raise FileNotFoundError(f"{name}: {source} is missing")
        cairosvg.svg2pdf(url=str(source), write_to=str(out / f"{name}.pdf"))
        cairosvg.svg2png(url=str(source), write_to=str(out / f"{name}.png"), scale=DPI_SCALE)
        written += [out / f"{name}.pdf", out / f"{name}.png"]
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=pathlib.Path, default=pathlib.Path(__file__).parent)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    for path in build(args.paper, args.out):
        print(path)


if __name__ == "__main__":
    main()
