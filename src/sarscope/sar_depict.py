"""Annotated, selectable scaffold SVGs; all chemistry is drawn by RDKit."""

from __future__ import annotations

import html
import re
import textwrap

from rdkit import Chem

from sarscope.analysis.sar_map import SarMap
from sarscope.depict import rdMolDraw2D


def sar_map_svg(result: SarMap, *, selected: str | None = None) -> str | None:
    """Export a self-contained SVG with attachment-site targets and evidence cards."""
    if rdMolDraw2D is None:
        return None
    molecule = Chem.MolFromSmiles(result.series.core)
    if molecule is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(660, 440)
    options = drawer.drawOptions()
    options.padding = 0.12
    positions = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum():
            name = f"R{atom.GetAtomMapNum()}"
            options.atomLabels[atom.GetIdx()] = name
            positions[name] = atom.GetIdx()
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    coords = {name: drawer.GetDrawCoords(index) for name, index in positions.items()}
    drawer.FinishDrawing()
    drawing = drawer.GetDrawingText()
    drawing = re.sub(r"^.*?<svg\b[^>]*>", "", drawing, flags=re.S)
    drawing = re.sub(r"</svg>\s*$", "", drawing)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        'viewBox="0 0 1100 670" width="1100" height="670" role="img" '
        'aria-label="Scaffold SAR map with selectable substitution positions">',
        '<rect width="1100" height="670" rx="16" fill="#fafafc"/>',
        '<text x="26" y="36" font-family="sans-serif" font-size="22" fill="#182235">'
        "Series SAR · click a position to inspect its evidence</text>",
        '<text x="26" y="62" font-family="sans-serif" font-size="13" fill="#526072">'
        "Green: supported increase · Red: supported decrease · Amber: mixed · "
        "Grey: sparse</text>",
        f'<g transform="translate(220,96)">{drawing}</g>',
    ]
    notes = result.annotations.to_dict("records")
    visible = [note for note in notes if note["position"] in coords]
    side_notes = {
        left: sorted(
            [note for note in visible if (coords[note["position"]].x < 330) == left],
            key=lambda note: coords[note["position"]].y,
        )
        for left in (True, False)
    }
    slots = {}
    for left, cards in side_notes.items():
        for index, note in enumerate(cards[:5]):
            slots[note["position"]] = (
                left,
                96 + max(0, (440 - len(cards[:5]) * 105) / 2) + index * 105,
            )
    for note in visible:
        name = str(note["position"])
        if name not in coords:
            continue
        point = coords[name]
        px, py = point.x + 220, point.y + 96
        if name not in slots:
            # Every position remains selectable, even when side cards are full.
            parts.append(
                f'<circle data-position="{html.escape(name)}" role="button" tabindex="0" '
                f'aria-label="Inspect {html.escape(name)}" cx="{px:.1f}" cy="{py:.1f}" '
                'r="23" fill="#9ca3af" fill-opacity="0.12" stroke="#9ca3af" '
                'style="cursor:pointer"/>'
            )
            continue
        left, y = slots[name]
        x = 14 if left else 886
        edge = x + 200 if left else x
        color = str(note["color"])
        title = html.escape(name + ": " + str(note["label"]) + "; " + str(note["detail"]))
        parts.append(
            f'<g data-position="{html.escape(name)}" role="button" tabindex="0" '
            f'aria-label="Inspect {html.escape(name)}" style="cursor:pointer">'
            f"<title>{title}</title>"
            f'<path d="M {edge} {y + 45} L {px:.1f} {py:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="1.5" opacity="0.55"/>'
            f'<rect x="{x}" y="{y}" width="200" height="94" rx="10" fill="white" '
            f'stroke="{color}" stroke-width="{3 if selected == name else 1.5}"/>'
            f'<text x="{x + 12}" y="{y + 22}" font-family="sans-serif" font-size="16" '
            f'font-weight="700" fill="{color}">{html.escape(name)}</text>'
        )
        lines = textwrap.wrap(str(note["label"]), width=25)[:1]
        for line_index, line in enumerate(lines):
            parts.append(
                f'<text x="{x + 12}" y="{y + 43 + line_index * 15}" font-family="sans-serif" '
                f'font-size="12" fill="#182235">{html.escape(line)}</text>'
            )
        endpoint = note.get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            parts.append(
                f'<text x="{x + 12}" y="{y + 65}" font-family="sans-serif" font-size="10" '
                f'fill="#526072">{html.escape(endpoint)} · same-assay evidence</text>'
            )
        parts.append(
            f'<text x="{x + 12}" y="{y + 81}" font-family="sans-serif" font-size="10" '
            f'fill="#526072">{html.escape(str(note["detail"]))}</text>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="23" fill="{color}" fill-opacity="0.12" '
            f'stroke="{color}" stroke-width="{3 if selected == name else 1.5}"/>'
            "</g>"
        )
    parts.append(
        f'<text x="26" y="627" font-family="sans-serif" font-size="13" fill="#182235">'
        f"{result.series.n_molecules} compounds · "
        f"{len(result.series.positions)} variable positions · "
        f"{len(result.pairs)} evidence rows</text>"
        '<text x="26" y="651" font-family="sans-serif" font-size="12" fill="#526072">'
        "Comparisons hold other labelled positions constant. Invariant core atoms have no "
        "measured SAR here.</text></svg>"
    )
    return "".join(parts)
