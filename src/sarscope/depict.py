"""Drawing molecules, scaffolds and R-group cores.

Two output formats, for two destinations:

  SVG  for the Streamlit app and anywhere text should stay selectable and
       scale cleanly.
  PNG  for the HTML report, which inlines its images as base64 so the file
       stays self-contained; a base64 SVG would work too, but PNG keeps the
       report's images consistent with its matplotlib figures.

**Attachment points are drawn, not stripped.** An R-group core comes back from
RDKit with ``[*:1]`` dummy atoms marking where substituents attach, and a
substituent like ``F[*:1]`` carries the matching label. Those labels are the
whole point of an R-group table, so they are kept and RDKit draws them as
numbered attachment points.

**Every function returns None rather than raising for an unparseable input.**
A depiction is decoration: a scaffold that will not parse should leave a gap in
a table, not take down the page that table is on.

**That extends to the drawing library itself.** ``rdkit.Chem.Draw.rdMolDraw2D``
is a compiled extension linked against the system X11 libraries (libXrender,
libX11, libXext). Those are not bundled in the RDKit wheel, so on a minimal
container - a slim Docker image, or Streamlit Community Cloud without a
``packages.txt`` - importing it raises ImportError and, imported at module
level, takes down the whole application. It is therefore imported defensively:
``available()`` reports whether drawing works, every function returns None when
it does not, and callers already handle None by falling back to the SMILES
string. ``packages.txt`` in the repository root installs those libraries on
Streamlit Cloud.
"""

from __future__ import annotations

import base64
from typing import Any

from rdkit import Chem

try:
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError as exc:  # pragma: no cover - depends on the host's system libraries
    rdMolDraw2D = None
    _IMPORT_ERROR: str | None = str(exc)
else:
    _IMPORT_ERROR = None

#: Default panel size. Wide enough for a kinase inhibitor at a readable scale.
DEFAULT_SIZE = (320, 220)


def available() -> bool:
    """Whether structures can be drawn on this host."""
    return rdMolDraw2D is not None


def unavailable_reason() -> str | None:
    """Why drawing is unavailable, or None when it works.

    Worth surfacing rather than silently showing text: the cause is a missing
    system package, which the person running the app can fix.
    """
    if rdMolDraw2D is not None:
        return None
    return (
        f"RDKit's drawing extension could not be imported ({_IMPORT_ERROR}). "
        "It needs the system libraries libxrender1, libx11-6 and libxext6; "
        "on Streamlit Cloud these come from packages.txt."
    )


def _prepare(smiles: str) -> Any | None:
    if rdMolDraw2D is None or not smiles or not isinstance(smiles, str):
        return None
    # sanitize=True is what PrepareAndDrawMolecule expects; dummy atoms survive it.
    return Chem.MolFromSmiles(smiles)


def _draw(drawer: Any, mol: Any, highlight: str | None) -> None:
    match = []
    if highlight:
        pattern = Chem.MolFromSmarts(highlight) or Chem.MolFromSmiles(highlight)
        if pattern is not None:
            match = list(mol.GetSubstructMatch(pattern))
    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=match)
    drawer.FinishDrawing()


def to_svg(
    smiles: str,
    size: tuple[int, int] = DEFAULT_SIZE,
    *,
    highlight: str | None = None,
) -> str | None:
    """Structure as an SVG string, or None if the SMILES will not parse.

    ``highlight`` is a substructure (SMARTS or SMILES) to shade - a scaffold
    inside a whole molecule, for instance.
    """
    mol = _prepare(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    _draw(drawer, mol, highlight)
    return str(drawer.GetDrawingText())


def to_png(
    smiles: str,
    size: tuple[int, int] = DEFAULT_SIZE,
    *,
    highlight: str | None = None,
) -> bytes | None:
    """Structure as PNG bytes, or None if the SMILES will not parse."""
    mol = _prepare(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    _draw(drawer, mol, highlight)
    return bytes(drawer.GetDrawingText())


def to_data_uri(
    smiles: str,
    size: tuple[int, int] = DEFAULT_SIZE,
    *,
    highlight: str | None = None,
) -> str | None:
    """PNG as a ``data:`` URI, for inlining into a self-contained HTML report."""
    png = to_png(smiles, size, highlight=highlight)
    if png is None:
        return None
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
