"""Render references.csl.json in the Wiley-VCH numeric style.

Molecular Informatics follows the Wiley-VCH chemistry house style, which differs
from the Springer style used for the Journal of Cheminformatics draft:

  Springer     Griffen E, Leach AG, Robb GR, Warner DJ (2011) Matched Molecular
               Pairs as a Medicinal Chemistry Tool. J Med Chem 54:7739-7750
  Wiley-VCH    E. Griffen, A. G. Leach, G. R. Robb, D. J. Warner, *J. Med. Chem.*
               **2011**, *54*, 7739-7750.

So: initials before the surname, no year in parentheses, article titles omitted,
the abbreviated journal in italics, the year in bold, the volume in italics.

Generated from the same structured library the Springer bibliography uses, so the
two cannot drift apart. Emphasis is written as Markdown and converted by
build_document.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: Journal abbreviations in CASSI/ISO 4 form, as the Wiley-VCH style requires.
#: Explicit rather than derived: algorithmic abbreviation gets single-letter words
#: and compound adjectives wrong ("J Med. Chem.", "Comput. Aided."). Only the
#: journals actually cited need an entry, and a missing one raises rather than
#: silently emitting an unabbreviated title.
JOURNAL_ABBREVIATIONS = {
    "Journal of Chemical Information and Modeling": "J. Chem. Inf. Model.",
    "Journal of Cheminformatics": "J. Cheminform.",
    "Journal of Computer-Aided Molecular Design": "J. Comput.-Aided Mol. Des.",
    "Journal of Medicinal Chemistry": "J. Med. Chem.",
    "Journal of the Royal Statistical Society Series B: Statistical Methodology": (
        "J. R. Stat. Soc. B"
    ),
    "Nucleic Acids Research": "Nucleic Acids Res.",
    "PLoS ONE": "PLoS One",
    "Scientific Data": "Sci. Data",
}


def initials(given: str) -> str:
    """ "Andrew G." -> "A. G."; handles hyphenated and already-initialled names."""
    parts = [p for p in given.replace(".", " ").split() if p]
    out = []
    for part in parts:
        if "-" in part:
            out.append("-".join(f"{piece[0]}." for piece in part.split("-") if piece))
        else:
            out.append(f"{part[0]}.")
    return " ".join(out)


def authors(entry: dict[str, Any]) -> str:
    people = entry.get("author") or []
    names = [
        f"{initials(p['given'])} {p['family']}" if p.get("given") else p["family"] for p in people
    ]
    return ", ".join(names)


def abbreviate(entry: dict[str, Any]) -> str:
    title = entry.get("container-title", "")
    try:
        return JOURNAL_ABBREVIATIONS[title]
    except KeyError:
        raise KeyError(
            f"no Wiley-VCH abbreviation for {title!r}; add it to JOURNAL_ABBREVIATIONS"
        ) from None


def year(entry: dict[str, Any], key: str = "issued") -> str:
    parts = entry.get(key, {}).get("date-parts") or [[]]
    return str(parts[0][0]) if parts[0] else ""


MONTHS = (
    "January February March April May June July August September October November December"
).split()


def render(entry: dict[str, Any]) -> str:
    if entry["type"] == "webpage":
        who = authors(entry)
        accessed = entry.get("accessed", {}).get("date-parts", [[]])[0]
        when = ""
        if len(accessed) == 3:
            when = f", accessed {accessed[2]} {MONTHS[accessed[1] - 1]} {accessed[0]}"
        lead = f"{who}, " if who else ""
        return f"{lead}{entry['title']}, {entry['URL']}{when}."

    pages = entry.get("page", "").replace("-", "–")
    volume = entry.get("volume", "")
    tail = [f"**{year(entry)}**"]
    if volume:
        tail.append(f"*{volume}*")
    if pages:
        tail.append(pages)
    return f"{authors(entry)}, *{abbreviate(entry)}* " + ", ".join(tail) + "."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library", type=Path, default=Path(__file__).parent / "references.csl.json"
    )
    args = parser.parse_args()
    entries = json.loads(args.library.read_text())
    for number, entry in enumerate(entries, start=1):
        print(f"[{number}] {render(entry)}")


if __name__ == "__main__":
    main()
