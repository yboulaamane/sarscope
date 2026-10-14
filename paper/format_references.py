"""Generate a consistent Springer Basic numeric bibliography and import files.

The CSL-JSON library is the reference source of truth. Full author lists are
retained in the library and exports; display follows et-al-min=5/use-first=3,
as in Springer Basic (numeric, brackets). This does not create live Zotero fields.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


def initials(given: str) -> str:
    return "".join(part[0] for part in re.findall(r"[^\W\d_]+", given, re.UNICODE))


def displayed_authors(authors: list[dict[str, str]]) -> str:
    shown = authors[:3] if len(authors) >= 5 else authors
    names = [
        a.get("literal") or f"{a['family']} {initials(a.get('given', ''))}".strip() for a in shown
    ]
    return ", ".join(names) + (" et al" if len(authors) >= 5 else "")


def locator(item: dict[str, Any]) -> str:
    pages = str(item.get("page") or item.get("article-number") or "")
    if not pages:
        raise ValueError(f"No page/article locator for {item['id']}")
    parts = pages.split("-")
    return parts[0] if len(parts) == 2 and parts[0] == parts[1] else pages.replace("-", "–")


def reference(item: dict[str, Any]) -> str:
    number = int(item["id"].removeprefix("ref"))
    authors = displayed_authors(item.get("author", []))
    if item["type"] == "article-journal":
        year = item["issued"]["date-parts"][0][0]
        return (
            f"{number}. {authors} ({year}) {item['title']}. "
            f"{item['container-title-short']} {item['volume']}:{locator(item)}. "
            f"https://doi.org/{item['DOI']}"
        )
    accessed_date = date(*item["accessed"]["date-parts"][0])
    accessed = f"{accessed_date.day} {accessed_date.strftime('%B %Y')}"
    return (
        f"{number}. {authors + '. ' if authors else ''}{item['title']}. "
        f"{item['URL']}. Accessed {accessed}"
    )


def bibtex(items: list[dict[str, Any]]) -> str:
    blocks = []
    for item in items:
        fields = {"title": "{" + item["title"] + "}"}
        if item.get("author"):
            fields["author"] = " and ".join(
                "{" + a["literal"] + "}"
                if "literal" in a
                else f"{a['family']}, {a.get('given', '')}".rstrip(", ")
                for a in item["author"]
            )
        if item["type"] == "article-journal":
            fields.update(
                journal=item["container-title"],
                year=str(item["issued"]["date-parts"][0][0]),
                volume=item["volume"],
                pages=locator(item).replace("–", "--"),
                DOI=item["DOI"],
            )
        fields["url"] = item["URL"]
        if "accessed" in item:
            fields["urldate"] = date(*item["accessed"]["date-parts"][0]).isoformat()
        kind = "article" if item["type"] == "article-journal" else "misc"
        # Braces preserve chemical identifiers, acronym case and author accents.
        content = [f"  {name} = {{{value}}}" for name, value in fields.items()]
        blocks.append(f"@{kind}{{{item['id']},\n" + ",\n".join(content) + "\n}")
    return "\n\n".join(blocks) + "\n"


def ris(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        lines += [
            f"TY  - {'JOUR' if item['type'] == 'article-journal' else 'ELEC'}",
            f"ID  - {item['id']}",
            f"TI  - {item['title']}",
        ]
        for author in item.get("author", []):
            lines.append(
                "AU  - "
                + (
                    author["literal"]
                    if "literal" in author
                    else f"{author['family']}, {author.get('given', '')}".rstrip(", ")
                )
            )
        if item["type"] == "article-journal":
            pages = locator(item).split("–")
            lines += [
                f"PY  - {item['issued']['date-parts'][0][0]}",
                f"JO  - {item['container-title']}",
                f"J2  - {item['container-title-short']}",
                f"VL  - {item['volume']}",
                f"SP  - {pages[0]}",
                f"DO  - {item['DOI']}",
            ]
            if len(pages) == 2:
                lines.append(f"EP  - {pages[1]}")
        if "accessed" in item:
            lines.append("Y2  - " + date(*item["accessed"]["date-parts"][0]).isoformat())
        lines += [f"UR  - {item['URL']}", "ER  -", ""]
    return "\n".join(lines)


def export(library: Path, out: Path) -> Path:
    if out.exists():
        raise ValueError("Use a new reference-export directory")
    items = json.loads(library.read_text())
    assert [item["id"] for item in items] == [f"ref{i}" for i in range(1, len(items) + 1)]
    dois = [item["DOI"] for item in items if item.get("DOI")]
    assert len(dois) == len(set(dois)), "Duplicate DOI in bibliography"
    out.mkdir(parents=True)
    (out / "bibliography.txt").write_text("\n".join(reference(item) for item in items) + "\n")
    (out / "references.bib").write_text(bibtex(items))
    (out / "references.ris").write_text(ris(items))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.library, args.out))
