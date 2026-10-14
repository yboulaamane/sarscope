"""The reference library, journal bibliography and citation numbering agree."""

import importlib.util
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
spec = importlib.util.spec_from_file_location("paper_references", PAPER / "format_references.py")
assert spec is not None and spec.loader is not None
formatter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formatter)


def test_springer_author_truncation_is_consistent_but_exports_retain_every_author():
    authors = [{"family": f"Family{i}", "given": "Anne B."} for i in range(6)]
    assert formatter.displayed_authors(authors[:4]) == (
        "Family0 AB, Family1 AB, Family2 AB, Family3 AB"
    )
    assert formatter.displayed_authors(authors) == "Family0 AB, Family1 AB, Family2 AB et al"
    library = json.loads((PAPER / "references.csl.json").read_text())
    item = next(row for row in library if row["DOI"] == "10.1038/sdata.2016.18")
    assert len(item["author"]) == 53
    assert formatter.ris([item]).count("AU  - ") == 53


def test_bibliography_exactly_matches_the_structured_library():
    library = json.loads((PAPER / "references.csl.json").read_text())
    bibliography = (
        (PAPER / "jcheminformatics-manuscript.md").read_text().split("## References\n")[1]
    )
    assert bibliography.strip() == "\n".join(formatter.reference(item) for item in library)
    assert [item["id"] for item in library] == [f"ref{i}" for i in range(1, 33)]
    assert len({item["DOI"] for item in library if item.get("DOI")}) == 29
    assert not any("local and awaits" in item["title"] for item in library)
    assert not any(item.get("DOI") == "10.1186/s13321-026-01181-x" for item in library)


def test_citations_are_numbered_in_first_appearance_order():
    content = (PAPER / "jcheminformatics-manuscript.md").read_text().split("## References\n")[0]
    seen = []
    for group in re.findall(r"\[([\d, –-]+)\]", content):
        for part in group.split(","):
            if "–" in part or "-" in part:
                left, right = re.split("[–-]", part)
                values = range(int(left), int(right) + 1)
            else:
                values = [int(part)]
            for value in values:
                if value not in seen:
                    seen.append(value)
    assert seen == list(range(1, 33))


def test_page_locators_and_export_records_remain_complete():
    library = json.loads((PAPER / "references.csl.json").read_text())
    article = next(row for row in library if row.get("DOI") == "10.1186/s13321-026-01156-y")
    assert formatter.locator(article) == "50"
    one_page = next(row for row in library if row.get("DOI") == "10.1021/ci060117s")
    assert formatter.locator(one_page) == "1535"
    bib = (PAPER / "reference-exports/references.bib").read_text()
    ris = (PAPER / "reference-exports/references.ris").read_text()
    assert bib == formatter.bibtex(library)
    assert ris == formatter.ris(library)
    assert ris.count("TY  - ") == ris.count("ER  -") == 32
