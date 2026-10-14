"""Validate the local journal draft without pretending to validate its science.

Optional document dependencies: python-docx and PyMuPDF. This checks references,
editable tables, packaged figures, source hashes and independent PDF rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
from docx import Document


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            row = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", cell) for cell in row):
                current.append(row)
        elif current:
            tables.append(current)
            current = []
    return tables


def check_document(source: Path, word: Path, pdf: Path) -> dict[str, Any]:
    text = source.read_text()
    with zipfile.ZipFile(word) as archive:
        assert archive.testzip() is None, "Word ZIP integrity failure"
        vectors = [name for name in archive.namelist() if name.endswith(".svg")]
    document = Document(word)
    word_tables = [
        [[cell.text for cell in row.cells] for row in table.rows] for table in document.tables
    ]
    assert word_tables == markdown_tables(text), "Editable Word table differs from Markdown"
    expected_figures = len(re.findall(r"^Figure (?:S?\d+):", text, re.M))
    assert len(vectors) == expected_figures, "Missing embedded vector figure"
    with fitz.open(pdf) as rendered:
        assert rendered.page_count > 0
        overflows = []
        for number, page in enumerate(rendered, start=1):
            for block in page.get_text("blocks"):
                if block[4].strip() and (
                    block[0] < -1
                    or block[1] < -1
                    or block[2] > page.rect.width + 1
                    or block[3] > page.rect.height + 1
                ):
                    overflows.append({"page": number, "bounds": list(block[:4])})
        assert not overflows, f"PDF text outside page bounds: {overflows}"
        rendered_text = "\n".join(page.get_text() for page in rendered)
        assert "SARscope" in rendered_text
        result = dict(
            pages=rendered.page_count,
            tables=len(word_tables),
            figures=len(vectors),
            pdf_page_bounds="passed",
            rendered_text_characters=len(rendered_text),
        )
    return {
        **result,
        "source_sha256": digest(source),
        "word_sha256": digest(word),
        "pdf_sha256": digest(pdf),
    }


def verify(paper: Path, out: Path) -> Path:
    if out.exists():
        raise ValueError("Use a new verification output file")
    main = (paper / "jcheminformatics-manuscript.md").read_text()
    content, references = main.split("## References\n", 1)
    defined = [int(value) for value in re.findall(r"^(\d+)\. ", references, re.M)]
    library = json.loads((paper / "references.csl.json").read_text())
    assert defined == list(range(1, len(library) + 1))
    from format_references import reference

    assert references.strip() == "\n".join(reference(item) for item in library)
    cited = set()
    for citation in re.findall(r"\[([\d, –-]+)\]", content):
        for part in citation.split(","):
            if "–" in part or "-" in part:
                first, last = re.split("[–-]", part)
                cited.update(range(int(first), int(last) + 1))
            else:
                cited.add(int(part))
    assert cited == set(defined), "Uncited or undefined bibliography entries"
    main_word = Document(paper / "SARscope_JCheminformatics_manuscript.docx")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    from lxml import etree

    xml = etree.fromstring(main_word.element.xml.encode())
    bookmarks = xml.xpath("//w:bookmarkStart/@w:name", namespaces=ns)
    assert set(bookmarks) == {item["id"] for item in library}
    targets = xml.xpath("//w:hyperlink/@w:anchor", namespaces=ns)
    assert targets and set(targets) <= set(bookmarks), "Broken Word citation link"
    abstract = content.split("## Abstract\n")[1].split("## Introduction\n")[0]
    assert len(abstract.split()) <= 350, "Conservative abstract-length target exceeded"
    contribution = abstract.split("### Scientific Contribution\n")[1].split("Keywords:")[0]
    assert len(re.findall(r"[.!?](?:\s|$)", contribution)) <= 3
    assert not re.search(r"\[\d", abstract), "Abstract contains numbered citations"
    assert content.count("\n## Results\n") == 1
    assert "Abdelmalek Essaadi University, Tetouan 93000, Morocco" in content
    legends = re.findall(r"^Figure (\d+): `([^`]+)`", content, re.M)
    assert [int(number) for number, _ in legends] == [1, 2, 3, 4]
    for _, path in legends:
        assert (paper / path).is_file()
    provenance = json.loads((paper / "workbench-audit/provenance.json").read_text())
    for path, sha in provenance["source_sha256"].items():
        assert digest(paper.parent / path) == sha, f"Source changed since audit: {path}"
    for path, sha in provenance["input_sha256"].items():
        assert digest(paper / path) == sha, f"Input changed since audit: {path}"
    for path, sha in provenance["output_sha256"].items():
        assert digest(paper / "workbench-audit" / path) == sha
    cases = pd.read_csv(paper / "workbench-audit/case-audit.csv")
    claims = pd.read_csv(paper / "workbench-audit/claim-audit.csv")
    assert cases.pairs.sum() == 263 and cases.selected_compounds.sum() == 92
    assert (
        cases.exhaustive_reference_agreement.all() and cases.scientific_manuscript_agreement.all()
    )
    assert len(claims) == 17 and claims.compound_sign_retained.sum() == 17
    assert claims.compound_directional_criterion_retained.sum() == 15
    assert claims.empty_publication_omissions.sum() == claims.empty_assay_omissions.sum() == 17
    result = {
        "scope": "Local manuscript/package QA; not independent scientific validation",
        "reference_entries": len(defined),
        "citation_integrity": "passed",
        "abstract_words": len(abstract.split()),
        "introduction_words": len(
            content.split("## Introduction\n")[1].split("## Implementation\n")[0].split()
        ),
        "workbench_hashes": "passed",
        "documents": {},
    }
    for label, source, stem in [
        ("main", "jcheminformatics-manuscript.md", "SARscope_JCheminformatics_manuscript"),
        ("supplement", "jcheminformatics-supplement.md", "SARscope_JCheminformatics_supplement"),
    ]:
        result["documents"][label] = check_document(
            paper / source,
            paper / f"{stem}.docx",
            paper / f"{stem}.pdf",
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.paper.resolve(), args.out.resolve()))
