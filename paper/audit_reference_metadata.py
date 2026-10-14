"""Check DOI/title metadata against Crossref; failures remain visible, not guessed.

Bibliographic metadata checks do not certify that a paper supports a claim or
replace reading the original article and supplementary tables.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx


def normalise(title: str) -> str:
    text = re.sub(r"<[^>]+>", "", html.unescape(title)).lower()
    text = text.replace("β", "beta").replace("α", "alpha")
    return re.sub(r"\W+", "", text)


def audit(source: Path, out: Path) -> Path:
    if out.exists():
        raise ValueError("Use a new metadata audit output file")
    references = source.read_text().split("## References\n")[1]
    rows = []
    started = time.monotonic()
    with httpx.Client(timeout=8, follow_redirects=True) as client:
        for line in references.splitlines():
            if not re.match(r"^\d+\. ", line):
                continue
            number = int(line.split(".", 1)[0])
            dois = re.findall(r"https://doi\.org/(\S+)", line)
            if not dois:
                continue
            title_and_journal = re.split(r"\(\d{4}\) ", line, maxsplit=1)[1]
            title = re.split(
                r"\. (?=J Med Chem |J Chem Inf Model |Nucleic Acids Res |"
                r"J Comput Aided Mol Des |J Cheminform |PLoS One |Sci Data |J R Stat Soc B )",
                title_and_journal,
                maxsplit=1,
            )[0]
            for index, raw_doi in enumerate(dois):
                doi = raw_doi.rstrip(".;")
                row = {"reference": number, "doi": doi}
                if time.monotonic() - started > 50:
                    rows.append({**row, "status": "not checked: bounded request time exceeded"})
                    continue
                try:
                    response = client.get("https://api.crossref.org/works/" + doi)
                    response.raise_for_status()
                    metadata = response.json()["message"]
                    retrieved_title = metadata["title"][0]
                    # A copyright correction appended to an entry has its own
                    # title; do not compare it against the original article title.
                    score = (
                        None
                        if index
                        else SequenceMatcher(
                            None, normalise(title), normalise(retrieved_title)
                        ).ratio()
                    )
                    if doi.casefold() != metadata["DOI"].casefold():
                        raise ValueError("Retrieved DOI does not match requested DOI")
                    rows.append(
                        {
                            **row,
                            "status": "retrieved",
                            "title": retrieved_title,
                            "title_similarity": score,
                            "authors": metadata.get("author"),
                            "journal": metadata.get("container-title"),
                            "volume": metadata.get("volume"),
                            "page": metadata.get("page"),
                            "article_number": metadata.get("article-number"),
                            "published": metadata.get("published"),
                            "published_print": metadata.get("published-print"),
                        }
                    )
                except (httpx.HTTPError, ValueError, KeyError) as error:
                    rows.append(
                        {**row, "status": "unverified in this API check", "error": str(error)}
                    )
                time.sleep(0.3)
    payload = {
        "scope": "Bibliographic DOI/title metadata, not claim support or scientific validation",
        "source": str(source.name),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checked_on": "2026-10-13",
        "results": rows,
        "retrieved": sum(row["status"] == "retrieved" for row in rows),
        "total_dois": len(rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.source, args.out))
