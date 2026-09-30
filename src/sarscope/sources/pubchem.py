"""Bounded PubChem BioAssay qualitative-outcome retrieval.

This client never interprets activity values as concentrations. AID is the
experimental unit; combining assay calls across AIDs requires a separate,
explicitly designed endpoint and is deliberately unsupported here.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from typing import Any

import httpx

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
MAX_CONCISE_ROWS = 10_000
CID_BATCH = 100


class PubChemError(RuntimeError):
    """PubChem could not provide a complete, usable assay snapshot."""


class PubChemClient:
    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 90.0,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)
        self._retries = retries
        self._sleep = sleep

    def __enter__(self) -> PubChemClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._http.close()

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._base}/{path}"
        for attempt in range(self._retries + 1):
            try:
                response = self._http.get(url)
                if response.status_code == 404:
                    raise PubChemError(f"PubChem resource not found: {path}")
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        "retryable PubChem response", request=response.request, response=response
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise PubChemError(f"invalid PubChem response: {path}")
                return payload
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                if attempt == self._retries:
                    raise PubChemError(f"PubChem request failed for {path}: {exc}") from exc
                self._sleep(2**attempt)
        raise AssertionError("unreachable")  # pragma: no cover

    def concise_assay(self, aid: int) -> tuple[list[dict[str, str]], dict[str, str]]:
        """Return all concise rows for a small AID, or refuse possible truncation."""
        if aid < 1:
            raise ValueError("AID must be a positive integer")
        payload = self._get(f"assay/aid/{aid}/concise/JSON")
        table = payload.get("Table", {})
        columns = table.get("Columns", {}).get("Column", [])
        raw_rows = table.get("Row", [])
        required = {"AID", "CID", "Activity Outcome", "Assay Name", "Assay Type"}
        if not isinstance(columns, list) or not required.issubset(columns):
            raise PubChemError("PubChem concise assay schema is missing required columns")
        if not isinstance(raw_rows, list) or len(raw_rows) >= MAX_CONCISE_ROWS:
            raise PubChemError(
                "assay reaches the 10,000-row concise retrieval safety limit; "
                "use a complete PubChem assay export rather than a truncated benchmark"
            )
        rows: list[dict[str, str]] = []
        for item in raw_rows:
            cells = item.get("Cell", [])
            if len(cells) != len(columns):
                raise PubChemError("PubChem concise assay row width does not match schema")
            row = dict(zip(columns, (str(x) if x is not None else "" for x in cells), strict=True))
            if row["AID"] != str(aid):
                raise PubChemError(f"assay {aid} response contains a different AID")
            rows.append(row)
        for field in ("Assay Name", "Assay Type", "Target Accession"):
            if field in columns and len({row[field] for row in rows}) > 1:
                raise PubChemError(f"assay {aid} has inconsistent {field} values")
        meta = {
            "aid": str(aid),
            "assay_name": rows[0]["Assay Name"] if rows else "",
            "assay_type": rows[0]["Assay Type"] if rows else "",
            "target_accession": rows[0].get("Target Accession", "") if rows else "",
            "source_url": f"{self._base}/assay/aid/{aid}/concise/JSON",
        }
        return rows, meta

    def smiles_for_cids(self, cids: Iterable[int]) -> dict[int, str]:
        ids = sorted(set(cids))
        if any(cid < 1 for cid in ids):
            raise ValueError("CID must be a positive integer")
        result: dict[int, str] = {}
        for start in range(0, len(ids), CID_BATCH):
            batch = ids[start : start + CID_BATCH]
            payload = self._get(f"compound/cid/{','.join(map(str, batch))}/property/SMILES/JSON")
            properties = payload.get("PropertyTable", {}).get("Properties", [])
            if not isinstance(properties, list):
                raise PubChemError("PubChem compound property schema changed")
            for item in properties:
                cid, smiles = item.get("CID"), item.get("SMILES")
                if isinstance(cid, int) and cid in batch and isinstance(smiles, str) and smiles:
                    result[cid] = smiles
        return result


def parse_cid(value: str) -> int | None:
    return int(value) if re.fullmatch(r"[1-9][0-9]*", value.strip()) else None
