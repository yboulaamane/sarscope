"""Bounded, on-demand disease discovery through the Open Targets GraphQL API.

Disease associations guide target selection only; scores are not probabilities
of therapeutic success and do not enter QSAR labels or potency measurements.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

SEARCH_QUERY = """
query DiseaseSearch($query: String!, $size: Int!) {
  search(queryString: $query, entityNames: ["disease"], page: {index: 0, size: $size}) {
    total hits { id name entity }
  }
}
"""

ASSOCIATIONS_QUERY = """
query DiseaseTargets($id: String!, $size: Int!) {
  disease(efoId: $id) {
    id name
    associatedTargets(page: {index: 0, size: $size}, enableIndirect: false) {
      count rows {
        score
        target { id approvedSymbol approvedName proteinIds { id source } }
      }
    }
  }
}
"""


class OpenTargetsError(RuntimeError):
    """Disease discovery failed; ChEMBL ID/name lookup remains independent."""


def _query(
    query: str, variables: dict[str, Any], transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=20, transport=transport) as client:
            response = client.post(API_URL, json={"query": query, "variables": variables})
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or payload.get("errors"):
            raise ValueError("GraphQL errors")
        data = payload["data"]
        if not isinstance(data, dict):
            raise ValueError("invalid data")
        return data
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise OpenTargetsError(
            "Open Targets is unavailable or returned an invalid response. "
            "Retry, or search directly by gene/protein or ChEMBL ID."
        ) from exc


def search_diseases(
    query: str, *, limit: int = 10, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Return candidate ontology terms, not an automatically chosen disease."""
    query = query.strip()
    if not query or len(query) > 200 or not 1 <= limit <= 25:
        raise ValueError("Enter a disease name (1–200 characters); result limit must be 1–25.")
    data = _query(SEARCH_QUERY, {"query": query, "size": limit}, transport)
    try:
        search = data["search"]
        hits = search["hits"]
        if not isinstance(hits, list) or any(
            not isinstance(hit, dict)
            or not hit.get("id")
            or not hit.get("name")
            or hit.get("entity") != "disease"
            for hit in hits
        ):
            raise ValueError("invalid disease hits")
        return {"hits": hits, "total": int(search["total"])}
    except (ValueError, KeyError, TypeError) as exc:
        raise OpenTargetsError("Open Targets returned an invalid disease search result.") from exc


def disease_targets(
    disease_id: str, *, limit: int = 25, transport: httpx.BaseTransport | None = None
) -> dict[str, Any]:
    """Top direct associations for the selected term, with the full count and timestamp."""
    if not disease_id.strip() or not 1 <= limit <= 50:
        raise ValueError("Choose a disease ID; association limit must be 1–50.")
    data = _query(ASSOCIATIONS_QUERY, {"id": disease_id, "size": limit}, transport)
    try:
        disease = data["disease"]
        if disease is None:
            raise OpenTargetsError("This disease ID is no longer available. Search again.")
        associations = disease["associatedTargets"]
        rows = []
        for row in associations["rows"]:
            target = row["target"]
            rows.append(
                {
                    "gene_id": target["id"],
                    "symbol": target["approvedSymbol"],
                    "name": target["approvedName"],
                    "score": float(row["score"]),
                    "accessions": sorted(
                        {
                            protein["id"]
                            for protein in target.get("proteinIds") or []
                            if protein.get("source") == "uniprot_swissprot"
                        }
                    ),
                }
            )
        return {
            "disease_id": disease["id"],
            "disease_name": disease["name"],
            "total": int(associations["count"]),
            "rows": rows,
            "source": "Open Targets Platform",
            "endpoint": API_URL,
            "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "enable_indirect": False,
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise OpenTargetsError("Open Targets returned invalid disease associations.") from exc
