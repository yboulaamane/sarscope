"""Bounded RCSB discovery and coordinate downloads, used only on explicit buttons."""

from __future__ import annotations

import re
from typing import Any

import httpx

from sarscope.docking import DockingError

MAX_STRUCTURE_BYTES = 20 * 1024 * 1024
METADATA_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id struct { title } exptl { method }
    rcsb_entry_info { resolution_combined }
    polymer_entities {
      entity_poly { type rcsb_mutation_count }
      rcsb_polymer_entity { pdbx_description pdbx_mutation }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids uniprot_ids }
    }
    nonpolymer_entities { pdbx_entity_nonpoly { comp_id name } }
  }
}
"""


def normalise_pdb_id(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[1-9][A-Z0-9]{3}", value):
        raise DockingError("Enter a four-character experimental PDB ID (not a URL).")
    return value


class PdbClient:
    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self.http = httpx.Client(timeout=20, transport=transport, follow_redirects=False)

    def __enter__(self) -> PdbClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.http.close()

    def _json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self.http.stream("POST", url, json=payload) as response:
                if response.status_code == 204:
                    return {}
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > 2 * 1024 * 1024:
                        raise DockingError("RCSB metadata response exceeds the browser limit.")
                import json

                return dict(json.loads(content))
        except (httpx.HTTPError, ValueError) as exc:
            raise DockingError(f"RCSB request failed: {exc}") from exc

    def search(self, accession: str) -> list[dict[str, Any]]:
        accession = accession.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", accession):
            raise DockingError("A valid UniProt accession is required for structure discovery.")
        result = self._json(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            {
                "query": {
                    "type": "group",
                    "logical_operator": "and",
                    "nodes": [
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": (
                                    "rcsb_polymer_entity_container_identifiers."
                                    "reference_sequence_identifiers.database_accession"
                                ),
                                "operator": "exact_match",
                                "value": accession,
                            },
                        },
                        {
                            "type": "terminal",
                            "service": "text",
                            "parameters": {
                                "attribute": (
                                    "rcsb_polymer_entity_container_identifiers."
                                    "reference_sequence_identifiers.database_name"
                                ),
                                "operator": "exact_match",
                                "value": "UniProt",
                            },
                        },
                    ],
                },
                "return_type": "entry",
                "request_options": {
                    "paginate": {"start": 0, "rows": 20},
                    "results_content_type": ["experimental"],
                    "sort": [
                        {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}
                    ],
                },
            },
        )
        identifiers = [normalise_pdb_id(row["identifier"]) for row in result.get("result_set", [])]
        return self.metadata(identifiers, accession=accession) if identifiers else []

    def metadata(self, ids: list[str], *, accession: str = "") -> list[dict[str, Any]]:
        ids = [normalise_pdb_id(x) for x in ids[:20]]
        data = self._json(
            "https://data.rcsb.org/graphql",
            {
                "query": METADATA_QUERY,
                "variables": {"ids": ids},
            },
        )
        if data.get("errors"):
            raise DockingError("RCSB metadata query failed; try again later.")
        output = []
        for entry in (data.get("data") or {}).get("entries", []) or []:
            if not entry:
                continue
            chains: list[str] = []
            variants: dict[str, str] = {}
            all_chains: list[str] = []
            for entity in entry.get("polymer_entities", []) or []:
                if not (entity.get("entity_poly") or {}).get("type", "").startswith("polypeptide"):
                    continue
                identity = entity.get("rcsb_polymer_entity_container_identifiers") or {}
                protein = entity.get("rcsb_polymer_entity") or {}
                for chain in identity.get("auth_asym_ids", []) or []:
                    all_chains.append(chain)
                    variants[chain] = protein.get("pdbx_mutation") or "Not reported"
                if accession in (identity.get("uniprot_ids") or []):
                    chains.extend(identity.get("auth_asym_ids") or [])
            resolutions = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            output.append(
                {
                    "pdb_id": normalise_pdb_id(entry["rcsb_id"]),
                    "title": (entry.get("struct") or {}).get("title", ""),
                    "method": ", ".join(e["method"] for e in entry.get("exptl", []) or []),
                    "resolution_A": min(resolutions) if resolutions else None,
                    "matched_chains": sorted(set(chains)),
                    "protein_chains": sorted(set(all_chains)),
                    "chain_mutations": variants,
                    "accession": accession,
                    "ligands": ", ".join(
                        (e.get("pdbx_entity_nonpoly") or {}).get("comp_id", "")
                        for e in entry.get("nonpolymer_entities", []) or []
                    ),
                }
            )
        return output

    def coordinates(self, pdb_id: str) -> str:
        identifier = normalise_pdb_id(pdb_id)
        try:
            with self.http.stream(
                "GET", f"https://files.rcsb.org/download/{identifier}.cif"
            ) as response:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_STRUCTURE_BYTES:
                        raise DockingError("PDB coordinate file exceeds the 20 MiB browser limit.")
                return content.decode("utf-8")
        except (httpx.HTTPError, UnicodeDecodeError) as exc:
            raise DockingError(f"Could not download {identifier}: {exc}") from exc
