"""ChEMBL REST client: target lookup, release version, and paged activity download.

Three decisions worth knowing about:

**Always resolve the target first.** A wrong target ID does not fail; it fetches
a different protein. The BRAF paper this tool reproduces gives "target ID: 5651",
but CHEMBL5651 is STK35 (3 IC50 records), not BRAF (CHEMBL5145, 11,017). The
CLI prints the resolved name and organism before downloading anything.

**The cache key includes the ChEMBL release.** Responses are cached as gzipped
JSON so a report can be regenerated offline, but a new release must never be
served stale data under the old key, and the release is part of provenance.

**Raw records are stored whole.** Curation decides which fields matter; the
download does not pre-filter, so every curation choice stays auditable against
what ChEMBL actually returned.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

#: Page size for activity downloads. 1000 is the API maximum.
PAGE_SIZE = 1000

_TARGET_ID = re.compile(r"^CHEMBL\d+$")


class ChemblError(RuntimeError):
    """The API returned something unusable, or could not be reached."""


class TargetNotFoundError(ChemblError):
    pass


def normalise_target_id(target: str) -> str:
    """Accept "CHEMBL5145", "chembl5145" or a bare "5145"; return "CHEMBL5145"."""
    text = target.strip().upper()
    if text.isdigit():
        text = f"CHEMBL{text}"
    if not _TARGET_ID.match(text):
        raise ValueError(f"not a ChEMBL target ID: {target!r}")
    return text


class ChemblClient:
    """Thin synchronous client. One instance per run.

    ``transport`` and ``sleep`` exist for tests: pass an ``httpx.MockTransport``
    and a no-op sleep to exercise paging and retries without the network.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        cache_dir: Path | None = None,
        timeout: float = 120.0,
        retries: int = 4,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base = httpx.URL(base_url.rstrip("/") + "/")
        self._cache_dir = cache_dir
        self._retries = retries
        self._sleep = sleep
        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)
        self._status: dict[str, Any] | None = None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ChemblClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------

    def _get(self, url: httpx.URL | str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with retries on transport errors and 5xx. 4xx is returned as-is."""
        delay = 1.0
        for attempt in range(self._retries + 1):
            try:
                response = self._http.get(url, params=params)
            except httpx.TransportError as exc:
                if attempt == self._retries:
                    raise ChemblError(f"ChEMBL unreachable: {exc}") from exc
            else:
                if response.status_code < 500:
                    return response
                if attempt == self._retries:
                    raise ChemblError(f"ChEMBL returned HTTP {response.status_code} for {url}")
            self._sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")  # pragma: no cover

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._get(self._base.join(path), params)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    # -- public ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Release metadata, e.g. {"chembl_db_version": "ChEMBL_37", ...}. Cached."""
        if self._status is None:
            self._status = self._get_json("status.json")
        return self._status

    @property
    def release(self) -> str:
        return str(self.status()["chembl_db_version"])

    def target(self, target_id: str) -> dict[str, Any]:
        """Target record (pref_name, organism, target_type, ...)."""
        target_id = normalise_target_id(target_id)
        response = self._get(self._base.join(f"target/{target_id}.json"))
        if response.status_code == 404:
            raise TargetNotFoundError(f"{target_id} does not exist in {self.release}")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data

    def count_activities(self, target_id: str, standard_types: Iterable[str] = ("IC50",)) -> int:
        """Number of activity records, from one single-row request. No download."""
        target_id = normalise_target_id(target_id)
        page = self._get_json(
            "activity.json",
            {
                "target_chembl_id": target_id,
                "standard_type__in": ",".join(sorted(set(standard_types))),
                "limit": 1,
            },
        )
        return int(page["page_meta"]["total_count"])

    def activities(
        self, target_id: str, standard_types: Iterable[str] = ("IC50",)
    ) -> list[dict[str, Any]]:
        """Every activity record for the target and types, all pages, cached."""
        target_id = normalise_target_id(target_id)
        types = sorted(set(standard_types))
        cache_file = self._cache_path(target_id, types)
        if cache_file is not None and cache_file.exists():
            with gzip.open(cache_file, "rt", encoding="utf-8") as fh:
                cached: list[dict[str, Any]] = json.load(fh)
            return cached

        records = self._download(target_id, types)

        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(records, fh)
            tmp.replace(cache_file)  # atomic: an interrupted run leaves no half-file
        return records

    def _download(self, target_id: str, types: list[str]) -> list[dict[str, Any]]:
        params: dict[str, Any] | None = {
            "target_chembl_id": target_id,
            "standard_type__in": ",".join(types),
            "limit": PAGE_SIZE,
        }
        url: httpx.URL | None = self._base.join("activity.json")
        records: list[dict[str, Any]] = []
        expected: int | None = None
        while url is not None:
            response = self._get(url, params)
            response.raise_for_status()
            page = response.json()
            meta = page["page_meta"]
            expected = meta["total_count"] if expected is None else expected
            records.extend(page["activities"])
            # "next" is an absolute path carrying the query, so drop our params.
            url = self._base.join(meta["next"]) if meta["next"] else None
            params = None
        if expected is not None and len(records) != expected:
            raise ChemblError(
                f"expected {expected} activities for {target_id}, received {len(records)}; "
                "the release may have changed mid-download"
            )
        return records

    def _cache_path(self, target_id: str, types: list[str]) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / self.release / f"{target_id}_{'-'.join(types)}.json.gz"
