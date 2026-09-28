"""What produced this report: versions, data source, parameters, time."""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from sarscope import __version__
from sarscope.params import RunParams

#: Distributions whose versions can change a result.
TRACKED_PACKAGES: tuple[str, ...] = (
    "sorbent",
    "rdkit",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "scikit-fingerprints",
)


def package_versions() -> dict[str, str | None]:
    """Installed version of each tracked package, None if absent."""
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chembl_source(target: dict[str, Any], release: str, n_records: int) -> dict[str, Any]:
    return {
        "kind": "chembl",
        "target_id": target.get("target_chembl_id"),
        "target_name": target.get("pref_name"),
        "organism": target.get("organism"),
        "release": release,
        "raw_records": n_records,
    }


def table_source(path: Path) -> dict[str, Any]:
    return {"kind": "table", "path": str(path), "sha256": file_sha256(path)}


def collect(params: RunParams, source: dict[str, Any]) -> dict[str, Any]:
    """JSON-serialisable provenance record."""
    return {
        "sarscope": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "source": source,
        "params": params.to_dict(),
    }
