"""Explicit, hash-verified retrieval of paper input data; no model fitting.

Uses only the Python standard library and never runs downloaded code.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

SOURCE_INDEX = Path(__file__).parent / "qsar3d/input-sources.json"
DATASETS = ("THR", "AChE", "STEROIDS_CBG")
MAX_BYTES = 8 * 1024 * 1024


def fetch_bytes(url: str, expected: str) -> bytes:
    """Bound one download and refuse any source drift rather than substituting it."""
    if not url.startswith("https://"):
        raise ValueError("Only the recorded HTTPS sources are supported.")
    request = Request(url, headers={"User-Agent": "SARscope-paper-data/1"})
    with urlopen(request, timeout=30) as response:
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Download exceeds the input size limit.")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Source hash differs; inspect the source instead of replacing it.")
    return payload


def verified_inputs(
    datasets: tuple[str, ...], source_index: Path = SOURCE_INDEX
) -> dict[str, bytes]:
    """Read only named data members; no filesystem archive extraction or execution."""
    if not datasets or set(datasets) - set(DATASETS):
        raise ValueError("Choose THR, AChE or STEROIDS_CBG.")
    sources = json.loads(source_index.read_text())
    labels = [source["dataset"] for source in sources]
    if len(labels) != len(set(labels)) or set(datasets) - set(labels):
        raise ValueError("Requested datasets are missing or duplicated in the source index.")
    result: dict[str, bytes] = {}
    archives: dict[tuple[str, str], bytes] = {}
    for source in sources:
        label = source["dataset"]
        if label not in datasets:
            continue
        if label == "STEROIDS_CBG":
            spec = source["historical_protocol"]
            for split in ("train", "test"):
                entry = spec[split]
                result[entry["file"]] = fetch_bytes(
                    spec["base_url"] + entry["file"], entry["sha256"]
                )
            continue
        key = (source["archive_url"], source["archive_sha256"])
        if key not in archives:
            archives[key] = fetch_bytes(*key)
        with ZipFile(io.BytesIO(archives[key])) as zipped:
            member = zipped.getinfo(source["nested_tar"])
            if member.file_size > MAX_BYTES:
                raise ValueError("Nested archive exceeds the input size limit.")
            nested = zipped.read(member)
        with tarfile.open(fileobj=io.BytesIO(nested)) as archive:
            member = archive.getmember(source["member"])
            if not member.isfile() or member.size > MAX_BYTES:
                raise ValueError("Expected a bounded, regular molecular data member.")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Expected molecular data member is missing.")
            payload = stream.read(MAX_BYTES + 1)
        if hashlib.sha256(payload).hexdigest() != source["member_sha256"]:
            raise ValueError("Molecular input hash differs; no substitute was written.")
        result[f"{label}_original.sdf"] = payload
    if not result:
        raise ValueError("Requested datasets are missing from the source index.")
    return result


def write_inputs(out: Path, inputs: dict[str, bytes]) -> list[Path]:
    """Preflight every destination; never overwrite a different existing file."""
    for name, payload in inputs.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Input destination must be a plain filename.")
        destination = out / name
        if destination.is_symlink():
            raise FileExistsError(f"Refusing a symlink destination: {destination}")
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace different data: {destination}")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, payload in inputs.items():
        destination = out / name
        if not destination.exists():
            with destination.open("xb") as handle:
                handle.write(payload)
        paths.append(destination)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory for verified input SDFs")
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    args = parser.parse_args()
    inputs = verified_inputs(tuple(args.dataset or DATASETS))
    for path in write_inputs(args.out, inputs):
        print(f"Verified {path}: {hashlib.sha256(inputs[path.name]).hexdigest()}")
    print("Data only: no coordinate changes, activity conversions or model fitting.")


if __name__ == "__main__":
    main()
