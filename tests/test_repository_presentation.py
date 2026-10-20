"""Keep the small reader-facing front page and its real assets usable."""

import hashlib
import json
import re
import struct
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def test_readme_is_short_and_all_local_links_resolve():
    text = (ROOT / "README.md").read_text()
    assert len(text.split()) <= 250
    assert "assets/sarscope-logo.svg" in text
    assert "assets/screenshots/streamlit-home.png" in text
    links = re.findall(r"\]\(([^)]+)\)", text)
    links += re.findall(r'(?:href|src)="([^"]+)"', text)
    for link in links:
        if link.startswith(("https://", "http://", "#")):
            continue
        assert (ROOT / link.split("#")[0]).is_file(), link


def test_logo_assets_are_local_accessible_svg():
    for name in ("sarscope-logo.svg", "sarscope-mark.svg"):
        tree = ET.parse(ROOT / "assets" / name).getroot()
        ns = {"svg": "http://www.w3.org/2000/svg"}
        assert tree.get("viewBox")
        assert tree.find("svg:title", ns) is not None
        assert tree.find("svg:desc", ns) is not None
        assert not tree.findall(".//svg:script", ns)


def test_app_screenshot_matches_browser_capture_record():
    root = ROOT / "assets/screenshots"
    metadata = json.loads((root / "capture.json").read_text())
    data = (root / "streamlit-home.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", data[16:24]) == (1600, 960)
    assert hashlib.sha256(data).hexdigest() == metadata["screenshot_sha256"]
    assert metadata["source"] == "Actual local Streamlit app; current working tree"
    assert metadata["http_status"] == 200
    assert metadata["postprocessing"] == "None"
    assert metadata["page_errors"] == []
