"""Code, package and citation metadata identify the same local candidate."""

import tomllib
from pathlib import Path

from sarscope import __version__


def test_release_candidate_version_is_consistent():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["version"] == __version__ == "0.2.0rc1"
    assert f'version: "{__version__}"' in (root / "CITATION.cff").read_text()
    assert "email" not in project["authors"][0]
