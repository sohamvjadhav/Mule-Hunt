import tomllib
from pathlib import Path

import upifraud


def test_version_matches_pyproject():
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert upifraud.__version__ == pyproject["project"]["version"]
