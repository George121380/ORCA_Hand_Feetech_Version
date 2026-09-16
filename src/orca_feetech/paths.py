from __future__ import annotations

import json
import os
from pathlib import Path


def workspace() -> Path:
    candidates = [Path(os.environ["ORCA_WORKSPACE"])] if "ORCA_WORKSPACE" in os.environ else []
    candidates.extend([Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents])
    for path in candidates:
        if (path / "upstream.lock.json").is_file():
            return path
    raise RuntimeError("Run inside the repository or set ORCA_WORKSPACE to its directory")


def upstream(name: str) -> Path:
    root = workspace()
    path = root / ".upstream" / name
    expected = json.loads((root / "upstream.lock.json").read_text())["repositories"][name]["commit"]
    stamp = path / ".orca-source.json"
    if not stamp.is_file() or json.loads(stamp.read_text()).get("commit") != expected:
        raise RuntimeError(f"Missing/mismatched {name} snapshot; run python scripts/bootstrap.py")
    return path


def provenance() -> dict:
    return json.loads((workspace() / "upstream.lock.json").read_text())
