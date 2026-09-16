"""Download exact upstream snapshots. No credentials or git installation required."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install(name: str, spec: dict) -> str:
    parent = ROOT / ".upstream"
    parent.mkdir(exist_ok=True)
    target = parent / name
    stamp = target / ".orca-source.json"
    if stamp.exists() and json.loads(stamp.read_text())["commit"] == spec["commit"]:
        return f"{name}: {spec['commit'][:12]} already present"
    if target.exists():
        raise RuntimeError(f"{target} exists without matching provenance; move it aside and retry")
    url = f"https://codeload.github.com/{spec['repository']}/tar.gz/{spec['commit']}"
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = response.read()
    with tempfile.TemporaryDirectory(dir=parent, prefix=f".{name}-") as tmp:
        stage = Path(tmp) / "content"
        stage.mkdir()
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                if ".." in parts or member.issym() or member.islnk():
                    raise RuntimeError(f"Unsupported archive member: {member.name}")
                member.name = str(Path(*parts))
                if member.isdir():
                    (stage / member.name).mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path = stage / member.name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, path.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
        (stage / ".orca-source.json").write_text(json.dumps({
            **spec, "archive_sha256": hashlib.sha256(payload).hexdigest()
        }, indent=2) + "\n")
        stage.rename(target)
    return f"{name}: {spec['commit'][:12]} installed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "upstream.lock.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(install, name, spec) for name, spec in manifest["repositories"].items()]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
