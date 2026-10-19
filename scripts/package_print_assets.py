"""Verify canonical dimensions, hash, and package the ready-to-print assets."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

import numpy as np

from orca_feetech.objects import OBJECT_VERSION, specification

ROOT = Path(__file__).resolve().parents[1]
assets = ROOT/"assets/objects/v3"
for kind in ("cube", "pen"):
    spec = json.loads((assets/f"{kind}.json").read_text())
    assert spec["version"] == OBJECT_VERSION
    assert np.allclose(spec["dimensions_m"], specification(kind)["dimensions_m"])
    assert spec["tags"] == specification(kind)["tags"]
paths = sorted(p for p in assets.iterdir() if p.is_file() and p.name not in
               ("a1mini-print-pack.zip", "SHA256SUMS"))
(assets/"SHA256SUMS").write_text("".join(
    hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.name+"\n" for p in paths))
with zipfile.ZipFile(assets/"a1mini-print-pack.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths+[assets/"SHA256SUMS"]:
        info = zipfile.ZipInfo("ORCA-"+OBJECT_VERSION+"/"+p.name)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, p.read_bytes())
for p in assets.iterdir():
    if p.is_file():
        shutil.copy2(p, ROOT/"docs/sim2real"/p.name)
print(f"Packaged {len(paths)+1} files, {OBJECT_VERSION}, and mirrored website downloads")
