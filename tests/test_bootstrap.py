"""Snapshot installation must support upstream mesh links without escaping."""
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bootstrap", Path(__file__).parents[1] / "scripts/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def snapshot(target):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        mesh = tarfile.TarInfo("repo/v1/assets/mesh.stl")
        mesh.size = 4
        archive.addfile(mesh, io.BytesIO(b"mesh"))
        link = tarfile.TarInfo("repo/v1/models/assets")
        link.type = tarfile.SYMTYPE
        link.linkname = target
        archive.addfile(link)
    return buffer.getvalue()


def test_official_relative_mesh_link(tmp_path):
    bootstrap.extract_snapshot(snapshot("../assets"), tmp_path)
    assert (tmp_path / "v1/models/assets").is_symlink()
    assert (tmp_path / "v1/models/assets/mesh.stl").read_bytes() == b"mesh"


@pytest.mark.parametrize("target", ["../../../outside", "/tmp/outside"])
def test_snapshot_rejects_external_links(tmp_path, target):
    with pytest.raises(RuntimeError, match="escapes snapshot"):
        bootstrap.extract_snapshot(snapshot(target), tmp_path)
