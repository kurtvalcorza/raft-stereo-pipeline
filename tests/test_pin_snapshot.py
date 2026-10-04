"""tools/pin_snapshot.py on a copy of the repository: a fake upstream archive holding a random-weight checkpoint of the
real architecture is pinned (manifest + MODEL_REVISION), and defective archives or checkpoints pin nothing."""

# ruff: noqa: E501  -- refusal messages and docstrings name the rule in full; they are kept on one line
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("pin_snapshot", ROOT / "tools" / "pin_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for rel in ("tools/notebook_template.py", "weights/raftstereo-middlebury/dimer-base-manifest.json"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    shutil.copytree(ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "STATUS.md").write_text("The checkpoint is not yet pinned.\n")
    # The tool imports the package from <repo>/src; make sure it is the copy, not the installed tree, and put the
    # original modules back afterwards so later tests never see the copy's `core` package.
    def ours(name):
        return name.split(".")[0] in ("raft_stereo_pipeline", "core")

    saved = {name: module for name, module in sys.modules.items() if ours(name)}
    for name in saved:
        del sys.modules[name]
    monkeypatch.setattr(sys, "path", [str(tmp_path / "src"), *sys.path])
    yield tmp_path
    for name in [name for name in sys.modules if ours(name)]:
        del sys.modules[name]
    sys.modules.update(saved)


def _archive(path, checkpoint_bytes, member="models/raftstereo-middlebury.pth"):
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(member, checkpoint_bytes)
        bundle.writestr("models/raftstereo-eth3d.pth", b"another checkpoint")
    return path


def _checkpoint(repo):
    sys.path.insert(0, str(repo / "src"))
    from raft_stereo_pipeline.pipeline import build_model

    torch.manual_seed(0)
    buffer = io.BytesIO()
    torch.save(torch.nn.DataParallel(build_model()).state_dict(), buffer)
    return buffer.getvalue()


def test_pin_records_the_checkpoint_digest_and_size(repo, tmp_path):
    data = _checkpoint(repo)
    archive = _archive(tmp_path / "models.zip", data)
    assert _tool().pin(repo, archive=archive) == 0
    digest = hashlib.sha256(data).hexdigest()
    manifest = json.loads((repo / "weights/raftstereo-middlebury/dimer-base-manifest.json").read_text())
    assert manifest["revision"] == digest and manifest["totalBytes"] == len(data)
    assert manifest["files"][0]["sha256"] == digest and manifest["files"][0]["archiveMemberPath"] == "models/raftstereo-middlebury.pth"
    assert manifest["archiveAtPin"]["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert f'MODEL_REVISION = "{digest}"' in (repo / "src/raft_stereo_pipeline/pipeline.py").read_text()
    assert (repo / "weights/raftstereo-middlebury/raftstereo-middlebury.pth").read_bytes() == data


def test_dry_run_writes_nothing(repo, tmp_path):
    before = (repo / "src/raft_stereo_pipeline/pipeline.py").read_text()
    assert _tool().pin(repo, archive=_archive(tmp_path / "models.zip", _checkpoint(repo)), dry_run=True) == 0
    assert (repo / "src/raft_stereo_pipeline/pipeline.py").read_text() == before
    assert json.loads((repo / "weights/raftstereo-middlebury/dimer-base-manifest.json").read_text())["revision"] == "unpinned"


def test_a_checkpoint_that_does_not_load_strictly_pins_nothing(repo, tmp_path):
    buffer = io.BytesIO()
    torch.save({"module.wrong": torch.zeros(1)}, buffer)
    before = (repo / "src/raft_stereo_pipeline/pipeline.py").read_text()
    assert _tool().pin(repo, archive=_archive(tmp_path / "models.zip", buffer.getvalue())) == 1
    assert (repo / "src/raft_stereo_pipeline/pipeline.py").read_text() == before


def test_an_archive_without_the_member_pins_nothing(repo, tmp_path):
    archive = _archive(tmp_path / "models.zip", b"x", member="models/raftstereo-sceneflow.pth")
    assert _tool().pin(repo, archive=archive) == 1
    assert json.loads((repo / "weights/raftstereo-middlebury/dimer-base-manifest.json").read_text())["revision"] == "unpinned"


def test_the_default_download_uses_the_manifest_url(repo, tmp_path):
    data = _checkpoint(repo)
    seen = []

    def download(url, target):
        seen.append(url)
        _archive(target, data)
        return target.stat().st_size

    assert _tool().pin(repo, download=download, dry_run=True) == 0
    manifest = json.loads((repo / "weights/raftstereo-middlebury/dimer-base-manifest.json").read_text())
    assert seen == [manifest["files"][0]["url"]]
