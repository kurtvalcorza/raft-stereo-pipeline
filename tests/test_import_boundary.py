"""Import-boundary contract: rejected requests never import model libraries (torch, cv2, scipy, the carried upstream).

Valid snapshots still reach them.
"""

import hashlib
import json

import pytest

from raft_stereo_pipeline import pipeline as pipeline_module
from raft_stereo_pipeline.pipeline import MANIFEST_NAME, MODEL_ID, WEIGHTS_ARCHIVE_URL, WEIGHTS_FILE, RaftStereoPipeline

_PAYLOAD = b"not-a-real-checkpoint"


def _snapshot(root, revision, tamper=False):
    (root / WEIGHTS_FILE).write_bytes(_PAYLOAD)
    digest = "0" * 64 if tamper else hashlib.sha256(_PAYLOAD).hexdigest()
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [{"path": WEIGHTS_FILE, "url": WEIGHTS_ARCHIVE_URL, "bytes": len(_PAYLOAD), "sha256": digest}],
        "totalBytes": len(_PAYLOAD),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_unpinned_package_refuses_before_model_imports(tmp_path, monkeypatch, forbid_model_imports):
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", "unpinned")
    with pytest.raises(RuntimeError, match="no pinned checkpoint digest"):
        RaftStereoPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)
    with pytest.raises(RuntimeError, match="no pinned checkpoint digest"):
        pipeline_module.stage_missing_files(tmp_path, allow_download=True)
    with pytest.raises(RuntimeError, match="no pinned checkpoint digest"):
        pipeline_module.verify_snapshot(tmp_path)


def test_from_pretrained_refuses_without_manifest_before_model_imports(tmp_path, pinned, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="no checkpoint manifest"):
        RaftStereoPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_from_pretrained_refuses_tampered_checkpoint_before_model_imports(tmp_path, pinned, forbid_model_imports):
    _snapshot(tmp_path, pinned, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        RaftStereoPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_from_pretrained_valid_snapshot_reaches_model_import(tmp_path, pinned, forbid_model_imports):
    _snapshot(tmp_path, pinned)
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        RaftStereoPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_rejected_inputs_never_import_model_libraries(forbid_model_imports):
    from PIL import Image

    with pytest.raises(ValueError, match="differ in size"):
        pipeline_module.validate_inputs(Image.new("RGB", (100, 80)), Image.new("RGB", (90, 80)))
