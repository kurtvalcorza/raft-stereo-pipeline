"""The real upstream RAFT-Stereo architecture with seeded random weights, on small rendered pairs (no checkpoint).

This proves the wiring — padding, the disparity sign, the sequence loss, what is frozen, and that an exported adapter
reloads to the same output — not the model's accuracy.
"""

import json

import numpy as np
import pytest
import torch

from conftest import tiny_records
from raft_stereo_pipeline import pipeline as P
from raft_stereo_pipeline.pipeline import EXPECTED_PARAMETERS, EXPECTED_STATE_TENSORS, RaftStereoPipeline


@pytest.fixture(scope="module")
def records():
    return tiny_records(4, seed=2)


def test_architecture_matches_the_checkpoint_layout():
    pipe = RaftStereoPipeline.from_random_init(device="cpu")
    assert sum(p.numel() for p in pipe.model.parameters()) == EXPECTED_PARAMETERS
    assert len(pipe.model.state_dict()) == EXPECTED_STATE_TENSORS
    # The upstream checkpoints are DataParallel state dicts: the prefix is stripped on load.
    assert all(not k.startswith("module.") for k in pipe.model.state_dict())


def test_checkpoint_state_loader_strips_dataparallel_prefix(tmp_path):
    pipe = RaftStereoPipeline.from_random_init(device="cpu")
    torch.save({"module." + k: v for k, v in pipe.model.state_dict().items()}, tmp_path / "c.pth")
    state = P.load_checkpoint_state(tmp_path / "c.pth")
    pipe.model.load_state_dict(state, strict=True)


def test_estimate_returns_left_image_disparity_of_the_input_size(records):
    pipe = RaftStereoPipeline.from_random_init(device="cpu")
    out = pipe.estimate(records[0]["left"], records[0]["right"], iters=2)
    assert out["disparity"].shape == (64, 96) and out["disparity"].dtype == np.float32
    assert np.isfinite(out["disparity"]).all() and out["iters"] == 2


def test_sequence_loss_matches_upstream_weighting():
    truth = torch.zeros(1, 1, 2, 2)
    valid = torch.ones(1, 2, 2)
    predictions = [torch.full((1, 1, 2, 2), 2.0), torch.full((1, 1, 2, 2), 1.0)]
    expected = 0.9 ** 15 * 2.0 + 1.0
    assert float(P.sequence_loss(predictions, truth, valid)) == pytest.approx(expected)


def test_finetune_trains_only_unfrozen_tensors_and_the_adapter_reloads(tmp_path, records, monkeypatch):
    pipe = RaftStereoPipeline.from_random_init(device="cpu", seed=0)
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    run = pipe.finetune(records[:2], epochs=1, batch_size=2, train_iters=2, seed=1, freeze_encoders=True)
    after = pipe.model.state_dict()
    assert run["steps"] == 1 and run["freeze_encoders"] and np.isfinite(run["final_loss"])
    assert run["trainable_parameters"] < run["total_parameters"] == EXPECTED_PARAMETERS
    for name, value in after.items():
        if name.startswith(P.ENCODER_PREFIXES):
            assert torch.equal(value, before[name]), f"frozen tensor changed: {name}"
    assert any(not torch.equal(after[k], before[k]) for k in after if k.startswith("update_block."))

    reference = pipe.estimate(records[2]["left"], records[2]["right"], iters=3)["disparity"]
    descriptor = pipe.save_artifact(tmp_path / "adapter.safetensors", notes="test")
    assert descriptor["frozen_prefixes"] == list(P.ENCODER_PREFIXES)
    assert descriptor["tensors"] < EXPECTED_STATE_TENSORS
    metadata = RaftStereoPipeline.read_artifact_metadata(tmp_path / "adapter.safetensors")
    assert metadata["upstream_commit"] == P.UPSTREAM_COMMIT and json.loads(metadata["model_args"]) == P.MODEL_ARGS

    fresh = RaftStereoPipeline.from_random_init(device="cpu", seed=0)  # same base weights as `pipe` before training
    fresh.apply_artifact(tmp_path / "adapter.safetensors")
    reloaded = fresh.estimate(records[2]["left"], records[2]["right"], iters=3)["disparity"]
    assert np.abs(reloaded - reference).max() <= 1e-4


def test_unfrozen_finetune_exports_every_tensor(tmp_path, records):
    pipe = RaftStereoPipeline.from_random_init(device="cpu", seed=0)
    run = pipe.finetune(records[:2], epochs=1, batch_size=2, train_iters=1, freeze_encoders=False)
    assert run["trainable_parameters"] == run["total_parameters"]
    assert pipe.save_artifact(tmp_path / "a.safetensors")["tensors"] == EXPECTED_STATE_TENSORS


def test_artifact_from_another_base_or_format_is_refused(tmp_path, records):
    from safetensors.torch import save_file

    pipe = RaftStereoPipeline.from_random_init(device="cpu", seed=0)
    with pytest.raises(RuntimeError, match="not been fine-tuned"):
        pipe.save_artifact(tmp_path / "x.safetensors")
    pipe.finetune(records[:2], epochs=1, batch_size=2, train_iters=1)
    path = tmp_path / "a.safetensors"
    pipe.save_artifact(path)
    tensors = {"update_block.flow_head.conv2.bias": torch.zeros(2)}
    base = RaftStereoPipeline.read_artifact_metadata(path)
    for change, message in (
        ({"format": "other"}, "artifact format"),
        ({"upstream_commit": "0" * 40}, "upstream code"),
        ({"model_revision": "f" * 64}, "package pins"),
    ):
        bad = tmp_path / "bad.safetensors"
        meta = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in base.items()}
        save_file(tensors, str(bad), metadata={**meta, **change})
        with pytest.raises(ValueError, match=message):
            RaftStereoPipeline.read_artifact_metadata(bad)
    other_base = RaftStereoPipeline.from_random_init(device="cpu", seed=0)
    other_base.base_state_digest = "e" * 64
    with pytest.raises(ValueError, match="exported against base"):
        other_base.apply_artifact(path)
