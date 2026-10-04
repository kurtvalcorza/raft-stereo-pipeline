"""The isolated-environment tutorial path (NOTEBOOK_SPEC 2.2 §25.13): the kernel's `run_stage` helper and the carried
stage runner.

* The kernel-side test executes the generated notebook's own carrier and `run_stage` code: the carried files are
  written and hash-verified into a run directory, and a failing stage stops the kernel with a RuntimeError that repeats
  the stage's own message (here: the checkpoint is not pinned).
* The CPU pre-flight runs every stage in order — sample path and BYOD dataset path — with the real upstream
  architecture under seeded random weights standing in for the checkpoint, on small rendered pairs. Each stage builds
  its own model, so everything a later stage uses crosses over through files in the run directory. It proves the stage
  plumbing and the hand-offs, not the model's accuracy.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from raft_stereo_pipeline import pipeline as P
from raft_stereo_pipeline.samples import stereo_dataset, write_pfm

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE


def _infrastructure_sources() -> tuple[str, str]:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    carrier = next(s for s in code if s.startswith("# @title Infrastructure: write and verify the carried"))
    install = next(s for s in code if s.startswith("# @title Infrastructure: install the locked runtime"))
    return carrier, install


def _kernel(tmp_path: Path) -> dict:
    """The kernel namespace after the carrier cell and the `run_stage` definition, with the current interpreter
    standing in for the isolated environment's Python."""
    carrier, install = _infrastructure_sources()
    run_root = tmp_path / "run"
    run_root.mkdir()
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("PYTHONPATH", None)
    namespace = {"ROOT": run_root, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env}
    exec("import hashlib\nimport json\nimport subprocess\n" + carrier, namespace)  # noqa: S102 - the notebook's own cell
    definition = install[install.index("def run_stage(") : install.index("def load_record(")]
    exec(definition, namespace)  # noqa: S102
    return namespace


def test_carrier_writes_and_verifies_every_carried_file(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    run_root = kernel["ROOT"]
    for dest, source in TEMPLATE["carried"].items():
        assert (run_root / dest).read_bytes() == (ROOT / source).read_text(encoding="utf-8").encode("utf-8"), dest
    assert kernel["NOTEBOOK_SOURCE"]["revision"] == "test-revision"
    # The carried upstream copy passes the package's own check from its carried location.
    assert P.verify_upstream(run_root / f"src/{TEMPLATE['package']}/third_party/raft_stereo")["files"] == 8


def test_carrier_refuses_a_changed_file(tmp_path: Path) -> None:
    carrier, _install = _infrastructure_sources()
    tampered = carrier.replace("UPSTREAM_COMMIT = ", "UPSTREAM_COMMIT  = ", 1)
    assert tampered != carrier
    with pytest.raises(RuntimeError, match="Carried file integrity failure"):
        exec("import hashlib\nimport json\n" + tampered, {"ROOT": tmp_path})  # noqa: S102


@pytest.mark.skipif(P.is_pinned(), reason="the checkpoint is pinned; the refusal applies to an unpinned revision only")
def test_unpinned_weights_stage_stops_the_kernel_with_the_stage_message(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match=r"Stage 'weights' failed .*no pinned checkpoint digest"):
        kernel["run_stage"]("weights")
    assert (kernel["ROOT"] / "logs" / "weights.log").is_file()


# --------------------------------------------------------------------------------------------------
# CPU pre-flight of every stage (random weights, small pairs)
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def stages(monkeypatch, tmp_path):
    ts = _load("tutorial_stages")

    def base(run):
        return P.RaftStereoPipeline.from_random_init(device="cpu", seed=0)

    def adapted(run, artifact):
        pipe = P.RaftStereoPipeline.from_random_init(device="cpu", seed=0)
        pipe.apply_artifact(artifact)
        return pipe

    monkeypatch.setattr(ts, "load_base", base)
    monkeypatch.setattr(ts, "load_adapted", adapted)
    monkeypatch.setattr(ts, "sample_records", lambda n, seed: stereo_dataset(n, seed=seed, width=96, height=64))
    root = tmp_path / "run"
    root.mkdir()
    (root / "source.json").write_text(json.dumps({"repository": "test", "revision": "test-revision"}), encoding="utf-8")

    def run(stage, *options):
        code = ts.main(["--root", str(root), "--weights", str(tmp_path / "weights"), "--stage", stage, *map(str, options)])
        if code:
            error = json.loads((root / "state" / f"{stage}.error.json").read_text(encoding="utf-8"))
            raise RuntimeError(f"{stage}: {error['type']}: {error['message']}")

    return run, root


def test_sample_path_runs_every_stage_in_order(stages) -> None:
    run, root = stages
    run("demo", "--iters", 2)
    run("probes", "--iters", 2)
    run("prepare", "--n-pairs", 4, "--dataset-seed", 0, "--holdout", 0.25, "--seed", 1, "--epochs", 1)
    run("baselines", "--iters", 2)
    run("adapt", "--epochs", 1, "--batch-size", 2, "--lr", 2e-5, "--train-iters", 1, "--freeze-encoders", 1, "--seed", 1, "--iters", 2)
    run("evaluate", "--iters", 2)
    run("infer", "--new-seed", 99, "--iters", 2)
    run("reload")
    out = root / "outputs"
    for name in ("raft_stereo_input_manifest.json", "raft_stereo_evaluation_report.json", "probes.json", "raft_stereo_dataset.json", "raft_stereo_baselines.json", "raft_stereo_adaptation.json", "raft_stereo_held_out_evaluation.json", "raft_stereo_new_pairs.csv", "raft_stereo_new_pairs.json", "raft_stereo_adapter.safetensors", "raft_stereo_result.json", "raft_stereo_demo.png", "raft_stereo_new_pairs.png"):
        assert (out / name).is_file(), name
    manifest = json.loads((out / "raft_stereo_input_manifest.json").read_text())
    assert manifest["verdict"] == "accepted" and "differ in size" in manifest["findings"][-1]["rejected"]
    dataset = json.loads((out / "raft_stereo_dataset.json").read_text())
    assert len(dataset["split"]["train"]) == 3 and len(dataset["split"]["held_out"]) == 1
    assert all("rejected" in probe for probe in dataset["refusal_probes"])
    result = json.loads((out / "raft_stereo_result.json").read_text())
    assert result["reload_parity"]["equivalent"] is True
    assert result["upstream_code"] == {"repository": P.UPSTREAM_REPOSITORY, "commit": P.UPSTREAM_COMMIT, "carried": True, "remote_code_fetched": False}
    assert set(result["held_out_table"]) == {"median", "sgbm", "pretrained", "adapted"}
    assert len(list((out / "raft_stereo_disparity").glob("*.npy"))) == 3
    evaluation = json.loads((out / "raft_stereo_held_out_evaluation.json").read_text())
    assert len(evaluation["run_history"]) == 1
    # Re-running adapt and evaluate (the Section 14 activity) starts from the checkpoint and adds a history row.
    run("adapt", "--epochs", 1, "--batch-size", 2, "--lr", 2e-5, "--train-iters", 1, "--freeze-encoders", 0, "--seed", 1, "--iters", 2)
    run("evaluate", "--iters", 2)
    history = json.loads((out / "raft_stereo_held_out_evaluation.json").read_text())["run_history"]
    assert [row["freeze_encoders"] for row in history] == [True, False]
    assert history[1]["trainable_parameters"] == P.EXPECTED_PARAMETERS


def test_evaluate_refuses_baselines_scored_with_other_iterations(stages) -> None:
    run, _root = stages
    run("prepare", "--n-pairs", 4, "--epochs", 1)
    run("baselines", "--iters", 2)
    run("adapt", "--epochs", 1, "--train-iters", 1, "--iters", 2)
    with pytest.raises(RuntimeError, match="baselines were scored with 2 iterations"):
        run("evaluate", "--iters", 3)


def test_a_stage_run_out_of_order_names_the_missing_stage(stages) -> None:
    run, _root = stages
    with pytest.raises(RuntimeError, match="data.json is missing"):
        run("baselines", "--iters", 2)


def _byod_folder(root: Path, n: int = 4) -> Path:
    root.mkdir(parents=True)
    entries = []
    for i, record in enumerate(stereo_dataset(n, seed=5, width=96, height=64)):
        record["left"].save(root / f"s{i}_im0.png")
        record["right"].save(root / f"s{i}_im1.png")
        write_pfm(root / f"s{i}_disp0.pfm", np.where(record["valid"], record["disparity"], np.inf).astype(np.float32))
        entries.append({"left": f"s{i}_im0.png", "right": f"s{i}_im1.png", "disparity": f"s{i}_disp0.pfm", "group": f"scene{i}"})
    (root / "pairs.json").write_text(json.dumps(entries), encoding="utf-8")
    return root


def test_byod_dataset_reaches_adaptation_evaluation_inference_and_reload(stages, tmp_path) -> None:
    """DAT14 / REL12: a compatible BYOD folder runs validate → split → baselines → adapt → evaluate → infer → reload."""
    run, root = stages
    folder = _byod_folder(tmp_path / "byod")
    common = ["--namespace", "byod"]
    run("prepare", *common, "--byod", folder, "--holdout", 0.25, "--seed", 1, "--epochs", 1)
    run("baselines", *common, "--iters", 2)
    run("adapt", *common, "--epochs", 1, "--batch-size", 2, "--train-iters", 1, "--iters", 2)
    run("evaluate", *common, "--iters", 2)
    run("infer", *common, "--iters", 2)
    run("reload", *common)
    out = root / "outputs" / "byod"
    result = json.loads((out / "byod_raft_stereo_result.json").read_text())
    assert result["data_source"] == "byod" and result["reload_parity"]["equivalent"] is True
    assert (out / "byod_raft_stereo_adapter.safetensors").is_file()
    assert not (root / "outputs" / "raft_stereo_result.json").exists(), "BYOD must not write the sample's files"


def test_byod_dataset_with_an_incompatible_record_is_refused_by_name(stages, tmp_path) -> None:
    run, _root = stages
    folder = _byod_folder(tmp_path / "bad")
    entries = json.loads((folder / "pairs.json").read_text())
    entries[1]["right"] = "../escape.png"
    (folder / "pairs.json").write_text(json.dumps(entries))
    with pytest.raises(RuntimeError, match="must be relative to the dataset folder"):
        run("prepare", "--namespace", "byod", "--byod", folder)
    Image.new("RGB", (50, 64)).save(folder / "s1_im1.png")
    entries[1]["right"] = "s1_im1.png"
    (folder / "pairs.json").write_text(json.dumps(entries))
    with pytest.raises(RuntimeError, match="record 1: left and right images differ in size"):
        run("prepare", "--namespace", "byod", "--byod", folder)


def test_byod_pair_branch_with_and_without_ground_truth(stages, tmp_path) -> None:
    run, root = stages
    folder = _byod_folder(tmp_path / "pair", n=1)
    run("byod_pair", "--namespace", "byod", "--left", folder / "s0_im0.png", "--right", folder / "s0_im1.png", "--disparity", folder / "s0_disp0.pfm", "--iters", 2)
    report = json.loads((root / "outputs" / "byod" / "byod_pair_report.json").read_text())
    assert report["verdict"] == "sample-sanity" and report["baselines"][0]["id"] == "sgbm"
    run("byod_pair", "--namespace", "byod", "--left", folder / "s0_im0.png", "--right", folder / "s0_im1.png", "--iters", 2)
    report = json.loads((root / "outputs" / "byod" / "byod_pair_report.json").read_text())
    assert report["verdict"] == "not-measurable"
    assert (root / "outputs" / "byod" / "byod_pair_disparity.npy").is_file()
