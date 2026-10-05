"""Regression tests for the 2026-10-05 notebook review findings (RST-M1, RST-m1..m4).

They need no torch: the stage runner runs with a **stand-in** pipeline (an integer block matcher that, like the recorded
diagnostic of the real checkpoint, answers far off for uniform shifts below 8 px), and the generated notebook's own cell
sources are executed with stand-ins. None of this is model evidence; the real-checkpoint behaviour is the recorded
Colab diagnostic in docs/execution-evidence/2026-10-05/probe_shift_diagnostic.md.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from raft_stereo_pipeline import pipeline as P
from raft_stereo_pipeline import stereo_dataset

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"rst_fix_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


class StandInPipeline:
    """Not a model. Picks the integer shift in 0..40 px that best aligns the two images and returns it everywhere;
    shifts below 8 px are answered as 250 px, mirroring the recorded failure of the real checkpoint."""

    device = "cpu"
    source = "stand-in"

    def __init__(self, adapted: bool = False, freeze: bool = True) -> None:
        self.adapted = adapted
        self.freeze = freeze

    def estimate(self, left, right, *, iters: int = 32) -> dict:
        a = np.asarray(left.convert("L"), dtype=np.float32)
        b = np.asarray(right.convert("L"), dtype=np.float32)
        costs = [np.abs(a[:, s:] - b[:, : a.shape[1] - s]).mean() for s in range(0, 41)]
        best = int(np.argmin(costs))
        value = 250.0 if best < 8 and np.std(a) > 0 and abs(np.corrcoef(a[:, 40:].ravel(), a[:, 39:-1].ravel())[0, 1]) < 0.9 else float(best)
        return {"disparity": np.full(a.shape, value + (0.01 if self.adapted else 0.0), dtype=np.float32)}

    def evaluate(self, records, *, iters: int = 32) -> dict:
        rows = []
        for index, record in enumerate(records):
            disparity = self.estimate(record["left"], record["right"], iters=iters)["disparity"]
            rows.append({"id": str(record.get("id", index)), **P.disparity_metrics(disparity, record["disparity"], P._valid(record))})
        return {**P.aggregate_metrics(rows), "per_pair": rows, "iters": iters, "adapted": self.adapted, "estimation": "stand-in"}

    def finetune(self, train, *, freeze_encoders: bool, epochs: int, progress=None, **kwargs) -> dict:
        self.adapted, self.freeze = True, freeze_encoders
        for epoch in range(1, epochs + 1):
            if progress:
                progress({"epoch": epoch, "epochs": epochs, "loss": 1.0 / epoch, "steps": 1})
        return {"freeze_encoders": freeze_encoders, "trainable_parameters": 1 if freeze_encoders else 2, "total_parameters": 2, "steps": epochs, "batch_size": kwargs.get("batch_size"),
                "learning_rate": kwargs.get("learning_rate"), "train_iters": kwargs.get("train_iters"), "precision": "float32", "epochs": epochs, "final_loss": 1.0 / epochs}

    def save_artifact(self, path, *, notes=None) -> dict:
        path = Path(path)
        path.write_bytes(b"stand-in adapter, encoders " + (b"frozen" if self.freeze else b"trained"))
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tensors": 1, "frozen_prefixes": ["fnet"] if self.freeze else []}


@pytest.fixture
def stages(monkeypatch, tmp_path):
    ts = _load("tutorial_stages")
    monkeypatch.setattr(ts, "load_base", lambda run: StandInPipeline())
    monkeypatch.setattr(ts, "load_adapted", lambda run, artifact: StandInPipeline(adapted=True, freeze=b"frozen" in Path(artifact).read_bytes()))
    monkeypatch.setattr(ts, "runtime_versions", lambda: {"python": "stand-in"})
    monkeypatch.setattr(ts, "sample_records", lambda n, seed: stereo_dataset(n, seed=seed, width=96, height=64))
    monkeypatch.setattr(P.RaftStereoPipeline, "read_artifact_metadata", staticmethod(lambda path: {"format": "stand-in", "model_id": "stand-in", "model_revision": "stand-in", "frozen_prefixes": []}))
    root = tmp_path / "run"
    root.mkdir()
    (root / "source.json").write_text(json.dumps({"repository": "test", "revision": "test-revision"}), encoding="utf-8")

    def run(stage, *options):
        code = ts.main(["--root", str(root), "--weights", str(tmp_path / "weights"), "--stage", stage, *map(str, options)])
        if code:
            error = json.loads((root / "state" / "sample" / f"{stage}.error.json").read_text(encoding="utf-8")) if (root / "state" / "sample" / f"{stage}.error.json").is_file() else json.loads((root / "state" / f"{stage}.error.json").read_text(encoding="utf-8"))
            raise RuntimeError(f"{stage}: {error['type']}: {error['message']}")

    return run, root


# ---- RST-M1 -------------------------------------------------------------------------------------------------------


def test_rst_M1_small_shift_probes_are_shown_with_a_verdict_and_never_stop_the_stage(stages) -> None:
    run, root = stages
    run("probes", "--iters", 2)
    record = json.loads((root / "outputs" / "probes.json").read_text(encoding="utf-8"))
    assert [p["true_disparity_px"] for p in record["probes"]] == [8, 16]
    assert [p["true_disparity_px"] for p in record["limitation_probes"]] == [0, 2, 4]
    assert all(p["verdict"].startswith("FAILED") for p in record["limitation_probes"])  # the stand-in mirrors the recorded failure


def test_rst_M1_untested_scope_claim_is_gone_and_the_limitation_is_shown_and_warned() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "not of small disparities in general" not in text
    md = "\n".join(c["source"] for c in _cells() if c["cell_type"] == "markdown")
    assert "known failure at 0–4 px" in md and "the cause has not been determined" in md
    byod = next(c["source"] for c in _cells() if c["source"].startswith("## 13."))
    assert "Near-zero disparity warning" in byod
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "rendered scenes with small disparities are read well" not in card


# ---- RST-m1 / m2 --------------------------------------------------------------------------------------------------


def test_rst_m1_no_stale_hosted_run_or_pin_statements() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "no recorded hosted run" not in text and "have not been measured yet" not in text
    assert "Section 3 says the checkpoint is not pinned" not in text
    prerequisites = next(c["source"] for c in _cells() if c["source"].startswith("## Prerequisites"))
    assert "Tesla T4, 5 October 2026" in prerequisites and "built in 68 s" in prerequisites and "adapt 31.9 s" in prerequisites


def test_rst_m2_sample_answers_match_the_recorded_run() -> None:
    md = "\n".join(c["source"] for c in _cells() if c["cell_type"] == "markdown")
    assert "modest fraction" not in md and "2.709" in md and "0.711" in md
    assert "the bad-pixel rates a little" not in md and "0.231 → 0.112" in md


# ---- RST-m3 -------------------------------------------------------------------------------------------------------


def test_rst_m3_activity_writes_beside_and_never_over_the_canonical_outputs(stages) -> None:
    run, root = stages
    run("prepare", "--n-pairs", 4, "--dataset-seed", 0, "--holdout", 0.25, "--seed", 1, "--epochs", 1)
    run("baselines", "--iters", 2)
    run("adapt", "--epochs", 1, "--batch-size", 2, "--lr", 2e-5, "--train-iters", 1, "--freeze-encoders", 1, "--seed", 1, "--iters", 2)
    for stage, options in (("evaluate", ("--iters", 2)), ("infer", ("--new-seed", 99, "--iters", 2)), ("reload", ())):
        run(stage, *options)
    out = root / "outputs"
    canonical = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in ("raft_stereo_adapter.safetensors", "raft_stereo_result.json", "raft_stereo_held_out_evaluation.json", "raft_stereo_new_pairs.csv")}
    # the Section 14 activity: FREEZE_ENCODERS = False, then Run after from Section 8
    run("adapt", "--epochs", 1, "--batch-size", 2, "--lr", 2e-5, "--train-iters", 1, "--freeze-encoders", 0, "--seed", 1, "--iters", 2)
    for stage, options in (("evaluate", ("--iters", 2)), ("infer", ("--new-seed", 99, "--iters", 2)), ("reload", ())):
        run(stage, *options)
    for name, digest in canonical.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest, f"the activity overwrote {name}"
    activity = out / "activity_unfrozen"
    assert b"trained" in (activity / "raft_stereo_adapter.safetensors").read_bytes()
    history = json.loads((activity / "raft_stereo_held_out_evaluation.json").read_text(encoding="utf-8"))["run_history"]
    assert [row["freeze_encoders"] for row in history] == [True, False]  # the run history keeps both
    assert json.loads((activity / "raft_stereo_result.json").read_text(encoding="utf-8"))["adaptation"]["freeze_encoders"] is False
    cells = "\n".join(c["source"] for c in _cells() if c["cell_type"] == "code")
    assert "ADAPT_OUT = '' if FREEZE_ENCODERS else 'activity_unfrozen/'" in cells and "load_record(ADAPT_OUT + 'raft_stereo_result.json')" in cells


# ---- RST-m4: the uv-template re-run fixes (idempotent Section 1, environment reuse, run_stage guard) ------------------


def _cells() -> list[dict]:
    return build.render(ROOT, TEMPLATE, "test-revision")["cells"]


def _check_cell() -> str:
    return next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: check the runtime"))


def _run_check_cell(namespace: dict, source: str | None = None) -> dict:
    exec(source or _check_cell(), namespace)  # noqa: S102 - the notebook's own cell
    return namespace


def test_rst_m4_section1_rerun_keeps_the_run_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=10**13))  # the disk check is not under test
    namespace = _run_check_cell({})
    first = namespace["ROOT"]
    assert first.is_dir() and first.parent == tmp_path / "outputs" / TEMPLATE["stem"]
    _run_check_cell(namespace)  # re-running Section 1 alone
    assert namespace["ROOT"] == first, "a Section 1 re-run must not strand the later cells in a new, empty run directory"
    _run_check_cell(namespace, _check_cell().replace("NEW_RUN_DIRECTORY = False", "NEW_RUN_DIRECTORY = True"))
    assert namespace["ROOT"] != first


def test_rst_m4_environment_is_keyed_on_the_lock_not_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=10**13))  # the disk check is not under test
    a = _run_check_cell({})
    b = _run_check_cell({})
    assert a["ROOT"] != b["ROOT"] and a["ENV_ROOT"] == b["ENV_ROOT"]
    assert a["ROOT"].name not in str(a["ENV_ROOT"])


def _fake_ipython(monkeypatch) -> None:
    display = types.ModuleType("IPython.display")
    display.Image = display.display = lambda *a, **k: None
    package = types.ModuleType("IPython")
    package.display = display
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display)


def test_rst_m4_install_cell_reuses_a_complete_environment_without_downloading(tmp_path: Path, monkeypatch) -> None:
    _fake_ipython(monkeypatch)
    install = next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: install the locked runtime"))
    env_root = tmp_path / "env"
    python = env_root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\necho '" + json.dumps({"python": "3.12.12", "torch": "x", "numpy": "x", "cuda": False}) + "'\n", encoding="utf-8")
    python.chmod(0o755)
    lock_sha = "a" * 64
    spec = {"lock_sha256": lock_sha, "python": TEMPLATE["managed_python"], "uv": TEMPLATE["uv"]["version"]}
    (env_root / "ready.json").write_text(json.dumps(spec), encoding="utf-8")

    def no_network(*_a, **_k):
        raise AssertionError("a matching environment must be reused, not rebuilt")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    namespace = {"ENV_ROOT": env_root, "ROOT": tmp_path, "CARRIED_HASHES": {TEMPLATE["lock"]: lock_sha}, "NOTEBOOK_SOURCE": {"revision": "r"}, "SESSION_START": time.perf_counter(), "Path": Path}
    exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert namespace["ENV_REUSED"] is True
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert name not in namespace["ENV"]
    assert namespace["ENV"]["MPLBACKEND"] == "Agg"
    namespace["CARRIED_HASHES"] = {TEMPLATE["lock"]: "b" * 64}  # a different lock is never reused
    with pytest.raises(AssertionError, match="must be reused"):
        exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert not (env_root / "ready.json").exists()


def test_rst_m4_run_stage_names_the_cells_to_rerun_when_the_run_directory_is_empty(tmp_path: Path) -> None:
    install = next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: install the locked runtime"))
    namespace = {"ROOT": tmp_path / "empty", "WEIGHTS": tmp_path / "w", "PYTHON": Path(sys.executable), "ENV": dict(os.environ)}
    exec(install[install.index("def run_stage(") : install.index("def load_record(")], namespace)  # noqa: S102
    with pytest.raises(RuntimeError, match=r"run the three Infrastructure cells again in order \(Sections 1, 2 and 3\)"):
        namespace["run_stage"]("prepare")
    troubleshooting = next(c["source"] for c in _cells() if "## Troubleshooting" in c["source"])
    assert "run the three Infrastructure cells again in order (Sections 1, 2 and 3)" in troubleshooting
