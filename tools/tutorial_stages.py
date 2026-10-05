"""Stage runner for the standalone RAFT-Stereo tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage demo [--iters 32]

Nothing is installed into the notebook kernel. Each stage is a separate process, so a stage starts from files only:
the verified checkpoint under ``--weights``, the carried upstream source, the dataset recorded by ``prepare``, the
adapter written by ``adapt`` and the JSON records of earlier stages. Learner-facing exports go to ``RUN_DIR/outputs``
(``RUN_DIR/outputs/byod`` for the BYOD namespace); hand-off state goes to ``RUN_DIR/state/<namespace>``. On failure a
stage writes ``RUN_DIR/state/<namespace>/<stage>.error.json`` with the exception type and message, which the notebook
re-raises in the kernel.

Stages: weights → demo → probes → prepare → baselines → adapt → evaluate → infer → reload, plus ``byod_pair``.
Every stage that needs a model builds it afresh from the verified checkpoint (or the exported adapter), so re-running
``adapt`` after a change always starts from the pretrained model.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

STEM = "raft_stereo"
DEMO_SEED = 7
DEMO_SGBM_RANGE = 64  # the sample contract's disparity range (at most 40 px) rounded up to a multiple of 16, plus margin
PROBE_SHIFTS = (8, 16)  # uniform random-dot shifts this checkpoint reads exactly; 0-4 px fail (docs/execution-evidence/2026-10-05/probe_shift_diagnostic.md)
N_NEW_PAIRS = 3
RELOAD_TOLERANCE_PX = {"mean_abs": 1e-3, "max_abs": 1e-2}


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources under ``root``, state and outputs per namespace, the checkpoint under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.namespace = options.namespace
        self.prefix = STEM if self.namespace == "sample" else "byod_" + STEM
        self.out = root / "outputs" if self.namespace == "sample" else root / "outputs" / "byod"
        self.state = root / "state" / self.namespace
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> Path:
        from raft_stereo_pipeline import MODEL_KEY

        return self.weights / MODEL_KEY

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2, default=_jsonable), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.write_text(json.dumps(value, indent=2, default=_jsonable), encoding="utf-8")
        return path


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def rounded(metrics: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    """The headline metrics of a metrics dictionary, rounded for printing."""
    keys = ("epe", "bad_1px", "bad_2px", "bad_3px", "d1_all")
    return {k: round(float(metrics[k]), digits) for k in keys if k in metrics}


def runtime_versions() -> dict[str, Any]:
    import cv2
    import numpy
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def records_digest(records: list[dict[str, Any]]) -> str:
    """SHA-256 over every record's id, image bytes, disparity and valid mask (the dataset identity)."""
    import numpy as np

    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.get("id")).encode("utf-8"))
        for key in ("left", "right"):
            digest.update(record[key].convert("RGB").tobytes())
        digest.update(np.ascontiguousarray(record["disparity"], dtype=np.float32).tobytes())
        valid = record.get("valid")
        if valid is not None:
            digest.update(np.ascontiguousarray(valid, dtype=bool).tobytes())
    return digest.hexdigest()


def strip(record: dict[str, Any]) -> dict[str, Any]:
    """A record's identity without its arrays (JSON-serialisable)."""
    return {"id": record.get("id"), "size": list(record["left"].size), **({"group": record["group"]} if "group" in record else {})}


# --------------------------------------------------------------------------------------------------
# model and data factories (the CPU pre-flight test replaces the model factories with random-weight stand-ins)
# --------------------------------------------------------------------------------------------------


def device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_base(run: Run) -> Any:
    """The pretrained model, from the verified checkpoint (verified again before loading)."""
    from raft_stereo_pipeline import RaftStereoPipeline

    return RaftStereoPipeline.from_pretrained(device=device(), weights_dir=run.snapshot())


def load_adapted(run: Run, artifact: Path) -> Any:
    """The adapted model rebuilt from files: verified checkpoint first, then the exported adapter."""
    from raft_stereo_pipeline import RaftStereoPipeline

    return RaftStereoPipeline.load_artifact(artifact, weights_dir=run.snapshot(), device=device())


def sample_records(n_pairs: int, seed: int) -> list[dict[str, Any]]:
    from raft_stereo_pipeline import stereo_dataset

    return stereo_dataset(n_pairs, seed=seed)


def load_data(run: Run, stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the split recorded by ``prepare`` from its source, and refuse if the records changed."""
    from raft_stereo_pipeline import read_stereo_records

    data = run.read_state("data.json", stage)
    records = read_stereo_records(data["byod_path"]) if data["source"] == "byod" else sample_records(data["n_pairs"], data["dataset_seed"])
    if records_digest(records) != data["dataset_digest"]:
        raise RuntimeError(f"the dataset changed since 'prepare'; re-run from the dataset cell (stage {stage!r})")
    by_id = {str(r["id"]): r for r in records}
    train = [by_id[i] for i in data["train_ids"]]
    held = [by_id[i] for i in data["held_out_ids"]]
    return train, held, data


def save_panel(run: Run, name: str, images: list[Any], labels: list[str], columns: int = 3) -> Path:
    from raft_stereo_pipeline.samples import panel

    path = run.out / name
    panel(images, labels, columns=columns).save(path)
    return path


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: verify the carried upstream source, install the carried manifest, fetch and verify the checkpoint."""
    from raft_stereo_pipeline import (
        MANIFEST_NAME,
        MODEL_ID,
        MODEL_KEY,
        MODEL_LICENSE,
        MODEL_REVISION,
        stage_missing_files,
        verify_snapshot,
        verify_upstream,
    )

    upstream = verify_upstream()
    print({"upstream": upstream["repository"], "commit": upstream["commit"], "carried_files_verified": upstream["files"]}, flush=True)
    carried = run.root / "weights" / MODEL_KEY / MANIFEST_NAME
    manifest = json.loads(carried.read_text(encoding="utf-8"))
    if (manifest["modelId"], manifest["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise RuntimeError("the carried manifest does not name the identity pinned by the package; regenerate the notebook")
    target = run.snapshot()
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(carried, target / MANIFEST_NAME)
    print({"model_id": MODEL_ID, "revision": MODEL_REVISION, "license": MODEL_LICENSE, "files": len(manifest["files"]), "total_bytes": manifest["totalBytes"]}, flush=True)
    fetched = stage_missing_files(target, allow_download=True)
    print({"weights_dir": str(target), "fetched": fetched}, flush=True)
    verified = verify_snapshot(target)
    print({"verified_files": verified["files"], "revision": verified["revision"]}, flush=True)
    run.write_output("weights.json", {"upstream": upstream, "checkpoint": verified, "fetched": fetched, "runtime": runtime_versions()})


def stage_demo(run: Run) -> None:
    """Section 4: validate the demonstration pair, run the pretrained model and SGBM, write the evaluation report."""
    import numpy as np

    from raft_stereo_pipeline import disparity_to_rgb, evaluation_report, render_stereo_scene, sgbm_disparity, validate_inputs
    from raft_stereo_pipeline.samples import error_to_rgb

    iters = run.options.iters
    pair = render_stereo_scene(DEMO_SEED)
    manifest = validate_inputs(pair["left"], pair["right"], iters=iters, name=pair["id"])
    try:
        validate_inputs(pair["left"], pair["right"].resize((160, 112)), iters=iters)
    except ValueError as exc:
        manifest["findings"].append({"probe": "right image of another size", "rejected": str(exc)})
    print({"pair": pair["id"], "size": manifest["inputs"][0]["size"], "iters": manifest["iters"], "padding": manifest["padding"], "verdict": manifest["verdict"]})
    print({"validation_probe": manifest["findings"][-1]})
    run.write_output(f"{STEM}_input_manifest.json", manifest)

    pipe = load_base(run)
    started = time.perf_counter()
    result = pipe.estimate(pair["left"], pair["right"], iters=iters)
    seconds = round(time.perf_counter() - started, 2)
    sgbm = sgbm_disparity(pair["left"], pair["right"], num_disparities=DEMO_SGBM_RANGE)
    report = evaluation_report(result, pair["disparity"], pair["valid"], baselines={"sgbm": sgbm["disparity"]}, sample_kind="synthetic")
    report["sgbm"] = {k: v for k, v in sgbm.items() if k != "disparity"}
    report["seconds"] = seconds
    report["device"] = pipe.device
    print({"device": pipe.device, "source": pipe.source, "seconds": seconds})
    print({"raft_stereo": rounded(report["metrics"]), "valid_pixels": report["metrics"]["valid_pixels"]})
    print({"sgbm": rounded(report["baselines"][0]), "sgbm_density": round(sgbm["density"], 4)})
    run.write_output(f"{STEM}_evaluation_report.json", report)
    error = np.abs(result["disparity"] - pair["disparity"])
    path = save_panel(
        run,
        f"{STEM}_demo.png",
        [pair["left"], pair["right"], disparity_to_rgb(pair["disparity"], vmax=40, valid=pair["valid"]), disparity_to_rgb(result["disparity"], vmax=40), disparity_to_rgb(sgbm["disparity"], vmax=40), error_to_rgb(error, pair["valid"])],
        ["left", "right", "ground truth (black = not scored)", f"RAFT-Stereo, {iters} iterations", "SGBM (filled)", "RAFT-Stereo |error| (red >= 3 px)"],
    )
    np.save(run.out / f"{STEM}_demo_disparity.npy", result["disparity"])
    print({"figure": str(path)})


def stage_probes(run: Run) -> None:
    """Section 5: two random-dot pairs whose answer is known exactly — shifted by 8 px and by 16 px."""
    import numpy as np

    from raft_stereo_pipeline import disparity_metrics, random_dot_pair

    iters = run.options.iters
    pipe = load_base(run)
    rows = []
    for shift in PROBE_SHIFTS:
        pair = random_dot_pair(shift)
        disparity = pipe.estimate(pair["left"], pair["right"], iters=iters)["disparity"]
        valid = pair["valid"]
        metrics = disparity_metrics(disparity, pair["disparity"], valid)
        row = {
            "probe": pair["id"],
            "true_disparity_px": shift,
            "predicted_median_px": round(float(np.median(disparity[valid])), 3),
            "predicted_p05_p95_px": [round(float(np.percentile(disparity[valid], q)), 3) for q in (5, 95)],
            **rounded(metrics),
        }
        rows.append(row)
        print(row)
    run.write_output("probes.json", {"iters": iters, "probes": rows})


def stage_prepare(run: Run) -> None:
    """Section 6: build the sample (or read the BYOD folder), validate, split, probe the refusals, record the split."""
    import numpy as np

    from raft_stereo_pipeline import read_stereo_records, split_dataset, validate_dataset

    opts = run.options
    if opts.byod:
        byod_path = Path(opts.byod).resolve()
        records = read_stereo_records(byod_path)
        source = "byod"
    else:
        byod_path = None
        records = sample_records(opts.n_pairs, opts.dataset_seed)
        source = "sample"
    manifest = validate_dataset(records, epochs=opts.epochs)
    print({"source": source if source == "sample" else f"byod ({byod_path.name})", **{k: v for k, v in manifest.items() if k not in ("findings", "verdict")}})
    for finding in manifest["findings"]:
        print({"finding": finding})
    train, held = split_dataset(records, holdout=opts.holdout, seed=opts.seed)
    train_ids, held_ids = [str(r["id"]) for r in train], [str(r["id"]) for r in held]
    overlap = sorted(set(train_ids) & set(held_ids))
    if overlap:
        raise RuntimeError(f"the split leaked records into both sides: {overlap[:5]}")
    groups = {"train": sorted({str(r.get("group", r["id"])) for r in train}), "held_out": sorted({str(r.get("group", r["id"])) for r in held})}
    if set(groups["train"]) & set(groups["held_out"]):
        raise RuntimeError("a group is on both sides of the split")
    print({"train": len(train), "held_out": len(held), "holdout": opts.holdout, "split_seed": opts.seed, "shared_ids": len(overlap)})

    probes = []
    first = records[0]
    broken = {
        "right image of another size": {**first, "right": first["right"].resize((first["right"].width // 2, first["right"].height))},
        "disparity of another shape": {**first, "disparity": np.zeros((10, 10), np.float32)},
        "negative disparity (left and right swapped)": {**first, "disparity": -np.asarray(first["disparity"]) - 1.0},
    }
    for name, record in broken.items():
        try:
            validate_dataset([record], epochs=opts.epochs)
            probes.append({"probe": name, "verdict": "accepted"})
        except (TypeError, ValueError) as exc:
            probes.append({"probe": name, "rejected": str(exc)[:160]})
        print(probes[-1])
    digest = records_digest(records)
    run.write_state(
        "data.json",
        {
            "source": source,
            "byod_path": str(byod_path) if byod_path else None,
            "n_pairs": opts.n_pairs,
            "dataset_seed": opts.dataset_seed,
            "holdout": opts.holdout,
            "split_seed": opts.seed,
            "epochs": opts.epochs,
            "dataset_digest": digest,
            "train_ids": train_ids,
            "held_out_ids": held_ids,
        },
    )
    run.write_output(
        f"{run.prefix}_dataset.json",
        {"source": source, "dataset": manifest, "dataset_sha256": digest, "split": {"train": [strip(r) for r in train], "held_out": [strip(r) for r in held], "seed": opts.seed, "holdout": opts.holdout}, "refusal_probes": probes},
    )
    print({"dataset_sha256": digest[:16] + "..."})


def stage_baselines(run: Run) -> None:
    """Section 7: three baselines on the held-out split — a median constant and SGBM (both configured on the training
    split only), and the pretrained RAFT-Stereo before any adaptation."""
    import numpy as np

    from raft_stereo_pipeline import aggregate_metrics, disparity_metrics, median_disparity, sgbm_disparity, sgbm_num_disparities
    from raft_stereo_pipeline.pipeline import _valid

    iters = run.options.iters
    train, held, data = load_data(run, "baselines")
    median = median_disparity(train)
    search = sgbm_num_disparities(train)
    median_rows, sgbm_rows, density = [], [], []
    for record in held:
        valid = _valid(record)
        truth = np.asarray(record["disparity"], np.float32)
        median_rows.append({"id": str(record["id"]), **disparity_metrics(np.full_like(truth, median), truth, valid)})
        sgbm = sgbm_disparity(record["left"], record["right"], num_disparities=search)
        density.append(sgbm["density"])
        sgbm_rows.append({"id": str(record["id"]), **disparity_metrics(sgbm["disparity"], truth, valid)})
    pipe = load_base(run)
    started = time.perf_counter()
    pretrained = pipe.evaluate(held, iters=iters)
    seconds = round(time.perf_counter() - started, 1)
    result = {
        "fitted_on": "training split only",
        "median_disparity_px": median,
        "sgbm_num_disparities": search,
        "median": {**aggregate_metrics(median_rows), "per_pair": median_rows},
        "sgbm": {**aggregate_metrics(sgbm_rows), "per_pair": sgbm_rows, "mean_density": float(np.mean(density))},
        "pretrained": pretrained,
        "iters": iters,
        "pretrained_seconds": seconds,
        "device": pipe.device,
    }
    print({"median_disparity_px": round(median, 3), "sgbm_num_disparities": search, "held_out_pairs": len(held), "iters": iters})
    for name in ("median", "sgbm", "pretrained"):
        print({name: rounded(result[name])})
    print({"sgbm_mean_density": round(result["sgbm"]["mean_density"], 4), "pretrained_seconds": seconds, "device": pipe.device})
    run.write_state("baselines.json", result)
    run.write_output(f"{run.prefix}_baselines.json", result)


def stage_adapt(run: Run) -> None:
    """Section 8: bounded fine-tuning from the pretrained checkpoint; record the in-memory reference; export the adapter."""
    import numpy as np

    opts = run.options
    train, held, data = load_data(run, "adapt")
    pipe = load_base(run)
    print({"start": "pretrained checkpoint, freshly loaded and verified", "source": pipe.source, "device": pipe.device}, flush=True)

    def report(row: dict[str, Any]) -> None:
        print({"epoch": f"{row['epoch']}/{row['epochs']}", "mean_loss": round(row["loss"], 4), "steps": row["steps"]}, flush=True)

    started = time.perf_counter()
    run_record = pipe.finetune(
        train,
        epochs=opts.epochs,
        batch_size=opts.batch_size,
        learning_rate=opts.lr,
        train_iters=opts.train_iters,
        seed=opts.seed,
        freeze_encoders=bool(opts.freeze_encoders),
        progress=report,
    )
    seconds = round(time.perf_counter() - started, 1)
    print({k: run_record[k] for k in ("freeze_encoders", "trainable_parameters", "total_parameters", "steps", "batch_size", "learning_rate", "train_iters", "precision")} | {"seconds": seconds})

    # The in-memory model's output on one held-out pair: the fresh-process reload in Section 11 must reproduce it.
    reference = held[0]
    np.save(run.state / "in_memory_disparity.npy", pipe.estimate(reference["left"], reference["right"], iters=opts.iters)["disparity"])
    in_memory = pipe.evaluate(held, iters=opts.iters)
    artifact = run.out / f"{run.prefix}_adapter.safetensors"
    descriptor = pipe.save_artifact(artifact, notes=f"RAFT-Stereo {data['source']} adaptation tutorial adapter")
    print({"artifact": artifact.name, "bytes": descriptor["bytes"], "tensors": descriptor["tensors"], "sha256": descriptor["sha256"][:16] + "...", "frozen_prefixes": descriptor["frozen_prefixes"]})
    record = {
        "finetune": run_record,
        "seconds": seconds,
        "artifact": descriptor,
        "in_memory": {"reference_pair": str(reference["id"]), "iters": opts.iters, "held_out_epe": in_memory["epe"], "held_out": rounded(in_memory, 6)},
    }
    run.write_state("adapt.json", record)
    run.write_output(f"{run.prefix}_adaptation.json", record)


def stage_evaluate(run: Run) -> None:
    """Section 9: a fresh process loads the exported adapter and scores it on the held-out split exactly as the
    baselines were scored; one row per evaluation is kept in the run history."""
    iters = run.options.iters
    train, held, data = load_data(run, "evaluate")
    baselines = run.read_state("baselines.json", "evaluate")
    adapted_state = run.read_state("adapt.json", "evaluate")
    if baselines["iters"] != iters:
        raise RuntimeError(f"the baselines were scored with {baselines['iters']} iterations, not {iters}; use the same VALID_ITERS or re-run the baselines cell")
    pipe = load_adapted(run, Path(adapted_state["artifact"]["path"]))
    adapted = pipe.evaluate(held, iters=iters)
    table = {name: rounded(baselines[name]) for name in ("median", "sgbm", "pretrained")}
    table["adapted"] = rounded(adapted)
    print({"held_out_pairs": len(held), "iters": iters, "loaded_from": "exported adapter, fresh process"})
    print(f"{'model':<12s}" + "".join(f"{k:>10s}" for k in ("epe", "bad_1px", "bad_2px", "bad_3px", "d1_all")))
    for name, row in table.items():
        print(f"{name:<12s}" + "".join(f"{row[k]:>10.4f}" for k in ("epe", "bad_1px", "bad_2px", "bad_3px", "d1_all")))
    better = sum(1 for a, b in zip(adapted["per_pair"], baselines["pretrained"]["per_pair"], strict=True) if a["epe"] < b["epe"])
    print({"pairs_where_adapted_epe_is_lower": f"{better}/{len(held)}"})
    history_path = run.state / "run_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    finetune = adapted_state["finetune"]
    history.append(
        {
            "freeze_encoders": finetune["freeze_encoders"],
            "epochs": finetune["epochs"],
            "learning_rate": finetune["learning_rate"],
            "trainable_parameters": finetune["trainable_parameters"],
            "final_loss": round(finetune["final_loss"], 4),
            "pretrained_epe": round(baselines["pretrained"]["epe"], 4),
            "adapted_epe": round(adapted["epe"], 4),
            "adapted_bad_3px": round(adapted["bad_3px"], 4),
            "train_seconds": adapted_state["seconds"],
        }
    )
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("run history (one row per evaluation in this session; row 0 is the first):")
    for index, row in enumerate(history):
        print(index, row)
    report = {
        "data_source": data["source"],
        "held_out_ids": data["held_out_ids"],
        "iters": iters,
        "baselines": {name: baselines[name] for name in ("median", "sgbm", "pretrained")},
        "baseline_configuration": {"median_disparity_px": baselines["median_disparity_px"], "sgbm_num_disparities": baselines["sgbm_num_disparities"], "fitted_on": baselines["fitted_on"]},
        "adapted": adapted,
        "table": table,
        "pairs_where_adapted_epe_is_lower": better,
        "run_history": history,
        "estimation": adapted["estimation"],
    }
    run.write_state("evaluate.json", {"adapted_epe": adapted["epe"], "table": table})
    run.write_output(f"{run.prefix}_held_out_evaluation.json", report)


def stage_infer(run: Run) -> None:
    """Section 10: the pretrained and the adapted model on pairs the adaptation never saw; disparity maps exported."""
    import numpy as np

    from raft_stereo_pipeline import disparity_metrics, disparity_to_rgb
    from raft_stereo_pipeline.pipeline import _valid
    from raft_stereo_pipeline.samples import write_disparity_png16

    opts = run.options
    train, held, data = load_data(run, "infer")
    adapted_state = run.read_state("adapt.json", "infer")
    if data["source"] == "byod":
        new_pairs, origin = held, "the BYOD held-out pairs (never used for training)"
    else:
        new_pairs, origin = sample_records(N_NEW_PAIRS, opts.new_seed), f"{N_NEW_PAIRS} new scenes from seed {opts.new_seed}, never part of the dataset"
    pretrained = load_base(run)
    adapted = load_adapted(run, Path(adapted_state["artifact"]["path"]))
    rows, tiles, labels = [], [], []
    disparity_dir = run.out / f"{run.prefix}_disparity"
    disparity_dir.mkdir(exist_ok=True)
    for record in new_pairs:
        valid = _valid(record)
        before = pretrained.estimate(record["left"], record["right"], iters=opts.iters)["disparity"]
        after = adapted.estimate(record["left"], record["right"], iters=opts.iters)["disparity"]
        name = Path(str(record["id"])).with_suffix("").as_posix().replace("/", "_")  # "scene2/im0.png" -> "scene2_im0"
        np.save(disparity_dir / f"{name}.npy", after)
        write_disparity_png16(disparity_dir / f"{name}.png", after)
        row = {
            "id": str(record["id"]),
            "pretrained": rounded(disparity_metrics(before, record["disparity"], valid)),
            "adapted": rounded(disparity_metrics(after, record["disparity"], valid)),
            "adapted_disparity_px": {"min": round(float(after.min()), 2), "median": round(float(np.median(after)), 2), "max": round(float(after.max()), 2)},
            "files": [f"{disparity_dir.name}/{name}.npy", f"{disparity_dir.name}/{name}.png"],
        }
        rows.append(row)
        print(row)
        if len(tiles) < 12:
            top = float(np.asarray(record["disparity"])[valid].max(initial=1.0))
            tiles += [record["left"], disparity_to_rgb(record["disparity"], vmax=top, valid=valid), disparity_to_rgb(before, vmax=top), disparity_to_rgb(after, vmax=top)]
            labels += [f"{name}: left", "ground truth", "pretrained", "adapted"]
    with open(run.out / f"{run.prefix}_new_pairs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair", "model", "epe", "bad_1px", "bad_2px", "bad_3px", "d1_all", "disparity_npy"])
        for row in rows:
            for model in ("pretrained", "adapted"):
                metrics = row[model]
                writer.writerow([row["id"], model, *(metrics[k] for k in ("epe", "bad_1px", "bad_2px", "bad_3px", "d1_all")), row["files"][0] if model == "adapted" else ""])
    lower = sum(1 for row in rows if row["adapted"]["epe"] < row["pretrained"]["epe"])
    print({"new_pairs": len(rows), "origin": origin, "adapted_epe_lower_on": f"{lower}/{len(rows)}"})
    figure = save_panel(run, f"{run.prefix}_new_pairs.png", tiles, labels, columns=4)
    print({"figure": str(figure), "csv": f"{run.prefix}_new_pairs.csv"})
    run.write_state("infer.json", {"origin": origin, "rows": rows})
    run.write_output(f"{run.prefix}_new_pairs.json", {"origin": origin, "iters": opts.iters, "rows": rows, "output_semantics": "disparity of the left image in pixels; .npy float32 (H, W); .png 16-bit KITTI convention (value / 256)"})


def stage_reload(run: Run) -> None:
    """Section 11: another fresh process rebuilds the adapted model from files and checks it reproduces the trained
    in-memory model within a stated tolerance; then the result record with provenance is written."""
    import numpy as np

    from raft_stereo_pipeline import (
        ARTIFACT_FORMAT,
        MODEL_ID,
        MODEL_KEY,
        MODEL_LICENSE,
        MODEL_REVISION,
        UPSTREAM_COMMIT,
        UPSTREAM_REPOSITORY,
        WEIGHTS_ARCHIVE_URL,
        WEIGHTS_FILE,
        RaftStereoPipeline,
    )

    train, held, data = load_data(run, "reload")
    adapted_state = run.read_state("adapt.json", "reload")
    evaluated = run.read_state("evaluate.json", "reload")
    artifact = Path(adapted_state["artifact"]["path"])
    metadata = RaftStereoPipeline.read_artifact_metadata(artifact)
    print({"artifact": artifact.name, "format": metadata["format"], "base": metadata["model_id"], "base_revision": metadata["model_revision"][:16], "frozen_prefixes": metadata["frozen_prefixes"]})
    reloaded = load_adapted(run, artifact)
    in_memory = adapted_state["in_memory"]
    reference = next(r for r in held if str(r["id"]) == in_memory["reference_pair"])
    before = np.load(run.state / "in_memory_disparity.npy")
    after = reloaded.estimate(reference["left"], reference["right"], iters=in_memory["iters"])["disparity"]
    difference = np.abs(before - after)
    parity = {
        "reference_pair": in_memory["reference_pair"],
        "mean_abs_diff_px": float(difference.mean()),
        "max_abs_diff_px": float(difference.max()),
        "tolerance_px": RELOAD_TOLERANCE_PX,
        "held_out_epe_in_memory": in_memory["held_out_epe"],
        "held_out_epe_fresh_process": evaluated["adapted_epe"],
        "compared_with": "the trained in-memory model of the 'adapt' process",
    }
    parity["equivalent"] = bool(
        parity["mean_abs_diff_px"] <= RELOAD_TOLERANCE_PX["mean_abs"]
        and parity["max_abs_diff_px"] <= RELOAD_TOLERANCE_PX["max_abs"]
        and abs(in_memory["held_out_epe"] - evaluated["adapted_epe"]) <= RELOAD_TOLERANCE_PX["mean_abs"]
    )
    print({"reload_parity": {k: (round(v, 8) if isinstance(v, float) else v) for k, v in parity.items()}})
    if not parity["equivalent"]:
        raise AssertionError(f"the reloaded adapter does not reproduce the in-memory model within tolerance: {parity}")

    source_path = run.root / "source.json"
    notebook_source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    weights_path = run.root / "outputs" / "weights.json"
    weights = json.loads(weights_path.read_text(encoding="utf-8")) if weights_path.is_file() else None
    result = {
        "notebook_source": notebook_source,
        "repository_revision": notebook_source["revision"] if notebook_source else None,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "license": MODEL_LICENSE, "checkpoint_file": WEIGHTS_FILE, "archive_url": WEIGHTS_ARCHIVE_URL, "device": reloaded.device, "precision": "float32"},
        "upstream_code": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT, "carried": True, "remote_code_fetched": False},
        "checkpoint_verification": weights["checkpoint"] if weights else None,
        "runtime": {**runtime_versions(), "environment": "isolated hash-locked environment (one process per stage)"},
        "data_source": data["source"],
        "dataset_sha256": data["dataset_digest"],
        "split": {"train": len(train), "held_out": len(held), "seed": data["split_seed"], "holdout": data["holdout"]},
        "adaptation": adapted_state["finetune"],
        "held_out_table": evaluated["table"],
        "artifact": {**adapted_state["artifact"], "format": ARTIFACT_FORMAT},
        "reload_parity": parity,
        "new_pairs": run.read_state("infer.json", "reload") if (run.state / "infer.json").is_file() else None,
        "evidence_class": "tutorial/sanity evidence on the stated sample; not a benchmark",
    }
    run.write_output(f"{run.prefix}_result.json", result)
    print(f"{run.out.relative_to(run.root).as_posix()}/:")
    for path in sorted(run.out.rglob("*")):
        if path.is_file() and (run.namespace != "sample" or "byod" not in path.parts):
            print(f"  - {path.relative_to(run.out).as_posix()} ({path.stat().st_size / 1024:.1f} KB)")


def stage_byod_pair(run: Run) -> None:
    """Section 13 (optional): your own rectified pair, through the same validation, model, baseline and report."""
    import numpy as np
    from PIL import Image

    from raft_stereo_pipeline import (
        MAX_DISPARITY,
        disparity_to_rgb,
        evaluation_report,
        read_disparity,
        sgbm_disparity,
        validate_inputs,
    )
    from raft_stereo_pipeline.samples import write_disparity_png16

    opts = run.options
    images = []
    for path in (opts.left, opts.right):
        with Image.open(path) as handle:
            images.append(handle.convert("RGB"))
    left, right = images
    manifest = validate_inputs(left, right, iters=opts.iters, name=Path(opts.left).name)
    print({"left": Path(opts.left).name, "right": Path(opts.right).name, "size": manifest["inputs"][0]["size"], "padding": manifest["padding"], "verdict": manifest["verdict"]})
    truth = valid = None
    if opts.disparity:
        truth, valid = read_disparity(opts.disparity)
        if truth.shape != (left.height, left.width):
            raise ValueError(f"ground-truth disparity shape {truth.shape} != image shape ({left.height}, {left.width})")
    pipe = load_base(run)
    result = pipe.estimate(left, right, iters=opts.iters)
    top = float(truth[valid].max()) if truth is not None and valid.any() else float(np.percentile(result["disparity"], 99))
    search = int(min(MAX_DISPARITY, 16 * np.ceil((top + 8.0) / 16.0)))
    sgbm = sgbm_disparity(left, right, num_disparities=max(16, search))
    report = evaluation_report(result, truth, valid, baselines={"sgbm": sgbm["disparity"]} if truth is not None else None, sample_kind="byod")
    report["sgbm"] = {k: v for k, v in sgbm.items() if k != "disparity"}
    report["input_manifest"] = manifest
    print({"verdict": report["verdict"], "raft_stereo": rounded(report["metrics"]) if report["metrics"] else None, "baselines": [rounded(b) | {"id": b["id"]} for b in report["baselines"]]})
    np.save(run.out / "byod_pair_disparity.npy", result["disparity"])
    write_disparity_png16(run.out / "byod_pair_disparity.png", result["disparity"])
    run.write_output("byod_pair_report.json", report)
    tiles = [left, right, disparity_to_rgb(result["disparity"], vmax=top), disparity_to_rgb(sgbm["disparity"], vmax=top)]
    labels = ["left", "right", "RAFT-Stereo (pretrained)", "SGBM (filled)"]
    if truth is not None:
        tiles.append(disparity_to_rgb(truth, vmax=top, valid=valid))
        labels.append("ground truth")
    path = save_panel(run, "byod_pair.png", tiles, labels, columns=3)
    print({"figure": str(path), "files": ["byod_pair_disparity.npy", "byod_pair_disparity.png", "byod_pair_report.json"]})


STAGES = {
    "weights": stage_weights,
    "demo": stage_demo,
    "probes": stage_probes,
    "prepare": stage_prepare,
    "baselines": stage_baselines,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "infer": stage_infer,
    "reload": stage_reload,
    "byod_pair": stage_byod_pair,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the staged checkpoint")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--namespace", choices=("sample", "byod"), default="sample", help="state/outputs namespace (BYOD results never overwrite the sample's)")
    parser.add_argument("--iters", type=int, default=32, help="refinement iterations at inference and evaluation")
    parser.add_argument("--byod", default="", help="prepare: a BYOD folder with pairs.json instead of the sample")
    parser.add_argument("--n-pairs", type=int, default=32, help="prepare: sample pairs to render")
    parser.add_argument("--dataset-seed", type=int, default=0, help="prepare: sample dataset seed")
    parser.add_argument("--holdout", type=float, default=0.25, help="prepare: held-out share")
    parser.add_argument("--seed", type=int, default=20261004, help="prepare/adapt: split and training seed")
    parser.add_argument("--epochs", type=int, default=3, help="prepare/adapt: training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="adapt: pairs per optimiser step")
    parser.add_argument("--lr", type=float, default=2e-5, help="adapt: AdamW learning rate")
    parser.add_argument("--train-iters", type=int, default=12, help="adapt: refinement iterations while training")
    parser.add_argument("--freeze-encoders", type=int, choices=(0, 1), default=1, help="adapt: 1 keeps the feature and context encoders frozen")
    parser.add_argument("--new-seed", type=int, default=99, help="infer: seed of the unseen sample scenes")
    parser.add_argument("--left", default="", help="byod_pair: left image")
    parser.add_argument("--right", default="", help="byod_pair: right image")
    parser.add_argument("--disparity", default="", help="byod_pair: optional ground-truth disparity (.pfm, .npy, 16-bit .png)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    # One error file per stage name, whatever the namespace: the notebook's run_stage reads it to re-raise the message.
    error_file = root / "state" / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "namespace": options.namespace, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "namespace": options.namespace, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
