"""Package-level contracts that need no model weights: metrics, baselines, validation, splitting, BYOD reading,
the carried-upstream check, archive-member extraction and checkpoint staging."""

import hashlib
import io
import json
import shutil
import sys
import types
import zipfile

import numpy as np
import pytest
from PIL import Image

from conftest import tiny_records
from raft_stereo_pipeline import pipeline as P
from raft_stereo_pipeline.samples import write_disparity_png16, write_pfm

# --------------------------------------------------------------------------- metrics and baselines


def test_disparity_metrics_definitions():
    truth = np.full((2, 4), 10.0)
    predicted = truth + np.array([[0.5, 1.5, 2.5, 3.5], [0.0, 0.0, 0.0, -4.0]])
    metrics = P.disparity_metrics(predicted, truth)
    assert metrics["epe"] == pytest.approx(np.abs(predicted - truth).mean())
    assert metrics["bad_1px"] == pytest.approx(4 / 8)
    assert metrics["bad_2px"] == pytest.approx(3 / 8)
    assert metrics["bad_3px"] == pytest.approx(2 / 8)
    assert metrics["d1_all"] == pytest.approx(2 / 8)  # > 3 px and > 5 % of 10 px
    assert metrics["valid_pixels"] == 8


def test_metrics_score_only_valid_pixels_below_the_ceiling():
    truth = np.array([[1.0, 2.0, 600.0]])
    predicted = np.array([[1.0, 9.0, 0.0]])
    valid = np.array([[True, False, True]])
    metrics = P.disparity_metrics(predicted, truth, valid)
    assert metrics["valid_pixels"] == 1 and metrics["epe"] == 0.0
    with pytest.raises(ValueError, match="no valid pixel"):
        P.disparity_metrics(predicted, truth, np.zeros_like(valid))
    with pytest.raises(ValueError, match="shape"):
        P.disparity_metrics(predicted[:, :2], truth)


def test_aggregate_is_the_pair_mean_with_pixel_weighted_epe():
    rows = [{**dict.fromkeys(P.METRIC_KEYS, 1.0), "valid_pixels": 1}, {**dict.fromkeys(P.METRIC_KEYS, 3.0), "valid_pixels": 3}]
    out = P.aggregate_metrics(rows)
    assert out["epe"] == 2.0 and out["pixel_weighted_epe"] == pytest.approx(2.5) and out["n_pairs"] == 2


def test_median_baseline_and_sgbm_range_are_fitted_on_the_given_records():
    records = tiny_records(3)
    pooled = np.concatenate([r["disparity"][r["valid"]] for r in records])
    assert P.median_disparity(records) == pytest.approx(float(np.median(pooled)))
    search = P.sgbm_num_disparities(records)
    assert search % 16 == 0 and search >= pooled.max() + 8


def test_sgbm_baseline_fills_every_pixel_and_reports_density():
    from raft_stereo_pipeline import render_stereo_scene

    record = render_stereo_scene(5)
    out = P.sgbm_disparity(record["left"], record["right"], num_disparities=64)
    assert out["disparity"].shape == record["disparity"].shape and np.isfinite(out["disparity"]).all()
    assert 0.5 < out["density"] <= 1.0
    assert P.disparity_metrics(out["disparity"], record["disparity"], record["valid"])["epe"] < 3.0
    with pytest.raises(ValueError, match="multiple of 16"):
        P.sgbm_disparity(record["left"], record["right"], num_disparities=20)


# --------------------------------------------------------------------------- validation and splitting


def test_validate_inputs_reports_padding_and_refuses_bad_pairs():
    left = Image.new("RGB", (100, 70))
    manifest = P.validate_inputs(left, left.copy(), iters=8)
    assert manifest["verdict"] == "accepted" and manifest["padding"] == [28, 26] and manifest["findings"]
    with pytest.raises(ValueError, match="differ in size"):
        P.validate_inputs(left, Image.new("RGB", (90, 70)))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        P.validate_pair(Image.new("RGB", (40, 70)), Image.new("RGB", (40, 70)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        P.validate_pair(Image.new("RGB", (1300, 70)), Image.new("RGB", (1300, 70)))
    with pytest.raises(TypeError, match="PIL"):
        P.validate_pair(np.zeros((70, 100, 3)), left)
    with pytest.raises(ValueError, match="iters"):
        P.validate_inputs(left, left.copy(), iters=0)


def test_validate_dataset_names_the_broken_rule():
    records = tiny_records(2)
    manifest = P.validate_dataset(records, epochs=2)
    assert manifest["n_records"] == 2 and manifest["verdict"] == "accepted"
    first = records[0]
    with pytest.raises(ValueError, match="disparity shape"):
        P.validate_dataset([{**first, "disparity": np.zeros((3, 3))}])
    with pytest.raises(ValueError, match="negative disparity"):
        P.validate_dataset([{**first, "disparity": -first["disparity"] - 1}])
    with pytest.raises(ValueError, match="no pixel has valid ground truth"):
        P.validate_dataset([{**first, "valid": np.zeros_like(first["valid"])}])
    with pytest.raises(ValueError, match="duplicate record ids"):
        P.validate_dataset([first, first])
    with pytest.raises(ValueError, match="'left', 'right' and 'disparity'"):
        P.validate_dataset([{"left": first["left"]}])
    with pytest.raises(ValueError, match="epochs"):
        P.validate_dataset(records, epochs=0)


def test_split_is_seeded_disjoint_and_keeps_groups_together():
    records = tiny_records(8)
    train, held = P.split_dataset(records, holdout=0.25, seed=3)
    assert len(train) == 6 and len(held) == 2
    assert not {r["id"] for r in train} & {r["id"] for r in held}
    assert [r["id"] for r in P.split_dataset(records, holdout=0.25, seed=3)[1]] == [r["id"] for r in held]
    grouped = [{**r, "group": f"scene-{i // 2}"} for i, r in enumerate(records)]
    train, held = P.split_dataset(grouped, holdout=0.25, seed=3)
    assert not {r["group"] for r in train} & {r["group"] for r in held}
    with pytest.raises(ValueError, match="at least 2 independent groups"):
        P.split_dataset([{**r, "group": "one"} for r in records])


def _byod_folder(root, records, *, disparity_format="pfm"):
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, record in enumerate(records):
        record["left"].save(root / f"p{i}_left.png")
        record["right"].save(root / f"p{i}_right.png")
        name = f"p{i}_disp.{disparity_format}"
        values = np.where(record["valid"], record["disparity"], np.inf if disparity_format == "pfm" else 0.0).astype(np.float32)
        if disparity_format == "pfm":
            write_pfm(root / name, values)
        elif disparity_format == "npy":
            np.save(root / name, np.where(record["valid"], record["disparity"], np.nan))
        else:
            write_disparity_png16(root / name, values)
            raw = np.asarray(Image.open(root / name)).copy()
            raw[~record["valid"]] = 0
            Image.fromarray(raw).save(root / name)
        entries.append({"left": f"p{i}_left.png", "right": f"p{i}_right.png", "disparity": name, "group": f"g{i}"})
    (root / "pairs.json").write_text(json.dumps(entries), encoding="utf-8")
    return root


@pytest.mark.parametrize("fmt", ["pfm", "npy", "png"])
def test_read_stereo_records_round_trips_every_disparity_format(tmp_path, fmt):
    records = tiny_records(2)
    read = P.read_stereo_records(_byod_folder(tmp_path / fmt, records, disparity_format=fmt))
    assert [r["group"] for r in read] == ["g0", "g1"]
    for original, loaded in zip(records, read, strict=True):
        assert np.array_equal(loaded["valid"], original["valid"])
        tolerance = 1 / 256 if fmt == "png" else 1e-6
        assert np.allclose(loaded["disparity"][loaded["valid"]], original["disparity"][original["valid"]], atol=tolerance)
    P.validate_dataset(read)


def test_read_stereo_records_refuses_unsafe_or_missing_files(tmp_path):
    root = _byod_folder(tmp_path / "d", tiny_records(2))
    entries = json.loads((root / "pairs.json").read_text())
    for bad, message in (("../outside.png", "relative"), ("/etc/passwd", "relative"), ("absent.png", "missing")):
        (root / "pairs.json").write_text(json.dumps([{**entries[0], "left": bad}]))
        with pytest.raises((ValueError, FileNotFoundError), match=message):
            P.read_stereo_records(root)
    (root / "pairs.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        P.read_stereo_records(root)
    (root / "pairs.json").write_text(json.dumps([{"left": "p0_left.png"}]))
    with pytest.raises(ValueError, match="must have 'left', 'right' and 'disparity'"):
        P.read_stereo_records(root)
    with pytest.raises(FileNotFoundError, match="pairs.json"):
        P.read_stereo_records(tmp_path / "empty")


def test_evaluation_report_without_ground_truth_is_not_measurable():
    report = P.evaluation_report({"disparity": np.ones((4, 4), np.float32), "iters": 4})
    assert report["verdict"] == "not-measurable" and report["metrics"] == {}
    truth = np.ones((4, 4))
    report = P.evaluation_report({"disparity": truth}, truth, baselines={"zero": np.zeros((4, 4))})
    assert report["verdict"] == "sample-sanity" and report["baselines"][0]["epe"] == 1.0


# --------------------------------------------------------------------------- carried upstream source


def test_carried_upstream_matches_its_recorded_digests():
    info = P.verify_upstream()
    assert info["files"] == len(P.UPSTREAM_SHA256) == 8 and info["commit"] == P.UPSTREAM_COMMIT


def test_tampered_upstream_file_is_refused(tmp_path):
    copy = tmp_path / "raft_stereo"
    shutil.copytree(P.DEFAULT_UPSTREAM_DIR, copy)
    (copy / "core" / "update.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="core/update.py"):
        P.verify_upstream(copy)
    (copy / "core" / "corr.py").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        P.verify_upstream(copy)


def test_a_foreign_core_module_is_refused(monkeypatch, tmp_path):
    foreign = types.ModuleType("core")
    foreign.__file__ = str(tmp_path / "core" / "__init__.py")
    monkeypatch.setitem(sys.modules, "core", foreign)
    with pytest.raises(RuntimeError, match="different module named 'core'"):
        P.load_upstream()


# --------------------------------------------------------------------------- archive member and staging


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return path


def test_extract_member_takes_only_the_named_file(tmp_path):
    archive = _zip(tmp_path / "m.zip", {"models/raftstereo-middlebury.pth": b"weights", "models/other.pth": b"x"})
    target = tmp_path / "out.pth"
    assert P.extract_member(archive, P.WEIGHTS_FILE, target) == "models/raftstereo-middlebury.pth"
    assert target.read_bytes() == b"weights"


@pytest.mark.parametrize(
    ("members", "message"),
    [
        ({"a/raftstereo-middlebury.pth": b"1", "b/raftstereo-middlebury.pth": b"2"}, "exactly one"),
        ({"other.pth": b"1"}, "exactly one"),
        ({"../raftstereo-middlebury.pth": b"1"}, "unsafe member path"),
    ],
)
def test_extract_member_refuses_ambiguous_or_unsafe_archives(tmp_path, members, message):
    archive = _zip(tmp_path / "m.zip", members)
    with pytest.raises(ValueError, match=message):
        P.extract_member(archive, P.WEIGHTS_FILE, tmp_path / "out.pth")


def test_extract_member_refuses_symlinks_and_oversized_members(tmp_path):
    archive = tmp_path / "s.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        info = zipfile.ZipInfo("raftstereo-middlebury.pth")
        info.external_attr = (0o120777 << 16)
        bundle.writestr(info, "target")
    with pytest.raises(ValueError, match="symbolic link"):
        P.extract_member(archive, P.WEIGHTS_FILE, tmp_path / "out.pth")
    big = _zip(tmp_path / "b.zip", {"raftstereo-middlebury.pth": b"x" * 100})
    with pytest.raises(ValueError, match="expands to"):
        P.extract_member(big, P.WEIGHTS_FILE, tmp_path / "out.pth", max_bytes=10)


def _manifest(root, revision, payload, *, tamper=False):
    entry = {
        "path": P.WEIGHTS_FILE,
        "url": P.WEIGHTS_ARCHIVE_URL,
        "archiveMember": P.WEIGHTS_FILE,
        "bytes": len(payload),
        "sha256": "0" * 64 if tamper else hashlib.sha256(payload).hexdigest(),
    }
    manifest = {"modelId": P.MODEL_ID, "revision": revision, "files": [entry], "totalBytes": len(payload)}
    root.mkdir(parents=True, exist_ok=True)
    (root / P.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_stage_missing_files_fetches_through_the_archive_and_verify_checks_it(tmp_path, pinned):
    payload = b"checkpoint-bytes"
    _manifest(tmp_path, pinned, payload)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as bundle:
        bundle.writestr("models/raftstereo-middlebury.pth", payload)

    def fake_download(url, target, **_):
        assert url == P.WEIGHTS_ARCHIVE_URL
        target.write_bytes(archive_bytes.getvalue())
        return len(archive_bytes.getvalue())

    with pytest.raises(FileNotFoundError, match="allow_download"):
        P.stage_missing_files(tmp_path)
    original = P.download_archive
    try:
        P.download_archive = fake_download
        assert P.stage_missing_files(tmp_path, allow_download=True) == [P.WEIGHTS_FILE]
    finally:
        P.download_archive = original
    assert (tmp_path / P.WEIGHTS_FILE).read_bytes() == payload
    assert not list(tmp_path.glob("*.archive.zip")), "the archive is deleted after extraction"
    assert P.stage_missing_files(tmp_path, allow_download=True) == []
    assert P.verify_snapshot(tmp_path)["files"] == 1


def test_verify_snapshot_names_size_and_digest_mismatches(tmp_path, pinned):
    payload = b"checkpoint-bytes"
    _manifest(tmp_path, pinned, payload, tamper=True)
    (tmp_path / P.WEIGHTS_FILE).write_bytes(payload)
    with pytest.raises(ValueError, match="sha256"):
        P.verify_snapshot(tmp_path)
    (tmp_path / P.WEIGHTS_FILE).write_bytes(payload + b"!")
    with pytest.raises(ValueError, match="size"):
        P.verify_snapshot(tmp_path)


def test_manifest_naming_another_checkpoint_is_refused(tmp_path, pinned):
    _manifest(tmp_path, "f" * 64, b"x")
    with pytest.raises(ValueError, match="package pins"):
        P.stage_missing_files(tmp_path, allow_download=True)


def test_committed_manifest_matches_the_package():
    manifest = json.loads((P.Path(__file__).resolve().parents[1] / "weights" / P.MODEL_KEY / P.MANIFEST_NAME).read_text())
    assert manifest["modelId"] == P.MODEL_ID and manifest["revision"] == P.MODEL_REVISION
    [entry] = manifest["files"]
    assert entry["url"] == P.WEIGHTS_ARCHIVE_URL and entry["path"] == P.WEIGHTS_FILE
    if not P.is_pinned():
        assert entry["sha256"] is None and entry["bytes"] is None and manifest["totalBytes"] is None
