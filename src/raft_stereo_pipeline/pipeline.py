"""Dense stereo disparity estimation and bounded stereo fine-tuning with RAFT-Stereo (Middlebury checkpoint).

RAFT-Stereo is not published as a Python package. The model code is the upstream ``core/`` package of
``princeton-vl/RAFT-Stereo`` at commit ``UPSTREAM_COMMIT``, carried verbatim under ``third_party/raft_stereo/``
(MIT licence). ``load_upstream`` refuses to import it unless every file matches ``UPSTREAM_SHA256``. The pure-PyTorch
correlation (``corr_implementation = "reg"``) is used, so the optional CUDA sampler extension is never needed and
the model runs on CPU and on any CUDA GPU.

The checkpoint is ``raftstereo-middlebury.pth`` from upstream's ``models.zip`` (``download_models.sh``). It is a
pickle-based PyTorch ``state_dict``; it is loaded only after its byte size and SHA-256 match the manifest, and only
with ``torch.load(..., weights_only=True)``, which refuses arbitrary pickled objects. Until
``tools/pin_snapshot.py`` has recorded the checkpoint's SHA-256 and byte size, the package refuses to stage, verify
or load the checkpoint: an unpinned checkpoint is never trusted.
"""

# ruff: noqa: E501  -- refusal messages and docstrings name the rule in full; they are kept on one line
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import urllib.request
import warnings
import zipfile
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .samples import read_disparity

MODEL_ID = "princeton-vl/RAFT-Stereo/raftstereo-middlebury"
# A URL-hosted checkpoint has no commit, so its pinned identity is the SHA-256 of the checkpoint's own bytes
# (the .pth file inside the archive, not the archive). "unpinned" until tools/pin_snapshot.py records it.
MODEL_REVISION = "d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819"
MODEL_LICENSE = "mit"
MODEL_KEY = "raftstereo-middlebury"
UNPINNED = "unpinned"
PIN_COMMAND = "python tools/pin_snapshot.py"
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "raft-stereo-adapter-v1"

# The upstream source tree carried verbatim, and the SHA-256 of every carried file at UPSTREAM_COMMIT.
UPSTREAM_REPOSITORY = "princeton-vl/RAFT-Stereo"
UPSTREAM_COMMIT = "6e93ed2169bd858dbb43033988563f3b0bb49506"
UPSTREAM_LICENSE = "mit"
DEFAULT_UPSTREAM_DIR = Path(__file__).resolve().parent / "third_party" / "raft_stereo"
UPSTREAM_SHA256 = {
    "core/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "core/raft_stereo.py": "107d836e37d341e4ca6f93446a448d2c3626fefb3145b1ca8ce5768fea55257a",
    "core/corr.py": "cf3a9dacbfbfd28656f793a9857d066c7ce2a05fbceb3a3f3206522649c7be19",
    "core/extractor.py": "7e384e51f70f705f8f46efc042afa5fc6fb848e07cf0314f2c9ec6a38a026972",
    "core/update.py": "e3d8a9b402f58109cf7e4990d8a14d0d2a675b0b1f108edc42ee085e57133ec9",
    "core/utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "core/utils/utils.py": "65968f7bab52e4744dc36b7fab1d56b47a91cc0180b6f594387d730677d602c1",
    "LICENSE": "f134ad9e8937e5b9571867acea371a53d1f3d4ff6babdbcddd8f6e554ad467aa",
}

# The checkpoint: one member of upstream's models.zip (the archive URL of download_models.sh at UPSTREAM_COMMIT).
WEIGHTS_FILE = "raftstereo-middlebury.pth"
WEIGHTS_ARCHIVE_URL = (
    "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1"
)
MAX_ARCHIVE_BYTES = 2 * 1024**3  # refuse a larger download outright
MAX_MEMBER_BYTES = 512 * 1024**2  # the checkpoint itself is about 45 MB
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY

# The architecture of the Middlebury checkpoint: upstream demo.py defaults (the Middlebury demo command passes no
# architecture flag). corr_implementation "reg" is upstream's default, pure-PyTorch correlation.
MODEL_ARGS: dict[str, Any] = {
    "hidden_dims": [128, 128, 128],
    "corr_implementation": "reg",
    "shared_backbone": False,
    "corr_levels": 4,
    "corr_radius": 4,
    "n_downsample": 2,
    "context_norm": "batch",
    "slow_fast_gru": False,
    "n_gru_layers": 3,
    "mixed_precision": False,
}
EXPECTED_PARAMETERS = 11_116_176  # parameters of that architecture (counted from the carried code)
EXPECTED_STATE_TENSORS = 337  # state_dict entries, BatchNorm buffers included

# Operational ceilings. The "reg" correlation volume holds H/4 * W/4 * W/4 values per level, so memory grows with
# width squared times height; inference is padded to a multiple of 32 px as upstream's demo and evaluation do.
MIN_IMAGE_SIDE = 64
MAX_IMAGE_SIDE = 1280
MAX_PIXELS = 1280 * 1024
MAX_DISPARITY = 512.0  # upstream treats ground truth at or above 512 px as invalid
MAX_RECORDS = 2000
PAD_DIVISOR = 32
DEFAULT_VALID_ITERS = 32  # upstream demo.py / evaluate_stereo.py default
MAX_ITERS = 64

# Bounded fine-tuning defaults. Optimiser and loss follow upstream train_stereo.py (AdamW eps 1e-8, weight decay
# 1e-5, sequence loss gamma 0.9 adjusted for the iteration count, ground truth >= 700 px ignored, gradient-norm clip
# 1.0, BatchNorm frozen); the learning rate is upstream's Middlebury fine-tuning value. Upstream's OneCycle schedule,
# augmentation and mixed precision are not used: the tutorial runs a constant learning rate in float32.
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_TRAIN_ITERS = 12
DEFAULT_GAMMA = 0.9
LOSS_MAX_FLOW = 700.0
GRAD_CLIP = 1.0
DEFAULT_SEED = 20261004
TRAIN_CROP = (320, 448)  # (height, width): larger training pairs are cropped at a seeded random position
ENCODER_PREFIXES = ("fnet.", "cnet.")  # feature encoder and context encoder

# Metric thresholds (pixels). "bad-t" is the share of valid pixels whose absolute disparity error exceeds t px:
# bad-1 is ETH3D's reported outlier rate, bad-2 Middlebury's, bad-3 the threshold of KITTI's D1.
BAD_THRESHOLDS = (1.0, 2.0, 3.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pinned() -> bool:
    """True once MODEL_REVISION is the checkpoint's 64-hex SHA-256."""
    revision = MODEL_REVISION
    return len(revision) == 64 and all(c in "0123456789abcdef" for c in revision)


def _require_pinned(action: str) -> None:
    if not is_pinned():
        raise RuntimeError(
            f"refusing to {action}: {MODEL_ID} has no pinned checkpoint digest yet (MODEL_REVISION = "
            f"{MODEL_REVISION!r}); run `{PIN_COMMAND}` where the upstream archive is reachable to record the "
            "SHA-256 and byte size, then regenerate the notebook"
        )


# --------------------------------------------------------------------------- carried upstream source


def verify_upstream(path: str | Path | None = None) -> dict[str, Any]:
    """Check every carried upstream file against UPSTREAM_SHA256; raise naming the first missing or changed file."""
    root = Path(path) if path is not None else DEFAULT_UPSTREAM_DIR
    for name, expected in UPSTREAM_SHA256.items():
        file_path = root / name
        if not file_path.is_file():
            raise FileNotFoundError(f"carried upstream file missing: {file_path}")
        digest = _sha256(file_path)
        if digest != expected:
            raise ValueError(
                f"carried upstream file {name}: sha256 {digest} != {expected} (RAFT-Stereo @ {UPSTREAM_COMMIT[:12]}); "
                "the carried source was changed — restore it from the repository"
            )
    return {"path": str(root), "repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT, "files": len(UPSTREAM_SHA256)}


def load_upstream(path: str | Path | None = None) -> Any:
    """Verify, then import the carried upstream ``core.raft_stereo`` module and return it.

    Upstream imports itself as the top-level package ``core``; the carried directory is put first on ``sys.path``
    and the imported module must come from it, so a different ``core`` package can never stand in for it.
    """
    root = Path(path) if path is not None else DEFAULT_UPSTREAM_DIR
    verify_upstream(root)
    root = root.resolve()
    loaded = sys.modules.get("core")
    if loaded is not None:
        origin = Path(getattr(loaded, "__file__", "") or "").resolve()
        if root not in origin.parents:
            raise RuntimeError(f"a different module named 'core' is already imported ({origin}); restart the process")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Two known, harmless deprecation warnings from the unmodified upstream code, filtered by exact message only:
    # torch.cuda.amp.autocast (mixed precision is off here) and torch.meshgrid without `indexing` ("ij" is the default).
    warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r"torch\.meshgrid: in an upcoming release", category=UserWarning)
    module = importlib.import_module("core.raft_stereo")
    if root not in Path(module.__file__).resolve().parents:
        raise RuntimeError(f"core.raft_stereo was imported from {module.__file__}, not from {root}")
    return module


def build_model(upstream_dir: str | Path | None = None) -> Any:
    """The upstream RAFTStereo architecture of the Middlebury checkpoint, randomly initialised (nothing downloaded)."""
    module = load_upstream(upstream_dir)
    return module.RAFTStereo(Namespace(**MODEL_ARGS))


# --------------------------------------------------------------------------- checkpoint manifest and staging


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def _check_manifest_identity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}"
        )


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the local checkpoint against its DIMER manifest; raise naming the first mismatch."""
    _require_pinned("verify the checkpoint")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    _check_manifest_identity(manifest)
    for entry in manifest["files"]:
        if not entry.get("sha256") or entry.get("bytes") is None:
            raise ValueError(f"{entry['path']}: manifest records no sha256/bytes; run `{PIN_COMMAND}`")
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"checkpoint file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def download_archive(url: str, target: Path, *, max_bytes: int = MAX_ARCHIVE_BYTES) -> int:
    """Stream ``url`` to ``target`` (no credential, redirects followed); refuse more than ``max_bytes``."""
    request = urllib.request.Request(url, headers={"User-Agent": "raft-stereo-pipeline/0.1 (DIMER tutorial)"})
    written = 0
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as handle:
        while chunk := response.read(1 << 20):
            written += len(chunk)
            if written > max_bytes:
                raise ValueError(f"{url}: download exceeds {max_bytes:,} bytes; refusing it")
            handle.write(chunk)
    return written


def extract_member(archive: Path, member_name: str, target: Path, *, max_bytes: int = MAX_MEMBER_BYTES) -> str:
    """Copy exactly one zip member whose file name is ``member_name`` to ``target``; return the member path.

    Nothing else in the archive is extracted. The member must be unique, a regular file, a relative path without
    ``..``, and no larger than ``max_bytes``.
    """
    with zipfile.ZipFile(archive) as bundle:
        matches = [
            info
            for info in bundle.infolist()
            if not info.is_dir() and info.filename.replace("\\", "/").rsplit("/", 1)[-1] == member_name
        ]
        if len(matches) != 1:
            raise ValueError(f"{archive.name}: expected exactly one member named {member_name}, found {len(matches)}")
        info = matches[0]
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ":" in name or ".." in name.split("/"):
            raise ValueError(f"{archive.name}: unsafe member path {info.filename!r}")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError(f"{archive.name}: member {info.filename!r} is a symbolic link")
        if info.file_size > max_bytes:
            raise ValueError(f"{archive.name}: member {info.filename!r} expands to {info.file_size:,} bytes > {max_bytes:,}")
        with bundle.open(info) as source, open(target, "wb") as handle:
            shutil.copyfileobj(source, handle, 1 << 20)
    return info.filename


def _archive_download(entry: Mapping[str, Any], root: Path) -> None:
    """Fetch the archive named in the manifest entry, keep only the checkpoint member, delete the archive."""
    archive = root / (entry["path"] + ".archive.zip")
    try:
        download_archive(entry["url"], archive)
        extract_member(archive, entry.get("archiveMember", entry["path"]), root / entry["path"])
    finally:
        archive.unlink(missing_ok=True)


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[Mapping[str, Any], Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but git-ignores the
    checkpoint). Returns the relative paths fetched; ``verify_snapshot`` still runs afterwards and decides."""
    _require_pinned("stage the checkpoint")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    _check_manifest_identity(manifest)
    missing = [entry for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"checkpoint at {root} is missing {[e['path'] for e in missing]}; pass allow_download=True to fetch it"
        )
    fetch = downloader or _archive_download
    for entry in missing:
        if entry.get("url") != WEIGHTS_ARCHIVE_URL or entry.get("path") != WEIGHTS_FILE:
            raise ValueError(f"manifest entry {entry.get('path')!r} @ {entry.get('url')!r} is not the package's checkpoint")
        root.mkdir(parents=True, exist_ok=True)
        fetch(entry, root)
    return [entry["path"] for entry in missing]


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    """Load a RAFT-Stereo checkpoint as a plain state_dict (``weights_only=True``); drop DataParallel's prefix."""
    import torch

    checkpoint = Path(path)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError(f"{Path(path).name}: expected a state_dict mapping, got {type(state).__name__}")
    return {(key[len("module.") :] if key.startswith("module.") else key): value for key, value in state.items()}


# --------------------------------------------------------------------------- metrics and baselines


def disparity_metrics(predicted: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> dict[str, Any]:
    """End-point error and bad-pixel rates of a disparity map against ground truth, over valid pixels.

    ``epe`` is the mean absolute disparity error in pixels (for disparity, the end-point error of the 1-D
    displacement). ``bad_1px``/``bad_2px``/``bad_3px`` are the shares of valid pixels whose error exceeds 1, 2 or 3 px.
    ``d1_all`` is KITTI's D1: error > 3 px **and** > 5 % of the true disparity. Lower is better for every value.
    Ground truth at or above ``MAX_DISPARITY`` is not scored, as upstream evaluation does.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape or truth.ndim != 2:
        raise ValueError(f"disparities must both have shape (H, W), got {predicted.shape} and {truth.shape}")
    mask = np.ones(truth.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != truth.shape:
        raise ValueError(f"valid mask shape {mask.shape} != disparity shape {truth.shape}")
    mask = mask & np.isfinite(truth) & (np.abs(truth) < MAX_DISPARITY)
    if not mask.any():
        raise ValueError("no valid pixel to score")
    error = np.abs(predicted - truth)[mask]
    out: dict[str, Any] = {"epe": float(error.mean())}
    for t in BAD_THRESHOLDS:
        out[f"bad_{t:g}px"] = float((error > t).mean())
    out["d1_all"] = float(((error > 3.0) & (error > 0.05 * np.abs(truth[mask]))).mean())
    out["valid_pixels"] = int(mask.sum())
    return out


METRIC_KEYS = ("epe", "bad_1px", "bad_2px", "bad_3px", "d1_all")


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean of per-pair metrics (each pair weighs the same), plus the pixel-weighted EPE."""
    if not rows:
        raise ValueError("no rows to aggregate")
    out: dict[str, Any] = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}
    pixels = sum(row["valid_pixels"] for row in rows)
    out["pixel_weighted_epe"] = float(sum(row["epe"] * row["valid_pixels"] for row in rows) / pixels)
    out["n_pairs"] = len(rows)
    return out


def median_disparity(records: Sequence[Mapping[str, Any]]) -> float:
    """The median ground-truth disparity over the valid pixels of ``records`` (fit on the training split only)."""
    values = [np.asarray(r["disparity"], np.float64)[_valid(r)] for r in records]
    pooled = np.concatenate([v for v in values if v.size])
    if not pooled.size:
        raise ValueError("no valid ground-truth pixel to fit the median baseline")
    return float(np.median(pooled))


def sgbm_num_disparities(records: Sequence[Mapping[str, Any]]) -> int:
    """The StereoSGBM search range for ``records``: their largest valid disparity, plus 8 px, rounded up to 16."""
    top = max(float(np.asarray(r["disparity"], np.float64)[_valid(r)].max(initial=0.0)) for r in records)
    return int(min(MAX_DISPARITY, 16 * np.ceil((top + 8.0) / 16.0)))


SGBM_BLOCK_SIZE = 5


def sgbm_disparity(left: Image.Image, right: Image.Image, *, num_disparities: int, block_size: int = SGBM_BLOCK_SIZE) -> dict[str, Any]:
    """Classical semi-global block matching (OpenCV StereoSGBM, 3-way mode) on the grey-level pair.

    SGBM leaves pixels it cannot match unassigned. They are filled from the nearest assigned pixel on the same row,
    taking the smaller (farther) of the left and right neighbours, so every pixel gets a disparity and the baseline is
    scored on exactly the pixels the network is scored on. ``density`` reports the share SGBM assigned itself.
    """
    import cv2

    if isinstance(num_disparities, bool) or not isinstance(num_disparities, int) or num_disparities < 16 or num_disparities % 16:
        raise ValueError(f"num_disparities must be a positive multiple of 16, got {num_disparities!r}")
    first, second = validate_pair(left, right)
    channels = 3
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * channels * block_size**2,
        P2=32 * channels * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(np.asarray(first), np.asarray(second)).astype(np.float32) / 16.0
    assigned = raw >= 0
    filled = _fill_rows(raw, assigned)
    return {
        "disparity": filled,
        "density": float(assigned.mean()),
        "num_disparities": num_disparities,
        "block_size": block_size,
        "method": "OpenCV StereoSGBM (3-way), unassigned pixels filled from the farther row neighbour",
    }


def _fill_rows(values: np.ndarray, assigned: np.ndarray) -> np.ndarray:
    height, width = values.shape
    columns = np.arange(width)
    out = np.zeros_like(values, dtype=np.float32)
    for y in range(height):
        known = np.flatnonzero(assigned[y])
        if not known.size:
            continue
        left_index = np.maximum.accumulate(np.where(assigned[y], columns, -1))
        right_index = np.minimum.accumulate(np.where(assigned[y], columns, width)[::-1])[::-1]
        left_value = np.where(left_index >= 0, values[y, np.clip(left_index, 0, width - 1)], np.inf)
        right_value = np.where(right_index < width, values[y, np.clip(right_index, 0, width - 1)], np.inf)
        out[y] = np.minimum(left_value, right_value)
    out[~np.isfinite(out)] = 0.0
    return out


# --------------------------------------------------------------------------- validation


def validate_pair(left: Any, right: Any) -> tuple[Image.Image, Image.Image]:
    """Two PIL images of one size inside the ceilings; returns them converted to RGB."""
    for name, image in (("left", left), ("right", right)):
        if not isinstance(image, Image.Image):
            raise TypeError(f"{name} must be a PIL.Image.Image, got {type(image).__name__}")
    if left.size != right.size:
        raise ValueError(f"left and right images differ in size: {left.size} vs {right.size}; a rectified pair has one size")
    width, height = left.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}; downscale the pair (and its disparity) first")
    if width * height > MAX_PIXELS:
        raise ValueError(f"image has {width * height} pixels > MAX_PIXELS {MAX_PIXELS}")
    return left.convert("RGB"), right.convert("RGB")


def check_iters(value: Any, name: str = "iters") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_ITERS:
        raise ValueError(f"{name} must be an int in 1..{MAX_ITERS}, got {value!r}")
    return value


INPUT_SCHEMA: dict[str, Any] = {
    "input": "a rectified stereo pair: two PIL.Image.Image of one size (any mode, converted to RGB), left then right",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "max_pixels": MAX_PIXELS,
    "iters": [1, MAX_ITERS],
    "output": (
        "disparity (H, W) float32 in pixels for the LEFT image: left pixel (x, y) matches right pixel (x - d, y); "
        "larger disparity = nearer surface"
    ),
    "preprocessing": (
        "RGB values 0..255 as float; the carried upstream model maps them to [-1, 1] itself. The pair is padded by edge "
        f"replication to a multiple of {PAD_DIVISOR} px (upstream InputPadder) and the disparity is cropped back"
    ),
    "ground_truth": (
        "optional: disparity (H, W) in pixels with a valid mask; formats .pfm (inf = unknown), .npy, or 16-bit .png "
        f"(KITTI, value / 256, 0 = unknown); values >= {MAX_DISPARITY:.0f} px are not scored"
    ),
}


def validate_inputs(left: Any, right: Any, *, iters: int = DEFAULT_VALID_ITERS, name: str = "pair") -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict) before any model runs."""
    first, _second = validate_pair(left, right)
    updates = check_iters(iters)
    pad = [(-first.width) % PAD_DIVISOR, (-first.height) % PAD_DIVISOR]
    findings = []
    if any(pad):
        findings.append(f"padded by {pad[0]} px in width and {pad[1]} px in height (edge replication); the output is cropped back")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": f"{name}:{side}", "mode": image.mode, "size": list(image.size)}
            for side, image in (("left", left), ("right", right))
        ],
        "iters": updates,
        "padding": pad,
        "verdict": "accepted",
        "findings": findings,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def _valid(record: Mapping[str, Any]) -> np.ndarray:
    disparity = np.asarray(record["disparity"])
    valid = record.get("valid")
    mask = np.ones(disparity.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return mask & np.isfinite(disparity) & (disparity < MAX_DISPARITY)


def validate_record(record: Any, index: int = 0) -> dict[str, Any]:
    """One labelled record ``{"left", "right", "disparity", "valid"?, "id"?, "group"?}``; raise naming the rule."""
    if not isinstance(record, Mapping) or not {"left", "right", "disparity"} <= set(record):
        raise ValueError(f"record {index} must be a mapping with 'left', 'right' and 'disparity'")
    try:
        first, _second = validate_pair(record["left"], record["right"])
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"record {index}: {exc}") from exc
    disparity = np.asarray(record["disparity"])
    if disparity.shape != (first.height, first.width):
        raise ValueError(f"record {index}: disparity shape {disparity.shape} != image shape ({first.height}, {first.width})")
    if not np.issubdtype(disparity.dtype, np.floating) and not np.issubdtype(disparity.dtype, np.integer):
        raise ValueError(f"record {index}: disparity must be numeric, got {disparity.dtype}")
    valid = record.get("valid")
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != disparity.shape or (valid.dtype != bool and not np.isin(valid, (0, 1)).all()):
            raise ValueError(f"record {index}: valid mask must be boolean with shape {disparity.shape}")
    mask = _valid(record)
    if not mask.any():
        raise ValueError(f"record {index}: no pixel has valid ground truth")
    values = disparity[mask].astype(np.float64)
    if (values < 0).any():
        raise ValueError(
            f"record {index}: {int((values < 0).sum())} valid pixels have negative disparity; this pipeline expects "
            "the LEFT image's disparity (left x matches right x - d), so swap left and right or negate the map"
        )
    return {"size": list(first.size), "valid_fraction": float(mask.mean()), "max_disparity": float(values.max()), "mean_disparity": float(values.mean())}


def validate_dataset(records: Sequence[Mapping[str, Any]], *, epochs: int = DEFAULT_EPOCHS) -> dict[str, Any]:
    """Validation stage for labelled stereo records: raise on the first broken record, else the dataset manifest."""
    if not records:
        raise ValueError("dataset must hold at least one record")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset holds {len(records)} records > MAX_RECORDS {MAX_RECORDS}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be an int in 1..100, got {epochs!r}")
    ids = [str(r.get("id", i)) if isinstance(r, Mapping) else str(i) for i, r in enumerate(records)]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate record ids: {duplicates[:5]}")
    rows = [validate_record(record, index) for index, record in enumerate(records)]
    sizes = sorted({tuple(row["size"]) for row in rows})
    findings = []
    if len(sizes) > 1:
        findings.append(f"{len(sizes)} different image sizes; training batches are grouped by size")
    crop_h, crop_w = TRAIN_CROP
    cropped = sum(1 for w, h in (row["size"] for row in rows) if h > crop_h or w > crop_w)
    if cropped:
        findings.append(f"{cropped} records are larger than the training crop {crop_w} x {crop_h} px (W x H) and are cropped at a seeded random position for training only; evaluation uses the full image")
    return {
        "n_records": len(records),
        "image_sizes": [list(s) for s in sizes],
        "mean_valid_fraction": float(np.mean([row["valid_fraction"] for row in rows])),
        "max_disparity_px": float(max(row["max_disparity"] for row in rows)),
        "mean_disparity_px": float(np.mean([row["mean_disparity"] for row in rows])),
        "epochs": epochs,
        "findings": findings,
        "verdict": "accepted",
    }


def split_dataset(
    records: Sequence[Mapping[str, Any]], *, holdout: float = 0.25, seed: int = DEFAULT_SEED
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Seeded random split into training and held-out records.

    Records that share a ``group`` value (for example several exposures or crops of one scene) always land on the
    same side, so a scene seen in training is never scored as held-out. Without groups every record is its own group,
    which assumes the pairs are independent.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout}")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(str(record.get("group", record.get("id", index))), []).append(index)
    if len(groups) < 2:
        raise ValueError(f"at least 2 independent groups are required to split, got {len(groups)}")
    keys = sorted(groups)
    order = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))]
    n_held = min(len(keys) - 1, max(1, int(round(holdout * len(keys)))))
    held_keys = set(order[:n_held])
    train = [records[i] for k in keys if k not in held_keys for i in groups[k]]
    held = [records[i] for k in keys if k in held_keys for i in groups[k]]
    return train, held


def read_stereo_records(directory: str | Path) -> list[dict[str, Any]]:
    """Read BYOD stereo records from ``<directory>/pairs.json`` plus the files it names.

    ``pairs.json`` is a list of objects such as
    ``{"left": "scene1/im0.png", "right": "scene1/im1.png", "disparity": "scene1/disp0.pfm", "valid": "scene1/mask.png",
    "group": "scene1"}``; ``disparity`` is a ``.pfm``, ``.npy`` or 16-bit KITTI ``.png`` (see ``read_disparity``),
    ``valid`` (a single-channel PNG whose non-zero pixels are valid) and ``group`` are optional. File names must stay
    inside ``directory``.
    """
    root = Path(directory).resolve()
    index = root / "pairs.json"
    if not index.is_file():
        raise FileNotFoundError(f"{index} not found; expected pairs.json next to the images")
    try:
        entries = json.loads(index.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"pairs.json is not valid JSON: {exc}") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError("pairs.json must hold a non-empty list of records")
    if len(entries) > MAX_RECORDS:
        raise ValueError(f"pairs.json lists {len(entries)} records > MAX_RECORDS {MAX_RECORDS}")

    def resolve(idx: int, name: Any) -> Path:
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"pairs.json entry {idx}: file {name!r} must be relative to the dataset folder")
        resolved = (root / relative).resolve()
        if root not in resolved.parents:
            raise ValueError(f"pairs.json entry {idx}: file {name!r} resolves outside the dataset folder")
        if not resolved.is_file():
            raise FileNotFoundError(f"pairs.json entry {idx}: file {name!r} is missing from the dataset folder")
        return resolved

    records = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"left", "right", "disparity"} <= set(entry):
            raise ValueError(f"pairs.json entry {idx} must have 'left', 'right' and 'disparity'")
        images = []
        for key in ("left", "right"):
            with Image.open(resolve(idx, entry[key])) as handle:
                images.append(handle.convert("RGB"))
        disparity, valid = read_disparity(resolve(idx, entry["disparity"]))
        if entry.get("valid"):
            with Image.open(resolve(idx, entry["valid"])) as handle:
                mask = np.asarray(handle.convert("L")) > 0
            if mask.shape != valid.shape:
                raise ValueError(f"pairs.json entry {idx}: valid mask shape {mask.shape} != disparity shape {valid.shape}")
            valid = valid & mask
        record = {"id": str(entry.get("id", entry["left"])), "left": images[0], "right": images[1], "disparity": disparity, "valid": valid}
        if entry.get("group") is not None:
            record["group"] = str(entry["group"])
        records.append(record)
    return records


def evaluation_report(
    result: Mapping[str, Any],
    truth: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    *,
    baselines: Mapping[str, np.ndarray] | None = None,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Single-pair evaluation stage: EPE and bad-pixel rates against ground truth next to the given baselines, or
    ``not-measurable`` when no ground truth is supplied."""
    disparity = np.asarray(result["disparity"])
    base = {
        "task": "dense stereo disparity for the left image of a rectified pair",
        "sample_kind": sample_kind,
        "disparity_shape": list(disparity.shape),
        "iters": result.get("iters"),
        "predicted_disparity_px": {"min": float(disparity.min()), "median": float(np.median(disparity)), "max": float(disparity.max())},
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if truth is None:
        return {
            **base,
            "metrics": {},
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth disparity was supplied for the evaluated pair",
            "needs": "rectified pairs from the deployment rig with ground-truth disparity (structured light, LiDAR or rendering), scored by EPE and bad-pixel rates",
        }
    return {
        **base,
        "metrics": disparity_metrics(disparity, truth, valid),
        "baselines": [{"id": name, **disparity_metrics(values, truth, valid)} for name, values in (baselines or {}).items()],
        "verdict": "sample-sanity",
        "reason": "one pair with exact ground truth; geometry sanity evidence, not a benchmark",
        "needs": "a labelled set of pairs from the deployment domain for any accuracy claim",
    }


# --------------------------------------------------------------------------- sequence loss


def sequence_loss(predictions: Sequence[Any], truth: Any, valid: Any, gamma: float = DEFAULT_GAMMA) -> Any:
    """Upstream train_stereo.py ``sequence_loss``: L1 on every refinement's flow (= -disparity), weighted
    ``gamma' ** (N - i - 1)`` with ``gamma' = gamma ** (15 / (N - 1))``, over valid pixels with |flow| < 700 px."""

    if not 0.0 < gamma < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    n = len(predictions)
    if n < 1:
        raise ValueError("no predictions")
    mask = (valid >= 0.5) & (truth.abs()[:, 0] < LOSS_MAX_FLOW)
    mask = mask[:, None]
    if not bool(mask.any()):
        raise ValueError("no valid pixel in the batch")
    adjusted = gamma ** (15 / (n - 1)) if n > 1 else 1.0
    loss = truth.new_zeros(())
    for i, prediction in enumerate(predictions):
        loss = loss + adjusted ** (n - i - 1) * (prediction - truth).abs()[mask].mean()
    return loss


# --------------------------------------------------------------------------- pipeline


@dataclass
class RaftStereoPipeline:
    """Dense stereo disparity, and bounded fine-tuning on labelled pairs, with RAFT-Stereo."""

    model: Any
    device: str
    padder: Any
    source: str = "checkpoint"
    base_state_digest: str | None = None
    adapted: bool = False
    frozen_prefixes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> RaftStereoPipeline:
        """Verify the carried upstream source and the pinned checkpoint, then load it strictly."""
        _require_pinned("load the model")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no checkpoint manifest at {root}; stage {MODEL_ID} under weights/{MODEL_KEY} "
                "(allow_download=True fetches the manifest-listed file)"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        model = build_model(upstream_dir)
        state = load_checkpoint_state(root / WEIGHTS_FILE)
        model.load_state_dict(state, strict=True)
        return cls._wrap(model, device, upstream_dir, source=str(root), base_state_digest=MODEL_REVISION)

    @classmethod
    def from_random_init(cls, device: str | None = None, upstream_dir: str | Path | None = None, *, seed: int = 0) -> RaftStereoPipeline:
        """The same architecture with seeded random weights. For offline tests only: it is not the model."""
        import torch

        torch.manual_seed(seed)
        return cls._wrap(build_model(upstream_dir), device, upstream_dir, source="random-init (test only)", base_state_digest=None)

    @classmethod
    def _wrap(cls, model: Any, device: str | None, upstream_dir: str | Path | None, **kwargs: Any) -> RaftStereoPipeline:
        import torch

        resolved = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        load_upstream(upstream_dir)  # verified and imported by build_model; the InputPadder comes from the same tree
        utils = importlib.import_module("core.utils.utils")
        return cls(model=model.to(resolved).eval(), device=resolved, padder=utils.InputPadder, **kwargs)

    # ---------------------------------------------------------------- inference

    def _tensor(self, images: Sequence[Image.Image]) -> Any:
        import torch

        arrays = np.stack([np.asarray(image, dtype=np.float32) for image in images])
        return torch.from_numpy(arrays).permute(0, 3, 1, 2).to(self.device)

    def estimate(self, left: Image.Image, right: Image.Image, *, iters: int = DEFAULT_VALID_ITERS) -> dict[str, Any]:
        """Disparity of the left image of a rectified pair; returns ``disparity`` as an (H, W) float32 array."""
        import torch

        first, second = validate_pair(left, right)
        updates = check_iters(iters)
        image1, image2 = self._tensor([first]), self._tensor([second])
        padder = self.padder(image1.shape, divis_by=PAD_DIVISOR)
        image1, image2 = padder.pad(image1, image2)
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            _low, flow_up = self.model(image1, image2, iters=updates, test_mode=True)
        if was_training:
            self.model.train()
        flow = padder.unpad(flow_up)[0, 0]
        disparity = (-flow).float().cpu().numpy()
        if disparity.shape != (first.height, first.width) or not np.isfinite(disparity).all():
            raise RuntimeError(f"model returned a malformed disparity of shape {disparity.shape}")
        return {
            "disparity": disparity,
            "iters": updates,
            "width": first.width,
            "height": first.height,
            "adapted": self.adapted,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, iters: int = DEFAULT_VALID_ITERS) -> dict[str, Any]:
        """Mean EPE and bad-pixel rates over labelled pairs, and the per-pair rows."""
        rows = []
        for index, record in enumerate(records):
            disparity = self.estimate(record["left"], record["right"], iters=iters)["disparity"]
            rows.append({"id": str(record.get("id", index)), **disparity_metrics(disparity, record["disparity"], _valid(record))})
        return {
            **aggregate_metrics(rows),
            "per_pair": rows,
            "iters": iters,
            "adapted": self.adapted,
            "estimation": f"one pass over {len(records)} pairs; each pair weighs the same; no resampling, no dispersion estimate",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---------------------------------------------------------------- adaptation

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        train_iters: int = DEFAULT_TRAIN_ITERS,
        gamma: float = DEFAULT_GAMMA,
        seed: int = DEFAULT_SEED,
        freeze_encoders: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded gradient fine-tuning with upstream's sequence loss, changing this pipeline's model in place.

        Every BatchNorm layer is held in eval mode (upstream ``freeze_bn``). With ``freeze_encoders`` the feature
        encoder (``fnet``) and context encoder (``cnet``) keep their weights and only the context projections, the
        multi-level GRU update block, the disparity head and the upsampling mask train. Gradients are clipped to
        norm 1.0. Pairs larger than ``TRAIN_CROP`` are cropped at a seeded random position; batches share a size.
        """
        import torch

        validate_dataset(records, epochs=epochs)
        updates = check_iters(train_iters, "train_iters")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float) or not 0.0 < float(learning_rate) <= 1.0:
            raise ValueError(f"learning_rate must be a number in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not (freeze_encoders and name.startswith(ENCODER_PREFIXES))
        self.frozen_prefixes = ENCODER_PREFIXES if freeze_encoders else ()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=float(weight_decay), eps=1e-8)

        crop_h, crop_w = TRAIN_CROP
        by_size: dict[tuple[int, int], list[int]] = {}
        for i, record in enumerate(records):
            width, height = record["left"].size
            by_size.setdefault((min(height, crop_h), min(width, crop_w)), []).append(i)
        epoch_losses: list[float] = []
        steps = 0
        for epoch in range(epochs):
            self.model.train()
            self.model.freeze_bn()
            batches = []
            for size, indices in by_size.items():
                order = [indices[int(j)] for j in rng.permutation(len(indices))]
                batches.extend((size, order[k : k + batch_size]) for k in range(0, len(order), batch_size))
            running, n_batches = 0.0, 0
            for b in rng.permutation(len(batches)):
                (height, width), members = batches[int(b)]
                lefts, rights, truths, valids = [], [], [], []
                for i in members:
                    record = records[i]
                    first, second = validate_pair(record["left"], record["right"])
                    full_h, full_w = first.height, first.width
                    top = int(rng.integers(0, full_h - height + 1))
                    left_x = int(rng.integers(0, full_w - width + 1))
                    box = (left_x, top, left_x + width, top + height)
                    lefts.append(first.crop(box))
                    rights.append(second.crop(box))
                    truths.append(np.asarray(record["disparity"], np.float32)[top : top + height, left_x : left_x + width])
                    valids.append(_valid(record)[top : top + height, left_x : left_x + width])
                image1, image2 = self._tensor(lefts), self._tensor(rights)
                padder = self.padder(image1.shape, divis_by=PAD_DIVISOR)
                image1, image2 = padder.pad(image1, image2)
                truth = -torch.from_numpy(np.stack(truths))[:, None].to(self.device)  # flow x = -disparity
                valid = torch.from_numpy(np.stack(valids).astype(np.float32)).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = [padder.unpad(p) for p in self.model(image1, image2, iters=updates)]
                loss = sequence_loss(predictions, truth, valid, gamma)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()
                running += float(loss.detach().cpu())
                n_batches += 1
                steps += 1
            epoch_losses.append(running / max(1, n_batches))
            if progress is not None:
                progress({"epoch": epoch + 1, "epochs": epochs, "loss": epoch_losses[-1], "steps": steps})
        self.model.eval()
        self.adapted = True
        return {
            "method": "gradient fine-tuning of the pretrained network (not PEFT): " + ("encoders frozen" if freeze_encoders else "all parameters trainable"),
            "epochs": epochs,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "lr_schedule": "constant",
            "weight_decay": float(weight_decay),
            "optimizer": "AdamW (eps 1e-8)",
            "loss": f"RAFT-Stereo sequence loss (L1 on flow = -disparity, gamma {gamma} adjusted to the iteration count, |flow| >= {LOSS_MAX_FLOW:.0f} px ignored)",
            "gradient_clip_norm": GRAD_CLIP,
            "batchnorm": "frozen (eval mode, upstream freeze_bn)",
            "train_iters": updates,
            "train_crop_hw": list(TRAIN_CROP),
            "augmentation": "none",
            "seed": seed,
            "precision": "float32",
            "freeze_encoders": freeze_encoders,
            "frozen_prefixes": list(self.frozen_prefixes),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "device": self.device,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---------------------------------------------------------------- artifact

    def save_artifact(self, path: str | Path, *, notes: str | None = None) -> dict[str, Any]:
        """Write the adapted tensors as one SafeTensors file with the provenance in its metadata.

        Tensors under ``frozen_prefixes`` are left out: they equal the verified base checkpoint, which
        ``load_artifact`` loads first. The artifact is therefore an adapter bound to the base checkpoint's digest.
        """
        from safetensors.torch import save_file

        if not self.adapted:
            raise RuntimeError("nothing to export: the pipeline has not been fine-tuned")
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in self.model.state_dict().items()
            if not any(name.startswith(prefix) for prefix in self.frozen_prefixes)
        }
        metadata = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": MODEL_KEY,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "model_args": json.dumps(MODEL_ARGS, sort_keys=True),
            "frozen_prefixes": json.dumps(list(self.frozen_prefixes)),
            "base_state_digest": self.base_state_digest or "",
            "notes": notes or "",
        }
        save_file(tensors, str(artifact_path), metadata=metadata)
        return {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "format": ARTIFACT_FORMAT,
            "tensors": len(tensors),
            "frozen_prefixes": list(self.frozen_prefixes),
            "base_state_digest": self.base_state_digest,
            "upstream_commit": UPSTREAM_COMMIT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @staticmethod
    def read_artifact_metadata(path: str | Path) -> dict[str, Any]:
        """Read and check the artifact's provenance header without loading any tensor."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {metadata.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if (metadata.get("model_id"), metadata.get("model_revision"), metadata.get("model_key")) != (MODEL_ID, MODEL_REVISION, MODEL_KEY):
            raise ValueError(
                f"artifact was built on {metadata.get('model_id')}@{metadata.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        if metadata.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError(f"artifact was built with upstream code {metadata.get('upstream_commit')!r}, package carries {UPSTREAM_COMMIT}")
        if json.loads(metadata.get("model_args", "{}")) != MODEL_ARGS:
            raise ValueError("artifact architecture arguments differ from the package's MODEL_ARGS")
        return {**metadata, "frozen_prefixes": json.loads(metadata["frozen_prefixes"])}

    def apply_artifact(self, path: str | Path) -> None:
        """Load adapter tensors onto this base pipeline; refuse any tensor set that does not fit."""
        from safetensors.torch import load_file

        metadata = self.read_artifact_metadata(path)
        expected_base = metadata.get("base_state_digest") or None
        if expected_base != (self.base_state_digest or None):
            raise ValueError(f"artifact was exported against base {expected_base!r}, this pipeline holds {self.base_state_digest!r}")
        tensors = load_file(str(path), device="cpu")
        prefixes = tuple(metadata["frozen_prefixes"])
        result = self.model.load_state_dict(tensors, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"artifact carries tensors the model does not have: {result.unexpected_keys[:5]}")
        stray = [key for key in result.missing_keys if not any(key.startswith(p) for p in prefixes)]
        if stray:
            raise ValueError(f"artifact is missing trainable tensors: {stray[:5]}")
        self.model.eval()
        self.adapted = True
        self.frozen_prefixes = prefixes
        self.source = f"artifact:{Path(path).name}"

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        weights_dir: str | Path | None = None,
        upstream_dir: str | Path | None = None,
        device: str | None = None,
    ) -> RaftStereoPipeline:
        """Rebuild an adapted pipeline from files: verified base checkpoint first, then the adapter tensors."""
        cls.read_artifact_metadata(path)
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, upstream_dir=upstream_dir)
        pipe.apply_artifact(path)
        return pipe
