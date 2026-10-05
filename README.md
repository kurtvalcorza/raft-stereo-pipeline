# RAFT-Stereo Stereo Disparity Pipeline

DIMER-oriented pipeline for **RAFT-Stereo dense disparity and stereo depth estimation** (`princeton-vl/RAFT-Stereo/raftstereo-middlebury`: the upstream RAFT-Stereo network with its Middlebury checkpoint). It estimates a disparity map for the left image of a rectified stereo pair, evaluates it by end-point error (EPE) and bad-pixel rates against a median constant, OpenCV StereoSGBM and the pretrained network, runs a bounded stereo fine-tune that exports a portable SafeTensors adapter, and ships a `MODEL_CARD.md` at DIMER Model Card Specification 1.2 and a standalone `E2E` / `GUIDED` tutorial at DIMER Notebook Specification 2.2.

## Upstream alignment

- Model: `princeton-vl/RAFT-Stereo/raftstereo-middlebury` — the checkpoint `raftstereo-middlebury.pth` from upstream's `models.zip` (the archive linked from `download_models.sh`), the checkpoint upstream recommends for in-the-wild images.
- Checkpoint revision: `d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819`, the SHA-256 of the checkpoint's own bytes (a URL-hosted file has no commit). The checkpoint was pinned on 2026-10-05 by running `python tools/pin_snapshot.py --dry-run` in a Google Colab runtime (Dropbox reachable): the extracted member is 44,617,876 bytes with SHA-256 `d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819`, and it strict-loaded into the carried architecture (337 tensors, 11,116,176 parameters). The manifest and `MODEL_REVISION` were written from that output. Every load verifies the size and digest first.
- Model code: upstream `princeton-vl/RAFT-Stereo` `core/` at commit `6e93ed2169bd858dbb43033988563f3b0bb49506`, carried verbatim under `src/raft_stereo_pipeline/third_party/raft_stereo/` with its MIT licence. RAFT-Stereo is not on PyPI; the code is never cloned or downloaded at runtime, and `load_upstream()` refuses to import it unless every file matches `UPSTREAM_SHA256`.
- Correlation: upstream's pure-PyTorch `corr_implementation = "reg"`. The optional CUDA sampler extension is never built, so the same code runs on a CUDA GPU or a CPU.
- Upstream licence: **MIT** (code and checkpoints, Princeton Vision & Learning Lab).
- Architecture: the upstream demo defaults (`hidden_dims` 128 × 3, 3 GRU levels, `n_downsample` 2, context norm `batch`, 4 correlation levels of radius 4): 11,116,176 parameters in 337 state tensors, counted from the carried code.
- Repository adaptation: **E2E** (bounded gradient fine-tuning on labelled pairs; encoders frozen by default).

## Two things to know before you start

**The checkpoint is a pickle-based `.pth` file.** Upstream publishes no safe-tensor format. The file is trusted only after its byte size and SHA-256 match the manifest, and it is read only with `torch.load(..., weights_only=True)`, which refuses arbitrary pickled objects. Exported adapters are SafeTensors.

**The numbers in the tutorial are tutorial evidence.** The default data are rendered in code with exact disparity (no download, no licence), so the held-out scores say how the model behaves on 8 rendered pairs, not on photographs or on any rig. No benchmark (KITTI, Middlebury, ETH3D, SceneFlow) is run, and the paper's numbers are not reproduced.

## Quick start

```python
from raft_stereo_pipeline import RaftStereoPipeline, stereo_dataset, split_dataset, render_stereo_scene

pipe = RaftStereoPipeline.from_pretrained(allow_download=True)  # verifies the carried upstream code and the pinned checkpoint
pair = render_stereo_scene(7)                                   # a rendered 320 x 224 pair with exact disparity
disparity = pipe.estimate(pair["left"], pair["right"], iters=32)["disparity"]  # (H, W) float32 px, left image
records = stereo_dataset(32, seed=0)
train, held_out = split_dataset(records, holdout=0.25, seed=20261004)
print(pipe.evaluate(held_out)["epe"])                           # pretrained, held-out EPE
pipe.finetune(train, epochs=3)                                  # bounded fine-tuning, encoders frozen
print(pipe.evaluate(held_out)["epe"])
pipe.save_artifact("outputs/raft_stereo_adapter.safetensors")
```

`from_pretrained` raises until the checkpoint is pinned (see above).

## Weights layout

```
weights/raftstereo-middlebury/   dimer-base-manifest.json          (committed)
                                 raftstereo-middlebury.pth         (git-ignored; staged from the archive on first use)
```

`stage_missing_files(allow_download=True)` downloads `models.zip` only if the checkpoint is absent, extracts the single member `raftstereo-middlebury.pth` (refusing ambiguous, unsafe, symlinked or oversized members), deletes the archive, and `verify_snapshot()` checks the byte size and SHA-256 before any load. `docs/WEIGHTS.md` records the provenance and the size question (the planned ~50.6 MB is not what any upstream configuration produces; the architecture serialises to 44,605,701 bytes, an estimate until the pin records the real size).

## Evaluation

`disparity_metrics` reports `epe` (mean absolute disparity error, px), `bad_1px` / `bad_2px` / `bad_3px` (share of valid pixels off by more than 1, 2 or 3 px; ETH3D reports bad-1, Middlebury bad-2) and `d1_all` (KITTI: > 3 px and > 5 % of the true disparity), over valid pixels only, ground truth ≥ 512 px excluded. Baselines: `median_disparity` (training-split median everywhere), `sgbm_disparity` (OpenCV StereoSGBM 3-way, search range from the training split, unmatched pixels filled from the farther row neighbour, density reported), and the pretrained network before adaptation.

## Adapter artifacts

`save_artifact(path)` writes one SafeTensors file holding the tensors the fine-tune could change (with the encoders frozen, `fnet.*` and `cnet.*` are left out because they equal the verified base) and a metadata header (format `raft-stereo-adapter-v1`, model id and revision, upstream commit, architecture arguments, frozen prefixes, base digest). `RaftStereoPipeline.load_artifact(path)` re-verifies the checkpoint, checks the header and refuses any adapter whose format, base, upstream commit, architecture or tensor set does not fit.

## Tests

```
pip install -e . --no-deps
pytest
```

Tests run offline on CPU: the real upstream architecture with seeded random weights on small rendered pairs, temporary manifests and fake archives; no network and no checkpoint. `python tools/validate_release_assets.py` performs the static release-asset checks, and `python tools/build_notebook.py --check` the notebook parity check.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/raft-stereo-pipeline/blob/main/tutorials/raft_stereo_colab.ipynb)

`tutorials/raft_stereo_colab.ipynb` is declared `E2E` / `GUIDED` and is **standalone** (DIMER Notebook Specification 2.2 §4): it is generated by `tools/build_notebook.py` (`build_notebook.py/3.1`) from `tools/notebook_template.py` and carries, byte for byte and SHA-256-verified, the package's three modules, the eight carried upstream files, the stage runner `tools/tutorial_stages.py`, the hash-locked requirements `tutorials/requirements-colab.lock.txt`, the checkpoint manifest and the licence.

The notebook installs nothing into its own kernel. It downloads a pinned `uv` wheel (checked by size and SHA-256), builds an isolated CPython 3.12.12 environment, and installs the lock (35 packages) into it with `--require-hashes --only-binary :all:`; each stage (`weights`, `demo`, `probes`, `prepare`, `baselines`, `adapt`, `evaluate`, `infer`, `reload`, and the optional `byod_pair`) then runs in its own process with `MPLBACKEND=Agg` and no Hugging Face token, and hands results to the next only through files. A hosted runtime's preloaded packages are never replaced, so `Run all` needs no restart. The notebook supports Linux x86_64 only and stops with a clear message elsewhere. A T4 GPU is the documented runtime; a CPU-only runtime completes the same path slowly. The lock is compiled from the `pyproject.toml` pins with transitive versions held at those of a lock already installed on Colab/Kaggle (`tools/lock-constraints.txt`):

```
uv pip compile pyproject.toml -c tools/lock-constraints.txt --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --index-url https://pypi.org/simple --only-binary :all: -o tutorials/requirements-colab.lock.txt
```

Recompile it and regenerate the notebook whenever a pin changes.

## Release status

**Candidate** — initial development. The checkpoint is pinned (44,617,876 bytes, SHA-256 `d22e84c0…5819`), and a Google Colab T4 execution with the real checkpoint completed every stage on 2026-10-05 (held-out EPE 0.231 px pretrained, 0.112 px adapted, 0.653 px SGBM). That run was not strictly clean-state, and small random-dot probes (0–4 px) turned out to be unreadable by this checkpoint, so Section 5 now probes 8 px and 16 px and awaits a fresh hosted run (see `STATUS.md`). The default path executed end to end on a local CPU with a random-weight stand-in checkpoint (plumbing evidence only); see `STATUS.md` and `docs/release-verification.md` for exactly what was and was not run.

## Licensing

- Upstream code and checkpoint: MIT (`princeton-vl/RAFT-Stereo`); the carried copy keeps upstream's `LICENSE`.
- Tutorial data: rendered in code by `samples.py`; no third-party data.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
