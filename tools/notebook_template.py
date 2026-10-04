"""Per-repository template for tools/build_notebook.py /3.1 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the list of carried files and the
learner cells. Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the
repository) runs one stage per process in an isolated, hash-locked environment, so nothing is installed into the
notebook kernel.

This template configures an E2E stereo workflow: the carried upstream RAFT-Stereo source is verified against its
SHA-256 digests, the pinned Middlebury checkpoint is staged and digest-verified, the pretrained model is run on a
rendered demonstration pair and on two probes with known answers, a 32-pair rendered dataset with exact disparity is
validated and split, three baselines are scored on the held-out split (a median constant, OpenCV StereoSGBM and the
pretrained network), a bounded fine-tune runs, the exported adapter is scored in a fresh process, the adapted model is
run on unseen pairs, and a second fresh process reloads the adapter and checks it against the trained in-memory model.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "raft-stereo-pipeline"
PKG = "raft_stereo_pipeline"
UPSTREAM_FILES = (
    "core/__init__.py",
    "core/raft_stereo.py",
    "core/corr.py",
    "core/extractor.py",
    "core/update.py",
    "core/utils/__init__.py",
    "core/utils/utils.py",
    "LICENSE",
)

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/raft_stereo_colab.ipynb",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-princeton--vl%2FRAFT--Stereo-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/princeton-vl/RAFT-Stereo",
    ),
    ("arXiv", "https://img.shields.io/badge/arXiv-2109.07547-b31b1b.svg", "https://arxiv.org/abs/2109.07547"),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", f"https://github.com/kurtvalcorza/{REPO}/blob/main/LICENSE"),
]

TEMPLATE = {
    "package": PKG,
    "repo_name": REPO,
    "stem": "raft_stereo",
    "notebook_name": "raft_stereo_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "accelerator": "recommended",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime (a Colab or Kaggle **T4 GPU** is the documented runtime; a CPU-only "
        "runtime completes the same path, much more slowly — see the Prerequisites) builds an isolated Python environment from the "
        "carried hash-locked requirements ({n_locked} packages: torch, numpy, scipy, opt-einsum, opencv-python-headless, safetensors, "
        "pillow and their dependencies) without touching the notebook kernel's own packages, then runs each stage below in its own "
        "process: it verifies the carried upstream RAFT-Stereo source against its SHA-256 digests, stages and digest-verifies the "
        "pinned Middlebury checkpoint (Section 3 refuses to load a checkpoint whose SHA-256 is not pinned), runs the pretrained model "
        "on a rendered demonstration pair and two probes with exact answers, renders and validates a 32-pair dataset with exact "
        "ground-truth disparity and splits it 24 / 8, scores three baselines on the held-out pairs, **runs a bounded fine-tune**, "
        "scores the exported adapter in a fresh process, runs the adapted model on three unseen pairs and exports their disparity "
        "maps, and reloads the adapter in another fresh process to check that it reproduces the trained model. Nothing is skipped "
        "behind a default-off flag, and no clone, DIMER worker, credential, upload dialog or configuration edit is needed, and no restart "
        "(NOTEBOOK_SPEC 2.2 §5, RUN7, FT2)."
    ),
    "byod": (
        "Two optional branches in Section 13 are off by default (`USE_BYOD_PAIR = False`, `USE_BYOD_DATASET = False`). The **pair** "
        "branch runs your own rectified pair (and, if you have it, its ground-truth disparity) through the same validation, model, "
        "SGBM baseline and evaluation report as the demonstration pair. The **dataset** branch takes a folder of your own labelled "
        "pairs through the full workflow — validate → split → baselines → fine-tune → evaluate → infer → export → reload — with the "
        "same stages as the sample (NOTEBOOK_SPEC 2.2 DAT14), writing to `outputs/byod/` so the sample results are kept. Location "
        "fields (`BYOD_LEFT_PATH`, `BYOD_RIGHT_PATH`, `BYOD_DISPARITY_PATH`, `BYOD_DATASET_DIR`) read from a path without an upload "
        "dialog (EXE2); left empty on Colab, they open one. Uploaded files stay in this runtime."
    ),
    "title": "RAFT-Stereo — DIMER guided notebook: dense stereo disparity and bounded stereo fine-tuning (standalone)",
    "badges": BADGES,
    "capability": (
        "dense disparity estimation for rectified stereo pairs with RAFT-Stereo (`princeton-vl/RAFT-Stereo`, Middlebury checkpoint), "
        "evaluated by end-point error and bad-pixel rates against a median constant, OpenCV StereoSGBM and the pretrained network, and "
        "a bounded stereo fine-tune that exports a reloadable SafeTensors adapter"
    ),
    "intro": (
        "A **rectified stereo pair** is two images taken side by side and aligned so that every scene point appears on the *same row* "
        "in both. The horizontal offset between the two appearances is the **disparity** `d`, in pixels: the left-image pixel `(x, y)` "
        "shows the same point as the right-image pixel `(x − d, y)`. Near surfaces have large disparity and far ones small, and with a "
        "calibrated rig depth follows as `focal length × baseline / d`. Stereo matching estimates `d` for every left-image pixel.\n\n"
        "**RAFT-Stereo** (Lipson, Teed and Deng, 2021) does this with a recurrent network: two convolutional encoders turn each image "
        "into features at 1/4 resolution, a **correlation volume** compares every left feature with every right feature on the same "
        "row, and a multi-level GRU starts from zero disparity and refines it over a chosen number of **iterations**, each time "
        "looking up the correlation around its current estimate; a learned convex upsampling returns full resolution. The upstream "
        "code is not on PyPI, so this notebook carries the upstream `core/` package verbatim (MIT licence, commit `6e93ed2`) and "
        "verifies every file's SHA-256 before importing it. It uses upstream's pure-PyTorch correlation (`corr_implementation = "
        "'reg'`), so the optional CUDA sampler extension is never built and the same code runs on a GPU or a CPU.\n\n"
        "**The default path really adapts the model.** The tutorial data are rendered in code — slanted textured backgrounds with "
        "textured objects in front, a right camera with a different gain and offset, and sensor noise — so the ground-truth disparity "
        "is exact and the data carry no licence or download. The checkpoint was fine-tuned on Middlebury photographs; Section 8 adapts "
        "it to this rendered domain with a short fine-tune, and Sections 9–11 measure, use, export and reload the result. Every number "
        "is measured in this runtime; none is copied from the paper."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) state the stereo task as *rectified pair → disparity of the left image* and explain "
        "what a disparity value means geometrically (Section 4); (2) explain what the refinement iterations do and use two probes "
        "with exact answers to check a model's sign and scale (Sections 4–5); (3) explain why a dataset is validated and split "
        "before any model runs, and why baselines are configured on the training split only (Sections 6–7); (4) compare end-point "
        "error (EPE) with bad-pixel rates and say what each hides (Sections 7 and 9); (5) read a training loss as optimisation "
        "evidence and the held-out comparison as task evidence (Sections 8–9); (6) interpret the adapted model's disparity on unseen "
        "pairs and check that an exported adapter reproduces the trained model (Sections 10–11); and (7) predict, then measure, what "
        "unfreezing the encoders changes when both runs start from the same checkpoint (Section 14). Along the way the notebook "
        "builds a locked runtime, verifies carried upstream code and a pinned checkpoint, and writes machine-readable outputs with "
        "provenance."
    ),
    "exclusions": (
        "accuracy on real photographs or any deployment rig (the default data are rendered, and the held-out score is measured on "
        "8 rendered pairs); benchmark results (no KITTI, Middlebury, ETH3D or SceneFlow evaluation is run, and the paper's numbers are "
        "not reproduced here); depth in metres (no camera calibration is used); stereo rectification of unrectified images; "
        "confidence or uncertainty estimates (RAFT-Stereo outputs none); upstream's full training recipe (OneCycle schedule, "
        "augmentation, mixed precision, 200,000 steps on two GPUs); and the faster CUDA correlation kernels."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime: Google Colab or Kaggle with a **T4 GPU** (the documented runtime), or a Linux Jupyter kernel. The kernel's own Python version does not matter: the notebook installs nothing into it and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (torch 2.14.0, whose Linux wheel is the CUDA 13.0 build). A **CPU-only runtime also completes the default path, but slowly**: in a local 4-thread CPU check, one 32-iteration inference on a 320 × 256 pair took about 40 s and one 12-iteration training step on two pairs about 69 s, so a CPU run takes on the order of an hour or more (an estimate). Hosted-runtime times have not been measured yet for this revision. Everything runs in float32. About 1 GiB of disk is needed for the checkpoint download and about 8 GiB for the isolated environment.",
        "- **Learner:** basic Python and NumPy, and Colab or Jupyter familiarity. No prior experience with stereo vision or fine-tuning is assumed; disparity, rectification, occlusion, the correlation volume, refinement iterations, EPE, bad-pixel rates and SGBM are explained where they first appear and again in the glossary.",
        "- **Model and code:** the model code is upstream `princeton-vl/RAFT-Stereo` `core/` at commit `6e93ed2169bd858dbb43033988563f3b0bb49506` (MIT), carried verbatim and verified file by file. The checkpoint `raftstereo-middlebury.pth` is a pickle-based PyTorch `state_dict` from upstream's `models.zip`; it is loaded only after its byte size and SHA-256 match the carried manifest, and only with `torch.load(..., weights_only=True)`. No remote code is fetched or executed.",
        "- **Data contract:** a record is a rectified pair (two images of one size, each side 64..1280 px, at most 1280 × 1024 pixels) with a ground-truth disparity map for the **left** image in pixels and an optional valid mask. Disparity files may be `.pfm` (Middlebury, ETH3D; `inf` = unknown), `.npy`, or 16-bit `.png` (KITTI convention, value / 256, 0 = unknown); ground truth at or above 512 px is not scored, and a negative disparity is refused (it usually means left and right are swapped). For BYOD, `pairs.json` lists `{{\"left\", \"right\", \"disparity\", \"valid\"?, \"group\"?}}` objects with file names relative to the folder.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — images of identifiable people, private property or client sites are exactly that. Uploaded files are written under this run's directory and are not sent to any service. The default path uploads nothing.",
    ],
    "dependencies_sentence": (
        "the upstream RAFT-Stereo checkpoint archive (`models.zip`, linked from upstream's `download_models.sh`, on Dropbox, no "
        "credential): only the `raftstereo-middlebury.pth` member is kept, and its {size_text}"
    ),
    "external_access": (
        "- **External access:** Dropbox, to fetch upstream's `models.zip` over HTTPS without a credential (the link in upstream "
        "`download_models.sh` at commit `6e93ed2`); only the `raftstereo-middlebury.pth` member is extracted and the archive is deleted. "
        "PyPI (`files.pythonhosted.org`), for the pinned `uv` wheel and the {n_locked} hash-locked packages; and the managed CPython "
        "build (python-build-standalone) that `uv` downloads for the isolated environment. The tutorial data are rendered in code, so "
        "no dataset is downloaded. No repository clone and no credentials are required; nothing is installed from this repository."
    ),
    "pin_notes": {
        "unpinned": (
            "**Pin state of this revision: not yet pinned.** The checkpoint's SHA-256 and byte size have not been recorded yet "
            "(`MODEL_REVISION = 'unpinned'`), so the `weights` stage stops here with a `RuntimeError` that says so, and the "
            "learner sections cannot run. A maintainer pins it once with `python tools/pin_snapshot.py` where the archive is "
            "reachable, then regenerates this notebook; see the repository's `docs/release-verification.md`."
        ),
        "pinned": (
            "**Pin state of this revision: pinned.** `MODEL_REVISION` is the SHA-256 of the checkpoint file, recorded by the "
            "repository's `tools/pin_snapshot.py` together with its byte size."
        ),
    },
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab or Jupyter), run cells in order and "
                "read short Python, and who want to see how a pretrained stereo network is evaluated honestly and adapted to a new "
                "domain. No prior stereo-vision experience is assumed: each term is explained where it is first needed and the glossary "
                "below collects them. A T4 GPU is recommended; a CPU-only runtime works but is slow. The **Prerequisites** give the "
                "details.\n\n"
                "**Running it.** In Colab, choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default path "
                "needs no edit, no upload, no account, no token and no runtime restart. Sections 1–3 build an isolated environment from "
                "hash-locked packages and download the checkpoint archive, so they take the longest before any model runs; read ahead "
                "while they finish.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls "
                "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated environment's "
                "Python, streams what it prints, and stops the notebook with the stage's own error message if it fails. Stages hand "
                "results to each other only through files in the run directory — the verified checkpoint, the recorded split, the "
                "adapter and JSON records — so each stage that needs a model loads it afresh from verified files.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–13) are the machine-learning workflow; each runs one stage and prints "
                "compact dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the runtime check, the carried code, the "
                "isolated install and the checkpoint staging) are collapsed and titled **Infrastructure**. You may run them without "
                "studying their implementation: they exist for reproducibility and provenance, not as prerequisite machine-learning "
                "knowledge.\n\n"
                "**Form controls.** Some learner cells start with fields that Colab renders as a form: `VALID_ITERS` (Section 4); "
                "`N_PAIRS`, `DATASET_SEED`, `HOLDOUT`, `SEED` and `EPOCHS` (Section 6); `LEARNING_RATE`, `BATCH_SIZE`, `TRAIN_ITERS` and "
                "`FREEZE_ENCODERS` (Section 8); `NEW_DATA_SEED` (Section 10); and the BYOD switches and locations (Section 13). Leave "
                "them at their defaults for the first run. Changing a field and choosing *Runtime → Run after* from that cell is how you "
                "experiment afterwards.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation "
                "practice]** — how the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and "
                "packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
                "expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** block gives "
                "worked guidance. Write your own answer first, then open it. The notes describe the shape of a normal result rather than "
                "fixed numbers: this revision has no recorded hosted run, and exact values vary with the runtime."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Inference** | a rectified pair (left, right), same size; `VALID_ITERS` | padded to a multiple of 32 px → feature and "
                "context encoders (1/4 resolution) → row-wise correlation volume → multi-level GRU refining the disparity `VALID_ITERS` "
                "times → convex upsampling → cropped back | a disparity map for the left image, (H, W) float32 pixels, larger = nearer |\n"
                "| **Adaptation** | 24 training pairs with exact disparity | the same network, BatchNorm frozen, encoders frozen by default; "
                "upstream's sequence loss on every refinement | updated weights, exported as a SafeTensors adapter |\n"
                "| **Evaluation** | 8 held-out pairs never used for training or for any choice | the adapted model next to three baselines, "
                "all scored on the same valid pixels | EPE and bad-pixel rates per pair and on average |\n\n"
                "The network outputs no confidence: every pixel gets a disparity, whether or not the scene allows a match there.\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux, GPU (optional) and disk checked; a fresh run directory | the GPU name or the CPU note |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions, `cuda` |\n"
                "| 3. Pin, stage and verify | [Engineering] | upstream source and checkpoint digest-checked | the verified file counts |\n"
                "| 4. The pretrained model on one pair | [Concept] | disparity of a rendered pair, next to SGBM | EPE, bad-pixel rates, the figure |\n"
                "| 5. Probes with exact answers | [Evaluation practice] | identical images and an 8 px shift | the predicted medians |\n"
                "| 6. Dataset, validation and split | [Evaluation practice] | 32 pairs validated, split 24 / 8, refusals shown | split sizes, refusals |\n"
                "| 7. Baselines on the held-out pairs | [Evaluation practice] | median constant, SGBM, pretrained network | the baseline table |\n"
                "| 8. Bounded fine-tuning | [Concept] | a short adaptation from the checkpoint | the epoch losses |\n"
                "| 9. Held-out comparison | [Evaluation practice] | the exported adapter scored in a fresh process | the principal result |\n"
                "| 10. Unseen pairs | [Concept] | pretrained vs adapted on three new scenes; disparity files | per-pair EPE, the figure |\n"
                "| 11. Fresh reload | [Engineering] | the adapter rebuilt from files in a new process | reload parity |\n"
                "| 12. Outputs and provenance | [Engineering] | the result record and the files written | identity, digests |\n"
                "| 13. Optional BYOD | [Evaluation practice] | your own pair or labelled folder (off by default) | your results |\n"
                "| 14. Change one thing | [Concept] | unfreeze the encoders and compare | the run history |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 4, 7 and 9 and the interpretation: they carry the principal results."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "- **Rectified stereo pair:** two images from side-by-side cameras, warped so that each scene point lies on the same row in both; matching then only searches along a row (the *epipolar line*).\n"
                "- **Disparity:** the horizontal offset `d` in pixels between a point's position in the left image and in the right image (left `x` ↔ right `x − d`). Larger disparity means a nearer surface.\n"
                "- **Depth from disparity:** with a calibrated rig, depth = focal length (pixels) × baseline / disparity. This notebook stays in pixels.\n"
                "- **Occlusion and the valid mask:** a point seen by one camera but hidden from (or outside) the other has no true match. The rendered data mark such pixels invalid, and every metric scores only valid pixels (the non-occluded protocol).\n"
                "- **Correlation volume:** for every left-image feature, its similarity with every right-image feature on the same row; RAFT-Stereo looks values up around its current estimate.\n"
                "- **Refinement iterations:** the number of GRU updates applied to the disparity estimate (`VALID_ITERS` at inference, `TRAIN_ITERS` while training); upstream evaluates with 32.\n"
                "- **EPE (end-point error):** the mean absolute disparity error in pixels over valid pixels. Sensitive to a few very large errors.\n"
                "- **Bad-pixel rate (bad-1 / bad-2 / bad-3):** the share of valid pixels whose error exceeds 1, 2 or 3 px (ETH3D reports bad-1, Middlebury bad-2). It counts failures and ignores how large they are.\n"
                "- **D1:** KITTI's outlier rate: error above 3 px **and** above 5 % of the true disparity.\n"
                "- **SGBM:** OpenCV's semi-global block matching (Hirschmüller, 2008), a classical, training-free matcher; pixels it cannot match are filled from a row neighbour here.\n"
                "- **Median baseline:** predicting the training split's median disparity everywhere — the floor any method has to beat.\n"
                "- **Fine-tuning / frozen encoders:** continuing gradient training of the pretrained network on new data; with the encoders frozen, the feature and context encoders keep their weights and only the update block and heads train.\n"
                "- **Sequence loss:** upstream's training loss: the L1 error of every refinement's estimate, with later iterations weighted more.\n"
                "- **Held-out split:** pairs never shown to the optimiser and never used to choose a setting; the task evidence.\n"
                "- **Adapter / reload equivalence:** the SafeTensors file of the tensors the fine-tune could change, and the check that a fresh model built from the checkpoint plus the adapter returns the same disparity.\n"
                "- **Digest (SHA-256):** a fingerprint of a file's bytes; a single changed byte changes it.\n"
                "- **Hash-locked environment:** a separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests; the installer refuses anything else.\n"
                "- **Stage:** one step of the workflow run as its own process by `run_stage`; it reads the files earlier stages wrote and writes its own.\n"
                "- **BYOD:** Bring Your Own Data — the optional Section 13 branches that run the same stages on your own pairs.\n\n"
                "</details>"
            ),
        ],
    },
    "carried": {
        f"src/{PKG}/__init__.py": f"src/{PKG}/__init__.py",
        f"src/{PKG}/samples.py": f"src/{PKG}/samples.py",
        f"src/{PKG}/pipeline.py": f"src/{PKG}/pipeline.py",
        **{f"src/{PKG}/third_party/raft_stereo/{name}": f"src/{PKG}/third_party/raft_stereo/{name}" for name in UPSTREAM_FILES},
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/raftstereo-middlebury/dimer-base-manifest.json": "weights/raftstereo-middlebury/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    # The fleet's pinned uv wheel (as kandinsky-generation-pipeline and detr-detection-pipeline): size and SHA-256 checked.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 1.0, "environment": 8.0},
    "runtime_modules": ["torch", "numpy", "scipy", "cv2"],
    "weights_key": "raftstereo-middlebury",
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64, looks for a CUDA GPU, checks the free "
                "disk, and creates a new run directory. **Output:** the GPU name (or a note that the stages will run on the CPU) and "
                "the directories this run uses. Each run writes to a new directory under `outputs/{stem}/`, so an earlier export cannot "
                "be mistaken for a current result. The verified checkpoint is kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the GPU (for example `Tesla T4, 15360 MiB`) or `None` after a CPU note, the "
                "kernel's Python version, the run directory, the weights directory, the isolated environment's directory and the free "
                "disk. If the cell stops with a platform or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the code this notebook runs, carried so "
                "that the notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text: the package's three modules under `src/raft_stereo_pipeline/` (identity constants, "
                "checkpoint staging and verification, validation, metrics, baselines, the pipeline class and the rendered samples); "
                "the upstream RAFT-Stereo `core/` files and their MIT licence under `src/raft_stereo_pipeline/third_party/raft_stereo/`, "
                "byte for byte as published at commit `6e93ed2`; the stage runner `tutorial_stages.py`; the hash-locked "
                "`requirements.txt` ({n_locked} packages); the checkpoint manifest; and the repository licence. It writes each file "
                "into the run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on any mismatch. The repository's "
                "parity test (`tests/test_notebook_parity.py`) fails whenever the carried text and the repository diverge, and the "
                "package checks the upstream files again against its own `UPSTREAM_SHA256` before importing them. Nothing in this cell "
                "runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer wheel, "
                "refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython 3.12.12, and "
                "installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must be the locked "
                "version, a prebuilt wheel, and match a locked digest (no source build happens, so no unpinned build dependency is "
                "fetched). The hosted runtime's own packages are never replaced, which is why no restart is needed. Stage processes get "
                "`MPLBACKEND=Agg` and no Hugging Face token; they write figures to files that the kernel displays. The cell also defines "
                "`run_stage`, `load_record` and `show_image`, the three helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few minutes "
                "(the CUDA build of PyTorch is the largest download). A failed download or a hash mismatch stops the cell; never remove "
                "a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), the "
                "`torch`, `numpy`, `scipy` and `cv2` versions, `cuda` (`True` on a GPU runtime), the number of locked "
                "packages and the setup time. On a CPU runtime a note says the stages will run on the CPU."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the model · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It verifies the carried upstream code, downloads the checkpoint "
                "archive once and checks the checkpoint's size and SHA-256; you may run it without studying its implementation.\n\n"
                "Two things are verified before any model exists. First, every carried upstream file is hashed against `UPSTREAM_SHA256` "
                "(the code identity: `princeton-vl/RAFT-Stereo` at `6e93ed2169bd858dbb43033988563f3b0bb49506`). Second, the checkpoint "
                "identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in `pipeline.py` and the manifest — and the `weights` stage "
                "checks they agree, downloads upstream's `models.zip` only if the checkpoint is absent, extracts the single member "
                "`raftstereo-middlebury.pth` (nothing else is unpacked; the archive is deleted), and re-hashes it against the manifest, "
                "raising on the first size or digest mismatch. A URL-hosted file has no commit, so the pinned revision of this checkpoint "
                "is the SHA-256 of its own bytes. There is no fallback to a different file, and every later stage verifies the checkpoint "
                "again before `torch.load(..., weights_only=True)` reads it.\n\n"
                "{pin_note}"
            ),
            "after": (
                "**What to notice:** the upstream repository, commit and `carried_files_verified: 8`; the model id, revision, licence "
                "(`mit`) and file count; a `fetched` list (the checkpoint on a first run, empty on a rerun, because staging only fetches "
                "absent files); and the verified file count. A size or SHA-256 mismatch stops the cell with a `ValueError` naming the "
                "file — see **Troubleshooting**, and never edit a manifest to get past one."
            ),
        },
    ],
    "cells": [
        # ---------------------------------------------------------------- 4. demonstration pair
        {
            "md": (
                "## 4. The pretrained model on one rendered pair · [Concept]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`; its printed dictionaries appear "
                "under the cell. This cell runs the `demo` stage on a rendered 320 × 224 pair (seed 7) whose disparity is known exactly.\n\n"
                "1. **Validate first.** `validate_inputs` checks the pair before any model runs — both images present, one size, each "
                "side within 64..1280 px — and writes `outputs/{stem}_input_manifest.json`, including a probe that is refused (a right "
                "image of another size) with a message naming the rule.\n"
                "2. **Run the network.** The pair is padded to a multiple of 32 px, the network refines its estimate `VALID_ITERS` times "
                "(upstream's default 32), and the disparity is cropped back to 320 × 224. The output's sign is flipped from upstream's "
                "internal horizontal flow (`flow = −disparity`), so positive values mean nearer.\n"
                "3. **Compare with a classical matcher.** OpenCV **StereoSGBM** searches the same rows with hand-designed matching costs "
                "and smoothness penalties, no training at all. Its search range here (64 px) comes from the sample's stated maximum "
                "disparity (40 px), not from this pair's ground truth.\n\n"
                "Both are scored with **EPE** (mean absolute error, px) and **bad-pixel rates** (share of valid pixels off by more than 1, "
                "2 or 3 px; `d1_all` is KITTI's version). On one pair the verdict is `sample-sanity`: geometry evidence, not a benchmark.\n\n"
                "**Predict before running:** which will have the lower EPE on this pair, RAFT-Stereo or SGBM? Where in the image do you "
                "expect the largest errors — object interiors, object edges, or low-texture surfaces?"
            ),
            "code": (
                'VALID_ITERS = 32  # @param {{type:"integer"}}\n\n'
                "run_stage('demo', '--iters', VALID_ITERS)\n"
                "show_image('{stem}_demo.png', 'Demonstration pair: inputs, ground truth, RAFT-Stereo, SGBM, RAFT-Stereo error')"
            ),
        },
        # ---------------------------------------------------------------- 5. probes
        {
            "md": (
                "**What to notice (Section 4):** the validation verdict and the refused probe; the two metric rows; SGBM's `density` "
                "(the share of pixels it matched itself before filling); and in the figure, where the red error pixels sit.\n\n"
                "<details><summary>Check your reasoning</summary>A pretrained RAFT-Stereo checkpoint usually beats SGBM on EPE and on "
                "bad-pixel rates for textured rendered scenes, because it matches learned features over a wide context and its "
                "iterations smooth surfaces without blurring their edges. Expect both methods' errors to concentrate along object "
                "boundaries (where a disparity jump meets occlusion) and on the deliberately low-texture objects, where neither method "
                "has anything to match; SGBM's unmatched pixels there are filled from a row neighbour, which is often wrong. If your "
                "run shows the opposite ordering, that is a finding about this checkpoint on this rendered domain — note it; it is "
                "what Sections 7–9 measure properly on held-out pairs.</details>\n\n"
                "## 5. Probes with exact answers: identical images and an 8 px shift · [Evaluation practice]\n\n"
                "Before trusting any score, check that the model's output means what you think it means. Two random-dot pairs have an "
                "answer that is known exactly and needs no rendering model at all:\n\n"
                "- **identical images**: every point sits at infinity, so the true disparity is **0** everywhere;\n"
                "- **a shift of 8 px**: the right image is the left image moved 8 px, so the true disparity is **8** everywhere (the left "
                "8 columns have no match and are not scored).\n\n"
                "A wrong sign would show up as a median near −8, a scale error as a median far from 8, and a bias as a non-zero median on "
                "the identical pair. Random dots are not natural images, so a small error here says nothing about scenes; a large one "
                "says the pipeline is wired wrongly.\n\n"
                "**Predict before running:** what median disparity will the model report for each probe, and will its 5th–95th "
                "percentile range be narrow (under 1 px) or wide?"
            ),
            "code": "run_stage('probes', '--iters', VALID_ITERS)",
        },
        # ---------------------------------------------------------------- 6. dataset
        {
            "md": (
                "**What to notice (Section 5):** `predicted_median_px` against `true_disparity_px`, and the width of `predicted_p05_p95_px`.\n\n"
                "<details><summary>Check your reasoning</summary>A correctly wired pipeline reports a median close to 0 for the identical "
                "pair and close to 8 for the shifted pair, with a narrow percentile range: the probes confirm the sign (positive = the "
                "right image's match lies to the left) and the pixel scale. Random dots are an easy, perfectly textured case, so EPE "
                "well below 1 px is normal; a median near −8 would mean left and right were swapped, which is exactly the mistake "
                "`validate_dataset` guards against in Section 6.</details>\n\n"
                "## 6. Dataset, validation and split · [Evaluation practice]\n\n"
                "This cell runs the `prepare` stage. It renders `N_PAIRS` scenes (seed `DATASET_SEED`; scene `i` uses seed "
                "`DATASET_SEED × 100003 + i`), each with exact disparity and a valid mask, and validates every record **before any model "
                "runs**: matching image sizes within the ceilings, a disparity map of the image's shape, finite non-negative values on "
                "valid pixels, unique record ids. It then shows three refusals — a right image of another size, a disparity map of the "
                "wrong shape, and a negative (left/right-swapped) disparity — each named by the rule it breaks.\n\n"
                "The split is a seeded random 75 / 25 split (`HOLDOUT`, `SEED`): 24 training and 8 held-out pairs. A random split is "
                "valid here because every scene is rendered independently; records that share a scene (several exposures or crops of one "
                "view) must be split by scene, which `pairs.json` can declare with a `group` field. The stage asserts that no record is "
                "on both sides, records the split and the dataset's SHA-256, and every later stage rebuilds exactly these records and "
                "refuses to run if they changed. `EPOCHS` is set here because the dataset manifest records it.\n\n"
                "**Sample provenance and licence:** generated in code by this package's `samples.py`; no third-party data, no licence "
                "terms beyond the repository's. **Pretraining overlap:** the checkpoint was trained on SceneFlow (rendered) and "
                "fine-tuned on Middlebury photographs; these scenes are rendered at runtime and cannot be in either set, but they "
                "resemble SceneFlow's random-object style, so the domain is not entirely new to the network.\n\n"
                "**Predict before running:** what share of each pair's pixels will have a valid ground truth — nearly all, about nine in "
                "ten, or about half — and which pixels lose it?"
            ),
            "code": (
                'N_PAIRS = 32  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 20261004  # @param {{type:"integer"}}\n'
                'EPOCHS = 3  # @param {{type:"integer"}}\n\n'
                "run_stage('prepare', '--n-pairs', N_PAIRS, '--dataset-seed', DATASET_SEED, '--holdout', HOLDOUT, '--seed', SEED, '--epochs', EPOCHS)"
            ),
        },
        # ---------------------------------------------------------------- 7. baselines
        {
            "md": (
                "**What to notice (Section 6):** `n_records`, the image size, `mean_valid_fraction` (the share of pixels with a visible "
                "match), `max_disparity_px`, the split sizes with `shared_ids: 0`, and the three refusals.\n\n"
                "<details><summary>Check your reasoning</summary>The renderer marks a pixel valid only when its surface point is inside "
                "the right image and not hidden there by a nearer object, so roughly nine pixels in ten stay valid: the losses are the "
                "left-edge strip whose matches fall outside the right frame, and the bands beside object edges that the right camera "
                "cannot see. Those pixels have a true disparity but no match, and every metric in this notebook leaves them out (the "
                "non-occluded protocol).</details>\n\n"
                "## 7. Three baselines on the held-out pairs · [Evaluation practice]\n\n"
                "A score means little without something to compare it with. The `baselines` stage scores the 8 held-out pairs three "
                "ways, on exactly the same valid pixels:\n\n"
                "- **median constant** — the training split's median disparity, predicted everywhere. Any method that does not beat it "
                "has learned nothing about the scene.\n"
                "- **SGBM** — classical semi-global matching. Its search range is the training split's largest disparity plus 8 px, "
                "rounded up to a multiple of 16; nothing about the held-out pairs is used to configure it.\n"
                "- **pretrained RAFT-Stereo** — the verified checkpoint before any adaptation, at `VALID_ITERS` iterations. This is the "
                "baseline the fine-tune has to beat.\n\n"
                "All three are fitted or configured on the training split only, so the held-out pairs stay untouched evidence. The "
                "averages give each pair the same weight; there is one pass and no dispersion estimate, so differences of a few "
                "hundredths of a pixel on 8 pairs are not resolved.\n\n"
                "**Predict before running:** order the three by held-out EPE. Will the ordering by `bad_3px` be the same?"
            ),
            "code": "run_stage('baselines', '--iters', VALID_ITERS)",
        },
        # ---------------------------------------------------------------- 8. fine-tune
        {
            "md": (
                "**What to notice (Section 7):** the three rows, and whether EPE and `bad_3px` rank them the same way; SGBM's mean density.\n\n"
                "<details><summary>Check your reasoning</summary>The median constant should be far behind both matchers — it ignores the "
                "images entirely. Between SGBM and the pretrained network, expect the network to lead on EPE and on the bad-pixel rates, "
                "but the two measures need not agree: SGBM's filled pixels can be badly wrong (inflating EPE) while most of its matched "
                "pixels are within 3 px, and a network can be slightly off almost everywhere (raising `bad_1px`) while rarely failing "
                "badly. That is why this notebook reports both: EPE weighs *how large* errors are, bad-pixel rates count *how many* "
                "fail.</details>\n\n"
                "## 8. Bounded stereo fine-tuning · [Concept]\n\n"
                "This cell runs the real adaptation in this runtime (`adapt` stage). It is **gradient fine-tuning** of the pretrained "
                "network — not a new head, not PEFT — with upstream's training loss: the L1 error between the true horizontal flow "
                "(`−disparity`) and the estimate after **every** refinement iteration, later iterations weighted more (`gamma` 0.9, "
                "adjusted to the iteration count), valid pixels only.\n\n"
                "- **What trains.** BatchNorm layers are frozen (upstream's `freeze_bn`). With `FREEZE_ENCODERS = True` the feature encoder "
                "(`fnet`) and context encoder (`cnet`) also keep their weights; the context projections, the multi-level GRU update block, "
                "the disparity head and the upsampling mask train. The cell prints the trainable and total parameter counts.\n"
                "- **Schedule.** `EPOCHS` epochs over the 24 training pairs, batch size `BATCH_SIZE`, AdamW at a constant `LEARNING_RATE` "
                "(2e-5, upstream's Middlebury fine-tuning rate; weight decay 1e-5, eps 1e-8), `TRAIN_ITERS` refinement iterations per "
                "step (upstream trains with 22; 12 keeps the tutorial short), gradient-norm clipping at 1.0, float32, seed `SEED`, no "
                "augmentation. Upstream's full recipe (OneCycle schedule, augmentation, mixed precision, 200,000 steps) is out of scope.\n"
                "- **Every run starts from the checkpoint.** The stage loads the verified checkpoint afresh in its own process, so "
                "re-running this cell after changing a field compares the new setting from the same starting point instead of training "
                "further.\n\n"
                "After training, the stage records the in-memory model's disparity on one held-out pair (the reference for Section 11) "
                "and exports `outputs/{stem}_adapter.safetensors`. **Read the loss as optimisation evidence only:** a falling loss says "
                "the optimiser is fitting the training pairs; Section 9 is the task evidence.\n\n"
                "**Predict before running:** will the mean loss fall from epoch 1 to epoch `EPOCHS`? By roughly how much — a few percent, "
                "a third, or most of it?"
            ),
            "code": (
                'LEARNING_RATE = 2e-5  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 2  # @param {{type:"integer"}}\n'
                'TRAIN_ITERS = 12  # @param {{type:"integer"}}\n'
                'FREEZE_ENCODERS = True  # @param {{type:"boolean"}}\n\n'
                "run_stage('adapt', '--epochs', EPOCHS, '--batch-size', BATCH_SIZE, '--lr', LEARNING_RATE, '--train-iters', TRAIN_ITERS,\n"
                "          '--freeze-encoders', int(FREEZE_ENCODERS), '--seed', SEED, '--iters', VALID_ITERS)"
            ),
        },
        # ---------------------------------------------------------------- 9. evaluate
        {
            "md": (
                "**What to notice (Section 8):** the `start` line (the checkpoint, freshly verified), the epoch losses, the trainable "
                "share of the parameters, and the adapter's size and tensor count.\n\n"
                "<details><summary>Check your reasoning</summary>The pretrained network already fits these scenes reasonably, so the loss "
                "starts moderate and typically falls by a modest fraction over three short epochs — not to zero. With the encoders "
                "frozen, roughly half of the 11.1 million parameters train (the encoders hold the rest), and the adapter holds only the "
                "tensors that could change. A loss that rises, or turns into `nan`, points at the learning rate; see "
                "Troubleshooting.</details>\n\n"
                "## 9. Held-out comparison: the exported adapter, in a fresh process · [Evaluation practice]\n\n"
                "The `evaluate` stage does not reuse the trained model in memory. It starts a new process, rebuilds the network from the "
                "verified checkpoint **plus the exported adapter file**, and scores the same 8 held-out pairs with the same `VALID_ITERS` "
                "as the baselines (it refuses to compare against baselines scored with a different iteration count). The table puts "
                "the adapted model under the three baselines; a per-pair count says on how many held-out pairs the adapted EPE is lower "
                "than the pretrained EPE; and a **run history** keeps one row per evaluation, so the Section 14 activity prints side by "
                "side with this run.\n\n"
                "**What these numbers are.** Tutorial metrics from one pass over 8 rendered pairs, with no dispersion estimate. No "
                "setting was chosen on the held-out pairs, so they are an honest check of this run; they are not evidence about real "
                "photographs or about other rigs.\n\n"
                "**Predict before running:** will the adapted EPE be lower than the pretrained EPE? On all 8 pairs, or only on most?"
            ),
            "code": "run_stage('evaluate', '--iters', VALID_ITERS)",
        },
        # ---------------------------------------------------------------- 10. new data
        {
            "md": (
                "**What to notice (Section 9):** the `adapted` row against the `pretrained` row for every column, the per-pair count, and "
                "the run history's row 0.\n\n"
                "<details><summary>Check your reasoning</summary>A short fine-tune on the target domain usually lowers held-out EPE and "
                "the bad-pixel rates a little, because the update block learns this domain's textures, gain difference and noise. Three "
                "things can legitimately differ from that: the improvement may be small compared with pair-to-pair spread (read the "
                "per-pair count, not only the mean); one metric may improve while another does not; and on some pairs the adapted model "
                "can be worse. With 8 held-out pairs and one run, report the direction and the count, not a precise gain. If the adapted "
                "model is worse overall, that is a negative result worth keeping — it is preserved in the outputs, not replaced.</details>\n\n"
                "## 10. Inference on unseen pairs · [Concept]\n\n"
                "Three new scenes come from a seed the dataset never used (`NEW_DATA_SEED = 99`), so neither the fine-tune nor any choice "
                "saw them. The `infer` stage runs the pretrained and the adapted model on each, prints per-pair metrics for both, and "
                "**exports the adapted disparity maps** — `outputs/{stem}_disparity/<pair>.npy` (float32 pixels, left image) and a 16-bit "
                "PNG in the KITTI convention (value / 256) — plus `outputs/{stem}_new_pairs.csv` and a figure (left image, ground truth, "
                "pretrained, adapted; one colour scale per row, warm = near).\n\n"
                "**Predict before running:** on how many of the three unseen pairs will the adapted model have the lower EPE?"
            ),
            "code": (
                'NEW_DATA_SEED = 99  # @param {{type:"integer"}}\n\n'
                "run_stage('infer', '--new-seed', NEW_DATA_SEED, '--iters', VALID_ITERS)\n"
                "show_image('{stem}_new_pairs.png', 'Unseen pairs: left, ground truth, pretrained, adapted')"
            ),
        },
        # ---------------------------------------------------------------- 11. reload
        {
            "md": (
                "**What to notice (Section 10):** `adapted_epe_lower_on`, each pair's two metric rows, and in the figure whether the "
                "adapted maps differ visibly from the pretrained ones (usually only at edges and on low-texture objects).\n\n"
                "<details><summary>Check your reasoning</summary>Three pairs are a demonstration, not an evaluation: the count can be "
                "3/3, 2/3 or worse even when the held-out average improved, because single scenes vary a lot (a large low-texture "
                "object can dominate a pair's EPE). The disparity files are the deliverable of this stage: machine-readable maps that a "
                "downstream user could convert to depth with their own calibration.</details>\n\n"
                "## 11. Fresh reload and equivalence check · [Engineering]\n\n"
                "Exporting a file is not the same as exporting the model. The `reload` stage starts yet another process, reads the "
                "adapter's provenance header (format, base model and revision, upstream commit, architecture arguments, frozen "
                "prefixes) and refuses one that does not fit, rebuilds the network from the verified checkpoint plus the adapter, and "
                "compares its disparity on the reference held-out pair with the disparity the **trained in-memory model** produced in "
                "Section 8, within a stated tolerance (mean absolute difference ≤ 0.001 px, maximum ≤ 0.01 px); it also checks that "
                "the held-out EPE measured in memory equals the fresh-process value of Section 9. Loading without error is not the "
                "check; reproducing the output is. The stage then writes `outputs/{stem}_result.json` with the provenance."
            ),
            "code": "run_stage('reload')",
        },
        # ---------------------------------------------------------------- 12. outputs
        {
            "md": (
                "**What to notice (Section 11):** `equivalent: True`, the two differences against their tolerances, and the list of "
                "files written.\n\n"
                "## 12. Outputs and provenance · [Engineering]\n\n"
                "Everything the stages wrote is under this run's `outputs/` directory:\n\n"
                "- `{stem}_input_manifest.json` and `{stem}_evaluation_report.json` — the demonstration pair's validation and report;\n"
                "- `probes.json` — the two exact-answer probes;\n"
                "- `{stem}_dataset.json` — dataset manifest, SHA-256, split ids and the refusal probes;\n"
                "- `{stem}_baselines.json`, `{stem}_adaptation.json`, `{stem}_held_out_evaluation.json` — baselines, fine-tuning "
                "configuration and losses, and the held-out comparison with per-pair rows and the run history;\n"
                "- `{stem}_new_pairs.json`, `{stem}_new_pairs.csv`, `{stem}_disparity/` — unseen-pair metrics and disparity maps;\n"
                "- `{stem}_adapter.safetensors` — the adapter (provenance in its header);\n"
                "- `{stem}_result.json` — identity (model, checkpoint revision, upstream commit), runtime versions, device, dataset "
                "digest, split, adaptation configuration, held-out table, artifact digest and reload parity;\n"
                "- `weights.json`, the PNG figures and `logs/` with every stage's full output.\n\n"
                "The next cell reads the result record in the kernel and prints its identity fields.\n\n"
                "**Expected result:** three dictionaries — the model id, the checkpoint revision, the upstream commit, the repository "
                "revision, the device and the dataset digest; the adapter's digest and size with `reload_equivalent: True`; and the "
                "held-out table — followed by the run directory."
            ),
            "code": (
                "result = load_record('{stem}_result.json')\n"
                "print({{'model': result['model']['id'], 'checkpoint_revision': result['model']['revision'], 'upstream_commit': result['upstream_code']['commit'],\n"
                "       'repository_revision': result['repository_revision'], 'device': result['model']['device'], 'dataset_sha256': result['dataset_sha256'][:16] + '...'}})\n"
                "print({{'artifact_sha256': result['artifact']['sha256'][:16] + '...', 'artifact_bytes': result['artifact']['bytes'], 'reload_equivalent': result['reload_parity']['equivalent']}})\n"
                "print({{'held_out_table': result['held_out_table']}})\n"
                "print('run directory:', ROOT)"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD) · [Evaluation practice]\n\n"
                "Both branches are off by default, so **Run all** never stops here. Read the contract before you switch one on:\n\n"
                "- **Pair branch** (`USE_BYOD_PAIR`): one **rectified** pair — two images of the same size, each side 64..1280 px, at "
                "most 1,310,720 pixels — and optionally its ground-truth disparity for the left image (`.pfm`, `.npy` or 16-bit KITTI "
                "`.png`). It runs `validate_inputs`, the pretrained model, SGBM and `evaluation_report` (`not-measurable` without ground "
                "truth) and writes `outputs/byod/byod_pair_*`. Unrectified photos (two phone shots, for example) violate the row "
                "assumption and give meaningless disparity. Through the upload dialog, upload two or three files whose names say which "
                "is which: `left`/`im0`, `right`/`im1`, and `disp` for the ground truth.\n"
                "- **Dataset branch** (`USE_BYOD_DATASET`): a folder (or one `.zip`) holding `pairs.json` — a list of `{{\"left\", "
                "\"right\", \"disparity\", \"valid\"?, \"group\"?}}` objects with paths relative to the folder — and the files it names; "
                "at least 2 independent groups. It runs the **same stages as the sample** — `prepare` (validate and split, by `group` "
                "when given) → `baselines` → `adapt` → `evaluate` → `infer` (on the held-out pairs) → `reload` — with the same fields as "
                "above (`HOLDOUT`, `SEED`, `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`, `TRAIN_ITERS`, `FREEZE_ENCODERS`, `VALID_ITERS`), "
                "and writes everything to `outputs/byod/` with a `byod_` prefix, so the sample results are kept. Pairs larger than "
                "448 × 320 px are cropped at a seeded random position for training only. Middlebury 2014 scenes at quarter or half "
                "resolution fit the ceilings; note that the checkpoint was fine-tuned on Middlebury, so scoring it there is not "
                "independent evidence.\n\n"
                "Set `BYOD_LEFT_PATH` and `BYOD_RIGHT_PATH` (and optionally `BYOD_DISPARITY_PATH`), or `BYOD_DATASET_DIR` (a folder or a "
                "`.zip`), to read from a mounted or local location; leave them empty on Colab to get an upload dialog. Uploads are "
                "written under this run's `byod/` directory and are not sent anywhere. A record that breaks the contract stops the cell "
                "with the validator's message naming the rule (wrong sizes, missing file, a path outside the folder, negative disparity, "
                "fewer than 2 groups). After changing a field here, re-run this cell only."
            ),
            "code": (
                'USE_BYOD_PAIR = False  # @param {{type:"boolean"}}\n'
                'BYOD_LEFT_PATH = \'\'  # @param {{type:"string"}}\n'
                'BYOD_RIGHT_PATH = \'\'  # @param {{type:"string"}}\n'
                'BYOD_DISPARITY_PATH = \'\'  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_DIR = \'\'  # @param {{type:"string"}}\n\n'
                "import shutil\n"
                "import zipfile\n\n"
                "BYOD_ROOT = ROOT / 'byod'\n"
                "BYOD_MAX_ARCHIVE_FILES = 4001\n"
                "BYOD_MAX_EXPANDED_BYTES = 4 * 1024**3\n\n\n"
                "def _upload_into(target):\n"
                "    \"\"\"Colab upload dialog into a fresh `target`, so an earlier upload never mixes with this one.\"\"\"\n"
                "    try:\n"
                "        from google.colab import files\n"
                "    except ImportError as exc:\n"
                "        raise RuntimeError('There is no upload dialog outside Google Colab: set the BYOD location fields to local paths.') from exc\n"
                "    shutil.rmtree(target, ignore_errors=True)\n"
                "    target.mkdir(parents=True)\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise ValueError('Nothing was uploaded: run the cell again and choose the files.')\n"
                "    for name, data in uploaded.items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    return target\n\n\n"
                "def _safe_unzip(archive, target):\n"
                "    \"\"\"Extract a BYOD .zip member by member into a fresh `target`, refusing unsafe paths and oversized archives.\"\"\"\n"
                "    shutil.rmtree(target, ignore_errors=True)\n"
                "    target.mkdir(parents=True)\n"
                "    with zipfile.ZipFile(archive) as bundle:\n"
                "        members = [m for m in bundle.infolist() if not m.is_dir()]\n"
                "        if len(members) > BYOD_MAX_ARCHIVE_FILES:\n"
                "            raise ValueError(f'{{Path(archive).name}} holds {{len(members)}} files, more than {{BYOD_MAX_ARCHIVE_FILES}}')\n"
                "        if sum(m.file_size for m in members) > BYOD_MAX_EXPANDED_BYTES:\n"
                "            raise ValueError(f'{{Path(archive).name}} expands to more than {{BYOD_MAX_EXPANDED_BYTES:,d}} bytes')\n"
                "        for member in members:\n"
                "            name = member.filename.replace('\\\\', '/')\n"
                "            parts = [p for p in name.split('/') if p not in ('', '.')]\n"
                "            if name.startswith('/') or ':' in name or '..' in parts:\n"
                "                raise ValueError(f'{{Path(archive).name}}: unsafe member path {{member.filename!r}}')\n"
                "            destination = target.joinpath(*parts)\n"
                "            destination.parent.mkdir(parents=True, exist_ok=True)\n"
                "            destination.write_bytes(bundle.read(member))\n"
                "    return target\n\n\n"
                "def _dataset_root(source):\n"
                "    \"\"\"The folder holding pairs.json: `source` itself, the inside of one .zip, or its single sub-folder that has one.\"\"\"\n"
                "    source = Path(source)\n"
                "    if source.is_file() and source.suffix.lower() == '.zip':\n"
                "        source = _safe_unzip(source, BYOD_ROOT / 'unzipped')\n"
                "    elif source.is_dir() and not (source / 'pairs.json').is_file():\n"
                "        archives = sorted(source.glob('*.zip'))\n"
                "        if len(archives) == 1:\n"
                "            source = _safe_unzip(archives[0], BYOD_ROOT / 'unzipped')\n"
                "    if (source / 'pairs.json').is_file():\n"
                "        return source\n"
                "    found = sorted(source.rglob('pairs.json'))\n"
                "    if len(found) != 1:\n"
                "        raise ValueError(f'expected one pairs.json under {{source}}, found {{len(found)}}: upload pairs.json and the files it names, or one .zip')\n"
                "    return found[0].parent\n\n\n"
                "def _pick(paths, keys, role):\n"
                "    \"\"\"The one uploaded file whose name contains one of `keys`.\"\"\"\n"
                "    chosen = [p for p in paths if any(k in p.stem.lower() for k in keys)]\n"
                "    if len(chosen) != 1:\n"
                "        raise ValueError(f'name exactly one uploaded file with {{\" or \".join(keys)}} for the {{role}}; got {{[p.name for p in chosen]}}')\n"
                "    return chosen[0]\n\n\n"
                "if USE_BYOD_PAIR:\n"
                "    if BYOD_LEFT_PATH and BYOD_RIGHT_PATH:\n"
                "        left_path, right_path = Path(BYOD_LEFT_PATH), Path(BYOD_RIGHT_PATH)\n"
                "        disparity_path = Path(BYOD_DISPARITY_PATH) if BYOD_DISPARITY_PATH else None\n"
                "    else:\n"
                "        uploaded = sorted(p for p in _upload_into(BYOD_ROOT / 'pair').iterdir() if p.is_file())\n"
                "        left_path, right_path = _pick(uploaded, ('left', 'im0'), 'left image'), _pick(uploaded, ('right', 'im1'), 'right image')\n"
                "        rest = [p for p in uploaded if p not in (left_path, right_path)]\n"
                "        disparity_path = _pick(rest, ('disp',), 'ground-truth disparity') if rest else None\n"
                "    pair_options = ['--namespace', 'byod', '--left', left_path, '--right', right_path, '--iters', VALID_ITERS]\n"
                "    if disparity_path is not None:\n"
                "        pair_options += ['--disparity', disparity_path]\n"
                "    run_stage('byod_pair', *pair_options)\n"
                "    show_image('byod/byod_pair.png', 'BYOD pair')\n"
                "else:\n"
                "    print('BYOD pair branch is off; set USE_BYOD_PAIR = True to run the pretrained model on your own rectified pair.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    dataset_dir = _dataset_root(BYOD_DATASET_DIR if BYOD_DATASET_DIR else _upload_into(BYOD_ROOT / 'dataset'))\n"
                "    for stage, options in (\n"
                "        ('prepare', ['--byod', dataset_dir, '--holdout', HOLDOUT, '--seed', SEED, '--epochs', EPOCHS]),\n"
                "        ('baselines', ['--iters', VALID_ITERS]),\n"
                "        ('adapt', ['--epochs', EPOCHS, '--batch-size', BATCH_SIZE, '--lr', LEARNING_RATE, '--train-iters', TRAIN_ITERS,\n"
                "                   '--freeze-encoders', int(FREEZE_ENCODERS), '--seed', SEED, '--iters', VALID_ITERS]),\n"
                "        ('evaluate', ['--iters', VALID_ITERS]),\n"
                "        ('infer', ['--iters', VALID_ITERS]),\n"
                "        ('reload', []),\n"
                "    ):\n"
                "        run_stage(stage, '--namespace', 'byod', *options)\n"
                "    byod_result = load_record('byod/byod_{stem}_result.json')\n"
                "    print({{'byod_held_out_table': byod_result['held_out_table'], 'reload_equivalent': byod_result['reload_parity']['equivalent']}})\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt RAFT-Stereo on your own labelled pairs.')"
            ),
        },
    ],
    "closing": (
        "## 14. Your turn — change one thing: unfreeze the encoders · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It re-runs the fine-tune with every parameter trainable, "
        "which takes longer than Section 8 did.\n\n"
        "1. **Predict:** with `FREEZE_ENCODERS = False`, will the held-out EPE go down further, stay about the same, or get worse than "
        "in row 0 of the Section 9 run history? Write your guess down.\n"
        "2. **Change:** in Section 8 set `FREEZE_ENCODERS = False`. Change nothing else.\n"
        "3. **Run:** select the Section 8 cell and choose **Runtime → Run after**. Section 8 loads the verified checkpoint afresh in a "
        "new process (its `start` line says so) and trains every parameter; Section 9 adds row 1 to the run history; Sections 10–12 "
        "re-run on the new adapter.\n"
        "4. **Observe:** in Section 8, the trainable parameter count rises to all 11,116,176 and the adapter grows (it now carries the "
        "encoders). In Section 9 compare rows 0 and 1: `adapted_epe`, `adapted_bad_3px`, `final_loss` and `train_seconds`.\n"
        "5. **Explain:** in one sentence, why can training more parameters lower the training loss without lowering the held-out EPE?\n\n"
        "<details><summary>Check your reasoning</summary>Unfreezing the encoders gives the optimiser more capacity, so the training loss "
        "often falls further; whether the held-out EPE follows depends on whether the encoders' changes generalise from 24 rendered "
        "pairs, and with so few pairs they can also start to fit the training scenes' particular textures. Both runs start from the same "
        "checkpoint with the same seed, so the comparison is fair; but it is one run each on 8 held-out pairs, so a difference of a few "
        "hundredths of a pixel is unresolved, not a ranking of the two settings. This notebook has no recorded hosted run of this "
        "comparison: your run history is the measurement.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Start from your own run: the Section 7 baseline table, the Section 9 adapted row and per-pair count, and the Section 10 count.\n\n"
        "**What this notebook established, in this runtime.** The carried upstream RAFT-Stereo source was verified file by file against "
        "its SHA-256 digests at a pinned commit, and the checkpoint was verified against a committed manifest before loading. The "
        "pretrained model was run on a rendered pair with exact ground truth and on two probes with exact answers, and compared with "
        "classical SGBM. A 32-pair rendered dataset was validated and split; the held-out pairs were scored against a median constant, "
        "SGBM and the pretrained network, all configured on the training split only; a bounded fine-tune ran; the exported adapter was "
        "scored in a fresh process on the same held-out pairs; the adapted model produced disparity maps for unseen pairs; and a "
        "second fresh process rebuilt the model from files and reproduced the trained model's output within a stated tolerance.\n\n"
        "**Reading the metrics.** EPE weighs the size of errors and is pulled up by a few large failures; bad-pixel rates count failures "
        "above a threshold and ignore their size; neither alone characterises a disparity map. Every metric is computed on valid "
        "(non-occluded, in-frame) pixels only, so occluded regions — where every stereo method must guess — are not scored. RAFT-Stereo "
        "returns no confidence: it assigns a disparity to every pixel, including those with no possible match.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set, the "
        "carried upstream source and the pinned checkpoint together reproduce these stages in a fresh runtime, without the repository "
        "being cloned or installed and without any DIMER worker or service. It does **not** establish benchmark superiority, accuracy "
        "on real photographs or on any particular rig, or fitness for any deployment. The held-out scores are tutorial metrics from one "
        "pass over 8 rendered pairs with no dispersion estimate; the rendered scenes resemble the network's synthetic pretraining data, "
        "and the checkpoint was fine-tuned on Middlebury, so neither domain is independent of its training.\n\n"
        "**Reproducibility.** Seeds are form fields (`DATASET_SEED`, `SEED`, `NEW_DATA_SEED`); rendering, the split and the training "
        "order are seeded; the run is float32 without augmentation. GPU kernels are not forced to be deterministic, so repeated GPU "
        "runs, and GPU against CPU runs, can differ in the last digits of the losses and metrics.\n\n"
        "**More experiments (optional).** Each names the cell to change and where to re-run from.\n\n"
        "- **Fewer refinement iterations:** set `VALID_ITERS = 8` in Section 4 and choose **Runtime → Run after** from Section 4; "
        "compare the demonstration and held-out EPE with 32 iterations, and the time per stage in the logs.\n"
        "- **Longer training:** set `EPOCHS = 6` in Section 6 and **Run after** from Section 6.\n"
        "- **A harder split:** set `N_PAIRS = 16` in Section 6 (12 training pairs) and **Run after** from Section 6.\n"
        "- **Your own labelled pairs:** turn on `USE_BYOD_DATASET` in Section 13 and re-run that cell only.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64\".** The locked environment is built from manylinux wheels; use Google Colab, "
        "Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 prints the CPU note.** No GPU was found. The notebook still completes, slowly; for the documented runtime choose "
        "*Runtime → Change runtime type → T4 GPU* and **Run all** again.\n"
        "- **Section 2 fails while downloading, or reports a size or hash mismatch.** The `uv` wheel, the managed Python and the locked "
        "packages come from PyPI and python-build-standalone; run the cell again. A mismatch is refused on purpose — if it repeats, the "
        "download is being altered.\n"
        "- **Not enough disk.** The isolated environment needs about 8 GiB. Start a fresh runtime, or delete old `outputs/` run "
        "directories and the temporary `raft_stereo_env_*` folders.\n"
        "- **Section 3 says the checkpoint is not pinned.** This revision of the notebook was generated before the checkpoint's "
        "SHA-256 was recorded; use a revision of the notebook generated after pinning (see the repository's "
        "`docs/release-verification.md`).\n"
        "- **Section 3 fails to download or stops on a digest mismatch.** The archive comes from upstream's Dropbox link without a "
        "credential; run the cell again. A size or SHA-256 mismatch is never loaded: if it repeats, the upstream archive has changed "
        "and the notebook must be re-pinned by a maintainer. Delete `weights/` before retrying a partial download.\n"
        "- **Out of memory in Section 8 or 14.** Lower `BATCH_SIZE` to 1 or `TRAIN_ITERS` to 8 in Section 8, or keep `FREEZE_ENCODERS = "
        "True`, and **Run after** from Section 8.\n"
        "- **The loss turns into `nan` or rises steadily.** Lower `LEARNING_RATE` (for example to 1e-5) and re-run from Section 8.\n"
        "- **A stage stops with \"... is missing: run the stage that writes it\".** A cell was run out of order; run the notebook from "
        "Section 4 again (Sections 1–3 need not be repeated).\n"
        "- **Section 9 refuses because the baselines used another iteration count.** You changed `VALID_ITERS` after Section 7 ran; "
        "**Run after** from Section 7.\n"
        "- **BYOD is refused.** Each refusal names the rule: images of different sizes, a side outside 64..1280 px, a disparity map of "
        "the wrong shape, a negative disparity (swap left and right), a file missing from the folder, a path with `..`, an "
        "unsupported disparity format, or fewer than 2 independent groups. Fix the data and re-run Section 13.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. On the demonstration pair, RAFT-Stereo's EPE was ___ px and SGBM's ___ px; the probes reported medians of ___ and ___ px.\n"
        "2. On the 8 held-out pairs, the median constant, SGBM and the pretrained network had EPE ___, ___ and ___ px.\n"
        "3. After the fine-tune, the held-out EPE was ___ px (bad-3 ___); the adapted model was better than the pretrained one on ___ of 8 pairs.\n"
        "4. On the unseen pairs it was better on ___ of 3; the reload check reported a maximum difference of ___ px.\n"
        "5. An important failure mode I saw (where, and in which metric): ___.\n"
        "6. What I would need before using this adapter on real images from my own rig: ___.\n\n"
        "## References\n\n"
        "- Lipson, L., Teed, Z., & Deng, J. (2021). RAFT-Stereo: Multilevel recurrent field transforms for stereo matching. *International Conference on 3D Vision (3DV)*. [arXiv:2109.07547](https://arxiv.org/abs/2109.07547)\n"
        "- Teed, Z., & Deng, J. (2020). RAFT: Recurrent all-pairs field transforms for optical flow. *European Conference on Computer Vision (ECCV)*. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)\n"
        "- Hirschmüller, H. (2008). Stereo processing by semiglobal matching and mutual information. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 30*(2), 328–341. https://doi.org/10.1109/TPAMI.2007.1166\n"
        "- Mayer, N., Ilg, E., Häusser, P., Fischer, P., Cremers, D., Dosovitskiy, A., & Brox, T. (2016). A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. [arXiv:1512.02134](https://arxiv.org/abs/1512.02134)\n"
        "- Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., & Westling, P. (2014). High-resolution stereo datasets with subpixel-accurate ground truth. *German Conference on Pattern Recognition (GCPR)*. https://doi.org/10.1007/978-3-319-11752-2_3\n"
        "- Upstream repository: [princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) — MIT; code carried at commit `6e93ed2169bd858dbb43033988563f3b0bb49506`; checkpoint from its `download_models.sh`.\n"
        "- Repository model card: https://github.com/kurtvalcorza/raft-stereo-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/raft-stereo-pipeline`](https://github.com/kurtvalcorza/raft-stereo-pipeline) — source repository for this pipeline."
    ),
}
