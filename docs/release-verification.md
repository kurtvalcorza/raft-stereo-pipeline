# Release verification

`tutorials/raft_stereo_colab.ipynb` (`E2E` / `GUIDED`, **standalone** carrier, isolated hash-locked environment) is a
**release candidate** in initial development. The checkpoint is pinned (44,617,876 bytes, SHA-256 `d22e84c0…5819`), but the exact notebook revision has
not executed top-to-bottom in a clean hosted runtime. Unit tests, JSON validation, code-cell compilation, the generator
parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER
Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static and offline, every pull request)

CI (`.github/workflows/ci.yml`) installs CPU PyTorch and the other pins, then runs `ruff check src tests tools`, the
offline unit suite, `tools/validate_release_assets.py` and `tools/build_notebook.py --check`. No network weights are
used anywhere in CI.

`tools/validate_release_assets.py` checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or execution
  counts; no unresolved placeholder markers; every code cell follows an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version, the
  standalone carrier and the isolated environment; `metadata.dimer` declares that profile, spec `2.2`, `GUIDED`,
  `standalone: true`, `requires_dimer_worker: false` and `generated_from` (repository, revision, package-module SHA-256,
  third-party files, per-file hashes, generator);
- the carrier (ST1–ST8, SRC4, PAR1–PAR4): exactly one carrier cell whose `CARRIED_FILES` equal the repository files the
  template names (the three package modules, the eight upstream RAFT-Stereo files, `tools/tutorial_stages.py`,
  `tutorials/requirements-colab.lock.txt`, the checkpoint manifest, `LICENSE`) plus the generated `source.json`; every
  `CARRIED_HASHES` entry is the SHA-256 of its text and is recorded in the cell and notebook metadata; the cell writes
  each file and raises on a mismatch; the notebook is byte-identical (on LF) to the generator's output;
- the carried upstream source (MOD6, RC3): the files under `src/raft_stereo_pipeline/third_party/raft_stereo/` are
  exactly the `UPSTREAM_SHA256` set and match it; `UPSTREAM_COMMIT` is the commit cited in `README.md`,
  `MODEL_CARD.md` and `docs/WEIGHTS.md`;
- the lock (ENV1, ENV2): it pins every `pyproject.toml` pin at the same version, every entry carries `--hash=sha256:`,
  and its header records `--generate-hashes --only-binary :all:` and the manylinux x86_64 target;
- the isolated install (RUN10, ENV6, §25.13): no kernel cell runs `pip` except `uv pip install --python <isolated env>
  --require-hashes --only-binary :all:`; no `sys.executable`, `importlib`, `pip install`, `-m pip`, `import torch`,
  `import cv2` or `import numpy` in a kernel cell; kernel imports limited to the standard library, `IPython.display` and
  `google.colab`; downloads only in the install cell; the pinned `uv` wheel URL, size and SHA-256; `--managed-python`
  CPython 3.12.12; the Hugging Face token and `PYTHONPATH` removal; `MPLBACKEND='Agg'`; and a `run_stage` that re-raises
  the stage's own error message;
- the learner path: the stages called in order (`weights`, `demo`, `probes`, `prepare`, `baselines`, `adapt`,
  `evaluate`, `infer`, `reload`, then the gated `byod_pair`); every BYOD form field exactly as an empty-string or `False`
  `# @param` line in the one BYOD cell, whose dataset branch runs `prepare`, `baselines`, `adapt`, `evaluate`, `infer`
  and `reload` (DAT14); no learner prose asking for a runtime restart;
- the carried stage runner's required calls and exports (input manifest, evaluation report, probes, dataset manifest
  with refusal probes, baselines configured on the training split, fine-tune, adapter export, fresh-process evaluation
  with the iteration-count guard and run history, unseen-pair disparity export, fresh-process reload with the tolerance
  assertion, provenance with `remote_code_fetched: False`, BYOD pair report, error record);
- `MODEL_ID`/`MODEL_REVISION` never rebound or cited in a kernel cell; the revision is a 64-hex SHA-256 or the
  `"unpinned"` sentinel, and the pin state agrees across `pipeline.py`, the manifest (no digest or size while
  unpinned), `README.md`, `MODEL_CARD.md`, `docs/WEIGHTS.md`, `STATUS.md` and this file (each stated the pending pin
  while unpinned, and none may do so once pinned); an unpinned checkpoint can only be Candidate;
- every SHA-256 and byte count quoted in the weight documents comes from the manifest or a labelled allowlist entry;
- forbidden patterns in the kernel and in every carried file (credential-in-URL, `git clone` / `github.com`, editable
  install, mutable revision, `trust_remote_code=True`, unsafe deserialization including any `torch.load` without
  `weights_only=True`, `extractall`, magics, source or unhashed installs);
- the guided layer (GDL1–GDL15): orientation markers, at least 7 predictions (one before each of Sections 4, 5, 7, 8, 9
  and 10), 8 "What to notice" notes, 7 worked answers, the glossary terms, section tags on every numbered heading, the
  infrastructure cells titled and collapsed and the learner cells not, and the change-one-thing field
  `FREEZE_ENCODERS = True` in the adaptation cell;
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token; `MODEL_CARD.md` front matter
  (`model_card_spec: "1.2"`), single H1, the 19 required headings in order, no badge linking back to this repository,
  no non-public or maintenance vocabulary, and the immutable provenance section.

The offline unit suite (`tests/`) adds: metric definitions; baselines; validation refusals; seeded, group-preserving
splits; BYOD reading of `.pfm`, `.npy` and 16-bit `.png` disparity with path-traversal refusal; the carried-upstream
digest check and the foreign-`core` guard; archive-member extraction (ambiguous, unsafe, symlinked and oversized members
refused); staging through a fake archive and size/digest mismatches; the import boundary (rejections before `torch` is
imported); the real architecture with random weights (parameter and tensor counts, `module.` prefix stripping, the
disparity sign and shape, upstream's sequence-loss weighting, frozen encoders unchanged by fine-tuning, adapter
round-trip within 1e-4 px, adapters from another base, format or upstream commit refused); the pin tool on a copy of
the repository with a fake archive (pins a strictly loadable checkpoint; refuses a missing member or a non-loading
checkpoint; dry run writes nothing); the generated carrier and `run_stage` (a changed carried file is refused; the
unpinned `weights` stage stops the kernel with the stage's message); and a CPU pre-flight of every stage with
random-weight stand-ins: the sample path in order, the change-one-thing re-run adding a run-history row, a compatible
BYOD folder through `prepare` → `reload`, two incompatible BYOD folders refused by name, and the BYOD pair branch with
and without ground truth.

## Local execution of the notebook (2026-10-04; not hosted runtime evidence)

Environment: Linux x86_64 container, 4 CPU threads, no GPU, no access to Dropbox, Hugging Face, Google Drive or
download.pytorch.org. Kernel: CPython 3.12.12 with `nbclient` 0.10.2.

Procedure: a scratch copy of this commit's working tree, in which a **random-weight** checkpoint of the real
architecture (saved as a DataParallel `state_dict` inside a fake `models.zip` with a second member) was pinned with
`tools/pin_snapshot.py --archive` (strict load and parameter count passed) and the notebook regenerated; the checkpoint
file was pre-staged in `weights/` because Dropbox is not reachable here. Each run executed an executed copy of the
notebook whose form fields were set by an executor that fails on a missing field (EXE6): `N_PAIRS = 6`, `EPOCHS = 1`,
`VALID_ITERS = 4`, `TRAIN_ITERS = 2`, every other learner field at its default.

- **Run A — default path and both BYOD branches: PASS**, all 14 code cells, 347 s wall time. Extra fields:
  `USE_BYOD_PAIR = True` with `BYOD_LEFT_PATH`, `BYOD_RIGHT_PATH` and `BYOD_DISPARITY_PATH` (one rendered 160 × 128 pair,
  `.pfm` ground truth), and `USE_BYOD_DATASET = True` with `BYOD_DATASET_DIR` (four rendered scenes in sub-folders,
  `pairs.json` with `group` per scene). The runtime check printed the CPU note; the carrier wrote and verified 16 files;
  the real install cell downloaded and verified the uv 0.12.15 wheel, built a managed CPython 3.12.12 environment and
  installed the 35 hash-locked packages (`torch 2.14.0+cu130`, `cuda: False`); the `weights` stage verified the 8
  upstream files and the checkpoint (`fetched: []`); `demo`, `probes`, `prepare` (three refusal probes rejected by
  name), `baselines`, `adapt` (5,728,144 of 11,116,176 parameters trainable with the encoders frozen), `evaluate`
  (fresh process, from the exported adapter), `infer` (three unseen pairs, disparity `.npy` and 16-bit `.png` written),
  `reload` (`equivalent: True`) and the outputs cell completed; the BYOD pair branch wrote a `sample-sanity` report with
  the SGBM baseline; the BYOD dataset branch ran `prepare` → `baselines` → `adapt` → `evaluate` → `infer` → `reload`
  under `outputs/byod/` (reload differences 0.0 px) without touching the sample outputs.
- **Run B — incompatible BYOD folder: refused as intended.** Same fields with `USE_BYOD_DATASET = True` and a copy of the
  folder whose `pairs.json` names `../outside/im1.png`: the default path completed (13 cells), and the BYOD cell stopped
  with `RuntimeError: Stage 'prepare' failed (exit 2): ValueError: pairs.json entry 2: file '../outside/im1.png' must be
  relative to the dataset folder` before any model ran (374 s).

Because the weights were random, **no metric from these runs means anything** and none is reported here. The runs prove
the infrastructure, the stage plumbing, the BYOD branches and the file hand-offs on a CPU runtime, nothing about the
model. An earlier attempt of run B stopped at the notebook's disk check (`Not enough free disk`) because previous
isolated environments filled the container's temporary directory; that is the guard working, and the run was repeated
after the old environments were removed. CPU timing, measured separately with nothing else running: one 32-iteration
inference on a 320 × 224 pair took 3.3 s and one 12-iteration training step on two pairs 10 s.

Not exercised locally: the real checkpoint (download from Dropbox, its digest, its strict load, its accuracy); a GPU;
Google Colab's kernel and upload dialog; the full default field values (32 pairs, 3 epochs, 32/12 iterations).

## Release gate (to be completed in order)

1. **Pin the checkpoint (done 2026-10-05).** Pinned on 2026-10-05 by running `python tools/pin_snapshot.py --dry-run` in a Google Colab runtime (Dropbox reachable): the extracted member is 44,617,876 bytes with SHA-256 `d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819`, and it strict-loaded into the carried architecture (337 tensors, 11,116,176 parameters). The manifest and `MODEL_REVISION` were written from that output. Original procedure: in a runtime that reaches Dropbox (a Colab CPU runtime is enough): clone the repository,
   install the pins (`pip install torch==2.14.0 numpy==2.5.3 scipy==1.18.1 opt-einsum==3.4.0 safetensors==0.8.0
   pillow==11.3.0`), and run `python tools/pin_snapshot.py`. Check that the printed size is plausible (about 44.6 MB is
   expected from the architecture; see `docs/WEIGHTS.md`), that the strict load passed, and commit the manifest and
   `pipeline.py` change. Then regenerate the notebook (`python tools/build_notebook.py`), replace every pending-pin
   statement (README, MODEL_CARD, WEIGHTS, STATUS, this file) with the digest and byte size, and run `pytest` and
   `python tools/validate_release_assets.py`.
2. **Hosted Run all.** Open the regenerated notebook at that commit in a fresh Colab **T4** runtime and choose
   *Run all* without editing anything. Record below: commit, notebook blob, runtime and GPU, start time, total time and
   per-stage times, the setup dictionary, and the principal outputs (demonstration metrics, probes, baseline table,
   adapted row, unseen-pair counts, reload parity). Keep the executed copy under `docs/execution-evidence/<date>/`.
3. **Confirm the standalone contract** from that run (REL3–REL7): no clone, no DIMER worker or service, no credential,
   the sample generated automatically, every stage completed, export and fresh reload in the same run.
4. **REL12 BYOD.** In the same or a second hosted run, set `USE_BYOD_DATASET = True` with `BYOD_DATASET_DIR` pointing at
   a small compatible folder (for example a few Middlebury 2014 scenes at quarter resolution with `pairs.json`, noting
   that the checkpoint was fine-tuned on Middlebury) and confirm it reaches `reload`; then point it at an incompatible
   `pairs.json` (a path with `..`, or images of different sizes) and confirm the refusal names the rule. Also run the
   pair branch once.
5. **Replace the guided-layer worked answers' expectations** with the recorded values where they differ, regenerate,
   and re-run `--check`.
6. Only then may `STATUS.md`, `README.md` and `tutorials/README.md` be promoted, and only to what the evidence shows.

## Recorded executions

| Date | Commit / notebook | Runtime | Scope | Outcome |
|---|---|---|---|---|
| 2026-10-04 | this commit's working tree, regenerated in a scratch copy with a random-weight checkpoint pinned | local Linux CPU container (4 threads), `nbclient` | run A: default path + both BYOD branches, reduced fields, random weights | PASS (plumbing only; metrics meaningless) |
| 2026-10-04 | same | same | run B: default path + incompatible BYOD folder | default path PASS; BYOD refused by name, as intended |

No hosted (Colab/Kaggle) execution has been recorded yet.
