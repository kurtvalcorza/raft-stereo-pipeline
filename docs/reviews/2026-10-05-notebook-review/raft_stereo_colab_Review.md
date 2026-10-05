# RAFT-Stereo E2E Notebook — Review

**Verdict: Needs revision** (one Major, four Minors)  
**Review date:** 5 October 2026  
**Repository:** `kurtvalcorza/raft-stereo-pipeline`  
**Notebook:** `tutorials/raft_stereo_colab.ipynb`  
**Reviewed commit:** `f51d8b9` (`main`, the merge of PR #1)  
**Notebook Git blob:** `44474b8480b4ee6a0099fa678c6b4efb4711906b`, generated from `ac3cc07`. This is the blob executed in the clean recorded Colab T4 run of 2026-10-05. At the reviewed commit `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `RST`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2, `ml-worker` `origin/main` at `b9fdd1f`.

## Executive assessment

The engineering is strong:

- **Isolated runtime.** A pinned `uv` builds managed CPython 3.12.12 from a hash lock of 35 packages, and every stage runs in its own subprocess.
- **Verified code and weights.** The upstream `core/` is carried verbatim and hash-checked before import. The checkpoint is pinned by its own SHA-256 and loaded with `weights_only=True`.
- **Clean evaluation design.** The 32-pair rendered dataset has exact ground truth and a leakage-checked split. Baselines (median constant, SGBM) are configured on the training split only. Fine-tuning always restarts from the checkpoint.
- **Artifact checks.** The exported adapter is scored in a fresh process and reloaded in a second fresh process against stated tolerances. The evaluation refuses a mismatched iteration count.
- **Guided layer.** It is complete.

The recorded Colab run is clean and the numbers are coherent:

| Measure (Colab T4, blob `44474b8`, 2026-10-05) | Value |
|---|---|
| Code cells | 14/14, one pass, no restart; environment build 68 s, stages ≈ 94 s |
| Probes | 8 px → 8.001 (EPE 0.032); 16 px → 16.01 (EPE 0.030) |
| Demonstration pair | RAFT-Stereo EPE 0.158 vs SGBM 0.469 |
| Held-out EPE / bad-3 (8 pairs) | median 8.223 / 0.632 · SGBM 0.653 / 0.028 · pretrained 0.231 / 0.0079 · adapted 0.112 / 0.0046 |
| Adapted EPE lower than pretrained | 8/8 held-out, 3/3 unseen |
| Fine-tune | 5,728,144 of 11,116,176 parameters trainable; loss 2.709 → 0.933 → 0.711; 23 s |
| Reload parity | mean / max difference 0.0 / 0.0 px; in-memory EPE equals fresh-process EPE |

The Major is this:

- **The failure is real and unexplained.** In the maintainer's diagnostic, this checkpoint reports **128–367 px** disparity, more than the 320 px image width, for random-dot pairs shifted by 0–4 px. That includes two identical images.
- **The notebook's response.** It switched its probes to 8 and 16 px, and it tells the learner the failure is "not of small disparities in general".
- **That claim is untested.** The rendered sample never contains a disparity below 2 px (`SAMPLE_MIN_DISPARITY = 2.0`; P3 found 0 % of evaluated pixels under 2 px). No scene with near-zero disparity has been run.
- **The cause is unknown.** It may be the model or the pipeline.
- **BYOD users are not warned.** Distant backgrounds have exactly these near-zero disparities.

Four Minors follow:

- stale "no hosted run / not pinned" statements (RST-m1);
- sample answers that understate the recorded fine-tuning effect (RST-m2);
- a Section 14 activity that overwrites the canonical adapter and result record (RST-m3);
- the shared partial re-run and environment-rebuild behaviour (RST-m4).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` |
| Declared spec | DIMER Notebook Specification **2.2**, standalone |
| Intended audience | Stated: learners who run hosted notebook cells and read short Python; no stereo background assumed |
| Supported runtime | Linux x86_64; Colab or Kaggle T4 documented, CPU completes slowly; float32; GPU kernels not forced deterministic (stated) |
| Promised outcomes | <ul><li>verified upstream code and checkpoint</li><li>demonstration pair against SGBM</li><li>two exact-answer probes that "check that the model's output means what you think it means"</li><li>validated 32-pair dataset split 24 / 8, with refusals</li><li>three baselines</li><li>bounded fine-tune</li><li>fresh-process held-out evaluation with a run history</li><li>unseen-pair inference with exported disparity maps</li><li>fresh reload equivalence</li><li>provenance record</li><li>two BYOD branches (pair; labelled dataset through the full workflow)</li><li>an unfreeze-encoders activity</li></ul> |
| Generator | `tools/build_notebook.py` (`build_notebook.py/3.1`) + `tools/notebook_template.py`; generating revision `ac3cc07` |
| Release status | `Candidate` (`STATUS.md`): default path recorded clean on Colab; REL12 hosted BYOD pending |

### Evidence actually obtained

- **Source inspection.**
  - All 38 cells (14 code; cell 9 is the 16-file carrier).
  - `tools/tutorial_stages.py`: every stage, the probe shifts, `adapt`, `evaluate`'s run history and the output names.
  - `src/raft_stereo_pipeline/samples.py`: the renderer, `SAMPLE_MIN_DISPARITY` and `random_dot_pair`.
  - `pipeline.py`: `estimate` follows upstream's demo path (raw 0–255 tensors, `InputPadder`, `test_mode=True`, `disparity = -flow`).
  - `STATUS.md`, `docs/release-verification.md`, and `docs/execution-evidence/2026-10-05/probe_shift_diagnostic.md`.
- **Documented execution evidence.**
  - Three Colab T4 runs are recorded; the reviewed blob's run (`…_44474b8_colab-t4.ipynb`) is clean and one-pass.
  - The maintainer's random-dot diagnostic covers 4 dot sizes × 6 shifts.
  - The local CPU runs with random weights covered BYOD plumbing. They are not model evidence.
- **Direct execution (this review), without torch or the checkpoint.**
  - Environment: Python 3.12 with `numpy` 2.5.3 and `pillow` 11.3.0.
  - P3: disparity statistics of the rendered sample (32 pairs, seed 0) and of the unseen pairs (3, seed 99), over the valid pixels that every metric scores.
  - P4: `read_stereo_records`, `validate_dataset` and `split_dataset` on 10 constructed BYOD folders, plus the BYOD cell's own `_pick` and `_safe_unzip` helpers, executed from the notebook source.
  - Scripts and results are in `raft_stereo_colab_Review_Probes.zip`.
- **Not executed here:** any torch stage. Dropbox, PyPI's CUDA torch and `download.pytorch.org` are unavailable or too large for this container, so model behaviour comes from the recorded runs.
- **Learner observation:** none.

## 2. Separate judgments

- **Technical correctness:**
  - The default path is correct and reproduces cleanly.
  - One open technical question remains: an output larger than the image width for identical images, either from the model or the pipeline.
  - The notebook routes around this question instead of resolving it (RST-M1).
- **Promise fulfilment:** every promised stage ran on Colab and wrote its outputs. Two gaps:
  - Section 5's promise ("check that the model's output means what you think it means") holds only for the 8–16 px range the probes were chosen to pass (RST-M1).
  - The activity's change does not leave the canonical outputs intact (RST-m3).
- **Learner experience:** strong.
  - Disparity geometry, the valid mask, EPE versus bad-pixel rates and loss-versus-task evidence are all explained well.
  - Friction: sample answers contradict the recorded magnitudes (RST-m2), and the "Why not a smaller shift?" box offers a scope claim with no evidence behind it (RST-M1).
- **Spec conformance:**
  - Unresolved MUSTs: SRC3 (RST-m1; RST-M1's unsupported claim), EVAL15 (RST-M1) and UX12 (RST-m1).
  - SHOULD deviations: GDL10 and UX7 (RST-m3); UX10 and GDL13 (RST-m4).
  - REL12 hosted BYOD evidence is absent.

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result (Colab unless noted) | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass Run all, no restart | cells 6, 9, 12 | 14/14, `setup_seconds` 68 | Section 2 prose | Met |
| Verified upstream code and checkpoint | cell 15 / `weights` | 8 files verified; checkpoint `d22e84c0…` fetched and verified | trust boundary stated | Met |
| Pretrained model beats SGBM on the demonstration pair | cell 18 / `demo` | EPE 0.158 vs 0.469; probe refused | sample answer matches | Met |
| Probes "check that the model's output means what you think it means" | cell 20 / `probes` | 8.001 / 16.01 px | "not of small disparities in general" | **Partly met** (RST-M1) |
| Dataset validated, split, refusals | cell 22 / `prepare` | 32 → 24 / 8, `shared_ids: 0`, 3 refusals; valid fraction 0.912 | sample answer matches | Met (sample floor of 2 px undisclosed; RST-M1) |
| Three baselines on held-out pairs | cell 24 / `baselines` | median 8.22, SGBM 0.65, pretrained 0.23 | EPE vs bad-px reasoning | Met |
| Bounded fine-tune; loss is optimisation evidence | cell 26 / `adapt` | 2.709 → 0.711 (−74 %), 51.5 % trainable | "falls by a modest fraction" | Met (RST-m2) |
| Fresh-process held-out comparison and run history | cell 28 / `evaluate` | 0.231 → 0.112 (−52 %), 8/8 | "usually lowers … a little" | Met (RST-m2) |
| Unseen pairs and disparity files | cell 30 / `infer` | 3/3 lower; `.npy` + KITTI PNG + CSV | "a demonstration, not an evaluation" | Met |
| Reload equivalence and provenance | cells 32, 34 | 0.0 / 0.0 px; `result.json` identity fields | "loading is not the check" | Met |
| BYOD pair and dataset branches | cell 36 | P4: 3 compatible folders and 8 incompatible ones refused naming the rule; `_safe_unzip` refuses `../`; `_pick` handles Middlebury and left/right names, refuses KITTI-style names | contract stated first | Met locally (reader); hosted REL12 pending |
| Activity: unfreeze the encoders | Section 14 via "Run after" from Section 8 | not run on Colab; run history row 1 | — | Met in source (RST-m3) |
| Runtime statements | cells 1, 4; Troubleshooting | "this revision has no recorded hosted run"; "Hosted-runtime times have not been measured yet"; a "checkpoint is not pinned" row | — | **Not met** (RST-m1) |

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 38 cells | The guided layer is complete. The learner meets: <ul><li>a scope claim about the small-shift failure that the evidence does not support (RST-M1);</li><li>sample answers saying the loss falls "by a modest fraction" and EPE improves "a little", when the run shows −74 % and −52 % (RST-m2);</li><li>a claim that no hosted run exists (RST-m1).</li></ul> |
| **Clean default** | Documented (Colab T4, reviewed blob) | 14/14 in one pass; numbers in the table above. Not re-executed here. |
| **Active learning** | Source | Section 8 always restarts from the checkpoint, so a changed field compares cleanly, and the run history keeps one row per evaluation. But "Run after" from Section 8 re-exports `raft_stereo_adapter.safetensors` and rewrites `raft_stereo_result.json` and the unseen-pair maps with the unfrozen run's results, so the default artifact is gone (RST-m3). |
| **Reuse and recovery** | Direct (P4) + source | See the two parts below. |

**Reuse and recovery, BYOD reader (P4).**

- Accepted:
  - 4 pairs in 2 groups (split 2 / 2);
  - 2 pairs with no groups (split 1 / 1: one training pair, one test pair; see RST-S2).
- Refused, each naming the entry and the rule:
  - a `../` path;
  - a negative disparity, with a "swap left and right" hint;
  - a missing file;
  - one group;
  - a size mismatch;
  - a wrong disparity shape;
  - a `.txt` disparity;
  - a 1400 px side.
- Cell helpers:
  - `_safe_unzip` refuses `../escape.png`;
  - `_pick` maps Middlebury `im0` / `im1` and `*_left` / `*_right` names;
  - KITTI-style `000000_10.png` names are refused with "name exactly one uploaded file with left or im0".

**Reuse and recovery, re-runs (source).** Cell 6 has the same `uuid4` run-directory pattern as the MediaPipe notebook, where re-running Section 1 alone was shown to strand the later cells (RST-m4).

## 5. Findings

### Major

#### RST-M1 — An unexplained failure on near-zero disparity is routed around, and the notebook asserts a scope it has not tested

- **Cell/section:**
  - Section 5 (cell 19, the "Why not a smaller shift?" box; cell 21 sample answer);
  - Section 6 (cells 21–23);
  - Section 13 (cell 35, BYOD contract);
  - generator `tools/notebook_template.py`, `tools/tutorial_stages.py` (`PROBE_SHIFTS = (8, 16)`), and `samples.py` (`SAMPLE_MIN_DISPARITY = 2.0`).
- **Observed issue:**
  - **The diagnostic.** In the maintainer's Colab diagnostic (`probe_shift_diagnostic.md`), the pinned checkpoint, through this repository's `RaftStereoPipeline`, reports medians of **128–367 px** for random-dot pairs shifted by 0, 1, 2 or 4 px, at every dot size from 1 to 8 px. The image is 320 px wide. For **shift 0**, the left and right images are identical and the correct answer is 0 everywhere.
  - **The notebook's response.** It changed its probes to 8 and 16 px and tells learners: "Rendered scenes with disparities of 2 px and more are read well, so this is a limitation on small uniform whole-frame shifts of random dots, not of small disparities in general; its cause was not investigated."
  - **Why the claim is untested.** The renderer deliberately keeps every scene at or above 2 px (`SAMPLE_MIN_DISPARITY = 2.0`, present since the first build). The evidence therefore covers only disparities ≥ 2 px. The notebook has never run a scene, synthetic or real, with a disparity between 0 and 2 px.
  - **Possible causes.** A pipeline-side cause has not been excluded, for example in tensor preparation or padding, although `estimate` mirrors upstream's demo. Neither has a checkpoint-side one.
  - **The missing warning.** The Section 13 BYOD contract does not warn that near-zero disparities (distant backgrounds, the sky, small baselines) may be read catastrophically wrong.
- **Consequence:**
  - **For learners.** Section 5 teaches that probes "check that the model's output means what you think it means", but the probes were chosen from the range the model was already known to pass. That models the anti-pattern of selecting sanity checks until they pass. The learner is also told something about the model's scope that no run has shown.
  - **For BYOD users.** Real scenes with distant content may receive confident disparities larger than the image is wide. RAFT-Stereo outputs no confidence, so nothing in the notebook would flag them.
- **Evidence:**
  - Documented: the diagnostic table (24 cells; every 0–4 px cell has EPE > 120 px, every 8 and 16 px cell has EPE ≤ 0.032 px), and the recorded probe outputs at 8 and 16 px.
  - Direct (P3), over valid pixels:

    | Data | Minimum disparity | Share < 2 px | Share < 4 px |
    |---|---|---|---|
    | Sample, 32 pairs | 2.003 px | **0.0** | 8.7 % |
    | Unseen pairs | 2.364 px | **0.0** | 3.6 % |
  - Source: `PROBE_SHIFTS = (8, 16)` with the comment "0-4 px fail".
- **Recommended correction:**
  1. **Investigate before claiming scope.** Run upstream's own `demo.py` on an identical-image pair and on a 2 px random-dot pair with the same checkpoint. That separates a pipeline cause from a checkpoint cause. Then run a rendered scene whose background sits at 0–2 px.
  2. **Replace the scope sentence with what was measured.** If small real disparities fail, say so in Section 5, in the BYOD contract and in `MODEL_CARD.md`.
  3. **Keep a visible small-shift probe**, for example 0 px (identical images), reported as a known failure with its interpretation, beside the 8 and 16 px probes. The learner then sees the limitation instead of a curated pass.
  4. **Optionally lower `SAMPLE_MIN_DISPARITY`**, or add one near-zero-background scene, so that the held-out evaluation covers the regime.
- **Acceptance check:**
  - The repository records a run (Colab is fine) of upstream `demo.py`, or an equivalent, on an identical-image pair with the pinned checkpoint, and states whether this pipeline reproduces upstream's output.
  - The notebook no longer contains "not of small disparities in general" unless a recorded run with 0–2 px scene disparities supports it.
  - Either the near-zero-disparity behaviour is shown in Section 5, or the BYOD contract warns about it, according to the result.
- **Spec:** EVAL15, SRC3, UX4, DAT12.

### Minor

#### RST-m1 — "No recorded hosted run" and "not pinned" statements survive in a pinned, hosted-run revision

- **Cell/section:**
  - cell 1 (*Predict, then check*: "this revision has no recorded hosted run");
  - cell 4 (*Runtime*: "Hosted-runtime times have not been measured yet for this revision", with only CPU estimates; "about 8 GiB" for the environment);
  - Troubleshooting ("Section 3 says the checkpoint is not pinned. This revision of the notebook was generated before the checkpoint's SHA-256 was recorded…").
- **Observed issue:**
  - The clean Colab T4 run of this blob is recorded: environment 68 s; weights 12.5 s, demo 5.6, probes 3.5, prepare 5.3, baselines 8.9, adapt 31.9 (23.0 s of training), evaluate 9.5, infer 9.4, reload 7.1.
  - The checkpoint is pinned. Section 3 itself says "Pin state of this revision: pinned".
- **Consequence:** the learner reads that no hosted measurement exists, plans by a 10–20 minute CPU estimate, and meets a troubleshooting row that cannot apply.
- **Evidence:** documented (the Colab outputs above); source.
- **Recommended correction:**
  - Quote the T4 timings, with runtime, date and revision.
  - Drop the stale sentence from cell 1.
  - Make the "not pinned" row conditional, or remove it from pinned builds.
  - Label the disk figures as estimates.
- **Acceptance check:** `grep -n "no recorded hosted run\|have not been measured yet" tutorials/raft_stereo_colab.ipynb` returns nothing, and the Prerequisites quote a T4 timing that matches `docs/release-verification.md`.
- **Spec:** SRC3, UX12.

#### RST-m2 — Sample answers understate the recorded fine-tuning effect

- **Cell/section:** Section 8 (cell 25 predict prompt; cell 27 sample answer) and Section 9 (cell 29 sample answer).
- **Observed issue:**
  - The Section 8 prompt offers "a few percent, a third, or most of it". The sample answer says the loss "typically falls by a modest fraction over three short epochs". The recorded loss falls from 2.709 to 0.711, which is **−74 %, most of it**.
  - The Section 9 answer says a short fine-tune "usually lowers held-out EPE and the bad-pixel rates a little". The recorded EPE goes from 0.231 to 0.112, which is **−52 %**, with bad-3 falling from 0.0079 to 0.0046 and 8/8 pairs improving.
- **Consequence:** a learner who predicted correctly is told by the sample answer that they were wrong, and the guidance misdescribes the shape of a normal result on the documented runtime.
- **Evidence:** documented (Colab outputs).
- **Recommended correction:** describe the recorded shape. The epoch-1 loss is high because the checkpoint starts out of domain, then it falls steeply. Held-out EPE roughly halves on this rendered domain. Keep the caveats (8 pairs, one run, rendered data).
- **Acceptance check:** the Section 8 and Section 9 sample answers are consistent with the recorded default run in `docs/release-verification.md`.
- **Spec:** SRC3, UX4.

#### RST-m3 — The Section 14 activity overwrites the canonical adapter, result record and disparity maps

- **Cell/section:** Section 14 (cell 37: "select the Section 8 cell and choose **Runtime → Run after**"); `stage_adapt` (`run.out / f"{run.prefix}_adapter.safetensors"`); `stage_reload` (`{prefix}_result.json`).
- **Observed issue:**
  - The activity re-runs Section 8 with `FREEZE_ENCODERS = False`, and Sections 9–12 follow.
  - The new adapter, unseen-pair maps, CSV and `raft_stereo_result.json` replace the default run's files under the same names. Only the run history keeps both evaluations.
  - The default configuration's exported artifact is no longer on disk.
- **Consequence:**
  - A learner who downloads `outputs/` after the activity gets the unfrozen adapter, without noticing.
  - The "change one thing" exercise is not optional with respect to the canonical outputs, which GDL10 asks for.
- **Evidence:** source; not executed here.
- **Recommended correction:** give the activity its own stage or output namespace (for example `outputs/activity/`), as the other fleet notebooks do, or have `adapt` write per-configuration file names. Keep the run history.
- **Acceptance check:** after a default Run all and the Section 14 activity, the default `raft_stereo_adapter.safetensors` digest and `raft_stereo_result.json` are unchanged, and the activity's results sit beside them.
- **Spec:** GDL10, UX7.

#### RST-m4 — Re-running the Section 1 cell alone strands later cells; every Run all rebuilds the several-GB environment

- **Cell/section:** cell 6; cells 9 and 12; Troubleshooting.
- **Observed issue:**
  - `ROOT` and `ENV_ROOT` come from a fresh `uuid4`. Re-running only cell 6 points `ROOT` at an empty directory while `PYTHON` names the old environment. The next learner cell fails with `can't open file '<new ROOT>/tutorial_stages.py'` and "see the log above".
  - Every Run all builds a new environment with its own `uv` cache. The CUDA torch wheels are downloaded again (68 s on Colab), and the 8 GiB disk check is applied again.
- **Consequence:**
  - A natural re-check after changing the runtime type yields an unguided error.
  - Repeated runs cost time and disk.
- **Evidence:** inferred from source. The code path is identical to the MediaPipe notebook's, where this review executed it (`mediapipe-face-landmarker-pipeline` review, P5); not executed here.
- **Recommended correction:**
  - Have `run_stage` check that the carried runner and `PYTHON` exist, and name the cells to re-run.
  - Add a Troubleshooting row.
  - Key `ENV_ROOT` on the lock digest.
- **Acceptance check:** re-running cell 6 and then cell 18 stops with a message naming Sections 2–3, and a second Run all reuses the environment.
- **Spec:** SRC2, UX10, GDL13.

### Suggestions

- **RST-S1** — The BYOD pair upload requires `left` / `right` or `im0` / `im1` in file names (P4). Say so next to the KITTI convention, whose files are named `000000_10.png` under `image_2/` and `image_3/`, so KITTI users rename their files before uploading.
- **RST-S2** — A 2-pair BYOD folder is accepted and split 1 / 1 (P4), so the fine-tune trains on a single pair and is scored on a single pair. Print a warning when the training split has fewer pairs than `BATCH_SIZE` × 2, or raise the minimum.
- **RST-S3** — Disclose in Section 6 that the renderer keeps every disparity between 2 and 40 px, next to `max_disparity_px`.
- **RST-S4** — Add the per-pair EPE spread (minimum and maximum of adapted minus pretrained) beside the 8/8 count, as the NAFNet notebook does.

## 6. Readiness

**Needs revision.**

RST-M1 must be resolved before the notebook can claim that its probes validate the pipeline, or say anything about small-disparity scope. Resolving it needs one hosted investigation run, with upstream `demo.py` on an identical-image pair.

The four Minors are generator edits.

Remaining gates after the fixes:

- a one-pass hosted Run all of the regenerated blob;
- the REL12 BYOD journey on a hosted runtime: one compatible folder and one incompatible `pairs.json`, as `STATUS.md` specifies.

## 7. Verified versus inferred

- **Verified by direct execution (torch-free):**
  - the sample's and the unseen pairs' disparity ranges over the scored pixels;
  - the BYOD reader, validation and split matrix;
  - the BYOD cell's `_pick` and `_safe_unzip` behaviour, using the notebook's own source.
- **Verified from documented evidence:** every model number (Colab T4, reviewed blob); the small-shift failure (the maintainer's diagnostic).
- **Inferred from source:**
  - the activity overwriting canonical outputs (RST-m3);
  - the partial re-run behaviour (RST-m4; executed on the MediaPipe notebook's identical code);
  - that `estimate` mirrors upstream's demo path. This was read, not executed.
- **Only Kurt can confirm:** whether a near-zero-disparity scene is in scope for this tutorial's audience, and therefore whether RST-M1 is resolved by a warning or needs a fix.
- **Most likely to be wrong:** RST-M1's severity, if the investigation shows that the failure is a known property of this upstream checkpoint on identical inputs and that real scenes with small disparities are read correctly. It would then reduce to a Minor wording fix. The finding stands until that run exists.
