# RAFT-Stereo notebook — review fixes

**Review:** `raft_stereo_colab_Review.md` (5 October 2026, RST-M1, RST-m1..m4).
**Fixed in:** the generator (`tools/build_notebook.py`, `tools/notebook_template.py`, `tools/tutorial_stages.py`), `MODEL_CARD.md` and the tests; the notebook was regenerated and `--check` passes.
**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| RST-M1 | **Fixed** (acceptance checks met; a 0–2 px background scene remains a follow-up) | (a) The `probes` stage now also runs 0 px (identical images), 2 px and 4 px random-dot pairs and prints each with a `verdict` (`FAILED (known limitation…)` above 1 px EPE); they are recorded as `limitation_probes` in `probes.json` and never stop the stage. (b) Section 5 is retitled "Probes with exact answers, and a known failure at 0–4 px": the untested sentence "not of small disparities in general" is removed and replaced with what was measured (the diagnostic), what is unknown (checkpoint vs pipeline cause; no scene below the renderer's 2 px floor has been evaluated) and the instruction to treat near-zero regions as suspect; the sample answer explains why probes must be able to fail. (c) The Section 13 BYOD contract carries a near-zero disparity warning. (d) `MODEL_CARD.md` drops "rendered scenes with small disparities are read well" and adds an input-boundary bullet. **Added 2026-10-10 (relay verification):** (e) a local CPU comparison against upstream `demo.py`'s inference path at the carried commit with the pinned checkpoint (`docs/execution-evidence/2026-10-10-rst-m1-upstream-comparison/`) gives output identical to this pipeline (max difference 0.0 px) on 0, 1, 2, 4 and 8 px random-dot pairs and on an identical-image rendered scene (about 416 px for a true 0); upstream's `alt` correlation fails the same way. Section 5, the stage message, the BYOD warning, `MODEL_CARD.md` and `STATUS.md` now say the checkpoint causes it. **Not done:** a rendered scene whose background alone sits at 0–2 px (follow-up). | `stage_probes`; Sections 5 and 13; `MODEL_CARD.md` | `test_rst_M1_small_shift_probes_are_shown_with_a_verdict_and_never_stop_the_stage` (stand-in matcher), `test_rst_M1_untested_scope_claim_is_gone_and_the_limitation_is_shown_and_warned`, `test_rst_M1_cause_claim_is_backed_by_the_recorded_upstream_comparison` |
| RST-m1 | Fixed | The Run-all line and the Prerequisites quote the recorded T4 run (5 October 2026, revision `ac3cc07`, 68 s environment, per-stage times); "have not been measured yet" and "no recorded hosted run" are gone; the disk figures are labelled estimates; the unreachable "checkpoint is not pinned" Troubleshooting row is removed (the pinned/unpinned Section 3 note already covers it). | `run_all`, Prerequisites, How to use, Section 14 note, Troubleshooting | `test_rst_m1_no_stale_hosted_run_or_pin_statements` |
| RST-m2 | Fixed | Section 8 sample answer: "most of it" — the out-of-domain epoch-1 loss falls steeply, 2.709 → 0.711 (about −74 %) in the recorded run. Section 9: held-out EPE roughly halves (0.231 → 0.112 px), bad-3 0.0079 → 0.0046, 8/8 pairs; caveats kept. | Sections 8–9 sample answers | `test_rst_m2_sample_answers_match_the_recorded_run` |
| RST-m3 | Fixed | With `FREEZE_ENCODERS = False`, `adapt`, `evaluate`, `infer` and `reload` write the same file names under `outputs/activity_unfrozen/`; the canonical adapter, result record, evaluation and disparity maps in `outputs/` stay as the default run left them. The run history keeps both rows. The Section 8 cell sets `ADAPT_OUT`, which Sections 10 and 12 use to show the matching files; Section 14 says where the activity's files go. | `use_configuration_outputs`, `stage_adapt/evaluate/infer/reload`; Sections 8, 10, 12, 14; validator marker | `test_rst_m3_activity_writes_beside_and_never_over_the_canonical_outputs` (canonical digests unchanged after the activity); `test_sample_path_runs_every_stage_in_order` updated to read the activity history from its directory |
| RST-m4 | Fixed (stronger than the acceptance check) | The uv-template change piloted in the MediaPipe notebook: Section 1 keeps this session's run directory when re-run (`NEW_RUN_DIRECTORY` for a fresh one); the environment directory is keyed on the lock digest, managed Python and `uv` version and reused through a `ready.json` marker; `run_stage` names Sections 1–3 when the run directory or interpreter is missing; Troubleshooting entry; `PYTHONSTARTUP` dropped from the stage environment. | `CHECK_CELL`, `INSTALL_CELL`, `run_stage`; validator | `test_rst_m4_*` (4 tests) |

## User-visible changes

- Section 5 prints three extra probe lines (0, 2, 4 px) with `FAILED` verdicts on the real checkpoint (expected), and `probes.json` gains `limitation_probes`.
- The Section 14 activity writes to `outputs/activity_unfrozen/`; `ADAPT_OUT` is a new kernel variable set by Section 8.
- Section 1 has a new form field, `NEW_RUN_DIRECTORY` (off); re-runs keep the run directory; the environment is reused (`environment_reused`) and lives at `<tmp>/raft_stereo_env_<lock key>`.
- `MODEL_CARD.md` gains a near-zero-disparity input boundary.
- Tests: without torch, `test_tiny_model.py` and `test_pin_snapshot.py` are skipped (via `collect_ignore`) instead of erroring, and the torch stage tests skip; with torch (CI) they run as before.

## Verification (offline, not clean-runtime evidence)

- `build_notebook.py --check`: OK. `validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` **without torch** (this container): before, 2 collection errors, and with those modules ignored 61 passed / 4 failed / 1 skipped; after, **68 passed / 7 skipped**. CI installs torch; the torch tests were not run here, and `test_sample_path_runs_every_stage_in_order` was edited (activity history path) without being executed.
- The RST-M1 and RST-m3 stage tests use a **stand-in** block matcher, not the checkpoint: they prove the plumbing (verdicts recorded, nothing stops, canonical files untouched), not model behaviour. The real-checkpoint behaviour is the recorded Colab diagnostic.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated notebook on a fresh T4 runtime (no restart); the 0/2/4 px probe lines will show the real verdicts. Then re-run the reload/export cell once.
2. The REL12 BYOD journey on a hosted runtime (pair branch and dataset branch, one refusal), including the upload dialog.
3. **RST-M1 root cause — done 2026-10-10** (local CPU, `docs/execution-evidence/2026-10-10-rst-m1-upstream-comparison/`): this pipeline reproduces upstream's inference path exactly; the failure is the checkpoint's. Still open: one rendered scene with a 0–2 px background only.
