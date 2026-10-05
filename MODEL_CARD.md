---
license: mit
model_card_spec: "1.2"
pipeline_tag: depth-estimation
base_model: princeton-vl/RAFT-Stereo/raftstereo-middlebury
date_published: "2021-09-15"
date_published_source: "date of upstream commit 5c13878 ('Initial Commit.'), the first public commit of princeton-vl/RAFT-Stereo; its README and download_models.sh already name raftstereo-middlebury.pth. Whether the bytes in today's models.zip equal that first release is not established: upstream changed the archive link in download_models.sh in 2023 (fa8ed9d) and 2026 (6e93ed2); the pin of 2026-10-05 records the bytes served then."
---

# RAFT-Stereo (Middlebury checkpoint) — Dense Stereo Disparity with Bounded Fine-Tuning

[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-princeton--vl%2FRAFT--Stereo-181717?style=flat&logo=github&logoColor=white)](https://github.com/princeton-vl/RAFT-Stereo)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2109.07547-b31b1b.svg)](https://arxiv.org/abs/2109.07547)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/princeton-vl/RAFT-Stereo/blob/main/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are obtained unmodified from the upstream release under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream checkpoint is pinned by its SHA-256 (`d22e84c0…5819`, 44,617,876 bytes); every loader verifies size and digest before reading it. No metric of this model has been measured by this repository yet: the Metrics sections describe what the code reports, not results.

---

## Interactive Colab Tutorials

- **End-to-end stereo disparity and adaptation notebook**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/raft-stereo-pipeline/blob/main/tutorials/raft_stereo_colab.ipynb) [`raft_stereo_colab.ipynb`](https://github.com/kurtvalcorza/raft-stereo-pipeline/blob/main/tutorials/raft_stereo_colab.ipynb)
  *Disparity on a rendered pair with exact ground truth next to OpenCV StereoSGBM, two probes with exact answers, a validated and split rendered dataset, three baselines on held-out pairs, a bounded fine-tune, held-out EPE and bad-pixel rates from the exported adapter in a fresh process, unseen-pair disparity export, and a fresh-process reload check. It runs in an isolated hash-locked environment and verifies the pinned checkpoint before any model runs.*

---

#### Description

This repository packages **RAFT-Stereo** (Lipson, Teed and Deng, 3DV 2021) with the upstream **Middlebury checkpoint**, `raftstereo-middlebury.pth`, identified here as `princeton-vl/RAFT-Stereo/raftstereo-middlebury`. The checkpoint comes from upstream's `models.zip`, the archive linked from `download_models.sh` at commit `6e93ed2169bd858dbb43033988563f3b0bb49506`.

RAFT-Stereo is a recurrent convolutional network for stereo matching. Its input is a rectified image pair, and its output is a dense disparity map for the left image: left pixel `(x, y)` matches right pixel `(x − d, y)`.

At inference, two convolutional encoders compute features at 1/4 resolution. A correlation volume compares each left feature with every right feature on the same row. A three-level GRU starts from zero disparity and refines it for a chosen number of iterations (`iters`, upstream default 32), each time reading the correlation around its current estimate. A learned convex upsampling then returns full resolution. Adaptation in this repository is gradient fine-tuning; there is no in-context conditioning.

RAFT-Stereo is not a Python package. This repository adds:

- the upstream `core/` source at commit `6e93ed2169bd858dbb43033988563f3b0bb49506`, carried verbatim under `src/raft_stereo_pipeline/third_party/raft_stereo/` with its MIT licence, and imported only after every file matches `UPSTREAM_SHA256`;
- a checkpoint loader that refuses an unpinned or unverified file and reads a verified one with `torch.load(..., weights_only=True)` and `load_state_dict(strict=True)`;
- input and dataset validation, EPE and bad-pixel metrics, a median-constant baseline and an OpenCV StereoSGBM baseline;
- `RaftStereoPipeline.finetune`, a bounded fine-tuning entry point, and a SafeTensors adapter format with a provenance header that `load_artifact` checks;
- a pin tool (`tools/pin_snapshot.py`), a standalone tutorial notebook, and rendered sample data with exact disparity.

The upstream weights do the matching. This repository verifies them, measures them and adapts them.

#### Intended Use and Limitations

The uses below are those the repository code supports and was written for.

###### Primary Intended Uses

- **Task.** Dense disparity estimation for the left image of a rectified stereo pair. Input: two RGB images of one size, each side 64 to 1280 px and at most 1280 × 1024 pixels. Output: a `(H, W)` float32 disparity map in pixels, with larger values meaning nearer surfaces.
- **Evaluation and adaptation.** Measuring the model on the user's own labelled pairs with `epe`, `bad_1px`, `bad_2px`, `bad_3px` and `d1_all` next to the median and SGBM baselines. Then adapting it with a bounded fine-tune and exporting a SafeTensors adapter that reloads onto the verified checkpoint.
- **Application domains envisioned.** Teaching and research on stereo matching. Prototyping depth from calibrated, rectified stereo rigs, for example robotics or mapping rigs, where the operator holds ground truth from structured light, LiDAR or rendering. A rendered scene benchmark of the operator's own design.
- **Role in a larger system.** A pretrained stereo baseline that an application embeds as a library, together with a measured comparison against a classical matcher. The pipeline returns disparity only. Conversion to depth (`focal length × baseline / disparity`) needs the operator's calibration and is not performed.

###### Primary Intended Users

- **Roles.** Machine learning engineers, computer vision researchers and students, and robotics or mapping developers who already operate a calibrated stereo rig.
- **Deployment setting.** Research and teaching notebooks on Google Colab, Kaggle or a Linux Jupyter server. Self-hosted experiments in which a user installs the package and runs it on their own hardware.
- **Assumed competencies.** Users are expected to know what stereo rectification is and to supply rectified pairs. They should know the disparity sign convention (left image reference) and that disparity is not depth without calibration. They should read EPE and bad-pixel rates as complementary measures, and know that a held-out score on a few pairs is not a benchmark. The pipeline cannot detect an unrectified pair, so a user who cannot rectify their own images is not supported.

###### Out-of-scope use cases

- **Capability boundary: unrectified images.** The model searches along image rows only. Two photographs taken without rectification (for example two phone shots) produce disparity that has no geometric meaning, and nothing in the pipeline detects this.
- **Capability boundary: metric depth.** The pipeline outputs disparity in pixels. It does not output depth in metres, and it holds no camera calibration.
- **Capability boundary: confidence.** RAFT-Stereo assigns a disparity to every pixel, including occluded and textureless pixels with no possible match. It returns no confidence or uncertainty value.
- **Capability boundary: monocular input.** One image is refused. For monocular depth, see the public sibling pipelines [`depth-anything-depth-estimation-pipeline`](https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline) and [`zoedepth-metric-depth-pipeline`](https://github.com/kurtvalcorza/zoedepth-metric-depth-pipeline).
- **Input boundary: size.** A side below 64 px or above 1280 px, more than 1,310,720 pixels, or left and right images of different sizes are refused by `validate_pair`. More than 2000 records in one dataset are refused. Ground truth at or above 512 px disparity is not scored, and negative ground-truth disparity is refused.
- **Input boundary: distribution.** Behaviour on night, rain, fog, glare, reflective or transparent surfaces, very large disparities, or sensors other than visible-light cameras is not measured.
- **Decision boundary.** Not for autonomous navigation, collision avoidance, or any safety-relevant distance decision taken without human oversight and independent validation on the deployment rig.

#### Factors

Behaviour varies with scene content and with the stereo rig, not with properties of people.

###### Groups

The pipeline is not human-centric. Its unit of data is a stereo image pair of a scene, and it makes no prediction about people. No group-level evaluation was performed. The upstream training data (SceneFlow, rendered; Middlebury 2014, indoor photographs of objects) was not audited by this repository for the presence of people. A downstream operator whose scenes contain people should check whether disparity quality differs systematically across the people and settings in their own data (for example clothing, skin tone under their lighting, posture, or distance) before relying on it.

###### Instrumentation

- **Training data, per the upstream paper:** SceneFlow was produced by a 3D renderer with exact disparity. Middlebury 2014 was captured with a calibrated stereo rig and structured-light ground truth (Scharstein et al., 2014).
- **Tutorial data:** `samples.py` renders every pair in code at 320 × 224 px. The renderer uses textured slanted backgrounds and textured fronto-parallel objects at up to 40 px disparity. The right camera has a gain of 0.85 to 1.15 and an offset of ±12 grey levels, and both views receive Gaussian sensor noise (standard deviation 2). Ground truth is exact by construction, and occluded pixels are marked invalid.
- **User data:** a rectified pair from the user's own rig, with optional ground-truth disparity as `.pfm`, `.npy` or 16-bit KITTI `.png`.
- **Error propagation:** rectification error (vertical misalignment), unsynchronised capture, rolling-shutter motion, different exposure between the cameras, or a swapped left/right order reach the model directly as matching error. The pipeline refuses negative ground-truth disparity and pairs of different sizes. It cannot detect vertical misalignment, unsynchronised capture or a changed calibration.

###### Environment

- **Operating environment.** Linux x86_64 with CPython 3.12. A CUDA GPU (a Colab or Kaggle T4 is the documented runtime) is recommended. The same code runs on a CPU, because the pure-PyTorch `corr_implementation = "reg"` correlation is used and the optional CUDA sampler is never built. CPU runs are slower: in a local 4-thread CPU check with random weights and nothing else running, one 32-iteration inference on a 320 × 224 pair took 3.3 s and one 12-iteration training step on two pairs took 10 s (one observation each). Everything runs in float32, with mixed precision off. The tutorial notebook installs its pins into an isolated environment (35 hash-locked packages) and supports Linux x86_64 only.
- **Data environment.** The model assumes rectified pairs whose content resembles its training data: rendered objects (SceneFlow) and indoor photographs (Middlebury 2014). Disparity quality is expected to degrade on textureless, reflective or transparent surfaces, at occlusion boundaries, under different exposure between the cameras, and outside the disparity range seen in training. None of this is measured by this repository on real data. The tutorial's rendered pairs resemble SceneFlow's style, so tutorial scores are not evidence about real photographs.

#### Metrics

The measures are chosen for a dense regression output scored against per-pixel ground truth.

###### Performance Measures

The pipeline reports, per pair and averaged over pairs (`disparity_metrics`, `aggregate_metrics`), over valid pixels only:

- `epe` — the mean absolute disparity error in pixels. It captures the size of errors and is pulled up by a few large failures.
- `bad_1px`, `bad_2px`, `bad_3px` — the share of valid pixels whose error exceeds 1, 2 or 3 px. They count failures and ignore their size. ETH3D reports bad-1 and Middlebury bad-2.
- `d1_all` — KITTI's outlier rate: error above 3 px **and** above 5 % of the true disparity.
- `pixel_weighted_epe` — EPE weighted by valid pixels across pairs.

Reading only `epe` hides whether errors are rare and large or common and small. Reading only a bad-pixel rate hides how wrong the failures are. Both are needed to compare a network with SGBM, whose unmatched pixels are filled and can be badly wrong. Every measure is printed next to three baselines scored on the same pixels: `median` (the training split's median disparity everywhere), `sgbm` (OpenCV StereoSGBM, with its matching density reported), and `pretrained` (the checkpoint before adaptation).

Measured in a Google Colab T4 run of the tutorial with the pinned checkpoint (8 held-out rendered pairs, 32 refinement iterations; see Verification records); a second, clean one-pass run reproduced every value except the adapted `bad_3px` and `d1_all`, 0.0046:

| Method | `epe` (px) | `bad_1px` | `bad_3px` | `d1_all` |
|---|---|---|---|---|
| median | 8.2234 | 0.8528 | 0.6321 | 0.6321 |
| sgbm | 0.6533 | 0.0365 | 0.0283 | 0.0283 |
| pretrained | 0.2310 | 0.0122 | 0.0079 | 0.0079 |
| adapted (36 steps, encoders frozen) | 0.1117 | 0.0086 | 0.0045 | 0.0045 |

The adapted model's EPE was lower on 8 of 8 held-out pairs and on 3 of 3 unseen pairs. These are tutorial-sample values on rendered scenes from one seed, not benchmark results. On a degenerate probe of two identical random-dot images (true disparity 0) the pretrained checkpoint returned a median of 235.2 px, while an 8 px shifted probe returned 8.001 px. A diagnostic sweep then showed that the checkpoint reads every uniform random-dot shift of 0–4 px wildly wrong (medians 128–367 px) at dot sizes of 1–8 px, and reads 8 px and 16 px exactly; rendered scenes with small disparities are read well. The cause was not investigated; the tutorial probes 8 px and 16 px. Upstream benchmark values in the paper are reported by the upstream authors and were not reproduced here.

###### Decision thresholds

- The pipeline makes no decision: every pixel receives a continuous disparity, and no threshold is applied to the output.
- The bad-pixel thresholds (1, 2 and 3 px) and the D1 rule (> 3 px and > 5 %) are evaluation conventions taken from ETH3D, Middlebury and KITTI. They are not acceptance thresholds, and no accuracy level was set as a development target.
- The ground-truth ceiling of 512 px follows upstream evaluation. The 700 px loss ceiling follows upstream training.
- The reload check accepts an exported adapter only if its disparity on a reference pair differs from the trained in-memory model by at most 0.001 px on average and 0.01 px at most.
- A deployment that turns disparity into a decision (an obstacle closer than a distance, for example) owns that threshold. The operator should set it on their own labelled pairs. A missed near obstacle (a false negative) usually costs more than a false alarm, so the threshold should be chosen for recall at the distances that matter.

###### Approaches to uncertainty and variability

- **Estimation procedure.** The tutorial uses one seeded holdout split: 32 rendered pairs, 24 for training and 8 held out (`split_dataset`, group-preserving when records carry a `group`). It makes one pass, with each pair weighted equally. No cross-validation, repeated runs or bootstrap are performed.
- **Dispersion.** None is reported. Per-pair rows and the count of held-out pairs on which the adapted EPE is lower are printed, so the spread across pairs is visible but not summarised.
- **Sources of variability.** Rendering, the split, crop positions and the training order are controlled by seeds (`DATASET_SEED`, `SEED`, `NEW_DATA_SEED`). GPU kernels are not forced to be deterministic, so repeated GPU runs, and GPU against CPU runs, can differ in the last digits.
- **Confidence outputs.** The model emits no probability or confidence. A caller who needs per-pixel uncertainty must estimate it separately, for example with left-right consistency checks or an ensemble. Neither is provided.

#### Ethical considerations and biases

No external ethics board or group-specific testing reviewed this model or pipeline.

###### Data

- **Upstream training data, per the upstream paper and README:** SceneFlow (FlyingThings3D, Driving, Monkaa; rendered), then Middlebury 2014 training scenes (indoor photographs of arranged objects). The upstream authors do not enumerate the content beyond these dataset names. This repository did not audit them for personal or restricted content. Their sensitivity is not known, although neither dataset is described by its authors as containing personal data.
- **What this repository distributes:** code, the carried upstream `core/` source with its licence, the checkpoint manifest, and code that renders synthetic sample pairs. It does **not** distribute the checkpoint (the file is git-ignored and fetched from the upstream archive on first use) or any third-party image data.
- **Operator obligation:** images supplied at inference or for fine-tuning may show people, vehicles, property or private places. The pipeline performs no audit for personal or sensitive content. The operator must confirm they may process the images in their environment. Uploads in the notebook stay in the runtime, and an exported adapter carries weights fitted to the operator's data.

###### Human Life

The pipeline is not intended for decisions about health, safety, criminal justice, employment, credit or housing. It has not been validated for any such use by anyone; the only executions recorded are tutorial runs on rendered pairs. Stereo depth is foreseeable in robotics and driving, where errors affect physical safety. Such use would be admissible only with a certified perception stack around it, independent validation on the deployment rig and conditions, and human oversight. None of that is provided or implied here.

###### Mitigations

- **Supply-chain integrity.** `load_upstream` verifies the eight carried upstream files against `UPSTREAM_SHA256` (commit `6e93ed2169bd858dbb43033988563f3b0bb49506`) before import. It refuses to run if a different module named `core` is already imported. `from_pretrained`, `stage_missing_files` and `verify_snapshot` raise while `MODEL_REVISION` is `unpinned`. Once pinned, the checkpoint's byte size and SHA-256 are checked before every load. `extract_member` takes exactly one archive member named `raftstereo-middlebury.pth` and refuses absolute paths, `..`, symbolic links and members over 512 MiB. A mismatch is never replaced from another source.
- **Deserialisation.** The `.pth` file is read with `torch.load(..., weights_only=True)` and `load_state_dict(strict=True)`. Exported adapters are SafeTensors. `read_artifact_metadata` refuses an adapter whose format, model id, revision, upstream commit or architecture arguments differ, and `apply_artifact` refuses a different base digest or a mismatched tensor set.
- **Input integrity.** `validate_pair` refuses non-image inputs, size mismatches and sizes outside the ceilings. `validate_dataset` refuses wrongly shaped or negative disparity, records without valid pixels and duplicate ids. `read_stereo_records` refuses paths outside the dataset folder, missing files and unsupported disparity formats. Each refusal names the rule.
- **Statistical mitigations.** Baselines are fitted or configured on the training split only. The split keeps declared groups on one side. The held-out comparison runs the exported adapter in a fresh process, and `evaluate` refuses baselines scored with a different iteration count.
- **Reproducibility.** Seeds are explicit. The tutorial installs a hash-locked lock (`tutorials/requirements-colab.lock.txt`) with `--require-hashes --only-binary :all:` into an isolated environment. Results carry the model id, revision, upstream commit, dataset SHA-256 and runtime versions.
- **Refusals.** No confidence is reported. The pipeline converts no disparity to depth and performs no rectification.

###### Risks and harms

- **Overconfident disparity outside the training distribution.** The model returns a plausible-looking map even where no match exists (textureless walls, glass, reflections, occluded regions). The operator, and anyone affected by a downstream decision, bears the harm. The risk is likely under normal use on real scenes unlike the training data, and its magnitude depends on the decision the map feeds.
- **Wrong geometry from an unrectified or swapped pair.** Silent garbage results if the pair is not rectified, and nothing detects it. A swapped pair is caught only when ground truth is supplied, through negative disparity. The user bears the harm; it is likely for first-time users.
- **Automation bias from tutorial numbers.** A good held-out score on 8 rendered pairs can be mistaken for real-world accuracy. Downstream users bear this harm, and its magnitude is high if the score drives a deployment choice.
- **Bias from the user's own data.** A fine-tune on a narrow set of scenes can degrade accuracy on other scenes, lighting or rigs. The held-out split measures only the user's own distribution.
- **Undetected data leakage.** Pairs from the same scene placed on both sides of the split inflate the held-out score, unless the user declares a `group`. This is likely with sequences or multi-exposure captures.
- **Privacy.** Stereo images can show identifiable people or places. An adapter fitted to them encodes information about them. The data subjects bear the risk when adapters or outputs are shared.
- **Supply-chain change upstream.** The archive is hosted on Dropbox, and upstream has changed its link twice (2023, 2026). Once pinned, a changed file is refused, which can make the tutorial fail rather than load something else; the operator bears that availability cost.

###### Use cases

The developers consider the following unacceptable even where the model would work:

- covert surveillance or tracking of people, for example estimating the positions or movements of people from stereo cameras without their knowledge or a lawful basis;
- biometric or demographic profiling from depth or shape;
- weapons targeting, or any use that selects people or vehicles for harm;
- deceptive uses, such as presenting tutorial-level disparity as certified measurements in safety, insurance or legal contexts.

The MIT licence of the upstream code and checkpoint imposes no use restrictions. These exclusions are the developers' own. A deployment's terms may add more.

---

## Immutable provenance

| Item | Value |
|---|---|
| Model id | `princeton-vl/RAFT-Stereo/raftstereo-middlebury` |
| Checkpoint revision (SHA-256 of `raftstereo-middlebury.pth`) | `d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819` |
| Checkpoint source | `https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1` (member `raftstereo-middlebury.pth`) |
| Checkpoint size | 44,617,876 bytes (measured at pin time) |
| Upstream code | `princeton-vl/RAFT-Stereo` at `6e93ed2169bd858dbb43033988563f3b0bb49506`, carried and verified against `UPSTREAM_SHA256` |
| Architecture | upstream demo defaults; 11,116,176 parameters, 337 state tensors |
| Licence | MIT (upstream code and checkpoint); this repository's code Apache-2.0 |
| Manifest | `weights/raftstereo-middlebury/dimer-base-manifest.json` (format `dimer_url_snapshot`) |

## Input/output contract

- `RaftStereoPipeline.estimate(left, right, iters=32)` takes two `PIL.Image.Image` of one size and returns `{"disparity": (H, W) float32, "iters", "width", "height", "adapted", "model_id", "model_revision"}`.
- `RaftStereoPipeline.evaluate(records, iters=32)` takes records `{"left", "right", "disparity", "valid"?}` and returns the aggregated metrics with `per_pair` rows.
- `RaftStereoPipeline.finetune(records, epochs=3, batch_size=2, learning_rate=2e-5, train_iters=12, freeze_encoders=True, seed=20261004)` uses AdamW (weight decay 1e-5), upstream's sequence loss, gradient clipping at 1.0 and frozen BatchNorm. Pairs larger than 448 × 320 px are cropped for training only.
- `save_artifact(path)` / `load_artifact(path)` write and read the SafeTensors adapter (format `raft-stereo-adapter-v1`).

## Verification records

- **Date:** 2026-10-05
- **Subject:** `tutorials/raft_stereo_colab.ipynb`, blob `c9bd882` at commit `f0d56f1` (2 px and 8 px probes), with the pinned checkpoint
- **Runtime:** fresh Google Colab runtime, Tesla T4; kernel CPython 3.13.15; isolated environment CPython 3.12.12, `torch 2.14.0+cu130` with CUDA
- **Procedure:** `Run all` with no field edited, one pass (execution counts 1–14)
- **Observed result:** every stage completed without error; held-out values as in Performance Measures; reload parity exact; the 2 px probe was read as 149.8 px and the 8 px probe as 8.001 px. The executed copy is in `docs/execution-evidence/2026-10-05/`
- **Caveats:** the 2 px probe failure led to the 8 px and 16 px probes of the current notebook, which has not yet been run on a hosted runtime; BYOD branches not run


- **Date:** 2026-10-05
- **Subject:** `tutorials/raft_stereo_colab.ipynb`, blob `b6ffed7` at commit `49e50cc`, with the pinned checkpoint
- **Runtime:** Google Colab, Tesla T4; kernel CPython 3.13.15; isolated environment CPython 3.12.12, `torch 2.14.0+cu130` with CUDA
- **Procedure:** default fields; the notebook's cells executed in the Colab session that had run the pin dry run, with three extra executions between the inference and reload cells
- **Observed result:** every stage completed without error; checkpoint verified; the values in Performance Measures; adapter reload parity exact (0.0 px). The executed copy is in `docs/execution-evidence/2026-10-05/`
- **Caveats:** not a strict clean-state one-pass `Run all`; BYOD branches not run; zero-disparity probe failed as described in Performance Measures


- **Date:** 2026-10-04
- **Subject:** `tutorials/raft_stereo_colab.ipynb`, regenerated in a scratch copy of the repository after a random-weight checkpoint of the carried architecture had been pinned there with `tools/pin_snapshot.py --archive`
- **Runtime:** Linux x86_64 CPU container, 4 threads, no GPU; kernel CPython 3.12.12; isolated environment built by the notebook (uv 0.12.15, CPython 3.12.12, `torch 2.14.0+cu130` running on CPU, `opencv-python-headless 5.0.0.93`)
- **Procedure:** all cells executed in order with `nbclient`; checkpoint pre-staged because the archive host is unreachable there; non-default fields `N_PAIRS = 6`, `EPOCHS = 1`, `VALID_ITERS = 4`, `TRAIN_ITERS = 2`, and both BYOD branches on with rendered BYOD data; a second run pointed the BYOD dataset branch at a `pairs.json` naming `../outside/im1.png`
- **Observed result:** first run: every cell completed in 347 s; 16 carried files and 8 upstream files verified; 35 locked packages installed; sample and BYOD reload checks reported `equivalent: True`. Second run: the default path completed, and the BYOD `prepare` stage refused the folder with `pairs.json entry 2: file '../outside/im1.png' must be relative to the dataset folder`
- **Caveats:** random weights, so no metric describes the model; not a hosted runtime; the real checkpoint, a GPU and Colab's kernel were not exercised

The full record and the remaining release steps are in [`docs/release-verification.md`](https://github.com/kurtvalcorza/raft-stereo-pipeline/blob/main/docs/release-verification.md).

## References

- Lipson, L., Teed, Z., & Deng, J. (2021). RAFT-Stereo: Multilevel recurrent field transforms for stereo matching. *International Conference on 3D Vision (3DV)*. [arXiv:2109.07547](https://arxiv.org/abs/2109.07547)
- Hirschmüller, H. (2008). Stereo processing by semiglobal matching and mutual information. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 30*(2), 328–341. https://doi.org/10.1109/TPAMI.2007.1166
- Mayer, N., Ilg, E., Häusser, P., Fischer, P., Cremers, D., Dosovitskiy, A., & Brox, T. (2016). A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. *CVPR*. [arXiv:1512.02134](https://arxiv.org/abs/1512.02134)
- Scharstein, D., Hirschmüller, H., Kitajima, Y., Krathwohl, G., Nešić, N., Wang, X., & Westling, P. (2014). High-resolution stereo datasets with subpixel-accurate ground truth. *GCPR*. https://doi.org/10.1007/978-3-319-11752-2_3
- Upstream repository: [princeton-vl/RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) (MIT).
