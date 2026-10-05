# Random-dot probe diagnostic — 2026-10-05

Run by the maintainer in a Google Colab T4 runtime, outside the tutorial: the pinned checkpoint
(`raftstereo-middlebury.pth`, SHA-256 `d22e84c0…5819`) loaded through `RaftStereoPipeline.from_pretrained`
from branch `ccr-24656dfc-ax1ln2` at `f0d56f1`, 32 refinement iterations, 320 × 224 random-dot pairs whose right
image is the left image shifted by `shift` px; `block` is the dot size in px. Median and EPE over the columns that
have a match (x ≥ shift).

| block | shift | median_px | epe |
|---:|---:|---:|---:|
| 1 | 0 | 250.528 | 249.667 |
| 1 | 1 | 214.325 | 213.037 |
| 1 | 2 | 146.575 | 143.689 |
| 1 | 4 | 169.012 | 165.430 |
| 1 | 8 | 8.004 | 0.032 |
| 1 | 16 | 16.013 | 0.030 |
| 2 | 0 | 241.467 | 240.866 |
| 2 | 1 | 169.382 | 168.174 |
| 2 | 2 | 136.286 | 133.753 |
| 2 | 4 | 128.399 | 124.645 |
| 2 | 8 | 8.001 | 0.026 |
| 2 | 16 | 16.009 | 0.019 |
| 4 | 0 | 346.525 | 346.583 |
| 4 | 1 | 272.551 | 271.826 |
| 4 | 2 | 146.821 | 144.858 |
| 4 | 4 | 153.336 | 149.207 |
| 4 | 8 | 8.005 | 0.030 |
| 4 | 16 | 16.008 | 0.027 |
| 8 | 0 | 367.328 | 364.298 |
| 8 | 1 | 311.413 | 303.846 |
| 8 | 2 | 322.081 | 318.252 |
| 8 | 4 | 219.941 | 214.553 |
| 8 | 8 | 8.002 | 0.028 |
| 8 | 16 | 16.015 | 0.030 |

Every uniform shift of 0–4 px failed for every dot size (medians 128–367 px); 8 px and 16 px were read exactly
(EPE ≤ 0.032 px) for every dot size. Rendered scenes with disparities from about 2 px to 40 px were read
accurately in the same runtime, so the failure is specific to small uniform whole-frame shifts of random dots. Its
cause was not investigated. The tutorial's probes were changed to 8 px and 16 px as a result.
