# RST-M1 — does this pipeline reproduce upstream's output on near-zero disparity?

**Date:** 10 October 2026. **Where:** a local CPU (Windows, CPython 3.12, `torch 2.14.0+cpu`), in a venv installed from
`tutorials/requirements-colab.lock.txt` with `--require-hashes` (the CUDA/NVIDIA/triton entries and the CUDA torch wheel
replaced by the CPU torch wheel of the same version). This is **not clean-runtime notebook evidence**; it answers one
question from the review's RST-M1 acceptance check.

**Inputs:** the pinned checkpoint `raftstereo-middlebury.pth`, SHA-256 `d22e84c0…5819` (fetched from the manifest URL and
digest-verified by `RaftStereoPipeline.from_pretrained`); the carried upstream `core/` at `princeton-vl/RAFT-Stereo@6e93ed2`
(verified against `UPSTREAM_SHA256`); upstream `demo.py` at the same commit (SHA-256 `a00d7af2…22cc`, read, not carried).

## Method

`rst_m1_upstream_comparison.py` (in this folder) runs every pair three ways at 32 update iterations:

1. **this pipeline:** `RaftStereoPipeline.estimate`;
2. **upstream `demo.py`, re-implemented line for line:** the pair is written to PNG and read back with demo.py's
   `load_image` (`np.array(Image.open(f)).astype(np.uint8)` → `permute(2, 0, 1).float()[None]`), padded with upstream's
   `InputPadder(divis_by=32)`, run with `model(image1, image2, iters=32, test_mode=True)` and unpadded; a separately built
   `RAFTStereo` loads the raw checkpoint state (`module.` prefix stripped, `strict=True`) with the package's architecture
   arguments and `corr_implementation="reg"`;
3. the same as 2 with `corr_implementation="alt"`, the correlation named in upstream README's Middlebury demo command
   (`--mixed_precision` omitted: CPU).

`demo.py` itself was not executed: it wraps the model in `DataParallel(device_ids=[0])`, which needs a CUDA device.

## Result

Image 320 × 224 px. Medians and EPE over the scored pixels (the left `shift` columns of a random-dot pair are unscored;
the identical rendered scene is scored everywhere).

| Pair | True px | Pipeline median / EPE | Upstream (reg) median / EPE | Upstream (alt) median / EPE | max \|pipeline − upstream reg\| |
|---|---|---|---|---|---|
| random dots, 0 px (identical images) | 0 | 235.666 / 235.320 | 235.666 / 235.320 | 235.755 / 235.495 | 0.0 |
| random dots, 1 px | 1 | 199.323 / 197.948 | 199.323 / 197.948 | 199.301 / 197.926 | 0.0 |
| random dots, 2 px | 2 | 149.825 / 147.173 | 149.825 / 147.173 | 149.825 / 147.173 | 0.0 |
| random dots, 4 px | 4 | 131.932 / 127.590 | 131.932 / 127.590 | 131.933 / 127.591 | 0.0 |
| random dots, 8 px | 8 | 8.001 / 0.032 | 8.001 / 0.032 | 8.001 / 0.032 | 0.0 |
| rendered scene 0 (seed 0) as both images | 0 | 416.005 / 416.267 | 416.005 / 416.267 | 415.953 / 416.227 | 0.0 |

Raw values: `rst_m1_upstream_comparison.json`.

## Conclusion

- **This pipeline reproduces upstream's inference path exactly** (maximum absolute difference 0.0 px on every pair), and
  upstream's README correlation (`alt`) gives the same failure. The near-zero failure is therefore a property of the
  pinned checkpoint under upstream's own code, not of this pipeline's tensor preparation, padding or loading.
- **It is not specific to random dots:** a rendered tutorial scene given as both images (true disparity 0 everywhere) is
  read as about 416 px, more than the image is wide.
- The 0 px CPU median (235.666 px) agrees with the recorded Colab T4 value through the pipeline (235.2 px).
- **Not measured:** a scene whose *background only* sits at 0–2 px while the foreground has larger disparity, and real
  photographs. Uniform 0–4 px whole-frame shifts fail; how partial near-zero regions behave remains unknown.
