"""RST-M1 investigation: pipeline vs upstream-demo-equivalent path on identical and small-shift pairs (CPU)."""
import json, sys, time, argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image
import raft_stereo_pipeline as R
from raft_stereo_pipeline import pipeline as P
from raft_stereo_pipeline.samples import random_dot_pair, stereo_dataset

W = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
torch.set_num_threads(8)
pipe = R.RaftStereoPipeline.from_pretrained(device="cpu", weights_dir=W, allow_download=True)
P.load_upstream()
from core.raft_stereo import RAFTStereo
from core.utils.utils import InputPadder

raw = torch.load(W / "raftstereo-middlebury.pth", map_location="cpu")
raw = {k.removeprefix("module."): v for k, v in raw.items()}
def upstream_model(corr):
    args = argparse.Namespace(**{**P.MODEL_ARGS, "corr_implementation": corr})
    m = RAFTStereo(args); m.load_state_dict(raw, strict=True); return m.eval()
ups = {c: upstream_model(c) for c in ("reg", "alt")}

def demo_load(path):  # upstream demo.py load_image, verbatim logic
    img = np.array(Image.open(path)).astype(np.uint8)
    return torch.from_numpy(img).permute(2, 0, 1).float()[None]

def demo_run(model, l, r, iters=32):
    i1, i2 = demo_load(l), demo_load(r)
    padder = InputPadder(i1.shape, divis_by=32); i1, i2 = padder.pad(i1, i2)
    with torch.no_grad():
        _, flow_up = model(i1, i2, iters=iters, test_mode=True)
    return (-padder.unpad(flow_up).squeeze()).numpy()

cases = []
for s in (0, 1, 2, 4, 8):
    p = random_dot_pair(s); cases.append((f"random_dot_{s}px", p["left"], p["right"], float(s), p["valid"]))
scene = stereo_dataset(1, seed=0)[0]
cases.append(("rendered_scene_identical", scene["left"], scene["left"], 0.0, None))
rows = []
for name, left, right, truth, valid in cases:
    lp, rp = OUT / f"{name}_l.png", OUT / f"{name}_r.png"; left.save(lp); right.save(rp)
    v = np.ones((left.height, left.width), bool) if valid is None else valid
    row = {"case": name, "true_px": truth}
    t = time.time(); d = pipe.estimate(left, right)["disparity"]; row["pipeline_median"] = round(float(np.median(d[v])), 3); row["pipeline_epe"] = round(float(np.abs(d[v] - truth).mean()), 3)
    for c, m in ups.items():
        u = demo_run(m, lp, rp); row[f"upstream_{c}_median"] = round(float(np.median(u[v])), 3); row[f"upstream_{c}_epe"] = round(float(np.abs(u[v] - truth).mean()), 3)
        if c == "reg": row["max_abs_diff_pipeline_vs_upstream_reg"] = float(np.abs(u - d).max())
    row["seconds"] = round(time.time() - t, 1); print(json.dumps(row), flush=True); rows.append(row)
meta = {"torch": torch.__version__, "upstream_commit": P.UPSTREAM_COMMIT, "checkpoint_sha256": P.MODEL_REVISION, "device": "cpu", "iters": 32, "image_size": [cases[0][1].width, cases[0][1].height]}
(OUT / "m1_probe_results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
print(json.dumps(meta))
