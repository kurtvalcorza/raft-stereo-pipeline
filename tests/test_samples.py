"""The rendered tutorial data: exact geometry, determinism, and the disparity file formats."""

# ruff: noqa: E501  -- refusal messages and docstrings name the rule in full; they are kept on one line
import numpy as np
import pytest
from PIL import Image

from raft_stereo_pipeline.samples import (
    disparity_to_rgb,
    random_dot_pair,
    read_disparity,
    read_pfm,
    render_stereo_scene,
    stereo_dataset,
    write_disparity_png16,
    write_pfm,
)


def _sgbm(record, num_disparities=48):
    import cv2

    matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities, blockSize=5, P1=200, P2=800, uniquenessRatio=10)
    left = np.asarray(record["left"].convert("L"))
    right = np.asarray(record["right"].convert("L"))
    return matcher.compute(left, right).astype(np.float32) / 16.0


def test_rendered_pair_geometry_is_exact():
    """A classical matcher recovers the rendered disparity on valid pixels: the right view really is the left view at x - d."""
    record = render_stereo_scene(7)
    disparity = _sgbm(record)
    matched = (disparity >= 0) & record["valid"]
    error = np.abs(disparity - record["disparity"])[matched]
    assert matched.sum() > 0.6 * record["valid"].sum()
    assert error.mean() < 1.5
    assert (error > 3).mean() < 0.1


def test_rendering_is_deterministic_and_seed_dependent():
    a, b, c = render_stereo_scene(3), render_stereo_scene(3), render_stereo_scene(4)
    assert a["left"].tobytes() == b["left"].tobytes() and np.array_equal(a["disparity"], b["disparity"])
    assert a["left"].tobytes() != c["left"].tobytes()
    assert a["disparity"].dtype == np.float32 and a["valid"].dtype == bool
    assert a["disparity"][a["valid"]].min() >= 2.0 and a["disparity"].max() <= 40.0


def test_dataset_scenes_are_distinct_across_seeds():
    first = {r["id"] for r in stereo_dataset(3, seed=0, width=96, height=64)}
    second = {r["id"] for r in stereo_dataset(3, seed=1, width=96, height=64)}
    assert not first & second


@pytest.mark.parametrize("shift", [0, 2, 8])
def test_random_dot_pair_has_the_stated_disparity(shift):
    pair = random_dot_pair(shift)
    left, right = np.asarray(pair["left"]), np.asarray(pair["right"])
    # left pixel x reappears at right pixel x - shift
    assert np.array_equal(left[:, shift:], right[:, : right.shape[1] - shift])
    assert not pair["valid"][:, :shift].any() and pair["valid"][:, shift:].all()
    assert np.all(pair["disparity"] == shift)


def test_pfm_round_trip_and_inf_is_invalid(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    values[0, 0] = np.inf
    write_pfm(tmp_path / "d.pfm", values)
    assert np.array_equal(read_pfm(tmp_path / "d.pfm"), values)
    disparity, valid = read_disparity(tmp_path / "d.pfm")
    assert not valid[0, 0] and valid.sum() == 11 and disparity[0, 0] == 0


def test_kitti_png_round_trip_and_zero_is_invalid(tmp_path):
    values = np.array([[0.0, 1.5], [10.25, 200.0]], dtype=np.float32)
    write_disparity_png16(tmp_path / "d.png", values)
    disparity, valid = read_disparity(tmp_path / "d.png")
    assert np.allclose(disparity[valid], values[valid], atol=1 / 256)
    raw = np.zeros((2, 2), dtype=np.uint16)
    Image.fromarray(raw).save(tmp_path / "z.png")
    assert not read_disparity(tmp_path / "z.png")[1].any()


def test_read_disparity_refuses_unsupported_inputs(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "rgb.png")
    with pytest.raises(ValueError, match="16-bit"):
        read_disparity(tmp_path / "rgb.png")
    (tmp_path / "d.txt").write_text("1")
    with pytest.raises(ValueError, match="unsupported disparity format"):
        read_disparity(tmp_path / "d.txt")
    np.save(tmp_path / "d.npy", np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="2-D"):
        read_disparity(tmp_path / "d.npy")


def test_colour_map_shape():
    image = disparity_to_rgb(np.linspace(0, 40, 64 * 32).reshape(32, 64), vmax=40)
    assert image.size == (64, 32) and image.mode == "RGB"
