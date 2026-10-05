"""Tutorial stereo samples with exact ground-truth disparity, and disparity file readers/writers.

Every sample is rendered in code from a seed, so it needs no download, no licence and no credential, and its
ground truth is exact by construction. A scene is a slanted, textured background plane and a few textured
fronto-parallel objects (rectangles and ellipses) at larger disparities. The left view samples each surface at its
own coordinates; the right view samples it at ``x + d``, where ``d`` is that surface's disparity, and the front-most
surface (largest disparity) wins. The right camera then gets a different gain and offset, and both views get sensor
noise, so the two images do not match pixel for pixel even where the geometry does.

Conventions (the same as the pipeline and the upstream RAFT-Stereo code):

* ``disparity[y, x] = d`` means left pixel ``(x, y)`` shows the same surface point as right pixel ``(x - d, y)``;
  disparities are positive pixels and the arrays are float32 of shape ``(H, W)``.
* ``valid`` marks the pixels whose point is visible in both views (in frame and not occluded); the metrics score
  only those pixels (the "non-occluded" protocol).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 224
SAMPLE_MAX_DISPARITY = 40.0
SAMPLE_MIN_DISPARITY = 2.0
SAMPLE_KIND = "synthetic"
SAMPLE_LICENSE = "generated in code by this package (no third-party data)"


def _value_noise(rng: np.random.Generator, height: int, width: int, scales: tuple[int, ...]) -> np.ndarray:
    """Multi-octave value noise in [0, 1]: random grids at several cell sizes, bicubically upsampled and summed."""
    total = np.zeros((height, width), dtype=np.float64)
    weight_sum = 0.0
    for scale in scales:
        grid = rng.random((max(2, height // scale + 2), max(2, width // scale + 2)))
        layer = Image.fromarray((grid * 255).astype(np.uint8)).resize((width, height), Image.BICUBIC)
        weight = float(scale) ** 0.5
        total += weight * np.asarray(layer, dtype=np.float64) / 255.0
        weight_sum += weight
    total /= weight_sum
    low, high = total.min(), total.max()
    return (total - low) / max(high - low, 1e-6)


def _texture(rng: np.random.Generator, height: int, width: int, *, low_texture: bool) -> np.ndarray:
    """An RGB texture (H, W, 3) in [0, 255]: a base colour modulated by noise; low-texture surfaces vary little."""
    base = rng.uniform(40, 220, size=3)
    noise = _value_noise(rng, height, width, (2, 4, 8, 16, 32))
    amplitude = 0.08 if low_texture else rng.uniform(0.6, 0.95)
    tint = _value_noise(rng, height, width, (16, 48))[..., None] * rng.uniform(-40, 40, size=3)
    shade = 1.0 - amplitude / 2 + amplitude * noise[..., None]
    return np.clip(base * shade + tint, 0, 255)


def _sample(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear lookup of ``texture`` at float columns ``u`` and rows ``v`` (edge-clamped)."""
    from scipy.ndimage import map_coordinates

    channels = [map_coordinates(texture[..., c], [v, u], order=1, mode="nearest") for c in range(3)]
    return np.stack(channels, axis=-1)


def render_stereo_scene(
    seed: int,
    *,
    width: int = SAMPLE_WIDTH,
    height: int = SAMPLE_HEIGHT,
    max_disparity: float = SAMPLE_MAX_DISPARITY,
) -> dict[str, Any]:
    """Render one rectified stereo pair with exact disparity; the same seed always gives the same record."""
    if width < 64 or height < 64:
        raise ValueError(f"scene must be at least 64 x 64 px, got {width} x {height}")
    if not SAMPLE_MIN_DISPARITY + 4 <= max_disparity <= width / 2:
        raise ValueError(f"max_disparity must be in [{SAMPLE_MIN_DISPARITY + 4}, {width / 2}], got {max_disparity}")
    rng = np.random.default_rng(seed)
    pad = int(np.ceil(max_disparity)) + 8
    canvas_w = width + pad
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)

    # Background: a slanted plane d = a + b*u + c*y (nearer, so larger disparity, towards the bottom of the image).
    a = rng.uniform(SAMPLE_MIN_DISPARITY, SAMPLE_MIN_DISPARITY + 4)
    b = rng.uniform(-(a - SAMPLE_MIN_DISPARITY) / width, 0.01)  # keeps the plane at or above SAMPLE_MIN_DISPARITY
    c = rng.uniform(0.0, 0.3 * max_disparity) / height
    background = _texture(rng, height, canvas_w, low_texture=False)

    objects = []
    for _ in range(int(rng.integers(3, 7))):
        half_w = rng.uniform(0.06, 0.2) * width
        half_h = rng.uniform(0.08, 0.25) * height
        disparity = rng.uniform(0.35 * max_disparity, max_disparity)
        objects.append(
            {
                "shape": "ellipse" if rng.random() < 0.4 else "rectangle",
                "cx": rng.uniform(0.1, 0.95) * width,
                "cy": rng.uniform(0.1, 0.9) * height,
                "hw": half_w,
                "hh": half_h,
                "d": disparity,
                "texture": _texture(rng, height, canvas_w, low_texture=bool(rng.random() < 0.25)),
            }
        )
    objects.sort(key=lambda o: o["d"])  # back to front

    def inside(obj: dict[str, Any], u: np.ndarray, v: np.ndarray) -> np.ndarray:
        du, dv = (u - obj["cx"]) / obj["hw"], (v - obj["cy"]) / obj["hh"]
        if obj["shape"] == "ellipse":
            return du**2 + dv**2 <= 1.0
        return (np.abs(du) <= 1.0) & (np.abs(dv) <= 1.0)

    # Left view: surfaces sampled at their own coordinates.
    left_layer = np.zeros((height, width), dtype=np.int16)  # 0 = background, i + 1 = objects[i]
    disparity = a + b * xs + c * ys
    left = _sample(background, xs, ys)
    for index, obj in enumerate(objects):
        mask = inside(obj, xs, ys)
        left_layer[mask] = index + 1
        disparity[mask] = obj["d"]
        left[mask] = _sample(obj["texture"], xs[mask], ys[mask])

    # Right view: right pixel x sees the surface point whose left coordinate is u = x + d.
    right_layer = np.zeros((height, width), dtype=np.int16)
    u_background = (xs + a + c * ys) / (1.0 - b)
    right = _sample(background, u_background, ys)
    for index, obj in enumerate(objects):
        u = xs + obj["d"]
        mask = inside(obj, u, ys)
        right_layer[mask] = index + 1
        right[mask] = _sample(obj["texture"], u[mask], ys[mask])

    # A left pixel is valid when its point is in the right frame and the same surface is front-most there.
    target = np.rint(xs - disparity).astype(np.int64)
    in_frame = target >= 0
    rows = ys.astype(np.int64)
    same_surface = np.zeros_like(in_frame)
    same_surface[in_frame] = right_layer[rows[in_frame], target[in_frame]] == left_layer[in_frame]
    valid = in_frame & same_surface

    # Photometric differences between the cameras: gain and offset on the right view, sensor noise on both.
    gain, offset = rng.uniform(0.85, 1.15), rng.uniform(-12, 12)
    right = right * gain + offset
    left = left + rng.normal(0.0, 2.0, left.shape)
    right = right + rng.normal(0.0, 2.0, right.shape)
    return {
        "id": f"scene-{seed}",
        "left": Image.fromarray(np.clip(np.rint(left), 0, 255).astype(np.uint8)),
        "right": Image.fromarray(np.clip(np.rint(right), 0, 255).astype(np.uint8)),
        "disparity": disparity.astype(np.float32),
        "valid": valid,
        "seed": int(seed),
        "kind": SAMPLE_KIND,
    }


def stereo_dataset(
    n_pairs: int, *, seed: int = 0, width: int = SAMPLE_WIDTH, height: int = SAMPLE_HEIGHT
) -> list[dict[str, Any]]:
    """``n_pairs`` independent scenes; scene ``i`` uses seed ``seed * 100003 + i`` so datasets never share scenes."""
    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs < 1:
        raise ValueError(f"n_pairs must be a positive int, got {n_pairs!r}")
    return [render_stereo_scene(seed * 100003 + i, width=width, height=height) for i in range(n_pairs)]


def random_dot_pair(
    shift: int, *, width: int = SAMPLE_WIDTH, height: int = SAMPLE_HEIGHT, seed: int = 0
) -> dict[str, Any]:
    """A random-dot pair in which left pixel ``x`` reappears at right pixel ``x - shift``: disparity = ``shift``.

    ``shift = 0`` gives two identical images (every point at infinity). The left ``shift`` columns have no match in
    the right view and are marked invalid.
    """
    if isinstance(shift, bool) or not isinstance(shift, int) or not 0 <= shift < width // 2:
        raise ValueError(f"shift must be an int in [0, {width // 2}), got {shift!r}")
    rng = np.random.default_rng(seed)
    dots = (rng.random((height, width + shift)) > 0.5).astype(np.float64)
    texture = np.repeat((40 + 175 * dots)[..., None], 3, axis=2)
    left = texture[:, :width]  # left pixel x shows texture column x
    right = texture[:, shift : shift + width]  # right pixel x - shift shows the same column
    valid = np.ones((height, width), dtype=bool)
    valid[:, :shift] = False
    return {
        "id": f"random-dots-shift-{shift}",
        "left": Image.fromarray(left.astype(np.uint8)),
        "right": Image.fromarray(right.astype(np.uint8)),
        "disparity": np.full((height, width), float(shift), dtype=np.float32),
        "valid": valid,
        "kind": "probe",
    }


# --------------------------------------------------------------------------- disparity files


def read_pfm(path: str | Path) -> np.ndarray:
    """Read a single-channel PFM file (Middlebury, ETH3D, SceneFlow) as a float32 (H, W) array, top row first."""
    with open(path, "rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"{path}: not a PFM file (header {header!r})")
        if header == "PF":
            raise ValueError(f"{path}: a 3-channel PFM is not a disparity map")
        dims = handle.readline().decode("ascii").split()
        while not dims:
            dims = handle.readline().decode("ascii").split()
        width, height = (int(v) for v in dims)
        scale = float(handle.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.fromfile(handle, dtype=dtype, count=width * height)
    if data.size != width * height:
        raise ValueError(f"{path}: PFM holds {data.size} values, header says {width} x {height}")
    return np.flipud(data.reshape(height, width)).astype(np.float32)


def write_pfm(path: str | Path, array: np.ndarray) -> None:
    """Write a float32 (H, W) array as a little-endian single-channel PFM."""
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"PFM disparity must be 2-D, got shape {array.shape}")
    with open(path, "wb") as handle:
        handle.write(f"Pf\n{array.shape[1]} {array.shape[0]}\n-1\n".encode("ascii"))
        np.flipud(array).astype("<f4").tofile(handle)


def read_disparity(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a ground-truth disparity file; return ``(disparity, valid)``.

    * ``.pfm`` — float disparity; ``inf``/``nan`` (Middlebury's "unknown") are invalid.
    * ``.npy`` — float (H, W) array; non-finite values are invalid.
    * ``.png`` — 16-bit KITTI convention, disparity = value / 256, and 0 means "no ground truth".
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pfm":
        disparity = read_pfm(path)
        valid = np.isfinite(disparity)
    elif suffix == ".npy":
        disparity = np.load(path, allow_pickle=False).astype(np.float32)
        if disparity.ndim != 2:
            raise ValueError(f"{path.name}: disparity array must be 2-D (H, W), got shape {disparity.shape}")
        valid = np.isfinite(disparity)
    elif suffix == ".png":
        with Image.open(path) as handle:
            raw = np.asarray(handle)
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError(
                f"{path.name}: a PNG disparity map must be single-channel 16-bit (KITTI convention, value / 256), "
                f"got dtype {raw.dtype} with shape {raw.shape}"
            )
        disparity = raw.astype(np.float32) / 256.0
        valid = raw > 0
    else:
        raise ValueError(f"{path.name}: unsupported disparity format {suffix!r}; use .pfm, .npy or 16-bit .png")
    disparity = np.where(valid, disparity, 0.0).astype(np.float32)
    return disparity, valid


def write_disparity_png16(path: str | Path, disparity: np.ndarray) -> None:
    """Write disparity in the KITTI 16-bit PNG convention (value = round(d * 256), clipped to 1..65535)."""
    values = np.clip(np.rint(np.asarray(disparity, dtype=np.float64) * 256.0), 1, 65535).astype(np.uint16)
    Image.fromarray(values).save(path)


# A perceptually ordered colour ramp (dark blue -> cyan -> yellow -> red); near surfaces (large disparity) are warm.
_RAMP = np.array(
    [[20, 20, 90], [30, 90, 200], [40, 180, 220], [120, 220, 120], [240, 220, 60], [240, 120, 40], [180, 20, 30]],
    dtype=np.float64,
)


def disparity_to_rgb(disparity: np.ndarray, *, vmax: float | None = None, valid: np.ndarray | None = None) -> Image.Image:
    """Colour-code a disparity map (0 -> dark blue, ``vmax`` -> dark red); invalid pixels are drawn black."""
    disparity = np.asarray(disparity, dtype=np.float64)
    top = float(vmax) if vmax else float(np.nanmax(np.where(np.isfinite(disparity), disparity, 0.0)) or 1.0)
    scaled = np.clip(np.nan_to_num(disparity / max(top, 1e-6)), 0.0, 1.0) * (len(_RAMP) - 1)
    low = np.floor(scaled).astype(int)
    high = np.minimum(low + 1, len(_RAMP) - 1)
    frac = (scaled - low)[..., None]
    rgb = _RAMP[low] * (1 - frac) + _RAMP[high] * frac
    if valid is not None:
        rgb[~np.asarray(valid, dtype=bool)] = 0
    return Image.fromarray(rgb.astype(np.uint8))


def error_to_rgb(error: np.ndarray, valid: np.ndarray, *, limit: float = 3.0) -> Image.Image:
    """Absolute disparity error as grey (0 px) to red (``limit`` px or more); invalid pixels are black."""
    level = np.clip(np.asarray(error, dtype=np.float64) / limit, 0.0, 1.0)[..., None]
    rgb = np.array([235.0, 235.0, 235.0]) * (1 - level) + np.array([200.0, 0.0, 0.0]) * level
    rgb[~np.asarray(valid, dtype=bool)] = 0
    return Image.fromarray(rgb.astype(np.uint8))


def panel(images: list[Image.Image], labels: list[str], *, columns: int = 3) -> Image.Image:
    """Tile same-size images into a grid with a one-line caption above each tile."""
    from PIL import ImageDraw

    if not images or len(images) != len(labels):
        raise ValueError("panel needs one label per image")
    width, height = images[0].size
    caption = 18
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        x, y = (index % columns) * width, (index // columns) * (height + caption)
        sheet.paste(image.convert("RGB").resize((width, height)), (x, y + caption))
        draw.text((x + 4, y + 3), label, fill="black")
    return sheet
