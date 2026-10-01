"""Seeded warp-pair dataset on the warp mirror.

Pipeline follows the reference foundry protocol exactly:
stride-selected surface samples in the unit box, uniform coefficient
draws from mulberry32 streams, forward warp, train-split mean/scale
normalization. float64 throughout; the cascade casts to float32 for
training."""

import json
import os

import numpy as np

from warp import warp_points

from train import config
from train.rng import mulberry32


def mesh_positions() -> np.ndarray:
    mesh = json.load(open(config.MESH_PATH))
    assert mesh["meshId"] == config.MESH_ID, mesh["meshId"]
    return np.array(mesh["positions"], dtype=np.float64).reshape(-1, 3)


def mesh_vert_count() -> int:
    return mesh_positions().shape[0]


def sample_indices(count: int = config.SAMPLE_COUNT) -> list:
    """Stride selection, mirroring the gallery sampler."""
    verts = mesh_vert_count()
    step = verts / count
    return [int(i * step) % verts for i in range(count)]


def rest_unit() -> np.ndarray:
    """The 64 sample points, unit-box normalized (64, 3)."""
    pos = mesh_positions()
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    return ((pos - lo) / (hi - lo))[sample_indices()]


def _mode_cache_path() -> str:
    from train import config as _c

    return os.path.join(_c.ROOT, "fixtures", "mode-weights.json")


def mode_weights(h: float = 1e-3, n_warps: int = 64,
                 cache: bool = True) -> np.ndarray:
    """Per-coefficient loss weights: RMS vertex motion per unit
    coefficient (central differences on the rest samples), averaged
    over draws from the training warp distribution and normalized to
    mean 1. Sensitivity spreads with amplitude (0.60–1.69x at a hard
    warp vs near-uniform at rest), so rest-centered weights would aim
    pressure at the wrong modes. Coefficient MSE then tracks vertex
    MSE to first order, so training pressure lands where the eye lives.

    Deterministic for fixed (h, n_warps); the full-size result is
    cached in fixtures/mode-weights.json (≈2 min to recompute)."""
    from train import config as _c

    unit = rest_unit()
    if cache and h == 1e-3 and n_warps == 64:
        path = _mode_cache_path()
        if os.path.exists(path):
            w = np.array(json.load(open(path))["weights"])
            assert w.shape == (_c.WARP_K,)
            return w
    rng = mulberry32(_c.TRAIN_SEED + 999)
    acc = np.zeros(_c.WARP_K)
    for _ in range(n_warps):
        c = random_coeffs(rng)
        for k in range(_c.WARP_K):
            ek = np.zeros(_c.WARP_K)
            ek[k] = h
            fwd = warp_points(unit, c + ek)
            bwd = warp_points(unit, c - ek)
            acc[k] += float(np.sqrt((((fwd - bwd) / (2 * h)) ** 2).mean()))
    acc /= n_warps
    w = acc / acc.mean()
    if cache and h == 1e-3 and n_warps == 64:
        with open(_mode_cache_path(), "w") as f:
            json.dump({"weights": w.tolist(), "h": h, "n_warps": n_warps,
                       "seed": _c.TRAIN_SEED + 999}, f)
    return w


def random_coeffs(rng, scale: float = config.WARP_SCALE) -> np.ndarray:
    return np.array([(rng() * 2 - 1) * scale for _ in range(config.WARP_K)])


def _pairs(rng, n: int, unit: np.ndarray):
    # Batched single call: the per-pair Python loop pays numpy overhead
    # 1756 times (minutes); one (n, 64, 3) flow call takes seconds.
    coeffs = np.stack([random_coeffs(rng) for _ in range(n)])
    pts = np.tile(unit[None, :, :], (n, 1, 1))
    w = warp_points(pts, coeffs)
    return w.reshape(n, -1), coeffs / config.COEFF_SCALE


def build_dataset(n_train: int = config.TRAIN_N, n_held: int = config.HELD_N):
    unit = rest_unit()
    tx, ty = _pairs(mulberry32(config.TRAIN_SEED), n_train, unit)
    hx, hy = _pairs(mulberry32(config.HELD_SEED), n_held, unit)
    mean = tx.mean(axis=0)
    var = ((tx - mean) ** 2).mean(axis=0)
    scale = np.sqrt(var)
    scale[scale == 0] = 1.0
    return {
        "train_x": (tx - mean) / scale,
        "train_y": ty,
        "held_x": (hx - mean) / scale,
        "held_y": hy,
        "train_raw": tx,
        "held_raw": hx,
        "mean": mean,
        "scale": scale,
        "rest_unit": unit,
    }
