"""Dataset contract: seeded pairs on the warp mirror, stride-selected
samples exactly like the gallery sampler, train-split normalization."""

import math

import numpy as np

from train import dataset
from train.config import COEFF_SCALE, HELD_SEED, SAMPLE_COUNT, TRAIN_SEED, WARP_SCALE


def test_same_seed_same_tensors():
    a = dataset.build_dataset(n_train=64, n_held=16)
    b = dataset.build_dataset(n_train=64, n_held=16)
    for k in ("train_x", "train_y", "held_x", "held_y", "mean", "scale"):
        np.testing.assert_array_equal(a[k], b[k])


def test_shapes_and_scales():
    ds = dataset.build_dataset(n_train=64, n_held=16)
    assert ds["train_x"].shape == (64, SAMPLE_COUNT * 3)
    assert ds["train_y"].shape == (64, 9)
    assert ds["held_x"].shape == (16, SAMPLE_COUNT * 3)
    assert ds["held_y"].shape == (16, 9)
    # Targets are COEFF_SCALE-normalized coefficients: |y| <= WARP_SCALE/COEFF_SCALE.
    assert float(np.abs(ds["train_y"]).max()) <= WARP_SCALE / COEFF_SCALE + 1e-9
    assert bool((ds["scale"] > 0).all())


def test_train_split_normalized_to_zero_mean():
    ds = dataset.build_dataset(n_train=128, n_held=16)
    assert float(np.abs(ds["train_x"].mean(axis=0)).max()) < 1e-9


def test_stride_selection_mirrors_mesh_sample_points():
    verts = dataset.mesh_vert_count()
    step = verts / SAMPLE_COUNT
    want = [math.floor(i * step) % verts for i in range(SAMPLE_COUNT)]
    assert dataset.sample_indices() == want
    assert want[0] == 0
    assert len(set(want)) == SAMPLE_COUNT


def test_seed_split_separation():
    assert TRAIN_SEED != HELD_SEED
    ds = dataset.build_dataset(n_train=32, n_held=32)
    assert not np.array_equal(ds["train_x"], ds["held_x"])


def test_mode_weights_positive_unit_mean_deterministic():
    w = dataset.mode_weights(h=1e-3, n_warps=4, cache=False)
    assert w.shape == (9,)
    assert bool((w > 0).all())
    assert abs(float(w.mean()) - 1.0) < 1e-12
    np.testing.assert_array_equal(
        w, dataset.mode_weights(h=1e-3, n_warps=4, cache=False))


def test_mode_weights_cache_roundtrip():
    w = dataset.mode_weights()
    assert w.shape == (9,)
    assert abs(float(w.mean()) - 1.0) < 1e-9
    import json
    cached = json.load(open(dataset._mode_cache_path()))
    assert cached["seed"] == TRAIN_SEED + 999
    np.testing.assert_array_equal(w, np.array(cached["weights"]))
