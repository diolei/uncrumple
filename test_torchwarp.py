"""Torch-warp mirror contract: the training measurement warp must
agree with the numpy reference (warp.py) on real batch shapes, or
every vertex-space number it reports points the wrong way."""

import numpy as np
import torch

from warp import warp_points as np_warp

from train import dataset
from train.torchwarp import unwarp_points, warp_points


def test_forward_agrees_with_numpy():
    unit = dataset.rest_unit()
    rng = np.random.default_rng(7)
    coeffs = (rng.random((8, 9)) * 2 - 1) * 0.45
    pts = np.tile(unit[None, :, :], (8, 1, 1))
    ref = np_warp(pts, coeffs)
    got = warp_points(torch.from_numpy(pts).float(),
                      torch.from_numpy(coeffs).float()).numpy()
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5)


def test_roundtrip_recovers_rest_points():
    unit = dataset.rest_unit()
    c = np.array([0.21, -0.33, 0.12, 0.4, -0.18, 0.27, -0.09, 0.31, -0.22])
    w = warp_points(torch.from_numpy(unit).float(),
                    torch.from_numpy(c).float())
    back = unwarp_points(w, torch.from_numpy(c).float()).numpy()
    np.testing.assert_allclose(back, unit, rtol=0, atol=1e-4)
