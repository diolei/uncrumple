"""SIREN contract: sine hidden units evaluate differently from tanh,
snapshots carry the activation family, and rebuilds preserve it."""

import numpy as np
import torch

from train.model import Mlp, nets_from_snapshots, snapshot


def test_sine_differs_from_tanh():
    x = torch.randn(4, 12)
    a = Mlp([12, 8, 9], 5, "tanh")
    b = Mlp([12, 8, 9], 5, "sine")
    with torch.no_grad():
        pa, pb = a(x).numpy(), b(x).numpy()
    assert not np.allclose(pa, pb)
    assert np.isfinite(pb).all()


def test_snapshot_carries_act():
    net = Mlp([12, 8, 9], 5, "sine")
    snap = snapshot(net)
    assert snap["act"] == "sine"
    rebuilt = nets_from_snapshots([snap])[0]
    assert rebuilt.act == "sine"
    x = torch.randn(4, 12)
    with torch.no_grad():
        np.testing.assert_array_equal(rebuilt(x).numpy(), net(x).numpy())


def test_tanh_default_preserved():
    snap = {"sizes": [12, 8, 9],
            "weights": np.random.default_rng(0).uniform(-0.1, 0.1, (8, 12)).tolist(),
            "biases": [np.zeros(8).tolist()]}
    snap["weights"] = [snap["weights"],
                       np.random.default_rng(1).uniform(-0.1, 0.1, (9, 8)).tolist()]
    snap["biases"] = [snap["biases"][0], np.zeros(9).tolist()]
    rebuilt = nets_from_snapshots([snap])[0]
    assert rebuilt.act == "tanh"
