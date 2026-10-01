"""Cascade contract: greedy residual stages, each one's held-out error
recorded; tiny smoke runs check shapes, finiteness, and stage count."""

import numpy as np

from train import cascade, dataset


def test_tiny_cascade_runs_two_stages():
    ds = dataset.build_dataset(n_train=48, n_held=16)
    out = cascade.train_cascade(
        ds, n_stages=2, hidden=(16,), hidden_late=(16,), epochs=4, lr=3e-3,
        batch=16, seed=99, residual_feat=False,
    )
    assert out["residual_feat"] is False
    assert len(out["stages"]) == 2
    assert len(out["held_rmse"]) == 2
    for v in out["held_rmse"]:
        assert np.isfinite(v)
    pred = cascade.predict_stages(out["stages"], ds["held_x"])
    assert pred.shape == (16, 9)
    assert np.isfinite(pred).all()


def test_tiny_cascade_with_residual_features():
    ds = dataset.build_dataset(n_train=32, n_held=16)
    out = cascade.train_cascade(
        ds, n_stages=2, hidden=(16,), hidden_late=(16,), epochs=3, lr=3e-3,
        batch=16, seed=99, residual_feat=True,
    )
    assert out["residual_feat"] is True
    assert out["stages"][1]["sizes"][0] == 192 + 9 + 192
    assert len(out["held_rmse"]) == 2
    for v in out["held_rmse"]:
        assert np.isfinite(v)
