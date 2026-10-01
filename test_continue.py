"""Continuation contract: frozen prefix stages load as-is and a new
sculptor appends on top with its own hidden sizes."""

import numpy as np

from train import cascade, dataset


def test_continuation_appends_bigger_refiner():
    ds = dataset.build_dataset(n_train=48, n_held=16)
    base = cascade.train_cascade(
        ds, n_stages=2, hidden=(16,), hidden_late=(16,), epochs=3, lr=3e-3,
        batch=16, seed=99, residual_feat=True,
    )
    out = cascade.train_cascade(
        ds, n_stages=1, hidden=(16,), epochs=2, lr=3e-3,
        batch=16, seed=99, residual_feat=True,
        frozen=base["stages"], extra_hidden=(24, 12),
    )
    assert len(out["stages"]) == 3
    assert out["stages"][2]["sizes"] == [393, 24, 12, 9]
    assert len(out["held_rmse"]) == 3
    assert len(out["held_vertex"]) == 3
    for v in out["held_rmse"] + out["held_vertex"]:
        assert np.isfinite(v)
