"""Distillation contract: the teacher step solves the damped normal
equations exactly, and an appended stage learns it as a pure function."""

import numpy as np

from train import cascade, dataset


def test_teacher_step_descends_vertex():
    ds = dataset.build_dataset(n_train=32, n_held=8)
    unit = ds["rest_unit"]
    rng = np.random.default_rng(3)
    c0 = (rng.random((8, 9)) * 2 - 1) * 0.3
    from warp import warp_points

    obs = warp_points(np.tile(unit[None], (8, 1, 1)), c0)
    # Start away from truth (the actual distillation regime) with
    # Jacobians at full resolution so the residual is exact.
    pred = c0 + 0.05
    base, J = cascade.sample_jacobian(unit, pred, steps=64)
    r = base - obs.reshape(8, -1)
    d = cascade.gn_teacher_delta(J, r)
    assert d.shape == (8, 9)
    assert np.isfinite(d).all()
    from warp import unwarp_points

    v0 = float(np.sqrt(((unwarp_points(obs, pred) - unit) ** 2).mean()))
    v1 = float(np.sqrt(((unwarp_points(obs, pred + d) - unit) ** 2).mean()))
    assert v1 < v0


def test_tiny_distill_stage_runs():
    ds = dataset.build_dataset(n_train=48, n_held=16)
    base = cascade.train_cascade(
        ds, n_stages=2, hidden=(16,), hidden_late=(16,), epochs=3, lr=3e-3,
        batch=16, seed=99, residual_feat=True,
    )
    out = cascade.train_cascade(
        ds, n_stages=1, hidden=(16,), epochs=2, lr=3e-3,
        batch=16, seed=99, residual_feat=True,
        frozen=base["stages"], extra_hidden=(24, 12), distill=True,
    )
    assert len(out["stages"]) == 3
    assert out["stages"][2]["sizes"] == [393, 24, 12, 9]
    for v in out["held_rmse"] + out["held_vertex"]:
        assert np.isfinite(v)
