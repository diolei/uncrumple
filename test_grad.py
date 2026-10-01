"""Descent-era contract: gradient+curvature features from the offline
Jacobian, per-net column layouts including the 411-wide era, and gnorm
travel through the artifact."""

import numpy as np
import torch

from train import cascade, export as ex


def test_grad_feats_match_manual():
    torch.manual_seed(0)
    J = torch.randn(5, 20, 9)
    r = torch.randn(5, 20)
    g, d = cascade.grad_feats(J, r)
    assert g.shape == (5, 9)
    assert d.shape == (5, 9)
    np.testing.assert_allclose(
        g.numpy(), (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1).numpy(),
        rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        d.numpy(), ((J**2).sum(dim=1)).numpy(), rtol=0, atol=1e-6)


def test_late_cols_descent_era():
    x = torch.randn(4, 12)
    pred = torch.randn(4, 9)
    feat = torch.randn(4, 12)
    grad = torch.randn(4, 18)
    cols = cascade.late_cols(2 * 12 + 9 + 18, x, pred, feat, x, grad)
    assert [c.shape[1] for c in cols] == [12, 9, 12, 18]
    cols = cascade.late_cols(2 * 12 + 9, x, pred, feat, x)
    assert [c.shape[1] for c in cols] == [12, 9, 12]


def test_gnorm_roundtrip():
    from train import export as ex2

    art = ex2.build_artifact(
        stages=[], mean=[0.0], scale=[1.0], stage_rmse=[], stage_vertex=[],
        residual_feat=True,
        gnorm={"gmean": [0.1] * 9, "gscale": [2.0] * 9,
               "dmean": [0.2] * 9, "dscale": [3.0] * 9},
        provenance={"repo": "t", "commit": "t", "seed": 1},
    )
    import json as _json

    doc = _json.loads(ex2.render_json(art))
    assert "gnorm" in doc
    import tempfile, os
    p = os.path.join(tempfile.mkdtemp(), "w.json")
    open(p, "w").write(ex2.render_json(art))
    back = ex2.load_json_artifact(p)
    assert back["gnorm"]["gscale"] == [2.0] * 9


def test_tiny_grad_stage_runs():
    from train import dataset

    ds = dataset.build_dataset(n_train=48, n_held=16)
    out = cascade.train_cascade(
        ds, n_stages=2, hidden=(16,), hidden_late=(16,), epochs=3, lr=3e-3,
        batch=16, seed=99, residual_feat=True, grad_feat=True,
    )
    assert out["stages"][1]["sizes"][0] == 2 * 192 + 9 + 18
    assert out["gnorm"] is not None
    assert len(out["held_vertex"]) == 2
    import numpy as np

    for v in out["held_rmse"] + out["held_vertex"]:
        assert np.isfinite(v)


def test_descent_fn_accepts_serialized_stats():
    import numpy as np
    import torch

    from train import cascade

    gnorm_lists = {"gmean": [0.0] * 9, "gscale": [1.0] * 9,
                   "dmean": [0.0] * 9, "dscale": [1.0] * 9}
    ds = cascade.dataset.build_dataset(n_train=16, n_held=8)
    fn = cascade.make_descent_fn(ds["rest_unit"], ds["held_raw"], gnorm_lists)
    pred = torch.zeros(8, 9)
    out = fn(pred)
    assert out.shape == (8, 18)
    assert np.isfinite(out.numpy()).all()
