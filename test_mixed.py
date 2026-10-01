"""Mixed-vintage contract: staged_predictions chooses input columns
per net by input width, so a cascade may mix sculptor eras."""

import numpy as np
import torch

from train import cascade
from train.model import Mlp


def _net(sizes, seed):
    net = Mlp(sizes, seed)
    net.eval()
    return net


def test_mixed_vintage_cols():
    torch.manual_seed(0)
    x = torch.randn(4, 12)
    nets = [_net([12, 8, 9], 1), _net([21, 8, 9], 2), _net([33, 8, 9], 3)]

    def feat_fn(pred):
        assert pred.shape == (4, 9)
        return torch.randn(4, 12)

    with torch.no_grad():
        preds = cascade.staged_predictions(nets, x, feat_fn)
    assert len(preds) == 3
    for p in preds:
        assert p.shape == (4, 9)
        assert np.isfinite(p.numpy()).all()


def test_unknown_width_raises():
    nets = [_net([12, 8, 9], 1), _net([99, 8, 9], 2)]
    x = torch.randn(4, 12)
    try:
        with torch.no_grad():
            cascade.staged_predictions(nets, x, lambda p: torch.randn(4, 12))
    except AssertionError:
        return
    raise SystemExit("expected AssertionError for width 99")
