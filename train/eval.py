"""Eval: per-stage held-out errors plus the frozen-mystery probe.
The mystery warp is fixed (never trained on, never redrawn): warp the
rest samples, normalize with the train split, run the cascade, and
report coefficient error per stage. A stage that cannot beat its
predecessor on held-out restored vertices fails the run."""

import numpy as np
import torch

from warp import unwarp_points, warp_points

from train import cascade, config
from train.model import nets_from_snapshots


def _rebuild(stages: list) -> list:
    return nets_from_snapshots(stages)


def _mystery_running(ds: dict, stages: list, residual_feat: bool,
                     gnorm: dict | None = None):
    """Running predictions (raw coeff units, one per stage) on the
    frozen mystery, through the same recurrence training used."""
    nets = _rebuild(stages)
    rest = np.asarray(ds["rest_unit"])
    c_true = np.asarray(config.MYSTERY)
    obs = warp_points(rest, c_true)
    raw = obs.reshape(-1)
    scale = np.asarray(ds["scale"])
    x = torch.from_numpy(((raw - np.asarray(ds["mean"])) / scale)[None, :]).float()
    feat = cascade.make_feat_fn(rest, raw[None, :], scale) if residual_feat else None
    need_grad = any(n.sizes[0] == 2 * x.shape[1] + 9 + 18 for n in nets)
    if need_grad:
        assert gnorm is not None, "gradient stages need gnorm"
    grad = (cascade.make_descent_fn(rest, raw[None, :], gnorm)
            if need_grad else None)
    with torch.no_grad():
        preds = cascade.staged_predictions(nets, x, feat, grad_fn=grad)
    running = [p.numpy()[0] * config.COEFF_SCALE for p in preds]
    return rest, c_true, obs, running


def report(ds: dict, stages: list, held_rmse: list,
           residual_feat: bool = False, gnorm: dict | None = None) -> dict:
    rest, c_true, obs, running = _mystery_running(ds, stages, residual_feat, gnorm)
    per_stage = [float(np.sqrt(((p - c_true) ** 2).mean())) for p in running]
    return {
        "stage_held_rmse": [float(v) for v in held_rmse],
        "mystery_coeff_rmse_per_stage": per_stage,
    }


def vertex_report(ds: dict, stages: list, residual_feat: bool = False,
                    gnorm: dict | None = None) -> dict:
    """Restored-vertex RMSE per stage on the frozen mystery, in unit-box
    units: unwarp the observed dots with each stage's running prediction
    and compare against rest. This is the showcase acceptance number —
    the staged beats under test are these exact states."""
    rest, c_true, obs, running = _mystery_running(ds, stages, residual_feat, gnorm)
    per_stage = []
    for p in running:
        back = unwarp_points(obs, p)
        per_stage.append(float(np.sqrt(((back - rest) ** 2).mean())))
    return {"mystery_vertex_rmse_per_stage": per_stage}
