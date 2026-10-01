"""Torch mirror of the flow warp (warp.py): sine-mode velocity field,
RK4 over unit time, float32. Used ONLY for held-out measurement inside
training (never in the inference path, never in the loss loop).
Agreement target vs numpy is 1e-5 (float32 rounding over 64 RK4
steps), verified by test_torchwarp."""

import torch

TAU = 2.0 * torch.pi
FLOW_STEPS = 64

# (axis, along, n, phase). Frozen order, warpFamily "uncrumple-buckle/1".
BASIS = [
    (2, 0, 1, 0.0),
    (1, 0, 1, 0.7),
    (2, 0, 2, 1.4),
    (1, 0, 2, 2.1),
    (0, 0, 1, 2.8),
    (2, 1, 1, 3.5),
    (1, 2, 1, 4.2),
    (0, 2, 2, 4.9),
    (1, 1, 3, 5.6),
]


def velocity(pts: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(pts)
    for k, (axis, along, n, phase) in enumerate(BASIS):
        ck = c[k] if c.dim() == 1 else c[..., k : k + 1]
        out[..., axis] = out[..., axis] + ck * torch.sin(
            TAU * n * pts[..., along] + phase
        )
    return out


def warp_points(pts: torch.Tensor, c: torch.Tensor,
                steps: int = FLOW_STEPS) -> torch.Tensor:
    dt = 1.0 / steps
    cur = pts.clone()
    for _ in range(steps):
        k1 = velocity(cur, c)
        k2 = velocity(cur + 0.5 * dt * k1, c)
        k3 = velocity(cur + 0.5 * dt * k2, c)
        k4 = velocity(cur + dt * k3, c)
        cur = cur + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return cur


def unwarp_points(pts: torch.Tensor, c: torch.Tensor, steps: int = 64) -> torch.Tensor:
    return warp_points(pts, -c, steps)
