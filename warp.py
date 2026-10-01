"""Reference buckle warp core: sine-mode fields, RK4 over unit time,
float64. Numpy and reference implementations of the uncrumple-buckle/1
family must agree to ~1e-9 on the shared fixture (see test_mirror.py);
the BASIS table below is the frozen definition. Five of nine modes vary
along the beam length, so the dominant shapes read as lateral bow, sag,
and second-harmonic ripple — the elastic buckling family, slightly
exaggerated — while staying in the reversible spring-back range."""

import numpy as np

WARP_K = 9
FLOW_STEPS = 64
TAU = 2.0 * np.pi

# (axis, along, n, phase). Order frozen: warpFamily "uncrumple-buckle/1".
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


def velocity(pts: np.ndarray, c: np.ndarray) -> np.ndarray:
    """pts (..., 3) in unit box, c (9,) or (..., 9) → velocity (..., 3)."""
    out = np.zeros_like(pts, dtype=np.float64)
    for k, (axis, along, n, phase) in enumerate(BASIS):
        ck = c[k] if c.ndim == 1 else c[..., k : k + 1]
        out[..., axis] = out[..., axis] + ck * np.sin(
            TAU * n * pts[..., along] + phase
        )
    return out


def warp_points(pts: np.ndarray, c: np.ndarray, steps: int = FLOW_STEPS) -> np.ndarray:
    """Forward flow by midpoint-free RK4 over unit time. Vectorized over
    leading dims: pts (N, 3) with c (9,) or (B, N, 3) with c (B, 9)."""
    dt = 1.0 / steps
    cur = np.array(pts, dtype=np.float64)
    for _ in range(steps):
        k1 = velocity(cur, c)
        k2 = velocity(cur + 0.5 * dt * k1, c)
        k3 = velocity(cur + 0.5 * dt * k2, c)
        k4 = velocity(cur + dt * k3, c)
        cur = cur + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return cur


def unwarp_points(pts: np.ndarray, c: np.ndarray, steps: int = FLOW_STEPS) -> np.ndarray:
    return warp_points(pts, -np.asarray(c, dtype=np.float64), steps)


def random_coeffs(rng: np.random.Generator, n: int, scale: float = 0.45) -> np.ndarray:
    return (rng.random((n, WARP_K)) * 2.0 - 1.0) * scale
