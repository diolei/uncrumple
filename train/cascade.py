"""Greedy cascaded residual regression (Dollar-style): stage 0 maps
dots to coefficients; each later stage reads the running prediction
plus the current sample-space mismatch and regresses the correction.
Earlier stages freeze while later ones train, so every stage boundary
is a real, independently-evaluable sculpting step.

Late stages minimize the linearized vertex mismatch ||J·delta + r||²
(Jacobians computed offline against the exact flow at 32 steps,
~1e-4 relative to full resolution) plus a small residual anchor.
Selection is on the TRUE held vertex: the linearization proposes,
the exact map disposes."""

import numpy as np
import torch

from warp import unwarp_points as np_unwarp
from warp import warp_points

from train import config, dataset
from train import torchwarp
from train.model import Mlp, snapshot

try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass


def _tensors(ds: dict):
    return (
        torch.from_numpy(ds["train_x"]).float(),
        torch.from_numpy(ds["train_y"]).float(),
        torch.from_numpy(ds["held_x"]).float(),
        torch.from_numpy(ds["held_y"]).float(),
    )


def _fit(net: Mlp, x: torch.Tensor, y: torch.Tensor, hx: torch.Tensor,
         hy: torch.Tensor, epochs: int, lr: float, batch: int, seed: int,
         loss_w: torch.Tensor, log=None) -> float:
    """AdamW with held-out early stopping: the cascade eats its own
    training residuals, so a stage that memorizes teaches the next one
    noise. Best-on-held snapshot is restored at the end; returns the
    best held loss. The loss is vertex-weighted coefficient MSE."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    def loss_fn(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (((p - t) ** 2) * loss_w).mean()

    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    best = None
    best_state = None
    with torch.no_grad():
        best = float(loss_fn(net(hx), hy).item())
        best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    for e in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss_fn(net(x[idx]), y[idx]).backward()
            opt.step()
        with torch.no_grad():
            hv = float(loss_fn(net(hx), hy).item())
        if hv < best:
            best = hv
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        if log and (e + 1) % max(1, epochs // 4) == 0:
            with torch.no_grad():
                tv = float(loss_fn(net(x), y).item())
            log(f"epoch {e + 1}/{epochs} train {tv:.5f} held {hv:.5f}")
    net.load_state_dict(best_state)
    return best


@torch.no_grad()
def held_vertex_rmse(nets: list, hx: torch.Tensor, raw_hx: np.ndarray,
                     rest: np.ndarray, feat_fn=None, grad_fn=None) -> float:
    """Held-out restored-vertex RMSE in unit-box units: the product
    metric, computed with the differentiable training warp."""
    preds = staged_predictions(nets, hx, feat_fn, grad_fn=grad_fn)[-1]
    pred_raw = preds * config.COEFF_SCALE
    obs3 = torch.from_numpy(np.asarray(raw_hx).reshape(-1, 64, 3)).float()
    rest_t = torch.from_numpy(np.asarray(rest)).float()
    back = torchwarp.unwarp_points(obs3, pred_raw)
    return float(torch.sqrt(((back - rest_t) ** 2).mean()).item())


def sample_jacobian(rest: np.ndarray, preds_raw: np.ndarray,
                    h: float = 1e-4, steps: int = 32) -> tuple:
    """Base warp plus forward-difference Jacobian dv/dc at the running
    predictions: base (B, 192), J (B, 192, 9). One batched flow call per
    mode plus one for the base (which doubles as the residual warp).
    32 integration steps match the 64-step Jacobian to ~1e-4 relative —
    plenty for a loss shaper — at half the offline cost."""
    b = preds_raw.shape[0]
    tiling = np.tile(np.asarray(rest)[None, :, :], (b, 1, 1))
    base = warp_points(tiling, preds_raw, steps=steps).reshape(b, -1)
    J = np.empty((b, base.shape[1], config.WARP_K))
    for k in range(config.WARP_K):
        ek = np.zeros(config.WARP_K)
        ek[k] = h
        w = warp_points(tiling, preds_raw + ek, steps=steps).reshape(b, -1)
        J[:, :, k] = (w - base) / h
    return base, J


def true_vertex_rmse(preds_raw: np.ndarray, raw_obs: np.ndarray,
                     rest: np.ndarray) -> float:
    """Exact restored-vertex RMSE (numpy, one batched call): unwarp the
    observations with the predicted warps, compare against rest."""
    b = preds_raw.shape[0]
    back = np_unwarp(np.asarray(raw_obs).reshape(b, -1, 3), preds_raw)
    return float(np.sqrt(((back - np.asarray(rest)) ** 2).mean()))


def gn_teacher_delta(J: np.ndarray, r: np.ndarray,
                     lam: float = 1e-8) -> np.ndarray:
    """One damped Gauss-Newton step per sample, batched: solves
    (JᵀJ + λI)·delta = -Jᵀr for the exact downhill correction in raw
    coefficient units. The teacher for distillation — exact local
    structure the residual target cannot see."""
    jtj = J.transpose(0, 2, 1) @ J
    jtj += lam * np.eye(jtj.shape[-1])[None, :, :]
    jtr = -(J.transpose(0, 2, 1) @ r[:, :, None])
    return np.linalg.solve(jtj, jtr).squeeze(-1)


def _fit_vertex(net: Mlp, fx: torch.Tensor, fhx: torch.Tensor,
                target: torch.Tensor, target_hx: torch.Tensor,
                Jtr: torch.Tensor, Jhx: torch.Tensor,
                rtr: torch.Tensor, rhx: torch.Tensor,
                prev_hx_raw: np.ndarray, raw_hx: np.ndarray,
                rest: np.ndarray, loss_w: torch.Tensor,
                epochs: int, lr: float, batch: int, seed: int,
                log=None, true_every: int = 25) -> float:
    """Sculptor stage: gradient descent on the linearized vertex
    mismatch ||J·delta + r||² (plus a small residual anchor). All warps
    happen offline in numpy; the loop is pure matmuls, so it runs in
    seconds per epoch. Selection is on the TRUE held vertex, checked
    every true_every epochs — the linearization proposes, the exact
    map disposes."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    def linloss(out, J, r):
        d = (out * config.COEFF_SCALE).unsqueeze(-1)
        return ((torch.bmm(J, d).squeeze(-1) + r) ** 2).mean()

    def anchored(out, J, r, t):
        return linloss(out, J, r) + 0.05 * ((((out - t) ** 2) * loss_w).mean())

    def true_of(out_hx: torch.Tensor) -> float:
        cur = prev_hx_raw + out_hx.detach().cpu().double().numpy() * config.COEFF_SCALE
        return true_vertex_rmse(cur, raw_hx, rest)

    gen = torch.Generator().manual_seed(seed)
    n = fx.shape[0]
    with torch.no_grad():
        best = true_of(net(fhx))
    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    for e in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            anchored(net(fx[idx]), Jtr[idx], rtr[idx], target[idx]).backward()
            opt.step()
        if (e + 1) % true_every == 0 or e == epochs - 1:
            with torch.no_grad():
                hv = true_of(net(fhx))
                lin = float(linloss(net(fhx), Jhx, rhx).item())
            if hv < best:
                best = hv
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            if log and (e + 1) % max(1, epochs // 4) == 0:
                log(f"epoch {e + 1}/{epochs} linheld {lin:.5f} vheld {hv:.5f}")
    net.load_state_dict(best_state)
    return best


def _fit_distill(net: Mlp, fx: torch.Tensor, fhx: torch.Tensor,
                 target: torch.Tensor, target_hx: torch.Tensor,
                 prev_hx_raw: np.ndarray, raw_hx: np.ndarray,
                 rest: np.ndarray, epochs: int, lr: float, batch: int,
                 seed: int, wd: float = 1e-4,
                 log=None, true_every: int = 25) -> float:
    """Distillation: plain MSE against the teacher's Gauss-Newton step
    (targets arrive well-scaled by construction — no normalization
    needed). Selection is on the TRUE held vertex, same as sculpting."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    n = fx.shape[0]

    def true_of(out_hx: torch.Tensor) -> float:
        cur = prev_hx_raw + out_hx.detach().cpu().double().numpy() * config.COEFF_SCALE
        return true_vertex_rmse(cur, raw_hx, rest)

    with torch.no_grad():
        best = true_of(net(fhx))
    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    for e in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss_fn(net(fx[idx]), target[idx]).backward()
            opt.step()
        if (e + 1) % true_every == 0 or e == epochs - 1:
            with torch.no_grad():
                hv = true_of(net(fhx))
                tv = float(loss_fn(net(fx), target).item())
                thv = float(loss_fn(net(fhx), target_hx).item())
            if hv < best:
                best = hv
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            if log and (e + 1) % max(1, epochs // 4) == 0:
                log(f"epoch {e + 1}/{epochs} teachmse {tv:.5f} teachheld {thv:.5f} vheld {hv:.5f}")
    net.load_state_dict(best_state)
    return best


@torch.no_grad()
def _held_rmse(nets: list, hx: torch.Tensor, hy: torch.Tensor,
               ds: dict | None = None, grad_fn=None) -> float:
    """Held-out RMSE in raw warp-coefficient units."""
    if ds is None:
        pred = predict_stages_raw(nets, hx)
    else:
        feat = make_feat_fn(
            np.asarray(ds["rest_unit"]),
            np.asarray(ds["held_raw"]),
            np.asarray(ds["scale"]),
        )
        pred = staged_predictions(nets, hx, feat, grad_fn=grad_fn)[-1]
    err = (pred - hy) * config.COEFF_SCALE
    return float(torch.sqrt((err**2).mean()).item())


@torch.no_grad()
def late_cols(sizes0: int, x: torch.Tensor, pred: torch.Tensor,
              feat: torch.Tensor | None, raw_x: torch.Tensor,
              grad: torch.Tensor | None = None) -> list:
    """Input columns for one sculptor net, chosen by its input width so
    mixed-vintage cascades run: xdim dots only (stage 0 never reaches
    here), xdim+9 dots+prediction (legacy), 9+xdim prediction+mismatch
    (first sculptor era), 2*xdim+9 dots+prediction+mismatch, and
    2*xdim+9+18 with gradient+curvature columns (descent era). feat is
    None exactly when the stage trains without mismatch."""
    xdim = x.shape[1]
    if feat is None:
        if sizes0 == xdim:
            return [x]
        assert sizes0 == xdim + 9, sizes0
        return [x, pred]
    if sizes0 == 9 + xdim:
        return [pred, feat]
    if sizes0 == 2 * xdim + 9:
        return [raw_x, pred, feat]
    assert sizes0 == 2 * xdim + 9 + 18, sizes0
    assert grad is not None, sizes0
    return [raw_x, pred, feat, grad]


def staged_predictions(nets: list, x: torch.Tensor, feat_fn=None,
                       raw_x: torch.Tensor | None = None,
                       grad_fn=None) -> list:
    """Running predictions in COEFF_SCALE-normalized units, one entry
    per stage. Columns are chosen per net by late_cols, so a cascade
    may mix stages trained under different feature eras. raw_x carries
    the dots (defaults to x); grad_fn maps the running prediction to
    normalized gradient+curvature columns for descent-era stages."""
    if raw_x is None:
        raw_x = x
    preds = [nets[0](x)]
    for net in nets[1:]:
        feat = feat_fn(preds[-1]).to(x.device, x.dtype) if feat_fn is not None else None
        grad = grad_fn(preds[-1]).to(x.device, x.dtype) if grad_fn is not None else None
        cols = late_cols(net.sizes[0], x, preds[-1], feat, raw_x, grad)
        preds.append(preds[-1] + net(torch.cat(cols, dim=1)))
    return preds


@torch.no_grad()
def predict_stages_raw(nets: list, x: torch.Tensor) -> torch.Tensor:
    return staged_predictions(nets, x)[-1]


def residual_feats(rest: np.ndarray, raw_obs: np.ndarray,
                   preds_raw: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    """Sample-space mismatch of the running prediction, per-dim scaled
    like the dots: warp rest by each predicted warp, subtract what was
    observed. One batched flow call — no Python loop over pairs."""
    b = preds_raw.shape[0]
    w = warp_points(np.tile(rest[None, :, :], (b, 1, 1)), preds_raw)
    return torch.from_numpy(((w.reshape(b, -1) - raw_obs) / scale).astype(np.float32))


def grad_feats(J: torch.Tensor, r: torch.Tensor) -> tuple:
    """Analytic gradient and curvature per sample from the offline
    Jacobian: G = JTr (B,9) points downhill, D = diag(JTJ) (B,9) scales
    it. Pure matmuls — the warps already happened. The net learns a
    correction to descent instead of the pseudoinverse itself."""
    g = (J.transpose(1, 2) @ r.unsqueeze(-1)).squeeze(-1)
    d = (J**2).sum(dim=1)
    return g, d


def norm_stats(t: torch.Tensor) -> tuple:
    """Per-dim mean/scale for standardizing gradient features."""
    mean = t.double().mean(dim=0)
    scale = t.double().std(dim=0)
    scale[scale == 0] = 1.0
    return mean.float(), scale.float()


def _gnorm_tensors(gnorm: dict) -> dict:
    out = {}
    for k, v in gnorm.items():
        out[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
    return out


def make_descent_fn(rest: np.ndarray, raw_obs: np.ndarray,
                    gnorm: dict):
    """Eval-side gradient+curvature columns for descent-era stages:
    Jacobian at the running prediction (32 steps, matching training),
    standardized with the frozen training stats. One batched flow call
    per mode per invocation — trivial on eval sets, minutes on full
    train sets (only future prefix scorings pay that)."""
    gnorm = _gnorm_tensors(gnorm)
    def grad_fn(pred_normed: torch.Tensor) -> torch.Tensor:
        preds_raw = (pred_normed.detach().cpu().double().numpy()) * config.COEFF_SCALE
        base, J = sample_jacobian(np.asarray(rest), preds_raw)
        r = base - np.asarray(raw_obs)
        g, d = grad_feats(torch.from_numpy(J).float(),
                          torch.from_numpy(r).float())
        return torch.cat([
            (g - gnorm["gmean"]) / gnorm["gscale"],
            (d - gnorm["dmean"]) / gnorm["dscale"],
        ], dim=1).float()
    return grad_fn


def make_feat_fn(rest: np.ndarray, raw_obs: np.ndarray,
                 scale: np.ndarray):
    def feat_fn(pred_normed: torch.Tensor) -> torch.Tensor:
        preds_raw = (pred_normed.detach().cpu().double().numpy()) * config.COEFF_SCALE
        return residual_feats(rest, raw_obs, preds_raw, scale)

    return feat_fn


def train_cascade(ds: dict, n_stages: int = config.N_STAGES,
                  hidden: tuple = config.HIDDEN,
                  hidden_late: tuple = config.HIDDEN_LATE,
                  epochs: int = 200, lr: float = 3e-3, batch: int = 256,
                  seed: int = config.TRAIN_SEED,
                  residual_feat: bool = config.RESIDUAL_FEAT,
                  frozen: list | None = None,
                  extra_hidden: tuple | None = None,
                  distill: bool = False,
                  act: str = "tanh",
                  late_act: str = "tanh",
                  grad_feat: bool = False,
                  grad_gnorm: dict | None = None,
                  log=None) -> dict:
    """Greedy cascade. frozen: snapshots trained elsewhere, loaded as-is
    and scored (continuation training appends new sculptor stages on
    top, e.g. a bigger refiner). n_stages counts NEW stages only."""
    from train.model import nets_from_snapshots

    tx, ty, hx, hy = _tensors(ds)
    in0 = tx.shape[1]
    rest = np.asarray(ds["rest_unit"])
    scale = np.asarray(ds["scale"])
    raw_tr = np.asarray(ds["train_raw"])
    raw_hx = np.asarray(ds["held_raw"])
    loss_w = torch.from_numpy(dataset.mode_weights()).float()
    gnorm: dict | None = grad_gnorm
    nets: list = nets_from_snapshots(frozen) if frozen else []
    rmses: list = []
    vrmses: list = []
    if nets:
        obs_hx3 = torch.from_numpy(raw_hx.reshape(-1, 64, 3)).float()
        rest_t = torch.from_numpy(rest).float()
        feat_tr0 = make_feat_fn(rest, raw_tr, scale) if residual_feat else None
        feat_hx0 = make_feat_fn(rest, raw_hx, scale) if residual_feat else None
        need_grad0 = any(n.sizes[0] == 2 * in0 + 9 + 18 for n in nets)
        if need_grad0:
            assert gnorm is not None, "gradient prefix needs loaded gnorm"
        grad_tr0 = make_descent_fn(rest, raw_tr, gnorm) if need_grad0 else None
        grad_hx0 = make_descent_fn(rest, raw_hx, gnorm) if need_grad0 else None
        with torch.no_grad():
            run_tr = staged_predictions(nets, tx, feat_tr0, grad_fn=grad_tr0)
            run_hx = staged_predictions(nets, hx, feat_hx0, grad_fn=grad_hx0)
            for ptr, phx in zip(run_tr, run_hx):
                err = (ptr - ty) * config.COEFF_SCALE
                rmses.append(float(torch.sqrt((err**2).mean()).item()))
                back = torchwarp.unwarp_points(obs_hx3, phx * config.COEFF_SCALE)
                vrmses.append(float(torch.sqrt(((back - rest_t) ** 2).mean()).item()))
        if log:
            log(f"prefix: {len(nets)} frozen stages, held coeff {rmses[-1]:.5f} vertex {vrmses[-1]:.5f}")
    start = len(nets)
    late_hidden = extra_hidden or hidden_late
    for s in range(start, start + n_stages):
        if not nets:
            sizes = [in0, *hidden, config.WARP_K]
            net = Mlp(sizes, seed + 1000 * s, act)
            _fit(net, tx, ty, hx, hy, epochs, lr, batch, seed + 1000 * s,
                 loss_w, log)
        else:
            feat_tr = make_feat_fn(rest, raw_tr, scale) if residual_feat else None
            feat_hx = make_feat_fn(rest, raw_hx, scale) if residual_feat else None
            need_grad = any(n.sizes[0] == 2 * in0 + 9 + 18 for n in nets)
            if need_grad:
                assert gnorm is not None, "gradient prefix needs loaded gnorm"
            grad_tr = make_descent_fn(rest, raw_tr, gnorm) if need_grad else None
            grad_hx = make_descent_fn(rest, raw_hx, gnorm) if need_grad else None
            with torch.no_grad():
                prev_tr = staged_predictions(nets, tx, feat_tr, grad_fn=grad_tr)[-1]
                prev_hx = staged_predictions(nets, hx, feat_hx, grad_fn=grad_hx)[-1]
                target = ty - prev_tr
                target_hx = hy - prev_hx
            prev_tr_raw = prev_tr.detach().cpu().double().numpy() * config.COEFF_SCALE
            prev_hx_raw = prev_hx.detach().cpu().double().numpy() * config.COEFF_SCALE
            # Offline warps: base residuals + Jacobians at the running
            # predictions. Minutes per stage, once — everything after
            # stays pure matmuls, including the gradient features below.
            if log:
                log(f"stage {s}: offline Jacobians ...")
            base_tr, J_tr = sample_jacobian(rest, prev_tr_raw)
            if log:
                log(f"stage {s}: train Jacobians done")
            base_hx, J_hx = sample_jacobian(rest, prev_hx_raw)
            if log:
                log(f"stage {s}: held Jacobians done")
            Jtr_t = torch.from_numpy(J_tr).float()
            Jhx_t = torch.from_numpy(J_hx).float()
            rtr_t = torch.from_numpy(base_tr - raw_tr).float()
            rhx_t = torch.from_numpy(base_hx - raw_hx).float()
            with torch.no_grad():
                cols_tr = [tx, prev_tr]
                cols_hx = [hx, prev_hx]
                if residual_feat:
                    assert feat_tr is not None and feat_hx is not None
                    cols_tr.append(feat_tr(prev_tr))
                    cols_hx.append(feat_hx(prev_hx))
                if grad_feat:
                    g_tr, d_tr = grad_feats(Jtr_t, rtr_t)
                    g_hx, d_hx = grad_feats(Jhx_t, rhx_t)
                    if gnorm is None:
                        gm, gs = norm_stats(g_tr)
                        dm, ds_ = norm_stats(d_tr)
                        gnorm = {"gmean": gm, "gscale": gs,
                                 "dmean": dm, "dscale": ds_}
                        if log:
                            log(f"stage {s}: gradient stats frozen")
                    cols_tr.append(torch.cat([
                        (g_tr - gnorm["gmean"]) / gnorm["gscale"],
                        (d_tr - gnorm["dmean"]) / gnorm["dscale"],
                    ], dim=1))
                    cols_hx.append(torch.cat([
                        (g_hx - gnorm["gmean"]) / gnorm["gscale"],
                        (d_hx - gnorm["dmean"]) / gnorm["dscale"],
                    ], dim=1))
                fx = torch.cat(cols_tr, dim=1)
                fhx = torch.cat(cols_hx, dim=1)
            late_in = fx.shape[1]
            sizes = [late_in, *late_hidden, config.WARP_K]
            net = Mlp(sizes, seed + 1000 * s, late_act)
            # Distillation versus sculpting. Distillation learns one
            # damped Gauss-Newton step from the running prediction —
            # the exact downhill correction from the true local
            # Jacobians above. Sculpting descends the linearized vertex
            # mismatch with a residual anchor. Either way the snapshots
            # stay pure forward passes.
            if distill:
                teach_tr = gn_teacher_delta(J_tr, base_tr - raw_tr)
                teach_hx = gn_teacher_delta(J_hx, base_hx - raw_hx)
                _fit_distill(
                    net,
                    fx,
                    fhx,
                    torch.from_numpy(teach_tr / config.COEFF_SCALE).float(),
                    torch.from_numpy(teach_hx / config.COEFF_SCALE).float(),
                    prev_hx_raw,
                    raw_hx,
                    rest,
                    epochs,
                    lr,
                    batch,
                    seed + 1000 * s,
                    log=log,
                )
            else:
                _fit_vertex(
                    net,
                    fx,
                    fhx,
                    target,
                    target_hx,
                    torch.from_numpy(J_tr).float(),
                    torch.from_numpy(J_hx).float(),
                    torch.from_numpy(base_tr - raw_tr).float(),
                    torch.from_numpy(base_hx - raw_hx).float(),
                    prev_hx_raw,
                    raw_hx,
                    rest,
                    loss_w,
                    epochs,
                    lr,
                    batch,
                    seed + 1000 * s,
                    log,
                )
        nets.append(net)
        need_all = grad_feat and gnorm is not None and any(
            n.sizes[0] == 2 * in0 + 9 + 18 for n in nets)
        grad_all = (make_descent_fn(rest, raw_hx, gnorm)
                    if need_all else None)
        r = _held_rmse(nets, hx, hy, ds if residual_feat else None,
                       grad_fn=grad_all)
        rmses.append(r)
        feat_all = make_feat_fn(rest, raw_hx, scale) if residual_feat else None
        vr = held_vertex_rmse(nets, hx, raw_hx, rest, feat_all, grad_all)
        vrmses.append(vr)
        if log:
            log(f"stage {s}: held coeff rmse {r:.5f} held vertex {vr:.5f}")
    def _ser(t):
        return t.detach().cpu().double().numpy().tolist()
    return {
        "stages": [snapshot(n) for n in nets],
        "held_rmse": rmses,
        "held_vertex": vrmses,
        "residual_feat": residual_feat,
        "grad_feat": grad_feat,
        "gnorm": (None if gnorm is None else {
            "gmean": _ser(gnorm["gmean"]), "gscale": _ser(gnorm["gscale"]),
            "dmean": _ser(gnorm["dmean"]), "dscale": _ser(gnorm["dscale"]),
        }),
    }


def predict_stages(stages: list, x_np):
    """Numpy entry point used by tests: rebuild nets from snapshots."""
    import numpy as np

    nets = []
    for st in stages:
        net = Mlp(st["sizes"], 0, st.get("act", "tanh"))
        sd = {}
        for i in range(len(st["sizes"]) - 1):
            sd[f"layers.{i}.weight"] = torch.tensor(st["weights"][i])
            sd[f"layers.{i}.bias"] = torch.tensor(st["biases"][i])
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    with torch.no_grad():
        out = predict_stages_raw(nets, torch.from_numpy(np.asarray(x_np)).float())
    return out.numpy()
