"""Cascade training entry point. Trains greedily stage by stage,
prints the eval report, and writes the exported weights artifact — but
only when every stage beats its predecessor on held-out restored
vertices (the showcase metric).

Usage: ./.venv/bin/python -m train.run [--epochs N] [--stages S]
"""

import argparse
import os
import sys

from train import cascade, config, dataset, eval as ev, export as ex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--stages", type=int, default=config.N_STAGES)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--train-pairs", type=int, default=config.TRAIN_N)
    ap.add_argument("--held-pairs", type=int, default=config.HELD_N)
    ap.add_argument("--from-ts", default=None,
                    help="continue from an exported artifact: prefix stages load frozen")
    ap.add_argument("--extra-hidden", default=None,
                    help="hidden sizes for appended stages, e.g. 128,64")
    ap.add_argument("--distill", action="store_true",
                    help="appended stages learn teacher GN steps, not residuals")
    ap.add_argument("--act", default="tanh")
    ap.add_argument("--late-act", default="tanh")
    ap.add_argument("--grad-feat", action="store_true",
                    help="sculptors also read gradient+curvature columns")
    ap.add_argument(
        "--out", default=os.path.join(config.ROOT, "dist", "uncrumple-weights.json")
    )
    args = ap.parse_args()

    print("dataset: building seeded pairs ...", flush=True)
    ds = dataset.build_dataset(n_train=args.train_pairs, n_held=args.held_pairs)
    frozen = None
    extra_hidden = None
    loaded_gnorm = None
    if args.extra_hidden:
        extra_hidden = tuple(int(v) for v in args.extra_hidden.split(","))
    if args.from_ts:
        import numpy as np

        pre = ex.load_json_artifact(args.from_ts)
        np.testing.assert_allclose(pre["mean"], ds["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(pre["scale"], ds["scale"], rtol=0, atol=1e-12)
        frozen = pre["stages"]
        loaded_gnorm = pre.get("gnorm")
        print(f"prefix: {len(frozen)} frozen stages from {args.from_ts}", flush=True)
    print(
        f"dataset: train {ds['train_x'].shape} held {ds['held_x'].shape}",
        flush=True,
    )
    out = cascade.train_cascade(
        ds,
        n_stages=args.stages,
        epochs=args.epochs,
        lr=args.lr,
        batch=args.batch,
        frozen=frozen,
        extra_hidden=extra_hidden,
        distill=args.distill,
        act=args.act,
        late_act=args.late_act,
        grad_feat=args.grad_feat,
        grad_gnorm=loaded_gnorm,
        log=lambda m: print(f"cascade: {m}", flush=True),
    )
    rep = ev.report(ds, out["stages"], out["held_rmse"], out["residual_feat"], out["gnorm"])
    for s, r in enumerate(out["held_rmse"]):
        print(f"eval: stage {s} held coeff rmse {r:.5f} held vertex {out['held_vertex'][s]:.5f}")
    print(f"eval: mystery per-stage {['%.5f' % v for v in rep['mystery_coeff_rmse_per_stage']]}")
    vert = ev.vertex_report(ds, out["stages"], out["residual_feat"], out["gnorm"])
    print(f"eval: mystery vertex per-stage {['%.5f' % v for v in vert['mystery_vertex_rmse_per_stage']]}")

    for s in range(1, len(out["held_vertex"])):
        if out["held_vertex"][s] >= out["held_vertex"][s - 1]:
            print(
                f"FAIL: stage {s} vertex ({out['held_vertex'][s]:.5f}) did not beat "
                f"stage {s - 1} ({out['held_vertex'][s - 1]:.5f}); no artifact written"
            )
            return 1
    art = ex.build_artifact(
        stages=out["stages"],
        mean=ds["mean"],
        scale=ds["scale"],
        stage_rmse=out["held_rmse"],
        stage_vertex=out["held_vertex"],
        residual_feat=out["residual_feat"],
        gnorm=out["gnorm"],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(ex.render_json(art))
    print(f"artifact: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
