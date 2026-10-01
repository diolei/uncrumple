# uncrease — learning to invert smooth 3D warps

A stationary sine-mode velocity field integrated over unit time is a
diffeomorphism: dramatic bends and twists with no fold ever forming,
invertible exactly by backward integration. That makes it the perfect
playground for learned inversion — the forward map is exact, the
inverse exists, and small MLPs can learn it from point samples.

This repo is the canonical trainer for the warp-inversion showcase.
Training happens here, in PyTorch, and only the exported weights file
travels downstream — hand-carried as a single committed file. All
training lives here; inference lives elsewhere.

## Honesty boundary

Only smooth, fold-free warps are learnable. True self-contact crumple
destroys information irreversibly: many rest shapes map to
indistinguishable crumples, so no net recovers "the" original, and this
repo does not claim to. Folded probes are exhibited failing, labeled
out-of-scope.

## Method: cascaded residual regression

One net cannot sculpt precisely enough — a single forward pass lands
near the answer but stalls wide. So the trainer is greedy and staged:
stage 0 maps the 192 normalized dots to the nine warp coefficients;
each sculptor stage reads the running prediction plus its current
sample-space mismatch and regresses the correction. Late stages
minimize the linearized restored-vertex error (Jacobians computed
offline against the exact flow), with each stage frozen before the
next trains — so every stage boundary is a real,
independently-evaluable sculpting step, and the staged beats replay
exactly those outputs. Selection is on the true held-out restored vertex; the
run is accepted only when every stage beats its predecessor, otherwise
no artifact is written.

## Layout

- `warp.py` — the buckle-warp core: bow-dominated sine-mode fields,
  RK4 integration. Numpy mirror of the uncrumple-buckle/1 reference
  warp, verified against the shared fixture (forward agreement is
  bit-identical).
- `mirror.py` — regenerates `fixtures/mirror.json`, the shared fixture
  that downstream suites cross-check.
- `mesh/` — gallery meshes: `uncrumple-beam-1.json` is the frozen
  training mesh (ID-checked against the weights on arrival).
- `train/` — the cascade: `config.py` (frozen seeds/scales/mystery),
  `rng.py` (exact mulberry32 port), `dataset.py` (seeded pairs +
  sensitivity weights), `model.py` (tanh MLP core), `torchwarp.py`
  (torch measurement warp), `cascade.py` (greedy staging +
  vertex sculptors), `eval.py` (held-out + frozen-mystery report),
  `export.py` (weights artifact), `run.py` (entry point).
- `dist/` — the exported `uncrumple-weights.json`, written only by
  `train/run.py`. This is the one file that moves downstream.
- `fixtures/` — the shared mirror fixture plus the cached
  sensitivity weights.

## Reproduce

```sh
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python mirror.py
.venv/bin/python -m train.run
```

Same seed → same dataset, bit for bit (mulberry32 port verified
against node). Training itself is seeded AdamW with held-out early
stopping; the committed artifact is what downstream gates, not the
training run.

## Frozen contract

- Mesh `uncrumple-beam/1`, mystery warp (seed-7 draw, copied verbatim
  into `train/config.py`), 64 stride-selected samples, 1500/256
  train/held splits, warp scale 0.30, coefficient scale 0.5.
- Change any of these and the mesh-ID gate fails on arrival —
  that is the point.

The venv is gitignored and never committed.
