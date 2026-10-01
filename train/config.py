"""Frozen contract shared by the dataset, the cascade, and the export.
Every value is frozen: the dataset draws, the stage shapes, and the
artifact must all agree on it. The mystery coefficients are a fixed
seed-7 draw, never redrawn."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MESH_ID = "uncrumple-beam/1"
MESH_PATH = os.path.join(ROOT, "mesh", "uncrumple-beam-1.json")

WARP_K = 9
SAMPLE_COUNT = 64

# Dataset draws: same seeds, scales, and RNG family as the reference
# foundry (mulberry32 + uniform coeff draws).
TRAIN_SEED = 20261101
HELD_SEED = 20261102
TRAIN_N = 1500
HELD_N = 256
WARP_SCALE = 0.30
COEFF_SCALE = 0.5

# Cascade shape: stage 0 reads the 192 normalized dots; sculptor
# stages read dots + running prediction + sample-space mismatch
# ([X, P, R], 393 dims). The dots ground the correction independently
# of the running estimate; the mismatch carries the error signal.
N_STAGES = 5
HIDDEN = (128, 64)
HIDDEN_LATE = (64, 32)
RESIDUAL_FEAT = True

# Frozen mystery warp: same seed-7 stream as the retired bracket draw,
# kept because its strong leading modes bow the beam laterally on the
# buckle table (max |c| 0.29).
MYSTERY = [
    -0.2929771481081843,
    -0.26282504545524715,
    0.286144579667598,
    0.1194172234274447,
    0.01286716111935675,
    -0.05668698716908693,
    -0.02026042048819363,
    -0.1560448884498328,
    0.03199536236934364,
]
