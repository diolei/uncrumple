"""Artifact: the single file that travels downstream. One staged
snapshot per cascade level plus every constant the inference runtime
needs — mesh id, sizes, normalization, frozen mystery, per-stage
errors, and provenance pointing back at the exact training commit.

Stage input layout: stage 0 reads the 192 normalized dots;
residual-feat stages read [dots (192), running prediction (9,
normalized coeff units), sample-space mismatch (192, dot-scaled)].
The inference runtime rebuilds the same columns — sizes[0] tells it
which layout.
The snapshot shape is declared inline so the artifact depends on
nothing outside itself."""

import json
import subprocess

import torch

from train import config


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_json_artifact(path: str) -> dict:
    """Read back an exported JSON artifact (continuation training):
    validates mesh id and stage shapes, returns flat snapshots plus
    normalization and optional gnorm."""
    art = json.load(open(path))
    assert art["mesh_id"] == config.MESH_ID, art["mesh_id"]
    for st in art["stages"]:
        sizes = st["sizes"]
        assert len(st["weights"]) == len(sizes) - 1
        assert len(st["biases"]) == len(sizes) - 1
    import numpy as np

    return {
        "stages": art["stages"],
        "mean": np.array(art["mean"]),
        "scale": np.array(art["scale"]),
        "gnorm": art.get("gnorm"),
    }

def build_artifact(stages: list[dict], mean, scale, stage_rmse: list,
                   stage_vertex: list, residual_feat: bool,
                   gnorm: dict | None = None,
                   provenance: dict | None = None) -> dict:
    import numpy as np

    art = {
        "mesh_id": config.MESH_ID,
        "seed": config.TRAIN_SEED,
        "stages": stages,
        "stage_vertex": [float(v) for v in stage_vertex],
        "residual_feat": residual_feat,
        "gnorm": gnorm,
        "mean": [float(v) for v in np.asarray(mean).tolist()],
        "scale": [float(v) for v in np.asarray(scale).tolist()],
        "mystery": list(config.MYSTERY),
        "stage_rmse": [float(v) for v in stage_rmse],
        "provenance": provenance
        or {
            "repo": "uncrease",
            "commit": _git_commit(),
            "seed": config.TRAIN_SEED,
            "torch": torch.__version__,
        },
    }
    json.dumps(art)  # plain floats only, no tensors
    return art


def render_json(art: dict) -> str:
    """Compact JSON artifact: the single file that travels downstream.
    Stages carry sizes/act/weights/biases; top level carries mesh id,
    normalization, frozen mystery, per-stage errors, and provenance."""
    return json.dumps(art)

