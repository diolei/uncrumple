"""Export contract: the artifact dict carries staged snapshots with
consistent layer dims, and the rendered TS file carries every constant
the inference runtime needs (mesh id, sizes, mean/scale, mystery,
per-stage errors, provenance) — with no imports, no outside refs."""

import json

import numpy as np

from train import config, export as ex


def _fake_stage(in_dim, seed):
    rng = np.random.default_rng(seed)
    sizes = [in_dim, 8, 9]
    weights = [
        rng.uniform(-0.1, 0.1, size=(8, in_dim)).tolist(),
        rng.uniform(-0.1, 0.1, size=(9, 8)).tolist(),
    ]
    biases = [np.zeros(8).tolist(), np.zeros(9).tolist()]
    return {"sizes": sizes, "act": "tanh", "weights": weights, "biases": biases}


def test_artifact_schema_and_dim_consistency():
    stages = [_fake_stage(12, 1), _fake_stage(21, 2)]
    art = ex.build_artifact(
        stages=stages,
        mean=[0.0] * 12,
        scale=[1.0] * 12,
        stage_rmse=[0.09, 0.05],
        stage_vertex=[0.08, 0.04],
        residual_feat=False,
        provenance={"repo": "uncrease", "commit": "test", "seed": 7},
    )
    assert art["mesh_id"] == config.MESH_ID
    assert art["residual_feat"] is False
    assert len(art["stages"]) == 2
    for st in art["stages"]:
        sizes = st["sizes"]
        assert len(st["weights"]) == len(sizes) - 1
        assert len(st["biases"]) == len(sizes) - 1
        for li, (w, b) in enumerate(zip(st["weights"], st["biases"])):
            assert len(w) == sizes[li + 1]
            assert all(len(row) == sizes[li] for row in w)
            assert len(b) == sizes[li + 1]
    json.dumps(art)  # plain floats only


def test_rendered_ts_carries_runtime_constants():
    stages = [_fake_stage(12, 1)]
    art = ex.build_artifact(
        stages=stages,
        mean=[0.0] * 12,
        scale=[1.0] * 12,
        stage_rmse=[0.09],
        stage_vertex=[0.08],
        residual_feat=True,
        provenance={"repo": "uncrease", "commit": "test", "seed": 7},
    )
    import json as _json

    text = ex.render_json(art)
    assert "@/" not in text
    doc = _json.loads(text)
    for key in (
        "stages",
        "mesh_id",
        "seed",
        "mean",
        "scale",
        "mystery",
        "stage_rmse",
        "stage_vertex",
        "residual_feat",
        "provenance",
    ):
        assert key in doc
    assert doc["mesh_id"] == config.MESH_ID
    assert doc["residual_feat"] is True


def test_load_json_artifact_roundtrip(tmp_path):
    stages = [_fake_stage(12, 1), _fake_stage(21, 2)]
    art = ex.build_artifact(
        stages=stages,
        mean=[0.5] * 12,
        scale=[2.0] * 12,
        stage_rmse=[0.09, 0.05],
        stage_vertex=[0.08, 0.04],
        residual_feat=True,
        provenance={"repo": "uncrease", "commit": "test", "seed": 7},
    )
    p = tmp_path / "w.json"
    p.write_text(ex.render_json(art))
    back = ex.load_json_artifact(str(p))
    assert back["stages"] == stages
    np.testing.assert_allclose(back["mean"], [0.5] * 12)
    np.testing.assert_allclose(back["scale"], [2.0] * 12)
