import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../app")))

from consolidated_clustering_v2.routing.psa_calibration import PSARoutingCalibration


def _calibration_payload(position_hash: str, star_positions_b64: str) -> dict:
    return {
        "axis_edges": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        "bins_per_axis": 1,
        "calibration_id": "test_calibration",
        "frame_hash": "test_frame",
        "label_prefix": "RQ1",
        "mode": "quantile",
        "native_embedding_dimension": 2048,
        "position_hash": position_hash,
        "star_positions_b64": [star_positions_b64],
        "star_positions_dtype": "float32",
        "star_positions_shape": [2, 3],
    }


def test_calibration_loads_embedded_star_positions(tmp_path):
    positions = np.arange(6, dtype=np.float32).reshape(2, 3)
    position_hash = hashlib.sha256(positions.tobytes()).hexdigest()[:12]
    payload = _calibration_payload(
        position_hash,
        base64.b64encode(positions.tobytes()).decode("ascii"),
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))

    calibration = PSARoutingCalibration.from_json_file(path)

    assert np.array_equal(calibration.get_star_positions(), positions)


def test_calibration_rejects_embedded_star_position_hash_mismatch(tmp_path):
    positions = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = _calibration_payload(
        "badbadbadbad",
        base64.b64encode(positions.tobytes()).decode("ascii"),
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))

    calibration = PSARoutingCalibration.from_json_file(path)

    with pytest.raises(ValueError, match="Bundled PSA star positions hash mismatch"):
        calibration.get_star_positions()


def test_checked_in_calibration_bundles_pinned_star_positions():
    repo_root = Path(__file__).resolve().parents[3]
    calibration = PSARoutingCalibration.from_json_file(
        repo_root / "app/consolidated_clustering_v2/config/psa_calibration_v2_115stars.json"
    )
    positions = calibration.get_star_positions()

    assert positions.shape == (115, 3)
    assert hashlib.sha256(positions.tobytes()).hexdigest()[:12] == "fe9486b3774d"


def test_calibration_requires_declared_native_embedding_dimension(tmp_path):
    positions = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = _calibration_payload(
        hashlib.sha256(positions.tobytes()).hexdigest()[:12],
        base64.b64encode(positions.tobytes()).decode("ascii"),
    )
    payload.pop("native_embedding_dimension")
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="must declare native_embedding_dimension"):
        PSARoutingCalibration.from_json_file(path)


def test_checked_in_calibration_declares_the_width_it_was_fit_in():
    repo_root = Path(__file__).resolve().parents[3]
    calibration = PSARoutingCalibration.from_json_file(
        repo_root / "app/consolidated_clustering_v2/config/psa_calibration_v2_115stars.json"
    )

    from mcp_servers.mcp_lib.semantic_routing_contract import NATIVE_EMBEDDING_DIMENSION

    assert calibration.native_embedding_dimension == NATIVE_EMBEDDING_DIMENSION == 2048
    assert calibration.frame_hash == "4855acf53eef"


@pytest.fixture
def refit_inputs():
    from consolidated_clustering_v2.routing.guide_stars import GuideStarManager

    rng = np.random.default_rng(20261004)
    stars = rng.normal(size=(115, 2048)).astype(np.float32)
    training = rng.normal(size=(504, 2048)).astype(np.float32)
    # Synthetic heldout rows stay within the training hull. Actual corpus
    # coverage is separately validated by the retained production receipt.
    heldout = training[:120].copy()
    repo_root = Path(__file__).resolve().parents[3]
    template = json.loads(
        (
            repo_root / "app/consolidated_clustering_v2/config/psa_calibration_v2_115stars.json"
        ).read_text()
    )
    return dict(
        vectors=np.concatenate([training, heldout]),
        stars=stars,
        holdout=np.array([False] * 504 + [True] * 120),
        template=template,
        calibration_id="synthetic_refit",
        expected_frame_hash=GuideStarManager(star_embeddings=stars).version_hash,
    )


def test_refit_fits_training_only_and_initializes_production_router(refit_inputs, tmp_path):
    from scripts.research.psa.refit_routing_calibration import fit_calibration

    from consolidated_clustering_v2.routing.adaptive_grid import AdaptiveVoxelRouter

    artifact, receipt = fit_calibration(**refit_inputs)
    changed = dict(refit_inputs)
    changed["vectors"] = refit_inputs["vectors"].copy()
    changed["vectors"][504:] = refit_inputs["vectors"][120:240]
    other, _ = fit_calibration(**changed)
    assert artifact["axis_edges"] == other["axis_edges"]
    assert receipt["training_samples"] == 504
    assert receipt["heldout_samples"] == 120
    assert artifact["native_embedding_dimension"] == 2048
    assert artifact["position_hash"] == refit_inputs["template"]["position_hash"]
    assert artifact["axis_edges"] != refit_inputs["template"]["axis_edges"]
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(artifact))
    router = AdaptiveVoxelRouter(routing_calibration_path=str(path))
    router.initialize_guide_stars(refit_inputs["stars"])
    _, metadata = router.get_psa_projection(refit_inputs["vectors"][-1].tolist())
    assert metadata["psa_frame_hash"] == artifact["frame_hash"]
    assert metadata["psa_calibration_id"] == "synthetic_refit"


@pytest.mark.parametrize("field", ["vectors", "stars"])
def test_refit_rejects_old_width_instead_of_relabelling_it(refit_inputs, field):
    from scripts.research.psa.refit_routing_calibration import fit_calibration

    refit_inputs[field] = np.tile(refit_inputs[field], (1, 2))
    with pytest.raises(ValueError, match="native 2048-dimensional"):
        fit_calibration(**refit_inputs)


def test_refit_rejects_changed_stars_even_at_the_same_width(refit_inputs):
    from scripts.research.psa.refit_routing_calibration import fit_calibration

    refit_inputs["stars"][0, 0] += 1
    with pytest.raises(ValueError, match="expected runtime frame"):
        fit_calibration(**refit_inputs)


def test_refit_rejects_nonfinite_samples(refit_inputs):
    from scripts.research.psa.refit_routing_calibration import fit_calibration

    refit_inputs["vectors"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite nonzero"):
        fit_calibration(**refit_inputs)


def test_refit_requires_independent_validation_rows(refit_inputs):
    from scripts.research.psa.refit_routing_calibration import fit_calibration

    refit_inputs["holdout"][:] = False
    with pytest.raises(ValueError, match="100 heldout"):
        fit_calibration(**refit_inputs)
