#!/usr/bin/env python3
"""Refit PSA visual quantiles from retained native vectors without network I/O.

The NPZ must contain ``vectors``, ``stars`` and a boolean ``holdout`` mask.
Keep that archive in its approved data environment; only the fitted calibration
and aggregate validation receipt need to leave it. No source text is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[3] / "app"
if APP_ROOT.is_dir():
    sys.path.insert(0, str(APP_ROOT))

from consolidated_clustering_v2.routing.adaptive_grid import AdaptiveVoxelRouter
from consolidated_clustering_v2.routing.guide_stars import (
    GUIDE_STAR_CATALOG,
    GuideStarManager,
)
from consolidated_clustering_v2.routing.psa_calibration import PSARoutingCalibration
from consolidated_clustering_v2.routing.psa_mapper import PSAMapper
from mcp_servers.mcp_lib.semantic_routing_contract import NATIVE_EMBEDDING_DIMENSION


def fit_calibration(
    *,
    vectors: np.ndarray,
    stars: np.ndarray,
    holdout: np.ndarray,
    template: dict,
    calibration_id: str,
    expected_frame_hash: str,
    max_outside_fraction: float = 0.1,
) -> tuple[dict, dict]:
    """Use the production projection, fitting edges on training rows only."""
    width = NATIVE_EMBEDDING_DIMENSION
    for name, matrix, count in (
        ("vectors", vectors, None),
        ("stars", stars, len(GUIDE_STAR_CATALOG)),
    ):
        if matrix.ndim != 2 or matrix.shape[1] != width:
            raise ValueError(f"{name} must contain native {width}-dimensional rows")
        if count is not None and matrix.shape[0] != count:
            raise ValueError(f"{name} must contain exactly {count} rows")
        if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError(f"{name} must contain finite nonzero vectors")
    if holdout.dtype != np.bool_ or holdout.shape != (len(vectors),):
        raise ValueError("holdout must be one boolean per vector")
    if np.count_nonzero(~holdout) < 400 or np.count_nonzero(holdout) < 100:
        raise ValueError("at least 400 training and 100 heldout rows are required")
    if not 0 <= max_outside_fraction <= 0.1:
        raise ValueError("heldout outside fraction must be between zero and 0.1")
    if not calibration_id or calibration_id == template["calibration_id"]:
        raise ValueError("a new calibration_id is required for a refit")

    # Read the pinned positions through the same checksum validation as runtime.
    positions = PSARoutingCalibration(
        calibration_id=template["calibration_id"],
        mode="quantile",
        bins_per_axis=int(template["bins_per_axis"]),
        axis_edges=((), (), ()),
        position_hash=template["position_hash"],
        star_positions_b64=tuple(template["star_positions_b64"]),
        star_positions_dtype=template["star_positions_dtype"],
        star_positions_shape=tuple(template["star_positions_shape"]),
    ).get_star_positions()
    if positions is None or positions.shape != (len(GUIDE_STAR_CATALOG), 3):
        raise ValueError("the pinned 115-star positions are required")
    manager = GuideStarManager(star_embeddings=stars)
    if manager.version_hash != expected_frame_hash:
        raise ValueError("retained stars do not match the expected runtime frame")
    mapper = PSAMapper()
    mapper.set_star_positions(positions)
    coordinates = []
    for vector in vectors:
        point = mapper.signature_to_position(manager.compute_signature(vector))
        coordinates.append((point.x, point.y, point.z))
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("projection produced non-finite coordinates")

    fit = coordinates[~holdout]
    bins = int(template["bins_per_axis"])
    edges = np.quantile(fit, np.linspace(0, 1, bins + 1), axis=0).T
    edges[:, 0] -= 1e-9
    edges[:, -1] += 1e-9
    if not np.all(np.diff(edges, axis=1) > 0):
        raise ValueError("quantile edges must be strictly increasing")
    outside = ((coordinates[holdout] < edges[:, 0]) | (coordinates[holdout] > edges[:, -1])).mean(
        axis=0
    )
    if np.any(outside > max_outside_fraction):
        raise ValueError(f"heldout coverage failed: {outside.tolist()}")

    # Deliberately copy only geometry; do not retain obsolete source or width labels.
    artifact = {
        key: template[key]
        for key in (
            "bins_per_axis",
            "frame_version",
            "label_prefix",
            "position_hash",
            "star_positions_b64",
            "star_positions_dtype",
            "star_positions_shape",
        )
    }
    artifact.update(
        calibration_id=calibration_id,
        mode="quantile",
        native_embedding_dimension=width,
        frame_hash=manager.version_hash,
        axis_edges=edges.tolist(),
        n_sentences=int((~holdout).sum()),
        heldout_count=int(holdout.sum()),
        heldout_outside_fraction_by_axis=outside.tolist(),
        raw_coord_min=fit.min(axis=0).tolist(),
        raw_coord_max=fit.max(axis=0).tolist(),
        normalization="Per-axis empirical quantiles of production PSA projections; training rows only",
    )
    receipt = {
        "native_embedding_dimension": width,
        "frame_hash": manager.version_hash,
        "position_hash": artifact["position_hash"],
        "training_samples": int((~holdout).sum()),
        "heldout_samples": int(holdout.sum()),
        "heldout_outside_fraction_by_axis": outside.tolist(),
        "guide_stars_sha256": hashlib.sha256(stars.tobytes()).hexdigest(),
        "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "holdout_sha256": hashlib.sha256(holdout.tobytes()).hexdigest(),
    }
    return artifact, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--expected-frame-hash", required=True)
    parser.add_argument("--source", required=True, help="Provenance URI, not source text")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    with np.load(args.vectors, allow_pickle=False) as archive:
        vectors, stars, holdout = (archive[key] for key in ("vectors", "stars", "holdout"))
    artifact, receipt = fit_calibration(
        vectors=vectors,
        stars=stars,
        holdout=holdout,
        template=json.loads(args.template.read_text()),
        calibration_id=args.calibration_id,
        expected_frame_hash=args.expected_frame_hash,
    )
    artifact["provenance"] = {
        "generator": "scripts/research/psa/refit_routing_calibration.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_archive_sha256": hashlib.sha256(args.vectors.read_bytes()).hexdigest(),
        "source": args.source,
        "model": args.model,
        "model_revision": args.model_revision,
        "representation": "first_2048_float32; last_token; cosine normalization in GuideStarManager",
        "scope": "Enterprise canary corpus, 504 fit fragments and 120 fragments held out by document; not a universal quality benchmark",
        "guide_stars_sha256": receipt["guide_stars_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    router = AdaptiveVoxelRouter(routing_calibration_path=str(args.output))
    router.initialize_guide_stars(stars)
    for vector in vectors[holdout]:
        label, metadata = router.get_psa_projection(vector.tolist())
        if not label.startswith(artifact["label_prefix"] + "_v_"):
            raise ValueError("heldout projection did not produce a calibrated label")
        if metadata["psa_frame_hash"] != artifact["frame_hash"]:
            raise ValueError("heldout projection emitted a different frame hash")
    receipt.update(
        validated=True,
        runtime_projection_count=int(holdout.sum()),
        calibration_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
    )
    args.output.with_suffix(".validation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"calibration": artifact, "validation": receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
