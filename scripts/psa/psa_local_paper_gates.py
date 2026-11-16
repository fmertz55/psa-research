#!/usr/bin/env python3
"""Local PSA paper gates over checked-in embedding artifacts.

This script intentionally avoids Neo4j/AWS. It runs the paper-critical checks
that can be computed from the exported 3,424-sentence embedding matrix:

1. Random-projection nulls for the fidelity ladder and disjoint-frame claim.
2. Signature-space blocking at the same candidate fractions as calibrated PSA.
3. Held-out retuning of the 115-star quantile routing radius.
4. Affine-vs-quantile calibration comparison for the 115-star frame.
5. Random-sentence anchor nulls for the embedding anisotropy/cone confound.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from consolidated_clustering_v2.routing.guide_stars import GUIDE_STAR_CATALOG  # noqa: E402
from consolidated_clustering_v2.routing.psa_calibration import (  # noqa: E402
    PSARoutingCalibration,
)

DEFAULT_SENTENCES = REPO_ROOT / "temp/psa_frame_overlap/neo4j_sentence_embeddings_3424_current.npz"
DEFAULT_STARS_75 = REPO_ROOT / "temp/psa_frame_overlap/v1_75_star_embeddings.npy"
DEFAULT_STARS_115 = REPO_ROOT / "temp/psa_frame_overlap/v2_115_star_embeddings.npy"
DEFAULT_RADIUS_SWEEP = REPO_ROOT / "temp/psa_frame_overlap/psa_quantile_radius_sweep_results.json"
DEFAULT_CALIBRATION_115 = (
    REPO_ROOT / "app/consolidated_clustering_v2/config/psa_calibration_v2_115stars.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "temp/psa_frame_overlap"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def tier_weights(count: int) -> np.ndarray:
    weights = np.ones(count, dtype=np.float32)
    if count > 69:
        weights[69 : min(75, count)] = 0.4
    return weights


def signature_matrix(
    embeddings_norm: np.ndarray,
    star_embeddings: np.ndarray,
    star_ids: list[int] | None = None,
) -> np.ndarray:
    stars_norm = l2_normalize(star_embeddings.astype(np.float32))
    if star_ids is None:
        ids = list(range(stars_norm.shape[0]))
        selected = stars_norm
    else:
        ids = list(star_ids)
        selected = stars_norm[ids]

    sig = embeddings_norm @ selected.T
    for col, star_id in enumerate(ids):
        if 69 <= star_id <= 74:
            sig[:, col] *= 0.4
    return sig.astype(np.float32)


def pairwise_cosine_distance_vector(
    matrix: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    normalized = l2_normalize(matrix.astype(np.float32))
    similarities = normalized @ normalized.T
    return (1.0 - similarities[tri]).astype(np.float32, copy=False)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    x_centered = x64 - float(np.mean(x64))
    y_centered = y64 - float(np.mean(y64))
    denom = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denom)


def corr_summary(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def split_tier_alternating() -> tuple[list[int], list[int]]:
    left: list[int] = []
    right: list[int] = []
    for tier in sorted({int(star.tier) for star in GUIDE_STAR_CATALOG}):
        ids = [star.id for star in GUIDE_STAR_CATALOG if int(star.tier) == tier]
        for offset, star_id in enumerate(ids):
            (left if offset % 2 == 0 else right).append(star_id)
    return left, right


def contrast_pairs() -> tuple[list[tuple[int, int]], list[int]]:
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    singles: list[int] = []
    by_id = {star.id: star for star in GUIDE_STAR_CATALOG}
    for star_id in sorted(by_id):
        if star_id in used:
            continue
        contrast = by_id[star_id].contrast_with
        if contrast is not None and contrast not in used:
            pairs.append((star_id, int(contrast)))
            used.add(star_id)
            used.add(int(contrast))
        else:
            singles.append(star_id)
            used.add(star_id)
    return pairs, singles


def split_contrast_halves() -> tuple[list[int], list[int]]:
    """Split contrast pairs, flipping every other pair to balance semantics.

    This reproduces the paper's contrast-halves Spearman row and keeps the
    two sides disjoint. Unpaired modal/comparison anchors stay on the first side.
    """
    pairs, singles = contrast_pairs()
    left: list[int] = []
    right: list[int] = []
    for offset, (a, b) in enumerate(pairs):
        if offset % 2 == 0:
            left.append(a)
            right.append(b)
        else:
            left.append(b)
            right.append(a)
    left.extend(singles)
    return left, right


@dataclass(frozen=True)
class SpaceDistances:
    name: str
    distances: np.ndarray
    to_full_r: float
    to_full_rho: float | None = None


def compute_semantic_spaces(
    embeddings_norm: np.ndarray,
    stars_75: np.ndarray,
    stars_115: np.ndarray,
    full_dist: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    spaces: dict[str, SpaceDistances] = {}
    for name, stars, ids in (
        ("75_signature", stars_75, None),
        ("115_signature", stars_115, None),
        ("tier_split_a", stars_115, split_tier_alternating()[0]),
        ("tier_split_b", stars_115, split_tier_alternating()[1]),
        ("contrast_split_a", stars_115, split_contrast_halves()[0]),
        ("contrast_split_b", stars_115, split_contrast_halves()[1]),
    ):
        sig = signature_matrix(embeddings_norm, stars, ids)
        dist = pairwise_cosine_distance_vector(sig, tri)
        spaces[name] = SpaceDistances(
            name=name,
            distances=dist,
            to_full_r=pearson(full_dist, dist),
        )

    pair_rows: dict[str, Any] = {}
    for name, left, right in (
        ("75_vs_115", "75_signature", "115_signature"),
        ("tier_alternating", "tier_split_a", "tier_split_b"),
        ("contrast_halves", "contrast_split_a", "contrast_split_b"),
    ):
        ldist = spaces[left].distances
        rdist = spaces[right].distances
        pair_rows[name] = {
            "left": left,
            "right": right,
            "left_dims": int(signature_dim(left, stars_75.shape[0])),
            "right_dims": int(signature_dim(right, stars_75.shape[0])),
            "left_to_full_r": spaces[left].to_full_r,
            "right_to_full_r": spaces[right].to_full_r,
            "pair_r": pearson(ldist, rdist),
            "pair_spearman_rho": float(spearmanr(ldist, rdist).statistic),
        }
        pair_rows[name]["independence_floor"] = float(
            pair_rows[name]["left_to_full_r"] * pair_rows[name]["right_to_full_r"]
        )
        pair_rows[name]["excess_over_floor"] = float(
            pair_rows[name]["pair_r"] - pair_rows[name]["independence_floor"]
        )

    return {
        "single_space": {
            name: {
                "to_full_r": space.to_full_r,
                "to_full_spearman_rho": float(spearmanr(full_dist, space.distances).statistic)
                if name in {"75_signature", "115_signature"}
                else None,
            }
            for name, space in spaces.items()
        },
        "pairs": pair_rows,
        "_distances": {name: space.distances for name, space in spaces.items()},
    }


def signature_dim(name: str, stars_75_count: int) -> int:
    if name == "75_signature":
        return stars_75_count
    if name == "115_signature":
        return len(GUIDE_STAR_CATALOG)
    if name == "tier_split_a":
        return len(split_tier_alternating()[0])
    if name == "tier_split_b":
        return len(split_tier_alternating()[1])
    if name == "contrast_split_a":
        return len(split_contrast_halves()[0])
    if name == "contrast_split_b":
        return len(split_contrast_halves()[1])
    return 0


def random_unit_projection(
    rng: np.random.Generator,
    input_dim: int,
    output_dim: int,
) -> np.ndarray:
    projection = rng.normal(size=(input_dim, output_dim)).astype(np.float32)
    norms = np.linalg.norm(projection, axis=0, keepdims=True)
    return projection / np.maximum(norms, 1e-12)


def run_random_projection_nulls(
    embeddings_norm: np.ndarray,
    full_dist: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    trials: int,
    spearman_trials: int,
) -> dict[str, Any]:
    if trials <= 0:
        return {"skipped": True, "reason": "random projection trials <= 0"}

    full_115: list[float] = []
    full_115_rho: list[float] = []
    pair_115: list[float] = []
    pair_115_left_full: list[float] = []
    pair_115_right_full: list[float] = []
    pair_58_57: list[float] = []
    pair_58_57_left_full: list[float] = []
    pair_58_57_right_full: list[float] = []
    pair_59_56: list[float] = []
    pair_59_56_left_full: list[float] = []
    pair_59_56_right_full: list[float] = []

    input_dim = embeddings_norm.shape[1]
    for trial in range(trials):
        if trial == 0 or (trial + 1) % 100 == 0 or trial + 1 == trials:
            print(f"  random projection trial {trial + 1}/{trials}", flush=True)

        r115 = random_unit_projection(rng, input_dim, 115)
        d115 = pairwise_cosine_distance_vector(embeddings_norm @ r115, tri)
        full_115.append(pearson(full_dist, d115))
        if trial < spearman_trials:
            full_115_rho.append(float(spearmanr(full_dist, d115).statistic))

        l115 = random_unit_projection(rng, input_dim, 115)
        r115b = random_unit_projection(rng, input_dim, 115)
        dl115 = pairwise_cosine_distance_vector(embeddings_norm @ l115, tri)
        dr115 = pairwise_cosine_distance_vector(embeddings_norm @ r115b, tri)
        pair_115_left_full.append(pearson(full_dist, dl115))
        pair_115_right_full.append(pearson(full_dist, dr115))
        pair_115.append(pearson(dl115, dr115))

        l58 = random_unit_projection(rng, input_dim, 58)
        r57 = random_unit_projection(rng, input_dim, 57)
        dl58 = pairwise_cosine_distance_vector(embeddings_norm @ l58, tri)
        dr57 = pairwise_cosine_distance_vector(embeddings_norm @ r57, tri)
        pair_58_57_left_full.append(pearson(full_dist, dl58))
        pair_58_57_right_full.append(pearson(full_dist, dr57))
        pair_58_57.append(pearson(dl58, dr57))

        l59 = random_unit_projection(rng, input_dim, 59)
        r56 = random_unit_projection(rng, input_dim, 56)
        dl59 = pairwise_cosine_distance_vector(embeddings_norm @ l59, tri)
        dr56 = pairwise_cosine_distance_vector(embeddings_norm @ r56, tri)
        pair_59_56_left_full.append(pearson(full_dist, dl59))
        pair_59_56_right_full.append(pearson(full_dist, dr56))
        pair_59_56.append(pearson(dl59, dr56))

    def paired_summary(
        pair_values: list[float],
        left_full: list[float],
        right_full: list[float],
    ) -> dict[str, Any]:
        products = np.asarray(left_full) * np.asarray(right_full)
        return {
            "frame_to_frame_r": corr_summary(pair_values),
            "left_to_full_r": corr_summary(left_full),
            "right_to_full_r": corr_summary(right_full),
            "left_right_product_r_floor": corr_summary(products.astype(float).tolist()),
        }

    return {
        "skipped": False,
        "trials": int(trials),
        "spearman_trials": int(spearman_trials),
        "single_115_to_full": {
            "pearson_r": corr_summary(full_115),
            "spearman_rho": corr_summary(full_115_rho) if full_115_rho else None,
            "trial_values_pearson_r": full_115,
        },
        "independent_115x115": {
            **paired_summary(pair_115, pair_115_left_full, pair_115_right_full),
            "trial_values_frame_to_frame_r": pair_115,
        },
        "independent_58x57": {
            **paired_summary(pair_58_57, pair_58_57_left_full, pair_58_57_right_full),
            "trial_values_frame_to_frame_r": pair_58_57,
        },
        "independent_59x56": {
            **paired_summary(pair_59_56, pair_59_56_left_full, pair_59_56_right_full),
            "trial_values_frame_to_frame_r": pair_59_56,
        },
    }


def random_sentence_anchor_frame(
    embeddings_norm: np.ndarray,
    anchor_indices: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    signatures = embeddings_norm @ embeddings_norm[anchor_indices].T
    return pairwise_cosine_distance_vector(signatures, tri)


def random_sentence_pair_trial(
    embeddings_norm: np.ndarray,
    full_dist: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
    left_size: int,
    right_size: int,
    rng: np.random.Generator,
    nonanchor_pairs: bool,
) -> dict[str, float]:
    n = embeddings_norm.shape[0]
    anchor_indices = rng.choice(n, size=left_size + right_size, replace=False)
    left_indices = anchor_indices[:left_size]
    right_indices = anchor_indices[left_size:]
    left_dist = random_sentence_anchor_frame(embeddings_norm, left_indices, tri)
    right_dist = random_sentence_anchor_frame(embeddings_norm, right_indices, tri)

    if nonanchor_pairs:
        anchor_mask = np.zeros(n, dtype=bool)
        anchor_mask[anchor_indices] = True
        pair_mask = (~anchor_mask[tri[0]]) & (~anchor_mask[tri[1]])
        full = full_dist[pair_mask]
        left = left_dist[pair_mask]
        right = right_dist[pair_mask]
    else:
        full = full_dist
        left = left_dist
        right = right_dist

    left_full = pearson(full, left)
    right_full = pearson(full, right)
    pair_r = pearson(left, right)
    return {
        "left_to_full_r": left_full,
        "right_to_full_r": right_full,
        "frame_to_frame_r": pair_r,
        "independence_floor": float(left_full * right_full),
        "excess_over_floor": float(pair_r - (left_full * right_full)),
        "evaluated_pairs": int(full.size),
    }


def run_random_sentence_anchor_nulls(
    embeddings_norm: np.ndarray,
    full_dist: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    trials: int,
    semantic_reference: dict[str, float],
    include_nonanchor_pairs: bool,
) -> dict[str, Any]:
    if trials <= 0:
        return {"skipped": True, "reason": "random sentence trials <= 0"}

    configs = {
        "disjoint_58x57_all_pairs": (
            58,
            57,
            False,
            semantic_reference["tier_alternating_pair_r"],
            semantic_reference["tier_alternating_excess_over_floor"],
        ),
        "disjoint_59x56_all_pairs": (
            59,
            56,
            False,
            semantic_reference["contrast_halves_pair_r"],
            semantic_reference["contrast_halves_excess_over_floor"],
        ),
    }
    if include_nonanchor_pairs:
        configs.update(
            {
                "disjoint_58x57_nonanchor_pairs": (
                    58,
                    57,
                    True,
                    semantic_reference["tier_alternating_pair_r"],
                    semantic_reference["tier_alternating_excess_over_floor"],
                ),
                "disjoint_59x56_nonanchor_pairs": (
                    59,
                    56,
                    True,
                    semantic_reference["contrast_halves_pair_r"],
                    semantic_reference["contrast_halves_excess_over_floor"],
                ),
            }
        )

    trial_rows: dict[str, list[dict[str, float]]] = {name: [] for name in configs}
    for trial in range(trials):
        if trial == 0 or (trial + 1) % 100 == 0 or trial + 1 == trials:
            print(f"  random sentence-anchor trial {trial + 1}/{trials}", flush=True)
        for name, (left_size, right_size, nonanchor_pairs, _, _) in configs.items():
            trial_rows[name].append(
                random_sentence_pair_trial(
                    embeddings_norm=embeddings_norm,
                    full_dist=full_dist,
                    tri=tri,
                    left_size=left_size,
                    right_size=right_size,
                    rng=rng,
                    nonanchor_pairs=nonanchor_pairs,
                )
            )

    by_config: dict[str, Any] = {}
    for name, rows in trial_rows.items():
        _, _, nonanchor_pairs, semantic_r, semantic_excess = configs[name]
        frame_values = [row["frame_to_frame_r"] for row in rows]
        floor_values = [row["independence_floor"] for row in rows]
        excess_values = [row["excess_over_floor"] for row in rows]
        left_values = [row["left_to_full_r"] for row in rows]
        right_values = [row["right_to_full_r"] for row in rows]
        by_config[name] = {
            "trials": int(trials),
            "nonanchor_pairs": bool(nonanchor_pairs),
            "semantic_reference_pair_r": float(semantic_r),
            "semantic_reference_excess_over_floor": float(semantic_excess),
            "empirical_p_random_ge_semantic_pair_r": float(
                np.mean(np.asarray(frame_values, dtype=np.float64) >= semantic_r)
            ),
            "empirical_p_random_ge_semantic_excess": float(
                np.mean(np.asarray(excess_values, dtype=np.float64) >= semantic_excess)
            ),
            "frame_to_frame_r": corr_summary(frame_values),
            "left_to_full_r": corr_summary(left_values),
            "right_to_full_r": corr_summary(right_values),
            "independence_floor": corr_summary(floor_values),
            "excess_over_floor": corr_summary(excess_values),
            "evaluated_pairs": corr_summary([float(row["evaluated_pairs"]) for row in rows]),
            "trial_values_frame_to_frame_r": frame_values,
            "trial_values_excess_over_floor": excess_values,
        }

    return {
        "skipped": False,
        "trials": int(trials),
        "source": "random disjoint anchor frames sampled from evaluated sentence embeddings",
        "by_config": by_config,
    }


def load_radius_rows(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def lookup_radius_row(radius_sweep: dict[str, Any], frame: str, radius: int) -> dict[str, Any]:
    rows = radius_sweep["frames"][frame]["coarse"]["radii"]
    for row in rows:
        if int(row["radius"]) == int(radius):
            return row
    raise KeyError(f"No radius {radius} for frame {frame}")


def matched_cost_signature_blocking(
    full_sim: np.ndarray,
    signature_distances: dict[str, np.ndarray],
    radius_sweep: dict[str, Any],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    total_pairs = int(full_sim.size)
    plan = [
        ("75", "75_signature", 0.88, 6),
        ("115", "115_signature", 0.88, 6),
        ("75", "75_signature", 0.85, 8),
        ("115", "115_signature", 0.85, 8),
        ("75", "75_signature", 0.77, 10),
        ("115", "115_signature", 0.77, 11),
    ]
    rows: list[dict[str, Any]] = []
    for frame, signature_name, threshold, radius in plan:
        radius_row = lookup_radius_row(radius_sweep, frame, radius)
        target_pairs = int(radius_row["candidate_pairs"])
        sig_dist = signature_distances[signature_name]
        cutoff = float(np.partition(sig_dist, target_pairs - 1)[target_pairs - 1])
        candidates = sig_dist <= cutoff
        high = full_sim >= threshold
        in_candidates = int(np.count_nonzero(candidates & high))
        high_pairs = int(np.count_nonzero(high))
        original = radius_row["thresholds"][f"{threshold:.2f}"]
        rows.append(
            {
                "frame": frame,
                "threshold": threshold,
                "matched_radius": int(radius),
                "candidate_pairs": target_pairs,
                "candidate_fraction": float(target_pairs / total_pairs),
                "signature_distance_cutoff": cutoff,
                "signature_recall": float(in_candidates / high_pairs) if high_pairs else 0.0,
                "signature_in_candidates": in_candidates,
                "high_similarity_pairs": high_pairs,
                "quantile_radius_recall": float(original["recall"]),
                "quantile_radius_in_shell": int(original["in_shell"]),
                "delta_recall_signature_minus_quantile": float(
                    (in_candidates / high_pairs if high_pairs else 0.0) - float(original["recall"])
                ),
            }
        )

    return {
        "method": "signature_distance_threshold_at_matched_candidate_fraction",
        "thresholds": list(thresholds),
        "rows": rows,
    }


def raw_115_positions(
    embeddings_norm: np.ndarray,
    stars_115: np.ndarray,
    calibration_path: Path,
) -> np.ndarray:
    calibration = PSARoutingCalibration.from_json_file(calibration_path)
    star_positions = calibration.get_star_positions()
    if star_positions is None:
        raise RuntimeError(f"Calibration file {calibration_path} lacks star positions")
    signatures = signature_matrix(embeddings_norm, stars_115)
    weights = np.maximum(signatures, 0.0)
    denom = weights.sum(axis=1, keepdims=True)
    return np.divide(
        weights @ star_positions,
        denom,
        out=np.zeros((weights.shape[0], 3), dtype=np.float32),
        where=denom > 0,
    )


def quantile_indices_from_calibration(positions: np.ndarray, calibration_path: Path) -> np.ndarray:
    calibration = PSARoutingCalibration.from_json_file(calibration_path)
    labels = [calibration.coord_to_indices(float(x), float(y), float(z)) for x, y, z in positions]
    return np.asarray(labels, dtype=np.int16)


def affine_indices(positions: np.ndarray, bins: int = 20) -> np.ndarray:
    lo = positions.min(axis=0, keepdims=True)
    hi = positions.max(axis=0, keepdims=True)
    span = np.maximum(hi - lo, 1e-12)
    scaled = (positions - lo) / span
    return np.clip(np.floor(scaled * bins).astype(np.int16), 0, bins - 1)


def chebyshev_pair_distances(indices: np.ndarray, tri: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    diff = np.abs(indices[:, None, :] - indices[None, :, :])
    return np.max(diff, axis=2)[tri].astype(np.int16, copy=False)


def radius_sweep_from_pair_distances(
    pair_distances: np.ndarray,
    full_sim: np.ndarray,
    thresholds: tuple[float, ...],
    max_radius: int = 19,
) -> list[dict[str, Any]]:
    total_pairs = int(pair_distances.size)
    rows: list[dict[str, Any]] = []
    for radius in range(max_radius + 1):
        candidates = pair_distances <= radius
        candidate_pairs = int(np.count_nonzero(candidates))
        threshold_rows: dict[str, Any] = {}
        for threshold in thresholds:
            high = full_sim >= threshold
            high_pairs = int(np.count_nonzero(high))
            in_shell = int(np.count_nonzero(candidates & high))
            threshold_rows[f"{threshold:.2f}"] = {
                "high_similarity_pairs": high_pairs,
                "in_shell": in_shell,
                "recall": float(in_shell / high_pairs) if high_pairs else 0.0,
            }
        rows.append(
            {
                "radius": radius,
                "candidate_pairs": candidate_pairs,
                "candidate_fraction": float(candidate_pairs / total_pairs),
                "pruning_fraction": float(1.0 - candidate_pairs / total_pairs),
                "thresholds": threshold_rows,
            }
        )
    return rows


def heldout_radius_retune(
    pair_distances: np.ndarray,
    full_sim: np.ndarray,
    rng: np.random.Generator,
    trials: int,
    thresholds_to_targets: dict[float, float],
    max_radius: int = 19,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    radius_masks = [pair_distances <= radius for radius in range(max_radius + 1)]
    for threshold, target_recall in thresholds_to_targets.items():
        high_idx = np.flatnonzero(full_sim >= threshold)
        trial_rows: list[dict[str, Any]] = []
        for trial in range(trials):
            shuffled = rng.permutation(high_idx)
            midpoint = len(shuffled) // 2
            train = shuffled[:midpoint]
            test = shuffled[midpoint:]
            selected_radius = max_radius
            train_recall = 0.0
            for radius, mask in enumerate(radius_masks):
                train_recall = (
                    float(np.count_nonzero(mask[train]) / len(train)) if len(train) else 0.0
                )
                if train_recall >= target_recall:
                    selected_radius = radius
                    break
            selected_mask = radius_masks[selected_radius]
            test_recall = (
                float(np.count_nonzero(selected_mask[test]) / len(test)) if len(test) else 0.0
            )
            trial_rows.append(
                {
                    "trial": trial,
                    "selected_radius": selected_radius,
                    "train_recall": train_recall,
                    "test_recall": test_recall,
                    "candidate_fraction": float(np.mean(selected_mask)),
                }
            )

        selected = [row["selected_radius"] for row in trial_rows]
        test_recalls = [row["test_recall"] for row in trial_rows]
        candidate_fractions = [row["candidate_fraction"] for row in trial_rows]
        rows[f"{threshold:.2f}"] = {
            "threshold": threshold,
            "target_train_recall": target_recall,
            "high_similarity_pairs": int(len(high_idx)),
            "trials": int(trials),
            "selected_radius": corr_summary([float(v) for v in selected]),
            "test_recall": corr_summary(test_recalls),
            "candidate_fraction": corr_summary(candidate_fractions),
        }
    return {
        "method": "split_high_similarity_pairs_select_min_radius_for_train_recall",
        "trials": int(trials),
        "rows": rows,
    }


def calibration_comparison(
    positions_115: np.ndarray,
    full_sim: np.ndarray,
    tri: tuple[np.ndarray, np.ndarray],
    calibration_path: Path,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    quantile_idx = quantile_indices_from_calibration(positions_115, calibration_path)
    affine_idx = affine_indices(positions_115, bins=20)
    quantile_pair_dist = chebyshev_pair_distances(quantile_idx, tri)
    affine_pair_dist = chebyshev_pair_distances(affine_idx, tri)
    return {
        "quantile": {
            "occupied_cells": int(np.unique(quantile_idx, axis=0).shape[0]),
            "radius_sweep": radius_sweep_from_pair_distances(
                quantile_pair_dist, full_sim, thresholds
            ),
            "_pair_distances": quantile_pair_dist,
        },
        "affine_minmax": {
            "occupied_cells": int(np.unique(affine_idx, axis=0).shape[0]),
            "radius_sweep": radius_sweep_from_pair_distances(
                affine_pair_dist, full_sim, thresholds
            ),
            "_pair_distances": affine_pair_dist,
        },
    }


def strip_private_arrays(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            key: strip_private_arrays(value)
            for key, value in data.items()
            if not key.startswith("_")
        }
    if isinstance(data, list):
        return [strip_private_arrays(value) for value in data]
    return data


def strip_trial_values(data: Any) -> Any:
    if isinstance(data, dict):
        return {
            key: strip_trial_values(value)
            for key, value in data.items()
            if not key.startswith("trial_values")
        }
    if isinstance(data, list):
        return [strip_trial_values(value) for value in data]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-npz", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--stars-75", type=Path, default=DEFAULT_STARS_75)
    parser.add_argument("--stars-115", type=Path, default=DEFAULT_STARS_115)
    parser.add_argument("--radius-sweep", type=Path, default=DEFAULT_RADIUS_SWEEP)
    parser.add_argument("--calibration-115", type=Path, default=DEFAULT_CALIBRATION_115)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--random-spearman-trials", type=int, default=25)
    parser.add_argument("--random-sentence-trials", type=int, default=0)
    parser.add_argument(
        "--skip-random-sentence-nonanchor",
        action="store_true",
        help="Skip the slower non-anchor-pair leakage control for random sentence anchors.",
    )
    parser.add_argument("--holdout-trials", type=int, default=200)
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    thresholds = (0.77, 0.85, 0.88)

    print("Loading local PSA artifacts...")
    sentence_npz = np.load(args.sentence_npz, allow_pickle=True)
    embeddings = sentence_npz["embeddings"].astype(np.float32)
    embeddings_norm = l2_normalize(embeddings)
    stars_75 = np.load(args.stars_75).astype(np.float32)
    stars_115 = np.load(args.stars_115).astype(np.float32)
    tri = np.triu_indices(embeddings.shape[0], k=1)

    print("Computing full embedding pair distances...")
    full_dist = pairwise_cosine_distance_vector(embeddings_norm, tri)
    full_sim = (1.0 - full_dist).astype(np.float32, copy=False)

    print("Computing semantic signature spaces...")
    semantic = compute_semantic_spaces(embeddings_norm, stars_75, stars_115, full_dist, tri)
    semantic_distances = semantic.pop("_distances")

    print(f"Running random-projection nulls ({args.random_trials} trials)...")
    random_nulls = run_random_projection_nulls(
        embeddings_norm=embeddings_norm,
        full_dist=full_dist,
        tri=tri,
        rng=rng,
        trials=args.random_trials,
        spearman_trials=args.random_spearman_trials,
    )

    tier_r = semantic["pairs"]["tier_alternating"]["pair_r"]
    contrast_r = semantic["pairs"]["contrast_halves"]["pair_r"]
    if not random_nulls.get("skipped"):
        random_nulls["empirical_p_random_ge_semantic"] = {
            "tier_vs_independent_58x57": float(
                np.mean(
                    np.asarray(
                        random_nulls["independent_58x57"]["trial_values_frame_to_frame_r"],
                        dtype=np.float64,
                    )
                    >= tier_r
                )
            ),
            "contrast_vs_independent_59x56": float(
                np.mean(
                    np.asarray(
                        random_nulls["independent_59x56"]["trial_values_frame_to_frame_r"],
                        dtype=np.float64,
                    )
                    >= contrast_r
                )
            ),
        }
    random_nulls["semantic_reference"] = {
        "tier_alternating_pair_r": tier_r,
        "contrast_halves_pair_r": contrast_r,
        "tier_alternating_excess_over_floor": semantic["pairs"]["tier_alternating"][
            "excess_over_floor"
        ],
        "contrast_halves_excess_over_floor": semantic["pairs"]["contrast_halves"][
            "excess_over_floor"
        ],
    }

    print(f"Running random-sentence anchor nulls ({args.random_sentence_trials} trials)...")
    random_sentence_nulls = run_random_sentence_anchor_nulls(
        embeddings_norm=embeddings_norm,
        full_dist=full_dist,
        tri=tri,
        rng=rng,
        trials=args.random_sentence_trials,
        semantic_reference=random_nulls["semantic_reference"],
        include_nonanchor_pairs=not args.skip_random_sentence_nonanchor,
    )

    print("Running matched-cost signature-space blocking...")
    radius_sweep = load_radius_rows(args.radius_sweep)
    blocking = matched_cost_signature_blocking(
        full_sim=full_sim,
        signature_distances=semantic_distances,
        radius_sweep=radius_sweep,
        thresholds=thresholds,
    )

    print("Running 115-star calibration and held-out radius checks...")
    positions_115 = raw_115_positions(embeddings_norm, stars_115, args.calibration_115)
    calibration = calibration_comparison(
        positions_115, full_sim, tri, args.calibration_115, thresholds
    )
    quantile_pair_distances = calibration["quantile"]["_pair_distances"]
    heldout = heldout_radius_retune(
        pair_distances=quantile_pair_distances,
        full_sim=full_sim,
        rng=rng,
        trials=args.holdout_trials,
        thresholds_to_targets={0.88: 0.97, 0.85: 0.965, 0.77: 0.965},
    )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {
            "sentence_npz": str(args.sentence_npz),
            "stars_75": str(args.stars_75),
            "stars_115": str(args.stars_115),
            "radius_sweep": str(args.radius_sweep),
            "calibration_115": str(args.calibration_115),
            "sentences": int(embeddings.shape[0]),
            "embedding_dims": int(embeddings.shape[1]),
            "total_pairs": int(full_dist.size),
            "seed": int(args.seed),
        },
        "semantic": semantic,
        "random_projection_nulls": random_nulls,
        "random_sentence_anchor_nulls": random_sentence_nulls,
        "matched_cost_signature_blocking": blocking,
        "calibration_comparison_115": calibration,
        "heldout_radius_retune_115": heldout,
        "runtime_seconds": round(time.time() - started, 3),
    }

    serializable = strip_private_arrays(result)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"psa_local_paper_gates_{stamp}.json"
    latest_path = args.out_dir / "psa_local_paper_gates_latest.json"
    out_path.write_text(json.dumps(serializable, indent=2) + "\n")
    latest_path.write_text(json.dumps(serializable, indent=2) + "\n")

    print(f"Saved: {out_path}")
    print(f"Saved: {latest_path}")
    print(
        json.dumps(
            {
                "semantic_pairs": serializable["semantic"]["pairs"],
                "random_single_115_to_full": strip_trial_values(
                    serializable["random_projection_nulls"].get("single_115_to_full")
                ),
                "random_sentence_anchor_nulls": strip_trial_values(
                    serializable["random_sentence_anchor_nulls"]
                ),
                "matched_cost_signature_blocking": serializable["matched_cost_signature_blocking"][
                    "rows"
                ],
                "heldout_radius_retune_115": serializable["heldout_radius_retune_115"]["rows"],
                "runtime_seconds": serializable["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
