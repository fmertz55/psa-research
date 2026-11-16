#!/usr/bin/env python3
"""Combined PSA paper baselines that keep large vectors out of SSM stdout.

This harness is intended to run where Neo4j is reachable, preferably on the
Neo4j EC2 host. It loads sentence embeddings once over Bolt, computes the
paper-critical baselines in process, and writes compact JSON results:

1. Full-4096D voxel-centroid locality baseline.
2. Optional first-200D truncation validation against full-D cosine.
3. Random-anchor PSA controls using randomly sampled sentence embeddings as
   guide stars.

Only aggregate metrics are written or printed; full vectors never transit SSM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # Allow --help to work before runtime dependency checks.
    np = None


PSA_TEST_RUNS = Path(os.environ.get("PSA_TEST_RUNS", Path.cwd() / "psa_test_runs"))
DEFAULT_SSM_PARAM_NAME = "/example/neo4j/password"
VOXEL_RE = re.compile(r"L(\d+)_v_(\d+)_(\d+)_(\d+)")


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    voxel: str
    coord: tuple[float, float, float]


def parse_voxel_coords(
    visual_voxel_id: str,
) -> tuple[float, float, float] | None:
    """Parse a PSA voxel label into L1-normalized coordinate indexes."""
    match = VOXEL_RE.search(visual_voxel_id)
    if not match:
        return None
    level = int(match.group(1))
    scale = 2 ** max(level - 1, 0)
    return (
        int(match.group(2)) / scale,
        int(match.group(3)) / scale,
        int(match.group(4)) / scale,
    )


def resolve_password(args: argparse.Namespace) -> str:
    if args.password:
        return args.password

    env_value = os.environ.get(args.password_env)
    if env_value:
        return env_value

    if args.ssm_param:
        result = subprocess.run(
            [
                "aws",
                "ssm",
                "get-parameter",
                "--name",
                args.ssm_param,
                "--with-decryption",
                "--query",
                "Parameter.Value",
                "--output",
                "text",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    raise RuntimeError(
        f"No Neo4j password found. Set {args.password_env}, pass --password, or pass --ssm-param."
    )


def get_driver(uri: str, user: str, password: str):
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python package 'neo4j'. Install it where this script runs."
        ) from exc

    return GraphDatabase.driver(uri, auth=(user, password))


def valid_node_label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
        raise ValueError(f"Unsafe Neo4j node label: {label!r}")
    return label


def fetch_sentence_embeddings(
    driver: Any,
    database: str | None,
    node_label: str,
    limit: int | None,
) -> tuple[list[SentenceRecord], np.ndarray]:
    """Fetch embedded sentence nodes with parseable PSA coordinates."""
    label = valid_node_label(node_label)
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
    MATCH (s:{label})
    WHERE s.visual_voxel_id IS NOT NULL AND s.embedding IS NOT NULL
    RETURN coalesce(s.sentence_id, s.id, elementId(s)) AS sentence_id,
           s.visual_voxel_id AS voxel,
           s.embedding AS embedding
    ORDER BY id(s)
    {limit_clause}
    """

    records: list[SentenceRecord] = []
    embeddings: list[list[float]] = []
    bad_voxels = 0
    bad_embeddings = 0
    dims: int | None = None

    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        result = session.run(query)
        for row in result:
            coord = parse_voxel_coords(str(row["voxel"]))
            if coord is None:
                bad_voxels += 1
                continue

            embedding = row["embedding"]
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    bad_embeddings += 1
                    continue

            if not isinstance(embedding, Iterable):
                bad_embeddings += 1
                continue

            embedding_list = list(embedding)
            if not embedding_list:
                bad_embeddings += 1
                continue

            if dims is None:
                dims = len(embedding_list)
            elif len(embedding_list) != dims:
                bad_embeddings += 1
                continue

            records.append(
                SentenceRecord(
                    sentence_id=str(row["sentence_id"]),
                    voxel=str(row["voxel"]),
                    coord=coord,
                )
            )
            embeddings.append(embedding_list)

    if not embeddings:
        raise RuntimeError(
            "No usable sentence embeddings found. Neo4j must contain "
            f":{label} nodes with visual_voxel_id and embedding."
        )

    matrix = np.asarray(embeddings, dtype=np.float32)
    print(
        f"Loaded {len(records)} embedded {label} nodes "
        f"({matrix.shape[1]} dims, skipped {bad_voxels} bad voxels, "
        f"{bad_embeddings} bad embeddings)."
    )
    return records, matrix


def fetch_sentence_embeddings_cypher_shell(
    args: argparse.Namespace,
    password: str,
) -> tuple[list[SentenceRecord], np.ndarray]:
    """Fetch embeddings through cypher-shell, usually via docker exec.

    This mode avoids the Python Neo4j driver on minimal EC2 hosts. The script
    still runs next to Neo4j and captures the large vector payload locally;
    SSM only receives compact progress/result output.
    """
    label = valid_node_label(args.node_label)
    limit_clause = f"LIMIT {int(args.limit)}" if args.limit else ""
    database_args = ["-d", args.neo4j_database] if args.neo4j_database else []
    query = f"""
    MATCH (s:{label})
    WHERE s.visual_voxel_id IS NOT NULL AND s.embedding IS NOT NULL
    RETURN coalesce(s.sentence_id, s.id, elementId(s)) AS sentence_id,
           s.visual_voxel_id AS voxel,
           apoc.convert.toJson(s.embedding) AS embedding_json
    ORDER BY id(s)
    {limit_clause}
    """

    shell_cmd = shlex.split(args.cypher_shell_command)
    if args.docker_container:
        command = ["docker", "exec", args.docker_container] + shell_cmd
    else:
        command = shell_cmd
    command += [
        "-u",
        args.neo4j_user,
        "-p",
        password,
        "--format",
        "plain",
        "--wrap",
        "false",
        "--non-interactive",
        *database_args,
        query,
    ]

    print("Fetching embeddings via cypher-shell...")
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    records: list[SentenceRecord] = []
    embeddings: list[list[float]] = []
    bad_voxels = 0
    bad_embeddings = 0
    dims: int | None = None

    assert proc.stdout is not None
    reader = csv.reader(proc.stdout, skipinitialspace=True)
    header_seen = False
    for row in reader:
        if not row:
            continue
        if not header_seen:
            header_seen = True
            if row[:3] == ["sentence_id", "voxel", "embedding_json"]:
                continue

        if len(row) < 3:
            bad_embeddings += 1
            continue

        sentence_id, voxel, embedding_json = row[0], row[1], row[2]
        coord = parse_voxel_coords(voxel)
        if coord is None:
            bad_voxels += 1
            continue

        try:
            embedding = json.loads(embedding_json)
        except json.JSONDecodeError:
            bad_embeddings += 1
            continue

        if dims is None:
            dims = len(embedding)
        elif len(embedding) != dims:
            bad_embeddings += 1
            continue

        records.append(SentenceRecord(sentence_id=sentence_id, voxel=voxel, coord=coord))
        embeddings.append(embedding)
        if len(records) == 1 or len(records) % 500 == 0:
            print(f"  fetched {len(records)} embeddings", flush=True)

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"cypher-shell failed with exit code {returncode}: {stderr[-4000:]}")

    if not embeddings:
        raise RuntimeError(
            f"No usable sentence embeddings found through cypher-shell. stderr={stderr[-1000:]}"
        )

    matrix = np.asarray(embeddings, dtype=np.float32)
    print(
        f"Loaded {len(records)} embedded {label} nodes "
        f"({matrix.shape[1]} dims, skipped {bad_voxels} bad voxels, "
        f"{bad_embeddings} bad embeddings)."
    )
    return records, matrix


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return matrix / norms


def pair_metrics(
    coords: np.ndarray,
    centroid_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compute locality metrics from coordinate and centroid matrices."""
    n = coords.shape[0]
    if n < 2:
        return {
            "n_voxels": int(n),
            "n_pairs": 0,
            "correlation": 0.0,
            "slope": 0.0,
            "intercept": 0.0,
            "mean_similarity": 0.0,
        }

    centroids_norm = l2_normalize(centroid_matrix)
    sim_matrix = centroids_norm @ centroids_norm.T

    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    triu = np.triu_indices(n, k=1)
    distances = dist_matrix[triu]
    similarities = sim_matrix[triu]

    if len(distances) > 1 and np.std(distances) > 0 and np.std(similarities) > 0:
        correlation = float(np.corrcoef(distances, similarities)[0, 1])
        slope, intercept = np.polyfit(distances, similarities, 1)
    else:
        correlation = 0.0
        slope = 0.0
        intercept = float(np.mean(similarities)) if len(similarities) else 0.0

    return {
        "n_voxels": int(n),
        "n_pairs": int(len(distances)),
        "correlation": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
        "mean_similarity": float(np.mean(similarities)) if len(similarities) else 0.0,
        "distance_min": float(np.min(distances)) if len(distances) else 0.0,
        "distance_max": float(np.max(distances)) if len(distances) else 0.0,
        "similarity_min": float(np.min(similarities)) if len(similarities) else 0.0,
        "similarity_max": float(np.max(similarities)) if len(similarities) else 0.0,
    }


def centroids_from_string_labels(
    records: list[SentenceRecord],
    embeddings: np.ndarray,
    min_sentences: int,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    by_voxel: dict[str, list[int]] = {}
    coords_by_voxel: dict[str, tuple[float, float, float]] = {}
    for i, record in enumerate(records):
        by_voxel.setdefault(record.voxel, []).append(i)
        coords_by_voxel[record.voxel] = record.coord

    voxel_names: list[str] = []
    coords: list[tuple[float, float, float]] = []
    centroids: list[np.ndarray] = []
    sizes: dict[str, int] = {}

    for voxel, idxs in sorted(by_voxel.items()):
        if len(idxs) < min_sentences:
            continue
        voxel_names.append(voxel)
        coords.append(coords_by_voxel[voxel])
        centroids.append(np.mean(embeddings[idxs], axis=0))
        sizes[voxel] = len(idxs)

    return (
        voxel_names,
        np.asarray(coords, dtype=np.float64),
        np.asarray(centroids, dtype=np.float32),
        sizes,
    )


def occupancy_stats_from_counts(counts: np.ndarray | list[int]) -> dict[str, Any]:
    counts_arr = np.asarray(counts, dtype=np.float64)
    if len(counts_arr) == 0:
        return {
            "occupied_voxels": 0,
            "occupancy_entropy": 0.0,
            "effective_occupied_voxels": 0.0,
            "mean_voxel_size": 0.0,
            "max_voxel_size": 0,
            "max_voxel_fraction": 0.0,
        }
    total = float(np.sum(counts_arr))
    probs = counts_arr / total if total > 0 else counts_arr
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) else 0.0
    return {
        "occupied_voxels": int(len(counts_arr)),
        "occupancy_entropy": entropy,
        "effective_occupied_voxels": float(math.exp(entropy)),
        "mean_voxel_size": float(np.mean(counts_arr)),
        "max_voxel_size": int(np.max(counts_arr)),
        "max_voxel_fraction": float(np.max(counts_arr) / total) if total > 0 else 0.0,
    }


def centroids_from_coord_labels(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_sentences: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centroids for integer coordinate labels using vectorized sums."""
    unique, inverse, counts = np.unique(
        labels.astype(np.int32),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    valid = counts >= min_sentences
    if not np.any(valid):
        return unique[:0].astype(np.float64), embeddings[:0], counts[:0]

    sums = np.zeros((unique.shape[0], embeddings.shape[1]), dtype=np.float32)
    np.add.at(sums, inverse, embeddings)
    centroids = sums[valid] / counts[valid, None]
    return unique[valid].astype(np.float64), centroids, counts[valid]


def full_dim_centroid_baseline(
    records: list[SentenceRecord],
    embeddings: np.ndarray,
    min_sentences: int,
) -> dict[str, Any]:
    voxels, coords, centroids, sizes = centroids_from_string_labels(
        records, embeddings, min_sentences=min_sentences
    )
    metrics = pair_metrics(coords, centroids)
    metrics.update(
        {
            "method": "full_dim_voxel_centroid",
            "embedding_dims": int(embeddings.shape[1]),
            "min_sentences_per_voxel": min_sentences,
            "occupancy": occupancy_stats_from_counts(list(sizes.values())),
            "voxel_sizes": sizes,
            "voxels": voxels,
        }
    )
    return metrics


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation for Spearman correlation."""
    values = np.asarray(values)
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[sorter]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[sorter[start:end]] = avg_rank
        start = end

    return ranks


def truncation_validation(
    embeddings: np.ndarray,
    rng: np.random.Generator,
    dims: int,
    sample_pairs: int,
) -> dict[str, Any]:
    if embeddings.shape[1] <= dims:
        return {
            "method": "truncation_validation",
            "skipped": True,
            "reason": f"embedding dims <= truncation dims ({embeddings.shape[1]} <= {dims})",
        }

    n = embeddings.shape[0]
    total_possible = n * (n - 1) // 2
    sample_pairs = min(sample_pairs, total_possible)
    left = rng.integers(0, n, size=sample_pairs)
    right = rng.integers(0, n - 1, size=sample_pairs)
    right = np.where(right >= left, right + 1, right)

    full_norm = l2_normalize(embeddings)
    trunc_norm = l2_normalize(embeddings[:, :dims])
    full_cos = np.sum(full_norm[left] * full_norm[right], axis=1)
    trunc_cos = np.sum(trunc_norm[left] * trunc_norm[right], axis=1)

    pearson = float(np.corrcoef(full_cos, trunc_cos)[0, 1])
    full_ranks = rankdata(full_cos)
    trunc_ranks = rankdata(trunc_cos)
    spearman = float(np.corrcoef(full_ranks, trunc_ranks)[0, 1])
    abs_diff = np.abs(full_cos - trunc_cos)

    return {
        "method": "truncation_validation",
        "skipped": False,
        "full_dims": int(embeddings.shape[1]),
        "truncated_dims": int(dims),
        "sample_pairs": int(sample_pairs),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "mean_abs_cosine_delta": float(np.mean(abs_diff)),
        "p95_abs_cosine_delta": float(np.percentile(abs_diff, 95)),
        "max_abs_cosine_delta": float(np.max(abs_diff)),
    }


def compute_anchor_positions(
    anchor_norm: np.ndarray,
    method: str,
    random_state: int,
) -> tuple[np.ndarray, str]:
    distance_matrix = 1.0 - (anchor_norm @ anchor_norm.T)
    distance_matrix = np.maximum(distance_matrix, 0.0)
    np.fill_diagonal(distance_matrix, 0.0)

    if method == "sklearn":
        try:
            from sklearn.manifold import MDS

            mds = MDS(
                n_components=3,
                dissimilarity="precomputed",
                random_state=random_state,
                max_iter=500,
                n_init=4,
            )
            positions = mds.fit_transform(distance_matrix)
            used = "sklearn_mds"
        except ImportError:
            positions = classical_mds(distance_matrix, n_components=3)
            used = "classical_mds_fallback"
    elif method == "classical":
        positions = classical_mds(distance_matrix, n_components=3)
        used = "classical_mds"
    else:
        raise ValueError(f"Unsupported MDS method: {method}")

    positions = positions.astype(np.float32)
    positions -= positions.mean(axis=0, keepdims=True)
    max_radius = np.linalg.norm(positions, axis=1).max()
    if max_radius > 0:
        positions /= max_radius
    return positions, used


def classical_mds(distance_matrix: np.ndarray, n_components: int = 3) -> np.ndarray:
    n = distance_matrix.shape[0]
    d2 = distance_matrix * distance_matrix
    identity = np.eye(n)
    ones = np.ones((n, n), dtype=np.float64) / n
    centering = identity - ones
    gram = -0.5 * centering @ d2 @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = np.maximum(values[:n_components], 0.0)
    coords = vectors[:, :n_components] * np.sqrt(positive)
    if coords.shape[1] < n_components:
        coords = np.pad(coords, ((0, 0), (0, n_components - coords.shape[1])))
    return coords


def positions_to_l1_voxels(positions: np.ndarray, grid_size: float) -> np.ndarray:
    divisions = int(2.0 / grid_size)
    labels = ((positions + 1.0) / grid_size).astype(np.int32)
    return np.clip(labels, 0, divisions - 1)


def random_anchor_trial(
    embeddings: np.ndarray,
    embeddings_norm: np.ndarray,
    anchor_count: int,
    min_sentences: int,
    grid_size: float,
    mds_method: str,
    anchor_source: str,
    rng: np.random.Generator,
    random_state: int,
) -> dict[str, Any]:
    n_sentences = embeddings.shape[0]
    if anchor_source == "sentence" and anchor_count >= n_sentences:
        raise ValueError(
            f"anchor_count must be less than sentence count ({anchor_count} >= {n_sentences})"
        )

    if anchor_source == "sentence":
        anchor_indices = rng.choice(n_sentences, size=anchor_count, replace=False)
        anchor_norm = embeddings_norm[anchor_indices]
    elif anchor_source == "random-unit":
        random_vectors = rng.normal(size=(anchor_count, embeddings.shape[1])).astype(np.float32)
        anchor_norm = l2_normalize(random_vectors)
    else:
        raise ValueError(f"Unsupported anchor source: {anchor_source}")

    anchor_positions, used_mds = compute_anchor_positions(
        anchor_norm, method=mds_method, random_state=random_state
    )

    signatures = embeddings_norm @ anchor_norm.T
    weights = np.maximum(signatures, 0.0)
    denom = weights.sum(axis=1, keepdims=True)
    positions = np.divide(
        weights @ anchor_positions,
        denom,
        out=np.zeros((n_sentences, 3), dtype=np.float32),
        where=denom > 0,
    )

    labels = positions_to_l1_voxels(positions, grid_size=grid_size)
    coords, centroids, counts = centroids_from_coord_labels(
        labels, embeddings, min_sentences=min_sentences
    )
    metrics = pair_metrics(coords, centroids)
    metrics.update(
        {
            "anchor_count": int(anchor_count),
            "anchor_source": anchor_source,
            "mds_method": used_mds,
            "assigned_sentences": int(n_sentences),
            "min_sentences_per_voxel": int(min_sentences),
            "occupancy": occupancy_stats_from_counts(counts),
        }
    )
    return metrics


def random_anchor_baseline(
    embeddings: np.ndarray,
    real_slope: float,
    anchor_counts: list[int],
    trials: int,
    min_sentences: int,
    grid_size: float,
    mds_method: str,
    anchor_source: str,
    rng: np.random.Generator,
    seed: int,
    keep_trials: bool,
) -> dict[str, Any]:
    embeddings_norm = l2_normalize(embeddings)
    by_count: dict[str, Any] = {}

    for anchor_count in anchor_counts:
        trial_results: list[dict[str, Any]] = []
        print(
            f"Running random-anchor control: source={anchor_source}, "
            f"G={anchor_count}, trials={trials}"
        )
        for trial in range(trials):
            if trial == 0 or (trial + 1) % 10 == 0 or trial + 1 == trials:
                print(f"  trial {trial + 1}/{trials}")
            trial_results.append(
                random_anchor_trial(
                    embeddings=embeddings,
                    embeddings_norm=embeddings_norm,
                    anchor_count=anchor_count,
                    min_sentences=min_sentences,
                    grid_size=grid_size,
                    mds_method=mds_method,
                    anchor_source=anchor_source,
                    rng=rng,
                    random_state=seed + trial,
                )
            )

        slopes = np.asarray([r["slope"] for r in trial_results], dtype=np.float64)
        correlations = np.asarray([r["correlation"] for r in trial_results], dtype=np.float64)
        n_voxels = np.asarray([r["n_voxels"] for r in trial_results], dtype=np.float64)
        effective_voxels = np.asarray(
            [r["occupancy"]["effective_occupied_voxels"] for r in trial_results],
            dtype=np.float64,
        )
        max_fractions = np.asarray(
            [r["occupancy"]["max_voxel_fraction"] for r in trial_results],
            dtype=np.float64,
        )
        empirical_p = float(np.mean(slopes <= real_slope))
        by_count[str(anchor_count)] = {
            "anchor_count": int(anchor_count),
            "trials": int(trials),
            "real_full_dim_centroid_slope": float(real_slope),
            "empirical_p_random_as_steep_or_steeper": empirical_p,
            "slope_mean": float(np.mean(slopes)),
            "slope_std": float(np.std(slopes)),
            "slope_min": float(np.min(slopes)),
            "slope_p05": float(np.percentile(slopes, 5)),
            "slope_median": float(np.median(slopes)),
            "slope_p95": float(np.percentile(slopes, 95)),
            "slope_max": float(np.max(slopes)),
            "correlation_mean": float(np.mean(correlations)),
            "correlation_median": float(np.median(correlations)),
            "n_voxels_mean": float(np.mean(n_voxels)),
            "n_voxels_min": int(np.min(n_voxels)),
            "n_voxels_max": int(np.max(n_voxels)),
            "effective_occupied_voxels_mean": float(np.mean(effective_voxels)),
            "effective_occupied_voxels_median": float(np.median(effective_voxels)),
            "max_voxel_fraction_mean": float(np.mean(max_fractions)),
            "max_voxel_fraction_median": float(np.median(max_fractions)),
            "trial_results": trial_results if keep_trials else None,
        }

    return {
        "method": "random_sentence_anchor_psa_control",
        "anchor_source": anchor_source,
        "anchor_counts": anchor_counts,
        "trials_per_anchor_count": int(trials),
        "grid_size": float(grid_size),
        "min_sentences_per_voxel": int(min_sentences),
        "by_anchor_count": by_count,
    }


def result_path(run_dir: str, name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    out_dir = PSA_TEST_RUNS / run_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{ts}_{name}.json"


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "timestamp": result["timestamp"],
        "run_dir": result["run_dir"],
        "sentences": result["input"]["sentences"],
        "embedding_dims": result["input"]["embedding_dims"],
        "full_dim_centroid": {
            k: result["full_dim_centroid_baseline"][k]
            for k in (
                "n_voxels",
                "n_pairs",
                "correlation",
                "slope",
                "intercept",
                "mean_similarity",
            )
        },
    }
    summary["full_dim_centroid"]["occupancy"] = result["full_dim_centroid_baseline"].get(
        "occupancy"
    )
    if result.get("truncation_validation"):
        tv = result["truncation_validation"]
        summary["truncation_validation"] = {
            k: tv.get(k)
            for k in (
                "skipped",
                "sample_pairs",
                "pearson_r",
                "spearman_rho",
                "mean_abs_cosine_delta",
                "p95_abs_cosine_delta",
            )
        }
    if result.get("random_anchor_baseline"):
        summary["random_anchor_baseline"] = {}
        for count, data in result["random_anchor_baseline"]["by_anchor_count"].items():
            summary["random_anchor_baseline"][count] = {
                k: data[k]
                for k in (
                    "trials",
                    "real_full_dim_centroid_slope",
                    "empirical_p_random_as_steep_or_steeper",
                    "slope_mean",
                    "slope_p05",
                    "slope_median",
                    "slope_p95",
                    "correlation_median",
                    "n_voxels_mean",
                    "effective_occupied_voxels_median",
                    "max_voxel_fraction_median",
                )
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-D PSA centroid and random-anchor paper baselines.",
    )
    parser.add_argument("--run-dir", required=True, help="Run directory under psa_test_runs/")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-database", default=None)
    parser.add_argument("--node-label", default="SentenceNode")
    parser.add_argument(
        "--fetch-mode",
        choices=["driver", "cypher-shell"],
        default="driver",
        help="Use Python Neo4j driver or local cypher-shell export.",
    )
    parser.add_argument(
        "--cypher-shell-command",
        default="cypher-shell",
        help="cypher-shell command used with --fetch-mode cypher-shell.",
    )
    parser.add_argument(
        "--docker-container",
        default=None,
        help="Optional container name for docker exec cypher-shell, e.g. neo4j.",
    )
    parser.add_argument("--password", default=None)
    parser.add_argument("--password-env", default="NEO4J_PASSWORD")
    parser.add_argument(
        "--ssm-param",
        default=DEFAULT_SSM_PARAM_NAME,
        help="SSM parameter for Neo4j password; use empty string to disable.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-sentences-per-voxel", type=int, default=2)
    parser.add_argument("--grid-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-anchor-counts",
        type=int,
        nargs="+",
        default=[115],
        help="Guide-star counts to use for random-anchor controls.",
    )
    parser.add_argument(
        "--anchor-source",
        choices=["sentence", "random-unit"],
        default="sentence",
        help="Use random sentence embeddings or random unit vectors as anchors.",
    )
    parser.add_argument("--random-trials", type=int, default=25)
    parser.add_argument(
        "--mds-method",
        choices=["sklearn", "classical"],
        default="sklearn",
        help="Use sklearn MDS when available, or NumPy classical MDS.",
    )
    parser.add_argument(
        "--skip-random-anchors",
        action="store_true",
        help="Only run the full-D centroid and truncation checks.",
    )
    parser.add_argument(
        "--skip-truncation-validation",
        action="store_true",
        help="Skip first-N-dim cosine validation.",
    )
    parser.add_argument("--truncation-dims", type=int, default=200)
    parser.add_argument("--truncation-sample-pairs", type=int, default=10000)
    parser.add_argument(
        "--keep-random-trials",
        action="store_true",
        help="Include every random-anchor trial in the result JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit JSON output path. Defaults to psa_test_runs/<run>/results.",
    )
    args = parser.parse_args()

    if np is None:
        raise SystemExit("Missing Python package 'numpy'. Install it where this script runs.")

    if args.ssm_param == "":
        args.ssm_param = None

    started = time.time()
    rng = np.random.default_rng(args.seed)
    password = resolve_password(args)

    if args.fetch_mode == "cypher-shell":
        records, embeddings = fetch_sentence_embeddings_cypher_shell(args, password)
    else:
        driver = get_driver(args.neo4j_uri, args.neo4j_user, password)
        try:
            records, embeddings = fetch_sentence_embeddings(
                driver=driver,
                database=args.neo4j_database,
                node_label=args.node_label,
                limit=args.limit,
            )
        finally:
            driver.close()

    full_dim = full_dim_centroid_baseline(
        records=records,
        embeddings=embeddings,
        min_sentences=args.min_sentences_per_voxel,
    )
    print(
        "Full-D centroid baseline: "
        f"voxels={full_dim['n_voxels']} pairs={full_dim['n_pairs']} "
        f"r={full_dim['correlation']:.4f} slope={full_dim['slope']:.6f}"
    )

    truncation = None
    if not args.skip_truncation_validation:
        truncation = truncation_validation(
            embeddings=embeddings,
            rng=rng,
            dims=args.truncation_dims,
            sample_pairs=args.truncation_sample_pairs,
        )
        if not truncation.get("skipped"):
            print(
                "Truncation validation: "
                f"Pearson={truncation['pearson_r']:.4f} "
                f"Spearman={truncation['spearman_rho']:.4f}"
            )

    random_baseline = None
    if not args.skip_random_anchors:
        random_baseline = random_anchor_baseline(
            embeddings=embeddings,
            real_slope=full_dim["slope"],
            anchor_counts=args.random_anchor_counts,
            trials=args.random_trials,
            min_sentences=args.min_sentences_per_voxel,
            grid_size=args.grid_size,
            mds_method=args.mds_method,
            anchor_source=args.anchor_source,
            rng=rng,
            seed=args.seed,
            keep_trials=args.keep_random_trials,
        )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "input": {
            "neo4j_uri": args.neo4j_uri,
            "fetch_mode": args.fetch_mode,
            "node_label": args.node_label,
            "sentences": len(records),
            "embedding_dims": int(embeddings.shape[1]),
            "limit": args.limit,
            "seed": args.seed,
        },
        "full_dim_centroid_baseline": full_dim,
        "truncation_validation": truncation,
        "random_anchor_baseline": random_baseline,
        "runtime_seconds": round(time.time() - started, 3),
    }

    out_path = Path(args.output) if args.output else result_path(args.run_dir, "paper_baselines")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved: {out_path}")
    print(json.dumps(compact_summary(result), indent=2))


if __name__ == "__main__":
    main()
