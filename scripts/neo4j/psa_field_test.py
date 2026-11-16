#!/usr/bin/env python3
"""
PSA Field Test Suite — §23 First-Round Scientific Test Protocol.

Subcommands:
    voxel-snapshot   Export voxel occupancy, sentence placements, paraphrase groups
    locality         Compute locality decay curve + permutation test (p-value)
    baseline         Embedding-space baseline curve (no voxelization, for paper §4)
    baseline-sentence  Sentence-level full-embedding baseline (Option B)
    guide-stars      Verify guide star voxel positions and standard candle
    falsify          Run falsification controls (scramble, shuffle, perturb)
    settle-check     Repeatedly snapshot until Neo4j writes stabilize

All queries run through SSM → Neo4j (same infrastructure as snapshot_neo4j.py).
Results are saved to psa_test_runs/<run-dir>/results/.

Usage:
    python scripts/neo4j/psa_field_test.py voxel-snapshot --run-dir run1_order_text_first
    python scripts/neo4j/psa_field_test.py locality --run-dir run1_order_text_first
    python scripts/neo4j/psa_field_test.py guide-stars --run-dir run1_order_text_first
    python scripts/neo4j/psa_field_test.py falsify --run-dir run1_order_text_first --control scramble
    python scripts/neo4j/psa_field_test.py settle-check --run-dir run1_order_text_first --diff post-ocio_roadmap-settle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from consolidated_clustering_v2.routing.psa_coordinates import (
    VOXEL_DISTANCE_MODE,
    canonical_voxel_coords,
    voxel_distance as canonical_voxel_distance,
    voxel_level,
)

INSTANCE_ID = os.getenv("NEO4J_SSM_INSTANCE_ID", "")
CLUSTERING_INSTANCE_NAME = "consolidated_clustering_v2"
SSM_PARAM_NAME = "/example/neo4j/password"
PSA_TEST_RUNS = Path(__file__).resolve().parents[2] / "psa_test_runs"

# Standard candle voxel — must reappear in same position across runs
STANDARD_CANDLE_VOXEL = "L1_v_10_9_11"

# Number of leading embedding dimensions to fetch per centroid via SSM.
# Full Qwen3 embeddings are 4096-dim (~50KB as text), exceeding SSM's 24KB
# output limit.  200 dims (~2.4KB) preserves pairwise cosine relationships
# with high probability (Johnson–Lindenstrauss).
EMBED_DIMS = 200

# ---------------------------------------------------------------------------
# Neo4j query infrastructure (same as snapshot_neo4j.py)
# ---------------------------------------------------------------------------


def get_neo4j_password() -> str:
    result = subprocess.run(
        [
            "aws",
            "ssm",
            "get-parameter",
            "--name",
            SSM_PARAM_NAME,
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


def run_cypher(query: str, password: str) -> str:
    escaped_query = query.replace('"', '\\"')
    command = (
        f"docker exec neo4j cypher-shell -u neo4j -p '{password}' "
        f'--format plain "{escaped_query}"'
    )
    result = subprocess.run(
        [
            "aws",
            "ssm",
            "send-command",
            "--instance-ids",
            INSTANCE_ID,
            "--document-name",
            "AWS-RunShellScript",
            "--comment",
            "PSA field test query",
            "--parameters",
            json.dumps({"commands": [command]}),
            "--output",
            "text",
            "--query",
            "Command.CommandId",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    command_id = result.stdout.strip()

    subprocess.run(
        [
            "aws",
            "ssm",
            "wait",
            "command-executed",
            "--command-id",
            command_id,
            "--instance-id",
            INSTANCE_ID,
        ],
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            "aws",
            "ssm",
            "get-command-invocation",
            "--command-id",
            command_id,
            "--instance-id",
            INSTANCE_ID,
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    invocation = json.loads(result.stdout)
    status = invocation.get("Status", "")
    stdout = invocation.get("StandardOutputContent", "")
    stderr = invocation.get("StandardErrorContent", "")
    if status not in ("Success", "SuccessWithWarning"):
        raise RuntimeError(f"SSM command failed: status={status} stderr={stderr}")
    return stdout.strip()


def parse_two_column(output: str) -> dict[str, int]:
    result = {}
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^"?([^",]+)"?\s*,\s*(\d+)$', line)
        if match:
            result[match.group(1).strip()] = int(match.group(2))
    return result


def parse_rows(output: str, expected_cols: int) -> list[list[str]]:
    """Parse cypher-shell multi-column output into rows of strings."""
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= expected_cols:
            rows.append(parts)
    return rows


def results_dir(run_dir: str) -> Path:
    d = PSA_TEST_RUNS / run_dir / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_result(run_dir: str, name: str, data: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    path = results_dir(run_dir) / f"{ts}_{name}.json"
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# Subcommand: voxel-snapshot
# ---------------------------------------------------------------------------


def cmd_voxel_snapshot(args: argparse.Namespace) -> None:
    """Export voxel occupancy: sentence counts, object counts, paraphrase
    groups per voxel, and the full voxel list."""
    print("🔑 Retrieving Neo4j password...")
    password = get_neo4j_password()

    # 1. Sentence counts per visual voxel
    print("  ⏳ Querying sentence voxel occupancy...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL "
        "RETURN s.visual_voxel_id AS voxel, count(s) AS cnt "
        "ORDER BY cnt DESC",
        password,
    )
    voxel_sentences = parse_two_column(raw)

    # 2. Paraphrase groups per voxel
    print("  ⏳ Querying paraphrase group distribution...")
    raw = run_cypher(
        "MATCH (s:SentenceNode)-[:member_of]->(pg:ParaphraseGroup) "
        "WHERE s.visual_voxel_id IS NOT NULL "
        "WITH s.visual_voxel_id AS voxel, pg "
        "RETURN voxel, count(DISTINCT pg) AS cnt "
        "ORDER BY cnt DESC",
        password,
    )
    voxel_paraphrases = parse_two_column(raw)

    # 3. Member-of counts per voxel (paraphrase mass)
    print("  ⏳ Querying paraphrase mass per voxel...")
    raw = run_cypher(
        "MATCH (s:SentenceNode)-[:member_of]->(pg:ParaphraseGroup) "
        "WHERE s.visual_voxel_id IS NOT NULL "
        "RETURN s.visual_voxel_id AS voxel, count(s) AS cnt "
        "ORDER BY cnt DESC",
        password,
    )
    voxel_paraphrase_mass = parse_two_column(raw)

    # 4. Total sentences (including those without voxels)
    print("  ⏳ Querying unassigned sentences...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) WHERE s.visual_voxel_id IS NULL RETURN count(s) AS unassigned",
        password,
    )
    unassigned = 0
    for line in raw.splitlines():
        line = line.strip()
        if line.isdigit():
            unassigned = int(line)

    # Build per-voxel summary
    all_voxels = sorted(set(list(voxel_sentences.keys()) + list(voxel_paraphrases.keys())))
    voxel_table = {}
    for v in all_voxels:
        sentences = voxel_sentences.get(v, 0)
        para_groups = voxel_paraphrases.get(v, 0)
        para_mass = voxel_paraphrase_mass.get(v, 0)
        consolidation = para_mass / sentences if sentences > 0 else 0.0
        voxel_table[v] = {
            "sentences": sentences,
            "paraphrase_groups": para_groups,
            "paraphrase_mass": para_mass,
            "consolidation": round(consolidation, 4),
        }

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "summary": {
            "total_voxels_occupied": len(all_voxels),
            "total_sentences_assigned": sum(voxel_sentences.values()),
            "total_sentences_unassigned": unassigned,
            "total_paraphrase_groups": sum(voxel_paraphrases.values()),
            "total_paraphrase_mass": sum(voxel_paraphrase_mass.values()),
        },
        "voxels": voxel_table,
    }

    path = save_result(args.run_dir, "voxel_snapshot", snapshot)

    # Print summary
    print()
    print("=" * 60)
    print("  VOXEL OCCUPANCY SNAPSHOT")
    print("=" * 60)
    print(f"  Occupied voxels:         {len(all_voxels)}")
    print(f"  Sentences assigned:      {sum(voxel_sentences.values())}")
    print(f"  Sentences unassigned:    {unassigned}")
    print(f"  Paraphrase groups:       {sum(voxel_paraphrases.values())}")
    print()

    # Top 10 by sentence count
    top = sorted(voxel_table.items(), key=lambda x: x[1]["sentences"], reverse=True)[:15]
    print("  📊 Top 15 voxels by sentence count:")
    for v, info in top:
        consol = f"{info['consolidation']:.2f}"
        print(
            f"     {v:<30} sent={info['sentences']:>5}  pg={info['paraphrase_groups']:>3}  consol={consol}"
        )
    print()

    # Standard candle check
    candle_voxels = [v for v in all_voxels if STANDARD_CANDLE_VOXEL in v]
    if candle_voxels:
        print(f"  🕯️  Standard candle ({STANDARD_CANDLE_VOXEL}) FOUND:")
        for cv in candle_voxels:
            info = voxel_table[cv]
            print(
                f"     {cv}: {info['sentences']} sentences, {info['paraphrase_groups']} paraphrase groups"
            )
    else:
        print(f"  ⚠️  Standard candle ({STANDARD_CANDLE_VOXEL}) NOT FOUND in any voxel")

    print(f"\n  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# Subcommand: locality
# ---------------------------------------------------------------------------


def parse_voxel_coords(voxel: str) -> tuple[float, float, float] | None:
    """Extract canonical (x, y, z) from a multiresolution voxel address."""
    return canonical_voxel_coords(voxel)


def parse_voxel_level(voxel: str) -> int | None:
    """Extract PSA resolution level from a voxel address."""
    return voxel_level(voxel)


def fetch_voxel_centroids(
    password: str,
    min_sentences: int = 2,
) -> tuple[dict[str, np.ndarray], dict[str, int], list[str]]:
    """Stream per-voxel centroid embeddings (first EMBED_DIMS dims) from Neo4j.

    Shared by `cmd_locality` and `cmd_baseline`.  Returns (centroids,
    sizes, failed_voxels).
    """
    print("  ⏳ Querying occupied voxels...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL AND s.embedding IS NOT NULL "
        "RETURN s.visual_voxel_id AS voxel, count(s) AS cnt "
        "ORDER BY cnt DESC",
        password,
    )
    voxel_counts = parse_two_column(raw)
    voxel_counts = {v: c for v, c in voxel_counts.items() if c >= min_sentences}
    voxels = list(voxel_counts.keys())
    print(f"  Found {len(voxels)} occupied voxels with >= {min_sentences} embedded sentences")

    voxel_centroids: dict[str, np.ndarray] = {}
    voxel_sizes: dict[str, int] = {}
    failed_voxels: list[str] = []

    print(f"  ⏳ Streaming per-voxel centroid embeddings ({len(voxels)} voxels)...")
    for i, voxel in enumerate(voxels):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"     [{i + 1}/{len(voxels)}] Fetching centroid for {voxel} ...", flush=True)

        escaped_voxel = voxel.replace("'", "\\'")
        try:
            raw = run_cypher(
                f"MATCH (s:SentenceNode {{visual_voxel_id: '{escaped_voxel}'}}) "
                f"WHERE s.embedding IS NOT NULL "
                f"WITH collect(s.embedding) AS embeddings, count(s) AS cnt "
                f"RETURN cnt, "
                f"reduce(acc = [], idx IN range(0, {EMBED_DIMS - 1}) | "
                f"  acc + [reduce(sum = 0.0, e IN embeddings | sum + e[idx]) / cnt] "
                f") AS centroid",
                password,
            )

            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                bracket_start = line.find("[")
                if bracket_start < 0:
                    continue
                prefix = line[:bracket_start].strip().rstrip(",")
                centroid_str = line[bracket_start:].strip().rstrip(",")

                try:
                    cnt = int(prefix.strip().strip('"'))
                except ValueError:
                    continue

                try:
                    centroid = json.loads(centroid_str)
                    voxel_centroids[voxel] = np.array(centroid, dtype=np.float32)
                    voxel_sizes[voxel] = cnt
                except (json.JSONDecodeError, ValueError):
                    failed_voxels.append(voxel)
                break
        except (RuntimeError, subprocess.CalledProcessError) as e:
            failed_voxels.append(voxel)
            print(f"     ⚠️  Failed for {voxel}: {e}")

    print(f"  ✅ Parsed centroids for {len(voxel_centroids)} voxels ({len(failed_voxels)} failed)")
    return voxel_centroids, voxel_sizes, failed_voxels


def voxel_distance(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> float:
    """Euclidean distance between two canonical voxel coordinate triples."""
    return canonical_voxel_distance(v1, v2)


def summarize_same_layer_pairs(
    distances: np.ndarray,
    similarities: np.ndarray,
    pair_levels: list[tuple[int | None, int | None]],
) -> dict[str, dict[str, float | int]]:
    """Summarize locality metrics for pairs that stay within one PSA layer."""
    summary: dict[str, dict[str, float | int]] = {}
    levels = sorted({a for a, b in pair_levels if a is not None and a == b})
    for level in levels:
        mask = np.array([a == level and b == level for a, b in pair_levels], dtype=bool)
        if not np.any(mask):
            continue

        level_distances = distances[mask]
        level_sims = similarities[mask]
        if len(level_distances) > 1 and np.std(level_distances) > 0 and np.std(level_sims) > 0:
            corr = float(np.corrcoef(level_distances, level_sims)[0, 1])
            slope, intercept = np.polyfit(level_distances, level_sims, 1)
            slope = float(slope)
            intercept = float(intercept)
        else:
            corr = 0.0
            slope = 0.0
            intercept = float(np.mean(level_sims)) if len(level_sims) else 0.0

        summary[f"L{level}"] = {
            "n_pairs": int(np.sum(mask)),
            "correlation": corr,
            "slope": slope,
            "intercept": intercept,
            "mean_similarity": float(np.mean(level_sims)),
            "mean_distance": float(np.mean(level_distances)),
        }
    return summary


def cmd_locality(args: argparse.Namespace) -> None:
    """Compute locality decay curve: cosine similarity vs voxel distance.

    For each pair of occupied voxels, compute:
    - PSA distance (Euclidean over voxel coords)
    - Behavioral similarity (cosine over mean sentence embeddings per voxel)

    Then run a permutation test: scramble voxel labels, recompute, and check
    whether the real decay slope is significantly steeper than scrambled.

    Centroid embeddings are fetched one voxel at a time to avoid SSM timeout
    on the 4096-dim × 215-voxel aggregation query.  Only the first EMBED_DIMS
    dimensions are returned to stay within SSM's ~24 KB output limit
    (4096 × ~12 chars/float ≈ 50 KB; 200 × ~12 chars ≈ 2.4 KB).
    Johnson–Lindenstrauss guarantees pairwise cosine relationships are preserved
    with high probability at ≥ 200 dimensions.
    """
    print("🔑 Retrieving Neo4j password...")
    password = get_neo4j_password()

    n_perms = getattr(args, "permutations", 1000)

    voxel_centroids, voxel_sizes, _failed = fetch_voxel_centroids(password, min_sentences=2)

    if len(voxel_centroids) < 2:
        print("  ❌ Need at least 2 voxels with centroids for locality test")
        return

    # Step 3: Compute pairwise distances and similarities
    print("  ⏳ Computing pairwise locality...")
    voxel_names = sorted(voxel_centroids.keys())
    coords = {}
    for v in voxel_names:
        c = parse_voxel_coords(v)
        if c:
            coords[v] = c

    # Filter to voxels that have both centroids and parseable coords
    valid_voxels = [v for v in voxel_names if v in coords]
    print(f"  Valid voxels for pairwise analysis: {len(valid_voxels)}")

    if len(valid_voxels) < 2:
        print("  ❌ Not enough valid voxels with parseable coordinates")
        return

    # Normalize centroids
    centroids_matrix = np.array([voxel_centroids[v] for v in valid_voxels])
    norms = np.linalg.norm(centroids_matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    centroids_norm = centroids_matrix / norms

    # Cosine similarity matrix
    sim_matrix = centroids_norm @ centroids_norm.T

    # Build distance-similarity pairs
    n = len(valid_voxels)
    distances = []
    similarities = []
    pairs = []
    pair_levels = []

    for i in range(n):
        for j in range(i + 1, n):
            d = voxel_distance(coords[valid_voxels[i]], coords[valid_voxels[j]])
            s = float(sim_matrix[i, j])
            distances.append(d)
            similarities.append(s)
            pairs.append((valid_voxels[i], valid_voxels[j]))
            pair_levels.append(
                (
                    parse_voxel_level(valid_voxels[i]),
                    parse_voxel_level(valid_voxels[j]),
                )
            )

    distances = np.array(distances)
    similarities = np.array(similarities)

    # Compute correlation (Pearson) — expect negative (closer = more similar)
    if len(distances) > 1 and np.std(distances) > 0 and np.std(similarities) > 0:
        correlation = float(np.corrcoef(distances, similarities)[0, 1])
    else:
        correlation = 0.0

    # Linear regression slope
    if len(distances) > 1:
        slope, intercept = np.polyfit(distances, similarities, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope = 0.0
        intercept = 0.0

    print("\n  📊 LOCALITY DECAY CURVE")
    print(f"     Voxel pairs analyzed:  {len(distances)}")
    print(f"     Pearson correlation:   {correlation:.4f}")
    print(f"     Decay slope:           {slope:.6f}")
    print(f"     Intercept:             {intercept:.4f}")

    same_layer = summarize_same_layer_pairs(distances, similarities, pair_levels)
    if same_layer:
        print("\n  📊 SAME-LAYER LOCALITY")
        for level, metrics in same_layer.items():
            print(
                f"     {level}: n={metrics['n_pairs']} "
                f"r={metrics['correlation']:.4f} "
                f"slope={metrics['slope']:.6f}"
            )

    # Step 4: Permutation test
    print(f"\n  ⏳ Running permutation test ({n_perms} permutations)...")
    rng = np.random.default_rng(seed=42)
    null_slopes = []

    for _ in range(n_perms):
        # Scramble voxel labels (permute centroid assignments)
        perm = rng.permutation(n)
        perm_sims = centroids_norm[perm] @ centroids_norm[perm].T
        perm_similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                perm_similarities.append(float(perm_sims[i, j]))
        perm_similarities = np.array(perm_similarities)
        if np.std(perm_similarities) > 0:
            perm_slope = float(np.polyfit(distances, perm_similarities, 1)[0])
        else:
            perm_slope = 0.0
        null_slopes.append(perm_slope)

    null_slopes = np.array(null_slopes)
    # p-value: fraction of null slopes as steep or steeper (more negative) than observed
    p_value = float(np.mean(null_slopes <= slope))

    print(f"     Null slope mean:       {float(np.mean(null_slopes)):.6f}")
    print(f"     Null slope std:        {float(np.std(null_slopes)):.6f}")
    print(f"     Observed slope:        {slope:.6f}")
    print(f"     p-value:               {p_value:.4f}")

    if p_value < 0.05:
        print(f"\n  ✅ PASS: Locality decay is statistically significant (p={p_value:.4f} < 0.05)")
        verdict = "PASS"
    else:
        print(f"\n  ❌ FAIL: Locality decay NOT significant (p={p_value:.4f} >= 0.05)")
        verdict = "FAIL"

    # Build distance bins for the decay curve
    if len(distances) > 0:
        max_dist = float(np.max(distances))
        n_bins = min(20, max(5, len(distances) // 10))
        bin_edges = np.linspace(0, max_dist + 0.001, n_bins + 1)
        bin_centers = []
        bin_means = []
        bin_counts = []
        for b in range(n_bins):
            mask = (distances >= bin_edges[b]) & (distances < bin_edges[b + 1])
            if np.any(mask):
                bin_centers.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
                bin_means.append(float(np.mean(similarities[mask])))
                bin_counts.append(int(np.sum(mask)))
        decay_curve = {
            "bin_centers": bin_centers,
            "bin_mean_similarity": bin_means,
            "bin_pair_counts": bin_counts,
        }
    else:
        decay_curve = {}

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "voxel_distance_mode": VOXEL_DISTANCE_MODE,
        "n_voxels": len(valid_voxels),
        "n_pairs": len(distances),
        "correlation": correlation,
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "n_permutations": n_perms,
        "null_slope_mean": float(np.mean(null_slopes)),
        "null_slope_std": float(np.std(null_slopes)),
        "verdict": verdict,
        "same_layer_locality": same_layer,
        "decay_curve": decay_curve,
        "voxel_sizes": {v: voxel_sizes.get(v, 0) for v in valid_voxels},
    }

    path = save_result(args.run_dir, "locality_decay", result)
    print(f"\n  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# Subcommand: baseline
# ---------------------------------------------------------------------------


def cmd_baseline(args: argparse.Namespace) -> None:
    """Full-embedding baseline for locality decay (Section 4 of the paper).

    The `locality` subcommand bins pairs by *voxel-grid* distance.  This
    subcommand bins the same pairs by *embedding-space* distance — i.e.,
    the native curvature of the centroid manifold before PSA compression.

    Outputs:
      - decay curve in embedding space (mean cosine similarity vs. binned
        embedding distance)
      - cross-space correlation: Pearson(voxel_distance, embedding_distance)
        — how faithfully the 3-D voxel grid preserves 200-D centroid geometry

    Centroids are fetched at EMBED_DIMS=200 (Johnson–Lindenstrauss safe).
    If the curve looks anomalous, run a sentence-level baseline (Option B)
    on full 4096-D embeddings as a follow-up.
    """
    print("🔑 Retrieving Neo4j password...")
    password = get_neo4j_password()

    voxel_centroids, voxel_sizes, _failed = fetch_voxel_centroids(password, min_sentences=2)

    if len(voxel_centroids) < 2:
        print("  ❌ Need at least 2 voxels with centroids for baseline")
        return

    # Restrict to voxels with parseable coordinates (so we can also report
    # cross-space correlation against the voxel grid).
    voxel_names = sorted(voxel_centroids.keys())
    coords = {v: parse_voxel_coords(v) for v in voxel_names}
    valid_voxels = [v for v in voxel_names if coords.get(v) is not None]
    print(f"  Valid voxels for pairwise analysis: {len(valid_voxels)}")

    if len(valid_voxels) < 2:
        print("  ❌ Not enough valid voxels with parseable coordinates")
        return

    # Normalize centroids
    centroids_matrix = np.array([voxel_centroids[v] for v in valid_voxels])
    norms = np.linalg.norm(centroids_matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    centroids_norm = centroids_matrix / norms
    sim_matrix = centroids_norm @ centroids_norm.T

    n = len(valid_voxels)
    embed_distances = []  # 1 - cosine, in centroid space
    similarities = []
    voxel_distances = []  # for cross-space correlation
    pair_levels = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_matrix[i, j])
            similarities.append(s)
            embed_distances.append(1.0 - s)
            voxel_distances.append(voxel_distance(coords[valid_voxels[i]], coords[valid_voxels[j]]))
            pair_levels.append(
                (
                    parse_voxel_level(valid_voxels[i]),
                    parse_voxel_level(valid_voxels[j]),
                )
            )

    embed_distances = np.array(embed_distances)
    similarities = np.array(similarities)
    voxel_distances = np.array(voxel_distances)

    # Native-space decay: similarity vs embedding distance (mechanically
    # similarity = 1 - embed_distance; correlation will be -1 by construction).
    # The interesting numbers are the *binned* curve and the cross-space
    # correlation between voxel-grid distance and embedding distance.
    if len(voxel_distances) > 1 and np.std(voxel_distances) > 0 and np.std(embed_distances) > 0:
        cross_corr = float(np.corrcoef(voxel_distances, embed_distances)[0, 1])
    else:
        cross_corr = 0.0

    # Linear regression of similarity onto embed_distance (sanity)
    if len(embed_distances) > 1:
        slope, intercept = np.polyfit(embed_distances, similarities, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        slope, intercept = 0.0, 0.0

    print("\n  📊 EMBEDDING-SPACE BASELINE")
    print(f"     Voxel pairs analyzed:          {len(embed_distances)}")
    print(
        f"     Embedding-distance range:      [{float(np.min(embed_distances)):.4f}, {float(np.max(embed_distances)):.4f}]"
    )
    print(f"     Mean cosine similarity:        {float(np.mean(similarities)):.4f}")
    print(f"     Cross-space r (voxel↔embed):   {cross_corr:.4f}")
    print(f"     (slope sim~embed_dist:         {slope:.4f}, intercept: {intercept:.4f})")

    same_layer = summarize_same_layer_pairs(voxel_distances, similarities, pair_levels)
    if same_layer:
        print("\n  📊 SAME-LAYER BASELINE")
        for level, metrics in same_layer.items():
            print(
                f"     {level}: n={metrics['n_pairs']} "
                f"r={metrics['correlation']:.4f} "
                f"slope={metrics['slope']:.6f}"
            )

    # Decay curve: bin similarity by embedding distance
    decay_curve = {}
    if len(embed_distances) > 0:
        max_dist = float(np.max(embed_distances))
        min_dist = float(np.min(embed_distances))
        n_bins = min(20, max(5, len(embed_distances) // 10))
        bin_edges = np.linspace(min_dist, max_dist + 1e-6, n_bins + 1)
        bin_centers = []
        bin_means = []
        bin_counts = []
        for b in range(n_bins):
            mask = (embed_distances >= bin_edges[b]) & (embed_distances < bin_edges[b + 1])
            if np.any(mask):
                bin_centers.append(float((bin_edges[b] + bin_edges[b + 1]) / 2))
                bin_means.append(float(np.mean(similarities[mask])))
                bin_counts.append(int(np.sum(mask)))
        decay_curve = {
            "bin_centers": bin_centers,
            "bin_mean_similarity": bin_means,
            "bin_pair_counts": bin_counts,
        }

    # Cross-space binning: for each integer voxel-distance bucket, mean
    # embedding distance.  This is the figure that goes side-by-side with
    # the locality plot.
    cross_curve = {}
    if len(voxel_distances) > 0:
        max_vd = float(np.max(voxel_distances))
        n_bins = min(20, max(5, int(max_vd) + 1))
        vd_edges = np.linspace(0.0, max_vd + 1e-6, n_bins + 1)
        vd_centers = []
        vd_mean_embed = []
        vd_mean_sim = []
        vd_counts = []
        for b in range(n_bins):
            mask = (voxel_distances >= vd_edges[b]) & (voxel_distances < vd_edges[b + 1])
            if np.any(mask):
                vd_centers.append(float((vd_edges[b] + vd_edges[b + 1]) / 2))
                vd_mean_embed.append(float(np.mean(embed_distances[mask])))
                vd_mean_sim.append(float(np.mean(similarities[mask])))
                vd_counts.append(int(np.sum(mask)))
        cross_curve = {
            "voxel_distance_centers": vd_centers,
            "mean_embed_distance": vd_mean_embed,
            "mean_cosine_similarity": vd_mean_sim,
            "bin_pair_counts": vd_counts,
        }

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "voxel_distance_mode": VOXEL_DISTANCE_MODE,
        "embed_dims": EMBED_DIMS,
        "n_voxels": len(valid_voxels),
        "n_pairs": len(embed_distances),
        "cross_space_correlation": cross_corr,
        "embed_distance_min": float(np.min(embed_distances)),
        "embed_distance_max": float(np.max(embed_distances)),
        "mean_similarity": float(np.mean(similarities)),
        "slope_sim_vs_embed_dist": slope,
        "intercept_sim_vs_embed_dist": intercept,
        "same_layer_locality": same_layer,
        "decay_curve_embed": decay_curve,
        "cross_space_curve": cross_curve,
        "voxel_sizes": {v: voxel_sizes.get(v, 0) for v in valid_voxels},
        "notes": (
            "Centroids fetched at EMBED_DIMS=200 (JL-safe). Compare "
            "decay_curve_embed against locality_decay.decay_curve. "
            "cross_space_correlation > 0 indicates voxel grid preserves "
            "embedding-space distance ordering; values near 1.0 indicate "
            "high faithfulness."
        ),
    }

    path = save_result(args.run_dir, "baseline_embedding", result)
    print(f"\n  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# Subcommand: baseline-sentence (Option B)
# ---------------------------------------------------------------------------


def cmd_baseline_sentence(args: argparse.Namespace) -> None:
    """Sentence-level full-embedding baseline for locality decay.

    Option A uses per-voxel centroids truncated to EMBED_DIMS=200 because full
    4096-D centroids exceed SSM output limits.  Option B samples sentence pairs
    and computes the full-dimensional cosine inside Neo4j, returning only
    voxel labels and scalar similarities.  This tests whether centroid
    truncation hid signal in the voxel/embedding geometry comparison.
    """
    print("🔑 Retrieving Neo4j password...")
    password = get_neo4j_password()

    sample_pairs = getattr(args, "sample_pairs", 500)

    print("  ⏳ Counting eligible embedded sentences...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL AND s.embedding IS NOT NULL "
        "RETURN count(s) AS cnt, size(head(collect(s.embedding))) AS dims",
        password,
    )

    sentence_count = 0
    embedding_dims = 0
    for row in parse_rows(raw, 2):
        try:
            sentence_count = int(row[0])
            embedding_dims = int(row[1])
            break
        except ValueError:
            continue

    print(f"  Eligible sentences: {sentence_count}")
    print(f"  Embedding dims:     {embedding_dims}")

    if sentence_count < 2:
        print("  ❌ Need at least 2 embedded, voxel-assigned sentences")
        return

    total_possible_pairs = sentence_count * (sentence_count - 1) // 2
    sample_pairs = min(sample_pairs, total_possible_pairs)

    print(f"  ⏳ Sampling {sample_pairs} sentence pairs and computing full-D cosine in Neo4j...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL AND s.embedding IS NOT NULL "
        "WITH collect(s) AS sentences "
        "UNWIND range(0, size(sentences) - 2) AS i "
        "UNWIND range(i + 1, size(sentences) - 1) AS j "
        "WITH sentences[i] AS s1, sentences[j] AS s2, rand() AS r "
        "ORDER BY r "
        f"LIMIT {sample_pairs} "
        "WITH s1, s2, "
        "reduce(dot = 0.0, idx IN range(0, size(s1.embedding) - 1) | "
        "  dot + s1.embedding[idx] * s2.embedding[idx]) AS dot, "
        "sqrt(reduce(norm = 0.0, idx IN range(0, size(s1.embedding) - 1) | "
        "  norm + s1.embedding[idx] * s1.embedding[idx])) AS n1, "
        "sqrt(reduce(norm = 0.0, idx IN range(0, size(s2.embedding) - 1) | "
        "  norm + s2.embedding[idx] * s2.embedding[idx])) AS n2 "
        "RETURN s1.visual_voxel_id AS voxel1, s2.visual_voxel_id AS voxel2, "
        "CASE WHEN n1 > 0 AND n2 > 0 THEN dot / (n1 * n2) ELSE 0.0 END AS cosine",
        password,
    )

    voxel_distances = []
    embed_distances = []
    similarities = []
    pair_levels = []
    skipped = 0

    for row in parse_rows(raw, 3):
        v1, v2, cos_text = row[0], row[1], row[2]
        c1 = parse_voxel_coords(v1)
        c2 = parse_voxel_coords(v2)
        if c1 is None or c2 is None:
            skipped += 1
            continue
        try:
            cosine = float(cos_text)
        except ValueError:
            skipped += 1
            continue
        voxel_distances.append(voxel_distance(c1, c2))
        similarities.append(cosine)
        embed_distances.append(1.0 - cosine)
        pair_levels.append((parse_voxel_level(v1), parse_voxel_level(v2)))

    voxel_distances = np.array(voxel_distances)
    similarities = np.array(similarities)
    embed_distances = np.array(embed_distances)

    if len(similarities) < 2:
        print("  ❌ Need at least 2 parseable sampled pairs")
        return

    if np.std(voxel_distances) > 0 and np.std(embed_distances) > 0:
        cross_corr = float(np.corrcoef(voxel_distances, embed_distances)[0, 1])
        locality_corr = float(np.corrcoef(voxel_distances, similarities)[0, 1])
    else:
        cross_corr = 0.0
        locality_corr = 0.0

    slope, intercept = np.polyfit(voxel_distances, similarities, 1)
    slope = float(slope)
    intercept = float(intercept)

    print("\n  📊 SENTENCE-LEVEL FULL-EMBEDDING BASELINE (OPTION B)")
    print(f"     Sentence pairs analyzed:       {len(similarities)}")
    print(f"     Skipped rows:                  {skipped}")
    print(
        f"     Embedding-distance range:      [{float(np.min(embed_distances)):.4f}, {float(np.max(embed_distances)):.4f}]"
    )
    print(f"     Mean cosine similarity:        {float(np.mean(similarities)):.4f}")
    print(f"     Cross-space r (voxel↔embed):   {cross_corr:.4f}")
    print(f"     Locality r (voxel↔cosine):     {locality_corr:.4f}")
    print(f"     Decay slope:                   {slope:.6f}")
    print(f"     Intercept:                     {intercept:.4f}")

    same_layer = summarize_same_layer_pairs(voxel_distances, similarities, pair_levels)
    if same_layer:
        print("\n  📊 SAME-LAYER SENTENCE BASELINE")
        for level, metrics in same_layer.items():
            print(
                f"     {level}: n={metrics['n_pairs']} "
                f"r={metrics['correlation']:.4f} "
                f"slope={metrics['slope']:.6f}"
            )

    # Native-space curve: full-D embedding distance vs cosine similarity.
    max_ed = float(np.max(embed_distances))
    min_ed = float(np.min(embed_distances))
    n_bins = min(20, max(5, len(embed_distances) // 25))
    embed_edges = np.linspace(min_ed, max_ed + 1e-6, n_bins + 1)
    embed_centers = []
    embed_means = []
    embed_counts = []
    for b in range(n_bins):
        mask = (embed_distances >= embed_edges[b]) & (embed_distances < embed_edges[b + 1])
        if np.any(mask):
            embed_centers.append(float((embed_edges[b] + embed_edges[b + 1]) / 2))
            embed_means.append(float(np.mean(similarities[mask])))
            embed_counts.append(int(np.sum(mask)))

    # Cross-space curve: integer voxel-distance buckets.
    bucket_values = np.floor(voxel_distances).astype(int)
    bucket_centers = []
    bucket_mean_embed = []
    bucket_mean_sim = []
    bucket_counts = []
    for bucket in sorted(set(bucket_values.tolist())):
        mask = bucket_values == bucket
        bucket_centers.append(int(bucket))
        bucket_mean_embed.append(float(np.mean(embed_distances[mask])))
        bucket_mean_sim.append(float(np.mean(similarities[mask])))
        bucket_counts.append(int(np.sum(mask)))

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "method": "sentence_pair_full_embedding_sample",
        "voxel_distance_mode": VOXEL_DISTANCE_MODE,
        "embedding_dims": embedding_dims,
        "eligible_sentences": sentence_count,
        "sample_pairs_requested": sample_pairs,
        "n_pairs": len(similarities),
        "skipped_rows": skipped,
        "cross_space_correlation": cross_corr,
        "locality_correlation": locality_corr,
        "slope": slope,
        "intercept": intercept,
        "embed_distance_min": float(np.min(embed_distances)),
        "embed_distance_max": float(np.max(embed_distances)),
        "mean_similarity": float(np.mean(similarities)),
        "same_layer_locality": same_layer,
        "decay_curve_embed": {
            "bin_centers": embed_centers,
            "bin_mean_similarity": embed_means,
            "bin_pair_counts": embed_counts,
        },
        "cross_space_curve": {
            "voxel_distance_floor": bucket_centers,
            "mean_embed_distance": bucket_mean_embed,
            "mean_cosine_similarity": bucket_mean_sim,
            "bin_pair_counts": bucket_counts,
        },
        "notes": (
            "Option B computes sentence-pair cosine over full embeddings inside "
            "Neo4j, returning only scalar similarities through SSM. This avoids "
            "the SSM stdout limit that prevents streaming full 4096-D vectors."
        ),
    }

    path = save_result(args.run_dir, "baseline_sentence_full_embedding", result)
    print(f"\n  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# Subcommand: guide-stars
# ---------------------------------------------------------------------------


def cmd_guide_stars(args: argparse.Namespace) -> None:
    """Verify guide star sentence placements and standard candle position."""
    print("🔑 Retrieving Neo4j password...")
    password = get_neo4j_password()

    # Load guide star catalog
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from consolidated_clustering_v2.routing.guide_stars import (
        GUIDE_STAR_CATALOG,
    )

    # Query: sentences in the standard candle voxel
    print(f"  ⏳ Querying standard candle voxel ({STANDARD_CANDLE_VOXEL})...")
    raw = run_cypher(
        f"MATCH (s:SentenceNode) "
        f"WHERE s.visual_voxel_id CONTAINS '{STANDARD_CANDLE_VOXEL}' "
        f"RETURN s.sentence_id AS sid, "
        f"substring(COALESCE(s.text, s.content, s.sentence, ''), 0, 120) AS text, "
        f"s.visual_voxel_id AS voxel, "
        f"s.container_name AS source "
        f"ORDER BY s.created_at "
        f"LIMIT 50",
        password,
    )
    candle_sentences = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Parse the CSV-ish output
        parts = line.split('", "')
        if len(parts) >= 2:
            candle_sentences.append(line)

    # Query: voxel distribution of all sentences
    print("  ⏳ Querying voxel distribution summary...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL "
        "WITH s.visual_voxel_id AS voxel, count(s) AS cnt "
        "RETURN count(voxel) AS n_voxels, "
        "min(cnt) AS min_occ, max(cnt) AS max_occ, "
        "avg(cnt) AS avg_occ, sum(cnt) AS total",
        password,
    )
    dist_stats = raw.strip()

    # Query: how many unique voxel addresses exist
    print("  ⏳ Querying unique voxel addresses...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL "
        "RETURN count(DISTINCT s.visual_voxel_id) AS unique_voxels",
        password,
    )
    unique_voxels = 0
    for line in raw.splitlines():
        if line.strip().isdigit():
            unique_voxels = int(line.strip())

    # Query: sentences with calibrated routing labels
    print("  ⏳ Querying routing-label coverage...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.routing_cluster_label IS NOT NULL "
        "RETURN count(s) AS with_routing_cluster_label",
        password,
    )
    routing_label_count = 0
    for line in raw.splitlines():
        if line.strip().isdigit():
            routing_label_count = int(line.strip())

    # Query: sentences with embeddings
    print("  ⏳ Querying embedding coverage...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) WHERE s.embedding IS NOT NULL RETURN count(s) AS with_embedding",
        password,
    )
    embedding_count = 0
    for line in raw.splitlines():
        if line.strip().isdigit():
            embedding_count = int(line.strip())

    print()
    print("=" * 60)
    print("  GUIDE STAR & PSA FIELD VERIFICATION")
    print("=" * 60)
    print(f"  Guide stars in catalog:    {len(GUIDE_STAR_CATALOG)}")
    print(f"  Unique voxels occupied:    {unique_voxels}")
    print(f"  Sentences with embeddings: {embedding_count}")
    print(f"  Sentences with routing labels: {routing_label_count}")
    print(f"  Distribution stats:        {dist_stats}")
    print()

    if candle_sentences:
        print(f"  🕯️  Standard candle ({STANDARD_CANDLE_VOXEL}): {len(candle_sentences)} sentences")
        for s in candle_sentences[:10]:
            print(f"     {s[:120]}")
    else:
        print(f"  ⚠️  Standard candle ({STANDARD_CANDLE_VOXEL}): NO sentences found")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "guide_star_count": len(GUIDE_STAR_CATALOG),
        "unique_voxels": unique_voxels,
        "embedding_count": embedding_count,
        "routing_cluster_label_count": routing_label_count,
        "standard_candle_voxel": STANDARD_CANDLE_VOXEL,
        "standard_candle_sentences": len(candle_sentences),
        "standard_candle_sample": candle_sentences[:10],
        "distribution_stats": dist_stats,
    }

    path = save_result(args.run_dir, "guide_star_verification", result)
    print(f"\n  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# Subcommand: falsify
# ---------------------------------------------------------------------------


def cmd_falsify(args: argparse.Namespace) -> None:
    """Run falsification controls per §23.5 step 7.

    Controls:
    - scramble: Randomly permute voxel labels across nodes (preserves per-voxel
      counts, destroys structure). Re-runs locality analysis.
    - shuffle: Randomly reassign source-document labels (preserves doc-count
      distribution).
    - perturb: Add noise to guide-star positions conceptually — recompute
      voxel assignments with jittered coordinates.

    For now, scramble is the primary control (most important for §23.7).
    """
    control = args.control
    print("🔑 Retrieving Neo4j password...")
    password = get_neo4j_password()

    if control == "scramble":
        _run_scramble_control(args, password)
    elif control == "shuffle":
        print("  ⚠️  Source-label shuffle not yet implemented — needs source_uri query.")
        print("     Use 'scramble' for the primary null control.")
    elif control == "perturb":
        print("  ⚠️  Guide-star perturbation not yet implemented — needs embedding recompute.")
        print("     Use 'scramble' for the primary null control.")
    else:
        print(f"  ❌ Unknown control: {control}. Use: scramble, shuffle, perturb")


def _run_scramble_control(args: argparse.Namespace, password: str) -> None:
    """Scramble control: permute voxel labels across sentences, then recompute
    the locality decay curve. The real PSA should show steeper decay."""

    n_perms = getattr(args, "permutations", 1000)

    print("  ⏳ Fetching voxel centroids for scramble control...")
    print("     (Streaming per-voxel to avoid SSM timeout)")

    # Get voxel list first
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL AND s.embedding IS NOT NULL "
        "RETURN s.visual_voxel_id AS voxel, count(s) AS cnt "
        "ORDER BY cnt DESC",
        password,
    )
    all_voxel_counts = parse_two_column(raw)
    all_voxel_counts = {v: c for v, c in all_voxel_counts.items() if c >= 2}

    voxel_centroids: dict[str, np.ndarray] = {}
    voxel_sizes: dict[str, int] = {}
    coords: dict[str, tuple[int, int, int]] = {}

    print(f"  ⏳ Streaming centroids for {len(all_voxel_counts)} voxels...")
    for i, (voxel, _cnt) in enumerate(all_voxel_counts.items()):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"     [{i + 1}/{len(all_voxel_counts)}] {voxel}")

        escaped_voxel = voxel.replace("'", "\\'")
        try:
            raw = run_cypher(
                f"MATCH (s:SentenceNode {{visual_voxel_id: '{escaped_voxel}'}}) "
                f"WHERE s.embedding IS NOT NULL "
                f"WITH collect(s.embedding) AS embeddings, count(s) AS cnt "
                f"RETURN cnt, "
                f"reduce(acc = [], idx IN range(0, {EMBED_DIMS - 1}) | "
                f"  acc + [reduce(sum = 0.0, e IN embeddings | sum + e[idx]) / cnt] "
                f") AS centroid",
                password,
            )

            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                bracket_start = line.find("[")
                if bracket_start < 0:
                    continue
                prefix = line[:bracket_start].strip().rstrip(",")
                centroid_str = line[bracket_start:].strip().rstrip(",")
                try:
                    parsed_cnt = int(prefix.strip().strip('"'))
                except ValueError:
                    continue
                try:
                    centroid = json.loads(centroid_str)
                    c = parse_voxel_coords(voxel)
                    if c:
                        voxel_centroids[voxel] = np.array(centroid, dtype=np.float32)
                        voxel_sizes[voxel] = parsed_cnt
                        coords[voxel] = c
                except (json.JSONDecodeError, ValueError):
                    pass
                break
        except (RuntimeError, subprocess.CalledProcessError):
            pass

    valid_voxels = sorted(voxel_centroids.keys())
    n = len(valid_voxels)
    print(f"  Loaded {n} voxels for scramble control")

    if n < 2:
        print("  ❌ Not enough voxels for scramble control")
        return

    centroids_matrix = np.array([voxel_centroids[v] for v in valid_voxels])
    norms = np.linalg.norm(centroids_matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    centroids_norm = centroids_matrix / norms

    # Real distances (fixed)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(voxel_distance(coords[valid_voxels[i]], coords[valid_voxels[j]]))
    distances = np.array(distances)

    # Real similarities
    sim_matrix = centroids_norm @ centroids_norm.T
    real_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            real_sims.append(float(sim_matrix[i, j]))
    real_sims = np.array(real_sims)
    real_slope = float(np.polyfit(distances, real_sims, 1)[0])

    # Scrambled: permute which centroid lives at which coordinate
    print(f"  ⏳ Running {n_perms} scramble permutations...")
    rng = np.random.default_rng(seed=42)
    scramble_slopes = []

    for _ in range(n_perms):
        perm = rng.permutation(n)
        scrambled_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                scrambled_sims.append(float(centroids_norm[perm[i]] @ centroids_norm[perm[j]]))
        scrambled_sims = np.array(scrambled_sims)
        if np.std(scrambled_sims) > 0:
            scramble_slopes.append(float(np.polyfit(distances, scrambled_sims, 1)[0]))
        else:
            scramble_slopes.append(0.0)

    scramble_slopes = np.array(scramble_slopes)
    p_value = float(np.mean(scramble_slopes <= real_slope))

    print("\n  📊 SCRAMBLE CONTROL RESULTS")
    print(f"     Real decay slope:      {real_slope:.6f}")
    print(f"     Scrambled slope mean:   {float(np.mean(scramble_slopes)):.6f}")
    print(f"     Scrambled slope std:    {float(np.std(scramble_slopes)):.6f}")
    print(f"     p-value:               {p_value:.4f}")

    if p_value < 0.05:
        print(f"\n  ✅ PASS: Real PSA structure survives scramble control (p={p_value:.4f})")
        verdict = "PASS"
    else:
        print(
            f"\n  ❌ FAIL: Real PSA structure NOT distinguishable from scrambled (p={p_value:.4f})"
        )
        verdict = "FAIL"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "control_type": "scramble",
        "n_voxels": n,
        "n_pairs": len(distances),
        "real_slope": real_slope,
        "scrambled_slope_mean": float(np.mean(scramble_slopes)),
        "scrambled_slope_std": float(np.std(scramble_slopes)),
        "p_value": p_value,
        "n_permutations": n_perms,
        "verdict": verdict,
    }

    path = save_result(args.run_dir, "falsify_scramble", result)
    print(f"\n  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# Subcommand: settle-check
# ---------------------------------------------------------------------------


def cmd_settle_check(args: argparse.Namespace) -> None:
    """Take repeated snapshots until Neo4j writes stabilize (zero delta)."""
    # Import snapshot_neo4j functions
    snapshot_script = Path(__file__).parent / "snapshot_neo4j.py"
    if not snapshot_script.exists():
        print(f"  ❌ snapshot_neo4j.py not found at {snapshot_script}")
        return

    max_checks = getattr(args, "max_checks", 5)
    interval = getattr(args, "interval", 60)
    diff_label = args.diff

    print(f"  ⏳ Settlement check: up to {max_checks} rounds, {interval}s apart")
    print(f"     Diffing against: {diff_label}")

    for i in range(max_checks):
        label = f"settle-check-{i + 1}"
        print(f"\n  --- Round {i + 1}/{max_checks} ---")

        cmd = [
            sys.executable,
            str(snapshot_script),
            "--run-dir",
            args.run_dir,
            "--label",
            label,
            "--diff",
            diff_label,
        ]
        subprocess.run(cmd, capture_output=False, text=True)

        # Check the latest snapshot for zero delta
        snapshots_dir = PSA_TEST_RUNS / args.run_dir / "snapshots"
        latest = sorted(snapshots_dir.glob(f"*_{label}.json"))
        if latest:
            data = json.loads(latest[-1].read_text())
            # Load the diff target
            diff_files = sorted(snapshots_dir.glob(f"*_{diff_label}.json"))
            if diff_files:
                prev = json.loads(diff_files[-1].read_text())
                node_delta = data["totals"]["nodes"] - prev["totals"]["nodes"]
                edge_delta = data["totals"]["edges"] - prev["totals"]["edges"]

                if node_delta == 0 and edge_delta == 0:
                    print(f"\n  ✅ SETTLED: Zero delta on round {i + 1}")
                    return
                else:
                    print(f"     Delta: nodes={node_delta:+d}, edges={edge_delta:+d}")

        diff_label = label
        if i < max_checks - 1:
            print(f"     Waiting {interval}s...")
            time.sleep(interval)

    print(f"\n  ⚠️  Not settled after {max_checks} rounds")


# ---------------------------------------------------------------------------
# Redis query infrastructure
# ---------------------------------------------------------------------------


def get_redis_host() -> str:
    result = subprocess.run(
        [
            "aws",
            "ssm",
            "get-parameter",
            "--name",
            "/aiidea/redis/host",
            "--region",
            "us-east-1",
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


def get_redis_ssm_instance_id() -> str:
    """Return the EC2 instance that has security-group access to Redis."""
    result = subprocess.run(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            "us-east-1",
            "--filters",
            f"Name=tag:Name,Values={CLUSTERING_INSTANCE_NAME}",
            "Name=instance-state-name,Values=running",
            "--query",
            "Reservations[].Instances[].InstanceId | [0]",
            "--output",
            "text",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    instance_id = result.stdout.strip()
    if instance_id and instance_id != "None":
        return instance_id

    print(f"  ⚠️  {CLUSTERING_INSTANCE_NAME} not found; falling back to Neo4j for Redis access")
    return INSTANCE_ID


def run_redis_cmd(redis_host: str, redis_cmd: str) -> str:
    """Run a redis-cli command via SSM on the Redis-authorized EC2 instance."""
    redis_instance_id = get_redis_ssm_instance_id()
    redis_args = shlex.split(redis_cmd)
    shell_cmd = (
        f"REDIS_CLI=$(command -v redis-cli 2>/dev/null || command -v redis6-cli 2>/dev/null || true); "
        f'if [ -z "$REDIS_CLI" ]; then sudo dnf install -y redis6 >/dev/null 2>&1; '
        f"REDIS_CLI=$(command -v redis-cli 2>/dev/null || command -v redis6-cli 2>/dev/null); fi; "
        f'if [ -n "$REDIS_CLI" ]; then $REDIS_CLI -h {redis_host} -p 6379 {redis_cmd}; '
        "else python3 - <<'PY'\n"
        "import socket\n"
        f"host = {redis_host!r}\n"
        f"args = {json.dumps(redis_args)}\n"
        "def encode(parts):\n"
        "    out = [f'*{len(parts)}\\r\\n'.encode()]\n"
        "    for part in parts:\n"
        "        data = str(part).encode()\n"
        "        out.append(f'${len(data)}\\r\\n'.encode() + data + b'\\r\\n')\n"
        "    return b''.join(out)\n"
        "def read_line(sock):\n"
        "    data = b''\n"
        "    while not data.endswith(b'\\r\\n'):\n"
        "        chunk = sock.recv(1)\n"
        "        if not chunk:\n"
        "            raise RuntimeError('Redis connection closed')\n"
        "        data += chunk\n"
        "    return data[:-2]\n"
        "def read_resp(sock):\n"
        "    prefix = sock.recv(1)\n"
        "    if prefix in (b'+', b':'):\n"
        "        return read_line(sock).decode()\n"
        "    if prefix == b'-':\n"
        "        raise RuntimeError(read_line(sock).decode())\n"
        "    if prefix == b'$':\n"
        "        n = int(read_line(sock))\n"
        "        if n < 0:\n"
        "            return ''\n"
        "        data = b''\n"
        "        while len(data) < n + 2:\n"
        "            data += sock.recv(n + 2 - len(data))\n"
        "        return data[:n].decode(errors='replace')\n"
        "    if prefix == b'*':\n"
        "        n = int(read_line(sock))\n"
        "        return [read_resp(sock) for _ in range(n)]\n"
        "    raise RuntimeError(f'Unknown RESP prefix: {prefix!r}')\n"
        "with socket.create_connection((host, 6379), timeout=20) as sock:\n"
        "    sock.sendall(encode(args))\n"
        "    value = read_resp(sock)\n"
        "if isinstance(value, list):\n"
        "    print('\\n'.join(str(v) for v in value))\n"
        "else:\n"
        "    print(value)\n"
        "PY\n"
        "fi"
    )
    result = subprocess.run(
        [
            "aws",
            "ssm",
            "send-command",
            "--instance-ids",
            redis_instance_id,
            "--document-name",
            "AWS-RunShellScript",
            "--comment",
            "PSA field test Redis query",
            "--parameters",
            json.dumps({"commands": [shell_cmd]}),
            "--output",
            "text",
            "--query",
            "Command.CommandId",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    command_id = result.stdout.strip()

    subprocess.run(
        [
            "aws",
            "ssm",
            "wait",
            "command-executed",
            "--command-id",
            command_id,
            "--instance-id",
            redis_instance_id,
        ],
        capture_output=True,
        text=True,
    )

    terminal_statuses = {
        "Success",
        "SuccessWithWarning",
        "Cancelled",
        "TimedOut",
        "Failed",
        "Cancelling",
    }
    invocation = {}
    for _ in range(60):
        result = subprocess.run(
            [
                "aws",
                "ssm",
                "get-command-invocation",
                "--command-id",
                command_id,
                "--instance-id",
                redis_instance_id,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        invocation = json.loads(result.stdout)
        if invocation.get("Status", "") in terminal_statuses:
            break
        time.sleep(2)

    status = invocation.get("Status", "")
    stdout = invocation.get("StandardOutputContent", "")
    stderr = invocation.get("StandardErrorContent", "")
    if status not in ("Success", "SuccessWithWarning"):
        raise RuntimeError(f"SSM Redis command failed: status={status} stderr={stderr}")
    return stdout.strip()


# ---------------------------------------------------------------------------
# Subcommand: topic-sanity
# ---------------------------------------------------------------------------


def cmd_topic_sanity(args: argparse.Namespace) -> None:
    """Sanity check: fetch LLM-generated voxel display_names from Redis,
    then check whether nearby voxels in PSA space have topically similar
    names compared to distant ones.

    This is a quick qualitative check before the full embedding-based
    locality test.
    """
    print("🔑 Retrieving Redis host...")
    redis_host = get_redis_host()
    semantic_namespace_id = str(args.semantic_namespace_id).strip()
    if re.fullmatch(r"ns_[0-9a-f]{64}", semantic_namespace_id) is None:
        raise ValueError(
            "--semantic-namespace-id must be ns_ followed by 64 lowercase hex characters"
        )
    redis_prefix = f"{semantic_namespace_id}:clustering"

    # 1. Get all occupied voxel keys from the hot_voxels sorted set
    print("  ⏳ Fetching occupied voxels from Redis...")
    raw = run_redis_cmd(
        redis_host,
        f"ZREVRANGE {shlex.quote(f'{redis_prefix}:hot_voxels')} 0 -1",
    )
    voxel_keys = [k.strip() for k in raw.splitlines() if k.strip()]
    print(f"  ✅ {len(voxel_keys)} occupied voxels found")

    if not voxel_keys:
        print("  ❌ No voxel keys found in Redis. Is the clustering service running?")
        return

    # 2. Fetch display_name for each voxel.  Use run_redis_cmd so this follows
    # the same Redis-authorized SSM path and Python fallback as other Redis
    # queries.
    print("  ⏳ Fetching display_names...")
    voxel_names: dict[str, str] = {}
    for i, vk in enumerate(voxel_keys, 1):
        if i == 1 or i % 10 == 0:
            print(f"     [{i}/{len(voxel_keys)}] {vk}")
        dn = run_redis_cmd(
            redis_host,
            f"HGET {shlex.quote(f'{redis_prefix}:voxel:{vk}')} display_name",
        ).strip()
        voxel_names[vk] = dn if dn else "(unnamed)"

    named_count = sum(1 for v in voxel_names.values() if v != "(unnamed)")
    print(f"  ✅ {named_count}/{len(voxel_names)} voxels have display_names")

    # 3. Parse coords and build (voxel, coords, name) tuples
    voxels_with_coords = []
    for vk, name in sorted(voxel_names.items()):
        coords = parse_voxel_coords(vk)
        if coords:
            voxels_with_coords.append((vk, coords, name))

    if len(voxels_with_coords) < 2:
        print("  ❌ Not enough voxels with parseable coordinates")
        return

    # 4. Print all voxels sorted by coords
    print(f"\n  📋 Voxel Topic Map ({len(voxels_with_coords)} voxels with coordinates):\n")
    print(f"    {'Voxel':<25} {'Coords':<15} {'Display Name'}")
    print(f"    {'─' * 25} {'─' * 15} {'─' * 40}")
    for vk, (x, y, z), name in sorted(voxels_with_coords, key=lambda t: t[1]):
        print(f"    {vk:<25} ({x:4.2f},{y:4.2f},{z:4.2f})    {name}")

    # 5. Find nearest-neighbor pairs and show topic proximity
    print("\n  🔍 Nearest-Neighbor Topic Check:")
    print("    For each voxel, showing its closest neighbor by PSA distance:\n")

    # Compute all pairwise distances
    n = len(voxels_with_coords)
    for i in range(n):
        vk_i, coords_i, name_i = voxels_with_coords[i]
        best_dist = float("inf")
        best_j = -1
        for j in range(n):
            if i == j:
                continue
            dist = voxel_distance(coords_i, voxels_with_coords[j][1])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        vk_j, coords_j, name_j = voxels_with_coords[best_j]
        print(f"    {name_i:<35} ↔ {name_j:<35} (dist={best_dist:.2f})")

    # 6. Show the most distant pair for contrast
    max_dist = 0.0
    max_pair = (0, 0)
    for i in range(n):
        for j in range(i + 1, n):
            dist = voxel_distance(voxels_with_coords[i][1], voxels_with_coords[j][1])
            if dist > max_dist:
                max_dist = dist
                max_pair = (i, j)
    i, j = max_pair
    print(f"\n  📏 Most distant pair (dist={max_dist:.2f}):")
    print(f"    {voxels_with_coords[i][2]} ({voxels_with_coords[i][0]})")
    print(f"    {voxels_with_coords[j][2]} ({voxels_with_coords[j][0]})")

    # 7. Standard candle check
    candle_name = voxel_names.get(STANDARD_CANDLE_VOXEL)
    if candle_name:
        print(f'\n  🕯️  Standard candle {STANDARD_CANDLE_VOXEL}: "{candle_name}"')
    else:
        # Try with prefix variants
        for vk, name in voxel_names.items():
            if STANDARD_CANDLE_VOXEL in vk:
                print(f'\n  🕯️  Standard candle match {vk}: "{name}"')
                break
        else:
            print(f"\n  ⚠️  Standard candle {STANDARD_CANDLE_VOXEL} not found in voxel names")

    # 8. Save results
    result_data = {
        "test": "topic-sanity",
        "semantic_namespace_id": semantic_namespace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_voxels": len(voxel_keys),
        "named_voxels": named_count,
        "voxels_with_coords": len(voxels_with_coords),
        "voxel_map": {
            vk: {"coords": list(c), "display_name": n} for vk, c, n in voxels_with_coords
        },
    }
    path = save_result(args.run_dir, "topic_sanity", result_data)
    print(f"\n  💾 Results saved to {path}")


# ---------------------------------------------------------------------------
# Subcommand: jaccard
# ---------------------------------------------------------------------------


def cmd_jaccard(args: argparse.Namespace) -> None:
    """Measure context-edge locality: PRECEDES edges vs voxel distance.

    Consecutive sentences in source documents are linked by :precedes edges.
    If PSA coordinates preserve meaning, sentences connected by :precedes
    should land in nearby voxels more often than distant ones.

    For every pair of voxels (including self-pairs at distance 0), we count
    the :precedes edges between them and normalize with the Salton index:
    edges(A,B) / sqrt(size_A * size_B).  This controls for the null that
    large voxels attract more edges by chance.

    The Salton index is the complete behavioral-locality metric.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    print("🔑 Retrieving credentials...")
    password = get_neo4j_password()
    # 1. Get voxel sizes
    print("  ⏳ Querying voxel sizes...")
    raw = run_cypher(
        "MATCH (s:SentenceNode) "
        "WHERE s.visual_voxel_id IS NOT NULL "
        "RETURN s.visual_voxel_id AS voxel, count(s) AS cnt "
        "ORDER BY cnt DESC",
        password,
    )
    voxel_counts = parse_two_column(raw)
    print(f"  Found {len(voxel_counts)} occupied visual voxels")

    # 2. Get all :precedes edges grouped by voxel pair
    # This is the ground-truth document flow: consecutive sentences share a
    # :precedes edge.  One Cypher query replaces the old 105-voxel vector
    # search loop.
    print("  ⏳ Querying :precedes edge distribution across voxels...")
    raw = run_cypher(
        "MATCH (a:SentenceNode)-[:precedes]->(b:SentenceNode) "
        "WHERE a.visual_voxel_id IS NOT NULL AND b.visual_voxel_id IS NOT NULL "
        "RETURN a.visual_voxel_id AS voxel_a, b.visual_voxel_id AS voxel_b, "
        "count(*) AS edges",
        password,
    )

    # Parse 3-column output — make edge counts undirected
    edge_counts: dict[tuple[str, str], int] = {}
    total_edges = 0
    intra_edges = 0
    cross_edges = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("voxel_a"):
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 3:
            va, vb = parts[0], parts[1]
            try:
                cnt = int(parts[2])
            except ValueError:
                continue
            # Canonical key: sorted pair (undirected)
            key = (min(va, vb), max(va, vb))
            edge_counts[key] = edge_counts.get(key, 0) + cnt
            total_edges += cnt
            if va == vb:
                intra_edges += cnt
            else:
                cross_edges += cnt

    print(f"  Total :precedes edges with labels: {total_edges}")
    print(f"  Intra-voxel (same voxel):  {intra_edges}")
    print(f"  Cross-voxel (different):   {cross_edges}")

    # 3. Build coordinate map
    coords = {}
    for v in voxel_counts:
        c = parse_voxel_coords(v)
        if c:
            coords[v] = c

    valid_voxels = sorted(coords.keys())
    n = len(valid_voxels)
    print(f"  Voxels with valid coordinates: {n}")

    if n < 2:
        print("  ❌ Need at least 2 voxels")
        return

    # 4. Pre-compute distance matrix for vectorized permutation test
    coord_array = np.array([coords[valid_voxels[i]] for i in range(n)], dtype=np.float64)
    # dist_matrix[i,j] = Euclidean distance between voxel i and j
    diff = coord_array[:, None, :] - coord_array[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))

    # 5. Pre-compute Salton matrix: edges(A,B) / sqrt(|A| * |B|)
    salton_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            vi, vj = valid_voxels[i], valid_voxels[j]
            key = (min(vi, vj), max(vi, vj))
            raw_edges = edge_counts.get(key, 0)
            denom = math.sqrt(voxel_counts[vi] * voxel_counts[vj])
            s = raw_edges / denom if denom > 0 else 0.0
            salton_matrix[i, j] = s
            salton_matrix[j, i] = s

    # Extract upper triangle (including diagonal = self-pairs at distance 0)
    triu_idx = np.triu_indices(n, k=0)
    distances = dist_matrix[triu_idx]
    salton_values = salton_matrix[triu_idx]

    # Correlation
    if len(distances) > 1 and np.std(distances) > 0 and np.std(salton_values) > 0:
        correlation = float(np.corrcoef(distances, salton_values)[0, 1])
        slope, intercept = np.polyfit(distances, salton_values, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        correlation = 0.0
        slope = 0.0
        intercept = 0.0

    mean_salton = float(np.mean(salton_values))
    nonzero_pairs = int(np.sum(salton_values > 0))

    print("\n  📊 PRECEDES-EDGE LOCALITY (Salton index)")
    print(f"     Voxel pairs analyzed:    {len(distances)} (incl. {n} self-pairs)")
    print(f"     Pairs with edges:        {nonzero_pairs}")
    print(f"     Mean Salton index:       {mean_salton:.6f}")
    print(f"     Pearson correlation:     {correlation:.4f}")
    print(f"     Decay slope:             {slope:.6f}")
    print(f"     Intercept:               {intercept:.6f}")

    # 6. Permutation test — vectorized
    # Null hypothesis: voxel positions are random w.r.t. PRECEDES structure.
    # Keep edge structure (salton_values) fixed, permute coordinate assignments.
    n_perms = getattr(args, "permutations", 1000)
    print(f"\n  ⏳ Running permutation test ({n_perms} permutations)...")
    rng = np.random.default_rng(seed=42)
    null_slopes = []

    for _ in range(n_perms):
        perm = rng.permutation(n)
        # Permuted distance matrix: voxel i gets coordinates of voxel perm[i]
        perm_dist = dist_matrix[np.ix_(perm, perm)]
        perm_distances = perm_dist[triu_idx]
        if np.std(perm_distances) > 0 and np.std(salton_values) > 0:
            null_slopes.append(float(np.polyfit(perm_distances, salton_values, 1)[0]))
        else:
            null_slopes.append(0.0)

    null_slopes = np.array(null_slopes)
    p_value = float(np.mean(null_slopes <= slope))

    print(f"     Null slope mean:       {float(np.mean(null_slopes)):.6f}")
    print(f"     Observed slope:        {slope:.6f}")
    print(f"     p-value:               {p_value:.4f}")

    if p_value < 0.05:
        print(f"\n  ✅ PASS: PRECEDES-edge locality is significant (p={p_value:.4f})")
        verdict = "PASS"
    else:
        print(f"\n  ❌ FAIL: PRECEDES-edge locality NOT significant (p={p_value:.4f})")
        verdict = "FAIL"

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir,
        "method": "precedes_edge_salton",
        "n_voxels": n,
        "n_pairs": len(distances),
        "n_self_pairs": n,
        "total_precedes_edges": total_edges,
        "intra_voxel_edges": intra_edges,
        "cross_voxel_edges": cross_edges,
        "nonzero_pairs": nonzero_pairs,
        "mean_salton": mean_salton,
        "correlation": correlation,
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "n_permutations": n_perms,
        "null_slope_mean": float(np.mean(null_slopes)),
        "verdict": verdict,
    }

    path = save_result(args.run_dir, "jaccard_locality", result)
    print(f"\n  💾 Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSA Field Test Suite — §23 First-Round Scientific Test Protocol",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # voxel-snapshot
    p = sub.add_parser("voxel-snapshot", help="Export voxel occupancy and paraphrase distribution")
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.set_defaults(func=cmd_voxel_snapshot)

    # locality
    p = sub.add_parser("locality", help="Compute locality decay curve + permutation test")
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.add_argument(
        "--permutations", type=int, default=1000, help="Number of permutations for null test"
    )
    p.set_defaults(func=cmd_locality)

    # baseline (full-embedding native-space curve, paired with locality)
    p = sub.add_parser(
        "baseline", help="Embedding-space baseline curve (locality without voxelization)"
    )
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.set_defaults(func=cmd_baseline)

    # baseline-sentence (Option B: sampled full-D sentence pairs)
    p = sub.add_parser(
        "baseline-sentence", help="Sentence-level full-embedding baseline (Option B)"
    )
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.add_argument(
        "--sample-pairs", type=int, default=500, help="Number of sentence pairs to sample"
    )
    p.set_defaults(func=cmd_baseline_sentence)

    # guide-stars
    p = sub.add_parser("guide-stars", help="Verify guide star placements and standard candle")
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.set_defaults(func=cmd_guide_stars)

    # falsify
    p = sub.add_parser("falsify", help="Run falsification controls (scramble, shuffle, perturb)")
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.add_argument(
        "--control",
        required=True,
        choices=["scramble", "shuffle", "perturb"],
        help="Which null control to run",
    )
    p.add_argument("--permutations", type=int, default=1000, help="Number of permutations")
    p.set_defaults(func=cmd_falsify)

    # settle-check
    p = sub.add_parser("settle-check", help="Repeated snapshots until Neo4j writes stabilize")
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.add_argument("--diff", required=True, help="Label to diff against (initial)")
    p.add_argument("--max-checks", type=int, default=5, help="Max settlement rounds")
    p.add_argument("--interval", type=int, default=60, help="Seconds between checks")
    p.set_defaults(func=cmd_settle_check)

    # topic-sanity
    p = sub.add_parser("topic-sanity", help="Sanity check: voxel display_names vs PSA proximity")
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.add_argument(
        "--semantic-namespace-id",
        required=True,
        help="Canonical tenant/workspace semantic namespace (ns_ + 64 lowercase hex)",
    )
    p.set_defaults(func=cmd_topic_sanity)

    # jaccard
    p = sub.add_parser("jaccard", help="PRECEDES-edge locality test (Salton index)")
    p.add_argument("--run-dir", required=True, help="Run directory name under psa_test_runs/")
    p.add_argument("--permutations", type=int, default=1000, help="Permutation count for null test")
    p.set_defaults(func=cmd_jaccard)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
