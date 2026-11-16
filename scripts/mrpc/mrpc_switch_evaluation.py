#!/usr/bin/env python3
"""
MRPC Switch Evaluation — Semantic Switch Paraphrase Detection Benchmark

Evaluates whether semantic switches can augment embedding-based paraphrase
detection. The hypothesis: switch-guided traversal through the meaning graph
provides additional signal beyond raw cosine similarity.

Evaluation methodology:
1. Embed MRPC sentence pairs using the embedding service
2. For each pair, simulate switch-chain traversal:
   a. Embed sentence 1 → find nearest paraphrase nodes (seed resolution)
   b. Execute switch chain from seeds
   c. Check if sentence 2's nearest node appears on the traversal path
3. Combine switch-path features with embedding similarity for classification
4. Compare F1 against embedding-only baseline

Modes:
- embedding_only: Cosine similarity + MLP (baseline, should match mrpc_embedding_benchmark.py)
- switch_augmented: Embedding features + switch traversal features
- switch_only: Pure switch-based detection (path overlap + confidence)

Usage:
    python scripts/research/mrpc/mrpc_switch_evaluation.py --tei-url http://localhost:11434
    python scripts/research/mrpc/mrpc_switch_evaluation.py --mode switch_augmented

AWS:
    python scripts/research/mrpc/mrpc_switch_evaluation.py --tei-url http://embedding.aiidea.local:11434 --neo4j-uri bolt://neo4j.aiidea.local:7687
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# MRPC data directory
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from mcp_servers.mcp_lib.qwen_typed_field_test import load_mrpc_pairs

MRPC_DIR = APP_ROOT / "utils" / "msrpc_split"

# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading (shared with mrpc_embedding_benchmark.py)
# ═══════════════════════════════════════════════════════════════════════════════


def load_mrpc_data() -> Tuple[Dict[int, str], Dict[int, str], List[Dict]]:
    """Load clean MRPC text by source sentence ID."""
    records = load_mrpc_pairs(MRPC_DIR)
    first_sentences = {pair.pair_id: pair.sentence1 for pair in records}
    second_sentences = {pair.pair_id: pair.sentence2 for pair in records}
    pairs = [
        {
            "pair_id": pair.pair_id,
            "quality": int(pair.is_paraphrase),
            "id1": pair.pair_id,
            "id2": pair.pair_id,
            "is_paraphrase": pair.is_paraphrase,
        }
        for pair in records
    ]

    print(
        f"  MRPC Dataset: {len(first_sentences)} first, {len(second_sentences)} second, {len(pairs)} pairs"
    )
    return first_sentences, second_sentences, pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════════════


def embed_batch(texts: List[str], tei_url: str, batch_size: int = 32) -> np.ndarray:
    """Embed texts using TEI endpoint."""
    import requests

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(f"{tei_url}/embed", json={"inputs": batch}, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"TEI error: {resp.status_code}")
        all_embeddings.extend(resp.json())
        if i > 0 and i % 200 == 0:
            print(f"   Embedded {i}/{len(texts)}...")
    return np.array(all_embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
# Switch Traversal Simulation
# ═══════════════════════════════════════════════════════════════════════════════


def simulate_switch_traversal(
    emb1: np.ndarray,
    emb2: np.ndarray,
    all_embeddings: np.ndarray,
    all_texts: List[str],
    max_depth: int = 10,
    k_neighbors: int = 5,
) -> Dict[str, float]:
    """
    Simulate switch-chain traversal for a sentence pair.

    Since we don't have a live Neo4j graph during offline evaluation,
    we simulate the traversal using the embedding space:
    1. Find k-nearest neighbors of sentence 1 (seed resolution)
    2. From each seed, follow nearest-neighbor chains (simulating PRECEDES)
    3. Check if any node on the path is near sentence 2
    4. Compute path-based features

    Returns dict of switch traversal features.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Seed resolution: nearest neighbors to emb1
    sims1 = cosine_similarity(emb1.reshape(1, -1), all_embeddings)[0]
    seed_indices = np.argsort(sims1)[-k_neighbors:][::-1]

    # Target: similarity of emb2 to all embeddings
    sims2 = cosine_similarity(emb2.reshape(1, -1), all_embeddings)[0]
    target_indices = set(np.argsort(sims2)[-k_neighbors:][::-1])

    # Simulated switch chain: greedy nearest-neighbor walk from each seed
    path_overlap = 0
    min_distance_to_target = float("inf")
    total_confidence = 0.0
    path_length = 0
    visited = set()

    for seed_idx in seed_indices:
        current = seed_idx
        for _depth in range(max_depth):
            if current in visited:
                break
            visited.add(current)
            path_length += 1

            # Check if current is near sentence 2's neighborhood
            if current in target_indices:
                path_overlap += 1

            # Compute distance from current to target
            dist = (
                1.0
                - cosine_similarity(all_embeddings[current].reshape(1, -1), emb2.reshape(1, -1))[
                    0, 0
                ]
            )
            min_distance_to_target = min(min_distance_to_target, dist)

            # Switch decision: find nearest unvisited neighbor (simulate PRECEDES)
            current_sims = cosine_similarity(
                all_embeddings[current].reshape(1, -1), all_embeddings
            )[0]
            # Mask visited
            for v in visited:
                current_sims[v] = -1
            next_idx = np.argmax(current_sims)
            confidence = current_sims[next_idx]
            total_confidence += confidence

            if confidence < 0.3:  # Simulated confidence threshold
                break
            current = next_idx

    avg_confidence = total_confidence / max(path_length, 1)

    return {
        "path_overlap": path_overlap,
        "path_overlap_ratio": path_overlap / max(len(target_indices), 1),
        "min_distance_to_target": min_distance_to_target,
        "path_length": path_length,
        "avg_path_confidence": avg_confidence,
        "reachable": 1.0 if path_overlap > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_embedding_features(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Prepare embedding-based features for a pair."""
    cosine = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
    diff = emb1 - emb2
    hadamard = emb1 * emb2
    return np.concatenate([diff, hadamard, [cosine]])


def prepare_switch_features(traversal: Dict[str, float]) -> np.ndarray:
    """Prepare switch-based features."""
    return np.array(
        [
            traversal["path_overlap"],
            traversal["path_overlap_ratio"],
            traversal["min_distance_to_target"],
            traversal["path_length"],
            traversal["avg_path_confidence"],
            traversal["reachable"],
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_with_cv(
    features: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate using stratified cross-validation with MLP classifier."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s, precisions, recalls, accuracies = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=42 + fold,
            early_stopping=True,
            validation_fraction=0.1,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        f1s.append(f1_score(y_test, preds))
        precisions.append(precision_score(y_test, preds))
        recalls.append(recall_score(y_test, preds))
        accuracies.append(accuracy_score(y_test, preds))

    results = {
        "method": method_name,
        "f1": np.mean(f1s),
        "f1_std": np.std(f1s),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "accuracy": np.mean(accuracies),
    }

    print(f"\n  {method_name}:")
    print(f"    F1:        {results['f1']:.4f} (+/- {results['f1_std']:.4f})")
    print(f"    Precision: {results['precision']:.4f}")
    print(f"    Recall:    {results['recall']:.4f}")
    print(f"    Accuracy:  {results['accuracy']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="MRPC Switch Evaluation")
    parser.add_argument(
        "--tei-url", default="http://localhost:11434", help="TEI embedding endpoint"
    )
    parser.add_argument(
        "--mode",
        choices=["embedding_only", "switch_augmented", "switch_only", "all"],
        default="all",
        help="Evaluation mode",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit pairs for quick testing")
    parser.add_argument("--k-neighbors", type=int, default=5, help="Neighbors for seed resolution")
    parser.add_argument("--max-depth", type=int, default=8, help="Max traversal depth")
    args = parser.parse_args()

    print("=" * 70)
    print("MRPC SWITCH EVALUATION — Semantic Switch Paraphrase Detection")
    print("=" * 70)

    # Load data
    first_sentences, second_sentences, pairs = load_mrpc_data()

    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
        print(f"  Limited to {args.max_pairs} pairs for quick evaluation")

    # Collect unique texts
    text_set = set()
    valid_pairs = []
    for p in pairs:
        idx1 = int(p["id1"])
        idx2 = int(p["id2"])
        if idx1 in first_sentences and idx2 in second_sentences:
            text_set.add(first_sentences[idx1])
            text_set.add(second_sentences[idx2])
            valid_pairs.append(p)

    all_texts = sorted(text_set)
    text_to_idx = {t: i for i, t in enumerate(all_texts)}
    print(f"  Unique texts to embed: {len(all_texts)}")

    # Embed all texts
    print("\n  Embedding texts...")
    t0 = time.time()
    all_embeddings = embed_batch(all_texts, args.tei_url)
    print(f"  Embedded {len(all_texts)} texts in {time.time() - t0:.1f}s")

    # Build features
    labels = []
    embedding_features = []
    switch_features = []

    print("\n  Computing features...")
    for i, p in enumerate(valid_pairs):
        idx1 = int(p["id1"])
        idx2 = int(p["id2"])
        s1 = first_sentences[idx1]
        s2 = second_sentences[idx2]
        emb1 = all_embeddings[text_to_idx[s1]]
        emb2 = all_embeddings[text_to_idx[s2]]

        # Embedding features
        emb_feat = prepare_embedding_features(emb1, emb2)
        embedding_features.append(emb_feat)

        # Switch traversal features (simulated)
        if args.mode in ("switch_augmented", "switch_only", "all"):
            traversal = simulate_switch_traversal(
                emb1,
                emb2,
                all_embeddings,
                all_texts,
                max_depth=args.max_depth,
                k_neighbors=args.k_neighbors,
            )
            switch_features.append(prepare_switch_features(traversal))

        labels.append(1 if p["is_paraphrase"] else 0)

        if (i + 1) % 500 == 0:
            print(f"    Processed {i + 1}/{len(valid_pairs)} pairs...")

    labels_arr = np.array(labels)
    emb_features_arr = np.array(embedding_features)
    switch_features_arr = np.array(switch_features) if switch_features else None

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    results = []

    if args.mode in ("embedding_only", "all"):
        r = evaluate_with_cv(emb_features_arr, labels_arr, "Embedding Only (MLP)")
        results.append(r)

    if switch_features_arr is not None and args.mode in ("switch_only", "all"):
        r = evaluate_with_cv(switch_features_arr, labels_arr, "Switch Only (MLP)")
        results.append(r)

    if switch_features_arr is not None and args.mode in ("switch_augmented", "all"):
        combined = np.hstack([emb_features_arr, switch_features_arr])
        r = evaluate_with_cv(combined, labels_arr, "Switch Augmented (MLP)")
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<30} {'F1':>8} {'P':>8} {'R':>8} {'Acc':>8}")
    print(f"  {'-' * 62}")
    for r in results:
        print(
            f"  {r['method']:<30} {r['f1']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['accuracy']:>8.4f}"
        )

    # Target: Beat 90.8% F1 (Qwen3-Embedding-8B baseline)
    best_f1 = max(r["f1"] for r in results) if results else 0
    print("\n  Target F1: 0.9080 (Qwen3-Embedding-8B baseline)")
    print(f"  Best F1:   {best_f1:.4f} {'BEAT' if best_f1 > 0.908 else 'NOT YET'}")


if __name__ == "__main__":
    main()
