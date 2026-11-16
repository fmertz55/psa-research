#!/usr/bin/env python3
"""
MRPC Embedding Benchmark - Beat 90.8% F1!
==========================================

Evaluates embedding models on MRPC paraphrase detection using:
1. Cosine similarity threshold (unsupervised baseline)
2. 3-layer MLP classifier (supervised - matches Qwen3 benchmark methodology)
3. Cross-validation for robust evaluation

The Qwen3-Embedding-8B baseline is 90.8% F1 with 3-layer MLP.
Our goal: Beat this using optimal classifier configuration.

Usage:
    python scripts/research/mrpc/mrpc_embedding_benchmark.py --tei-url http://localhost:11434
    python scripts/research/mrpc/mrpc_embedding_benchmark.py --tei-url http://embedding.aiidea.local:11434

AWS TEI endpoints (update as needed):
    - Local dev: http://localhost:11434
    - ECS: http://embedding.aiidea.local:11434
"""

import argparse
import hashlib
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# MRPC data directory
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from mcp_servers.mcp_lib.qwen_typed_field_test import load_mrpc_pairs

MRPC_DIR = APP_ROOT / "utils" / "msrpc_split"


@dataclass
class BenchmarkResult:
    """Stores benchmark evaluation metrics"""

    method: str
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    config: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.method}: F1={self.f1_score:.4f} P={self.precision:.4f} R={self.recall:.4f} Acc={self.accuracy:.4f}"


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

    print("📊 MRPC Dataset loaded:")
    print(f"   First sentences: {len(first_sentences)}")
    print(f"   Second sentences: {len(second_sentences)}")
    print(f"   Pairs: {len(pairs)}")

    return first_sentences, second_sentences, pairs


def embed_batch(texts: List[str], tei_url: str, batch_size: int = 32) -> np.ndarray:
    """Embed texts using TEI endpoint with batching"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        response = requests.post(f"{tei_url}/embed", json={"inputs": batch}, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(f"TEI error: {response.status_code} - {response.text}")

        embeddings = response.json()
        all_embeddings.extend(embeddings)

        if i > 0 and i % 200 == 0:
            print(f"   Embedded {i}/{len(texts)} texts...")

    return np.array(all_embeddings)


def prepare_features(
    emb1: np.ndarray, emb2: np.ndarray, feature_type: str = "concat"
) -> np.ndarray:
    """Prepare features for classifier from embedding pairs

    Args:
        emb1: Embeddings for first sentences [N, D]
        emb2: Embeddings for second sentences [N, D]
        feature_type: One of:
            - "concat": [emb1, emb2] - 2D features
            - "diff": |emb1 - emb2| - D features
            - "hadamard": emb1 * emb2 - D features
            - "combined": [|diff|, hadamard, cosine] - 2D+1 features
            - "full": [emb1, emb2, |diff|, hadamard] - 4D features
    """
    # Normalize embeddings
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)

    if feature_type == "concat":
        return np.concatenate([emb1_norm, emb2_norm], axis=1)

    elif feature_type == "diff":
        return np.abs(emb1_norm - emb2_norm)

    elif feature_type == "hadamard":
        return emb1_norm * emb2_norm

    elif feature_type == "combined":
        diff = np.abs(emb1_norm - emb2_norm)
        hadamard = emb1_norm * emb2_norm
        cosine = np.sum(emb1_norm * emb2_norm, axis=1, keepdims=True)
        return np.concatenate([diff, hadamard, cosine], axis=1)

    elif feature_type == "full":
        diff = np.abs(emb1_norm - emb2_norm)
        hadamard = emb1_norm * emb2_norm
        return np.concatenate([emb1_norm, emb2_norm, diff, hadamard], axis=1)

    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def evaluate_cosine_threshold(
    emb1: np.ndarray, emb2: np.ndarray, labels: np.ndarray, thresholds: List[float] = None
) -> BenchmarkResult:
    """Evaluate using cosine similarity threshold (unsupervised baseline)"""

    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.01)

    # Compute cosine similarities
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-10)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-10)
    cosines = np.sum(emb1_norm * emb2_norm, axis=1)

    # Grid search for best threshold
    best_f1 = 0
    best_threshold = 0.5
    best_preds = None

    for thresh in thresholds:
        preds = (cosines >= thresh).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_preds = preds

    return BenchmarkResult(
        method="Cosine Threshold",
        f1_score=f1_score(labels, best_preds),
        precision=precision_score(labels, best_preds),
        recall=recall_score(labels, best_preds),
        accuracy=accuracy_score(labels, best_preds),
        config={"threshold": best_threshold, "mean_cosine": float(np.mean(cosines))},
    )


def evaluate_mlp_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    hidden_layers: Tuple[int, ...] = (256, 128, 64),
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> BenchmarkResult:
    """Evaluate 3-layer MLP classifier with cross-validation

    This matches the Qwen3 benchmark methodology for fair comparison.
    """

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(features_scaled, labels)):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Train MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=random_state + fold,
        )

        mlp.fit(X_train, y_train)
        preds = mlp.predict(X_test)

        f1_scores.append(f1_score(y_test, preds))
        precision_scores.append(precision_score(y_test, preds))
        recall_scores.append(recall_score(y_test, preds))
        accuracy_scores.append(accuracy_score(y_test, preds))

    return BenchmarkResult(
        method=f"MLP {hidden_layers}",
        f1_score=np.mean(f1_scores),
        precision=np.mean(precision_scores),
        recall=np.mean(recall_scores),
        accuracy=np.mean(accuracy_scores),
        config={
            "hidden_layers": hidden_layers,
            "n_folds": n_folds,
            "f1_std": float(np.std(f1_scores)),
            "fold_f1s": [float(f) for f in f1_scores],
        },
    )


def evaluate_logistic_regression(
    features: np.ndarray, labels: np.ndarray, n_folds: int = 5, random_state: int = 42
) -> BenchmarkResult:
    """Evaluate logistic regression (simpler baseline)"""

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(features_scaled, labels)):
        X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        lr = LogisticRegression(max_iter=1000, random_state=random_state + fold)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)

        f1_scores.append(f1_score(y_test, preds))
        precision_scores.append(precision_score(y_test, preds))
        recall_scores.append(recall_score(y_test, preds))
        accuracy_scores.append(accuracy_score(y_test, preds))

    return BenchmarkResult(
        method="Logistic Regression",
        f1_score=np.mean(f1_scores),
        precision=np.mean(precision_scores),
        recall=np.mean(recall_scores),
        accuracy=np.mean(accuracy_scores),
        config={
            "n_folds": n_folds,
            "f1_std": float(np.std(f1_scores)),
            "fold_f1s": [float(f) for f in f1_scores],
        },
    )


def truncate_to_dimensions(embeddings: np.ndarray, dimensions: int) -> np.ndarray:
    """Apply Matryoshka truncation, then restore unit norm.

    Order matters: normalizing before truncation leaves the kept prefix off the
    unit sphere, which silently shifts every cosine.
    """

    if dimensions > embeddings.shape[1]:
        raise ValueError(f"cannot truncate {embeddings.shape[1]} dims to {dimensions}")
    sliced = embeddings[:, :dimensions]
    norms = np.linalg.norm(sliced, axis=1, keepdims=True)
    return sliced / (norms + 1e-10)


def run_full_benchmark(
    tei_url: str,
    output_path: Optional[Path] = None,
    feature_types: List[str] = None,
    cache_path: Optional[Path] = None,
    dimensions: Optional[int] = None,
) -> List[BenchmarkResult]:
    """Run complete MRPC benchmark suite"""

    print("\n" + "=" * 70)
    print("🎯 MRPC EMBEDDING BENCHMARK - Target: Beat 90.8% F1")
    print("=" * 70)

    if feature_types is None:
        feature_types = ["combined", "diff", "hadamard", "concat"]

    # Load data
    first_sentences, second_sentences, pairs = load_mrpc_data()

    # Prepare sentence lists and labels
    sent1_list = []
    sent2_list = []
    labels = []

    for pair in pairs:
        pair_id = pair["pair_id"]
        s1 = first_sentences.get(pair_id)
        s2 = second_sentences.get(pair_id)

        if s1 and s2:
            sent1_list.append(s1)
            sent2_list.append(s2)
            labels.append(1 if pair["is_paraphrase"] else 0)

    labels = np.array(labels)

    print(f"\n📝 Prepared {len(labels)} valid pairs")
    print(f"   Paraphrases: {np.sum(labels)} ({100 * np.mean(labels):.1f}%)")
    print(
        f"   Non-paraphrases: {len(labels) - np.sum(labels)} ({100 * (1 - np.mean(labels)):.1f}%)"
    )

    # Bind the cache to the exact corpus so a stale file cannot be reused.
    corpus_digest = hashlib.sha256(
        json.dumps([sent1_list, sent2_list], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    emb1 = emb2 = None
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["corpus_digest"]) == corpus_digest:
            emb1 = cached["emb1"]
            emb2 = cached["emb2"]
            print(f"\n💾 Loaded cached embeddings from {cache_path}")
            print(f"   Shape: {emb1.shape}")
        else:
            print(f"\n⚠️  Cache {cache_path} is for a different corpus; re-embedding")

    if emb1 is None:
        print(f"\n🔮 Generating embeddings from TEI: {tei_url}")
        start_time = time.time()

        print("   Embedding first sentences...")
        emb1 = embed_batch(sent1_list, tei_url)
        print(f"   Shape: {emb1.shape}")

        print("   Embedding second sentences...")
        emb2 = embed_batch(sent2_list, tei_url)
        print(f"   Shape: {emb2.shape}")

        embed_time = time.time() - start_time
        print(
            f"   ⏱️  Embedding time: {embed_time:.1f}s ({len(sent1_list) * 2 / embed_time:.1f} sentences/sec)"
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, emb1=emb1, emb2=emb2, corpus_digest=corpus_digest)
            print(f"   💾 Cached embeddings to {cache_path}")

    if dimensions and dimensions != emb1.shape[1]:
        print(
            f"\n✂️  Truncating {emb1.shape[1]} → {dimensions} dims (Matryoshka), then L2 renormalizing"
        )
        emb1 = truncate_to_dimensions(emb1, dimensions)
        emb2 = truncate_to_dimensions(emb2, dimensions)

    results = []

    # Baseline: Cosine similarity
    print("\n" + "-" * 50)
    print("📏 BASELINE: Cosine Similarity Threshold")
    print("-" * 50)

    cosine_result = evaluate_cosine_threshold(emb1, emb2, labels)
    results.append(cosine_result)
    print(f"   {cosine_result}")
    print(f"   Best threshold: {cosine_result.config['threshold']:.3f}")
    print(f"   Mean cosine: {cosine_result.config['mean_cosine']:.3f}")

    # MLP configurations to try
    mlp_configs = [
        (256, 128, 64),  # Standard 3-layer (Qwen benchmark config)
        (512, 256, 128),  # Larger 3-layer
        (1024, 512, 256),  # Even larger 3-layer
        (256, 128),  # 2-layer
        (512, 256),  # Larger 2-layer
        (128,),  # 1-layer (baseline)
    ]

    for feature_type in feature_types:
        print("\n" + "-" * 50)
        print(f"🧠 FEATURE TYPE: {feature_type}")
        print("-" * 50)

        features = prepare_features(emb1, emb2, feature_type)
        print(f"   Feature shape: {features.shape}")

        # Logistic regression baseline
        print("\n   📊 Logistic Regression:")
        lr_result = evaluate_logistic_regression(features, labels)
        lr_result.method = f"LR ({feature_type})"
        results.append(lr_result)
        print(f"      {lr_result}")

        # MLP with different configurations
        for hidden_layers in mlp_configs:
            print(f"\n   🔬 MLP {hidden_layers}:")
            mlp_result = evaluate_mlp_classifier(features, labels, hidden_layers)
            mlp_result.method = f"MLP {hidden_layers} ({feature_type})"
            results.append(mlp_result)
            print(f"      {mlp_result}")
            print(f"      Fold F1s: {[f'{f:.3f}' for f in mlp_result.config['fold_f1s']]}")

            # Check if we beat 90.8%
            if mlp_result.f1_score > 0.908:
                print("      🎉 BEATS BASELINE 90.8%!")

    # Summary
    print("\n" + "=" * 70)
    print("🏆 BENCHMARK SUMMARY")
    print("=" * 70)

    # Sort by F1
    sorted_results = sorted(results, key=lambda x: x.f1_score, reverse=True)

    print("\nTop 10 Results:")
    print(f"{'Rank':<5} {'Method':<40} {'F1':<8} {'P':<8} {'R':<8} {'Acc':<8}")
    print("-" * 70)

    for i, r in enumerate(sorted_results[:10], 1):
        beat_marker = "🎯" if r.f1_score > 0.908 else ""
        print(
            f"{i:<5} {r.method:<40} {r.f1_score:.4f}   {r.precision:.4f}   {r.recall:.4f}   {r.accuracy:.4f} {beat_marker}"
        )

    best = sorted_results[0]
    baseline = 0.908

    print("\n" + "-" * 70)
    if best.f1_score > baseline:
        print(f"✅ SUCCESS! Best F1 ({best.f1_score:.4f}) beats baseline ({baseline:.4f})")
        print(f"   Improvement: +{(best.f1_score - baseline) * 100:.2f}%")
    else:
        print(f"❌ Did not beat baseline. Best: {best.f1_score:.4f} vs Target: {baseline:.4f}")
        print(f"   Gap: {(baseline - best.f1_score) * 100:.2f}%")

    print(f"\nBest configuration: {best.method}")
    print(f"   Config: {best.config}")

    # Save results
    if output_path:
        output_data = {
            "baseline_f1": baseline,
            "best_f1": best.f1_score,
            "beat_baseline": best.f1_score > baseline,
            "embedding_dimensions": int(emb1.shape[1]),
            "num_pairs": len(labels),
            "results": [
                {
                    "method": r.method,
                    "f1_score": r.f1_score,
                    "precision": r.precision,
                    "recall": r.recall,
                    "accuracy": r.accuracy,
                    "config": r.config,
                }
                for r in sorted_results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MRPC Embedding Benchmark - Beat 90.8% F1!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--tei-url", default="http://localhost:11434", help="TEI embedding endpoint URL"
    )

    parser.add_argument("--output", "-o", type=Path, help="Save results to JSON file")

    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=["combined", "diff", "hadamard"],
        help="Feature combination types to test",
    )

    parser.add_argument(
        "--cache", type=Path, help="Load/save full-dimension embeddings here to avoid re-embedding"
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        help="Truncate embeddings to N dims and L2 renormalize before evaluating",
    )

    args = parser.parse_args()

    # Verify TEI endpoint
    try:
        info_resp = requests.get(f"{args.tei_url}/info", timeout=10)
        if info_resp.status_code == 200:
            info = info_resp.json()
            print(f"✅ TEI endpoint ready: {info.get('model_id', 'unknown')}")
            print(f"   Embedding dim: {info.get('max_batch_tokens', 'unknown')}")
    except Exception as e:
        print(f"⚠️  Could not get TEI info: {e}")
        print("   Continuing anyway...")

    # Run benchmark
    run_full_benchmark(
        args.tei_url,
        args.output,
        args.feature_types,
        cache_path=args.cache,
        dimensions=args.dimensions,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
