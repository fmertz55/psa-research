#!/usr/bin/env python3
"""Evaluate a learned diagnostic semantic-relation lane on the frozen MRPC split.

Only sentence-component calibration labels fit the model and choose thresholds.
Although the shared MRPC loader materializes every row, held-out labels are not
accessed for model or threshold selection; evaluation reads them only after the
artifact and policies are frozen.  The resolver is swap-invariant, may abstain,
and has no paraphrase-membership authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import RidgeClassifier

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
for import_path in (REPOSITORY_ROOT, APP_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.research.mrpc.run_mrpc_current_pipeline_benchmark import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    _load_pinned_frame,
    _project_representations,
)
from scripts.research.mrpc.run_mrpc_gate_verifier_benchmark import _psa_neighborhoods  # noqa: E402
from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import (  # noqa: E402
    _atomic_json,
    _load_cache_evidence,
)
from scripts.research.mrpc.run_mrpc_proposition_graph_benchmark import (  # noqa: E402
    BLIND_SPLIT_SEED,
    DEFAULT_CACHE,
    DEFAULT_GRAPH_CACHE,
    DEFAULT_SENSE_CACHE,
    PINNED_GRAPH_CACHE_SHA256,
    PINNED_GRAPH_PARSER_IDENTITY,
    _build_or_load_graphs,
)

from mcp_servers.mcp_lib.proposition_graph import PropositionGraph  # noqa: E402
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    MRPCPair,
    _binary_metrics,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)
from mcp_servers.mcp_lib.semantic_relation_features import (  # noqa: E402
    COMPACT_GENERIC_FEATURE_DIMENSION,
    COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
    CompactGenericLinearArtifact,
    compact_generic_pair_matrix,
    load_compact_generic_linear_artifact,
)
from mcp_servers.mcp_lib.semantic_relation_resolver import (  # noqa: E402
    SYMMETRIC_PAIR_FEATURE_SCHEMA_VERSION,
    SemanticRelationState,
    SemanticResolverProvenance,
    SemanticResolverThresholds,
    SymmetricEmbeddingRelationResolver,
    symmetric_embedding_pair_features,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    PSA_SIGNATURE_CANDIDATE_LIMIT,
    QWEN3_EMBEDDING_MODEL_ID,
)

REPORT_SCHEMA_VERSION = "aiidea_mrpc_semantic_relation_resolver_benchmark_v3"
RESOLVER_ID = "compact_generic_ridge_mrpc_diagnostic_v1"
GRAPH_PAIR_FEATURE_SCHEMA_VERSION = "proposition_graph_multiset_overlap_v1"
MODEL_SELECTION_FOLDS = 5
RIDGE_ALPHA_GRID = (100.0, 1_000.0, 10_000.0, 100_000.0)
SELECTIVE_TARGET_PRECISION = 0.95
PAIRED_BOOTSTRAP_SEED = 20260803
PAIRED_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_MINIMUM_F1_GAIN = 0.01
EXPECTED_MRPC_DATASET_SHA256 = {
    "msr_paraphrase_data.txt": ("0daa19928665d29551d93c42fbd48200d272818b5e91fe589a98ed7467e69519"),
    "msr_paraphrase_test.txt": ("9b4c96307f6e9276f4c32614dea1f1e65fa6777d0e4451c0c4e87c3d048636ef"),
}
EXPECTED_BASELINE_REPORT_SHA256 = "c40eea84bce3eea527198ef1e31de69c2eb7285b3be55f2c6616e35c747ebffe"
EXPECTED_BASELINE_SCHEMA_VERSION = "aiidea_mrpc_proposition_graph_benchmark_v2"
EXPECTED_BASELINE_POLICY_ID = "legacy_etpc_mrpc_diagnostic_v1"
EXPECTED_BASELINE_SPLIT = {
    "seed": BLIND_SPLIT_SEED,
    "unit": "connected_component_of_normalized_sentence_identity",
    "calibration_pair_count": 618,
    "held_out_pair_count": 1107,
}
EXPECTED_COMPARABLE_GATE_METRICS = {
    "accuracy": 0.7353206865401988,
    "count": 1107,
    "f1": 0.8339943342776205,
    "false_negative": 5,
    "false_positive": 288,
    "precision": 0.71875,
    "recall": 0.9932523616734144,
    "true_negative": 78,
    "true_positive": 736,
}
EXPECTED_CROSS_LANE_VERIFIER_METRICS = {
    "accuracy": 0.5004516711833785,
    "count": 1107,
    "f1": 0.4072883172561629,
    "false_negative": 551,
    "false_positive": 2,
    "precision": 0.9895833333333334,
    "recall": 0.2564102564102564,
    "true_negative": 364,
    "true_positive": 190,
}
EXPECTED_FROZEN_SELECTION_FILE_SHA256 = (
    "fa92b37f978f3fb955cf097c984ac386ff28a8439b3a9d10bf10d076a91a8d3f"
)
EXPECTED_FROZEN_SELECTION_SHA256 = (
    "6906e04234640628e4f42a3b227271e856a032bd4038776d2fff2e067f42908e"
)
EXPECTED_FROZEN_CONTRACT_FILE_SHA256 = (
    "58ab913adfcaf48dc06c0b0a911b97706ad9d9af109feebb80e0ed1d830f55cf"
)
EXPECTED_FROZEN_CONTRACT_SHA256 = "29fb4c35d61c8e3f5bdf29160b9f7f7d52b307444adac59bddea597e0de24956"
EXPECTED_FROZEN_ARTIFACT_SHA256 = "8cffff04158091455029b70af9c59adfaebcd11b9cd0c13aafb5c818e8844350"
EXPECTED_FROZEN_REVEAL_FILE_SHA256 = (
    "a5b228d25affc998a0df71471d115369312c958b6534096ebe637fb57ff0d79a"
)
EXPECTED_FROZEN_EXPERIMENT_SOURCE_SHA256 = (
    "d5036695993c18ac33886fbad1c32229da35ed0045a34f205f74a7b2cfbc154a"
)
EXPECTED_FROZEN_PROPOSITION_GRAPH_SOURCE_SHA256 = (
    "c8917f3c2296667b9ac0f0d5a6ad24c5e8aaa0d8c84fea0c10a6c6e83ed08b76"
)
EXPECTED_FROZEN_EXPERIMENT_ID = "generic_symmetric_models_component_cv_v1"
EXPECTED_FROZEN_EXPERIMENT_SCHEMA = "mrpc_semantic_relation_model_search_v1"
EXPECTED_FROZEN_WINNER = {
    "oof_metrics": {
        "accuracy": 0.8009708737864077,
        "count": 618,
        "f1": 0.8619528619528619,
        "false_negative": 22,
        "false_positive": 101,
        "precision": 0.7917525773195876,
        "recall": 0.9458128078817734,
        "true_negative": 111,
        "true_positive": 384,
    },
    "spec": {
        "candidate_id": "ridge:compact_generic_v1:fb340838fc4d",
        "family": "ridge",
        "feature_set": COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
        "parameters": {"alpha": 1000.0, "class_weight": None},
    },
    "threshold": -0.12455821638457187,
}
EXPECTED_FROZEN_HELD_OUT_METRICS = {
    "accuracy": 0.7904245709123758,
    "count": 1107,
    "f1": 0.8564356435643564,
    "false_negative": 49,
    "false_positive": 183,
    "precision": 0.7908571428571428,
    "recall": 0.9338731443994602,
    "true_negative": 183,
    "true_positive": 692,
}
DEFAULT_BASELINE_REPORT = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_rule_policy_ablation_2026-08-03"
    / "legacy"
    / "mrpc_proposition_graph_report.json"
)
DEFAULT_FROZEN_EXPERIMENT_DIR = (
    REPOSITORY_ROOT / "reports" / "mrpc_semantic_relation_model_search_2026-08-03"
)
DEFAULT_FROZEN_SELECTION = DEFAULT_FROZEN_EXPERIMENT_DIR / "frozen_selection.json"
DEFAULT_FROZEN_CONTRACT = DEFAULT_FROZEN_EXPERIMENT_DIR / "calibration_resolver_contract.json"
DEFAULT_FROZEN_ARTIFACT = DEFAULT_FROZEN_EXPERIMENT_DIR / "fitted_resolver_artifact.npz"
DEFAULT_FROZEN_REVEAL = DEFAULT_FROZEN_EXPERIMENT_DIR / "held_out_reveal.json"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_semantic_relation_resolver_2026-08-03"


@dataclass(frozen=True)
class _FittedAblation:
    name: str
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    oof_margins: np.ndarray
    binary_threshold: float
    oof_metrics: Mapping[str, Any]

    def margins(self, features: np.ndarray) -> np.ndarray:
        standardized = (features - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _dataset_identity(mrpc_dir: Path) -> dict[str, Any]:
    paths = (
        mrpc_dir.parent / "msr_paraphrase_data.txt",
        mrpc_dir.parent / "msr_paraphrase_test.txt",
    )
    files: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing MRPC dataset file: {path}")
        sha256 = _file_sha256(path)
        expected_sha256 = EXPECTED_MRPC_DATASET_SHA256[path.name]
        if sha256 != expected_sha256:
            raise ValueError(
                f"MRPC dataset SHA-256 mismatch for {path.name}: "
                f"expected {expected_sha256}, got {sha256}"
            )
        files.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256,
                "expected_sha256": expected_sha256,
            }
        )
    portable_identity = [
        {key: file[key] for key in ("name", "size_bytes", "sha256")} for file in files
    ]
    return {
        "files": files,
        "identity_sha256": _canonical_sha256(portable_identity),
        "matches_frozen_comparator_dataset": True,
    }


def _benchmark_source_identity() -> dict[str, Any]:
    sources = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "semantic_relation_resolver.py",
        REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "semantic_relation_features.py",
    )
    files = [
        {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": _file_sha256(path),
        }
        for path in sources
    ]
    return {"files": files, "identity_sha256": _canonical_sha256(files)}


def _validated_embedding_model_id(embedding_evidence: Mapping[str, Any]) -> str:
    model_id = str(embedding_evidence.get("model_id") or "")
    if model_id != QWEN3_EMBEDDING_MODEL_ID:
        raise ValueError(
            "embedding cache model identity mismatch: "
            f"expected {QWEN3_EMBEDDING_MODEL_ID!r}, got {model_id!r}"
        )
    return model_id


def _nonnegative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite nonnegative float") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite nonnegative float")
    return parsed


def _rowwise_f1(
    sampled_labels: np.ndarray,
    sampled_predictions: np.ndarray,
) -> np.ndarray:
    true_positive = np.count_nonzero(
        sampled_predictions & sampled_labels,
        axis=1,
    )
    false_positive = np.count_nonzero(
        sampled_predictions & ~sampled_labels,
        axis=1,
    )
    false_negative = np.count_nonzero(
        ~sampled_predictions & sampled_labels,
        axis=1,
    )
    denominator = (2 * true_positive) + false_positive + false_negative
    result = np.zeros(sampled_labels.shape[0], dtype=np.float64)
    np.divide(
        2 * true_positive,
        denominator,
        out=result,
        where=denominator != 0,
    )
    return result


def _paired_f1_bootstrap(
    labels: np.ndarray,
    candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    *,
    sample_count: int = PAIRED_BOOTSTRAP_SAMPLES,
    seed: int = PAIRED_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Estimate a pair-aligned confidence interval for candidate minus baseline F1."""

    labels = np.asarray(labels, dtype=bool)
    candidate_predictions = np.asarray(candidate_predictions, dtype=bool)
    baseline_predictions = np.asarray(baseline_predictions, dtype=bool)
    if (
        labels.ndim != 1
        or not labels.size
        or candidate_predictions.shape != labels.shape
        or baseline_predictions.shape != labels.shape
    ):
        raise ValueError("paired bootstrap inputs must be aligned non-empty vectors")
    if sample_count <= 0:
        raise ValueError("paired bootstrap sample count must be positive")

    random = np.random.default_rng(seed)
    deltas = np.empty(sample_count, dtype=np.float64)
    batch_size = 256
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        indexes = random.integers(
            0,
            labels.size,
            size=(stop - start, labels.size),
        )
        sampled_labels = labels[indexes]
        deltas[start:stop] = _rowwise_f1(
            sampled_labels,
            candidate_predictions[indexes],
        ) - _rowwise_f1(
            sampled_labels,
            baseline_predictions[indexes],
        )

    lower_bound, upper_bound = np.quantile(
        deltas,
        (0.025, 0.975),
        method="linear",
    )
    candidate_metrics = _binary_metrics(labels, candidate_predictions)
    baseline_metrics = _binary_metrics(labels, baseline_predictions)
    return {
        "method": "deterministic_pair_aligned_nonparametric_bootstrap",
        "resampling_unit": "held_out_mrpc_pair",
        "sample_count": sample_count,
        "seed": seed,
        "confidence_level": 0.95,
        "point_f1_delta": float(candidate_metrics["f1"] - baseline_metrics["f1"]),
        "mean_bootstrap_f1_delta": float(np.mean(deltas)),
        "lower_bound_95": float(lower_bound),
        "upper_bound_95": float(upper_bound),
        "lower_bound_strictly_positive": bool(lower_bound > 0.0),
    }


def _promotion_gate(
    *,
    f1_gain: float,
    minimum_f1_gain: float,
    paired_bootstrap: Mapping[str, Any],
) -> dict[str, bool]:
    passed_material_gain_gate = f1_gain >= minimum_f1_gain
    passed_bootstrap_confidence_gate = bool(paired_bootstrap.get("lower_bound_strictly_positive"))
    return {
        "passed_material_gain_gate": passed_material_gain_gate,
        "passed_bootstrap_confidence_gate": passed_bootstrap_confidence_gate,
        "passed_gain_gate": (passed_material_gain_gate and passed_bootstrap_confidence_gate),
    }


def _normalized_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _pair_is_psa_routed(
    pair: MRPCPair,
    neighborhoods: Mapping[str, frozenset[str]],
) -> bool:
    """Route a pair symmetrically when either endpoint nominates the other."""

    return (
        pair.sentence2 in neighborhoods[pair.sentence1]
        or pair.sentence1 in neighborhoods[pair.sentence2]
    )


def _partition_psa_routing(
    pairs: Sequence[MRPCPair],
    signatures: Mapping[str, np.ndarray],
    *,
    top_k: int,
) -> tuple[dict[int, bool], dict[str, Any]]:
    """Build routing geometry solely from the supplied evaluation partition."""

    texts = {text for pair in pairs for text in (pair.sentence1, pair.sentence2)}
    missing = sorted(text for text in texts if text not in signatures)
    if missing:
        raise ValueError("PSA signatures are missing partition sentences")
    partition_signatures = {text: signatures[text] for text in texts}
    neighborhoods, evidence = _psa_neighborhoods(
        partition_signatures,
        top_k=top_k,
    )
    routed = {pair.pair_id: _pair_is_psa_routed(pair, neighborhoods) for pair in pairs}
    hashed_neighborhoods = {
        sha256_text(text): sorted(sha256_text(neighbor) for neighbor in neighbors)
        for text, neighbors in neighborhoods.items()
    }
    routing_payload = {
        **evidence,
        "partition_isolated": True,
        "symmetric_either_direction": True,
        "pair_count": len(pairs),
        "routed_pair_count": sum(routed.values()),
        "sentence_set_sha256": _canonical_sha256(sorted(sha256_text(text) for text in texts)),
        "neighborhoods_sha256": _canonical_sha256(hashed_neighborhoods),
        "routed_pair_ids_sha256": _canonical_sha256(
            sorted(pair_id for pair_id, is_routed in routed.items() if is_routed)
        ),
    }
    return routed, {
        **routing_payload,
        "identity_sha256": _canonical_sha256(routing_payload),
    }


def _component_groups(pairs: Sequence[MRPCPair]) -> tuple[np.ndarray, int]:
    """Return deterministic sentence-component groups for cross-validation."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        union(_normalized_text(pair.sentence1), _normalized_text(pair.sentence2))
    roots = [find(_normalized_text(pair.sentence1)) for pair in pairs]
    identifiers = {root: index for index, root in enumerate(sorted(set(roots)))}
    return np.asarray([identifiers[root] for root in roots], dtype=np.int64), len(identifiers)


def _group_folds(groups: np.ndarray, fold_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = sorted({int(value) for value in groups})
    if len(unique_groups) < fold_count:
        raise ValueError("not enough sentence components for resolver cross-validation")
    fold_by_group = {
        group: int(
            hashlib.sha256(f"semantic-resolver-fold-v1:{group}".encode("utf-8")).hexdigest(),
            16,
        )
        % fold_count
        for group in unique_groups
    }
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(fold_count):
        validation = np.asarray([fold_by_group[int(group)] == fold for group in groups], dtype=bool)
        if not np.any(validation) or np.all(validation):
            raise ValueError("resolver component fold is empty")
        result.append((np.flatnonzero(~validation), np.flatnonzero(validation)))
    return result


def _standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(features.mean(axis=0), dtype=np.float64)
    scale = np.asarray(features.std(axis=0), dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _threshold_metrics(labels: np.ndarray, margins: np.ndarray, threshold: float) -> dict[str, Any]:
    return _binary_metrics(labels, margins >= threshold)


def _f1_threshold(labels: np.ndarray, margins: np.ndarray) -> tuple[float, dict[str, Any]]:
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in np.unique(margins):
        metrics = _threshold_metrics(labels, margins, float(threshold))
        key = (float(metrics["f1"]), float(metrics["precision"]), float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise ValueError("cannot learn a resolver threshold from no margins")
    return best[1], best[2]


def _selective_positive_threshold(
    labels: np.ndarray,
    margins: np.ndarray,
    *,
    binary_threshold: float,
    target_precision: float,
) -> tuple[float, dict[str, Any]]:
    candidates: list[tuple[tuple[int, float, float], float, dict[str, Any]]] = []
    for threshold in np.unique(margins):
        if threshold <= binary_threshold:
            continue
        selected = margins >= threshold
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        precision = float(np.mean(labels[selected]))
        if precision < target_precision:
            continue
        candidates.append(
            (
                (count, precision, float(threshold)),
                float(threshold),
                {
                    "count": count,
                    "precision": precision,
                    "true_positive": int(np.count_nonzero(labels[selected])),
                    "false_positive": int(np.count_nonzero(~labels[selected])),
                },
            )
        )
    if not candidates:
        raise ValueError("no positive selective threshold meets the calibration target")
    _, threshold, evidence = max(candidates, key=lambda item: item[0])
    return threshold, evidence


def _selective_negative_threshold(
    labels: np.ndarray,
    margins: np.ndarray,
    *,
    binary_threshold: float,
    target_precision: float,
) -> tuple[float, dict[str, Any]]:
    candidates: list[tuple[tuple[int, float, float], float, dict[str, Any]]] = []
    for threshold in np.unique(margins):
        if threshold >= binary_threshold:
            continue
        selected = margins <= threshold
        count = int(np.count_nonzero(selected))
        if not count:
            continue
        negative_precision = float(np.mean(~labels[selected]))
        if negative_precision < target_precision:
            continue
        candidates.append(
            (
                (count, negative_precision, -float(threshold)),
                float(threshold),
                {
                    "count": count,
                    "negative_predictive_value": negative_precision,
                    "true_negative": int(np.count_nonzero(~labels[selected])),
                    "false_negative": int(np.count_nonzero(labels[selected])),
                },
            )
        )
    if not candidates:
        raise ValueError("no negative selective threshold meets the calibration target")
    _, threshold, evidence = max(candidates, key=lambda item: item[0])
    return threshold, evidence


def _fit_ridge_ablation(
    *,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> _FittedAblation:
    folds = _group_folds(groups, MODEL_SELECTION_FOLDS)
    candidates: list[_FittedAblation] = []
    for alpha in RIDGE_ALPHA_GRID:
        oof_margins = np.zeros(len(labels), dtype=np.float64)
        for training_indexes, validation_indexes in folds:
            training_features = features[training_indexes]
            mean, scale = _standardization(training_features)
            model = RidgeClassifier(
                alpha=alpha,
                class_weight="balanced",
                solver="cholesky",
            )
            model.fit((training_features - mean) / scale, labels[training_indexes])
            oof_margins[validation_indexes] = model.decision_function(
                (features[validation_indexes] - mean) / scale
            )
        threshold, metrics = _f1_threshold(labels, oof_margins)
        mean, scale = _standardization(features)
        model = RidgeClassifier(
            alpha=alpha,
            class_weight="balanced",
            solver="cholesky",
        )
        model.fit((features - mean) / scale, labels)
        raw_coefficients = np.asarray(model.coef_, dtype=np.float64)
        coefficients = raw_coefficients[0] if raw_coefficients.ndim == 2 else raw_coefficients
        raw_intercept = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
        candidates.append(
            _FittedAblation(
                name=name,
                alpha=alpha,
                feature_mean=mean,
                feature_scale=scale,
                coefficients=coefficients,
                intercept=float(raw_intercept[0]),
                oof_margins=oof_margins,
                binary_threshold=threshold,
                oof_metrics=metrics,
            )
        )
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate.oof_metrics["f1"]),
            float(candidate.oof_metrics["precision"]),
            candidate.alpha,
        ),
    )


def _embedding_pair_matrix(
    pairs: Sequence[MRPCPair],
    embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [
            symmetric_embedding_pair_features(
                embeddings[pair.sentence1], embeddings[pair.sentence2]
            )
            for pair in pairs
        ]
    )


_GRAPH_INVENTORY_NAMES = (
    "argument_lemma",
    "argument_sense",
    "clause_role",
    "entity",
    "owner",
    "predicate_lemma",
    "predicate_sense",
    "property_key",
    "property_value",
    "quantity",
    "role",
    "role_entity",
    "role_lemma",
)


def _graph_inventory(graph: PropositionGraph) -> tuple[dict[str, Counter[str]], np.ndarray]:
    inventory = {name: Counter() for name in _GRAPH_INVENTORY_NAMES}
    propositions = graph.semantic_value.get("propositions") or []
    argument_count = 0
    for proposition in propositions:
        predicate = proposition.get("predicate") or {}
        predicate_lemma = str(predicate.get("lemma") or "").casefold()
        if predicate_lemma:
            inventory["predicate_lemma"][predicate_lemma] += 1
        for synset_id in predicate.get("synset_ids") or []:
            inventory["predicate_sense"][str(synset_id)] += 1
        clause_role = str((proposition.get("clause") or {}).get("role") or "").casefold()
        if clause_role:
            inventory["clause_role"][clause_role] += 1
        for property_name, records in (proposition.get("properties") or {}).items():
            normalized_name = str(property_name).casefold()
            inventory["property_key"][normalized_name] += 1
            if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
                for record in records:
                    inventory["property_value"][f"{normalized_name}:{_canonical_json(record)}"] += 1
        for argument in proposition.get("arguments") or []:
            argument_count += 1
            role = str(argument.get("role") or "").casefold()
            node = argument.get("node") or {}
            lemma = str(node.get("lemma") or "").casefold()
            entity = str(node.get("normalized_entity") or "").casefold()
            if role:
                inventory["role"][role] += 1
            if lemma:
                inventory["argument_lemma"][lemma] += 1
            if role and lemma:
                inventory["role_lemma"][f"{role}:{lemma}"] += 1
            if entity:
                inventory["entity"][entity] += 1
            if role and entity:
                inventory["role_entity"][f"{role}:{entity}"] += 1
            for synset_id in node.get("synset_ids") or []:
                inventory["argument_sense"][str(synset_id)] += 1
            if node.get("normalized_quantity") is not None:
                inventory["quantity"][_canonical_json(node["normalized_quantity"])] += 1
            owner = node.get("owner") or {}
            owner_entity = str(owner.get("normalized_entity") or "").casefold()
            if owner_entity:
                inventory["owner"][owner_entity] += 1
    counts = np.asarray(
        (
            len(propositions),
            argument_count,
            len(graph.value.get("edges") or []),
            len(graph.semantic_value.get("coreference_chains") or []),
            len(graph.unresolved),
        ),
        dtype=np.float64,
    )
    return inventory, counts


def _counter_pair_features(left: Counter[str], right: Counter[str]) -> tuple[float, ...]:
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    left_count = sum(left.values())
    right_count = sum(right.values())
    total = left_count + right_count
    minimum = min(left_count, right_count)
    return (
        intersection / union if union else 1.0,
        (2.0 * intersection) / total if total else 1.0,
        intersection / minimum if minimum else float(left_count == right_count),
        abs(left_count - right_count) / max(left_count, right_count, 1),
    )


def _graph_pair_features(
    left_graph: PropositionGraph,
    right_graph: PropositionGraph,
) -> np.ndarray:
    left_inventory, left_counts = _graph_inventory(left_graph)
    right_inventory, right_counts = _graph_inventory(right_graph)
    values: list[float] = []
    for name in _GRAPH_INVENTORY_NAMES:
        values.extend(_counter_pair_features(left_inventory[name], right_inventory[name]))
    scale = np.maximum(np.maximum(left_counts, right_counts), 1.0)
    values.extend((np.abs(left_counts - right_counts) / scale).tolist())
    values.extend((np.minimum(left_counts, right_counts) / scale).tolist())
    return np.asarray(values, dtype=np.float64)


def _graph_pair_matrix(
    pairs: Sequence[MRPCPair],
    graphs: Mapping[str, PropositionGraph],
) -> np.ndarray:
    return np.stack(
        [_graph_pair_features(graphs[pair.sentence1], graphs[pair.sentence2]) for pair in pairs]
    )


def _cosine_ablation(
    calibration_pairs: Sequence[MRPCPair],
    held_out_pairs: Sequence[MRPCPair],
    embeddings: Mapping[str, np.ndarray],
    held_out_routed: np.ndarray,
) -> dict[str, Any]:
    calibration_labels = np.asarray([pair.is_paraphrase for pair in calibration_pairs], dtype=bool)
    calibration_scores = np.asarray(
        [
            float(np.dot(embeddings[pair.sentence1], embeddings[pair.sentence2]))
            for pair in calibration_pairs
        ]
    )
    threshold, calibration_metrics = _f1_threshold(calibration_labels, calibration_scores)
    held_out_scores = np.asarray(
        [
            float(np.dot(embeddings[pair.sentence1], embeddings[pair.sentence2]))
            for pair in held_out_pairs
        ]
    )
    held_out_labels = np.asarray([pair.is_paraphrase for pair in held_out_pairs], dtype=bool)
    held_out_predictions = held_out_routed & (held_out_scores >= threshold)
    return {
        "feature_schema_version": "native_cosine_v1",
        "threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "held_out_metrics": _binary_metrics(held_out_labels, held_out_predictions),
    }


def _ablation_report(
    fitted: _FittedAblation,
    *,
    held_out_features: np.ndarray,
    held_out_labels: np.ndarray,
    held_out_routed: np.ndarray,
    feature_schema_version: str,
) -> tuple[dict[str, Any], np.ndarray]:
    held_out_margins = fitted.margins(held_out_features)
    predictions = held_out_routed & (held_out_margins >= fitted.binary_threshold)
    return (
        {
            "feature_schema_version": feature_schema_version,
            "alpha_grid": list(RIDGE_ALPHA_GRID),
            "selected_alpha": fitted.alpha,
            "selection": "maximum component-cross-fitted F1, then precision, then regularization",
            "binary_threshold": fitted.binary_threshold,
            "calibration_oof_metrics": dict(fitted.oof_metrics),
            "held_out_metrics": _binary_metrics(held_out_labels, predictions),
        },
        held_out_margins,
    )


def _selective_report(
    *,
    resolutions: Sequence[Any | None],
    labels: np.ndarray,
    routed: np.ndarray,
    target_precision: float,
) -> dict[str, Any]:
    states = np.asarray(
        [
            "not_routed" if resolution is None else resolution.state.value
            for resolution in resolutions
        ],
        dtype=object,
    )
    resolved = np.isin(
        states,
        (
            SemanticRelationState.LIKELY_EQUIVALENT.value,
            SemanticRelationState.LIKELY_NOT_EQUIVALENT.value,
        ),
    )
    predicted = states == SemanticRelationState.LIKELY_EQUIVALENT.value
    actual_for_resolved = labels[resolved]
    predicted_for_resolved = predicted[resolved]
    risk = (
        float(np.mean(actual_for_resolved != predicted_for_resolved)) if np.any(resolved) else None
    )
    positive_selected = predicted
    negative_selected = states == SemanticRelationState.LIKELY_NOT_EQUIVALENT.value
    positive_precision = (
        float(np.mean(labels[positive_selected])) if np.any(positive_selected) else None
    )
    negative_predictive_value = (
        float(np.mean(~labels[negative_selected])) if np.any(negative_selected) else None
    )
    positive_target_met = positive_precision is not None and positive_precision >= target_precision
    negative_target_met = (
        negative_predictive_value is not None and negative_predictive_value >= target_precision
    )
    selective_target_met = positive_target_met and negative_target_met
    routed_count = int(np.count_nonzero(routed))
    return {
        "state_counts": dict(sorted(Counter(states.tolist()).items())),
        "resolved_pair_count": int(np.count_nonzero(resolved)),
        "abstained_routed_pair_count": int(
            np.count_nonzero(states == SemanticRelationState.ABSTAIN.value)
        ),
        "coverage_all_pairs": float(np.mean(resolved)),
        "coverage_routed_pairs": (
            float(np.count_nonzero(resolved) / routed_count) if routed_count else 0.0
        ),
        "selective_risk": risk,
        "likely_equivalent_precision": positive_precision,
        "likely_equivalent_count": int(np.count_nonzero(positive_selected)),
        "likely_not_equivalent_npv": negative_predictive_value,
        "likely_not_equivalent_count": int(np.count_nonzero(negative_selected)),
        "target_precision": target_precision,
        "likely_equivalent_target_met": positive_target_met,
        "likely_not_equivalent_target_met": negative_target_met,
        "selective_target_met": selective_target_met,
        "selective_target_status": "passed" if selective_target_met else "failed",
        "authority": "none",
    }


def _validated_frozen_baseline(
    path: Path,
    held_out_pairs: Sequence[MRPCPair],
) -> tuple[dict[str, Any], np.ndarray]:
    """Authenticate the one baseline whose lane metrics are used by this benchmark."""

    report_sha256 = _file_sha256(path)
    if report_sha256 != EXPECTED_BASELINE_REPORT_SHA256:
        raise ValueError(
            "frozen baseline report SHA-256 mismatch: "
            f"expected {EXPECTED_BASELINE_REPORT_SHA256}, got {report_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_BASELINE_SCHEMA_VERSION:
        raise ValueError("frozen baseline schema version mismatch")
    execution_policy = (payload.get("verifier") or {}).get("execution_policy") or {}
    if execution_policy.get("policy_id") != EXPECTED_BASELINE_POLICY_ID:
        raise ValueError("frozen baseline execution policy mismatch")
    split_policy = payload.get("split_policy") or {}
    for key, expected in EXPECTED_BASELINE_SPLIT.items():
        if split_policy.get(key) != expected:
            raise ValueError(f"frozen baseline split policy mismatch for {key}")
    try:
        learned_gate = payload["held_out"]["profiles"]["mrpc_compatible_v1"]["learned_gate"]
        comparable_gate_metrics = learned_gate["gate_only_metrics"]
        verifier_metrics = learned_gate["metrics"]
        profile = payload["verifier"]["profiles"]["mrpc_compatible_v1"]
        baseline_traces = payload["held_out_pair_traces"]["profiles"]["mrpc_compatible_v1"][
            "learned_gate"
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen baseline is missing the MRPC-compatible lane") from exc
    if comparable_gate_metrics != EXPECTED_COMPARABLE_GATE_METRICS:
        raise ValueError("frozen baseline learned gate-only metrics mismatch")
    if verifier_metrics != EXPECTED_CROSS_LANE_VERIFIER_METRICS:
        raise ValueError("frozen baseline verifier metrics mismatch")
    if profile != {
        "contract": "document_paraphrase_compatibility",
        "hard_merge_authority": False,
    }:
        raise ValueError("frozen baseline MRPC profile contract mismatch")
    if not isinstance(baseline_traces, list):
        raise ValueError("frozen baseline learned-gate traces must be a list")
    current_alignment = [
        {"pair_id": int(pair.pair_id), "label": bool(pair.is_paraphrase)} for pair in held_out_pairs
    ]
    trace_alignment: list[dict[str, Any]] = []
    baseline_predictions: list[bool] = []
    for trace in baseline_traces:
        if not isinstance(trace, Mapping):
            raise ValueError("frozen baseline learned-gate trace is malformed")
        pair_id = trace.get("pair_id")
        label = trace.get("label")
        psa_nominated = trace.get("psa_nominated")
        native_gate_passed = trace.get("native_gate_passed")
        if isinstance(pair_id, bool) or not isinstance(pair_id, int):
            raise ValueError("frozen baseline trace pair ID is malformed")
        if not isinstance(label, bool):
            raise ValueError("frozen baseline trace label is malformed")
        if not isinstance(psa_nominated, bool) or not isinstance(native_gate_passed, bool):
            raise ValueError("frozen baseline gate prediction fields are malformed")
        trace_alignment.append({"pair_id": pair_id, "label": label})
        baseline_predictions.append(psa_nominated and native_gate_passed)
    if trace_alignment != current_alignment:
        raise ValueError(
            "current MRPC held-out pair IDs/labels do not match the authenticated baseline"
        )
    baseline_predictions_array = np.asarray(baseline_predictions, dtype=bool)
    current_labels = np.asarray(
        [pair.is_paraphrase for pair in held_out_pairs],
        dtype=bool,
    )
    if _binary_metrics(current_labels, baseline_predictions_array) != comparable_gate_metrics:
        raise ValueError("frozen baseline traces do not reproduce learned gate-only metrics")
    evidence = {
        "report_path": str(path),
        "report_sha256": report_sha256,
        "schema_version": payload["schema_version"],
        "execution_policy_id": execution_policy["policy_id"],
        "split_policy": {key: split_policy[key] for key in EXPECTED_BASELINE_SPLIT},
        "comparable_learned_gate_only_metrics": comparable_gate_metrics,
        "cross_lane_legacy_verifier_metrics": verifier_metrics,
        "held_out_pair_alignment": {
            "validated": True,
            "pair_count": len(current_alignment),
            "pair_id_label_sha256": _canonical_sha256(current_alignment),
            "baseline_gate_definition": "psa_nominated && native_gate_passed",
            "baseline_gate_prediction_sha256": _canonical_sha256(baseline_predictions),
        },
        "validated": True,
    }
    return evidence, baseline_predictions_array


@dataclass(frozen=True)
class _FrozenExperimentBundle:
    evidence: Mapping[str, Any]
    expected_scores: np.ndarray
    expected_predictions: np.ndarray
    expected_routes: np.ndarray
    expected_labels: np.ndarray


def _load_exact_json(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    path = path.resolve()
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"frozen {name} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"frozen {name} must be a JSON object")
    return payload


def _validated_frozen_experiment_bundle(
    *,
    selection_path: Path,
    contract_path: Path,
    artifact_path: Path,
    reveal_path: Path,
    artifact: CompactGenericLinearArtifact,
    held_out_pairs: Sequence[MRPCPair],
    dataset_identity: Mapping[str, Any],
    embedding_evidence: Mapping[str, Any],
    graph_evidence: Mapping[str, Any],
    calibration_routing_evidence: Mapping[str, Any],
    held_out_routing_evidence: Mapping[str, Any],
    calibration_component_count: int,
) -> _FrozenExperimentBundle:
    """Authenticate the frozen search, contract, artifact, and one-shot reveal."""

    selection = _load_exact_json(
        selection_path,
        EXPECTED_FROZEN_SELECTION_FILE_SHA256,
        name="compact-generic selection",
    )
    selection_without_hash = dict(selection)
    selection_sha256 = str(selection_without_hash.pop("frozen_selection_sha256", ""))
    if (
        selection_sha256 != EXPECTED_FROZEN_SELECTION_SHA256
        or _canonical_sha256(selection_without_hash) != selection_sha256
    ):
        raise ValueError("frozen compact-generic selection identity mismatch")
    if (
        selection.get("schema_version") != EXPECTED_FROZEN_EXPERIMENT_SCHEMA
        or selection.get("experiment_id") != EXPECTED_FROZEN_EXPERIMENT_ID
        or selection.get("phase") != "selection_frozen_before_held_out_label_materialization"
        or selection.get("diagnostic_only") is not True
        or selection.get("hard_membership_authorized") is not False
    ):
        raise ValueError("frozen compact-generic selection contract mismatch")
    if selection.get("winner") != EXPECTED_FROZEN_WINNER:
        raise ValueError("frozen compact-generic winner mismatch")
    if (selection.get("feature_dimensions") or {}).get(
        COMPACT_GENERIC_FEATURE_SCHEMA_VERSION
    ) != COMPACT_GENERIC_FEATURE_DIMENSION:
        raise ValueError("frozen compact-generic feature dimension mismatch")
    split = selection.get("split") or {}
    if (
        split.get("seed") != BLIND_SPLIT_SEED
        or split.get("calibration_pair_count") != 618
        or split.get("calibration_component_count") != calibration_component_count
        or split.get("held_out_labels_materialized") is not False
    ):
        raise ValueError("frozen compact-generic selection split mismatch")
    routing = selection.get("routing") or {}
    expected_routing_subset = {
        "kind": "partition_local_symmetric_psa_top_k_union",
        "direction_policy": "route if either sentence nominates the other",
        "top_k": PSA_SIGNATURE_CANDIDATE_LIMIT,
        "calibration_routed_pair_count": 603,
    }
    if any(routing.get(key) != value for key, value in expected_routing_subset.items()):
        raise ValueError("frozen compact-generic selection routing mismatch")
    frozen_calibration_neighborhood = routing.get("calibration_neighborhood_evidence") or {}
    for key in ("implementation", "sentence_count", "signature_dimension", "top_k"):
        if frozen_calibration_neighborhood.get(key) != calibration_routing_evidence.get(key):
            raise ValueError("current calibration routing does not match the frozen experiment")
    if calibration_routing_evidence.get("routed_pair_count") != 603:
        raise ValueError("current calibration routed-pair count drifted")

    dataset_files = {file["name"]: file["sha256"] for file in dataset_identity.get("files") or []}
    current_input_identity = {
        "mrpc_corpus_sha256": dataset_files.get("msr_paraphrase_data.txt"),
        "mrpc_test_sha256": dataset_files.get("msr_paraphrase_test.txt"),
        "embedding_cache_sha256": embedding_evidence.get("cache_sha256"),
        "graph_cache_sha256": graph_evidence.get("cache_file_sha256"),
    }
    if selection.get("input_identity") != current_input_identity:
        raise ValueError("current input identities do not match the frozen experiment")
    frozen_graph = (selection.get("evidence") or {}).get("graph") or {}
    for key in (
        "cache_file_sha256",
        "parser_identity",
        "projected_graph_schema_version",
        "fact_extractor",
        "source_graph_schema_version",
        "source_evidence_schema_versions",
    ):
        if frozen_graph.get(key) != graph_evidence.get(key):
            raise ValueError(f"current graph evidence drifted for {key}")

    contract = artifact.contract
    if (
        artifact.contract_file_sha256 != EXPECTED_FROZEN_CONTRACT_FILE_SHA256
        or artifact.contract_sha256 != EXPECTED_FROZEN_CONTRACT_SHA256
        or artifact.artifact_file_sha256 != EXPECTED_FROZEN_ARTIFACT_SHA256
    ):
        raise ValueError("loaded compact-generic artifact identity mismatch")
    if (
        contract.get("experiment_id") != EXPECTED_FROZEN_EXPERIMENT_ID
        or contract.get("frozen_selection_sha256") != selection_sha256
        or contract.get("winner") != selection.get("winner")
        or contract.get("input_identity") != selection.get("input_identity")
        or contract.get("routing") != selection.get("routing")
    ):
        raise ValueError("frozen calibration contract does not bind the selection")
    dependency_identity = contract.get("dependency_and_source_identity") or {}
    package_identity = dependency_identity.get("packages") or {}
    current_packages = {
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    if package_identity != current_packages:
        raise ValueError("runtime package identity differs from the frozen artifact")
    if dependency_identity.get("python") != platform.python_version():
        raise ValueError("Python runtime identity differs from the frozen artifact")
    frozen_sources = dependency_identity.get("sources") or {}
    experiment_source = frozen_sources.get("experiment_script") or {}
    graph_source = frozen_sources.get("proposition_graph") or {}
    if (
        experiment_source.get("sha256") != EXPECTED_FROZEN_EXPERIMENT_SOURCE_SHA256
        or _file_sha256(
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "experiment_mrpc_semantic_relation_models.py"
        )
        != EXPECTED_FROZEN_EXPERIMENT_SOURCE_SHA256
        or graph_source.get("sha256") != EXPECTED_FROZEN_PROPOSITION_GRAPH_SOURCE_SHA256
        or _file_sha256(
            REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "proposition_graph.py"
        )
        != EXPECTED_FROZEN_PROPOSITION_GRAPH_SOURCE_SHA256
    ):
        raise ValueError("frozen experiment source identity mismatch")

    reveal = _load_exact_json(
        reveal_path,
        EXPECTED_FROZEN_REVEAL_FILE_SHA256,
        name="compact-generic held-out reveal",
    )
    if (
        reveal.get("schema_version") != EXPECTED_FROZEN_EXPERIMENT_SCHEMA
        or reveal.get("experiment_id") != EXPECTED_FROZEN_EXPERIMENT_ID
        or reveal.get("phase") != "single_held_out_reveal"
        or reveal.get("diagnostic_only") is not True
        or reveal.get("hard_membership_authorized") is not False
        or reveal.get("selection_file_sha256") != EXPECTED_FROZEN_SELECTION_FILE_SHA256
        or reveal.get("frozen_selection_sha256") != selection_sha256
        or reveal.get("winner") != EXPECTED_FROZEN_WINNER
    ):
        raise ValueError("frozen compact-generic reveal contract mismatch")
    reveal_contract = reveal.get("calibration_contract") or {}
    reveal_artifact = reveal.get("fitted_artifact") or {}
    if (
        reveal_contract.get("file_sha256") != EXPECTED_FROZEN_CONTRACT_FILE_SHA256
        or reveal_contract.get("contract_sha256") != EXPECTED_FROZEN_CONTRACT_SHA256
        or reveal_artifact.get("sha256") != EXPECTED_FROZEN_ARTIFACT_SHA256
        or (reveal.get("held_out") or {}).get("metrics") != EXPECTED_FROZEN_HELD_OUT_METRICS
        or (reveal.get("held_out") or {}).get("pair_count") != len(held_out_pairs)
        or (reveal.get("held_out") or {}).get("routed_pair_count") != 1065
    ):
        raise ValueError("frozen compact-generic reveal evidence mismatch")
    frozen_held_routing = (reveal.get("routing") or {}).get("held_out_neighborhood_evidence") or {}
    for key in ("implementation", "sentence_count", "signature_dimension", "top_k"):
        if frozen_held_routing.get(key) != held_out_routing_evidence.get(key):
            raise ValueError("current held-out routing does not match the frozen reveal")
    traces = reveal.get("held_out_pair_traces")
    if not isinstance(traces, list) or len(traces) != len(held_out_pairs):
        raise ValueError("frozen compact-generic reveal traces are missing")
    expected_scores: list[float] = []
    expected_predictions: list[bool] = []
    expected_routes: list[bool] = []
    expected_labels: list[bool] = []
    for pair, trace in zip(held_out_pairs, traces, strict=True):
        if trace.get("pair_id") != pair.pair_id or trace.get("label") is not pair.is_paraphrase:
            raise ValueError("frozen compact-generic reveal pair alignment mismatch")
        score = trace.get("resolver_score")
        prediction = trace.get("resolver_prediction")
        is_routed = trace.get("routed")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not isinstance(prediction, bool)
            or not isinstance(is_routed, bool)
        ):
            raise ValueError("frozen compact-generic reveal trace is malformed")
        expected_scores.append(float(score))
        expected_predictions.append(prediction)
        expected_routes.append(is_routed)
        expected_labels.append(bool(trace["label"]))
    evidence = {
        "selection": {
            "path": str(selection_path.resolve()),
            "file_sha256": EXPECTED_FROZEN_SELECTION_FILE_SHA256,
            "selection_sha256": selection_sha256,
        },
        "calibration_contract": {
            "path": str(contract_path.resolve()),
            "file_sha256": artifact.contract_file_sha256,
            "contract_sha256": artifact.contract_sha256,
        },
        "numeric_artifact": {
            "path": str(artifact_path.resolve()),
            "file_sha256": artifact.artifact_file_sha256,
        },
        "held_out_reveal": {
            "path": str(reveal_path.resolve()),
            "file_sha256": EXPECTED_FROZEN_REVEAL_FILE_SHA256,
            "metrics": EXPECTED_FROZEN_HELD_OUT_METRICS,
        },
        "winner": EXPECTED_FROZEN_WINNER,
        "input_identity": current_input_identity,
        "source_identity": {
            "experiment_script_sha256": EXPECTED_FROZEN_EXPERIMENT_SOURCE_SHA256,
            "proposition_graph_sha256": (EXPECTED_FROZEN_PROPOSITION_GRAPH_SOURCE_SHA256),
        },
        "validated": True,
        "diagnostic_only": True,
        "hard_membership_authorized": False,
    }
    return _FrozenExperimentBundle(
        evidence=evidence,
        expected_scores=np.asarray(expected_scores, dtype=np.float64),
        expected_predictions=np.asarray(expected_predictions, dtype=bool),
        expected_routes=np.asarray(expected_routes, dtype=bool),
        expected_labels=np.asarray(expected_labels, dtype=bool),
    )


def _validate_frozen_controlling_parity(
    *,
    scores: np.ndarray,
    predictions: np.ndarray,
    routes: np.ndarray,
    labels: np.ndarray,
    bundle: _FrozenExperimentBundle,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=bool)
    routes = np.asarray(routes, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    expected_shape = bundle.expected_scores.shape
    if any(value.shape != expected_shape for value in (scores, predictions, routes, labels)):
        raise ValueError("controlling compact-generic parity vectors are misaligned")
    if not np.array_equal(routes, bundle.expected_routes):
        raise ValueError("controlling compact-generic routing differs from frozen reveal")
    if not np.array_equal(labels, bundle.expected_labels):
        raise ValueError("controlling compact-generic labels differ from frozen reveal")
    if not np.array_equal(scores, bundle.expected_scores):
        maximum_delta = float(np.max(np.abs(scores - bundle.expected_scores)))
        raise ValueError(
            "controlling compact-generic scores differ from frozen reveal; "
            f"maximum absolute delta {maximum_delta}"
        )
    if not np.array_equal(predictions, bundle.expected_predictions):
        raise ValueError("controlling compact-generic predictions differ from frozen reveal")
    metrics = _binary_metrics(labels, predictions)
    if metrics != EXPECTED_FROZEN_HELD_OUT_METRICS:
        raise ValueError("controlling compact-generic metrics differ from frozen reveal")
    return {
        "exact_score_parity": True,
        "maximum_absolute_score_delta": 0.0,
        "exact_prediction_parity": True,
        "exact_route_parity": True,
        "exact_label_alignment": True,
        "score_sha256": _canonical_sha256([float(value) for value in scores]),
        "prediction_sha256": _canonical_sha256([bool(value) for value in predictions]),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mrpc-dir", type=Path, default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split"
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sense-cache", type=Path, default=DEFAULT_SENSE_CACHE)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--graph-cache-sha256", default=PINNED_GRAPH_CACHE_SHA256)
    parser.add_argument("--graph-parser-identity", default=PINNED_GRAPH_PARSER_IDENTITY)
    parser.add_argument(
        "--pinned-guide-embeddings", type=Path, default=DEFAULT_PINNED_GUIDE_EMBEDDINGS
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--psa-top-k", type=int, default=PSA_SIGNATURE_CANDIDATE_LIMIT)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--frozen-selection", type=Path, default=DEFAULT_FROZEN_SELECTION)
    parser.add_argument("--frozen-contract", type=Path, default=DEFAULT_FROZEN_CONTRACT)
    parser.add_argument("--frozen-artifact", type=Path, default=DEFAULT_FROZEN_ARTIFACT)
    parser.add_argument("--frozen-reveal", type=Path, default=DEFAULT_FROZEN_REVEAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--minimum-f1-gain",
        type=_nonnegative_finite_float,
        default=DEFAULT_MINIMUM_F1_GAIN,
        help=(
            "minimum F1 gain over the comparable frozen learned gate-only lane; "
            "the legacy verifier lane is never used for this gate"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mrpc_dir = args.mrpc_dir.resolve()
    dataset_identity = _dataset_identity(mrpc_dir)
    source_identity = _benchmark_source_identity()
    pairs = load_mrpc_pairs(mrpc_dir)
    split = split_mrpc_by_sentence_component(pairs, seed=BLIND_SPLIT_SEED)
    calibration_all = [pair for pair in pairs if split[pair.pair_id] == "calibration"]
    held_out_pairs = [pair for pair in pairs if split[pair.pair_id] == "held_out"]
    texts = sorted(
        {text for pair in pairs for text in (pair.sentence1, pair.sentence2)},
        key=lambda text: (sha256_text(text), text),
    )

    embeddings, _cached_guides, embedding_evidence = _load_cache_evidence(
        args.cache.resolve(), pairs
    )
    embedding_model_id = _validated_embedding_model_id(embedding_evidence)
    pinned_guides, calibration, frame_router, frame_evidence = _load_pinned_frame(
        args.pinned_guide_embeddings.resolve(), args.calibration.resolve()
    )
    signatures, _labels, signature_evidence = _project_representations(
        embeddings, pinned_guides, frame_router, calibration
    )
    calibration_routed, calibration_neighborhood_evidence = _partition_psa_routing(
        calibration_all,
        signatures,
        top_k=args.psa_top_k,
    )
    held_out_routed_by_id, held_out_neighborhood_evidence = _partition_psa_routing(
        held_out_pairs,
        signatures,
        top_k=args.psa_top_k,
    )
    routed = {**calibration_routed, **held_out_routed_by_id}
    calibration_pairs = [pair for pair in calibration_all if calibration_routed[pair.pair_id]]
    held_out_routed = np.asarray(
        [held_out_routed_by_id[pair.pair_id] for pair in held_out_pairs],
        dtype=bool,
    )
    calibration_labels = np.asarray([pair.is_paraphrase for pair in calibration_pairs], dtype=bool)
    groups, ablation_component_count = _component_groups(calibration_pairs)
    _all_calibration_groups, calibration_component_count = _component_groups(calibration_all)

    graphs, graph_evidence = _build_or_load_graphs(
        texts,
        sense_path=args.sense_cache.resolve(),
        cache_path=args.graph_cache.resolve(),
        expected_cache_sha256=args.graph_cache_sha256,
        expected_parser_identity=args.graph_parser_identity,
    )
    calibration_embedding_features = _embedding_pair_matrix(calibration_pairs, embeddings)
    held_out_embedding_features = _embedding_pair_matrix(held_out_pairs, embeddings)
    calibration_graph_features = _graph_pair_matrix(calibration_pairs, graphs)
    held_out_graph_features = _graph_pair_matrix(held_out_pairs, graphs)
    calibration_compact_features = compact_generic_pair_matrix(
        calibration_all,
        embeddings=embeddings,
        signatures=signatures,
        graphs=graphs,
    )
    held_out_compact_features = compact_generic_pair_matrix(
        held_out_pairs,
        embeddings=embeddings,
        signatures=signatures,
        graphs=graphs,
    )
    frozen_artifact = load_compact_generic_linear_artifact(
        artifact_path=args.frozen_artifact.resolve(),
        contract_path=args.frozen_contract.resolve(),
        expected_artifact_sha256=EXPECTED_FROZEN_ARTIFACT_SHA256,
        expected_contract_file_sha256=EXPECTED_FROZEN_CONTRACT_FILE_SHA256,
    )
    frozen_bundle = _validated_frozen_experiment_bundle(
        selection_path=args.frozen_selection.resolve(),
        contract_path=args.frozen_contract.resolve(),
        artifact_path=args.frozen_artifact.resolve(),
        reveal_path=args.frozen_reveal.resolve(),
        artifact=frozen_artifact,
        held_out_pairs=held_out_pairs,
        dataset_identity=dataset_identity,
        embedding_evidence=embedding_evidence,
        graph_evidence=graph_evidence,
        calibration_routing_evidence=calibration_neighborhood_evidence,
        held_out_routing_evidence=held_out_neighborhood_evidence,
        calibration_component_count=calibration_component_count,
    )
    if calibration_compact_features.shape != (
        len(calibration_all),
        COMPACT_GENERIC_FEATURE_DIMENSION,
    ):
        raise ValueError("calibration compact-generic feature matrix shape mismatch")

    embedding_ablation = _fit_ridge_ablation(
        name="symmetric_embedding",
        features=calibration_embedding_features,
        labels=calibration_labels,
        groups=groups,
    )
    graph_ablation = _fit_ridge_ablation(
        name="symmetric_embedding_plus_graph_overlap",
        features=np.column_stack((calibration_embedding_features, calibration_graph_features)),
        labels=calibration_labels,
        groups=groups,
    )
    learned_candidates = (
        (
            embedding_ablation,
            calibration_embedding_features,
            held_out_embedding_features,
            SYMMETRIC_PAIR_FEATURE_SCHEMA_VERSION,
        ),
        (
            graph_ablation,
            np.column_stack((calibration_embedding_features, calibration_graph_features)),
            np.column_stack((held_out_embedding_features, held_out_graph_features)),
            f"{SYMMETRIC_PAIR_FEATURE_SCHEMA_VERSION}+{GRAPH_PAIR_FEATURE_SCHEMA_VERSION}",
        ),
    )
    (
        selected_ablation,
        _selected_calibration_features,
        selected_held_out_features,
        selected_feature_schema,
    ) = max(
        learned_candidates,
        key=lambda candidate: (
            float(candidate[0].oof_metrics["f1"]),
            float(candidate[0].oof_metrics["precision"]),
            candidate[0].name,
        ),
    )
    likely_equivalent_threshold, positive_threshold_evidence = _selective_positive_threshold(
        calibration_labels,
        selected_ablation.oof_margins,
        binary_threshold=selected_ablation.binary_threshold,
        target_precision=SELECTIVE_TARGET_PRECISION,
    )
    likely_not_equivalent_threshold, negative_threshold_evidence = _selective_negative_threshold(
        calibration_labels,
        selected_ablation.oof_margins,
        binary_threshold=selected_ablation.binary_threshold,
        target_precision=SELECTIVE_TARGET_PRECISION,
    )

    training_identity_payload = {
        "split_seed": BLIND_SPLIT_SEED,
        "candidate_pair_ids": [pair.pair_id for pair in calibration_pairs],
        "candidate_labels": [pair.is_paraphrase for pair in calibration_pairs],
        "mrpc_dataset_identity_sha256": dataset_identity["identity_sha256"],
        "embedding_cache_sha256": embedding_evidence["cache_sha256"],
        "graph_cache_sha256": args.graph_cache_sha256,
        "selected_feature_schema": selected_feature_schema,
        "calibration_psa_neighborhood_identity": calibration_neighborhood_evidence,
        "fold_count": MODEL_SELECTION_FOLDS,
    }
    training_split_identity = hashlib.sha256(
        _canonical_json(training_identity_payload).encode("utf-8")
    ).hexdigest()
    ablation_thresholds = SemanticResolverThresholds(
        binary_equivalence_margin=selected_ablation.binary_threshold,
        likely_equivalence_margin=likely_equivalent_threshold,
        likely_not_equivalence_margin=likely_not_equivalent_threshold,
        selective_target_precision=SELECTIVE_TARGET_PRECISION,
    )
    ablation_provenance = SemanticResolverProvenance(
        resolver_id="dynamic_ridge_ablation_v1",
        embedding_model_id=embedding_model_id,
        training_split_identity=training_split_identity,
        training_pair_count=len(calibration_pairs),
        feature_schema_version=selected_feature_schema,
    )
    ablation_resolver = SymmetricEmbeddingRelationResolver(
        feature_mean=selected_ablation.feature_mean,
        feature_scale=selected_ablation.feature_scale,
        coefficients=selected_ablation.coefficients,
        intercept=selected_ablation.intercept,
        thresholds=ablation_thresholds,
        provenance=ablation_provenance,
        extra_artifact_identity={
            "ridge_alpha": selected_ablation.alpha,
            "selected_ablation": selected_ablation.name,
            "fold_count": MODEL_SELECTION_FOLDS,
            "alpha_grid": list(RIDGE_ALPHA_GRID),
            "embedding_cache_sha256": embedding_evidence["cache_sha256"],
            "mrpc_dataset_identity_sha256": dataset_identity["identity_sha256"],
            "graph_cache_sha256": args.graph_cache_sha256,
            "routing_top_k": args.psa_top_k,
            "calibration_routing_identity_sha256": (
                calibration_neighborhood_evidence["identity_sha256"]
            ),
        },
    )
    ablation_resolutions = [
        (
            ablation_resolver.resolve_features(selected_held_out_features[index])
            if routed[pair.pair_id]
            else None
        )
        for index, pair in enumerate(held_out_pairs)
    ]

    resolver = SymmetricEmbeddingRelationResolver(
        feature_mean=frozen_artifact.feature_mean,
        feature_scale=frozen_artifact.feature_scale,
        coefficients=frozen_artifact.coefficients,
        intercept=frozen_artifact.intercept,
        thresholds=frozen_artifact.thresholds,
        provenance=SemanticResolverProvenance(
            resolver_id=RESOLVER_ID,
            embedding_model_id=embedding_model_id,
            training_split_identity=EXPECTED_FROZEN_SELECTION_SHA256,
            training_pair_count=603,
            feature_schema_version=COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
        ),
        extra_artifact_identity={
            "source": "authenticated_frozen_compact_generic_experiment",
            "numeric_artifact_sha256": frozen_artifact.artifact_file_sha256,
            "calibration_contract_sha256": frozen_artifact.contract_sha256,
            "selection_sha256": EXPECTED_FROZEN_SELECTION_SHA256,
            "held_out_reveal_sha256": EXPECTED_FROZEN_REVEAL_FILE_SHA256,
            "routing_separate_from_resolver": True,
            "hard_membership_authorized": False,
        },
    )
    held_out_margins = frozen_artifact.margins(held_out_compact_features)
    resolutions = [
        (
            resolver.resolve_features(held_out_compact_features[index])
            if routed[pair.pair_id]
            else None
        )
        for index, pair in enumerate(held_out_pairs)
    ]

    # The fitted artifact, binary threshold, selective thresholds, and ablation
    # choice are frozen before held-out labels are accessed for evaluation below.
    held_out_labels = np.asarray([pair.is_paraphrase for pair in held_out_pairs], dtype=bool)
    embedding_report, _embedding_held_out_margins = _ablation_report(
        embedding_ablation,
        held_out_features=held_out_embedding_features,
        held_out_labels=held_out_labels,
        held_out_routed=held_out_routed,
        feature_schema_version=SYMMETRIC_PAIR_FEATURE_SCHEMA_VERSION,
    )
    graph_report, _graph_held_out_margins = _ablation_report(
        graph_ablation,
        held_out_features=np.column_stack((held_out_embedding_features, held_out_graph_features)),
        held_out_labels=held_out_labels,
        held_out_routed=held_out_routed,
        feature_schema_version=(
            f"{SYMMETRIC_PAIR_FEATURE_SCHEMA_VERSION}+{GRAPH_PAIR_FEATURE_SCHEMA_VERSION}"
        ),
    )
    cosine_report = _cosine_ablation(calibration_pairs, held_out_pairs, embeddings, held_out_routed)
    selected_ablation_report = (
        graph_report if selected_ablation.name == graph_ablation.name else embedding_report
    )
    ablation_selective_report = _selective_report(
        resolutions=ablation_resolutions,
        labels=held_out_labels,
        routed=held_out_routed,
        target_precision=SELECTIVE_TARGET_PRECISION,
    )
    selective_report = _selective_report(
        resolutions=resolutions,
        labels=held_out_labels,
        routed=held_out_routed,
        target_precision=SELECTIVE_TARGET_PRECISION,
    )

    baseline_path = args.baseline_report.resolve()
    baseline_evidence, baseline_gate_predictions = _validated_frozen_baseline(
        baseline_path,
        held_out_pairs,
    )
    comparable_baseline_metrics = baseline_evidence["comparable_learned_gate_only_metrics"]
    cross_lane_verifier_metrics = baseline_evidence["cross_lane_legacy_verifier_metrics"]
    selected_predictions = held_out_routed & (
        held_out_margins >= frozen_artifact.thresholds.binary_equivalence_margin
    )
    selected_metrics = _binary_metrics(held_out_labels, selected_predictions)
    frozen_parity = _validate_frozen_controlling_parity(
        scores=held_out_margins,
        predictions=selected_predictions,
        routes=held_out_routed,
        labels=held_out_labels,
        bundle=frozen_bundle,
    )
    compact_report = {
        "feature_schema_version": COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
        "feature_dimension": COMPACT_GENERIC_FEATURE_DIMENSION,
        "selection": "authenticated frozen calibration-only model search winner",
        "selected_family": "ridge",
        "selected_alpha": 1000.0,
        "binary_threshold": (frozen_artifact.thresholds.binary_equivalence_margin),
        "calibration_oof_metrics": EXPECTED_FROZEN_WINNER["oof_metrics"],
        "held_out_metrics": selected_metrics,
        "frozen_parity": frozen_parity,
        "diagnostic_only": True,
        "hard_membership_authorized": False,
    }
    comparable_f1_gain = float(selected_metrics["f1"] - comparable_baseline_metrics["f1"])
    paired_bootstrap = _paired_f1_bootstrap(
        held_out_labels,
        selected_predictions,
        baseline_gate_predictions,
    )
    if paired_bootstrap["point_f1_delta"] != comparable_f1_gain:
        raise ValueError("paired bootstrap point estimate does not match the F1 comparison")
    promotion_gate = _promotion_gate(
        f1_gain=comparable_f1_gain,
        minimum_f1_gain=args.minimum_f1_gain,
        paired_bootstrap=paired_bootstrap,
    )
    passed_material_gain_gate = promotion_gate["passed_material_gain_gate"]
    passed_bootstrap_confidence_gate = promotion_gate["passed_bootstrap_confidence_gate"]
    passed_gain_gate = promotion_gate["passed_gain_gate"]

    traces = []
    for pair, margin, resolution in zip(held_out_pairs, held_out_margins, resolutions, strict=True):
        traces.append(
            {
                "pair_id": pair.pair_id,
                "label": pair.is_paraphrase,
                "routed": routed[pair.pair_id],
                "margin": float(margin),
                "binary_prediction": bool(
                    routed[pair.pair_id]
                    and margin >= frozen_artifact.thresholds.binary_equivalence_margin
                ),
                "selective_resolution": (
                    resolution.trace_dict() if resolution is not None else None
                ),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "semantic_relation_resolver_artifact.json"
    artifact_envelope = resolver.export_artifact()
    _atomic_json(artifact_path, artifact_envelope)
    replayed_resolver = SymmetricEmbeddingRelationResolver.from_artifact(
        json.loads(artifact_path.read_text(encoding="utf-8"))
    )
    if replayed_resolver.artifact_sha256 != resolver.artifact_sha256:
        raise ValueError("persisted semantic resolver artifact did not replay exactly")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_class": "post_hoc_iterative_development_diagnostic",
        "eligible_for_blind_performance_claim": False,
        "resolver": {
            **resolver.artifact_identity(),
            "selected_ablation": "frozen_compact_generic_v1",
            "selection_basis": (
                "authenticated frozen calibration-only 246-candidate model search; "
                "held-out scores and predictions must exactly replay the one-shot reveal"
            ),
            "swap_invariant": True,
            "directional_limitation": (
                "forward and reverse scores are identical; this first lane is a symmetric "
                "pair classifier, not a directional NLI model"
            ),
            "membership_effect": "none",
            "frozen_numeric_artifact": frozen_artifact.identity(),
            "frozen_experiment": frozen_bundle.evidence,
            "artifact": {
                "path": str(artifact_path),
                "canonical_payload_sha256": resolver.artifact_sha256,
                "file_sha256": _file_sha256(artifact_path),
                "load_replay_verified": True,
            },
        },
        "split_policy": {
            "seed": BLIND_SPLIT_SEED,
            "unit": "connected_component_of_normalized_sentence_identity",
            "calibration_pair_count": len(calibration_all),
            "calibration_routed_pair_count": len(calibration_pairs),
            "calibration_routed_positive_count": int(np.count_nonzero(calibration_labels)),
            "calibration_component_count": calibration_component_count,
            "calibration_routed_component_count_for_dynamic_ablations": (ablation_component_count),
            "model_selection_folds": MODEL_SELECTION_FOLDS,
            "held_out_pair_count": len(held_out_pairs),
            "held_out_labels_materialized_by_shared_loader": True,
            "held_out_labels_accessed_by_fit_or_threshold_selection": False,
            "held_out_labels_previously_inspected_in_repository_development": True,
        },
        "routing": {
            "kind": "partition_isolated_symmetric_psa_signature_top_k_neighborhood",
            "top_k": args.psa_top_k,
            "calibration": calibration_neighborhood_evidence,
            "held_out": held_out_neighborhood_evidence,
            "held_out_routed_pair_count": int(np.count_nonzero(held_out_routed)),
            "held_out_routed_positive_count": int(
                np.count_nonzero(held_out_routed & held_out_labels)
            ),
        },
        "evidence": {
            "embedding_cache": embedding_evidence,
            "mrpc_dataset": dataset_identity,
            "frame": frame_evidence,
            "signature_distribution": signature_evidence,
            "psa_neighborhoods": {
                "calibration": calibration_neighborhood_evidence,
                "held_out": held_out_neighborhood_evidence,
            },
            "graph": graph_evidence,
            "runtime_versions": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "scipy": scipy.__version__,
                "python": platform.python_version(),
            },
            "benchmark_source": source_identity,
            "graph_usage": (
                "generic multiset-overlap features in the learned diagnostic scorer and "
                "dynamic ablations; graph evidence grants no resolver or membership authority"
            ),
        },
        "calibration": {
            "controlling_artifact": {
                "feature_schema_version": COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
                "selected_alpha": 1000.0,
                "binary_threshold": (frozen_artifact.thresholds.binary_equivalence_margin),
                "threshold_contract": (frozen_artifact.contract["threshold_contract"]),
                "calibration_oof_metrics": EXPECTED_FROZEN_WINNER["oof_metrics"],
                "source": "authenticated frozen calibration contract",
            },
            "dynamic_ablation_alpha_grid": list(RIDGE_ALPHA_GRID),
            "dynamic_ablation_selected_alpha": selected_ablation.alpha,
            "dynamic_ablation_binary_threshold_source": (
                "component-cross-fitted calibration margins"
            ),
            "selective_target_precision": SELECTIVE_TARGET_PRECISION,
            "dynamic_ablation_positive_selective_threshold_evidence": (positive_threshold_evidence),
            "dynamic_ablation_negative_selective_threshold_evidence": (negative_threshold_evidence),
        },
        "ablations": {
            "frozen_compact_generic_controlling": compact_report,
            "native_cosine": cosine_report,
            "symmetric_embedding": embedding_report,
            "symmetric_embedding_plus_graph_overlap": graph_report,
            "dynamic_ridge_selected": {
                **selected_ablation_report,
                "name": selected_ablation.name,
                "selective": ablation_selective_report,
                "controlling": False,
            },
        },
        "held_out": {
            "binary_metrics": selected_metrics,
            "selective": selective_report,
            "comparable_learned_gate_only_baseline": {
                "metrics": comparable_baseline_metrics,
                "comparison_lane": "candidate_relation_binary_f1",
                "comparable": True,
            },
            "frozen_legacy_verifier_reference": {
                "metrics": cross_lane_verifier_metrics,
                "comparison_lane": "deterministic_verifier_after_candidate_gate",
                "comparable": False,
                "reason": (
                    "verifier precision/coverage is a different dashboard lane; "
                    "its 0.407 F1 cannot be used as the learned classifier baseline"
                ),
            },
            "frozen_baseline_validation": baseline_evidence,
            "f1_gain_over_comparable_learned_gate_only": comparable_f1_gain,
            "paired_bootstrap_f1_difference": paired_bootstrap,
            "minimum_required_f1_gain": args.minimum_f1_gain,
            "passed_material_gain_gate": passed_material_gain_gate,
            "passed_bootstrap_confidence_gate": passed_bootstrap_confidence_gate,
            "passed_gain_gate": passed_gain_gate,
            "frozen_experiment_parity": frozen_parity,
            "controlling_lane": "frozen_compact_generic_v1",
            "promotion_requirement": (
                "material F1 gain and paired-bootstrap 95% lower bound strictly above zero"
            ),
        },
        "caveats": [
            "MRPC held-out labels were inspected during earlier repository development.",
            "The result is a diagnostic development score, not a fresh blind claim.",
            "The fitted model is MRPC-calibrated and requires cross-dataset and OOD evaluation.",
            "A symmetric score does not establish directional textual entailment.",
            "Pinned cached graph evidence feeds diagnostic scoring and is not proof authority.",
            "Compact generic features are learned diagnostic evidence, not proof authority.",
            "The 0.407 legacy verifier F1 is cross-lane context, not a classifier baseline.",
            "No resolver output creates or upgrades paraphrase membership.",
        ],
        "held_out_pair_traces": traces,
    }
    output_path = output_dir / "semantic_relation_resolver_report.json"
    _atomic_json(output_path, report)
    summary = {
        "report": str(output_path),
        "artifact_sha256": resolver.artifact_sha256,
        "frozen_numeric_artifact_sha256": frozen_artifact.artifact_file_sha256,
        "selected_alpha": 1000.0,
        "selected_ablation": "frozen_compact_generic_v1",
        "binary_metrics": selected_metrics,
        "selective": selective_report,
        "comparable_gate_only_baseline_f1": comparable_baseline_metrics["f1"],
        "cross_lane_legacy_verifier_f1": cross_lane_verifier_metrics["f1"],
        "f1_gain_over_comparable_gate_only": comparable_f1_gain,
        "paired_bootstrap_f1_difference": paired_bootstrap,
        "minimum_required_f1_gain": args.minimum_f1_gain,
        "passed_material_gain_gate": passed_material_gain_gate,
        "passed_bootstrap_confidence_gate": passed_bootstrap_confidence_gate,
        "passed_gain_gate": passed_gain_gate,
        "selective_target_status": selective_report["selective_target_status"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed_gain_gate:
        raise SystemExit(
            "semantic resolver did not clear the material-gain and paired-bootstrap "
            "promotion gate against the comparable learned gate-only lane; "
            "do not promote this benchmark lane"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
