#!/usr/bin/env python3
"""Calibration-only model search for the diagnostic MRPC relation lane.

This experiment is intentionally separate from the resolver benchmark.  The
``select`` phase never materializes held-out labels: it reads sentence IDs and
texts, computes the frozen sentence-component split and fixed PSA route, then
loads labels only for calibration pair IDs.  It writes one frozen selection
artifact.  The ``reveal`` phase accepts that artifact, fits exactly the selected
configuration, and materializes held-out labels once for evaluation.

All features are swap-invariant and generic.  No lemma, phrase, or MRPC-example
rules are present.  Outputs are diagnostic and have no membership authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

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
from scripts.research.mrpc.run_mrpc_semantic_relation_resolver_benchmark import (  # noqa: E402
    _component_groups,
    _graph_pair_matrix,
)

from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    _binary_metrics,
    sha256_text,
    split_mrpc_by_sentence_component,
)

EXPERIMENT_SCHEMA_VERSION = "mrpc_semantic_relation_model_search_v1"
EXPERIMENT_ID = "generic_symmetric_models_component_cv_v1"
MODEL_SELECTION_FOLDS = 5
FOLD_SEED = "mrpc-semantic-resolver-model-search-fold-v1"
SELECTIVE_TARGET_PRECISION = 0.95
BOOTSTRAP_SEED = 2_026_080_3
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_BASELINE_REPORT_SHA256 = "c40eea84bce3eea527198ef1e31de69c2eb7285b3be55f2c6616e35c747ebffe"
EXPECTED_BASELINE_SCHEMA_VERSION = "aiidea_mrpc_proposition_graph_benchmark_v2"
EXPECTED_BASELINE_POLICY_ID = "legacy_etpc_mrpc_diagnostic_v1"
EXPECTED_BASELINE_METRICS = {
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
DEFAULT_BASELINE_REPORT = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_rule_policy_ablation_2026-08-03"
    / "legacy"
    / "mrpc_proposition_graph_report.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_semantic_relation_model_search_2026-08-03"
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class PairInput:
    pair_id: int
    sentence1: str
    sentence2: str


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    family: str
    feature_set: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateResult:
    spec: CandidateSpec
    oof_scores: np.ndarray
    threshold: float
    metrics: Mapping[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pair_inputs(mrpc_dir: Path) -> list[PairInput]:
    """Load pair inputs without reading the Quality field."""

    corpus_path = mrpc_dir.parent / "msr_paraphrase_data.txt"
    test_path = mrpc_dir.parent / "msr_paraphrase_test.txt"
    sentence_by_id: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        sentence_id_index = header.index("Sentence ID")
        string_index = header.index("String")
        for row in reader:
            sentence_by_id[row[sentence_id_index].strip()] = row[string_index]

    pairs: list[PairInput] = []
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        first_id_index = header.index("#1 ID")
        second_id_index = header.index("#2 ID")
        for pair_id, row in enumerate(reader, start=1):
            pairs.append(
                PairInput(
                    pair_id=pair_id,
                    sentence1=sentence_by_id[row[first_id_index].strip()],
                    sentence2=sentence_by_id[row[second_id_index].strip()],
                )
            )
    if not pairs:
        raise ValueError("MRPC input is empty")
    return pairs


def _load_selected_labels(
    mrpc_dir: Path,
    selected_pair_ids: frozenset[int],
) -> dict[int, bool]:
    """Materialize labels for exactly the requested pair IDs."""

    test_path = mrpc_dir.parent / "msr_paraphrase_test.txt"
    labels: dict[int, bool] = {}
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        quality_index = header.index("Quality")
        for pair_id, row in enumerate(reader, start=1):
            if pair_id not in selected_pair_ids:
                continue
            value = row[quality_index].strip()
            if value not in {"0", "1"}:
                raise ValueError(f"invalid MRPC label for pair {pair_id}")
            labels[pair_id] = value == "1"
    if labels.keys() != selected_pair_ids:
        raise ValueError("requested MRPC labels were not all found")
    return labels


def _group_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = sorted({int(value) for value in groups})
    fold_by_group = {
        group: int(
            hashlib.sha256(f"{FOLD_SEED}:{group}".encode("utf-8")).hexdigest(),
            16,
        )
        % MODEL_SELECTION_FOLDS
        for group in unique_groups
    }
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(MODEL_SELECTION_FOLDS):
        validation = np.asarray([fold_by_group[int(group)] == fold for group in groups], dtype=bool)
        if not np.any(validation) or np.all(validation):
            raise ValueError("component fold is empty")
        result.append((np.flatnonzero(~validation), np.flatnonzero(validation)))
    return result


def _counter_overlap_features(left: Counter[str], right: Counter[str]) -> tuple[float, ...]:
    intersection = sum((left & right).values())
    union = sum((left | right).values())
    left_count = sum(left.values())
    right_count = sum(right.values())
    total = left_count + right_count
    minimum = min(left_count, right_count)
    maximum = max(left_count, right_count, 1)
    return (
        intersection / union if union else 1.0,
        2.0 * intersection / total if total else 1.0,
        intersection / minimum if minimum else float(left_count == right_count),
        abs(left_count - right_count) / maximum,
    )


def _ngrams(values: Sequence[str], size: int) -> Counter[str]:
    return Counter(
        "\0".join(values[index : index + size]) for index in range(len(values) - size + 1)
    )


def _surface_pair_features(left: str, right: str) -> np.ndarray:
    left_tokens = TOKEN_PATTERN.findall(left.casefold())
    right_tokens = TOKEN_PATTERN.findall(right.casefold())
    values: list[float] = []
    for size in (1, 2, 3):
        values.extend(
            _counter_overlap_features(_ngrams(left_tokens, size), _ngrams(right_tokens, size))
        )
    left_chars = tuple(" ".join(left.casefold().split()))
    right_chars = tuple(" ".join(right.casefold().split()))
    for size in (2, 3, 4):
        values.extend(
            _counter_overlap_features(_ngrams(left_chars, size), _ngrams(right_chars, size))
        )
    counts_left = np.asarray((len(left_tokens), len(left_chars)), dtype=np.float64)
    counts_right = np.asarray((len(right_tokens), len(right_chars)), dtype=np.float64)
    scale = np.maximum(np.maximum(counts_left, counts_right), 1.0)
    values.extend((np.abs(counts_left - counts_right) / scale).tolist())
    values.extend((np.minimum(counts_left, counts_right) / scale).tolist())
    return np.asarray(values, dtype=np.float64)


def _distribution_summary(values: np.ndarray) -> list[float]:
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        *[float(value) for value in np.quantile(values, (0.1, 0.25, 0.5, 0.75, 0.9))],
    ]


def _compact_embedding_pair_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    absolute_difference = np.abs(left - right)
    hadamard = left * right
    values = [
        float(np.dot(left, right)),
        float(np.linalg.norm(left - right)),
        float(np.mean(absolute_difference)),
    ]
    values.extend(_distribution_summary(absolute_difference))
    values.extend(_distribution_summary(hadamard))
    return np.asarray(values, dtype=np.float64)


def _symmetric_vector_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate((np.abs(left - right), left * right)).astype(np.float64, copy=False)


def _feature_matrices(
    pairs: Sequence[PairInput],
    *,
    embeddings: Mapping[str, np.ndarray],
    signatures: Mapping[str, np.ndarray],
    graphs: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    raw_embedding = np.stack(
        [
            _symmetric_vector_features(embeddings[pair.sentence1], embeddings[pair.sentence2])
            for pair in pairs
        ]
    )
    compact_embedding = np.stack(
        [
            _compact_embedding_pair_features(embeddings[pair.sentence1], embeddings[pair.sentence2])
            for pair in pairs
        ]
    )
    psa = np.stack(
        [
            _symmetric_vector_features(signatures[pair.sentence1], signatures[pair.sentence2])
            for pair in pairs
        ]
    )
    surface = np.stack([_surface_pair_features(pair.sentence1, pair.sentence2) for pair in pairs])
    graph = _graph_pair_matrix(pairs, graphs)
    compact = np.column_stack((compact_embedding, psa, surface, graph))
    return {
        "compact_generic_v1": compact,
        "raw_embedding_v1": raw_embedding,
        "raw_plus_compact_v1": np.column_stack((raw_embedding, compact)),
    }


def _candidate_specs() -> tuple[CandidateSpec, ...]:
    result: list[CandidateSpec] = []

    def add(
        family: str,
        feature_set: str,
        **parameters: Any,
    ) -> None:
        identity = _sha256_payload(
            {"family": family, "feature_set": feature_set, "parameters": parameters}
        )[:12]
        result.append(
            CandidateSpec(
                candidate_id=f"{family}:{feature_set}:{identity}",
                family=family,
                feature_set=feature_set,
                parameters=parameters,
            )
        )

    for feature_set in ("raw_embedding_v1", "raw_plus_compact_v1", "compact_generic_v1"):
        for alpha in (10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0):
            for class_weight in (None, "balanced"):
                add("ridge", feature_set, alpha=alpha, class_weight=class_weight)
        for c_value in (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0):
            for class_weight in (None, "balanced"):
                add(
                    "logistic",
                    feature_set,
                    C=c_value,
                    class_weight=class_weight,
                )
        for c_value in (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0):
            for class_weight in (None, "balanced"):
                add(
                    "linear_svc",
                    feature_set,
                    C=c_value,
                    class_weight=class_weight,
                )

    for c_value in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        for gamma in ("scale", 0.01, 0.03, 0.1, 0.3, 1.0):
            for class_weight in (None, "balanced"):
                add(
                    "rbf_svc",
                    "compact_generic_v1",
                    C=c_value,
                    gamma=gamma,
                    class_weight=class_weight,
                )

    for max_features in ("sqrt", 0.5, 1.0):
        for min_samples_leaf in (1, 2, 4, 8):
            for class_weight in (None, "balanced"):
                add(
                    "extra_trees",
                    "compact_generic_v1",
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                )

    for learning_rate in (0.03, 0.1):
        for max_leaf_nodes in (3, 7, 15):
            for l2_regularization in (0.0, 1.0, 10.0):
                add(
                    "hist_gradient_boosting",
                    "compact_generic_v1",
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                )

    for learning_rate in (0.03, 0.1):
        for max_depth in (1, 2, 3):
            for n_estimators in (100, 300):
                add(
                    "gradient_boosting",
                    "compact_generic_v1",
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                )
    return tuple(result)


def _make_estimator(spec: CandidateSpec) -> BaseEstimator:
    parameters = dict(spec.parameters)
    if spec.family == "ridge":
        return make_pipeline(
            StandardScaler(),
            RidgeClassifier(solver="cholesky", **parameters),
        )
    if spec.family == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                penalty="l2",
                max_iter=2_000,
                random_state=31,
                **parameters,
            ),
        )
    if spec.family == "linear_svc":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(dual="auto", max_iter=20_000, random_state=31, **parameters),
        )
    if spec.family == "rbf_svc":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", **parameters))
    if spec.family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            random_state=31,
            n_jobs=-1,
            **parameters,
        )
    if spec.family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=300,
            min_samples_leaf=10,
            random_state=31,
            **parameters,
        )
    if spec.family == "gradient_boosting":
        return GradientBoostingClassifier(random_state=31, **parameters)
    raise ValueError(f"unsupported model family {spec.family!r}")


def _estimator_scores(estimator: BaseEstimator, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=np.float64)
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    return probabilities[:, 1]


def _best_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    routed: np.ndarray,
) -> tuple[float, Mapping[str, Any]]:
    finite_scores = scores[routed & np.isfinite(scores)]
    if not len(finite_scores):
        raise ValueError("no routed OOF scores")
    best: tuple[tuple[float, float, float, float], float, Mapping[str, Any]] | None = None
    for threshold in np.unique(finite_scores):
        predictions = routed & (scores >= threshold)
        metrics = _binary_metrics(labels, predictions)
        key = (
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
            float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    if best is None:
        raise ValueError("threshold search failed")
    return best[1], best[2]


def _selective_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
    routed: np.ndarray,
    *,
    binary_threshold: float,
    target_precision: float,
) -> tuple[float, float, Mapping[str, Any]]:
    """Fit non-overlapping abstention bounds around a frozen binary threshold."""

    finite = routed & np.isfinite(scores)
    positive_candidates: list[tuple[int, float, float]] = []
    negative_candidates: list[tuple[int, float, float]] = []
    for threshold in np.unique(scores[finite]):
        threshold = float(threshold)
        if threshold > binary_threshold:
            selected = finite & (scores >= threshold)
            count = int(np.count_nonzero(selected))
            precision = float(np.mean(labels[selected]))
            if precision >= target_precision:
                positive_candidates.append((count, precision, threshold))
        if threshold < binary_threshold:
            selected = finite & (scores <= threshold)
            count = int(np.count_nonzero(selected))
            negative_predictive_value = float(np.mean(~labels[selected]))
            if negative_predictive_value >= target_precision:
                negative_candidates.append((count, negative_predictive_value, threshold))
    if not positive_candidates or not negative_candidates:
        raise ValueError("calibration cannot satisfy the selective precision contract")
    positive_count, positive_precision, positive_threshold = max(
        positive_candidates,
        key=lambda value: (value[0], value[1], -value[2]),
    )
    negative_count, negative_precision, negative_threshold = max(
        negative_candidates,
        key=lambda value: (value[0], value[1], value[2]),
    )
    if not negative_threshold < binary_threshold < positive_threshold:
        raise ValueError("selective threshold order is invalid")
    return (
        negative_threshold,
        positive_threshold,
        {
            "target_precision": target_precision,
            "negative": {
                "threshold": negative_threshold,
                "count": negative_count,
                "negative_predictive_value": negative_precision,
            },
            "binary": {"threshold": binary_threshold},
            "positive": {
                "threshold": positive_threshold,
                "count": positive_count,
                "precision": positive_precision,
            },
            "ordering_enforced": "negative < binary < positive",
        },
    )


def _crossfit(
    spec: CandidateSpec,
    features: np.ndarray,
    labels: np.ndarray,
    routed: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> CandidateResult:
    scores = np.full(len(labels), -math.inf, dtype=np.float64)
    for training_indexes, validation_indexes in folds:
        training_indexes = training_indexes[routed[training_indexes]]
        validation_routed = validation_indexes[routed[validation_indexes]]
        estimator = _make_estimator(spec)
        estimator.fit(features[training_indexes], labels[training_indexes])
        scores[validation_routed] = _estimator_scores(estimator, features[validation_routed])
    threshold, metrics = _best_threshold(labels, scores, routed)
    return CandidateResult(
        spec=spec,
        oof_scores=scores,
        threshold=threshold,
        metrics=metrics,
    )


def _select_models(
    *,
    feature_matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    routed: np.ndarray,
    groups: np.ndarray,
) -> list[CandidateResult]:
    folds = _group_folds(groups)
    results: list[CandidateResult] = []
    specs = _candidate_specs()
    for index, spec in enumerate(specs, start=1):
        result = _crossfit(
            spec,
            feature_matrices[spec.feature_set],
            labels,
            routed,
            folds,
        )
        results.append(result)
        if index % 25 == 0:
            current = max(results, key=_result_key)
            print(
                f"evaluated {index}/{len(specs)}; best "
                f"{current.spec.candidate_id} OOF F1={current.metrics['f1']:.6f}",
                flush=True,
            )
    return results


def _result_key(result: CandidateResult) -> tuple[float, float, float, str]:
    return (
        float(result.metrics["f1"]),
        float(result.metrics["precision"]),
        float(result.metrics["recall"]),
        result.spec.candidate_id,
    )


def _candidate_report(result: CandidateResult) -> dict[str, Any]:
    return {
        "spec": asdict(result.spec),
        "threshold": result.threshold,
        "oof_metrics": dict(result.metrics),
    }


def _partition_neighborhoods(
    pairs: Sequence[PairInput],
    signatures: Mapping[str, np.ndarray],
    *,
    top_k: int,
) -> tuple[dict[str, frozenset[str]], Mapping[str, Any]]:
    """Build a route from only the sentence geometry in one split partition."""

    texts = {text for pair in pairs for text in (pair.sentence1, pair.sentence2)}
    partition_signatures = {text: signatures[text] for text in texts}
    return _psa_neighborhoods(
        partition_signatures,
        top_k=min(top_k, len(partition_signatures) - 1),
    )


def _symmetric_route(
    pairs: Sequence[PairInput],
    neighborhoods: Mapping[str, frozenset[str]],
) -> np.ndarray:
    return np.asarray(
        [
            pair.sentence2 in neighborhoods[pair.sentence1]
            or pair.sentence1 in neighborhoods[pair.sentence2]
            for pair in pairs
        ],
        dtype=bool,
    )


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[PairInput],
    Mapping[int, str],
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    pairs = _load_pair_inputs(args.mrpc_dir.resolve())
    split = split_mrpc_by_sentence_component(pairs, seed=BLIND_SPLIT_SEED)
    embeddings, _cached_guides, embedding_evidence = _load_cache_evidence(
        args.cache.resolve(), pairs
    )
    pinned_guides, calibration, frame_router, _frame_evidence = _load_pinned_frame(
        args.pinned_guide_embeddings.resolve(), args.calibration.resolve()
    )
    signatures, _routing_labels, _signature_evidence = _project_representations(
        embeddings, pinned_guides, frame_router, calibration
    )
    texts = sorted(
        {text for pair in pairs for text in (pair.sentence1, pair.sentence2)},
        key=lambda text: (sha256_text(text), text),
    )
    graphs, graph_evidence = _build_or_load_graphs(
        texts,
        sense_path=args.sense_cache.resolve(),
        cache_path=args.graph_cache.resolve(),
        expected_cache_sha256=args.graph_cache_sha256,
        expected_parser_identity=args.graph_parser_identity,
    )
    evidence = {
        "embedding_cache": embedding_evidence,
        "graph": graph_evidence,
    }
    return pairs, split, embeddings, signatures, graphs, evidence


def _select(args: argparse.Namespace) -> int:
    pairs, split, embeddings, signatures, graphs, evidence = _load_inputs(args)
    calibration_pairs = [pair for pair in pairs if split[pair.pair_id] == "calibration"]
    calibration_ids = frozenset(pair.pair_id for pair in calibration_pairs)
    calibration_label_map = _load_selected_labels(args.mrpc_dir.resolve(), calibration_ids)
    labels = np.asarray(
        [calibration_label_map[pair.pair_id] for pair in calibration_pairs], dtype=bool
    )
    neighborhoods, neighborhood_evidence = _partition_neighborhoods(
        calibration_pairs,
        signatures,
        top_k=args.psa_top_k,
    )
    routed = _symmetric_route(calibration_pairs, neighborhoods)
    groups, component_count = _component_groups(calibration_pairs)
    feature_matrices = _feature_matrices(
        calibration_pairs,
        embeddings=embeddings,
        signatures=signatures,
        graphs=graphs,
    )
    results = _select_models(
        feature_matrices=feature_matrices,
        labels=labels,
        routed=routed,
        groups=groups,
    )
    ordered = sorted(results, key=_result_key, reverse=True)
    winner = ordered[0]
    candidate_grid = [asdict(spec) for spec in _candidate_specs()]
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "selection_frozen_before_held_out_label_materialization",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "split": {
            "seed": BLIND_SPLIT_SEED,
            "calibration_pair_count": len(calibration_pairs),
            "calibration_component_count": component_count,
            "component_folds": MODEL_SELECTION_FOLDS,
            "fold_seed": FOLD_SEED,
            "held_out_labels_materialized": False,
        },
        "routing": {
            "kind": "partition_local_symmetric_psa_top_k_union",
            "direction_policy": "route if either sentence nominates the other",
            "partition_policy": (
                "calibration and held-out neighborhoods are built independently; "
                "held-out geometry is absent from selection"
            ),
            "top_k": args.psa_top_k,
            "calibration_routed_pair_count": int(np.count_nonzero(routed)),
            "calibration_neighborhood_evidence": neighborhood_evidence,
        },
        "candidate_grid_sha256": _sha256_payload(candidate_grid),
        "candidate_count": len(candidate_grid),
        "selection_objective": "maximum component-cross-fitted F1, then precision and recall",
        "winner": _candidate_report(winner),
        "top_candidates": [_candidate_report(result) for result in ordered[:25]],
        "family_winners": {
            family: _candidate_report(
                max(
                    (result for result in results if result.spec.family == family),
                    key=_result_key,
                )
            )
            for family in sorted({result.spec.family for result in results})
        },
        "input_identity": {
            "mrpc_corpus_sha256": _file_sha256(
                args.mrpc_dir.resolve().parent / "msr_paraphrase_data.txt"
            ),
            "mrpc_test_sha256": _file_sha256(
                args.mrpc_dir.resolve().parent / "msr_paraphrase_test.txt"
            ),
            "embedding_cache_sha256": evidence["embedding_cache"]["cache_sha256"],
            "graph_cache_sha256": args.graph_cache_sha256,
        },
        "evidence": evidence,
        "feature_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in feature_matrices.items()
        },
        "caveats": [
            "Selection is MRPC-calibrated and is not a fresh blind benchmark claim.",
            "The fixed route can omit pairs before resolver scoring.",
            "Every feature is generic and swap-invariant; no phrase or lemma rules are used.",
            "The resolver experiment has no paraphrase membership authority.",
        ],
    }
    payload["frozen_selection_sha256"] = _sha256_payload(payload)
    output_path = args.output_dir.resolve() / "frozen_selection.json"
    _atomic_json(output_path, payload)
    print(
        json.dumps(
            {
                "frozen_selection": str(output_path),
                "frozen_selection_sha256": payload["frozen_selection_sha256"],
                "winner": payload["winner"],
                "family_winners": payload["family_winners"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _spec_from_payload(value: Mapping[str, Any]) -> CandidateSpec:
    return CandidateSpec(
        candidate_id=str(value["candidate_id"]),
        family=str(value["family"]),
        feature_set=str(value["feature_set"]),
        parameters=dict(value["parameters"]),
    )


def _validate_selection_inputs(
    selection: Mapping[str, Any],
    args: argparse.Namespace,
    evidence: Mapping[str, Any],
) -> None:
    expected = selection["input_identity"]
    actual = {
        "mrpc_corpus_sha256": _file_sha256(
            args.mrpc_dir.resolve().parent / "msr_paraphrase_data.txt"
        ),
        "mrpc_test_sha256": _file_sha256(
            args.mrpc_dir.resolve().parent / "msr_paraphrase_test.txt"
        ),
        "embedding_cache_sha256": evidence["embedding_cache"]["cache_sha256"],
        "graph_cache_sha256": args.graph_cache_sha256,
    }
    if actual != expected:
        raise ValueError("reveal inputs do not match the frozen selection")
    if int(selection["routing"]["top_k"]) != int(args.psa_top_k):
        raise ValueError("reveal routing top-k does not match the frozen selection")
    if selection["candidate_grid_sha256"] != _sha256_payload(
        [asdict(spec) for spec in _candidate_specs()]
    ):
        raise ValueError("candidate grid source changed after selection freeze")


def _validate_and_load_baseline(
    path: Path,
    held_out_pair_ids: Sequence[int],
) -> tuple[Mapping[str, Any], np.ndarray]:
    if _file_sha256(path) != EXPECTED_BASELINE_REPORT_SHA256:
        raise ValueError("comparable baseline report hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPECTED_BASELINE_SCHEMA_VERSION:
        raise ValueError("comparable baseline schema mismatch")
    split_policy = payload.get("split_policy") or {}
    expected_split = {
        "seed": BLIND_SPLIT_SEED,
        "unit": "connected_component_of_normalized_sentence_identity",
        "calibration_pair_count": 618,
        "held_out_pair_count": 1107,
    }
    if any(split_policy.get(key) != value for key, value in expected_split.items()):
        raise ValueError("comparable baseline split mismatch")
    policy_id = ((payload.get("verifier") or {}).get("execution_policy") or {}).get("policy_id")
    if policy_id != EXPECTED_BASELINE_POLICY_ID:
        raise ValueError("comparable baseline policy mismatch")
    metrics = payload["held_out"]["profiles"]["mrpc_compatible_v1"]["learned_gate"][
        "gate_only_metrics"
    ]
    if metrics != EXPECTED_BASELINE_METRICS:
        raise ValueError("comparable baseline metrics mismatch")
    traces = payload["held_out_pair_traces"]["profiles"]["mrpc_compatible_v1"]["learned_gate"]
    prediction_by_pair = {
        int(trace["pair_id"]): bool(trace["psa_nominated"] and trace["native_gate_passed"])
        for trace in traces
    }
    if prediction_by_pair.keys() != set(held_out_pair_ids):
        raise ValueError("comparable baseline trace pair IDs mismatch")
    predictions = np.asarray(
        [prediction_by_pair[pair_id] for pair_id in held_out_pair_ids], dtype=bool
    )
    return payload, predictions


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(np.count_nonzero(labels & predictions))
    false_positive = int(np.count_nonzero(~labels & predictions))
    false_negative = int(np.count_nonzero(labels & ~predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _paired_bootstrap(
    labels: np.ndarray,
    candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> Mapping[str, Any]:
    random = np.random.default_rng(BOOTSTRAP_SEED)
    candidate_values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    baseline_values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indexes = random.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indexes]
        candidate_values[replicate] = _f1(sampled_labels, candidate_predictions[indexes])
        baseline_values[replicate] = _f1(sampled_labels, baseline_predictions[indexes])
    deltas = candidate_values - baseline_values
    return {
        "method": "paired_nonparametric_pair_bootstrap_percentile",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "candidate_f1_95_percent_ci": [
            float(value) for value in np.quantile(candidate_values, (0.025, 0.975))
        ],
        "baseline_f1_95_percent_ci": [
            float(value) for value in np.quantile(baseline_values, (0.025, 0.975))
        ],
        "paired_delta_f1_95_percent_ci": [
            float(value) for value in np.quantile(deltas, (0.025, 0.975))
        ],
        "paired_probability_delta_above_zero": float(np.mean(deltas > 0.0)),
    }


def _serialize_linear_artifact(
    estimator: BaseEstimator,
    path: Path,
    *,
    threshold_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not hasattr(estimator, "named_steps"):
        raise ValueError("selected winner is not a serializable linear pipeline")
    steps = estimator.named_steps
    scaler = steps.get("standardscaler")
    model = next((value for key, value in steps.items() if key != "standardscaler"), None)
    if scaler is None or model is None or not hasattr(model, "coef_"):
        raise ValueError("selected winner lacks scaler or linear coefficients")
    arrays = {
        "feature_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "feature_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "coefficients": np.asarray(model.coef_, dtype=np.float64).reshape(-1),
        "intercept": np.asarray(model.intercept_, dtype=np.float64).reshape(-1),
        "thresholds": np.asarray(
            (
                threshold_contract["negative"]["threshold"],
                threshold_contract["binary"]["threshold"],
                threshold_contract["positive"]["threshold"],
            ),
            dtype=np.float64,
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "format": "numpy_npz_compressed_numeric_arrays_only",
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "allow_pickle_required": False,
    }


def _dependency_identity() -> Mapping[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        },
        "sources": {
            "experiment_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
            "resolver_benchmark": {
                "path": str(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "research"
                    / "mrpc"
                    / "run_mrpc_semantic_relation_resolver_benchmark.py"
                ),
                "sha256": _file_sha256(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "research"
                    / "mrpc"
                    / "run_mrpc_semantic_relation_resolver_benchmark.py"
                ),
            },
            "proposition_graph": {
                "path": str(APP_ROOT / "mcp_servers" / "mcp_lib" / "proposition_graph.py"),
                "sha256": _file_sha256(
                    APP_ROOT / "mcp_servers" / "mcp_lib" / "proposition_graph.py"
                ),
            },
        },
    }


def _selective_held_out_report(
    labels: np.ndarray,
    scores: np.ndarray,
    routed: np.ndarray,
    *,
    negative_threshold: float,
    positive_threshold: float,
) -> Mapping[str, Any]:
    positive = routed & (scores >= positive_threshold)
    negative = routed & (scores <= negative_threshold)
    resolved = positive | negative
    return {
        "state_counts": {
            "likely_equivalent": int(np.count_nonzero(positive)),
            "likely_not_equivalent": int(np.count_nonzero(negative)),
            "abstain": int(np.count_nonzero(routed & ~resolved)),
            "not_routed": int(np.count_nonzero(~routed)),
        },
        "coverage_all_pairs": float(np.mean(resolved)),
        "coverage_routed_pairs": float(np.count_nonzero(resolved) / np.count_nonzero(routed)),
        "likely_equivalent_precision": (
            float(np.mean(labels[positive])) if np.any(positive) else None
        ),
        "likely_not_equivalent_npv": (
            float(np.mean(~labels[negative])) if np.any(negative) else None
        ),
        "selective_risk": (
            float(np.mean(labels[resolved] != positive[resolved])) if np.any(resolved) else None
        ),
    }


def _reveal(args: argparse.Namespace) -> int:
    selection_path = args.selection.resolve()
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "held_out_reveal.json"
    contract_path = output_dir / "calibration_resolver_contract.json"
    artifact_path = output_dir / "fitted_resolver_artifact.npz"
    existing_outputs = [
        path for path in (report_path, contract_path, artifact_path) if path.exists()
    ]
    if existing_outputs:
        raise ValueError(
            "reveal outputs already exist; refusing a second reveal: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("selection schema mismatch")
    expected_hash = selection.pop("frozen_selection_sha256")
    actual_hash = _sha256_payload(selection)
    selection["frozen_selection_sha256"] = expected_hash
    if actual_hash != expected_hash:
        raise ValueError("frozen selection hash mismatch")
    if selection.get("phase") != "selection_frozen_before_held_out_label_materialization":
        raise ValueError("selection was not frozen in the calibration-only phase")

    pairs, split, embeddings, signatures, graphs, evidence = _load_inputs(args)
    _validate_selection_inputs(selection, args, evidence)
    calibration_pairs = [pair for pair in pairs if split[pair.pair_id] == "calibration"]
    held_out_pairs = [pair for pair in pairs if split[pair.pair_id] == "held_out"]
    calibration_ids = frozenset(pair.pair_id for pair in calibration_pairs)
    calibration_label_map = _load_selected_labels(args.mrpc_dir.resolve(), calibration_ids)
    calibration_labels = np.asarray(
        [calibration_label_map[pair.pair_id] for pair in calibration_pairs], dtype=bool
    )
    calibration_neighborhoods, _calibration_neighborhood_evidence = _partition_neighborhoods(
        calibration_pairs,
        signatures,
        top_k=args.psa_top_k,
    )
    held_out_neighborhoods, held_out_neighborhood_evidence = _partition_neighborhoods(
        held_out_pairs,
        signatures,
        top_k=args.psa_top_k,
    )
    calibration_routed = _symmetric_route(calibration_pairs, calibration_neighborhoods)
    held_out_routed = _symmetric_route(held_out_pairs, held_out_neighborhoods)
    if int(np.count_nonzero(calibration_routed)) != int(
        selection["routing"]["calibration_routed_pair_count"]
    ):
        raise ValueError("calibration route changed after selection freeze")
    winner = selection["winner"]
    spec = _spec_from_payload(winner["spec"])
    all_features = _feature_matrices(
        [*calibration_pairs, *held_out_pairs],
        embeddings=embeddings,
        signatures=signatures,
        graphs=graphs,
    )[spec.feature_set]
    calibration_features = all_features[: len(calibration_pairs)]
    held_out_features = all_features[len(calibration_pairs) :]

    # Replay the winner's OOF scores without reopening model selection.  This
    # verifies the frozen binary threshold and supplies calibration-only
    # abstention bounds before any held-out label is materialized.
    groups, _component_count = _component_groups(calibration_pairs)
    replay = _crossfit(
        spec,
        calibration_features,
        calibration_labels,
        calibration_routed,
        _group_folds(groups),
    )
    if replay.threshold != float(winner["threshold"]):
        raise ValueError("winner threshold does not reproduce exactly")
    if dict(replay.metrics) != winner["oof_metrics"]:
        raise ValueError("winner OOF metrics do not reproduce exactly")
    negative_threshold, positive_threshold, threshold_contract = _selective_thresholds(
        calibration_labels,
        replay.oof_scores,
        calibration_routed,
        binary_threshold=float(winner["threshold"]),
        target_precision=SELECTIVE_TARGET_PRECISION,
    )

    calibration_cosines = np.asarray(
        [
            float(np.dot(embeddings[pair.sentence1], embeddings[pair.sentence2]))
            for pair in calibration_pairs
        ],
        dtype=np.float64,
    )
    corrected_gate_threshold, corrected_gate_oof_metrics = _best_threshold(
        calibration_labels,
        calibration_cosines,
        calibration_routed,
    )
    held_out_cosines = np.asarray(
        [
            float(np.dot(embeddings[pair.sentence1], embeddings[pair.sentence2]))
            for pair in held_out_pairs
        ],
        dtype=np.float64,
    )
    corrected_gate_predictions = held_out_routed & (held_out_cosines >= corrected_gate_threshold)

    estimator = _make_estimator(spec)
    estimator.fit(
        calibration_features[calibration_routed],
        calibration_labels[calibration_routed],
    )
    held_out_scores = _estimator_scores(estimator, held_out_features)
    held_out_predictions = held_out_routed & (held_out_scores >= float(winner["threshold"]))

    dependency_identity = _dependency_identity()
    artifact_identity = _serialize_linear_artifact(
        estimator,
        artifact_path,
        threshold_contract=threshold_contract,
    )
    calibration_contract = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "calibration_contract_frozen_before_held_out_label_materialization",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "frozen_selection_sha256": expected_hash,
        "winner": winner,
        "replayed_oof_metrics": dict(replay.metrics),
        "threshold_contract": threshold_contract,
        "corrected_route_cosine_baseline": {
            "threshold": corrected_gate_threshold,
            "calibration_metrics": corrected_gate_oof_metrics,
        },
        "artifact": artifact_identity,
        "dependency_and_source_identity": dependency_identity,
        "input_identity": selection["input_identity"],
        "routing": selection["routing"],
    }
    calibration_contract["contract_sha256"] = _sha256_payload(calibration_contract)
    _atomic_json(contract_path, calibration_contract)

    # This is the sole held-out label materialization in the experiment.
    held_out_ids = frozenset(pair.pair_id for pair in held_out_pairs)
    held_out_label_map = _load_selected_labels(args.mrpc_dir.resolve(), held_out_ids)
    held_out_labels = np.asarray(
        [held_out_label_map[pair.pair_id] for pair in held_out_pairs], dtype=bool
    )
    metrics = _binary_metrics(held_out_labels, held_out_predictions)
    corrected_gate_metrics = _binary_metrics(held_out_labels, corrected_gate_predictions)
    always_positive_predictions = np.ones(len(held_out_labels), dtype=bool)
    always_positive_metrics = _binary_metrics(held_out_labels, always_positive_predictions)
    baseline_path = args.baseline_report.resolve()
    _baseline_payload, historical_baseline_predictions = _validate_and_load_baseline(
        baseline_path,
        [pair.pair_id for pair in held_out_pairs],
    )
    reconstructed_historical_metrics = _binary_metrics(
        held_out_labels, historical_baseline_predictions
    )
    if reconstructed_historical_metrics != EXPECTED_BASELINE_METRICS:
        raise ValueError("historical baseline traces do not reproduce its metrics")
    paired_historical = _paired_bootstrap(
        held_out_labels,
        held_out_predictions,
        historical_baseline_predictions,
    )
    paired_corrected_gate = _paired_bootstrap(
        held_out_labels,
        held_out_predictions,
        corrected_gate_predictions,
    )
    selective_report = _selective_held_out_report(
        held_out_labels,
        held_out_scores,
        held_out_routed,
        negative_threshold=negative_threshold,
        positive_threshold=positive_threshold,
    )
    pair_traces = [
        {
            "pair_id": pair.pair_id,
            "label": bool(held_out_labels[index]),
            "routed": bool(held_out_routed[index]),
            "resolver_score": float(held_out_scores[index]),
            "resolver_prediction": bool(held_out_predictions[index]),
            "corrected_route_cosine": float(held_out_cosines[index]),
            "corrected_route_cosine_prediction": bool(corrected_gate_predictions[index]),
            "authenticated_historical_gate_prediction": bool(
                historical_baseline_predictions[index]
            ),
        }
        for index, pair in enumerate(held_out_pairs)
    ]
    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "single_held_out_reveal",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "selection_path": str(selection_path),
        "selection_file_sha256": _file_sha256(selection_path),
        "frozen_selection_sha256": expected_hash,
        "winner": winner,
        "calibration_contract": {
            "path": str(contract_path),
            "file_sha256": _file_sha256(contract_path),
            "contract_sha256": calibration_contract["contract_sha256"],
            "threshold_contract": threshold_contract,
        },
        "fitted_artifact": artifact_identity,
        "dependency_and_source_identity": dependency_identity,
        "held_out": {
            "pair_count": len(held_out_pairs),
            "routed_pair_count": int(np.count_nonzero(held_out_routed)),
            "metrics": metrics,
            "selective": selective_report,
            "labels_materialized_after_artifact_freeze": True,
            "reveal_count": 1,
        },
        "comparators": {
            "corrected_partition_local_symmetric_cosine_gate": {
                "threshold": corrected_gate_threshold,
                "calibration_metrics": corrected_gate_oof_metrics,
                "held_out_metrics": corrected_gate_metrics,
                "f1_delta": float(metrics["f1"] - corrected_gate_metrics["f1"]),
                "paired_bootstrap": paired_corrected_gate,
            },
            "authenticated_historical_mrpc_compatible_gate": {
                "report_path": str(baseline_path),
                "report_sha256": _file_sha256(baseline_path),
                "schema_version": EXPECTED_BASELINE_SCHEMA_VERSION,
                "execution_policy_id": EXPECTED_BASELINE_POLICY_ID,
                "held_out_metrics": EXPECTED_BASELINE_METRICS,
                "f1_delta": float(metrics["f1"] - EXPECTED_BASELINE_METRICS["f1"]),
                "paired_bootstrap": paired_historical,
                "route_caveat": (
                    "historical report used a corpus-global directional nomination; "
                    "it is authenticated as the dashboard reference but not the corrected route"
                ),
            },
            "always_positive": {"held_out_metrics": always_positive_metrics},
        },
        "routing": {
            "kind": "partition_local_symmetric_psa_top_k_union",
            "direction_policy": "route if either sentence nominates the other",
            "partition_policy": "held-out neighborhoods use held-out sentence geometry only",
            "top_k": args.psa_top_k,
            "held_out_neighborhood_evidence": held_out_neighborhood_evidence,
        },
        "held_out_pair_traces": pair_traces,
        "caveats": [
            *selection["caveats"],
            "The held-out reveal was run once after winner, binary threshold, selective thresholds, and fitted artifact were frozen.",
            "The historical dashboard comparator has a different, now-corrected routing topology.",
        ],
    }
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "oof_metrics": winner["oof_metrics"],
                "held_out_metrics": metrics,
                "corrected_route_gate_metrics": corrected_gate_metrics,
                "authenticated_historical_gate_metrics": EXPECTED_BASELINE_METRICS,
                "always_positive_metrics": always_positive_metrics,
                "paired_bootstrap_vs_historical": paired_historical,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    for phase in ("select", "reveal"):
        subparser = subparsers.add_parser(phase)
        subparser.add_argument(
            "--mrpc-dir",
            type=Path,
            default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split",
        )
        subparser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
        subparser.add_argument("--sense-cache", type=Path, default=DEFAULT_SENSE_CACHE)
        subparser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
        subparser.add_argument("--graph-cache-sha256", default=PINNED_GRAPH_CACHE_SHA256)
        subparser.add_argument("--graph-parser-identity", default=PINNED_GRAPH_PARSER_IDENTITY)
        subparser.add_argument(
            "--pinned-guide-embeddings",
            type=Path,
            default=DEFAULT_PINNED_GUIDE_EMBEDDINGS,
        )
        subparser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
        subparser.add_argument("--psa-top-k", type=int, default=32)
        subparser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        if phase == "reveal":
            subparser.add_argument(
                "--selection",
                type=Path,
                default=DEFAULT_OUTPUT_DIR / "frozen_selection.json",
            )
            subparser.add_argument(
                "--baseline-report",
                type=Path,
                default=DEFAULT_BASELINE_REPORT,
            )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "select":
        return _select(args)
    return _reveal(args)


if __name__ == "__main__":
    raise SystemExit(main())
