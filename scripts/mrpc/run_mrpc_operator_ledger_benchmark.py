#!/usr/bin/env python3
"""Run the bounded grammatical-operator ledger veto on frozen MRPC.

The experiment is deliberately narrower than a replacement paraphrase model.
It keeps the authenticated compact-resolver prediction as the ceiling and may
only veto an existing positive prediction.  The veto sees a fixed vector of
named integer counts and category one-hots emitted by the directional ledger;
raw text, lemmas, parser markers, IDs, hashes, embeddings, PSA values, and
corpus-specific rules are forbidden scorer inputs.

There are two phases.  ``select`` uses only the sentence-component calibration
labels to cross-fit one fixed additive ridge scorer and freeze its threshold.
``reveal`` authenticates that artifact and the immutable 1,107-pair comparator,
scores the held-out rows before loading their labels, and reports every changed
prediction plus a deterministic paired-bootstrap F1 interval.  Nothing in this
script has membership, edge-write, service, or database authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import RidgeClassifier

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
for import_path in (REPOSITORY_ROOT, APP_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import scripts.research.mrpc.experiment_mrpc_semantic_relation_models as model_experiment  # noqa: E402
from scripts.research.mrpc.run_mrpc_current_pipeline_benchmark import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_PINNED_GUIDE_EMBEDDINGS,
)
from scripts.research.mrpc.run_mrpc_proposition_graph_benchmark import (  # noqa: E402
    BLIND_SPLIT_SEED,
    DEFAULT_CACHE,
    DEFAULT_GRAPH_CACHE,
    DEFAULT_SENSE_CACHE,
    PINNED_GRAPH_CACHE_SHA256,
    PINNED_GRAPH_PARSER_IDENTITY,
)
from scripts.research.mrpc.run_mrpc_semantic_relation_resolver_benchmark import (  # noqa: E402
    EXPECTED_FROZEN_HELD_OUT_METRICS,
    EXPECTED_FROZEN_REVEAL_FILE_SHA256,
    EXPECTED_FROZEN_WINNER,
    PAIRED_BOOTSTRAP_SAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    _component_groups,
    _paired_f1_bootstrap,
)

from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    _binary_metrics,
    sha256_text,
)

SELECTION_SCHEMA_VERSION = "mrpc_operator_ledger_veto_selection_v1"
REPORT_SCHEMA_VERSION = "mrpc_operator_ledger_veto_report_v1"
MODEL_SCHEMA_VERSION = "transparent_additive_operator_veto_v1"
FEATURE_SCHEMA_VERSION = "directional_ledger_named_integer_features_v1"
EXPERIMENT_ID = "grammatical_operator_directional_ledger_veto_v1"
FIXED_RIDGE_ALPHA = 1_000.0
COMPONENT_FOLD_COUNT = 5
COMPONENT_FOLD_SEED = "operator-ledger-veto-fold-v1"
DEFAULT_MINIMUM_F1_GAIN = 0.01
EXPECTED_MRPC_DATASET_SHA256 = {
    "msr_paraphrase_data.txt": ("0daa19928665d29551d93c42fbd48200d272818b5e91fe589a98ed7467e69519"),
    "msr_paraphrase_test.txt": ("9b4c96307f6e9276f4c32614dea1f1e65fa6777d0e4451c0c4e87c3d048636ef"),
}
EXPECTED_HELD_PAIR_COUNT = 1_107
EXPECTED_HELD_ROUTED_PAIR_COUNT = 1_065
EXPECTED_HELD_PAIR_ID_SHA256 = "80ee7384db0dff94dde5641601f39ad02dae4e8b886e34526f780451bfe9c962"
EXPECTED_HELD_LABEL_SHA256 = "5f3ac11e6301f139afcd1cd808b010667420e2c6e54dacf3013e2cbb5b2573e9"
EXPECTED_HELD_ROUTE_SHA256 = "b89e31dac79a3e7f834a9a80759e3143b99212a664ccbf7d188ebabc0a7dfa1f"
EXPECTED_HELD_SCORE_SHA256 = "c90e657cdb93deb1dd3426fa21b2d92e3fdfe5a9b5bea0b957bc8d3a83ee2fd5"
EXPECTED_HELD_PREDICTION_SHA256 = "f49cb531be3b5fc722c2c6764b710435738b0240a9c985a57028b539a03a32c1"
EXPECTED_FROZEN_SELECTION_FILE_SHA256 = (
    "fa92b37f978f3fb955cf097c984ac386ff28a8439b3a9d10bf10d076a91a8d3f"
)
EXPECTED_FROZEN_SELECTION_SHA256 = (
    "6906e04234640628e4f42a3b227271e856a032bd4038776d2fff2e067f42908e"
)
DEFAULT_BASELINE_REVEAL = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_semantic_relation_model_search_2026-08-03"
    / "held_out_reveal.json"
)
DEFAULT_WORK_DIR = REPOSITORY_ROOT / "temp" / "mrpc_operator_ledger_veto"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_operator_ledger_veto_2026-08-04"
DEFAULT_SELECTION = DEFAULT_WORK_DIR / "frozen_selection.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "operator_ledger_veto_report.json"
DEFAULT_REVEAL_MARKER = DEFAULT_WORK_DIR / "held_out_reveal_started.json"
DEFAULT_FROZEN_COMPACT_SELECTION = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_semantic_relation_model_search_2026-08-03"
    / "frozen_selection.json"
)

_SAFE_FEATURE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_FORBIDDEN_FEATURE_FRAGMENTS = frozenset(
    {
        "text",
        "token",
        "lemma",
        "marker",
        "phrase",
        "sentence",
        "pair_id",
        "example",
        "corpus",
        "mrpc",
        "hash",
        "sha",
        "embedding",
        "cosine",
        "psa",
        "qwen",
        "url",
        "author",
        "source",
    }
)


@dataclass(frozen=True, slots=True)
class FrozenComparator:
    pair_ids: np.ndarray
    routes: np.ndarray
    scores: np.ndarray
    predictions: np.ndarray
    trace_by_pair: Mapping[int, Mapping[str, Any]]
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LedgerFeatureTrace:
    feature_names: tuple[str, ...]
    values: tuple[int, ...]
    feature_sha256: str
    microgpt_shadow_tokens: tuple[str, ...]
    ledger: Mapping[str, Any]
    edges: Mapping[str, Any]
    coverage: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AdditiveReplay:
    score: float
    intercept: float
    contributions: tuple[dict[str, float | int | str], ...]


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


def _optional_score_vector(values: np.ndarray) -> list[float | None]:
    """Serialize deliberately unscored veto rows without non-JSON infinities."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("score vector is malformed")
    return [float(value) if math.isfinite(float(value)) else None for value in scores]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create one immutable phase artifact; never replace an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing experiment artifact: {path}") from exc


def _payload_with_sha(value: Mapping[str, Any], *, key: str) -> dict[str, Any]:
    payload = dict(value)
    if key in payload:
        raise ValueError(f"payload already contains identity key {key!r}")
    payload[key] = _canonical_sha256(payload)
    return payload


def _validate_payload_sha(value: Mapping[str, Any], *, key: str, name: str) -> None:
    expected = str(value.get(key) or "")
    payload = {item_key: item for item_key, item in value.items() if item_key != key}
    if len(expected) != 64 or expected != _canonical_sha256(payload):
        raise ValueError(f"{name} SHA-256 mismatch")


def _nonnegative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a finite nonnegative float") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite nonnegative float")
    return parsed


def _validate_feature_name(name: str) -> None:
    if not _SAFE_FEATURE_NAME.fullmatch(name):
        raise ValueError(f"unsafe ledger feature name {name!r}")
    parts = frozenset(name.split("_"))
    forbidden = sorted(parts & _FORBIDDEN_FEATURE_FRAGMENTS)
    if forbidden:
        raise ValueError(
            f"ledger feature {name!r} contains forbidden scorer input {forbidden[0]!r}"
        )


def _validated_feature_row(
    names: Sequence[Any],
    values: Sequence[Any],
    *,
    expected_names: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Accept only a fixed, unique vector of named Python integers."""

    normalized_names = tuple(str(name) for name in names)
    if not normalized_names or len(normalized_names) != len(values):
        raise ValueError("ledger feature names and values must be aligned and non-empty")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("ledger feature names must be unique")
    for name in normalized_names:
        _validate_feature_name(name)
    if expected_names is not None and normalized_names != tuple(expected_names):
        raise ValueError("ledger feature schema changed after selection freeze")

    normalized_values: list[int] = []
    for name, value in zip(normalized_names, values, strict=True):
        if isinstance(value, bool):
            raise ValueError(f"ledger feature {name!r} must be an integer, not bool")
        if not isinstance(value, (int, np.integer)):
            raise ValueError(f"ledger feature {name!r} is not an integer count/category")
        normalized_values.append(int(value))
    return normalized_names, tuple(normalized_values)


def _microgpt_shadow_tokens(names: Sequence[str], values: Sequence[int]) -> tuple[str, ...]:
    """Expose the allowlisted count/category vocabulary without running SwitchCore."""

    validated_names, validated_values = _validated_feature_row(names, values)
    return tuple(
        f"{name}={value}" for name, value in zip(validated_names, validated_values, strict=True)
    )


def _component_folds(groups: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    groups = np.asarray(groups, dtype=np.int64)
    if groups.ndim != 1 or not groups.size:
        raise ValueError("component groups must be a non-empty vector")
    unique_groups = sorted({int(value) for value in groups})
    if len(unique_groups) < COMPONENT_FOLD_COUNT:
        raise ValueError("not enough sentence components for operator cross-validation")
    fold_by_group = {
        group: int(
            hashlib.sha256(f"{COMPONENT_FOLD_SEED}:{group}".encode("utf-8")).hexdigest(),
            16,
        )
        % COMPONENT_FOLD_COUNT
        for group in unique_groups
    }
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(COMPONENT_FOLD_COUNT):
        validation = np.asarray([fold_by_group[int(group)] == fold for group in groups], dtype=bool)
        if not np.any(validation) or np.all(validation):
            raise ValueError("operator component fold is empty")
        folds.append((np.flatnonzero(~validation), np.flatnonzero(validation)))
    return tuple(folds)


def _standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("operator feature matrix must be non-empty and two-dimensional")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator feature matrix contains non-finite values")
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    scale = np.asarray(matrix.std(axis=0), dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def _fit_additive_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    """Fit the one declared linear model and serialize every numeric parameter."""

    names, _ = _validated_feature_row(feature_names, [0] * len(feature_names))
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if matrix.ndim != 2 or matrix.shape != (labels.size, len(names)) or not labels.size:
        raise ValueError("operator model inputs are not pair-aligned")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("operator model feature matrix contains non-finite values")
    if np.unique(labels).size != 2:
        raise ValueError("operator model training rows must contain both labels")
    mean, scale = _standardization(matrix)
    standardized = (matrix - mean) / scale
    estimator = RidgeClassifier(
        alpha=FIXED_RIDGE_ALPHA,
        class_weight=None,
        solver="cholesky",
    )
    estimator.fit(standardized, labels)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)[0])
    if coefficients.shape != (len(names),) or not np.all(np.isfinite(coefficients)):
        raise ValueError("operator model coefficients are malformed")
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(names),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "intercept": intercept,
        "fixed_ridge_alpha": FIXED_RIDGE_ALPHA,
        "class_weight": None,
        "solver": "cholesky",
        "truth_surface": "transparent_additive_linear_veto",
        "microgpt_role": "shadow_tokens_only_not_truth_controlling",
    }
    return _payload_with_sha(payload, key="model_sha256")


def _validate_model(
    model: Mapping[str, Any], *, expected_feature_names: Sequence[str] | None = None
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, float]:
    _validate_payload_sha(model, key="model_sha256", name="operator additive model")
    if (
        model.get("schema_version") != MODEL_SCHEMA_VERSION
        or model.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
        or float(model.get("fixed_ridge_alpha", math.nan)) != FIXED_RIDGE_ALPHA
        or model.get("class_weight") is not None
        or model.get("solver") != "cholesky"
        or model.get("truth_surface") != "transparent_additive_linear_veto"
        or model.get("microgpt_role") != "shadow_tokens_only_not_truth_controlling"
    ):
        raise ValueError("operator additive model contract mismatch")
    names, _ = _validated_feature_row(
        model.get("feature_names") or (),
        [0] * len(model.get("feature_names") or ()),
        expected_names=expected_feature_names,
    )
    arrays: list[np.ndarray] = []
    for key in ("feature_mean", "feature_scale", "coefficients"):
        values = np.asarray(model.get(key), dtype=np.float64)
        if values.shape != (len(names),) or not np.all(np.isfinite(values)):
            raise ValueError(f"operator additive model {key} is malformed")
        arrays.append(values)
    mean, scale, coefficients = arrays
    if np.any(scale <= 0.0):
        raise ValueError("operator additive model feature scale must be positive")
    intercept = float(model.get("intercept", math.nan))
    if not math.isfinite(intercept):
        raise ValueError("operator additive model intercept is malformed")
    return names, mean, scale, coefficients, intercept


def _score_additive_model(
    model: Mapping[str, Any], features: np.ndarray
) -> tuple[np.ndarray, tuple[AdditiveReplay, ...]]:
    names, mean, scale, coefficients, intercept = _validate_model(model)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or not np.all(np.isfinite(matrix)):
        raise ValueError("operator scoring matrix is malformed")
    if np.any(matrix != np.floor(matrix)):
        raise ValueError("operator scoring matrix contains non-integer inputs")
    standardized = (matrix - mean) / scale
    contribution_matrix = standardized * coefficients
    scores = np.sum(contribution_matrix, axis=1, dtype=np.float64) + intercept
    replays: list[AdditiveReplay] = []
    for row_index in range(matrix.shape[0]):
        contributions = tuple(
            {
                "feature": name,
                "raw_value": int(matrix[row_index, column]),
                "standardized_value": float(standardized[row_index, column]),
                "coefficient": float(coefficients[column]),
                "contribution": float(contribution_matrix[row_index, column]),
            }
            for column, name in enumerate(names)
        )
        replay_score = float(np.sum(contribution_matrix[row_index], dtype=np.float64) + intercept)
        if replay_score != float(scores[row_index]):
            raise ValueError("operator contribution replay differs from score")
        replays.append(
            AdditiveReplay(
                score=replay_score,
                intercept=intercept,
                contributions=contributions,
            )
        )
    return scores, tuple(replays)


def _apply_veto(
    baseline_predictions: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    baseline = np.asarray(baseline_predictions, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if baseline.ndim != 1 or scores.shape != baseline.shape or not math.isfinite(threshold):
        raise ValueError("operator veto inputs are not aligned and finite")
    candidate = baseline & (scores >= threshold)
    if np.any(candidate & ~baseline):
        raise AssertionError("operator veto attempted a false-to-true transition")
    return candidate


def _best_veto_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    baseline_predictions: np.ndarray,
) -> tuple[float, Mapping[str, Any]]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    baseline = np.asarray(baseline_predictions, dtype=bool)
    if labels.shape != scores.shape or labels.shape != baseline.shape or labels.ndim != 1:
        raise ValueError("operator threshold inputs are not pair-aligned")
    finite = baseline & np.isfinite(scores)
    if not np.any(finite):
        raise ValueError("no baseline-positive operator scores are available")
    best: tuple[tuple[float, float, float, float], float, Mapping[str, Any]] | None = None
    for raw_threshold in np.unique(scores[finite]):
        threshold = float(raw_threshold)
        predictions = _apply_veto(baseline, scores, threshold=threshold)
        metrics = _binary_metrics(labels, predictions)
        key = (
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    if best is None:
        raise ValueError("operator veto threshold selection failed")
    return best[1], best[2]


def _crossfit_veto(
    features: np.ndarray,
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    groups: np.ndarray,
    *,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, float, Mapping[str, Any]]:
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    baseline = np.asarray(baseline_predictions, dtype=bool)
    groups = np.asarray(groups, dtype=np.int64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != labels.size
        or baseline.shape != labels.shape
        or groups.shape != labels.shape
    ):
        raise ValueError("operator crossfit inputs are not pair-aligned")
    oof_scores = np.full(labels.size, -math.inf, dtype=np.float64)
    for training_indexes, validation_indexes in _component_folds(groups):
        training = training_indexes[baseline[training_indexes]]
        validation = validation_indexes[baseline[validation_indexes]]
        if not len(validation):
            continue
        fold_model = _fit_additive_model(
            matrix[training], labels[training], feature_names=feature_names
        )
        fold_scores, _ = _score_additive_model(fold_model, matrix[validation])
        oof_scores[validation] = fold_scores
    if np.any(baseline & ~np.isfinite(oof_scores)):
        raise ValueError("operator crossfit did not score every baseline-positive row")
    threshold, metrics = _best_veto_threshold(labels, oof_scores, baseline)
    return oof_scores, threshold, metrics


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _dataset_identity(mrpc_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name, expected_sha256 in EXPECTED_MRPC_DATASET_SHA256.items():
        path = mrpc_dir.resolve().parent / name
        if not path.is_file():
            raise ValueError(f"missing MRPC dataset file: {path}")
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"MRPC dataset SHA-256 mismatch for {name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        files.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    return {"files": files, "identity_sha256": _canonical_sha256(files)}


def _load_frozen_comparator(
    path: Path,
    held_pairs: Sequence[Any],
) -> FrozenComparator:
    """Authenticate pair/route/score/prediction vectors without reading labels."""

    actual_sha256 = _file_sha256(path)
    if actual_sha256 != EXPECTED_FROZEN_REVEAL_FILE_SHA256:
        raise ValueError(
            "frozen compact comparator file identity mismatch: "
            f"expected {EXPECTED_FROZEN_REVEAL_FILE_SHA256}, got {actual_sha256}"
        )
    reveal = _load_json(path)
    if (
        reveal.get("schema_version") != "mrpc_semantic_relation_model_search_v1"
        or reveal.get("experiment_id") != "generic_symmetric_models_component_cv_v1"
        or reveal.get("phase") != "single_held_out_reveal"
        or reveal.get("diagnostic_only") is not True
        or reveal.get("hard_membership_authorized") is not False
    ):
        raise ValueError("frozen compact comparator contract mismatch")
    traces = reveal.get("held_out_pair_traces")
    if not isinstance(traces, list) or len(traces) != EXPECTED_HELD_PAIR_COUNT:
        raise ValueError("frozen compact comparator traces are missing or incomplete")
    trace_by_pair: dict[int, Mapping[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("frozen compact comparator trace is malformed")
        pair_id = trace.get("pair_id")
        if isinstance(pair_id, bool) or not isinstance(pair_id, int) or pair_id <= 0:
            raise ValueError("frozen compact comparator pair identity is malformed")
        if pair_id in trace_by_pair:
            raise ValueError("frozen compact comparator pair identity is duplicated")
        prediction = trace.get("resolver_prediction")
        routed = trace.get("routed")
        score = trace.get("resolver_score")
        if (
            not isinstance(prediction, bool)
            or not isinstance(routed, bool)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("frozen compact comparator prediction trace is malformed")
        trace_by_pair[pair_id] = trace

    pair_ids = np.asarray([int(pair.pair_id) for pair in held_pairs], dtype=np.int64)
    if set(trace_by_pair) != set(pair_ids.tolist()):
        raise ValueError("frozen compact comparator pair alignment mismatch")
    routes = np.asarray(
        [bool(trace_by_pair[int(pair_id)]["routed"]) for pair_id in pair_ids],
        dtype=bool,
    )
    scores = np.asarray(
        [float(trace_by_pair[int(pair_id)]["resolver_score"]) for pair_id in pair_ids],
        dtype=np.float64,
    )
    predictions = np.asarray(
        [bool(trace_by_pair[int(pair_id)]["resolver_prediction"]) for pair_id in pair_ids],
        dtype=bool,
    )
    identities = {
        "pair_id_sha256": _canonical_sha256(pair_ids.tolist()),
        "route_sha256": _canonical_sha256(routes.tolist()),
        "score_sha256": _canonical_sha256(scores.tolist()),
        "prediction_sha256": _canonical_sha256(predictions.tolist()),
    }
    expected_identities = {
        "pair_id_sha256": EXPECTED_HELD_PAIR_ID_SHA256,
        "route_sha256": EXPECTED_HELD_ROUTE_SHA256,
        "score_sha256": EXPECTED_HELD_SCORE_SHA256,
        "prediction_sha256": EXPECTED_HELD_PREDICTION_SHA256,
    }
    if identities != expected_identities:
        raise ValueError("frozen compact comparator vector identity mismatch")
    if int(np.count_nonzero(routes)) != EXPECTED_HELD_ROUTED_PAIR_COUNT:
        raise ValueError("frozen compact comparator routed-pair count mismatch")
    return FrozenComparator(
        pair_ids=pair_ids,
        routes=routes,
        scores=scores,
        predictions=predictions,
        trace_by_pair=trace_by_pair,
        evidence={
            "path": str(path.resolve()),
            "file_sha256": actual_sha256,
            **identities,
            "pair_count": len(pair_ids),
            "routed_pair_count": int(np.count_nonzero(routes)),
            "labels_accessed": False,
        },
    )


def _validate_comparator_labels(
    comparator: FrozenComparator,
    held_pairs: Sequence[Any],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Access comparator labels only after candidate predictions are frozen."""

    labels = np.asarray(labels, dtype=bool)
    if labels.shape != comparator.predictions.shape or len(held_pairs) != labels.size:
        raise ValueError("held labels are not pair-aligned with the comparator")
    baseline_module = importlib.import_module(
        "scripts.research.mrpc.run_mrpc_directional_nli_benchmark"
    )
    shared_validator = getattr(baseline_module, "_validated_baseline", None)
    if not callable(shared_validator):
        raise RuntimeError("shared frozen baseline validator is unavailable")
    (
        authenticated_labels,
        authenticated_predictions,
        authenticated_routes,
        authenticated_evidence,
    ) = shared_validator(Path(str(comparator.evidence["path"])), held_pairs)
    if not np.array_equal(authenticated_predictions, comparator.predictions) or not np.array_equal(
        authenticated_routes, comparator.routes
    ):
        raise ValueError("shared frozen baseline validation differs from comparator vectors")
    trace_labels = np.asarray(
        [bool(comparator.trace_by_pair[int(pair.pair_id)]["label"]) for pair in held_pairs],
        dtype=bool,
    )
    if not np.array_equal(labels, trace_labels) or not np.array_equal(labels, authenticated_labels):
        raise ValueError("official held labels differ from the frozen comparator")
    label_sha256 = _canonical_sha256(labels.tolist())
    if label_sha256 != EXPECTED_HELD_LABEL_SHA256:
        raise ValueError("frozen held label-vector identity mismatch")
    metrics = _binary_metrics(labels, comparator.predictions)
    if metrics != EXPECTED_FROZEN_HELD_OUT_METRICS:
        raise ValueError("frozen compact comparator metrics changed")
    return {
        "metrics": metrics,
        "label_sha256": label_sha256,
        "exact_label_alignment": True,
        "labels_accessed_after_candidate_freeze": True,
        "shared_validated_baseline": authenticated_evidence,
    }


def _frozen_compact_calibration_predictions(
    *,
    args: argparse.Namespace,
    calibration_pairs: Sequence[Any],
    embeddings: Mapping[str, np.ndarray],
    signatures: Mapping[str, np.ndarray],
    graphs: Mapping[str, Any],
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay the authenticated compact winner's calibration OOF predictions."""

    selection_path = args.frozen_compact_selection.resolve()
    if _file_sha256(selection_path) != EXPECTED_FROZEN_SELECTION_FILE_SHA256:
        raise ValueError("frozen compact selection file identity mismatch")
    selection = _load_json(selection_path)
    if selection.get("frozen_selection_sha256") != EXPECTED_FROZEN_SELECTION_SHA256:
        raise ValueError("frozen compact internal selection identity mismatch")
    winner = selection.get("winner")
    if winner != EXPECTED_FROZEN_WINNER:
        raise ValueError("frozen compact winner contract changed")
    neighborhoods, route_evidence = model_experiment._partition_neighborhoods(
        calibration_pairs,
        signatures,
        top_k=args.psa_top_k,
    )
    routed = model_experiment._symmetric_route(calibration_pairs, neighborhoods)
    if int(np.count_nonzero(routed)) != int(selection["routing"]["calibration_routed_pair_count"]):
        raise ValueError("frozen compact calibration route changed")
    spec = model_experiment._spec_from_payload(winner["spec"])
    features = model_experiment._feature_matrices(
        calibration_pairs,
        embeddings=embeddings,
        signatures=signatures,
        graphs=graphs,
    )[spec.feature_set]
    groups, component_count = _component_groups(calibration_pairs)
    replay = model_experiment._crossfit(
        spec,
        features,
        np.asarray(labels, dtype=bool),
        routed,
        model_experiment._group_folds(groups),
    )
    if (
        float(replay.threshold) != float(winner["threshold"])
        or dict(replay.metrics) != winner["oof_metrics"]
    ):
        raise ValueError("frozen compact calibration OOF replay changed")
    predictions = routed & (replay.oof_scores >= float(replay.threshold))
    return predictions, {
        "selection_path": str(selection_path),
        "selection_file_sha256": EXPECTED_FROZEN_SELECTION_FILE_SHA256,
        "selection_sha256": EXPECTED_FROZEN_SELECTION_SHA256,
        "oof_metrics": dict(replay.metrics),
        "binary_threshold": float(replay.threshold),
        "prediction_sha256": _canonical_sha256(predictions.tolist()),
        "routed_pair_count": int(np.count_nonzero(routed)),
        "component_count": component_count,
        "route": dict(route_evidence),
    }


def _trace_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        trace = dict(value)
    else:
        trace_method = getattr(value, "trace_dict", None)
        if not callable(trace_method):
            raise ValueError(f"{name} must be a mapping or expose trace_dict()")
        raw_trace = trace_method()
        if not isinstance(raw_trace, Mapping):
            raise ValueError(f"{name} trace must be a mapping")
        trace = dict(raw_trace)
    detached = json.loads(_canonical_json(trace))
    if not isinstance(detached, dict):
        raise ValueError(f"{name} trace must be a JSON object")
    return detached


def _ledger_api() -> tuple[Any, Any, Any]:
    try:
        module = importlib.import_module("mcp_servers.mcp_lib.directional_proposition_ledger")
    except ImportError as exc:
        raise RuntimeError(
            "directional proposition ledger is unavailable; complete its bounded "
            "library module before running this benchmark"
        ) from exc
    required = (
        "build_directional_proposition_ledger",
        "propose_membership_edges",
        "build_ledger_feature_bundle",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            "directional proposition ledger API is incomplete: " + ", ".join(missing)
        )
    return tuple(getattr(module, name) for name in required)  # type: ignore[return-value]


def _feature_bundle_row(
    bundle: Any,
    *,
    expected_names: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[int, ...], str, tuple[str, ...]]:
    names = getattr(bundle, "feature_names", None)
    microgpt_inputs_method = getattr(bundle, "microgpt_inputs", None)
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise ValueError("ledger feature bundle is missing feature_names")
    if not callable(microgpt_inputs_method):
        raise ValueError("ledger feature bundle is missing microgpt_inputs()")
    microgpt_inputs = microgpt_inputs_method()
    if not isinstance(microgpt_inputs, Mapping):
        raise ValueError("ledger MicroGPT inputs must be a named mapping")
    normalized_names = tuple(str(name) for name in names)
    if set(microgpt_inputs) != set(normalized_names):
        raise ValueError("ledger MicroGPT inputs differ from the scorer feature schema")
    values = tuple(microgpt_inputs[name] for name in normalized_names)
    safe_names, safe_values = _validated_feature_row(
        normalized_names,
        values,
        expected_names=expected_names,
    )
    feature_sha256 = _canonical_sha256(
        {"feature_names": list(safe_names), "values": list(safe_values)}
    )
    declared_sha = (
        getattr(bundle, "feature_sha256", None)
        or getattr(bundle, "bundle_sha256", None)
        or getattr(bundle, "sha256", None)
    )
    if declared_sha is not None and str(declared_sha) != feature_sha256:
        # A bundle may hash a larger trace contract. Preserve that identity in
        # its own trace, but the scorer row always uses this independently
        # reproducible schema/value hash.
        bundle_trace = _trace_mapping(bundle, name="ledger feature bundle")
        declared_from_trace = str(
            bundle_trace.get("feature_sha256")
            or bundle_trace.get("bundle_sha256")
            or bundle_trace.get("sha256")
            or ""
        )
        if declared_from_trace and declared_from_trace != str(declared_sha):
            raise ValueError("ledger feature bundle declared identity is inconsistent")
    shadow_tokens = _microgpt_shadow_tokens(safe_names, safe_values)
    return safe_names, safe_values, feature_sha256, shadow_tokens


def _load_fresh_enriched_graphs(
    texts: Sequence[str],
    *,
    graph_cache_path: Path,
    expected_graph_cache_sha256: str,
    parser_identity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach a fresh pinned operator sidecar without mutating the baseline cache."""

    if _file_sha256(graph_cache_path) != expected_graph_cache_sha256:
        raise ValueError("pinned proposition graph evidence cache identity mismatch")
    cache = _load_json(graph_cache_path)
    sentences = cache.get("sentences")
    if not isinstance(sentences, Mapping):
        raise ValueError("pinned proposition graph cache has no sentence evidence")
    try:
        extraction_module = importlib.import_module(
            "mcp_servers.tokenization_service.extractors.spacy_extractor"
        )
        evidence_module = importlib.import_module(
            "mcp_servers.mcp_lib.paraphrase_evidence_extraction"
        )
        graph_module = importlib.import_module("mcp_servers.mcp_lib.proposition_graph")
        operator_module = importlib.import_module(
            "mcp_servers.mcp_lib.grammatical_operator_inventory"
        )
    except ImportError as exc:
        raise RuntimeError("fresh grammatical-operator sidecar dependencies are missing") from exc
    extractor_class = getattr(extraction_module, "SpacyExtractor", None)
    attach_inventory = getattr(evidence_module, "attach_logical_operator_inventory", None)
    build_graph = getattr(graph_module, "build_proposition_graph", None)
    fallback_analyzer = getattr(operator_module, "analyze_grammatical_operator_inventory", None)
    if not all(
        callable(value)
        for value in (extractor_class, attach_inventory, build_graph, fallback_analyzer)
    ):
        raise RuntimeError("fresh grammatical-operator sidecar API is incomplete")
    extractor = extractor_class()
    if not getattr(extractor, "available", False):
        raise RuntimeError(
            "pinned spaCy model is unavailable; run with "
            "temp/wordnet-venv/bin/python (spaCy 3.8, en_core_web_sm 3.8.0)"
        )
    actual_parser_identity = str(getattr(extractor, "model_identity", ""))
    if actual_parser_identity != parser_identity:
        raise RuntimeError(
            "grammatical-operator parser identity mismatch: "
            f"expected {parser_identity!r}, got {actual_parser_identity!r}"
        )
    extract_inventory = getattr(extractor, "extract_grammatical_operator_inventory", None)
    if not callable(extract_inventory):
        raise RuntimeError(
            "SpacyExtractor lacks extract_grammatical_operator_inventory(); "
            "the sidecar must fail closed"
        )

    enriched: dict[str, Any] = {}
    inventory_hashes: list[str] = []
    coverage_counts: dict[str, int] = {}
    for text in sorted(set(texts)):
        text_hash = sha256_text(text)
        entry = sentences.get(text_hash)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("evidence"), Mapping):
            raise ValueError(f"pinned graph evidence is missing for sentence {text_hash}")
        try:
            inventory = extract_inventory(text)
        except Exception as exc:  # fail closed per occurrence, never silently complete
            inventory = fallback_analyzer(
                text,
                None,
                parser_identity=parser_identity,
                extraction_error=f"{type(exc).__name__}:{exc}",
            )
        inventory_trace = _trace_mapping(inventory, name="grammatical operator inventory")
        if inventory_trace.get("text_sha256") != text_hash:
            raise ValueError("grammatical operator inventory text identity mismatch")
        if inventory_trace.get("parser_identity") != parser_identity:
            raise ValueError("grammatical operator inventory parser identity mismatch")
        inventory_hash = str(inventory_trace.get("inventory_sha256") or "")
        if len(inventory_hash) != 64:
            raise ValueError("grammatical operator inventory identity is missing")
        inventory_hashes.append(inventory_hash)
        coverage = str(inventory_trace.get("coverage") or "partial")
        coverage_counts[coverage] = coverage_counts.get(coverage, 0) + 1
        projection = attach_inventory(entry["evidence"], inventory)
        enriched[text] = build_graph(projection)

    parser_evidence = {
        "parser_identity": actual_parser_identity,
        "spacy_package_version": importlib.metadata.version("spacy"),
        "spacy_model_package_version": importlib.metadata.version("en-core-web-sm"),
        "sentence_count": len(enriched),
        "inventory_hashes_sha256": _canonical_sha256(sorted(inventory_hashes)),
        "coverage_counts": dict(sorted(coverage_counts.items())),
        "baseline_graph_cache_path": str(graph_cache_path.resolve()),
        "baseline_graph_cache_sha256": expected_graph_cache_sha256,
        "baseline_graph_cache_mutated": False,
    }
    parser_evidence["identity_sha256"] = _canonical_sha256(parser_evidence)
    return enriched, parser_evidence


def _ledger_feature_traces(
    pairs: Sequence[Any],
    graphs: Mapping[str, Any],
    *,
    expected_feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[LedgerFeatureTrace, ...]]:
    build_ledger, propose_edges, build_bundle = _ledger_api()
    matrix_rows: list[tuple[int, ...]] = []
    traces: list[LedgerFeatureTrace] = []
    schema: tuple[str, ...] | None = (
        tuple(expected_feature_names) if expected_feature_names is not None else None
    )
    for pair in pairs:
        left_graph = graphs[pair.sentence1]
        right_graph = graphs[pair.sentence2]
        ledger = build_ledger(left_graph, right_graph)
        edges = propose_edges(ledger)
        bundle = build_bundle(ledger, edge_proposals=edges)
        names, values, feature_sha256, shadow_tokens = _feature_bundle_row(
            bundle, expected_names=schema
        )
        if schema is None:
            schema = names
        reverse_ledger = build_ledger(right_graph, left_graph)
        reverse_edges = propose_edges(reverse_ledger)
        reverse_bundle = build_bundle(reverse_ledger, edge_proposals=reverse_edges)
        reverse_names, reverse_values, _reverse_sha, reverse_tokens = _feature_bundle_row(
            reverse_bundle, expected_names=schema
        )
        if reverse_names != names or reverse_values != values or reverse_tokens != shadow_tokens:
            raise ValueError("ledger scorer features are not endpoint-order invariant")
        ledger_trace = _trace_mapping(ledger, name="directional proposition ledger")
        edge_trace = _trace_mapping(edges, name="ledger edge proposal set")
        coverage = {
            key: value
            for key, value in ledger_trace.items()
            if "coverage" in str(key) or key in {"status", "unresolved"}
        }
        matrix_rows.append(values)
        traces.append(
            LedgerFeatureTrace(
                feature_names=names,
                values=values,
                feature_sha256=feature_sha256,
                microgpt_shadow_tokens=shadow_tokens,
                ledger=ledger_trace,
                edges=edge_trace,
                coverage=coverage,
            )
        )
    if schema is None or not matrix_rows:
        raise ValueError("operator ledger emitted no feature rows")
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.shape != (len(pairs), len(schema)):
        raise ValueError("operator ledger feature matrix is malformed")
    return matrix, tuple(traces)


def _source_identity() -> dict[str, str]:
    paths = {
        "benchmark": Path(__file__).resolve(),
        "grammatical_operator_inventory": (
            REPOSITORY_ROOT
            / "app"
            / "mcp_servers"
            / "mcp_lib"
            / "grammatical_operator_inventory.py"
        ),
        "directional_proposition_ledger": (
            REPOSITORY_ROOT
            / "app"
            / "mcp_servers"
            / "mcp_lib"
            / "directional_proposition_ledger.py"
        ),
        "operator_attachment": (
            REPOSITORY_ROOT
            / "app"
            / "mcp_servers"
            / "mcp_lib"
            / "paraphrase_evidence_extraction.py"
        ),
        "proposition_graph": (
            REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "proposition_graph.py"
        ),
        "spacy_adapter": (
            REPOSITORY_ROOT
            / "app"
            / "mcp_servers"
            / "tokenization_service"
            / "extractors"
            / "spacy_extractor.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError("operator benchmark source is missing: " + ", ".join(missing))
    return {name: _file_sha256(path) for name, path in paths.items()}


def _load_label_free_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[Any],
    Mapping[int, str],
    Mapping[str, np.ndarray],
    Mapping[str, np.ndarray],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    """Use the established loader whose PairInput intentionally omits Quality."""

    pairs, split, embeddings, signatures, graphs, evidence = model_experiment._load_inputs(args)
    if len(pairs) != 1_725:
        raise ValueError("MRPC pair inventory changed")
    calibration_count = sum(split[pair.pair_id] == "calibration" for pair in pairs)
    held_count = sum(split[pair.pair_id] == "held_out" for pair in pairs)
    if (calibration_count, held_count) != (618, EXPECTED_HELD_PAIR_COUNT):
        raise ValueError("MRPC sentence-component partition changed")
    return pairs, split, embeddings, signatures, graphs, evidence


def _load_calibration_labels(
    mrpc_dir: Path,
    calibration_pairs: Sequence[Any],
) -> tuple[np.ndarray, frozenset[int]]:
    """Materialize exactly the 618 calibration labels and no held labels."""

    calibration_ids = frozenset(int(pair.pair_id) for pair in calibration_pairs)
    if len(calibration_pairs) != 618 or len(calibration_ids) != 618:
        raise ValueError("operator selection requires exactly 618 calibration pairs")
    label_map = model_experiment._load_selected_labels(mrpc_dir.resolve(), calibration_ids)
    if label_map.keys() != calibration_ids:
        raise ValueError("operator selection label loader crossed its calibration boundary")
    return (
        np.asarray([label_map[pair.pair_id] for pair in calibration_pairs], dtype=bool),
        calibration_ids,
    )


def _freeze_then_load_held_labels(
    *,
    marker_path: Path,
    marker: Mapping[str, Any],
    mrpc_dir: Path,
    held_pairs: Sequence[Any],
) -> tuple[np.ndarray, frozenset[int]]:
    """Persist the candidate marker before requesting exactly the held labels."""

    held_ids = frozenset(int(pair.pair_id) for pair in held_pairs)
    if len(held_pairs) != EXPECTED_HELD_PAIR_COUNT or len(held_ids) != len(held_pairs):
        raise ValueError("operator reveal requires exactly 1,107 unique held pairs")
    _exclusive_json(marker_path.resolve(), marker)
    label_map = model_experiment._load_selected_labels(mrpc_dir.resolve(), held_ids)
    if label_map.keys() != held_ids:
        raise ValueError("operator reveal label loader crossed its held boundary")
    return (
        np.asarray([label_map[pair.pair_id] for pair in held_pairs], dtype=bool),
        held_ids,
    )


def _feature_matrix_sha256(feature_names: Sequence[str], matrix: np.ndarray) -> str:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("feature matrix identity inputs are malformed")
    integer_rows = [[int(value) for value in row] for row in values]
    if any(
        float(integer_rows[i][j]) != values[i, j]
        for i in range(len(integer_rows))
        for j in range(len(feature_names))
    ):
        raise ValueError("operator feature matrix contains a non-integer value")
    return _canonical_sha256({"feature_names": list(feature_names), "rows": integer_rows})


def _selection_payload(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _dataset_identity(args.mrpc_dir.resolve())
    pairs, split, embeddings, signatures, frozen_graphs, input_evidence = _load_label_free_inputs(
        args
    )
    calibration_pairs = [pair for pair in pairs if split[pair.pair_id] == "calibration"]
    # This is the only label materialization in selection. Held IDs never
    # enter this call and PairInput carries no Quality field.
    labels, _calibration_ids = _load_calibration_labels(args.mrpc_dir.resolve(), calibration_pairs)
    baseline_predictions, baseline_evidence = _frozen_compact_calibration_predictions(
        args=args,
        calibration_pairs=calibration_pairs,
        embeddings=embeddings,
        signatures=signatures,
        graphs=frozen_graphs,
        labels=labels,
    )
    texts = [text for pair in calibration_pairs for text in (pair.sentence1, pair.sentence2)]
    enriched_graphs, parser_evidence = _load_fresh_enriched_graphs(
        texts,
        graph_cache_path=args.graph_cache.resolve(),
        expected_graph_cache_sha256=args.graph_cache_sha256,
        parser_identity=args.operator_parser_identity,
    )
    feature_matrix, feature_traces = _ledger_feature_traces(
        calibration_pairs,
        enriched_graphs,
    )
    feature_names = feature_traces[0].feature_names
    if any(trace.feature_names != feature_names for trace in feature_traces):
        raise ValueError("calibration ledger feature schema is not fixed")
    groups, component_count = _component_groups(calibration_pairs)
    oof_scores, threshold, oof_metrics = _crossfit_veto(
        feature_matrix,
        labels,
        baseline_predictions,
        groups,
        feature_names=feature_names,
    )
    oof_predictions = _apply_veto(
        baseline_predictions,
        oof_scores,
        threshold=threshold,
    )
    if _binary_metrics(labels, oof_predictions) != oof_metrics:
        raise ValueError("operator OOF metric replay mismatch")
    training = baseline_predictions
    model = _fit_additive_model(
        feature_matrix[training],
        labels[training],
        feature_names=feature_names,
    )
    full_scores, full_replays = _score_additive_model(model, feature_matrix)
    if len(full_replays) != len(calibration_pairs):
        raise ValueError("operator additive calibration replay is incomplete")
    baseline_metrics = _binary_metrics(labels, baseline_predictions)
    if baseline_metrics != EXPECTED_FROZEN_WINNER["oof_metrics"]:
        raise ValueError("operator selection comparator is not the frozen OOF lane")
    microgpt_vocabulary = sorted(
        {token for trace in feature_traces for token in trace.microgpt_shadow_tokens}
    )
    source_identity = _source_identity()
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "calibration_operator_veto_frozen_before_held_out_label_loading",
        "evaluation_class": "post_hoc_mrpc_development_diagnostic",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "edge_write_authorized": False,
        "membership_effect": "none",
        "scope_constraints": {
            "candidate_policy": "frozen_baseline_prediction AND operator_score>=threshold",
            "allowed_transition": "true_to_false_only",
            "new_edge_types": False,
            "triplets": False,
            "stitch_integration": False,
            "group_propagation": False,
            "cross_encoder": False,
            "corpus_specific_phrase_rules": False,
            "microgpt_truth_controlling": False,
        },
        "split": {
            "seed": BLIND_SPLIT_SEED,
            "unit": "connected_component_of_normalized_sentence_identity",
            "calibration_pair_count": len(calibration_pairs),
            "calibration_component_count": component_count,
            "fold_count": COMPONENT_FOLD_COUNT,
            "fold_seed": COMPONENT_FOLD_SEED,
            "held_out_labels_materialized": False,
        },
        "input_identity": {
            "dataset": dataset,
            "graph_cache_sha256": args.graph_cache_sha256,
            "graph_parser_identity": args.graph_parser_identity,
            "operator_parser_identity": args.operator_parser_identity,
            "model_input_evidence": input_evidence,
            "source_identity": source_identity,
        },
        "fresh_operator_sidecar": parser_evidence,
        "feature_contract": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(feature_names),
            "feature_count": len(feature_names),
            "values": "nonnegative_integer_counts_and_category_one_hots_only",
            "forbidden_inputs": sorted(_FORBIDDEN_FEATURE_FRAGMENTS),
            "matrix_sha256": _feature_matrix_sha256(feature_names, feature_matrix),
            "pair_id_sha256": _canonical_sha256([pair.pair_id for pair in calibration_pairs]),
        },
        "microgpt_shadow": {
            "truth_controlling": False,
            "switch_core_invoked": False,
            "tokens": microgpt_vocabulary,
            "token_count": len(microgpt_vocabulary),
            "tokens_sha256": _canonical_sha256(microgpt_vocabulary),
        },
        "frozen_calibration_comparator": {
            **baseline_evidence,
            "metrics": baseline_metrics,
        },
        "selection": {
            "model_family": "ridge",
            "fixed_alpha": FIXED_RIDGE_ALPHA,
            "class_weight": None,
            "solver": "cholesky",
            "training_surface": "frozen_baseline_positive_rows_only",
            "training_row_count": int(np.count_nonzero(training)),
            "threshold_objective": "maximum component-cross-fitted overlay F1, then precision, recall, threshold",
            "binary_threshold": threshold,
            "baseline_oof_metrics": baseline_metrics,
            "candidate_oof_metrics": dict(oof_metrics),
            "candidate_oof_f1_gain": float(oof_metrics["f1"] - baseline_metrics["f1"]),
            "oof_score_sha256": _canonical_sha256(_optional_score_vector(oof_scores)),
            "oof_unscored_baseline_negative_count": int(np.count_nonzero(~np.isfinite(oof_scores))),
            "oof_prediction_sha256": _canonical_sha256(oof_predictions.tolist()),
            "fitted_all_calibration_score_sha256": _canonical_sha256(full_scores.tolist()),
            "model": model,
        },
        "authority": {
            "classification": "diagnostic_only",
            "membership": "none",
            "edge_write": "none",
            "database_write": "none",
            "service_integration": "none",
        },
    }
    return _payload_with_sha(payload, key="selection_sha256")


def _validate_selection(
    selection: Mapping[str, Any], args: argparse.Namespace
) -> tuple[tuple[str, ...], Mapping[str, Any], float]:
    _validate_payload_sha(selection, key="selection_sha256", name="operator selection")
    if (
        selection.get("schema_version") != SELECTION_SCHEMA_VERSION
        or selection.get("experiment_id") != EXPERIMENT_ID
        or selection.get("phase")
        != "calibration_operator_veto_frozen_before_held_out_label_loading"
        or selection.get("diagnostic_only") is not True
        or selection.get("hard_membership_authorized") is not False
        or selection.get("edge_write_authorized") is not False
        or selection.get("membership_effect") != "none"
    ):
        raise ValueError("operator selection contract mismatch")
    split = selection.get("split") or {}
    if (
        split.get("calibration_pair_count") != 618
        or split.get("held_out_labels_materialized") is not False
        or split.get("fold_count") != COMPONENT_FOLD_COUNT
        or split.get("fold_seed") != COMPONENT_FOLD_SEED
    ):
        raise ValueError("operator selection label-isolation contract mismatch")
    scope = selection.get("scope_constraints") or {}
    if scope.get("allowed_transition") != "true_to_false_only" or any(
        scope.get(key) is not False
        for key in (
            "new_edge_types",
            "triplets",
            "stitch_integration",
            "group_propagation",
            "cross_encoder",
            "corpus_specific_phrase_rules",
            "microgpt_truth_controlling",
        )
    ):
        raise ValueError("operator selection experiment scope changed")
    input_identity = selection.get("input_identity") or {}
    if (
        input_identity.get("graph_cache_sha256") != args.graph_cache_sha256
        or input_identity.get("graph_parser_identity") != args.graph_parser_identity
        or input_identity.get("operator_parser_identity") != args.operator_parser_identity
        or input_identity.get("source_identity") != _source_identity()
    ):
        raise ValueError("operator selection source/input identity changed")
    dataset = _dataset_identity(args.mrpc_dir.resolve())
    if input_identity.get("dataset") != dataset:
        raise ValueError("operator selection MRPC dataset identity changed")
    feature_contract = selection.get("feature_contract") or {}
    feature_names, _ = _validated_feature_row(
        feature_contract.get("feature_names") or (),
        [0] * len(feature_contract.get("feature_names") or ()),
    )
    if feature_contract.get("schema_version") != FEATURE_SCHEMA_VERSION or feature_contract.get(
        "feature_count"
    ) != len(feature_names):
        raise ValueError("operator selection feature contract mismatch")
    microgpt = selection.get("microgpt_shadow") or {}
    if (
        microgpt.get("truth_controlling") is not False
        or microgpt.get("switch_core_invoked") is not False
    ):
        raise ValueError("operator selection improperly authorizes MicroGPT truth")
    selected = selection.get("selection") or {}
    if (
        selected.get("model_family") != "ridge"
        or float(selected.get("fixed_alpha", math.nan)) != FIXED_RIDGE_ALPHA
        or selected.get("class_weight") is not None
        or selected.get("solver") != "cholesky"
    ):
        raise ValueError("operator selection model contract mismatch")
    model = selected.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("operator selection model is missing")
    _validate_model(model, expected_feature_names=feature_names)
    threshold = float(selected.get("binary_threshold", math.nan))
    if not math.isfinite(threshold):
        raise ValueError("operator selection threshold is invalid")
    return feature_names, model, threshold


def _transition_name(label: bool, baseline: bool, candidate: bool) -> str:
    if baseline == candidate:
        return "unchanged"
    if not baseline or candidate:
        raise ValueError("operator veto emitted a forbidden prediction transition")
    return "true_positive_to_false_negative" if label else "false_positive_to_true_negative"


def _changed_prediction_records(
    *,
    pairs: Sequence[Any],
    labels: np.ndarray,
    comparator: FrozenComparator,
    candidate_predictions: np.ndarray,
    candidate_scores: np.ndarray,
    threshold: float,
    feature_traces: Sequence[LedgerFeatureTrace],
    additive_replays: Sequence[AdditiveReplay],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels = np.asarray(labels, dtype=bool)
    candidate = np.asarray(candidate_predictions, dtype=bool)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    expected_shape = comparator.predictions.shape
    if (
        labels.shape != expected_shape
        or candidate.shape != expected_shape
        or scores.shape != expected_shape
        or len(pairs) != len(feature_traces)
        or len(pairs) != len(additive_replays)
        or len(pairs) != labels.size
    ):
        raise ValueError("changed-prediction inputs are not pair-aligned")
    changed_indexes = np.flatnonzero(candidate != comparator.predictions)
    records: list[dict[str, Any]] = []
    transition_counts = {
        "false_positive_to_true_negative": 0,
        "true_positive_to_false_negative": 0,
        "false_negative_to_true_positive": 0,
        "true_negative_to_false_positive": 0,
    }
    for index in changed_indexes:
        pair = pairs[int(index)]
        label = bool(labels[index])
        baseline = bool(comparator.predictions[index])
        prediction = bool(candidate[index])
        transition = _transition_name(label, baseline, prediction)
        transition_counts[transition] += 1
        trace = feature_traces[int(index)]
        replay = additive_replays[int(index)]
        if replay.score != float(scores[index]):
            raise ValueError("changed prediction additive replay is not exact")
        contribution_total = float(
            np.sum(
                np.asarray(
                    [float(value["contribution"]) for value in replay.contributions],
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
            + replay.intercept
        )
        if contribution_total != replay.score:
            raise ValueError("changed prediction contribution sum is not exact")
        named_inputs = dict(zip(trace.feature_names, trace.values, strict=True))
        records.append(
            {
                "pair_id": int(pair.pair_id),
                "label": label,
                "transition": transition,
                "sentence1": pair.sentence1,
                "sentence2": pair.sentence2,
                "sentence1_sha256": sha256_text(pair.sentence1),
                "sentence2_sha256": sha256_text(pair.sentence2),
                "routed": bool(comparator.routes[index]),
                "baseline_score": float(comparator.scores[index]),
                "baseline_prediction": baseline,
                "candidate_score": float(scores[index]),
                "candidate_threshold": threshold,
                "candidate_prediction": prediction,
                "directional_ledger": trace.ledger,
                "plural_existing_edge_set": trace.edges,
                "coverage_and_fail_closed_state": trace.coverage,
                "named_integer_inputs": named_inputs,
                "feature_row_sha256": trace.feature_sha256,
                "microgpt_shadow_tokens": list(trace.microgpt_shadow_tokens),
                "microgpt_truth_controlling": False,
                "additive_scoring_replay": {
                    "intercept": replay.intercept,
                    "contributions": list(replay.contributions),
                    "replayed_score": replay.score,
                    "exact": True,
                },
                "membership_effect": "none",
                "edge_write_authorized": False,
            }
        )
    records.sort(key=lambda value: int(value["pair_id"]))
    if (
        len(records) != int(np.count_nonzero(candidate != comparator.predictions))
        or len({int(record["pair_id"]) for record in records}) != len(records)
        or sum(transition_counts.values()) != len(records)
    ):
        raise ValueError("changed prediction report is incomplete")
    return records, transition_counts


def _all_pair_score_replays(
    *,
    pairs: Sequence[Any],
    comparator: FrozenComparator,
    candidate_predictions: np.ndarray,
    candidate_scores: np.ndarray,
    threshold: float,
    feature_traces: Sequence[LedgerFeatureTrace],
    additive_replays: Sequence[AdditiveReplay],
) -> list[dict[str, Any]]:
    """Emit compact but complete additive evidence for every held pair."""

    candidate = np.asarray(candidate_predictions, dtype=bool)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if (
        candidate.shape != comparator.predictions.shape
        or scores.shape != comparator.predictions.shape
        or len(pairs) != len(feature_traces)
        or len(pairs) != len(additive_replays)
        or len(pairs) != candidate.size
    ):
        raise ValueError("all-pair scoring replay inputs are not aligned")
    records: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        trace = feature_traces[index]
        replay = additive_replays[index]
        if replay.score != float(scores[index]):
            raise ValueError("all-pair additive replay differs from candidate score")
        standardized = [float(value["standardized_value"]) for value in replay.contributions]
        contributions = [float(value["contribution"]) for value in replay.contributions]
        if len(trace.values) != len(standardized) or len(trace.values) != len(contributions):
            raise ValueError("all-pair contribution vector is incomplete")
        replayed_score = float(
            np.sum(np.asarray(contributions, dtype=np.float64), dtype=np.float64) + replay.intercept
        )
        if replayed_score != replay.score:
            raise ValueError("all-pair contribution sum does not exactly replay")
        record = {
            "pair_id": int(pair.pair_id),
            "baseline_prediction": bool(comparator.predictions[index]),
            "candidate_prediction": bool(candidate[index]),
            "candidate_score": float(scores[index]),
            "candidate_threshold": threshold,
            "raw_integer_values": list(trace.values),
            "standardized_values": standardized,
            "feature_contributions": contributions,
            "intercept": replay.intercept,
            "exact_contribution_replay": True,
        }
        record["score_replay_sha256"] = _canonical_sha256(record)
        records.append(record)
    if (
        len(records) != EXPECTED_HELD_PAIR_COUNT
        or len({int(record["pair_id"]) for record in records}) != len(records)
        or [int(record["pair_id"]) for record in records] != [int(pair.pair_id) for pair in pairs]
    ):
        raise ValueError("all-pair scoring replay is incomplete")
    return records


def _report_payload(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = args.selection.resolve()
    selection = _load_json(selection_path)
    feature_names, model, threshold = _validate_selection(selection, args)
    dataset = _dataset_identity(args.mrpc_dir.resolve())
    pairs, split, _embeddings, _signatures, _graphs, input_evidence = _load_label_free_inputs(args)
    held_pairs = [pair for pair in pairs if split[pair.pair_id] == "held_out"]
    comparator = _load_frozen_comparator(args.baseline_reveal.resolve(), held_pairs)
    texts = [text for pair in held_pairs for text in (pair.sentence1, pair.sentence2)]
    enriched_graphs, parser_evidence = _load_fresh_enriched_graphs(
        texts,
        graph_cache_path=args.graph_cache.resolve(),
        expected_graph_cache_sha256=args.graph_cache_sha256,
        parser_identity=args.operator_parser_identity,
    )
    selected_parser = selection.get("fresh_operator_sidecar") or {}
    for key in (
        "parser_identity",
        "spacy_package_version",
        "spacy_model_package_version",
        "baseline_graph_cache_sha256",
    ):
        if selected_parser.get(key) != parser_evidence.get(key):
            raise ValueError(f"held operator sidecar changed selection identity for {key}")
    feature_matrix, feature_traces = _ledger_feature_traces(
        held_pairs,
        enriched_graphs,
        expected_feature_names=feature_names,
    )
    scores, additive_replays = _score_additive_model(model, feature_matrix)
    candidate_predictions = _apply_veto(
        comparator.predictions,
        scores,
        threshold=threshold,
    )
    if np.any(candidate_predictions & ~comparator.predictions):
        raise ValueError("operator candidate violated the one-way veto contract")
    prediction_sha256 = _canonical_sha256(candidate_predictions.tolist())

    # Predictions and their complete numeric replay are frozen before the only
    # held-label loader is invoked below.
    marker = _payload_with_sha(
        {
            "schema_version": "mrpc_operator_ledger_reveal_marker_v1",
            "experiment_id": EXPERIMENT_ID,
            "selection_sha256": selection["selection_sha256"],
            "comparator_file_sha256": EXPECTED_FROZEN_REVEAL_FILE_SHA256,
            "held_pair_id_sha256": _canonical_sha256([pair.pair_id for pair in held_pairs]),
            "held_feature_matrix_sha256": _feature_matrix_sha256(feature_names, feature_matrix),
            "candidate_score_sha256": _canonical_sha256(scores.tolist()),
            "candidate_prediction_sha256": prediction_sha256,
            "meaning": "candidate predictions are frozen; held labels may now be loaded once",
        },
        key="marker_sha256",
    )
    labels, _held_ids = _freeze_then_load_held_labels(
        marker_path=args.reveal_marker.resolve(),
        marker=marker,
        mrpc_dir=args.mrpc_dir.resolve(),
        held_pairs=held_pairs,
    )
    comparator_label_evidence = _validate_comparator_labels(comparator, held_pairs, labels)
    metrics = _binary_metrics(labels, candidate_predictions)
    baseline_metrics = comparator_label_evidence["metrics"]
    f1_gain = float(metrics["f1"] - baseline_metrics["f1"])
    bootstrap = _paired_f1_bootstrap(
        labels,
        candidate_predictions,
        comparator.predictions,
        sample_count=PAIRED_BOOTSTRAP_SAMPLES,
        seed=PAIRED_BOOTSTRAP_SEED,
    )
    if float(bootstrap["point_f1_delta"]) != f1_gain:
        raise ValueError("paired bootstrap point estimate differs from metric delta")
    changed_records, transition_counts = _changed_prediction_records(
        pairs=held_pairs,
        labels=labels,
        comparator=comparator,
        candidate_predictions=candidate_predictions,
        candidate_scores=scores,
        threshold=threshold,
        feature_traces=feature_traces,
        additive_replays=additive_replays,
    )
    all_pair_replays = _all_pair_score_replays(
        pairs=held_pairs,
        comparator=comparator,
        candidate_predictions=candidate_predictions,
        candidate_scores=scores,
        threshold=threshold,
        feature_traces=feature_traces,
        additive_replays=additive_replays,
    )
    held_microgpt_tokens = sorted(
        {token for trace in feature_traces for token in trace.microgpt_shadow_tokens}
    )
    changed_ids = [int(record["pair_id"]) for record in changed_records]
    material_gain = f1_gain >= args.minimum_f1_gain
    confidence_gain = bool(bootstrap["lower_bound_strictly_positive"])
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_class": "post_hoc_mrpc_development_diagnostic",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "edge_write_authorized": False,
        "membership_effect": "none",
        "selection": {
            "path": str(selection_path),
            "selection_sha256": selection["selection_sha256"],
            "model_sha256": model["model_sha256"],
            "binary_threshold": threshold,
        },
        "held_out": {
            "pair_count": len(held_pairs),
            "baseline_metrics": baseline_metrics,
            "candidate_metrics": metrics,
            "f1_gain_over_frozen_0_8564": f1_gain,
            "paired_bootstrap_f1_difference": bootstrap,
            "minimum_required_f1_gain": args.minimum_f1_gain,
            "passed_material_gain_gate": material_gain,
            "passed_bootstrap_confidence_gate": confidence_gain,
            "passed_diagnostic_promotion_gate": material_gain and confidence_gain,
            "candidate_prediction_sha256": prediction_sha256,
            "changed_prediction_count": len(changed_records),
            "transition_counts": transition_counts,
            "changed_pair_ids_sha256": _canonical_sha256(changed_ids),
            "one_way_veto_verified": True,
        },
        "evidence": {
            "dataset": dataset,
            "immutable_comparator": {
                **dict(comparator.evidence),
                **comparator_label_evidence,
            },
            "label_isolation": {
                "pair_inputs_loaded_without_quality": True,
                "candidate_predictions_frozen_before_held_label_loader": True,
                "reveal_marker_path": str(args.reveal_marker.resolve()),
                "reveal_marker_sha256": _file_sha256(args.reveal_marker.resolve()),
            },
            "fresh_operator_sidecar": parser_evidence,
            "feature_contract": {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": list(feature_names),
                "held_matrix_sha256": _feature_matrix_sha256(feature_names, feature_matrix),
                "held_score_sha256": _canonical_sha256(scores.tolist()),
                "named_integer_inputs_only": True,
                "raw_text_or_lexical_inputs_to_scorer": False,
            },
            "source_identity": _source_identity(),
            "model_input_evidence": input_evidence,
        },
        "microgpt_shadow": {
            "truth_controlling": False,
            "switch_core_invoked": False,
            "input_surface": "same named integer counts and category one-hots",
            "tokens": held_microgpt_tokens,
            "token_count": len(held_microgpt_tokens),
            "tokens_sha256": _canonical_sha256(held_microgpt_tokens),
        },
        "transparent_scoring_contract": {
            "feature_names": list(feature_names),
            "feature_mean": list(model["feature_mean"]),
            "feature_scale": list(model["feature_scale"]),
            "coefficients": list(model["coefficients"]),
            "intercept": float(model["intercept"]),
            "binary_threshold": threshold,
            "equation": "score = intercept + sum(((raw-mean)/scale)*coefficient)",
            "all_pair_replay_count": len(all_pair_replays),
            "all_pair_replay_sha256": _canonical_sha256(all_pair_replays),
        },
        "held_out_pair_score_replays": all_pair_replays,
        "changed_predictions": changed_records,
        "authority": {
            "classification": "diagnostic_only",
            "membership": "none",
            "edge_write": "none",
            "database_write": "none",
            "service_integration": "none",
        },
        "caveats": [
            "MRPC held labels were visible during earlier repository development.",
            "This is an operator-only diagnostic comparison, not external generalization.",
            "The scorer is a transparent calibration-trained veto and cannot add positives.",
            "MicroGPT tokens are exposed as a shadow interface and never control truth.",
            "Ledger edge proposals are diagnostic existing-type categories and are never written.",
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "reveal"), required=True)
    parser.add_argument(
        "--mrpc-dir",
        type=Path,
        default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sense-cache", type=Path, default=DEFAULT_SENSE_CACHE)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--graph-cache-sha256", default=PINNED_GRAPH_CACHE_SHA256)
    parser.add_argument("--graph-parser-identity", default=PINNED_GRAPH_PARSER_IDENTITY)
    parser.add_argument("--operator-parser-identity", default=PINNED_GRAPH_PARSER_IDENTITY)
    parser.add_argument(
        "--pinned-guide-embeddings",
        type=Path,
        default=DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--psa-top-k", type=int, default=32)
    parser.add_argument(
        "--frozen-compact-selection",
        type=Path,
        default=DEFAULT_FROZEN_COMPACT_SELECTION,
    )
    parser.add_argument("--baseline-reveal", type=Path, default=DEFAULT_BASELINE_REVEAL)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reveal-marker", type=Path, default=DEFAULT_REVEAL_MARKER)
    parser.add_argument(
        "--minimum-f1-gain",
        type=_nonnegative_finite_float,
        default=DEFAULT_MINIMUM_F1_GAIN,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.psa_top_k <= 0:
        raise ValueError("psa-top-k must be positive")
    if args.phase == "select":
        selection = _selection_payload(args)
        _exclusive_json(args.selection.resolve(), selection)
        print(
            json.dumps(
                {
                    "status": "operator_veto_selection_frozen",
                    "selection_path": str(args.selection.resolve()),
                    "selection_file_sha256": _file_sha256(args.selection.resolve()),
                    "selection_sha256": selection["selection_sha256"],
                    "calibration": selection["selection"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    report_path = args.report.resolve()
    if report_path.exists() or args.reveal_marker.resolve().exists():
        raise ValueError(
            "refusing a second operator held-out reveal; remove neither authenticated "
            "artifact and start a newly named experiment instead"
        )
    report = _report_payload(args)
    _exclusive_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "operator_veto_held_out_revealed_once",
                "report_path": str(report_path),
                "report_file_sha256": _file_sha256(report_path),
                "metrics": report["held_out"]["candidate_metrics"],
                "f1_gain": report["held_out"]["f1_gain_over_frozen_0_8564"],
                "paired_bootstrap": report["held_out"]["paired_bootstrap_f1_difference"],
                "changed_prediction_count": report["held_out"]["changed_prediction_count"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
