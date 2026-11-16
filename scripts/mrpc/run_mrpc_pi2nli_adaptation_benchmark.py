#!/usr/bin/env python3
"""Freeze and reveal one calibration-adapted PI2NLI fusion on MRPC.

The controlling candidate keeps the frozen split, PSA route, 341 compact
features, two sorted log-entailment features, and fixed ridge estimator.  The
only changed signal is that the entailment values come from a calibration-
adapted clone of the exact pinned NLI model.

The phases deliberately separate calibration training, selection, final model
freezing, label-free held inference, and the one permitted held-label reveal.
The adapted clone is a symmetric paraphrase feature source only.  Its arbitrary
negative-direction conversion has no directional-relation, atomic-support,
membership, service, database, or edge-write authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

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
from scripts.research.mrpc.run_mrpc_directional_nli_benchmark import (  # noqa: E402
    _require_route_parity,
    _validated_baseline,
)
from scripts.research.mrpc.run_mrpc_directional_nli_fusion_benchmark import (  # noqa: E402
    FIXED_CLASS_WEIGHT,
    FIXED_RIDGE_ALPHA,
    FUSION_FEATURE_DIMENSION,
    FUSION_FEATURE_SCHEMA_VERSION,
    _array_sha256,
    _authenticated_baseline_margins,
    _fit_numeric_model,
    _fixed_spec,
    _frozen_compact_parity,
    _score_numeric_model,
    _validate_numeric_model,
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
    EXPECTED_FROZEN_ARTIFACT_SHA256,
    EXPECTED_FROZEN_CONTRACT_FILE_SHA256,
    EXPECTED_FROZEN_HELD_OUT_METRICS,
    EXPECTED_FROZEN_REVEAL_FILE_SHA256,
    EXPECTED_FROZEN_WINNER,
    PAIRED_BOOTSTRAP_SAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    _dataset_identity,
    _paired_f1_bootstrap,
)

from mcp_servers.mcp_lib.directional_nli_features import (  # noqa: E402
    DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION,
    DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_NAMES,
    DIRECTIONAL_NLI_LOG_ENTAILMENT_SCHEMA_VERSION,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import _binary_metrics  # noqa: E402
from mcp_servers.mcp_lib.semantic_relation_features import (  # noqa: E402
    COMPACT_GENERIC_FEATURE_DIMENSION,
    COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
    load_compact_generic_linear_artifact,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    PSA_SIGNATURE_CANDIDATE_LIMIT,
)

EXPERIMENT_ID = "mrpc_pi2nli_preserved_head_fusion_v1"
TRAINING_MANIFEST_SCHEMA_VERSION = "mrpc_pi2nli_training_manifest_v1"
CROSSFIT_CACHE_SCHEMA_VERSION = "mrpc_pi2nli_crossfit_cache_v1"
SELECTION_SCHEMA_VERSION = "mrpc_pi2nli_selection_v1"
FINAL_FIT_CACHE_SCHEMA_VERSION = "mrpc_pi2nli_final_fit_cache_v1"
RESOLVER_SCHEMA_VERSION = "mrpc_pi2nli_frozen_resolver_v1"
HELD_MANIFEST_SCHEMA_VERSION = "mrpc_pi2nli_held_request_manifest_v1"
HELD_CACHE_SCHEMA_VERSION = "mrpc_pi2nli_held_inference_cache_v1"
REPORT_SCHEMA_VERSION = "mrpc_pi2nli_benchmark_v1"
REVEAL_MARKER_SCHEMA_VERSION = "mrpc_pi2nli_reveal_started_v1"

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
MODEL_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
MODEL_WEIGHTS_RELATIVE_PATH = "model.safetensors"
MODEL_WEIGHTS_SHA256 = "4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9"
BASE_MODEL_FILE_SHA256 = {
    "README.md": "4aeb5585a9eeb71b2d572249f9faa22ef3378c6abee3eeaffaae8c961068a2c6",
    "added_tokens.json": ("a4b6bfe668f2b3cf6f0cd535e98a0663d2d0d4a4a15f13075ad3597d33985a23"),
    "config.json": "8d9f07bf7ba54a6fc3b1962483056f94c39dcf188db4cf61843e1c88f94b2342",
    "model.safetensors": MODEL_WEIGHTS_SHA256,
    "special_tokens_map.json": ("ed7c099c988dbb414b18a6980d20cb57b91b7cd119f6f6941eb364b0e892e712"),
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "tokenizer.json": ("5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964"),
    "tokenizer_config.json": ("f3eecd07c370ef0bf7dd3780d3cd68cf9c8b00c267e21a208ddcd8f82bfec1a6"),
}

PI2NLI_MAPPING_SEED = "mrpc-pi2nli-deterministic-mapping-v1"
TRAINING_EPOCHS = tuple(range(1, 13))
TRAINING_PROTOCOL = {
    "protocol": "pi2nli_deterministic_three_way_v1",
    "mapping_seed": PI2NLI_MAPPING_SEED,
    "optimization_seed": 0,
    "learning_rate": 1e-5,
    "weight_decay": 0.001,
    "adam_epsilon": 1e-8,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "max_length": 156,
    "epochs": list(TRAINING_EPOCHS),
    "head_initialization": "preserve_pretrained_three_way_nli_head",
    "optimizer": "torch.optim.AdamW",
    "scheduler": "none",
    "gradient_accumulation_steps": 1,
    "mixed_precision": False,
}

DEFAULT_MINIMUM_F1_GAIN = 0.01
DEFAULT_MODEL_DIR = (
    REPOSITORY_ROOT / "temp" / "models" / "cross-encoder-nli-deberta-v3-xsmall-a150876"
)
DEFAULT_WORK_DIR = REPOSITORY_ROOT / "temp" / "mrpc_pi2nli_adaptation_v1"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_pi2nli_adaptation_2026-08-04"
DEFAULT_BASELINE_REVEAL = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_semantic_relation_model_search_2026-08-03"
    / "held_out_reveal.json"
)
DEFAULT_FROZEN_COMPACT_DIR = (
    REPOSITORY_ROOT / "reports" / "mrpc_semantic_relation_model_search_2026-08-03"
)
DEFAULT_FROZEN_COMPACT_ARTIFACT = DEFAULT_FROZEN_COMPACT_DIR / "fitted_resolver_artifact.npz"
DEFAULT_FROZEN_COMPACT_CONTRACT = DEFAULT_FROZEN_COMPACT_DIR / "calibration_resolver_contract.json"
DEFAULT_PREREGISTRATION = (
    REPOSITORY_ROOT / "docs" / "audits" / "PARAPHRASE_MRPC_PI2NLI_PREREGISTRATION_2026-08-04.md"
)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one canonical JSON object and refuse replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical_json(dict(value)))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A reveal marker intentionally remains if its write started.  For other
        # artifacts, a partial exclusive file is safer than silent replacement.
        raise


def _refuse_existing_outputs(paths: Iterable[Path]) -> None:
    existing = sorted(str(path) for path in paths if path.exists())
    if existing:
        raise ValueError(
            "experiment output already exists; refusing replacement: " + ", ".join(existing)
        )


def _adapted_model_contract() -> dict[str, Any]:
    return {
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "membership_effect": "none",
        "directional_relation_authorized": False,
        "atomic_support_authorized": False,
        "directional_edge_authorized": False,
        "edge_write_authorized": False,
        "service_consumers": [],
        "database_consumers": [],
    }


def _base_model_identity(model_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    actual: dict[str, str] = {}
    for relative_path, expected_sha256 in BASE_MODEL_FILE_SHA256.items():
        path = model_dir / relative_path
        if not path.is_file():
            raise ValueError(f"missing pinned base-model file: {path}")
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"pinned base-model file SHA-256 mismatch for {relative_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        actual[relative_path] = actual_sha256
    model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weights_relative_path": MODEL_WEIGHTS_RELATIVE_PATH,
        "weights_sha256": MODEL_WEIGHTS_SHA256,
    }
    return model, actual


def _pair_value(pair: Any, name: str) -> Any:
    if isinstance(pair, Mapping):
        return pair[name]
    return getattr(pair, name)


def _validate_component_folds(
    pairs: Sequence[Any],
    folds: Sequence[Mapping[str, Any] | tuple[np.ndarray, np.ndarray]],
) -> None:
    pair_ids = [int(_pair_value(pair, "pair_id")) for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)) or not pair_ids:
        raise ValueError("training pairs must have unique non-empty pair IDs")
    if all(
        (isinstance(pair, Mapping) and "group" in pair) or hasattr(pair, "group") for pair in pairs
    ):
        group_by_pair = {
            int(_pair_value(pair, "pair_id")): int(_pair_value(pair, "group")) for pair in pairs
        }
    else:
        groups, _component_count = model_experiment._component_groups(pairs)
        group_by_pair = {pair_id: int(groups[index]) for index, pair_id in enumerate(pair_ids)}
    expected = set(pair_ids)
    validation_coverage: list[int] = []
    seen_fold_ids: set[int] = set()
    for fold_position, fold in enumerate(folds):
        if isinstance(fold, Mapping):
            fold_id = int(fold["fold"])
            training = [int(value) for value in fold["training_pair_ids"]]
            validation = [int(value) for value in fold["validation_pair_ids"]]
        else:
            training_indexes, validation_indexes = fold
            fold_id = fold_position
            training = [pair_ids[int(index)] for index in training_indexes]
            validation = [pair_ids[int(index)] for index in validation_indexes]
        if fold_id in seen_fold_ids:
            raise ValueError("component fold IDs must be unique")
        seen_fold_ids.add(fold_id)
        training_set = set(training)
        validation_set = set(validation)
        if (
            not validation
            or len(training) != len(training_set)
            or len(validation) != len(validation_set)
            or training_set & validation_set
            or training_set | validation_set != expected
        ):
            raise ValueError("component fold pair partition is invalid")
        training_groups = {group_by_pair[pair_id] for pair_id in training_set}
        validation_groups = {group_by_pair[pair_id] for pair_id in validation_set}
        if training_groups & validation_groups:
            raise ValueError("component fold leaks a sentence component")
        validation_coverage.extend(validation)
    if sorted(validation_coverage) != sorted(pair_ids):
        raise ValueError("component validation folds must cover every pair exactly once")


def _training_manifest(
    *,
    pairs: Sequence[Any],
    labels: Mapping[int, bool],
    groups: Sequence[int] | np.ndarray,
    routed: Sequence[bool] | np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    model: Mapping[str, str],
    base_model_files: Mapping[str, str],
    dataset_identity: Mapping[str, Any],
    route_evidence: Mapping[str, Any],
    preregistration_path: Path,
) -> dict[str, Any]:
    if len(pairs) != len(groups) or len(pairs) != len(routed):
        raise ValueError("training pair metadata must be pair-aligned")
    pair_ids = [int(_pair_value(pair, "pair_id")) for pair in pairs]
    if set(labels) != set(pair_ids):
        raise ValueError("calibration labels must align exactly with training pairs")
    pair_rows = [
        {
            "pair_id": pair_id,
            "sentence1": str(_pair_value(pair, "sentence1")),
            "sentence2": str(_pair_value(pair, "sentence2")),
            "label": bool(labels[pair_id]),
            "group": int(groups[index]),
            "routed": bool(routed[index]),
        }
        for index, (pair, pair_id) in enumerate(zip(pairs, pair_ids, strict=True))
    ]
    fold_rows = [
        {
            "fold": fold_id,
            "training_pair_ids": [pair_ids[int(index)] for index in training_indexes],
            "validation_pair_ids": [pair_ids[int(index)] for index in validation_indexes],
        }
        for fold_id, (training_indexes, validation_indexes) in enumerate(folds)
    ]
    _validate_component_folds(pair_rows, fold_rows)
    payload = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "calibration_training_manifest_before_adapted_model_training",
        "model": dict(model),
        "base_model_files": dict(base_model_files),
        "training": dict(TRAINING_PROTOCOL),
        "authority": _adapted_model_contract(),
        "split": {
            "seed": BLIND_SPLIT_SEED,
            "unit": "connected_component_of_normalized_sentence_identity",
            "calibration_pair_count": len(pair_rows),
            "component_fold_count": len(fold_rows),
            "fold_seed": model_experiment.FOLD_SEED,
            "held_out_labels_loaded": False,
            "held_out_pairs_included": False,
            "held_out_text_included": False,
        },
        "dataset_identity": dict(dataset_identity),
        "routing": {
            "top_k": int(route_evidence["top_k"]),
            "routed_pair_count": int(np.count_nonzero(routed)),
            "route_vector_sha256": _canonical_sha256(np.asarray(routed, dtype=bool).tolist()),
            "evidence": dict(route_evidence),
        },
        "preregistration": {
            "path": str(preregistration_path.relative_to(REPOSITORY_ROOT)),
            "sha256": _file_sha256(preregistration_path),
        },
        "pairs": pair_rows,
        "folds": fold_rows,
    }
    return {**payload, "manifest_sha256": _canonical_sha256(payload)}


def _validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported PI2NLI training manifest schema")
    if manifest.get("manifest_sha256") != _canonical_sha256(payload):
        raise ValueError("PI2NLI training manifest SHA-256 mismatch")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PI2NLI training experiment identity mismatch")
    if manifest.get("training") != TRAINING_PROTOCOL:
        raise ValueError("PI2NLI training protocol changed")
    if manifest.get("authority") != _adapted_model_contract():
        raise ValueError("adapted model authority contract changed")
    split = manifest.get("split") or {}
    if (
        split.get("held_out_labels_loaded") is not False
        or split.get("held_out_pairs_included") is not False
        or split.get("held_out_text_included") is not False
    ):
        raise ValueError("training manifest is not calibration-only")
    pairs = manifest.get("pairs")
    folds = manifest.get("folds")
    if not isinstance(pairs, list) or not isinstance(folds, list):
        raise ValueError("training manifest pairs and folds must be lists")
    expected_pair_keys = {
        "pair_id",
        "sentence1",
        "sentence2",
        "label",
        "group",
        "routed",
    }
    for pair in pairs:
        if not isinstance(pair, Mapping) or set(pair) != expected_pair_keys:
            raise ValueError("training manifest pair schema changed")
        if type(pair["label"]) is not bool or type(pair["routed"]) is not bool:
            raise ValueError("training manifest labels and routes must be booleans")
    _validate_component_folds(pairs, folds)


def _load_common_inputs(args: argparse.Namespace) -> dict[str, Any]:
    pairs, split, embeddings, signatures, graphs, evidence = model_experiment._load_inputs(args)
    return {
        "pairs": pairs,
        "split": split,
        "embeddings": embeddings,
        "signatures": signatures,
        "graphs": graphs,
        "evidence": evidence,
    }


def _partition_inputs(
    common: Mapping[str, Any],
    *,
    partition: str,
    top_k: int,
) -> dict[str, Any]:
    pairs = [pair for pair in common["pairs"] if common["split"][pair.pair_id] == partition]
    neighborhoods, route_evidence = model_experiment._partition_neighborhoods(
        pairs,
        common["signatures"],
        top_k=top_k,
    )
    routed = model_experiment._symmetric_route(pairs, neighborhoods)
    compact = model_experiment._feature_matrices(
        pairs,
        embeddings=common["embeddings"],
        signatures=common["signatures"],
        graphs=common["graphs"],
    )[COMPACT_GENERIC_FEATURE_SCHEMA_VERSION]
    return {
        "pairs": pairs,
        "routed": routed,
        "route_evidence": route_evidence,
        "compact_features": compact,
    }


def _fold_indexes_from_manifest(
    pairs: Sequence[Any],
    fold_rows: Sequence[Mapping[str, Any]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    index_by_pair = {int(_pair_value(pair, "pair_id")): index for index, pair in enumerate(pairs)}
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in sorted(fold_rows, key=lambda value: int(value["fold"])):
        result.append(
            (
                np.asarray(
                    [index_by_pair[int(value)] for value in fold["training_pair_ids"]],
                    dtype=np.int64,
                ),
                np.asarray(
                    [index_by_pair[int(value)] for value in fold["validation_pair_ids"]],
                    dtype=np.int64,
                ),
            )
        )
    return result


def _epoch_outer_cv(
    epoch_feature_matrices: Mapping[int, np.ndarray],
    compact_features: np.ndarray,
    labels: np.ndarray,
    routed: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[EpochResult, ...]:
    compact_features = np.asarray(compact_features, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    routed = np.asarray(routed, dtype=bool)
    pair_count = len(labels)
    if (
        compact_features.shape != (pair_count, COMPACT_GENERIC_FEATURE_DIMENSION)
        or routed.shape != labels.shape
    ):
        raise ValueError("outer-CV compact features, labels, and route are not aligned")
    results: list[EpochResult] = []
    for epoch in sorted(epoch_feature_matrices):
        per_fold = np.asarray(epoch_feature_matrices[epoch], dtype=np.float64)
        expected_shape = (
            len(folds),
            pair_count,
            DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION,
        )
        if per_fold.shape != expected_shape or not np.all(np.isfinite(per_fold)):
            raise ValueError("adapted epoch features have the wrong shape or values")
        scores = np.full(pair_count, -math.inf, dtype=np.float64)
        for fold_id, (training_indexes, validation_indexes) in enumerate(folds):
            training_routed = training_indexes[routed[training_indexes]]
            validation_routed = validation_indexes[routed[validation_indexes]]
            if not len(training_routed) or not len(validation_routed):
                raise ValueError("outer component fold has no routed training or validation pair")
            training_features = np.column_stack(
                (compact_features[training_routed], per_fold[fold_id, training_routed])
            )
            validation_features = np.column_stack(
                (compact_features[validation_routed], per_fold[fold_id, validation_routed])
            )
            estimator = model_experiment._make_estimator(_fixed_spec(FUSION_FEATURE_SCHEMA_VERSION))
            estimator.fit(training_features, labels[training_routed])
            scores[validation_routed] = model_experiment._estimator_scores(
                estimator,
                validation_features,
            )
        threshold, metrics = model_experiment._best_threshold(labels, scores, routed)
        results.append(
            EpochResult(
                epoch=int(epoch),
                oof_scores=scores,
                threshold=float(threshold),
                metrics=dict(metrics),
            )
        )
    if {result.epoch for result in results} != set(TRAINING_EPOCHS):
        raise ValueError("crossfit cache does not cover the preregistered epochs")
    return tuple(results)


def _select_epoch(results: Sequence[EpochResult]) -> EpochResult:
    if not results:
        raise ValueError("cannot select an epoch from no calibration results")
    return max(
        results,
        key=lambda result: (
            float(result.metrics["f1"]),
            float(result.metrics["precision"]),
            float(result.metrics["recall"]),
            -int(result.epoch),
        ),
    )


def _feature_rows_to_matrix(
    rows: Sequence[Mapping[str, Any]],
    pair_ids: Sequence[int],
) -> np.ndarray:
    by_pair: dict[int, tuple[float, float]] = {}
    for row in rows:
        pair_id = int(row["pair_id"])
        forward = float(row["forward_log_entailment"])
        reverse = float(row["reverse_log_entailment"])
        if pair_id in by_pair or not math.isfinite(forward) or not math.isfinite(reverse):
            raise ValueError("adapted feature row is duplicate or non-finite")
        by_pair[pair_id] = (min(forward, reverse), max(forward, reverse))
    if set(by_pair) != set(pair_ids):
        raise ValueError("adapted feature rows do not align with the pair inventory")
    result = np.asarray([by_pair[pair_id] for pair_id in pair_ids], dtype=np.float64)
    if result.shape != (len(pair_ids), DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION):
        raise ValueError("adapted feature matrix shape changed")
    return result


def _validate_cache_alignment(
    cache: Mapping[str, Any],
    *,
    expected_schema_version: str,
    manifest_sha256: str,
    pair_ids: Sequence[int],
) -> Mapping[str, Any]:
    payload = cache.get("payload")
    if cache.get("schema_version") != expected_schema_version or not isinstance(payload, Mapping):
        raise ValueError("unsupported PI2NLI cache schema")
    expected_hash = _canonical_sha256(
        {"schema_version": cache["schema_version"], "payload": payload}
    )
    if cache.get("cache_sha256") != expected_hash:
        raise ValueError("PI2NLI cache SHA-256 mismatch")
    manifest_identity = payload.get("manifest") or {}
    if manifest_identity.get("manifest_sha256") != manifest_sha256:
        raise ValueError("PI2NLI cache is not bound to the expected manifest")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PI2NLI cache experiment identity mismatch")
    if payload.get("authority") != _adapted_model_contract():
        raise ValueError("PI2NLI cache authority contract changed")
    cache_pair_ids = payload.get("pair_ids")
    if cache_pair_ids is not None and [int(value) for value in cache_pair_ids] != list(pair_ids):
        raise ValueError("PI2NLI cache pair order changed")
    return payload


def _crossfit_feature_matrices(
    payload: Mapping[str, Any],
    *,
    pair_ids: Sequence[int],
    fold_count: int,
) -> dict[int, np.ndarray]:
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != fold_count:
        raise ValueError("crossfit cache fold count changed")
    by_epoch = {
        epoch: np.empty(
            (
                fold_count,
                len(pair_ids),
                DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION,
            ),
            dtype=np.float64,
        )
        for epoch in TRAINING_EPOCHS
    }
    for expected_fold, fold in enumerate(sorted(folds, key=lambda row: int(row["fold"]))):
        if int(fold["fold"]) != expected_fold:
            raise ValueError("crossfit cache fold IDs changed")
        epochs = fold.get("epochs")
        if not isinstance(epochs, list):
            raise ValueError("crossfit cache fold epochs must be a list")
        seen: set[int] = set()
        for epoch_row in epochs:
            epoch = int(epoch_row["epoch"])
            if epoch not in by_epoch or epoch in seen:
                raise ValueError("crossfit cache epoch inventory changed")
            seen.add(epoch)
            by_epoch[epoch][expected_fold] = _feature_rows_to_matrix(
                epoch_row["features"],
                pair_ids,
            )
        if seen != set(TRAINING_EPOCHS):
            raise ValueError("crossfit cache fold is missing an epoch")
    return by_epoch


def _epoch_evidence(result: EpochResult) -> dict[str, Any]:
    return {
        "epoch": result.epoch,
        "threshold": float(result.threshold),
        "oof_metrics": dict(result.metrics),
        "oof_score_sha256": _array_sha256(result.oof_scores),
    }


def _source_identity() -> dict[str, str]:
    paths = {
        "benchmark": Path(__file__).resolve(),
        "runtime_builder": REPOSITORY_ROOT
        / "scripts"
        / "research"
        / "mrpc"
        / "build_mrpc_pi2nli_adaptation_cache.py",
        "model_experiment": REPOSITORY_ROOT
        / "scripts"
        / "research"
        / "mrpc"
        / "experiment_mrpc_semantic_relation_models.py",
        "fusion_benchmark": REPOSITORY_ROOT
        / "scripts"
        / "research"
        / "mrpc"
        / "run_mrpc_directional_nli_fusion_benchmark.py",
        "directional_features": APP_ROOT
        / "mcp_servers"
        / "mcp_lib"
        / "directional_nli_features.py",
        "compact_features": APP_ROOT / "mcp_servers" / "mcp_lib" / "semantic_relation_features.py",
        "preregistration": DEFAULT_PREREGISTRATION,
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _prepare(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    manifest_path = work_dir / "calibration_training_manifest.json"
    _refuse_existing_outputs([manifest_path])
    common = _load_common_inputs(args)
    calibration = _partition_inputs(common, partition="calibration", top_k=args.psa_top_k)
    pairs = calibration["pairs"]
    pair_ids = frozenset(pair.pair_id for pair in pairs)
    labels = model_experiment._load_selected_labels(args.mrpc_dir.resolve(), pair_ids)
    groups, _component_count = model_experiment._component_groups(pairs)
    folds = model_experiment._group_folds(groups)
    model, base_model_files = _base_model_identity(args.model_dir.resolve())
    manifest = _training_manifest(
        pairs=pairs,
        labels=labels,
        groups=groups,
        routed=calibration["routed"],
        folds=folds,
        model=model,
        base_model_files=base_model_files,
        dataset_identity=_dataset_identity(args.mrpc_dir.resolve()),
        route_evidence=calibration["route_evidence"],
        preregistration_path=args.preregistration.resolve(),
    )
    _validate_training_manifest(manifest)
    _exclusive_json(manifest_path, manifest)
    print(
        _canonical_json(
            {
                "status": "calibration_training_manifest_frozen",
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "calibration_pair_count": len(pairs),
                "held_out_labels_loaded": False,
                "next": (
                    "run build_mrpc_pi2nli_adaptation_cache.py --phase crossfit "
                    "in the pinned torch runtime"
                ),
            }
        )
    )
    return 0


def _select(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    manifest_path = work_dir / "calibration_training_manifest.json"
    cache_path = work_dir / "crossfit_cache.json"
    selection_path = work_dir / "frozen_selection.json"
    _refuse_existing_outputs([selection_path])
    manifest = _load_json(manifest_path)
    _validate_training_manifest(manifest)
    common = _load_common_inputs(args)
    calibration = _partition_inputs(common, partition="calibration", top_k=args.psa_top_k)
    pairs = calibration["pairs"]
    pair_ids = [pair.pair_id for pair in pairs]
    labels_by_pair = model_experiment._load_selected_labels(
        args.mrpc_dir.resolve(), frozenset(pair_ids)
    )
    labels = np.asarray([labels_by_pair[pair_id] for pair_id in pair_ids], dtype=bool)
    routed = np.asarray(calibration["routed"], dtype=bool)
    fold_indexes = _fold_indexes_from_manifest(pairs, manifest["folds"])
    cache = _load_json(cache_path)
    payload = _validate_cache_alignment(
        cache,
        expected_schema_version=CROSSFIT_CACHE_SCHEMA_VERSION,
        manifest_sha256=str(manifest["manifest_sha256"]),
        pair_ids=pair_ids,
    )
    epoch_matrices = _crossfit_feature_matrices(
        payload,
        pair_ids=pair_ids,
        fold_count=len(fold_indexes),
    )
    results = _epoch_outer_cv(
        epoch_matrices,
        calibration["compact_features"],
        labels,
        routed,
        fold_indexes,
    )
    selected = _select_epoch(results)
    base_result = model_experiment._crossfit(
        _fixed_spec(COMPACT_GENERIC_FEATURE_SCHEMA_VERSION),
        calibration["compact_features"],
        labels,
        routed,
        fold_indexes,
    )
    if dict(base_result.metrics) != EXPECTED_FROZEN_WINNER["oof_metrics"] or float(
        base_result.threshold
    ) != float(EXPECTED_FROZEN_WINNER["threshold"]):
        raise ValueError("341-feature calibration control no longer replays")
    selection_payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "calibration_selection_frozen_before_final_fit_or_held_inference",
        "training_manifest_sha256": manifest["manifest_sha256"],
        "crossfit_cache_sha256": cache["cache_sha256"],
        "selected_epoch": selected.epoch,
        "selected_threshold": float(selected.threshold),
        "selected_oof_metrics": dict(selected.metrics),
        "selected_oof_score_sha256": _array_sha256(selected.oof_scores),
        "epoch_selection_rule": ("max(oof_f1,oof_precision,oof_recall,negative_epoch)"),
        "all_epoch_results": [_epoch_evidence(result) for result in results],
        "calibration_control": {
            "compact_341": {
                "threshold": float(base_result.threshold),
                "oof_metrics": dict(base_result.metrics),
                "oof_score_sha256": _array_sha256(base_result.oof_scores),
            },
            "oof_f1_gain": float(selected.metrics["f1"] - base_result.metrics["f1"]),
        },
        "feature_contract": {
            "compact_schema_version": COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
            "compact_dimension": COMPACT_GENERIC_FEATURE_DIMENSION,
            "adapted_directional_schema_version": (DIRECTIONAL_NLI_LOG_ENTAILMENT_SCHEMA_VERSION),
            "adapted_directional_feature_names": list(DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_NAMES),
            "adapted_directional_dimension": (DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION),
            "fusion_schema_version": FUSION_FEATURE_SCHEMA_VERSION,
            "fusion_dimension": FUSION_FEATURE_DIMENSION,
            "atomic_features_control_score": False,
            "phrase_or_lemma_exception_features": False,
        },
        "fixed_estimator": {
            "family": "ridge",
            "alpha": FIXED_RIDGE_ALPHA,
            "class_weight": FIXED_CLASS_WEIGHT,
            "solver": "cholesky",
            "standardization": "StandardScaler",
        },
        "authority": _adapted_model_contract(),
        "frozen_comparator": {
            "report_file_sha256": EXPECTED_FROZEN_REVEAL_FILE_SHA256,
            "metrics": EXPECTED_FROZEN_HELD_OUT_METRICS,
        },
        "minimum_required_f1_gain": args.minimum_f1_gain,
        "bootstrap_policy": {
            "sample_count": PAIRED_BOOTSTRAP_SAMPLES,
            "seed": PAIRED_BOOTSTRAP_SEED,
        },
        "split": manifest["split"],
        "routing": manifest["routing"],
        "runtime": payload.get("runtime"),
        "source_identity": _source_identity(),
        "caveats": [
            "MRPC held labels were visible during earlier repository development.",
            "Epoch and threshold selection use calibration data only.",
            "PI2NLI negative direction and class assignments are not semantic facts.",
            "The adapted clone cannot supply directional or edge evidence.",
        ],
    }
    selection = {
        **selection_payload,
        "selection_sha256": _canonical_sha256(selection_payload),
    }
    _exclusive_json(selection_path, selection)
    print(
        _canonical_json(
            {
                "status": "calibration_selection_frozen",
                "selection_path": str(selection_path),
                "selection_sha256": selection["selection_sha256"],
                "selected_epoch": selected.epoch,
                "oof_metrics": selected.metrics,
                "oof_f1_gain": selection["calibration_control"]["oof_f1_gain"],
                "held_out_labels_loaded": False,
                "next": ("run build_mrpc_pi2nli_adaptation_cache.py --phase fit-final"),
            }
        )
    )
    return 0


def _validate_selection(selection: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported PI2NLI selection schema")
    if selection.get("selection_sha256") != _canonical_sha256(payload):
        raise ValueError("PI2NLI selection SHA-256 mismatch")
    if selection.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PI2NLI selection experiment identity mismatch")
    if selection.get("authority") != _adapted_model_contract():
        raise ValueError("PI2NLI selection authority contract changed")
    if int(selection.get("selected_epoch", 0)) not in TRAINING_EPOCHS:
        raise ValueError("PI2NLI selected epoch is invalid")


def _held_manifest(
    *,
    pairs: Sequence[Any],
    groups: Sequence[int] | np.ndarray,
    routed: Sequence[bool] | np.ndarray,
    model: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    resolver_sha256: str,
    route_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if len(pairs) != len(groups) or len(pairs) != len(routed):
        raise ValueError("held manifest metadata must be pair-aligned")
    pair_rows = [
        {
            "pair_id": int(_pair_value(pair, "pair_id")),
            "sentence1": str(_pair_value(pair, "sentence1")),
            "sentence2": str(_pair_value(pair, "sentence2")),
            "group": int(groups[index]),
            "routed": bool(routed[index]),
        }
        for index, pair in enumerate(pairs)
    ]
    payload = {
        "schema_version": HELD_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "label_free_held_requests_after_final_resolver_freeze",
        "model": dict(model),
        "checkpoint": dict(checkpoint),
        "resolver_sha256": resolver_sha256,
        "authority": _adapted_model_contract(),
        "routing": {
            "top_k": int(route_evidence["top_k"]),
            "routed_pair_count": int(np.count_nonzero(routed)),
            "route_vector_sha256": _canonical_sha256(np.asarray(routed, dtype=bool).tolist()),
            "evidence": dict(route_evidence),
        },
        "labels_included": False,
        "pairs": pair_rows,
    }
    return {**payload, "manifest_sha256": _canonical_sha256(payload)}


def _freeze(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    manifest_path = work_dir / "calibration_training_manifest.json"
    selection_path = work_dir / "frozen_selection.json"
    final_cache_path = work_dir / "final_fit_cache.json"
    resolver_path = work_dir / "frozen_resolver.json"
    held_manifest_path = work_dir / "held_request_manifest.json"
    _refuse_existing_outputs([resolver_path, held_manifest_path])
    manifest = _load_json(manifest_path)
    _validate_training_manifest(manifest)
    selection = _load_json(selection_path)
    _validate_selection(selection)
    if selection["training_manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("selection is not bound to the training manifest")
    common = _load_common_inputs(args)
    calibration = _partition_inputs(common, partition="calibration", top_k=args.psa_top_k)
    calibration_pairs = calibration["pairs"]
    calibration_pair_ids = [pair.pair_id for pair in calibration_pairs]
    labels_by_pair = model_experiment._load_selected_labels(
        args.mrpc_dir.resolve(), frozenset(calibration_pair_ids)
    )
    labels = np.asarray([labels_by_pair[pair_id] for pair_id in calibration_pair_ids], dtype=bool)
    routed = np.asarray(calibration["routed"], dtype=bool)
    final_cache = _load_json(final_cache_path)
    final_payload = _validate_cache_alignment(
        final_cache,
        expected_schema_version=FINAL_FIT_CACHE_SCHEMA_VERSION,
        manifest_sha256=str(manifest["manifest_sha256"]),
        pair_ids=calibration_pair_ids,
    )
    if int(final_payload.get("selected_epoch", 0)) != int(selection["selected_epoch"]):
        raise ValueError("final checkpoint epoch differs from frozen selection")
    adapted_features = _feature_rows_to_matrix(
        final_payload["features"],
        calibration_pair_ids,
    )
    fusion_features = np.column_stack((calibration["compact_features"], adapted_features))
    numeric_model = _fit_numeric_model(
        fusion_features,
        labels,
        routed,
        binary_threshold=float(selection["selected_threshold"]),
    )
    _validate_numeric_model(numeric_model)
    checkpoint = final_payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("artifact_sha256"):
        raise ValueError("final fit cache has no authenticated checkpoint")
    held = _partition_inputs(common, partition="held_out", top_k=args.psa_top_k)
    held_groups, _held_component_count = model_experiment._component_groups(held["pairs"])
    resolver_payload = {
        "schema_version": RESOLVER_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "phase": "final_resolver_frozen_before_adapted_held_inference",
        "training_manifest_sha256": manifest["manifest_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "final_fit_cache_sha256": final_cache["cache_sha256"],
        "selected_epoch": selection["selected_epoch"],
        "model": manifest["model"],
        "checkpoint": dict(checkpoint),
        "numeric_model": numeric_model,
        "authority": _adapted_model_contract(),
        "calibration_evidence": {
            "pair_ids_sha256": _canonical_sha256(calibration_pair_ids),
            "labels_sha256": _canonical_sha256(labels.tolist()),
            "route_sha256": _canonical_sha256(routed.tolist()),
            "compact_feature_sha256": _array_sha256(calibration["compact_features"]),
            "adapted_feature_sha256": _array_sha256(adapted_features),
            "fusion_feature_sha256": _array_sha256(fusion_features),
        },
        "held_out_labels_loaded": False,
    }
    resolver = {
        **resolver_payload,
        "resolver_sha256": _canonical_sha256(resolver_payload),
    }
    held_manifest = _held_manifest(
        pairs=held["pairs"],
        groups=held_groups,
        routed=held["routed"],
        model=manifest["model"],
        checkpoint={
            "artifact_sha256": checkpoint["artifact_sha256"],
            "selected_epoch": int(selection["selected_epoch"]),
        },
        resolver_sha256=resolver["resolver_sha256"],
        route_evidence=held["route_evidence"],
    )
    _exclusive_json(resolver_path, resolver)
    _exclusive_json(held_manifest_path, held_manifest)
    print(
        _canonical_json(
            {
                "status": "final_resolver_frozen",
                "resolver_path": str(resolver_path),
                "resolver_sha256": resolver["resolver_sha256"],
                "held_manifest_path": str(held_manifest_path),
                "held_manifest_sha256": held_manifest["manifest_sha256"],
                "held_out_labels_loaded": False,
                "next": ("run build_mrpc_pi2nli_adaptation_cache.py --phase infer-held"),
            }
        )
    )
    return 0


def _validate_resolver(resolver: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in resolver.items() if key != "resolver_sha256"}
    if resolver.get("schema_version") != RESOLVER_SCHEMA_VERSION:
        raise ValueError("unsupported frozen PI2NLI resolver schema")
    if resolver.get("resolver_sha256") != _canonical_sha256(payload):
        raise ValueError("frozen PI2NLI resolver SHA-256 mismatch")
    if resolver.get("authority") != _adapted_model_contract():
        raise ValueError("frozen PI2NLI resolver authority changed")
    _validate_numeric_model(resolver["numeric_model"])


def _validate_held_manifest(manifest: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != HELD_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported held request manifest schema")
    if manifest.get("manifest_sha256") != _canonical_sha256(payload):
        raise ValueError("held request manifest SHA-256 mismatch")
    if manifest.get("labels_included") is not False:
        raise ValueError("held request manifest contains labels")
    if manifest.get("authority") != _adapted_model_contract():
        raise ValueError("held request manifest authority changed")
    pairs = manifest.get("pairs")
    expected_keys = {"pair_id", "sentence1", "sentence2", "group", "routed"}
    if not isinstance(pairs, list) or any(
        not isinstance(pair, Mapping) or set(pair) != expected_keys for pair in pairs
    ):
        raise ValueError("held request pair schema changed")


def _reveal(args: argparse.Namespace) -> int:
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = work_dir / "calibration_training_manifest.json"
    selection_path = work_dir / "frozen_selection.json"
    resolver_path = work_dir / "frozen_resolver.json"
    held_manifest_path = work_dir / "held_request_manifest.json"
    held_cache_path = work_dir / "held_inference_cache.json"
    reveal_marker_path = work_dir / "reveal_started.json"
    report_path = output_dir / "pi2nli_adaptation_report.json"
    _refuse_existing_outputs([reveal_marker_path, report_path])

    training_manifest = _load_json(manifest_path)
    _validate_training_manifest(training_manifest)
    selection = _load_json(selection_path)
    _validate_selection(selection)
    resolver = _load_json(resolver_path)
    _validate_resolver(resolver)
    held_manifest = _load_json(held_manifest_path)
    _validate_held_manifest(held_manifest)
    if (
        resolver["training_manifest_sha256"] != training_manifest["manifest_sha256"]
        or resolver["selection_sha256"] != selection["selection_sha256"]
        or held_manifest["resolver_sha256"] != resolver["resolver_sha256"]
        or held_manifest["checkpoint"]["artifact_sha256"]
        != resolver["checkpoint"]["artifact_sha256"]
    ):
        raise ValueError("PI2NLI frozen artifact chain is inconsistent")

    common = _load_common_inputs(args)
    held = _partition_inputs(common, partition="held_out", top_k=args.psa_top_k)
    pairs = held["pairs"]
    pair_ids = [pair.pair_id for pair in pairs]
    manifest_pairs = held_manifest["pairs"]
    if [int(row["pair_id"]) for row in manifest_pairs] != pair_ids:
        raise ValueError("held request manifest pair order changed")
    if any(
        row["sentence1"] != pair.sentence1
        or row["sentence2"] != pair.sentence2
        or bool(row["routed"]) != bool(held["routed"][index])
        for index, (row, pair) in enumerate(zip(manifest_pairs, pairs, strict=True))
    ):
        raise ValueError("held request content or route changed")
    held_cache = _load_json(held_cache_path)
    held_payload = _validate_cache_alignment(
        held_cache,
        expected_schema_version=HELD_CACHE_SCHEMA_VERSION,
        manifest_sha256=str(held_manifest["manifest_sha256"]),
        pair_ids=pair_ids,
    )
    if held_payload.get("checkpoint") != resolver["checkpoint"]:
        raise ValueError("held inference cache checkpoint differs from frozen resolver")
    adapted_features = _feature_rows_to_matrix(held_payload["features"], pair_ids)
    fusion_features = np.column_stack((held["compact_features"], adapted_features))
    scores = _score_numeric_model(resolver["numeric_model"], fusion_features)
    routed = np.asarray(held["routed"], dtype=bool)

    baseline_path = args.baseline_reveal.resolve()
    if _file_sha256(baseline_path) != EXPECTED_FROZEN_REVEAL_FILE_SHA256:
        raise ValueError("frozen compact comparator file identity changed")
    marker = {
        "schema_version": REVEAL_MARKER_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "resolver_sha256": resolver["resolver_sha256"],
        "held_cache_sha256": held_cache["cache_sha256"],
        "meaning": "held labels may have been loaded; this experiment cannot be retried",
    }
    _exclusive_json(reveal_marker_path, marker)

    held_ids = frozenset(pair_ids)
    label_map = model_experiment._load_selected_labels(args.mrpc_dir.resolve(), held_ids)
    labels = np.asarray([label_map[pair_id] for pair_id in pair_ids], dtype=bool)
    baseline_labels, baseline_predictions, baseline_routes, baseline = _validated_baseline(
        baseline_path,
        pairs,
    )
    if not np.array_equal(labels, baseline_labels):
        raise ValueError("held labels do not align with frozen comparator")
    _require_route_parity(routed, baseline_routes)
    if {
        "report_file_sha256": baseline["report_file_sha256"],
        "metrics": baseline["metrics"],
    } != selection["frozen_comparator"]:
        raise ValueError("frozen compact comparator identity changed after selection")

    frozen_compact = load_compact_generic_linear_artifact(
        artifact_path=args.frozen_compact_artifact.resolve(),
        contract_path=args.frozen_compact_contract.resolve(),
        expected_artifact_sha256=EXPECTED_FROZEN_ARTIFACT_SHA256,
        expected_contract_file_sha256=EXPECTED_FROZEN_CONTRACT_FILE_SHA256,
    )
    trace_margins = _authenticated_baseline_margins(baseline_path, pairs)
    baseline_margins, compact_parity = _frozen_compact_parity(
        artifact=frozen_compact,
        compact_features=held["compact_features"],
        current_routes=routed,
        trace_routes=baseline_routes,
        trace_margins=trace_margins,
        trace_predictions=baseline_predictions,
    )
    threshold = float(resolver["numeric_model"]["binary_threshold"])
    predictions = routed & (scores >= threshold)
    metrics = _binary_metrics(labels, predictions)
    f1_gain = float(metrics["f1"] - baseline["metrics"]["f1"])
    bootstrap = _paired_f1_bootstrap(
        labels,
        predictions,
        baseline_predictions,
        sample_count=int(selection["bootstrap_policy"]["sample_count"]),
        seed=int(selection["bootstrap_policy"]["seed"]),
    )
    minimum_gain = float(selection["minimum_required_f1_gain"])
    passed_material = f1_gain >= minimum_gain
    passed_confidence = bool(bootstrap["lower_bound_strictly_positive"])
    traces = [
        {
            "pair_id": pair.pair_id,
            "label": bool(labels[index]),
            "routed": bool(routed[index]),
            "resolver_score": float(scores[index]),
            "resolver_prediction": bool(predictions[index]),
            "baseline_score": float(baseline_margins[index]),
            "baseline_prediction": bool(baseline_predictions[index]),
            "adapted_log_entailment_features": adapted_features[index].tolist(),
            "membership_effect": "none",
            "directional_relation_authorized": False,
            "edge_write_authorized": False,
        }
        for index, pair in enumerate(pairs)
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "evaluation_class": "post_hoc_mrpc_development_diagnostic",
        "authority": _adapted_model_contract(),
        "selection": selection,
        "resolver": {
            "resolver_sha256": resolver["resolver_sha256"],
            "checkpoint": resolver["checkpoint"],
            "numeric_model_sha256": resolver["numeric_model"]["numeric_model_sha256"],
        },
        "held_out": {
            "pair_count": len(pairs),
            "routed_pair_count": int(np.count_nonzero(routed)),
            "metrics": metrics,
            "baseline": baseline,
            "f1_gain_over_frozen_compact_resolver": f1_gain,
            "paired_bootstrap_f1_difference": bootstrap,
            "minimum_required_f1_gain": minimum_gain,
            "passed_material_gain_gate": passed_material,
            "passed_bootstrap_confidence_gate": passed_confidence,
            "passed_promotion_gate": passed_material and passed_confidence,
        },
        "evidence": {
            "training_manifest_sha256": training_manifest["manifest_sha256"],
            "held_manifest_sha256": held_manifest["manifest_sha256"],
            "held_cache_sha256": held_cache["cache_sha256"],
            "held_route": held["route_evidence"],
            "held_compact_feature_matrix_sha256": _array_sha256(held["compact_features"]),
            "held_adapted_feature_matrix_sha256": _array_sha256(adapted_features),
            "held_fusion_feature_matrix_sha256": _array_sha256(fusion_features),
            "held_score_sha256": _array_sha256(scores),
            "held_prediction_sha256": _canonical_sha256(predictions.tolist()),
            "frozen_compact_parity": compact_parity,
            "reveal_marker_sha256": _file_sha256(reveal_marker_path),
        },
        "held_out_pair_traces": traces,
        "caveats": [
            "Held MRPC labels were visible during earlier repository development.",
            "This dashboard result is not an external-generalization claim.",
            "PI2NLI negative targets cannot be read as directional semantics.",
            "The adapted clone has no atomic, membership, service, database, or edge authority.",
        ],
    }
    _exclusive_json(report_path, report)
    print(
        _canonical_json(
            {
                "status": "held_out_revealed_once",
                "report_path": str(report_path),
                "report_file_sha256": _file_sha256(report_path),
                "metrics": metrics,
                "f1_gain": f1_gain,
                "paired_bootstrap": bootstrap,
                "passed_promotion_gate": passed_material and passed_confidence,
            }
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "select", "freeze", "reveal"),
        required=True,
    )
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
    parser.add_argument(
        "--pinned-guide-embeddings",
        type=Path,
        default=DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--psa-top-k", type=int, default=PSA_SIGNATURE_CANDIDATE_LIMIT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-reveal", type=Path, default=DEFAULT_BASELINE_REVEAL)
    parser.add_argument(
        "--frozen-compact-artifact",
        type=Path,
        default=DEFAULT_FROZEN_COMPACT_ARTIFACT,
    )
    parser.add_argument(
        "--frozen-compact-contract",
        type=Path,
        default=DEFAULT_FROZEN_COMPACT_CONTRACT,
    )
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--minimum-f1-gain", type=float, default=DEFAULT_MINIMUM_F1_GAIN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.minimum_f1_gain) or args.minimum_f1_gain < 0.0:
        raise ValueError("minimum F1 gain must be finite and nonnegative")
    if args.psa_top_k <= 0:
        raise ValueError("PSA top-k must be positive")
    phases = {
        "prepare": _prepare,
        "select": _select,
        "freeze": _freeze,
        "reveal": _reveal,
    }
    return phases[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
