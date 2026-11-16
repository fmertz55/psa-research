#!/usr/bin/env python3
"""Freeze and reveal one compact-resolver plus directional-NLI fusion.

The existing 341 generic features, PSA route, component folds, ridge family,
ridge parameters, model snapshot, and inference cache remain fixed.  The only
candidate change is two swap-invariant values: the log minimum and log maximum
directional entailment probability.  Atomic support remains authenticated
diagnostic evidence because calibration did not justify making it controlling.

Selection loads labels only for the calibration pair IDs.  Reveal first
recomputes the complete selection and fitted numeric model exactly, then loads
held-out labels and compares against the authenticated frozen compact resolver.
Everything here is diagnostic and has no paraphrase-membership authority.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    DEFAULT_BASELINE_REVEAL,
    DEFAULT_WORK_DIR as DEFAULT_NLI_WORK_DIR,
    EXPECTED_FROZEN_HELD_OUT_METRICS,
    EXPECTED_FROZEN_REVEAL_FILE_SHA256,
    PAIRED_BOOTSTRAP_SAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    _atomic_records,
    _dataset_identity,
    _load_cache,
    _load_json,
    _paired_f1_bootstrap,
    _probabilities,
    _validated_baseline,
)
from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import _atomic_json  # noqa: E402
from scripts.research.mrpc.run_mrpc_proposition_graph_benchmark import (  # noqa: E402
    BLIND_SPLIT_SEED,
    DEFAULT_CACHE,
    DEFAULT_GRAPH_CACHE,
    DEFAULT_SENSE_CACHE,
    PINNED_GRAPH_CACHE_SHA256,
    PINNED_GRAPH_PARSER_IDENTITY,
)
from scripts.research.mrpc.run_mrpc_semantic_relation_resolver_benchmark import (  # noqa: E402
    DEFAULT_FROZEN_ARTIFACT,
    DEFAULT_FROZEN_CONTRACT,
    EXPECTED_FROZEN_ARTIFACT_SHA256,
    EXPECTED_FROZEN_CONTRACT_FILE_SHA256,
)

from mcp_servers.mcp_lib.directional_nli_features import (  # noqa: E402
    DIRECTIONAL_NLI_FEATURE_DIMENSION,
    DIRECTIONAL_NLI_FEATURE_NAMES,
    DIRECTIONAL_NLI_FEATURE_SCHEMA_VERSION,
    DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION,
    DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_NAMES,
    DIRECTIONAL_NLI_LOG_ENTAILMENT_SCHEMA_VERSION,
    directional_nli_log_entailment_features,
    directional_nli_pair_features,
)
from mcp_servers.mcp_lib.directional_semantic_relation import (  # noqa: E402
    AtomicAssertionInventory,
    project_atomic_assertions,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import _binary_metrics  # noqa: E402
from mcp_servers.mcp_lib.semantic_relation_features import (  # noqa: E402
    COMPACT_GENERIC_FEATURE_DIMENSION,
    COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
    CompactGenericLinearArtifact,
    load_compact_generic_linear_artifact,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    PSA_SIGNATURE_CANDIDATE_LIMIT,
)

SELECTION_SCHEMA_VERSION = "mrpc_directional_nli_fusion_selection_v1"
REPORT_SCHEMA_VERSION = "mrpc_directional_nli_fusion_benchmark_v1"
FUSION_FEATURE_DIMENSION = (
    COMPACT_GENERIC_FEATURE_DIMENSION + DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION
)
FUSION_FEATURE_SCHEMA_VERSION = (
    f"{COMPACT_GENERIC_FEATURE_SCHEMA_VERSION}+{DIRECTIONAL_NLI_LOG_ENTAILMENT_SCHEMA_VERSION}"
)
FIXED_RIDGE_ALPHA = 1_000.0
FIXED_CLASS_WEIGHT: None = None
DEFAULT_MINIMUM_F1_GAIN = 0.01
DEFAULT_WORK_DIR = REPOSITORY_ROOT / "temp" / "mrpc_directional_nli_fusion"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_directional_nli_fusion_2026-08-03"


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": "float64", "shape": list(array.shape)}).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fixed_spec(feature_set: str) -> model_experiment.CandidateSpec:
    return model_experiment.CandidateSpec(
        candidate_id=f"ridge:{feature_set}:fixed_v1",
        family="ridge",
        feature_set=feature_set,
        parameters={"alpha": FIXED_RIDGE_ALPHA, "class_weight": FIXED_CLASS_WEIGHT},
    )


def _candidate_evidence(
    result: model_experiment.CandidateResult,
    *,
    dimension: int,
) -> dict[str, Any]:
    return {
        "candidate_id": result.spec.candidate_id,
        "feature_set": result.spec.feature_set,
        "feature_dimension": dimension,
        "ridge_alpha": FIXED_RIDGE_ALPHA,
        "class_weight": FIXED_CLASS_WEIGHT,
        "threshold": float(result.threshold),
        "oof_metrics": dict(result.metrics),
        "oof_score_sha256": _array_sha256(result.oof_scores),
    }


def _inventory_identity(
    inventories: Mapping[str, AtomicAssertionInventory],
) -> dict[str, Any]:
    rows = sorted(
        (
            {
                "source_text_sha256": inventory.source_text_sha256,
                "inventory_sha256": inventory.inventory_sha256,
            }
            for inventory in inventories.values()
        ),
        key=lambda row: (row["source_text_sha256"], row["inventory_sha256"]),
    )
    return {
        "inventory_count": len(rows),
        "inventory_set_sha256": _canonical_sha256(rows),
    }


def _pair_nli_features(
    pair: Any,
    *,
    inventories: Mapping[str, AtomicAssertionInventory],
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    left_to_right = _probabilities(results, pair.sentence1, pair.sentence2)
    right_to_left = _probabilities(results, pair.sentence2, pair.sentence1)
    right_assertion_support = _atomic_records(
        direction="left_to_right",
        premise=pair.sentence1,
        inventory=inventories[pair.sentence2],
        results=results,
    )
    left_assertion_support = _atomic_records(
        direction="right_to_left",
        premise=pair.sentence2,
        inventory=inventories[pair.sentence1],
        results=results,
    )
    controlling = directional_nli_log_entailment_features(
        left_to_right_probabilities=left_to_right,
        right_to_left_probabilities=right_to_left,
    )
    diagnostic = directional_nli_pair_features(
        left_to_right_probabilities=left_to_right,
        right_to_left_probabilities=right_to_left,
        left_inventory=inventories[pair.sentence1],
        right_inventory=inventories[pair.sentence2],
        left_to_right_atom_records=[record for _assertion, record in right_assertion_support],
        right_to_left_atom_records=[record for _assertion, record in left_assertion_support],
    )
    return controlling, diagnostic


def _nli_feature_matrix(
    pairs: Sequence[Any],
    *,
    routed: np.ndarray,
    inventories: Mapping[str, AtomicAssertionInventory],
    results: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    routed = np.asarray(routed, dtype=bool)
    if routed.shape != (len(pairs),):
        raise ValueError("directional NLI feature route is not pair-aligned")
    controlling = np.zeros(
        (len(pairs), DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION),
        dtype=np.float64,
    )
    diagnostic = np.zeros(
        (len(pairs), DIRECTIONAL_NLI_FEATURE_DIMENSION),
        dtype=np.float64,
    )
    for index, pair in enumerate(pairs):
        if routed[index]:
            controlling[index], diagnostic[index] = _pair_nli_features(
                pair,
                inventories=inventories,
                results=results,
            )
    if not np.all(np.isfinite(controlling)) or not np.all(np.isfinite(diagnostic)):
        raise ValueError("directional NLI feature matrix contains non-finite values")
    return controlling, diagnostic


def _fit_numeric_model(
    features: np.ndarray,
    labels: np.ndarray,
    routed: np.ndarray,
    *,
    binary_threshold: float,
) -> dict[str, Any]:
    if not math.isfinite(binary_threshold):
        raise ValueError("fusion numeric model binary threshold must be finite")
    spec = _fixed_spec(FUSION_FEATURE_SCHEMA_VERSION)
    estimator = model_experiment._make_estimator(spec)
    estimator.fit(features[routed], labels[routed])
    steps = estimator.named_steps
    scaler = steps["standardscaler"]
    ridge = steps["ridgeclassifier"]
    payload = {
        "feature_schema": FUSION_FEATURE_SCHEMA_VERSION,
        "feature_dimension": FUSION_FEATURE_DIMENSION,
        "feature_mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
        "feature_scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        "coefficients": np.asarray(ridge.coef_, dtype=np.float64).reshape(-1).tolist(),
        "intercept": float(np.asarray(ridge.intercept_).reshape(-1)[0]),
        "binary_threshold": float(binary_threshold),
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "membership_effect": "none",
    }
    payload["numeric_model_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_numeric_model(model: Mapping[str, Any]) -> None:
    payload = dict(model)
    observed_sha256 = payload.pop("numeric_model_sha256", None)
    if observed_sha256 != _canonical_sha256(payload):
        raise ValueError("fusion numeric model SHA-256 mismatch")
    if (
        model.get("feature_schema") != FUSION_FEATURE_SCHEMA_VERSION
        or model.get("feature_dimension") != FUSION_FEATURE_DIMENSION
        or model.get("diagnostic_only") is not True
        or model.get("hard_membership_authorized") is not False
        or model.get("membership_effect") != "none"
    ):
        raise ValueError("fusion numeric model contract mismatch")
    for name in ("feature_mean", "feature_scale", "coefficients"):
        values = np.asarray(model.get(name), dtype=np.float64)
        if values.shape != (FUSION_FEATURE_DIMENSION,) or not np.all(np.isfinite(values)):
            raise ValueError(f"fusion numeric model {name} is invalid")
        if name == "feature_scale" and np.any(values <= 0.0):
            raise ValueError("fusion numeric model scale must be positive")
    if not math.isfinite(float(model.get("intercept", math.nan))):
        raise ValueError("fusion numeric model intercept is invalid")
    try:
        binary_threshold = float(model.get("binary_threshold", math.nan))
    except (TypeError, ValueError) as exc:
        raise ValueError("fusion numeric model binary threshold is invalid") from exc
    if not math.isfinite(binary_threshold):
        raise ValueError("fusion numeric model binary threshold is invalid")


def _score_numeric_model(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    _validate_numeric_model(model)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FUSION_FEATURE_DIMENSION:
        raise ValueError("fusion scoring matrix has the wrong shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("fusion scoring matrix contains non-finite values")
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    return ((matrix - mean) / scale) @ coefficients + float(model["intercept"])


def _authenticated_baseline_margins(
    path: Path,
    held_out_pairs: Sequence[Any],
) -> np.ndarray:
    """Load pair-aligned compact margins from the authenticated frozen reveal."""

    if _file_sha256(path) != EXPECTED_FROZEN_REVEAL_FILE_SHA256:
        raise ValueError("frozen compact resolver reveal file identity mismatch")
    reveal = _load_json(path)
    traces = reveal.get("held_out_pair_traces")
    if not isinstance(traces, list) or len(traces) != len(held_out_pairs):
        raise ValueError("frozen compact resolver margin traces are not pair-aligned")
    by_pair: dict[int, Mapping[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("frozen compact resolver margin trace is malformed")
        pair_id = int(trace.get("pair_id", 0))
        if pair_id <= 0 or pair_id in by_pair:
            raise ValueError("frozen compact resolver margin trace identity is invalid")
        by_pair[pair_id] = trace
    if set(by_pair) != {pair.pair_id for pair in held_out_pairs}:
        raise ValueError("frozen compact resolver margin trace alignment mismatch")
    margins: list[float] = []
    for pair in held_out_pairs:
        raw_margin = by_pair[pair.pair_id].get("resolver_score")
        if isinstance(raw_margin, bool):
            raise ValueError("frozen compact resolver margin trace is malformed")
        try:
            margin = float(raw_margin)
        except (TypeError, ValueError) as exc:
            raise ValueError("frozen compact resolver margin trace is malformed") from exc
        if not math.isfinite(margin):
            raise ValueError("frozen compact resolver margin trace is non-finite")
        margins.append(margin)
    return np.asarray(margins, dtype=np.float64)


def _frozen_compact_parity(
    *,
    artifact: CompactGenericLinearArtifact,
    compact_features: np.ndarray,
    current_routes: np.ndarray,
    trace_routes: np.ndarray,
    trace_margins: np.ndarray,
    trace_predictions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recompute the frozen compact lane and require exact trace parity."""

    current_routes = np.asarray(current_routes, dtype=bool)
    trace_routes = np.asarray(trace_routes, dtype=bool)
    trace_margins = np.asarray(trace_margins, dtype=np.float64)
    trace_predictions = np.asarray(trace_predictions, dtype=bool)
    expected_shape = current_routes.shape
    if (
        current_routes.ndim != 1
        or trace_routes.shape != expected_shape
        or trace_margins.shape != expected_shape
        or trace_predictions.shape != expected_shape
        or not np.all(np.isfinite(trace_margins))
    ):
        raise ValueError("frozen compact parity inputs are not pair-aligned and finite")
    if not np.array_equal(current_routes, trace_routes):
        raise ValueError("held-out route differs from the frozen compact trace")

    recomputed_margins = artifact.margins(compact_features)
    if not np.array_equal(recomputed_margins, trace_margins):
        raise ValueError("recomputed compact margins differ from the frozen compact trace")
    binary_threshold = artifact.thresholds.binary_equivalence_margin
    recomputed_predictions = current_routes & (recomputed_margins >= binary_threshold)
    if not np.array_equal(recomputed_predictions, trace_predictions):
        raise ValueError("recomputed compact predictions differ from the frozen compact trace")

    route_sha256 = _canonical_sha256(current_routes.tolist())
    margin_sha256 = _array_sha256(recomputed_margins)
    prediction_sha256 = _canonical_sha256(recomputed_predictions.tolist())
    return recomputed_margins, {
        "exact_route_match": True,
        "exact_margin_match": True,
        "exact_prediction_match": True,
        "compact_feature_matrix_sha256": _array_sha256(compact_features),
        "recomputed_route_sha256": route_sha256,
        "trace_route_sha256": _canonical_sha256(trace_routes.tolist()),
        "recomputed_margin_sha256": margin_sha256,
        "trace_margin_sha256": _array_sha256(trace_margins),
        "recomputed_prediction_sha256": prediction_sha256,
        "trace_prediction_sha256": _canonical_sha256(trace_predictions.tolist()),
        "binary_threshold": binary_threshold,
        "artifact": artifact.identity(),
    }


def _load_common_inputs(args: argparse.Namespace) -> dict[str, Any]:
    pairs, split, embeddings, signatures, graphs, evidence = model_experiment._load_inputs(args)
    manifest_path = args.nli_work_dir.resolve() / "request_manifest.json"
    cache_path = args.nli_work_dir.resolve() / "inference_cache.json"
    manifest = _load_json(manifest_path)
    results, cache = _load_cache(cache_path, manifest=manifest)
    return {
        "pairs": pairs,
        "split": split,
        "embeddings": embeddings,
        "signatures": signatures,
        "graphs": graphs,
        "evidence": evidence,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "cache": cache,
        "cache_path": cache_path,
        "results": results,
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
    texts = {text for pair in pairs for text in (pair.sentence1, pair.sentence2)}
    inventories = {
        text: project_atomic_assertions(common["graphs"][text], source_text=text) for text in texts
    }
    compact = model_experiment._feature_matrices(
        pairs,
        embeddings=common["embeddings"],
        signatures=common["signatures"],
        graphs=common["graphs"],
    )[COMPACT_GENERIC_FEATURE_SCHEMA_VERSION]
    controlling_nli, diagnostic_nli = _nli_feature_matrix(
        pairs,
        routed=routed,
        inventories=inventories,
        results=common["results"],
    )
    return {
        "pairs": pairs,
        "routed": routed,
        "route_evidence": route_evidence,
        "inventories": inventories,
        "inventory_identity": _inventory_identity(inventories),
        "compact_features": compact,
        "controlling_nli_features": controlling_nli,
        "diagnostic_nli_features": diagnostic_nli,
        "fusion_features": np.column_stack((compact, controlling_nli)),
    }


def _source_identity() -> dict[str, str]:
    paths = {
        "fusion_benchmark": Path(__file__).resolve(),
        "directional_nli_features": (
            APP_ROOT / "mcp_servers" / "mcp_lib" / "directional_nli_features.py"
        ),
        "model_experiment": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "experiment_mrpc_semantic_relation_models.py"
        ),
        "directional_nli_benchmark": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "run_mrpc_directional_nli_benchmark.py"
        ),
        "gate_verifier_benchmark": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "run_mrpc_gate_verifier_benchmark.py"
        ),
        "directional_semantic_relation": (
            APP_ROOT / "mcp_servers" / "mcp_lib" / "directional_semantic_relation.py"
        ),
        "semantic_relation_features": (
            APP_ROOT / "mcp_servers" / "mcp_lib" / "semantic_relation_features.py"
        ),
        "semantic_routing_contract": (
            APP_ROOT / "mcp_servers" / "mcp_lib" / "semantic_routing_contract.py"
        ),
        "semantic_relation_resolver_benchmark": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "run_mrpc_semantic_relation_resolver_benchmark.py"
        ),
        "qwen_typed_field_test": (
            APP_ROOT / "mcp_servers" / "mcp_lib" / "qwen_typed_field_test.py"
        ),
        "embedding_pipeline": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "run_mrpc_current_pipeline_benchmark.py"
        ),
        "graph_pipeline": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "run_mrpc_proposition_graph_benchmark.py"
        ),
        "cache_evidence_pipeline": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "run_mrpc_production_pipeline_benchmark.py"
        ),
        "cache_builder": (
            REPOSITORY_ROOT
            / "scripts"
            / "research"
            / "mrpc"
            / "build_mrpc_directional_nli_inference_cache.py"
        ),
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _compute_selection(args: argparse.Namespace) -> dict[str, Any]:
    common = _load_common_inputs(args)
    calibration = _partition_inputs(
        common,
        partition="calibration",
        top_k=args.psa_top_k,
    )
    calibration_pairs = calibration["pairs"]
    calibration_ids = frozenset(pair.pair_id for pair in calibration_pairs)
    label_map = model_experiment._load_selected_labels(args.mrpc_dir.resolve(), calibration_ids)
    labels = np.asarray([label_map[pair.pair_id] for pair in calibration_pairs], dtype=bool)
    groups, component_count = model_experiment._component_groups(calibration_pairs)
    folds = model_experiment._group_folds(groups)
    routed = calibration["routed"]
    base_result = model_experiment._crossfit(
        _fixed_spec(COMPACT_GENERIC_FEATURE_SCHEMA_VERSION),
        calibration["compact_features"],
        labels,
        routed,
        folds,
    )
    fusion_result = model_experiment._crossfit(
        _fixed_spec(FUSION_FEATURE_SCHEMA_VERSION),
        calibration["fusion_features"],
        labels,
        routed,
        folds,
    )
    frozen_compact = load_compact_generic_linear_artifact(
        artifact_path=args.frozen_compact_artifact.resolve(),
        contract_path=args.frozen_compact_contract.resolve(),
        expected_artifact_sha256=EXPECTED_FROZEN_ARTIFACT_SHA256,
        expected_contract_file_sha256=EXPECTED_FROZEN_CONTRACT_FILE_SHA256,
    )
    frozen_winner = frozen_compact.contract["winner"]
    if dict(base_result.metrics) != frozen_winner["oof_metrics"] or float(
        base_result.threshold
    ) != float(frozen_winner["threshold"]):
        raise ValueError("341-feature calibration control no longer replays")
    numeric_model = _fit_numeric_model(
        calibration["fusion_features"],
        labels,
        routed,
        binary_threshold=float(fusion_result.threshold),
    )
    _validate_numeric_model(numeric_model)
    dependency_identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        },
    }
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "phase": "calibration_fusion_frozen_before_held_out_label_loading",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "membership_effect": "none",
        "split": {
            "seed": BLIND_SPLIT_SEED,
            "calibration_pair_count": len(calibration_pairs),
            "calibration_component_count": component_count,
            "fold_count": model_experiment.MODEL_SELECTION_FOLDS,
            "fold_seed": model_experiment.FOLD_SEED,
            "held_out_labels_loaded": False,
        },
        "routing": {
            "top_k": args.psa_top_k,
            "routed_pair_count": int(np.count_nonzero(routed)),
            "route_vector_sha256": _canonical_sha256(routed.tolist()),
            "evidence": calibration["route_evidence"],
        },
        "fixed_estimator": {
            "family": "ridge",
            "alpha": FIXED_RIDGE_ALPHA,
            "class_weight": FIXED_CLASS_WEIGHT,
            "solver": "cholesky",
            "standardization": "StandardScaler",
        },
        "feature_contract": {
            "compact_schema_version": COMPACT_GENERIC_FEATURE_SCHEMA_VERSION,
            "compact_dimension": COMPACT_GENERIC_FEATURE_DIMENSION,
            "controlling_directional_schema_version": (
                DIRECTIONAL_NLI_LOG_ENTAILMENT_SCHEMA_VERSION
            ),
            "controlling_directional_dimension": (DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_DIMENSION),
            "controlling_directional_feature_names": list(
                DIRECTIONAL_NLI_LOG_ENTAILMENT_FEATURE_NAMES
            ),
            "diagnostic_atomic_schema_version": DIRECTIONAL_NLI_FEATURE_SCHEMA_VERSION,
            "diagnostic_atomic_dimension": DIRECTIONAL_NLI_FEATURE_DIMENSION,
            "diagnostic_atomic_feature_names": list(DIRECTIONAL_NLI_FEATURE_NAMES),
            "fusion_dimension": FUSION_FEATURE_DIMENSION,
            "swap_invariant": True,
            "phrase_or_lemma_exception_features": False,
            "atomic_features_control_score": False,
        },
        "calibration_controls": {
            "compact_only": _candidate_evidence(
                base_result, dimension=COMPACT_GENERIC_FEATURE_DIMENSION
            ),
            "compact_plus_log_directional_entailment": _candidate_evidence(
                fusion_result, dimension=FUSION_FEATURE_DIMENSION
            ),
            "atomic_exclusion_reason": (
                "Calibration did not establish incremental atomic-aggregate value; "
                "atomic evidence remains explanatory and cannot control this score."
            ),
        },
        "model": numeric_model,
        "minimum_required_f1_gain": args.minimum_f1_gain,
        "bootstrap_policy": {
            "sample_count": PAIRED_BOOTSTRAP_SAMPLES,
            "seed": PAIRED_BOOTSTRAP_SEED,
        },
        "frozen_comparator": {
            "report_file_sha256": EXPECTED_FROZEN_REVEAL_FILE_SHA256,
            "metrics": EXPECTED_FROZEN_HELD_OUT_METRICS,
        },
        "input_identity": {
            "dataset": _dataset_identity(args.mrpc_dir.resolve()),
            "embedding_and_graph_evidence": common["evidence"],
            "request_manifest_file_sha256": _file_sha256(common["manifest_path"]),
            "request_manifest_sha256": common["manifest"]["manifest_sha256"],
            "nli_cache_file_sha256": _file_sha256(common["cache_path"]),
            "nli_cache_sha256": common["cache"]["cache_sha256"],
            "nli_model": common["cache"]["payload"]["model"],
            "calibration_inventory": calibration["inventory_identity"],
            "compact_feature_matrix_sha256": _array_sha256(calibration["compact_features"]),
            "controlling_directional_feature_matrix_sha256": _array_sha256(
                calibration["controlling_nli_features"]
            ),
            "diagnostic_atomic_feature_matrix_sha256": _array_sha256(
                calibration["diagnostic_nli_features"]
            ),
            "calibration_pair_ids_sha256": _canonical_sha256(
                [pair.pair_id for pair in calibration_pairs]
            ),
            "calibration_labels_sha256": _canonical_sha256(labels.tolist()),
        },
        "frozen_compact_artifact": frozen_compact.identity(),
        "source_identity": _source_identity(),
        "dependency_identity": dependency_identity,
        "caveats": [
            "MRPC is a repeatedly observed development dashboard, not a fresh blind test.",
            "The fusion hypothesis is post-hoc to earlier MRPC reveals and needs OOD evaluation.",
            "Atomic support is retained in diagnostics but excluded from the controlling model.",
            "Learned fusion output has no hard-membership or database-write authority.",
        ],
    }
    return {**payload, "selection_sha256": _canonical_sha256(payload)}


def _reveal_report(
    args: argparse.Namespace,
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    common = _load_common_inputs(args)
    held = _partition_inputs(common, partition="held_out", top_k=args.psa_top_k)
    pairs = held["pairs"]
    held_ids = frozenset(pair.pair_id for pair in pairs)
    held_label_map = model_experiment._load_selected_labels(args.mrpc_dir.resolve(), held_ids)
    labels = np.asarray([held_label_map[pair.pair_id] for pair in pairs], dtype=bool)
    baseline_path = args.baseline_reveal.resolve()
    baseline_labels, baseline_predictions, baseline_routes, baseline = _validated_baseline(
        baseline_path, pairs
    )
    if not np.array_equal(labels, baseline_labels):
        raise ValueError("held-out labels do not align with the frozen comparator")
    routed = np.asarray(held["routed"], dtype=bool)
    if not np.array_equal(routed, baseline_routes):
        raise ValueError("held-out route differs from the frozen comparator")
    if {
        "report_file_sha256": baseline["report_file_sha256"],
        "metrics": baseline["metrics"],
    } != selection["frozen_comparator"]:
        raise ValueError("frozen compact comparator identity changed")
    frozen_compact = load_compact_generic_linear_artifact(
        artifact_path=args.frozen_compact_artifact.resolve(),
        contract_path=args.frozen_compact_contract.resolve(),
        expected_artifact_sha256=EXPECTED_FROZEN_ARTIFACT_SHA256,
        expected_contract_file_sha256=EXPECTED_FROZEN_CONTRACT_FILE_SHA256,
    )
    if frozen_compact.identity() != selection["frozen_compact_artifact"]:
        raise ValueError("frozen compact artifact identity changed after selection")
    baseline_trace_margins = _authenticated_baseline_margins(baseline_path, pairs)
    baseline_margins, compact_parity = _frozen_compact_parity(
        artifact=frozen_compact,
        compact_features=held["compact_features"],
        current_routes=routed,
        trace_routes=baseline_routes,
        trace_margins=baseline_trace_margins,
        trace_predictions=baseline_predictions,
    )
    scores = _score_numeric_model(selection["model"], held["fusion_features"])
    predictions = routed & (scores >= float(selection["model"]["binary_threshold"]))
    metrics = _binary_metrics(labels, predictions)
    f1_gain = float(metrics["f1"] - baseline["metrics"]["f1"])
    bootstrap_policy = selection["bootstrap_policy"]
    bootstrap = _paired_f1_bootstrap(
        labels,
        predictions,
        baseline_predictions,
        sample_count=int(bootstrap_policy["sample_count"]),
        seed=int(bootstrap_policy["seed"]),
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
            "controlling_log_entailment_features": held["controlling_nli_features"][index].tolist(),
            "whole_sentence_nli_diagnostics": held["diagnostic_nli_features"][index, :6].tolist(),
            "atomic_nli_diagnostics": held["diagnostic_nli_features"][index, 6:].tolist(),
            "membership_effect": "none",
        }
        for index, pair in enumerate(pairs)
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_class": "post_hoc_mrpc_development_diagnostic",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "membership_effect": "none",
        "selection": dict(selection),
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
            "held_route": held["route_evidence"],
            "held_inventory": held["inventory_identity"],
            "held_compact_feature_matrix_sha256": _array_sha256(held["compact_features"]),
            "held_controlling_directional_feature_matrix_sha256": _array_sha256(
                held["controlling_nli_features"]
            ),
            "held_diagnostic_atomic_feature_matrix_sha256": _array_sha256(
                held["diagnostic_nli_features"]
            ),
            "held_score_sha256": _array_sha256(scores),
            "held_prediction_sha256": _canonical_sha256(predictions.tolist()),
            "frozen_compact_parity": compact_parity,
        },
        "held_out_pair_traces": traces,
        "caveats": [
            "Held MRPC labels have been visible in prior repository development.",
            "The result is diagnostic and does not establish external generalization.",
            "Atomic axis deficits are diagnostics, not semantic rules or score inputs.",
            "No prediction or atom feature writes a membership or satellite edge.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("select", "reveal"), required=True)
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
    parser.add_argument("--nli-work-dir", type=Path, default=DEFAULT_NLI_WORK_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-reveal", type=Path, default=DEFAULT_BASELINE_REVEAL)
    parser.add_argument("--frozen-compact-artifact", type=Path, default=DEFAULT_FROZEN_ARTIFACT)
    parser.add_argument("--frozen-compact-contract", type=Path, default=DEFAULT_FROZEN_CONTRACT)
    parser.add_argument("--minimum-f1-gain", type=float, default=DEFAULT_MINIMUM_F1_GAIN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.minimum_f1_gain) or args.minimum_f1_gain < 0.0:
        raise ValueError("minimum F1 gain must be finite and nonnegative")
    selection = _compute_selection(args)
    work_dir = args.work_dir.resolve()
    selection_path = work_dir / "frozen_selection.json"
    if args.phase == "select":
        _atomic_json(selection_path, selection)
        print(
            _canonical_json(
                {
                    "status": "fusion_selection_frozen",
                    "selection_path": str(selection_path),
                    "selection_sha256": selection["selection_sha256"],
                    "calibration_controls": selection["calibration_controls"],
                }
            )
        )
        return 0
    frozen_selection = _load_json(selection_path)
    if frozen_selection != selection:
        raise ValueError("frozen fusion selection no longer replays exactly")
    report = _reveal_report(args, selection=selection)
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "fusion_report.json"
    _atomic_json(report_path, report)
    print(
        _canonical_json(
            {
                "status": "fusion_held_out_revealed",
                "report_path": str(report_path),
                "report_file_sha256": _file_sha256(report_path),
                "metrics": report["held_out"]["metrics"],
                "f1_gain": report["held_out"]["f1_gain_over_frozen_compact_resolver"],
                "paired_bootstrap": report["held_out"]["paired_bootstrap_f1_difference"],
                "passed_promotion_gate": report["held_out"]["passed_promotion_gate"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
