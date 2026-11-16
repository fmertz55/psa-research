#!/usr/bin/env python3
"""Run one pinned bidirectional-NLI experiment on the frozen MRPC dashboard.

The only controlling score is the minimum of forward and reverse entailment
probability from one pinned cross-encoder.  The threshold is selected from the
existing component-disjoint calibration partition, frozen, and only then
applied to the historically visible held-out development partition.

Proposition atoms are scored in the same immutable request cache, but they do
not affect this first controlling score.  They explain which independent axes
failed in a direction and provide evidence for later plural satellite-edge and
MicroGPT work without adding another score-tuning variable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
for import_path in (REPOSITORY_ROOT, APP_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.research.mrpc.build_mrpc_directional_nli_inference_cache import (  # noqa: E402
    load_and_validate_inference_cache,
)
from scripts.research.mrpc.run_mrpc_current_pipeline_benchmark import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    _load_pinned_frame,
    _project_representations,
)
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
    EXPECTED_FROZEN_HELD_OUT_METRICS,
    EXPECTED_FROZEN_REVEAL_FILE_SHA256,
    PAIRED_BOOTSTRAP_SAMPLES,
    PAIRED_BOOTSTRAP_SEED,
    _dataset_identity,
    _f1_threshold,
    _paired_f1_bootstrap,
    _partition_psa_routing,
)

from mcp_servers.mcp_lib.directional_semantic_relation import (  # noqa: E402
    AtomicAssertion,
    AtomicAssertionInventory,
    AtomicSupportRecord,
    DirectionalRelationDecision,
    DirectionalRelationOutcome,
    NliRelation,
    SemanticAxis,
    project_atomic_assertions,
    unsupported_axes,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    MRPCPair,
    _binary_metrics,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    PSA_SIGNATURE_CANDIDATE_LIMIT,
)

REQUEST_MANIFEST_SCHEMA_VERSION = "mrpc_directional_nli_request_manifest_v1"
NLI_CACHE_SCHEMA_VERSION = "mrpc_directional_nli_inference_cache_v1"
SELECTION_SCHEMA_VERSION = "mrpc_directional_nli_selection_v1"
REPORT_SCHEMA_VERSION = "mrpc_directional_nli_benchmark_v1"
MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
MODEL_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
MODEL_ONNX_RELATIVE_PATH = "onnx/model_qint8_arm64.onnx"
MODEL_ONNX_SHA256 = "ef41fe8474f6070373f97fbe8362461d55108f387723280fcdb64f8015133db2"
MUTUAL_SCORE_POLICY = "min(forward_entailment,reverse_entailment)"
DEFAULT_MINIMUM_F1_GAIN = 0.01
DEFAULT_WORK_DIR = REPOSITORY_ROOT / "temp" / "mrpc_directional_nli"
DEFAULT_BASELINE_REVEAL = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_semantic_relation_model_search_2026-08-03"
    / "held_out_reveal.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_directional_nli_2026-08-03"


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


def _request_id(premise: str, hypothesis: str) -> str:
    return _canonical_sha256(
        {
            "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "premise": premise,
            "hypothesis": hypothesis,
        }
    )


def _request(premise: str, hypothesis: str) -> dict[str, str]:
    return {
        "request_id": _request_id(premise, hypothesis),
        "premise": premise,
        "hypothesis": hypothesis,
    }


def _manifest(
    pairs: Sequence[MRPCPair],
    *,
    routed: Mapping[int, bool],
    inventories: Mapping[str, AtomicAssertionInventory],
) -> dict[str, Any]:
    requests: dict[str, dict[str, str]] = {}
    for pair in pairs:
        if not routed[pair.pair_id]:
            continue
        pair_requests = (
            _request(pair.sentence1, pair.sentence2),
            _request(pair.sentence2, pair.sentence1),
        )
        for request in pair_requests:
            requests[request["request_id"]] = request
        for assertion in inventories[pair.sentence2].assertions:
            request = _request(pair.sentence1, assertion.text)
            requests[request["request_id"]] = request
        for assertion in inventories[pair.sentence1].assertions:
            request = _request(pair.sentence2, assertion.text)
            requests[request["request_id"]] = request
    payload = {
        "schema_version": REQUEST_MANIFEST_SCHEMA_VERSION,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "onnx_relative_path": MODEL_ONNX_RELATIVE_PATH,
        },
        "requests": sorted(requests.values(), key=lambda value: value["request_id"]),
    }
    return {**payload, "manifest_sha256": _canonical_sha256(payload)}


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("schema_version") != REQUEST_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported directional NLI request manifest schema")
    if manifest.get("manifest_sha256") != _canonical_sha256(payload):
        raise ValueError("directional NLI request manifest SHA-256 mismatch")
    requests = manifest.get("requests")
    if not isinstance(requests, list):
        raise ValueError("directional NLI request manifest requests must be a list")
    seen: set[str] = set()
    for request in requests:
        if not isinstance(request, Mapping):
            raise ValueError("directional NLI request must be a mapping")
        expected = _request_id(str(request.get("premise")), str(request.get("hypothesis")))
        if request.get("request_id") != expected or expected in seen:
            raise ValueError("directional NLI request identity mismatch")
        seen.add(expected)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_cache(
    path: Path,
    *,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    cache = load_and_validate_inference_cache(path, manifest=dict(manifest))
    payload = cache.get("payload")
    if cache.get("schema_version") != NLI_CACHE_SCHEMA_VERSION or not isinstance(payload, Mapping):
        raise ValueError("unsupported directional NLI cache schema")
    manifest_identity = payload.get("request_manifest") or {}
    if (
        manifest_identity.get("schema_version") != REQUEST_MANIFEST_SCHEMA_VERSION
        or manifest_identity.get("manifest_sha256") != manifest.get("manifest_sha256")
        or manifest_identity.get("request_count") != len(manifest["requests"])
    ):
        raise ValueError("directional NLI cache does not bind the request manifest")
    model = payload.get("model") or {}
    if (
        model.get("id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or model.get("onnx_relative_path") != MODEL_ONNX_RELATIVE_PATH
        or (model.get("files") or {}).get(MODEL_ONNX_RELATIVE_PATH) != MODEL_ONNX_SHA256
    ):
        raise ValueError("directional NLI cache model identity mismatch")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("directional NLI cache results must be a list")
    results: dict[str, Mapping[str, Any]] = {}
    for result in raw_results:
        if not isinstance(result, Mapping):
            raise ValueError("directional NLI cache result must be a mapping")
        request_id = str(result.get("request_id") or "")
        probabilities = result.get("probabilities") or {}
        values = tuple(
            float(probabilities.get(label, math.nan))
            for label in ("contradiction", "entailment", "neutral")
        )
        if (
            not request_id
            or request_id in results
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
            or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError("directional NLI cache result is malformed")
        results[request_id] = result
    expected_ids = {str(request["request_id"]) for request in manifest["requests"]}
    if set(results) != expected_ids:
        raise ValueError("directional NLI cache request coverage mismatch")
    return results, cache


def _probabilities(
    results: Mapping[str, Mapping[str, Any]],
    premise: str,
    hypothesis: str,
) -> tuple[float, float, float]:
    result = results[_request_id(premise, hypothesis)]
    probabilities = result["probabilities"]
    return (
        float(probabilities["contradiction"]),
        float(probabilities["entailment"]),
        float(probabilities["neutral"]),
    )


def _mutual_score(
    pair: MRPCPair,
    results: Mapping[str, Mapping[str, Any]],
) -> float:
    forward = _probabilities(results, pair.sentence1, pair.sentence2)[1]
    reverse = _probabilities(results, pair.sentence2, pair.sentence1)[1]
    return min(forward, reverse)


def _relation(probabilities: tuple[float, float, float]) -> NliRelation:
    values = {
        NliRelation.CONTRADICTION: probabilities[0],
        NliRelation.ENTAILMENT: probabilities[1],
        NliRelation.NEUTRAL: probabilities[2],
    }
    return max(values, key=lambda relation: (values[relation], relation.value))


def _atomic_records(
    *,
    direction: str,
    premise: str,
    inventory: AtomicAssertionInventory,
    results: Mapping[str, Mapping[str, Any]],
) -> list[tuple[AtomicAssertion, AtomicSupportRecord]]:
    records: list[tuple[AtomicAssertion, AtomicSupportRecord]] = []
    premise_sha256 = hashlib.sha256(premise.encode("utf-8")).hexdigest()
    for assertion in inventory.assertions:
        contradiction, entailment, neutral = _probabilities(results, premise, assertion.text)
        records.append(
            (
                assertion,
                AtomicSupportRecord(
                    direction=direction,
                    premise_text_sha256=premise_sha256,
                    hypothesis_inventory_sha256=inventory.inventory_sha256,
                    hypothesis_assertion_id=assertion.assertion_id,
                    hypothesis_assertion_sha256=assertion.canonical_payload_sha256,
                    contradiction_probability=contradiction,
                    entailment_probability=entailment,
                    neutral_probability=neutral,
                    request_id=_request_id(premise, assertion.text),
                    model_id=MODEL_ID,
                    model_revision=MODEL_REVISION,
                ),
            )
        )
    return records


def _directional_trace(
    pair: MRPCPair,
    *,
    results: Mapping[str, Mapping[str, Any]],
    inventories: Mapping[str, AtomicAssertionInventory],
) -> dict[str, Any]:
    forward = _probabilities(results, pair.sentence1, pair.sentence2)
    reverse = _probabilities(results, pair.sentence2, pair.sentence1)
    forward_relation = _relation(forward)
    reverse_relation = _relation(reverse)
    right_atoms = _atomic_records(
        direction="left_to_right",
        premise=pair.sentence1,
        inventory=inventories[pair.sentence2],
        results=results,
    )
    left_atoms = _atomic_records(
        direction="right_to_left",
        premise=pair.sentence2,
        inventory=inventories[pair.sentence1],
        results=results,
    )
    unsupported_left = unsupported_axes(left_atoms)
    unsupported_right = unsupported_axes(right_atoms)
    if NliRelation.CONTRADICTION in {forward_relation, reverse_relation}:
        outcome = DirectionalRelationOutcome.LIKELY_CONTRADICTION
    elif forward_relation is NliRelation.ENTAILMENT and reverse_relation is NliRelation.ENTAILMENT:
        outcome = DirectionalRelationOutcome.LIKELY_EQUIVALENT
    elif forward_relation is NliRelation.ENTAILMENT:
        outcome = DirectionalRelationOutcome.LEFT_ENTAILS_RIGHT
    elif reverse_relation is NliRelation.ENTAILMENT:
        outcome = DirectionalRelationOutcome.RIGHT_ENTAILS_LEFT
    else:
        outcome = DirectionalRelationOutcome.ABSTAIN
    evidence = {
        "whole_sentence_request_ids": [
            _request_id(pair.sentence1, pair.sentence2),
            _request_id(pair.sentence2, pair.sentence1),
        ],
        "left_atom_request_ids": [record.request_id for _, record in left_atoms],
        "right_atom_request_ids": [record.request_id for _, record in right_atoms],
    }
    decision = DirectionalRelationDecision(
        left_text_sha256=hashlib.sha256(pair.sentence1.encode("utf-8")).hexdigest(),
        right_text_sha256=hashlib.sha256(pair.sentence2.encode("utf-8")).hexdigest(),
        left_to_right_entailment=forward[1],
        right_to_left_entailment=reverse[1],
        contradiction_probability=max(forward[0], reverse[0]),
        outcome=outcome,
        unsupported_left_axes=unsupported_left,
        unsupported_right_axes=unsupported_right,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        evidence_sha256=_canonical_sha256(evidence),
    )
    proposed_satellite_edges: list[str] = []
    richer_side: str | None = None
    axes: tuple[SemanticAxis, ...] = ()
    if outcome is DirectionalRelationOutcome.LEFT_ENTAILS_RIGHT:
        richer_side = "left"
        axes = unsupported_left
    elif outcome is DirectionalRelationOutcome.RIGHT_ENTAILS_LEFT:
        richer_side = "right"
        axes = unsupported_right
    edge_for_axis = {
        SemanticAxis.SCALE: "member_scale",
        SemanticAxis.TEMPORAL: "member_temporal",
        SemanticAxis.ASPECTUAL: "member_aspectual",
    }
    proposed_satellite_edges = sorted(
        {edge_for_axis[axis] for axis in axes if axis in edge_for_axis}
    )
    return {
        "decision": decision.trace_dict(),
        "forward_relation": forward_relation.value,
        "reverse_relation": reverse_relation.value,
        "forward_atom_count": len(right_atoms),
        "reverse_atom_count": len(left_atoms),
        "forward_atom_entailment_coverage": (
            sum(record.relation is NliRelation.ENTAILMENT for _, record in right_atoms)
            / len(right_atoms)
            if right_atoms
            else 0.0
        ),
        "reverse_atom_entailment_coverage": (
            sum(record.relation is NliRelation.ENTAILMENT for _, record in left_atoms)
            / len(left_atoms)
            if left_atoms
            else 0.0
        ),
        "richer_side": richer_side,
        "proposed_satellite_edge_types": proposed_satellite_edges,
        "membership_effect": "none",
    }


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    mrpc_dir = args.mrpc_dir.resolve()
    dataset_identity = _dataset_identity(mrpc_dir)
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
    pinned_guides, calibration, frame_router, _frame_evidence = _load_pinned_frame(
        args.pinned_guide_embeddings.resolve(), args.calibration.resolve()
    )
    signatures, _labels, _signature_evidence = _project_representations(
        embeddings, pinned_guides, frame_router, calibration
    )
    calibration_routed, calibration_route_evidence = _partition_psa_routing(
        calibration_all, signatures, top_k=args.psa_top_k
    )
    held_routed, held_route_evidence = _partition_psa_routing(
        held_out_pairs, signatures, top_k=args.psa_top_k
    )
    routed = {**calibration_routed, **held_routed}
    graphs, graph_evidence = _build_or_load_graphs(
        texts,
        sense_path=args.sense_cache.resolve(),
        cache_path=args.graph_cache.resolve(),
        expected_cache_sha256=args.graph_cache_sha256,
        expected_parser_identity=args.graph_parser_identity,
    )
    inventories = {
        text: project_atomic_assertions(graphs[text], source_text=text) for text in texts
    }
    return {
        "dataset_identity": dataset_identity,
        "pairs": pairs,
        "calibration_all": calibration_all,
        "held_out_pairs": held_out_pairs,
        "routed": routed,
        "calibration_routed": calibration_routed,
        "held_routed": held_routed,
        "calibration_route_evidence": calibration_route_evidence,
        "held_route_evidence": held_route_evidence,
        "embedding_evidence": embedding_evidence,
        "graph_evidence": graph_evidence,
        "inventories": inventories,
    }


def _selection(
    *,
    inputs: Mapping[str, Any],
    manifest_path: Path,
    cache_path: Path,
    cache: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    minimum_f1_gain: float,
) -> dict[str, Any]:
    calibration_pairs = [
        pair for pair in inputs["calibration_all"] if inputs["calibration_routed"][pair.pair_id]
    ]
    scores = np.asarray(
        [_mutual_score(pair, results) for pair in calibration_pairs], dtype=np.float64
    )
    labels = np.asarray([pair.is_paraphrase for pair in calibration_pairs], dtype=bool)
    threshold, metrics = _f1_threshold(labels, scores)
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "phase": "calibration_selection_before_directional_held_out_evaluation",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "membership_effect": "none",
        "model": cache["payload"]["model"],
        "score_policy": MUTUAL_SCORE_POLICY,
        "threshold": threshold,
        "minimum_required_f1_gain": minimum_f1_gain,
        "bootstrap_policy": {
            "sample_count": PAIRED_BOOTSTRAP_SAMPLES,
            "seed": PAIRED_BOOTSTRAP_SEED,
        },
        "frozen_comparator": {
            "report_file_sha256": EXPECTED_FROZEN_REVEAL_FILE_SHA256,
            "metrics": EXPECTED_FROZEN_HELD_OUT_METRICS,
        },
        "calibration_metrics": metrics,
        "calibration_pair_count": len(calibration_pairs),
        "calibration_pair_ids_sha256": _canonical_sha256(
            [pair.pair_id for pair in calibration_pairs]
        ),
        "split_seed": BLIND_SPLIT_SEED,
        "routing": inputs["calibration_route_evidence"],
        "dataset_identity": inputs["dataset_identity"],
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": cache["payload"]["request_manifest"]["manifest_sha256"],
        "cache_file_sha256": _file_sha256(cache_path),
        "cache_sha256": cache["cache_sha256"],
        "source_identity": {
            "benchmark_sha256": _file_sha256(Path(__file__).resolve()),
            "directional_module_sha256": _file_sha256(
                APP_ROOT / "mcp_servers" / "mcp_lib" / "directional_semantic_relation.py"
            ),
            "cache_builder_sha256": _file_sha256(
                REPOSITORY_ROOT
                / "scripts"
                / "research"
                / "mrpc"
                / "build_mrpc_directional_nli_inference_cache.py"
            ),
            "metric_and_bootstrap_source_sha256": _file_sha256(
                REPOSITORY_ROOT
                / "scripts"
                / "research"
                / "mrpc"
                / "run_mrpc_semantic_relation_resolver_benchmark.py"
            ),
            "binary_metric_source_sha256": _file_sha256(
                APP_ROOT / "mcp_servers" / "mcp_lib" / "qwen_typed_field_test.py"
            ),
            "embedding_pipeline_source_sha256": _file_sha256(
                REPOSITORY_ROOT
                / "scripts"
                / "research"
                / "mrpc"
                / "run_mrpc_current_pipeline_benchmark.py"
            ),
            "graph_pipeline_source_sha256": _file_sha256(
                REPOSITORY_ROOT
                / "scripts"
                / "research"
                / "mrpc"
                / "run_mrpc_proposition_graph_benchmark.py"
            ),
            "cache_evidence_source_sha256": _file_sha256(
                REPOSITORY_ROOT
                / "scripts"
                / "research"
                / "mrpc"
                / "run_mrpc_production_pipeline_benchmark.py"
            ),
        },
        "caveats": [
            "MRPC held-out labels were visible in prior repository development.",
            "The single threshold is selected only from routed calibration pairs.",
            "Atomic NLI evidence is explanatory and does not control this first score.",
            "No learned output authorizes or writes membership edges.",
        ],
    }
    return {**payload, "selection_sha256": _canonical_sha256(payload)}


def _validated_baseline(
    path: Path,
    held_out_pairs: Sequence[MRPCPair],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if _file_sha256(path) != EXPECTED_FROZEN_REVEAL_FILE_SHA256:
        raise ValueError("frozen compact resolver reveal file identity mismatch")
    reveal = _load_json(path)
    traces = reveal.get("held_out_pair_traces") or []
    by_pair = {int(trace["pair_id"]): trace for trace in traces}
    if set(by_pair) != {pair.pair_id for pair in held_out_pairs}:
        raise ValueError("frozen compact resolver pair alignment mismatch")
    labels = np.asarray([bool(by_pair[pair.pair_id]["label"]) for pair in held_out_pairs])
    predictions = np.asarray(
        [bool(by_pair[pair.pair_id]["resolver_prediction"]) for pair in held_out_pairs]
    )
    routes = np.asarray([bool(by_pair[pair.pair_id]["routed"]) for pair in held_out_pairs])
    metrics = _binary_metrics(labels, predictions)
    if metrics != EXPECTED_FROZEN_HELD_OUT_METRICS:
        raise ValueError("frozen compact resolver metrics changed")
    return (
        labels,
        predictions,
        routes,
        {
            "report_file_sha256": EXPECTED_FROZEN_REVEAL_FILE_SHA256,
            "metrics": metrics,
            "routed_pair_count": int(np.count_nonzero(routes)),
            "route_vector_sha256": _canonical_sha256(routes.tolist()),
        },
    )


def _require_route_parity(current: np.ndarray, frozen: np.ndarray) -> None:
    current = np.asarray(current, dtype=bool)
    frozen = np.asarray(frozen, dtype=bool)
    if current.ndim != 1 or current.shape != frozen.shape or not np.array_equal(current, frozen):
        raise ValueError("directional NLI routing differs from the frozen compact-resolver route")


def _reveal_report(
    *,
    args: argparse.Namespace,
    inputs: Mapping[str, Any],
    selection: Mapping[str, Any],
    cache: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    held_out_pairs = inputs["held_out_pairs"]
    baseline_labels, baseline_predictions, baseline_routes, baseline = _validated_baseline(
        args.baseline_reveal.resolve(), held_out_pairs
    )
    held_labels = np.asarray([pair.is_paraphrase for pair in held_out_pairs], dtype=bool)
    if not np.array_equal(held_labels, baseline_labels):
        raise ValueError("held-out labels do not align with frozen compact resolver")
    if {
        "report_file_sha256": baseline["report_file_sha256"],
        "metrics": baseline["metrics"],
    } != selection["frozen_comparator"]:
        raise ValueError("frozen compact-resolver comparator identity changed")
    routed = np.asarray(
        [inputs["held_routed"][pair.pair_id] for pair in held_out_pairs], dtype=bool
    )
    _require_route_parity(routed, baseline_routes)
    scores = np.zeros(len(held_out_pairs), dtype=np.float64)
    for index, pair in enumerate(held_out_pairs):
        if routed[index]:
            scores[index] = _mutual_score(pair, results)
    predictions = routed & (scores >= float(selection["threshold"]))
    metrics = _binary_metrics(held_labels, predictions)
    f1_gain = float(metrics["f1"] - baseline["metrics"]["f1"])
    bootstrap_policy = selection["bootstrap_policy"]
    bootstrap = _paired_f1_bootstrap(
        held_labels,
        predictions,
        baseline_predictions,
        sample_count=int(bootstrap_policy["sample_count"]),
        seed=int(bootstrap_policy["seed"]),
    )
    traces: list[dict[str, Any]] = []
    edge_ledger: Counter[tuple[str, ...]] = Counter()
    outcome_ledger: Counter[str] = Counter()
    for index, pair in enumerate(held_out_pairs):
        directional = None
        if routed[index]:
            directional = _directional_trace(
                pair,
                results=results,
                inventories=inputs["inventories"],
            )
            edge_types = tuple(directional["proposed_satellite_edge_types"])
            edge_ledger[edge_types] += 1
            outcome_ledger[directional["decision"]["outcome"]] += 1
        traces.append(
            {
                "pair_id": pair.pair_id,
                "label": bool(held_labels[index]),
                "routed": bool(routed[index]),
                "mutual_support_score": float(scores[index]),
                "prediction": bool(predictions[index]),
                "baseline_prediction": bool(baseline_predictions[index]),
                "directional_evidence": directional,
            }
        )
    minimum_f1_gain = float(selection["minimum_required_f1_gain"])
    passed_gain = f1_gain >= minimum_f1_gain
    passed_confidence = bool(bootstrap["lower_bound_strictly_positive"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_class": "post_hoc_mrpc_development_diagnostic",
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "membership_effect": "none",
        "model": cache["payload"]["model"],
        "selection": dict(selection),
        "held_out": {
            "metrics": metrics,
            "baseline": baseline,
            "f1_gain_over_frozen_compact_resolver": f1_gain,
            "paired_bootstrap_f1_difference": bootstrap,
            "minimum_required_f1_gain": minimum_f1_gain,
            "passed_material_gain_gate": passed_gain,
            "passed_bootstrap_confidence_gate": passed_confidence,
            "passed_promotion_gate": passed_gain and passed_confidence,
        },
        "directional_diagnostics": {
            "outcome_counts": dict(sorted(outcome_ledger.items())),
            "proposed_plural_satellite_edge_set_counts": {
                ("+".join(edge_types) if edge_types else "(none)"): count
                for edge_types, count in sorted(edge_ledger.items())
            },
            "atomic_evidence_controls_score": False,
            "reason": "first reveal changes one variable: mutual directional NLI support",
        },
        "routing": {
            "calibration": inputs["calibration_route_evidence"],
            "held_out": inputs["held_route_evidence"],
        },
        "evidence": {
            "dataset_identity": inputs["dataset_identity"],
            "embedding": inputs["embedding_evidence"],
            "graph": inputs["graph_evidence"],
            "nli_cache_sha256": cache["cache_sha256"],
        },
        "held_out_pair_traces": traces,
        "caveats": [
            "The MRPC held-out partition is a development dashboard, not a fresh blind test.",
            "MRPC labels equivalence only; directional outcomes need external entailment evaluation.",
            "Atomic verbalizations are frozen diagnostic evidence and are not proof authority.",
            "Proposed satellite edge sets are not persisted by this benchmark.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "select", "reveal"), required=True)
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
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--baseline-reveal", type=Path, default=DEFAULT_BASELINE_REVEAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--minimum-f1-gain", type=float, default=DEFAULT_MINIMUM_F1_GAIN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.minimum_f1_gain) or args.minimum_f1_gain < 0.0:
        raise ValueError("minimum F1 gain must be finite and nonnegative")
    inputs = _load_inputs(args)
    manifest = _manifest(
        inputs["pairs"],
        routed=inputs["routed"],
        inventories=inputs["inventories"],
    )
    _validate_manifest(manifest)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "request_manifest.json"
    _atomic_json(manifest_path, manifest)
    if args.phase == "prepare":
        print(
            _canonical_json(
                {
                    "status": "prepared",
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "request_count": len(manifest["requests"]),
                    "next": (
                        "run scripts/research/mrpc/build_mrpc_directional_nli_inference_cache.py in the "
                        "isolated Python 3.13 ONNX environment"
                    ),
                }
            )
        )
        return 0
    cache_path = work_dir / "inference_cache.json"
    results, cache = _load_cache(cache_path, manifest=manifest)
    selection = _selection(
        inputs=inputs,
        manifest_path=manifest_path,
        cache_path=cache_path,
        cache=cache,
        results=results,
        minimum_f1_gain=args.minimum_f1_gain,
    )
    selection_path = work_dir / "frozen_selection.json"
    if args.phase == "select":
        _atomic_json(selection_path, selection)
        print(
            _canonical_json(
                {
                    "status": "selection_frozen",
                    "selection_path": str(selection_path),
                    "selection_sha256": selection["selection_sha256"],
                    "threshold": selection["threshold"],
                    "calibration_metrics": selection["calibration_metrics"],
                }
            )
        )
        return 0
    frozen_selection = _load_json(selection_path)
    if frozen_selection != selection:
        raise ValueError("frozen directional NLI selection no longer replays exactly")
    report = _reveal_report(
        args=args,
        inputs=inputs,
        selection=selection,
        cache=cache,
        results=results,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "directional_nli_report.json"
    _atomic_json(report_path, report)
    print(
        _canonical_json(
            {
                "status": "held_out_revealed",
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
