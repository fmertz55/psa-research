#!/usr/bin/env python3
"""Audit topology compression in the frozen compact MRPC false positives.

This is a read-only research diagnostic.  It authenticates the frozen compact
resolver prediction vector and the pinned proposition-evidence cache, rebuilds
graphs only from that cache, and compares the 183 false positives with all 692
true positives.  It does not fit a threshold, change a prediction, or expose a
new scorer input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
for import_path in (REPOSITORY_ROOT, APP_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts.research.mrpc.run_mrpc_proposition_graph_benchmark import (  # noqa: E402
    BLIND_SPLIT_SEED,
)
from scripts.research.mrpc.run_mrpc_semantic_relation_resolver_benchmark import (  # noqa: E402
    _GRAPH_INVENTORY_NAMES,
    _graph_inventory,
)

from mcp_servers.mcp_lib.proposition_graph import (  # noqa: E402
    PropositionGraph,
    build_proposition_graph,
    compare_proposition_graphs,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    MRPCPair,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)

REPORT_SCHEMA_VERSION = "mrpc_topology_collision_audit_v1"
AUDIT_ID = "frozen_compact_false_positive_topology_collision_v1"
EXPECTED_REVEAL_SHA256 = "a5b228d25affc998a0df71471d115369312c958b6534096ebe637fb57ff0d79a"
EXPECTED_GRAPH_CACHE_SHA256 = "9625dd25e9ed67beb5832ec38dccf95b66415f8c49333e2e1477ef8ff7f927fc"
EXPECTED_PAIR_ID_SHA256 = "80ee7384db0dff94dde5641601f39ad02dae4e8b886e34526f780451bfe9c962"
EXPECTED_LABEL_SHA256 = "5f3ac11e6301f139afcd1cd808b010667420e2c6e54dacf3013e2cbb5b2573e9"
EXPECTED_ROUTE_SHA256 = "b89e31dac79a3e7f834a9a80759e3143b99212a664ccbf7d188ebabc0a7dfa1f"
EXPECTED_SCORE_SHA256 = "c90e657cdb93deb1dd3426fa21b2d92e3fdfe5a9b5bea0b957bc8d3a83ee2fd5"
EXPECTED_PREDICTION_SHA256 = "f49cb531be3b5fc722c2c6764b710435738b0240a9c985a57028b539a03a32c1"
EXPECTED_BASELINE_METRICS = {
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
EXPECTED_DATASET_SHA256 = {
    "msr_paraphrase_data.txt": ("0daa19928665d29551d93c42fbd48200d272818b5e91fe589a98ed7467e69519"),
    "msr_paraphrase_test.txt": ("9b4c96307f6e9276f4c32614dea1f1e65fa6777d0e4451c0c4e87c3d048636ef"),
}

# These are descriptive sensitivity bands, not learned thresholds.  A pair is
# near-colliding only when tagged inventory multiset Jaccard reaches the band
# and each of the five existing graph-count values differs by at most one.
NEAR_INVENTORY_BANDS = (0.90, 0.80, 0.70)
NEAR_MAX_GRAPH_COUNT_DELTA = 1

TEMPORAL_PROPERTY_KEYS = frozenset({"tense", "aspect"})
SCALE_PROPERTY_KEYS = frozenset({"direction"})
OTHER_EVENT_PROPERTY_KEYS = frozenset({"polarity", "modality", "attribution"})
SCALE_NODE_KEYS = frozenset({"comparison_degree", "scalar_operator", "comparator", "percentage"})
PROPERTY_PROVENANCE_KEYS = frozenset(
    {
        "binding_source",
        "object_id",
        "object_node_id",
        "proposition_index",
        "role_source",
        "sentence_id",
        "source_id",
        "surface",
        "target_id",
        "text",
    }
)

DEFAULT_MRPC_DIR = REPOSITORY_ROOT / "app" / "utils" / "msrpc_split"
DEFAULT_REVEAL = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_semantic_relation_model_search_2026-08-03"
    / "held_out_reveal.json"
)
DEFAULT_GRAPH_CACHE = (
    REPOSITORY_ROOT / "temp" / "mrpc_proposition_graph" / "spacy_oewn_evidence_v1.json"
)
DEFAULT_REPORT = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_topology_collision_audit_2026-08-04"
    / "topology_collision_audit.json"
)


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    if len(labels) != len(predictions):
        raise ValueError("label and prediction vectors are not aligned")
    true_positive = sum(
        label and prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    false_positive = sum(
        not label and prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    false_negative = sum(
        label and not prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    true_negative = sum(
        not label and not prediction for label, prediction in zip(labels, predictions, strict=False)
    )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (true_positive + true_negative) / max(len(labels), 1),
        "count": len(labels),
        "f1": f1,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "precision": precision,
        "recall": recall,
        "true_negative": true_negative,
        "true_positive": true_positive,
    }


def _authenticate_dataset(mrpc_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for name, expected_sha256 in EXPECTED_DATASET_SHA256.items():
        path = mrpc_dir.resolve().parent / name
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"MRPC dataset identity changed for {name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        files.append(
            {
                "name": name,
                "path": str(path),
                "sha256": actual_sha256,
                "size_bytes": path.stat().st_size,
            }
        )
    return {"files": files, "identity_sha256": _canonical_sha256(files)}


def _authenticate_reveal(
    path: Path,
    held_pairs: Sequence[MRPCPair],
) -> tuple[dict[int, Mapping[str, Any]], dict[str, Any]]:
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != EXPECTED_REVEAL_SHA256:
        raise ValueError(
            "frozen compact reveal identity changed: "
            f"expected {EXPECTED_REVEAL_SHA256}, got {actual_sha256}"
        )
    reveal = _load_json(path)
    traces = reveal.get("held_out_pair_traces")
    if not isinstance(traces, list) or len(traces) != 1107:
        raise ValueError("frozen compact reveal must contain 1,107 pair traces")
    trace_by_pair: dict[int, Mapping[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("frozen compact trace is malformed")
        pair_id = trace.get("pair_id")
        if isinstance(pair_id, bool) or not isinstance(pair_id, int):
            raise ValueError("frozen compact trace has an invalid pair ID")
        if pair_id in trace_by_pair:
            raise ValueError("frozen compact trace has a duplicate pair ID")
        trace_by_pair[pair_id] = trace

    pair_ids = [int(pair.pair_id) for pair in held_pairs]
    if set(pair_ids) != set(trace_by_pair):
        raise ValueError("frozen compact reveal and held partition are not aligned")
    labels = [bool(trace_by_pair[pair_id]["label"]) for pair_id in pair_ids]
    official_labels = [bool(pair.is_paraphrase) for pair in held_pairs]
    routes = [bool(trace_by_pair[pair_id]["routed"]) for pair_id in pair_ids]
    scores = [float(trace_by_pair[pair_id]["resolver_score"]) for pair_id in pair_ids]
    predictions = [bool(trace_by_pair[pair_id]["resolver_prediction"]) for pair_id in pair_ids]
    identities = {
        "pair_id_sha256": _canonical_sha256(pair_ids),
        "label_sha256": _canonical_sha256(labels),
        "route_sha256": _canonical_sha256(routes),
        "score_sha256": _canonical_sha256(scores),
        "prediction_sha256": _canonical_sha256(predictions),
    }
    expected = {
        "pair_id_sha256": EXPECTED_PAIR_ID_SHA256,
        "label_sha256": EXPECTED_LABEL_SHA256,
        "route_sha256": EXPECTED_ROUTE_SHA256,
        "score_sha256": EXPECTED_SCORE_SHA256,
        "prediction_sha256": EXPECTED_PREDICTION_SHA256,
    }
    if identities != expected:
        raise ValueError("frozen compact reveal vectors changed")
    if labels != official_labels:
        raise ValueError("frozen reveal labels differ from the authenticated MRPC rows")
    metrics = _binary_metrics(labels, predictions)
    if metrics != EXPECTED_BASELINE_METRICS:
        raise ValueError("frozen compact metrics changed")
    return trace_by_pair, {
        "path": str(path.resolve()),
        "file_sha256": actual_sha256,
        **identities,
        "metrics": metrics,
        "pair_count": len(pair_ids),
        "prediction_positive_count": sum(predictions),
        "validated": True,
    }


def _load_graphs(
    path: Path,
    texts: Iterable[str],
) -> tuple[dict[str, PropositionGraph], dict[str, Mapping[str, Any]], dict[str, Any]]:
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != EXPECTED_GRAPH_CACHE_SHA256:
        raise ValueError(
            "pinned graph cache identity changed: "
            f"expected {EXPECTED_GRAPH_CACHE_SHA256}, got {actual_sha256}"
        )
    payload = _load_json(path)
    entries = payload.get("sentences")
    if not isinstance(entries, Mapping):
        raise ValueError("pinned graph cache has no sentence mapping")
    expected_hashes = payload.get("sentence_hashes")
    if not isinstance(expected_hashes, list) or set(expected_hashes) != set(entries):
        raise ValueError("pinned graph cache sentence inventory is malformed")
    ordered_texts = sorted(set(texts), key=lambda text: (sha256_text(text), text))
    graphs: dict[str, PropositionGraph] = {}
    evidence_by_text: dict[str, Mapping[str, Any]] = {}
    for text in ordered_texts:
        text_sha256 = sha256_text(text)
        entry = entries.get(text_sha256)
        if not isinstance(entry, Mapping) or entry.get("text_sha256") != text_sha256:
            raise ValueError(f"pinned graph cache is missing text {text_sha256}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"pinned graph cache entry {text_sha256} is malformed")
        graphs[text] = build_proposition_graph(evidence)
        evidence_by_text[text] = evidence
    graph_source = APP_ROOT / "mcp_servers" / "mcp_lib" / "proposition_graph.py"
    return (
        graphs,
        evidence_by_text,
        {
            "path": str(path.resolve()),
            "file_sha256": actual_sha256,
            "cache_schema_version": payload.get("schema_version"),
            "source_graph_schema_version": payload.get("graph_schema_version"),
            "source_evidence_sentence_count": payload.get("sentence_count"),
            "source_fact_extractor": payload.get("fact_extractor"),
            "parser_identity": payload.get("parser_identity"),
            "projected_graph_schema_version": next(iter(graphs.values())).semantic_value.get(
                "schema_version"
            ),
            "projected_graph_count": len(graphs),
            "graph_builder_source_sha256": _file_sha256(graph_source),
            "validated": True,
        },
    )


def _counter_intersection_union(
    source: Counter[str],
    target: Counter[str],
) -> tuple[int, int]:
    return sum((source & target).values()), sum((source | target).values())


def _counter_similarity(source: Counter[str], target: Counter[str]) -> float:
    intersection, union = _counter_intersection_union(source, target)
    return intersection / union if union else 1.0


def _flatten_inventory(inventory: Mapping[str, Counter[str]]) -> Counter[str]:
    return Counter(
        {
            f"{name}\u0000{value}": count
            for name in _GRAPH_INVENTORY_NAMES
            for value, count in inventory[name].items()
        }
    )


def _position_insensitive_inventory(
    inventory: Mapping[str, Counter[str]],
) -> dict[str, Counter[str]]:
    normalized = {name: Counter(values) for name, values in inventory.items()}
    normalized_properties: Counter[str] = Counter()
    for value, count in inventory["property_value"].items():
        property_name, separator, payload = value.partition(":")
        if not separator:
            normalized_properties[value] += count
            continue
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            normalized_properties[value] += count
            continue
        normalized_properties[
            f"{property_name}:{_canonical_json(_semantic_property_value(record))}"
        ] += count
    normalized["property_value"] = normalized_properties
    return normalized


def _near_inventory_collision(
    similarity: float,
    graph_count_deltas: Sequence[int],
    *,
    band: float,
) -> bool:
    return similarity >= band and max(graph_count_deltas, default=0) <= (NEAR_MAX_GRAPH_COUNT_DELTA)


def _labeled_path_inventory(graph: PropositionGraph) -> Counter[str]:
    paths: Counter[str] = Counter()
    for proposition in graph.semantic_value.get("propositions") or []:
        predicate = proposition.get("predicate") or {}
        for argument in proposition.get("arguments") or []:
            path = {
                "predicate": predicate,
                "role": str(argument.get("role") or ""),
                "entity": argument.get("node") or {},
            }
            paths[_canonical_json(path)] += 1
    return paths


def _counter_witnesses(
    values: Counter[str],
    *,
    limit: int = 64,
) -> list[dict[str, Any]]:
    return [
        {"count": int(count), "value": json.loads(value)}
        for value, count in sorted(values.items())[:limit]
    ]


def _node_is_scale_bearing(node: Mapping[str, Any]) -> bool:
    if SCALE_NODE_KEYS & set(node):
        return True
    normalized_quantity = node.get("normalized_quantity")
    return isinstance(normalized_quantity, Mapping) and any(
        key in normalized_quantity for key in ("comparator", "percentage")
    )


def _semantic_property_value(value: Any) -> Any:
    """Remove parser binding coordinates/provenance, preserving semantic facts."""

    if isinstance(value, Mapping):
        return {
            str(key): _semantic_property_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in PROPERTY_PROVENANCE_KEYS
            and not str(key).endswith("_position")
            and not str(key).endswith("_index")
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sorted(
            (_semantic_property_value(item) for item in value),
            key=_canonical_json,
        )
    return value


def _material_event_property_conflicts(
    differences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    material: list[dict[str, Any]] = []
    for difference in differences:
        source = _semantic_property_value(difference.get("source") or {})
        target = _semantic_property_value(difference.get("target") or {})
        if source == target:
            continue
        material.append(
            {
                **dict(difference),
                "source": source,
                "target": target,
                "raw_kernel_source": difference.get("source") or {},
                "raw_kernel_target": difference.get("target") or {},
            }
        )
    return material


def _changed_property_keys(difference: Mapping[str, Any]) -> set[str]:
    source = difference.get("source") or {}
    target = difference.get("target") or {}
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        return set()
    return {str(key) for key in set(source) | set(target) if source.get(key) != target.get(key)}


def _kernel_conflicts(
    source: PropositionGraph,
    target: PropositionGraph,
) -> dict[str, Any]:
    proof = compare_proposition_graphs(source, target)
    differences = [dict(value) for value in proof.differences]
    role_binding = [value for value in differences if value.get("kind") == "role_binding_conflict"]
    event_property = [
        value for value in differences if value.get("kind") == "event_property_conflict"
    ]
    predicate_inventory = [
        value
        for value in differences
        if value.get("kind")
        in {"proposition_missing_from_source", "proposition_missing_from_target"}
    ]
    coreference = [
        value for value in differences if value.get("kind") == "coreference_topology_conflict"
    ]
    semantic_event_property = _material_event_property_conflicts(event_property)
    temporal_properties = [
        value
        for value in semantic_event_property
        if _changed_property_keys(value) & TEMPORAL_PROPERTY_KEYS
    ]
    scale_properties = [
        value
        for value in semantic_event_property
        if _changed_property_keys(value) & SCALE_PROPERTY_KEYS
    ]
    other_properties = [
        value
        for value in semantic_event_property
        if _changed_property_keys(value) & OTHER_EVENT_PROPERTY_KEYS
    ]
    argument_differences = [
        value
        for value in differences
        if value.get("kind") in {"argument_coverage_conflict", "role_binding_conflict"}
    ]
    temporal_arguments: list[dict[str, Any]] = []
    quantity_arguments: list[dict[str, Any]] = []
    scale_arguments: list[dict[str, Any]] = []
    for value in argument_differences:
        node = value.get("node") or {}
        source_roles = {str(role) for role in value.get("source_roles") or []}
        target_roles = {str(role) for role in value.get("target_roles") or []}
        roles = source_roles | target_roles
        if "time" in roles or (isinstance(node, Mapping) and node.get("kind") == "time"):
            temporal_arguments.append(value)
        if "quantity" in roles or (isinstance(node, Mapping) and node.get("kind") == "quantity"):
            quantity_arguments.append(value)
        if isinstance(node, Mapping) and _node_is_scale_bearing(node):
            scale_arguments.append(value)
    return {
        "proof_reason": proof.reason,
        "proof_unresolved": list(proof.unresolved),
        "matched_propositions": int(proof.matched_propositions),
        "role_binding_conflicts": role_binding,
        "predicate_inventory_differences": predicate_inventory,
        "event_property_conflicts": event_property,
        "semantic_event_property_conflicts": semantic_event_property,
        "temporal_property_conflicts": temporal_properties,
        "temporal_argument_conflicts": temporal_arguments,
        "scale_property_conflicts": scale_properties,
        "scale_argument_conflicts": scale_arguments,
        "quantity_argument_conflicts": quantity_arguments,
        "other_event_property_conflicts": other_properties,
        "coreference_topology_conflicts": coreference,
    }


def _predicate_binding_differences(
    source_paths: Counter[str],
    target_paths: Counter[str],
) -> list[dict[str, Any]]:
    source_bindings: dict[str, Counter[str]] = defaultdict(Counter)
    target_bindings: dict[str, Counter[str]] = defaultdict(Counter)
    decoded: dict[str, dict[str, Any]] = {}
    for paths, destination in (
        (source_paths, source_bindings),
        (target_paths, target_bindings),
    ):
        for path_json, count in paths.items():
            path = json.loads(path_json)
            binding = _canonical_json({"role": path.get("role"), "entity": path.get("entity")})
            predicate = _canonical_json(path.get("predicate") or {})
            destination[binding][predicate] += count
            decoded[binding] = {
                "role": path.get("role"),
                "entity": path.get("entity"),
            }
    witnesses: list[dict[str, Any]] = []
    for binding in sorted(set(source_bindings) & set(target_bindings)):
        source_predicates = source_bindings[binding]
        target_predicates = target_bindings[binding]
        if source_predicates == target_predicates:
            continue
        witnesses.append(
            {
                **decoded[binding],
                "source_predicates": [
                    {"count": int(count), "predicate": json.loads(predicate)}
                    for predicate, count in sorted(source_predicates.items())
                ],
                "target_predicates": [
                    {"count": int(count), "predicate": json.loads(predicate)}
                    for predicate, count in sorted(target_predicates.items())
                ],
            }
        )
    return witnesses[:64]


def _frame_conflicts(
    source: PropositionGraph,
    target: PropositionGraph,
) -> list[dict[str, Any]]:
    def frames_by_lemma(graph: PropositionGraph) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        for proposition in graph.semantic_value.get("propositions") or []:
            predicate = proposition.get("predicate") or {}
            lemma = str(predicate.get("lemma") or "")
            result[lemma].update(str(value) for value in predicate.get("frame_ids") or [])
        return result

    source_frames = frames_by_lemma(source)
    target_frames = frames_by_lemma(target)
    witnesses: list[dict[str, Any]] = []
    for lemma in sorted(set(source_frames) & set(target_frames)):
        left = source_frames[lemma]
        right = target_frames[lemma]
        if left and right and left.isdisjoint(right):
            witnesses.append(
                {
                    "predicate_lemma": lemma,
                    "source_frame_ids": sorted(left),
                    "target_frame_ids": sorted(right),
                }
            )
    return witnesses


def _coverage_trace(
    source: PropositionGraph,
    target: PropositionGraph,
    source_evidence: Mapping[str, Any],
    target_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    source_coverage = source_evidence.get("coverage") or {}
    target_coverage = target_evidence.get("coverage") or {}
    required_path_families = ("objects", "roles")
    labeled_paths_observed = all(
        source_coverage.get(family) == "complete" and target_coverage.get(family) == "complete"
        for family in required_path_families
    )
    logical_operator_complete = (
        source.logical_operator_coverage.value == "complete"
        and target.logical_operator_coverage.value == "complete"
    )
    coreference_complete = all(
        coverage.get("coreference_topology") == "complete"
        for coverage in (source_coverage, target_coverage)
    )
    frame_complete = all(
        coverage.get("frames") == "complete" for coverage in (source_coverage, target_coverage)
    ) and all(
        any(
            (proposition.get("predicate") or {}).get("frame_ids")
            for proposition in graph.semantic_value.get("propositions") or []
        )
        for graph in (source, target)
    )
    graph_unresolved = sorted(
        {f"source:{value}" for value in source.unresolved}
        | {f"target:{value}" for value in target.unresolved}
    )
    return {
        "labeled_path_families_complete": labeled_paths_observed,
        "logical_operator_scope_complete": logical_operator_complete,
        "coreference_topology_complete": coreference_complete,
        "frame_grounding_observed_on_both_sides": frame_complete,
        "graph_unresolved": graph_unresolved,
        "graph_locally_resolved": not graph_unresolved,
        "strict_proof_ready": not graph_unresolved and logical_operator_complete,
        "source_coverage": dict(source_coverage),
        "target_coverage": dict(target_coverage),
    }


def _audit_pair(
    pair: MRPCPair,
    *,
    source: PropositionGraph,
    target: PropositionGraph,
    source_evidence: Mapping[str, Any],
    target_evidence: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    source_inventory, source_counts_array = _graph_inventory(source)
    target_inventory, target_counts_array = _graph_inventory(target)
    source_counts = tuple(int(value) for value in source_counts_array.tolist())
    target_counts = tuple(int(value) for value in target_counts_array.tolist())
    graph_count_deltas = tuple(
        abs(left - right) for left, right in zip(source_counts, target_counts, strict=False)
    )
    flattened_source = _flatten_inventory(source_inventory)
    flattened_target = _flatten_inventory(target_inventory)
    inventory_intersection, inventory_union = _counter_intersection_union(
        flattened_source, flattened_target
    )
    inventory_similarity = _counter_similarity(flattened_source, flattened_target)
    exact_inventory = (
        all(source_inventory[name] == target_inventory[name] for name in _GRAPH_INVENTORY_NAMES)
        and source_counts == target_counts
    )
    semantic_source_inventory = _position_insensitive_inventory(source_inventory)
    semantic_target_inventory = _position_insensitive_inventory(target_inventory)
    flattened_semantic_source = _flatten_inventory(semantic_source_inventory)
    flattened_semantic_target = _flatten_inventory(semantic_target_inventory)
    semantic_inventory_intersection, semantic_inventory_union = _counter_intersection_union(
        flattened_semantic_source,
        flattened_semantic_target,
    )
    semantic_inventory_similarity = _counter_similarity(
        flattened_semantic_source,
        flattened_semantic_target,
    )
    semantic_exact_inventory = (
        all(
            semantic_source_inventory[name] == semantic_target_inventory[name]
            for name in _GRAPH_INVENTORY_NAMES
        )
        and source_counts == target_counts
    )

    source_paths = _labeled_path_inventory(source)
    target_paths = _labeled_path_inventory(target)
    path_intersection, path_union = _counter_intersection_union(source_paths, target_paths)
    source_path_count = sum(source_paths.values())
    target_path_count = sum(target_paths.values())
    path_exact = source_paths == target_paths
    coverage = _coverage_trace(
        source,
        target,
        source_evidence,
        target_evidence,
    )
    kernel = _kernel_conflicts(source, target)
    frame_conflicts = (
        _frame_conflicts(source, target)
        if coverage["frame_grounding_observed_on_both_sides"]
        else []
    )
    operators = {
        "observed": coverage["logical_operator_scope_complete"],
        "conflict": False,
        "source": [],
        "target": [],
    }
    if operators["observed"]:
        left_operators = source.semantic_value.get("logical_operators") or []
        right_operators = target.semantic_value.get("logical_operators") or []
        operators = {
            "observed": True,
            "conflict": left_operators != right_operators,
            "source": left_operators,
            "target": right_operators,
        }

    localized_conflict = (
        any(
            kernel[name]
            for name in (
                "role_binding_conflicts",
                "semantic_event_property_conflicts",
                "coreference_topology_conflicts",
            )
        )
        or bool(frame_conflicts)
        or bool(operators["conflict"])
    )
    return {
        "pair_id": int(pair.pair_id),
        "label": bool(pair.is_paraphrase),
        "baseline_prediction": bool(trace["resolver_prediction"]),
        "diagnostic_class": "true_positive" if pair.is_paraphrase else "false_positive",
        "semantic_hash": {
            "exact_match": source.semantic_sha256 == target.semantic_sha256,
            "source": source.semantic_sha256,
            "target": target.semantic_sha256,
            "position_insensitive_exact_match": (
                _semantic_property_value(source.semantic_value)
                == _semantic_property_value(target.semantic_value)
            ),
            "position_insensitive_source": _canonical_sha256(
                _semantic_property_value(source.semantic_value)
            ),
            "position_insensitive_target": _canonical_sha256(
                _semantic_property_value(target.semantic_value)
            ),
        },
        "inventory": {
            "exact_collision": exact_inventory,
            "weighted_intersection": inventory_intersection,
            "weighted_union": inventory_union,
            "weighted_jaccard": inventory_similarity,
            "graph_count_names": [
                "propositions",
                "arguments",
                "edges",
                "coreference_chains",
                "unresolved",
            ],
            "source_graph_counts": list(source_counts),
            "target_graph_counts": list(target_counts),
            "absolute_graph_count_deltas": list(graph_count_deltas),
            "near_collision_bands": {
                f"{band:.2f}": _near_inventory_collision(
                    inventory_similarity,
                    graph_count_deltas,
                    band=band,
                )
                for band in NEAR_INVENTORY_BANDS
            },
            "position_insensitive": {
                "exact_collision": semantic_exact_inventory,
                "weighted_intersection": semantic_inventory_intersection,
                "weighted_union": semantic_inventory_union,
                "weighted_jaccard": semantic_inventory_similarity,
                "near_collision_bands": {
                    f"{band:.2f}": _near_inventory_collision(
                        semantic_inventory_similarity,
                        graph_count_deltas,
                        band=band,
                    )
                    for band in NEAR_INVENTORY_BANDS
                },
            },
        },
        "labeled_paths": {
            "exact_match": path_exact,
            "source_count": source_path_count,
            "target_count": target_path_count,
            "intersection_count": path_intersection,
            "union_count": path_union,
            "jaccard": path_intersection / path_union if path_union else 1.0,
            "source_directional_coverage": (
                path_intersection / source_path_count if source_path_count else 1.0
            ),
            "target_directional_coverage": (
                path_intersection / target_path_count if target_path_count else 1.0
            ),
            "source_only_witnesses": _counter_witnesses(source_paths - target_paths),
            "target_only_witnesses": _counter_witnesses(target_paths - source_paths),
        },
        "conflicts": {
            **kernel,
            "predicate_binding_differences": _predicate_binding_differences(
                source_paths, target_paths
            ),
            "frame_conflicts": frame_conflicts,
            "operator_scope": operators,
            "localized_existing_conflict_observed": localized_conflict,
            "strict_fail_closed_separation": (
                localized_conflict and coverage["strict_proof_ready"]
            ),
        },
        "coverage": coverage,
    }


def _condition_values(record: Mapping[str, Any]) -> dict[str, bool]:
    inventory = record["inventory"]
    paths = record["labeled_paths"]
    conflicts = record["conflicts"]
    coverage = record["coverage"]
    values = {
        "exact_semantic_hash_match": bool(record["semantic_hash"]["exact_match"]),
        "position_insensitive_semantic_hash_match": bool(
            record["semantic_hash"]["position_insensitive_exact_match"]
        ),
        "exact_inventory_collision": bool(inventory["exact_collision"]),
        "exact_inventory_topology_collision": bool(
            inventory["exact_collision"] and not paths["exact_match"]
        ),
        "exact_labeled_path_match": bool(paths["exact_match"]),
        "position_insensitive_exact_inventory_collision": bool(
            inventory["position_insensitive"]["exact_collision"]
        ),
        "position_insensitive_exact_inventory_topology_collision": bool(
            inventory["position_insensitive"]["exact_collision"] and not paths["exact_match"]
        ),
        "zero_labeled_path_intersection": paths["intersection_count"] == 0,
        "role_binding_conflict": bool(conflicts["role_binding_conflicts"]),
        "predicate_inventory_difference": bool(conflicts["predicate_inventory_differences"]),
        "predicate_binding_difference": bool(conflicts["predicate_binding_differences"]),
        "raw_kernel_event_property_conflict": bool(conflicts["event_property_conflicts"]),
        "semantic_event_property_conflict": bool(conflicts["semantic_event_property_conflicts"]),
        "frame_conflict": bool(conflicts["frame_conflicts"]),
        "operator_scope_conflict": bool(conflicts["operator_scope"]["conflict"]),
        "temporal_property_conflict": bool(conflicts["temporal_property_conflicts"]),
        "temporal_argument_conflict": bool(conflicts["temporal_argument_conflicts"]),
        "scale_property_conflict": bool(conflicts["scale_property_conflicts"]),
        "scale_argument_conflict": bool(conflicts["scale_argument_conflicts"]),
        "quantity_argument_conflict": bool(conflicts["quantity_argument_conflicts"]),
        "other_event_property_conflict": bool(conflicts["other_event_property_conflicts"]),
        "coreference_topology_conflict": bool(conflicts["coreference_topology_conflicts"]),
        "localized_existing_conflict_observed": bool(
            conflicts["localized_existing_conflict_observed"]
        ),
        "strict_fail_closed_separation": bool(conflicts["strict_fail_closed_separation"]),
        "graph_locally_resolved": bool(coverage["graph_locally_resolved"]),
        "strict_proof_ready": bool(coverage["strict_proof_ready"]),
        "logical_operator_scope_complete": bool(coverage["logical_operator_scope_complete"]),
        "frame_grounding_observed_on_both_sides": bool(
            coverage["frame_grounding_observed_on_both_sides"]
        ),
        "coreference_topology_complete": bool(coverage["coreference_topology_complete"]),
    }
    for band in NEAR_INVENTORY_BANDS:
        key = f"{band:.2f}"
        values[f"near_inventory_collision_{key}"] = bool(inventory["near_collision_bands"][key])
        values[f"near_inventory_topology_collision_{key}"] = bool(
            inventory["near_collision_bands"][key] and not paths["exact_match"]
        )
        values[f"position_insensitive_near_inventory_collision_{key}"] = bool(
            inventory["position_insensitive"]["near_collision_bands"][key]
        )
        values[f"position_insensitive_near_inventory_topology_collision_{key}"] = bool(
            inventory["position_insensitive"]["near_collision_bands"][key]
            and not paths["exact_match"]
        )
    return values


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lanes = {
        "false_positive": [
            value for value in records if value["diagnostic_class"] == "false_positive"
        ],
        "true_positive": [
            value for value in records if value["diagnostic_class"] == "true_positive"
        ],
    }
    all_conditions = sorted(
        {condition for record in records for condition in _condition_values(record)}
    )
    condition_counts = {
        condition: {
            lane: sum(_condition_values(record)[condition] for record in lane_records)
            for lane, lane_records in lanes.items()
        }
        for condition in all_conditions
    }
    inventory_similarity = {
        lane: sorted(float(value["inventory"]["weighted_jaccard"]) for value in lane_records)
        for lane, lane_records in lanes.items()
    }

    def quantiles(values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {}
        indexes = {
            "min": 0,
            "p25": round((len(values) - 1) * 0.25),
            "median": round((len(values) - 1) * 0.50),
            "p75": round((len(values) - 1) * 0.75),
            "p90": round((len(values) - 1) * 0.90),
            "max": len(values) - 1,
        }
        return {name: values[index] for name, index in indexes.items()}

    baseline_tp = int(EXPECTED_BASELINE_METRICS["true_positive"])
    baseline_fp = int(EXPECTED_BASELINE_METRICS["false_positive"])
    baseline_fn = int(EXPECTED_BASELINE_METRICS["false_negative"])
    minimum_repairs = next(
        repairs
        for repairs in range(baseline_fp + 1)
        if (2.0 * baseline_tp) / (2.0 * baseline_tp + (baseline_fp - repairs) + baseline_fn) >= 0.9
    )
    return {
        "lane_counts": {lane: len(values) for lane, values in lanes.items()},
        "condition_counts": condition_counts,
        "inventory_weighted_jaccard_quantiles": {
            lane: quantiles(values) for lane, values in inventory_similarity.items()
        },
        "f1_0_9_materiality_reference": {
            "minimum_false_positive_repairs_if_no_true_positive_is_lost": minimum_repairs,
            "baseline_true_positive": baseline_tp,
            "baseline_false_positive": baseline_fp,
            "baseline_false_negative": baseline_fn,
            "interpretation": (
                "This is an arithmetic ceiling for materiality, not a scored candidate."
            ),
        },
    }


def _representative_cases(
    records: Sequence[Mapping[str, Any]],
    pairs_by_id: Mapping[int, MRPCPair],
) -> dict[str, Any]:
    false_positives = [
        record for record in records if record["diagnostic_class"] == "false_positive"
    ]

    def select(
        predicate: Any,
        *,
        order: Any = lambda record: record["pair_id"],
    ) -> Mapping[str, Any] | None:
        candidates = [record for record in false_positives if predicate(record)]
        return min(candidates, key=order) if candidates else None

    selections = {
        "highest_inventory_similarity_false_positive": select(
            lambda _record: True,
            order=lambda record: (-record["inventory"]["weighted_jaccard"], record["pair_id"]),
        ),
        "role_binding_conflict": select(
            lambda record: bool(record["conflicts"]["role_binding_conflicts"])
        ),
        "predicate_binding_difference": select(
            lambda record: bool(record["conflicts"]["predicate_binding_differences"])
        ),
        "temporal_property_conflict": select(
            lambda record: bool(record["conflicts"]["temporal_property_conflicts"])
        ),
        "scale_conflict": select(
            lambda record: (
                bool(record["conflicts"]["scale_property_conflicts"])
                or bool(record["conflicts"]["scale_argument_conflicts"])
            )
        ),
        "quantity_conflict": select(
            lambda record: bool(record["conflicts"]["quantity_argument_conflicts"])
        ),
        "other_event_property_conflict": select(
            lambda record: bool(record["conflicts"]["other_event_property_conflicts"])
        ),
    }
    result: dict[str, Any] = {}
    for name, record in selections.items():
        if record is None:
            continue
        pair = pairs_by_id[int(record["pair_id"])]
        result[name] = {
            "pair_id": int(pair.pair_id),
            "sentence1": pair.sentence1,
            "sentence2": pair.sentence2,
            "inventory": record["inventory"],
            "labeled_paths": record["labeled_paths"],
            "conflicts": record["conflicts"],
            "coverage": record["coverage"],
        }
    return result


def run_audit(
    *,
    mrpc_dir: Path,
    reveal_path: Path,
    graph_cache_path: Path,
) -> dict[str, Any]:
    dataset_evidence = _authenticate_dataset(mrpc_dir)
    pairs = load_mrpc_pairs(mrpc_dir.resolve())
    split = split_mrpc_by_sentence_component(pairs, seed=BLIND_SPLIT_SEED)
    held_pairs = [pair for pair in pairs if split[pair.pair_id] == "held_out"]
    if len(pairs) != 1725 or len(held_pairs) != 1107:
        raise ValueError("MRPC sentence-component split changed")
    trace_by_pair, comparator_evidence = _authenticate_reveal(reveal_path, held_pairs)
    positive_pairs = [
        pair for pair in held_pairs if bool(trace_by_pair[pair.pair_id]["resolver_prediction"])
    ]
    if len(positive_pairs) != 875:
        raise ValueError("frozen compact positive prediction count changed")
    graphs, evidence_by_text, graph_evidence = _load_graphs(
        graph_cache_path,
        (text for pair in positive_pairs for text in (pair.sentence1, pair.sentence2)),
    )
    records = [
        _audit_pair(
            pair,
            source=graphs[pair.sentence1],
            target=graphs[pair.sentence2],
            source_evidence=evidence_by_text[pair.sentence1],
            target_evidence=evidence_by_text[pair.sentence2],
            trace=trace_by_pair[pair.pair_id],
        )
        for pair in positive_pairs
    ]
    summary = _summarize(records)
    pairs_by_id = {pair.pair_id: pair for pair in positive_pairs}
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "evaluation_class": "post_hoc_frozen_mrpc_collision_diagnostic",
        "diagnostic_only": True,
        "predictions_changed": 0,
        "thresholds_fitted": 0,
        "hard_membership_authorized": False,
        "service_integration_authorized": False,
        "graph_intersection_operator_proposed": False,
        "evidence": {
            "dataset": dataset_evidence,
            "immutable_comparator": comparator_evidence,
            "pinned_graph_cache": graph_evidence,
            "source_identity": {
                "audit_script_sha256": _file_sha256(Path(__file__).resolve()),
                "focused_test_sha256": _file_sha256(
                    REPOSITORY_ROOT / "tests" / "unit" / "test_mrpc_topology_collision_audit.py"
                ),
                "controlling_inventory_implementation_sha256": _file_sha256(
                    REPOSITORY_ROOT
                    / "scripts"
                    / "research"
                    / "mrpc"
                    / "run_mrpc_semantic_relation_resolver_benchmark.py"
                ),
            },
        },
        "definitions": {
            "exact_inventory_collision": (
                "All 13 controlling graph inventory Counters and all five "
                "controlling graph count values are identical."
            ),
            "position_insensitive_inventory_sensitivity": (
                "The same inventory after parser positions, indexes, and "
                "binding provenance are removed from property-value records."
            ),
            "near_inventory_collision": {
                "weighted_tagged_inventory_jaccard_bands": list(NEAR_INVENTORY_BANDS),
                "maximum_absolute_delta_for_each_graph_count": (NEAR_MAX_GRAPH_COUNT_DELTA),
                "threshold_selection": "fixed descriptive sensitivity bands; not fitted",
            },
            "labeled_path": (
                "Exact canonical predicate identity -> exact role label -> exact "
                "semantic participant node identity, counted as a multiset."
            ),
            "localized_existing_conflict": (
                "A role-binding or coreference conflict emitted by "
                "compare_proposition_graphs; an event-property conflict after "
                "parser coordinates/provenance are removed; or an observed "
                "frame or operator-scope conflict."
            ),
            "strict_fail_closed_separation": (
                "A localized conflict only when both graph-local unresolved state "
                "and logical-operator coverage permit the existing strict proof."
            ),
        },
        "summary": summary,
        "representative_cases": _representative_cases(records, pairs_by_id),
        "pair_audits": records,
        "conclusion_contract": {
            "operator_should_be_preregistered_only_if_collision_bucket_is_material": True,
            "operator_implemented_or_scored": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrpc-dir", type=Path, default=DEFAULT_MRPC_DIR)
    parser.add_argument("--reveal", type=Path, default=DEFAULT_REVEAL)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_audit(
        mrpc_dir=args.mrpc_dir,
        reveal_path=args.reveal,
        graph_cache_path=args.graph_cache,
    )
    _atomic_json(args.report.resolve(), report)
    summary = report["summary"]
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "lane_counts": summary["lane_counts"],
                "condition_counts": summary["condition_counts"],
                "f1_0_9_materiality_reference": summary["f1_0_9_materiality_reference"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
