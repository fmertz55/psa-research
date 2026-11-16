#!/usr/bin/env python3
"""Probe narrow MRPC-compatible predicate pairs against the pinned v24 graphs.

This is a post-hoc development diagnostic.  It does not change the verifier,
and its results are not eligible for a blind performance claim.  Each candidate
is added in isolation to the existing MRPC-only contextual predicate inventory,
then evaluated on every held-out graph pair whose predicates could use it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
for path in (REPOSITORY_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import _atomic_json  # noqa: E402
from scripts.research.mrpc.run_mrpc_proposition_graph_benchmark import (
    BLIND_SPLIT_SEED,  # noqa: E402
)

from mcp_servers.mcp_lib import proposition_graph_rules as graph_rules  # noqa: E402
from mcp_servers.mcp_lib.proposition_graph import (  # noqa: E402
    PropositionGraph,
    build_proposition_graph,
)
from mcp_servers.mcp_lib.proposition_graph_rules import (  # noqa: E402
    LEGACY_EXPERIMENTAL_EXECUTION_POLICY,
    DeterministicGraphRuleSuite,
    GraphVerificationProfile,
    LexicalRelations,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    MRPCPair,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)

DEFAULT_TAXONOMY = (
    REPOSITORY_ROOT / "reports" / "mrpc_v24_unproven_positive_taxonomy_2026-08-01" / "taxonomy.json"
)
DEFAULT_BASELINE_REPORT = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_proposition_graph_occurrence_context_v24_2026-07-31"
    / "mrpc_proposition_graph_report.json"
)
DEFAULT_GRAPH_CACHE = (
    REPOSITORY_ROOT / "temp" / "mrpc_proposition_graph" / "spacy_oewn_evidence_v1.json"
)
DEFAULT_LEXICAL_CACHE = (
    REPOSITORY_ROOT / "temp" / "mrpc_proposition_graph" / "oewn_lexical_relations_v1.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_v24_unproven_positive_taxonomy_2026-08-01"
    / "lexical_candidate_probe.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _load_relations(payload: Mapping[str, Any]) -> LexicalRelations:
    return LexicalRelations(
        antonym_synset_pairs=frozenset(tuple(value) for value in payload["antonyms"]),
        derivational_synset_pairs=frozenset(tuple(value) for value in payload["derivations"]),
        hypernym_synset_pairs=frozenset(tuple(value) for value in payload["hypernyms"]),
        lemma_synset_pairs=frozenset(tuple(value) for value in payload["lemma_synsets"]),
        resource_identity=f"oewn:2024:{payload['cache_id']}",
    )


def _load_graphs(pairs: list[MRPCPair], payload: Mapping[str, Any]) -> dict[str, PropositionGraph]:
    sentences = payload.get("sentences") or {}
    texts = {text for pair in pairs for text in (pair.sentence1, pair.sentence2)}
    return {
        text: build_proposition_graph(sentences[sha256_text(text)]["evidence"]) for text in texts
    }


def _predicates(graph: PropositionGraph) -> frozenset[str]:
    return frozenset(
        str((proposition.get("predicate") or {}).get("lemma") or "").casefold()
        for proposition in graph.semantic_value.get("propositions", [])
        if (proposition.get("predicate") or {}).get("lemma")
    )


def _candidate_pairs(taxonomy: Mapping[str, Any]) -> tuple[set[int], list[tuple[str, str]]]:
    sole_ids: set[int] = set()
    candidates: set[tuple[str, str]] = set()
    for record in taxonomy.get("records") or []:
        if record.get("proposed_repair_family") != "rules.lexical_predicate_transformation":
            continue
        if record.get("implicated_repair_families") != ["rules.lexical_predicate_transformation"]:
            continue
        if record.get("secondary_blockers"):
            continue
        sole_ids.add(int(record["pair_id"]))
        predicates = record.get("implicated_predicates") or {}
        for source in predicates.get("source") or []:
            for target in predicates.get("target") or []:
                source_lemma = str(source).casefold()
                target_lemma = str(target).casefold()
                if source_lemma and target_lemma and source_lemma != target_lemma:
                    candidates.add(LexicalRelations.canonical_pair(source_lemma, target_lemma))
    existing = graph_rules._MRPC_COMPATIBLE_CONTEXTUAL_PREDICATE_PAIRS
    return sole_ids, sorted(candidates - set(existing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mrpc-dir",
        type=Path,
        default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split",
    )
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--lexical-cache", type=Path, default=DEFAULT_LEXICAL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    taxonomy = _load_json(args.taxonomy.resolve())
    baseline_report = _load_json(args.baseline_report.resolve())
    graph_cache = _load_json(args.graph_cache.resolve())
    lexical_cache = _load_json(args.lexical_cache.resolve())
    all_pairs = load_mrpc_pairs(args.mrpc_dir.resolve())
    split = split_mrpc_by_sentence_component(all_pairs, seed=BLIND_SPLIT_SEED)
    pairs = [pair for pair in all_pairs if split[pair.pair_id] == "held_out"]
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    graphs = _load_graphs(pairs, graph_cache)
    relations = _load_relations(lexical_cache)
    suite = DeterministicGraphRuleSuite(
        lexical_relations=relations,
        execution_policy=LEGACY_EXPERIMENTAL_EXECUTION_POLICY,
    )
    profile = GraphVerificationProfile.MRPC_COMPATIBLE
    base_inventory = graph_rules._MRPC_COMPATIBLE_CONTEXTUAL_PREDICATE_PAIRS
    live_market_close_inventory = graph_rules._MRPC_MARKET_CLOSE_DIRECTION_PREDICATE_PAIRS
    live_reporting_content_inventory = graph_rules._MRPC_REPORTING_CONTENT_PREDICATE_PAIRS
    # Reproduce v24 even when this diagnostic is run from a later worktree.
    graph_rules._MRPC_MARKET_CLOSE_DIRECTION_PREDICATE_PAIRS = frozenset()
    graph_rules._MRPC_REPORTING_CONTENT_PREDICATE_PAIRS = frozenset()
    baseline = {
        pair.pair_id: suite.verify(
            graphs[pair.sentence1],
            graphs[pair.sentence2],
            source_text=pair.sentence1,
            target_text=pair.sentence2,
            profile=profile,
        )
        for pair in pairs
    }
    baseline_traces = {
        int(trace["pair_id"]): trace
        for trace in baseline_report["held_out_pair_traces"]["profiles"][profile.value][
            "learned_gate"
        ]
    }
    drift = [
        pair_id
        for pair_id, decision in baseline.items()
        if decision.state.value != baseline_traces[pair_id]["graph_state"]
        or decision.reason != baseline_traces[pair_id]["graph_reason"]
    ]
    if drift:
        raise RuntimeError(f"live verifier does not reproduce v24 baseline: {drift[:10]}")

    sole_ids, candidates = _candidate_pairs(taxonomy)
    predicate_sets = {
        pair.pair_id: (_predicates(graphs[pair.sentence1]), _predicates(graphs[pair.sentence2]))
        for pair in pairs
    }
    gate_eligible = {
        pair_id
        for pair_id, trace in baseline_traces.items()
        if trace["psa_nominated"] and trace["native_gate_passed"]
    }
    rows: list[dict[str, Any]] = []
    try:
        for candidate in candidates:
            left, right = candidate
            affected = [
                pair_id
                for pair_id, (source, target) in predicate_sets.items()
                if (left in source and right in target) or (right in source and left in target)
            ]
            graph_rules._MRPC_COMPATIBLE_CONTEXTUAL_PREDICATE_PAIRS = base_inventory | {candidate}
            changed: dict[int, Any] = {}
            for pair_id in affected:
                pair = pair_by_id[pair_id]
                decision = suite.verify(
                    graphs[pair.sentence1],
                    graphs[pair.sentence2],
                    source_text=pair.sentence1,
                    target_text=pair.sentence2,
                    profile=profile,
                )
                if (
                    decision.state != baseline[pair_id].state
                    or decision.reason != baseline[pair_id].reason
                ):
                    changed[pair_id] = decision
            newly_accepted = {
                pair_id
                for pair_id, decision in changed.items()
                if decision.accepted and not baseline[pair_id].accepted
            }
            lost_acceptance = {
                pair_id
                for pair_id, decision in changed.items()
                if not decision.accepted and baseline[pair_id].accepted
            }
            positive = {pair_id for pair_id in newly_accepted if pair_by_id[pair_id].is_paraphrase}
            negative = newly_accepted - positive
            rows.append(
                {
                    "candidate": list(candidate),
                    "affected_held_out_pair_count": len(affected),
                    "changed_state_or_reason_pair_ids": sorted(changed),
                    "newly_accepted_positive_pair_ids": sorted(positive),
                    "newly_accepted_negative_pair_ids": sorted(negative),
                    "gate_eligible_positive_pair_ids": sorted(positive & gate_eligible),
                    "gate_eligible_negative_pair_ids": sorted(negative & gate_eligible),
                    "sole_lexical_positive_pair_ids": sorted(positive & sole_ids),
                    "lost_acceptance_pair_ids": sorted(lost_acceptance),
                }
            )
    finally:
        graph_rules._MRPC_COMPATIBLE_CONTEXTUAL_PREDICATE_PAIRS = base_inventory
        graph_rules._MRPC_MARKET_CLOSE_DIRECTION_PREDICATE_PAIRS = live_market_close_inventory
        graph_rules._MRPC_REPORTING_CONTENT_PREDICATE_PAIRS = live_reporting_content_inventory

    ranked = sorted(
        rows,
        key=lambda row: (
            len(row["gate_eligible_negative_pair_ids"]),
            -len(row["gate_eligible_positive_pair_ids"]),
            row["candidate"],
        ),
    )
    zero_exposure = [
        row
        for row in ranked
        if row["gate_eligible_positive_pair_ids"]
        and not row["gate_eligible_negative_pair_ids"]
        and not row["lost_acceptance_pair_ids"]
    ]
    output = {
        "schema_version": "aiidea_mrpc_v24_lexical_candidate_probe_v1",
        "evaluation_class": "post_hoc_iterative_development_diagnostic",
        "eligible_for_blind_performance_claim": False,
        "baseline_report": str(args.baseline_report.resolve()),
        "candidate_source": str(args.taxonomy.resolve()),
        "runtime_override": "v24_before_anchored_lexical_context",
        "held_out_pair_count": len(pairs),
        "sole_lexical_pair_count": len(sole_ids),
        "candidate_count": len(candidates),
        "zero_observed_gate_exposure_candidate_count": len(zero_exposure),
        "zero_observed_gate_exposure_candidates": zero_exposure,
        "candidates": ranked,
    }
    _atomic_json(args.output.resolve(), output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_count": len(candidates),
                "zero_observed_gate_exposure_candidate_count": len(zero_exposure),
                "top_candidates": zero_exposure[:20],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
