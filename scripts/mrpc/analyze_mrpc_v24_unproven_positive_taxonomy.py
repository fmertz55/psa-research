#!/usr/bin/env python3
"""Build the deterministic v24 MRPC unproven-positive failure census.

This script is analysis-only. It reads the frozen v24 report, its pinned graph
cache, and the MRPC sentence registry. It does not mutate extraction,
verification, matching, persistence, or service behavior.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = ROOT / "app"
for import_path in (ROOT, APP_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from mcp_servers.mcp_lib.paraphrase_evidence_extraction import (  # noqa: E402
    build_initial_evidence,
)
from mcp_servers.mcp_lib.proposition_graph import (  # noqa: E402
    PropositionGraph,
    build_proposition_graph,
)
from mcp_servers.mcp_lib.proposition_graph_rules import (  # noqa: E402
    LexicalRelations,
    _canonicalize_compatible_pair,
    _compatible_predicate_edge_cost,
    _compatible_predicate_match,
    _content_alignment_proof,
    _content_reference_topology,
    _maximum_predicate_matching,
    _predicate_identity,
)
from mcp_servers.tokenization_service.extractors import spacy_extractor  # noqa: E402
from mcp_servers.tokenization_service.extractors.spacy_extractor import (  # noqa: E402
    SpacyExtractor,
)

SOURCE_REPORT = (
    ROOT
    / "reports"
    / "mrpc_proposition_graph_occurrence_context_v24_2026-07-31"
    / "mrpc_proposition_graph_report.json"
)
GRAPH_CACHE = ROOT / "temp" / "mrpc_proposition_graph" / "spacy_oewn_evidence_v1.json"
LEXICAL_CACHE = ROOT / "temp" / "mrpc_proposition_graph" / "oewn_lexical_relations_v1.json"
MRPC_CORPUS = ROOT / "app" / "utils" / "msr_paraphrase_data.txt"
MRPC_TEST = ROOT / "app" / "utils" / "msr_paraphrase_test.txt"
OUTPUT_DIR = ROOT / "reports" / "mrpc_v24_unproven_positive_taxonomy_2026-08-01"

PROFILE = "mrpc_compatible_v1"
TRACE_BASE = "/held_out_pair_traces/profiles/mrpc_compatible_v1/learned_gate"
EXPECTED_CENSUS = 581
SAMPLE_SEED = "mrpc-v24-taxonomy-parser-probe-2026-08-01-v1"
SAMPLE_PER_PRIMARY = 4
NEGATIVE_CONTROLS_PER_PRIMARY = 2

ALLOWED_BLOCKERS = (
    "extraction_incompleteness",
    "role_or_quantity_grouping_loss",
    "topology_or_tie_selection_failure",
    "missing_licensed_transformation",
    "authenticated_context_required",
    "material_semantic_conflict_or_loose_label",
)

MATERIAL_CONFLICT_KINDS = frozenset(
    {
        "constraint_scope_conflict",
        "coordinated_predicative_complement_omission",
        "incompatible_supplemental_replacement",
        "location_conflict",
        "material_discourse_relation_omission",
        "modality_conflict",
        "polarity_conflict",
        "quantity_conflict",
        "referent_conflict",
        "reported_patient_or_order_conflict",
        "role_binding_conflict",
        "scalar_conflict",
        "time_conflict",
    }
)
GROUPING_CONFLICT_KINDS = frozenset(
    {
        "constraint_scope_conflict",
        "location_conflict",
        "quantity_conflict",
        "referent_conflict",
        "role_binding_conflict",
        "scalar_conflict",
    }
)
DEICTIC_TERMS = frozenset(
    {
        "currently",
        "earlier",
        "friday",
        "last",
        "later",
        "monday",
        "now",
        "recently",
        "saturday",
        "sunday",
        "thursday",
        "today",
        "tomorrow",
        "tuesday",
        "wednesday",
        "yesterday",
    }
)
PRONOUNS = frozenset(
    {
        "he",
        "her",
        "hers",
        "him",
        "his",
        "it",
        "its",
        "she",
        "their",
        "theirs",
        "them",
        "they",
        "we",
        "who",
    }
)
NEGATIVE_LEXEMES = frozenset(
    {
        "decline",
        "deny",
        "fail",
        "lack",
        "miss",
        "refuse",
        "reject",
        "unable",
        "without",
    }
)


FAMILY_DEFINITIONS: dict[str, dict[str, Any]] = {
    "parser.currency_approximation_retention": {
        "title": "Currency and approximation retention",
        "implementation_breadth": "medium",
        "proof_risk": "medium",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "implement_after_bounded_root_and_owner_edges",
    },
    "parser.appositive_group_composition": {
        "title": "Appositive and group composition",
        "implementation_breadth": "broad",
        "proof_risk": "medium",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "implement_after_possessor_edges",
    },
    "parser.possessor_owner_edges": {
        "title": "Possessor and owner edges",
        "implementation_breadth": "medium",
        "proof_risk": "medium",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "implement_early_with_exact_dependency_guards",
    },
    "parser.grouped_quantity_location_metric_tuples": {
        "title": "Grouped quantity, location, and metric tuples",
        "implementation_breadth": "broad",
        "proof_risk": "high",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "implement_incrementally_after_atomic_quantity_retention",
    },
    "parser.nested_participial_noun_modifier_events": {
        "title": "Nested participial and noun-modifier events",
        "implementation_breadth": "broad",
        "proof_risk": "high",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "defer_until_lower_risk_parser_primitives_are_frozen",
    },
    "parser.bounded_nonverbal_root_recovery": {
        "title": "Bounded non-verbal-root recovery",
        "implementation_breadth": "narrow",
        "proof_risk": "medium",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "do_not_implement_from_this_census_zero_positive_yield",
    },
    "parser.vp_ellipsis": {
        "title": "VP ellipsis",
        "implementation_breadth": "medium",
        "proof_risk": "high",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "defer_pending_authenticated_antecedent_design",
    },
    "parser.reporting_content_attachment": {
        "title": "Reporting and content attachment",
        "implementation_breadth": "broad",
        "proof_risk": "high",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "implement_only_after_occurrence_identity_tests_expand",
    },
    "parser.operator_scope_attachment": {
        "title": "Operator and polarity scope attachment",
        "implementation_breadth": "medium",
        "proof_risk": "high",
        "semantic_primitive": "reusable_if_occurrence_bound",
        "implementation_recommendation": "diagnose_before_any_negation_rule_expansion",
    },
    "matcher.topology_aware_tie_search": {
        "title": "Bounded topology-aware predicate search",
        "implementation_breadth": "medium",
        "proof_risk": "high",
        "semantic_primitive": "reusable",
        "implementation_recommendation": "do_not_implement_from_this_census_zero_equal_cost_cases",
    },
    "rules.lexical_predicate_transformation": {
        "title": "Lexical predicate and argument transformation",
        "implementation_breadth": "broad",
        "proof_risk": "high",
        "semantic_primitive": "reusable_if_versioned_and_anchored",
        "implementation_recommendation": "add_only_narrow_rules_with_negative_controls",
    },
    "rules.negation_scalar_normalization": {
        "title": "Negation and scalar normalization",
        "implementation_breadth": "medium",
        "proof_risk": "high",
        "semantic_primitive": "reusable_if_scope_preserving",
        "implementation_recommendation": "split_safe_scope_normalization_from_unsafe_approximation",
    },
    "rules.nominalization_voice_reporting": {
        "title": "Nominalization, voice, and reporting transformations",
        "implementation_breadth": "broad",
        "proof_risk": "high",
        "semantic_primitive": "reusable_if_role_complete",
        "implementation_recommendation": "defer_broad_rules_use_typed_templates_only",
    },
    "rules.modality_equivalence": {
        "title": "Modality equivalence",
        "implementation_breadth": "broad",
        "proof_risk": "unacceptable_for_general_equivalence",
        "semantic_primitive": "not_a_safe_equivalence_primitive",
        "implementation_recommendation": "do_not_implement_as_equivalence",
    },
    "context.authenticated_coreference_alias": {
        "title": "Authenticated coreference and alias resolution",
        "implementation_breadth": "broad",
        "proof_risk": "high",
        "semantic_primitive": "reusable_only_with_authenticated_context",
        "implementation_recommendation": "defer_mrpc_has_no_authenticated_context_field",
    },
    "no_repair.material_conflict_or_loose_label": {
        "title": "Material conflict or loose MRPC label",
        "implementation_breadth": "none",
        "proof_risk": "unacceptable",
        "semantic_primitive": "not_a_repair_family",
        "implementation_recommendation": "do_not_implement_preserve_fail_closed_behavior",
    },
}


@dataclass(frozen=True)
class PairText:
    sentence1: str
    sentence2: str
    label: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_pairs() -> dict[int, PairText]:
    sentence_by_id: dict[str, str] = {}
    with MRPC_CORPUS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            sentence_by_id[str(row["Sentence ID"])] = str(row["String"])
    result: dict[int, PairText] = {}
    with MRPC_TEST.open(encoding="utf-8-sig", newline="") as handle:
        for pair_id, row in enumerate(
            csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), start=1
        ):
            result[pair_id] = PairText(
                sentence1=sentence_by_id[str(row["#1 ID"])],
                sentence2=sentence_by_id[str(row["#2 ID"])],
                label=str(row["Quality"]) == "1",
            )
    return result


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def numeric_atoms(text: str) -> set[str]:
    return {
        normalized.lstrip("0") or "0"
        for raw_value in re.findall(r"(?<![a-z])\d[\d,.]*(?:\.\d+)?", text.casefold())
        if (normalized := raw_value.replace(",", ""))
    }


def nested_records(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from nested_records(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from nested_records(child)


def evidence_predicates(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(value) for value in evidence.get("propositions") or []]


def predicate_lemmas(evidence: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(value.get("predicate_lemma") or "").casefold()
        for value in evidence_predicates(evidence)
        if value.get("predicate_lemma")
    )


def dependency_roles(evidence: Mapping[str, Any]) -> set[str]:
    return {
        str(value.get("dependency_role") or "").casefold()
        for value in evidence.get("object_occurrences") or []
        if value.get("dependency_role")
    }


def dependency_attachment_terms(
    evidence: Mapping[str, Any],
    dependency: str,
) -> set[str]:
    result: set[str] = set()
    for value in evidence.get("object_occurrences") or []:
        if str(value.get("dependency_role") or "").casefold() != dependency:
            continue
        result.update(
            words(
                " ".join(
                    str(value.get(field) or "")
                    for field in (
                        "surface",
                        "lemma",
                        "normalized_entity",
                        "head_lemma",
                        "head_text",
                    )
                )
            )
        )
    return result


def clause_roles(evidence: Mapping[str, Any]) -> set[str]:
    return {
        str(value.get("clause_role") or "").casefold()
        for value in evidence_predicates(evidence)
        if value.get("clause_role")
    }


def entity_surfaces(evidence: Mapping[str, Any], labels: set[str]) -> set[str]:
    return {
        str(value.get("normalized_entity") or value.get("surface") or "").casefold()
        for value in evidence.get("object_occurrences") or []
        if str(value.get("entity_label") or "").upper() in labels
        and (value.get("normalized_entity") or value.get("surface"))
    }


def evidence_pronouns(evidence: Mapping[str, Any]) -> set[str]:
    return {
        str(value.get("lemma") or "").casefold()
        for value in evidence.get("object_occurrences") or []
        if str(value.get("occurrence_type") or "").casefold() == "pronoun"
        and str(value.get("lemma") or "").casefold() in PRONOUNS
    }


def relevant_quantities(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in nested_records(evidence.get("propositions") or []):
        node = value.get("node") if isinstance(value.get("node"), Mapping) else value
        if node.get("normalized_quantity") or str(node.get("entity_label") or "").upper() in {
            "CARDINAL",
            "MONEY",
            "ORDINAL",
            "PERCENT",
            "QUANTITY",
        }:
            result[canonical_json(node)] = dict(node)
    return [result[key] for key in sorted(result)]


def relevant_content_edges(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for proposition in evidence_predicates(evidence):
        for argument in proposition.get("arguments") or []:
            if str(argument.get("role") or "").casefold() != "content":
                continue
            result.append(
                {
                    "parent_predicate": proposition.get("predicate_lemma"),
                    "parent_position": proposition.get("predicate_token_position"),
                    "content": argument,
                }
            )
    return result


def unresolved_prefixes(unresolved: Sequence[str]) -> set[str]:
    return {str(value).split(":", 1)[0] for value in unresolved}


def conflict_nodes(conflicts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for conflict in conflicts:
        for value in nested_records(conflict):
            node = value.get("node")
            if isinstance(node, Mapping):
                result.append(node)
    return result


def conflict_numeric_overlap(conflicts: Sequence[Mapping[str, Any]]) -> bool:
    for conflict in conflicts:
        source_atoms = numeric_atoms(canonical_json(conflict.get("source")))
        target_atoms = numeric_atoms(canonical_json(conflict.get("target")))
        if source_atoms and target_atoms and source_atoms & target_atoms:
            return True
    return False


def conflict_lexical_overlap(conflicts: Sequence[Mapping[str, Any]]) -> bool:
    for conflict in conflicts:
        source_atoms = words(canonical_json(conflict.get("source")))
        target_atoms = words(canonical_json(conflict.get("target")))
        material_source = source_atoms - {"kind", "node", "role", "entity", "lemma"}
        material_target = target_atoms - {"kind", "node", "role", "entity", "lemma"}
        if material_source and material_target:
            overlap = material_source & material_target
            if (
                len(overlap) >= 2
                or len(overlap) / min(len(material_source), len(material_target)) >= 0.5
            ):
                return True
    return False


def relative_time_context(
    sentence1: str,
    sentence2: str,
    conflict_kinds: set[str],
) -> bool:
    if "time_conflict" not in conflict_kinds:
        return False
    left = words(sentence1) & DEICTIC_TERMS
    right = words(sentence2) & DEICTIC_TERMS
    return bool(left != right and (left or right))


def negative_semantic_markers(text: str, evidence: Mapping[str, Any]) -> set[str]:
    result = words(text) & NEGATIVE_LEXEMES
    if re.search(
        r"\bno\b(?!\s*\.\s*\d)|\b(?:not|never|neither|nor)\b|n't\b",
        text.casefold(),
    ):
        result.add("syntactic_negation")
    for proposition in evidence_predicates(evidence):
        lemma = str(proposition.get("predicate_lemma") or "").casefold()
        if lemma in NEGATIVE_LEXEMES:
            result.add(lemma)
    return result


def stable_rank(namespace: str, pair_id: int) -> str:
    return sha256_text(f"{SAMPLE_SEED}|{namespace}|{pair_id}")


def primary_family_for(
    primary: str,
    implicated: Sequence[str],
) -> str:
    family_priority = {
        "extraction_incompleteness": (
            "parser.bounded_nonverbal_root_recovery",
            "parser.vp_ellipsis",
            "parser.operator_scope_attachment",
            "parser.reporting_content_attachment",
            "parser.nested_participial_noun_modifier_events",
            "parser.grouped_quantity_location_metric_tuples",
        ),
        "role_or_quantity_grouping_loss": (
            "parser.possessor_owner_edges",
            "parser.currency_approximation_retention",
            "parser.grouped_quantity_location_metric_tuples",
            "parser.appositive_group_composition",
        ),
        "topology_or_tie_selection_failure": ("matcher.topology_aware_tie_search",),
        "missing_licensed_transformation": (
            "rules.negation_scalar_normalization",
            "rules.nominalization_voice_reporting",
            "rules.lexical_predicate_transformation",
        ),
        "authenticated_context_required": ("context.authenticated_coreference_alias",),
        "material_semantic_conflict_or_loose_label": (
            "no_repair.material_conflict_or_loose_label",
            "rules.modality_equivalence",
        ),
    }
    for family in family_priority[primary]:
        if family in implicated:
            return family
    fallback = {
        "extraction_incompleteness": "parser.nested_participial_noun_modifier_events",
        "role_or_quantity_grouping_loss": "parser.grouped_quantity_location_metric_tuples",
        "topology_or_tie_selection_failure": "matcher.topology_aware_tie_search",
        "missing_licensed_transformation": "rules.lexical_predicate_transformation",
        "authenticated_context_required": "context.authenticated_coreference_alias",
        "material_semantic_conflict_or_loose_label": "no_repair.material_conflict_or_loose_label",
    }
    return fallback[primary]


def parser_family_features(
    sentence1: str,
    sentence2: str,
    evidence1: Mapping[str, Any],
    evidence2: Mapping[str, Any],
    conflict_kinds: set[str],
    conflicts: Sequence[Mapping[str, Any]],
    unresolved: Sequence[str],
) -> list[str]:
    families: list[str] = []
    combined = f"{sentence1} {sentence2}".casefold()
    propositions1 = evidence_predicates(evidence1)
    propositions2 = evidence_predicates(evidence2)
    quantities1, quantities2 = relevant_quantities(evidence1), relevant_quantities(evidence2)
    conflict_terms = words(canonical_json(conflicts)) - {
        "entity",
        "explanation",
        "kind",
        "lemma",
        "node",
        "role",
        "source",
        "target",
    }
    shared_numbers = numeric_atoms(sentence1) & numeric_atoms(sentence2)

    if (
        (
            "$" in combined
            or " us$" in combined
            or any(
                value in combined for value in ("million", "billion", "percent", "per cent", "yen")
            )
        )
        and conflict_kinds & {"constraint_scope_conflict", "quantity_conflict", "scalar_conflict"}
        and shared_numbers
    ):
        families.append("parser.currency_approximation_retention")
    appos_terms = dependency_attachment_terms(evidence1, "appos") | dependency_attachment_terms(
        evidence2, "appos"
    )
    if appos_terms & conflict_terms and conflict_kinds & {
        "referent_conflict",
        "role_binding_conflict",
        "location_conflict",
    }:
        families.append("parser.appositive_group_composition")
    poss_terms = dependency_attachment_terms(evidence1, "poss") | dependency_attachment_terms(
        evidence2, "poss"
    )
    if poss_terms & conflict_terms and conflict_kinds & {
        "referent_conflict",
        "role_binding_conflict",
        "quantity_conflict",
        "location_conflict",
    }:
        families.append("parser.possessor_owner_edges")
    tuple_conflict = bool(
        conflict_kinds
        & {
            "constraint_scope_conflict",
            "location_conflict",
            "quantity_conflict",
            "role_binding_conflict",
            "time_conflict",
        }
        or ("referent_conflict" in conflict_kinds and poss_terms & conflict_terms)
    )
    if tuple_conflict and (
        (len(quantities1) >= 2 and len(quantities2) >= 2 and shared_numbers)
        or (
            len(entity_surfaces(evidence1, {"FAC", "GPE", "LOC"})) >= 2
            and len(entity_surfaces(evidence2, {"FAC", "GPE", "LOC"})) >= 2
            and bool(
                words(" ".join(entity_surfaces(evidence1, {"FAC", "GPE", "LOC"})))
                & words(" ".join(entity_surfaces(evidence2, {"FAC", "GPE", "LOC"})))
            )
        )
        or any("quantities_scope_unresolved" in value for value in unresolved)
    ):
        families.append("parser.grouped_quantity_location_metric_tuples")
    unmatched_source = {
        value.split(":", 1)[1]
        for value in unresolved
        if value.startswith("unmatched_source_proposition:")
    }
    unmatched_target = {
        value.split(":", 1)[1]
        for value in unresolved
        if value.startswith("unmatched_target_proposition:")
    }
    source_words = words(sentence1)
    target_words = words(sentence2)
    nested_unmatched = any(
        str(value.get("predicate_lemma") or "").casefold() in unmatched_source
        and str(value.get("clause_role") or "").casefold() in {"acl", "relcl"}
        and str(value.get("predicate_lemma") or "").casefold() in target_words
        for value in propositions1
    ) or any(
        str(value.get("predicate_lemma") or "").casefold() in unmatched_target
        and str(value.get("clause_role") or "").casefold() in {"acl", "relcl"}
        and str(value.get("predicate_lemma") or "").casefold() in source_words
        for value in propositions2
    )
    if nested_unmatched:
        families.append("parser.nested_participial_noun_modifier_events")
    if not propositions1 or not propositions2:
        families.append("parser.bounded_nonverbal_root_recovery")
    if re.search(
        r"\b(?:do|does|did|doing|done)\s+(?:so|it|that)\b|"
        r"\b(?:has|have|had|is|are|was|were|will|would|could|should|can|may|might|must)\s+not\s*[.,;:]",
        combined,
    ):
        families.append("parser.vp_ellipsis")
    if (
        any(value.startswith("content_reference_") for value in unresolved)
        or "proposition_reference_topology_conflict" in conflict_kinds
    ) and (evidence1.get("attribution") or evidence2.get("attribution")):
        families.append("parser.reporting_content_attachment")
    syntactic_negative1 = bool(
        re.search(
            r"\bno\b(?!\s*\.\s*\d)|\b(?:not|never|neither|nor)\b|n't\b",
            sentence1.casefold(),
        )
    )
    syntactic_negative2 = bool(
        re.search(
            r"\bno\b(?!\s*\.\s*\d)|\b(?:not|never|neither|nor)\b|n't\b",
            sentence2.casefold(),
        )
    )
    if (
        "polarity_conflict" in conflict_kinds
        and not syntactic_negative1
        and not syntactic_negative2
    ):
        families.append("parser.operator_scope_attachment")
    return list(dict.fromkeys(families))


def transformation_families(
    sentence1: str,
    sentence2: str,
    evidence1: Mapping[str, Any],
    evidence2: Mapping[str, Any],
    conflict_kinds: set[str],
    unresolved: Sequence[str],
) -> list[str]:
    families: list[str] = []
    negative1 = negative_semantic_markers(sentence1, evidence1)
    negative2 = negative_semantic_markers(sentence2, evidence2)
    if (
        "polarity_conflict" in conflict_kinds
        and negative1
        and negative2
        and ("syntactic_negation" in negative1 | negative2 or negative1 != negative2)
    ) or "scalar_conflict" in conflict_kinds:
        families.append("rules.negation_scalar_normalization")
    if "modality_conflict" in conflict_kinds:
        families.append("rules.modality_equivalence")
    prefixes = unresolved_prefixes(unresolved)
    if prefixes & {
        "unmatched_source_proposition",
        "unmatched_target_proposition",
        "unmatched_source_argument",
        "unmatched_target_argument",
    }:
        families.append("rules.lexical_predicate_transformation")
    if (
        evidence1.get("attribution")
        or evidence2.get("attribution")
        or {"acl", "advcl", "relcl"} & (clause_roles(evidence1) | clause_roles(evidence2))
    ) and prefixes & {"unmatched_source_proposition", "unmatched_target_proposition"}:
        families.append("rules.nominalization_voice_reporting")
    return list(dict.fromkeys(families))


def classify_pair(
    trace: Mapping[str, Any],
    pair: PairText,
    evidence1: Mapping[str, Any],
    evidence2: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    proof = trace.get("proof") or {}
    unresolved = [str(value) for value in proof.get("unresolved") or []]
    conflicts = [dict(value) for value in proof.get("conflicts") or []]
    conflict_kinds = {str(value) for value in trace.get("conflict_kinds") or []}
    parsers = parser_family_features(
        pair.sentence1,
        pair.sentence2,
        evidence1,
        evidence2,
        conflict_kinds,
        conflicts,
        unresolved,
    )
    transformations = transformation_families(
        pair.sentence1,
        pair.sentence2,
        evidence1,
        evidence2,
        conflict_kinds,
        unresolved,
    )

    names1 = entity_surfaces(evidence1, {"ORG", "PERSON"})
    names2 = entity_surfaces(evidence2, {"ORG", "PERSON"})
    person_names1 = entity_surfaces(evidence1, {"PERSON"})
    person_names2 = entity_surfaces(evidence2, {"PERSON"})
    pronouns1 = evidence_pronouns(evidence1)
    pronouns2 = evidence_pronouns(evidence2)
    deictic = relative_time_context(pair.sentence1, pair.sentence2, conflict_kinds)
    source_starts_pronoun = bool(
        re.match(
            r"^[\s\"']*(?:he|her|him|his|it|its|she|their|them|they|we)\b",
            pair.sentence1.casefold(),
        )
    )
    target_starts_pronoun = bool(
        re.match(
            r"^[\s\"']*(?:he|her|him|his|it|its|she|their|them|they|we)\b",
            pair.sentence2.casefold(),
        )
    )
    pronoun_name_gap = bool(
        conflict_kinds & {"referent_conflict", "role_binding_conflict"}
        and (
            (source_starts_pronoun and not target_starts_pronoun and names2)
            or (target_starts_pronoun and not source_starts_pronoun and names1)
            or (pronouns1 - pronouns2 and not names1 and names2)
            or (pronouns2 - pronouns1 and not names2 and names1)
        )
    )
    alias_gap = bool(
        "referent_conflict" in conflict_kinds
        and person_names1
        and person_names2
        and not any(words(left) & words(right) for left in person_names1 for right in person_names2)
    )
    context_signal = deictic or pronoun_name_gap or alias_gap

    lexical_overlap = conflict_lexical_overlap(conflicts)
    numeric_overlap = conflict_numeric_overlap(conflicts)
    same_sentence_numbers = bool(numeric_atoms(pair.sentence1) & numeric_atoms(pair.sentence2))
    grouping_signal = bool(
        set(parsers)
        & {
            "parser.appositive_group_composition",
            "parser.currency_approximation_retention",
            "parser.grouped_quantity_location_metric_tuples",
            "parser.possessor_owner_edges",
        }
    )
    extraction_signal = bool(
        trace.get("graph_reason") == "incomplete_proposition_graph"
        or any(
            family
            in {
                "parser.bounded_nonverbal_root_recovery",
                "parser.operator_scope_attachment",
                "parser.vp_ellipsis",
                "parser.reporting_content_attachment",
                "parser.nested_participial_noun_modifier_events",
            }
            for family in parsers
        )
    )
    negative1 = negative_semantic_markers(pair.sentence1, evidence1)
    negative2 = negative_semantic_markers(pair.sentence2, evidence2)
    polarity_transform = bool(
        "polarity_conflict" in conflict_kinds
        and negative1
        and negative2
        and ("syntactic_negation" in negative1 | negative2 or negative1 != negative2)
    )
    operator_scope_signal = "parser.operator_scope_attachment" in parsers
    transformation_signal = bool(
        transformations
        and (
            trace.get("graph_reason") == "compatibility_not_established"
            or polarity_transform
            or (conflict_kinds <= {"polarity_conflict", "scalar_conflict"} and bool(conflict_kinds))
        )
    )

    semantic_reasons: list[str] = []
    if "modality_conflict" in conflict_kinds:
        semantic_reasons.append("modal operator classes differ")
    if "time_conflict" in conflict_kinds and not deictic and not lexical_overlap:
        semantic_reasons.append("explicit time values differ")
    if "quantity_conflict" in conflict_kinds and not numeric_overlap:
        semantic_reasons.append("explicit quantity values differ")
    if "location_conflict" in conflict_kinds and not lexical_overlap and not context_signal:
        semantic_reasons.append("explicit locations differ")
    if "referent_conflict" in conflict_kinds and not lexical_overlap and not context_signal:
        semantic_reasons.append("explicit referents differ")
    if conflict_kinds & {
        "coordinated_predicative_complement_omission",
        "incompatible_supplemental_replacement",
        "material_discourse_relation_omission",
        "reported_patient_or_order_conflict",
    }:
        semantic_reasons.append("material clause or list content differs")
    if "constraint_scope_conflict" in conflict_kinds and not (
        numeric_overlap or same_sentence_numbers
    ):
        semantic_reasons.append("one-sided hard constraint is explicit")
    if (
        "polarity_conflict" in conflict_kinds
        and not polarity_transform
        and not operator_scope_signal
    ):
        semantic_reasons.append("polarity differs without a scope-preserving normalization")
    semantic_signal = bool(semantic_reasons)

    equally_ranked_alternate = bool(topology.get("equal_cost_topology_valid_alternatives"))
    if trace.get("graph_reason") == "incomplete_proposition_graph":
        primary = "extraction_incompleteness"
    elif equally_ranked_alternate:
        primary = "topology_or_tie_selection_failure"
    elif context_signal and not semantic_signal:
        primary = "authenticated_context_required"
    elif grouping_signal and not semantic_signal:
        primary = "role_or_quantity_grouping_loss"
    elif extraction_signal and not semantic_signal:
        primary = "extraction_incompleteness"
    elif transformation_signal and not semantic_signal:
        primary = "missing_licensed_transformation"
    elif trace.get("graph_state") == "do_not_merge" and semantic_signal:
        primary = "material_semantic_conflict_or_loose_label"
    elif trace.get("graph_reason") == "compatibility_not_established":
        primary = "missing_licensed_transformation"
    elif transformation_signal:
        primary = "missing_licensed_transformation"
    else:
        primary = "material_semantic_conflict_or_loose_label"

    secondary: list[str] = []
    signals = {
        "extraction_incompleteness": extraction_signal,
        "role_or_quantity_grouping_loss": grouping_signal,
        "topology_or_tie_selection_failure": equally_ranked_alternate,
        "missing_licensed_transformation": transformation_signal,
        "authenticated_context_required": context_signal,
        "material_semantic_conflict_or_loose_label": semantic_signal,
    }
    for blocker in ALLOWED_BLOCKERS:
        if blocker != primary and signals[blocker]:
            secondary.append(blocker)

    implicated_families = [*parsers, *transformations]
    if equally_ranked_alternate:
        implicated_families.append("matcher.topology_aware_tie_search")
    if context_signal:
        implicated_families.append("context.authenticated_coreference_alias")
    if semantic_signal:
        implicated_families.append("no_repair.material_conflict_or_loose_label")
    implicated_families = list(dict.fromkeys(implicated_families))
    proposed_family = primary_family_for(primary, implicated_families)
    if proposed_family not in implicated_families:
        implicated_families.insert(0, proposed_family)

    blocker_evidence: dict[str, list[dict[str, Any]]] = {}
    for blocker in [primary, *secondary]:
        values: list[dict[str, Any]] = []
        if blocker == "extraction_incompleteness":
            values.append(
                {
                    "type": "graph_and_parser_shape",
                    "graph_reason": trace.get("graph_reason"),
                    "unresolved": unresolved,
                    "parser_families": parsers,
                    "source_predicate_count": len(evidence_predicates(evidence1)),
                    "target_predicate_count": len(evidence_predicates(evidence2)),
                    "source_clause_roles": sorted(clause_roles(evidence1)),
                    "target_clause_roles": sorted(clause_roles(evidence2)),
                }
            )
        elif blocker == "role_or_quantity_grouping_loss":
            values.append(
                {
                    "type": "shared_surface_conflicting_graph_binding",
                    "parser_families": parsers,
                    "conflicts": [
                        value for value in conflicts if value.get("kind") in GROUPING_CONFLICT_KINDS
                    ],
                    "conflict_lexical_overlap": lexical_overlap,
                    "conflict_numeric_overlap": numeric_overlap,
                    "shared_sentence_numbers": sorted(
                        numeric_atoms(pair.sentence1) & numeric_atoms(pair.sentence2)
                    ),
                }
            )
        elif blocker == "topology_or_tie_selection_failure":
            values.append(
                {
                    "type": "demonstrated_equal_cost_occurrence_consistent_mapping",
                    "selected_mapping": topology.get("selected_mapping"),
                    "alternatives": topology.get("equal_cost_topology_valid_alternatives"),
                }
            )
        elif blocker == "missing_licensed_transformation":
            values.append(
                {
                    "type": "complete_but_unlicensed_graph_difference",
                    "graph_reason": trace.get("graph_reason"),
                    "unresolved": unresolved,
                    "candidate_rule_families": transformations,
                    "negative_semantic_markers": {
                        "source": sorted(negative1),
                        "target": sorted(negative2),
                    },
                }
            )
        elif blocker == "authenticated_context_required":
            values.append(
                {
                    "type": "sentence_local_reference_gap",
                    "deictic_time_gap": deictic,
                    "pronoun_name_gap": pronoun_name_gap,
                    "alias_gap": alias_gap,
                    "source_pronouns": sorted(pronouns1),
                    "target_pronouns": sorted(pronouns2),
                    "source_named_entities": sorted(names1),
                    "target_named_entities": sorted(names2),
                }
            )
        elif blocker == "material_semantic_conflict_or_loose_label":
            values.append(
                {
                    "type": "explicit_material_difference",
                    "reasons": semantic_reasons
                    or [
                        "current proof contains a material conflict not assigned to a safer repair"
                    ],
                    "conflicts": conflicts,
                }
            )
        blocker_evidence[blocker] = values

    if (
        (
            primary == "extraction_incompleteness"
            and trace.get("graph_reason") == "incomplete_proposition_graph"
        )
        or (
            primary == "material_semantic_conflict_or_loose_label"
            and len(semantic_reasons) >= 2
            and not secondary
        )
        or (primary == "authenticated_context_required" and deictic and not secondary)
    ):
        confidence = "high"
    elif primary in {
        "role_or_quantity_grouping_loss",
        "authenticated_context_required",
        "material_semantic_conflict_or_loose_label",
    }:
        confidence = "medium"
    else:
        confidence = "low" if len(secondary) >= 2 else "medium"

    implicated_predicates = {
        "source": predicate_lemmas(evidence1),
        "target": predicate_lemmas(evidence2),
        "unmatched": sorted(
            value.split(":", 1)[1]
            for value in unresolved
            if value.startswith("unmatched_source_proposition:")
            or value.startswith("unmatched_target_proposition:")
        ),
    }
    implicated_roles = sorted(
        {
            str(value.get("role"))
            for conflict in conflicts
            for value in nested_records(conflict)
            if value.get("role")
        }
    )
    context_dependent = primary == "authenticated_context_required" or (
        "authenticated_context_required" in secondary
    )
    return {
        "primary_blocker": primary,
        "secondary_blockers": secondary,
        "blocker_evidence": blocker_evidence,
        "proposed_repair_family": proposed_family,
        "implicated_repair_families": implicated_families,
        "repair_scope": "context-dependent" if context_dependent else "sentence-local",
        "classification_confidence": confidence,
        "manual_review_required": confidence != "high" or bool(secondary),
        "implicated_predicates": implicated_predicates,
        "implicated_roles": implicated_roles,
        "relevant_quantities": {
            "source": relevant_quantities(evidence1),
            "target": relevant_quantities(evidence2),
        },
        "relevant_content_edges": {
            "source": relevant_content_edges(evidence1),
            "target": relevant_content_edges(evidence2),
        },
        "feature_trace": {
            "parser_families": parsers,
            "transformation_families": transformations,
            "context_signal": context_signal,
            "grouping_signal": grouping_signal,
            "extraction_signal": extraction_signal,
            "semantic_signal": semantic_signal,
            "semantic_reasons": semantic_reasons,
        },
    }


def lexical_relations() -> LexicalRelations:
    payload = json.loads(LEXICAL_CACHE.read_text(encoding="utf-8"))
    return LexicalRelations(
        antonym_synset_pairs=frozenset(tuple(value) for value in payload["antonyms"]),
        derivational_synset_pairs=frozenset(tuple(value) for value in payload["derivations"]),
        hypernym_synset_pairs=frozenset(tuple(value) for value in payload["hypernyms"]),
        lemma_synset_pairs=frozenset(tuple(value) for value in payload["lemma_synsets"]),
        resource_identity=f"oewn:2024:{payload['cache_id']}",
    )


def mapping_description(
    assignments: Sequence[tuple[int, int]],
    source_propositions: Sequence[Mapping[str, Any]],
    target_propositions: Sequence[Mapping[str, Any]],
    source_origins: Sequence[Sequence[int]],
    target_origins: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    return [
        {
            "source_index": source_index,
            "source_predicate": str(
                _predicate_identity(source_propositions[source_index]).get("lemma") or ""
            ),
            "source_origins": list(source_origins[source_index]),
            "target_index": target_index,
            "target_predicate": str(
                _predicate_identity(target_propositions[target_index]).get("lemma") or ""
            ),
            "target_origins": list(target_origins[target_index]),
        }
        for source_index, target_index in assignments
    ]


def analyze_topology(
    graph1: PropositionGraph,
    graph2: PropositionGraph,
    resources: LexicalRelations,
    conflict_kinds: set[str],
) -> dict[str, Any]:
    if "proposition_reference_topology_conflict" not in conflict_kinds:
        return {
            "status": "not_applicable",
            "equal_cost_topology_valid_alternatives": [],
            "maximum_cardinality_topology_valid_alternatives": [],
        }

    projection = _canonicalize_compatible_pair(graph1, graph2, resources)
    source = list(projection.source_propositions)
    target = list(projection.target_propositions)
    edges: dict[tuple[int, int], tuple[Any, tuple[int, ...]]] = {}
    for source_index, source_proposition in enumerate(source):
        for target_index, target_proposition in enumerate(target):
            predicate_match = _compatible_predicate_match(
                source_proposition,
                target_proposition,
                resources,
            )
            if predicate_match is None:
                continue
            edges[(source_index, target_index)] = (
                predicate_match,
                _compatible_predicate_edge_cost(
                    source_proposition,
                    target_proposition,
                    predicate_match,
                    resources,
                    source_index=source_index,
                    target_index=target_index,
                ),
            )

    candidates: list[tuple[int, tuple[int, ...], tuple[tuple[int, int], ...]]] = []
    zero_cost = (0,) * 13

    def visit(
        source_index: int,
        used_targets: frozenset[int],
        assignments: tuple[tuple[int, int], ...],
        cost: tuple[int, ...],
    ) -> None:
        if source_index >= len(source):
            candidates.append((len(assignments), cost, assignments))
            return
        visit(source_index + 1, used_targets, assignments, cost)
        for target_index in range(len(target)):
            edge = edges.get((source_index, target_index))
            if edge is None or target_index in used_targets:
                continue
            edge_cost = edge[1]
            visit(
                source_index + 1,
                used_targets | {target_index},
                (*assignments, (source_index, target_index)),
                tuple(left + right for left, right in zip(cost, edge_cost, strict=True)),
            )

    visit(0, frozenset(), (), zero_cost)
    maximum_cardinality = max(value[0] for value in candidates)
    maximum_mappings = [value for value in candidates if value[0] == maximum_cardinality]
    optimal_cost = min(value[1] for value in maximum_mappings)

    selected, _edge_matches = _maximum_predicate_matching(
        source,
        target,
        resources,
        compatible=True,
    )
    selected_assignments = tuple(
        sorted(
            (source_index, target_index)
            for source_index, (target_index, _predicate_match) in selected.items()
        )
    )
    source_topology = _content_reference_topology(graph1)
    target_topology = _content_reference_topology(graph2)

    def topology_result(
        assignments: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        mapping = {
            source_index: (target_index, edges[(source_index, target_index)][0])
            for source_index, target_index in assignments
        }
        proof = _content_alignment_proof(
            source_topology,
            target_topology,
            mapping,
            projection.source_origins,
            projection.target_origins,
        )
        return {
            "exact": not proof.conflicts and not proof.unresolved,
            "conflicts": [
                {
                    "kind": conflict.kind,
                    "source": conflict.source,
                    "target": conflict.target,
                    "explanation": conflict.explanation,
                }
                for conflict in proof.conflicts
            ],
            "unresolved": list(proof.unresolved),
        }

    equal_cost_valid: list[dict[str, Any]] = []
    maximum_valid: list[dict[str, Any]] = []
    for _count, cost, assignments in sorted(
        maximum_mappings,
        key=lambda value: (value[1], value[2]),
    ):
        topology_result_value = topology_result(assignments)
        if not topology_result_value["exact"] or assignments == selected_assignments:
            continue
        value = {
            "cost_vector": list(cost),
            "mapping": mapping_description(
                assignments,
                source,
                target,
                projection.source_origins,
                projection.target_origins,
            ),
            "topology": topology_result_value,
        }
        maximum_valid.append(value)
        if cost == optimal_cost:
            equal_cost_valid.append(value)

    return {
        "status": "analyzed",
        "candidate_state_count": len(candidates),
        "maximum_cardinality_mapping_count": len(maximum_mappings),
        "maximum_cardinality": maximum_cardinality,
        "optimal_cost_vector": list(optimal_cost),
        "selected_mapping": mapping_description(
            selected_assignments,
            source,
            target,
            projection.source_origins,
            projection.target_origins,
        ),
        "selected_topology": topology_result(selected_assignments),
        "equal_cost_topology_valid_alternatives": equal_cost_valid,
        "maximum_cardinality_topology_valid_alternatives": maximum_valid,
        "interpretation": (
            "taxonomy category 3 requires at least one exact equal-cost alternative; "
            "worse-cost alternatives are diagnostic only"
        ),
    }


def make_taxonomy_record(
    trace: Mapping[str, Any],
    trace_index: int,
    pair: PairText,
    evidence1: Mapping[str, Any],
    evidence2: Mapping[str, Any],
    topology: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    classified = classify_pair(trace, pair, evidence1, evidence2, topology)
    proof = trace.get("proof") or {}
    return {
        "pair_id": int(trace["pair_id"]),
        "sentence1": pair.sentence1,
        "sentence2": pair.sentence2,
        "label": bool(trace["label"]),
        "similarity": float(trace["native_similarity"]),
        "gate": {
            "profile": PROFILE,
            "threshold": threshold,
            "psa_nominated": bool(trace["psa_nominated"]),
            "native_gate_passed": bool(trace["native_gate_passed"]),
            "predicted_paraphrase": bool(trace["predicted_paraphrase"]),
        },
        "graph_state": trace["graph_state"],
        "graph_reason": trace["graph_reason"],
        "matched_propositions": int(proof.get("matched_propositions") or 0),
        "unresolved_conditions": list(proof.get("unresolved") or []),
        "conflict_kinds": list(trace.get("conflict_kinds") or []),
        "conflicts": list(proof.get("conflicts") or []),
        "rule_type_ids": list(trace.get("rule_type_ids") or []),
        "topology_analysis": topology,
        "analogous_negative_controls": [],
        "estimated_negative_exposure": None,
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "trace_location": f"{TRACE_BASE}/{trace_index}",
        **classified,
    }


def classify_negative_controls(
    traces: Sequence[Mapping[str, Any]],
    pairs: Mapping[int, PairText],
    evidence_by_hash: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for trace_index, trace in enumerate(traces):
        if not (
            trace.get("label") is False
            and trace.get("psa_nominated") is True
            and trace.get("native_gate_passed") is True
        ):
            continue
        pair_id = int(trace["pair_id"])
        pair = pairs[pair_id]
        evidence1 = evidence_by_hash[sha256_text(pair.sentence1)]["evidence"]
        evidence2 = evidence_by_hash[sha256_text(pair.sentence2)]["evidence"]
        classified = classify_pair(
            trace,
            pair,
            evidence1,
            evidence2,
            {
                "status": "negative_control_not_topology_enumerated",
                "equal_cost_topology_valid_alternatives": [],
            },
        )
        result.append(
            {
                "pair_id": pair_id,
                "sentence1": pair.sentence1,
                "sentence2": pair.sentence2,
                "similarity": float(trace["native_similarity"]),
                "graph_state": trace["graph_state"],
                "graph_reason": trace["graph_reason"],
                "conflict_kinds": list(trace.get("conflict_kinds") or []),
                "trace_location": f"{TRACE_BASE}/{trace_index}",
                "structural_families": classified["implicated_repair_families"],
                "structural_primary_blocker": classified["primary_blocker"],
            }
        )
    assert len(result) == 288, len(result)
    return result


def attach_negative_controls(
    records: list[dict[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
) -> None:
    negatives_by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for negative in negatives:
        for family in negative["structural_families"]:
            negatives_by_family[family].append(negative)
    for family in negatives_by_family:
        negatives_by_family[family].sort(
            key=lambda value: (-float(value["similarity"]), int(value["pair_id"]))
        )
    for record in records:
        family = record["proposed_repair_family"]
        controls = negatives_by_family.get(family, [])
        record["estimated_negative_exposure"] = len(controls)
        record["analogous_negative_controls"] = [
            {
                "pair_id": value["pair_id"],
                "similarity": value["similarity"],
                "graph_state": value["graph_state"],
                "conflict_kinds": value["conflict_kinds"],
                "trace_location": value["trace_location"],
            }
            for value in controls[:5]
        ]


def repair_yield(
    records: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[list[Any]]]:
    pairs_by_family: dict[str, set[int]] = defaultdict(set)
    sole_by_family: dict[str, set[int]] = defaultdict(set)
    primary_by_family: dict[str, set[int]] = defaultdict(set)
    for record in records:
        pair_id = int(record["pair_id"])
        primary_family = str(record["proposed_repair_family"])
        primary_by_family[primary_family].add(pair_id)
        implicated = list(record["implicated_repair_families"])
        for family in implicated:
            pairs_by_family[family].add(pair_id)
        if not record["secondary_blockers"] and len(implicated) == 1:
            sole_by_family[primary_family].add(pair_id)

    negative_by_family: dict[str, set[int]] = defaultdict(set)
    negative_rows_by_id = {int(value["pair_id"]): value for value in negatives}
    for negative in negatives:
        for family in negative["structural_families"]:
            negative_by_family[family].add(int(negative["pair_id"]))

    families = sorted(FAMILY_DEFINITIONS)
    rows: list[dict[str, Any]] = []
    for family in families:
        upper = pairs_by_family[family]
        overlaps = {
            other: len(upper & pairs_by_family[other])
            for other in families
            if other != family and upper & pairs_by_family[other]
        }
        controls = sorted(
            (negative_rows_by_id[pair_id] for pair_id in negative_by_family[family]),
            key=lambda value: (-float(value["similarity"]), int(value["pair_id"])),
        )
        rows.append(
            {
                "repair_family": family,
                **FAMILY_DEFINITIONS[family],
                "repair_character": (
                    "reusable_semantic_primitive"
                    if "reusable" in FAMILY_DEFINITIONS[family]["semantic_primitive"]
                    else "unsafe_or_not_a_repair_primitive"
                ),
                "mrpc_specific_template": False,
                "sole_blocker_pair_count": len(sole_by_family[family]),
                "sole_blocker_pair_ids": sorted(sole_by_family[family]),
                "primary_pair_count": len(primary_by_family[family]),
                "primary_pair_ids": sorted(primary_by_family[family]),
                "all_implicated_pair_count": len(upper),
                "all_implicated_pair_ids": sorted(upper),
                "overlap_by_family": dict(
                    sorted(overlaps.items(), key=lambda value: (-value[1], value[0]))
                ),
                "candidate_negative_exposure_count": len(negative_by_family[family]),
                "candidate_negative_exposure_rate_of_288": round(
                    len(negative_by_family[family]) / 288, 6
                ),
                "analogous_negative_controls": [
                    {
                        "pair_id": value["pair_id"],
                        "similarity": value["similarity"],
                        "graph_state": value["graph_state"],
                        "conflict_kinds": value["conflict_kinds"],
                        "sentence1": value["sentence1"],
                        "sentence2": value["sentence2"],
                        "trace_location": value["trace_location"],
                    }
                    for value in controls[:10]
                ],
            }
        )

    matrix: list[list[Any]] = [["repair_family", *families]]
    for family in families:
        matrix.append(
            [
                family,
                *(len(pairs_by_family[family] & pairs_by_family[other]) for other in families),
            ]
        )

    unsafe = {
        "context.authenticated_coreference_alias",
        "matcher.topology_aware_tie_search",
        "no_repair.material_conflict_or_loose_label",
        "rules.modality_equivalence",
    }
    remaining = {family for family in families if family not in unsafe}
    covered: set[int] = set()
    cumulative: list[dict[str, Any]] = []
    while remaining:
        family = min(
            remaining,
            key=lambda value: (
                -len(pairs_by_family[value] - covered),
                value,
            ),
        )
        new_pairs = pairs_by_family[family] - covered
        covered.update(new_pairs)
        cumulative.append(
            {
                "order": len(cumulative) + 1,
                "repair_family": family,
                "new_upper_bound_pairs": len(new_pairs),
                "new_pair_ids": sorted(new_pairs),
                "cumulative_upper_bound_pairs": len(covered),
                "cumulative_fraction_of_581": round(len(covered) / EXPECTED_CENSUS, 6),
                "cumulative_reaches_452": len(covered) >= 452,
            }
        )
        remaining.remove(family)

    topology_records = [
        record for record in records if record["topology_analysis"].get("status") == "analyzed"
    ]
    equal_cost_topology = [
        int(record["pair_id"])
        for record in topology_records
        if record["topology_analysis"].get("equal_cost_topology_valid_alternatives")
    ]
    broader_topology = [
        int(record["pair_id"])
        for record in topology_records
        if record["topology_analysis"].get("maximum_cardinality_topology_valid_alternatives")
    ]
    safe_sentence_local_families = set(families) - unsafe
    safe_sole_ids = set().union(
        *(sole_by_family[family] for family in safe_sentence_local_families)
    )
    safe_upper_ids = set().union(
        *(pairs_by_family[family] for family in safe_sentence_local_families)
    )
    payload = {
        "schema_version": "aiidea_mrpc_v24_repair_family_yield_v1",
        "source_report": str(SOURCE_REPORT.relative_to(ROOT)),
        "yield_semantics": {
            "sole_blocker_pair_count": (
                "strict lower-bound stratum: family is proposed primary, no secondary "
                "blocker, and no other repair family is implicated; this is not a "
                "guaranteed accepted-pair count"
            ),
            "all_implicated_pair_count": (
                "diagnostic upper-bound stratum: family appears anywhere in the "
                "deterministic evidence trace; multi-family pairs are counted once per family"
            ),
            "candidate_negative_exposure_count": (
                "gate-passing held-out negatives with the same structural feature; it is "
                "exposure, not an estimated false-positive count"
            ),
        },
        "families": rows,
        "cumulative_yield_ordering": cumulative,
        "shortest_defensible_route_assessment": {
            "additional_true_positive_requirement": 452,
            "sentence_local_non_preexcluded_sole_blocker_pair_count": len(safe_sole_ids),
            "sentence_local_non_preexcluded_sole_blocker_pair_ids": sorted(safe_sole_ids),
            "sentence_local_non_preexcluded_diagnostic_upper_bound_count": len(safe_upper_ids),
            "defensible_452_pair_route_demonstrated": len(safe_sole_ids) >= 452,
            "assessment": (
                "no: the broad-family union can exceed 452 only as an overlapping "
                "post-hoc upper bound; the strict sole-blocker strata do not"
            ),
        },
        "topology_recovery_estimate": {
            "topology_conflict_pair_count": len(topology_records),
            "defensible_equal_cost_recoverable_pair_count": len(equal_cost_topology),
            "defensible_equal_cost_recoverable_pair_ids": sorted(equal_cost_topology),
            "exploratory_worse_cost_maximum_cardinality_candidate_count": len(broader_topology),
            "exploratory_worse_cost_maximum_cardinality_candidate_ids": sorted(broader_topology),
            "estimated_safe_recovery": 0,
            "reason": (
                "no topology-conflict trace has an equal-cost exact alternative; the "
                "three broader candidates are not category-3 cases and retain unsafe "
                "argument, rule, or semantic concerns"
            ),
        },
    }
    return payload, matrix


def select_probe_sample(
    records: Sequence[Mapping[str, Any]],
    negatives: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    records_by_primary: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_primary[str(record["primary_blocker"])].append(record)
    for primary in ALLOWED_BLOCKERS:
        values = sorted(
            records_by_primary.get(primary, []),
            key=lambda value: stable_rank(primary, int(value["pair_id"])),
        )[:SAMPLE_PER_PRIMARY]
        for value in values:
            sample.append(
                {
                    "pair_id": int(value["pair_id"]),
                    "sample_role": "unproven_positive",
                    "stratum": primary,
                    "proposed_repair_family": value["proposed_repair_family"],
                    "sentence1": value["sentence1"],
                    "sentence2": value["sentence2"],
                    "trace_location": value["trace_location"],
                }
            )

    already_positive = {(value["sample_role"], int(value["pair_id"])) for value in sample}
    topology_diagnostic = [
        value
        for value in sorted(
            (value for value in records if value["topology_analysis"].get("status") == "analyzed"),
            key=lambda value: stable_rank("topology_diagnostic", int(value["pair_id"])),
        )
        if ("unproven_positive", int(value["pair_id"])) not in already_positive
    ][:3]
    for value in topology_diagnostic:
        identity = ("unproven_positive", int(value["pair_id"]))
        if identity in already_positive:
            continue
        sample.append(
            {
                "pair_id": int(value["pair_id"]),
                "sample_role": "unproven_positive",
                "stratum": "diagnostic_topology_conflict_not_category_3",
                "proposed_repair_family": value["proposed_repair_family"],
                "sentence1": value["sentence1"],
                "sentence2": value["sentence2"],
                "trace_location": value["trace_location"],
            }
        )

    used_negative_ids: set[int] = set()
    for primary in ALLOWED_BLOCKERS:
        positive_values = records_by_primary.get(primary, [])
        if not positive_values:
            continue
        family_targets = {
            family for value in positive_values for family in value["implicated_repair_families"]
        }
        conflict_targets = {kind for value in positive_values for kind in value["conflict_kinds"]}
        ranked_negatives = sorted(
            (value for value in negatives if int(value["pair_id"]) not in used_negative_ids),
            key=lambda value: (
                -len(family_targets & set(value["structural_families"])),
                -len(conflict_targets & set(value["conflict_kinds"])),
                stable_rank(f"negative:{primary}", int(value["pair_id"])),
            ),
        )[:NEGATIVE_CONTROLS_PER_PRIMARY]
        for value in ranked_negatives:
            used_negative_ids.add(int(value["pair_id"]))
            overlap = sorted(family_targets & set(value["structural_families"]))
            sample.append(
                {
                    "pair_id": int(value["pair_id"]),
                    "sample_role": "gate_passing_negative_control",
                    "stratum": f"matched_to:{primary}",
                    "proposed_repair_family": overlap[0] if overlap else None,
                    "structural_family_overlap": overlap,
                    "sentence1": value["sentence1"],
                    "sentence2": value["sentence2"],
                    "trace_location": value["trace_location"],
                }
            )
    return sample


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_tree(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for candidate in sorted(value for value in path.rglob("*") if value.is_file()):
        if "__pycache__" in candidate.parts or candidate.suffix == ".pyc":
            continue
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = candidate.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        file_count += 1
        byte_count += len(data)
    return digest.hexdigest(), file_count, byte_count


def governing_predicate(token: Any) -> Any | None:
    current = token
    visited: set[int] = set()
    for _ in range(12):
        if current.i in visited:
            return None
        visited.add(current.i)
        if current.pos_ in {"AUX", "VERB"} and current.dep_ not in {
            "aux",
            "auxpass",
            "cop",
        }:
            return current
        if current.head.i == current.i:
            return None
        current = current.head
    return None


def argument_signatures(evidence: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for proposition in evidence_predicates(evidence):
        predicate = str(proposition.get("predicate_lemma") or "").casefold()
        for argument in proposition.get("arguments") or []:
            result.add(
                canonical_json(
                    {
                        "predicate": predicate,
                        "role": argument.get("role"),
                        "lemma": argument.get("lemma"),
                        "normalized_entity": argument.get("normalized_entity"),
                        "normalized_quantity": argument.get("normalized_quantity"),
                    }
                )
            )
    return result


def parse_once(
    text: str,
    extractor: SpacyExtractor,
    cached_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    doc = spacy_extractor.nlp(text)
    tokens = [
        {
            "i": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head_i": token.head.i,
            "head_text": token.head.text,
            "entity_label": token.ent_type_,
            "morph": str(token.morph),
        }
        for token in doc
    ]
    entities = [
        {
            "text": entity.text,
            "label": entity.label_,
            "start": entity.start,
            "end": entity.end,
            "root_i": entity.root.i,
            "root_dep": entity.root.dep_,
            "head_i": entity.root.head.i,
            "head_lemma": entity.root.head.lemma_,
            "governing_predicate": (
                governing_predicate(entity.root).lemma_
                if governing_predicate(entity.root) is not None
                else None
            ),
        }
        for entity in doc.ents
    ]
    occurrences = extractor.extract_objects(text)
    projection = build_initial_evidence(
        sentence_id=f"parser-probe:{sha256_text(text)}",
        text=text,
        occurrences=occurrences,
        promoted_object_ids=(),
        extractor_identity={
            "tokenizer": "aiidea_sentence_tokenizer@v1",
            "spacy_model": extractor.model_identity,
            "fact_extractor": (
                "paraphrase_evidence_extraction@v18-occurrence-bound-operators-apostrophe-safe"
            ),
            "sense_resolver": "not_run_parser_diagnostic",
        },
        extraction_complete=True,
    )
    fresh_evidence = projection.evidence
    fresh_graph = build_proposition_graph(fresh_evidence)
    predicate_tokens = [
        {
            "i": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "dep": token.dep_,
            "pos": token.pos_,
        }
        for token in doc
        if token.pos_ in {"AUX", "VERB"} and token.dep_ not in {"aux", "auxpass", "cop"}
    ]
    role_tokens = [
        {
            "i": token.i,
            "text": token.text,
            "dep": token.dep_,
            "head_i": token.head.i,
            "head_lemma": token.head.lemma_,
        }
        for token in doc
        if token.dep_ in {"nsubj", "nsubjpass", "nsubj:pass", "csubj", "dobj", "obj", "iobj"}
    ]
    poss_appos = [
        {
            "i": token.i,
            "text": token.text,
            "dep": token.dep_,
            "head_i": token.head.i,
            "head_text": token.head.text,
        }
        for token in doc
        if token.dep_ in {"poss", "appos"}
    ]
    modifier_attachments = [
        {
            "i": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head_i": token.head.i,
            "head_text": token.head.text,
        }
        for token in doc
        if token.dep_ in {"acl", "relcl", "amod", "compound"}
    ]
    content_attachments = [
        {
            "i": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "dep": token.dep_,
            "head_i": token.head.i,
            "head_lemma": token.head.lemma_,
        }
        for token in doc
        if token.dep_ in {"ccomp", "xcomp", "parataxis"}
    ]
    quantity_location_associations = [
        entity
        for entity in entities
        if entity["label"]
        in {"CARDINAL", "MONEY", "ORDINAL", "PERCENT", "QUANTITY", "FAC", "GPE", "LOC"}
    ]
    fresh_predicates = set(predicate_lemmas(fresh_evidence))
    cached_predicates = set(predicate_lemmas(cached_evidence))
    fresh_arguments = argument_signatures(fresh_evidence)
    cached_arguments = argument_signatures(cached_evidence)
    return {
        "runtime_seconds": time.perf_counter() - started,
        "tokens": tokens,
        "entities": entities,
        "sentences": [sentence.text for sentence in doc.sents],
        "predicate_recovery": {
            "parser_predicate_tokens": predicate_tokens,
            "extracted_predicate_lemmas": sorted(fresh_predicates),
            "cached_v24_predicate_lemmas": sorted(cached_predicates),
            "new_against_v24_cache": sorted(fresh_predicates - cached_predicates),
            "missing_against_v24_cache": sorted(cached_predicates - fresh_predicates),
        },
        "subject_object_role_recovery": role_tokens,
        "possessive_appositive_attachments": poss_appos,
        "participial_modifier_attachments": modifier_attachments,
        "quantity_location_associations": quantity_location_associations,
        "proposition_content_topology": {
            "parser_content_attachments": content_attachments,
            "extracted_content_edges": relevant_content_edges(fresh_evidence),
        },
        "new_erroneous_or_changed_attachments_against_v24_cache": {
            "new_predicates": sorted(fresh_predicates - cached_predicates),
            "new_argument_signatures": sorted(fresh_arguments - cached_arguments),
            "interpretation": (
                "drift against the pinned v24 small-model cache, not manual gold error"
            ),
        },
        "unresolved_conditions": list(fresh_graph.unresolved),
        "cached_v24_unresolved_conditions": list(
            build_proposition_graph(cached_evidence).unresolved
        ),
        "fresh_graph_sha256": fresh_graph.sha256,
        "fresh_semantic_sha256": fresh_graph.semantic_sha256,
    }


def parser_probe(
    sample: Sequence[Mapping[str, Any]],
    evidence_by_hash: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    extractor = SpacyExtractor()
    if not extractor.available or spacy_extractor.nlp is None:
        raise RuntimeError("pinned en_core_web_sm is unavailable in the analysis environment")

    import en_core_web_sm
    import spacy

    model_package_path = Path(en_core_web_sm.__file__).resolve().parent
    model_tree_sha, model_file_count, model_byte_count = sha256_tree(model_package_path)
    unique_texts = sorted(
        {str(value[key]) for value in sample for key in ("sentence1", "sentence2")}
    )
    memory_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sentence_results: list[dict[str, Any]] = []
    all_started = time.perf_counter()
    for text in unique_texts:
        cached_evidence = evidence_by_hash[sha256_text(text)]["evidence"]
        runs = [parse_once(text, extractor, cached_evidence) for _ in range(2)]
        comparable_runs = [
            {key: value for key, value in run.items() if key != "runtime_seconds"} for run in runs
        ]
        sentence_results.append(
            {
                "text_sha256": sha256_text(text),
                "text": text,
                "repeat_count": 2,
                "repeated_run_deterministic": (
                    canonical_json(comparable_runs[0]) == canonical_json(comparable_runs[1])
                ),
                "runtime_seconds_by_run": [run["runtime_seconds"] for run in runs],
                "parser_output": comparable_runs[0],
            }
        )
    total_runtime = time.perf_counter() - all_started
    memory_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    total_predicates = sum(
        len(value["parser_output"]["predicate_recovery"]["parser_predicate_tokens"])
        for value in sentence_results
    )
    total_roles = sum(
        len(value["parser_output"]["subject_object_role_recovery"]) for value in sentence_results
    )
    total_poss_appos = sum(
        len(value["parser_output"]["possessive_appositive_attachments"])
        for value in sentence_results
    )
    total_modifiers = sum(
        len(value["parser_output"]["participial_modifier_attachments"])
        for value in sentence_results
    )
    total_quantity_locations = sum(
        len(value["parser_output"]["quantity_location_associations"]) for value in sentence_results
    )
    total_content = sum(
        len(value["parser_output"]["proposition_content_topology"]["parser_content_attachments"])
        for value in sentence_results
    )
    new_predicates = sum(
        len(value["parser_output"]["predicate_recovery"]["new_against_v24_cache"])
        for value in sentence_results
    )
    new_arguments = sum(
        len(
            value["parser_output"]["new_erroneous_or_changed_attachments_against_v24_cache"][
                "new_argument_signatures"
            ]
        )
        for value in sentence_results
    )
    unresolved_changed = sum(
        value["parser_output"]["unresolved_conditions"]
        != value["parser_output"]["cached_v24_unresolved_conditions"]
        for value in sentence_results
    )

    trf_modules = {
        "en_core_web_trf": importlib.util.find_spec("en_core_web_trf") is not None,
        "spacy_transformers": importlib.util.find_spec("spacy_transformers") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
    }
    transformer_available = all(trf_modules.values())
    assert not transformer_available, "probe script expects the documented blocked environment"
    return {
        "schema_version": "aiidea_mrpc_v24_parser_probe_v1",
        "status": "blocked_transformer_dependency_missing_small_model_complete",
        "diagnostic_only": True,
        "production_evidence_changed": False,
        "sample_selection": {
            "seed": SAMPLE_SEED,
            "algorithm": (
                "SHA-256 rank within each observed primary blocker, four positives per "
                "stratum, plus three topology-conflict diagnostics and two structurally "
                "matched gate-passing negatives per observed primary blocker"
            ),
            "pair_count": len(sample),
            "unique_sentence_count": len(unique_texts),
            "sample": list(sample),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "spacy_version": spacy.__version__,
            "small_model": {
                "identity": extractor.model_identity,
                "module_version": getattr(en_core_web_sm, "__version__", None),
                "distribution_version": distribution_version("en-core-web-sm"),
                "metadata": dict(spacy_extractor.nlp.meta),
                "pipeline": list(spacy_extractor.nlp.pipe_names),
                "package_path": str(model_package_path),
                "artifact_tree_sha256_excluding_pycache": model_tree_sha,
                "artifact_file_count": model_file_count,
                "artifact_byte_count": model_byte_count,
            },
            "transformer_model": {
                "requested_identity": "en_core_web_trf@3.8.0",
                "official_release_url": (
                    "https://github.com/explosion/spacy-models/releases/tag/en_core_web_trf-3.8.0"
                ),
                "official_wheel_url": (
                    "https://github.com/explosion/spacy-models/releases/download/"
                    "en_core_web_trf-3.8.0/"
                    "en_core_web_trf-3.8.0-py3-none-any.whl"
                ),
                "official_wheel_sha256": (
                    "272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5"
                ),
                "approval_required_install_command": (
                    "temp/wordnet-venv/bin/python -m pip install "
                    "'https://github.com/explosion/spacy-models/releases/download/"
                    "en_core_web_trf-3.8.0/"
                    "en_core_web_trf-3.8.0-py3-none-any.whl#sha256="
                    "272a31e9d8530d1e075351d30a462d7e80e31da23574f1b274e200f3fff35bf5'"
                ),
                "available": transformer_available,
                "module_availability": trf_modules,
                "distribution_versions": {
                    "en-core-web-trf": distribution_version("en-core-web-trf"),
                    "spacy-transformers": distribution_version("spacy-transformers"),
                    "torch": distribution_version("torch"),
                    "transformers": distribution_version("transformers"),
                },
                "exact_blocker": (
                    "en_core_web_trf and its runtime dependencies spacy-transformers, "
                    "torch, and transformers are not installed"
                ),
                "substitution_used": False,
            },
            "execution_settings": {
                "device": "cpu",
                "batching": "one sentence per parser call",
                "repeat_count": 2,
                "random_seed_dependency": "none; deterministic sample uses SHA-256",
                "small_model_pipeline_disabled_components": [],
            },
        },
        "small_model_measurements": {
            "sentence_count": len(sentence_results),
            "predicate_recovery_count": total_predicates,
            "subject_object_role_recovery_count": total_roles,
            "possessive_appositive_attachment_count": total_poss_appos,
            "participial_modifier_attachment_count": total_modifiers,
            "quantity_location_association_count": total_quantity_locations,
            "proposition_content_attachment_count": total_content,
            "new_predicates_against_v24_cache_count": new_predicates,
            "new_argument_signatures_against_v24_cache_count": new_arguments,
            "unresolved_condition_changed_sentence_count": unresolved_changed,
            "repeated_run_deterministic_sentence_count": sum(
                bool(value["repeated_run_deterministic"]) for value in sentence_results
            ),
            "repeated_run_nondeterministic_sentence_count": sum(
                not bool(value["repeated_run_deterministic"]) for value in sentence_results
            ),
            "total_runtime_seconds_two_runs_plus_extraction": total_runtime,
            "mean_runtime_seconds_per_parser_extraction_run": (
                sum(
                    runtime
                    for value in sentence_results
                    for runtime in value["runtime_seconds_by_run"]
                )
                / (2 * len(sentence_results))
                if sentence_results
                else 0.0
            ),
            "ru_maxrss_before": memory_before,
            "ru_maxrss_after": memory_after,
            "ru_maxrss_delta": memory_after - memory_before,
            "ru_maxrss_units": "platform_native_getrusage_units",
            "memory_measurement_caveat": (
                "the small model is imported before the probe and ru_maxrss is a "
                "process high-water mark, so the delta is not isolated model-load memory"
            ),
            "error_measurement_caveat": (
                "new predicate and attachment counts measure drift from the v24 cache, "
                "not correctness against manually annotated parser gold"
            ),
        },
        "transformer_measurements": None,
        "comparison": {
            "status": "not_run_missing_dependency",
            "fields_unavailable": [
                "transformer predicate recovery delta",
                "transformer role recovery delta",
                "transformer possessive and appositive attachment delta",
                "transformer participial and modifier attachment delta",
                "transformer quantity and location association delta",
                "transformer proposition and content topology delta",
                "new transformer erroneous predicates or attachments",
                "transformer unresolved-condition changes",
                "transformer repeated-run determinism",
                "transformer runtime and memory cost",
            ],
        },
        "sentence_results": sentence_results,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def top_overlaps(matrix: Sequence[Sequence[Any]], limit: int = 12) -> list[tuple[str, str, int]]:
    headers = list(matrix[0][1:])
    result: list[tuple[str, str, int]] = []
    for row_index, row in enumerate(matrix[1:]):
        for column_index in range(row_index + 1, len(headers)):
            count = int(row[column_index + 1])
            if count:
                result.append((str(row[0]), headers[column_index], count))
    return sorted(result, key=lambda value: (-value[2], value[0], value[1]))[:limit]


def taxonomy_summary_markdown(
    taxonomy: Mapping[str, Any],
    repair: Mapping[str, Any],
    matrix: Sequence[Sequence[Any]],
    probe: Mapping[str, Any],
) -> str:
    counts = taxonomy["aggregate_counts"]["primary_blockers"]
    count_rows = [
        (blocker, counts.get(blocker, 0), f"{counts.get(blocker, 0) / EXPECTED_CENSUS:.1%}")
        for blocker in ALLOWED_BLOCKERS
    ]
    yield_rows = [
        (
            value["repair_family"],
            value["sole_blocker_pair_count"],
            value["all_implicated_pair_count"],
            value["candidate_negative_exposure_count"],
            value["implementation_breadth"],
            value["proof_risk"],
        )
        for value in sorted(
            repair["families"],
            key=lambda item: (-item["sole_blocker_pair_count"], item["repair_family"]),
        )
    ]
    overlap_rows = top_overlaps(matrix)
    topology = repair["topology_recovery_estimate"]
    measurements = probe["small_model_measurements"]
    parser_order = [
        "parser.possessor_owner_edges",
        "parser.appositive_group_composition",
        "parser.currency_approximation_retention",
        "parser.grouped_quantity_location_metric_tuples",
        "parser.operator_scope_attachment",
        "parser.reporting_content_attachment",
        "parser.nested_participial_noun_modifier_events",
        "parser.bounded_nonverbal_root_recovery",
        "parser.vp_ellipsis",
    ]
    by_family = {value["repair_family"]: value for value in repair["families"]}
    order_lines = []
    for family in parser_order:
        value = by_family.get(family)
        if value is None:
            continue
        order_lines.append(
            f"{len(order_lines) + 1}. `{family}`: "
            f"{value['sole_blocker_pair_count']} sole-blocker, "
            f"{value['all_implicated_pair_count']} implicated, "
            f"{value['candidate_negative_exposure_count']} structurally analogous negatives."
        )

    route = repair["shortest_defensible_route_assessment"]
    lexical_sole = by_family["rules.lexical_predicate_transformation"]["sole_blocker_pair_count"]
    route_text = (
        f"No 452-pair route is demonstrated. The strict sentence-local, non-precluded "
        f"sole-blocker strata contain only "
        f"{route['sentence_local_non_preexcluded_sole_blocker_pair_count']} pairs. The "
        f"corresponding overlapping diagnostic upper bound is "
        f"{route['sentence_local_non_preexcluded_diagnostic_upper_bound_count']}, but it "
        "is dominated by broad lexical and nominalization families with high negative "
        "exposure and is not a yield prediction. The shortest defensible route toward "
        f"the target is to split the {lexical_sole} sole lexical-predicate cases into "
        "narrow typed "
        "rules, then take the appositive, possessor, quantity, reporting, and operator "
        "sole strata in the stated parser order, freezing after each negative-control "
        "and sealed-evaluation pass. Reaching 452 would still require new authenticated "
        "evidence or reusable primitives not demonstrated by this census; broad "
        "allowlisting cannot close the gap safely."
    )

    return f"""# MRPC v24 unproven-positive taxonomy

## Outcome

The exact compatible recall-floor-gate census contains **581 unique held-out positive pairs**. Every record has one primary blocker from the required vocabulary, trace-preserving evidence, and a stable pointer into the v24 report. This is deterministic post-hoc development analysis, not blind performance evidence.

{markdown_table(("Primary blocker", "Pairs", "Percent"), count_rows)}

Category 3 is deliberately empty. All ten current topology-conflict traces have a unique optimal 13-component matcher cost vector; none has the required equally ranked exact alternative.

## Repair-family yield and negative exposure

The lower column is the strict sole-family stratum. The upper column is every pair implicating the family and therefore overlaps other rows. Exposure counts are structurally analogous gate-passing negatives, not predicted false positives.

{markdown_table(("Repair family", "Sole", "Implicated", "Negative exposure", "Breadth", "Proof risk"), yield_rows)}

## Largest overlaps

{markdown_table(("Family A", "Family B", "Shared pairs"), overlap_rows)}

The cumulative ordering in `repair_family_yield.json` uses set union, so multi-blocker pairs are never double-counted.

## Topology-in-matcher estimate

- Current topology-conflict pairs: {topology["topology_conflict_pair_count"]}.
- Equal-cost exact alternatives: {topology["defensible_equal_cost_recoverable_pair_count"]}.
- Worse-cost maximum-cardinality mappings with exact content topology: {topology["exploratory_worse_cost_maximum_cardinality_candidate_count"]} (pairs {topology["exploratory_worse_cost_maximum_cardinality_candidate_ids"]}).
- Estimated safe recovery from topology search alone: **{topology["estimated_safe_recovery"]}**.

The broader three are diagnostic candidates only. They are not category-3 cases and retain unsafe argument, transformation, or semantic concerns. Implementing topology-aware search from this census would add proof surface without a demonstrated equal-rank recovery.

## Parser probe

The pinned small model completed on {measurements["sentence_count"]} unique sampled sentences. All {measurements["repeated_run_deterministic_sentence_count"]} repeated outputs were byte-equivalent after excluding runtime, with {measurements["repeated_run_nondeterministic_sentence_count"]} nondeterministic sentences. The run observed {measurements["predicate_recovery_count"]} predicate tokens, {measurements["subject_object_role_recovery_count"]} subject/object role tokens, {measurements["possessive_appositive_attachment_count"]} possessive/appositive attachments, {measurements["participial_modifier_attachment_count"]} modifier attachments, {measurements["quantity_location_association_count"]} quantity/location associations, and {measurements["proposition_content_attachment_count"]} content attachments.

`en_core_web_trf@3.8.0` was not run. `en_core_web_trf`, `spacy-transformers`, `torch`, and `transformers` are absent. No model was installed or substituted. Exact per-token small-model outputs, runtime, memory, cache drift, and the unavailable transformer fields are in `parser_probe.json`.

## Recommended parser implementation order

{chr(10).join(order_lines)}

The order favors bounded, auditable semantic primitives before broad attachment and ellipsis behavior. Each step requires negative-control regression tests and a fresh sealed evaluation before the next step.

## Families not to implement from this tranche

- `no_repair.material_conflict_or_loose_label`: these traces contain explicit incompatible facts or loose document-level MRPC labels. Turning them into equivalence would violate fail-closed behavior.
- `rules.modality_equivalence`: asserted, prospective, potential, and counterfactual operators are not generally interchangeable.
- `context.authenticated_coreference_alias`: MRPC provides no authenticated article-order, predecessor, or document-context field. Adjacent rows, URLs, and dates cannot be manufactured as evidence.
- `matcher.topology_aware_tie_search`: zero pairs satisfy the required equal-cost alternate-mapping criterion.
- `parser.bounded_nonverbal_root_recovery`: no positive pair in this census implicates it; retain the two negative controls but do not build from zero demonstrated yield.
- `parser.vp_ellipsis`: do not give it proof authority until antecedent identity and scope are authenticated and bounded.
- Broad lexical, nominalization, or reporting allowlists: only narrow typed rules with complete roles and matched negative controls are defensible.

## Shortest defensible route toward 452 additional TP

{route_text}

## Limitations

- The labels and all results have already been inspected; this is post-hoc development evidence.
- The deterministic rules create an auditable taxonomy but do not replace manual linguistic adjudication. Records marked `manual_review_required` must be reviewed before implementation.
- A structural negative exposure is not a simulated verifier false-positive count.
- Sole-blocker counts are conservative strata, not guaranteed recovered TP.
- The transformer comparison is blocked, so no transformer quality or cost conclusion is present.
"""


def parser_probe_summary_markdown(probe: Mapping[str, Any]) -> str:
    measurements = probe["small_model_measurements"]
    small = probe["environment"]["small_model"]
    transformer = probe["environment"]["transformer_model"]
    return f"""# MRPC v24 parser-quality probe

## Status

**Blocked for transformer comparison; pinned small-model baseline complete.** The transformer model is diagnostic only and no production evidence or verifier decision was changed.

## Deterministic sample

- Seed: `{probe["sample_selection"]["seed"]}`
- Pair count: {probe["sample_selection"]["pair_count"]}
- Unique sentence count: {probe["sample_selection"]["unique_sentence_count"]}
- Selection: {probe["sample_selection"]["algorithm"]}

The sample covers every observed primary taxonomy stratum, three additional topology-conflict diagnostics, and structurally matched gate-passing negative controls.

## Pinned small model

- Identity: `{small["identity"]}`
- spaCy: `{probe["environment"]["spacy_version"]}`
- Model module/distribution version: `{small["module_version"]}` / `{small["distribution_version"]}`
- Artifact tree SHA-256: `{small["artifact_tree_sha256_excluding_pycache"]}`
- Pipeline: `{small["pipeline"]}`
- Repeated-run deterministic: {measurements["repeated_run_deterministic_sentence_count"]} of {measurements["sentence_count"]} sentences
- Total wall time for two parser-plus-extraction runs per sentence: {measurements["total_runtime_seconds_two_runs_plus_extraction"]:.6f} seconds
- Mean measured run time: {measurements["mean_runtime_seconds_per_parser_extraction_run"]:.6f} seconds
- Peak-RSS delta in platform-native units: {measurements["ru_maxrss_delta"]}
- Memory caveat: {measurements["memory_measurement_caveat"]}

### Structural measurements

- Predicate recovery tokens: {measurements["predicate_recovery_count"]}
- Subject/object role tokens: {measurements["subject_object_role_recovery_count"]}
- Possessive/appositive attachments: {measurements["possessive_appositive_attachment_count"]}
- Participial/modifier attachments: {measurements["participial_modifier_attachment_count"]}
- Quantity/location associations: {measurements["quantity_location_association_count"]}
- Proposition/content attachments: {measurements["proposition_content_attachment_count"]}
- New predicates versus the v24 cache: {measurements["new_predicates_against_v24_cache_count"]}
- New argument signatures versus the v24 cache: {measurements["new_argument_signatures_against_v24_cache_count"]}
- Sentences with unresolved-condition changes versus the v24 cache: {measurements["unresolved_condition_changed_sentence_count"]}

Cache deltas measure environment or extraction drift against the pinned v24 small-model artifact; they are not manual gold parser-error judgments.

## Exact blocker

Requested model: `{transformer["requested_identity"]}`. {transformer["exact_blocker"]}. Module availability is `{transformer["module_availability"]}` and installed distribution versions are `{transformer["distribution_versions"]}`. No substitute model was used and no installation was attempted.

Approval-required exact command: `{transformer["approval_required_install_command"]}`. The official model-wheel SHA-256 is `{transformer["official_wheel_sha256"]}`; resolved dependency versions would be recorded after installation and before the diagnostic run.

Consequently, transformer deltas for predicates, semantic roles, possessives/appositives, modifiers, quantities/locations, content topology, new errors, unresolved conditions, determinism, runtime, and memory remain explicitly unavailable. Full token and attachment outputs for the completed small-model baseline are preserved in `parser_probe.json`.
"""


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def validate(
    records: Sequence[Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
    repair: Mapping[str, Any],
    matrix: Sequence[Sequence[Any]],
    probe: Mapping[str, Any],
) -> None:
    pair_ids = [int(value["pair_id"]) for value in records]
    assert len(records) == EXPECTED_CENSUS
    assert len(set(pair_ids)) == EXPECTED_CENSUS
    assert all(value["label"] is True for value in records)
    assert all(value["gate"]["psa_nominated"] is True for value in records)
    assert all(value["gate"]["native_gate_passed"] is True for value in records)
    assert all(value["gate"]["predicted_paraphrase"] is False for value in records)
    assert all(value["primary_blocker"] in ALLOWED_BLOCKERS for value in records)
    assert all(
        all(blocker in ALLOWED_BLOCKERS for blocker in value["secondary_blockers"])
        for value in records
    )
    assert all(value["primary_blocker"] not in value["secondary_blockers"] for value in records)
    assert all(
        set(value["blocker_evidence"]) == {value["primary_blocker"], *value["secondary_blockers"]}
        for value in records
    )
    assert all(value["trace_location"].startswith(TRACE_BASE + "/") for value in records)
    assert all(
        value["proposed_repair_family"] in FAMILY_DEFINITIONS
        and all(family in FAMILY_DEFINITIONS for family in value["implicated_repair_families"])
        for value in records
    )
    primary_counts = Counter(str(value["primary_blocker"]) for value in records)
    assert sum(primary_counts.values()) == EXPECTED_CENSUS
    assert taxonomy["aggregate_counts"]["primary_blockers"] == {
        blocker: primary_counts.get(blocker, 0) for blocker in ALLOWED_BLOCKERS
    }
    family_rows = {value["repair_family"]: value for value in repair["families"]}
    matrix_headers = list(matrix[0][1:])
    assert set(matrix_headers) == set(family_rows)
    for row in matrix[1:]:
        family = str(row[0])
        diagonal_index = matrix_headers.index(family) + 1
        assert int(row[diagonal_index]) == family_rows[family]["all_implicated_pair_count"]
    assert repair["topology_recovery_estimate"][
        "defensible_equal_cost_recoverable_pair_count"
    ] == primary_counts.get("topology_or_tie_selection_failure", 0)
    observed_primaries = {value["primary_blocker"] for value in records}
    sampled_primaries = {
        value["stratum"]
        for value in probe["sample_selection"]["sample"]
        if value["sample_role"] == "unproven_positive"
    }
    assert observed_primaries <= sampled_primaries
    assert probe["production_evidence_changed"] is False
    assert probe["transformer_measurements"] is None


def main() -> int:
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    graph_cache_payload = json.loads(GRAPH_CACHE.read_text(encoding="utf-8"))
    evidence_by_hash = graph_cache_payload["sentences"]
    pairs = load_pairs()
    traces = report["held_out_pair_traces"]["profiles"][PROFILE]["learned_gate"]
    selected = [
        (trace_index, trace)
        for trace_index, trace in enumerate(traces)
        if trace.get("label") is True
        and trace.get("psa_nominated") is True
        and trace.get("native_gate_passed") is True
        and trace.get("predicted_paraphrase") is False
    ]
    assert len(selected) == EXPECTED_CENSUS
    threshold = float(report["threshold"]["profiles"][PROFILE]["learned"])
    resources = lexical_relations()
    graph_by_hash: dict[str, PropositionGraph] = {}

    def graph_for(text: str) -> PropositionGraph:
        text_hash = sha256_text(text)
        if text_hash not in graph_by_hash:
            graph_by_hash[text_hash] = build_proposition_graph(
                evidence_by_hash[text_hash]["evidence"]
            )
        return graph_by_hash[text_hash]

    records: list[dict[str, Any]] = []
    for trace_index, trace in selected:
        pair_id = int(trace["pair_id"])
        pair = pairs[pair_id]
        evidence1 = evidence_by_hash[sha256_text(pair.sentence1)]["evidence"]
        evidence2 = evidence_by_hash[sha256_text(pair.sentence2)]["evidence"]
        topology = analyze_topology(
            graph_for(pair.sentence1),
            graph_for(pair.sentence2),
            resources,
            set(trace.get("conflict_kinds") or []),
        )
        records.append(
            make_taxonomy_record(
                trace,
                trace_index,
                pair,
                evidence1,
                evidence2,
                topology,
                threshold,
            )
        )
    records.sort(key=lambda value: int(value["pair_id"]))

    negatives = classify_negative_controls(traces, pairs, evidence_by_hash)
    attach_negative_controls(records, negatives)
    repair, matrix = repair_yield(records, negatives)
    sample = select_probe_sample(records, negatives)
    probe = parser_probe(sample, evidence_by_hash)

    primary_counts = Counter(str(value["primary_blocker"]) for value in records)
    secondary_counts = Counter(
        blocker for value in records for blocker in value["secondary_blockers"]
    )
    confidence_counts = Counter(str(value["classification_confidence"]) for value in records)
    family_counts = Counter(str(value["proposed_repair_family"]) for value in records)
    taxonomy = {
        "schema_version": "aiidea_mrpc_v24_unproven_positive_taxonomy_v1",
        "generated_at_date": "2026-08-01",
        "analysis_only": True,
        "post_hoc_development_evidence": True,
        "source": {
            "report": str(SOURCE_REPORT.relative_to(ROOT)),
            "report_sha256": sha256_file(SOURCE_REPORT),
            "graph_cache": str(GRAPH_CACHE.relative_to(ROOT)),
            "graph_cache_sha256": sha256_file(GRAPH_CACHE),
            "lexical_cache": str(LEXICAL_CACHE.relative_to(ROOT)),
            "lexical_cache_sha256": sha256_file(LEXICAL_CACHE),
            "mrpc_corpus_sha256": sha256_file(MRPC_CORPUS),
            "mrpc_test_sha256": sha256_file(MRPC_TEST),
        },
        "selection": {
            "trace_base": TRACE_BASE,
            "predicate": (
                "label == true && psa_nominated == true && native_gate_passed == true "
                "&& predicted_paraphrase == false"
            ),
            "profile": PROFILE,
            "threshold": threshold,
            "expected_count": EXPECTED_CENSUS,
        },
        "allowed_blockers": list(ALLOWED_BLOCKERS),
        "classification_method": {
            "kind": "deterministic_inspectable_rules_not_statistical_or_generative",
            "priority": [
                "incomplete graph",
                "demonstrated equal-cost exact topology alternative",
                "authenticated-context signal",
                "shared-surface grouping signal",
                "parser-shape extraction signal",
                "complete but unlicensed transformation",
                "explicit material semantic conflict",
                "compatibility abstention fallback to missing transformation",
            ],
            "manual_review_policy": (
                "all medium/low-confidence or multi-blocker records require manual review"
            ),
            "topology_category_rule": (
                "requires an exact content-topology alternative with identical maximum "
                "cardinality and identical 13-component matcher cost vector"
            ),
        },
        "aggregate_counts": {
            "census": len(records),
            "unique_pair_ids": len({int(value["pair_id"]) for value in records}),
            "primary_blockers": {
                blocker: primary_counts.get(blocker, 0) for blocker in ALLOWED_BLOCKERS
            },
            "secondary_blockers": {
                blocker: secondary_counts.get(blocker, 0) for blocker in ALLOWED_BLOCKERS
            },
            "classification_confidence": dict(sorted(confidence_counts.items())),
            "manual_review_required": sum(
                bool(value["manual_review_required"]) for value in records
            ),
            "proposed_repair_families": dict(sorted(family_counts.items())),
            "graph_states": dict(
                sorted(Counter(str(value["graph_state"]) for value in records).items())
            ),
            "graph_reasons": dict(
                sorted(Counter(str(value["graph_reason"]) for value in records).items())
            ),
        },
        "records": records,
    }
    validate(records, taxonomy, repair, matrix, probe)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT_DIR / "taxonomy.json", taxonomy)
    atomic_json(OUTPUT_DIR / "repair_family_yield.json", repair)
    atomic_json(OUTPUT_DIR / "parser_probe.json", probe)
    write_csv(
        OUTPUT_DIR / "taxonomy.csv",
        (
            "pair_id",
            "sentence1",
            "sentence2",
            "label",
            "similarity",
            "gate_threshold",
            "graph_state",
            "graph_reason",
            "primary_blocker",
            "secondary_blockers_json",
            "blocker_evidence_json",
            "unresolved_conditions_json",
            "conflict_kinds_json",
            "conflicts_json",
            "implicated_predicates_json",
            "implicated_roles_json",
            "relevant_quantities_json",
            "relevant_content_edges_json",
            "proposed_repair_family",
            "implicated_repair_families_json",
            "repair_scope",
            "classification_confidence",
            "manual_review_required",
            "analogous_negative_controls_json",
            "estimated_negative_exposure",
            "source_report",
            "trace_location",
        ),
        (
            (
                value["pair_id"],
                value["sentence1"],
                value["sentence2"],
                value["label"],
                value["similarity"],
                value["gate"]["threshold"],
                value["graph_state"],
                value["graph_reason"],
                value["primary_blocker"],
                canonical_json(value["secondary_blockers"]),
                canonical_json(value["blocker_evidence"]),
                canonical_json(value["unresolved_conditions"]),
                canonical_json(value["conflict_kinds"]),
                canonical_json(value["conflicts"]),
                canonical_json(value["implicated_predicates"]),
                canonical_json(value["implicated_roles"]),
                canonical_json(value["relevant_quantities"]),
                canonical_json(value["relevant_content_edges"]),
                value["proposed_repair_family"],
                canonical_json(value["implicated_repair_families"]),
                value["repair_scope"],
                value["classification_confidence"],
                value["manual_review_required"],
                canonical_json(value["analogous_negative_controls"]),
                value["estimated_negative_exposure"],
                value["source_report"],
                value["trace_location"],
            )
            for value in records
        ),
    )
    with (OUTPUT_DIR / "overlap_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(matrix)
    (OUTPUT_DIR / "summary.md").write_text(
        taxonomy_summary_markdown(taxonomy, repair, matrix, probe),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "parser_probe_summary.md").write_text(
        parser_probe_summary_markdown(probe),
        encoding="utf-8",
    )

    output_hashes = {
        path.name: sha256_file(path) for path in sorted(OUTPUT_DIR.iterdir()) if path.is_file()
    }
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "census": len(records),
                "primary_blockers": taxonomy["aggregate_counts"]["primary_blockers"],
                "manual_review_required": taxonomy["aggregate_counts"]["manual_review_required"],
                "topology_recovery": repair["topology_recovery_estimate"],
                "parser_probe_status": probe["status"],
                "output_sha256": output_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
