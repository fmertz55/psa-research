#!/usr/bin/env python3
"""Evaluate MRPC through the current repaired paraphrase-membership pipeline.

This is an offline, serial-ingestion replay of the production policy:

1. Map each sentence into the pinned calibrated PSA routing grid.
2. Expand the sentence's routing cell for a non-excluding locality prior.
3. Search native vectors globally for up to 32 free sentences and 32 true
   groups in the exact semantic namespace.
4. Freeze a bounded inventory from the benchmark's global-native recall
   oracle, then schedule every comparison through native-cosine and
   PSA-Manhattan proximity queues. The queues change evaluation order only;
   they never choose membership.
5. Join a verified group, create a true two-member group from a verified free
   sentence, or leave the sentence durable and free-range.

The local cache does not contain object-to-synset annotations. The production
verifier therefore follows its explicit ``coverage_empty_side`` pass branch.
MRPC labels are used only after routing to score held-out published pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import (  # noqa: E402
    DEFAULT_CACHE_PATHS,
    DEFAULT_ORDER_SEEDS,
    BenchmarkError,
    _aggregate_orders,
    _atomic_json,
    _baseline_report,
    _canonical_json,
    _load_cache_evidence,
    _mean_summary,
)

from consolidated_clustering_v2.routing.psa_calibration import (  # noqa: E402
    PSARoutingCalibration,
)
from consolidated_clustering_v2.routing.psa_mapper import (  # noqa: E402
    GuideStarPSARouter,
)
from mcp_servers.mcp_lib.comparison_triage import (  # noqa: E402
    COMPARISON_TRIAGE_SCHEMA_VERSION,
    build_comparison_geometry,
    build_comparison_queues,
)
from mcp_servers.mcp_lib.paraphrase_verification import (  # noqa: E402
    PROBE_IDS,
    ParaphraseVerificationController,
)
from mcp_servers.mcp_lib.psa_routing_labels import (  # noqa: E402
    expand_routing_labels,
    normalize_corpus_routing_label,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    DEFAULT_MRPC_CALIBRATION_PERCENT,
    DEFAULT_SPLIT_SEED,
    MRPCPair,
    _binary_metrics,
    compute_psa_signatures,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    NATIVE_EMBEDDING_DIMENSION,
    PSA_SIGNATURE_CANDIDATE_LIMIT,
    PSA_SIGNATURE_DIMENSION,
    QWEN3_EMBEDDING_MODEL_ID,
    paraphrase_native_cosine_threshold,
)
from mcp_servers.sentence_node_service.semantic_signature import (  # noqa: E402
    SemanticSignatureSummary,
    build_signature_summary,
    normalize_signature,
    update_signature_summary,
)

DEFAULT_CALIBRATION_PATH = (
    APP_ROOT / "consolidated_clustering_v2" / "config" / "psa_calibration_v2_115stars.json"
)
DEFAULT_PINNED_GUIDE_EMBEDDINGS = (
    REPOSITORY_ROOT / "temp" / "psa_frame_overlap" / "v2_115_star_embeddings.npy"
)
DEFAULT_LOCALITY_PRIOR_RADIUS = 1
DEFAULT_FAST_VERIFY_PAGE_SIZE = 8
WORDNET_SENSE_CACHE_SCHEMA_VERSION = "aiidea_mrpc_wordnet_sense_cache_v1"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator == 0.0:
        raise BenchmarkError("cosine received a zero or invalid vector")
    return float(np.dot(left, right) / denominator)


def _normalize_senses(
    senses: Mapping[str, Iterable[str]] | None,
) -> dict[str, frozenset[str]]:
    normalized: dict[str, frozenset[str]] = {}
    for lemma, synset_ids in (senses or {}).items():
        lemma_text = str(lemma or "").strip().casefold()
        ids = frozenset(
            str(synset_id).strip() for synset_id in synset_ids if str(synset_id or "").strip()
        )
        if lemma_text and ids:
            normalized[lemma_text] = ids
    return normalized


def _merge_senses(
    *sense_maps: Mapping[str, Iterable[str]],
) -> dict[str, frozenset[str]]:
    merged: dict[str, set[str]] = {}
    for sense_map in sense_maps:
        for lemma, synset_ids in sense_map.items():
            merged.setdefault(lemma, set()).update(synset_ids)
    return {
        lemma: frozenset(sorted(synset_ids)) for lemma, synset_ids in merged.items() if synset_ids
    }


def _wordnet_gate(
    source: Mapping[str, Iterable[str]],
    target: Mapping[str, Iterable[str]],
    *,
    threshold: float = 0.5,
) -> tuple[bool, float, str]:
    if not source or not target:
        return True, 1.0, "coverage_empty_side"
    shared = set(source) & set(target)
    if not shared:
        return True, 1.0, "coverage_no_shared_lemmas"
    agreement = sum(1 for lemma in shared if set(source[lemma]) & set(target[lemma])) / len(shared)
    return agreement >= threshold, agreement, "applied"


def _load_wordnet_sense_cache(
    path: Path,
    texts: Iterable[str],
) -> tuple[dict[str, dict[str, frozenset[str]]], dict[str, Any]]:
    if not path.is_file():
        raise BenchmarkError(f"WordNet sense cache does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != WORDNET_SENSE_CACHE_SCHEMA_VERSION:
        raise BenchmarkError("WordNet sense cache schema is not supported")
    resolver = payload.get("resolver")
    wordnet = payload.get("wordnet")
    sentences = payload.get("sentences")
    if not all(isinstance(value, dict) for value in (resolver, wordnet, sentences)):
        raise BenchmarkError("WordNet sense cache evidence is incomplete")
    if wordnet.get("lexicon") != "oewn:2024":
        raise BenchmarkError("WordNet sense cache must use pinned oewn:2024")
    if resolver.get("production_equivalent") is not False:
        raise BenchmarkError("offline WordNet cache must state that it is not production WSD")

    expected = {sha256_text(text): text for text in texts}
    if set(sentences) != set(expected):
        raise BenchmarkError("WordNet sense cache does not exactly cover MRPC sentence hashes")
    senses_by_text: dict[str, dict[str, frozenset[str]]] = {}
    for text_hash, text in expected.items():
        entry = sentences[text_hash]
        if (
            not isinstance(entry, dict)
            or entry.get("text_sha256") != text_hash
            or not isinstance(entry.get("senses"), dict)
        ):
            raise BenchmarkError(f"invalid WordNet sense cache entry {text_hash}")
        senses_by_text[text] = _normalize_senses(entry["senses"])
    return senses_by_text, {
        "cache_path": str(path.resolve()),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": payload["schema_version"],
        "resolver": resolver,
        "wordnet": wordnet,
        "coverage": payload.get("coverage"),
    }


@dataclass
class FreeSentenceState:
    member_id: str
    text: str
    native: np.ndarray
    signature: np.ndarray
    routing_label: str
    wordnet_senses: dict[str, frozenset[str]]


@dataclass
class CurrentGroupState:
    group_id: str
    members: list[str]
    representative_text: str
    native_centroid: np.ndarray
    signature_summary: SemanticSignatureSummary
    routing_label: str
    wordnet_senses: dict[str, frozenset[str]]


@dataclass(frozen=True)
class CurrentRouteTrace:
    winner_type: str | None
    winner_id: str | None
    global_free_count: int
    global_group_count: int
    eligible_count: int
    native_first_candidate_id: str | None
    scheduled_first_candidate_id: str | None


@dataclass(frozen=True)
class CurrentCandidate:
    candidate_type: str
    candidate_id: str
    native_score: float
    member_count: int
    psa_local: bool
    native: np.ndarray
    signature: np.ndarray

    @property
    def observation_id(self) -> str:
        prefix = "group" if self.candidate_type == "paraphrase_group" else "sentence"
        return f"{prefix}:{self.candidate_id}"


class CurrentProductionRouter:
    """Serial replay of the repaired free-range/group membership policy."""

    def __init__(
        self,
        *,
        native_threshold: float,
        locality_prior_radius: int = DEFAULT_LOCALITY_PRIOR_RADIUS,
    ) -> None:
        self.native_threshold = native_threshold
        self.locality_prior_radius = locality_prior_radius
        self.free_sentences: dict[str, FreeSentenceState] = {}
        self.groups: dict[str, CurrentGroupState] = {}
        self.assignment: dict[str, str] = {}
        self.text_by_member: dict[str, str] = {}
        self.verifier = ParaphraseVerificationController(
            require_complete_evidence=False,
            enforce_fact_conflicts=False,
        )
        self.stats = {
            "query_count": 0,
            "global_free_candidates": 0,
            "global_group_candidates": 0,
            "eligible_candidates": 0,
            "comparison_triage_candidates": 0,
            "comparison_triage_schedules_changed": 0,
            "comparison_triage_first_candidate_changed": 0,
            "comparison_triage_rank_displacement_count": 0,
            "comparison_triage_rank_displacement_total": 0,
            "comparison_triage_max_rank_displacement": 0,
            "free_sentence_wins": 0,
            "existing_group_wins": 0,
            "unmatched_free_range": 0,
            "multiple_verified_candidate_sets": 0,
            "synset_empty_coverage_accepts": 0,
            "wordnet_reason": {
                "applied": 0,
                "coverage_empty_side": 0,
                "coverage_no_shared_lemmas": 0,
            },
            "verification_candidates_evaluated": 0,
            "verification_rejects": 0,
            "bandit_first_probe": {probe_id: 0 for probe_id in PROBE_IDS},
            "probe_status": {
                probe_id: {
                    "pass": 0,
                    "conflict": 0,
                    "unknown": 0,
                }
                for probe_id in PROBE_IDS
            },
            "shadow_recommendation": {
                "accept": 0,
                "defer": 0,
                "reject": 0,
            },
        }

    @staticmethod
    def _native_inventory_sort_key(
        candidate: CurrentCandidate,
    ) -> tuple[float, int, int, str]:
        """Match the pre-triage production nomination inventory order."""

        return (
            -candidate.native_score,
            0 if candidate.psa_local else 1,
            -candidate.member_count,
            candidate.observation_id,
        )

    def _global_free_candidates(
        self,
        native: np.ndarray,
        allowed_labels: set[str],
    ) -> list[CurrentCandidate]:
        candidates = [
            CurrentCandidate(
                candidate_type="free_sentence",
                candidate_id=state.member_id,
                native_score=_cosine(native, state.native),
                member_count=0,
                psa_local=state.routing_label in allowed_labels,
                native=state.native,
                signature=state.signature,
            )
            for state in self.free_sentences.values()
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.native_score,
                candidate.candidate_id,
            )
        )
        return candidates[:PSA_SIGNATURE_CANDIDATE_LIMIT]

    def _global_group_candidates(
        self,
        native: np.ndarray,
        allowed_labels: set[str],
    ) -> list[CurrentCandidate]:
        candidates = [
            CurrentCandidate(
                candidate_type="paraphrase_group",
                candidate_id=group.group_id,
                native_score=_cosine(native, group.native_centroid),
                member_count=len(group.members),
                psa_local=group.routing_label in allowed_labels,
                native=group.native_centroid,
                signature=np.asarray(
                    group.signature_summary.centroid,
                    dtype=np.float64,
                ),
            )
            for group in self.groups.values()
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.native_score,
                -candidate.member_count,
                candidate.candidate_id,
            )
        )
        return candidates[:PSA_SIGNATURE_CANDIDATE_LIMIT]

    def _create_group(
        self,
        source: FreeSentenceState,
        target: FreeSentenceState,
    ) -> CurrentGroupState:
        ordered = sorted((source.member_id, target.member_id))
        representative_id = min(
            ordered,
            key=lambda member_id: (
                len(self.text_by_member[member_id]),
                member_id,
            ),
        )
        group_id = "mrpc-group:" + hashlib.sha256("\0".join(ordered).encode("utf-8")).hexdigest()
        summary = build_signature_summary(
            [
                (source.member_id, source.signature),
                (target.member_id, target.signature),
            ],
            expected_dimension=PSA_SIGNATURE_DIMENSION,
        )
        group = CurrentGroupState(
            group_id=group_id,
            members=ordered,
            representative_text=self.text_by_member[representative_id],
            native_centroid=np.asarray(
                (source.native + target.native) / 2.0,
                dtype=np.float32,
            ),
            signature_summary=summary,
            routing_label=(
                source.routing_label
                if representative_id == source.member_id
                else target.routing_label
            ),
            wordnet_senses=_merge_senses(
                source.wordnet_senses,
                target.wordnet_senses,
            ),
        )
        self.groups[group_id] = group
        for member_id in ordered:
            self.assignment[member_id] = group_id
            self.free_sentences.pop(member_id, None)
        return group

    def _join_group(
        self,
        source: FreeSentenceState,
        group: CurrentGroupState,
    ) -> None:
        previous_count = len(group.members)
        group.native_centroid = np.asarray(
            (group.native_centroid * previous_count + source.native) / (previous_count + 1),
            dtype=np.float32,
        )
        group.signature_summary = update_signature_summary(
            group.signature_summary,
            [(source.member_id, source.signature)],
        )
        group.members.append(source.member_id)
        group.wordnet_senses = _merge_senses(
            group.wordnet_senses,
            source.wordnet_senses,
        )
        self.assignment[source.member_id] = group.group_id
        self.free_sentences.pop(source.member_id, None)

    def route(
        self,
        *,
        member_id: str,
        text: str,
        native: np.ndarray,
        signature: np.ndarray,
        routing_label: str,
        wordnet_senses: Mapping[str, Iterable[str]] | None = None,
    ) -> CurrentRouteTrace:
        if (
            member_id in self.text_by_member
            or member_id in self.assignment
            or member_id in self.free_sentences
        ):
            raise BenchmarkError(f"member {member_id!r} was ingested twice")
        source = FreeSentenceState(
            member_id=member_id,
            text=text,
            native=np.asarray(native, dtype=np.float32),
            # Production persistence canonicalizes every sentence signature
            # before candidate routing. Mirror that boundary here so L1
            # compares semantic coordinates, not float32 centering residue.
            signature=np.asarray(
                normalize_signature(
                    signature,
                    expected_dimension=PSA_SIGNATURE_DIMENSION,
                ),
                dtype=np.float64,
            ),
            routing_label=routing_label,
            wordnet_senses=_normalize_senses(wordnet_senses),
        )
        self.text_by_member[member_id] = text
        self.stats["query_count"] += 1

        allowed_labels = set(
            expand_routing_labels(
                [routing_label],
                content_type="knowledge",
                radius=self.locality_prior_radius,
            )
        )
        free_candidates = self._global_free_candidates(
            source.native,
            allowed_labels,
        )
        group_candidates = self._global_group_candidates(
            source.native,
            allowed_labels,
        )
        self.stats["global_free_candidates"] += len(free_candidates)
        self.stats["global_group_candidates"] += len(group_candidates)

        deduplicated: dict[
            tuple[str, str],
            CurrentCandidate,
        ] = {}
        for candidate in (*free_candidates, *group_candidates):
            identity = candidate.candidate_type, candidate.candidate_id
            prior = deduplicated.get(identity)
            if (
                prior is None
                or candidate.native_score > prior.native_score
                or (
                    candidate.native_score == prior.native_score
                    and candidate.psa_local
                    and not prior.psa_local
                )
            ):
                deduplicated[identity] = candidate
        native_inventory = sorted(
            (
                candidate
                for candidate in deduplicated.values()
                if candidate.native_score >= self.native_threshold
            ),
            key=self._native_inventory_sort_key,
        )[:PSA_SIGNATURE_CANDIDATE_LIMIT]
        geometry_by_observation_id = {
            candidate.observation_id: build_comparison_geometry(
                source_observation_id=f"sentence:{member_id}",
                target_observation_id=candidate.observation_id,
                source_native_embedding=source.native,
                target_native_embedding=candidate.native,
                source_signature=source.signature,
                target_signature=candidate.signature,
                expected_psa_dimension=PSA_SIGNATURE_DIMENSION,
            )
            for candidate in native_inventory
        }
        triage_queues = build_comparison_queues(geometry_by_observation_id.values())
        candidate_by_observation_id = {
            candidate.observation_id: candidate for candidate in native_inventory
        }
        scheduled = [
            candidate_by_observation_id[geometry.target_observation_id]
            for geometry in triage_queues.comparison_schedule
        ]
        native_positions = {
            candidate.observation_id: index for index, candidate in enumerate(native_inventory)
        }
        scheduled_ids = [candidate.observation_id for candidate in scheduled]
        native_ids = [candidate.observation_id for candidate in native_inventory]
        displacements = [
            abs(index - native_positions[candidate_id])
            for index, candidate_id in enumerate(scheduled_ids)
        ]
        if scheduled_ids != native_ids:
            self.stats["comparison_triage_schedules_changed"] += 1
        if scheduled_ids[:1] != native_ids[:1]:
            self.stats["comparison_triage_first_candidate_changed"] += 1
        self.stats["comparison_triage_rank_displacement_count"] += len(displacements)
        self.stats["comparison_triage_rank_displacement_total"] += sum(displacements)
        self.stats["comparison_triage_max_rank_displacement"] = max(
            int(self.stats["comparison_triage_max_rank_displacement"]),
            max(displacements, default=0),
        )
        self.stats["eligible_candidates"] += len(scheduled)
        self.stats["comparison_triage_candidates"] += len(geometry_by_observation_id)

        # The optional offline cache supplies lemma/synset assignments without
        # requiring the object graph. The microGPT policy ranks evidence
        # probes; it does not receive MRPC labels or select truth. Every fixed
        # candidate is evaluated: triage order cannot select membership by
        # stopping at the first accepted comparison.
        accepted: dict[str, tuple[CurrentCandidate, str]] = {}
        for candidate in scheduled:
            candidate_type = candidate.candidate_type
            candidate_id = candidate.candidate_id
            native_score = candidate.native_score
            psa_local = candidate.psa_local
            if candidate_type == "paraphrase_group":
                target_state = self.groups[candidate_id]
                target_text = target_state.representative_text
                target_senses = target_state.wordnet_senses
            else:
                target_state = self.free_sentences[candidate_id]
                target_text = target_state.text
                target_senses = target_state.wordnet_senses
            (
                wordnet_passes,
                wordnet_agreement,
                wordnet_reason,
            ) = _wordnet_gate(
                source.wordnet_senses,
                target_senses,
            )
            decision = self.verifier.verify(
                source_text=text,
                target_text=target_text,
                native_similarity=native_score,
                candidate_type=candidate_type,
                psa_local=psa_local,
                wordnet_passes=wordnet_passes,
                wordnet_agreement=wordnet_agreement,
                wordnet_reason=wordnet_reason,
            )
            self.stats["verification_candidates_evaluated"] += 1
            self.stats["wordnet_reason"][wordnet_reason] += 1
            self.stats["bandit_first_probe"][decision.bandit_selected_probe] += 1
            self.stats["shadow_recommendation"][decision.shadow_recommendation] += 1
            for probe_result in decision.probe_results:
                self.stats["probe_status"][probe_result.probe_id][probe_result.status] += 1
            if decision.accepted:
                accepted[candidate.observation_id] = (
                    candidate,
                    wordnet_reason,
                )
            else:
                self.stats["verification_rejects"] += 1

        winner_record = next(
            (
                accepted[candidate.observation_id]
                for candidate in native_inventory
                if candidate.observation_id in accepted
            ),
            None,
        )
        native_first_candidate_id = native_ids[0] if native_ids else None
        scheduled_first_candidate_id = scheduled_ids[0] if scheduled_ids else None
        if winner_record is None:
            self.free_sentences[member_id] = source
            self.stats["unmatched_free_range"] += 1
            return CurrentRouteTrace(
                winner_type=None,
                winner_id=None,
                global_free_count=len(free_candidates),
                global_group_count=len(group_candidates),
                eligible_count=len(scheduled),
                native_first_candidate_id=native_first_candidate_id,
                scheduled_first_candidate_id=(scheduled_first_candidate_id),
            )

        if len(accepted) > 1:
            self.stats["multiple_verified_candidate_sets"] += 1
        winner, winner_wordnet_reason = winner_record
        if winner_wordnet_reason == "coverage_empty_side":
            self.stats["synset_empty_coverage_accepts"] += 1
        winner_type = winner.candidate_type
        winner_id = winner.candidate_id
        if winner_type == "paraphrase_group":
            self._join_group(source, self.groups[winner_id])
            self.stats["existing_group_wins"] += 1
        else:
            target = self.free_sentences[winner_id]
            self._create_group(source, target)
            self.stats["free_sentence_wins"] += 1
        return CurrentRouteTrace(
            winner_type=winner_type,
            winner_id=winner_id,
            global_free_count=len(free_candidates),
            global_group_count=len(group_candidates),
            eligible_count=len(scheduled),
            native_first_candidate_id=native_first_candidate_id,
            scheduled_first_candidate_id=scheduled_first_candidate_id,
        )

    def report(self) -> dict[str, Any]:
        group_sizes = [len(group.members) for group in self.groups.values()]
        displacement_count = int(self.stats["comparison_triage_rank_displacement_count"])
        return {
            **self.stats,
            "comparison_triage_mean_rank_displacement": (
                float(self.stats["comparison_triage_rank_displacement_total"]) / displacement_count
                if displacement_count
                else 0.0
            ),
            "free_sentence_count": len(self.free_sentences),
            "true_group_count": len(self.groups),
            "grouped_sentence_count": len(self.assignment),
            "group_size": _mean_summary([float(size) for size in group_sizes]),
            "maximum_group_size": max(group_sizes, default=0),
        }


def _ordered_texts(texts: Iterable[str], seed: str) -> list[str]:
    return sorted(
        texts,
        key=lambda text: (
            hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest(),
            text,
        ),
    )


def _load_pinned_frame(
    guide_embeddings_path: Path,
    calibration_path: Path,
) -> tuple[np.ndarray, PSARoutingCalibration, GuideStarPSARouter, dict[str, Any]]:
    if not guide_embeddings_path.is_file():
        raise BenchmarkError(f"pinned guide embeddings do not exist: {guide_embeddings_path}")
    if not calibration_path.is_file():
        raise BenchmarkError(f"PSA calibration does not exist: {calibration_path}")
    guide_embeddings = np.asarray(
        np.load(guide_embeddings_path, allow_pickle=False),
        dtype=np.float32,
    )
    if guide_embeddings.shape != (
        PSA_SIGNATURE_DIMENSION,
        NATIVE_EMBEDDING_DIMENSION,
    ):
        raise BenchmarkError(f"invalid pinned guide matrix shape {guide_embeddings.shape}")
    calibration = PSARoutingCalibration.from_json_file(calibration_path)
    star_positions = calibration.get_star_positions()
    if star_positions is None:
        raise BenchmarkError("PSA calibration does not bundle star positions")
    router = GuideStarPSARouter(guide_embeddings, star_positions)
    if router.get_projection_hash() != calibration.frame_hash:
        raise BenchmarkError(
            "pinned guide embeddings do not match PSA calibration frame hash: "
            f"{router.get_projection_hash()} != {calibration.frame_hash}"
        )
    if router.psa_mapper.position_hash != calibration.position_hash:
        raise BenchmarkError("bundled star positions do not match PSA calibration position hash")
    evidence = {
        "guide_embeddings_path": str(guide_embeddings_path.resolve()),
        "guide_embeddings_sha256": hashlib.sha256(guide_embeddings_path.read_bytes()).hexdigest(),
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        "calibration_id": calibration.calibration_id,
        "frame_hash": calibration.frame_hash,
        "position_hash": calibration.position_hash,
        "bins_per_axis": calibration.bins_per_axis,
        "label_prefix": calibration.label_prefix,
    }
    return guide_embeddings, calibration, router, evidence


def _project_representations(
    native_by_text: Mapping[str, np.ndarray],
    pinned_guides: np.ndarray,
    frame_router: GuideStarPSARouter,
    calibration: PSARoutingCalibration,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    texts = sorted(native_by_text, key=lambda text: (sha256_text(text), text))
    native_matrix = np.stack([native_by_text[text] for text in texts])
    signatures = compute_psa_signatures(native_matrix, pinned_guides)
    signatures_by_text = {text: signatures[index] for index, text in enumerate(texts)}
    labels_by_text: dict[str, str] = {}
    label_counts: dict[str, int] = {}
    for text in texts:
        coordinate = frame_router.get_full_psa_coordinate(native_by_text[text].tolist())
        raw_label = calibration.coord_to_label(
            coordinate.x,
            coordinate.y,
            coordinate.z,
        )
        label = normalize_corpus_routing_label(
            raw_label,
            content_type="knowledge",
        )
        labels_by_text[text] = label
        label_counts[label] = label_counts.get(label, 0) + 1
    distribution = {
        "unique_routing_label_count": len(label_counts),
        "maximum_sentences_in_one_label": max(label_counts.values(), default=0),
        "mean_sentences_per_occupied_label": (
            float(statistics.fmean(label_counts.values())) if label_counts else 0.0
        ),
    }
    return signatures_by_text, labels_by_text, distribution


def _rate_by_label(
    values: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    positive = labels
    negative = ~labels
    return {
        "all": float(np.mean(values)),
        "positive": float(np.mean(values[positive])),
        "negative": float(np.mean(values[negative])),
    }


def _pair_gate_diagnostics(
    held_out_pairs: Sequence[MRPCPair],
    native_by_text: Mapping[str, np.ndarray],
    labels_by_text: Mapping[str, str],
    *,
    native_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(
        [pair.is_paraphrase for pair in held_out_pairs],
        dtype=bool,
    )
    native_cosines = np.asarray(
        [
            float(
                np.dot(
                    native_by_text[pair.sentence1],
                    native_by_text[pair.sentence2],
                )
            )
            for pair in held_out_pairs
        ]
    )
    local = np.asarray(
        [
            labels_by_text[pair.sentence2]
            in set(
                expand_routing_labels(
                    [labels_by_text[pair.sentence1]],
                    content_type="knowledge",
                    radius=DEFAULT_LOCALITY_PRIOR_RADIUS,
                )
            )
            for pair in held_out_pairs
        ]
    )
    native_gate = native_cosines >= native_threshold
    joint_gate = local & native_gate
    quantiles = (0.1, 0.5, 0.9)
    return {
        "psa_neighborhood_target_rate": _rate_by_label(local, labels),
        "native_threshold_target_rate": _rate_by_label(
            native_gate,
            labels,
        ),
        "joint_pair_gate_target_rate": _rate_by_label(
            joint_gate,
            labels,
        ),
        "joint_pair_gate_metrics": _binary_metrics(labels, joint_gate),
        "native_cosine_quantiles": {
            label: {
                str(quantile): float(np.quantile(native_cosines[mask], quantile))
                for quantile in quantiles
            }
            for label, mask in (
                ("positive", labels),
                ("negative", ~labels),
            )
        },
    }


def _run_order(
    *,
    pairs: Sequence[MRPCPair],
    held_out_pairs: Sequence[MRPCPair],
    native_by_text: Mapping[str, np.ndarray],
    signatures_by_text: Mapping[str, np.ndarray],
    labels_by_text: Mapping[str, str],
    wordnet_senses_by_text: Mapping[
        str,
        Mapping[str, Iterable[str]],
    ],
    order_seed: str,
    native_threshold: float,
) -> dict[str, Any]:
    router = CurrentProductionRouter(native_threshold=native_threshold)
    texts = _ordered_texts(native_by_text, order_seed)
    for text in texts:
        router.route(
            member_id=f"mrpc-sentence:{sha256_text(text)}",
            text=text,
            native=native_by_text[text],
            signature=signatures_by_text[text],
            routing_label=labels_by_text[text],
            wordnet_senses=wordnet_senses_by_text.get(text),
        )

    def evaluate(
        evaluation_pairs: Sequence[MRPCPair],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        labels = np.asarray(
            [pair.is_paraphrase for pair in evaluation_pairs],
            dtype=bool,
        )
        predictions: list[bool] = []
        pair_predictions: list[dict[str, Any]] = []
        for pair in evaluation_pairs:
            first_id = f"mrpc-sentence:{sha256_text(pair.sentence1)}"
            second_id = f"mrpc-sentence:{sha256_text(pair.sentence2)}"
            first_group = router.assignment.get(first_id)
            second_group = router.assignment.get(second_id)
            predicted = bool(first_group and second_group and first_group == second_group)
            predictions.append(predicted)
            pair_predictions.append(
                {
                    "pair_id": pair.pair_id,
                    "label": pair.is_paraphrase,
                    "predicted_paraphrase": predicted,
                    "first_group_id": first_group,
                    "second_group_id": second_group,
                }
            )
        return (
            _binary_metrics(labels, np.asarray(predictions, dtype=bool)),
            pair_predictions,
        )

    held_out_metrics, pair_predictions = evaluate(held_out_pairs)
    official_test_metrics, official_pair_predictions = evaluate(pairs)
    return {
        "order_seed": order_seed,
        "ingested_sentence_count": len(texts),
        "held_out_metrics": held_out_metrics,
        "official_test_metrics": official_test_metrics,
        "router": router.report(),
        "pair_predictions": pair_predictions,
        "official_test_pair_predictions": official_pair_predictions,
    }


def _aggregate_official_orders(order_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pair_votes: dict[int, list[bool]] = {}
    pair_labels: dict[int, bool] = {}
    for report in order_reports:
        for row in report["official_test_pair_predictions"]:
            pair_id = int(row["pair_id"])
            pair_votes.setdefault(pair_id, []).append(bool(row["predicted_paraphrase"]))
            pair_labels[pair_id] = bool(row["label"])
    ordered_ids = sorted(pair_votes)
    consensus = np.asarray(
        [sum(pair_votes[pair_id]) * 2 >= len(pair_votes[pair_id]) for pair_id in ordered_ids],
        dtype=bool,
    )
    labels = np.asarray(
        [pair_labels[pair_id] for pair_id in ordered_ids],
        dtype=bool,
    )
    return {
        "majority_vote_official_test_metrics": _binary_metrics(labels, consensus),
        "order_sensitive_official_test_pair_count": sum(
            len(set(votes)) > 1 for votes in pair_votes.values()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        action="append",
        dest="caches",
        help="Complete qwen_embeddings.sqlite3 cache; repeat for reproducibility",
    )
    parser.add_argument(
        "--mrpc-dir",
        type=Path,
        default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "reports" / "mrpc_current_pipeline",
    )
    parser.add_argument(
        "--pinned-guide-embeddings",
        type=Path,
        default=DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
    )
    parser.add_argument(
        "--order-seed",
        action="append",
        dest="order_seeds",
        help="Deterministic serial-ingestion order; repeat to test order sensitivity",
    )
    parser.add_argument(
        "--wordnet-sense-cache",
        type=Path,
        help=(
            "Optional offline OEWN 2024 sense cache produced by build_mrpc_wordnet_sense_cache.py"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = load_mrpc_pairs(args.mrpc_dir.resolve())
    splits = split_mrpc_by_sentence_component(pairs)
    held_out_pairs = [pair for pair in pairs if splits[pair.pair_id] == "held_out"]
    all_texts = {text for pair in pairs for text in (pair.sentence1, pair.sentence2)}
    if args.wordnet_sense_cache is not None:
        (
            wordnet_senses_by_text,
            wordnet_evidence,
        ) = _load_wordnet_sense_cache(
            args.wordnet_sense_cache.resolve(),
            all_texts,
        )
    else:
        wordnet_senses_by_text = {}
        wordnet_evidence = None
    split_counts = {
        split: {
            "pair_count": sum(value == split for value in splits.values()),
            "positive_count": sum(
                splits[pair.pair_id] == split and pair.is_paraphrase for pair in pairs
            ),
            "negative_count": sum(
                splits[pair.pair_id] == split and not pair.is_paraphrase for pair in pairs
            ),
        }
        for split in ("calibration", "held_out")
    }
    (
        pinned_guides,
        calibration,
        frame_router,
        frame_evidence,
    ) = _load_pinned_frame(
        args.pinned_guide_embeddings.resolve(),
        args.calibration.resolve(),
    )
    caches = tuple(path.resolve() for path in (args.caches or DEFAULT_CACHE_PATHS))
    order_seeds = tuple(args.order_seeds or DEFAULT_ORDER_SEEDS)
    native_threshold = paraphrase_native_cosine_threshold(QWEN3_EMBEDDING_MODEL_ID)
    report: dict[str, Any] = {
        "schema_version": "aiidea_mrpc_current_pipeline_benchmark_v6",
        "evidence_class": "cached_qwen_current_pipeline_offline_replay",
        "routing_contract": {
            "architecture": "repaired_free_range_membership",
            "candidate_blocking": "none_within_exact_semantic_namespace",
            "candidate_locality_prior": (
                "pinned calibrated PSA routing cell with Chebyshev radius 1"
            ),
            "locality_prior_radius": DEFAULT_LOCALITY_PRIOR_RADIUS,
            "free_sentence_native_limit": PSA_SIGNATURE_CANDIDATE_LIMIT,
            "true_group_native_limit": PSA_SIGNATURE_CANDIDATE_LIMIT,
            "attractor_candidates_enabled": False,
            "verification_inventory_limit": (PSA_SIGNATURE_CANDIDATE_LIMIT),
            "fast_verification_page_size": (DEFAULT_FAST_VERIFY_PAGE_SIZE),
            "comparison_triage": {
                "schema_version": COMPARISON_TRIAGE_SCHEMA_VERSION,
                "label_independent": True,
                "native_queue_order": ("native_cosine_desc_then_psa_mad_asc"),
                "manhattan_queue_order": ("psa_mean_absolute_delta_asc_then_native_cosine_desc"),
                "joint_schedule": ("native_first_stable_round_robin_union"),
                "pre_triage_inventory_preserved": True,
                "comparison_geometry_filters_inventory": False,
                "complete_inventory_evaluated": True,
                "first_accept_stops_evaluation": False,
                "scheduling_only": True,
                "hard_membership_authorized": False,
                "automatic_promotion_enabled": False,
                "membership_action": "none",
            },
            "multiple_verified_candidate_policy": (
                "record_ambiguity_then_preserve_pretriage_native_order"
            ),
            "native_embedding_model": QWEN3_EMBEDDING_MODEL_ID,
            "native_embedding_dimension": NATIVE_EMBEDDING_DIMENSION,
            "native_cosine_threshold": native_threshold,
            "unmatched_policy": "durable_free_range_sentence",
            "minimum_group_size": 2,
            "synset_verifier_branch": (
                str(
                    wordnet_evidence["resolver"].get(
                        "resolver_id",
                        "offline_oewn_unknown_resolver",
                    )
                )
                if wordnet_evidence is not None
                else "coverage_empty_side"
            ),
            "synset_annotations_available": (wordnet_evidence is not None),
            "synset_gate_threshold": 0.5,
            "synset_group_policy": "union_of_member_senses",
            "verification_policy": ("wordnet_hard_conflict_plus_microgpt_ranked_shadow_probes"),
            "microgpt_selects_truth_label": False,
            "microgpt_training_labels_used": False,
            "require_complete_verification_evidence": False,
            "enforce_fact_conflicts": False,
            "candidate_search": (
                "global native-vector recall upper bound; not an exact "
                "replay of Neo4j PSA HNSW nomination"
            ),
            "candidate_search_fidelity": ("upper_bound_not_production_candidate_set_identity"),
            "serial_ingestion_only": True,
        },
        "wordnet_evidence": wordnet_evidence,
        "frame_evidence": frame_evidence,
        "split_policy": {
            "unit": "connected_component_of_normalized_sentence_identity",
            "seed": DEFAULT_SPLIT_SEED,
            "calibration_percent": DEFAULT_MRPC_CALIBRATION_PERCENT,
            "exact_sentence_leakage_allowed": False,
            "labels_used_as_non_excluding_tie_break_prior": True,
        },
        "split_counts": split_counts,
        "cache_reports": [],
    }
    for cache_path in caches:
        print(f"Loading complete cached evidence: {cache_path}", flush=True)
        native_by_text, cached_guides, cache_evidence = _load_cache_evidence(
            cache_path,
            pairs,
        )
        cached_guide_router = GuideStarPSARouter(
            cached_guides,
            calibration.get_star_positions(),
        )
        cache_evidence["cached_guide_frame_hash"] = cached_guide_router.get_projection_hash()
        cache_evidence["pinned_frame_used_for_routing"] = True
        (
            signatures_by_text,
            labels_by_text,
            label_distribution,
        ) = _project_representations(
            native_by_text,
            pinned_guides,
            frame_router,
            calibration,
        )
        cache_report: dict[str, Any] = {
            "evidence": cache_evidence,
            "routing_label_distribution": label_distribution,
            "held_out_pair_gate_diagnostics": _pair_gate_diagnostics(
                held_out_pairs,
                native_by_text,
                labels_by_text,
                native_threshold=native_threshold,
            ),
            "native_qwen_pair_baselines": _baseline_report(
                pairs,
                splits,
                native_by_text,
            ),
            "orders": [],
        }
        for order_seed in order_seeds:
            print(
                f"Routing {len(native_by_text)} sentences for {order_seed}",
                flush=True,
            )
            order_report = _run_order(
                pairs=pairs,
                held_out_pairs=held_out_pairs,
                native_by_text=native_by_text,
                signatures_by_text=signatures_by_text,
                labels_by_text=labels_by_text,
                wordnet_senses_by_text=wordnet_senses_by_text,
                order_seed=order_seed,
                native_threshold=native_threshold,
            )
            cache_report["orders"].append(order_report)
            metrics = order_report["held_out_metrics"]
            print(
                f"  held-out F1={metrics['f1']:.4f} "
                f"P={metrics['precision']:.4f} "
                f"R={metrics['recall']:.4f} "
                f"official F1={order_report['official_test_metrics']['f1']:.4f} "
                f"groups={order_report['router']['true_group_count']} "
                f"free={order_report['router']['free_sentence_count']}",
                flush=True,
            )
            partial = {
                **report,
                "cache_reports": [*report["cache_reports"], cache_report],
            }
            _atomic_json(
                args.output_dir / "mrpc_current_pipeline_report.json",
                partial,
            )
        cache_report["aggregate"] = _aggregate_orders(cache_report["orders"])
        cache_report["aggregate"].update(_aggregate_official_orders(cache_report["orders"]))
        report["cache_reports"].append(cache_report)
        _atomic_json(
            args.output_dir / "mrpc_current_pipeline_report.json",
            report,
        )

    report["cross_cache"] = {
        "cache_count": len(report["cache_reports"]),
        "majority_vote_f1": _mean_summary(
            [
                float(cache_report["aggregate"]["majority_vote_held_out_metrics"]["f1"])
                for cache_report in report["cache_reports"]
            ]
        ),
        "per_order_f1": _mean_summary(
            [
                float(order["held_out_metrics"]["f1"])
                for cache_report in report["cache_reports"]
                for order in cache_report["orders"]
            ]
        ),
        "majority_vote_official_test_f1": _mean_summary(
            [
                float(cache_report["aggregate"]["majority_vote_official_test_metrics"]["f1"])
                for cache_report in report["cache_reports"]
            ]
        ),
        "per_order_official_test_f1": _mean_summary(
            [
                float(order["official_test_metrics"]["f1"])
                for cache_report in report["cache_reports"]
                for order in cache_report["orders"]
            ]
        ),
    }
    report_material = {key: value for key, value in report.items() if key != "report_id"}
    report["report_id"] = (
        "mrpc-current-pipeline:"
        + hashlib.sha256(_canonical_json(report_material).encode("utf-8")).hexdigest()
    )
    output_path = args.output_dir / "mrpc_current_pipeline_report.json"
    _atomic_json(output_path, report)
    print(json.dumps(report["cross_cache"], indent=2, sort_keys=True))
    print(f"Report: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
