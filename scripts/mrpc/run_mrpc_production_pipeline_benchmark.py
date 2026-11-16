#!/usr/bin/env python3
"""Evaluate MRPC through AIidea's production paraphrase-routing funnel.

This benchmark does not replace Qwen with PSA and does not blend the two
representations. It simulates serial production ingestion:

1. Search 115D PSA group-signature centroids for 32 candidates.
2. Rerank candidates against centroid, medoid, and boundary signatures to 8.
3. Verify the remaining groups with native 4096D Qwen cosine at the production
   threshold.
4. Join the winner or create a singleton semantic group.

The evaluator reads the content-addressed SQLite caches created by the Qwen
typed field test. MRPC labels are consulted only after routing and never affect
candidate search, group construction, thresholds, or ingestion order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from consolidated_clustering_v2.routing.guide_stars import (  # noqa: E402
    GUIDE_STAR_CATALOG,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    DEFAULT_MRPC_CALIBRATION_PERCENT,
    DEFAULT_SPLIT_SEED,
    EMBEDDING_CACHE_SCHEMA_VERSION,
    QWEN_TYPED_NATIVE_DIMENSION,
    MRPCPair,
    _binary_metrics,
    compute_psa_signatures,
    fit_cosine_threshold,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    NATIVE_EMBEDDING_DIMENSION,
    PSA_SIGNATURE_CANDIDATE_LIMIT,
    PSA_SIGNATURE_DIMENSION,
    PSA_SIGNATURE_RERANK_LIMIT,
    PSA_SIGNATURE_SCHEMA_VERSION,
    QWEN3_EMBEDDING_MODEL_ID,
    paraphrase_native_cosine_threshold,
)
from mcp_servers.sentence_node_service.semantic_signature import (  # noqa: E402
    SemanticSignatureSummary,
    build_signature_summary,
    update_signature_summary,
)

DEFAULT_CACHE_PATHS = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30282117353" / "qwen_embeddings.sqlite3",
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30279973693" / "qwen_embeddings.sqlite3",
)
DEFAULT_ORDER_SEEDS = (
    "mrpc-production-order-v1",
    "mrpc-production-order-v2",
    "mrpc-production-order-v3",
    "mrpc-production-order-v4",
    "mrpc-production-order-v5",
)
PARAPHRASE_NATIVE_COSINE_THRESHOLD = paraphrase_native_cosine_threshold(QWEN3_EMBEDDING_MODEL_ID)


class BenchmarkError(RuntimeError):
    """Raised when the offline evidence cannot support a trustworthy result."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator == 0.0:
        raise BenchmarkError("cosine received a zero or invalid vector")
    return float(np.dot(left, right) / denominator)


def _mean_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": float(statistics.fmean(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "population_stddev": float(statistics.pstdev(values)),
    }


class ReadOnlyEmbeddingCache:
    """Resolve raw text to normalized Qwen vectors without endpoint access."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise BenchmarkError(f"embedding cache does not exist: {path}")
        self.path = path.resolve()
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.metadata = dict(
            self.connection.execute("SELECT key, value FROM cache_meta").fetchall()
        )
        required = {"schema_version", "model_id", "endpoint_fingerprint"}
        missing = required - set(self.metadata)
        if missing:
            raise BenchmarkError(f"cache metadata is missing {sorted(missing)}")
        if self.metadata["schema_version"] != EMBEDDING_CACHE_SCHEMA_VERSION:
            raise BenchmarkError(
                f"cache schema {self.metadata['schema_version']!r} does not match "
                f"{EMBEDDING_CACHE_SCHEMA_VERSION!r}"
            )

    def _cache_key(self, text: str) -> str:
        return _sha256_payload(
            {
                "schema_version": self.metadata["schema_version"],
                "model_id": self.metadata["model_id"],
                "endpoint_fingerprint": self.metadata["endpoint_fingerprint"],
                "namespace": "primary",
                "text": text,
            }
        )

    def get(self, text: str) -> np.ndarray:
        row = self.connection.execute(
            """
            SELECT text_sha256, dimension, vector
            FROM embeddings
            WHERE cache_key = ?
            """,
            (self._cache_key(text),),
        ).fetchone()
        if row is None:
            raise BenchmarkError(
                f"cache {self.path} has no primary vector for text {sha256_text(text)}"
            )
        text_hash, dimension, blob = row
        if text_hash != sha256_text(text):
            raise BenchmarkError("embedding cache text hash does not match requested text")
        vector = np.frombuffer(blob, dtype="<f4").copy()
        if int(dimension) != NATIVE_EMBEDDING_DIMENSION or vector.shape != (
            NATIVE_EMBEDDING_DIMENSION,
        ):
            raise BenchmarkError(f"invalid cached vector dimension {dimension!r}")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise BenchmarkError("embedding cache contains a zero or invalid vector")
        return np.asarray(vector / norm, dtype=np.float32)

    def contains(self, text: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM embeddings WHERE cache_key = ?",
            (self._cache_key(text),),
        ).fetchone()
        return row is not None

    @property
    def row_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def close(self) -> None:
        self.connection.close()


@dataclass
class GroupState:
    group_id: str
    integer_label: int
    members: list[str]
    native_centroid: np.ndarray
    signature_summary: SemanticSignatureSummary


class CandidateIndex:
    """Small interface shared by exact and HNSW candidate engines."""

    name = "abstract"

    def add_or_update(self, group: GroupState) -> None:
        raise NotImplementedError

    def query(
        self,
        signature: np.ndarray,
        *,
        limit: int,
    ) -> list[tuple[int, float]]:
        raise NotImplementedError


class ExactCandidateIndex(CandidateIndex):
    """Deterministic exact oracle for the production PSA candidate stage."""

    name = "exact"

    def __init__(self) -> None:
        self.vectors: dict[int, np.ndarray] = {}

    def add_or_update(self, group: GroupState) -> None:
        self.vectors[group.integer_label] = np.asarray(
            group.signature_summary.centroid,
            dtype=np.float32,
        )

    def query(
        self,
        signature: np.ndarray,
        *,
        limit: int,
    ) -> list[tuple[int, float]]:
        scored = [
            (label, float(np.dot(signature, centroid))) for label, centroid in self.vectors.items()
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]


class HNSWCandidateIndex(CandidateIndex):
    """hnswlib implementation of the production signature ANN stage."""

    name = "hnswlib"

    def __init__(
        self,
        *,
        max_elements: int,
        random_seed: int,
        construction_ef: int,
        search_ef: int,
        connections: int,
    ) -> None:
        try:
            import hnswlib
        except ImportError as exc:
            raise BenchmarkError(
                "hnswlib is required for --candidate-engine hnsw; "
                "run with a benchmark environment that provides hnswlib==0.8.0"
            ) from exc
        self.index = hnswlib.Index(space="cosine", dim=PSA_SIGNATURE_DIMENSION)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=construction_ef,
            M=connections,
            random_seed=random_seed,
        )
        self.index.set_ef(search_ef)
        self.count = 0
        self.labels: set[int] = set()

    def add_or_update(self, group: GroupState) -> None:
        vector = np.asarray(
            group.signature_summary.centroid,
            dtype=np.float32,
        ).reshape(1, -1)
        label = np.asarray([group.integer_label], dtype=np.int64)
        self.index.add_items(vector, label)
        if group.integer_label not in self.labels:
            self.labels.add(group.integer_label)
            self.count += 1

    def query(
        self,
        signature: np.ndarray,
        *,
        limit: int,
    ) -> list[tuple[int, float]]:
        if self.count == 0:
            return []
        count = min(limit, self.count)
        labels, distances = self.index.knn_query(
            np.asarray(signature, dtype=np.float32).reshape(1, -1),
            k=count,
        )
        return [
            (int(label), float(1.0 - distance))
            for label, distance in zip(labels[0], distances[0], strict=True)
        ]


@dataclass(frozen=True)
class RouteTrace:
    ann_group_ids: tuple[str, ...]
    reranked_group_ids: tuple[str, ...]
    native_scores: Mapping[str, float]
    winner_group_id: str | None


class ProductionPSARouter:
    """Serial in-memory realization of the production semantic routing contract."""

    def __init__(
        self,
        *,
        candidate_engine: str,
        max_elements: int,
        random_seed: int,
        exact_audit_every: int,
        hnsw_construction_ef: int,
        hnsw_search_ef: int,
        hnsw_connections: int,
    ) -> None:
        self.groups: dict[str, GroupState] = {}
        self.groups_by_label: dict[int, GroupState] = {}
        self.assignment: dict[str, str] = {}
        self.next_integer_label = 0
        self.exact_audit_every = exact_audit_every
        self.query_count = 0
        self.audit_overlap: list[float] = []
        self.audit_best_present: list[float] = []
        if candidate_engine == "exact":
            self.index: CandidateIndex = ExactCandidateIndex()
        elif candidate_engine == "hnsw":
            self.index = HNSWCandidateIndex(
                max_elements=max_elements,
                random_seed=random_seed,
                construction_ef=hnsw_construction_ef,
                search_ef=hnsw_search_ef,
                connections=hnsw_connections,
            )
        else:
            raise BenchmarkError(f"unsupported candidate engine {candidate_engine!r}")

    @staticmethod
    def _signature_rerank_score(
        signature: np.ndarray,
        group: GroupState,
    ) -> float:
        representatives = (
            group.signature_summary.centroid,
            group.signature_summary.medoid.signature,
            *(prototype.signature for prototype in group.signature_summary.boundary_prototypes),
        )
        return max(
            float(np.dot(signature, np.asarray(representative, dtype=np.float32)))
            for representative in representatives
        )

    def _exact_candidates(
        self,
        signature: np.ndarray,
        *,
        limit: int,
    ) -> list[tuple[int, float]]:
        scored = [
            (
                group.integer_label,
                float(
                    np.dot(
                        signature,
                        np.asarray(group.signature_summary.centroid, dtype=np.float32),
                    )
                ),
            )
            for group in self.groups.values()
        ]
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]

    def _audit_ann(
        self,
        signature: np.ndarray,
        ann_candidates: Sequence[tuple[int, float]],
    ) -> None:
        if (
            self.index.name == "exact"
            or self.exact_audit_every <= 0
            or self.query_count % self.exact_audit_every
        ):
            return
        exact = self._exact_candidates(
            signature,
            limit=min(PSA_SIGNATURE_CANDIDATE_LIMIT, len(self.groups)),
        )
        if not exact:
            return
        exact_labels = {label for label, _score in exact}
        ann_labels = {label for label, _score in ann_candidates}
        self.audit_overlap.append(len(exact_labels & ann_labels) / len(exact_labels))
        self.audit_best_present.append(float(exact[0][0] in ann_labels))

    def route(
        self,
        *,
        member_id: str,
        native: np.ndarray,
        signature: np.ndarray,
    ) -> RouteTrace:
        if member_id in self.assignment:
            raise BenchmarkError(f"member {member_id!r} was ingested twice")
        self.query_count += 1
        ann = self.index.query(
            signature,
            limit=min(PSA_SIGNATURE_CANDIDATE_LIMIT, len(self.groups)),
        )
        self._audit_ann(signature, ann)

        candidate_rows = []
        for integer_label, ann_score in ann:
            group = self.groups_by_label[integer_label]
            signature_score = self._signature_rerank_score(signature, group)
            candidate_rows.append((group, ann_score, signature_score))
        candidate_rows.sort(
            key=lambda row: (
                -row[2],
                -row[1],
                row[0].group_id,
            )
        )
        reranked = candidate_rows[:PSA_SIGNATURE_RERANK_LIMIT]

        verified = []
        native_scores: dict[str, float] = {}
        for group, ann_score, signature_score in reranked:
            native_score = _cosine(group.native_centroid, native)
            native_scores[group.group_id] = native_score
            if native_score >= PARAPHRASE_NATIVE_COSINE_THRESHOLD:
                verified.append((group, native_score, signature_score, ann_score))
        verified.sort(
            key=lambda row: (
                -row[1],
                -row[2],
                row[0].group_id,
            )
        )
        winner = verified[0][0] if verified else None

        if winner is None:
            group_id = f"mrpc-group:{member_id}"
            summary = build_signature_summary(
                [(member_id, signature)],
                expected_dimension=PSA_SIGNATURE_DIMENSION,
            )
            winner = GroupState(
                group_id=group_id,
                integer_label=self.next_integer_label,
                members=[member_id],
                native_centroid=np.asarray(native, dtype=np.float32),
                signature_summary=summary,
            )
            self.next_integer_label += 1
            self.groups[group_id] = winner
            self.groups_by_label[winner.integer_label] = winner
        else:
            previous_count = winner.signature_summary.member_count
            winner.native_centroid = np.asarray(
                (winner.native_centroid * previous_count + np.asarray(native, dtype=np.float32))
                / (previous_count + 1),
                dtype=np.float32,
            )
            winner.signature_summary = update_signature_summary(
                winner.signature_summary,
                [(member_id, signature)],
            )
            winner.members.append(member_id)

        self.assignment[member_id] = winner.group_id
        self.index.add_or_update(winner)
        return RouteTrace(
            ann_group_ids=tuple(self.groups_by_label[label].group_id for label, _score in ann),
            reranked_group_ids=tuple(row[0].group_id for row in reranked),
            native_scores=native_scores,
            winner_group_id=(verified[0][0].group_id if verified else None),
        )

    def audit_report(self) -> dict[str, Any]:
        return {
            "candidate_engine": self.index.name,
            "exact_audit_every": self.exact_audit_every,
            "audited_query_count": len(self.audit_overlap),
            "top32_set_overlap": _mean_summary(self.audit_overlap),
            "exact_best_present_rate": (
                float(statistics.fmean(self.audit_best_present))
                if self.audit_best_present
                else None
            ),
        }


def _stable_member_id(text: str) -> str:
    return f"mrpc-sentence:{sha256_text(text)}"


def _ordered_texts(texts: Iterable[str], seed: str) -> list[str]:
    return sorted(
        texts,
        key=lambda text: (
            hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).hexdigest(),
            text,
        ),
    )


def _rank(group_id: str, values: Sequence[str]) -> int | None:
    try:
        return values.index(group_id) + 1
    except ValueError:
        return None


def _stage_rates(
    events: Sequence[Mapping[str, Any]],
    *,
    label: bool,
) -> dict[str, Any]:
    selected = [event for event in events if bool(event["label"]) is label]
    if not selected:
        return {"count": 0}

    def rate(field: str) -> float:
        return sum(bool(event[field]) for event in selected) / len(selected)

    return {
        "count": len(selected),
        "target_recall_at_32": rate("target_at_32"),
        "target_recall_at_8": rate("target_at_8"),
        "target_native_gate_rate": rate("target_passed_native_gate"),
        "target_selected_rate": rate("target_selected"),
    }


def _load_cache_evidence(
    cache_path: Path,
    pairs: Sequence[MRPCPair],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    cache = ReadOnlyEmbeddingCache(cache_path)
    try:
        texts = sorted(
            {text for pair in pairs for text in (pair.sentence1, pair.sentence2)},
            key=lambda text: (sha256_text(text), text),
        )
        missing_texts = [text for text in texts if not cache.contains(text)]
        if missing_texts:
            sample_hashes = [sha256_text(text) for text in missing_texts[:5]]
            raise BenchmarkError(
                f"cache {cache.path} is missing {len(missing_texts)} of "
                f"{len(texts)} source-joined MRPC sentence vectors; first "
                f"missing SHA-256 values: {sample_hashes}"
            )
        native = {text: cache.get(text) for text in texts}
        guide_embeddings = np.stack([cache.get(star.text) for star in GUIDE_STAR_CATALOG])
        if guide_embeddings.shape != (
            PSA_SIGNATURE_DIMENSION,
            NATIVE_EMBEDDING_DIMENSION,
        ):
            raise BenchmarkError("guide-star embedding matrix has the wrong shape")
        evidence = {
            "cache_path": str(cache.path),
            "cache_sha256": hashlib.sha256(cache.path.read_bytes()).hexdigest(),
            "cache_row_count": cache.row_count,
            "cache_schema_version": cache.metadata["schema_version"],
            "model_id": cache.metadata["model_id"],
            "endpoint_fingerprint": cache.metadata["endpoint_fingerprint"],
            "mrpc_unique_sentence_count": len(texts),
            "mrpc_covered_sentence_count": len(native),
            "guide_star_count": len(guide_embeddings),
            "native_dimension": NATIVE_EMBEDDING_DIMENSION,
        }
        return native, guide_embeddings, evidence
    finally:
        cache.close()


def _baseline_report(
    pairs: Sequence[MRPCPair],
    splits: Mapping[int, str],
    native_by_text: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    cosines = np.asarray(
        [
            float(np.dot(native_by_text[pair.sentence1], native_by_text[pair.sentence2]))
            for pair in pairs
        ],
        dtype=np.float32,
    )
    labels = np.asarray([pair.is_paraphrase for pair in pairs], dtype=bool)
    calibration_mask = np.asarray(
        [splits[pair.pair_id] == "calibration" for pair in pairs],
        dtype=bool,
    )
    held_out_mask = ~calibration_mask
    threshold, calibration_metrics = fit_cosine_threshold(
        cosines[calibration_mask],
        labels[calibration_mask],
    )
    return {
        "calibrated_pair_cosine": {
            "threshold": float(threshold),
            "threshold_selected_on": "calibration_only",
            "calibration": calibration_metrics,
            "held_out": _binary_metrics(
                labels[held_out_mask],
                cosines[held_out_mask] >= threshold,
            ),
        },
        "production_0_77_pair_cosine": {
            "threshold": PARAPHRASE_NATIVE_COSINE_THRESHOLD,
            "held_out": _binary_metrics(
                labels[held_out_mask],
                cosines[held_out_mask] >= PARAPHRASE_NATIVE_COSINE_THRESHOLD,
            ),
        },
    }


def _run_order(
    *,
    pairs: Sequence[MRPCPair],
    held_out_pairs: Sequence[MRPCPair],
    native_by_text: Mapping[str, np.ndarray],
    signatures_by_text: Mapping[str, np.ndarray],
    order_seed: str,
    candidate_engine: str,
    exact_audit_every: int,
    hnsw_construction_ef: int,
    hnsw_search_ef: int,
    hnsw_connections: int,
) -> dict[str, Any]:
    texts = {text for pair in pairs for text in (pair.sentence1, pair.sentence2)}
    random_seed = int(hashlib.sha256(order_seed.encode("utf-8")).hexdigest()[:8], 16)
    router = ProductionPSARouter(
        candidate_engine=candidate_engine,
        max_elements=len(texts),
        random_seed=random_seed,
        exact_audit_every=exact_audit_every,
        hnsw_construction_ef=hnsw_construction_ef,
        hnsw_search_ef=hnsw_search_ef,
        hnsw_connections=hnsw_connections,
    )

    incident: dict[str, list[tuple[MRPCPair, str]]] = {}
    held_out_pair_ids = {pair.pair_id for pair in held_out_pairs}
    for pair in held_out_pairs:
        if pair.sentence1 == pair.sentence2:
            continue
        incident.setdefault(pair.sentence1, []).append((pair, pair.sentence2))
        incident.setdefault(pair.sentence2, []).append((pair, pair.sentence1))

    stage_events: dict[int, dict[str, Any]] = {}
    started = time.monotonic()
    for text in _ordered_texts(texts, order_seed):
        member_id = _stable_member_id(text)
        trace = router.route(
            member_id=member_id,
            native=native_by_text[text],
            signature=signatures_by_text[text],
        )
        for pair, other_text in incident.get(text, ()):
            other_member_id = _stable_member_id(other_text)
            target_group_id = router.assignment.get(other_member_id)
            if target_group_id is None:
                continue
            native_score = trace.native_scores.get(target_group_id)
            stage_events[pair.pair_id] = {
                "pair_id": pair.pair_id,
                "label": pair.is_paraphrase,
                "query_sentence_sha256": sha256_text(text),
                "target_sentence_sha256": sha256_text(other_text),
                "target_ann_rank": _rank(target_group_id, trace.ann_group_ids),
                "target_rerank_rank": _rank(target_group_id, trace.reranked_group_ids),
                "target_native_similarity": native_score,
                "target_at_32": target_group_id in trace.ann_group_ids,
                "target_at_8": target_group_id in trace.reranked_group_ids,
                "target_passed_native_gate": (
                    native_score is not None and native_score >= PARAPHRASE_NATIVE_COSINE_THRESHOLD
                ),
                "target_selected": trace.winner_group_id == target_group_id,
            }

    labels = []
    predicted = []
    pair_predictions = []
    identity_shortcuts = 0
    for pair in held_out_pairs:
        left = _stable_member_id(pair.sentence1)
        right = _stable_member_id(pair.sentence2)
        same_group = router.assignment[left] == router.assignment[right]
        labels.append(pair.is_paraphrase)
        predicted.append(same_group)
        identity_shortcut = pair.sentence1 == pair.sentence2
        identity_shortcuts += int(identity_shortcut)
        pair_predictions.append(
            {
                "pair_id": pair.pair_id,
                "label": pair.is_paraphrase,
                "predicted_paraphrase": same_group,
                "identity_shortcut": identity_shortcut,
            }
        )

    missing_stage_events = held_out_pair_ids - set(stage_events)
    non_identity_missing = [
        pair_id
        for pair_id in missing_stage_events
        if next(pair for pair in held_out_pairs if pair.pair_id == pair_id).sentence1
        != next(pair for pair in held_out_pairs if pair.pair_id == pair_id).sentence2
    ]
    if non_identity_missing:
        raise BenchmarkError(
            f"order {order_seed!r} omitted {len(non_identity_missing)} non-identity stage events"
        )

    group_sizes = [len(group.members) for group in router.groups.values()]
    events = list(stage_events.values())
    return {
        "order_seed": order_seed,
        "candidate_engine": router.index.name,
        "elapsed_seconds": time.monotonic() - started,
        "ingested_unique_sentence_count": len(texts),
        "final_group_count": len(router.groups),
        "group_size": {
            "mean": float(statistics.fmean(group_sizes)),
            "maximum": max(group_sizes),
            "singleton_count": sum(size == 1 for size in group_sizes),
        },
        "identity_shortcut_pair_count": identity_shortcuts,
        "held_out_metrics": _binary_metrics(
            np.asarray(labels, dtype=bool),
            np.asarray(predicted, dtype=bool),
        ),
        "positive_stage": _stage_rates(events, label=True),
        "negative_stage": _stage_rates(events, label=False),
        "ann_audit": router.audit_report(),
        "pair_predictions": pair_predictions,
        "stage_events": sorted(events, key=lambda event: int(event["pair_id"])),
    }


def _aggregate_orders(order_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = ("precision", "recall", "f1", "accuracy")
    aggregate = {
        metric: _mean_summary(
            [
                float(report["held_out_metrics"][metric])
                for report in order_reports
                if report["held_out_metrics"][metric] is not None
            ]
        )
        for metric in metric_names
    }
    pair_votes: dict[int, list[bool]] = {}
    pair_labels: dict[int, bool] = {}
    for report in order_reports:
        for row in report["pair_predictions"]:
            pair_id = int(row["pair_id"])
            pair_votes.setdefault(pair_id, []).append(bool(row["predicted_paraphrase"]))
            pair_labels[pair_id] = bool(row["label"])
    ordered_ids = sorted(pair_votes)
    consensus = np.asarray(
        [sum(pair_votes[pair_id]) * 2 >= len(pair_votes[pair_id]) for pair_id in ordered_ids],
        dtype=bool,
    )
    labels = np.asarray([pair_labels[pair_id] for pair_id in ordered_ids], dtype=bool)
    aggregate["majority_vote_held_out_metrics"] = _binary_metrics(labels, consensus)
    aggregate["order_count"] = len(order_reports)
    aggregate["order_sensitive_pair_count"] = sum(
        len(set(votes)) > 1 for votes in pair_votes.values()
    )
    return aggregate


def _validate_contract() -> None:
    if NATIVE_EMBEDDING_DIMENSION != QWEN_TYPED_NATIVE_DIMENSION:
        raise BenchmarkError("field-test and production native dimensions disagree")
    if len(GUIDE_STAR_CATALOG) != PSA_SIGNATURE_DIMENSION:
        raise BenchmarkError("guide-star catalog and routing signature dimensions disagree")
    if PSA_SIGNATURE_CANDIDATE_LIMIT != 32 or PSA_SIGNATURE_RERANK_LIMIT != 8:
        raise BenchmarkError("production candidate funnel is not the expected 32-to-8 contract")


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
        default=REPOSITORY_ROOT / "reports" / "mrpc_production_pipeline",
    )
    parser.add_argument(
        "--candidate-engine",
        choices=("hnsw", "exact"),
        default="hnsw",
    )
    parser.add_argument(
        "--order-seed",
        action="append",
        dest="order_seeds",
        help="Deterministic serial-ingestion order; repeat to test order sensitivity",
    )
    parser.add_argument("--exact-audit-every", type=int, default=25)
    parser.add_argument("--hnsw-construction-ef", type=int, default=200)
    parser.add_argument("--hnsw-search-ef", type=int, default=128)
    parser.add_argument("--hnsw-connections", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_contract()
    pairs = load_mrpc_pairs(args.mrpc_dir.resolve())
    splits = split_mrpc_by_sentence_component(pairs)
    held_out_pairs = [pair for pair in pairs if splits[pair.pair_id] == "held_out"]
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
    caches = tuple(path.resolve() for path in (args.caches or DEFAULT_CACHE_PATHS))
    order_seeds = tuple(args.order_seeds or DEFAULT_ORDER_SEEDS)
    if not order_seeds:
        raise BenchmarkError("at least one ingestion order is required")

    report: dict[str, Any] = {
        "schema_version": "aiidea_mrpc_production_pipeline_benchmark_v1",
        "evidence_class": "cached_qwen_production_pipeline_offline_replay",
        "routing_contract": {
            "signature_dimension": PSA_SIGNATURE_DIMENSION,
            "signature_schema_version": PSA_SIGNATURE_SCHEMA_VERSION,
            "candidate_limit": PSA_SIGNATURE_CANDIDATE_LIMIT,
            "rerank_limit": PSA_SIGNATURE_RERANK_LIMIT,
            "native_dimension": NATIVE_EMBEDDING_DIMENSION,
            "native_cosine_threshold": PARAPHRASE_NATIVE_COSINE_THRESHOLD,
            "composition": (
                "PSA HNSW candidate routing -> bounded signature representative "
                "rerank -> native Qwen verification"
            ),
            "qwen_psa_weighted_fusion_used": False,
        },
        "split_policy": {
            "unit": "connected_component_of_normalized_sentence_identity",
            "seed": DEFAULT_SPLIT_SEED,
            "calibration_percent": DEFAULT_MRPC_CALIBRATION_PERCENT,
            "exact_sentence_leakage_allowed": False,
            "labels_used_during_routing": False,
            "all_sentences_ingested_as_unlabeled_candidate_distractors": True,
        },
        "split_counts": split_counts,
        "candidate_engine_request": args.candidate_engine,
        "hnsw_parameters": (
            {
                "implementation": "hnswlib",
                "version": "0.8.0",
                "construction_ef": args.hnsw_construction_ef,
                "search_ef": args.hnsw_search_ef,
                "connections": args.hnsw_connections,
            }
            if args.candidate_engine == "hnsw"
            else None
        ),
        "serial_ingestion_only": True,
        "cache_reports": [],
    }

    for cache_path in caches:
        print(f"Loading complete cached evidence: {cache_path}", flush=True)
        native_by_text, guide_embeddings, evidence = _load_cache_evidence(
            cache_path,
            pairs,
        )
        texts = sorted(native_by_text, key=lambda text: (sha256_text(text), text))
        signatures = compute_psa_signatures(
            np.stack([native_by_text[text] for text in texts]),
            guide_embeddings,
        )
        signatures_by_text = {text: signatures[index] for index, text in enumerate(texts)}
        cache_report: dict[str, Any] = {
            "evidence": evidence,
            "native_qwen_pair_baselines": _baseline_report(
                pairs,
                splits,
                native_by_text,
            ),
            "orders": [],
        }
        for order_seed in order_seeds:
            print(
                f"Routing {len(texts)} sentences with {args.candidate_engine} for {order_seed}",
                flush=True,
            )
            order_report = _run_order(
                pairs=pairs,
                held_out_pairs=held_out_pairs,
                native_by_text=native_by_text,
                signatures_by_text=signatures_by_text,
                order_seed=order_seed,
                candidate_engine=args.candidate_engine,
                exact_audit_every=args.exact_audit_every,
                hnsw_construction_ef=args.hnsw_construction_ef,
                hnsw_search_ef=args.hnsw_search_ef,
                hnsw_connections=args.hnsw_connections,
            )
            cache_report["orders"].append(order_report)
            print(
                f"  F1={order_report['held_out_metrics']['f1']:.4f} "
                f"P={order_report['held_out_metrics']['precision']} "
                f"R={order_report['held_out_metrics']['recall']:.4f} "
                f"groups={order_report['final_group_count']}",
                flush=True,
            )
            partial = {
                **report,
                "cache_reports": [*report["cache_reports"], cache_report],
            }
            _atomic_json(args.output_dir / "mrpc_production_pipeline_report.json", partial)
        cache_report["aggregate"] = _aggregate_orders(cache_report["orders"])
        report["cache_reports"].append(cache_report)
        _atomic_json(args.output_dir / "mrpc_production_pipeline_report.json", report)

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
    }
    report_material = {key: value for key, value in report.items() if key != "report_id"}
    report["report_id"] = (
        "mrpc-production-pipeline:"
        + hashlib.sha256(_canonical_json(report_material).encode("utf-8")).hexdigest()
    )
    output_path = args.output_dir / "mrpc_production_pipeline_report.json"
    _atomic_json(output_path, report)
    print(json.dumps(report["cross_cache"], indent=2, sort_keys=True))
    print(f"Report: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
