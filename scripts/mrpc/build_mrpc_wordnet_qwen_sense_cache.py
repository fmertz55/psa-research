#!/usr/bin/env python3
"""Build MRPC WordNet senses from a scoped set of Qwen synset vectors.

This runner keeps the application offline. It:

1. extracts noun/verb lemma and POS pairs from the 3,393 MRPC sentences;
2. drops lemmas that occur in only one sentence because the paraphrase gate
   can never compare them with the same lemma on another sentence;
3. reproduces the production WordNet importer's exact example/gloss points;
4. embeds only those points with Qwen3-Embedding-8B;
5. converts raw cosine to Neo4j's bounded cosine-index score and resolves each
   context at the production 0.70 index-score threshold; and
6. emits the versioned sentence-to-sense cache consumed by the offline MRPC
   current-pipeline benchmark.

Production queries a global approximate HNSW index before applying lemma/POS
filters. This offline test scores the complete relevant lemma/POS subset
directly. The embedding and point-text contracts match production, while the
candidate retrieval algorithm is deliberately exact and scoped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from scripts.research.mrpc.build_mrpc_wordnet_sense_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    OEWN_VERSION,
    WORD_RE,
    OfflineOEWNResolver,
    OfflineWordNetError,
    _atomic_json,
    _load_mrpc_texts,
    _sha256_file,
    _sha256_text,
)

from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    NATIVE_EMBEDDING_DIMENSION,
)

MANIFEST_SCHEMA_VERSION = "aiidea_mrpc_wordnet_qwen_manifest_v1"
REPORT_SCHEMA_VERSION = "aiidea_mrpc_wordnet_qwen_wsd_report_v1"
RESOLVER_ID = "oewn_2024_qwen_synset_example_exact_lemma_pos_v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
DEFAULT_WN_DATA = REPOSITORY_ROOT / "temp" / "wordnet"
DEFAULT_NLTK_DATA = REPOSITORY_ROOT / "temp" / "nltk_data"
DEFAULT_MRPC_DIR = REPOSITORY_ROOT / "app" / "utils" / "msrpc_split"
DEFAULT_SENTENCE_CACHE = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "qwen_embeddings.sqlite3"
)
DEFAULT_SENTENCE_ENDPOINT = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "endpoint.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "temp" / "mrpc_wordnet_qwen"
DEFAULT_COSINE_THRESHOLD = 0.70
DEFAULT_MINIMUM_LEMMA_SENTENCE_COUNT = 2
DEFAULT_COMPATIBILITY_SAMPLE_SIZE = 8
EXPECTED_MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
EMBEDDING_CACHE_SCHEMA_VERSION = "qwen_embedding_cache_v1"
NEO4J_COSINE_SCORE_CONTRACT = "neo4j_vector_2_0_cosine_v1"
DIAGNOSTIC_SCORE_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


class ScopedQwenWsdError(RuntimeError):
    """Raised when scoped Qwen WSD evidence is incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class SynsetPoint:
    example_id: str
    synset_id: str
    text: str
    kind: str


@dataclass(frozen=True, slots=True)
class QwenSenseAssignment:
    lemma: str
    pos: str
    synset_id: str
    score: float
    raw_cosine: float
    example_id: str
    example_kind: str
    candidate_synset_count: int
    candidate_example_count: int
    resolution_method: str = "qwen_synset_example_neo4j_cosine_score"


@dataclass(frozen=True, slots=True)
class ScopedEmbeddingInput:
    key: str
    text: str
    purpose: str


@dataclass(frozen=True, slots=True)
class ScopedInventory:
    sentence_pairs: Mapping[str, tuple[tuple[str, str], ...]]
    lemma_sentence_counts: Mapping[str, int]
    relevant_pairs: tuple[tuple[str, str], ...]
    candidate_synsets: Mapping[tuple[str, str], tuple[str, ...]]
    points_by_synset: Mapping[str, tuple[SynsetPoint, ...]]

    @property
    def points(self) -> tuple[SynsetPoint, ...]:
        return tuple(
            point
            for synset_id in sorted(self.points_by_synset)
            for point in self.points_by_synset[synset_id]
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalized(vector: np.ndarray, *, context: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    if value.shape != (NATIVE_EMBEDDING_DIMENSION,) or not np.isfinite(value).all():
        raise ScopedQwenWsdError(f"{context} is not a finite 4,096D vector")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm == 0.0:
        raise ScopedQwenWsdError(f"{context} is a zero or invalid vector")
    return np.asarray(value / norm, dtype=np.float32)


def _neo4j_cosine_score(raw_cosine: float) -> float:
    """Match the bounded cosine score returned by Neo4j vector indexes."""

    if not math.isfinite(raw_cosine):
        raise ScopedQwenWsdError("raw cosine score is not finite")
    return min(1.0, max(0.0, (1.0 + raw_cosine) / 2.0))


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    probabilities = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    scores = np.asarray(values, dtype=np.float64)
    return {
        f"{probability:g}": float(np.quantile(scores, probability)) for probability in probabilities
    }


def _extract_sentence_pairs(
    texts: Sequence[str],
    tagged_sentences: Sequence[Sequence[tuple[str, str]]],
    resolver: OfflineOEWNResolver,
) -> tuple[
    dict[str, tuple[tuple[str, str], ...]],
    Counter[str],
]:
    if len(texts) != len(tagged_sentences):
        raise ScopedQwenWsdError("POS tagging did not preserve sentence count")
    by_text: dict[str, tuple[tuple[str, str], ...]] = {}
    lemma_sentence_counts: Counter[str] = Counter()
    for text, tagged in zip(texts, tagged_sentences, strict=True):
        pairs: set[tuple[str, str]] = set()
        for token, tag in tagged:
            pos = "n" if str(tag).startswith("NN") else "v" if str(tag).startswith("VB") else None
            if pos is None or len(token) <= 2 or not token.isalpha():
                continue
            lemma = resolver._lemma(token, pos)
            if lemma:
                pairs.add((lemma, pos))
        ordered = tuple(sorted(pairs))
        by_text[text] = ordered
        lemma_sentence_counts.update({lemma for lemma, _pos in ordered})
    return by_text, lemma_sentence_counts


def _production_points(synset: Any) -> tuple[SynsetPoint, ...]:
    synset_id = str(synset.id)
    examples = [
        str(example).strip()
        for example in (synset.examples() or [])
        if example and str(example).strip()
    ]
    if examples:
        return tuple(
            SynsetPoint(
                example_id=f"{synset_id}#ex{index}",
                synset_id=synset_id,
                text=text,
                kind="example",
            )
            for index, text in enumerate(examples)
        )
    definition = str(synset.definition() or "").strip()
    if not definition:
        return ()
    lemmas = [str(lemma).lower() for lemma in (synset.lemmas() or []) if lemma]
    primary = lemmas[0] if lemmas else ""
    text = f"{primary}: {definition}" if primary else definition
    return (
        SynsetPoint(
            example_id=f"{synset_id}#gloss",
            synset_id=synset_id,
            text=text,
            kind="gloss",
        ),
    )


def _build_inventory(
    *,
    texts: Sequence[str],
    tagged_sentences: Sequence[Sequence[tuple[str, str]]],
    resolver: OfflineOEWNResolver,
    minimum_lemma_sentence_count: int,
) -> ScopedInventory:
    sentence_pairs, lemma_sentence_counts = _extract_sentence_pairs(
        texts,
        tagged_sentences,
        resolver,
    )
    relevant_pairs = tuple(
        sorted(
            {
                pair
                for pairs in sentence_pairs.values()
                for pair in pairs
                if lemma_sentence_counts[pair[0]] >= minimum_lemma_sentence_count
            }
        )
    )
    synset_objects: dict[str, Any] = {}
    candidate_synsets: dict[tuple[str, str], tuple[str, ...]] = {}
    for lemma, pos in relevant_pairs:
        candidates = tuple(resolver._synsets(lemma, pos))
        ids = tuple(sorted({str(candidate.id) for candidate in candidates}))
        candidate_synsets[(lemma, pos)] = ids
        for candidate in candidates:
            synset_objects[str(candidate.id)] = candidate
    points_by_synset = {
        synset_id: _production_points(synset)
        for synset_id, synset in sorted(synset_objects.items())
    }
    points_by_synset = {
        synset_id: points for synset_id, points in points_by_synset.items() if points
    }
    return ScopedInventory(
        sentence_pairs=sentence_pairs,
        lemma_sentence_counts=dict(lemma_sentence_counts),
        relevant_pairs=relevant_pairs,
        candidate_synsets=candidate_synsets,
        points_by_synset=points_by_synset,
    )


def _inventory_manifest(
    *,
    inventory: ScopedInventory,
    mrpc_evidence: Mapping[str, Any],
    wordnet: Any,
    database_path: Path,
    minimum_lemma_sentence_count: int,
) -> dict[str, Any]:
    points = inventory.points
    unique_text_count = len({point.text for point in points})
    body = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selection_policy": {
            "parts_of_speech": ["noun", "verb"],
            "minimum_lemma_sentence_count": minimum_lemma_sentence_count,
            "reason": (
                "the production paraphrase gate only evaluates lemmas shared by two candidate sides"
            ),
        },
        "wordnet": {
            "lexicon": OEWN_VERSION,
            "synset_count": len(wordnet.synsets()),
            "database_path": str(database_path.resolve()),
            "database_sha256": _sha256_file(database_path),
        },
        "mrpc": dict(mrpc_evidence),
        "counts": {
            "sentence_count": len(inventory.sentence_pairs),
            "sentences_with_relevant_lemmas": sum(
                any(
                    inventory.lemma_sentence_counts[lemma] >= minimum_lemma_sentence_count
                    for lemma, _pos in pairs
                )
                for pairs in inventory.sentence_pairs.values()
            ),
            "gate_relevant_lemma_count": len({lemma for lemma, _pos in inventory.relevant_pairs}),
            "gate_relevant_lemma_pos_count": len(inventory.relevant_pairs),
            "candidate_synset_count": len(inventory.points_by_synset),
            "synset_example_point_count": len(points),
            "unique_embedding_text_count": unique_text_count,
        },
        "candidate_synsets": [
            {
                "lemma": lemma,
                "pos": pos,
                "synset_ids": list(inventory.candidate_synsets[(lemma, pos)]),
            }
            for lemma, pos in inventory.relevant_pairs
        ],
        "points": [asdict(point) for point in points],
    }
    return {
        **body,
        "manifest_sha256": _sha256_payload(body),
    }


def _read_endpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScopedQwenWsdError(f"endpoint metadata does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScopedQwenWsdError("endpoint metadata must be an object")
    return value


class ReadOnlyScopedEmbeddingCache:
    """Read endpoint-bound vectors without contacting the original endpoint."""

    def __init__(self, path: Path) -> None:
        if not path.is_file():
            raise ScopedQwenWsdError(f"scoped embedding cache does not exist: {path}")
        self.path = path.resolve()
        self.connection = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
        )
        self.metadata = dict(
            self.connection.execute("SELECT key, value FROM cache_meta").fetchall()
        )
        required = {
            "schema_version",
            "model_id",
            "endpoint_fingerprint",
        }
        missing = required - set(self.metadata)
        if missing:
            raise ScopedQwenWsdError(f"scoped cache metadata is missing {sorted(missing)}")
        if self.metadata["schema_version"] != EMBEDDING_CACHE_SCHEMA_VERSION:
            raise ScopedQwenWsdError("scoped cache schema does not match the Qwen cache contract")
        if self.metadata["model_id"] != DEFAULT_MODEL_ID:
            raise ScopedQwenWsdError(f"scoped cache uses {self.metadata['model_id']!r}")

    def get(
        self,
        text: str,
        *,
        namespace: str = "synset_example",
    ) -> np.ndarray | None:
        cache_key = _sha256_payload(
            {
                "schema_version": self.metadata["schema_version"],
                "model_id": self.metadata["model_id"],
                "endpoint_fingerprint": self.metadata["endpoint_fingerprint"],
                "namespace": namespace,
                "text": text,
            }
        )
        row = self.connection.execute(
            """
            SELECT text_sha256, dimension, vector
            FROM embeddings
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        text_hash, dimension, blob = row
        if text_hash != _sha256_text(text):
            raise ScopedQwenWsdError("scoped cache text hash does not match requested text")
        vector = np.frombuffer(blob, dtype="<f4").copy()
        if int(dimension) != NATIVE_EMBEDDING_DIMENSION or vector.shape != (
            NATIVE_EMBEDDING_DIMENSION,
        ):
            raise ScopedQwenWsdError(f"scoped cache contains invalid dimension {dimension!r}")
        return _normalized(
            vector,
            context=f"scoped cache text {_sha256_text(text)}",
        )

    @property
    def row_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()


def _read_replay_source(
    *,
    source_sense_cache_path: Path,
    scoped_manifest_path: Path,
    sentence_cache_path: Path,
    synset_vector_cache_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = json.loads(source_sense_cache_path.read_text(encoding="utf-8"))
    scoped_manifest = json.loads(scoped_manifest_path.read_text(encoding="utf-8"))
    if source.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ScopedQwenWsdError("replay source sense-cache schema is not supported")
    if scoped_manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ScopedQwenWsdError("replay scoped-manifest schema is not supported")
    expected_manifest_sha = source.get("scoped_manifest", {}).get("sha256")
    if _sha256_file(scoped_manifest_path) != expected_manifest_sha:
        raise ScopedQwenWsdError("replay scoped manifest does not match source evidence")
    embedding_evidence = source.get("embedding_evidence") or {}
    expected_sentence_sha = embedding_evidence.get("sentence_cache", {}).get("sha256")
    expected_synset_sha = embedding_evidence.get("synset_vector_cache", {}).get("sha256")
    if _sha256_file(sentence_cache_path) != expected_sentence_sha:
        raise ScopedQwenWsdError("MRPC sentence cache does not match replay source evidence")
    if _sha256_file(synset_vector_cache_path) != expected_synset_sha:
        raise ScopedQwenWsdError("synset vector cache does not match replay source evidence")
    model = source.get("resolver", {}).get("model") or {}
    if (
        model.get("repository") != DEFAULT_MODEL_ID
        or model.get("revision") != EXPECTED_MODEL_REVISION
    ):
        raise ScopedQwenWsdError("replay source does not use the pinned Qwen model revision")
    return source, scoped_manifest


def _validate_replay_inventory(
    generated: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    for key in ("counts", "candidate_synsets", "points"):
        if generated.get(key) != source.get(key):
            raise ScopedQwenWsdError(f"local OEWN inventory does not match replay {key}")


def _validate_model_lineage(
    *,
    prior_endpoint: Mapping[str, Any],
    repository: str | None,
    revision: str | None,
) -> dict[str, str]:
    prior_repository = str(prior_endpoint.get("repository") or "").strip()
    prior_revision = str(prior_endpoint.get("revision") or "").strip()
    current_repository = str(repository or "").strip()
    current_revision = str(revision or "").strip()
    if prior_repository != DEFAULT_MODEL_ID:
        raise ScopedQwenWsdError(f"MRPC sentence cache endpoint used {prior_repository!r}")
    if current_repository != DEFAULT_MODEL_ID:
        raise ScopedQwenWsdError(f"synset endpoint uses {current_repository!r}")
    if (
        not prior_revision
        or not current_revision
        or prior_revision != current_revision
        or current_revision != EXPECTED_MODEL_REVISION
    ):
        raise ScopedQwenWsdError(
            "sentence and synset endpoints do not share the pinned Qwen revision"
        )
    return {
        "repository": current_repository,
        "revision": current_revision,
    }


def _compatibility_check(
    *,
    texts: Sequence[str],
    sentence_vectors: Mapping[str, np.ndarray],
    client: Any,
    sample_size: int,
) -> dict[str, Any]:
    sample = tuple(texts[: min(sample_size, len(texts))])
    inputs = [
        ScopedEmbeddingInput(
            key=f"compatibility:{_sha256_text(text)}",
            text=text,
            purpose="cross_endpoint_vector_compatibility",
        )
        for text in sample
    ]
    fresh = client.embed_many(inputs, namespace="compatibility")
    comparisons = []
    for text, item in zip(sample, inputs, strict=True):
        cached = _normalized(
            sentence_vectors[text],
            context=f"cached sentence {_sha256_text(text)}",
        )
        current = _normalized(
            fresh[item.key],
            context=f"fresh sentence {_sha256_text(text)}",
        )
        comparisons.append(
            {
                "text_sha256": _sha256_text(text),
                "cosine": float(np.dot(cached, current)),
                "maximum_absolute_difference": float(np.max(np.abs(cached - current))),
            }
        )
    minimum_cosine = min(
        (row["cosine"] for row in comparisons),
        default=1.0,
    )
    if minimum_cosine < 0.999:
        raise ScopedQwenWsdError(
            "new endpoint is not vector-compatible with cached MRPC sentences: "
            f"minimum cosine {minimum_cosine:.9f}"
        )
    return {
        "sample_count": len(comparisons),
        "minimum_cosine": minimum_cosine,
        "maximum_absolute_difference": max(
            (row["maximum_absolute_difference"] for row in comparisons),
            default=0.0,
        ),
        "comparisons": comparisons,
    }


def _resolve_one(
    *,
    sentence_vector: np.ndarray,
    lemma: str,
    pos: str,
    candidate_synsets: Mapping[tuple[str, str], tuple[str, ...]],
    points_by_synset: Mapping[str, tuple[SynsetPoint, ...]],
    vectors_by_text: Mapping[str, np.ndarray],
    threshold: float,
) -> QwenSenseAssignment | None:
    synset_ids = candidate_synsets.get((lemma, pos), ())
    scored: list[tuple[float, float, str, str, str]] = []
    candidate_example_count = 0
    for synset_id in synset_ids:
        for point in points_by_synset.get(synset_id, ()):
            point_vector = vectors_by_text.get(point.text)
            if point_vector is None:
                continue
            candidate_example_count += 1
            raw_cosine = float(
                np.dot(
                    sentence_vector,
                    point_vector,
                )
            )
            score = _neo4j_cosine_score(raw_cosine)
            scored.append(
                (
                    score,
                    raw_cosine,
                    synset_id,
                    point.example_id,
                    point.kind,
                )
            )
    if not scored:
        return None
    score, raw_cosine, synset_id, example_id, example_kind = sorted(
        scored,
        key=lambda row: (-row[0], row[2], row[3]),
    )[0]
    if score < threshold:
        return None
    return QwenSenseAssignment(
        lemma=lemma,
        pos=pos,
        synset_id=synset_id,
        score=score,
        raw_cosine=raw_cosine,
        example_id=example_id,
        example_kind=example_kind,
        candidate_synset_count=len(synset_ids),
        candidate_example_count=candidate_example_count,
    )


def _resolve_sentences(
    *,
    texts: Sequence[str],
    inventory: ScopedInventory,
    sentence_vectors: Mapping[str, np.ndarray],
    vectors_by_text: Mapping[str, np.ndarray],
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sentences: dict[str, Any] = {}
    assignment_count = 0
    resolved_sentence_count = 0
    below_threshold_count = 0
    missing_candidate_point_count = 0
    best_scores: list[float] = []
    best_raw_cosines: list[float] = []
    best_example_kinds: Counter[str] = Counter()
    scores_by_sentence: list[tuple[float, ...]] = []
    relevant_pairs = set(inventory.relevant_pairs)
    for index, text in enumerate(texts, start=1):
        vector = _normalized(
            sentence_vectors[text],
            context=f"MRPC sentence {_sha256_text(text)}",
        )
        assignments: list[QwenSenseAssignment] = []
        sentence_best_scores: list[float] = []
        for lemma, pos in inventory.sentence_pairs[text]:
            if (lemma, pos) not in relevant_pairs:
                continue
            assignment = _resolve_one(
                sentence_vector=vector,
                lemma=lemma,
                pos=pos,
                candidate_synsets=inventory.candidate_synsets,
                points_by_synset=inventory.points_by_synset,
                vectors_by_text=vectors_by_text,
                threshold=0.0,
            )
            if assignment is None:
                missing_candidate_point_count += 1
                continue
            sentence_best_scores.append(assignment.score)
            best_scores.append(assignment.score)
            best_raw_cosines.append(assignment.raw_cosine)
            best_example_kinds[assignment.example_kind] += 1
            if assignment.score < threshold:
                below_threshold_count += 1
                continue
            assignments.append(assignment)
        scores_by_sentence.append(tuple(sentence_best_scores))
        senses: dict[str, list[str]] = {}
        for assignment in assignments:
            senses.setdefault(assignment.lemma, []).append(assignment.synset_id)
        senses = {lemma: sorted(set(synset_ids)) for lemma, synset_ids in sorted(senses.items())}
        text_hash = _sha256_text(text)
        sentences[text_hash] = {
            "text_sha256": text_hash,
            "senses": senses,
            "assignments": [asdict(assignment) for assignment in assignments],
        }
        assignment_count += len(assignments)
        if assignments:
            resolved_sentence_count += 1
        if index % 250 == 0 or index == len(texts):
            print(
                f"Resolved {index}/{len(texts)} MRPC sentences; assignments={assignment_count}",
                flush=True,
            )
    coverage = {
        "sentence_count": len(texts),
        "resolved_sentence_count": resolved_sentence_count,
        "resolved_sentence_rate": (resolved_sentence_count / len(texts) if texts else 0.0),
        "sense_assignment_count": assignment_count,
        "mean_assignments_per_sentence": (assignment_count / len(texts) if texts else 0.0),
        "candidate_resolution_attempt_count": (len(best_scores) + missing_candidate_point_count),
        "below_threshold_count": below_threshold_count,
        "missing_candidate_point_count": missing_candidate_point_count,
        "below_threshold_or_missing_point_count": (
            below_threshold_count + missing_candidate_point_count
        ),
        "score_contract": NEO4J_COSINE_SCORE_CONTRACT,
        "best_neo4j_score_quantiles": _quantiles(best_scores),
        "best_raw_cosine_quantiles": _quantiles(best_raw_cosines),
        "best_example_kind_count": dict(sorted(best_example_kinds.items())),
        "diagnostic_threshold_coverage": {
            f"{diagnostic_threshold:.2f}": {
                "sense_assignment_count": sum(
                    score >= diagnostic_threshold for score in best_scores
                ),
                "resolved_sentence_count": sum(
                    any(score >= diagnostic_threshold for score in sentence_scores)
                    for sentence_scores in scores_by_sentence
                ),
            }
            for diagnostic_threshold in DIAGNOSTIC_SCORE_THRESHOLDS
        },
        "resolution_method_count": {
            ("qwen_synset_example_neo4j_cosine_score"): assignment_count,
        },
    }
    return sentences, coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrpc-dir", type=Path, default=DEFAULT_MRPC_DIR)
    parser.add_argument("--wn-data", type=Path, default=DEFAULT_WN_DATA)
    parser.add_argument("--nltk-data", type=Path, default=DEFAULT_NLTK_DATA)
    parser.add_argument(
        "--sentence-embedding-cache",
        type=Path,
        default=DEFAULT_SENTENCE_CACHE,
    )
    parser.add_argument(
        "--sentence-endpoint-metadata",
        type=Path,
        default=DEFAULT_SENTENCE_ENDPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tei-url")
    parser.add_argument(
        "--replay-synset-vector-cache",
        type=Path,
        help=(
            "Reuse a completed endpoint-bound synset vector cache without contacting Hugging Face."
        ),
    )
    parser.add_argument(
        "--replay-source-sense-cache",
        type=Path,
        help=("Original sense cache whose hashes and model lineage authorize offline replay."),
    )
    parser.add_argument(
        "--replay-scoped-manifest",
        type=Path,
        help="Original scoped manifest referenced by the source sense cache.",
    )
    parser.add_argument("--auth-token-env")
    parser.add_argument("--served-model-alias", action="append", default=[])
    parser.add_argument("--declared-model-repository")
    parser.add_argument("--declared-model-revision")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-qwen-seconds", type=float, default=3600.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--minimum-lemma-sentence-count",
        type=int,
        default=DEFAULT_MINIMUM_LEMMA_SENTENCE_COUNT,
    )
    parser.add_argument(
        "--compatibility-sample-size",
        type=int,
        default=DEFAULT_COMPATIBILITY_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=DEFAULT_COSINE_THRESHOLD,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate the scoped manifest without calling Qwen.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_lemma_sentence_count < 2:
        raise ScopedQwenWsdError("minimum lemma sentence count must be at least two")
    if args.compatibility_sample_size < 1:
        raise ScopedQwenWsdError("compatibility sample size must be positive")
    if not 0.0 <= args.cosine_threshold <= 1.0:
        raise ScopedQwenWsdError("cosine threshold must be in [0, 1]")

    import nltk
    import wn
    from nltk.stem import PorterStemmer
    from wn.morphy import Morphy

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wn.config.data_directory = args.wn_data.resolve()
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    wordnet = wn.Wordnet(OEWN_VERSION)
    if len(wordnet.synsets()) != 120_630:
        raise OfflineWordNetError(
            "OEWN 2024 inventory does not contain the expected 120,630 synsets"
        )
    texts, mrpc_evidence = _load_mrpc_texts(args.mrpc_dir.resolve())
    tagged = nltk.pos_tag_sents([WORD_RE.findall(text) for text in texts])
    lexical_resolver = OfflineOEWNResolver(
        wordnet,
        Morphy(wordnet),
        PorterStemmer(),
    )
    inventory = _build_inventory(
        texts=texts,
        tagged_sentences=tagged,
        resolver=lexical_resolver,
        minimum_lemma_sentence_count=args.minimum_lemma_sentence_count,
    )
    database_path = Path(wn.config.database_path)
    manifest = _inventory_manifest(
        inventory=inventory,
        mrpc_evidence=mrpc_evidence,
        wordnet=wordnet,
        database_path=database_path,
        minimum_lemma_sentence_count=args.minimum_lemma_sentence_count,
    )
    manifest_path = output_dir / "scoped_synset_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"Scoped manifest: {manifest_path}", flush=True)
    if args.prepare_only:
        return 0
    replay_args = (
        args.replay_synset_vector_cache,
        args.replay_source_sense_cache,
        args.replay_scoped_manifest,
    )
    replay_mode = any(value is not None for value in replay_args)
    if replay_mode and not all(value is not None for value in replay_args):
        raise ScopedQwenWsdError(
            "offline replay requires the synset cache, source sense cache, "
            "and original scoped manifest"
        )
    if not replay_mode and not args.tei_url:
        raise ScopedQwenWsdError("--tei-url is required unless --prepare-only or replaying")

    prior_endpoint = _read_endpoint_metadata(args.sentence_endpoint_metadata.resolve())
    if replay_mode:
        read_cache = ReadOnlyScopedEmbeddingCache(args.sentence_embedding_cache.resolve())
        try:
            sentence_vectors = {}
            for text in texts:
                vector = read_cache.get(text, namespace="primary")
                if vector is None:
                    raise ScopedQwenWsdError(
                        f"sentence cache is incomplete for {_sha256_text(text)}"
                    )
                sentence_vectors[text] = vector
            sentence_cache_evidence = {
                "path": str(read_cache.path),
                "sha256": _sha256_file(read_cache.path),
                "row_count": read_cache.row_count,
                "metadata": read_cache.metadata,
            }
        finally:
            read_cache.close()
    else:
        from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import (
            BenchmarkError,
            ReadOnlyEmbeddingCache,
        )

        from mcp_servers.mcp_lib.qwen_typed_field_test import (
            QwenFieldTestError,
            QwenTEIClient,
            SQLiteEmbeddingCache,
        )

        read_cache = ReadOnlyEmbeddingCache(args.sentence_embedding_cache.resolve())
        try:
            sentence_vectors = {text: read_cache.get(text) for text in texts}
            sentence_cache_evidence = {
                "path": str(read_cache.path),
                "sha256": _sha256_file(read_cache.path),
                "row_count": read_cache.row_count,
                "metadata": read_cache.metadata,
            }
        finally:
            read_cache.close()

    unique_texts = tuple(
        sorted(
            {point.text for point in inventory.points},
            key=lambda text: (_sha256_text(text), text),
        )
    )
    if replay_mode:
        replay_vector_path = args.replay_synset_vector_cache.resolve()
        source_sense_path = args.replay_source_sense_cache.resolve()
        source_manifest_path = args.replay_scoped_manifest.resolve()
        replay_source, replay_manifest = _read_replay_source(
            source_sense_cache_path=source_sense_path,
            scoped_manifest_path=source_manifest_path,
            sentence_cache_path=args.sentence_embedding_cache.resolve(),
            synset_vector_cache_path=replay_vector_path,
        )
        _validate_replay_inventory(manifest, replay_manifest)
        model_lineage = dict(replay_source["resolver"]["model"])
        _validate_model_lineage(
            prior_endpoint=prior_endpoint,
            repository=model_lineage["repository"],
            revision=model_lineage["revision"],
        )
        compatibility = dict(replay_source["embedding_evidence"]["compatibility"])
        replay_cache = ReadOnlyScopedEmbeddingCache(replay_vector_path)
        try:
            vectors_by_text = {
                text: vector
                for text in unique_texts
                if (
                    vector := replay_cache.get(
                        text,
                        namespace="synset_example",
                    )
                )
                is not None
            }
            replay_row_count = replay_cache.row_count
            replay_cache_metadata = dict(replay_cache.metadata)
        finally:
            replay_cache.close()
        vector_cache_path = replay_vector_path
        resource_usage = {
            "mode": "offline_vector_cache_replay",
            "endpoint_request_count": 0,
            "source_sense_cache": {
                "path": str(source_sense_path),
                "sha256": _sha256_file(source_sense_path),
            },
            "source_resource_usage": replay_source["embedding_evidence"].get("resource_usage"),
            "synset_vector_cache_row_count": replay_row_count,
            "synset_vector_cache_metadata": replay_cache_metadata,
            "requested_unique_text_count": len(unique_texts),
            "available_unique_text_count": len(vectors_by_text),
            "missing_unique_text_count": (len(unique_texts) - len(vectors_by_text)),
        }
        manifest_evidence = {
            "path": str(source_manifest_path),
            "sha256": _sha256_file(source_manifest_path),
            "manifest_sha256": replay_manifest["manifest_sha256"],
            "counts": replay_manifest["counts"],
        }
        wordnet_evidence = dict(replay_source["wordnet"])
    else:
        model_lineage = _validate_model_lineage(
            prior_endpoint=prior_endpoint,
            repository=args.declared_model_repository,
            revision=args.declared_model_revision,
        )
        auth_token = None
        if args.auth_token_env:
            auth_token = os.environ.get(args.auth_token_env)
            if not auth_token or not auth_token.strip():
                raise ScopedQwenWsdError(f"environment variable {args.auth_token_env!r} is empty")
        else:
            try:
                from huggingface_hub import get_token
            except ImportError:
                pass
            else:
                auth_token = get_token()

        vector_cache_path = output_dir / "qwen_synset_vectors.sqlite3"
        vector_cache = SQLiteEmbeddingCache(vector_cache_path)
        try:
            client = QwenTEIClient(
                tei_url=args.tei_url,
                cache=vector_cache,
                model_id=DEFAULT_MODEL_ID,
                batch_size=args.batch_size,
                request_timeout_seconds=args.request_timeout_seconds,
                max_qwen_seconds=args.max_qwen_seconds,
                auth_token=auth_token,
                served_model_aliases=args.served_model_alias,
                declared_model_repository=args.declared_model_repository,
                declared_model_revision=args.declared_model_revision,
                skip_persistent_non_finite=True,
            )
            compatibility = _compatibility_check(
                texts=texts,
                sentence_vectors=sentence_vectors,
                client=client,
                sample_size=args.compatibility_sample_size,
            )
            embedding_inputs = [
                ScopedEmbeddingInput(
                    key=f"synset-text:{_sha256_text(text)}",
                    text=text,
                    purpose="mrpc_scoped_wordnet_synset_example",
                )
                for text in unique_texts
            ]
            embedded = client.embed_many(
                embedding_inputs,
                namespace="synset_example",
            )
            vectors_by_text = {
                text: _normalized(
                    embedded[item.key],
                    context=f"synset text {_sha256_text(text)}",
                )
                for text, item in zip(
                    unique_texts,
                    embedding_inputs,
                    strict=True,
                )
                if item.key in embedded
            }
            resource_usage = client.resource_report()
        except (BenchmarkError, QwenFieldTestError) as exc:
            raise ScopedQwenWsdError(str(exc)) from exc
        finally:
            vector_cache.close()
        manifest_evidence = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "counts": manifest["counts"],
        }
        wordnet_evidence = {
            **manifest["wordnet"],
            "wn_package_version": wn.__version__,
        }

    sentences, coverage = _resolve_sentences(
        texts=texts,
        inventory=inventory,
        sentence_vectors=sentence_vectors,
        vectors_by_text=vectors_by_text,
        threshold=args.cosine_threshold,
    )
    sense_cache = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "resolver": {
            "resolver_id": RESOLVER_ID,
            "cosine_threshold": args.cosine_threshold,
            "threshold_score_contract": NEO4J_COSINE_SCORE_CONTRACT,
            "raw_cosine_equivalent_threshold": ((2.0 * args.cosine_threshold) - 1.0),
            "parts_of_speech": ["noun", "verb"],
            "minimum_lemma_sentence_count": (args.minimum_lemma_sentence_count),
            "production_equivalent": False,
            "embedding_contract_equivalent": True,
            "production_difference": (
                "scores all MRPC-relevant lemma/POS candidate points exactly; "
                "production uses a global approximate HNSW top-128 query "
                "before applying lemma/POS filters"
            ),
            "model": model_lineage,
        },
        "wordnet": wordnet_evidence,
        "mrpc": mrpc_evidence,
        "coverage": coverage,
        "scoped_manifest": manifest_evidence,
        "embedding_evidence": {
            "sentence_cache": sentence_cache_evidence,
            "synset_vector_cache": {
                "path": str(vector_cache_path),
                "sha256": _sha256_file(vector_cache_path),
            },
            "compatibility": compatibility,
            "resource_usage": resource_usage,
        },
        "sentences": sentences,
    }
    sense_cache_path = output_dir / "qwen_wordnet_senses.json"
    _atomic_json(sense_cache_path, sense_cache)
    report_body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest": manifest_evidence,
        "sense_cache": {
            "path": str(sense_cache_path),
            "sha256": _sha256_file(sense_cache_path),
            "coverage": coverage,
        },
        "model": model_lineage,
        "compatibility": compatibility,
        "resource_usage": resource_usage,
    }
    report = {
        **report_body,
        "report_id": ("mrpc-wordnet-qwen-wsd:" + _sha256_payload(report_body)),
    }
    report_path = output_dir / "qwen_wordnet_wsd_report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "sense_cache": str(sense_cache_path),
                "coverage": coverage,
                "compatibility": {
                    "sample_count": compatibility["sample_count"],
                    "minimum_cosine": compatibility["minimum_cosine"],
                },
                "resource_usage": resource_usage,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
