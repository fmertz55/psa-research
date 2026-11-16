#!/usr/bin/env python3
"""Census every directed candidate in MRPC's full-signature PSA top 32.

This is an offline, non-mutating analysis. Official MRPC relations are marked
as known positive or known negative; every other neighbor remains explicitly
unlabeled and is never counted as a negative.

The funnel is:

1. exact 115-dimensional PSA top-32 retrieval,
2. cold 0.77 and calibration-learned native gates,
3. production-default and strict-shadow verifier probes,
4. WordNet sense agreement,
5. ordered-synset De Bruijn-style overlap features.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from scripts.research.mrpc.analyze_mrpc_wordnet_debruijn_shadow import (  # noqa: E402
    sequence_features,
)
from scripts.research.mrpc.run_mrpc_current_pipeline_benchmark import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    _load_pinned_frame,
    _load_wordnet_sense_cache,
    _project_representations,
    _wordnet_gate,
)
from scripts.research.mrpc.run_mrpc_gate_verifier_benchmark import (  # noqa: E402
    DEFAULT_CACHE,
)
from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import (  # noqa: E402
    BenchmarkError,
    _atomic_json,
    _load_cache_evidence,
)

from mcp_servers.mcp_lib.paraphrase_verification import (  # noqa: E402
    ParaphraseVerificationController,
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
    QWEN3_EMBEDDING_MODEL_ID,
    paraphrase_native_cosine_threshold,
)

DEFAULT_SENSE_CACHE = (
    REPOSITORY_ROOT
    / "temp"
    / "mrpc_wordnet_qwen_replay"
    / "30475782422-neo4j-score"
    / "qwen_wordnet_senses.json"
)
DEFAULT_ORDERED_SENSE_CACHE = (
    REPOSITORY_ROOT
    / "temp"
    / "mrpc_wordnet_qwen_replay"
    / "30475782422-neo4j-score"
    / "ordered_synset_sequences.json"
)
DEFAULT_GATE_REPORT = (
    REPOSITORY_ROOT / "reports" / "mrpc_gate_verifier_2026-07-30" / "mrpc_gate_verifier_report.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_psa_top32_analysis_2026-07-30"
ORDERED_SENSE_SCHEMA_VERSION = "aiidea_mrpc_ordered_synset_cache_v1"
REPORT_SCHEMA_VERSION = "aiidea_mrpc_psa_top32_candidate_analysis_v1"
RELATION_POSITIVE = "official_positive"
RELATION_NEGATIVE = "official_negative"
RELATION_UNLABELED = "unlabeled_neighbor"
RELATION_TYPES = (
    RELATION_POSITIVE,
    RELATION_NEGATIVE,
    RELATION_UNLABELED,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((sha256_text(left), sha256_text(right))))


def _relation_index(
    pairs: Sequence[MRPCPair],
    splits: Mapping[int, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        key = _pair_key(pair.sentence1, pair.sentence2)
        value = {
            "relation": (RELATION_POSITIVE if pair.is_paraphrase else RELATION_NEGATIVE),
            "label": bool(pair.is_paraphrase),
            "split": splits[pair.pair_id],
            "pair_ids": [pair.pair_id],
        }
        prior = index.get(key)
        if prior is None:
            index[key] = value
            continue
        if prior["relation"] != value["relation"] or prior["split"] != value["split"]:
            raise BenchmarkError("duplicate MRPC relation has conflicting label or split")
        prior["pair_ids"].append(pair.pair_id)
    return index


def _ranked_psa_neighborhoods(
    signatures_by_text: Mapping[str, np.ndarray],
    *,
    top_k: int,
) -> tuple[
    dict[str, tuple[tuple[str, float], ...]],
    dict[str, Any],
]:
    texts = sorted(signatures_by_text, key=lambda text: (sha256_text(text), text))
    if top_k < 1 or top_k >= len(texts):
        raise BenchmarkError("PSA top_k must be positive and below corpus size")
    matrix = np.stack([signatures_by_text[text] for text in texts]).astype(
        np.float32,
        copy=False,
    )
    neighborhoods: dict[str, tuple[tuple[str, float], ...]] = {}
    for source_index, text in enumerate(texts):
        scores = matrix[source_index] @ matrix.T
        scores[source_index] = -math.inf
        candidate_indexes = np.argpartition(
            -scores,
            kth=top_k - 1,
        )[:top_k]
        ordered = sorted(
            (int(index) for index in candidate_indexes),
            key=lambda index: (-float(scores[index]), texts[index]),
        )
        neighborhoods[text] = tuple((texts[index], float(scores[index])) for index in ordered)
    return neighborhoods, {
        "implementation": "exact_psa_cosine_ranked_top_k",
        "top_k": top_k,
        "sentence_count": len(texts),
        "directed_candidate_count": len(texts) * top_k,
        "signature_dimension": int(matrix.shape[1]),
    }


def _load_ordered_sequences(
    path: Path,
    texts: Iterable[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ORDERED_SENSE_SCHEMA_VERSION:
        raise BenchmarkError("ordered synset cache schema is not supported")
    sentences = payload.get("sentences")
    if not isinstance(sentences, dict):
        raise BenchmarkError("ordered synset cache has no sentence mapping")
    expected = {sha256_text(text): text for text in texts}
    if set(sentences) != set(expected):
        raise BenchmarkError("ordered synset cache does not exactly cover MRPC sentences")
    sequences: dict[str, tuple[str, ...]] = {}
    for text_hash, text in expected.items():
        row = sentences[text_hash]
        raw_sequence = row.get("synset_sequence") if isinstance(row, dict) else None
        if not isinstance(raw_sequence, list) or any(
            not isinstance(value, str) for value in raw_sequence
        ):
            raise BenchmarkError(f"invalid ordered synset cache row {text_hash}")
        sequences[text] = tuple(raw_sequence)
    return sequences, {
        "path": str(path),
        "sha256": _file_sha256(path),
        "cache_id": payload.get("cache_id"),
        "sentence_count": payload.get("sentence_count"),
        "zero_sequence_sentence_count": payload.get("zero_sequence_sentence_count"),
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row[field]) for row in rows) / len(rows)


def _quantiles(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, float] | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None and math.isfinite(float(row[field]))
    ]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        str(probability): float(np.quantile(array, probability))
        for probability in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def _candidate_group_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    learned_gated = [row for row in rows if row["learned_gate_passed"]]
    sequence_fields = (
        "unigram_jaccard",
        "unigram_containment",
        "bigram_jaccard",
        "bigram_containment",
        "ordered_sequence_ratio",
    )
    return {
        "count": len(rows),
        "rank_quantiles": _quantiles(rows, "rank"),
        "psa_similarity_quantiles": _quantiles(rows, "psa_similarity"),
        "native_similarity_quantiles": _quantiles(rows, "native_similarity"),
        "cold_gate_pass_rate": _rate(rows, "cold_gate_passed"),
        "learned_gate_pass_rate": _rate(rows, "learned_gate_passed"),
        "production_verifier_pass_rate_after_learned_gate": _rate(
            learned_gated,
            "production_verifier_passed",
        ),
        "strict_shadow_pass_rate_after_learned_gate": _rate(
            learned_gated,
            "strict_shadow_passed",
        ),
        "pipeline_accept_rate": _rate(rows, "pipeline_accepted"),
        "wordnet_reason_counts": dict(
            sorted(Counter(str(row["wordnet_reason"]) for row in rows).items())
        ),
        "production_decision_reason_counts_after_learned_gate": dict(
            sorted(Counter(str(row["production_verifier_reason"]) for row in learned_gated).items())
        ),
        "strict_shadow_reason_counts_after_learned_gate": dict(
            sorted(Counter(str(row["strict_shadow_reason"]) for row in learned_gated).items())
        ),
        "surface_conflict_counts": dict(
            sorted(Counter(reason for row in rows for reason in row["surface_conflicts"]).items())
        ),
        "sequence_feature_quantiles": {
            field: _quantiles(
                [row["sequence_features"] for row in rows],
                field,
            )
            for field in sequence_fields
        },
    }


def _directional_pair_report(
    pairs: Sequence[MRPCPair],
    splits: Mapping[int, str],
    edge_by_direction: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for split in ("calibration", "held_out"):
        split_pairs = [pair for pair in pairs if splits[pair.pair_id] == split]
        labels = np.asarray(
            [pair.is_paraphrase for pair in split_pairs],
            dtype=bool,
        )
        rows = []
        for pair in split_pairs:
            forward = edge_by_direction.get(
                (
                    sha256_text(pair.sentence1),
                    sha256_text(pair.sentence2),
                )
            )
            reverse = edge_by_direction.get(
                (
                    sha256_text(pair.sentence2),
                    sha256_text(pair.sentence1),
                )
            )

            def passed(
                row: Mapping[str, Any] | None,
                field: str,
            ) -> bool:
                return bool(row is not None and row[field])

            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "label": pair.is_paraphrase,
                    "forward_nominated": forward is not None,
                    "reverse_nominated": reverse is not None,
                    "forward_pipeline_accepted": passed(
                        forward,
                        "pipeline_accepted",
                    ),
                    "reverse_pipeline_accepted": passed(
                        reverse,
                        "pipeline_accepted",
                    ),
                    "forward_rank": (int(forward["rank"]) if forward is not None else None),
                    "reverse_rank": (int(reverse["rank"]) if reverse is not None else None),
                }
            )
        reports[split] = {
            "pair_count": len(rows),
            "positive_count": int(np.count_nonzero(labels)),
            "negative_count": int(np.count_nonzero(~labels)),
            "nomination": {},
            "pipeline_acceptance": {},
        }
        for name, left_field, right_field in (
            (
                "forward",
                "forward_nominated",
                "forward_nominated",
            ),
            (
                "reverse",
                "reverse_nominated",
                "reverse_nominated",
            ),
            (
                "any_direction",
                "forward_nominated",
                "reverse_nominated",
            ),
            (
                "both_directions",
                "forward_nominated",
                "reverse_nominated",
            ),
        ):
            if name == "both_directions":
                prediction = np.asarray(
                    [bool(row[left_field] and row[right_field]) for row in rows]
                )
            else:
                prediction = np.asarray([bool(row[left_field] or row[right_field]) for row in rows])
            reports[split]["nomination"][name] = _binary_metrics(
                labels,
                prediction,
            )
        for name, mode in (
            ("forward", "forward"),
            ("reverse", "reverse"),
            ("any_direction", "any"),
            ("both_directions", "both"),
        ):
            prediction = np.asarray(
                [
                    (
                        bool(row["forward_pipeline_accepted"])
                        if mode == "forward"
                        else bool(row["reverse_pipeline_accepted"])
                        if mode == "reverse"
                        else bool(
                            row["forward_pipeline_accepted"] or row["reverse_pipeline_accepted"]
                        )
                        if mode == "any"
                        else bool(
                            row["forward_pipeline_accepted"] and row["reverse_pipeline_accepted"]
                        )
                    )
                    for row in rows
                ]
            )
            reports[split]["pipeline_acceptance"][name] = _binary_metrics(
                labels,
                prediction,
            )
        reports[split]["positive_rank_quantiles"] = {
            direction: _quantiles(
                [{"rank": row[field]} for row in rows if row["label"] and row[field] is not None],
                "rank",
            )
            for direction, field in (
                ("forward", "forward_rank"),
                ("reverse", "reverse_rank"),
            )
        }
    return reports


def _example(
    row: Mapping[str, Any],
    text_by_hash: Mapping[str, str],
) -> dict[str, Any]:
    return {
        **{
            key: row[key]
            for key in (
                "source_sha256",
                "target_sha256",
                "rank",
                "relation",
                "split",
                "psa_similarity",
                "native_similarity",
                "cold_gate_passed",
                "learned_gate_passed",
                "production_verifier_reason",
                "strict_shadow_reason",
                "surface_conflicts",
                "wordnet_reason",
                "wordnet_agreement",
                "sequence_features",
            )
        },
        "source_text": text_by_hash[row["source_sha256"]],
        "target_text": text_by_hash[row["target_sha256"]],
    }


def _evidence_feature_groups(
    row: Mapping[str, Any],
) -> dict[str, list[float]]:
    sequence = row["sequence_features"]
    wordnet_reasons = (
        "applied",
        "coverage_empty_side",
        "coverage_no_shared_lemmas",
    )
    surface_reasons = (
        "numeric_multiset_mismatch",
        "explicit_polarity_mismatch",
        "modality_surface_mismatch",
        "surface_entity_disjoint",
        "wordnet_sense_conflict",
    )
    sequence_fields = (
        "unigram_jaccard",
        "unigram_containment",
        "bigram_jaccard",
        "bigram_containment",
        "ordered_sequence_ratio",
        "source_length",
        "target_length",
        "shared_synset_count",
        "shared_bigram_count",
    )
    return {
        "retrieval": [
            float(row["rank"]) / PSA_SIGNATURE_CANDIDATE_LIMIT,
            1.0 / float(row["rank"]),
            float(row["psa_similarity"]),
            float(row["native_similarity"]),
        ],
        "wordnet": [
            float(row["wordnet_agreement"]),
            *[1.0 if row["wordnet_reason"] == reason else 0.0 for reason in wordnet_reasons],
        ],
        "surface": [
            1.0 if reason in row["surface_conflicts"] else 0.0 for reason in surface_reasons
        ],
        "debruijn": [
            (0.0 if sequence.get(field) is None else float(sequence[field]))
            for field in sequence_fields
        ],
    }


def _calibrated_evidence_ablation(
    edge_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit small diagnostic combiners on calibration directed edges only.

    This is not a release classifier. It checks whether a feature family adds
    held-out discrimination beyond rank, PSA similarity, and native cosine.
    """

    known = [row for row in edge_rows if row["relation"] in {RELATION_POSITIVE, RELATION_NEGATIVE}]
    calibration = [row for row in known if row["split"] == "calibration"]
    held_out = [row for row in known if row["split"] == "held_out"]
    configurations = (
        ("retrieval",),
        ("retrieval", "wordnet"),
        ("retrieval", "surface"),
        ("retrieval", "debruijn"),
        ("retrieval", "wordnet", "surface", "debruijn"),
    )

    def matrix(
        rows: Sequence[Mapping[str, Any]],
        groups: Sequence[str],
    ) -> np.ndarray:
        return np.asarray(
            [
                [value for group in groups for value in _evidence_feature_groups(row)[group]]
                for row in rows
            ],
            dtype=np.float64,
        )

    calibration_labels = np.asarray([row["relation"] == RELATION_POSITIVE for row in calibration])
    held_out_labels = np.asarray([row["relation"] == RELATION_POSITIVE for row in held_out])
    calibration_gate = np.asarray([bool(row["learned_gate_passed"]) for row in calibration])
    held_out_gate = np.asarray([bool(row["learned_gate_passed"]) for row in held_out])
    results = {}
    for groups in configurations:
        calibration_features = matrix(calibration, groups)
        held_out_features = matrix(held_out, groups)
        best: (
            tuple[
                tuple[float, float, float],
                float,
                str | None,
                float,
                Any,
                dict[str, Any],
            ]
            | None
        ) = None
        for regularization in (
            0.001,
            0.003,
            0.01,
            0.03,
            0.1,
            0.3,
            1.0,
            3.0,
            10.0,
        ):
            for class_weight in (None, "balanced"):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=regularization,
                        class_weight=class_weight,
                        max_iter=5000,
                        random_state=42,
                    ),
                )
                model.fit(
                    calibration_features[calibration_gate],
                    calibration_labels[calibration_gate],
                )
                probabilities = model.predict_proba(calibration_features)[:, 1]
                for threshold in np.unique(probabilities[calibration_gate]):
                    metrics = _binary_metrics(
                        calibration_labels,
                        calibration_gate & (probabilities >= threshold),
                    )
                    score = (
                        float(metrics["f1"]),
                        float(metrics["precision"] or 0.0),
                        float(threshold),
                    )
                    if best is None or score > best[0]:
                        best = (
                            score,
                            regularization,
                            class_weight,
                            float(threshold),
                            model,
                            metrics,
                        )
        if best is None:
            raise BenchmarkError("evidence ablation produced no model")
        (
            _score,
            regularization,
            class_weight,
            threshold,
            model,
            calibration_metrics,
        ) = best
        held_out_probabilities = model.predict_proba(held_out_features)[:, 1]
        held_out_metrics = _binary_metrics(
            held_out_labels,
            held_out_gate & (held_out_probabilities >= threshold),
        )
        results["+".join(groups)] = {
            "feature_groups": list(groups),
            "regularization_c": regularization,
            "class_weight": class_weight,
            "decision_threshold": threshold,
            "calibration_metrics": calibration_metrics,
            "held_out_metrics": held_out_metrics,
        }
    return {
        "evaluation_unit": "known_directed_top32_candidate",
        "training_partition": "calibration_only",
        "held_out_used_for_model_selection": False,
        "release_metric": False,
        "baseline_learned_gate_held_out_metrics": _binary_metrics(
            held_out_labels,
            held_out_gate,
        ),
        "ablations": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--mrpc-dir",
        type=Path,
        default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split",
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
    parser.add_argument("--sense-cache", type=Path, default=DEFAULT_SENSE_CACHE)
    parser.add_argument(
        "--ordered-sense-cache",
        type=Path,
        default=DEFAULT_ORDERED_SENSE_CACHE,
    )
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--psa-top-k",
        type=int,
        default=PSA_SIGNATURE_CANDIDATE_LIMIT,
    )
    parser.add_argument("--example-limit", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = load_mrpc_pairs(args.mrpc_dir.resolve())
    splits = split_mrpc_by_sentence_component(pairs)
    relation_by_pair = _relation_index(pairs, splits)
    texts = sorted({text for pair in pairs for text in (pair.sentence1, pair.sentence2)})
    text_by_hash = {sha256_text(text): text for text in texts}
    native_by_text, _cached_guides, cache_evidence = _load_cache_evidence(
        args.cache.resolve(),
        pairs,
    )
    (
        pinned_guides,
        calibration,
        frame_router,
        frame_evidence,
    ) = _load_pinned_frame(
        args.pinned_guide_embeddings.resolve(),
        args.calibration.resolve(),
    )
    signatures_by_text, _labels_by_text, signature_distribution = _project_representations(
        native_by_text,
        pinned_guides,
        frame_router,
        calibration,
    )
    neighborhoods, neighborhood_evidence = _ranked_psa_neighborhoods(
        signatures_by_text,
        top_k=args.psa_top_k,
    )
    wordnet_senses, wordnet_evidence = _load_wordnet_sense_cache(
        args.sense_cache.resolve(),
        texts,
    )
    ordered_sequences, ordered_evidence = _load_ordered_sequences(
        args.ordered_sense_cache.resolve(),
        texts,
    )
    gate_report_path = args.gate_report.resolve()
    gate_report = json.loads(gate_report_path.read_text(encoding="utf-8"))
    learned_threshold = float(gate_report["threshold"]["learned"])
    cold_threshold = paraphrase_native_cosine_threshold(QWEN3_EMBEDDING_MODEL_ID)

    production_controller = ParaphraseVerificationController(
        require_complete_evidence=False,
        enforce_fact_conflicts=False,
    )
    strict_controller = ParaphraseVerificationController(
        require_complete_evidence=False,
        enforce_fact_conflicts=True,
    )
    edge_rows: list[dict[str, Any]] = []
    edge_by_direction: dict[tuple[str, str], dict[str, Any]] = {}
    for source_text in sorted(
        neighborhoods,
        key=lambda text: (sha256_text(text), text),
    ):
        source_hash = sha256_text(source_text)
        for rank, (target_text, psa_similarity) in enumerate(
            neighborhoods[source_text],
            start=1,
        ):
            target_hash = sha256_text(target_text)
            relation = relation_by_pair.get(tuple(sorted((source_hash, target_hash))))
            native_similarity = float(
                np.dot(
                    native_by_text[source_text],
                    native_by_text[target_text],
                )
            )
            wordnet_passes, wordnet_agreement, wordnet_reason = _wordnet_gate(
                wordnet_senses[source_text],
                wordnet_senses[target_text],
            )
            production = production_controller.verify(
                source_text=source_text,
                target_text=target_text,
                native_similarity=native_similarity,
                candidate_type="free_sentence",
                psa_local=True,
                wordnet_passes=wordnet_passes,
                wordnet_agreement=wordnet_agreement,
                wordnet_reason=wordnet_reason,
            )
            strict = strict_controller.verify(
                source_text=source_text,
                target_text=target_text,
                native_similarity=native_similarity,
                candidate_type="free_sentence",
                psa_local=True,
                wordnet_passes=wordnet_passes,
                wordnet_agreement=wordnet_agreement,
                wordnet_reason=wordnet_reason,
            )
            sequence = sequence_features(
                ordered_sequences[source_text],
                ordered_sequences[target_text],
            )
            surface_conflicts = sorted(
                {
                    result.reason
                    for result in production.probe_results
                    if result.status == "conflict"
                }
            )
            learned_gate_passed = native_similarity >= learned_threshold
            row = {
                "source_sha256": source_hash,
                "target_sha256": target_hash,
                "rank": rank,
                "relation": (relation["relation"] if relation is not None else RELATION_UNLABELED),
                "split": (relation["split"] if relation is not None else None),
                "pair_ids": (relation["pair_ids"] if relation is not None else []),
                "psa_similarity": psa_similarity,
                "native_similarity": native_similarity,
                "cold_gate_passed": native_similarity >= cold_threshold,
                "learned_gate_passed": learned_gate_passed,
                "wordnet_passed": wordnet_passes,
                "wordnet_agreement": wordnet_agreement,
                "wordnet_reason": wordnet_reason,
                "production_verifier_passed": production.accepted,
                "production_verifier_reason": production.reason,
                "strict_shadow_passed": strict.accepted,
                "strict_shadow_reason": strict.reason,
                "surface_conflicts": surface_conflicts,
                "sequence_features": asdict(sequence),
                "pipeline_accepted": bool(learned_gate_passed and production.accepted),
            }
            edge_rows.append(row)
            edge_by_direction[(source_hash, target_hash)] = row

    by_relation = {
        relation: [row for row in edge_rows if row["relation"] == relation]
        for relation in RELATION_TYPES
    }
    known_positive_failures = [
        row for row in by_relation[RELATION_POSITIVE] if not row["pipeline_accepted"]
    ]
    known_negative_accepts = [
        row for row in by_relation[RELATION_NEGATIVE] if row["pipeline_accepted"]
    ]
    strict_collateral = [
        row
        for row in by_relation[RELATION_POSITIVE]
        if row["learned_gate_passed"]
        and row["production_verifier_passed"]
        and not row["strict_shadow_passed"]
    ]
    examples = {
        "known_positive_pipeline_failures": [
            _example(row, text_by_hash)
            for row in sorted(
                known_positive_failures,
                key=lambda row: (
                    row["learned_gate_passed"],
                    row["native_similarity"],
                    -row["rank"],
                ),
                reverse=True,
            )[: args.example_limit]
        ],
        "known_negative_pipeline_accepts": [
            _example(row, text_by_hash)
            for row in sorted(
                known_negative_accepts,
                key=lambda row: (
                    row["native_similarity"],
                    row["psa_similarity"],
                ),
                reverse=True,
            )[: args.example_limit]
        ],
        "strict_surface_positive_collateral": [
            _example(row, text_by_hash)
            for row in sorted(
                strict_collateral,
                key=lambda row: (
                    row["native_similarity"],
                    row["psa_similarity"],
                ),
                reverse=True,
            )[: args.example_limit]
        ],
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_class": "offline_directed_candidate_census",
        "scope": {
            "production_mutation": False,
            "official_relations_are_labels": True,
            "all_other_neighbors_are_unlabeled": True,
            "cold_threshold": cold_threshold,
            "learned_threshold": learned_threshold,
            "embedding_model": QWEN3_EMBEDDING_MODEL_ID,
        },
        "evidence": {
            "embedding_cache": cache_evidence,
            "frame": frame_evidence,
            "gate_report": {
                "path": str(gate_report_path),
                "sha256": _file_sha256(gate_report_path),
            },
            "wordnet": wordnet_evidence,
            "ordered_synsets": ordered_evidence,
            "signature_distribution": signature_distribution,
            "neighborhood": neighborhood_evidence,
        },
        "relation_inventory": {
            "official_unique_relation_count": len(relation_by_pair),
            "official_pair_row_count": len(pairs),
            "directed_candidate_relation_counts": dict(
                sorted(Counter(row["relation"] for row in edge_rows).items())
            ),
            "unlabeled_neighbors_are_not_negatives": True,
        },
        "candidate_funnel_by_relation": {
            relation: _candidate_group_summary(by_relation[relation]) for relation in RELATION_TYPES
        },
        "official_pair_directionality": _directional_pair_report(
            pairs,
            splits,
            edge_by_direction,
        ),
        "failure_cohorts": {
            "known_positive_directed_pipeline_failure_count": len(known_positive_failures),
            "known_negative_directed_pipeline_accept_count": len(known_negative_accepts),
            "strict_surface_positive_collateral_count": len(strict_collateral),
            "known_positive_failure_stage_counts": dict(
                sorted(
                    Counter(
                        (
                            "below_learned_native_gate"
                            if not row["learned_gate_passed"]
                            else "production_verifier_rejected"
                        )
                        for row in known_positive_failures
                    ).items()
                )
            ),
        },
        "evidence_combiner_shadow": _calibrated_evidence_ablation(edge_rows),
        "examples": examples,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "candidate_census.jsonl.gz"
    with gzip.open(rows_path, "wt", encoding="utf-8") as handle:
        for row in edge_rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")
    report["candidate_rows"] = {
        "path": str(rows_path),
        "sha256": _file_sha256(rows_path),
        "row_count": len(edge_rows),
    }
    report_body = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    report["report_id"] = (
        "mrpc-psa-top32:" + hashlib.sha256(report_body.encode("utf-8")).hexdigest()
    )
    report_path = output_dir / "psa_top32_analysis.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_id": report["report_id"],
                "candidate_rows": len(edge_rows),
                "relation_counts": report["relation_inventory"][
                    "directed_candidate_relation_counts"
                ],
                "failure_cohorts": report["failure_cohorts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
