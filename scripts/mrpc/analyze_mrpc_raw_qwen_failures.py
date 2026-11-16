#!/usr/bin/env python3
"""Produce a pair-level audit of every raw-Qwen MRPC classification error.

This audit intentionally stops before PSA routing or paraphrase-group
construction. It evaluates only the normalized Qwen sentence vectors, the
production 0.77 raw-cosine threshold, and the held-out MRPC component split.
Every false positive and false negative is retained with score margin,
surface features, observable verifier probes, and optional offline WordNet
evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
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
    ReadOnlyEmbeddingCache,
    _atomic_json,
)

from mcp_servers.mcp_lib.paraphrase_verification import (  # noqa: E402
    extract_modality_markers,
    extract_named_entities,
    extract_numeric_facts,
    extract_polarity_markers,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    DEFAULT_MRPC_CALIBRATION_PERCENT,
    DEFAULT_SPLIT_SEED,
    MRPCPair,
    _binary_metrics,
    fit_cosine_threshold,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    paraphrase_native_cosine_threshold,
)

SCHEMA_VERSION = "aiidea_mrpc_raw_qwen_failure_audit_v1"
DEFAULT_CACHE = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "qwen_embeddings.sqlite3"
)
DEFAULT_ENDPOINT = REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "endpoint.json"
DEFAULT_WORDNET_CACHE = (
    REPOSITORY_ROOT
    / "temp"
    / "mrpc_wordnet_qwen_replay"
    / "30475782422-neo4j-score"
    / "qwen_wordnet_senses.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_raw_qwen_failure_audit_2026-07-29"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
QUOTE_RE = re.compile(r"[\"“”]")
LEGACY_HEX_ESCAPE_RE = re.compile(r"\\'([0-9A-Fa-f]{2})")
REPORTING_MARKERS = frozenset(
    {
        "according",
        "announced",
        "claimed",
        "confirmed",
        "reported",
        "said",
        "says",
        "stated",
        "told",
    }
)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)
SCORE_BINS = (
    (0.00, 0.60),
    (0.60, 0.70),
    (0.70, 0.74),
    (0.74, 0.77),
    (0.77, 0.80),
    (0.80, 0.85),
    (0.85, 0.90),
    (0.90, 0.95),
    (0.95, 1.01),
)


class RawQwenAuditError(RuntimeError):
    """Raised when raw-Qwen audit evidence is incomplete or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in TOKEN_RE.findall(text))


def _decode_legacy_hex_escapes(text: str) -> str:
    return LEGACY_HEX_ESCAPE_RE.sub(
        lambda match: bytes.fromhex(match.group(1)).decode("cp1252"),
        text,
    )


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in _tokens(text) if token not in STOPWORDS and len(token) > 1)


def _set_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _set_containment(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    denominator = min(len(left_set), len(right_set))
    return len(left_set & right_set) / denominator if denominator else 1.0


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{probability:g}": float(np.quantile(array, probability)) for probability in probabilities
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        raise RawQwenAuditError("ROC AUC requires both labels")
    comparisons = positives[:, None] - negatives[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    cumulative = np.cumsum(ordered_labels)
    positive_ranks = np.flatnonzero(ordered_labels) + 1
    if not len(positive_ranks):
        raise RawQwenAuditError("average precision requires positives")
    return float(np.mean(cumulative[positive_ranks - 1] / positive_ranks))


def _balanced_accuracy(metrics: Mapping[str, Any]) -> float:
    true_positive = int(metrics["true_positive"])
    false_negative = int(metrics["false_negative"])
    true_negative = int(metrics["true_negative"])
    false_positive = int(metrics["false_positive"])
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return (sensitivity + specificity) / 2.0


def _best_thresholds(
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    unique = np.unique(scores)
    thresholds = np.concatenate((unique, [np.nextafter(unique[-1], math.inf)]))
    objectives: dict[str, tuple[float, float, dict[str, Any]]] = {}
    for threshold in thresholds:
        metrics = _binary_metrics(labels, scores >= threshold)
        objective_values = {
            "f1": float(metrics["f1"]),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": _balanced_accuracy(metrics),
        }
        for objective, value in objective_values.items():
            candidate = (value, float(threshold), metrics)
            previous = objectives.get(objective)
            if previous is None or candidate[:2] > previous[:2]:
                objectives[objective] = candidate
    return {
        objective: {
            "threshold": threshold,
            "objective_value": value,
            "metrics": metrics,
        }
        for objective, (value, threshold, metrics) in objectives.items()
    }


def _load_wordnet_senses(
    path: Path | None,
    pairs: Sequence[MRPCPair],
) -> tuple[dict[str, dict[str, frozenset[str]]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    if not path.is_file():
        raise RawQwenAuditError(f"WordNet sense cache does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("sentences")
    if not isinstance(entries, dict):
        raise RawQwenAuditError("WordNet cache has no sentence mapping")
    texts = {text for pair in pairs for text in (pair.sentence1, pair.sentence2)}
    senses: dict[str, dict[str, frozenset[str]]] = {}
    for text in texts:
        entry = entries.get(sha256_text(text))
        if not isinstance(entry, dict):
            raise RawQwenAuditError(f"WordNet cache omitted sentence {sha256_text(text)}")
        senses[text] = {
            str(lemma): frozenset(str(value) for value in synset_ids)
            for lemma, synset_ids in (entry.get("senses") or {}).items()
            if synset_ids
        }
    return senses, {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "resolver": payload.get("resolver"),
        "coverage": payload.get("coverage"),
    }


def _validate_source_join(
    pairs: Sequence[MRPCPair],
    mrpc_dir: Path,
) -> dict[str, Any]:
    test_path = mrpc_dir.parent / "msr_paraphrase_test.txt"
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
                quoting=csv.QUOTE_NONE,
            )
        )
    if len(rows) != len(pairs):
        raise RawQwenAuditError("official test rows do not match loaded MRPC pair count")
    raw_mismatches = []
    normalized_mismatches = []
    trailing_export_slash_count = 0
    trailing_export_brace_count = 0
    for pair, row in zip(pairs, rows, strict=True):
        first = str(row.get("#1 String") or "")
        second_export = str(row.get("#2 String") or "")
        if second_export.endswith("\\"):
            trailing_export_slash_count += 1
            second = second_export[:-1]
        elif second_export.endswith("}"):
            trailing_export_brace_count += 1
            second = second_export[:-1]
        else:
            second = second_export
        if pair.sentence1 != first or pair.sentence2 != second:
            raw_mismatches.append(
                {
                    "pair_id": pair.pair_id,
                    "joined_sentence1": pair.sentence1,
                    "official_sentence1": first,
                    "sentence1_matches": pair.sentence1 == first,
                    "joined_sentence2": pair.sentence2,
                    "official_sentence2": second,
                    "sentence2_matches": pair.sentence2 == second,
                }
            )
        decoded_first = _decode_legacy_hex_escapes(first)
        decoded_second = _decode_legacy_hex_escapes(second)
        if pair.sentence1 != decoded_first or pair.sentence2 != decoded_second:
            normalized_mismatches.append(
                {
                    "pair_id": pair.pair_id,
                    "joined_sentence1": pair.sentence1,
                    "normalized_official_sentence1": decoded_first,
                    "joined_sentence2": pair.sentence2,
                    "normalized_official_sentence2": decoded_second,
                }
            )
    return {
        "official_test_path": str(test_path.resolve()),
        "official_test_sha256": _sha256_file(test_path),
        "pair_count": len(pairs),
        "joined_text_matches_official_export_after_known_slash_cleanup": (not raw_mismatches),
        "raw_mismatched_pair_count": len(raw_mismatches),
        "raw_mismatched_pairs": raw_mismatches,
        "joined_text_matches_after_export_cleanup_and_cp1252_decoding": (not normalized_mismatches),
        "normalized_mismatched_pair_count": len(normalized_mismatches),
        "normalized_mismatched_pairs": normalized_mismatches,
        "known_trailing_export_slash_count": trailing_export_slash_count,
        "known_trailing_export_brace_count": trailing_export_brace_count,
        "joined_sentence_with_trailing_export_slash_count": sum(
            text.endswith("\\") for pair in pairs for text in (pair.sentence1, pair.sentence2)
        ),
    }


def _wordnet_pair_features(
    left: Mapping[str, frozenset[str]],
    right: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    shared = sorted(set(left) & set(right))
    agreements = [lemma for lemma in shared if left[lemma] & right[lemma]]
    conflicts = [lemma for lemma in shared if not (left[lemma] & right[lemma])]
    return {
        "covered_both_sides": bool(left and right),
        "shared_lemmas": shared,
        "agreement_lemmas": agreements,
        "conflict_lemmas": conflicts,
        "agreement": (len(agreements) / len(shared) if shared else None),
        "gate_conflict": bool(shared and len(agreements) / len(shared) < 0.5),
    }


def _surface_features(
    left: str,
    right: str,
) -> dict[str, Any]:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    left_content = _content_tokens(left)
    right_content = _content_tokens(right)
    left_numbers = extract_numeric_facts(left)
    right_numbers = extract_numeric_facts(right)
    left_polarity = extract_polarity_markers(left)
    right_polarity = extract_polarity_markers(right)
    left_modality = extract_modality_markers(left)
    right_modality = extract_modality_markers(right)
    left_entities = extract_named_entities(left)
    right_entities = extract_named_entities(right)
    left_reporting = frozenset(left_tokens) & REPORTING_MARKERS
    right_reporting = frozenset(right_tokens) & REPORTING_MARKERS
    length_ratio = (
        min(len(left_tokens), len(right_tokens)) / max(len(left_tokens), len(right_tokens))
        if left_tokens or right_tokens
        else 1.0
    )
    features = {
        "source_token_count": len(left_tokens),
        "target_token_count": len(right_tokens),
        "token_length_ratio": length_ratio,
        "token_jaccard": _set_jaccard(left_tokens, right_tokens),
        "content_token_jaccard": _set_jaccard(
            left_content,
            right_content,
        ),
        "content_token_containment": _set_containment(
            left_content,
            right_content,
        ),
        "character_sequence_ratio": SequenceMatcher(
            None,
            " ".join(left_tokens),
            " ".join(right_tokens),
            autojunk=False,
        ).ratio(),
        "numeric_source": dict(left_numbers),
        "numeric_target": dict(right_numbers),
        "numeric_mismatch": bool(left_numbers and right_numbers and left_numbers != right_numbers),
        "numeric_one_sided": bool(left_numbers) != bool(right_numbers),
        "polarity_source": sorted(left_polarity),
        "polarity_target": sorted(right_polarity),
        "polarity_mismatch": bool(left_polarity) != bool(right_polarity),
        "modality_source": sorted(left_modality),
        "modality_target": sorted(right_modality),
        "modality_mismatch": bool(
            left_modality and right_modality and left_modality != right_modality
        ),
        "entity_source": sorted(left_entities),
        "entity_target": sorted(right_entities),
        "surface_entity_disjoint": bool(
            left_entities and right_entities and not (left_entities & right_entities)
        ),
        "reporting_source": sorted(left_reporting),
        "reporting_target": sorted(right_reporting),
        "reporting_mismatch": bool(left_reporting != right_reporting),
        "quote_mismatch": bool(QUOTE_RE.search(left)) != bool(QUOTE_RE.search(right)),
    }
    return features


def _observable_tags(
    *,
    error_type: str,
    cosine: float,
    threshold: float,
    surface: Mapping[str, Any],
    wordnet: Mapping[str, Any],
) -> list[str]:
    tags = []
    if abs(cosine - threshold) < 0.03:
        tags.append("score_near_threshold")
    if cosine >= 0.90:
        tags.append("very_high_cosine")
    if cosine < 0.70:
        tags.append("very_low_cosine")
    for key in (
        "numeric_mismatch",
        "numeric_one_sided",
        "polarity_mismatch",
        "modality_mismatch",
        "surface_entity_disjoint",
        "reporting_mismatch",
        "quote_mismatch",
    ):
        if surface[key]:
            tags.append(key)
    if surface["token_length_ratio"] < 0.70:
        tags.append("large_length_asymmetry")
    if surface["content_token_jaccard"] >= 0.65:
        tags.append("high_content_overlap")
    if surface["content_token_jaccard"] < 0.30:
        tags.append("low_content_overlap")
    if wordnet.get("gate_conflict"):
        tags.append("wordnet_gate_conflict")
    if not tags:
        tags.append(
            "unexplained_surface_false_positive"
            if error_type == "false_positive"
            else "unexplained_surface_false_negative"
        )
    return tags


def _pair_row(
    *,
    pair: MRPCPair,
    cosine: float,
    threshold: float,
    wordnet_senses: Mapping[str, Mapping[str, frozenset[str]]],
) -> dict[str, Any]:
    predicted = cosine >= threshold
    error_type = (
        "false_positive"
        if predicted and not pair.is_paraphrase
        else "false_negative"
        if not predicted and pair.is_paraphrase
        else "true_positive"
        if predicted
        else "true_negative"
    )
    surface = _surface_features(pair.sentence1, pair.sentence2)
    wordnet = _wordnet_pair_features(
        wordnet_senses.get(pair.sentence1, {}),
        wordnet_senses.get(pair.sentence2, {}),
    )
    return {
        "pair_id": pair.pair_id,
        "label": pair.is_paraphrase,
        "predicted_paraphrase": predicted,
        "outcome": error_type,
        "cosine": cosine,
        "threshold": threshold,
        "signed_margin": cosine - threshold,
        "absolute_margin": abs(cosine - threshold),
        "sentence1_sha256": sha256_text(pair.sentence1),
        "sentence2_sha256": sha256_text(pair.sentence2),
        "sentence1": pair.sentence1,
        "sentence2": pair.sentence2,
        "surface": surface,
        "wordnet": wordnet,
        "observable_tags": _observable_tags(
            error_type=error_type,
            cosine=cosine,
            threshold=threshold,
            surface=surface,
            wordnet=wordnet,
        ),
    }


def _tag_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["observable_tags"])
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _score_bins(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "lower_inclusive": lower,
            "upper_exclusive": upper,
            "false_positive_count": sum(
                row["outcome"] == "false_positive" and lower <= row["cosine"] < upper
                for row in rows
            ),
            "false_negative_count": sum(
                row["outcome"] == "false_negative" and lower <= row["cosine"] < upper
                for row in rows
            ),
            "positive_count": sum(row["label"] and lower <= row["cosine"] < upper for row in rows),
            "negative_count": sum(
                not row["label"] and lower <= row["cosine"] < upper for row in rows
            ),
        }
        for lower, upper in SCORE_BINS
    ]


def _shadow_verifier_diagnostics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in rows], dtype=bool)
    raw_predictions = np.asarray(
        [row["predicted_paraphrase"] for row in rows],
        dtype=bool,
    )
    rules = {
        "numeric_mismatch": lambda row: row["surface"]["numeric_mismatch"],
        "polarity_mismatch": lambda row: row["surface"]["polarity_mismatch"],
        "modality_mismatch": lambda row: row["surface"]["modality_mismatch"],
        "surface_entity_disjoint": lambda row: row["surface"]["surface_entity_disjoint"],
        "wordnet_gate_conflict": lambda row: row["wordnet"]["gate_conflict"],
        "numeric_or_polarity": lambda row: (
            row["surface"]["numeric_mismatch"] or row["surface"]["polarity_mismatch"]
        ),
        "any_observable_conflict": lambda row: (
            row["surface"]["numeric_mismatch"]
            or row["surface"]["polarity_mismatch"]
            or row["surface"]["modality_mismatch"]
            or row["surface"]["surface_entity_disjoint"]
            or row["wordnet"]["gate_conflict"]
        ),
    }
    diagnostics = {}
    for rule_id, rule in rules.items():
        conflicts = np.asarray([bool(rule(row)) for row in rows])
        rejected = raw_predictions & conflicts
        predictions = raw_predictions & ~conflicts
        removed_false_positives = int(np.count_nonzero(rejected & ~labels))
        removed_true_positives = int(np.count_nonzero(rejected & labels))
        diagnostics[rule_id] = {
            "raw_candidates_rejected": int(np.count_nonzero(rejected)),
            "removed_false_positives": removed_false_positives,
            "removed_true_positives": removed_true_positives,
            "conflict_negative_predictive_precision": (
                removed_false_positives / (removed_false_positives + removed_true_positives)
                if removed_false_positives + removed_true_positives
                else None
            ),
            "resulting_metrics": _binary_metrics(labels, predictions),
        }
    return diagnostics


def _write_error_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "pair_id",
        "outcome",
        "cosine",
        "signed_margin",
        "observable_tags",
        "content_token_jaccard",
        "content_token_containment",
        "token_length_ratio",
        "numeric_mismatch",
        "polarity_mismatch",
        "modality_mismatch",
        "surface_entity_disjoint",
        "wordnet_gate_conflict",
        "sentence1",
        "sentence2",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            surface = row["surface"]
            writer.writerow(
                {
                    **row,
                    "observable_tags": ",".join(row["observable_tags"]),
                    "content_token_jaccard": surface["content_token_jaccard"],
                    "content_token_containment": surface["content_token_containment"],
                    "token_length_ratio": surface["token_length_ratio"],
                    "numeric_mismatch": surface["numeric_mismatch"],
                    "polarity_mismatch": surface["polarity_mismatch"],
                    "modality_mismatch": surface["modality_mismatch"],
                    "surface_entity_disjoint": surface["surface_entity_disjoint"],
                    "wordnet_gate_conflict": row["wordnet"]["gate_conflict"],
                }
            )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mrpc-dir",
        type=Path,
        default=APP_ROOT / "utils" / "msrpc_split",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--endpoint-metadata",
        type=Path,
        default=DEFAULT_ENDPOINT,
    )
    parser.add_argument(
        "--wordnet-sense-cache",
        type=Path,
        default=DEFAULT_WORDNET_CACHE,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = load_mrpc_pairs(args.mrpc_dir.resolve())
    source_join_evidence = _validate_source_join(
        pairs,
        args.mrpc_dir.resolve(),
    )
    splits = split_mrpc_by_sentence_component(
        pairs,
        calibration_percent=DEFAULT_MRPC_CALIBRATION_PERCENT,
        seed=DEFAULT_SPLIT_SEED,
    )
    threshold = paraphrase_native_cosine_threshold("Qwen/Qwen3-Embedding-8B")
    cache = ReadOnlyEmbeddingCache(args.cache.resolve())
    try:
        texts = sorted({text for pair in pairs for text in (pair.sentence1, pair.sentence2)})
        vectors = {text: cache.get(text) for text in texts}
        cache_evidence = {
            "path": str(cache.path),
            "sha256": _sha256_file(cache.path),
            "row_count": cache.row_count,
            "metadata": cache.metadata,
        }
    finally:
        cache.close()
    endpoint_path = args.endpoint_metadata.resolve()
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    wordnet_path = (
        args.wordnet_sense_cache.resolve() if args.wordnet_sense_cache is not None else None
    )
    wordnet_senses, wordnet_evidence = _load_wordnet_senses(
        wordnet_path,
        pairs,
    )

    held_out_pairs = [pair for pair in pairs if splits[pair.pair_id] == "held_out"]
    calibration_pairs = [pair for pair in pairs if splits[pair.pair_id] == "calibration"]
    held_out_scores = np.asarray(
        [float(np.dot(vectors[pair.sentence1], vectors[pair.sentence2])) for pair in held_out_pairs]
    )
    held_out_labels = np.asarray(
        [pair.is_paraphrase for pair in held_out_pairs],
        dtype=bool,
    )
    calibration_scores = np.asarray(
        [
            float(np.dot(vectors[pair.sentence1], vectors[pair.sentence2]))
            for pair in calibration_pairs
        ]
    )
    calibration_labels = np.asarray(
        [pair.is_paraphrase for pair in calibration_pairs],
        dtype=bool,
    )
    unique_text_token_counts = [len(_tokens(text)) for text in texts]
    calibrated_threshold, calibration_metrics = fit_cosine_threshold(
        calibration_scores,
        calibration_labels,
    )
    rows = [
        _pair_row(
            pair=pair,
            cosine=float(cosine),
            threshold=threshold,
            wordnet_senses=wordnet_senses,
        )
        for pair, cosine in zip(
            held_out_pairs,
            held_out_scores,
            strict=True,
        )
    ]
    errors = [row for row in rows if row["outcome"] in {"false_positive", "false_negative"}]
    false_positives = [row for row in errors if row["outcome"] == "false_positive"]
    false_negatives = [row for row in errors if row["outcome"] == "false_negative"]
    metrics = _binary_metrics(
        held_out_labels,
        held_out_scores >= threshold,
    )
    if metrics["false_positive"] != 196 or metrics["false_negative"] != 71:
        raise RawQwenAuditError(f"baseline drifted from FP=196/FN=71: {metrics}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "classifier": "raw_qwen_pair_cosine",
            "psa_routing_applied": False,
            "paraphrase_grouping_applied": False,
            "production_threshold": threshold,
            "split": "held_out",
            "held_out_pair_count": len(held_out_pairs),
            "held_out_positive_count": int(np.count_nonzero(held_out_labels)),
            "held_out_negative_count": int(np.count_nonzero(~held_out_labels)),
        },
        "evidence": {
            "cache": cache_evidence,
            "endpoint_metadata": {
                "path": str(endpoint_path),
                "sha256": _sha256_file(endpoint_path),
                "repository": endpoint.get("repository"),
                "revision": endpoint.get("revision"),
                "task": endpoint.get("model", {}).get("task"),
                "arguments": endpoint.get("model", {}).get("args"),
                "runtime_image": (
                    endpoint.get("model", {}).get("image", {}).get("tei", {}).get("url")
                ),
            },
            "embedding_request_contract": {
                "input_transform": "verbatim_raw_mrpc_sentence",
                "task_instruction_applied": False,
                "tei_route": "/embed",
                "request_shape": {"inputs": ["<raw sentence>"]},
            },
            "input_length": {
                "unit": "audit_regex_token",
                "unique_sentence_count": len(texts),
                "maximum": max(unique_text_token_counts),
                "quantiles": _quantiles(unique_text_token_counts),
                "endpoint_auto_truncate_enabled": (
                    "--auto-truncate" in (endpoint.get("model", {}).get("args") or [])
                ),
            },
            "wordnet": wordnet_evidence,
            "source_join_validation": source_join_evidence,
            "split_policy": {
                "unit": "connected_component_of_normalized_sentence_identity",
                "seed": DEFAULT_SPLIT_SEED,
                "calibration_percent": DEFAULT_MRPC_CALIBRATION_PERCENT,
                "exact_sentence_leakage_allowed": False,
            },
        },
        "baseline": {
            "held_out_metrics": metrics,
            "all_positive_held_out_metrics": _binary_metrics(
                held_out_labels,
                np.ones(len(held_out_labels), dtype=bool),
            ),
            "specificity": (
                metrics["true_negative"] / (metrics["true_negative"] + metrics["false_positive"])
            ),
            "false_positive_rate": (
                metrics["false_positive"] / (metrics["true_negative"] + metrics["false_positive"])
            ),
            "balanced_accuracy": _balanced_accuracy(metrics),
            "roc_auc": _roc_auc(held_out_labels, held_out_scores),
            "average_precision": _average_precision(
                held_out_labels,
                held_out_scores,
            ),
            "calibration_selected_threshold": float(calibrated_threshold),
            "calibration_metrics": calibration_metrics,
            "calibration_threshold_held_out_metrics": _binary_metrics(
                held_out_labels,
                held_out_scores >= calibrated_threshold,
            ),
            "held_out_oracle_thresholds_diagnostic_only": _best_thresholds(
                held_out_labels,
                held_out_scores,
            ),
            "score_quantiles": {
                "positive": _quantiles(held_out_scores[held_out_labels].tolist()),
                "negative": _quantiles(held_out_scores[~held_out_labels].tolist()),
                "false_positive": _quantiles([row["cosine"] for row in false_positives]),
                "false_negative": _quantiles([row["cosine"] for row in false_negatives]),
            },
            "score_bins": _score_bins(rows),
            "shadow_verifier_diagnostics": (_shadow_verifier_diagnostics(rows)),
        },
        "error_inventory": {
            "error_count": len(errors),
            "false_positive_count": len(false_positives),
            "false_negative_count": len(false_negatives),
            "false_positive_observable_tag_counts": _tag_counts(false_positives),
            "false_negative_observable_tag_counts": _tag_counts(false_negatives),
            "errors": errors,
        },
    }
    report_id_body = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    report["report_id"] = (
        "mrpc-raw-qwen-failure-audit:" + hashlib.sha256(report_id_body.encode("utf-8")).hexdigest()
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "raw_qwen_failure_audit.json"
    tsv_path = output_dir / "raw_qwen_errors.tsv"
    _atomic_json(report_path, report)
    _write_error_tsv(tsv_path, errors)
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "metrics": metrics,
                "roc_auc": report["baseline"]["roc_auc"],
                "average_precision": report["baseline"]["average_precision"],
                "calibrated_threshold": calibrated_threshold,
                "error_count": len(errors),
                "false_positive_tag_counts": report["error_inventory"][
                    "false_positive_observable_tag_counts"
                ],
                "false_negative_tag_counts": report["error_inventory"][
                    "false_negative_observable_tag_counts"
                ],
                "report_path": str(report_path),
                "tsv_path": str(tsv_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
