#!/usr/bin/env python3
"""Evaluate an ordered WordNet/De Bruijn shadow verifier on raw Qwen MRPC.

The production De Bruijn compositor belongs to Stitch and does not participate
in paraphrase acceptance. This runner tests the related verifier hypothesis
without starting aiidea services or mutating the graph:

1. Reconstruct the clean official MRPC pairs.
2. Recover each sentence's ordered noun/verb synset sequence from the scoped
   Qwen WordNet sense cache.
3. Measure symmetric unigram and De Bruijn bigram overlap between each pair.
4. Select simple hard-rejection policies on the calibration split only.
5. Report their untouched held-out effect on the raw 0.77 Qwen decision.

Run this with ``temp/wordnet-venv/bin/python`` because ordered reconstruction
uses the pinned local NLTK and Open English WordNet installations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.research.mrpc.build_mrpc_wordnet_sense_cache import (
    OEWN_VERSION,
    WORD_RE,
    OfflineOEWNResolver,
)

DEFAULT_MRPC_DIR = REPOSITORY_ROOT / "app" / "utils" / "msrpc_split"
DEFAULT_SENTENCE_CACHE = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "qwen_embeddings.sqlite3"
)
DEFAULT_SENSE_CACHE = (
    REPOSITORY_ROOT
    / "temp"
    / "mrpc_wordnet_qwen_replay"
    / "30475782422-neo4j-score"
    / "qwen_wordnet_senses.json"
)
DEFAULT_PSA_REPORT = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "mrpc_report.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_raw_qwen_failure_audit_2026-07-29"
    / "wordnet_debruijn_shadow.json"
)
DEFAULT_WORDNET_DATA = REPOSITORY_ROOT / "temp" / "wordnet"
DEFAULT_NLTK_DATA = REPOSITORY_ROOT / "temp" / "nltk_data"
CALIBRATION_PERCENT = 35
SPLIT_SEED = "qwen-typed-field-test-mrpc-sentence-components-v1"
RAW_THRESHOLD = 0.77
SCHEMA_VERSION = "aiidea_mrpc_wordnet_debruijn_shadow_v1"


class ShadowVerifierError(RuntimeError):
    """Raised when shadow-verifier evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class Pair:
    pair_id: int
    sentence1: str
    sentence2: str
    label: bool


@dataclass(frozen=True, slots=True)
class SequenceFeatures:
    source_length: int
    target_length: int
    unigram_jaccard: float | None
    unigram_containment: float | None
    bigram_jaccard: float | None
    bigram_containment: float | None
    ordered_sequence_ratio: float | None
    shared_synset_count: int
    shared_bigram_count: int


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_pairs(mrpc_dir: Path) -> list[Pair]:
    corpus_path = mrpc_dir.parent / "msr_paraphrase_data.txt"
    test_path = mrpc_dir.parent / "msr_paraphrase_test.txt"
    sentence_by_id: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
        for row in reader:
            sentence_id = str(row.get("Sentence ID") or "").strip()
            text = str(row.get("String") or "")
            if sentence_id and text:
                sentence_by_id[sentence_id] = text
    pairs = []
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
        for pair_id, row in enumerate(reader, start=1):
            try:
                first = sentence_by_id[str(row["#1 ID"]).strip()]
                second = sentence_by_id[str(row["#2 ID"]).strip()]
            except KeyError as exc:
                raise ShadowVerifierError(
                    f"MRPC pair {pair_id} references a missing sentence"
                ) from exc
            label = str(row.get("Quality") or "").strip()
            if label not in {"0", "1"}:
                raise ShadowVerifierError(f"MRPC pair {pair_id} has invalid label {label!r}")
            pairs.append(Pair(pair_id, first, second, label == "1"))
    if len(pairs) != 1_725:
        raise ShadowVerifierError(f"expected 1,725 official MRPC pairs, found {len(pairs)}")
    return pairs


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _normalized_sentence(text: str) -> str:
    return " ".join(text.casefold().split())


def _component_split(pairs: Sequence[Pair]) -> dict[int, str]:
    graph = _UnionFind()
    normalized: dict[int, tuple[str, str]] = {}
    for pair in pairs:
        left = _normalized_sentence(pair.sentence1)
        right = _normalized_sentence(pair.sentence2)
        graph.union(left, right)
        normalized[pair.pair_id] = (left, right)
    members: dict[str, list[str]] = {}
    for left, right in normalized.values():
        members.setdefault(graph.find(left), []).extend((left, right))
    component_split = {}
    for root, values in members.items():
        digest = hashlib.sha256(
            _canonical_json(
                {
                    "seed": SPLIT_SEED,
                    "sentences": sorted(set(values)),
                }
            ).encode("utf-8")
        ).hexdigest()
        bucket = int(digest[:8], 16) % 100
        component_split[root] = "calibration" if bucket < CALIBRATION_PERCENT else "held_out"
    return {
        pair_id: component_split[graph.find(left)] for pair_id, (left, _right) in normalized.items()
    }


def build_debruijn_subsequences(
    sequence: Sequence[str],
    k: int = 2,
) -> frozenset[tuple[str, ...]]:
    """Return the same sliding k-window structure used by the Stitch helper."""

    if not sequence:
        return frozenset()
    effective_k = max(1, min(int(k), len(sequence)))
    return frozenset(
        tuple(sequence[index : index + effective_k])
        for index in range(len(sequence) - effective_k + 1)
    )


def _jaccard(
    left: Iterable[Any],
    right: Iterable[Any],
) -> float | None:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else None


def _containment(
    left: Iterable[Any],
    right: Iterable[Any],
) -> float | None:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return None
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def sequence_features(
    source: Sequence[str],
    target: Sequence[str],
) -> SequenceFeatures:
    source_unigrams = frozenset(source)
    target_unigrams = frozenset(target)
    source_bigrams = build_debruijn_subsequences(source, 2)
    target_bigrams = build_debruijn_subsequences(target, 2)
    ordered_ratio = (
        SequenceMatcher(
            None,
            tuple(source),
            tuple(target),
            autojunk=False,
        ).ratio()
        if source and target
        else None
    )
    return SequenceFeatures(
        source_length=len(source),
        target_length=len(target),
        unigram_jaccard=_jaccard(source_unigrams, target_unigrams),
        unigram_containment=_containment(
            source_unigrams,
            target_unigrams,
        ),
        bigram_jaccard=_jaccard(source_bigrams, target_bigrams),
        bigram_containment=_containment(
            source_bigrams,
            target_bigrams,
        ),
        ordered_sequence_ratio=ordered_ratio,
        shared_synset_count=len(source_unigrams & target_unigrams),
        shared_bigram_count=len(source_bigrams & target_bigrams),
    )


def _binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    true_positive = int(np.count_nonzero(labels & predictions))
    false_positive = int(np.count_nonzero(~labels & predictions))
    false_negative = int(np.count_nonzero(labels & ~predictions))
    true_negative = int(np.count_nonzero(~labels & ~predictions))
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(labels),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": ((true_positive + true_negative) / len(labels) if len(labels) else 0.0),
    }


def _load_vectors(
    path: Path,
    texts: Sequence[str],
) -> dict[str, np.ndarray]:
    connection = sqlite3.connect(path)
    try:
        vectors = {}
        for text in texts:
            text_hash = _sha256_text(text)
            row = connection.execute(
                """
                SELECT vector, dimension
                FROM embeddings
                WHERE namespace = 'primary' AND text_sha256 = ?
                """,
                (text_hash,),
            ).fetchone()
            if row is None:
                raise ShadowVerifierError(f"sentence cache omitted {text_hash}")
            vector = np.frombuffer(
                row[0],
                dtype=np.float32,
                count=int(row[1]),
            )
            vectors[text] = vector
        return vectors
    finally:
        connection.close()


def _ordered_synset_sequences(
    *,
    texts: Sequence[str],
    sense_payload: Mapping[str, Any],
    wordnet_data: Path,
    nltk_data: Path,
) -> dict[str, tuple[str, ...]]:
    import nltk
    import wn
    from nltk.stem import PorterStemmer
    from wn.morphy import Morphy

    wn.config.data_directory = wordnet_data.resolve()
    nltk.data.path.insert(0, str(nltk_data.resolve()))
    wordnet = wn.Wordnet(OEWN_VERSION)
    if len(wordnet.synsets()) != 120_630:
        raise ShadowVerifierError("local OEWN 2024 inventory is incomplete")
    resolver = OfflineOEWNResolver(
        wordnet,
        Morphy(wordnet),
        PorterStemmer(),
    )
    tagged = nltk.pos_tag_sents([WORD_RE.findall(text) for text in texts])
    sentence_entries = sense_payload.get("sentences")
    if not isinstance(sentence_entries, dict):
        raise ShadowVerifierError("sense cache has no sentence mapping")
    sequences = {}
    for text, tags in zip(texts, tagged, strict=True):
        text_hash = _sha256_text(text)
        entry = sentence_entries.get(text_hash)
        if not isinstance(entry, dict):
            raise ShadowVerifierError(f"sense cache omitted sentence {text_hash}")
        assignments = {
            (str(item["lemma"]), str(item["pos"])): str(item["synset_id"])
            for item in (entry.get("assignments") or [])
        }
        sequence = []
        for token, tag in tags:
            pos = "n" if str(tag).startswith("NN") else "v" if str(tag).startswith("VB") else None
            if pos is None:
                continue
            lemma = resolver._lemma(str(token), pos)
            synset_id = assignments.get((lemma, pos)) if lemma else None
            if synset_id:
                sequence.append(synset_id)
        sequences[text] = tuple(sequence)
    return sequences


def _evaluate_rule(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature: str,
    threshold: float,
    minimum_each: int,
    require_psa_top8: bool,
) -> tuple[dict[str, Any], dict[str, int]]:
    labels = np.asarray([bool(row["label"]) for row in rows])
    predictions = []
    rejected = []
    for row in rows:
        features = row["sequence_features"]
        value = features[feature]
        enough = (
            features["source_length"] >= minimum_each
            and features["target_length"] >= minimum_each
            and value is not None
        )
        sequence_reject = enough and value < threshold
        psa_reject = require_psa_top8 and row.get("psa_signature_rank_at_8") is None
        prediction = bool(row["raw_prediction"] and not sequence_reject and not psa_reject)
        predictions.append(prediction)
        rejected.append(bool(row["raw_prediction"] and not prediction))
    prediction_array = np.asarray(predictions)
    rejected_array = np.asarray(rejected)
    return _binary_metrics(labels, prediction_array), {
        "raw_candidates_rejected": int(np.count_nonzero(rejected_array)),
        "removed_false_positives": int(np.count_nonzero(rejected_array & ~labels)),
        "removed_true_positives": int(np.count_nonzero(rejected_array & labels)),
    }


def _calibrate_rule(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature: str,
) -> dict[str, Any]:
    best: tuple[Any, ...] | None = None
    for minimum_each in range(1, 6):
        for threshold in np.linspace(0.0, 1.0, 41):
            metrics, effects = _evaluate_rule(
                rows,
                feature=feature,
                threshold=float(threshold),
                minimum_each=minimum_each,
                require_psa_top8=False,
            )
            candidate = (
                float(metrics["f1"]),
                float(metrics["accuracy"]),
                -int(metrics["false_positive"]),
                -int(metrics["false_negative"]),
                -float(threshold),
                -minimum_each,
                float(threshold),
                minimum_each,
                metrics,
                effects,
            )
            if best is None or candidate[:6] > best[:6]:
                best = candidate
    if best is None:
        raise ShadowVerifierError("could not calibrate sequence rule")
    return {
        "feature": feature,
        "threshold": best[6],
        "minimum_synsets_each": best[7],
        "require_psa_signature_top8": False,
        "calibration_metrics": best[8],
        "calibration_effects": best[9],
    }


def _feature_summary(
    rows: Sequence[Mapping[str, Any]],
    feature: str,
) -> dict[str, Any]:
    values = [
        row["sequence_features"][feature]
        for row in rows
        if row["sequence_features"][feature] is not None
    ]
    positive = [
        row["sequence_features"][feature]
        for row in rows
        if row["label"] and row["sequence_features"][feature] is not None
    ]
    negative = [
        row["sequence_features"][feature]
        for row in rows
        if not row["label"] and row["sequence_features"][feature] is not None
    ]
    return {
        "covered_pair_count": len(values),
        "positive_median": (float(np.median(positive)) if positive else None),
        "negative_median": (float(np.median(negative)) if negative else None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrpc-dir", type=Path, default=DEFAULT_MRPC_DIR)
    parser.add_argument(
        "--sentence-cache",
        type=Path,
        default=DEFAULT_SENTENCE_CACHE,
    )
    parser.add_argument(
        "--sense-cache",
        type=Path,
        default=DEFAULT_SENSE_CACHE,
    )
    parser.add_argument(
        "--psa-report",
        type=Path,
        default=DEFAULT_PSA_REPORT,
    )
    parser.add_argument(
        "--wordnet-data",
        type=Path,
        default=DEFAULT_WORDNET_DATA,
    )
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=DEFAULT_NLTK_DATA,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = _load_pairs(args.mrpc_dir.resolve())
    splits = _component_split(pairs)
    texts = sorted({text for pair in pairs for text in (pair.sentence1, pair.sentence2)})
    vectors = _load_vectors(args.sentence_cache.resolve(), texts)
    sense_path = args.sense_cache.resolve()
    sense_payload = json.loads(sense_path.read_text(encoding="utf-8"))
    sequences = _ordered_synset_sequences(
        texts=texts,
        sense_payload=sense_payload,
        wordnet_data=args.wordnet_data.resolve(),
        nltk_data=args.nltk_data.resolve(),
    )
    psa_path = args.psa_report.resolve()
    psa_payload = json.loads(psa_path.read_text(encoding="utf-8"))
    psa_rows = {int(row["pair_id"]): row for row in psa_payload["held_out_rows"]}

    rows = []
    for pair in pairs:
        score = float(np.dot(vectors[pair.sentence1], vectors[pair.sentence2]))
        features = sequence_features(
            sequences[pair.sentence1],
            sequences[pair.sentence2],
        )
        psa_row = psa_rows.get(pair.pair_id, {})
        rows.append(
            {
                "pair_id": pair.pair_id,
                "split": splits[pair.pair_id],
                "label": pair.label,
                "raw_cosine": score,
                "raw_prediction": score >= RAW_THRESHOLD,
                "psa_signature_rank_at_8": psa_row.get("signature_rank_at_8"),
                "sequence_features": asdict(features),
            }
        )
    calibration = [row for row in rows if row["split"] == "calibration"]
    held_out = [row for row in rows if row["split"] == "held_out"]
    held_out_labels = np.asarray([bool(row["label"]) for row in held_out])
    raw_metrics = _binary_metrics(
        held_out_labels,
        np.asarray([bool(row["raw_prediction"]) for row in held_out]),
    )
    if (
        raw_metrics["count"] != 1_110
        or raw_metrics["false_positive"] != 196
        or raw_metrics["false_negative"] != 71
    ):
        raise ShadowVerifierError(f"raw baseline drifted from the audited result: {raw_metrics}")

    feature_names = (
        "unigram_jaccard",
        "unigram_containment",
        "bigram_jaccard",
        "bigram_containment",
        "ordered_sequence_ratio",
    )
    policies = []
    for feature in feature_names:
        policy = _calibrate_rule(
            calibration,
            feature=feature,
        )
        held_metrics, held_effects = _evaluate_rule(
            held_out,
            feature=feature,
            threshold=float(policy["threshold"]),
            minimum_each=int(policy["minimum_synsets_each"]),
            require_psa_top8=False,
        )
        policy["held_out_metrics"] = held_metrics
        policy["held_out_effects"] = held_effects
        psa_metrics, psa_effects = _evaluate_rule(
            held_out,
            feature=feature,
            threshold=float(policy["threshold"]),
            minimum_each=int(policy["minimum_synsets_each"]),
            require_psa_top8=True,
        )
        policy["held_out_with_unlabeled_psa_top8_metrics"] = psa_metrics
        policy["held_out_with_unlabeled_psa_top8_effects"] = psa_effects
        policies.append(policy)

    psa_only_metrics, psa_only_effects = _evaluate_rule(
        held_out,
        feature="unigram_jaccard",
        threshold=0.0,
        minimum_each=1,
        require_psa_top8=True,
    )

    zero_sequences = sum(not sequence for sequence in sequences.values())
    sequence_lengths = [len(sequence) for sequence in sequences.values()]
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "production_mutation": False,
            "aiidea_services_started": False,
            "classifier": "raw_qwen_cosine_plus_shadow_verifier",
            "raw_threshold": RAW_THRESHOLD,
            "split_seed": SPLIT_SEED,
            "calibration_percent": CALIBRATION_PERCENT,
        },
        "evidence": {
            "mrpc_pair_count": len(pairs),
            "unique_sentence_count": len(texts),
            "sentence_cache": {
                "path": str(args.sentence_cache.resolve()),
                "sha256": _sha256_file(args.sentence_cache.resolve()),
            },
            "sense_cache": {
                "path": str(sense_path),
                "sha256": _sha256_file(sense_path),
                "resolver": sense_payload.get("resolver"),
            },
            "psa_report": {
                "path": str(psa_path),
                "sha256": _sha256_file(psa_path),
            },
        },
        "sequence_coverage": {
            "zero_sequence_sentence_count": zero_sequences,
            "sequence_length_quantiles": {
                str(probability): float(np.quantile(sequence_lengths, probability))
                for probability in (
                    0.0,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    1.0,
                )
            },
            "held_out_feature_summaries": {
                feature: _feature_summary(held_out, feature) for feature in feature_names
            },
        },
        "raw_held_out_metrics": raw_metrics,
        "unlabeled_psa_top8_shadow_gate": {
            "calibrated_on_mrpc_labels": False,
            "policy": (
                "reject a raw-positive pair only when its paired target was "
                "absent from the PSA signature top-8 retrieval result"
            ),
            "held_out_metrics": psa_only_metrics,
            "held_out_effects": psa_only_effects,
        },
        "calibrated_shadow_policies": policies,
        "held_out_raw_errors": [
            row for row in held_out if bool(row["label"]) != bool(row["raw_prediction"])
        ],
    }
    report_body = _canonical_json(report)
    report["report_id"] = (
        "mrpc-wordnet-debruijn-shadow:" + hashlib.sha256(report_body.encode("utf-8")).hexdigest()
    )
    output_path = args.output.resolve()
    _atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "raw_held_out_metrics": raw_metrics,
                "sequence_coverage": report["sequence_coverage"],
                "calibrated_shadow_policies": policies,
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
