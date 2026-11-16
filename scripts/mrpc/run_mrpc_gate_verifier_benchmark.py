#!/usr/bin/env python3
"""Evaluate MRPC through the restored paraphrase decision funnel.

The offline replay follows the production order without requiring services:

1. Load cached native sentence embeddings.
2. Retrieve a bounded neighborhood in the 115-dimensional PSA signature.
3. Apply a native-cosine gate whose cold start is 0.77 and whose active value
   is learned on the calibration partition only.
4. Apply the paraphrase verifier probes to every non-exact gated pair.

MRPC labels are never used for PSA nomination or verification. They are used
only to learn the scalar gate on the calibration split and score held-out
pairs. This is a pair-decision diagnostic, not a graph-ingestion replay.
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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from scripts.research.mrpc.run_mrpc_current_pipeline_benchmark import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    _load_pinned_frame,
    _project_representations,
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
    DEFAULT_MRPC_CALIBRATION_PERCENT,
    DEFAULT_SPLIT_SEED,
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

DEFAULT_CACHE = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "qwen_embeddings.sqlite3"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "reports" / "mrpc_gate_verifier_2026-07-30"
REPORT_SCHEMA_VERSION = "aiidea_mrpc_gate_verifier_benchmark_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _psa_neighborhoods(
    signatures_by_text: Mapping[str, np.ndarray],
    *,
    top_k: int,
) -> tuple[dict[str, frozenset[str]], dict[str, Any]]:
    """Return exact deterministic PSA top-k neighbors for every sentence."""

    texts = sorted(signatures_by_text, key=lambda text: (sha256_text(text), text))
    if top_k < 1 or top_k >= len(texts):
        raise BenchmarkError("PSA top_k must be positive and below corpus size")
    matrix = np.stack([signatures_by_text[text] for text in texts]).astype(
        np.float32,
        copy=False,
    )
    neighborhoods: dict[str, frozenset[str]] = {}
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
        neighborhoods[text] = frozenset(texts[index] for index in ordered)
    return neighborhoods, {
        "implementation": "exact_psa_cosine_top_k",
        "top_k": top_k,
        "sentence_count": len(texts),
        "signature_dimension": int(matrix.shape[1]),
    }


def _learn_gate_threshold(
    cosines: np.ndarray,
    labels: np.ndarray,
    nominated: np.ndarray,
    verifier_passed: np.ndarray,
    *,
    cold_start: float,
) -> tuple[float, dict[str, Any]]:
    """Fit the scalar first gate on calibration decisions only."""

    cosines = np.asarray(cosines, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    nominated = np.asarray(nominated, dtype=bool)
    verifier_passed = np.asarray(verifier_passed, dtype=bool)
    if not (len(cosines) == len(labels) == len(nominated) == len(verifier_passed) and len(labels)):
        raise BenchmarkError("gate-learning arrays must be aligned and non-empty")
    if not math.isfinite(cold_start) or not 0.0 <= cold_start <= 1.0:
        raise BenchmarkError("cold-start threshold must be in [0, 1]")

    eligible = nominated & verifier_passed
    thresholds = np.unique(
        np.concatenate(
            (
                cosines[(cosines >= 0.0) & (cosines <= 1.0)],
                np.asarray([cold_start], dtype=np.float64),
            )
        )
    )
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in thresholds:
        predictions = eligible & (cosines >= threshold)
        metrics = _binary_metrics(labels, predictions)
        precision = float(metrics["precision"]) if metrics["precision"] is not None else -1.0
        score = (
            float(metrics["f1"]),
            precision,
            float(threshold),
        )
        if best is None or score > best[0]:
            best = (score, float(threshold), metrics)
    if best is None:
        raise BenchmarkError("gate learning produced no threshold")
    return best[1], best[2]


def _learn_recall_floor_gate_threshold(
    cosines: np.ndarray,
    labels: np.ndarray,
    nominated: np.ndarray,
    verifier_passed: np.ndarray,
    *,
    cold_start: float,
    minimum_positive_recall: float,
) -> tuple[float, dict[str, Any]]:
    """Select the strictest candidate gate meeting a calibration recall floor.

    This objective is intended for a cascade: the gate preserves nearly every
    labeled positive available in the PSA neighborhood and leaves precision
    work to the deterministic verifier.  Verifier outcomes must be represented
    by an all-true mask when the threshold is learned for candidate routing.
    """

    cosines = np.asarray(cosines, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    nominated = np.asarray(nominated, dtype=bool)
    verifier_passed = np.asarray(verifier_passed, dtype=bool)
    if not (len(cosines) == len(labels) == len(nominated) == len(verifier_passed) and len(labels)):
        raise BenchmarkError("gate-learning arrays must be aligned and non-empty")
    if not math.isfinite(cold_start) or not 0.0 <= cold_start <= 1.0:
        raise BenchmarkError("cold-start threshold must be in [0, 1]")
    if not 0.0 < minimum_positive_recall <= 1.0:
        raise BenchmarkError("minimum_positive_recall must be in (0, 1]")

    eligible = nominated & verifier_passed
    thresholds = np.unique(
        np.concatenate(
            (
                cosines[(cosines >= 0.0) & (cosines <= 1.0)],
                np.asarray([cold_start], dtype=np.float64),
            )
        )
    )
    evaluated: list[tuple[float, dict[str, Any]]] = []
    for threshold in thresholds:
        predictions = eligible & (cosines >= threshold)
        evaluated.append((float(threshold), _binary_metrics(labels, predictions)))

    feasible = [
        value for value in evaluated if float(value[1]["recall"] or 0.0) >= minimum_positive_recall
    ]
    if feasible:
        threshold, metrics = max(
            feasible,
            key=lambda value: (
                value[0],
                float(value[1]["precision"] or 0.0),
            ),
        )
        target_met = True
    else:
        threshold, metrics = max(
            evaluated,
            key=lambda value: (
                float(value[1]["recall"] or 0.0),
                float(value[1]["precision"] or 0.0),
                value[0],
            ),
        )
        target_met = False
    return threshold, {
        **metrics,
        "objective": "strictest_threshold_meeting_calibration_positive_recall_floor",
        "minimum_positive_recall": minimum_positive_recall,
        "target_met": target_met,
        "tie_break": "highest_threshold_then_precision",
        "psa_positive_recall_ceiling": float(np.mean(eligible[labels])),
    }


def _evaluate(
    pairs: Sequence[MRPCPair],
    *,
    native_by_text: Mapping[str, np.ndarray],
    neighborhoods: Mapping[str, frozenset[str]],
    threshold: float,
    verifier_passed: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([pair.is_paraphrase for pair in pairs], dtype=bool)
    cosines = np.asarray(
        [
            float(
                np.dot(
                    native_by_text[pair.sentence1],
                    native_by_text[pair.sentence2],
                )
            )
            for pair in pairs
        ]
    )
    exact = np.asarray(
        [sha256_text(pair.sentence1) == sha256_text(pair.sentence2) for pair in pairs]
    )
    nominated = np.asarray([pair.sentence2 in neighborhoods[pair.sentence1] for pair in pairs])
    gate_passed = cosines >= threshold
    predictions = exact | (nominated & gate_passed & verifier_passed)
    traces = [
        {
            "pair_id": pair.pair_id,
            "label": pair.is_paraphrase,
            "exact_text_identity": bool(exact[index]),
            "psa_nominated": bool(nominated[index]),
            "native_similarity": float(cosines[index]),
            "native_candidate_threshold": threshold,
            "native_gate_passed": bool(gate_passed[index]),
            "verifier_passed": bool(verifier_passed[index]),
            "predicted_paraphrase": bool(predictions[index]),
        }
        for index, pair in enumerate(pairs)
    ]
    return {
        "metrics": _binary_metrics(labels, predictions),
        "stage_counts": {
            "exact_identity": int(np.count_nonzero(exact)),
            "psa_nominated": int(np.count_nonzero(nominated)),
            "native_gate_passed": int(np.count_nonzero(nominated & gate_passed)),
            "verifier_passed": int(np.count_nonzero(nominated & gate_passed & verifier_passed)),
        },
        "positive_stage_recall": {
            "psa_nominated": float(np.mean(nominated[labels])),
            "native_gate_passed": float(np.mean((nominated & gate_passed)[labels])),
            "verifier_passed": float(np.mean((nominated & gate_passed & verifier_passed)[labels])),
        },
    }, traces


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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--psa-top-k",
        type=int,
        default=PSA_SIGNATURE_CANDIDATE_LIMIT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = load_mrpc_pairs(args.mrpc_dir.resolve())
    splits = split_mrpc_by_sentence_component(pairs)
    calibration_pairs = [pair for pair in pairs if splits[pair.pair_id] == "calibration"]
    held_out_pairs = [pair for pair in pairs if splits[pair.pair_id] == "held_out"]
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
    neighborhoods, neighborhood_evidence = _psa_neighborhoods(
        signatures_by_text,
        top_k=args.psa_top_k,
    )

    production_controller = ParaphraseVerificationController(
        require_complete_evidence=False,
        enforce_fact_conflicts=False,
    )
    strict_shadow_controller = ParaphraseVerificationController(
        require_complete_evidence=False,
        enforce_fact_conflicts=True,
    )

    def _decision(
        controller: ParaphraseVerificationController,
        pair: MRPCPair,
    ) -> Any:
        return controller.verify(
            source_text=pair.sentence1,
            target_text=pair.sentence2,
            native_similarity=float(
                np.dot(
                    native_by_text[pair.sentence1],
                    native_by_text[pair.sentence2],
                )
            ),
            candidate_type="free_sentence",
            psa_local=(pair.sentence2 in neighborhoods[pair.sentence1]),
            wordnet_passes=True,
            wordnet_agreement=1.0,
            wordnet_reason="coverage_empty_side",
        )

    production_decisions = [_decision(production_controller, pair) for pair in pairs]
    strict_shadow_decisions = [_decision(strict_shadow_controller, pair) for pair in pairs]
    verifier_passed = np.asarray([decision.accepted for decision in production_decisions])
    strict_shadow_passed = np.asarray([decision.accepted for decision in strict_shadow_decisions])
    verifier_reasons = Counter(decision.reason for decision in production_decisions)
    strict_shadow_reasons = Counter(decision.reason for decision in strict_shadow_decisions)
    surface_conflicts = Counter(
        result.reason
        for decision in production_decisions
        for result in decision.probe_results
        if result.status == "conflict"
    )

    calibration_cosines = np.asarray(
        [
            float(
                np.dot(
                    native_by_text[pair.sentence1],
                    native_by_text[pair.sentence2],
                )
            )
            for pair in calibration_pairs
        ]
    )
    calibration_labels = np.asarray(
        [pair.is_paraphrase for pair in calibration_pairs],
        dtype=bool,
    )
    calibration_nominated = np.asarray(
        [pair.sentence2 in neighborhoods[pair.sentence1] for pair in calibration_pairs]
    )
    verifier_by_pair_id = {
        pair.pair_id: bool(verifier_passed[index]) for index, pair in enumerate(pairs)
    }
    calibration_verifier_passed = np.asarray(
        [verifier_by_pair_id[pair.pair_id] for pair in calibration_pairs]
    )
    cold_start = paraphrase_native_cosine_threshold(QWEN3_EMBEDDING_MODEL_ID)
    learned_threshold, learned_calibration_metrics = _learn_gate_threshold(
        calibration_cosines,
        calibration_labels,
        calibration_nominated,
        calibration_verifier_passed,
        cold_start=cold_start,
    )

    held_out_verifier_passed = np.asarray(
        [verifier_by_pair_id[pair.pair_id] for pair in held_out_pairs]
    )
    strict_shadow_by_pair_id = {
        pair.pair_id: bool(strict_shadow_passed[index]) for index, pair in enumerate(pairs)
    }
    held_out_strict_shadow_passed = np.asarray(
        [strict_shadow_by_pair_id[pair.pair_id] for pair in held_out_pairs]
    )
    cold_report, cold_traces = _evaluate(
        held_out_pairs,
        native_by_text=native_by_text,
        neighborhoods=neighborhoods,
        threshold=cold_start,
        verifier_passed=held_out_verifier_passed,
    )
    learned_report, learned_traces = _evaluate(
        held_out_pairs,
        native_by_text=native_by_text,
        neighborhoods=neighborhoods,
        threshold=learned_threshold,
        verifier_passed=held_out_verifier_passed,
    )
    strict_shadow_report, strict_shadow_traces = _evaluate(
        held_out_pairs,
        native_by_text=native_by_text,
        neighborhoods=neighborhoods,
        threshold=learned_threshold,
        verifier_passed=held_out_strict_shadow_passed,
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_class": "offline_pair_decision_diagnostic",
        "pipeline": [
            "cached_native_embedding",
            "psa_signature_top_k_neighborhood",
            "learned_native_cosine_candidate_gate",
            "numeric_polarity_modality_entity_wordnet_verifier_suite",
        ],
        "split_policy": {
            "unit": "connected_component_of_normalized_sentence_identity",
            "seed": DEFAULT_SPLIT_SEED,
            "calibration_percent": DEFAULT_MRPC_CALIBRATION_PERCENT,
            "threshold_labels": "calibration_only",
            "held_out_labels_visible_during_decision": False,
        },
        "evidence": {
            "cache": cache_evidence,
            "cache_sha256": _file_sha256(args.cache.resolve()),
            "frame": frame_evidence,
            "psa_neighborhood": neighborhood_evidence,
            "signature_distribution": signature_distribution,
        },
        "threshold": {
            "cold_start": cold_start,
            "learned": learned_threshold,
            "learning_objective": "calibration_f1_then_precision_then_stricter",
            "learned_calibration_metrics": learned_calibration_metrics,
        },
        "verifier": {
            "require_complete_evidence": False,
            "production_enforce_fact_conflicts": False,
            "strict_shadow_enforce_fact_conflicts": True,
            "wordnet_coverage": "unavailable_offline",
            "production_decision_reason_counts": dict(sorted(verifier_reasons.items())),
            "strict_shadow_decision_reason_counts": dict(sorted(strict_shadow_reasons.items())),
            "surface_conflict_counts": dict(sorted(surface_conflicts.items())),
        },
        "held_out": {
            "pair_count": len(held_out_pairs),
            "cold_start_gate": cold_report,
            "learned_gate": learned_report,
            "strict_shadow_learned_gate": strict_shadow_report,
        },
        "held_out_pair_traces": {
            "cold_start_gate": cold_traces,
            "learned_gate": learned_traces,
            "strict_shadow_learned_gate": strict_shadow_traces,
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mrpc_gate_verifier_report.json"
    _atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "report": str(output_path),
                "cold_start_threshold": cold_start,
                "learned_threshold": learned_threshold,
                "cold_start_held_out": cold_report["metrics"],
                "learned_held_out": learned_report["metrics"],
                "strict_shadow_learned_held_out": (strict_shadow_report["metrics"]),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
