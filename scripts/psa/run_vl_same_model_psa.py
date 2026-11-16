#!/usr/bin/env python3
"""Evaluate text-only versus multimodal PSA anchors in one Qwen3-VL space.

The experiment reuses the retained Qwen3-VL image vectors from the completed
Flickr8k pilot. It embeds only the missing guide and evaluation captions with
the exact same model revision, then measures candidate recall for image-text
and text-text positive pairs against negative controls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
APP_ROOT = REPOSITORY_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from scripts.research.psa.run_image_caption_neighborhood_experiment import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_MULTIMODAL_MODEL_ID,
    DEFAULT_MULTIMODAL_MODEL_REVISION,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    EmbeddingCache,
    ExperimentError,
    HostedEmbeddingService,
    ServiceConfig,
    embed_with_cache,
    preflight_text_service,
    validate_manifest,
)

from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    NATIVE_EMBEDDING_DIMENSION,
)

SCHEMA_VERSION = "aiidea_qwen3_vl_same_model_psa_v1"
DEFAULT_SOURCE_DIR = REPOSITORY_ROOT / "temp/image_caption_neighborhood/runs/32050657545"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "temp/qwen3_vl_same_model_psa"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _normalize(rows: Sequence[Sequence[float]], label: str) -> np.ndarray:
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ExperimentError(f"{label} does not form a vector matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(norms == 0.0):
        raise ExperimentError(f"{label} contains zero or non-finite vectors")
    return matrix / norms


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_reused_image_vectors(
    cache_path: Path,
    expected_artifact_ids: Sequence[str],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    if not cache_path.is_file():
        raise ExperimentError(f"source embedding cache does not exist: {cache_path}")
    connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT artifact_id, vector_json, endpoint_url, api_style
            FROM embeddings
            WHERE cache_schema=? AND modality='image'
              AND model_id=? AND model_revision=?
            ORDER BY artifact_id
            """,
            (
                CACHE_SCHEMA_VERSION,
                DEFAULT_MULTIMODAL_MODEL_ID,
                DEFAULT_MULTIMODAL_MODEL_REVISION,
            ),
        ).fetchall()
    finally:
        connection.close()

    vectors: dict[str, list[float]] = {}
    endpoints: set[str] = set()
    api_styles: set[str] = set()
    for artifact_id, vector_json, endpoint_url, api_style in rows:
        vector = [float(value) for value in json.loads(vector_json)]
        previous = vectors.get(artifact_id)
        if previous is not None and previous != vector:
            raise ExperimentError(f"conflicting cached image vector for {artifact_id}")
        vectors[artifact_id] = vector
        endpoints.add(str(endpoint_url))
        api_styles.add(str(api_style))

    missing = sorted(set(expected_artifact_ids) - set(vectors))
    if missing:
        raise ExperimentError(
            f"source cache is missing {len(missing)} required image vectors; first={missing[0]}"
        )
    selected = {artifact_id: vectors[artifact_id] for artifact_id in expected_artifact_ids}
    dimensions = {len(vector) for vector in selected.values()}
    if dimensions != {NATIVE_EMBEDDING_DIMENSION}:
        raise ExperimentError(
            f"expected reused {NATIVE_EMBEDDING_DIMENSION}D image vectors, observed {dimensions}"
        )
    return selected, {
        "path": str(cache_path),
        "reused_image_vectors": len(selected),
        "model_id": DEFAULT_MULTIMODAL_MODEL_ID,
        "model_revision": DEFAULT_MULTIMODAL_MODEL_REVISION,
        "source_endpoints": sorted(endpoints),
        "api_styles": sorted(api_styles),
    }


def _rank(scores: np.ndarray) -> list[int]:
    return [int(value) for value in np.argsort(-scores, kind="stable")]


def _compare(scores_a: np.ndarray, scores_b: np.ndarray, top_k: int) -> dict[str, float]:
    rank_a = _rank(scores_a)
    rank_b = _rank(scores_b)
    set_a = set(rank_a[:top_k])
    set_b = set(rank_b[:top_k])
    intersection = set_a & set_b
    union = set_a | set_b
    return {
        "candidate_recall_either_direction": float(rank_a[0] in set_b or rank_b[0] in set_a),
        "candidate_recall_both_directions": float(rank_a[0] in set_b and rank_b[0] in set_a),
        "jaccard": len(intersection) / len(union),
        "containment": len(intersection) / top_k,
        "guide_bank_fraction_union": len(union) / len(scores_a),
    }


def _bootstrap_mean(
    values: Sequence[float],
    *,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[draws].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "sample_count": len(array),
        "bootstrap_samples": samples,
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["frame"]), str(row["pair_kind"]), int(row["top_k"]))].append(
            row["metrics"]
        )
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for (frame, pair_kind, top_k), metric_rows in sorted(grouped.items()):
        condition = f"{frame}|{pair_kind}|k={top_k}"
        result[condition] = {
            "frame": frame,
            "pair_kind": pair_kind,
            "top_k": top_k,
            **{
                metric: _bootstrap_mean(
                    [float(row[metric]) for row in metric_rows],
                    rng=rng,
                    samples=bootstrap_samples,
                )
                for metric in (
                    "candidate_recall_either_direction",
                    "candidate_recall_both_directions",
                    "jaccard",
                    "containment",
                    "guide_bank_fraction_union",
                )
            },
        }
    return result


def _render_report(summary: Mapping[str, Any]) -> str:
    conditions = summary["metrics"]
    lines = [
        "# Qwen3-VL Same-Model PSA Evaluation",
        "",
        f"**Completed:** {summary['completed_utc']}",
        "",
        "The test reuses the prior Qwen3-VL image vectors and embeds only text with the ",
        "same pinned Qwen3-VL model. No production model, graph membership, or native ",
        "embedding configuration is changed.",
        "",
        "## Candidate recall",
        "",
        "| Frame | Pair type | k | Recall | 95% CI | Both directions | Jaccard | Bank union |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pair_order = {
        "image_text_positive": 0,
        "text_text_positive": 1,
        "image_text_shuffled": 2,
        "image_text_hard_negative": 3,
        "text_text_shuffled": 4,
        "text_text_hard_negative": 5,
    }
    ordered = sorted(
        conditions.values(),
        key=lambda row: (
            str(row["frame"]),
            pair_order.get(str(row["pair_kind"]), 99),
            int(row["top_k"]),
        ),
    )
    for row in ordered:
        recall = row["candidate_recall_either_direction"]
        both = row["candidate_recall_both_directions"]
        jaccard = row["jaccard"]
        union = row["guide_bank_fraction_union"]
        lines.append(
            "| {frame} | {pair_kind} | {top_k} | {recall:.1%} | {low:.1%}-{high:.1%} "
            "| {both:.1%} | {jaccard:.3f} | {union:.1%} |".format(
                frame=row["frame"],
                pair_kind=row["pair_kind"],
                top_k=row["top_k"],
                recall=recall["estimate"],
                low=recall["ci95_low"],
                high=recall["ci95_high"],
                both=both["estimate"],
                jaccard=jaccard["estimate"],
                union=union["estimate"],
            )
        )

    lines.extend(
        [
            "",
            "## Gate interpretation",
            "",
            f"- Coverage target: {summary['decision']['coverage_target']:.1%}.",
            f"- Text-guide-only result: **{summary['decision']['text_guides_only']}**.",
            "- Do guide-star images rescue the target: "
            f"**{summary['decision']['image_guides_rescue_target']}**.",
            "- Overall promotion recommendation: "
            f"**{summary['decision']['promotion_recommendation']}**.",
            "",
            "Image-caption association remains typed visual evidence, not automatic "
            "hard paraphrase membership.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, manifest_path.parent)
    guides = manifest["guide_stars"]
    evaluations = manifest["evaluation_pairs"]

    expected_image_ids = [
        *[f"guide:{row['guide_star_id']}:image" for row in guides],
        *[f"evaluation:{row['record_id']}:image" for row in evaluations],
    ]
    image_vectors, source_provenance = _load_reused_image_vectors(
        Path(args.source_cache).resolve(),
        expected_image_ids,
    )

    endpoint_url = args.multimodal_endpoint_url or os.getenv(args.multimodal_endpoint_env)
    if not endpoint_url:
        raise ExperimentError("Qwen3-VL endpoint is required for the missing text vectors")
    config = ServiceConfig(
        model_id=DEFAULT_MULTIMODAL_MODEL_ID,
        model_revision=DEFAULT_MULTIMODAL_MODEL_REVISION,
        endpoint_url=endpoint_url,
        provider="hf-endpoint",
        api_style="hf-multimodal",
        token_env=args.hf_token_env,
        timeout=args.timeout,
    )
    service = HostedEmbeddingService(config)
    preflight = preflight_text_service(service)

    text_artifacts: dict[str, str] = {}
    for guide in guides:
        text_artifacts[f"guide:{guide['guide_star_id']}:caption:0"] = guide["captions"][0]
    for row in evaluations:
        for caption_index, caption in enumerate(row["captions"]):
            text_artifacts[f"evaluation:{row['record_id']}:caption:{caption_index}"] = caption

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(output_dir / "vl_text_embeddings.sqlite3")
    try:
        text_vectors, text_cache_stats = embed_with_cache(
            artifacts=sorted(text_artifacts.items()),
            modality="text",
            service=service,
            cache=cache,
            batch_size=args.text_batch_size,
        )
        text_cache_rows = cache.row_count
        text_cache_responses = cache.response_count
    finally:
        cache.close()

    text_guides = _normalize(
        [text_vectors[f"guide:{guide['guide_star_id']}:caption:0"] for guide in guides],
        "VL text guide vectors",
    )
    image_guides = _normalize(
        [image_vectors[f"guide:{guide['guide_star_id']}:image"] for guide in guides],
        "VL image guide vectors",
    )

    def scores(vector: Sequence[float], frame: str, modality: str) -> np.ndarray:
        query = _normalize([vector], f"{modality} query")[0]
        text_scores = query @ text_guides.T
        image_scores = query @ image_guides.T
        if frame == "text_guides_only":
            return text_scores
        if frame == "paired_native_guides":
            return image_scores if modality == "image" else text_scores
        if frame == "fused_mean_guides":
            return (text_scores + image_scores) / 2.0
        if frame == "fused_max_guides":
            return np.maximum(text_scores, image_scores)
        raise ExperimentError(f"unknown frame {frame}")

    frames = (
        "text_guides_only",
        "paired_native_guides",
        "fused_mean_guides",
        "fused_max_guides",
    )
    top_ks = sorted(set(args.top_k))
    if not top_ks or top_ks[0] < 1 or top_ks[-1] > len(guides):
        raise ExperimentError("top-k values must fit inside the guide bank")

    rows: list[dict[str, Any]] = []

    def record_pair(
        *,
        frame: str,
        pair_kind: str,
        pair_id: str,
        left_scores: np.ndarray,
        right_scores: np.ndarray,
    ) -> None:
        for top_k in top_ks:
            rows.append(
                {
                    "frame": frame,
                    "pair_kind": pair_kind,
                    "pair_id": pair_id,
                    "top_k": top_k,
                    "metrics": _compare(left_scores, right_scores, top_k),
                }
            )

    evaluation_by_record = {row["record_id"]: row for row in evaluations}
    for row in evaluations:
        record_id = row["record_id"]
        image_vector = image_vectors[f"evaluation:{record_id}:image"]
        caption_vectors = [
            text_vectors[f"evaluation:{record_id}:caption:{index}"]
            for index in range(len(row["captions"]))
        ]
        shuffled = row["shuffled_negative"]
        hard_negative = row["hard_negative"]
        shuffled_vector = text_vectors[
            f"evaluation:{shuffled['record_id']}:caption:{shuffled['caption_index']}"
        ]
        hard_vector = text_vectors[
            f"evaluation:{hard_negative['record_id']}:caption:{hard_negative['caption_index']}"
        ]
        # Validate controls against the immutable manifest before evaluation.
        if (
            evaluation_by_record[shuffled["record_id"]]["captions"][shuffled["caption_index"]]
            != shuffled["caption"]
            or evaluation_by_record[hard_negative["record_id"]]["captions"][
                hard_negative["caption_index"]
            ]
            != hard_negative["caption"]
        ):
            raise ExperimentError("manifest control caption is stale")

        for frame in frames:
            image_scores = scores(image_vector, frame, "image")
            text_scores = [scores(vector, frame, "text") for vector in caption_vectors]
            shuffled_scores = scores(shuffled_vector, frame, "text")
            hard_scores = scores(hard_vector, frame, "text")
            for caption_index, caption_scores in enumerate(text_scores):
                record_pair(
                    frame=frame,
                    pair_kind="image_text_positive",
                    pair_id=f"{record_id}:image-caption-{caption_index}",
                    left_scores=image_scores,
                    right_scores=caption_scores,
                )
            for caption_index, caption_scores in enumerate(text_scores[1:], start=1):
                record_pair(
                    frame=frame,
                    pair_kind="text_text_positive",
                    pair_id=f"{record_id}:caption-0-caption-{caption_index}",
                    left_scores=text_scores[0],
                    right_scores=caption_scores,
                )
            for pair_kind, negative_scores in (
                ("image_text_shuffled", shuffled_scores),
                ("image_text_hard_negative", hard_scores),
            ):
                record_pair(
                    frame=frame,
                    pair_kind=pair_kind,
                    pair_id=f"{record_id}:{pair_kind}",
                    left_scores=image_scores,
                    right_scores=negative_scores,
                )
            for pair_kind, negative_scores in (
                ("text_text_shuffled", shuffled_scores),
                ("text_text_hard_negative", hard_scores),
            ):
                record_pair(
                    frame=frame,
                    pair_kind=pair_kind,
                    pair_id=f"{record_id}:{pair_kind}",
                    left_scores=text_scores[0],
                    right_scores=negative_scores,
                )

    metrics = _aggregate(
        rows,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    gate_k = max(top_ks)
    coverage_target = args.coverage_target

    def recall(frame: str, pair_kind: str) -> float:
        condition = metrics[f"{frame}|{pair_kind}|k={gate_k}"]
        return float(condition["candidate_recall_either_direction"]["estimate"])

    text_only_passes = all(
        recall("text_guides_only", pair_kind) >= coverage_target
        for pair_kind in ("image_text_positive", "text_text_positive")
    )
    image_frame_passes = any(
        all(
            recall(frame, pair_kind) >= coverage_target
            for pair_kind in ("image_text_positive", "text_text_positive")
        )
        for frame in ("paired_native_guides", "fused_mean_guides", "fused_max_guides")
    )
    if text_only_passes:
        text_guide_decision = "PASS"
        image_guide_decision = "NO - text guide stars already clear the coverage target"
        promotion = "QUALIFY SELECTIVITY AND VERIFICATION BEFORE MODEL PROMOTION"
    elif image_frame_passes:
        text_guide_decision = "FAIL"
        image_guide_decision = "YES - a frame containing guide-star images clears the target"
        promotion = "DO NOT PROMOTE UNTIL THE MULTIMODAL GUIDE FRAME IS VERSIONED"
    else:
        text_guide_decision = "FAIL"
        image_guide_decision = "NO - guide-star images do not rescue the coverage target"
        promotion = "KEEP QWEN3-EMBEDDING-8B; REDESIGN OR REJECT THE VL PSA FRAME"

    summary = {
        "schema_version": SCHEMA_VERSION,
        "completed_utc": _utc_now(),
        "configuration": {
            "model_id": DEFAULT_MULTIMODAL_MODEL_ID,
            "model_revision": DEFAULT_MULTIMODAL_MODEL_REVISION,
            "guide_count": len(guides),
            "evaluation_image_count": len(evaluations),
            "image_text_positive_pairs": len(evaluations) * 5,
            "text_text_positive_pairs": len(evaluations) * 4,
            "top_k": top_ks,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "production_model_changed": False,
            "graph_membership_written": False,
        },
        "preflight": preflight,
        "source_image_cache": source_provenance,
        "vl_text_cache": {
            "path": str(output_dir / "vl_text_embeddings.sqlite3"),
            "row_count": text_cache_rows,
            "response_count": text_cache_responses,
            **text_cache_stats,
        },
        "metrics": metrics,
        "decision": {
            "gate_top_k": gate_k,
            "coverage_target": coverage_target,
            "text_guides_only": text_guide_decision,
            "image_guides_rescue_target": image_guide_decision,
            "promotion_recommendation": promotion,
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    report_path = output_dir / "report.md"
    report_path.write_text(_render_report(summary), encoding="utf-8")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_SOURCE_DIR / "data/manifest.json"),
    )
    parser.add_argument(
        "--source-cache",
        default=str(DEFAULT_SOURCE_DIR / "embeddings.sqlite3"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--multimodal-endpoint-url")
    parser.add_argument(
        "--multimodal-endpoint-env",
        default="HF_QWEN3_VL_EMBEDDING_ENDPOINT",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--top-k", nargs="+", type=int, default=list(DEFAULT_TOP_K))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--coverage-target", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(parse_args(argv))
    except ExperimentError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
