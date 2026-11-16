#!/usr/bin/env python3
"""Run the bounded Qwen text/image guide-star neighborhood pilot.

The experiment deliberately keeps the Qwen3 text vectors and Qwen3-VL image
vectors in separate native spaces.  Each query is compared only with guide
stars embedded by the same model.  Cross-model evidence consists solely of
stable guide-star IDs, ranks, and set overlap.

The CLI has three stages:

* ``prepare`` materializes a deterministic Flickr8k manifest and its images;
* ``preflight`` makes one minimal request to each configured hosted service;
* ``run`` preflights again, embeds through a resumable SQLite cache, evaluates
  controls, and writes JSONL, JSON, and Markdown outputs.

There is intentionally no local model-loading path and no graph write path.
Similarity nominates candidates; it never creates paraphrase membership.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
MCP_LIB_ROOT = APP_ROOT / "mcp_servers" / "mcp_lib"
if str(MCP_LIB_ROOT) not in sys.path:
    # Import the leaf contract without executing mcp_lib's service-client package
    # initializer; this pilot should not acquire Redis/aiohttp runtime dependencies.
    sys.path.insert(0, str(MCP_LIB_ROOT))

from semantic_routing_contract import (  # noqa: E402
    PSA_SIGNATURE_CANDIDATE_LIMIT,
    PSA_SIGNATURE_RERANK_LIMIT,
    QWEN3_EMBEDDING_MODEL_ID,
)

from consolidated_clustering_v2.routing.guide_stars import GuideStarManager  # noqa: E402

SCHEMA_VERSION = "aiidea_image_caption_neighborhood_v1"
MANIFEST_SCHEMA_VERSION = "aiidea_image_caption_manifest_v1"
CACHE_SCHEMA_VERSION = "aiidea_hosted_embedding_cache_v1"
DEFAULT_MULTIMODAL_MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_TEXT_MODEL_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
DEFAULT_MULTIMODAL_MODEL_REVISION = "2c4565515e0f265c6511776e7193b22c0968ddc7"
DEFAULT_DATASET = "intro/flickr8k"
DEFAULT_DATASET_REVISION = "c7a02cb487da0eca729e133194b0f842cb578c08"
DEFAULT_GUIDE_COUNT = 115
DEFAULT_EVALUATION_COUNT = 200
DEFAULT_RANDOM_SUBSET_COUNT = 64
DEFAULT_SEED = 20260817
DEFAULT_TOP_K = (4, PSA_SIGNATURE_RERANK_LIMIT, 16, PSA_SIGNATURE_CANDIDATE_LIMIT)
DEFAULT_WORK_DIR = REPOSITORY_ROOT / "temp" / "image_caption_neighborhood"
DEFAULT_DATASET_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_HF_ROUTER = "https://router.huggingface.co"
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)
_COUNT_WORDS = frozenset(
    {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "single",
        "couple",
        "few",
        "many",
        "several",
        "group",
    }
)


class ExperimentError(RuntimeError):
    """Raised for an explicit, non-recoverable experiment contract failure."""


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _stable_order(values: Iterable[int], seed: int, namespace: str) -> list[int]:
    rng = random.Random(f"{seed}:{namespace}")
    ordered = list(values)
    rng.shuffle(ordered)
    return ordered


def _stable_window(total: int, count: int, seed: int, namespace: str) -> list[int]:
    """Select a seeded contiguous window to keep Dataset Viewer reads bounded."""

    if count < 1 or count > total:
        raise ExperimentError("selection count must be positive and no larger than the split")
    rng = random.Random(f"{seed}:{namespace}:contiguous-window-v1")
    start = rng.randrange(total - count + 1)
    return list(range(start, start + count))


def _derangement(values: Sequence[int], seed: int, namespace: str) -> dict[int, int]:
    """Return a deterministic identity-to-source mapping with no fixed points."""

    ordered = list(values)
    if len(ordered) < 2:
        raise ExperimentError("a shuffled control requires at least two identities")
    shuffled = list(ordered)
    rng = random.Random(f"{seed}:{namespace}")
    for _attempt in range(100):
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(ordered, shuffled, strict=True)):
            return dict(zip(ordered, shuffled, strict=True))
    # A rotation is a deterministic, guaranteed fallback, not random augmentation.
    shuffled = ordered[1:] + ordered[:1]
    return dict(zip(ordered, shuffled, strict=True))


def _json_request(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 60.0,
    retries: int = 4,
) -> Any:
    headers = {
        "Accept": "application/json",
        "User-Agent": "aiidea-image-caption-neighborhood/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            if exc.code in {429, 502, 503, 504} and attempt < retries:
                time.sleep(min(2**attempt, 8))
                continue
            raise ExperimentError(
                f"hosted request failed with HTTP {exc.code} for {url}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
                continue
            raise ExperimentError(f"hosted request failed for {url}: {exc.reason}") from exc
    raise AssertionError("request retry loop exited unexpectedly")


def _download(url: str, path: Path, *, token: str | None, timeout: float) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    headers = {"User-Agent": "aiidea-image-caption-neighborhood/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            temporary.open("wb") as handle,
        ):
            while block := response.read(1024 * 1024):
                handle.write(block)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise ExperimentError(f"image download failed for {path.name}: {exc}") from exc
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ExperimentError(f"image download returned no bytes for {path.name}")
    os.replace(temporary, path)


def _hub_revision(dataset: str, *, token: str | None, timeout: float) -> str:
    url = "https://huggingface.co/api/datasets/" + urllib.parse.quote(dataset, safe="/")
    payload = _json_request(url, token=token, timeout=timeout)
    revision = payload.get("sha") if isinstance(payload, Mapping) else None
    if not isinstance(revision, str) or not revision:
        raise ExperimentError("Hugging Face dataset metadata did not include a revision")
    return revision


def _dataset_rows(
    *,
    dataset_server: str,
    dataset: str,
    config: str,
    split: str,
    row_indexes: Sequence[int],
    token: str | None,
    timeout: float,
) -> dict[int, Mapping[str, Any]]:
    """Fetch only pages containing selected row indexes from the Dataset Viewer."""

    pages = sorted({index // 100 for index in row_indexes})
    selected = set(row_indexes)
    rows: dict[int, Mapping[str, Any]] = {}
    for page in pages:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": page * 100,
                "length": 100,
            }
        )
        payload = _json_request(
            f"{dataset_server.rstrip('/')}/rows?{query}",
            token=token,
            timeout=timeout,
        )
        for item in payload.get("rows", []):
            row_index = item.get("row_idx")
            row = item.get("row")
            if row_index in selected and isinstance(row, Mapping):
                rows[int(row_index)] = row
    missing = selected - set(rows)
    if missing:
        raise ExperimentError(f"dataset viewer omitted selected rows: {sorted(missing)[:10]}")
    return rows


def _dataset_size(
    *,
    dataset_server: str,
    dataset: str,
    config: str,
    split: str,
    token: str | None,
    timeout: float,
) -> int:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": 0, "length": 1}
    )
    payload = _json_request(
        f"{dataset_server.rstrip('/')}/rows?{query}", token=token, timeout=timeout
    )
    total = payload.get("num_rows_total") if isinstance(payload, Mapping) else None
    if not isinstance(total, int) or total < 1:
        raise ExperimentError(f"dataset viewer returned an invalid size for split {split!r}")
    return total


def _caption_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token for token in _TOKEN_RE.findall(text.casefold()) if token not in _STOP_WORDS
    )


def _hard_negative(
    source: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_tokens = _caption_tokens(str(source["captions"][0]))
    source_counts = source_tokens & _COUNT_WORDS
    ranked: list[tuple[float, int, str, int, Mapping[str, Any]]] = []
    for candidate in candidates:
        if candidate["record_id"] == source["record_id"]:
            continue
        for caption_index, caption in enumerate(candidate["captions"]):
            tokens = _caption_tokens(str(caption))
            union = source_tokens | tokens
            lexical_overlap = len(source_tokens & tokens) / len(union) if union else 0.0
            count_mismatch = bool((tokens & _COUNT_WORDS) != source_counts)
            score = lexical_overlap + (0.05 if count_mismatch else 0.0)
            ranked.append(
                (
                    -score,
                    int(candidate["row_index"]),
                    str(caption),
                    caption_index,
                    candidate,
                )
            )
    if not ranked:
        raise ExperimentError("hard-negative construction has no distinct candidate")
    _neg_score, _row_index, caption, caption_index, candidate = min(ranked)
    overlap_tokens = sorted(source_tokens & _caption_tokens(caption))
    return {
        "record_id": candidate["record_id"],
        "caption_index": caption_index,
        "caption": caption,
        "method": "max_lexical_overlap_different_image_with_count_mismatch_bonus_v1",
        "shared_tokens": overlap_tokens,
        "lexical_jaccard": (
            len(source_tokens & _caption_tokens(caption))
            / len(source_tokens | _caption_tokens(caption))
            if source_tokens | _caption_tokens(caption)
            else 0.0
        ),
    }


def prepare_manifest(args: argparse.Namespace) -> Path:
    if args.guide_count != DEFAULT_GUIDE_COUNT:
        raise ExperimentError(
            f"this pilot reuses the repository's {DEFAULT_GUIDE_COUNT}-star frame contract"
        )
    if not 100 <= args.evaluation_count <= 250:
        raise ExperimentError("evaluation-count must be between 100 and 250")
    token = os.getenv(args.hf_token_env)
    observed_revision = _hub_revision(args.dataset, token=token, timeout=args.timeout)
    if observed_revision != args.dataset_revision:
        raise ExperimentError(
            "dataset main revision does not match the configured revision: "
            f"{observed_revision} != {args.dataset_revision}"
        )
    guide_total = _dataset_size(
        dataset_server=args.dataset_server,
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.guide_split,
        token=token,
        timeout=args.timeout,
    )
    evaluation_total = _dataset_size(
        dataset_server=args.dataset_server,
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.evaluation_split,
        token=token,
        timeout=args.timeout,
    )
    guide_indexes = _stable_window(guide_total, args.guide_count, args.seed, "guide")
    evaluation_indexes = _stable_window(
        evaluation_total,
        args.evaluation_count,
        args.seed,
        "evaluation",
    )
    guide_rows = _dataset_rows(
        dataset_server=args.dataset_server,
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.guide_split,
        row_indexes=guide_indexes,
        token=token,
        timeout=args.timeout,
    )
    evaluation_rows = _dataset_rows(
        dataset_server=args.dataset_server,
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.evaluation_split,
        row_indexes=evaluation_indexes,
        token=token,
        timeout=args.timeout,
    )
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"

    def materialize(
        split: str, row_index: int, row: Mapping[str, Any], *, guide_id: int | None
    ) -> dict[str, Any]:
        image = row.get("image")
        captions = [row.get(f"caption_{index}") for index in range(5)]
        if not isinstance(image, Mapping) or not isinstance(image.get("src"), str):
            raise ExperimentError(f"row {split}:{row_index} has no image URL")
        if any(not isinstance(caption, str) or not caption.strip() for caption in captions):
            raise ExperimentError(f"row {split}:{row_index} has incomplete captions")
        record_id = f"{args.dataset}:{args.dataset_config}:{split}:{row_index}"
        extension = Path(str(row.get("file_name") or "image.jpg")).suffix or ".jpg"
        local_name = f"{split}-{row_index}{extension.lower()}"
        local_path = image_dir / local_name
        _download(image["src"], local_path, token=token, timeout=args.timeout)
        result = {
            "record_id": record_id,
            "row_index": row_index,
            "split": split,
            "file_name": row.get("file_name") or local_name,
            "image_path": str(local_path.relative_to(output_dir)),
            "captions": captions,
        }
        if guide_id is not None:
            result["guide_star_id"] = guide_id
        return result

    guides = [
        materialize(args.guide_split, row_index, guide_rows[row_index], guide_id=index)
        for index, row_index in enumerate(guide_indexes)
    ]
    evaluations = [
        materialize(
            args.evaluation_split,
            row_index,
            evaluation_rows[row_index],
            guide_id=None,
        )
        for row_index in evaluation_indexes
    ]
    shuffled = _derangement(list(range(len(evaluations))), args.seed, "evaluation-caption-control")
    for index, evaluation in enumerate(evaluations):
        shuffled_source = evaluations[shuffled[index]]
        evaluation["positive_caption_index"] = 0
        evaluation["shuffled_negative"] = {
            "record_id": shuffled_source["record_id"],
            "caption_index": 0,
            "caption": shuffled_source["captions"][0],
            "method": "deterministic_derangement_v1",
        }
        evaluation["hard_negative"] = _hard_negative(evaluation, evaluations)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "dataset": {
            "name": args.dataset,
            "revision": args.dataset_revision,
            "config": args.dataset_config,
            "guide_split": args.guide_split,
            "evaluation_split": args.evaluation_split,
            "seed": args.seed,
            "selection_method": "seeded_contiguous_window_v1",
            "selected_guide_record_ids": [row["record_id"] for row in guides],
            "selected_evaluation_record_ids": [row["record_id"] for row in evaluations],
        },
        "guide_stars": guides,
        "evaluation_pairs": evaluations,
    }
    validate_manifest(manifest, output_dir)
    manifest_path = output_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def validate_manifest(manifest: Mapping[str, Any], base_dir: Path) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExperimentError("manifest schema is not supported")
    guides = manifest.get("guide_stars")
    evaluations = manifest.get("evaluation_pairs")
    if not isinstance(guides, list) or len(guides) != DEFAULT_GUIDE_COUNT:
        raise ExperimentError(f"manifest must contain exactly {DEFAULT_GUIDE_COUNT} guides")
    if not isinstance(evaluations, list) or not 100 <= len(evaluations) <= 250:
        raise ExperimentError("manifest must contain 100-250 evaluation pairs")
    guide_ids = [row.get("guide_star_id") for row in guides]
    if guide_ids != list(range(DEFAULT_GUIDE_COUNT)):
        raise ExperimentError("guide-star IDs must be stable contiguous rows 0-114")
    guide_records = {str(row.get("record_id")) for row in guides}
    evaluation_records = {str(row.get("record_id")) for row in evaluations}
    if len(guide_records) != len(guides) or len(evaluation_records) != len(evaluations):
        raise ExperimentError("guide and evaluation record IDs must be unique")
    if guide_records & evaluation_records:
        raise ExperimentError("guide stars and evaluation pairs are not disjoint")
    guide_splits = {str(row.get("split")) for row in guides}
    evaluation_splits = {str(row.get("split")) for row in evaluations}
    if guide_splits & evaluation_splits:
        raise ExperimentError("guide and evaluation splits must be disjoint")
    for row in [*guides, *evaluations]:
        image_path = base_dir / str(row.get("image_path") or "")
        if not image_path.is_file():
            raise ExperimentError(f"manifest image does not exist: {image_path}")
        captions = row.get("captions")
        if not isinstance(captions, list) or not captions:
            raise ExperimentError(f"manifest row {row.get('record_id')} has no captions")
    for row in evaluations:
        for control_name in ("shuffled_negative", "hard_negative"):
            control = row.get(control_name)
            if not isinstance(control, Mapping):
                raise ExperimentError(f"evaluation row lacks {control_name}")
            source_id = str(control.get("record_id") or "")
            if source_id == row["record_id"] or source_id not in evaluation_records:
                raise ExperimentError(
                    f"{control_name} must reference a different held-out evaluation record"
                )


@dataclass(frozen=True)
class ServiceConfig:
    model_id: str
    model_revision: str
    endpoint_url: str
    provider: str
    api_style: str
    token_env: str
    timeout: float

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.endpoint_url)
        hostname = (parsed.hostname or "").casefold()
        if parsed.scheme != "https" or not (
            hostname == "huggingface.co"
            or hostname.endswith((".huggingface.co", ".huggingface.cloud"))
        ):
            raise ExperimentError(
                "embedding endpoints must be HTTPS services hosted by Hugging Face"
            )
        if self.provider not in {"hf-inference", "hf-endpoint"}:
            raise ExperimentError(f"unsupported Hugging Face provider {self.provider!r}")
        if self.api_style not in {
            "hf-feature-extraction",
            "hf-multimodal",
            "openai-embeddings",
        }:
            raise ExperimentError(f"unsupported embedding API style {self.api_style!r}")

    @property
    def token(self) -> str:
        value = os.getenv(self.token_env)
        if not value:
            raise ExperimentError(
                f"Hugging Face credential environment variable {self.token_env!r} is not set"
            )
        return value


def _vectors_from_response(payload: Any, expected_count: int) -> list[list[float]]:
    candidate = payload
    if isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
        ordered = sorted(payload["data"], key=lambda row: int(row.get("index", 0)))
        candidate = [row.get("embedding") for row in ordered]
    elif isinstance(payload, Mapping):
        for key in ("embeddings", "vectors"):
            if key in payload:
                candidate = payload[key]
                break
        else:
            if "embedding" in payload:
                candidate = [payload["embedding"]]
    if (
        expected_count == 1
        and isinstance(candidate, list)
        and candidate
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in candidate
        )
    ):
        candidate = [candidate]
    if not isinstance(candidate, list) or len(candidate) != expected_count:
        raise ExperimentError(
            "hosted embedding response does not contain one vector per requested artifact"
        )
    vectors: list[list[float]] = []
    for index, vector in enumerate(candidate):
        if not isinstance(vector, list) or not vector:
            raise ExperimentError(f"embedding response item {index} is not a vector")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise ExperimentError(
                "embedding response is nested, empty, or contains non-finite values; "
                "the endpoint must return pooled vectors and the runner will not pool silently"
            )
        vectors.append([float(value) for value in vector])
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ExperimentError("embedding response vectors have inconsistent dimensions")
    return vectors


class HostedEmbeddingService:
    """Minimal client for one explicitly configured Hugging Face-hosted service."""

    def __init__(self, config: ServiceConfig):
        self.config = config

    def _payload(self, values: Sequence[str], *, modality: str) -> Mapping[str, Any]:
        if self.config.api_style == "hf-feature-extraction":
            if modality != "text":
                raise ExperimentError("hf-feature-extraction only accepts text in this pilot")
            return {"inputs": list(values), "options": {"wait_for_model": False}}
        if self.config.api_style == "hf-multimodal":
            if modality == "text":
                return {
                    "inputs": list(values),
                    "parameters": {"normalize": True},
                }
            if modality != "image":
                raise ExperimentError(f"unsupported modality {modality!r}")
            return {
                "inputs": [{"image": value} for value in values],
                "parameters": {"normalize": True},
            }
        if modality == "text":
            inputs: list[Any] = list(values)
        else:
            inputs = [
                {"content": [{"type": "image_url", "image_url": {"url": value}}]}
                for value in values
            ]
        return {"model": self.config.model_id, "input": inputs}

    def embed(self, values: Sequence[str], *, modality: str) -> tuple[list[list[float]], Any]:
        if not values:
            return [], []
        payload = _json_request(
            self.config.endpoint_url,
            method="POST",
            token=self.config.token,
            payload=self._payload(values, modality=modality),
            timeout=self.config.timeout,
        )
        return _vectors_from_response(payload, len(values)), payload


class EmbeddingCache:
    """Resumable per-artifact vector and raw-response cache."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_schema TEXT NOT NULL,
                modality TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                endpoint_url TEXT NOT NULL,
                api_style TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                PRIMARY KEY (
                    cache_schema, modality, model_id, model_revision,
                    endpoint_url, api_style, artifact_id
                )
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                request_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_schema TEXT NOT NULL,
                modality TEXT NOT NULL,
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                endpoint_url TEXT NOT NULL,
                api_style TEXT NOT NULL,
                artifact_ids_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_utc TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def _key(self, service: ServiceConfig, modality: str, artifact_id: str) -> tuple[str, ...]:
        return (
            CACHE_SCHEMA_VERSION,
            modality,
            service.model_id,
            service.model_revision,
            service.endpoint_url,
            service.api_style,
            artifact_id,
        )

    def get(self, service: ServiceConfig, modality: str, artifact_id: str) -> list[float] | None:
        row = self.connection.execute(
            """
            SELECT vector_json FROM embeddings
            WHERE cache_schema=? AND modality=? AND model_id=? AND model_revision=?
              AND endpoint_url=? AND api_style=? AND artifact_id=?
            """,
            self._key(service, modality, artifact_id),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self,
        service: ServiceConfig,
        modality: str,
        artifact_id: str,
        vector: Sequence[float],
        response: Any,
    ) -> None:
        values = [float(value) for value in vector]
        if not values or not all(math.isfinite(value) for value in values):
            raise ExperimentError("refusing to cache an empty or non-finite vector")
        self.connection.execute(
            """
            INSERT OR REPLACE INTO embeddings (
                cache_schema, modality, model_id, model_revision,
                endpoint_url, api_style, artifact_id,
                vector_json, response_json, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                *self._key(service, modality, artifact_id),
                json.dumps(values, separators=(",", ":")),
                json.dumps(response, separators=(",", ":")),
                _utc_now(),
            ),
        )
        self.connection.commit()

    def put_response(
        self,
        service: ServiceConfig,
        modality: str,
        artifact_ids: Sequence[str],
        response: Any,
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO responses (
                cache_schema, modality, model_id, model_revision,
                endpoint_url, api_style, artifact_ids_json,
                response_json, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                CACHE_SCHEMA_VERSION,
                modality,
                service.model_id,
                service.model_revision,
                service.endpoint_url,
                service.api_style,
                json.dumps(list(artifact_ids), separators=(",", ":")),
                json.dumps(response, separators=(",", ":")),
                _utc_now(),
            ),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    @property
    def row_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    @property
    def response_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM responses").fetchone()[0])


def embed_with_cache(
    *,
    artifacts: Sequence[tuple[str, str]],
    modality: str,
    service: HostedEmbeddingService,
    cache: EmbeddingCache,
    batch_size: int,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    if batch_size < 1:
        raise ExperimentError("embedding batch size must be positive")
    resolved: dict[str, list[float]] = {}
    missing: list[tuple[str, str]] = []
    hits = 0
    for artifact_id, value in artifacts:
        cached = cache.get(service.config, modality, artifact_id)
        if cached is None:
            missing.append((artifact_id, value))
        else:
            resolved[artifact_id] = cached
            hits += 1
    requests = 0
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors, response = service.embed(
            [value for _artifact_id, value in batch], modality=modality
        )
        requests += 1
        response_id = cache.put_response(
            service.config,
            modality,
            [artifact_id for artifact_id, _value in batch],
            response,
        )
        for index, ((artifact_id, _value), vector) in enumerate(zip(batch, vectors, strict=True)):
            row_response = {
                "response_id": response_id,
                "batch_index": index,
                "batch_size": len(batch),
            }
            cache.put(service.config, modality, artifact_id, vector, row_response)
            resolved[artifact_id] = vector
    return resolved, {"cache_hits": hits, "new_embeddings": len(missing), "requests": requests}


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.casefold()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def preflight_services(
    text_service: HostedEmbeddingService,
    image_service: HostedEmbeddingService,
) -> dict[str, Any]:
    text_result = preflight_text_service(text_service)
    image_vectors, _image_response = image_service.embed(
        ["data:image/png;base64," + base64.b64encode(_TINY_PNG).decode("ascii")],
        modality="image",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_utc": _utc_now(),
        "text": text_result,
        "image": {
            "model_id": image_service.config.model_id,
            "model_revision": image_service.config.model_revision,
            "provider": image_service.config.provider,
            "endpoint_url": image_service.config.endpoint_url,
            "api_style": image_service.config.api_style,
            "response_vector_count": len(image_vectors),
            "dimension": len(image_vectors[0]),
        },
        "native_vectors_compared_across_models": False,
    }


def preflight_text_service(text_service: HostedEmbeddingService) -> dict[str, Any]:
    """Validate the text service independently before paid image compute starts."""

    text_vectors, _text_response = text_service.embed(
        ["A person walks beside a dog outdoors."], modality="text"
    )
    return {
        "model_id": text_service.config.model_id,
        "model_revision": text_service.config.model_revision,
        "provider": text_service.config.provider,
        "endpoint_url": text_service.config.endpoint_url,
        "api_style": text_service.config.api_style,
        "response_vector_count": len(text_vectors),
        "dimension": len(text_vectors[0]),
    }


def _matrix(vectors: Sequence[Sequence[float]], label: str) -> np.ndarray:
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except ValueError as exc:
        raise ExperimentError(f"{label} vectors have inconsistent dimensions") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ExperimentError(f"{label} vectors do not form a matrix")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0.0):
        raise ExperimentError(f"{label} vectors contain zero or non-finite rows")
    return matrix


def _ranked(
    manager: GuideStarManager,
    scores: np.ndarray,
    guide_ids: Sequence[int],
) -> list[tuple[int, float]]:
    masked = np.full(DEFAULT_GUIDE_COUNT, -math.inf, dtype=np.float32)
    masked[list(guide_ids)] = scores[list(guide_ids)]
    ranked = manager.get_top_k_stars(masked, k=len(guide_ids))
    if {guide_id for guide_id, _score in ranked} != set(guide_ids):
        raise ExperimentError("guide-star candidate construction lost identity alignment")
    return ranked


def compare_neighborhoods(
    text_ranking: Sequence[tuple[int, float]],
    image_ranking: Sequence[tuple[int, float]],
    *,
    top_k: int,
    guide_bank_size: int,
) -> dict[str, Any]:
    if not 1 <= top_k <= min(len(text_ranking), len(image_ranking)):
        raise ExperimentError("top-k is outside the available guide bank")
    text_ids = [guide_id for guide_id, _score in text_ranking[:top_k]]
    image_ids = [guide_id for guide_id, _score in image_ranking[:top_k]]
    text_set = set(text_ids)
    image_set = set(image_ids)
    intersection = text_set & image_set
    union = text_set | image_set
    text_anchor = text_ids[0]
    image_anchor = image_ids[0]
    prefix_overlap = statistics.fmean(
        len(set(text_ids[:depth]) & set(image_ids[:depth])) / depth for depth in range(1, top_k + 1)
    )
    return {
        "top_k": top_k,
        "text_neighborhood_size": len(text_ids),
        "image_neighborhood_size": len(image_ids),
        "guide_bank_fraction_each": top_k / guide_bank_size,
        "guide_bank_fraction_union": len(union) / guide_bank_size,
        "intersection_size": len(intersection),
        "jaccard": len(intersection) / len(union) if union else 0.0,
        "containment_smaller": len(intersection) / min(len(text_set), len(image_set)),
        "rank_aware_prefix_overlap": prefix_overlap,
        "any_intersection": bool(intersection),
        # MRPC's either-direction recall asks whether either query nominated its
        # partner.  Here each modality's rank-1 anchor is the partner identity.
        "caption_nominates_image_anchor": image_anchor in text_set,
        "image_nominates_caption_anchor": text_anchor in image_set,
        "candidate_recall_either_direction": (image_anchor in text_set or text_anchor in image_set),
        "candidate_recall_both_directions": (image_anchor in text_set and text_anchor in image_set),
        "text_guide_star_ids": text_ids,
        "image_guide_star_ids": image_ids,
        "intersection_guide_star_ids": sorted(intersection),
    }


def _condition_result(
    *,
    text_scores: np.ndarray,
    image_scores: np.ndarray,
    guide_ids: Sequence[int],
    identity_to_image_source: Mapping[int, int],
    manager: GuideStarManager,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    aligned_image = np.full(DEFAULT_GUIDE_COUNT, -math.inf, dtype=np.float32)
    for identity in guide_ids:
        aligned_image[identity] = image_scores[identity_to_image_source[identity]]
    text_ranking = _ranked(manager, text_scores, guide_ids)
    image_ranking = _ranked(manager, aligned_image, guide_ids)
    text_rank = {guide_id: index + 1 for index, (guide_id, _score) in enumerate(text_ranking)}
    image_rank = {guide_id: index + 1 for index, (guide_id, _score) in enumerate(image_ranking)}
    decisions = []
    for guide_id in guide_ids:
        decisions.append(
            {
                "guide_star_id": guide_id,
                "image_source_guide_star_id": identity_to_image_source[guide_id],
                "text_similarity": float(text_scores[guide_id]),
                "image_similarity": float(aligned_image[guide_id]),
                "text_rank": text_rank[guide_id],
                "image_rank": image_rank[guide_id],
                "included_at_top_k": [
                    top_k
                    for top_k in top_ks
                    if text_rank[guide_id] <= top_k or image_rank[guide_id] <= top_k
                ],
                "included_text_at_top_k": [
                    top_k for top_k in top_ks if text_rank[guide_id] <= top_k
                ],
                "included_image_at_top_k": [
                    top_k for top_k in top_ks if image_rank[guide_id] <= top_k
                ],
            }
        )
    return {
        "guide_scores": decisions,
        "metrics_by_top_k": {
            str(top_k): compare_neighborhoods(
                text_ranking,
                image_ranking,
                top_k=top_k,
                guide_bank_size=len(guide_ids),
            )
            for top_k in top_ks
        },
    }


def _bootstrap_mean(
    values: Sequence[float], rng: np.random.Generator, samples: int
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ExperimentError("bootstrap requires a non-empty one-dimensional sample")
    draws = rng.integers(0, len(array), size=(samples, len(array)))
    estimates = array[draws].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_samples": samples,
    }


_AGGREGATE_FIELDS = (
    "caption_nominates_image_anchor",
    "image_nominates_caption_anchor",
    "candidate_recall_either_direction",
    "candidate_recall_both_directions",
    "any_intersection",
    "intersection_size",
    "jaccard",
    "containment_smaller",
    "rank_aware_prefix_overlap",
    "guide_bank_fraction_union",
)


def aggregate_results(
    rows: Sequence[Mapping[str, Any]], *, seed: int, bootstrap_samples: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    indexed: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        for top_k, metrics in row["metrics_by_top_k"].items():
            key = (row["pair_kind"], row["guide_bank"], row["guide_pairing"], int(top_k))
            grouped[key].append(metrics)
            indexed[
                (
                    row["evaluation_record_id"],
                    row["pair_kind"],
                    row["guide_bank"],
                    row["guide_pairing"],
                    int(top_k),
                )
            ] = metrics
    rng = np.random.default_rng(seed)
    conditions: dict[str, Any] = {}
    for key in sorted(grouped):
        pair_kind, bank, pairing, top_k = key
        metrics_rows = grouped[key]
        condition_id = f"{pair_kind}|{bank}|{pairing}|k={top_k}"
        aggregate = {
            field: _bootstrap_mean(
                [float(row[field]) for row in metrics_rows], rng, bootstrap_samples
            )
            for field in _AGGREGATE_FIELDS
        }
        aggregate.update(
            {
                "pair_kind": pair_kind,
                "guide_bank": bank,
                "guide_pairing": pairing,
                "top_k": top_k,
                "sample_count": len(metrics_rows),
                "mean_text_neighborhood_size": statistics.fmean(
                    row["text_neighborhood_size"] for row in metrics_rows
                ),
                "median_text_neighborhood_size": statistics.median(
                    row["text_neighborhood_size"] for row in metrics_rows
                ),
                "mean_image_neighborhood_size": statistics.fmean(
                    row["image_neighborhood_size"] for row in metrics_rows
                ),
                "median_image_neighborhood_size": statistics.median(
                    row["image_neighborhood_size"] for row in metrics_rows
                ),
            }
        )
        conditions[condition_id] = aggregate

    separations: dict[str, Any] = {}
    pair_kinds = ("shuffled_negative", "hard_negative")
    banks = sorted({row["guide_bank"] for row in rows})
    pairings = sorted({row["guide_pairing"] for row in rows})
    top_ks = sorted({int(value) for row in rows for value in row["metrics_by_top_k"]})
    record_ids = sorted({row["evaluation_record_id"] for row in rows})
    for negative_kind in pair_kinds:
        for bank in banks:
            for pairing in pairings:
                for top_k in top_ks:
                    differences: dict[str, list[float]] = defaultdict(list)
                    for record_id in record_ids:
                        positive = indexed[(record_id, "positive", bank, pairing, top_k)]
                        negative = indexed[(record_id, negative_kind, bank, pairing, top_k)]
                        for field in (
                            "candidate_recall_either_direction",
                            "jaccard",
                            "containment_smaller",
                            "rank_aware_prefix_overlap",
                        ):
                            differences[field].append(
                                float(positive[field]) - float(negative[field])
                            )
                    separation_id = f"positive-minus-{negative_kind}|{bank}|{pairing}|k={top_k}"
                    separations[separation_id] = {
                        field: _bootstrap_mean(values, rng, bootstrap_samples)
                        for field, values in differences.items()
                    }
    paired_identity_advantage: dict[str, Any] = {}
    for bank in banks:
        for top_k in top_ks:
            differences = defaultdict(list)
            for record_id in record_ids:
                paired = indexed[(record_id, "positive", bank, "paired", top_k)]
                shuffled = indexed[(record_id, "positive", bank, "shuffled", top_k)]
                for field in (
                    "candidate_recall_either_direction",
                    "jaccard",
                    "containment_smaller",
                    "rank_aware_prefix_overlap",
                ):
                    differences[field].append(float(paired[field]) - float(shuffled[field]))
            paired_identity_advantage[f"{bank}|k={top_k}"] = {
                field: _bootstrap_mean(values, rng, bootstrap_samples)
                for field, values in differences.items()
            }
    return {
        "conditions": conditions,
        "positive_negative_separation": separations,
        "paired_identity_advantage": paired_identity_advantage,
    }


def _percentage(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def render_report(summary: Mapping[str, Any]) -> str:
    top_ks = summary["configuration"]["top_k"]
    conditions = summary["metrics"]["conditions"]
    lines = [
        "# Image-Caption Guide-Star Neighborhood Pilot",
        "",
        f"**Run status:** completed at `{summary['completed_utc']}`",
        "",
        (
            "This feasibility pilot compares only same-model native vectors. Cross-model "
            "agreement is computed from guide-star identities and ranks; it does not authorize "
            "`member_of` or any typed membership edge."
        ),
        "",
        "## Primary results",
        "",
        (
            "| k | True candidate recall | Shuffled eval match | Hard-negative match | "
            "Mean intersection | Jaccard | Containment | Rank-aware overlap | Union of bank |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for top_k in top_ks:
        positive = conditions[f"positive|full|paired|k={top_k}"]
        shuffled = conditions[f"shuffled_negative|full|paired|k={top_k}"]
        hard = conditions[f"hard_negative|full|paired|k={top_k}"]
        lines.append(
            "| {k} | {positive} | {shuffled} | {hard} | {intersection:.2f} | "
            "{jaccard:.3f} | {containment:.3f} | {rank_overlap:.3f} | {fraction:.1%} |".format(
                k=top_k,
                positive=_percentage(positive["candidate_recall_either_direction"]["estimate"]),
                shuffled=_percentage(shuffled["candidate_recall_either_direction"]["estimate"]),
                hard=_percentage(hard["candidate_recall_either_direction"]["estimate"]),
                intersection=positive["intersection_size"]["estimate"],
                jaccard=positive["jaccard"]["estimate"],
                containment=positive["containment_smaller"]["estimate"],
                rank_overlap=positive["rank_aware_prefix_overlap"]["estimate"],
                fraction=positive["guide_bank_fraction_union"]["estimate"],
            )
        )
    primary_k = PSA_SIGNATURE_CANDIDATE_LIMIT
    positive = conditions[f"positive|full|paired|k={primary_k}"]
    shuffled = conditions[f"shuffled_negative|full|paired|k={primary_k}"]
    hard = conditions[f"hard_negative|full|paired|k={primary_k}"]
    identity = summary["metrics"]["paired_identity_advantage"][f"full|k={primary_k}"]
    shuffled_separation = summary["metrics"]["positive_negative_separation"][
        f"positive-minus-shuffled_negative|full|paired|k={primary_k}"
    ]
    hard_separation = summary["metrics"]["positive_negative_separation"][
        f"positive-minus-hard_negative|full|paired|k={primary_k}"
    ]
    candidate = positive["candidate_recall_either_direction"]
    caption_direction = positive["caption_nominates_image_anchor"]
    image_direction = positive["image_nominates_caption_anchor"]
    success = (
        candidate["estimate"] >= 0.99
        and shuffled_separation["candidate_recall_either_direction"]["ci95_low"] > 0.0
        and hard_separation["candidate_recall_either_direction"]["ci95_low"] > 0.0
        and identity["candidate_recall_either_direction"]["ci95_low"] > 0.0
        and positive["guide_bank_fraction_union"]["estimate"] <= 0.5
    )
    lines.extend(
        [
            "",
            f"## Ablations at k={primary_k}",
            "",
            (
                "| Guide bank | Identity pairing | True recall | Jaccard | Containment | "
                "Rank-aware overlap | Union of bank |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _condition_id, condition in sorted(conditions.items()):
        if condition["pair_kind"] != "positive" or condition["top_k"] != primary_k:
            continue
        lines.append(
            "| {bank} | {pairing} | {recall} | {jaccard:.3f} | {containment:.3f} | "
            "{rank_overlap:.3f} | {fraction:.1%} |".format(
                bank=condition["guide_bank"],
                pairing=condition["guide_pairing"],
                recall=_percentage(condition["candidate_recall_either_direction"]["estimate"]),
                jaccard=condition["jaccard"]["estimate"],
                containment=condition["containment_smaller"]["estimate"],
                rank_overlap=condition["rank_aware_prefix_overlap"]["estimate"],
                fraction=condition["guide_bank_fraction_union"]["estimate"],
            )
        )
    lines.extend(
        [
            "",
            "## Plain answers",
            "",
            (
                "1. **True pairs entering the same candidate neighborhood:** "
                f"{_percentage(candidate['estimate'])} at k={primary_k} "
                f"(95% bootstrap CI {_percentage(candidate['ci95_low'])} to "
                f"{_percentage(candidate['ci95_high'])}); caption-to-image-anchor recall was "
                f"{_percentage(caption_direction['estimate'])}, and image-to-caption-anchor "
                f"recall was {_percentage(image_direction['estimate'])}. The MRPC-analog "
                "definition is "
                "either modality's rank-1 anchor being nominated by the other modality's top-k."
            ),
            (
                "2. **Neighborhood size:** "
                f"{positive['mean_text_neighborhood_size']:.1f} text stars (median "
                f"{positive['median_text_neighborhood_size']:.1f}) and "
                f"{positive['mean_image_neighborhood_size']:.1f} image stars (median "
                f"{positive['median_image_neighborhood_size']:.1f}) on average; the "
                f"mean union covers {_percentage(positive['guide_bank_fraction_union']['estimate'])} "
                "of the guide bank."
            ),
            (
                "3. **Negative match rates:** "
                f"{_percentage(shuffled['candidate_recall_either_direction']['estimate'])} for "
                "shuffled captions (95% CI "
                f"{_percentage(shuffled['candidate_recall_either_direction']['ci95_low'])} to "
                f"{_percentage(shuffled['candidate_recall_either_direction']['ci95_high'])}) and "
                f"{_percentage(hard['candidate_recall_either_direction']['estimate'])} for hard "
                "negatives (95% CI "
                f"{_percentage(hard['candidate_recall_either_direction']['ci95_low'])} to "
                f"{_percentage(hard['candidate_recall_either_direction']['ci95_high'])})."
            ),
            (
                "4. **Paired versus shuffled guide identities:** paired identities changed true-pair "
                "candidate recall by "
                f"{identity['candidate_recall_either_direction']['estimate']:+.1%} "
                f"(95% CI {identity['candidate_recall_either_direction']['ci95_low']:+.1%} to "
                f"{identity['candidate_recall_either_direction']['ci95_high']:+.1%})."
            ),
            "5. **Larger-run decision:** "
            + (
                "YES. The preregistered feasibility rule passed."
                if success
                else "NO / NOT YET. At least one recall, selectivity, negative-separation, or "
                "paired-identity criterion did not pass."
            ),
            "",
            "## Reproduction evidence",
            "",
            (
                f"- Dataset: `{summary['dataset']['name']}` revision "
                f"`{summary['dataset']['revision']}`, guide split "
                f"`{summary['dataset']['guide_split']}`, evaluation split "
                f"`{summary['dataset']['evaluation_split']}`, seed "
                f"`{summary['dataset']['seed']}`."
            ),
            (
                f"- Held-out pairs: {summary['configuration']['evaluation_count']}; guide stars: "
                f"{summary['configuration']['guide_count']}."
            ),
            (
                f"- Text model: `{summary['preflight']['text']['model_id']}` revision "
                f"`{summary['preflight']['text']['model_revision']}`."
            ),
            (
                f"- Image model: `{summary['preflight']['image']['model_id']}` revision "
                f"`{summary['preflight']['image']['model_revision']}`."
            ),
            f"- Machine-readable rows: `{summary['outputs']['per_pair_jsonl']}`.",
            (
                "- Bootstrap: paired percentile bootstrap over evaluation image IDs; "
                f"{summary['configuration']['bootstrap_samples']} resamples."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _text_service_config(args: argparse.Namespace) -> ServiceConfig:
    text_endpoint = args.text_endpoint_url or (
        f"{DEFAULT_HF_ROUTER}/hf-inference/models/" + urllib.parse.quote(args.text_model, safe="/")
    )
    return ServiceConfig(
        model_id=args.text_model,
        model_revision=args.text_model_revision,
        endpoint_url=text_endpoint,
        provider=args.text_provider,
        api_style=args.text_api_style,
        token_env=args.hf_token_env,
        timeout=args.timeout,
    )


def _image_service_config(args: argparse.Namespace) -> ServiceConfig:
    image_endpoint = args.multimodal_endpoint_url or os.getenv(args.multimodal_endpoint_env)
    if not image_endpoint:
        raise ExperimentError(
            "no Qwen multimodal Hugging Face endpoint is configured; pass "
            "--multimodal-endpoint-url or set "
            f"{args.multimodal_endpoint_env}. No local fallback is permitted."
        )
    return ServiceConfig(
        model_id=args.multimodal_model,
        model_revision=args.multimodal_model_revision,
        endpoint_url=image_endpoint,
        provider=args.multimodal_provider,
        api_style=args.multimodal_api_style,
        token_env=args.hf_token_env,
        timeout=args.timeout,
    )


def _service_configs(args: argparse.Namespace) -> tuple[ServiceConfig, ServiceConfig]:
    return _text_service_config(args), _image_service_config(args)


def run_experiment(args: argparse.Namespace) -> Path:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, manifest_path.parent)
    text_config, image_config = _service_configs(args)
    text_service = HostedEmbeddingService(text_config)
    image_service = HostedEmbeddingService(image_config)

    # This always occurs before opening the bulk cache or issuing bulk requests.
    preflight = preflight_services(text_service, image_service)

    guides = manifest["guide_stars"]
    evaluations = manifest["evaluation_pairs"]
    cache = EmbeddingCache(Path(args.cache).resolve())
    try:
        text_artifacts: dict[str, str] = {}
        image_artifacts: dict[str, str] = {}
        for guide in guides:
            guide_id = int(guide["guide_star_id"])
            text_artifacts[f"guide:{guide_id}:caption:0"] = guide["captions"][0]
            image_artifacts[f"guide:{guide_id}:image"] = _image_data_url(
                manifest_path.parent / guide["image_path"]
            )
        evaluation_by_record = {row["record_id"]: row for row in evaluations}
        for row in evaluations:
            image_artifacts[f"evaluation:{row['record_id']}:image"] = _image_data_url(
                manifest_path.parent / row["image_path"]
            )
            text_artifacts[f"evaluation:{row['record_id']}:caption:0"] = row["captions"][0]
            for control_name in ("shuffled_negative", "hard_negative"):
                control = row[control_name]
                artifact_id = (
                    f"evaluation:{control['record_id']}:caption:{control['caption_index']}"
                )
                source = evaluation_by_record[control["record_id"]]
                source_caption = source["captions"][int(control["caption_index"])]
                if source_caption != control["caption"]:
                    raise ExperimentError(f"{control_name} caption source is stale")
                text_artifacts[artifact_id] = source_caption
        text_vectors, text_cache_stats = embed_with_cache(
            artifacts=sorted(text_artifacts.items()),
            modality="text",
            service=text_service,
            cache=cache,
            batch_size=args.text_batch_size,
        )
        image_vectors, image_cache_stats = embed_with_cache(
            artifacts=sorted(image_artifacts.items()),
            modality="image",
            service=image_service,
            cache=cache,
            batch_size=args.image_batch_size,
        )

        text_guides = _matrix(
            [
                text_vectors[f"guide:{guide_id}:caption:0"]
                for guide_id in range(DEFAULT_GUIDE_COUNT)
            ],
            "text guide",
        )
        image_guides = _matrix(
            [image_vectors[f"guide:{guide_id}:image"] for guide_id in range(DEFAULT_GUIDE_COUNT)],
            "image guide",
        )
        text_manager = GuideStarManager(text_guides)
        image_manager = GuideStarManager(image_guides)
        # Only same-model query-to-guide computations occur here.
        image_scores_by_record = {
            row["record_id"]: image_manager.compute_signature(
                np.asarray(image_vectors[f"evaluation:{row['record_id']}:image"], dtype=np.float32),
                apply_tier_weights=False,
            )
            for row in evaluations
        }
        text_scores_by_artifact = {
            artifact_id: text_manager.compute_signature(
                np.asarray(vector, dtype=np.float32), apply_tier_weights=False
            )
            for artifact_id, vector in text_vectors.items()
            if artifact_id.startswith("evaluation:")
        }

        full_ids = list(range(DEFAULT_GUIDE_COUNT))
        subset_count = min(args.random_guide_subset, DEFAULT_GUIDE_COUNT)
        subset_ids = sorted(
            _stable_order(full_ids, args.seed, "random-guide-subset")[:subset_count]
        )
        banks = {"full": full_ids}
        if subset_count < DEFAULT_GUIDE_COUNT:
            banks[f"random_{subset_count}"] = subset_ids
        mappings: dict[tuple[str, str], dict[int, int]] = {}
        for bank_name, guide_ids in banks.items():
            mappings[(bank_name, "paired")] = {guide_id: guide_id for guide_id in guide_ids}
            mappings[(bank_name, "shuffled")] = _derangement(
                guide_ids, args.seed, f"guide-identity:{bank_name}"
            )
        top_ks = sorted(set(args.top_k))
        if top_ks[-1] > min(len(values) for values in banks.values()):
            raise ExperimentError("top-k exceeds the smallest guide bank")

        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        per_pair_path = output_dir / "per_pair_results.jsonl"
        temporary = per_pair_path.with_suffix(".jsonl.tmp")
        aggregate_rows: list[dict[str, Any]] = []
        with temporary.open("w", encoding="utf-8") as handle:
            for evaluation in evaluations:
                record_id = evaluation["record_id"]
                pair_specs = [
                    (
                        "positive",
                        f"evaluation:{record_id}:caption:0",
                        record_id,
                        0,
                        True,
                    ),
                    (
                        "shuffled_negative",
                        "evaluation:{record_id}:caption:{caption_index}".format(
                            **evaluation["shuffled_negative"]
                        ),
                        evaluation["shuffled_negative"]["record_id"],
                        evaluation["shuffled_negative"]["caption_index"],
                        False,
                    ),
                    (
                        "hard_negative",
                        "evaluation:{record_id}:caption:{caption_index}".format(
                            **evaluation["hard_negative"]
                        ),
                        evaluation["hard_negative"]["record_id"],
                        evaluation["hard_negative"]["caption_index"],
                        False,
                    ),
                ]
                for (
                    pair_kind,
                    caption_artifact_id,
                    caption_record_id,
                    caption_index,
                    label,
                ) in pair_specs:
                    text_scores = text_scores_by_artifact[caption_artifact_id]
                    image_scores = image_scores_by_record[record_id]
                    for bank_name, guide_ids in banks.items():
                        for pairing in ("paired", "shuffled"):
                            result = _condition_result(
                                text_scores=text_scores,
                                image_scores=image_scores,
                                guide_ids=guide_ids,
                                identity_to_image_source=mappings[(bank_name, pairing)],
                                manager=text_manager,
                                top_ks=top_ks,
                            )
                            row = {
                                "schema_version": SCHEMA_VERSION,
                                "evaluation_record_id": record_id,
                                "image_record_id": record_id,
                                "caption_record_id": caption_record_id,
                                "caption_index": caption_index,
                                "pair_kind": pair_kind,
                                "original_association_positive_label": label,
                                "guide_bank": bank_name,
                                "guide_bank_size": len(guide_ids),
                                "guide_pairing": pairing,
                                **result,
                            }
                            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                            aggregate_rows.append(row)
        os.replace(temporary, per_pair_path)
        metrics = aggregate_results(
            aggregate_rows,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        summary_path = output_dir / "summary.json"
        report_path = output_dir / "report.md"
        summary = {
            "schema_version": SCHEMA_VERSION,
            "completed_utc": _utc_now(),
            "dataset": manifest["dataset"],
            "configuration": {
                "guide_count": len(guides),
                "evaluation_count": len(evaluations),
                "seed": args.seed,
                "top_k": top_ks,
                "random_guide_subset": subset_count,
                "bootstrap_samples": args.bootstrap_samples,
                "candidate_recall_definition": (
                    "either modality's rank-1 guide identity occurs in the other "
                    "modality's top-k guide neighborhood"
                ),
                "mrpc_reuse": {
                    "normalization_and_similarity": (
                        "GuideStarManager.compute_signature(apply_tier_weights=False)"
                    ),
                    "top_k_construction": "GuideStarManager.get_top_k_stars",
                    "production_candidate_limit": PSA_SIGNATURE_CANDIDATE_LIMIT,
                    "production_rerank_limit": PSA_SIGNATURE_RERANK_LIMIT,
                },
                "native_vectors_compared_across_models": False,
                "graph_membership_written": False,
            },
            "preflight": preflight,
            "cache": {
                "path": str(cache.path),
                "row_count": cache.row_count,
                "response_count": cache.response_count,
                "text": text_cache_stats,
                "image": image_cache_stats,
            },
            "guide_identity_mappings": {
                f"{bank}|{pairing}": {str(key): value for key, value in mapping.items()}
                for (bank, pairing), mapping in mappings.items()
            },
            "metrics": metrics,
            "outputs": {
                "manifest": str(manifest_path),
                "per_pair_jsonl": str(per_pair_path),
                "summary_json": str(summary_path),
                "report_markdown": str(report_path),
            },
        }
        _atomic_json(summary_path, summary)
        report_path.write_text(render_report(summary), encoding="utf-8")
        return report_path
    finally:
        cache.close()


def _add_text_hosted_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--text-model", default=QWEN3_EMBEDDING_MODEL_ID)
    parser.add_argument("--text-model-revision", default=DEFAULT_TEXT_MODEL_REVISION)
    parser.add_argument(
        "--text-provider", choices=("hf-inference", "hf-endpoint"), default="hf-inference"
    )
    parser.add_argument("--text-endpoint-url")
    parser.add_argument(
        "--text-api-style",
        choices=("hf-feature-extraction", "openai-embeddings"),
        default="hf-feature-extraction",
    )
    parser.add_argument("--timeout", type=float, default=120.0)


def _add_hosted_options(parser: argparse.ArgumentParser) -> None:
    _add_text_hosted_options(parser)
    parser.add_argument("--multimodal-model", default=DEFAULT_MULTIMODAL_MODEL_ID)
    parser.add_argument("--multimodal-model-revision", default=DEFAULT_MULTIMODAL_MODEL_REVISION)
    parser.add_argument(
        "--multimodal-provider",
        choices=("hf-inference", "hf-endpoint"),
        default="hf-endpoint",
    )
    parser.add_argument("--multimodal-endpoint-url")
    parser.add_argument("--multimodal-endpoint-env", default="HF_QWEN3_VL_EMBEDDING_ENDPOINT")
    parser.add_argument(
        "--multimodal-api-style",
        choices=("hf-multimodal", "openai-embeddings"),
        default="hf-multimodal",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="materialize the deterministic pilot data")
    prepare.add_argument("--dataset", default=DEFAULT_DATASET)
    prepare.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    prepare.add_argument("--dataset-config", default="default")
    prepare.add_argument("--dataset-server", default=DEFAULT_DATASET_SERVER)
    prepare.add_argument("--guide-split", default="train")
    prepare.add_argument("--evaluation-split", default="test")
    prepare.add_argument("--guide-count", type=int, default=DEFAULT_GUIDE_COUNT)
    prepare.add_argument("--evaluation-count", type=int, default=DEFAULT_EVALUATION_COUNT)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--hf-token-env", default="HF_TOKEN")
    prepare.add_argument("--timeout", type=float, default=120.0)
    prepare.add_argument("--output-dir", default=str(DEFAULT_WORK_DIR / "data"))

    preflight = commands.add_parser("preflight", help="validate both hosted model responses")
    _add_hosted_options(preflight)
    preflight.add_argument("--output", default=str(DEFAULT_WORK_DIR / "preflight.json"))

    text_preflight = commands.add_parser(
        "preflight-text", help="validate only the hosted text response"
    )
    _add_text_hosted_options(text_preflight)
    text_preflight.add_argument("--output", default=str(DEFAULT_WORK_DIR / "preflight_text.json"))

    run = commands.add_parser("run", help="run the cached experiment and produce a report")
    _add_hosted_options(run)
    run.add_argument("--manifest", default=str(DEFAULT_WORK_DIR / "data" / "manifest.json"))
    run.add_argument("--cache", default=str(DEFAULT_WORK_DIR / "embeddings.sqlite3"))
    run.add_argument("--output-dir", default=str(DEFAULT_WORK_DIR / "run"))
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--top-k", nargs="+", type=int, default=list(DEFAULT_TOP_K))
    run.add_argument("--random-guide-subset", type=int, default=DEFAULT_RANDOM_SUBSET_COUNT)
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    run.add_argument("--text-batch-size", type=int, default=16)
    run.add_argument("--image-batch-size", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare_manifest(args)
            print(f"Wrote deterministic dataset manifest to {path}")
            return 0
        if args.command == "preflight-text":
            result = preflight_text_service(HostedEmbeddingService(_text_service_config(args)))
            output = Path(args.output).resolve()
            _atomic_json(output, result)
            print(f"Hosted text preflight passed: dimension={result['dimension']} output={output}")
            return 0
        text_config, image_config = _service_configs(args)
        if args.command == "preflight":
            result = preflight_services(
                HostedEmbeddingService(text_config), HostedEmbeddingService(image_config)
            )
            output = Path(args.output).resolve()
            _atomic_json(output, result)
            print(
                "Hosted preflight passed: "
                f"text_dim={result['text']['dimension']} "
                f"image_dim={result['image']['dimension']} output={output}"
            )
            return 0
        report = run_experiment(args)
        print(f"Experiment completed; report written to {report}")
        return 0
    except ExperimentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
