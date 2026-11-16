#!/usr/bin/env python3
"""Build an authenticated, label-free directional NLI inference cache.

The input is a canonical manifest of premise/hypothesis requests.  This script
does not load MRPC labels, choose a threshold, import application services, or
make a paraphrase decision.  It only authenticates one pinned local ONNX model,
executes ordered batched inference, and writes raw logits plus stable softmax
probabilities in a hash-bound envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

REQUEST_MANIFEST_SCHEMA_VERSION = "mrpc_directional_nli_request_manifest_v1"
INFERENCE_CACHE_SCHEMA_VERSION = "mrpc_directional_nli_inference_cache_v1"

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
MODEL_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
ONNX_RELATIVE_PATH = "onnx/model_qint8_arm64.onnx"
LABEL_ORDER = ("contradiction", "entailment", "neutral")
MODEL_MAX_LENGTH = 512

EXPECTED_MODEL_FILE_SHA256: Mapping[str, str] = {
    "added_tokens.json": ("a4b6bfe668f2b3cf6f0cd535e98a0663d2d0d4a4a15f13075ad3597d33985a23"),
    "config.json": ("8d9f07bf7ba54a6fc3b1962483056f94c39dcf188db4cf61843e1c88f94b2342"),
    ONNX_RELATIVE_PATH: ("ef41fe8474f6070373f97fbe8362461d55108f387723280fcdb64f8015133db2"),
    "special_tokens_map.json": ("ed7c099c988dbb414b18a6980d20cb57b91b7cd119f6f6941eb364b0e892e712"),
    "spm.model": ("c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd"),
    "tokenizer.json": ("5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964"),
    "tokenizer_config.json": ("f3eecd07c370ef0bf7dd3780d3cd68cf9c8b00c267e21a208ddcd8f82bfec1a6"),
}

EXPECTED_RUNTIME_VERSIONS: Mapping[str, str] = {
    "python": "3.13.7",
    "numpy": "2.5.1",
    "onnxruntime": "1.28.0",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "huggingface-hub": "0.36.2",
    "sentencepiece": "0.2.1",
}

DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "temp" / "mrpc_directional_nli" / "request_manifest.json"
DEFAULT_MODEL_DIR = (
    REPOSITORY_ROOT / "temp" / "models" / "cross-encoder-nli-deberta-v3-xsmall-a150876"
)
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "temp" / "mrpc_directional_nli" / "inference_cache.json"


class DirectionalNLIInferenceCacheError(ValueError):
    """Raised when an inference-cache input or identity is not authenticated."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON encoding used by all cache digests."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_mapping(
    value: Any,
    *,
    expected_keys: frozenset[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise DirectionalNLIInferenceCacheError(
            f"{context} must contain exactly {sorted(expected_keys)!r}"
        )
    return value


def _manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "model": manifest["model"],
        "requests": manifest["requests"],
    }


def load_and_validate_request_manifest(path: Path) -> dict[str, Any]:
    """Load a request manifest and authenticate its canonical content digest."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionalNLIInferenceCacheError(
            f"cannot load directional NLI request manifest: {path}"
        ) from exc
    top = _exact_mapping(
        manifest,
        expected_keys=frozenset({"schema_version", "model", "requests", "manifest_sha256"}),
        context="directional NLI request manifest",
    )
    if top["schema_version"] != REQUEST_MANIFEST_SCHEMA_VERSION:
        raise DirectionalNLIInferenceCacheError(
            "unsupported directional NLI request manifest schema"
        )
    model = _exact_mapping(
        top["model"],
        expected_keys=frozenset({"id", "revision", "onnx_relative_path"}),
        context="directional NLI manifest model identity",
    )
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "onnx_relative_path": ONNX_RELATIVE_PATH,
    }
    if dict(model) != expected_model:
        raise DirectionalNLIInferenceCacheError(
            "directional NLI manifest model identity does not match the pinned model"
        )

    requests = top["requests"]
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence) or not requests:
        raise DirectionalNLIInferenceCacheError(
            "directional NLI request manifest must contain at least one request"
        )
    request_ids: set[str] = set()
    for index, raw_request in enumerate(requests):
        request = _exact_mapping(
            raw_request,
            expected_keys=frozenset({"request_id", "premise", "hypothesis"}),
            context=f"directional NLI request {index}",
        )
        request_id = request["request_id"]
        premise = request["premise"]
        hypothesis = request["hypothesis"]
        if not isinstance(request_id, str) or not request_id or request_id != request_id.strip():
            raise DirectionalNLIInferenceCacheError(
                f"directional NLI request {index} has an invalid request_id"
            )
        if request_id in request_ids:
            raise DirectionalNLIInferenceCacheError(
                f"duplicate directional NLI request_id: {request_id!r}"
            )
        request_ids.add(request_id)
        for name, text in (("premise", premise), ("hypothesis", hypothesis)):
            if not isinstance(text, str) or not text.strip():
                raise DirectionalNLIInferenceCacheError(
                    f"directional NLI request {request_id!r} has an invalid {name}"
                )

    manifest_sha256 = top["manifest_sha256"]
    actual_sha256 = canonical_sha256(_manifest_payload(top))
    if not _is_sha256(manifest_sha256) or manifest_sha256 != actual_sha256:
        raise DirectionalNLIInferenceCacheError("directional NLI request manifest SHA-256 mismatch")
    return json.loads(canonical_json(top))


def authenticate_model_snapshot(model_dir: Path) -> dict[str, Any]:
    """Authenticate every required model and tokenizer file plus label order."""

    resolved = model_dir.resolve()
    if not resolved.is_dir():
        raise DirectionalNLIInferenceCacheError(
            f"pinned directional NLI model directory is missing: {resolved}"
        )
    observed_hashes: dict[str, str] = {}
    for relative_path, expected_sha256 in EXPECTED_MODEL_FILE_SHA256.items():
        path = resolved / relative_path
        if not path.is_file():
            raise DirectionalNLIInferenceCacheError(
                f"pinned directional NLI model file is missing: {relative_path}"
            )
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise DirectionalNLIInferenceCacheError(
                f"pinned directional NLI model file SHA-256 mismatch: {relative_path}"
            )
        observed_hashes[relative_path] = actual_sha256

    try:
        config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionalNLIInferenceCacheError(
            "pinned directional NLI config.json is malformed"
        ) from exc
    if config.get("architectures") != ["DebertaV2ForSequenceClassification"]:
        raise DirectionalNLIInferenceCacheError(
            "pinned directional NLI model architecture mismatch"
        )
    expected_id2label = {str(index): label for index, label in enumerate(LABEL_ORDER)}
    expected_label2id = {label: index for index, label in enumerate(LABEL_ORDER)}
    if config.get("id2label") != expected_id2label:
        raise DirectionalNLIInferenceCacheError("pinned directional NLI id2label mapping mismatch")
    if config.get("label2id") != expected_label2id:
        raise DirectionalNLIInferenceCacheError("pinned directional NLI label2id mapping mismatch")
    if config.get("max_position_embeddings") != MODEL_MAX_LENGTH:
        raise DirectionalNLIInferenceCacheError(
            "pinned directional NLI maximum input length mismatch"
        )
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "onnx_relative_path": ONNX_RELATIVE_PATH,
        "files": dict(sorted(observed_hashes.items())),
    }


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise DirectionalNLIInferenceCacheError(
            f"required directional NLI runtime package is missing: {distribution}"
        ) from exc


def authenticate_runtime() -> dict[str, Any]:
    """Require the exact tested Python/package versions and ARM64 platform."""

    observed: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": _distribution_version("numpy"),
        "onnxruntime": _distribution_version("onnxruntime"),
        "transformers": _distribution_version("transformers"),
        "tokenizers": _distribution_version("tokenizers"),
        "huggingface-hub": _distribution_version("huggingface-hub"),
        "sentencepiece": _distribution_version("sentencepiece"),
    }
    if observed != dict(EXPECTED_RUNTIME_VERSIONS):
        raise DirectionalNLIInferenceCacheError(
            "directional NLI runtime version mismatch: "
            f"expected {dict(EXPECTED_RUNTIME_VERSIONS)!r}, got {observed!r}"
        )
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine != "arm64":
        raise DirectionalNLIInferenceCacheError(
            "the pinned quantized directional NLI artifact requires Darwin ARM64"
        )
    return {**observed, "system": system, "machine": machine}


def stable_softmax(logits: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """Compute deterministic float64 softmax for the three pinned NLI labels."""

    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(LABEL_ORDER),) or not np.all(np.isfinite(values)):
        raise DirectionalNLIInferenceCacheError(
            "directional NLI logits must be a finite three-value vector"
        )
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise DirectionalNLIInferenceCacheError("directional NLI softmax denominator is invalid")
    probabilities = exponentials / denominator
    return tuple(float(value) for value in probabilities)  # type: ignore[return-value]


def _validated_batch_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DirectionalNLIInferenceCacheError("batch_size must be a positive integer")
    return value


def run_batched_inference(
    requests: Sequence[Mapping[str, str]],
    *,
    model_dir: Path,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the pinned ONNX model locally while preserving exact request order."""

    batch_size = _validated_batch_size(batch_size)
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised by runtime integration
        raise DirectionalNLIInferenceCacheError(
            "directional NLI inference dependencies are unavailable"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir.resolve()),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.model_max_length != MODEL_MAX_LENGTH:
        raise DirectionalNLIInferenceCacheError("directional NLI tokenizer maximum length mismatch")

    session = ort.InferenceSession(
        str((model_dir.resolve() / ONNX_RELATIVE_PATH)),
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise DirectionalNLIInferenceCacheError("directional NLI ONNX provider identity mismatch")
    input_types = {item.name: item.type for item in session.get_inputs()}
    if input_types != {
        "input_ids": "tensor(int64)",
        "attention_mask": "tensor(int64)",
    }:
        raise DirectionalNLIInferenceCacheError("directional NLI ONNX input contract mismatch")
    outputs = session.get_outputs()
    if len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].type != "tensor(float)":
        raise DirectionalNLIInferenceCacheError("directional NLI ONNX output contract mismatch")

    results: list[dict[str, Any]] = []
    maximum_sequence_length = 0
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        encoded = tokenizer(
            [request["premise"] for request in batch],
            [request["hypothesis"] for request in batch],
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise DirectionalNLIInferenceCacheError(
                "directional NLI tokenized input shape mismatch"
            )
        sequence_length = int(input_ids.shape[1])
        maximum_sequence_length = max(maximum_sequence_length, sequence_length)
        if sequence_length > MODEL_MAX_LENGTH:
            raise DirectionalNLIInferenceCacheError(
                "directional NLI input exceeds the pinned 512-token limit; "
                "implicit truncation is forbidden"
            )
        raw_outputs = session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        if len(raw_outputs) != 1:
            raise DirectionalNLIInferenceCacheError(
                "directional NLI ONNX session returned an unexpected output count"
            )
        logits = np.asarray(raw_outputs[0], dtype=np.float64)
        if logits.shape != (len(batch), len(LABEL_ORDER)) or not np.all(np.isfinite(logits)):
            raise DirectionalNLIInferenceCacheError(
                "directional NLI ONNX logits shape or values are invalid"
            )
        for request, row in zip(batch, logits, strict=True):
            probabilities = stable_softmax(row)
            results.append(
                {
                    "request_id": request["request_id"],
                    "logits": [float(value) for value in row],
                    "probabilities": {
                        label: probabilities[index] for index, label in enumerate(LABEL_ORDER)
                    },
                }
            )
    return results, {
        "batch_size": batch_size,
        "provider": "CPUExecutionProvider",
        "label_order": list(LABEL_ORDER),
        "result_count": len(results),
        "maximum_sequence_length": maximum_sequence_length,
        "truncation": False,
    }


def _validate_cache_envelope(
    envelope: Any,
    *,
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    top = _exact_mapping(
        envelope,
        expected_keys=frozenset({"schema_version", "payload", "cache_sha256"}),
        context="directional NLI inference cache envelope",
    )
    if top["schema_version"] != INFERENCE_CACHE_SCHEMA_VERSION:
        raise DirectionalNLIInferenceCacheError(
            "unsupported directional NLI inference cache schema"
        )
    payload = _exact_mapping(
        top["payload"],
        expected_keys=frozenset({"request_manifest", "model", "runtime", "inference", "results"}),
        context="directional NLI inference cache payload",
    )
    if top["cache_sha256"] != canonical_sha256(payload):
        raise DirectionalNLIInferenceCacheError("directional NLI inference cache SHA-256 mismatch")
    request_manifest = _exact_mapping(
        payload["request_manifest"],
        expected_keys=frozenset({"schema_version", "manifest_sha256", "request_count"}),
        context="directional NLI cache request-manifest identity",
    )
    model = _exact_mapping(
        payload["model"],
        expected_keys=frozenset({"id", "revision", "onnx_relative_path", "files"}),
        context="directional NLI cache model identity",
    )
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "onnx_relative_path": ONNX_RELATIVE_PATH,
        "files": dict(sorted(EXPECTED_MODEL_FILE_SHA256.items())),
    }
    if dict(model) != expected_model:
        raise DirectionalNLIInferenceCacheError("directional NLI cache model identity mismatch")
    runtime = _exact_mapping(
        payload["runtime"],
        expected_keys=frozenset({*EXPECTED_RUNTIME_VERSIONS, "system", "machine"}),
        context="directional NLI cache runtime identity",
    )
    expected_runtime = {
        **EXPECTED_RUNTIME_VERSIONS,
        "system": "Darwin",
        "machine": "arm64",
    }
    if dict(runtime) != expected_runtime:
        raise DirectionalNLIInferenceCacheError("directional NLI cache runtime identity mismatch")
    inference = _exact_mapping(
        payload["inference"],
        expected_keys=frozenset(
            {
                "batch_size",
                "provider",
                "label_order",
                "result_count",
                "maximum_sequence_length",
                "truncation",
            }
        ),
        context="directional NLI cache inference identity",
    )
    if request_manifest["schema_version"] != REQUEST_MANIFEST_SCHEMA_VERSION:
        raise DirectionalNLIInferenceCacheError("directional NLI cache manifest schema mismatch")
    if not _is_sha256(request_manifest["manifest_sha256"]):
        raise DirectionalNLIInferenceCacheError("directional NLI cache manifest SHA-256 is invalid")
    if inference["provider"] != "CPUExecutionProvider":
        raise DirectionalNLIInferenceCacheError("directional NLI cache provider mismatch")
    if inference["label_order"] != list(LABEL_ORDER):
        raise DirectionalNLIInferenceCacheError("directional NLI cache label order mismatch")
    _validated_batch_size(inference["batch_size"])
    maximum_sequence_length = inference["maximum_sequence_length"]
    if (
        isinstance(maximum_sequence_length, bool)
        or not isinstance(maximum_sequence_length, int)
        or not 0 < maximum_sequence_length <= MODEL_MAX_LENGTH
        or inference["truncation"] is not False
    ):
        raise DirectionalNLIInferenceCacheError(
            "directional NLI cache tokenization identity is invalid"
        )

    results = payload["results"]
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise DirectionalNLIInferenceCacheError("directional NLI cache results must be a sequence")
    request_count = request_manifest["request_count"]
    result_count = inference["result_count"]
    if (
        isinstance(request_count, bool)
        or not isinstance(request_count, int)
        or request_count <= 0
        or isinstance(result_count, bool)
        or not isinstance(result_count, int)
        or result_count <= 0
        or request_count != len(results)
        or result_count != len(results)
    ):
        raise DirectionalNLIInferenceCacheError(
            "directional NLI cache request/result count mismatch"
        )
    result_ids: list[str] = []
    for index, raw_result in enumerate(results):
        result = _exact_mapping(
            raw_result,
            expected_keys=frozenset({"request_id", "logits", "probabilities"}),
            context=f"directional NLI cache result {index}",
        )
        request_id = result["request_id"]
        if not isinstance(request_id, str) or not request_id or request_id != request_id.strip():
            raise DirectionalNLIInferenceCacheError(
                f"directional NLI cache result {index} has an invalid request_id"
            )
        result_ids.append(request_id)
        logits = result["logits"]
        if isinstance(logits, (str, bytes)) or not isinstance(logits, Sequence):
            raise DirectionalNLIInferenceCacheError(
                f"directional NLI cache result {request_id!r} has invalid logits"
            )
        expected_probabilities = stable_softmax(logits)
        probabilities = _exact_mapping(
            result["probabilities"],
            expected_keys=frozenset(LABEL_ORDER),
            context=f"directional NLI cache result {request_id!r} probabilities",
        )
        observed_probabilities = tuple(probabilities[label] for label in LABEL_ORDER)
        if observed_probabilities != expected_probabilities:
            raise DirectionalNLIInferenceCacheError(
                f"directional NLI cache result {request_id!r} softmax mismatch"
            )
    if len(set(result_ids)) != len(result_ids):
        raise DirectionalNLIInferenceCacheError(
            "directional NLI cache contains duplicate request IDs"
        )

    if manifest is not None:
        expected_ids = [request["request_id"] for request in manifest["requests"]]
        if request_manifest != {
            "schema_version": manifest["schema_version"],
            "manifest_sha256": manifest["manifest_sha256"],
            "request_count": len(expected_ids),
        }:
            raise DirectionalNLIInferenceCacheError(
                "directional NLI cache does not match the request manifest identity"
            )
        if result_ids != expected_ids:
            raise DirectionalNLIInferenceCacheError(
                "directional NLI cache request coverage or order mismatch"
            )
    return json.loads(canonical_json(top))


def load_and_validate_inference_cache(
    path: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and authenticate a cache envelope, optionally against its manifest."""

    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectionalNLIInferenceCacheError(
            f"cannot load directional NLI inference cache: {path}"
        ) from exc
    return _validate_cache_envelope(envelope, manifest=manifest)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_inference_cache(
    *,
    manifest_path: Path,
    model_dir: Path,
    output_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Authenticate inputs, run inference, persist, and replay the cache."""

    manifest = load_and_validate_request_manifest(manifest_path)
    model_identity = authenticate_model_snapshot(model_dir)
    runtime_identity = authenticate_runtime()
    results, inference_identity = run_batched_inference(
        manifest["requests"],
        model_dir=model_dir,
        batch_size=batch_size,
    )
    payload = {
        "request_manifest": {
            "schema_version": manifest["schema_version"],
            "manifest_sha256": manifest["manifest_sha256"],
            "request_count": len(manifest["requests"]),
        },
        "model": model_identity,
        "runtime": runtime_identity,
        "inference": inference_identity,
        "results": results,
    }
    envelope = {
        "schema_version": INFERENCE_CACHE_SCHEMA_VERSION,
        "payload": payload,
        "cache_sha256": canonical_sha256(payload),
    }
    validated = _validate_cache_envelope(envelope, manifest=manifest)
    _atomic_json(output_path, validated)
    replayed = load_and_validate_inference_cache(output_path, manifest=manifest)
    if replayed != validated:
        raise DirectionalNLIInferenceCacheError(
            "persisted directional NLI inference cache did not replay exactly"
        )
    return replayed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    envelope = build_inference_cache(
        manifest_path=args.manifest.resolve(),
        model_dir=args.model_dir.resolve(),
        output_path=args.output.resolve(),
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "schema_version": envelope["schema_version"],
                "cache_sha256": envelope["cache_sha256"],
                "manifest_sha256": envelope["payload"]["request_manifest"]["manifest_sha256"],
                "request_count": envelope["payload"]["request_manifest"]["request_count"],
                "model_revision": envelope["payload"]["model"]["revision"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
