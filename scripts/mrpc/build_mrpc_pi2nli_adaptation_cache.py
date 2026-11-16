#!/usr/bin/env python3
"""Train and run one calibration-only, diagnostic PI2NLI adaptation.

The three phases are intentionally separate. ``crossfit`` trains only on the
component-fold training IDs supplied by an authenticated calibration manifest.
``fit-final`` trains on that complete calibration manifest after an epoch has
been frozen. ``infer-held`` accepts a separate, label-free held manifest only
after the final checkpoint exists. The resulting scores have no service,
database, membership, atomic-support, or edge-writing authority.

PyTorch and Transformers are imported lazily so the manifest/cache contracts
can be tested in the repository's ordinary (non-training) Python runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

CALIBRATION_MANIFEST_SCHEMA_VERSION = "mrpc_pi2nli_training_manifest_v1"
HELD_MANIFEST_SCHEMA_VERSION = "mrpc_pi2nli_held_request_manifest_v1"
SELECTION_SCHEMA_VERSION = "mrpc_pi2nli_selection_v1"
CROSSFIT_CACHE_SCHEMA_VERSION = "mrpc_pi2nli_crossfit_cache_v1"
FINAL_FIT_CACHE_SCHEMA_VERSION = "mrpc_pi2nli_final_fit_cache_v1"
HELD_INFERENCE_CACHE_SCHEMA_VERSION = "mrpc_pi2nli_held_inference_cache_v1"
CHECKPOINT_SCHEMA_VERSION = "mrpc_pi2nli_checkpoint_v1"
ADAPTATION_MAPPING_SCHEMA_VERSION = "mrpc_pi2nli_deterministic_mapping_v1"
EXPERIMENT_ID = "mrpc_pi2nli_preserved_head_fusion_v1"

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
MODEL_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
MODEL_WEIGHTS_RELATIVE_PATH = "model.safetensors"
MODEL_WEIGHTS_SHA256 = "4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9"
MODEL_MAX_POSITION_EMBEDDINGS = 512
LABEL_ORDER = ("contradiction", "entailment", "neutral")
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}

EXPECTED_BASE_MODEL_FILES: Mapping[str, str] = {
    "README.md": "4aeb5585a9eeb71b2d572249f9faa22ef3378c6abee3eeaffaae8c961068a2c6",
    "added_tokens.json": "a4b6bfe668f2b3cf6f0cd535e98a0663d2d0d4a4a15f13075ad3597d33985a23",
    "config.json": "8d9f07bf7ba54a6fc3b1962483056f94c39dcf188db4cf61843e1c88f94b2342",
    MODEL_WEIGHTS_RELATIVE_PATH: MODEL_WEIGHTS_SHA256,
    "special_tokens_map.json": ("ed7c099c988dbb414b18a6980d20cb57b91b7cd119f6f6941eb364b0e892e712"),
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "tokenizer.json": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    "tokenizer_config.json": ("f3eecd07c370ef0bf7dd3780d3cd68cf9c8b00c267e21a208ddcd8f82bfec1a6"),
}

BASE_MODEL_MANIFEST_IDENTITY: Mapping[str, Any] = {
    "id": MODEL_ID,
    "revision": MODEL_REVISION,
    "weights_relative_path": MODEL_WEIGHTS_RELATIVE_PATH,
    "weights_sha256": MODEL_WEIGHTS_SHA256,
}

INITIAL_HEAD_TENSOR_SHA256: Mapping[str, str] = {
    "classifier.bias": "3a65b35f8dea24a31fb3b2e2f8e3e6e785198e64bd7c9c7206126d11e472eacf",
    "classifier.weight": "4099d554b3f0555f802d818086d1eaa70b914347a69feb67e14b173fce84a078",
}

PI2NLI_MAPPING_SEED = "mrpc-pi2nli-deterministic-mapping-v1"
OPTIMIZATION_SEED = 0
TRAINING_EPOCHS = tuple(range(1, 13))
TRAINING_SPEC: Mapping[str, Any] = {
    "protocol": "pi2nli_deterministic_three_way_v1",
    "mapping_seed": PI2NLI_MAPPING_SEED,
    "optimization_seed": OPTIMIZATION_SEED,
    "learning_rate": 1e-5,
    "weight_decay": 0.001,
    "adam_epsilon": 1e-8,
    "train_batch_size": 32,
    "eval_batch_size": 64,
    "max_length": 156,
    "epochs": list(TRAINING_EPOCHS),
    "head_initialization": "preserve_pretrained_three_way_nli_head",
    "optimizer": "torch.optim.AdamW",
    "scheduler": "none",
    "gradient_accumulation_steps": 1,
    "mixed_precision": False,
}

AUTHORITY_CONTRACT: Mapping[str, Any] = {
    "diagnostic_only": True,
    "hard_membership_authorized": False,
    "membership_effect": "none",
    "directional_relation_authorized": False,
    "atomic_support_authorized": False,
    "directional_edge_authorized": False,
    "edge_write_authorized": False,
    "service_consumers": [],
    "database_consumers": [],
}

EXPECTED_RUNTIME_VERSIONS: Mapping[str, str] = {
    "python": "3.13.7",
    "numpy": "2.5.1",
    "torch": "2.8.0",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "safetensors": "0.6.2",
    "huggingface-hub": "0.36.2",
    "sentencepiece": "0.2.1",
}

DEFAULT_WORK_DIR = REPOSITORY_ROOT / "temp" / "mrpc_pi2nli_adaptation_v1"
DEFAULT_BASE_MODEL_DIR = (
    REPOSITORY_ROOT / "temp" / "models" / "cross-encoder-nli-deberta-v3-xsmall-a150876"
)


class PI2NLIAdaptationCacheError(ValueError):
    """Raised when an adaptation input, runtime, or artifact is not authenticated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_mapping(
    value: Any,
    *,
    keys: frozenset[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PI2NLIAdaptationCacheError(f"{context} must contain exactly {sorted(keys)!r}")
    return value


def _validated_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PI2NLIAdaptationCacheError(f"{context} must be a non-empty trimmed string")
    return value


def _validated_positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PI2NLIAdaptationCacheError(f"{context} must be a positive integer")
    return value


def _validate_model_manifest(value: Any) -> Mapping[str, Any]:
    model = _exact_mapping(
        value,
        keys=frozenset(BASE_MODEL_MANIFEST_IDENTITY),
        context="PI2NLI base-model identity",
    )
    if dict(model) != dict(BASE_MODEL_MANIFEST_IDENTITY):
        raise PI2NLIAdaptationCacheError("PI2NLI base-model identity is not pinned")
    return model


def _validate_authority(value: Any) -> Mapping[str, Any]:
    authority = _exact_mapping(
        value,
        keys=frozenset(AUTHORITY_CONTRACT),
        context="PI2NLI authority contract",
    )
    if dict(authority) != dict(AUTHORITY_CONTRACT):
        raise PI2NLIAdaptationCacheError("PI2NLI authority contract mismatch")
    return authority


def _validate_training_spec(value: Any) -> Mapping[str, Any]:
    training = _exact_mapping(
        value,
        keys=frozenset(TRAINING_SPEC),
        context="PI2NLI training specification",
    )
    if dict(training) != dict(TRAINING_SPEC):
        raise PI2NLIAdaptationCacheError("PI2NLI training specification is not frozen")
    return training


def _validate_calibration_pair(value: Any, *, index: int) -> Mapping[str, Any]:
    pair = _exact_mapping(
        value,
        keys=frozenset({"pair_id", "sentence1", "sentence2", "label", "group", "routed"}),
        context=f"PI2NLI calibration pair {index}",
    )
    _validated_positive_int(pair["pair_id"], context=f"calibration pair {index} pair_id")
    _validated_nonempty_string(pair["sentence1"], context=f"calibration pair {index} sentence1")
    _validated_nonempty_string(pair["sentence2"], context=f"calibration pair {index} sentence2")
    if not isinstance(pair["label"], bool):
        raise PI2NLIAdaptationCacheError(f"PI2NLI calibration pair {index} label must be Boolean")
    if isinstance(pair["group"], bool) or not isinstance(pair["group"], int):
        raise PI2NLIAdaptationCacheError(
            f"PI2NLI calibration pair {index} group must be an integer"
        )
    if not isinstance(pair["routed"], bool):
        raise PI2NLIAdaptationCacheError(f"PI2NLI calibration pair {index} routed must be Boolean")
    return pair


def _validate_held_pair(value: Any, *, index: int) -> Mapping[str, Any]:
    pair = _exact_mapping(
        value,
        keys=frozenset({"pair_id", "sentence1", "sentence2", "group", "routed"}),
        context=f"PI2NLI held pair {index}",
    )
    _validated_positive_int(pair["pair_id"], context=f"held pair {index} pair_id")
    _validated_nonempty_string(pair["sentence1"], context=f"held pair {index} sentence1")
    _validated_nonempty_string(pair["sentence2"], context=f"held pair {index} sentence2")
    if isinstance(pair["group"], bool) or not isinstance(pair["group"], int):
        raise PI2NLIAdaptationCacheError(f"PI2NLI held pair {index} group must be an integer")
    if not isinstance(pair["routed"], bool):
        raise PI2NLIAdaptationCacheError(f"PI2NLI held pair {index} routed must be Boolean")
    return pair


def _validate_pairs(
    values: Any,
    *,
    held: bool,
) -> list[Mapping[str, Any]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise PI2NLIAdaptationCacheError("PI2NLI manifest pairs must be a non-empty sequence")
    pairs: list[Mapping[str, Any]] = []
    for index, value in enumerate(values):
        pair = (
            _validate_held_pair(value, index=index)
            if held
            else _validate_calibration_pair(value, index=index)
        )
        pairs.append(pair)
    pair_ids = [int(pair["pair_id"]) for pair in pairs]
    if pair_ids != sorted(pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise PI2NLIAdaptationCacheError(
            "PI2NLI manifest pairs must have unique, ascending pair IDs"
        )
    return pairs


def _validate_component_folds(
    pairs: Sequence[Mapping[str, Any]],
    raw_folds: Any,
) -> list[Mapping[str, Any]]:
    if (
        isinstance(raw_folds, (str, bytes))
        or not isinstance(raw_folds, Sequence)
        or len(raw_folds) < 2
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI component folds must contain at least two folds")
    pair_ids = {int(pair["pair_id"]) for pair in pairs}
    group_by_pair = {int(pair["pair_id"]): int(pair["group"]) for pair in pairs}
    seen_validation: set[int] = set()
    folds: list[Mapping[str, Any]] = []
    for index, value in enumerate(raw_folds):
        fold = _exact_mapping(
            value,
            keys=frozenset({"fold", "training_pair_ids", "validation_pair_ids"}),
            context=f"PI2NLI component fold {index}",
        )
        if fold["fold"] != index:
            raise PI2NLIAdaptationCacheError("PI2NLI component folds must be numbered from zero")
        training_ids = fold["training_pair_ids"]
        validation_ids = fold["validation_pair_ids"]
        for name, values in (
            ("training_pair_ids", training_ids),
            ("validation_pair_ids", validation_ids),
        ):
            if (
                isinstance(values, (str, bytes))
                or not isinstance(values, Sequence)
                or not values
                or any(isinstance(item, bool) or not isinstance(item, int) for item in values)
                or list(values) != sorted(values)
                or len(set(values)) != len(values)
            ):
                raise PI2NLIAdaptationCacheError(
                    f"PI2NLI fold {index} {name} must be a non-empty sorted unique ID list"
                )
        training_set = set(training_ids)
        validation_set = set(validation_ids)
        if (
            training_set & validation_set
            or training_set | validation_set != pair_ids
            or training_set != pair_ids - validation_set
        ):
            raise PI2NLIAdaptationCacheError(
                f"PI2NLI fold {index} must partition all calibration pair IDs"
            )
        training_groups = {group_by_pair[pair_id] for pair_id in training_set}
        validation_groups = {group_by_pair[pair_id] for pair_id in validation_set}
        if training_groups & validation_groups:
            raise PI2NLIAdaptationCacheError(
                f"PI2NLI fold {index} crosses a sentence-component group"
            )
        if seen_validation & validation_set:
            raise PI2NLIAdaptationCacheError(
                "PI2NLI validation pair appears in more than one component fold"
            )
        seen_validation.update(validation_set)
        folds.append(fold)
    if seen_validation != pair_ids:
        raise PI2NLIAdaptationCacheError(
            "PI2NLI component-fold validation IDs do not cover calibration exactly once"
        )
    return folds


def _validate_checkpoint_manifest(value: Any) -> Mapping[str, Any]:
    checkpoint = _exact_mapping(
        value,
        keys=frozenset({"artifact_sha256", "selected_epoch"}),
        context="PI2NLI held checkpoint identity",
    )
    if not _is_sha256(checkpoint["artifact_sha256"]):
        raise PI2NLIAdaptationCacheError("PI2NLI held checkpoint SHA-256 is invalid")
    if checkpoint["selected_epoch"] not in TRAINING_EPOCHS:
        raise PI2NLIAdaptationCacheError("PI2NLI held checkpoint epoch is invalid")
    return checkpoint


def _validate_manifest(
    value: Any,
    *,
    expected_schema_version: str | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize a hash-bound calibration or held manifest."""

    if not isinstance(value, Mapping):
        raise PI2NLIAdaptationCacheError("PI2NLI manifest must be a mapping")
    schema_version = value.get("schema_version")
    if expected_schema_version is not None and schema_version != expected_schema_version:
        raise PI2NLIAdaptationCacheError("PI2NLI manifest schema does not match the phase")
    if schema_version == CALIBRATION_MANIFEST_SCHEMA_VERSION:
        manifest = _exact_mapping(
            value,
            keys=frozenset(
                {
                    "schema_version",
                    "manifest_sha256",
                    "experiment_id",
                    "phase",
                    "model",
                    "base_model_files",
                    "authority",
                    "training",
                    "split",
                    "dataset_identity",
                    "routing",
                    "preregistration",
                    "pairs",
                    "folds",
                }
            ),
            context="PI2NLI calibration manifest",
        )
        if manifest["experiment_id"] != EXPERIMENT_ID:
            raise PI2NLIAdaptationCacheError("PI2NLI training experiment identity mismatch")
        if manifest["phase"] != "calibration_training_manifest_before_adapted_model_training":
            raise PI2NLIAdaptationCacheError("PI2NLI training-manifest phase mismatch")
        _validate_model_manifest(manifest["model"])
        if manifest["base_model_files"] != dict(EXPECTED_BASE_MODEL_FILES):
            raise PI2NLIAdaptationCacheError("PI2NLI manifest base-model files are not pinned")
        _validate_authority(manifest["authority"])
        _validate_training_spec(manifest["training"])
        split = manifest["split"]
        if not isinstance(split, Mapping) or (
            split.get("held_out_labels_loaded") is not False
            or split.get("held_out_pairs_included") is not False
            or split.get("held_out_text_included") is not False
        ):
            raise PI2NLIAdaptationCacheError("PI2NLI training manifest is not calibration-only")
        for name in ("dataset_identity", "routing", "preregistration"):
            if not isinstance(manifest[name], Mapping) or not manifest[name]:
                raise PI2NLIAdaptationCacheError(
                    f"PI2NLI training manifest {name} identity is invalid"
                )
        pairs = _validate_pairs(manifest["pairs"], held=False)
        _validate_component_folds(pairs, manifest["folds"])
    elif schema_version == HELD_MANIFEST_SCHEMA_VERSION:
        manifest = _exact_mapping(
            value,
            keys=frozenset(
                {
                    "schema_version",
                    "manifest_sha256",
                    "experiment_id",
                    "phase",
                    "model",
                    "checkpoint",
                    "resolver_sha256",
                    "authority",
                    "routing",
                    "labels_included",
                    "pairs",
                }
            ),
            context="PI2NLI held manifest",
        )
        if manifest["experiment_id"] != EXPERIMENT_ID:
            raise PI2NLIAdaptationCacheError("PI2NLI held experiment identity mismatch")
        if manifest["phase"] != "label_free_held_requests_after_final_resolver_freeze":
            raise PI2NLIAdaptationCacheError("PI2NLI held-manifest phase mismatch")
        _validate_model_manifest(manifest["model"])
        _validate_checkpoint_manifest(manifest["checkpoint"])
        if not _is_sha256(manifest["resolver_sha256"]):
            raise PI2NLIAdaptationCacheError("PI2NLI held resolver SHA-256 is invalid")
        _validate_authority(manifest["authority"])
        if not isinstance(manifest["routing"], Mapping) or not manifest["routing"]:
            raise PI2NLIAdaptationCacheError("PI2NLI held routing identity is invalid")
        if manifest["labels_included"] is not False:
            raise PI2NLIAdaptationCacheError("PI2NLI held manifest contains labels")
        _validate_pairs(manifest["pairs"], held=True)
    else:
        raise PI2NLIAdaptationCacheError("unsupported PI2NLI manifest schema")

    manifest_sha256 = manifest["manifest_sha256"]
    payload = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    if not _is_sha256(manifest_sha256) or manifest_sha256 != _canonical_sha256(payload):
        raise PI2NLIAdaptationCacheError("PI2NLI manifest SHA-256 mismatch")
    return json.loads(_canonical_json(manifest))


def _load_validated_manifest(
    path: Path,
    *,
    expected_schema_version: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PI2NLIAdaptationCacheError(f"cannot load PI2NLI manifest: {path}") from exc
    return _validate_manifest(value, expected_schema_version=expected_schema_version)


def _mapping_choice(pair: Mapping[str, Any]) -> tuple[str, str]:
    payload = {
        "schema_version": ADAPTATION_MAPPING_SCHEMA_VERSION,
        "mapping_seed": PI2NLI_MAPPING_SEED,
        "pair_id": pair["pair_id"],
        "sentence1": pair["sentence1"],
        "sentence2": pair["sentence2"],
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()
    direction = "sentence1_to_sentence2" if digest[0] % 2 == 0 else "sentence2_to_sentence1"
    label = "contradiction" if digest[1] % 2 == 0 else "neutral"
    return direction, label


def _adapted_row(
    pair: Mapping[str, Any],
    *,
    direction: str,
    label: str,
) -> dict[str, Any]:
    if direction == "sentence1_to_sentence2":
        premise, hypothesis = pair["sentence1"], pair["sentence2"]
    elif direction == "sentence2_to_sentence1":
        premise, hypothesis = pair["sentence2"], pair["sentence1"]
    else:  # pragma: no cover - private callers are fixed
        raise PI2NLIAdaptationCacheError("unsupported PI2NLI direction")
    row_payload = {
        "schema_version": ADAPTATION_MAPPING_SCHEMA_VERSION,
        "mapping_seed": PI2NLI_MAPPING_SEED,
        "pair_id": pair["pair_id"],
        "direction": direction,
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "label_id": LABEL_TO_ID[label],
    }
    return {
        "row_id": _canonical_sha256(row_payload),
        **{
            key: item
            for key, item in row_payload.items()
            if key not in {"schema_version", "mapping_seed"}
        },
    }


def _adapted_rows(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map PI pairs deterministically onto the preserved three-way NLI head."""

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise PI2NLIAdaptationCacheError("PI2NLI pair must be a mapping")
        if not isinstance(pair.get("label"), bool):
            raise PI2NLIAdaptationCacheError("PI2NLI adapted-row input label must be Boolean")
        if pair["label"]:
            for direction in ("sentence1_to_sentence2", "sentence2_to_sentence1"):
                rows.append(_adapted_row(pair, direction=direction, label="entailment"))
        else:
            direction, label = _mapping_choice(pair)
            rows.append(_adapted_row(pair, direction=direction, label=label))
    rows.sort(key=lambda row: row["row_id"])
    if not rows or len({row["row_id"] for row in rows}) != len(rows):
        raise PI2NLIAdaptationCacheError("PI2NLI adapted rows are empty or not unique")
    return rows


def _validate_config_payload(config: Mapping[str, Any], *, context: str) -> None:
    expected_id2label = {str(index): label for index, label in enumerate(LABEL_ORDER)}
    expected_label2id = {label: index for index, label in enumerate(LABEL_ORDER)}
    raw_id2label = config.get("id2label")
    raw_label2id = config.get("label2id")
    normalized_id2label = (
        {str(key): value for key, value in raw_id2label.items()}
        if isinstance(raw_id2label, Mapping)
        else None
    )
    normalized_label2id = (
        {str(key): int(value) for key, value in raw_label2id.items()}
        if isinstance(raw_label2id, Mapping)
        and all(
            not isinstance(value, bool) and isinstance(value, (int, str))
            for value in raw_label2id.values()
        )
        else None
    )
    if (
        config.get("architectures") != ["DebertaV2ForSequenceClassification"]
        or normalized_id2label != expected_id2label
        or normalized_label2id != expected_label2id
        or config.get("max_position_embeddings") != MODEL_MAX_POSITION_EMBEDDINGS
    ):
        raise PI2NLIAdaptationCacheError(f"{context} three-way NLI configuration mismatch")


def authenticate_base_model_snapshot(model_dir: Path) -> dict[str, Any]:
    """Authenticate the exact pinned full-weight model and tokenizer snapshot."""

    resolved = model_dir.resolve()
    if not resolved.is_dir():
        raise PI2NLIAdaptationCacheError(f"pinned PI2NLI model directory is missing: {resolved}")
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in EXPECTED_BASE_MODEL_FILES.items():
        path = resolved / relative_path
        if not path.is_file():
            raise PI2NLIAdaptationCacheError(
                f"pinned PI2NLI model file is missing: {relative_path}"
            )
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise PI2NLIAdaptationCacheError(
                f"pinned PI2NLI model file SHA-256 mismatch: {relative_path}"
            )
        observed[relative_path] = actual_sha256
    try:
        config = json.loads((resolved / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PI2NLIAdaptationCacheError("pinned PI2NLI config.json is malformed") from exc
    if not isinstance(config, Mapping):
        raise PI2NLIAdaptationCacheError("pinned PI2NLI config.json must be a mapping")
    _validate_config_payload(config, context="pinned PI2NLI model")
    return {
        **dict(BASE_MODEL_MANIFEST_IDENTITY),
        "files": dict(sorted(observed.items())),
        "head": {
            "initialization": TRAINING_SPEC["head_initialization"],
            "label_order": list(LABEL_ORDER),
            "output_dimension": len(LABEL_ORDER),
            "tensor_sha256": dict(sorted(INITIAL_HEAD_TENSOR_SHA256.items())),
        },
    }


def _distribution_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise PI2NLIAdaptationCacheError(
            f"required PI2NLI runtime package is missing: {distribution}"
        ) from exc


def authenticate_runtime() -> dict[str, Any]:
    observed = {
        "python": platform.python_version(),
        "numpy": _distribution_version("numpy"),
        "torch": _distribution_version("torch"),
        "transformers": _distribution_version("transformers"),
        "tokenizers": _distribution_version("tokenizers"),
        "safetensors": _distribution_version("safetensors"),
        "huggingface-hub": _distribution_version("huggingface-hub"),
        "sentencepiece": _distribution_version("sentencepiece"),
    }
    if observed != dict(EXPECTED_RUNTIME_VERSIONS):
        raise PI2NLIAdaptationCacheError(
            "PI2NLI runtime version mismatch: "
            f"expected {dict(EXPECTED_RUNTIME_VERSIONS)!r}, got {observed!r}"
        )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise PI2NLIAdaptationCacheError("the pinned PI2NLI runtime requires Darwin ARM64")
    return {**observed, "system": "Darwin", "machine": "arm64"}


def _lazy_training_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime integration only
        raise PI2NLIAdaptationCacheError("PI2NLI training dependencies are unavailable") from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    header = {"dtype": str(value.dtype), "shape": list(value.shape)}
    digest = hashlib.sha256(_canonical_json(header).encode("utf-8"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _head_tensor_identity(model: Any) -> dict[str, str]:
    state = model.state_dict()
    result: dict[str, str] = {}
    for name in sorted(INITIAL_HEAD_TENSOR_SHA256):
        if name not in state:
            raise PI2NLIAdaptationCacheError(
                f"PI2NLI model is missing preserved head tensor {name}"
            )
        result[name] = _tensor_sha256(state[name])
    return result


def _validate_loaded_model_contract(model: Any, *, require_initial_head: bool) -> None:
    config = model.config.to_dict()
    _validate_config_payload(config, context="loaded PI2NLI model")
    classifier = getattr(model, "classifier", None)
    if (
        classifier is None
        or getattr(classifier, "in_features", None) != 384
        or getattr(classifier, "out_features", None) != len(LABEL_ORDER)
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI model did not preserve the three-way NLI head")
    if require_initial_head and _head_tensor_identity(model) != dict(INITIAL_HEAD_TENSOR_SHA256):
        raise PI2NLIAdaptationCacheError("PI2NLI pretrained three-way head tensor mismatch")


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested not in {"mps", "cpu"}:
        raise PI2NLIAdaptationCacheError("PI2NLI device must be 'mps' or 'cpu'")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise PI2NLIAdaptationCacheError("PI2NLI MPS device was requested but is unavailable")
    return torch.device(requested)


def _set_deterministic_runtime(torch: Any) -> None:
    random.seed(OPTIMIZATION_SEED)
    np.random.seed(OPTIMIZATION_SEED)
    torch.manual_seed(OPTIMIZATION_SEED)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(OPTIMIZATION_SEED)
    torch.use_deterministic_algorithms(True)


def _load_model_and_tokenizer(model_dir: Path, *, initial: bool) -> tuple[Any, Any, Any]:
    torch, model_class, tokenizer_class = _lazy_training_runtime()
    tokenizer = tokenizer_class.from_pretrained(
        str(model_dir.resolve()),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.model_max_length != MODEL_MAX_POSITION_EMBEDDINGS:
        raise PI2NLIAdaptationCacheError("PI2NLI tokenizer maximum length mismatch")
    model = model_class.from_pretrained(
        str(model_dir.resolve()),
        local_files_only=True,
        use_safetensors=True,
    )
    _validate_loaded_model_contract(model, require_initial_head=initial)
    return torch, model, tokenizer


def _encoded_batch(
    tokenizer: Any,
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    torch: Any,
    device: Any,
) -> tuple[dict[str, Any], int]:
    encoded = tokenizer(
        list(premises),
        list(hypotheses),
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    input_ids = encoded.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise PI2NLIAdaptationCacheError("PI2NLI tokenized input shape mismatch")
    sequence_length = int(input_ids.shape[1])
    if sequence_length > int(TRAINING_SPEC["max_length"]):
        raise PI2NLIAdaptationCacheError(
            "PI2NLI input exceeds the frozen 156-token limit; truncation is forbidden"
        )
    result = {
        key: value.to(device=device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    return result, sequence_length


def _train_one_epoch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    epoch: int,
) -> tuple[float, int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(OPTIMIZATION_SEED + epoch - 1)
    order = torch.randperm(len(rows), generator=generator).tolist()
    batch_size = int(TRAINING_SPEC["train_batch_size"])
    total_loss = 0.0
    total_rows = 0
    maximum_sequence_length = 0
    model.train()
    for start in range(0, len(order), batch_size):
        indexes = order[start : start + batch_size]
        batch = [rows[index] for index in indexes]
        encoded, sequence_length = _encoded_batch(
            tokenizer,
            [str(row["premise"]) for row in batch],
            [str(row["hypothesis"]) for row in batch],
            torch=torch,
            device=device,
        )
        maximum_sequence_length = max(maximum_sequence_length, sequence_length)
        labels = torch.tensor(
            [int(row["label_id"]) for row in batch],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(**encoded, labels=labels)
        loss = output.loss
        loss_value = float(loss.detach().to(device="cpu"))
        if not math.isfinite(loss_value) or loss_value < 0.0:
            raise PI2NLIAdaptationCacheError("PI2NLI training loss is invalid")
        loss.backward()
        optimizer.step()
        total_loss += loss_value * len(batch)
        total_rows += len(batch)
    if total_rows != len(rows):
        raise PI2NLIAdaptationCacheError("PI2NLI training row accounting mismatch")
    return total_loss / total_rows, maximum_sequence_length


def _stable_entailment_probability(logits: Sequence[float]) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (len(LABEL_ORDER),) or not np.all(np.isfinite(values)):
        raise PI2NLIAdaptationCacheError("PI2NLI logits must be a finite three-value vector")
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials, dtype=np.float64))
    probability = float(exponentials[LABEL_TO_ID["entailment"]] / denominator)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise PI2NLIAdaptationCacheError("PI2NLI entailment probability is invalid")
    return probability


def _score_pairs(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    pairs: Sequence[Mapping[str, Any]],
    device: Any,
) -> tuple[list[dict[str, Any]], int]:
    requests: list[tuple[int, str, str, str]] = []
    for pair in pairs:
        pair_id = int(pair["pair_id"])
        requests.append((pair_id, "forward", str(pair["sentence1"]), str(pair["sentence2"])))
        requests.append((pair_id, "reverse", str(pair["sentence2"]), str(pair["sentence1"])))
    batch_size = int(TRAINING_SPEC["eval_batch_size"])
    probabilities: dict[tuple[int, str], float] = {}
    maximum_sequence_length = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            encoded, sequence_length = _encoded_batch(
                tokenizer,
                [request[2] for request in batch],
                [request[3] for request in batch],
                torch=torch,
                device=device,
            )
            maximum_sequence_length = max(maximum_sequence_length, sequence_length)
            logits = (
                model(**encoded).logits.detach().to(device="cpu").to(dtype=torch.float64).numpy()
            )
            if logits.shape != (len(batch), len(LABEL_ORDER)):
                raise PI2NLIAdaptationCacheError("PI2NLI inference logits shape mismatch")
            for request, row in zip(batch, logits, strict=True):
                probabilities[(request[0], request[1])] = _stable_entailment_probability(row)
    if was_training:
        model.train()
    smallest_positive = np.finfo(np.float64).tiny
    features = [
        {
            "pair_id": int(pair["pair_id"]),
            "forward_log_entailment": math.log(
                max(probabilities[(int(pair["pair_id"]), "forward")], smallest_positive)
            ),
            "reverse_log_entailment": math.log(
                max(probabilities[(int(pair["pair_id"]), "reverse")], smallest_positive)
            ),
        }
        for pair in pairs
    ]
    features.sort(key=lambda row: row["pair_id"])
    _validate_features(features, expected_pair_ids=[int(pair["pair_id"]) for pair in pairs])
    return features, maximum_sequence_length


def _validate_features(values: Any, *, expected_pair_ids: Sequence[int] | None = None) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise PI2NLIAdaptationCacheError("PI2NLI features must be a non-empty sequence")
    pair_ids: list[int] = []
    for index, value in enumerate(values):
        feature = _exact_mapping(
            value,
            keys=frozenset({"pair_id", "forward_log_entailment", "reverse_log_entailment"}),
            context=f"PI2NLI feature {index}",
        )
        pair_id = _validated_positive_int(feature["pair_id"], context=f"feature {index} pair_id")
        pair_ids.append(pair_id)
        for name in ("forward_log_entailment", "reverse_log_entailment"):
            score = feature[name]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) > 1e-12
            ):
                raise PI2NLIAdaptationCacheError(f"PI2NLI feature {index} {name} is invalid")
    if pair_ids != sorted(pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise PI2NLIAdaptationCacheError("PI2NLI features must have unique, ascending pair IDs")
    if expected_pair_ids is not None and pair_ids != sorted(expected_pair_ids):
        raise PI2NLIAdaptationCacheError("PI2NLI feature pair coverage mismatch")


def _validate_manifest_identity(
    value: Any,
    *,
    held: bool,
) -> Mapping[str, Any]:
    keys = {"schema_version", "manifest_sha256", "pair_count"}
    if not held:
        keys.add("fold_count")
    identity = _exact_mapping(
        value,
        keys=frozenset(keys),
        context="PI2NLI cache manifest identity",
    )
    expected_schema = HELD_MANIFEST_SCHEMA_VERSION if held else CALIBRATION_MANIFEST_SCHEMA_VERSION
    if identity["schema_version"] != expected_schema or not _is_sha256(identity["manifest_sha256"]):
        raise PI2NLIAdaptationCacheError("PI2NLI cache manifest identity is invalid")
    _validated_positive_int(identity["pair_count"], context="cache manifest pair_count")
    if not held:
        _validated_positive_int(identity["fold_count"], context="cache manifest fold_count")
    return identity


def _validate_runtime_identity(value: Any) -> None:
    runtime = _exact_mapping(
        value,
        keys=frozenset({*EXPECTED_RUNTIME_VERSIONS, "system", "machine"}),
        context="PI2NLI runtime identity",
    )
    expected = {**EXPECTED_RUNTIME_VERSIONS, "system": "Darwin", "machine": "arm64"}
    if dict(runtime) != expected:
        raise PI2NLIAdaptationCacheError("PI2NLI cache runtime identity mismatch")


def _validate_execution(value: Any) -> None:
    execution = _exact_mapping(
        value,
        keys=frozenset({"device", "deterministic_algorithms"}),
        context="PI2NLI execution identity",
    )
    if (
        execution["device"] not in {"mps", "cpu"}
        or execution["deterministic_algorithms"] is not True
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI execution identity mismatch")


def _validate_base_model_cache_identity(value: Any) -> None:
    model = _exact_mapping(
        value,
        keys=frozenset({*BASE_MODEL_MANIFEST_IDENTITY, "files", "head"}),
        context="PI2NLI cache base-model identity",
    )
    if {key: model[key] for key in BASE_MODEL_MANIFEST_IDENTITY} != dict(
        BASE_MODEL_MANIFEST_IDENTITY
    ) or model["files"] != dict(EXPECTED_BASE_MODEL_FILES):
        raise PI2NLIAdaptationCacheError("PI2NLI cache base-model identity mismatch")
    head = _exact_mapping(
        model["head"],
        keys=frozenset({"initialization", "label_order", "output_dimension", "tensor_sha256"}),
        context="PI2NLI base-model head identity",
    )
    if head != {
        "initialization": TRAINING_SPEC["head_initialization"],
        "label_order": list(LABEL_ORDER),
        "output_dimension": len(LABEL_ORDER),
        "tensor_sha256": dict(INITIAL_HEAD_TENSOR_SHA256),
    }:
        raise PI2NLIAdaptationCacheError("PI2NLI base-model head identity mismatch")


def _validate_adaptation_identity(value: Any) -> None:
    adaptation = _exact_mapping(
        value,
        keys=frozenset({"schema_version", "mapping_seed", "row_count", "rows_sha256"}),
        context="PI2NLI adaptation identity",
    )
    if (
        adaptation["schema_version"] != ADAPTATION_MAPPING_SCHEMA_VERSION
        or adaptation["mapping_seed"] != PI2NLI_MAPPING_SEED
        or not _is_sha256(adaptation["rows_sha256"])
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI adaptation identity mismatch")
    _validated_positive_int(adaptation["row_count"], context="adaptation row_count")


def _validate_checkpoint_identity(value: Any) -> Mapping[str, Any]:
    checkpoint = _exact_mapping(
        value,
        keys=frozenset({"schema_version", "artifact_sha256", "files", "selected_epoch", "head"}),
        context="PI2NLI checkpoint identity",
    )
    files = checkpoint["files"]
    if not isinstance(files, Mapping) or not files:
        raise PI2NLIAdaptationCacheError("PI2NLI checkpoint files identity is invalid")
    for name, sha256 in files.items():
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in Path(name).parts
            or not _is_sha256(sha256)
        ):
            raise PI2NLIAdaptationCacheError("PI2NLI checkpoint file identity is invalid")
    artifact_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "files": dict(sorted(files.items())),
    }
    if (
        checkpoint["schema_version"] != CHECKPOINT_SCHEMA_VERSION
        or checkpoint["artifact_sha256"] != _canonical_sha256(artifact_payload)
        or checkpoint["selected_epoch"] not in TRAINING_EPOCHS
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI checkpoint artifact identity mismatch")
    head = _exact_mapping(
        checkpoint["head"],
        keys=frozenset(
            {
                "initialization",
                "label_order",
                "output_dimension",
                "initial_tensor_sha256",
                "final_tensor_sha256",
            }
        ),
        context="PI2NLI checkpoint head identity",
    )
    if (
        head["initialization"] != TRAINING_SPEC["head_initialization"]
        or head["label_order"] != list(LABEL_ORDER)
        or head["output_dimension"] != len(LABEL_ORDER)
        or head["initial_tensor_sha256"] != dict(INITIAL_HEAD_TENSOR_SHA256)
        or not isinstance(head["final_tensor_sha256"], Mapping)
        or set(head["final_tensor_sha256"]) != set(INITIAL_HEAD_TENSOR_SHA256)
        or not all(_is_sha256(item) for item in head["final_tensor_sha256"].values())
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI checkpoint head contract mismatch")
    return checkpoint


def _validate_selection(
    value: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PI2NLIAdaptationCacheError("PI2NLI frozen selection must be a mapping")
    required = {
        "schema_version",
        "selection_sha256",
        "experiment_id",
        "training_manifest_sha256",
        "crossfit_cache_sha256",
        "selected_epoch",
    }
    if not required.issubset(value):
        raise PI2NLIAdaptationCacheError("PI2NLI frozen selection is missing required keys")
    payload = {key: item for key, item in value.items() if key != "selection_sha256"}
    if (
        value["schema_version"] != SELECTION_SCHEMA_VERSION
        or not _is_sha256(value["selection_sha256"])
        or value["selection_sha256"] != _canonical_sha256(payload)
        or not _is_sha256(value["training_manifest_sha256"])
        or not _is_sha256(value["crossfit_cache_sha256"])
        or value["selected_epoch"] not in TRAINING_EPOCHS
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI frozen selection identity mismatch")
    _validated_nonempty_string(value["experiment_id"], context="selection experiment_id")
    if manifest is not None and (
        value["experiment_id"] != manifest["experiment_id"]
        or value["training_manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI selection does not bind the training manifest")
    return json.loads(_canonical_json(value))


def _validate_cache(
    value: Any,
    *,
    expected_schema_version: str | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize one hash-bound PI2NLI cache envelope."""

    envelope = _exact_mapping(
        value,
        keys=frozenset({"schema_version", "payload", "cache_sha256"}),
        context="PI2NLI cache envelope",
    )
    schema_version = envelope["schema_version"]
    supported = {
        CROSSFIT_CACHE_SCHEMA_VERSION,
        FINAL_FIT_CACHE_SCHEMA_VERSION,
        HELD_INFERENCE_CACHE_SCHEMA_VERSION,
    }
    if schema_version not in supported or (
        expected_schema_version is not None and schema_version != expected_schema_version
    ):
        raise PI2NLIAdaptationCacheError("unsupported PI2NLI cache schema")
    payload = envelope["payload"]
    if not isinstance(payload, Mapping) or envelope["cache_sha256"] != _canonical_sha256(
        {"schema_version": schema_version, "payload": payload}
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI cache SHA-256 mismatch")

    common_keys = {
        "experiment_id",
        "manifest",
        "pair_ids",
        "base_model",
        "runtime",
        "execution",
        "authority",
    }
    if schema_version == CROSSFIT_CACHE_SCHEMA_VERSION:
        expected_keys = common_keys | {
            "training",
            "adaptation",
            "folds",
        }
    elif schema_version == FINAL_FIT_CACHE_SCHEMA_VERSION:
        expected_keys = common_keys | {
            "selection",
            "selected_epoch",
            "training",
            "adaptation",
            "training_epochs",
            "checkpoint",
            "features",
            "features_sha256",
        }
    else:
        expected_keys = common_keys | {
            "final_fit_cache",
            "checkpoint",
            "features",
            "features_sha256",
        }
    _exact_mapping(payload, keys=frozenset(expected_keys), context="PI2NLI cache payload")
    _validated_nonempty_string(payload["experiment_id"], context="cache experiment_id")
    _validate_base_model_cache_identity(payload["base_model"])
    _validate_runtime_identity(payload["runtime"])
    _validate_execution(payload["execution"])
    _validate_authority(payload["authority"])
    pair_ids = payload["pair_ids"]
    if (
        isinstance(pair_ids, (str, bytes))
        or not isinstance(pair_ids, Sequence)
        or not pair_ids
        or any(isinstance(pair_id, bool) or not isinstance(pair_id, int) for pair_id in pair_ids)
        or list(pair_ids) != sorted(pair_ids)
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI cache pair IDs are invalid")

    if schema_version == CROSSFIT_CACHE_SCHEMA_VERSION:
        identity = _validate_manifest_identity(payload["manifest"], held=False)
        _validate_training_spec(payload["training"])
        _validate_adaptation_identity(payload["adaptation"])
        folds = payload["folds"]
        if isinstance(folds, (str, bytes)) or not isinstance(folds, Sequence) or not folds:
            raise PI2NLIAdaptationCacheError("PI2NLI crossfit cache folds are invalid")
        expected_pair_ids = list(pair_ids)
        for fold_index, raw_fold in enumerate(folds):
            fold = _exact_mapping(
                raw_fold,
                keys=frozenset(
                    {
                        "fold",
                        "training_pair_count",
                        "validation_pair_count",
                        "training_row_count",
                        "training_rows_sha256",
                        "epochs",
                    }
                ),
                context=f"PI2NLI crossfit fold {fold_index}",
            )
            if fold["fold"] != fold_index or not _is_sha256(fold["training_rows_sha256"]):
                raise PI2NLIAdaptationCacheError("PI2NLI crossfit fold identity mismatch")
            for name in ("training_pair_count", "validation_pair_count", "training_row_count"):
                _validated_positive_int(fold[name], context=f"crossfit fold {fold_index} {name}")
            epochs = fold["epochs"]
            if isinstance(epochs, (str, bytes)) or not isinstance(epochs, Sequence):
                raise PI2NLIAdaptationCacheError("PI2NLI crossfit epochs are invalid")
            if [item.get("epoch") for item in epochs if isinstance(item, Mapping)] != list(
                TRAINING_EPOCHS
            ):
                raise PI2NLIAdaptationCacheError("PI2NLI crossfit must capture epochs 1 through 12")
            for raw_epoch in epochs:
                epoch = _exact_mapping(
                    raw_epoch,
                    keys=frozenset(
                        {
                            "epoch",
                            "mean_training_loss",
                            "maximum_sequence_length",
                            "features",
                            "features_sha256",
                        }
                    ),
                    context="PI2NLI crossfit epoch",
                )
                if (
                    isinstance(epoch["mean_training_loss"], bool)
                    or not isinstance(epoch["mean_training_loss"], (int, float))
                    or not math.isfinite(float(epoch["mean_training_loss"]))
                    or float(epoch["mean_training_loss"]) < 0.0
                    or isinstance(epoch["maximum_sequence_length"], bool)
                    or not isinstance(epoch["maximum_sequence_length"], int)
                    or not 0 < epoch["maximum_sequence_length"] <= TRAINING_SPEC["max_length"]
                ):
                    raise PI2NLIAdaptationCacheError("PI2NLI crossfit epoch evidence is invalid")
                _validate_features(epoch["features"], expected_pair_ids=expected_pair_ids)
                if epoch["features_sha256"] != _canonical_sha256(epoch["features"]):
                    raise PI2NLIAdaptationCacheError("PI2NLI crossfit feature SHA-256 mismatch")
        if len(folds) != identity["fold_count"]:
            raise PI2NLIAdaptationCacheError("PI2NLI crossfit fold count mismatch")
    elif schema_version == FINAL_FIT_CACHE_SCHEMA_VERSION:
        _validate_manifest_identity(payload["manifest"], held=False)
        _validate_training_spec(payload["training"])
        _validate_adaptation_identity(payload["adaptation"])
        selection = payload["selection"]
        _exact_mapping(
            selection,
            keys=frozenset(
                {"schema_version", "selection_sha256", "crossfit_cache_sha256", "selected_epoch"}
            ),
            context="PI2NLI final-fit selection identity",
        )
        if (
            selection["schema_version"] != SELECTION_SCHEMA_VERSION
            or not _is_sha256(selection["selection_sha256"])
            or not _is_sha256(selection["crossfit_cache_sha256"])
            or selection["selected_epoch"] not in TRAINING_EPOCHS
        ):
            raise PI2NLIAdaptationCacheError("PI2NLI final-fit selection identity mismatch")
        if payload["selected_epoch"] != selection["selected_epoch"]:
            raise PI2NLIAdaptationCacheError("PI2NLI final-fit selected epoch mismatch")
        epochs = payload["training_epochs"]
        if (
            isinstance(epochs, (str, bytes))
            or not isinstance(epochs, Sequence)
            or [item.get("epoch") for item in epochs if isinstance(item, Mapping)]
            != list(range(1, int(selection["selected_epoch"]) + 1))
        ):
            raise PI2NLIAdaptationCacheError("PI2NLI final-fit epoch evidence mismatch")
        for raw_epoch in epochs:
            epoch = _exact_mapping(
                raw_epoch,
                keys=frozenset({"epoch", "mean_training_loss", "maximum_sequence_length"}),
                context="PI2NLI final-fit epoch",
            )
            if (
                not math.isfinite(float(epoch["mean_training_loss"]))
                or float(epoch["mean_training_loss"]) < 0.0
                or not 0 < epoch["maximum_sequence_length"] <= TRAINING_SPEC["max_length"]
            ):
                raise PI2NLIAdaptationCacheError("PI2NLI final-fit epoch evidence is invalid")
        _validate_checkpoint_identity(payload["checkpoint"])
        expected_pair_ids = list(pair_ids)
        _validate_features(payload["features"], expected_pair_ids=expected_pair_ids)
        if payload["features_sha256"] != _canonical_sha256(payload["features"]):
            raise PI2NLIAdaptationCacheError("PI2NLI calibration feature SHA-256 mismatch")
    else:
        _validate_manifest_identity(payload["manifest"], held=True)
        final_fit_cache = _exact_mapping(
            payload["final_fit_cache"],
            keys=frozenset({"schema_version", "cache_sha256"}),
            context="PI2NLI held final-fit cache identity",
        )
        if final_fit_cache["schema_version"] != FINAL_FIT_CACHE_SCHEMA_VERSION or not _is_sha256(
            final_fit_cache["cache_sha256"]
        ):
            raise PI2NLIAdaptationCacheError("PI2NLI held final-fit cache identity mismatch")
        _validate_checkpoint_identity(payload["checkpoint"])
        expected_pair_ids = list(pair_ids)
        _validate_features(payload["features"], expected_pair_ids=expected_pair_ids)
        if payload["features_sha256"] != _canonical_sha256(payload["features"]):
            raise PI2NLIAdaptationCacheError("PI2NLI held feature SHA-256 mismatch")

    if manifest is not None:
        manifest_identity = payload["manifest"]
        expected_identity = {
            "schema_version": manifest["schema_version"],
            "manifest_sha256": manifest["manifest_sha256"],
            "pair_count": len(manifest["pairs"]),
        }
        if manifest["schema_version"] == CALIBRATION_MANIFEST_SCHEMA_VERSION:
            expected_identity["fold_count"] = len(manifest["folds"])
        if (
            manifest_identity != expected_identity
            or payload["experiment_id"] != manifest["experiment_id"]
            or list(payload["pair_ids"]) != [int(pair["pair_id"]) for pair in manifest["pairs"]]
        ):
            raise PI2NLIAdaptationCacheError("PI2NLI cache does not bind its manifest")
    return json.loads(_canonical_json(envelope))


def _load_validated_cache(
    path: Path,
    *,
    expected_schema_version: str,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PI2NLIAdaptationCacheError(f"cannot load PI2NLI cache: {path}") from exc
    return _validate_cache(
        value,
        expected_schema_version=expected_schema_version,
        manifest=manifest,
    )


def _validate_final_cache_alignment(
    final_cache: Mapping[str, Any],
    held_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require a label-free held manifest to bind the frozen final checkpoint."""

    validated_final = _validate_cache(
        final_cache,
        expected_schema_version=FINAL_FIT_CACHE_SCHEMA_VERSION,
    )
    validated_held = _validate_manifest(
        held_manifest,
        expected_schema_version=HELD_MANIFEST_SCHEMA_VERSION,
    )
    checkpoint = validated_final["payload"]["checkpoint"]
    held_checkpoint = validated_held["checkpoint"]
    if (
        validated_final["payload"]["experiment_id"] != validated_held["experiment_id"]
        or held_checkpoint["artifact_sha256"] != checkpoint["artifact_sha256"]
        or held_checkpoint["selected_epoch"] != checkpoint["selected_epoch"]
        or validated_held["model"] != dict(BASE_MODEL_MANIFEST_IDENTITY)
    ):
        raise PI2NLIAdaptationCacheError(
            "PI2NLI held manifest does not align with the frozen final cache"
        )
    return checkpoint


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest["manifest_sha256"],
        "pair_count": len(manifest["pairs"]),
    }
    if manifest["schema_version"] == CALIBRATION_MANIFEST_SCHEMA_VERSION:
        result["fold_count"] = len(manifest["folds"])
    return result


def _adaptation_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": ADAPTATION_MAPPING_SCHEMA_VERSION,
        "mapping_seed": PI2NLI_MAPPING_SEED,
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
    }


def _execution_identity(device: str) -> dict[str, Any]:
    return {"device": device, "deterministic_algorithms": True}


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


def _refuse_existing(path: Path, *, context: str) -> None:
    if path.exists():
        raise PI2NLIAdaptationCacheError(f"{context} already exists; refusing to overwrite: {path}")


def _envelope(schema_version: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "payload": payload,
        "cache_sha256": _canonical_sha256({"schema_version": schema_version, "payload": payload}),
    }


def run_crossfit(
    *,
    manifest_path: Path,
    base_model_dir: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    _refuse_existing(output_path, context="PI2NLI crossfit cache")
    manifest = _load_validated_manifest(
        manifest_path,
        expected_schema_version=CALIBRATION_MANIFEST_SCHEMA_VERSION,
    )
    base_model = authenticate_base_model_snapshot(base_model_dir)
    runtime = authenticate_runtime()
    all_pairs = manifest["pairs"]
    all_rows = _adapted_rows(all_pairs)
    pair_by_id = {int(pair["pair_id"]): pair for pair in all_pairs}
    fold_results: list[dict[str, Any]] = []
    for fold in manifest["folds"]:
        fold_index = int(fold["fold"])
        training_pairs = [pair_by_id[int(pair_id)] for pair_id in fold["training_pair_ids"]]
        training_rows = _adapted_rows(training_pairs)
        torch, model, tokenizer = _load_model_and_tokenizer(base_model_dir, initial=True)
        _set_deterministic_runtime(torch)
        device = _resolve_device(torch, device_name)
        model.to(device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(TRAINING_SPEC["learning_rate"]),
            weight_decay=float(TRAINING_SPEC["weight_decay"]),
            eps=float(TRAINING_SPEC["adam_epsilon"]),
        )
        epochs: list[dict[str, Any]] = []
        for epoch in TRAINING_EPOCHS:
            mean_loss, training_maximum = _train_one_epoch(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                rows=training_rows,
                device=device,
                epoch=epoch,
            )
            features, inference_maximum = _score_pairs(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                pairs=all_pairs,
                device=device,
            )
            epochs.append(
                {
                    "epoch": epoch,
                    "mean_training_loss": mean_loss,
                    "maximum_sequence_length": max(training_maximum, inference_maximum),
                    "features": features,
                    "features_sha256": _canonical_sha256(features),
                }
            )
            print(
                f"PI2NLI crossfit fold={fold_index} epoch={epoch} loss={mean_loss:.8f}",
                flush=True,
            )
        fold_results.append(
            {
                "fold": fold_index,
                "training_pair_count": len(training_pairs),
                "validation_pair_count": len(fold["validation_pair_ids"]),
                "training_row_count": len(training_rows),
                "training_rows_sha256": _canonical_sha256(training_rows),
                "epochs": epochs,
            }
        )
        del optimizer, model, tokenizer
        if device_name == "mps":
            torch.mps.empty_cache()
    payload = {
        "experiment_id": manifest["experiment_id"],
        "manifest": _manifest_identity(manifest),
        "pair_ids": [int(pair["pair_id"]) for pair in all_pairs],
        "base_model": base_model,
        "runtime": runtime,
        "execution": _execution_identity(device_name),
        "authority": dict(AUTHORITY_CONTRACT),
        "training": dict(TRAINING_SPEC),
        "adaptation": _adaptation_identity(all_rows),
        "folds": fold_results,
    }
    result = _validate_cache(
        _envelope(CROSSFIT_CACHE_SCHEMA_VERSION, payload),
        expected_schema_version=CROSSFIT_CACHE_SCHEMA_VERSION,
        manifest=manifest,
    )
    _atomic_json(output_path, result)
    replayed = _load_validated_cache(
        output_path,
        expected_schema_version=CROSSFIT_CACHE_SCHEMA_VERSION,
        manifest=manifest,
    )
    if replayed != result:
        raise PI2NLIAdaptationCacheError("persisted PI2NLI crossfit cache did not replay")
    return result


def _checkpoint_files(model_dir: Path) -> dict[str, str]:
    required = {
        "config.json",
        MODEL_WEIGHTS_RELATIVE_PATH,
        "added_tokens.json",
        "special_tokens_map.json",
        "spm.model",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    files: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            if path.is_dir():
                continue
            raise PI2NLIAdaptationCacheError("PI2NLI checkpoint contains an unsupported entry")
        relative = path.relative_to(model_dir).as_posix()
        files[relative] = _file_sha256(path)
    if not required.issubset(files):
        raise PI2NLIAdaptationCacheError("PI2NLI checkpoint is missing model/tokenizer files")
    return dict(sorted(files.items()))


def _checkpoint_identity(
    model_dir: Path,
    *,
    selected_epoch: int,
    final_head_sha256: Mapping[str, str],
) -> dict[str, Any]:
    files = _checkpoint_files(model_dir)
    artifact_payload = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "files": files}
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_sha256": _canonical_sha256(artifact_payload),
        "files": files,
        "selected_epoch": selected_epoch,
        "head": {
            "initialization": TRAINING_SPEC["head_initialization"],
            "label_order": list(LABEL_ORDER),
            "output_dimension": len(LABEL_ORDER),
            "initial_tensor_sha256": dict(INITIAL_HEAD_TENSOR_SHA256),
            "final_tensor_sha256": dict(sorted(final_head_sha256.items())),
        },
    }


def _save_checkpoint_atomically(model: Any, tokenizer: Any, output_dir: Path) -> None:
    _refuse_existing(output_dir, context="PI2NLI final model directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        model.save_pretrained(temporary_dir, safe_serialization=True)
        tokenizer.save_pretrained(temporary_dir)
        _checkpoint_files(temporary_dir)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise


def run_fit_final(
    *,
    manifest_path: Path,
    selection_path: Path,
    base_model_dir: Path,
    model_output_dir: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    _refuse_existing(output_path, context="PI2NLI final-fit cache")
    _refuse_existing(model_output_dir, context="PI2NLI final model directory")
    manifest = _load_validated_manifest(
        manifest_path,
        expected_schema_version=CALIBRATION_MANIFEST_SCHEMA_VERSION,
    )
    try:
        raw_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PI2NLIAdaptationCacheError(
            f"cannot load PI2NLI frozen selection: {selection_path}"
        ) from exc
    selection = _validate_selection(raw_selection, manifest=manifest)
    base_model = authenticate_base_model_snapshot(base_model_dir)
    runtime = authenticate_runtime()
    rows = _adapted_rows(manifest["pairs"])
    selected_epoch = int(selection["selected_epoch"])
    torch, model, tokenizer = _load_model_and_tokenizer(base_model_dir, initial=True)
    _set_deterministic_runtime(torch)
    device = _resolve_device(torch, device_name)
    model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(TRAINING_SPEC["learning_rate"]),
        weight_decay=float(TRAINING_SPEC["weight_decay"]),
        eps=float(TRAINING_SPEC["adam_epsilon"]),
    )
    epoch_evidence: list[dict[str, Any]] = []
    for epoch in range(1, selected_epoch + 1):
        mean_loss, maximum_sequence_length = _train_one_epoch(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            rows=rows,
            device=device,
            epoch=epoch,
        )
        epoch_evidence.append(
            {
                "epoch": epoch,
                "mean_training_loss": mean_loss,
                "maximum_sequence_length": maximum_sequence_length,
            }
        )
        print(f"PI2NLI final epoch={epoch} loss={mean_loss:.8f}", flush=True)
    features, inference_maximum = _score_pairs(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        pairs=manifest["pairs"],
        device=device,
    )
    epoch_evidence[-1]["maximum_sequence_length"] = max(
        int(epoch_evidence[-1]["maximum_sequence_length"]), inference_maximum
    )
    final_head_sha256 = _head_tensor_identity(model)
    model.to(device="cpu")
    _save_checkpoint_atomically(model, tokenizer, model_output_dir)
    checkpoint = _checkpoint_identity(
        model_output_dir,
        selected_epoch=selected_epoch,
        final_head_sha256=final_head_sha256,
    )
    payload = {
        "experiment_id": manifest["experiment_id"],
        "manifest": _manifest_identity(manifest),
        "pair_ids": [int(pair["pair_id"]) for pair in manifest["pairs"]],
        "selection": {
            "schema_version": selection["schema_version"],
            "selection_sha256": selection["selection_sha256"],
            "crossfit_cache_sha256": selection["crossfit_cache_sha256"],
            "selected_epoch": selected_epoch,
        },
        "selected_epoch": selected_epoch,
        "base_model": base_model,
        "runtime": runtime,
        "execution": _execution_identity(device_name),
        "authority": dict(AUTHORITY_CONTRACT),
        "training": dict(TRAINING_SPEC),
        "adaptation": _adaptation_identity(rows),
        "training_epochs": epoch_evidence,
        "checkpoint": checkpoint,
        "features": features,
        "features_sha256": _canonical_sha256(features),
    }
    result = _validate_cache(
        _envelope(FINAL_FIT_CACHE_SCHEMA_VERSION, payload),
        expected_schema_version=FINAL_FIT_CACHE_SCHEMA_VERSION,
        manifest=manifest,
    )
    _atomic_json(output_path, result)
    replayed = _load_validated_cache(
        output_path,
        expected_schema_version=FINAL_FIT_CACHE_SCHEMA_VERSION,
        manifest=manifest,
    )
    if replayed != result:
        raise PI2NLIAdaptationCacheError("persisted PI2NLI final-fit cache did not replay")
    return result


def run_infer_held(
    *,
    manifest_path: Path,
    final_fit_cache_path: Path,
    model_dir: Path,
    output_path: Path,
    device_name: str,
) -> dict[str, Any]:
    _refuse_existing(output_path, context="PI2NLI held inference cache")
    manifest = _load_validated_manifest(
        manifest_path,
        expected_schema_version=HELD_MANIFEST_SCHEMA_VERSION,
    )
    final_cache = _load_validated_cache(
        final_fit_cache_path,
        expected_schema_version=FINAL_FIT_CACHE_SCHEMA_VERSION,
    )
    checkpoint = _validate_final_cache_alignment(final_cache, manifest)
    observed_files = _checkpoint_files(model_dir)
    artifact_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "files": observed_files,
    }
    if (
        observed_files != checkpoint["files"]
        or _canonical_sha256(artifact_payload) != checkpoint["artifact_sha256"]
    ):
        raise PI2NLIAdaptationCacheError("PI2NLI final checkpoint files do not match the cache")
    runtime = authenticate_runtime()
    torch, model, tokenizer = _load_model_and_tokenizer(model_dir, initial=False)
    if _head_tensor_identity(model) != checkpoint["head"]["final_tensor_sha256"]:
        raise PI2NLIAdaptationCacheError("PI2NLI final checkpoint head tensor mismatch")
    _set_deterministic_runtime(torch)
    device = _resolve_device(torch, device_name)
    model.to(device=device)
    features, _maximum_sequence_length = _score_pairs(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        pairs=manifest["pairs"],
        device=device,
    )
    payload = {
        "experiment_id": manifest["experiment_id"],
        "manifest": _manifest_identity(manifest),
        "pair_ids": [int(pair["pair_id"]) for pair in manifest["pairs"]],
        "final_fit_cache": {
            "schema_version": final_cache["schema_version"],
            "cache_sha256": final_cache["cache_sha256"],
        },
        "base_model": final_cache["payload"]["base_model"],
        "runtime": runtime,
        "execution": _execution_identity(device_name),
        "authority": dict(AUTHORITY_CONTRACT),
        "checkpoint": checkpoint,
        "features": features,
        "features_sha256": _canonical_sha256(features),
    }
    result = _validate_cache(
        _envelope(HELD_INFERENCE_CACHE_SCHEMA_VERSION, payload),
        expected_schema_version=HELD_INFERENCE_CACHE_SCHEMA_VERSION,
        manifest=manifest,
    )
    _atomic_json(output_path, result)
    replayed = _load_validated_cache(
        output_path,
        expected_schema_version=HELD_INFERENCE_CACHE_SCHEMA_VERSION,
        manifest=manifest,
    )
    if replayed != result:
        raise PI2NLIAdaptationCacheError("persisted PI2NLI held cache did not replay")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("crossfit", "fit-final", "infer-held"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_WORK_DIR / "frozen_selection.json",
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=DEFAULT_WORK_DIR / "final_model",
    )
    parser.add_argument(
        "--final-fit-cache",
        type=Path,
        default=DEFAULT_WORK_DIR / "final_fit_cache.json",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_WORK_DIR / "final_model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase == "crossfit":
        manifest_path = args.manifest or DEFAULT_WORK_DIR / "calibration_training_manifest.json"
        output_path = args.output or DEFAULT_WORK_DIR / "crossfit_cache.json"
        result = run_crossfit(
            manifest_path=manifest_path.resolve(),
            base_model_dir=args.base_model_dir.resolve(),
            output_path=output_path.resolve(),
            device_name=args.device,
        )
    elif args.phase == "fit-final":
        manifest_path = args.manifest or DEFAULT_WORK_DIR / "calibration_training_manifest.json"
        output_path = args.output or DEFAULT_WORK_DIR / "final_fit_cache.json"
        result = run_fit_final(
            manifest_path=manifest_path.resolve(),
            selection_path=args.selection.resolve(),
            base_model_dir=args.base_model_dir.resolve(),
            model_output_dir=args.model_output_dir.resolve(),
            output_path=output_path.resolve(),
            device_name=args.device,
        )
    else:
        manifest_path = args.manifest or DEFAULT_WORK_DIR / "held_request_manifest.json"
        output_path = args.output or DEFAULT_WORK_DIR / "held_inference_cache.json"
        result = run_infer_held(
            manifest_path=manifest_path.resolve(),
            final_fit_cache_path=args.final_fit_cache.resolve(),
            model_dir=args.model_dir.resolve(),
            output_path=output_path.resolve(),
            device_name=args.device,
        )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "schema_version": result["schema_version"],
                "cache_sha256": result["cache_sha256"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
