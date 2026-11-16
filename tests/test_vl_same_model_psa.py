from pathlib import Path

import numpy as np
from scripts.research.psa.run_image_caption_neighborhood_experiment import (
    DEFAULT_MULTIMODAL_MODEL_ID,
    DEFAULT_MULTIMODAL_MODEL_REVISION,
    EmbeddingCache,
    ServiceConfig,
)
from scripts.research.psa.run_vl_same_model_psa import (
    _compare,
    _load_reused_image_vectors,
)


def _vl_config() -> ServiceConfig:
    return ServiceConfig(
        model_id=DEFAULT_MULTIMODAL_MODEL_ID,
        model_revision=DEFAULT_MULTIMODAL_MODEL_REVISION,
        endpoint_url="https://pilot.endpoints.huggingface.cloud/embed",
        provider="hf-endpoint",
        api_style="hf-multimodal",
        token_env="HF_TOKEN",
        timeout=1.0,
    )


def test_reused_image_cache_is_model_and_revision_bound(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.sqlite3"
    cache = EmbeddingCache(cache_path)
    try:
        cache.put(
            _vl_config(),
            "image",
            "guide:0:image",
            [1.0] * 2048,
            {"fixture": True},
        )
    finally:
        cache.close()

    vectors, provenance = _load_reused_image_vectors(
        cache_path,
        ["guide:0:image"],
    )

    assert len(vectors["guide:0:image"]) == 2048
    assert provenance["reused_image_vectors"] == 1
    assert provenance["model_revision"] == DEFAULT_MULTIMODAL_MODEL_REVISION


def test_candidate_recall_uses_the_mrpc_either_direction_rule() -> None:
    left = np.asarray([0.9, 0.8, 0.1], dtype=np.float32)
    right = np.asarray([0.7, 0.95, 0.2], dtype=np.float32)

    at_one = _compare(left, right, top_k=1)
    at_two = _compare(left, right, top_k=2)

    assert at_one["candidate_recall_either_direction"] == 0.0
    assert at_two["candidate_recall_either_direction"] == 1.0
    assert at_two["candidate_recall_both_directions"] == 1.0
