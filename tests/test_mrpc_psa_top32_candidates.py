from __future__ import annotations

import numpy as np
from scripts.research.mrpc.analyze_mrpc_psa_top32_candidates import (
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    _ranked_psa_neighborhoods,
    _relation_index,
)

from mcp_servers.mcp_lib.qwen_typed_field_test import MRPCPair


def test_ranked_psa_neighborhoods_are_deterministic_and_exclude_self() -> None:
    signatures = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.9, 0.1], dtype=np.float32),
        "c": np.asarray([0.0, 1.0], dtype=np.float32),
    }

    neighborhoods, evidence = _ranked_psa_neighborhoods(
        signatures,
        top_k=2,
    )

    assert [text for text, _score in neighborhoods["a"]] == ["b", "c"]
    assert all(text != "a" for text, _score in neighborhoods["a"])
    assert evidence["directed_candidate_count"] == 6


def test_relation_index_preserves_official_labels_and_splits() -> None:
    pairs = [
        MRPCPair(1, "alpha", "beta", True),
        MRPCPair(2, "gamma", "delta", False),
    ]

    index = _relation_index(
        pairs,
        {1: "calibration", 2: "held_out"},
    )

    relations = {value["relation"] for value in index.values()}
    assert relations == {RELATION_POSITIVE, RELATION_NEGATIVE}
    assert sorted(value["split"] for value in index.values()) == [
        "calibration",
        "held_out",
    ]
