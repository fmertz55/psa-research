"""Measure difference-vector geometry per membership edge type.

Tests one falsifiable hypothesis from the anisotropy program: single-axis
departures from a paraphrase attractor (member_scale, member_temporal, ...)
should differ from the core in a *concentrated* set of embedding dimensions,
while unrelated or multi-axis pairs differ *diffusely*.  Geometry proposes,
evidence types — this script only measures whether the embedding could
pre-sort edge types, using the pinned caches and the latest report traces.

Usage:
    temp/wordnet-venv/bin/python scripts/research/mrpc/analyze_membership_edge_geometry.py \
        --report reports/<run>/mrpc_proposition_graph_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import (  # noqa: E402
    _load_cache_evidence,
)
from scripts.research.mrpc.run_mrpc_proposition_graph_benchmark import DEFAULT_CACHE  # noqa: E402

from mcp_servers.mcp_lib.qwen_typed_field_test import load_mrpc_pairs  # noqa: E402

_TOP_MASS_DIMENSIONS = 32
DEFAULT_MRPC_DIR = REPOSITORY_ROOT / "app" / "utils" / "msrpc_split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mrpc-dir", type=Path, default=DEFAULT_MRPC_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    traces = report["held_out_pair_traces"]["profiles"]["mrpc_compatible_v1"]["learned_gate"]
    pairs = {pair.pair_id: pair for pair in load_mrpc_pairs(args.mrpc_dir.resolve())}
    trace_pairs = [pairs[int(trace["pair_id"])] for trace in traces]
    native, _guides, _evidence = _load_cache_evidence(args.cache.resolve(), trace_pairs)

    rows = defaultdict(list)
    for trace in traces:
        pair = pairs[int(trace["pair_id"])]
        a = np.asarray(native[pair.sentence1], dtype=np.float64)
        b = np.asarray(native[pair.sentence2], dtype=np.float64)
        d = a - b
        l1 = float(np.abs(d).sum())
        l2 = float(np.linalg.norm(d))
        if l2 == 0.0:
            continue
        dimension = d.shape[0]
        # 1.0 = perfectly diffuse, -> 1/sqrt(dim) = single-dimension spike.
        concentration = l1 / (np.sqrt(dimension) * l2)
        magnitudes = np.sort(np.abs(d))[::-1]
        top_mass = float(magnitudes[:_TOP_MASS_DIMENSIONS].sum() / magnitudes.sum())
        key = (
            trace.get("membership_edge_type") or "(none)",
            bool(trace["label"]),
        )
        rows[key].append((float(np.dot(a, b)), l1, l2, concentration, top_mass))

    print(
        f"{'edge type':24s} {'label':>5s} {'n':>5s} {'cosine':>8s} "
        f"{'L2':>7s} {'concen':>7s} {'top32%':>7s}"
    )
    for (edge_type, label), values in sorted(rows.items()):
        data = np.asarray(values)
        print(
            f"{edge_type:24s} {str(label):>5s} {len(values):5d} "
            f"{np.median(data[:, 0]):8.3f} {np.median(data[:, 2]):7.3f} "
            f"{np.median(data[:, 3]):7.3f} {np.median(data[:, 4]):7.3f}"
        )

    # Separability probe: member_scale vs (none), concentration AUC.
    def auc(positive: list[float], negative: list[float]) -> float:
        if not positive or not negative:
            return float("nan")
        wins = sum((1.0 if p < n else 0.5 if p == n else 0.0) for p in positive for n in negative)
        return wins / (len(positive) * len(negative))

    for target in ("member_scale", "member_temporal", "member_of"):
        target_values = [
            row[3]
            for (edge_type, _label), values in rows.items()
            if edge_type == target
            for row in values
        ]
        none_values = [
            row[3]
            for (edge_type, _label), values in rows.items()
            if edge_type == "(none)"
            for row in values
        ]
        print(
            f"AUC(concentration lower for {target} vs none): {auc(target_values, none_values):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
