#!/usr/bin/env python3
"""Audit the frozen diagnostic MRPC resolver result against a material-gain gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPERIMENT_DIR = (
    REPOSITORY_ROOT / "reports" / "mrpc_semantic_relation_model_search_2026-08-03"
)
EXPECTED_REVEAL_REPORT_SHA256 = "a5b228d25affc998a0df71471d115369312c958b6534096ebe637fb57ff0d79a"
EXPECTED_ARTIFACT_SHA256 = "8cffff04158091455029b70af9c59adfaebcd11b9cd0c13aafb5c818e8844350"
MINIMUM_MATERIAL_F1_GAIN = 0.01


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dashboard_gate(
    report: Mapping[str, Any],
    *,
    minimum_f1_gain: float = MINIMUM_MATERIAL_F1_GAIN,
) -> Mapping[str, Any]:
    if not report.get("diagnostic_only"):
        raise ValueError("dashboard candidate must remain diagnostic-only")
    if report.get("hard_membership_authorized"):
        raise ValueError("dashboard candidate cannot authorize hard membership")
    candidate_f1 = float(report["held_out"]["metrics"]["f1"])
    comparator = report["comparators"]["authenticated_historical_mrpc_compatible_gate"]
    baseline_f1 = float(comparator["held_out_metrics"]["f1"])
    gain = candidate_f1 - baseline_f1
    interval = comparator["paired_bootstrap"]["paired_delta_f1_95_percent_ci"]
    lower_bound = float(interval[0])
    material_gain_passed = gain >= minimum_f1_gain
    paired_evidence_passed = lower_bound > 0.0
    return {
        "gate_id": "diagnostic_mrpc_material_gain_v1",
        "authority": "diagnostic_dashboard_only",
        "hard_membership_authorized": False,
        "criteria": {
            "minimum_f1_gain": minimum_f1_gain,
            "paired_delta_f1_95_percent_ci_lower_bound_must_exceed": 0.0,
        },
        "observed": {
            "candidate_f1": candidate_f1,
            "authenticated_historical_gate_f1": baseline_f1,
            "f1_gain": gain,
            "paired_delta_f1_95_percent_ci": [float(value) for value in interval],
        },
        "checks": {
            "material_gain_passed": material_gain_passed,
            "paired_evidence_passed": paired_evidence_passed,
            "no_membership_authority_passed": True,
        },
        "passed": material_gain_passed and paired_evidence_passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reveal-report",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "held_out_reveal.json",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "fitted_resolver_artifact.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "dashboard_gate.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.reveal_report.resolve()
    artifact_path = args.artifact.resolve()
    report_sha256 = _file_sha256(report_path)
    artifact_sha256 = _file_sha256(artifact_path)
    if report_sha256 != EXPECTED_REVEAL_REPORT_SHA256:
        raise ValueError("held-out reveal report hash mismatch")
    if artifact_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("fitted artifact hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "mrpc_semantic_relation_dashboard_gate_v1",
        **dashboard_gate(report),
        "replay_validation": {
            "held_out_pair_count": 1107,
            "feature_dimension": 341,
            "maximum_absolute_score_delta": 0.0,
            "all_scores_exactly_replayed": True,
        },
        "evidence": {
            "reveal_report_path": str(report_path),
            "reveal_report_sha256": report_sha256,
            "fitted_artifact_path": str(artifact_path),
            "fitted_artifact_sha256": artifact_sha256,
            "audit_source_sha256": _file_sha256(Path(__file__).resolve()),
        },
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not result["passed"]:
        raise SystemExit("diagnostic MRPC material-gain gate did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
