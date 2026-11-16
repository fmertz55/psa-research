#!/usr/bin/env python3
"""Run the typed candidate/verifier calibration harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_servers.mcp_lib.typed_calibration import (
    load_and_run_typed_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure typed candidate Recall@32/Recall@8 separately from "
            "relation-verifier precision and coverage."
        )
    )
    parser.add_argument("corpus", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; stdout is always emitted.",
    )
    args = parser.parse_args()

    report = load_and_run_typed_calibration(args.corpus)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
