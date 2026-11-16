#!/usr/bin/env python3
"""Run the one-hour, resumable Qwen typed-routing field test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    DEFAULT_CASES_PER_RELATION_PAIR,
    DEFAULT_MAX_QWEN_SECONDS,
    DEFAULT_MODEL_ID,
    DEFAULT_TYPED_CORPUS_PROFILE,
    TYPED_CORPUS_PROFILES,
    QwenBudgetExhausted,
    QwenFieldTestError,
    run_qwen_typed_field_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MRPC plus a 4x100 typed microcorpus against a Qwen3 "
            "Embedding 8B TEI endpoint. Completed batches resume from SQLite."
        )
    )
    parser.add_argument(
        "--tei-url",
        required=True,
        help="Base URL of a Qwen3-Embedding-8B TEI endpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "temp" / "qwen_typed_field_test",
    )
    parser.add_argument(
        "--mrpc-dir",
        type=Path,
        default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--cases-per-relation-pair",
        type=int,
        default=DEFAULT_CASES_PER_RELATION_PAIR,
    )
    parser.add_argument(
        "--typed-corpus-profile",
        choices=TYPED_CORPUS_PROFILES,
        default=DEFAULT_TYPED_CORPUS_PROFILE,
        help=(
            "Frozen typed corpus construction profile. The hostile profile "
            "plants a wrong-target lexical collision for every source."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-qwen-seconds",
        type=float,
        default=DEFAULT_MAX_QWEN_SECONDS,
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--stability-sample-size", type=int, default=32)
    parser.add_argument(
        "--auth-token-env",
        help=(
            "Name of an environment variable containing a bearer token. "
            "When omitted, use the credential saved by `hf auth login`. "
            "The token value is never written to reports or caches."
        ),
    )
    parser.add_argument(
        "--served-model-alias",
        action="append",
        default=[],
        help=(
            "Exact TEI /info model ID alias. An alias is accepted only with "
            "a matching declared repository and a non-empty revision."
        ),
    )
    parser.add_argument("--declared-model-repository")
    parser.add_argument("--declared-model-revision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_token = None
    if args.auth_token_env:
        auth_token = os.environ.get(args.auth_token_env)
        if not auth_token or not auth_token.strip():
            print(
                f"Field test failed: environment variable "
                f"{args.auth_token_env!r} is missing or empty",
                file=sys.stderr,
            )
            return 1
    else:
        try:
            from huggingface_hub import get_token
        except ImportError:
            pass
        else:
            auth_token = get_token()
    try:
        report = run_qwen_typed_field_test(
            tei_url=args.tei_url,
            output_dir=args.output_dir.resolve(),
            mrpc_dir=args.mrpc_dir.resolve(),
            model_id=args.model_id,
            cases_per_relation_pair=args.cases_per_relation_pair,
            typed_corpus_profile=args.typed_corpus_profile,
            batch_size=args.batch_size,
            max_qwen_seconds=args.max_qwen_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            stability_sample_size=args.stability_sample_size,
            auth_token=auth_token,
            served_model_aliases=args.served_model_alias,
            declared_model_repository=args.declared_model_repository,
            declared_model_revision=args.declared_model_revision,
        )
    except QwenBudgetExhausted as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"Resume with the same --output-dir: {args.output_dir.resolve()}",
            file=sys.stderr,
        )
        return 2
    except QwenFieldTestError as exc:
        print(f"Field test failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_status": report["run_status"],
                "report_id": report["report_id"],
                "report_path": str(args.output_dir.resolve() / "field_test_report.json"),
                "qwen_wall_seconds": report["resource_usage"]["qwen_wall_seconds"],
                "rollout_gate_open": report["rollout_gate"]["open"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
