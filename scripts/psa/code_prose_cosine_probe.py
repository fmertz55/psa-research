#!/usr/bin/env python3
"""Qwen-only code-to-prose cosine probe.

This pilot deliberately skips PSA routing, clustering, graph persistence, and
the WordNet paraphrase gate. It embeds a small set of prose claims and exact
repository code spans as raw documents, computes the complete cosine matrix,
and reports candidate alignments at an exploratory threshold.

Usage:
    python3 scripts/research/psa/code_prose_cosine_probe.py --validate-only
    python3 scripts/research/psa/code_prose_cosine_probe.py \
        --endpoint http://embedding.aiidea.local:11434 \
        --threshold 0.84
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tokenize
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = (
    REPO_ROOT / "scripts" / "research" / "test_rigs" / "code_prose_cosine" / "aiidea_v1.json"
)
DEFAULT_ENDPOINT = os.getenv("EMBEDDING_SERVICE_URL") or "http://embedding.aiidea.local:11434"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    source_path: str
    source_line: int
    status: str = "implemented"


@dataclass(frozen=True)
class CodeUnit:
    code_id: str
    source_path: str
    start_line: int
    end_line: int
    symbol: str
    language: str
    text: str
    sha256: str


@dataclass(frozen=True)
class PairLabel:
    claim_id: str
    code_id: str
    label: str
    note: str = ""


def _safe_repo_path(relative_path: str) -> Path:
    candidate = (REPO_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {relative_path}") from exc
    if not candidate.is_file():
        raise ValueError(f"Repository file does not exist: {relative_path}")
    return candidate


def extract_code_span(
    source_path: str,
    start_line: int,
    end_line: int,
) -> tuple[str, str]:
    """Return an exact one-indexed inclusive source span and its SHA-256."""
    if start_line < 1 or end_line < start_line:
        raise ValueError(f"Invalid line span for {source_path}: {start_line}-{end_line}")
    path = _safe_repo_path(source_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if end_line > len(lines):
        raise ValueError(
            f"Line span {start_line}-{end_line} exceeds {source_path} ({len(lines)} lines)"
        )
    text = "\n".join(lines[start_line - 1 : end_line]).rstrip()
    if not text.strip():
        raise ValueError(f"Empty source span for {source_path}:{start_line}-{end_line}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest


def resolve_python_symbol_span(source_path: str, symbol: str) -> tuple[int, int]:
    """Resolve a complete callable/class without relying on drifting line numbers."""
    tree = ast.parse(_safe_repo_path(source_path).read_text(encoding="utf-8"))
    matches: list[tuple[int, int]] = []

    def visit(node: ast.AST, scope: tuple[str, ...] = ()) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = (*scope, node.name)
            if ".".join(scope) == symbol:
                matches.append((node.lineno, node.end_lineno or node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree)
    if len(matches) != 1:
        raise ValueError(f"Python symbol must resolve exactly once: {source_path}:{symbol}")
    return matches[0]


def strip_python_comments(text: str) -> str:
    """Remove COMMENT tokens while preserving strings and executable syntax."""
    tokens: list[tokenize.TokenInfo] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type != tokenize.COMMENT:
                tokens.append(token)
    except tokenize.TokenError as exc:
        # Repository spans may intentionally end inside an enclosing function or
        # collection. The tokenizer still yields every usable token before EOF.
        if "eof" not in str(exc).lower() or not tokens:
            raise ValueError(f"Could not tokenize Python source span: {exc}") from exc
    except IndentationError as exc:
        raise ValueError(f"Could not tokenize Python source span: {exc}") from exc
    cleaned = tokenize.untokenize(tokens)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
    if not cleaned:
        raise ValueError("Comment stripping removed the entire source span")
    return cleaned


def _python_module_name(source_path: str) -> str:
    """Convert a repository Python path to its import-style module name."""
    parts = list(Path(source_path).with_suffix("").parts)
    if parts and parts[0] == "app":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _python_scope_header(item: CodeUnit) -> str:
    """Build compact module/class/callable context from the repository AST."""
    source = _safe_repo_path(item.source_path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=item.source_path)
    except SyntaxError as exc:
        raise ValueError(f"Could not parse scoped Python source: {item.source_path}") from exc

    symbol_parts = [part for part in item.symbol.split(".") if part]
    expected_class = symbol_parts[-2] if len(symbol_parts) >= 2 else None
    expected_leaf = symbol_parts[-1] if symbol_parts else ""
    containing_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and node.lineno <= item.start_line <= (node.end_lineno or node.lineno)
    ]
    class_node = next(
        (node for node in containing_classes if node.name == expected_class),
        min(
            containing_classes,
            key=lambda node: (node.end_lineno or node.lineno) - node.lineno,
            default=None,
        ),
    )
    callable_node = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == expected_leaf
            and node.lineno <= item.start_line <= (node.end_lineno or node.lineno)
        ),
        None,
    )

    lines = [
        f"Language: {item.language.lower()}",
        f"Module: {_python_module_name(item.source_path)}",
    ]
    if class_node is not None:
        bases = ", ".join(ast.unparse(base) for base in class_node.bases)
        declaration = f"class {class_node.name}({bases})" if bases else f"class {class_node.name}"
        lines.append(f"Class: {declaration}")
    else:
        lines.append("Class: module scope")

    if callable_node is not None:
        async_prefix = "async " if isinstance(callable_node, ast.AsyncFunctionDef) else ""
        declaration = f"{async_prefix}def {callable_node.name}({ast.unparse(callable_node.args)})"
        if callable_node.returns is not None:
            declaration += f" -> {ast.unparse(callable_node.returns)}"
        lines.append(f"Callable: {declaration}")
    else:
        lines.append(f"Symbol: {item.symbol or expected_leaf}")
    return "\n".join(lines)


def code_embedding_text(item: CodeUnit, code_mode: str) -> str:
    if code_mode == "raw":
        return item.text
    if code_mode == "comments-stripped":
        if item.language.lower() != "python":
            raise ValueError(
                f"comments-stripped mode does not support {item.language}: {item.code_id}"
            )
        return strip_python_comments(item.text)
    if code_mode == "scoped":
        if item.language.lower() != "python":
            raise ValueError(f"scoped mode does not support {item.language}: {item.code_id}")
        return f"{_python_scope_header(item)}\n\n{item.text}"
    raise ValueError(f"Unsupported code input mode: {code_mode}")


def load_corpus(path: Path) -> tuple[dict[str, Any], list[Claim], list[CodeUnit], list[PairLabel]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = [
        Claim(
            claim_id=str(item["id"]),
            text=str(item["text"]).strip(),
            source_path=str(item["source_path"]),
            source_line=int(item["source_line"]),
            status=str(item.get("status") or "implemented"),
        )
        for item in data.get("claims", [])
    ]

    code_units: list[CodeUnit] = []
    for item in data.get("code_units", []):
        source_path = str(item["source_path"])
        selector = item.get("selector", "line_span")
        if selector == "python_symbol" and item.get("language") == "python":
            start_line, end_line = resolve_python_symbol_span(source_path, str(item["symbol"]))
        elif selector == "line_span":
            start_line = int(item["start_line"])
            end_line = int(item["end_line"])
        else:
            raise ValueError(f"Unsupported source selector: {selector}")
        text, digest = extract_code_span(source_path, start_line, end_line)
        code_units.append(
            CodeUnit(
                code_id=str(item["id"]),
                source_path=source_path,
                start_line=start_line,
                end_line=end_line,
                symbol=str(item.get("symbol") or ""),
                language=str(item.get("language") or "unknown"),
                text=text,
                sha256=digest,
            )
        )

    labels = [
        PairLabel(
            claim_id=str(item["claim_id"]),
            code_id=str(item["code_id"]),
            label=str(item["label"]),
            note=str(item.get("note") or ""),
        )
        for item in data.get("labeled_pairs", [])
    ]
    validate_corpus(claims, code_units, labels)
    return data, claims, code_units, labels


def validate_corpus(
    claims: Sequence[Claim],
    code_units: Sequence[CodeUnit],
    labels: Sequence[PairLabel],
) -> None:
    if not claims:
        raise ValueError("Corpus must contain at least one prose claim")
    if not code_units:
        raise ValueError("Corpus must contain at least one code unit")

    claim_ids = [item.claim_id for item in claims]
    code_ids = [item.code_id for item in code_units]
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("Claim IDs must be unique")
    if len(set(code_ids)) != len(code_ids):
        raise ValueError("Code unit IDs must be unique")
    if any(not item.text for item in claims):
        raise ValueError("Claim text must not be empty")

    known_claims = set(claim_ids)
    known_code = set(code_ids)
    seen_pairs: set[tuple[str, str]] = set()
    for label in labels:
        pair = (label.claim_id, label.code_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate labeled pair: {pair}")
        seen_pairs.add(pair)
        if label.claim_id not in known_claims:
            raise ValueError(f"Unknown claim in labeled pair: {label.claim_id}")
        if label.code_id not in known_code:
            raise ValueError(f"Unknown code unit in labeled pair: {label.code_id}")
        if label.label not in {"match", "non_match"}:
            raise ValueError(f"Unsupported pair label: {label.label}")


def request_headers(auth_token: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    return headers


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    auth_token: str | None = None,
) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers(auth_token),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TEI returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach TEI endpoint {url}: {exc.reason}") from exc


def _get_json(url: str, timeout_seconds: int, auth_token: str | None = None) -> Any:
    request = urllib.request.Request(url, headers=request_headers(auth_token))
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # Model info is useful but not required by every TEI build.
        return {"unavailable": True, "error": str(exc)}


def normalize_vector(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0.0:
        raise ValueError("Embedding vector has zero norm")
    return [float(value) / norm for value in vector]


def embed_documents(
    endpoint: str,
    texts: Sequence[str],
    *,
    batch_size: int,
    timeout_seconds: int,
    expected_dimension: int,
    auth_token: str | None = None,
) -> list[list[float]]:
    """Embed raw document inputs through TEI without a query instruction."""
    endpoint = endpoint.rstrip("/")
    rows: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        data = _post_json(
            f"{endpoint}/embed",
            {"inputs": batch},
            timeout_seconds,
            auth_token,
        )
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError(
                f"TEI returned an unexpected embedding shape for batch starting at {start}"
            )
        for offset, row in enumerate(data):
            if not isinstance(row, list) or not row:
                raise RuntimeError(f"TEI returned an empty vector for item {start + offset}")
            if expected_dimension and len(row) != expected_dimension:
                raise RuntimeError(
                    f"Expected {expected_dimension} dimensions, got {len(row)} "
                    f"for item {start + offset}"
                )
            rows.append(normalize_vector(row))
    return rows


def cosine_matrix(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> list[list[float]]:
    if not left or not right:
        return []
    dimension = len(left[0])
    if dimension == 0 or any(len(row) != dimension for row in left):
        raise ValueError("Left embedding rows must share a non-zero dimension")
    if any(len(row) != dimension for row in right):
        raise ValueError("Left and right embeddings must share a dimension")
    return [
        [sum(a * b for a, b in zip(left_row, right_row, strict=True)) for right_row in right]
        for left_row in left
    ]


def rank_claims(
    claims: Sequence[Claim],
    code_units: Sequence[CodeUnit],
    matrix: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for claim, row in zip(claims, matrix, strict=True):
        order = sorted(range(len(code_units)), key=lambda index: row[index], reverse=True)
        ranked.append(
            {
                "claim_id": claim.claim_id,
                "ranking": [
                    {
                        "rank": rank,
                        "code_id": code_units[index].code_id,
                        "score": row[index],
                    }
                    for rank, index in enumerate(order, start=1)
                ],
            }
        )
    return ranked


def evaluate_labeled_pairs(
    labels: Sequence[PairLabel],
    claim_index: dict[str, int],
    code_index: dict[str, int],
    matrix: Sequence[Sequence[float]],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    rows: list[dict[str, Any]] = []
    for item in labels:
        score = matrix[claim_index[item.claim_id]][code_index[item.code_id]]
        predicted_match = score >= threshold
        expected_match = item.label == "match"
        if expected_match and predicted_match:
            outcome = "tp"
        elif expected_match:
            outcome = "fn"
        elif predicted_match:
            outcome = "fp"
        else:
            outcome = "tn"
        counts[outcome] += 1
        rows.append(
            {
                "claim_id": item.claim_id,
                "code_id": item.code_id,
                "expected": item.label,
                "predicted": "match" if predicted_match else "non_match",
                "score": score,
                "outcome": outcome,
                "note": item.note,
            }
        )

    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    accuracy_denominator = sum(counts.values())
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (counts["tp"] + counts["tn"]) / accuracy_denominator if accuracy_denominator else 0.0
    metrics = {
        **counts,
        "labeled_pair_count": accuracy_denominator,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
    return metrics, rows


def retrieval_metrics(
    labels: Sequence[PairLabel],
    rankings: Sequence[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    positives: dict[str, set[str]] = {}
    for item in labels:
        if item.label == "match":
            positives.setdefault(item.claim_id, set()).add(item.code_id)

    ranking_by_claim = {item["claim_id"]: item["ranking"] for item in rankings}
    reciprocal_ranks: list[float] = []
    hit_at_1 = 0
    hit_at_k = 0
    for claim_id, expected_ids in positives.items():
        ranking = ranking_by_claim[claim_id]
        expected_ranks = [item["rank"] for item in ranking if item["code_id"] in expected_ids]
        if not expected_ranks:
            reciprocal_ranks.append(0.0)
            continue
        best_rank = min(expected_ranks)
        reciprocal_ranks.append(1.0 / best_rank)
        hit_at_1 += int(best_rank == 1)
        hit_at_k += int(best_rank <= top_k)

    count = len(positives)
    return {
        "claims_with_positive_labels": count,
        "hit_at_1": hit_at_1 / count if count else 0.0,
        f"hit_at_{top_k}": hit_at_k / count if count else 0.0,
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _write_matrix_csv(
    path: Path,
    claims: Sequence[Claim],
    code_units: Sequence[CodeUnit],
    matrix: Sequence[Sequence[float]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", *[item.code_id for item in code_units]])
        for claim, row in zip(claims, matrix, strict=True):
            writer.writerow([claim.claim_id, *[f"{score:.8f}" for score in row]])


def _score(value: float) -> str:
    return f"{value:.4f}"


def _write_summary_markdown(
    path: Path,
    *,
    threshold: float,
    code_mode: str,
    top_k: int,
    claims: Sequence[Claim],
    code_units: Sequence[CodeUnit],
    rankings: Sequence[dict[str, Any]],
    metrics: dict[str, Any],
    retrieval: dict[str, Any],
    pair_rows: Sequence[dict[str, Any]],
) -> None:
    claim_by_id = {item.claim_id: item for item in claims}
    code_by_id = {item.code_id: item for item in code_units}
    lines = [
        "# Qwen Code-to-Prose Cosine Pilot",
        "",
        f"Exploratory cosine threshold: `{threshold:.4f}`",
        "",
        f"Code input mode: `{code_mode}`",
        "",
        "This run uses raw document embeddings only. Threshold crossings are candidates, not proven paraphrases.",
        "",
        "## Labeled-pair metrics",
        "",
        f"- Precision: `{metrics['precision']:.4f}`",
        f"- Recall: `{metrics['recall']:.4f}`",
        f"- F1: `{metrics['f1']:.4f}`",
        f"- Accuracy: `{metrics['accuracy']:.4f}`",
        f"- Confusion counts: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}",
        "",
        "## Retrieval metrics",
        "",
        f"- Hit@1: `{retrieval['hit_at_1']:.4f}`",
        f"- Hit@{top_k}: `{retrieval[f'hit_at_{top_k}']:.4f}`",
        f"- Mean reciprocal rank: `{retrieval['mean_reciprocal_rank']:.4f}`",
        "",
        f"## Top {top_k} code spans per claim",
        "",
    ]

    for result in rankings:
        claim = claim_by_id[result["claim_id"]]
        lines.extend(
            [
                f"### `{claim.claim_id}` ({claim.status})",
                "",
                claim.text,
                "",
            ]
        )
        for item in result["ranking"][:top_k]:
            code = code_by_id[item["code_id"]]
            marker = " **≥ threshold**" if item["score"] >= threshold else ""
            lines.append(
                f"- {_score(item['score'])} — `{code.code_id}` — "
                f"`{code.source_path}:{code.start_line}` ({code.symbol}){marker}"
            )
        lines.append("")

    lines.extend(["## Labeled pair outcomes", ""])
    for item in sorted(pair_rows, key=lambda row: row["score"], reverse=True):
        lines.append(
            f"- `{item['outcome'].upper()}` {_score(item['score'])}: "
            f"`{item['claim_id']}` ↔ `{item['code_id']}` "
            f"(expected {item['expected']})"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _serializable_claim(item: Claim) -> dict[str, Any]:
    return {
        "id": item.claim_id,
        "text": item.text,
        "source_path": item.source_path,
        "source_line": item.source_line,
        "status": item.status,
    }


def _serializable_code(item: CodeUnit) -> dict[str, Any]:
    return {
        "id": item.code_id,
        "source_path": item.source_path,
        "start_line": item.start_line,
        "end_line": item.end_line,
        "symbol": item.symbol,
        "language": item.language,
        "sha256": item.sha256,
        "character_count": len(item.text),
    }


def _serializable_embedded_code(item: CodeUnit, embedded_text: str) -> dict[str, Any]:
    result = _serializable_code(item)
    result["embedded_sha256"] = hashlib.sha256(embedded_text.encode("utf-8")).hexdigest()
    result["embedded_character_count"] = len(embedded_text)
    return result


def _default_output_dir(code_mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "reports" / "code_prose_cosine" / f"{timestamp}-{code_mode}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--threshold", type=float, default=0.84)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--expected-dimension", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--auth-token-env",
        help=(
            "Environment variable containing a Bearer token. When omitted, "
            "use the credential saved by `hf auth login`; the value is never "
            "reported."
        ),
    )
    parser.add_argument(
        "--code-mode",
        choices=("raw", "comments-stripped", "scoped"),
        default="raw",
        help="Embed exact, comment-stripped, or structurally scoped code spans",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate corpus paths/spans without contacting Qwen",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("--top-k and --batch-size must be positive")

    corpus_path = args.corpus.resolve()
    data, claims, code_units, labels = load_corpus(corpus_path)
    print(
        f"Validated {len(claims)} claims, {len(code_units)} code spans, "
        f"and {len(labels)} labeled pairs from {corpus_path}"
    )
    if args.validate_only:
        for item in code_units:
            print(
                f"  {item.code_id}: {item.source_path}:{item.start_line}-{item.end_line} "
                f"sha256={item.sha256[:12]}"
            )
        return 0

    endpoint = args.endpoint.rstrip("/")
    auth_token = None
    if args.auth_token_env:
        auth_token = os.getenv(args.auth_token_env)
        if not auth_token:
            raise ValueError(
                f"Authentication token environment variable is empty: {args.auth_token_env}"
            )
    else:
        try:
            from huggingface_hub import get_token
        except ImportError:
            pass
        else:
            auth_token = get_token()
    print(
        f"Embedding {len(claims) + len(code_units)} documents via {endpoint} "
        f"(code_mode={args.code_mode})"
    )
    model_info = _get_json(
        f"{endpoint}/info",
        min(args.timeout_seconds, 15),
        auth_token,
    )
    code_texts = [code_embedding_text(item, args.code_mode) for item in code_units]
    texts = [item.text for item in claims] + code_texts
    embeddings = embed_documents(
        endpoint,
        texts,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        expected_dimension=args.expected_dimension,
        auth_token=auth_token,
    )
    claim_embeddings = embeddings[: len(claims)]
    code_embeddings = embeddings[len(claims) :]
    matrix = cosine_matrix(claim_embeddings, code_embeddings)
    rankings = rank_claims(claims, code_units, matrix)
    claim_index = {item.claim_id: index for index, item in enumerate(claims)}
    code_index = {item.code_id: index for index, item in enumerate(code_units)}
    metrics, pair_rows = evaluate_labeled_pairs(
        labels,
        claim_index,
        code_index,
        matrix,
        args.threshold,
    )
    retrieval = retrieval_metrics(labels, rankings, args.top_k)

    output_dir = (args.output_dir or _default_output_dir(args.code_mode)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "code_prose_cosine_pilot.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "corpus": {
            "path": _display_path(corpus_path),
            "name": data.get("name"),
            "version": data.get("version"),
        },
        "embedding": {
            "endpoint": endpoint,
            "model_info": model_info,
            "dimension": len(embeddings[0]) if embeddings else None,
            "input_type": "raw_document",
            "query_instruction_used": False,
            "code_mode": args.code_mode,
            "authentication": "bearer" if auth_token else "none",
        },
        "threshold": args.threshold,
        "claims": [_serializable_claim(item) for item in claims],
        "code_units": [
            _serializable_embedded_code(item, embedded_text)
            for item, embedded_text in zip(code_units, code_texts, strict=True)
        ],
        "matrix": matrix,
        "rankings": rankings,
        "labeled_pair_metrics": metrics,
        "retrieval_metrics": retrieval,
        "labeled_pair_results": pair_rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_matrix_csv(output_dir / "matrix.csv", claims, code_units, matrix)
    _write_summary_markdown(
        output_dir / "summary.md",
        threshold=args.threshold,
        code_mode=args.code_mode,
        top_k=args.top_k,
        claims=claims,
        code_units=code_units,
        rankings=rankings,
        metrics=metrics,
        retrieval=retrieval,
        pair_rows=pair_rows,
    )

    print(
        "Labeled pairs: "
        f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
        f"f1={metrics['f1']:.3f} accuracy={metrics['accuracy']:.3f}"
    )
    print(
        f"Retrieval: hit@1={retrieval['hit_at_1']:.3f} "
        f"hit@{args.top_k}={retrieval[f'hit_at_{args.top_k}']:.3f} "
        f"MRR={retrieval['mean_reciprocal_rank']:.3f}"
    )
    print(f"Wrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
