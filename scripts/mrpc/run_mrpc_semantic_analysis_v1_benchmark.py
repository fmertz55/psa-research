#!/usr/bin/env python3
"""Compile deterministic semantic evidence and evaluate the full MRPC test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from mcp_servers.mcp_lib.grammatical_operator_inventory import (  # noqa: E402
    analyze_grammatical_operator_inventory,
)
from mcp_servers.mcp_lib.semantic_analysis import (  # noqa: E402
    ResourceIdentity,
    SemanticAnalysisArtifact,
)
from mcp_servers.mcp_lib.semantic_content import canonical_sha256  # noqa: E402
from mcp_servers.mcp_lib.semantic_evidence_compiler import (  # noqa: E402
    SEMANTIC_EVIDENCE_COMPILER_REVISION,
    CompilerResources,
    DeterministicSemanticEvidenceCompiler,
    OccurrenceSenseResolver,
    VerbNetValenceIndex,
)
from mcp_servers.mcp_lib.semantic_proof import (  # noqa: E402
    SemanticLicenseRegistry,
)
from mcp_servers.mcp_lib.semantic_resolution import (  # noqa: E402
    build_meaning_symbol_artifact,
    prove_invariant_semantic_relation,
)

SCHEMA_VERSION = "mrpc_semantic_analysis_engineering_benchmark_v3"
BLIND_SPLIT_SEED = "mrpc-proposition-graph-blind-sentence-components-v1"
CALIBRATION_PERCENT = 35
DEFAULT_CORPUS = REPOSITORY_ROOT / "app" / "utils" / "msr_paraphrase_data.txt"
DEFAULT_TEST = REPOSITORY_ROOT / "app" / "utils" / "msr_paraphrase_test.txt"
DEFAULT_WORDNET_DATA = REPOSITORY_ROOT / "temp" / "wordnet"
DEFAULT_VERBNET_DIR = REPOSITORY_ROOT / "temp" / "nltk_data" / "corpora" / "verbnet3"
DEFAULT_VERBNET_ZIP = REPOSITORY_ROOT / "temp" / "nltk_data" / "corpora" / "verbnet3.zip"
DEFAULT_REPORT = (
    REPOSITORY_ROOT / "reports" / "mrpc_semantic_analysis" / "semantic_proof_report.json"
)
DEFAULT_CURRENT_PIPELINE_REPORT = (
    REPOSITORY_ROOT
    / "reports"
    / "mrpc_comparison_triage_2026-08-05"
    / "mrpc_current_pipeline_report.json"
)


class BenchmarkError(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tree_sha256(path: Path) -> str:
    records = [
        {"path": str(value.relative_to(path)), "sha256": _file_sha256(value)}
        for value in sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    ]
    return canonical_sha256(records)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sentence_registry(corpus_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames is None or not {"Sentence ID", "String"} <= set(reader.fieldnames):
            raise BenchmarkError("MRPC sentence registry schema is invalid")
        for line_number, row in enumerate(reader, start=2):
            sentence_id = str(row.get("Sentence ID") or "").strip()
            text = str(row.get("String") or "")
            if not sentence_id or not text.strip():
                raise BenchmarkError(f"malformed MRPC corpus row {line_number}")
            previous = result.get(sentence_id)
            if previous is not None and previous != text:
                raise BenchmarkError(f"MRPC sentence ID {sentence_id!r} is not unique")
            result[sentence_id] = text
    return result


def _label_free_pair_ids(test_path: Path) -> list[tuple[int, str, str]]:
    """Read only the ID suffix of each TSV row; the leading label is discarded."""

    pairs: list[tuple[int, str, str]] = []
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        try:
            first_index = header.index("#1 ID") - 1
            second_index = header.index("#2 ID") - 1
        except ValueError as exc:
            raise BenchmarkError("MRPC test ID schema is invalid") from exc
        if not header or header[0] != "Quality":
            raise BenchmarkError("MRPC test label must be the isolated leading column")
        for pair_id, line in enumerate(handle, start=1):
            _discarded_prefix, separator, label_free_suffix = line.rstrip("\r\n").partition("\t")
            if not separator:
                raise BenchmarkError(f"malformed MRPC test row {pair_id + 1}")
            fields = label_free_suffix.split("\t")
            try:
                first_id = fields[first_index].strip()
                second_id = fields[second_index].strip()
            except IndexError as exc:
                raise BenchmarkError(f"malformed MRPC test row {pair_id + 1}") from exc
            if not first_id or not second_id:
                raise BenchmarkError(f"missing MRPC sentence ID on row {pair_id + 1}")
            pairs.append((pair_id, first_id, second_id))
    if not pairs:
        raise BenchmarkError("MRPC test split is empty")
    return pairs


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            self.parent[high] = low


def _component_split(pair_texts: Sequence[tuple[int, str, str]]) -> dict[int, str]:
    graph = _UnionFind()
    normalized: dict[int, tuple[str, str]] = {}
    for pair_id, left, right in pair_texts:
        values = (" ".join(left.casefold().split()), " ".join(right.casefold().split()))
        graph.union(*values)
        normalized[pair_id] = values
    members: dict[str, list[str]] = defaultdict(list)
    for left, right in normalized.values():
        members[graph.find(left)].extend((left, right))
    component_splits: dict[str, str] = {}
    for root, values in members.items():
        component_id = canonical_sha256(
            {"seed": BLIND_SPLIT_SEED, "sentences": sorted(set(values))}
        )
        bucket = int(component_id[:8], 16) % 100
        component_splits[root] = "calibration" if bucket < CALIBRATION_PERCENT else "held_out"
    return {
        pair_id: component_splits[graph.find(values[0])] for pair_id, values in normalized.items()
    }


def _resource_identities() -> tuple[CompilerResources, dict[str, Any]]:
    import en_core_web_sm

    compiler_path = (
        REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "semantic_evidence_compiler.py"
    )
    analysis_path = REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "semantic_analysis.py"
    proof_path = REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "semantic_proof.py"
    resolution_path = REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "semantic_resolution.py"
    operator_path = (
        REPOSITORY_ROOT / "app" / "mcp_servers" / "mcp_lib" / "grammatical_operator_inventory.py"
    )
    model_root = Path(en_core_web_sm.__file__).resolve().parent / "en_core_web_sm-3.8.0"
    resources = CompilerResources(
        analyzer=ResourceIdentity(
            "analyzer:deterministic-semantic-evidence",
            SEMANTIC_EVIDENCE_COMPILER_REVISION,
            _file_sha256(compiler_path),
        ),
        tokenizer=ResourceIdentity(
            "parser:spacy-en-core-web-sm", "v3.8.0", _tree_sha256(model_root)
        ),
        wordnet=ResourceIdentity(
            "lexicon:oewn", "v2024", _file_sha256(DEFAULT_WORDNET_DATA / "wn.db")
        ),
        verbnet=ResourceIdentity("lexicon:verbnet", "v3.3", _file_sha256(DEFAULT_VERBNET_ZIP)),
        operator_analyzer=ResourceIdentity(
            "analyzer:grammatical-operator", "v1", _file_sha256(operator_path)
        ),
    )
    source_hashes = {
        "benchmark": _file_sha256(Path(__file__).resolve()),
        "compiler": _file_sha256(compiler_path),
        "semantic_analysis": _file_sha256(analysis_path),
        "semantic_proof": _file_sha256(proof_path),
        "semantic_resolution": _file_sha256(resolution_path),
        "grammatical_operator_inventory": _file_sha256(operator_path),
    }
    return resources, source_hashes


def _operator_records(doc: Any) -> list[dict[str, Any]]:
    sentence_by_position: dict[int, int] = {}
    for sentence_index, sentence in enumerate(doc.sents):
        for token in sentence:
            sentence_by_position[token.i] = sentence_index
    return [
        {
            "position": token.i,
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dependency": token.dep_,
            "head_position": token.head.i,
            "sentence_index": sentence_by_position.get(token.i, 0),
        }
        for token in doc
    ]


def _reason_category(reason: str) -> str:
    parts = [part for part in reason.split(":") if not part.isdigit()]
    return ":".join(parts).replace("atom:atom", "atom")


def _branch_readiness(
    artifact: SemanticAnalysisArtifact,
) -> tuple[int, tuple[str, ...]]:
    ready = 0
    reasons: set[str] = set()
    for interpretation in artifact.interpretations:
        selected = replace(
            artifact,
            selected_interpretation_id=interpretation.interpretation_id,
            selection_evidence_sha256=canonical_sha256(
                {
                    "schema_version": "mrpc-branch-readiness-v1",
                    "artifact_sha256": artifact.artifact_sha256,
                    "interpretation_id": interpretation.interpretation_id,
                }
            ),
        )
        readiness = selected.proof_readiness
        if readiness.ready:
            ready += 1
        else:
            reasons.update(readiness.reasons)
    return ready, tuple(sorted(reasons))


def _compile_evidence(args: argparse.Namespace) -> dict[str, Any]:
    sentence_by_id = _sentence_registry(args.corpus)
    pair_ids = _label_free_pair_ids(args.test)
    pair_texts: list[tuple[int, str, str]] = []
    for pair_id, first_id, second_id in pair_ids:
        try:
            pair_texts.append((pair_id, sentence_by_id[first_id], sentence_by_id[second_id]))
        except KeyError as exc:
            raise BenchmarkError(f"MRPC test references missing sentence {exc.args[0]!r}") from exc
    splits = _component_split(pair_texts)
    split_counts = Counter(splits.values())
    if split_counts != Counter({"calibration": 618, "held_out": 1107}):
        raise BenchmarkError(f"component split drifted: {dict(split_counts)}")

    import spacy
    import wn
    from nltk.stem import PorterStemmer
    from wn.morphy import Morphy

    resources, source_hashes = _resource_identities()
    wn.config.data_directory = args.wordnet_data.resolve()
    wordnet = wn.Wordnet("oewn:2024")
    if len(wordnet.synsets()) != 120_630:
        raise BenchmarkError("OEWN 2024 inventory cardinality mismatch")
    nlp = spacy.load("en_core_web_sm")
    valence_index = VerbNetValenceIndex.from_directory(
        args.verbnet_dir,
        resource=resources.verbnet,
    )
    compiler = DeterministicSemanticEvidenceCompiler(
        resources=resources,
        sense_resolver=OccurrenceSenseResolver(
            wordnet,
            Morphy(wordnet),
            PorterStemmer(),
            resource=resources.wordnet,
        ),
        valence_index=valence_index,
    )
    registry = SemanticLicenseRegistry.build(registry_id="registry:identity-only")
    unique_texts = sorted(
        {text for _pair_id, left, right in pair_texts for text in (left, right)},
        key=lambda value: (_text_sha256(value), value),
    )
    sentences: dict[str, Any] = {}
    artifacts: dict[str, SemanticAnalysisArtifact] = {}
    coverage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved_counts: Counter[str] = Counter()
    for index, (text_value, doc) in enumerate(
        zip(unique_texts, nlp.pipe(unique_texts, batch_size=64), strict=True), start=1
    ):
        text_id = _text_sha256(text_value)
        try:
            inventory = analyze_grammatical_operator_inventory(
                text_value,
                _operator_records(doc),
                parser_identity=(
                    f"{resources.tokenizer.resource_id}@{resources.tokenizer.revision}:"
                    f"{resources.tokenizer.content_sha256}"
                ),
            )
            artifact = compiler.compile(
                text=text_value,
                doc=doc,
                operator_inventory=inventory,
                enumerate_interpretations=True,
            )
            artifacts[text_id] = artifact
            meaning_symbols = build_meaning_symbol_artifact(artifact)
            ready_interpretation_count, readiness_reasons = _branch_readiness(artifact)
            interpretation = artifact.interpretations[0]
            coverage = interpretation.coverage.trace_dict()
            for field, status in coverage.items():
                coverage_counts[field][status] += 1
            for reason in readiness_reasons:
                unresolved_counts[_reason_category(reason)] += 1
            all_interpretations_ready = ready_interpretation_count == len(artifact.interpretations)
            sentences[text_id] = {
                "artifact_sha256": artifact.artifact_sha256,
                "meaning_symbol_artifact_sha256": meaning_symbols.artifact_sha256,
                "meaning_symbol_count": sum(
                    len(value.symbols) for value in meaning_symbols.interpretations
                ),
                "graph_reference_metadata": meaning_symbols.graph_reference_metadata,
                "interpretation_count": len(artifact.interpretations),
                "ready_interpretation_count": ready_interpretation_count,
                "interpretation_domain_complete": all_interpretations_ready,
                "proof_readiness": {
                    "ready": all_interpretations_ready,
                    "interpretation_id": (
                        artifact.selected_interpretation_id
                        if len(artifact.interpretations) == 1
                        else None
                    ),
                    "reasons": list(readiness_reasons),
                },
                "coverage": coverage,
                "predicate_count": len(interpretation.predicates),
                "atom_count": len(interpretation.atoms),
            }
        except Exception as exc:
            reason = f"compiler-error:{type(exc).__name__}"
            unresolved_counts[reason] += 1
            sentences[text_id] = {
                "artifact_sha256": None,
                "meaning_symbol_artifact_sha256": None,
                "meaning_symbol_count": 0,
                "graph_reference_metadata": None,
                "interpretation_count": 0,
                "ready_interpretation_count": 0,
                "interpretation_domain_complete": False,
                "proof_readiness": {"ready": False, "interpretation_id": None, "reasons": [reason]},
                "coverage": None,
                "predicate_count": 0,
                "atom_count": 0,
            }
        if index % 250 == 0 or index == len(unique_texts):
            ready_count = sum(value["proof_readiness"]["ready"] for value in sentences.values())
            print(
                f"compiled {index}/{len(unique_texts)} sentences; proof-ready={ready_count}",
                flush=True,
            )

    pair_outcomes: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    for pair_id, left, right in pair_texts:
        left_id, right_id = _text_sha256(left), _text_sha256(right)
        left_artifact, right_artifact = artifacts.get(left_id), artifacts.get(right_id)
        proof_trace = None
        proof_sha256 = None
        branch_pair_count = 0
        if left_artifact is not None and right_artifact is not None:
            proof = prove_invariant_semantic_relation(
                left_artifact,
                right_artifact,
                registry=registry,
            )
            outcome = proof.outcome.value
            proof_sha256 = proof.resolution_sha256
            branch_pair_count = proof.evaluated_branch_pair_count
            if outcome != "abstain":
                proof_trace = proof.trace_dict()
            pair_reasons = list(proof.blockers)
            if outcome == "abstain" and not pair_reasons:
                pair_reasons = [f"invariance:{proof.reason}"]
        else:
            outcome = "abstain"
            pair_reasons = sorted(
                {
                    *(
                        f"left:{value}"
                        for value in sentences[left_id]["proof_readiness"]["reasons"]
                    ),
                    *(
                        f"right:{value}"
                        for value in sentences[right_id]["proof_readiness"]["reasons"]
                    ),
                }
            )
        outcome_counts[outcome] += 1
        pair_outcomes.append(
            {
                "pair_id": pair_id,
                "left_source_text_sha256": left_id,
                "right_source_text_sha256": right_id,
                "split": splits[pair_id],
                "outcome": outcome,
                "proof_sha256": proof_sha256,
                "proof": proof_trace,
                "evaluated_branch_pair_count": branch_pair_count,
                "abstention_reasons": pair_reasons,
            }
        )

    payload = {
        "schema_version": "mrpc_semantic_analysis_compilation_v3",
        "input_files": {
            "corpus_sha256": _file_sha256(args.corpus),
            "test_sha256": _file_sha256(args.test),
        },
        "source_hashes": source_hashes,
        "resources": [
            value.trace_dict()
            for value in (
                resources.analyzer,
                resources.tokenizer,
                resources.wordnet,
                resources.verbnet,
                resources.operator_analyzer,
            )
        ],
        "registry": registry.trace_dict(),
        "split": {
            "seed": BLIND_SPLIT_SEED,
            "calibration_percent": CALIBRATION_PERCENT,
            "counts": dict(sorted(split_counts.items())),
        },
        "sentence_count": len(unique_texts),
        "pair_count": len(pair_texts),
        "sentences": sentences,
        "pair_outcomes": pair_outcomes,
        "summary": {
            "proof_ready_sentences": sum(
                value["proof_readiness"]["ready"] for value in sentences.values()
            ),
            "enumerated_domain_sentences": sum(
                value["interpretation_count"] > 1 for value in sentences.values()
            ),
            "interpretation_count": sum(
                value["interpretation_count"] for value in sentences.values()
            ),
            "meaning_symbol_count": sum(
                value["meaning_symbol_count"] for value in sentences.values()
            ),
            "coverage": {
                field: dict(sorted(counts.items()))
                for field, counts in sorted(coverage_counts.items())
            },
            "outcomes": dict(sorted(outcome_counts.items())),
            "unresolved_categories": dict(unresolved_counts.most_common()),
        },
    }
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    return payload


def _labels(test_path: Path) -> dict[int, bool]:
    result: dict[int, bool] = {}
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames is None or "Quality" not in reader.fieldnames:
            raise BenchmarkError("MRPC labels are missing")
        for pair_id, row in enumerate(reader, start=1):
            value = str(row.get("Quality") or "").strip()
            if value not in {"0", "1"}:
                raise BenchmarkError(f"invalid MRPC label on pair {pair_id}")
            result[pair_id] = value == "1"
    return result


def _metrics(records: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for expected, predicted in records:
        if expected and predicted:
            tp += 1
        elif predicted:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
    }


def _current_pipeline_majority_predictions(
    path: Path,
) -> tuple[dict[int, bool], dict[str, Any]]:
    if not path.is_file():
        raise BenchmarkError(f"current pipeline report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cache_reports = payload.get("cache_reports")
    if not isinstance(cache_reports, list) or not cache_reports:
        raise BenchmarkError("current pipeline report has no cache reports")
    order_rows: list[Mapping[str, Any]] = []
    for cache_report in cache_reports:
        if not isinstance(cache_report, Mapping):
            raise BenchmarkError("current pipeline cache report is malformed")
        orders = cache_report.get("orders")
        if not isinstance(orders, list) or not orders:
            raise BenchmarkError("current pipeline report has no order reports")
        order_rows.extend(value for value in orders if isinstance(value, Mapping))
    votes: dict[int, list[bool]] = defaultdict(list)
    labels: dict[int, bool] = {}
    for order in order_rows:
        predictions = order.get("official_test_pair_predictions")
        if not isinstance(predictions, list):
            raise BenchmarkError(
                "current pipeline report predates full official pair predictions; rerun it"
            )
        for row in predictions:
            if not isinstance(row, Mapping):
                raise BenchmarkError("current pipeline pair prediction is malformed")
            pair_id = int(row["pair_id"])
            label = bool(row["label"])
            prior = labels.setdefault(pair_id, label)
            if prior != label:
                raise BenchmarkError("current pipeline labels disagree across orders")
            votes[pair_id].append(bool(row["predicted_paraphrase"]))
    expected_vote_count = len(order_rows)
    if any(len(value) != expected_vote_count for value in votes.values()):
        raise BenchmarkError("current pipeline vote coverage differs across orders")
    majority = {pair_id: sum(values) * 2 > len(values) for pair_id, values in votes.items()}
    return majority, {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "order_count": expected_vote_count,
        "pair_count": len(majority),
    }


def evaluate(args: argparse.Namespace) -> int:
    payload = _compile_evidence(args)
    labels = _labels(args.test)
    pairs = payload["pair_outcomes"]
    if set(labels) != {int(value["pair_id"]) for value in pairs}:
        raise BenchmarkError("prepared pair IDs do not match revealed labels")
    current_predictions, current_evidence = _current_pipeline_majority_predictions(
        args.current_pipeline_report
    )
    if set(current_predictions) != set(labels):
        raise BenchmarkError("current pipeline predictions do not cover the official test")

    by_scope: dict[str, list[Mapping[str, Any]]] = {
        "official_test": list(pairs),
        "held_out": [value for value in pairs if value["split"] == "held_out"],
        "calibration": [value for value in pairs if value["split"] == "calibration"],
    }
    evaluations: dict[str, Any] = {}
    for scope, records in by_scope.items():
        outcomes = Counter(str(value["outcome"]) for value in records)
        binary = _metrics(
            (labels[int(value["pair_id"])], value["outcome"] == "equivalent") for value in records
        )
        decided = [value for value in records if value["outcome"] != "abstain"]
        baseline = _metrics(
            (
                labels[int(value["pair_id"])],
                current_predictions[int(value["pair_id"])],
            )
            for value in records
        )
        combined_rows: list[tuple[bool, bool]] = []
        override_counts: Counter[str] = Counter()
        repaired = harmed = 0
        for value in records:
            pair_id = int(value["pair_id"])
            label = labels[pair_id]
            baseline_prediction = current_predictions[pair_id]
            outcome = str(value["outcome"])
            if outcome == "abstain":
                combined_prediction = baseline_prediction
            else:
                combined_prediction = outcome == "equivalent"
                override_counts[outcome] += 1
                before_correct = baseline_prediction == label
                after_correct = combined_prediction == label
                repaired += int(not before_correct and after_correct)
                harmed += int(before_correct and not after_correct)
            combined_rows.append((label, combined_prediction))
        label_outcomes: dict[str, Counter[str]] = {
            "positive": Counter(),
            "negative": Counter(),
        }
        for value in records:
            label_outcomes["positive" if labels[int(value["pair_id"])] else "negative"][
                str(value["outcome"])
            ] += 1
        evaluations[scope] = {
            "pair_count": len(records),
            "outcomes": dict(sorted(outcomes.items())),
            "binary_equivalent_positive": binary,
            "current_pipeline_majority": baseline,
            "current_pipeline_plus_invariant_resolution": _metrics(combined_rows),
            "semantic_override": {
                "count": sum(override_counts.values()),
                "by_outcome": dict(sorted(override_counts.items())),
                "repaired": repaired,
                "harmed": harmed,
            },
            "selective_coverage": len(decided) / len(records),
            "selective_accuracy": (
                sum(
                    (value["outcome"] == "equivalent") == labels[int(value["pair_id"])]
                    for value in decided
                )
                / len(decided)
                if decided
                else 0.0
            ),
            "outcomes_by_true_label": {
                key: dict(sorted(counts.items())) for key, counts in label_outcomes.items()
            },
        }

    registry = _sentence_registry(args.corpus)
    text_by_hash = {_text_sha256(text): text for text in registry.values()}
    examples: dict[str, list[dict[str, Any]]] = {}
    for outcome in (
        "equivalent",
        "left_entails_right",
        "right_entails_left",
        "incompatible",
        "abstain",
    ):
        selected = [value for value in pairs if value["outcome"] == outcome][:5]
        examples[outcome] = [
            {
                "pair_id": value["pair_id"],
                "left": text_by_hash[value["left_source_text_sha256"]],
                "right": text_by_hash[value["right_source_text_sha256"]],
                "true_label": labels[int(value["pair_id"])],
                "proof_sha256": value["proof_sha256"],
                "abstention_reasons": value["abstention_reasons"][:12],
            }
            for value in selected
        ]
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "hard_membership_authorized": False,
        "current_pipeline_evidence": current_evidence,
        "preparation_summary": payload["summary"],
        "evaluations": evaluations,
        "examples": examples,
    }
    _atomic_json(args.report, report_payload)
    print(json.dumps(evaluations, indent=2, sort_keys=True), flush=True)
    print(f"report {args.report}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--wordnet-data", type=Path, default=DEFAULT_WORDNET_DATA)
    parser.add_argument("--verbnet-dir", type=Path, default=DEFAULT_VERBNET_DIR)
    parser.add_argument("--verbnet-zip", type=Path, default=DEFAULT_VERBNET_ZIP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--current-pipeline-report",
        type=Path,
        default=DEFAULT_CURRENT_PIPELINE_REPORT,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
