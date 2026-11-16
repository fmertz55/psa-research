#!/usr/bin/env python3
"""Run MRPC through PSA, the learned native gate, and graph proof rules.

This is an offline, non-mutating development diagnostic. MRPC labels are
invisible to the parser and verifier during a run, but the corpus and its error
cases have already influenced development. No result from this script is a
blind performance claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPOSITORY_ROOT / "app"
for path in (REPOSITORY_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.research.mrpc.run_mrpc_current_pipeline_benchmark import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    _load_pinned_frame,
    _project_representations,
)
from scripts.research.mrpc.run_mrpc_gate_verifier_benchmark import (  # noqa: E402
    _learn_gate_threshold,
    _learn_recall_floor_gate_threshold,
    _psa_neighborhoods,
)
from scripts.research.mrpc.run_mrpc_production_pipeline_benchmark import (  # noqa: E402
    BenchmarkError,
    _atomic_json,
    _load_cache_evidence,
)

from mcp_servers.mcp_lib.paraphrase_contracts import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    LEGACY_EVIDENCE_SCHEMA_VERSIONS,
    normalize_evidence,
)
from mcp_servers.mcp_lib.paraphrase_evidence_extraction import (  # noqa: E402
    FACT_EXTRACTOR_IDENTITY,
    build_initial_evidence,
)
from mcp_servers.mcp_lib.paraphrase_membership_edges import (  # noqa: E402
    MEMBERSHIP_EDGE_SCHEMA_VERSION,
    classify_membership_edges,
)
from mcp_servers.mcp_lib.proposition_graph import (  # noqa: E402
    PROPOSITION_GRAPH_SCHEMA_VERSION,
    PropositionGraph,
    build_proposition_graph,
)
from mcp_servers.mcp_lib.proposition_graph_rules import (  # noqa: E402
    LEGACY_EXPERIMENTAL_EXECUTION_POLICY,
    PROOF_KERNEL_EXECUTION_POLICY,
    RULE_DEFINITIONS,
    RULE_PATTERN_DEFINITIONS,
    RULE_REGISTRY_SHA256,
    RULE_REGISTRY_VERSION,
    DeterministicGraphRuleSuite,
    GraphRuleDecision,
    GraphVerificationProfile,
    LexicalRelations,
)
from mcp_servers.mcp_lib.qwen_typed_field_test import (  # noqa: E402
    DEFAULT_MRPC_CALIBRATION_PERCENT,
    MRPCPair,
    _binary_metrics,
    load_mrpc_pairs,
    sha256_text,
    split_mrpc_by_sentence_component,
)
from mcp_servers.mcp_lib.semantic_routing_contract import (  # noqa: E402
    PSA_SIGNATURE_CANDIDATE_LIMIT,
    QWEN3_EMBEDDING_MODEL_ID,
    paraphrase_native_cosine_threshold,
)
from mcp_servers.tokenization_service.extractors.spacy_extractor import (  # noqa: E402
    SpacyExtractor,
)

REPORT_SCHEMA_VERSION = "aiidea_mrpc_proposition_graph_benchmark_v4"
GRAPH_CACHE_SCHEMA_VERSION = "aiidea_mrpc_proposition_graph_evidence_cache_v1"
LEXICAL_CACHE_SCHEMA_VERSION = "aiidea_oewn_graph_lexical_relations_v3"
FRAMEBASE_CACHE_SCHEMA_VERSION = "aiidea_framebase_lex_units_v1"
BLIND_SPLIT_SEED = "mrpc-proposition-graph-blind-sentence-components-v1"
MRPC_COMPATIBLE_GATE_MINIMUM_RECALL = 0.99
_INVALID_XML_CHARACTER_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

DEFAULT_CACHE = (
    REPOSITORY_ROOT / "temp" / "qwen_typed_runs" / "30399991644" / "qwen_embeddings.sqlite3"
)
DEFAULT_SENSE_CACHE = (
    REPOSITORY_ROOT
    / "temp"
    / "mrpc_wordnet_qwen_replay"
    / "30475782422-neo4j-score"
    / "qwen_wordnet_senses.json"
)
DEFAULT_GRAPH_CACHE = (
    REPOSITORY_ROOT / "temp" / "mrpc_proposition_graph" / "spacy_oewn_evidence_v1.json"
)
PINNED_GRAPH_CACHE_SHA256 = "9625dd25e9ed67beb5832ec38dccf95b66415f8c49333e2e1477ef8ff7f927fc"
PINNED_GRAPH_PARSER_IDENTITY = "core_web_sm@3.8.0"
REPLAYABLE_GRAPH_SCHEMA_VERSIONS = frozenset(
    {
        "paraphrase_proposition_graph_v2",
        PROPOSITION_GRAPH_SCHEMA_VERSION,
    }
)
DEFAULT_LEXICAL_CACHE = (
    REPOSITORY_ROOT / "temp" / "mrpc_proposition_graph" / "oewn_lexical_relations_v1.json"
)
PINNED_LEXICAL_CACHE_SHA256 = "5fc286576a9ba15c9f2d371386dc38af8265fb39510a1ad2fe90916a7a1ffa1d"
DEFAULT_FRAMEBASE_CACHE = (
    REPOSITORY_ROOT / "temp" / "mrpc_proposition_graph" / "framebase_lex_units_v1.json"
)
PINNED_FRAMEBASE_CACHE_SHA256 = "a54b7dcea28a8093bda0c0ecc078a6817ac87d05aabcd153b793acd7027bb6e6"
DEFAULT_WORDNET_DATA = REPOSITORY_ROOT / "temp" / "wordnet"
DEFAULT_ETPC_DIR = REPOSITORY_ROOT / "temp" / "ETPC" / "Corpus"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "reports" / "mrpc_rule_policy_dashboard"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_graph_cache(
    cache_path: Path,
    *,
    expected_sha256: str,
    expected_parser_identity: str,
) -> tuple[dict[str, Any], str]:
    """Load graph evidence only when both artifact and parser are pinned."""

    actual_sha256 = _file_sha256(cache_path)
    if actual_sha256 != expected_sha256:
        raise BenchmarkError(
            f"pinned graph cache SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkError("pinned graph cache must contain a JSON object")
    if payload.get("parser_identity") != expected_parser_identity:
        raise BenchmarkError(
            "pinned graph cache parser mismatch: "
            f"expected {expected_parser_identity!r}, got "
            f"{payload.get('parser_identity')!r}"
        )
    return payload, actual_sha256


def _normalized_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _cache_identity_sha256(identity: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_cached_graph_evidence(
    cached: Mapping[str, Any],
    *,
    expected_sentence_hashes: Sequence[str],
    expected_sense_sha256: str,
    expected_parser_identity: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    """Validate a pinned evidence artifact independently of graph projection schema."""

    expected_metadata = {
        "schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "fact_extractor": FACT_EXTRACTOR_IDENTITY,
        "sense_cache_sha256": expected_sense_sha256,
        "sentence_hashes": list(expected_sentence_hashes),
        "parser_identity": expected_parser_identity,
    }
    for key, expected in expected_metadata.items():
        if cached.get(key) != expected:
            if key == "sentence_hashes":
                actual = cached.get(key)
                actual_count = len(actual) if isinstance(actual, list) else None
                raise BenchmarkError(
                    "pinned graph cache sentence_hashes mismatch: "
                    f"expected {len(expected_sentence_hashes)} entries, got "
                    f"{actual_count!r}"
                )
            raise BenchmarkError(
                f"pinned graph cache {key} mismatch: expected {expected!r}, got {cached.get(key)!r}"
            )

    source_graph_schema_version = str(cached.get("graph_schema_version") or "")
    if source_graph_schema_version not in REPLAYABLE_GRAPH_SCHEMA_VERSIONS:
        raise BenchmarkError(
            f"pinned graph cache graph schema is not replayable: {source_graph_schema_version!r}"
        )
    if cached.get("sentence_count") != len(expected_sentence_hashes):
        raise BenchmarkError(
            "pinned graph cache sentence_count mismatch: "
            f"expected {len(expected_sentence_hashes)!r}, got "
            f"{cached.get('sentence_count')!r}"
        )

    source_identity = {
        **expected_metadata,
        "graph_schema_version": source_graph_schema_version,
    }
    source_cache_id = _cache_identity_sha256(source_identity)
    if cached.get("cache_id") != source_cache_id:
        raise BenchmarkError(
            "pinned graph cache identity mismatch: "
            f"expected {source_cache_id}, got {cached.get('cache_id')!r}"
        )

    evidence_by_hash = cached.get("sentences")
    if not isinstance(evidence_by_hash, Mapping):
        raise BenchmarkError("pinned graph cache sentences must be a JSON object")
    expected_hash_set = set(expected_sentence_hashes)
    if set(evidence_by_hash) != expected_hash_set:
        raise BenchmarkError("pinned graph cache sentence entries do not match metadata")

    validated_evidence: dict[str, Mapping[str, Any]] = {}
    source_evidence_schema_versions: set[str] = set()
    for sentence_hash in expected_sentence_hashes:
        entry = evidence_by_hash.get(sentence_hash)
        if not isinstance(entry, Mapping):
            raise BenchmarkError(f"pinned graph cache sentence entry {sentence_hash} is invalid")
        if entry.get("text_sha256") != sentence_hash:
            raise BenchmarkError(
                f"pinned graph cache sentence entry hash mismatch: {sentence_hash}"
            )
        evidence = entry.get("evidence")
        if not isinstance(evidence, Mapping):
            raise BenchmarkError(f"pinned graph cache evidence {sentence_hash} is invalid")
        evidence_schema_version = str(evidence.get("schema_version") or "")
        if evidence_schema_version not in {
            EVIDENCE_SCHEMA_VERSION,
            *LEGACY_EVIDENCE_SCHEMA_VERSIONS,
        }:
            raise BenchmarkError(f"pinned graph cache evidence schema mismatch: {sentence_hash}")
        source_evidence_schema_versions.add(evidence_schema_version)
        if evidence.get("text_sha256") != sentence_hash:
            raise BenchmarkError(f"pinned graph cache evidence hash mismatch: {sentence_hash}")
        extractor_identity = evidence.get("extractor_identity")
        if not isinstance(extractor_identity, Mapping):
            raise BenchmarkError(
                f"pinned graph cache extractor identity is missing: {sentence_hash}"
            )
        if extractor_identity.get("fact_extractor") != FACT_EXTRACTOR_IDENTITY:
            raise BenchmarkError(f"pinned graph cache evidence extractor mismatch: {sentence_hash}")
        if extractor_identity.get("spacy_model") != expected_parser_identity:
            raise BenchmarkError(f"pinned graph cache evidence parser mismatch: {sentence_hash}")
        validated_evidence[sentence_hash] = evidence

    return validated_evidence, {
        **source_identity,
        "cache_id": source_cache_id,
        "source_evidence_schema_versions": sorted(source_evidence_schema_versions),
    }


def _sense_assignments(
    payload: Mapping[str, Any],
    text: str,
) -> dict[tuple[str, str], set[str]]:
    sentences = payload.get("sentences") or {}
    entry = sentences.get(sha256_text(text)) if isinstance(sentences, Mapping) else None
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    if not isinstance(entry, Mapping):
        return result
    for assignment in entry.get("assignments") or []:
        if not isinstance(assignment, Mapping):
            continue
        lemma = str(assignment.get("lemma") or "").casefold()
        pos = str(assignment.get("pos") or "").casefold()
        synset_id = str(assignment.get("synset_id") or "")
        if lemma and pos and synset_id:
            result[(lemma, pos)].add(synset_id)
    return result


def _attach_senses(
    evidence: Mapping[str, Any],
    assignments: Mapping[tuple[str, str], set[str]],
) -> dict[str, Any]:
    enriched = copy.deepcopy(dict(evidence))
    by_lemma: dict[str, set[str]] = defaultdict(set)
    for (lemma, _pos), synsets in assignments.items():
        by_lemma[lemma].update(synsets)
    propositions = []
    for raw_proposition in enriched.get("propositions") or []:
        proposition = dict(raw_proposition)
        predicate = str(proposition.get("predicate_lemma") or "").casefold()
        predicate_synsets = sorted(assignments.get((predicate, "v"), set()))
        if predicate_synsets:
            proposition["synset_ids"] = predicate_synsets
        arguments = []
        for raw_argument in proposition.get("arguments") or []:
            argument = dict(raw_argument)
            lemma = str(argument.get("lemma") or "").casefold()
            synsets = sorted(by_lemma.get(lemma, set()))
            if synsets:
                argument["synset_ids"] = synsets
            arguments.append(argument)
        proposition["arguments"] = arguments
        propositions.append(proposition)
    enriched["propositions"] = propositions
    wordnet = []
    for (lemma, pos), synsets in sorted(assignments.items()):
        for synset_id in sorted(synsets):
            wordnet.append(
                {
                    "lemma": lemma,
                    "pos": pos,
                    "synset_id": synset_id,
                }
            )
    enriched["wordnet"] = wordnet
    return normalize_evidence(enriched).evidence


def _build_or_load_graphs(
    texts: Sequence[str],
    *,
    sense_path: Path,
    cache_path: Path,
    expected_cache_sha256: str,
    expected_parser_identity: str,
) -> tuple[dict[str, PropositionGraph], dict[str, Any]]:
    sense_sha256 = _file_sha256(sense_path)
    sentence_hashes = sorted(sha256_text(text) for text in texts)
    projected_cache_identity = {
        "schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "graph_schema_version": PROPOSITION_GRAPH_SCHEMA_VERSION,
        "fact_extractor": FACT_EXTRACTOR_IDENTITY,
        "sense_cache_sha256": sense_sha256,
        "sentence_hashes": sentence_hashes,
        "parser_identity": expected_parser_identity,
    }
    if cache_path.is_file():
        cached, cache_file_sha256 = _load_pinned_graph_cache(
            cache_path,
            expected_sha256=expected_cache_sha256,
            expected_parser_identity=expected_parser_identity,
        )
        evidence_by_hash, source_cache_identity = _validate_cached_graph_evidence(
            cached,
            expected_sentence_hashes=sentence_hashes,
            expected_sense_sha256=sense_sha256,
            expected_parser_identity=expected_parser_identity,
        )
        graphs = {
            text: build_proposition_graph(evidence_by_hash[sha256_text(text)]) for text in texts
        }
        return graphs, {
            **projected_cache_identity,
            "cache_id": source_cache_identity["cache_id"],
            "source_cache_id": source_cache_identity["cache_id"],
            "projected_cache_id": _cache_identity_sha256(projected_cache_identity),
            "source_graph_schema_version": source_cache_identity["graph_schema_version"],
            "projected_graph_schema_version": PROPOSITION_GRAPH_SCHEMA_VERSION,
            "source_evidence_schema_versions": source_cache_identity[
                "source_evidence_schema_versions"
            ],
            "projected_evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "cache_path": str(cache_path),
            "cache_file_sha256": cache_file_sha256,
            "cache_hit": True,
        }
    extractor = SpacyExtractor()
    if not extractor.available:
        raise BenchmarkError(
            "spaCy en_core_web_sm is required to rebuild a missing or stale graph cache"
        )
    if extractor.model_identity != expected_parser_identity:
        raise BenchmarkError(
            "graph parser identity mismatch: "
            f"expected {expected_parser_identity!r}, got "
            f"{extractor.model_identity!r}"
        )
    cache_identity = {
        **projected_cache_identity,
        "parser_identity": extractor.model_identity,
    }
    cache_id = _cache_identity_sha256(cache_identity)

    sense_payload = json.loads(sense_path.read_text(encoding="utf-8"))
    evidence_by_hash: dict[str, dict[str, Any]] = {}
    graphs: dict[str, PropositionGraph] = {}
    for index, text in enumerate(texts, start=1):
        occurrences = extractor.extract_objects(text)
        projection = build_initial_evidence(
            sentence_id=f"mrpc-sentence:{sha256_text(text)}",
            text=text,
            occurrences=occurrences,
            promoted_object_ids=(),
            extractor_identity={
                "tokenizer": "aiidea_sentence_tokenizer@v1",
                "spacy_model": extractor.model_identity,
                "fact_extractor": FACT_EXTRACTOR_IDENTITY,
                "sense_resolver": str(
                    (sense_payload.get("resolver") or {}).get("method") or "qwen_oewn_pinned"
                ),
            },
            extraction_complete=True,
        )
        evidence = _attach_senses(
            projection.evidence,
            _sense_assignments(sense_payload, text),
        )
        graph = build_proposition_graph(evidence)
        graphs[text] = graph
        evidence_by_hash[sha256_text(text)] = {
            "text_sha256": sha256_text(text),
            "evidence": evidence,
        }
        if index % 500 == 0:
            print(f"parsed {index}/{len(texts)} MRPC sentences", flush=True)
    payload = {
        **cache_identity,
        "cache_id": cache_id,
        "sentence_count": len(texts),
        "sentences": evidence_by_hash,
    }
    _atomic_json(cache_path, payload)
    return graphs, {
        **cache_identity,
        "cache_id": cache_id,
        "source_cache_id": cache_id,
        "projected_cache_id": cache_id,
        "source_graph_schema_version": PROPOSITION_GRAPH_SCHEMA_VERSION,
        "projected_graph_schema_version": PROPOSITION_GRAPH_SCHEMA_VERSION,
        "source_evidence_schema_versions": [EVIDENCE_SCHEMA_VERSION],
        "projected_evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "cache_path": str(cache_path),
        "cache_file_sha256": _file_sha256(cache_path),
        "cache_hit": False,
    }


def _framebase_relations(
    framebase_cache: Path | None,
    *,
    expected_sha256: str | None = None,
) -> tuple[frozenset[tuple[str, str]], frozenset[tuple[str, str]], dict[str, Any]]:
    """Load pinned lemma/synset frame evocations from the ontology seed snapshot."""

    if framebase_cache is None or not framebase_cache.is_file():
        if expected_sha256:
            raise BenchmarkError(f"the pinned FrameBase cache is missing: {framebase_cache}")
        return (
            frozenset(),
            frozenset(),
            {
                "schema_version": FRAMEBASE_CACHE_SCHEMA_VERSION,
                "cache_path": str(framebase_cache) if framebase_cache else None,
                "cache_hit": False,
                "identity": "framebase:none",
            },
        )
    cache_file_sha256 = _file_sha256(framebase_cache)
    if expected_sha256 and cache_file_sha256 != expected_sha256:
        raise BenchmarkError(
            f"FrameBase cache SHA-256 mismatch: expected {expected_sha256}, got {cache_file_sha256}"
        )
    payload = json.loads(framebase_cache.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FRAMEBASE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported framebase cache schema: {payload.get('schema_version')!r}")
    lemma_frames = frozenset(
        (str(lemma_key), str(frame_id))
        for lemma_key, frame_id in payload.get("lemma_frames", [])
        if str(lemma_key) and str(frame_id)
    )
    synset_frames = frozenset(
        (str(synset_id), str(frame_id))
        for synset_id, frame_id in payload.get("synset_frames", [])
        if str(synset_id) and str(frame_id)
    )
    identity = f"framebase:{cache_file_sha256}"
    return (
        lemma_frames,
        synset_frames,
        {
            "schema_version": FRAMEBASE_CACHE_SCHEMA_VERSION,
            "cache_path": str(framebase_cache),
            "cache_hit": True,
            "cache_file_sha256": cache_file_sha256,
            "identity": identity,
            "source": payload.get("source"),
            "lemma_frame_count": len(lemma_frames),
            "synset_frame_count": len(synset_frames),
        },
    )


def _lexical_relations(
    graphs: Mapping[str, PropositionGraph],
    *,
    wordnet_data: Path,
    cache_path: Path,
    framebase_cache: Path | None = None,
    expected_cache_sha256: str | None = None,
    expected_framebase_sha256: str | None = None,
) -> tuple[LexicalRelations, dict[str, Any]]:
    observed_synset_ids = sorted(
        {
            str(synset_id)
            for graph in graphs.values()
            for proposition in graph.semantic_value.get("propositions", [])
            for value in (
                proposition.get("predicate", {}),
                *(argument.get("node", {}) for argument in proposition.get("arguments", [])),
            )
            for synset_id in value.get("synset_ids", [])
        }
    )
    lemma_queries = sorted(
        {
            (str(value.get("lemma") or "").casefold(), pos)
            for graph in graphs.values()
            for proposition in graph.semantic_value.get("propositions", [])
            for value, pos in (
                (proposition.get("predicate", {}), "v"),
                *(
                    (argument.get("node", {}), candidate_pos)
                    for argument in proposition.get("arguments", [])
                    for candidate_pos in ("a", "n", "r", "v")
                ),
            )
            if value.get("lemma")
        }
    )
    identity = {
        "schema_version": LEXICAL_CACHE_SCHEMA_VERSION,
        "wordnet": "oewn:2024",
        "observed_synset_ids": observed_synset_ids,
        "lemma_queries": lemma_queries,
    }
    cache_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    lemma_frames, synset_frames, framebase_evidence = _framebase_relations(
        framebase_cache,
        expected_sha256=expected_framebase_sha256,
    )
    resource_identity_suffix = f"+{framebase_evidence['identity']}"
    if cache_path.is_file():
        cache_file_sha256 = _file_sha256(cache_path)
        if expected_cache_sha256 and cache_file_sha256 != expected_cache_sha256:
            raise BenchmarkError(
                "OEWN lexical cache SHA-256 mismatch: "
                f"expected {expected_cache_sha256}, got {cache_file_sha256}"
            )
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_id") == cache_id:
            relations = LexicalRelations(
                antonym_synset_pairs=frozenset(tuple(value) for value in cached["antonyms"]),
                derivational_synset_pairs=frozenset(
                    tuple(value) for value in cached["derivations"]
                ),
                hypernym_synset_pairs=frozenset(tuple(value) for value in cached["hypernyms"]),
                hypernym_directed_pairs=frozenset(
                    tuple(value) for value in cached["hypernyms_directed"]
                ),
                lemma_synset_pairs=frozenset(tuple(value) for value in cached["lemma_synsets"]),
                synset_frame_pairs=synset_frames,
                lemma_frame_pairs=lemma_frames,
                resource_identity=(f"oewn:2024:{cache_file_sha256}{resource_identity_suffix}"),
            )
            return relations, {
                **identity,
                "cache_id": cache_id,
                "cache_path": str(cache_path),
                "cache_file_sha256": cache_file_sha256,
                "cache_hit": True,
                "framebase": framebase_evidence,
            }
    elif expected_cache_sha256:
        raise BenchmarkError(f"the pinned OEWN lexical cache is missing: {cache_path}")

    import wn

    wn.config.data_directory = wordnet_data.resolve()
    wordnet = wn.Wordnet("oewn:2024")
    lemma_synsets: set[tuple[str, str]] = set()
    for lemma, pos in lemma_queries:
        try:
            synsets = wordnet.synsets(lemma, pos=pos)
        except Exception:
            continue
        lemma_key = f"{pos}:{lemma}"
        lemma_synsets.update((lemma_key, synset.id) for synset in synsets)
    synset_ids = sorted(
        set(observed_synset_ids) | {synset_id for _lemma, synset_id in lemma_synsets}
    )
    antonyms: set[tuple[str, str]] = set()
    derivations: set[tuple[str, str]] = set()
    hypernyms: set[tuple[str, str]] = set()
    hypernyms_directed: set[tuple[str, str]] = set()
    for synset_id in synset_ids:
        try:
            synset = wordnet.synset(synset_id)
        except Exception:
            continue
        for hypernym in synset.relations().get("hypernym", []):
            hypernyms.add(LexicalRelations.canonical_pair(synset.id, hypernym.id))
            hypernyms_directed.add((synset.id, hypernym.id))
        for word in synset.words():
            for sense in word.senses():
                try:
                    if sense.synset().id != synset.id:
                        continue
                except Exception:
                    continue
                for relation_name, destination in (
                    ("antonym", antonyms),
                    ("derivation", derivations),
                ):
                    for related_sense in sense.relations().get(relation_name, []):
                        try:
                            related_id = related_sense.synset().id
                        except Exception:
                            continue
                        destination.add(LexicalRelations.canonical_pair(synset.id, related_id))
    payload = {
        **identity,
        "cache_id": cache_id,
        "antonyms": sorted(antonyms),
        "derivations": sorted(derivations),
        "hypernyms": sorted(hypernyms),
        "hypernyms_directed": sorted(hypernyms_directed),
        "lemma_synsets": sorted(lemma_synsets),
    }
    _atomic_json(cache_path, payload)
    cache_file_sha256 = _file_sha256(cache_path)
    if expected_cache_sha256 and cache_file_sha256 != expected_cache_sha256:
        raise BenchmarkError(
            "rebuilt OEWN lexical cache does not match the pinned SHA-256: "
            f"expected {expected_cache_sha256}, got {cache_file_sha256}"
        )
    relations = LexicalRelations(
        antonym_synset_pairs=frozenset(antonyms),
        derivational_synset_pairs=frozenset(derivations),
        hypernym_synset_pairs=frozenset(hypernyms),
        hypernym_directed_pairs=frozenset(hypernyms_directed),
        lemma_synset_pairs=frozenset(lemma_synsets),
        synset_frame_pairs=synset_frames,
        lemma_frame_pairs=lemma_frames,
        resource_identity=f"oewn:2024:{cache_file_sha256}{resource_identity_suffix}",
    )
    return relations, {
        **identity,
        "cache_id": cache_id,
        "cache_path": str(cache_path),
        "cache_file_sha256": cache_file_sha256,
        "cache_hit": False,
        "framebase": framebase_evidence,
    }


def _load_etpc(etpc_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    pairs_path = etpc_dir / "text_pairs.xml"
    relations_path = etpc_dir / "textual_paraphrases.xml"
    if not pairs_path.is_file() or not relations_path.is_file():
        return {}, {"available": False}

    def sanitized_root(path: Path) -> ET.Element:
        source = path.read_text(encoding="utf-8")
        return ET.fromstring(_INVALID_XML_CHARACTER_RE.sub("", source))

    types_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in sanitized_root(relations_path).findall("relation"):
        pair_id = str(relation.findtext("pair_id") or "")
        types_by_pair[pair_id].append(
            {
                "type_id": int(relation.findtext("type_id") or 0),
                "type_name": str(relation.findtext("type_name") or ""),
                "sense_preserving": str(relation.findtext("sense_preserving") or "").casefold(),
            }
        )
    by_text: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in sanitized_root(pairs_path).findall("text_pair"):
        pair_id = str(pair.findtext("pair_id") or "")
        source = _normalized_text(pair.findtext("sent1_raw") or "")
        target = _normalized_text(pair.findtext("sent2_raw") or "")
        entry = {
            "pair_id": int(pair_id),
            "mrpc_label": str(pair.findtext("mrpc_label") or "") == "1",
            "etpc_label": str(pair.findtext("etpc_label") or "") == "1",
            "negation": str(pair.findtext("negation") or "") == "1",
            "relations": sorted(types_by_pair.get(pair_id, []), key=lambda item: item["type_id"]),
        }
        by_text[(source, target)] = entry
        by_text[(target, source)] = entry
    return by_text, {
        "available": True,
        "version": "0.95",
        "pair_count": len(by_text) // 2,
        "text_pairs_sha256": _file_sha256(pairs_path),
        "relations_sha256": _file_sha256(relations_path),
    }


def _evaluate(
    pairs: Sequence[MRPCPair],
    *,
    native_by_text: Mapping[str, np.ndarray],
    neighborhoods: Mapping[str, frozenset[str]],
    threshold: float,
    decisions: Mapping[int, GraphRuleDecision],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = np.asarray([pair.is_paraphrase for pair in pairs], dtype=bool)
    cosines = np.asarray(
        [
            float(np.dot(native_by_text[pair.sentence1], native_by_text[pair.sentence2]))
            for pair in pairs
        ]
    )
    nominated = np.asarray(
        [pair.sentence2 in neighborhoods[pair.sentence1] for pair in pairs],
        dtype=bool,
    )
    exact = np.asarray(
        [sha256_text(pair.sentence1) == sha256_text(pair.sentence2) for pair in pairs],
        dtype=bool,
    )
    gate_passed = cosines >= threshold
    verifier_passed = np.asarray([decisions[pair.pair_id].accepted for pair in pairs])
    predictions = exact | (nominated & gate_passed & verifier_passed)
    gate_only = exact | (nominated & gate_passed)
    traces = []
    for index, pair in enumerate(pairs):
        decision = decisions[pair.pair_id]
        proof = decision.trace_dict()
        membership_edges = list(classify_membership_edges(proof))
        membership_edge_types = [edge["edge_type"] for edge in membership_edges]
        singular_membership_edge = membership_edges[0] if len(membership_edges) == 1 else None
        traces.append(
            {
                "pair_id": pair.pair_id,
                "label": pair.is_paraphrase,
                "psa_nominated": bool(nominated[index]),
                "native_similarity": float(cosines[index]),
                "native_gate_passed": bool(gate_passed[index]),
                "graph_state": decision.state.value,
                "graph_reason": decision.reason,
                "rule_type_ids": sorted(
                    {application.etpc_type_id for application in decision.applications}
                ),
                "conflict_kinds": sorted({conflict.kind for conflict in decision.conflicts}),
                "predicted_paraphrase": bool(predictions[index]),
                "membership_edge_type": (
                    singular_membership_edge["edge_type"] if singular_membership_edge else None
                ),
                "membership_edge": singular_membership_edge,
                "membership_edge_types": membership_edge_types,
                "membership_edges": membership_edges,
                "proof": proof,
            }
        )
    edge_type_ledger: dict[str, dict[str, int]] = {}
    edge_set_ledger: dict[str, dict[str, int]] = {}
    for trace in traces:
        keys = trace["membership_edge_types"] or ["(none)"]
        for key in keys:
            row = edge_type_ledger.setdefault(
                key,
                {"mrpc_paraphrase": 0, "mrpc_not_paraphrase": 0},
            )
            row["mrpc_paraphrase" if trace["label"] else "mrpc_not_paraphrase"] += 1
        set_key = "+".join(trace["membership_edge_types"]) or "(none)"
        set_row = edge_set_ledger.setdefault(
            set_key,
            {"mrpc_paraphrase": 0, "mrpc_not_paraphrase": 0},
        )
        set_row["mrpc_paraphrase" if trace["label"] else "mrpc_not_paraphrase"] += 1
    edge_predictions = np.asarray(
        [bool(trace["membership_edge_types"]) for trace in traces],
        dtype=bool,
    )
    return {
        "metrics": _binary_metrics(labels, predictions),
        "gate_only_metrics": _binary_metrics(labels, gate_only),
        "membership_edge_schema_version": MEMBERSHIP_EDGE_SCHEMA_VERSION,
        "membership_edge_ledger": dict(sorted(edge_type_ledger.items())),
        "membership_edge_set_ledger": dict(sorted(edge_set_ledger.items())),
        "membership_edge_union_metrics": _binary_metrics(labels, edge_predictions),
        "state_counts": dict(
            sorted(Counter(decisions[pair.pair_id].state.value for pair in pairs).items())
        ),
        "reason_counts": dict(
            sorted(Counter(decisions[pair.pair_id].reason for pair in pairs).items())
        ),
        "stage_counts": {
            "psa_nominated": int(np.count_nonzero(nominated)),
            "native_gate_passed": int(np.count_nonzero(nominated & gate_passed)),
            "graph_verified": int(np.count_nonzero(nominated & gate_passed & verifier_passed)),
        },
    }, traces


def _etpc_ledger(
    traces: Sequence[Mapping[str, Any]],
    pair_by_id: Mapping[int, MRPCPair],
    etpc_by_text: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    by_type: dict[int, dict[str, Any]] = {}
    mapped = 0
    for trace in traces:
        pair = pair_by_id[int(trace["pair_id"])]
        etpc = etpc_by_text.get(
            (_normalized_text(pair.sentence1), _normalized_text(pair.sentence2))
        )
        if etpc is None:
            continue
        mapped += 1
        for relation in etpc.get("relations", []):
            type_id = int(relation["type_id"])
            row = by_type.setdefault(
                type_id,
                {
                    "type_id": type_id,
                    "type_name": relation["type_name"],
                    "pair_count": 0,
                    "sense_preserving_count": 0,
                    "verified_count": 0,
                    "do_not_merge_count": 0,
                    "abstain_count": 0,
                    "rule_applied_count": 0,
                },
            )
            row["pair_count"] += 1
            row["sense_preserving_count"] += int(relation.get("sense_preserving") == "yes")
            state_key = {
                "verified_paraphrase": "verified_count",
                "do_not_merge": "do_not_merge_count",
                "abstain_and_keep_free_range": "abstain_count",
            }[str(trace["graph_state"])]
            row[state_key] += 1
            row["rule_applied_count"] += int(type_id in trace["rule_type_ids"])
    return {
        "mapped_held_out_pairs": mapped,
        "types": [by_type[key] for key in sorted(by_type)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--mrpc-dir",
        type=Path,
        default=REPOSITORY_ROOT / "app" / "utils" / "msrpc_split",
    )
    parser.add_argument("--sense-cache", type=Path, default=DEFAULT_SENSE_CACHE)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument(
        "--graph-cache-sha256",
        default=PINNED_GRAPH_CACHE_SHA256,
        help="Required SHA-256 for an existing pinned graph-evidence cache.",
    )
    parser.add_argument(
        "--graph-parser-identity",
        default=PINNED_GRAPH_PARSER_IDENTITY,
        help="Required parser identity for cached or rebuilt graph evidence.",
    )
    parser.add_argument("--lexical-cache", type=Path, default=DEFAULT_LEXICAL_CACHE)
    parser.add_argument(
        "--lexical-cache-sha256",
        default=PINNED_LEXICAL_CACHE_SHA256,
        help="Required SHA-256 for the pinned OEWN relation artifact.",
    )
    parser.add_argument("--framebase-cache", type=Path, default=DEFAULT_FRAMEBASE_CACHE)
    parser.add_argument(
        "--framebase-cache-sha256",
        default=PINNED_FRAMEBASE_CACHE_SHA256,
        help="Required SHA-256 for the pinned FrameBase relation artifact.",
    )
    parser.add_argument("--wordnet-data", type=Path, default=DEFAULT_WORDNET_DATA)
    parser.add_argument("--etpc-dir", type=Path, default=DEFAULT_ETPC_DIR)
    parser.add_argument(
        "--pinned-guide-embeddings",
        type=Path,
        default=DEFAULT_PINNED_GUIDE_EMBEDDINGS,
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Policy-qualified output directory. Defaults to "
            "reports/mrpc_rule_policy_dashboard/<execution-policy-id>."
        ),
    )
    parser.add_argument("--psa-top-k", type=int, default=PSA_SIGNATURE_CANDIDATE_LIMIT)
    parser.add_argument(
        "--rule-execution-policy",
        choices=(
            PROOF_KERNEL_EXECUTION_POLICY.policy_id,
            LEGACY_EXPERIMENTAL_EXECUTION_POLICY.policy_id,
        ),
        default=PROOF_KERNEL_EXECUTION_POLICY.policy_id,
        help=(
            "Proof-only authority boundary or the explicitly quarantined legacy "
            "MRPC diagnostic policy."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = load_mrpc_pairs(args.mrpc_dir.resolve())
    split = split_mrpc_by_sentence_component(pairs, seed=BLIND_SPLIT_SEED)
    calibration_pairs = [pair for pair in pairs if split[pair.pair_id] == "calibration"]
    held_out_pairs = [pair for pair in pairs if split[pair.pair_id] == "held_out"]
    texts = sorted(
        {text for pair in pairs for text in (pair.sentence1, pair.sentence2)},
        key=lambda text: (sha256_text(text), text),
    )
    native_by_text, _cached_guides, embedding_evidence = _load_cache_evidence(
        args.cache.resolve(), pairs
    )
    pinned_guides, calibration, frame_router, frame_evidence = _load_pinned_frame(
        args.pinned_guide_embeddings.resolve(), args.calibration.resolve()
    )
    signatures_by_text, _labels, signature_distribution = _project_representations(
        native_by_text,
        pinned_guides,
        frame_router,
        calibration,
    )
    neighborhoods, neighborhood_evidence = _psa_neighborhoods(
        signatures_by_text,
        top_k=args.psa_top_k,
    )
    graphs, graph_evidence = _build_or_load_graphs(
        texts,
        sense_path=args.sense_cache.resolve(),
        cache_path=args.graph_cache.resolve(),
        expected_cache_sha256=args.graph_cache_sha256,
        expected_parser_identity=args.graph_parser_identity,
    )
    lexical_relations, lexical_evidence = _lexical_relations(
        graphs,
        wordnet_data=args.wordnet_data.resolve(),
        cache_path=args.lexical_cache.resolve(),
        framebase_cache=args.framebase_cache.resolve(),
        expected_cache_sha256=args.lexical_cache_sha256,
        expected_framebase_sha256=args.framebase_cache_sha256,
    )
    execution_policies = {
        PROOF_KERNEL_EXECUTION_POLICY.policy_id: PROOF_KERNEL_EXECUTION_POLICY,
        LEGACY_EXPERIMENTAL_EXECUTION_POLICY.policy_id: (LEGACY_EXPERIMENTAL_EXECUTION_POLICY),
    }
    execution_policy = execution_policies[args.rule_execution_policy]
    suite = DeterministicGraphRuleSuite(
        lexical_relations=lexical_relations,
        execution_policy=execution_policy,
    )
    profiles = tuple(GraphVerificationProfile)
    decisions_by_profile = {
        profile: {
            pair.pair_id: suite.verify(
                graphs[pair.sentence1],
                graphs[pair.sentence2],
                source_text=pair.sentence1,
                target_text=pair.sentence2,
                profile=profile,
            )
            for pair in pairs
        }
        for profile in profiles
    }
    calibration_cosines = np.asarray(
        [
            float(np.dot(native_by_text[pair.sentence1], native_by_text[pair.sentence2]))
            for pair in calibration_pairs
        ]
    )
    calibration_labels = np.asarray([pair.is_paraphrase for pair in calibration_pairs], dtype=bool)
    calibration_nominated = np.asarray(
        [pair.sentence2 in neighborhoods[pair.sentence1] for pair in calibration_pairs],
        dtype=bool,
    )
    cold_start = paraphrase_native_cosine_threshold(QWEN3_EMBEDDING_MODEL_ID)
    learned_threshold, learned_calibration_metrics = _learn_gate_threshold(
        calibration_cosines,
        calibration_labels,
        calibration_nominated,
        np.ones(len(calibration_pairs), dtype=bool),
        cold_start=cold_start,
    )
    compatible_threshold, compatible_calibration_metrics = _learn_recall_floor_gate_threshold(
        calibration_cosines,
        calibration_labels,
        calibration_nominated,
        np.ones(len(calibration_pairs), dtype=bool),
        cold_start=cold_start,
        minimum_positive_recall=MRPC_COMPATIBLE_GATE_MINIMUM_RECALL,
    )
    learned_threshold_by_profile = {
        GraphVerificationProfile.STRICT_MERGE: learned_threshold,
        GraphVerificationProfile.MRPC_COMPATIBLE: compatible_threshold,
    }
    cold_by_profile: dict[
        GraphVerificationProfile, tuple[dict[str, Any], list[dict[str, Any]]]
    ] = {}
    learned_by_profile: dict[
        GraphVerificationProfile, tuple[dict[str, Any], list[dict[str, Any]]]
    ] = {}
    for profile in profiles:
        cold_by_profile[profile] = _evaluate(
            held_out_pairs,
            native_by_text=native_by_text,
            neighborhoods=neighborhoods,
            threshold=cold_start,
            decisions=decisions_by_profile[profile],
        )
        learned_by_profile[profile] = _evaluate(
            held_out_pairs,
            native_by_text=native_by_text,
            neighborhoods=neighborhoods,
            threshold=learned_threshold_by_profile[profile],
            decisions=decisions_by_profile[profile],
        )
    strict_profile = GraphVerificationProfile.STRICT_MERGE
    cold_report, cold_traces = cold_by_profile[strict_profile]
    learned_report, learned_traces = learned_by_profile[strict_profile]
    etpc_by_text, etpc_evidence = _load_etpc(args.etpc_dir.resolve())
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    held_out_ledgers = {
        profile.value: _etpc_ledger(
            learned_by_profile[profile][1],
            pair_by_id,
            etpc_by_text,
        )
        for profile in profiles
    }
    held_out_ledger = held_out_ledgers[strict_profile.value]
    graph_unresolved = Counter(value for graph in graphs.values() for value in graph.unresolved)
    rule_applications_by_profile = {
        profile.value: Counter(
            application.etpc_type_id
            for decision in decisions_by_profile[profile].values()
            for application in decision.applications
        )
        for profile in profiles
    }
    rule_applications = rule_applications_by_profile[strict_profile.value]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluation_class": "post_hoc_iterative_development_diagnostic",
        "pipeline": [
            "cached_native_embedding",
            "psa_signature_top_32_neighborhood",
            "learned_native_cosine_candidate_gate",
            "spacy_oewn_proposition_graph",
            "policy_gated_proposition_graph_relation",
        ],
        "split_policy": {
            "unit": "connected_component_of_normalized_sentence_identity",
            "seed": BLIND_SPLIT_SEED,
            "calibration_percent": DEFAULT_MRPC_CALIBRATION_PERCENT,
            "calibration_pair_count": len(calibration_pairs),
            "held_out_pair_count": len(held_out_pairs),
            "held_out_labels_visible_during_decision": False,
            "held_out_labels_previously_inspected_during_rule_development": True,
            "eligible_for_blind_performance_claim": False,
            "required_next_evaluation": "fresh_sealed_sentence_component_split",
        },
        "evidence": {
            "embedding_cache": embedding_evidence,
            "frame": frame_evidence,
            "psa_neighborhood": neighborhood_evidence,
            "signature_distribution": signature_distribution,
            "graph": graph_evidence,
            "lexical_relations": lexical_evidence,
            "etpc": etpc_evidence,
        },
        "threshold": {
            "cold_start": cold_start,
            "learned": learned_threshold,
            "learning_objective": (
                "candidate_gate_calibration_f1_then_precision_then_stricter_before_verification"
            ),
            "verifier_outcomes_visible_during_gate_learning": False,
            "learned_calibration_metrics": learned_calibration_metrics,
            "profiles": {
                GraphVerificationProfile.STRICT_MERGE.value: {
                    "learned": learned_threshold,
                    "objective": ("candidate_gate_calibration_f1_then_precision_then_stricter"),
                    "calibration_metrics": learned_calibration_metrics,
                },
                GraphVerificationProfile.MRPC_COMPATIBLE.value: {
                    "learned": compatible_threshold,
                    "objective": compatible_calibration_metrics["objective"],
                    "minimum_positive_recall": MRPC_COMPATIBLE_GATE_MINIMUM_RECALL,
                    "calibration_metrics": compatible_calibration_metrics,
                },
            },
        },
        "verifier": {
            "generative_model": False,
            "statistical_truth_classifier": False,
            "registry_version": RULE_REGISTRY_VERSION,
            "registry_sha256": RULE_REGISTRY_SHA256,
            "execution_policy": {
                "policy_id": execution_policy.policy_id,
                "sha256": execution_policy.sha256,
                "enabled_families": [family.value for family in execution_policy.enabled_families],
                "enabled_authorities": [
                    authority.value for authority in execution_policy.enabled_authorities
                ],
                "enabled_pattern_ids": [
                    definition.pattern_id
                    for definition in RULE_PATTERN_DEFINITIONS
                    if execution_policy.allows(definition.pattern_id)
                ],
                "diagnostic_only": (execution_policy == LEGACY_EXPERIMENTAL_EXECUTION_POLICY),
            },
            "profiles": {
                GraphVerificationProfile.STRICT_MERGE.value: {
                    "hard_merge_authority": (execution_policy.authorizes_hard_membership),
                    "contract": "bidirectional_proposition_equivalence",
                },
                GraphVerificationProfile.MRPC_COMPATIBLE.value: {
                    "hard_merge_authority": False,
                    "contract": "document_paraphrase_compatibility",
                },
            },
            "rule_definitions": [
                {
                    "rule_id": rule.rule_id,
                    "etpc_type_id": rule.etpc_type_id,
                    "name": rule.name,
                    "category": rule.category,
                    "policy": rule.policy.value,
                    "explanation": rule.explanation,
                }
                for rule in RULE_DEFINITIONS
            ],
            "rule_pattern_definitions": [
                {
                    "pattern_id": definition.pattern_id,
                    "etpc_type_id": definition.etpc_type_id,
                    "license_scope": definition.license_scope.value,
                    "family": definition.family.value,
                    "authority": definition.authority.value,
                    "provenance": definition.provenance,
                    "default_runtime_enabled": definition.default_runtime_enabled,
                    "explanation": definition.explanation,
                }
                for definition in RULE_PATTERN_DEFINITIONS
            ],
            "rule_application_counts_all_pairs": dict(
                sorted((str(key), value) for key, value in rule_applications.items())
            ),
            "rule_application_counts_by_profile": {
                profile: dict(sorted((str(key), value) for key, value in counts.items()))
                for profile, counts in sorted(rule_applications_by_profile.items())
            },
            "graph_unresolved_counts": dict(sorted(graph_unresolved.items())),
        },
        "held_out": {
            "cold_start_gate": cold_report,
            "learned_gate": learned_report,
            "etpc_type_ledger": held_out_ledger,
            "profiles": {
                profile.value: {
                    "cold_start_gate": cold_by_profile[profile][0],
                    "learned_gate": learned_by_profile[profile][0],
                    "etpc_type_ledger": held_out_ledgers[profile.value],
                }
                for profile in profiles
            },
        },
        "held_out_pair_traces": {
            "cold_start_gate": cold_traces,
            "learned_gate": learned_traces,
            "profiles": {
                profile.value: {
                    "cold_start_gate": cold_by_profile[profile][1],
                    "learned_gate": learned_by_profile[profile][1],
                }
                for profile in profiles
            },
        },
    }
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT / execution_policy.policy_id
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mrpc_proposition_graph_report.json"
    if output_path.is_file():
        existing_report = json.loads(output_path.read_text(encoding="utf-8"))
        existing_policy_id = (
            (existing_report.get("verifier") or {}).get("execution_policy", {}).get("policy_id")
        )
        if existing_policy_id != execution_policy.policy_id:
            raise BenchmarkError(
                "refusing to overwrite an MRPC report from a different "
                f"execution policy: {existing_policy_id!r}"
            )
    _atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "report": str(output_path),
                "cold_start_threshold": cold_start,
                "learned_threshold": learned_threshold,
                "learned_threshold_by_profile": {
                    profile.value: learned_threshold_by_profile[profile] for profile in profiles
                },
                "cold_start_held_out": cold_report["metrics"],
                "learned_held_out": learned_report["metrics"],
                "learned_gate_only_held_out": learned_report["gate_only_metrics"],
                "state_counts": learned_report["state_counts"],
                "profile_held_out": {
                    profile.value: learned_by_profile[profile][0]["metrics"] for profile in profiles
                },
                "graph_unresolved_counts": dict(sorted(graph_unresolved.items())),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
