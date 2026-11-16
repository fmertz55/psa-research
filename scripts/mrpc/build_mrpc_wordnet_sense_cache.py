#!/usr/bin/env python3
"""Build an offline OEWN 2024 sense cache for the MRPC benchmark.

This deliberately does not start aiidea services. It mirrors the production
lexical contract where practical:

1. Extract noun and verb objects with NLTK POS tags.
2. Lemmatize against the pinned Open English WordNet inventory.
3. Resolve ambiguous lemmas with a deterministic extended-Lesk score over
   definitions, examples, synonyms, and bounded one-hop relations.

Production uses Qwen sentence embeddings against SynsetExample HNSW points.
The cache produced here is therefore an auditable offline approximation, not
a claim that lexical-overlap WSD is identical to the production resolver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MRPC_DIR = REPOSITORY_ROOT / "app" / "utils" / "msrpc_split"
DEFAULT_WN_DATA = REPOSITORY_ROOT / "temp" / "wordnet"
DEFAULT_NLTK_DATA = REPOSITORY_ROOT / "temp" / "nltk_data"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "temp" / "mrpc_wordnet" / "oewn_2024_extended_lesk_senses.json"

CACHE_SCHEMA_VERSION = "aiidea_mrpc_wordnet_sense_cache_v1"
RESOLVER_ID = "oewn_2024_nltk_nv_extended_lesk_v1"
OEWN_VERSION = "oewn:2024"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because
    been before being below between both but by can could did do does doing
    down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me
    more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves
    """.split()
)


class OfflineWordNetError(RuntimeError):
    """Raised when the offline WordNet evidence contract is incomplete."""


@dataclass(frozen=True, slots=True)
class SenseAssignment:
    lemma: str
    pos: str
    synset_id: str
    score: float
    runner_up_score: float | None
    candidate_count: int
    resolution_method: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_mrpc_texts(mrpc_dir: Path) -> tuple[list[str], dict[str, Any]]:
    corpus_path = mrpc_dir.parent / "msr_paraphrase_data.txt"
    test_path = mrpc_dir.parent / "msr_paraphrase_test.txt"
    for path in (corpus_path, test_path):
        if not path.is_file():
            raise OfflineWordNetError(f"missing MRPC source file: {path}")

    sentence_by_id: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        required = {"Sentence ID", "String"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise OfflineWordNetError("MRPC sentence registry schema is invalid")
        for row in reader:
            sentence_id = str(row.get("Sentence ID") or "").strip()
            text = str(row.get("String") or "")
            if sentence_id and text.strip():
                sentence_by_id[sentence_id] = text

    referenced_ids: set[str] = set()
    pair_count = 0
    with test_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        required = {"Quality", "#1 ID", "#2 ID"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise OfflineWordNetError("MRPC test split schema is invalid")
        for row in reader:
            first_id = str(row.get("#1 ID") or "").strip()
            second_id = str(row.get("#2 ID") or "").strip()
            if first_id not in sentence_by_id or second_id not in sentence_by_id:
                raise OfflineWordNetError("MRPC test split references a missing sentence")
            referenced_ids.update((first_id, second_id))
            pair_count += 1

    texts = sorted(
        {sentence_by_id[sentence_id] for sentence_id in referenced_ids},
        key=lambda text: (_sha256_text(text), text),
    )
    return texts, {
        "pair_count": pair_count,
        "unique_sentence_count": len(texts),
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": _sha256_file(corpus_path),
        "test_path": str(test_path.resolve()),
        "test_sha256": _sha256_file(test_path),
    }


class OfflineOEWNResolver:
    """Deterministic noun/verb OEWN resolver with bounded extended Lesk."""

    def __init__(
        self,
        wordnet: Any,
        morphy: Any,
        stemmer: Any,
        *,
        minimum_score: float = 1.0,
        minimum_margin: float = 1.0,
        related_limit: int = 64,
    ) -> None:
        self.wordnet = wordnet
        self.morphy = morphy
        self.stemmer = stemmer
        self.minimum_score = float(minimum_score)
        self.minimum_margin = float(minimum_margin)
        self.related_limit = int(related_limit)
        self._lemma_cache: dict[tuple[str, str], str | None] = {}
        self._synset_cache: dict[tuple[str, str], tuple[Any, ...]] = {}
        self._signature_cache: dict[tuple[str, str], Counter[str]] = {}

    def _terms(self, text: str) -> list[str]:
        return [
            self.stemmer.stem(token.casefold())
            for token in WORD_RE.findall(text or "")
            if token.casefold() not in STOP_WORDS
        ]

    def _lemma(self, token: str, pos: str) -> str | None:
        key = token.casefold(), pos
        if key in self._lemma_cache:
            return self._lemma_cache[key]
        result = self.morphy(key[0], pos)
        forms = set(result.get(pos, set())) if isinstance(result, dict) else set()
        forms.add(key[0])
        candidates = [form for form in forms if self._synsets(form, pos)]
        lemma = min(candidates, key=lambda form: (len(form), form)) if candidates else None
        self._lemma_cache[key] = lemma
        return lemma

    def _synsets(self, lemma: str, pos: str) -> tuple[Any, ...]:
        key = lemma, pos
        if key not in self._synset_cache:
            self._synset_cache[key] = tuple(self.wordnet.synsets(lemma, pos=pos))
        return self._synset_cache[key]

    @staticmethod
    def _add_weighted(
        target: Counter[str],
        terms: Iterable[str],
        weight: float,
    ) -> None:
        for term in terms:
            target[term] += weight

    def _signature(self, synset: Any, target_lemma: str) -> Counter[str]:
        cache_key = str(synset.id), target_lemma
        cached = self._signature_cache.get(cache_key)
        if cached is not None:
            return cached
        signature: Counter[str] = Counter()
        self._add_weighted(
            signature,
            self._terms(str(synset.definition() or "")),
            2.0,
        )
        for example in synset.examples() or []:
            self._add_weighted(signature, self._terms(str(example)), 3.0)
        for lemma in synset.lemmas() or []:
            self._add_weighted(
                signature,
                self._terms(str(lemma).replace("_", " ")),
                1.0,
            )
        try:
            related = sorted(
                synset.get_related(),
                key=lambda item: str(item.id),
            )[: self.related_limit]
        except Exception:
            related = []
        for relation in related:
            self._add_weighted(
                signature,
                self._terms(str(relation.definition() or "")),
                1.0,
            )
            for example in relation.examples() or []:
                self._add_weighted(
                    signature,
                    self._terms(str(example)),
                    1.0,
                )
            for lemma in relation.lemmas() or []:
                self._add_weighted(
                    signature,
                    self._terms(str(lemma).replace("_", " ")),
                    0.5,
                )
        for term in self._terms(target_lemma):
            signature.pop(term, None)
        self._signature_cache[cache_key] = signature
        return signature

    def resolve_tagged(
        self,
        text: str,
        tagged_tokens: Sequence[tuple[str, str]],
    ) -> list[SenseAssignment]:
        context = set(self._terms(text))
        assignments: list[SenseAssignment] = []
        seen: set[tuple[str, str]] = set()
        for token, tag in tagged_tokens:
            pos = "n" if str(tag).startswith("NN") else "v" if str(tag).startswith("VB") else None
            if pos is None or len(token) <= 2 or not token.isalpha():
                continue
            lemma = self._lemma(token, pos)
            if lemma is None or (lemma, pos) in seen:
                continue
            seen.add((lemma, pos))
            candidates = self._synsets(lemma, pos)
            if not candidates:
                continue
            local_context = context - set(self._terms(lemma)) - set(self._terms(token))
            scored = sorted(
                (
                    (
                        math.fsum(
                            self._signature(candidate, lemma).get(term, 0.0)
                            for term in local_context
                        ),
                        str(candidate.id),
                    )
                    for candidate in candidates
                ),
                key=lambda item: (-item[0], item[1]),
            )
            best_score, best_id = scored[0]
            runner_up = scored[1][0] if len(scored) > 1 else None
            if len(scored) == 1:
                method = "unique_candidate"
            elif (
                best_score >= self.minimum_score
                and best_score - float(runner_up or 0.0) >= self.minimum_margin
            ):
                method = "extended_lesk"
            else:
                continue
            assignments.append(
                SenseAssignment(
                    lemma=lemma,
                    pos=pos,
                    synset_id=best_id,
                    score=float(best_score),
                    runner_up_score=(float(runner_up) if runner_up is not None else None),
                    candidate_count=len(scored),
                    resolution_method=method,
                )
            )
        return assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrpc-dir", type=Path, default=DEFAULT_MRPC_DIR)
    parser.add_argument("--wn-data", type=Path, default=DEFAULT_WN_DATA)
    parser.add_argument("--nltk-data", type=Path, default=DEFAULT_NLTK_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-score", type=float, default=1.0)
    parser.add_argument("--minimum-margin", type=float, default=1.0)
    parser.add_argument("--related-limit", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_score < 0.0 or args.minimum_margin < 0.0:
        raise OfflineWordNetError("score and margin thresholds must be non-negative")
    if args.related_limit < 0:
        raise OfflineWordNetError("related-limit must be non-negative")

    import nltk
    import wn
    from nltk.stem import PorterStemmer
    from wn.morphy import Morphy

    wn.config.data_directory = args.wn_data.resolve()
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    wordnet = wn.Wordnet(OEWN_VERSION)
    if len(wordnet.synsets()) != 120_630:
        raise OfflineWordNetError(
            "OEWN 2024 inventory does not contain the expected 120,630 synsets"
        )

    texts, mrpc_evidence = _load_mrpc_texts(args.mrpc_dir.resolve())
    tokenized = [WORD_RE.findall(text) for text in texts]
    tagged_sentences = nltk.pos_tag_sents(tokenized)
    resolver = OfflineOEWNResolver(
        wordnet,
        Morphy(wordnet),
        PorterStemmer(),
        minimum_score=args.minimum_score,
        minimum_margin=args.minimum_margin,
        related_limit=args.related_limit,
    )

    sentences: dict[str, Any] = {}
    method_counts: Counter[str] = Counter()
    resolved_sentence_count = 0
    assignment_count = 0
    for index, (text, tagged) in enumerate(
        zip(texts, tagged_sentences, strict=True),
        start=1,
    ):
        assignments = resolver.resolve_tagged(text, tagged)
        senses: dict[str, list[str]] = {}
        for assignment in assignments:
            senses.setdefault(assignment.lemma, []).append(assignment.synset_id)
            method_counts[assignment.resolution_method] += 1
        for lemma in senses:
            senses[lemma] = sorted(set(senses[lemma]))
        text_sha256 = _sha256_text(text)
        sentences[text_sha256] = {
            "text_sha256": text_sha256,
            "senses": senses,
            "assignments": [asdict(item) for item in assignments],
        }
        if assignments:
            resolved_sentence_count += 1
            assignment_count += len(assignments)
        if index % 250 == 0 or index == len(texts):
            print(
                f"Resolved {index}/{len(texts)} sentences; assignments={assignment_count}",
                flush=True,
            )

    database_path = Path(wn.config.database_path)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "resolver": {
            "resolver_id": RESOLVER_ID,
            "minimum_score": args.minimum_score,
            "minimum_margin": args.minimum_margin,
            "related_limit": args.related_limit,
            "pos_policy": "nltk_nouns_and_verbs",
            "production_equivalent": False,
            "production_difference": (
                "lexical extended-Lesk approximation; production uses Qwen "
                "sentence embeddings against scoped SynsetExample HNSW points"
            ),
        },
        "wordnet": {
            "lexicon": OEWN_VERSION,
            "wn_package_version": wn.__version__,
            "synset_count": len(wordnet.synsets()),
            "database_path": str(database_path.resolve()),
            "database_sha256": _sha256_file(database_path),
        },
        "mrpc": mrpc_evidence,
        "coverage": {
            "sentence_count": len(texts),
            "resolved_sentence_count": resolved_sentence_count,
            "resolved_sentence_rate": (resolved_sentence_count / len(texts) if texts else 0.0),
            "sense_assignment_count": assignment_count,
            "mean_assignments_per_sentence": (assignment_count / len(texts) if texts else 0.0),
            "resolution_method_count": dict(sorted(method_counts.items())),
        },
        "sentences": sentences,
    }
    _atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload["coverage"], indent=2, sort_keys=True))
    print(f"Sense cache: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
