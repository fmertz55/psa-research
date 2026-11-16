#!/usr/bin/env python3
"""Build a reusable ordered-synset cache for the MRPC candidate census.

Run this script with the pinned offline WordNet environment:

    PYTHONPATH=app:. temp/wordnet-venv/bin/python \
      scripts/research/mrpc/build_mrpc_ordered_synset_cache.py

The source sense cache contains Qwen-resolved noun and verb senses. This
builder restores sentence order with the pinned OEWN/NLTK tokenizer and POS
tagger, then writes only immutable sentence-hash keyed synset sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.research.mrpc.analyze_mrpc_wordnet_debruijn_shadow import (
    DEFAULT_MRPC_DIR,
    DEFAULT_NLTK_DATA,
    DEFAULT_SENSE_CACHE,
    DEFAULT_WORDNET_DATA,
    _load_pairs,
    _ordered_synset_sequences,
    _sha256_file,
    _sha256_text,
)

DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "temp"
    / "mrpc_wordnet_qwen_replay"
    / "30475782422-neo4j-score"
    / "ordered_synset_sequences.json"
)
SCHEMA_VERSION = "aiidea_mrpc_ordered_synset_cache_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrpc-dir", type=Path, default=DEFAULT_MRPC_DIR)
    parser.add_argument("--sense-cache", type=Path, default=DEFAULT_SENSE_CACHE)
    parser.add_argument("--wordnet-data", type=Path, default=DEFAULT_WORDNET_DATA)
    parser.add_argument("--nltk-data", type=Path, default=DEFAULT_NLTK_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = _load_pairs(args.mrpc_dir.resolve())
    texts = sorted({text for pair in pairs for text in (pair.sentence1, pair.sentence2)})
    sense_path = args.sense_cache.resolve()
    sense_payload = json.loads(sense_path.read_text(encoding="utf-8"))
    sequences = _ordered_synset_sequences(
        texts=texts,
        sense_payload=sense_payload,
        wordnet_data=args.wordnet_data.resolve(),
        nltk_data=args.nltk_data.resolve(),
    )
    sentence_rows = {
        _sha256_text(text): {
            "text_sha256": _sha256_text(text),
            "synset_sequence": list(sequences[text]),
        }
        for text in texts
    }
    lengths = [len(sequence) for sequence in sequences.values()]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_sense_cache": {
            "path": str(sense_path),
            "sha256": _sha256_file(sense_path),
            "resolver": sense_payload.get("resolver"),
        },
        "wordnet": sense_payload.get("wordnet"),
        "sentence_count": len(sentence_rows),
        "zero_sequence_sentence_count": sum(not sequence for sequence in sequences.values()),
        "minimum_sequence_length": min(lengths, default=0),
        "maximum_sequence_length": max(lengths, default=0),
        "sentences": sentence_rows,
    }
    body = _canonical_json(payload)
    payload["cache_id"] = "mrpc-ordered-synsets:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    output_path = args.output.resolve()
    _atomic_json(output_path, payload)
    print(
        json.dumps(
            {
                "cache_id": payload["cache_id"],
                "sentence_count": payload["sentence_count"],
                "zero_sequence_sentence_count": (payload["zero_sequence_sentence_count"]),
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
