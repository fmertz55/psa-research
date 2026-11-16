# MRPC Offline WordNet Results - 2026-07-29

## Outcome

MRPC can be run with WordNet senses while all aiidea services are stopped.
The benchmark now accepts a versioned, content-addressed sentence-to-sense
cache and applies the same shared-lemma sense-agreement gate used by the
production sentence service.

The result does not support using the current WordNet gate as a broad hard
paraphrase rejection:

| Evaluation | F1 | Precision | Recall | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Qwen cosine at 0.77 | 0.835287 | 0.775487 | 0.905080 | 677 | 196 | 71 | 166 |
| Current pipeline, no WordNet | 0.836319 | 0.777268 | 0.905080 | 677 | 194 | 71 | 168 |
| Scoped Qwen OEWN, Neo4j score 0.70 | 0.830519 | 0.780259 | 0.887701 | 664 | 187 | 84 | 175 |
| Unique-sense-only OEWN | 0.835600 | 0.777011 | 0.903743 | 676 | 194 | 72 | 168 |
| Extended-Lesk OEWN | 0.828105 | 0.780142 | 0.882353 | 660 | 186 | 88 | 176 |

All pipeline values are majority votes across the same five deterministic
ingestion orders.

The conservative unique-sense-only cache made one hard rejection in every
order. It removed no false positives and added one false negative relative to
the pipeline without WordNet. The wider extended-Lesk cache removed eight
false positives but added seventeen false negatives, reducing F1 by 0.008213.

The test therefore confirms that WordNet evidence is available and active.
It also shows that hard sense disagreement is too blunt for MRPC. WordNet
should remain a verifier feature or shadow recommendation until a
high-confidence contradiction rule is calibrated on held-out data.

## Scoped Qwen result

GitHub Actions run `30475782422` created a protected Hugging Face endpoint
for the pinned `Qwen/Qwen3-Embedding-8B` revision and embedded only the
OEWN points needed by MRPC:

| Scoped artifact | Count |
| --- | ---: |
| MRPC sentences | 3,393 |
| Gate-relevant lemma/POS pairs | 3,568 |
| Candidate synsets | 12,454 |
| Synset example/gloss points | 16,202 |
| Unique texts requested | 16,201 |
| Finite synset vectors | 16,199 |

The two persistently non-finite inputs were both `import` glosses:

- `oewn-02237504-v`: `import: transfer (electronic data) into a database or document`
- `oewn-02351132-v`: `import: bring in from abroad`

Every sentence/lemma context still had at least one finite candidate point.
The vector cache SHA-256 is
`4ba9dfa13df95352dc6dca94f1d3233522bfa1cfe8285578bfc43506e67c05d5`.
The workflow used 195.34 seconds of Qwen inference time and paused the
endpoint after artifact upload.

The first offline interpretation incorrectly compared raw cosine directly
to `0.70`, producing only 10 assignments. [Neo4j's cosine vector-index
score](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/#vector-index-cosine-similarity)
is bounded to `[0, 1]` as `(1 + raw cosine) / 2`. Production therefore treats
its `0.70` threshold as raw cosine `0.40`. The hash-validated offline replay
now applies that score contract and produces:

| Coverage measure | Result |
| --- | ---: |
| Sense assignments | 13,291 / 27,602 |
| Sentences with at least one assignment | 3,214 / 3,393 (94.72%) |
| Median best raw cosine | 0.396291 |
| Median best Neo4j score | 0.698146 |

Against the no-WordNet majority vote, the active Qwen WordNet gate changes
24 held-out pair predictions. It corrects 9 false positives, introduces 13
false negatives, and introduces 2 false positives through clustering
cascade effects. The net confusion-matrix change is 7 fewer false positives
and 13 more false negatives, reducing F1 by `0.005800`.

## What is local

The local offline inventory is:

```text
Lexicon:             oewn:2024
Synsets:             120,630
wn package:          1.1.0
Database:            temp/wordnet/wn.db
Database size:       107 MB
Database SHA-256:    ea29868a4c57e869c1a3ac42d29ffd444c948f53f5c9a11f5d01cb4c7520ef51
```

Two generated caches were tested:

| Cache | Sentence coverage | Assignments | SHA-256 |
| --- | ---: | ---: | --- |
| Extended Lesk | 3,340 / 3,393 (98.44%) | 17,978 | `a9e1cd3d520ad3b2fd92dba3bf3b4a4d1b6bb15a36f222540af3d72b9c38e124` |
| Unique sense only | 2,524 / 3,393 (74.39%) | 4,806 | `139fed9969fc83f2d80e288337b26a1d3439df95792909202cf55f2526469fe5` |

The checked application buckets do not currently contain WordNet or OEWN:

- `s3://aiidea-ui-uploads/ontology/` has BFO, CCO, IAO, RO, and FrameBase
  assets.
- `s3://aiidea-default-state/ontologies/` has
  `framebase-schema.ttl`.
- A key-name search for `wordnet`, `WordNet`, `oewn`, and `OEWN` returned no
  matches in either application data bucket.

No S3 objects were created or changed during this test.

## Resolver boundary

This is an offline WordNet test, but it is not a claim of production WSD
equivalence.

Production extracts explicit lemma/POS objects and embeds the full sentence
with Qwen. It queries the global approximate `SynsetExample` HNSW index for
128 candidates, filters those results by lemma and POS, and applies a `0.70`
Neo4j vector-index score threshold.

The offline builder instead uses NLTK noun/verb tags, OEWN Morphy
lemmatization, and deterministic extended Lesk over definitions, examples,
synonyms, and bounded one-hop relations. The unique-sense-only control avoids
the ambiguous Lesk decisions entirely. Both variants use the pinned
production lexicon and then exercise the production sense-agreement gate.

The scoped Qwen run uses the production point-text and score contracts, but
it exhaustively scores all MRPC-relevant candidates after lemma/POS scoping.
That is more complete than production's global approximate top-128 retrieval,
so the cache explicitly records `production_equivalent: false`.

## Reproduction

Build the regular cache:

```bash
temp/wordnet-venv/bin/python \
  scripts/research/mrpc/build_mrpc_wordnet_sense_cache.py
```

Build the unique-sense-only control:

```bash
temp/wordnet-venv/bin/python \
  scripts/research/mrpc/build_mrpc_wordnet_sense_cache.py \
  --minimum-score 1000000 \
  --minimum-margin 1000000 \
  --output temp/mrpc_wordnet/oewn_2024_unique_only_senses.json
```

Run the five-order MRPC replay:

```bash
PYTHONPATH=app python3 scripts/research/mrpc/run_mrpc_current_pipeline_benchmark.py \
  --cache temp/qwen_typed_runs/30399991644/qwen_embeddings.sqlite3 \
  --wordnet-sense-cache \
    temp/mrpc_wordnet/oewn_2024_extended_lesk_senses.json \
  --output-dir reports/mrpc_offline_wordnet_2026-07-29
```

Evidence reports:

- Extended Lesk report ID:
  `mrpc-current-pipeline:f1c22d7fc60f8539b4f9b6acf7faf37624581b5558c3fbe8a761fad03b371cab`
- Unique-sense-only report ID:
  `mrpc-current-pipeline:c2cb9d890346d3a02a0447ed3aedc54dc9b49745a73fa63f1601f73af4c756f4`
- Scoped Qwen/Neo4j-score report ID:
  `mrpc-current-pipeline:13a42391163422dce6708538cc60f8f6dfcfc4448ce6b1cc9313676bb848fe46`

The report records the resolver contract, database and cache hashes, exact
MRPC sentence-hash coverage, and the explicit
`production_equivalent: false` boundary.

Replay the downloaded vectors without an endpoint:

```bash
PYTHONPATH=app:. temp/wordnet-venv/bin/python \
  scripts/research/mrpc/build_mrpc_wordnet_qwen_sense_cache.py \
  --sentence-embedding-cache \
    temp/qwen_typed_runs/30399991644/qwen_embeddings.sqlite3 \
  --sentence-endpoint-metadata \
    temp/qwen_typed_runs/30399991644/endpoint.json \
  --replay-synset-vector-cache \
    temp/mrpc_wordnet_qwen_runs/30475782422/qwen_synset_vectors.sqlite3 \
  --replay-source-sense-cache \
    temp/mrpc_wordnet_qwen_runs/30475782422/qwen_wordnet_senses.json \
  --replay-scoped-manifest \
    temp/mrpc_wordnet_qwen_runs/30475782422/scoped_synset_manifest.json \
  --output-dir \
    temp/mrpc_wordnet_qwen_replay/30475782422-neo4j-score
```

## Recommended next experiment

Do not call a public WordNet endpoint once per sentence. Such an endpoint
would provide the lexical inventory, not production-quality contextual sense
selection, and would add network variability to a benchmark.

Keep the 107 MB OEWN database local for development. If it needs to be shared
between runners, upload a pinned source artifact or the content-addressed
database to an application asset prefix in S3 and verify its SHA-256 before
use. The more valuable follow-up is to precompute the production Qwen
`SynsetExample` vectors and run the same MRPC replay offline, with WordNet
conflicts first treated as bandit-ranked evidence rather than unconditional
rejections.
