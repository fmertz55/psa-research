# Qwen Typed Hostile Field Test Results - 2026-07-27

> **MRPC retraction (2026-07-28):** All MRPC measurements in this document are
> invalid because the cached inputs came from corrupted line-oriented sentence
> exports. The hostile typed-corpus results and endpoint stability measurements
> are unaffected. MRPC references are retained only as an audit record.

## Run Identity

| Field | Value |
| --- | --- |
| Status | Complete |
| GitHub Actions run | [30282117353](https://github.com/aiidea-biz/aiidea/actions/runs/30282117353) |
| Report ID | `qwen-typed-field-report:362632816e80d00dce670f3a32c2c7ea03393c7ee73305f1183e4c72bdf9161f` |
| Corpus profile | `hostile_lexical_collision_v2` |
| Corpus ID | `qwen-typed-field:42009527808b4129b738` |
| Model | `Qwen/Qwen3-Embedding-8B` |
| Model revision | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| Native dimension | 4,096 |
| PSA frame | `qwen-psa-field:30f6613655c739e7a259` |
| Compute | Hugging Face optimized endpoint, AWS `nvidia-l4` x1 |
| TEI | 1.9.1, FP16, last-token pooling |
| Endpoint cleanup | Explicitly paused after artifact upload |

The complete downloaded artifact is retained locally under
`temp/qwen_typed_runs/30282117353/`. Its report and corpus content hashes
replay exactly, and its SQLite cache contains all 4,207 normalized vectors.

## Executive Result

The hostile run did its job: it removed the interpretation that the first
typed retrieval result was uniformly semantic.

Formula-to-formula retrieval survived the adversarial labels extremely well.
Code-to-code retained useful shortlist recall but was strongly attracted to
the planted wrong function names. The cross-modal routes were substantially
weaker, particularly formula-to-text.

The verifier safety boundary held completely:

- all 400 expected-pair isolated outcomes were exact;
- all 213 expected-pair end-to-end invocations were exact;
- a post-run replay verified every native top-8 candidate, including 2,924
  wrong targets, and accepted no wrong target;
- no graph writes or membership actions occurred.

This is a successful architecture result and a deliberately mixed embedding
result. Qwen is useful as a high-recall candidate generator, but these results
do not support nearest-neighbor equivalence, unverified membership, or one
global cosine gate.

## Hostile Candidate Retrieval

Every source's visible label belongs to one specific wrong target. The
`Expected beats decoy` column measures direct native cosine, independent of
whether either candidate entered the PSA shortlist.

| Relation pair | Recall@32 | Recall@8 | Verification eligible | Expected beats decoy | Top 8 and beats decoy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code to code | 92% | 69% | 69% | 27% | 24% |
| Formula to formula | 100% | 91% | 91% | 83% | 80% |
| Code to text | 98% | 64% | 1% | 51% | 43% |
| Formula to text | 88% | 52% | 52% | 22% | 12% |
| **Total** | **94.5%** | **69%** | **53.25%** | **45.75%** | **39.75%** |

The code-to-text eligibility collapse is the existing relation-specific
`0.77` native-cosine floor doing exactly what it was designed to do: fail
closed. Only one expected code-to-text target crossed that floor. Recall@8 is
therefore the more useful geometry measurement for that row.

## Comparison With The First Constructed Corpus

These are different corpora, so the deltas measure sensitivity to the hostile
construction rather than a model regression.

| Relation pair | First Recall@8 | Hostile Recall@8 | Change |
| --- | ---: | ---: | ---: |
| Code to code | 95% | 69% | -26 points |
| Formula to formula | 83% | 91% | +8 points |
| Code to text | 87% | 64% | -23 points |
| Formula to text | 95% | 52% | -43 points |

The decisive same-kind true positives split sharply:

| True-positive challenge | Expected top 32 | Expected top 8 | Expected beats decoy | Decoy top 8 |
| --- | ---: | ---: | ---: | ---: |
| Alpha-renamed code | 38/40 | 28/40 | 2/40 | 21/40 |
| Binder-renamed polynomial formula | 40/40 | 39/40 | 39/40 | 7/40 |

The formula result is strong evidence that Qwen captured the algebraic
structure despite the zero-weight wrong label and changed bound-variable
names. The code result shows that exact callable-name overlap is a powerful
signal: the planted decoy usually had higher cosine. Even so, the semantic
target remained in native top 8 for 70% of code true positives, which is why a
shortlist followed by deterministic verification remains viable.

## Category Results

| Challenge category | Cases | Recall@32 | Recall@8 | Expected beats decoy |
| --- | ---: | ---: | ---: | ---: |
| Code true positive, alpha-renamed | 40 | 95% | 70% | 5% |
| Formula true positive, binder-renamed polynomial | 40 | 100% | 97.5% | 97.5% |
| Code hard negative, comparator | 20 | 100% | 95% | 20% |
| Code hard negative, operator | 20 | 70% | 20% | 5% |
| Formula hard negative, sign | 20 | 100% | 100% | 80% |
| Formula hard negative, exponent | 20 | 100% | 100% | 85% |
| Code mandatory abstention, reordering | 20 | 100% | 90% | 100% |
| Formula mandatory abstention, unresolved domain | 20 | 100% | 60% | 55% |
| Cross-modal claim-required abstention | 200 | 93% | 58% | 36.5% |

Hard-negative retrieval is useful: a close but contradictory target must enter
the shortlist before the verifier can reject it. Missing a hard negative is
safe in this read-only test, but it reduces the observable coverage of the
verification boundary.

## Exhaustive Native Top-8 Verifier Audit

The primary report verifies every expected pair in isolation and verifies the
expected pair end to end when it passes the route's gate. After the run, the
retained cache was replayed through the same router and bounded verifier for
every native top-8 candidate, not just the expected target.

| Relation pair | Top-8 candidates | Wrong candidates | Wrong rejected | Wrong abstained | Wrong accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code to code | 800 | 731 | 597 | 134 | 0 |
| Formula to formula | 800 | 709 | 561 | 148 | 0 |
| Code to text | 800 | 736 | 0 | 736 | 0 |
| Formula to text | 800 | 748 | 0 | 748 | 0 |
| **Total** | **3,200** | **2,924** | **1,158** | **1,766** | **0** |

The planted decoy itself entered native top 8 in 95 cases. When each of the 400
planted source-decoy pairs was checked directly, the verifier:

- rejected all 100 code decoys;
- rejected 80 formula decoys and abstained on the 20 unsafe-domain sources;
- abstained on all 200 cross-modal decoys because no source-backed claim
  artifact exists;
- accepted none.

This does not estimate production false-accept probability because the corpus
is constructed. It does demonstrate that the current composition fails safe
on this specific adversarial family, including cases where the embedding ranks
the wrong label first.

## Bounded Verifier Results

| Relation pair | Isolated exact outcomes | End-to-end expected invocations | Exact when invoked |
| --- | ---: | ---: | ---: |
| Code to code | 100/100 | 69 | 100% |
| Formula to formula | 100/100 | 91 | 100% |
| Code to text | 100/100 | 1 | 100% |
| Formula to text | 100/100 | 52 | 100% |
| **Total** | **400/400** | **213** | **100%** |

The 187 non-invoked expected pairs caused no action. All cross-modal verifier
outcomes remain abstentions until a source-backed claim artifact exists.

## Resource, Identity, And Stability

| Measurement | Result |
| --- | ---: |
| New embeddings | 4,207 |
| TEI requests | 132 |
| Qwen session time | 66.82 seconds |
| Median batch latency | 0.412 seconds |
| Maximum batch latency | 0.648 seconds |
| Complete workflow time | 2 minutes 30 seconds |
| Stability repeat sample | 32 |
| Minimum repeat cosine | 0.99999565 |
| Maximum coordinate drift | 0.00058485 |

MRPC produced exactly the same reported classifier and retrieval metrics as the
first run, including the held-out F1 of `0.80154` and the fixed-`0.77` F1 of
`0.04021`. That repeat confirms the earlier conclusion: raw MRPC cosine is not
a useful prose classification gate.

The endpoint-management record and TEI observation agree on the pinned Qwen
repository and revision. The uploaded artifact contained seven files and was
finalized before the workflow logged:
`Paused clarkdd55/aiidea-qwen-hostile-30282117353`.

## Decision

The rollout gate remains closed.

The result supports three narrower conclusions:

1. Keep the two-stage architecture: Qwen candidate generation followed by the
   bounded verifier. Do not convert native cosine directly into membership.
2. Treat formula-to-formula as the strongest relation-specific candidate for
   held-out calibration.
3. Redesign or ablate code and cross-modal embedding recipes before setting
   production gates. High-value tests include a masked-callable structural
   code view, a dual original-plus-masked code representation, and
   query-instructed cross-modal retrieval.

The next authority-producing phase is still at least 100 independently
authored, adjudicated, held-out cases per relation pair. Automatic shadow
invocation, durable decision storage, and membership enablement remain
separate reviewed rollout phases.
