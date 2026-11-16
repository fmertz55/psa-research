# MRPC Current-Pipeline Benchmark Results

**Date:** 2026-07-28
**Status:** **ROUTING REGRESSION REPAIRED; PAIR VERIFIER AUDIT SUPERSEDES THE
RAW-QUALITY VERDICT**
**Verdict:** Global bounded native retrieval repaired the July candidate-loss
regression and restored the membership lifecycle to 83.56% mean held-out MRPC
F1. That does not establish healthy paraphrase classification. The subsequent
raw-Qwen audit found 196 false positives, 71 false negatives, only 45.86%
specificity, and a February 2026 cosine-only verifier rewrite. See
`MRPC_RAW_QWEN_FAILURE_AUDIT_2026-07-29.md`.

## Post-Repair Validation

The candidate router now searches globally within the exact semantic namespace
for up to 32 free sentences and 32 true groups. The radius-one PSA neighborhood
remains available only as a non-excluding `psa_local` tie-break prior. The
native threshold, top-eight verifier limit, high-mass attractors, durable
free-range policy, and minimum two-member group invariant are unchanged.

| Evaluation | F1 | Precision | Recall |
|---|---:|---:|---:|
| Corrected app route, five-order mean | **83.56%** | 77.70% | 90.37% |
| Corrected app route, majority vote | **83.63%** | 77.73% | 90.51% |
| Raw Qwen pair cosine at production `0.77` | 83.53% | 77.55% | 90.51% |

The five corrected ingestion orders range from 83.49% to 83.66% F1 with
population standard deviation 0.07 percentage points. Focused database,
sentence-service, and replay tests pass 24/24, including an explicit
cross-cell paraphrase regression test.

- Corrected report:
  `reports/mrpc_global_native_routing_2026-07-29/mrpc_current_pipeline_report.json`
- Report ID:
  `mrpc-current-pipeline:65379bb3d03465272ac4349f83bdb676c05d26117f018b8924ace756c79d27f1`

## Valid Rerun

GitHub Actions run `30399991644` executed commit `6c2ce17e` against a fresh
protected `Qwen/Qwen3-Embedding-8B` endpoint. The job embedded the repaired,
source-joined MRPC corpus from scratch, uploaded the cache, and paused the
endpoint successfully.

| Evidence | Result |
|---|---:|
| Official test pairs | 1,725 |
| Unique source-joined sentences | 3,393 |
| Held-out pairs | 1,110 |
| Held-out positives | 748 |
| Held-out negatives | 362 |
| Fresh cache rows | 4,340 |
| Cache hits | 0 |
| Qwen wall time | 71.27 seconds |
| Production native threshold | `0.77` |
| Deterministic ingestion orders | 5 |

### Valid Held-Out Results

| Evaluation | F1 | Precision | Recall |
|---|---:|---:|---:|
| Raw Qwen pair cosine at production `0.77` | **83.53%** | 77.55% | 90.51% |
| Current app membership route, radius 1, majority vote | **44.27%** | 87.06% | 29.68% |
| Current app membership route, radius 1, order mean | 44.29% | 87.27% | 29.68% |
| Diagnostic full membership replay, radius 7, order mean | **82.96%** | 79.27% | 87.01% |

The radius-seven row is a controlled diagnostic, not a production policy
change. It uses the same native threshold, pinned frame, stateful free-sentence
and true-group lifecycle, candidate limits, verifier branch, and five ingestion
orders. Only the PSA neighborhood radius changes.

### Root Cause

The valid vectors reproduce the paraphrase matches observed during product
ingestion:

- the production `0.77` threshold accepts 90.51% of held-out positive pairs;
- exact global native retrieval finds 100% of positive partners within 32
  candidates and 99.20% within eight;
- the radius-one PSA neighborhood admits only 31.28% of positive partners;
- the joint radius-one-plus-native gate has 44.62% F1, almost exactly the
  final stateful membership result.

The radius-one app therefore discarded most true paraphrases before native
comparison. The embeddings and the `0.77` threshold were not the cause of that
specific July routing regression. They remain an inadequate downstream
equivalence rule: even after routing is repaired, raw Qwen produces 196 false
positives and rejects only 45.86% of held-out negatives.

Frame drift is also ruled out. Fresh and pinned guide-star vectors have
per-guide cosine of at least `0.9999968`, 98.14% of sentence routing labels
match exactly, and the fresh-guide radius-one positive admission rate is
31.42% versus 31.28% for the pinned production frame.

### Radius Sensitivity

This table applies the unchanged `0.77` native gate after each PSA
neighborhood gate:

| PSA radius | Positive admission | Pair-gate F1 |
|---:|---:|---:|
| 1 | 31.28% | 44.62% |
| 2 | 52.01% | 62.29% |
| 3 | 68.45% | 72.44% |
| 4 | 80.08% | 78.13% |
| 5 | 85.96% | 80.16% |
| 6 | 90.11% | 81.52% |
| 7 | 94.92% | 83.00% |
| 8 | 97.59% | 83.28% |

At radius seven, the complete stateful replay scores from 82.91% to 83.03% F1
across five ingestion orders, with a mean of 82.96%.

### Valid Artifacts

- Fresh embedding cache:
  `temp/qwen_typed_runs/30399991644/qwen_embeddings.sqlite3`
- Qwen MRPC report:
  `temp/qwen_typed_runs/30399991644/mrpc_report.json`
- Current app replay:
  `reports/mrpc_current_pipeline_2026-07-28_valid/mrpc_current_pipeline_report.json`
- Current app report ID:
  `mrpc-current-pipeline:dafbc175f6f0f4f08226c54c919f297bead5de480be74dd43774e979804aebe6`

## Retraction

The prior `1.36%` result below is invalid and must not be cited. The evaluator loaded
MRPC texts from legacy line-oriented exports that are corrupted in four ways:

- all or nearly all second-sentence rows have a literal trailing backslash;
- complete five-column TSV records occur inside the sentence files;
- those insertions shift the physical line numbers used as pair IDs;
- the derived mapping omits 84 of the 1,725 official test pairs.

The two July Qwen caches contain vectors for those malformed strings, not the
source MRPC sentences. After joining the official test IDs to the clean
sentence registry, the test split has 1,725 pairs and 3,393 unique sentences.
Each July cache is missing 1,932 of those clean vectors. The corrected
evaluator now fails closed instead of producing a score.

The production graph export independently contains 809 sentences assigned
across 237 paraphrase groups. Among stored group memberships, 96.7% of members
have a same-group neighbor at native cosine `>= 0.77`, with median best-match
cosine `0.820`. That evidence is incompatible with treating the malformed
replay as a valid measurement of the app.

The archived randomized Wikipedia ingestion independently records 3,218
sentence nodes, 165 paraphrase groups, and 514 `member_of` edges immediately
after `ai_wikipedia`; after `ai_agent_wikipedia`, it records 3,378 sentences,
175 groups, and 539 memberships. Those snapshots corroborate the production
export and the observed `0.77` behavior.

Everything below is retained only as an audit record of the withdrawn run.

## What Was Tested

This benchmark replays the current post-repair prose membership path:

1. Project each sentence through the pinned 115-guide-star PSA frame.
2. Assign its calibrated `RQ20` routing cell and expand by the production
   Chebyshev radius of one.
3. Search native vectors for up to 32 free sentences and 32 true
   multi-member groups in that PSA neighborhood.
4. Add up to eight groups with at least eight members as global attractors.
5. Keep the eight strongest candidates at the model-scoped Qwen native-cosine
   threshold of `0.77`.
6. Join a group, create a true two-member group from a verified free sentence,
   or leave the sentence durable and free-range.
7. Score published held-out MRPC pairs only after unlabeled routing completes.

This is not the retired singleton-group replay from 2026-07-27. Unmatched
sentences remain free-range, and only verified pairs create a
`ParaphraseGroup`.

## Evidence

| Evidence | Result |
|---|---:|
| MRPC pairs | 1,641 |
| Unique MRPC sentences | 3,260 |
| Held-out pairs | 1,088 |
| Held-out positives | 729 |
| Held-out negatives | 359 |
| Pinned PSA guide stars | 115 |
| Native dimensions | 4,096 |
| Independent local Qwen caches | 2 |
| Deterministic ingestion orders per cache | 5 |
| Total current-path replays | 10 |

The pinned guide-star artifact matches the app's required PSA frame hash
`4e57d83ddafc`, position hash `fe9486b3774d`, and calibration
`v2_115stars_q20_neo4j3424_20260611`.

## Final Held-Out Result

Every cache and ingestion order produced the same pair predictions.

| Metric | Result |
|---|---:|
| F1 | **1.36%** |
| Precision | **100.00%** |
| Recall | **0.69%** |
| Accuracy | 33.46% |
| True positives | 5 |
| False positives | 0 |
| False negatives | 724 |
| True negatives | 359 |
| Order-sensitive held-out pairs | 0 |

The result is not suitable for a product-quality or marketing claim.

## Gate Diagnosis

The two main gates independently discard nearly all positive targets.

| Held-out pair condition | Positive rate | Negative rate |
|---|---:|---:|
| Paired sentence is inside the radius-1 PSA neighborhood | **2.06%** | 0.84% |
| Pair native cosine is at least `0.77` | **2.06%** | 0.56% |
| Both gates pass | **0.69%** | 0.00% |

Positive and negative native-cosine distributions are also not separated:

| Native cosine | Positive pairs | Negative pairs |
|---|---:|---:|
| Median | 0.2263 | 0.2350 |
| 90th percentile | 0.3413 | 0.3352 |

The five joint-gate positive pairs are exactly the five final true positives.
The verifier is not the measured bottleneck in this offline replay. The local
cache has no object-to-synset annotations, so the current production verifier
uses its explicit `coverage_empty_side` acceptance branch.

## Structural Outcome

Across orders, the repaired policy created 361 to 367 true groups and left
2,468 to 2,476 sentences free-range. No singleton groups were fabricated.
Maximum group size varied from 8 to 19 with ingestion order, while all scored
held-out pair predictions remained invariant.

## Limitations

- The replay is serial. Runtime multi-sentence batches were not exercised.
- Native candidate selection used exact search as an oracle for Neo4j HNSW.
  This avoids blaming ANN approximation but can only overstate candidate
  quality.
- Object extraction and synset enrichment were not available in the local
  evidence cache.
- No live Neo4j, Redis, Docker, AWS, or embedding endpoint was used.
- The fixed production policy used no MRPC labels. The component split remains
  in place for comparison with the prior benchmark.

## Artifacts

- Current evaluator:
  `scripts/research/mrpc/run_mrpc_current_pipeline_benchmark.py`
- Raw report:
  `reports/mrpc_current_pipeline_2026-07-28/mrpc_current_pipeline_report.json`
- Report ID:
  `mrpc-current-pipeline:cfe727b95af67c5f1a1effce078932d3d1ab5dbc2eb1478cd0f2a9bccefa887f`

## Full MTEB Readiness

The repository has an MTEB runner, but it currently measures native, PSA-only,
and experimental hybrid vector representations directly against TEI. It does
not exercise the repaired free-range/group membership pipeline.

Before a full app-backed MTEB run:

1. Add an AIidea sentence-embedding API transport to the MTEB encoder.
2. Load and hash-check the pinned guide-star artifact instead of re-embedding
   guide text during the run.
3. Treat native Qwen and the complete 115-coordinate PSA signature as the
   production-relevant scorecards. Keep hybrid clearly experimental unless the
   product adopts it.
4. Install the locked benchmark dependencies and download the MTEB English v2
   datasets.
5. Run task-by-task into a resumable cache with `only-missing`, then aggregate
   only after every task completes.

The current workstation does not yet contain the `mteb` or `datasets` Python
packages, a Hugging Face dataset cache, or full-MTEB raw embedding caches. The
two local Qwen caches are complete for this MRPC replay, not for full MTEB.
