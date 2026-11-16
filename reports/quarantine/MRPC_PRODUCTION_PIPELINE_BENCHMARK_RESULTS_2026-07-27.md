# MRPC Production-Pipeline Benchmark Results

**Date:** 2026-07-27  
**Status:** **WITHDRAWN - INVALID INPUTS**
**Verdict:** No MRPC score can be reported from this run.

## Retraction

The `4.03%` result below is invalid and must not be cited. This replay used
corrupted line-oriented MRPC exports rather than joining test sentence IDs to
the clean source registry. The exports contain trailing backslashes, embedded
complete TSV records, shifted line numbers, and only 1,641 of the 1,725
official test pairs.

The corrected source join produces 3,393 unique clean sentences. Each July
cache is missing 1,932 of their vectors, so the run cannot be recomputed from
those caches. The loader and offline benchmark now fail closed on that
incomplete evidence.

Everything below is retained only as an audit record of the withdrawn run.

## What Was Tested

This test replayed the production paraphrase route without the artificial
Qwen/PSA weighted fusion:

1. Compute the weighted, row-centered, L2-normalized 115D guide-star signature.
2. Search group signature centroids with HNSW and retain 32 candidates.
3. Rerank against each group's centroid, actual-member medoid, and two boundary
   representatives; retain eight.
4. Verify the remaining groups with exact native 4096D Qwen cosine at the
   production threshold of `0.77`.
5. Join the verified winner or create a singleton group.
6. After all unlabeled routing is complete, classify each held-out MRPC pair by
   whether its sentences resolved to the same group.

MRPC labels did not influence signatures, HNSW construction, group formation,
thresholds, candidate selection, reranking, or ingestion order.

The replay used the production constants and the production
`SemanticSignatureSummary`, `build_signature_summary`, and
`update_signature_summary` implementations.

## Evidence

The benchmark consumed two independent complete Qwen caches:

| Evidence | Result |
|---|---:|
| MRPC pairs | 1,641 |
| Unique MRPC sentences | 3,260 |
| Guide stars | 115 |
| Native dimensions | 4,096 |
| Calibration pairs | 553 |
| Held-out pairs | 1,088 |
| Held-out positives | 729 |
| Held-out negatives | 359 |
| Independent Qwen caches | 2 |
| Deterministic ingestion orders per cache | 5 |
| Total production replays | 10 |

Both caches use `Qwen/Qwen3-Embedding-8B` at declared revision
`1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`. Every MRPC sentence and guide
star was verified present by its content-addressed cache key.

## Final Held-Out Result

The five-order majority result was identical for both independent caches.

| Metric | Result |
|---|---:|
| F1 | **4.03%** |
| Precision | 93.75% |
| Recall | **2.06%** |
| Accuracy | 34.28% |
| True positives | 15 |
| False positives | 1 |
| False negatives | 714 |
| True negatives | 358 |

Across all ten individual replays, F1 ranged from 4.02% to 4.03%. Only one of
the 1,088 held-out pairs changed prediction across ingestion orders.

## Stage Diagnosis

Rates below are means across both caches and all five ingestion orders.

| Production stage | Positive target survives | Negative target survives |
|---|---:|---:|
| HNSW candidate set, top 32 | **4.55%** | 3.96% |
| Signature representative rerank, top 8 | **2.91%** | 2.03% |
| Native verifier at `0.77` | **2.06%** | 0.45% |
| Target selected | **2.06%** | 0.33% |

Two independent failures are present:

1. The 115D signature field does not place MRPC paraphrase targets reliably in
   the same local candidate neighborhood. More than 95% of positive targets are
   absent from the top 32.
2. The native `0.77` verifier threshold rejects approximately 98% of held-out
   positive MRPC pairs.

The held-out native-cosine distributions explain the verifier failure:

| Native cosine | Positive pairs | Negative pairs |
|---|---:|---:|
| Median | 0.2263 | 0.2350 |
| 90th percentile | 0.3413 | 0.3352 |
| Fraction at or above `0.77` | 2.06% | 0.56% |

On this corpus, raw Qwen cosine does not meaningfully separate the two classes,
and the fixed production threshold is far outside the main mass of either
distribution.

## HNSW Was Not the Failure

The benchmark used `hnswlib==0.8.0` with cosine distance, `M=16`,
`ef_construction=200`, and `ef_search=128`.

Across 1,300 audited searches:

| ANN audit | Result |
|---|---:|
| Exact best candidate present in HNSW top 32 | **100%** |
| Mean HNSW/exact top-32 set overlap | **99.98%** |

An exact-search oracle and HNSW also produced identical predictions for all
1,088 held-out pairs in the smoke comparison. The poor candidate recall is
therefore a property of the current signature geometry, not HNSW approximation.

## Qwen Pair-Cosine Control

A native-cosine threshold selected using calibration labels produced 80.15% F1
on held-out pairs. That number is not a competitive result: its threshold was
`0.0861`, it predicted 1,085 of 1,088 held-out pairs as paraphrases, and its
accuracy was only 66.91%. It mostly reflects MRPC's positive-class prevalence.

The direct native-cosine control at the production `0.77` threshold produced
4.02% F1, nearly identical to the complete production route.

## Interpretation

This result does not say that HNSW, guide-star visualization, or typed
code/formula routing is useless. It says something narrower and decisive:

> The currently configured prose paraphrase-membership route is not calibrated
> for Qwen3-Embedding-8B on MRPC and cannot support the claimed 95%+ MRPC F1
> target.

The earlier 80/20 Qwen/PSA benchmark was not a product benchmark. This replay
tests the actual funnel and supersedes it for claims about production
paraphrase resolution.

## Limitations

- Routing was serial and deterministic; transactional concurrency was not
  exercised.
- HNSW used `hnswlib`, not Neo4j's vector-index implementation. The exact-search
  audit shows that this implementation difference did not affect the measured
  failure.
- The repository-prepared MRPC pair graph is not a complete set of all semantic
  equivalence relationships among its sentences. Final metrics score only the
  published labeled pairs.
- This test used cached raw production-style sentence embeddings. It did not
  generate task-instructed replacement embeddings.

## Artifacts

- Evaluator:
  `scripts/research/mrpc/run_mrpc_production_pipeline_benchmark.py`
- Raw report:
  `reports/mrpc_production_pipeline/mrpc_production_pipeline_report.json`
- Report ID:
  `mrpc-production-pipeline:2fd060aecd1af93335c09f357033e39a502eaafda0e642d57a899b1de5a692ec`
