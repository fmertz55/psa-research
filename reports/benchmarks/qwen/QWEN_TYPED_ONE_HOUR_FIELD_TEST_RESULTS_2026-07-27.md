# Qwen Typed Field Test Results - 2026-07-27

> **MRPC retraction (2026-07-28):** All MRPC measurements in this document are
> invalid because the cached inputs came from corrupted line-oriented sentence
> exports. The typed code/formula results and endpoint stability measurements
> are unaffected. The MRPC section is retained only as an audit record.

> Follow-up: the lexical-collision challenge is complete. Read
> [Qwen Typed Hostile Field Test Results](QWEN_TYPED_HOSTILE_FIELD_TEST_RESULTS_2026-07-27.md)
> before interpreting the optimistic development retrieval rates below.

## Run Identity

| Field | Value |
| --- | --- |
| Status | Complete |
| GitHub Actions run | [30279973693](https://github.com/aiidea-biz/aiidea/actions/runs/30279973693) |
| Report ID | `qwen-typed-field-report:1e9e70bb47ff5970ac7fc29153d037939dac706582506e553d3ab22670fdf619` |
| Corpus ID | `qwen-typed-field:8b1f12ba2ebee0a17670` |
| Model | `Qwen/Qwen3-Embedding-8B` |
| Model revision | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| Native dimension | 4,096 |
| PSA frame | `qwen-psa-field:02f46f6192e626e0f52b` |
| Compute | Hugging Face optimized endpoint, AWS `nvidia-l4` x1 |
| TEI | 1.9.1, FP16, last-token pooling |
| Endpoint cleanup | Explicitly paused after artifact upload |

The complete downloaded artifact is retained locally under
`temp/qwen_typed_runs/30279973693/`. The 68 MB SQLite artifact contains all
4,207 normalized vectors. It is an experiment cache, not a production
decision store.

## Executive Result

The experiment produced two sharply different findings.

1. **Typed routing is strong enough to justify a larger held-out calibration.**
   Across 400 constructed code, formula, and cross-text cases, the expected
   target reached PSA top 32 in 393 cases and native top 8 in 360 cases. The
   bounded verifier made the exact expected decision in all 400 isolated
   cases. Every verifier invocation on the end-to-end shadow path was also
   correct.
2. **Raw MRPC cosine is not a usable classification gate.** On the held-out
   sentence-component split, positive and negative cosine distributions almost
   completely overlapped. The F1-selected threshold degenerated to almost
   always predicting paraphrase. The resulting F1 is a label-prevalence effect,
   not evidence of useful separation.

The experiment therefore validates the typed shadow architecture while
rejecting the idea that one global raw-cosine threshold can calibrate prose,
code, and formula membership.

## Resource And Stability Results

| Measurement | Result |
| --- | ---: |
| New embeddings | 4,207 |
| TEI requests | 132 |
| Qwen session time | 75.87 seconds |
| Median batch latency | 0.474 seconds |
| Maximum batch latency | 1.042 seconds |
| Complete workflow time | 2 minutes 41 seconds |
| Stability repeat sample | 32 |
| Minimum repeat cosine | 0.99999744 |
| Median repeat cosine | 0.99999821 |
| Maximum coordinate drift | 0.00053482 |

The endpoint was fast enough that the one-hour budget was not remotely
binding. Repeat drift is extremely small at the vector level.

## Typed Candidate Retrieval

| Relation pair | Cases | Recall@32 | Recall@8 | Verification eligible |
| --- | ---: | ---: | ---: | ---: |
| Code to code | 100 | 98% | 95% | 95% |
| Formula to formula | 100 | 96% | 83% | 83% |
| Code to text | 100 | 99% | 87% | 80% |
| Formula to text | 100 | 100% | 95% | 95% |
| **Total** | **400** | **98.25%** | **90.00%** | **88.25%** |

The aggregate formula-to-formula rate needs context. Its true-positive
commutative and distributive cases reached top 8 in 38 of 40 cases, or 95%.
Its direct hard negatives reached top 8 in all 40 cases. Most attrition came
from the mandatory-abstention `x/x` cases, where not retrieving a candidate is
a safe no-action outcome.

The same distinction applies across the corpus:

| Same-kind category | Cases | Recall@32 | Recall@8 |
| --- | ---: | ---: | ---: |
| True positives | 80 | 100% | 96.25% |
| Hard negatives | 80 | 97.50% | 95.00% |
| Mandatory abstentions | 40 | 90.00% | 62.50% |

For code-to-text, seven paired targets reached top 8 but fell below the current
native cosine floor of `0.77`. Their cosines ranged from `0.7288` to `0.7637`.
This is evidence that the old floor may be too strict for this relation pair,
but the constructed corpus is not sufficient authority to lower it.

The generated source and target pairs also share case-specific identifiers,
variable suffixes, or numeric constants. This guarantees distinct targets and
tests the complete routing implementation, but it supplies lexical breadcrumbs
that can inflate paired-target recall. Treat the typed retrieval rates as an
optimistic development result. The held-out corpus must counterbalance or
remove shared identifier and constant cues and add lexically similar
wrong-target distractors.

## Typed Verifier Results

| Relation pair | Isolated exact outcomes | End-to-end invocations | Exact when invoked |
| --- | ---: | ---: | ---: |
| Code to code | 100/100 | 95 | 100% |
| Formula to formula | 100/100 | 83 | 100% |
| Code to text | 100/100 | 80 | 100% |
| Formula to text | 100/100 | 95 | 100% |
| **Total** | **400/400** | **353** | **100%** |

The cross-text verifier correctly abstained because source-backed claim
artifacts do not exist yet. A missing candidate or failed native gate also
causes no action. Consequently, none of the 47 non-invoked cases created an
unsafe judgment.

This is strong implementation evidence, not an unbiased estimate of production
precision. The verifier expectations were generated from visible constructed
templates and must be repeated on independently authored, held-out cases.

## MRPC Results

The sentence-component split contained 553 calibration pairs and 1,088
held-out pairs. Exact normalized sentence identities could not cross the split.

| Operating point | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Calibration-selected threshold `0.08614` | 67.00% | 99.73% | 80.15% | 66.91% |
| Fixed threshold `0.77` | 88.24% | 2.06% | 4.02% | 34.19% |

The apparently high F1 at `0.08614` is misleading. An all-positive classifier
would score approximately `0.80242` F1 on the same held-out labels, slightly
better than the measured `0.80154`. A post-run rank calculation gives held-out
AUROC of approximately `0.48735`, effectively random.

Held-out paired native-cosine medians were:

- positive pairs: `0.22644`
- negative pairs: `0.23497`

The candidate-routing formulation was also weak:

| MRPC held-out measurement | Result |
| --- | ---: |
| Positive paired-target Recall@32 | 3.43% |
| Positive paired-target Recall@8 after native rerank | 2.74% |
| Negative paired-target rate@32 | 4.18% |
| Negative paired-target rate@8 | 1.39% |

MRPC is a pair-classification corpus, not a retrieval corpus. Ranking each
first sentence against all 1,619 distinct second sentences introduces
unlabeled alternate matches, so these retrieval numbers should not be treated
as conventional benchmark scores. They do show that this raw symmetric recipe
does not provide a useful prose membership gate.

The [official Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
supports raw sentence similarity, while recommending task-specific query
instructions for retrieval. It reports that instructions commonly improve
downstream performance by roughly 1% to 5%. An instructed-query ablation is
warranted, but the present MRPC failure is too large to explain away solely as
a missing instruction.

## Rollout Decision

The rollout gate remains closed:

- no graph writes were performed;
- automatic shadow invocation is still disabled;
- production decision-ledger storage is still absent;
- shared-core and supporting-halo membership remain disabled;
- the typed corpus is constructed development evidence rather than
  independently held-out adjudication.

The result supports proceeding to the next calibration phase. It does not
support membership enablement.

## Recommended Next Experiment

1. Preserve this exact model revision and PSA frame as the development
   baseline.
2. Add an MRPC ablation with a task-specific instruction on the query side and
   raw documents on the target side.
3. Evaluate pair classification separately from retrieval. If a supervised
   pair classifier is tested, fit it only on sentence-component-isolated
   calibration data and report held-out results.
4. Build at least 100 independently authored and adjudicated held-out cases per
   typed relation pair. Counterbalance identifiers and constants, include
   alpha-renamed positives, and include collision distractors that share names
   and numerals while differing semantically.
5. Calibrate relation-specific retrieval and native gates. Do not reuse one
   cosine threshold across prose, code, formula, and cross-text relations.
6. Only after those gates pass, add automatic shadow invocation and the
   append-only production decision ledger. Membership enablement remains a
   later reviewed phase.
