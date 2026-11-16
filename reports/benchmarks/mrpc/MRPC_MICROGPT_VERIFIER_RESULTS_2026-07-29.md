# MRPC MicroGPT Verifier Results - 2026-07-29

## Outcome

The current paraphrase pipeline now runs a microGPT policy that ranks
verification probes. The policy does not choose `paraphrase` or
`not-paraphrase`, and an untrained policy cannot override a hard verifier.

The five-order MRPC replay remains effectively unchanged:

| Evaluation | F1 | Precision | Recall | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Qwen cosine at 0.77 | 0.835287 | 0.775487 | 0.905080 | 677 | 196 | 71 | 166 |
| Current pipeline, five-order majority | 0.836319 | 0.777268 | 0.905080 | 677 | 194 | 71 | 168 |

The current pipeline therefore removes two net false positives and changes no
net false negatives relative to raw Qwen. This is a gain of only 0.001032 F1.
It is not evidence that the new bandit improves paraphrase accuracy.

Per-order F1 ranged from 0.834879 to 0.836634, with mean 0.835599 and population
standard deviation 0.000691. Twenty-six held-out pairs were order-sensitive.

## What was added

- Numeric, polarity, modality, entity, and WordNet verification probes.
- A microGPT switch that ranks only applicable probes.
- Hard rejection for an actual WordNet sense conflict.
- Shadow-only fact conflicts by default.
- A complete JSON verification trace on durable membership evidence.
- Exact tenant/workspace namespace isolation for learned switch state.
- Redis persistence across service restarts.
- A distributed feedback lock and adjudication idempotency markers.
- Explicit rewards from human merge/split or gold benchmark adjudications.
- Rejection of generic thumbs-up/down feedback for verifier training.
- Medoid sentence text for group verification instead of an LLM-generated
  group display name.
- A bounded inference cache keyed by switch training step and evidence context.

The two opt-in enforcement flags remain off:

```text
PARAPHRASE_VERIFICATION_REQUIRE_COMPLETE=false
PARAPHRASE_VERIFICATION_ENFORCE_FACT_CONFLICTS=false
```

## Why fact mismatches are not hard rejections

Across the five ingestion orders, the verifier evaluated 7,231 winning
candidates. WordNet was unavailable for all of them. The surface probes
reported the following conflicts:

| Probe | Conflicts | Conflict rate |
| --- | ---: | ---: |
| Numeric | 1,488 | 20.58% |
| Entity | 1,318 | 18.23% |
| Polarity | 272 | 3.76% |
| Modality | 209 | 2.89% |

On the 1,110 held-out MRPC pairs, a hard numeric rejection would remove 60
false positives but also 95 true positives. It would reduce F1 to 0.795082.
The other individual surface probes also reduce F1 when used as hard gates:

| Hard-rejection probe | Resulting F1 |
| --- | ---: |
| Numeric | 0.795082 |
| Polarity | 0.824197 |
| Modality | 0.818640 |
| Entity surface mismatch | 0.752909 |

MRPC considers many sentences paraphrases even when one sentence adds or
changes a fact. These probes are therefore evidence for adjudication, not
independent truth tests.

## Feedback contract

Verifier learning accepts a durable
`sentence_node_service:paraphrase_verification_feedback` stream event with:

```json
{
  "tenant_id": "canonical tenant",
  "workspace_id": "canonical workspace",
  "semantic_namespace_id": "ns_<sha256>",
  "source": "human_merge_split_adjudication",
  "adjudication_id": "stable unique id",
  "probe_utility_reward": 1.0,
  "verification_trace": {}
}
```

`probe_utility_reward` describes whether the selected first probe was useful
to the adjudicator. It does not encode the pair's paraphrase class. Valid
rewards are finite, non-zero values from -1 through 1. The other accepted
source is `gold_benchmark_adjudication`.

## Reproduction

```bash
PYTHONPATH=app python3 scripts/research/mrpc/run_mrpc_current_pipeline_benchmark.py \
  --cache temp/qwen_typed_runs/30399991644/qwen_embeddings.sqlite3 \
  --output-dir reports/mrpc_microgpt_verifier_2026-07-29
```

Evidence:

- Cache rows: 4,340
- MRPC unique sentences covered: 3,393 of 3,393
- Native model: `Qwen/Qwen3-Embedding-8B`
- Native dimension: 4,096
- PSA frame: pinned `v2_115stars`
- Report ID:
  `mrpc-current-pipeline:b125d7266bccbb7b29a0c23dc943fe778af250e6d9162a1e3e994401206ad775`

## Interpretation

The application still makes substantially more false-positive than
false-negative errors: 194 versus 71 in the five-order majority result.
The new bandit creates a safe, persistent route for learning which evidence to
surface first. It does not yet solve those false positives. The next verifier
accuracy experiment should add stronger relational evidence, such as
predicate-argument, temporal, quantity-role, and calibrated NLI probes, and
validate any enforcement rule on held-out data before enabling graph
membership mutations.
