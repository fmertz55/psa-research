# MRPC PSA Top-32 Candidate Analysis

**Date:** 2026-07-30

**Status:** Completed offline diagnostic

**Primary report:**
`reports/mrpc_psa_top32_analysis_2026-07-30/psa_top32_analysis.json`

**Candidate census:**
`reports/mrpc_psa_top32_analysis_2026-07-30/candidate_census.jsonl.gz`

## Outcome

The full 115-dimensional PSA top-32 route is healthy. It supplies essentially
complete positive recall and should remain a broad nomination stage. The
native gate performs almost all bulk rejection. The unresolved problem is
combining relation-sensitive evidence after the gate.

Plain ordered WordNet synsets and De Bruijn-style bigram overlap contain a
real positive signal, but they do not add held-out discrimination beyond PSA
rank, PSA similarity, and native cosine. The follow-on design decision dated
2026-07-31 retires De Bruijn evidence and learned truth combination from the
paraphrase verifier. The next representation is a deterministic proposition
graph with proof-producing, versioned rewrite rules.

## Scope and evidence

The analysis used:

- 3,393 unique MRPC sentences,
- the pinned Qwen 4,096-dimensional embedding cache,
- the pinned 115-star PSA frame,
- exact full-signature cosine top-32 retrieval,
- the cold `0.77` and calibration-learned `0.728510` native gates,
- the current surface verifier probes,
- the Qwen-resolved OEWN 2024 sense cache, and
- ordered noun/verb synset sequences reconstructed with pinned OEWN and NLTK.

It evaluated 108,576 directed candidate edges. Of those:

| Relation status | Directed edges |
| --- | ---: |
| Official MRPC positive | 2,287 |
| Official MRPC negative | 958 |
| Unlabeled neighbor | 105,331 |

An unlabeled neighbor is not a negative. MRPC labels only the published pair
inventory; it does not adjudicate arbitrary neighbors.

The ordered-sense cache covers all 3,393 sentences. A total of 179 sentences
have no resolved noun/verb sequence. The sense resolver is an offline
approximation of production WSD, so its outputs are evidence rather than
ground truth.

## PSA nomination

On the held-out 1,110 official pairs:

| Direction rule | Positive recall | Positive misses |
| --- | ---: | ---: |
| Published forward direction | 99.73% | 2 |
| Reverse direction | 99.87% | 1 |
| Either direction | 100.00% | 0 |
| Both directions | 99.60% | 3 |

For nominated official-positive directed edges, the median and 90th-percentile
rank are both 1. The corresponding official-negative median is also 1, but
its 90th-percentile rank is 11.3. Rank is useful evidence, but rank 1 alone
does not establish equivalence.

The top-32 neighborhood is deliberately broad. Its unlabeled-neighbor native
cosine median is only `0.3867`. The learned native gate passes 1.37% of
unlabeled neighbors, so it removes 98.63% before semantic verification. This
is the intended division of work.

## Gate and verifier census

The learned gate passes:

- 95.50% of nominated official-positive directed edges,
- 76.30% of nominated official-negative directed edges, and
- 1.37% of unlabeled directed neighbors.

There are 147 nominated official-positive directed pipeline failures:

- 103 fall below the learned native gate;
- 44 pass the gate but are rejected by the WordNet verifier.

There are 696 nominated official-negative directed edges accepted by the
learned-gate plus production-default verifier path. These are the primary
known false-positive analysis cohort.

### WordNet

After the learned gate, the current WordNet verifier passes 97.99% of known
positive edges and 95.21% of known negative edges. It rejects 44 known
positives while rejecting 35 known negatives. At pair level, adding the
available WordNet gate reduces held-out forward F1 from the earlier
gate-only `0.8369` diagnostic to `0.8331`.

The rejected-positive examples expose brittle WSD rather than proposition
conflicts. They include near-identical pairs such as `find` versus `found`,
`50 per cent` versus `half-owned`, and identical quoted propositions with
different attribution surfaces. WordNet disagreement must remain one input
to an evidence combiner rather than an independent hard rejection.

### Surface probes

Strict numeric and polarity enforcement rejects 371 known-positive directed
edges that the production-default verifier accepts. Numeric normalization
also confuses equivalent forms such as `3 to 5 percent` versus `three percent
to five percent`, `35 per cent` versus `35 percent`, and comma-formatted
quantities.

The strict surface path passes 81.00% of learned-gated positive edges and
67.03% of learned-gated negative edges. That separation is informative, but
the positive collateral is too large for a hard gate.

### Directionality

At pair level, requiring the learned-gate and production verifier to pass in
both top-32 directions scores held-out F1 `0.83665`, compared with `0.83313`
for the published forward direction. It removes seven false positives without
losing another true positive in this run.

Mutual nomination and mutual verification should therefore be explicit
features. They are not universal hard requirements because top-k
neighborhoods can be asymmetric.

## Ordered WordNet and De Bruijn evidence

The current shadow representation uses ordered noun/verb synsets and unique
sliding bigrams. Its candidate-level medians are:

| Feature | Official positive | Official negative | Unlabeled |
| --- | ---: | ---: | ---: |
| Unigram containment | 0.7500 | 0.5714 | 0.0000 |
| Bigram containment | 0.2500 | 0.0000 | 0.0000 |
| Bigram Jaccard | 0.0833 | 0.0000 | 0.0000 |
| Ordered-sequence ratio | 0.5714 | 0.4000 | 0.0000 |

The signal is genuine, but sparse: at least 25% of positive candidates have
zero bigram overlap. Active/passive alternation, argument movement,
morphology, attribution, and WSD errors destroy raw adjacency even when the
proposition is stable.

## Evidence-combination shadow

A small logistic shadow was fitted on calibration directed candidates only.
Its purpose was feature-family ablation, not release scoring:

| Feature families | Held-out directed-candidate F1 |
| --- | ---: |
| Learned gate only | 0.83930 |
| Retrieval features | **0.84412** |
| Retrieval + WordNet | 0.84348 |
| Retrieval + surface probes | 0.84351 |
| Retrieval + raw De Bruijn | 0.84310 |
| All current features | 0.84236 |

PSA rank, PSA similarity, and native cosine explain the available lift. Raw
synset adjacency does not yet add information after those features are known.
This negative result rules out promoting the old shadow implementation as-is.
The subsequent design decision also retires grammar-role De Bruijn evidence
from paraphrase verification; the useful role information is represented
directly as graph edges instead.

## Failure shapes

Accepted known negatives concentrate in four classes:

1. numeric or quantity changes inside otherwise identical sentences;
2. added or removed propositions, denials, and attributions;
3. changed entity roles or relative-clause attachment;
4. high lexical identity with a narrow referential distinction.

These are precisely the cases for which a role-canonical proposition graph can
add information:

```text
AGENT -> PREDICATE -> PATIENT
ATTRIBUTION_SOURCE -> CLAIM
QUANTITY -> UNIT -> ROLE
NEGATION -> PREDICATE
TIME -> EVENT
```

## Decision

Keep:

- full PSA signature top-32 nomination,
- the model-specific learned native candidate gate,
- rank, similarity, and mutual-direction features,
- WordNet and surface probes as shadow evidence.

Do not:

- narrow PSA retrieval to repair verifier precision,
- treat an unlabeled neighbor as a negative,
- use WordNet disagreement as an independent veto,
- use raw ordered-synset bigram overlap as an independent gate,
- let grammar-pattern popularity determine semantic weight.

Build next:

- dependency and semantic-role canonicalization into a proposition graph;
- explicit event, entity, quantity, temporal, location, attribution, modality,
  polarity, and coreference nodes, edges, and properties;
- deterministic, versioned graph rewrite rules grounded in ETPC paraphrase
  operations;
- complete-coverage proofs that reject or abstain on every unlicensed residual
  difference; and
- a per-operation MRPC audit with `verified_paraphrase`, `do_not_merge`, and
  `abstain_and_keep_free_range` outcomes.

Because this analysis inspected the current held-out partition in detail, the
next tuned verifier needs a newly locked blind evaluation partition.
