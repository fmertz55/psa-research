# Raw Qwen MRPC Failure Audit

**Date:** 2026-07-29
**Scope:** Raw `Qwen/Qwen3-Embedding-8B` pair cosine, before PSA routing or
paraphrase grouping
**Status:** Complete pair-level inventory; no MTEB work authorized or started

## Verdict

The current raw classifier is not a paraphrase verifier. It is a semantic
relatedness candidate generator followed by one global cutoff:

```text
paraphrase := cosine(sentence_a, sentence_b) >= 0.77
```

That rule has high positive recall but weak negative discrimination. On the
locked held-out MRPC split it accepts 90.51% of positives while rejecting only
45.86% of negatives. It makes 196 false-positive errors and 71 false-negative
errors. Its 83.53% F1 is only 3.01 percentage points above the 80.52% F1 of an
all-positive classifier because MRPC is 67.39% positive.

The July PSA repair did not cause this raw-Qwen behavior. The decisive
cosine-only rewrite happened on 2026-02-27. July changed candidate routing and
group lifecycle while retaining the inherited pair-truth rule. The repaired
application now routes candidates successfully, but it still has almost no
semantic verification after routing.

The historical "90+" result is not a valid counterexample. Its 100-pair
fixture is corrupted, its evaluator silently excludes missing pairs, and the
repository contains no persisted confusion matrix for that claim.

## Locked Result

| Metric | Raw Qwen at `0.77` |
| --- | ---: |
| Held-out pairs | 1,110 |
| Positive / negative | 748 / 362 |
| TP / FP / FN / TN | 677 / 196 / 71 / 166 |
| Precision | 0.775487 |
| Recall | 0.905080 |
| Specificity | 0.458564 |
| False-positive rate | 0.541436 |
| Accuracy | 0.759459 |
| Balanced accuracy | 0.681822 |
| F1 | 0.835287 |
| ROC AUC | 0.783212 |
| Average precision | 0.875602 |

The complete 267-error inventory is in:

- `reports/mrpc_raw_qwen_failure_audit_2026-07-29/raw_qwen_errors.tsv`
- `reports/mrpc_raw_qwen_failure_audit_2026-07-29/raw_qwen_failure_audit.json`

The report ID is
`mrpc-raw-qwen-failure-audit:000652718bdcaee442cdd9ebb6886f91e113a804914a7d1e2ff3c5fd971bf290`.

## What Was Ruled Out

### The current MRPC join is exact

The evaluator reads the 1,725 official test labels and sentence IDs, then
joins those IDs to clean text in `msr_paraphrase_data.txt`. The joined text
matches every official test sentence after removing the export terminator and
decoding its 38 literal CP1252 escapes.

The older 1,641-pair caches are malformed and were already retracted. They
cannot be used to diagnose model quality.

### Truncation is not the cause

The endpoint used `--auto-truncate`, but the longest of the 3,393 unique MRPC
sentences is only 35 audit tokens. Median length is 19. The model supports far
longer inputs, so no MRPC sentence is plausibly near its context limit.

### The model and vector contract are stable

- Model: `Qwen/Qwen3-Embedding-8B`
- Revision: `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`
- Dimension: 4,096
- Runtime:
  `ghcr.io/huggingface/text-embeddings-inference:89-1.9.1`
- TEI task: `feature-extraction`
- Request: raw sentence text sent directly to `/embed`
- Task instruction: none

The official Qwen implementation uses normalized last-token embeddings. The
endpoint and cached-vector behavior are consistent with that contract.

The absent task instruction remains a legitimate controlled experiment:
Qwen's model card says task-specific instructions can improve performance.
It is not an explanation for a sudden July regression because the existing
runner has always sent raw MRPC sentences.

### A better threshold cannot repair the classifier

The calibration-selected threshold is `0.743228`; on held-out data it lowers
F1 to `0.832143`.

Even an invalid held-out oracle that is allowed to see the answers selects
`0.776586` and reaches only `0.838107`, a gain of 0.002820. The score
distributions overlap too much for a scalar cutoff to solve the problem.

## Score Geometry

| Cosine band | Positive | Negative |
| --- | ---: | ---: |
| `<0.60` | 0 | 27 |
| `0.60-0.70` | 22 | 53 |
| `0.70-0.74` | 23 | 43 |
| `0.74-0.77` | 26 | 43 |
| `0.77-0.80` | 47 | 26 |
| `0.80-0.85` | 107 | 66 |
| `0.85-0.90` | 163 | 57 |
| `0.90-0.95` | 190 | 37 |
| `>=0.95` | 170 | 10 |

The median false-positive score is `0.856409`. Forty-seven false positives
score at least `0.90`, and ten score at least `0.95`. These are not threshold
edge cases.

The median false-negative score is `0.722867`. Twenty-two false negatives are
below `0.70`; only 26 are in the narrow `0.74-0.77` threshold band.

## Every Raw-Qwen Failure Mode

The modes below are not mutually exclusive. A pair can contain a number
change, a role change, and an added proposition simultaneously. Every error is
preserved at pair level with its sentence text, cosine, margin, surface
features, WordNet result, and observable tags.

### False positives: 196

#### 1. Factual-slot blindness

Qwen strongly represents the shared event template while underweighting the
values that determine whether the propositions are equivalent.

Observable counts:

| Signal | False positives |
| --- | ---: |
| Different numeric facts on both sides | 60 |
| Numeric facts on only one side | 17 |
| Polarity mismatch | 8 |
| Modality mismatch | 4 |

Examples:

- Pair 1508, cosine `0.977077`: 12 rebels killed versus 16, and six soldiers
  wounded versus one killed plus six wounded.
- Pair 1288, cosine `0.939789`: 21 arrests versus 99.
- Pair 937, cosine `0.926930`: stock rose 32% versus 38%.
- Pair 1387, cosine `0.841351`: one poll says Street leads; the other says
  Katz leads.
- Pair 1377, cosine `0.829271`: finasteride cohort at 18% versus placebo
  cohort at 24%.
- Pair 841, cosine `0.771225`: Treasury notes gained versus fell, with
  different yields.

This is the largest directly observable false-positive mode.

#### 2. Entity, role, and referent inversion

The same names and surrounding event dominate the vector even when grammatical
attachment changes who did what.

- Pair 1573, cosine `0.987384`: NBC is first in the demographic versus CBS is
  first.
- Pair 1135, cosine `0.971989`: the panel's two Democrats oppose Powell versus
  the panel's two other Republicans are also described.
- Pair 1020, cosine `0.847206`: Russin and his lawyer are separate people
  versus Basil Russin being the lawyer/cousin referent.

The surface entity probe marks 33 false positives, but it is an incomplete and
noisy proxy for this deeper predicate-argument problem.

#### 3. Partial entailment and added or omitted propositions

The pair shares one proposition, but one sentence adds a second proposition
that is not entailed by the other.

- Pair 32, cosine `0.972503`: the shared plot report plus an added denial.
- Pair 510, cosine `0.960803`: stop working at the hotel plus a demand that
  U.S. forces leave Iraq.
- Pair 1043, cosine `0.936430`: a similar attack plus an added al-Qaida
  attribution.

Large length asymmetry marks 32 false positives. Quote presence differs on 16,
and reporting markers differ on 32.

#### 4. Same topic or event, different claim

The sentences occupy the same event neighborhood but assert different
predicates or stages.

- Pair 1048, cosine `0.851865`: different victim sequence in a murder account.
- Pair 1681, cosine `0.871457`: accused and unrepentant versus standing trial.
- Pair 1699, cosine `0.813637`: leading an inquiry versus wanting to determine
  audit compliance.
- Pair 1606, cosine `0.773579`: different quotations from the same official.

This is the core reason cosine is a candidate signal rather than an
equivalence decision.

#### 5. Negation, direction, certainty, and attribution

- Pair 1695, cosine `0.926404`: a statement thanks the commission versus the
  claim that Bush did not discuss the issue in that statement.
- Pair 1622, cosine `0.913836`: pressure may rise versus will intensify, with
  additional claims.

The current regex probes only catch explicit markers. Implicit direction,
scope of negation, and source attribution require structured analysis.

#### 6. News-template dominance

Finance, polling, casualty, court, sports, and corporate-reporting sentences
reuse near-identical syntactic templates. Qwen encodes the event type and
lexical shell so strongly that changed participants, quantities, directions,
or outcomes can remain above `0.90`.

This is not confined to one topic. It is a general dense-embedding failure
mode for fine-grained proposition equivalence.

#### 7. MRPC annotation ambiguity and noise

Some official negatives are reasonable paraphrases under ordinary product
semantics:

- Pair 1470, cosine `0.961240`: the same engine-selection quote with a trailing
  rationale removed.
- Pair 1191, cosine `0.895919`: a speeding police van injured protesters.
- Pair 822, cosine `0.916819`: the same Altria downgrade.
- Pair 726, cosine `0.878352`: Apple sold more than one million songs in the
  first week.

These cases do not excuse the clear factual false positives above. They show
that MRPC mixes strict proposition equivalence with a looser
same-core-message policy. Any production verifier must define its own merge
semantics explicitly instead of inheriting every inconsistent MRPC label.

### False negatives: 71

#### 1. Coreference, named-to-generic substitution, and entity elision

- Pair 166, cosine `0.601984`: "Both" versus the named Bijani sisters.
- Pair 1662, cosine `0.632642`: named executives versus their company roles.
- Pair 689, cosine `0.684391`: "the author" versus Spitz.
- Pair 1340, cosine `0.700529`: "the plan" versus Sprint's named plan.

The surface entity detector marks 31 false negatives because valid
paraphrases often replace a name with a role, pronoun, or generic description.

#### 2. Reporting and quotation wrappers

- Pair 1218, cosine `0.625701`: the same quotation with its speaker attribution
  removed.
- Pair 681, cosine `0.674605`: the same quote with a different reporting
  wrapper.
- Pair 1588, cosine `0.697872`: a quoted fragment versus a sentence that
  introduces it.

Reporting markers differ in 21 false negatives. The model sometimes gives too
much weight to the wrapper and too little to the quoted proposition.

#### 3. Lexical, syntactic, and nominalization rewrites

- Pair 1151, cosine `0.674512`: agencies "have the final say" versus the deal
  "must get approval" from those agencies.
- Pair 1605, cosine `0.702968`: "the medicine" versus Erbitux, with the same
  study conclusion.
- Pair 186, cosine `0.700666`: Crohn's disease definition plus symptoms versus
  the symptom list.

These need alias resolution, predicate normalization, and entailment rather
than a lower global threshold.

#### 4. Preserved quantities expressed in different structures

- Pair 207, cosine `0.665253`: 19.5 million and 8.9 million with an explicit
  derived 120% increase versus the same totals without the percentage.
- Pair 1496, cosine `0.718922`: the same operating-income values rearranged.
- Pair 101, cosine `0.740604`: licensing revenue versus license sales with the
  same decline and value.

This explains why a blanket numeric-mismatch rejection destroys true
positives.

#### 5. Threshold-edge misses

Twenty-six false negatives are within `0.03` below the cutoff. These can be
recovered only if another verifier supplies positive evidence. Lowering the
threshold globally admits more negatives than it repairs.

#### 6. Positive-label contradictions and noise

Several official positives are not propositionally equivalent:

- Pair 1697, cosine `0.645353`: positive versus negative numbers compared with
  high versus low standards.
- Pair 1227, cosine `0.650261`: $2.3B expenses versus $4.4B R&D spending.
- Pair 1024, cosine `0.677531`: $21.6B sales versus Microsoft's $8.1B sales.
- Pair 1332, cosine `0.687139`: different stock moves and closing prices.
- Pair 1023, cosine `0.746646`: one alligator killed versus none killed.
- Pair 107, cosine `0.731317`: materially different Dow moves and closes.

A verifier that correctly rejects these pairs will look worse against the
published MRPC label. That conflict must be made explicit in any calibration
or product claim.

## Why the Existing Probes Do Not Fix It

Applied as independent hard rejections to raw-positive pairs:

| Probe | FP repaired | TP destroyed | Resulting F1 |
| --- | ---: | ---: | ---: |
| Numeric mismatch | 60 | 95 | 0.7940 |
| Polarity mismatch | 8 | 23 | 0.8226 |
| Modality mismatch | 4 | 27 | 0.8176 |
| Surface entity disjoint | 33 | 128 | 0.7521 |
| WordNet sense conflict | 8 | 13 | 0.8300 |
| Any observable conflict | 94 | 247 | 0.6719 |

The probes are useful evidence, but none is independently equivalent to
"not paraphrase." Quantity differences can be derived facts; names can be
replaced by roles; negation can sit outside the shared proposition; WordNet
coverage can be sparse or wrong.

The microGPT switch added on July 29 only orders applicable probes. It is not
trained to classify a pair, cannot create missing relational evidence, and
cannot override the hard-verifier contract. With fact enforcement disabled,
the five-order application majority improves over raw Qwen by only two false
positives and `0.001032` F1.

## The De Bruijn Verifier Question

The tracked De Bruijn implementation was added on 2026-04-18 for Stitch
response composition. It:

- extracts prompt surface symbols;
- builds sliding k-gram coverage windows;
- selects and orders response pieces.

It does not:

- convert SentenceNode text to ordered WordNet synsets;
- compare two sentences for paraphrase acceptance;
- consume PSA paired-target evidence as a merge verifier;
- run from the SentenceNode paraphrase path.

Repository history contains a design reference to "de Bruijn entropy
clustering," but no committed WordNet-to-De-Bruijn paraphrase verifier in any
branch or reflog inspected.

### Offline reconstruction

The proposed ordered-sense check was reconstructed from the scoped Qwen
WordNet cache without starting aiidea services or mutating Neo4j. It preserves
noun/verb sense order, then measures synset unigram overlap, De Bruijn bigram
overlap, and ordered-sequence similarity.

| Held-out feature | Positive median | Negative median |
| --- | ---: | ---: |
| Synset unigram Jaccard | 0.40 | 0.25 |
| Synset unigram containment | 0.75 | 0.50 |
| De Bruijn bigram Jaccard | 0.0714 | 0.00 |
| De Bruijn bigram containment | 0.25 | 0.00 |
| Ordered sequence ratio | 0.5714 | 0.40 |

The signal exists, but a calibration-only sweep chooses a zero rejection
threshold for every individual feature. Used alone as a hard gate, exact
sequence overlap loses more true positives than it repairs.

The existing unlabeled PSA signature top-8 paired-target condition is more
precise. As a shadow rejection on raw positives, it repairs eight false
positives and loses one true positive:

| Metric | Raw Qwen | Raw Qwen + PSA top-8 shadow gate |
| --- | ---: | ---: |
| TP / FP / FN / TN | 677 / 196 / 71 / 166 | 676 / 188 / 72 / 174 |
| F1 | 0.835287 | 0.838710 |

This is a useful input feature, not a complete solution. The full shadow
artifact is
`reports/mrpc_raw_qwen_failure_audit_2026-07-29/wordnet_debruijn_shadow.json`.

## Historical Reconstruction

### The invalid "90+" evidence

Commit `eb9198ea` on 2026-01-07 added `scripts/research/mrpc/quick_f1_eval.py` with the commit
message "Doug says 90+." That evaluator:

- uses only `sample_100_*`;
- reads prediction from same-group membership in whatever Neo4j state exists;
- silently drops a pair if either sentence is absent;
- reports no coverage-adjusted or missing-as-error score;
- has no persisted result artifact.

The fixture itself is unusable:

| Fixture check | Result |
| --- | ---: |
| Constructed pairs | 100 |
| Pairs matching any official MRPC pair | 1 |
| Claimed `source_pair_id` text matches | 1 |
| Fixture labels differing from the official source-row label | 53 |
| Fixture positives | 50 |

One fixture positive pairs a shuttle-funding sentence with an unrelated
SARS-quarantine sentence. The 90+ claim cannot establish a regression.

### What actually changed

1. `3a25808f` on 2026-01-07 added verb-contradiction penalties and
   entity-aware similarity.
2. `0447241f` lowered the default threshold from `0.82` to `0.77` and added the
   malformed 100-pair fixture.
3. `09d0ec66` later that day replaced the contradiction path with positive
   object-embedding and verb-SUMO boosts and lowered the final threshold to
   `0.73`.
4. `c5306ce2` on 2026-02-27 explicitly removed object boosts, verb-SUMO boosts,
   role/Jaccard logic, and the accumulated detection path. Its replacement
   docstring says: "No entity boost, no verb SUMO boost - the embedding model
   captures these semantics." It raised the threshold to `0.77` and made the
   detector pure cosine.
5. `9d5528ae` on 2026-05-05 added the current shared-lemma WordNet sense gate.
   Missing senses or no shared lemmas explicitly fall back to pure cosine.
6. `3a5020ba` on 2026-04-18 added De Bruijn response composition to Stitch,
   not paraphrase verification.
7. The July PSA changes altered bounded candidate routing and group lifecycle.
   They inherited the February pair decision.

The February assumption that the embedding "captures these semantics" is
falsified by the error inventory. The removed positive boosts could help some
false negatives, but they would not repair factual false positives. The
removed contradiction logic addressed one part of the correct problem, but
there is no valid MRPC evidence for its quality and it should not be restored
blindly.

## Root Causes

### 1. Relatedness is being treated as equivalence

Dense cosine answers whether sentences occupy a similar semantic region. The
application uses it as if it establishes bidirectional proposition
equivalence.

### 2. PSA is a blocker, not the downstream judge

The PSA frame can reduce candidate search while preserving neighborhood
recall. It does not determine that two propositions have the same quantities,
roles, polarity, modality, time, source, or entailments.

### 3. The verifier has no relational claim representation

The current verifier does not align:

- predicate and arguments;
- entity identity and grammatical role;
- quantities with units and the role they modify;
- dates and temporal scope;
- negation scope;
- modality and certainty;
- attribution and quotation scope;
- added or omitted propositions.

Those are exactly the dominant false-positive modes.

### 4. Evidence is not jointly calibrated

The current microGPT policy orders probes; it does not combine their values
into a learned pair decision. Individual hard gates fail because each probe is
context-dependent.

### 5. Single-link and centroid membership can amplify one pair error

One false-positive edge can join a sentence to a group, after which centroid
matching or transitive closure can propagate the mistake. A graph mutation
requires a higher-precision decision than an MRPC F1-optimized binary
classifier.

### 6. The benchmark contract was historically weak

The repository accepted malformed fixtures, missing-pair exclusion, no
persisted confusion matrix, and an unverified commit-message claim. That made
an invalid historical score feel like evidence of a severe recent regression.

## Correct Next Experiment

Do not run MTEB yet. Build and test one MRPC pair verifier whose feature vector
contains:

1. raw Qwen cosine and margin;
2. PSA paired-target rank, rank margin, and neighborhood agreement;
3. ordered WordNet/De Bruijn support and conflict features;
4. aligned predicate-argument and entity-role features;
5. quantity, unit, direction, date, polarity, modality, and attribution
   features;
6. bidirectional entailment or a calibrated cross-encoder/reranker score;
7. explicit coverage and uncertainty flags.

Fit the evidence-combination policy only on the 615-pair calibration component
split. Keep the 1,110 held-out pairs untouched. Report both published MRPC
labels and a separately adjudicated strict-equivalence subset for the
label-noise cases.

For graph mutation, introduce three outcomes:

```text
merge
do_not_merge
abstain_and_keep_free_range
```

Cosine should nominate candidates. It should never again be the only evidence
that creates durable paraphrase membership.
