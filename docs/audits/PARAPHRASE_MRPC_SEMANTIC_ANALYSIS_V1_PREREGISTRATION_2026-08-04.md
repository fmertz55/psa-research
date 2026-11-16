# MRPC Semantic Analysis V1 Preregistration

**Date:** 2026-08-04
**Status:** Frozen before compiler implementation and MRPC label access
**Authority:** Research diagnostic only; no membership or production authority

## Question

How many MRPC sentence pairs can a deterministic, occurrence-bound semantic
compiler prove equivalent, prove incompatible, or only abstain on when every
accepted decision is bound to exact source spans and pinned resources?

This experiment tests the new `semantic_analysis_v1` plus deterministic proof
kernel. It does not modify, blend with, or threshold the frozen learned MRPC
resolver.

## Frozen inputs

- Official MRPC corpus and test split already pinned by SHA-256 in the existing
  benchmark suite.
- spaCy `en_core_web_sm` 3.8.0 dependency parse.
- Open English WordNet 2024 with the local database content hash.
- VerbNet 3.3 distributed by the NLTK data project. The downloaded ZIP must
  match MD5 `60efc5ed90ab8a18ef4a436e4c39ffbf` and SHA-256
  `fa0136a7699c52f0bd532dc5adc0914745aa4369a52ae1465cb11841060ec1de`.
- The finite grammatical-operator analyzer already covered by its focused
  contract tests.

No MRPC label, baseline score, pair outcome, or phrase list may influence a
sense, valence, referent, coreference, operator, alignment, or decision.

## Deterministic compiler

### Predicate occurrences and senses

Every non-auxiliary verbal predicate receives an exact character-bound token
occurrence. OEWN candidates are retrieved for that occurrence's parser lemma.

- A single candidate is selected.
- Multiple candidates use the pre-existing bounded extended-Lesk procedure:
  definitions weight `2`, examples `3`, lemmas `1`, one-hop related
  definitions/examples `1`, and related lemmas `0.5`.
- A winner is selected only when its score is at least `1` and its margin over
  the runner-up is at least `1`.
- Ties, missing candidates, and insufficient margins remain explicit
  alternatives and make that atom unready.

The selected OEWN synset is the predicate concept. Qwen sense output is not
selection authority.

### Valence roles

The selected OEWN lexical sense key must map to VerbNet 3.3. The observed
dependency arguments must match at least one sense-specific VerbNet syntax
frame. Active and passive realization are normalized from the same frame.

Every matching frame must agree on the thematic role for every observed core
argument. No match or disagreement means unresolved valence. There are no
lemma exceptions or MRPC-shaped role rules.

### Referents and coreference

- Named entities and proper-name spans receive a global grounding derived from
  exact normalized name plus entity type.
- Repeated exact names and dependency-authenticated apposition share that
  grounding.
- Common nominals receive document-local discourse referents.
- A reflexive pronoun may share the unique bound subject referent of its
  governing predicate.
- Every other pronoun or ambiguous mention remains unresolved.
- Quantity, money, percent, ordinal, date, and time arguments are canonical
  literals rather than entity guesses.

No cross-sentence alias is created from pair similarity or MRPC co-occurrence.

### Operators

The complete dependency token stream is passed to the finite grammatical
operator analyzer. Predicate truth commitment is an exact composite of
polarity and modal force. Quantification, coordination, and conditional facts
retain canonical relative scope. Partial operator coverage makes the affected
interpretation unready.

## Pair decision

Only proof-ready sentence artifacts reach the kernel.

- mutual support: `equivalent`;
- one-way support: directional entailment;
- a validated material conflict: `incompatible`;
- every missing, ambiguous, unsupported, or competing analysis: `abstain`.

For the binary MRPC diagnostic, only `equivalent` is a positive prediction.
All other outcomes are non-positive. Selective coverage and abstention are
reported separately so this mapping cannot hide low proof coverage.

## Label isolation and stopping rule

The run has two phases:

1. `prepare` loads sentence IDs and text but never reads `Quality`; it freezes
   compiler identities, artifact/readiness hashes, pair outcomes, and the
   calibration/held component split.
2. `reveal` authenticates the frozen preparation, then and only then reads
   labels and computes metrics without changing any prediction.

Report both the complete official test split and the existing sentence-
component-held subset. Do not tune after either reveal. Stop after one reveal.

## Required report

- sentence and pair proof-readiness;
- coverage for spans, senses, valence, referents/coreference, and operators;
- counts of equivalent, directional, incompatible, and abstain outcomes;
- precision, recall, F1, confusion counts, and selective precision;
- readiness and outcome counts by true label;
- the largest unresolved categories; and
- representative complete proofs, conflicts, and abstentions with exact
  witnesses.
