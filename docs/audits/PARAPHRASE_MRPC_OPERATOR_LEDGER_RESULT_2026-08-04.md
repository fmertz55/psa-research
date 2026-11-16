# MRPC Grammatical-Operator Ledger Result

**Date:** 2026-08-04  
**Status:** Completed diagnostic experiment; score-changing overlay rejected  
**Authority:** None for membership, edge writes, services, or deployment

## Result

The finite grammatical-operator inventory and directional proposition ledger
are valid diagnostic infrastructure, but the calibration-trained additive veto
does **not** improve the controlling MRPC score.

| Measure | Frozen compact baseline | Operator-ledger veto | Difference |
|---|---:|---:|---:|
| Precision | 0.790857143 | 0.793503480 | +0.002646337 |
| Recall | 0.933873144 | 0.923076923 | -0.010796221 |
| F1 | **0.856435644** | 0.853399875 | **-0.003035768** |
| TP / FP / FN / TN | 692 / 183 / 49 / 183 | 684 / 178 / 57 / 188 | -8 / -5 / +8 / +5 |

The paired, pair-aligned 20,000-replicate bootstrap interval for candidate
minus baseline F1 is **[-0.007820295, +0.001376645]**. The point estimate is
negative and the interval includes zero. The candidate fails both the material
gain and confidence gates. The controlling dashboard therefore remains
**P=0.7909, R=0.9339, F1=0.8564**.

Calibration had moved from F1 `0.861952862` to `0.863892013` by removing two
false positives with no true-positive loss. That small gain did not generalize
to the held partition. No parameter was changed after reveal.

## Controlled scope

This experiment changed one rail only:

1. A pinned spaCy dependency stream produced a finite inventory for negation,
   modality, quantification, conjunction/disjunction, and conditionals.
2. Every fact carried an occurrence-bound scope. Missing, ambiguous, malformed,
   overflowed, or unmappable scope downgraded coverage rather than guessing.
3. Two proposition graphs produced directional support ledgers and zero or more
   diagnostic proposals from the existing membership-edge enum. The ledger
   cannot emit `member_of`.
4. The scoring boundary received 81 fixed, swap-invariant, nonnegative integer
   counts and category indicators. It received no text, phrase, lemma, example
   ID, embedding, PSA value, corpus identity, or semantic exception list.
5. A standardized linear ridge veto exposed its intercept, raw value,
   standardized value, coefficient, and exact contribution for every input.
   It could only change a frozen `true` prediction to `false`.

MicroGPT's declared interface is the same named integer/category surface. The
nonlinear `SwitchCore` was deliberately not allowed to control truth because
it cannot provide an exact additive contribution decomposition. No
cross-encoder, new edge type, triplet, Stitch integration, group propagation,
service connection, database write, or corpus-specific phrase rule entered the
experiment.

The full machine report contains all 1,107 raw input vectors, standardized
vectors, 81 per-feature contributions, exact score replays, predictions, and
SHA-256 identities. It also contains the complete sentences, directional
ledger, plural edge proposal set, and additive replay for every changed pair:

- `reports/mrpc_operator_ledger_veto_2026-08-04/operator_ledger_veto_report.json`

## Operator coverage and fail-closed audit

The independent pinned-parser audit processed all 3,393 unique MRPC sentences
without an exception:

| Audit item | Result |
|---|---:|
| Operator facts | 2,437 |
| Complete inventories | 2,727 |
| Partial inventories | 666 |
| Analyzer exceptions | 0 |
| Graph-projection exceptions | 0 |
| Facts unmappable to the frozen graph | 64 |

All 64 unmappable facts downgraded graph coverage: 35 had an unresolved
canonical argument role, 25 had unresolved coordination members, and 4 had an
unresolved nominal argument. The scanner's remaining closed-lexeme surfaces
were genuine non-operator homonyms such as the month `May`, noun `cans`,
ordinal `No.`, quoted response `No`, and the name `All Saints`.

The held partition contained 2,177 unique sentences: 1,743 complete operator
inventories and 434 partial inventories.

## Every changed prediction

The one-way veto changed 13 of 1,107 predictions. Five changes corrected false
positives; eight changes damaged true positives. No false prediction was
changed to true.

| Pair | MRPC label | Transition | Main audited signal |
|---:|---:|---|---|
| 187 | 0 | FP to TN | Added coordinated patient claim and proposition gaps |
| 277 | 1 | TP to FN | `will/likely` versus `could`; disjunction scopes failed exact alignment |
| 636 | 1 | TP to FN | Coordination and large graph gaps; MRPC accepts differing arrest details |
| 817 | 0 | FP to TN | Ban in force versus recommendation to enact one; disjunctive inventory differs |
| 943 | 0 | FP to TN | Extra call instruction and different coordinated participant structure |
| 984 | 1 | TP to FN | Shared `any` and disjunction facts did not align across attribution structure |
| 987 | 1 | TP to FN | `will` versus `would` and coordinated support clauses |
| 1039 | 1 | TP to FN | Equivalent participant lists produced unaligned conjunction scopes |
| 1107 | 1 | TP to FN | The same alternatives were restated with different disjunction scopes |
| 1250 | 0 | FP to TN | Different conference purpose and coordinated subject matter |
| 1645 | 1 | TP to FN | Equivalent attendee list produced an unaligned conjunction scope |
| 1648 | 0 | FP to TN | Possibility claim versus behavioral recommendation; modal scope unresolved |
| 1667 | 1 | TP to FN | Shared boot episode with differing surrounding proposition inventory |

The candidate proposed no membership edge for any of these 13 pairs. That is
correct fail-closed behavior, but it also means the plural edge indicators did
not cause any changed prediction in this run.

## What the result means

The failure is informative and architectural, not a reason to add rules.

First, exact proposition hashes are too brittle to align paraphrastic clauses.
Equivalent conjunctions and disjunctions frequently became unsupported merely
because their event or argument realization changed. Coordination mismatch and
generic proposition-gap counts dominated both the five good vetoes and the
eight bad vetoes.

Second, the unconstrained learned weights are fully replayable but not
semantically monotone. For example, an exactly supported quantifier count had
a negative fitted contribution in several changed pairs. That is transparent
correlation, not a defensible semantic reason to reject a paraphrase.

Third, MRPC's positive class sometimes accepts modal, numeric, attribution, and
scope differences that the stricter graph design intentionally preserves.
Improving strict semantic resolution can therefore lower MRPC F1 on individual
pairs. Chasing those labels with exceptions would recreate the rule cascade
this architecture is designed to avoid.

## Decision and bounded breadcrumbs

- Keep the finite operator analyzer, authenticated sidecar, directional ledger,
  plural existing-edge proposals, and named-count boundary. They are reusable,
  deterministic, and auditable.
- Do **not** promote or integrate the learned veto. The frozen compact resolver
  remains the controlling MRPC lane at F1 `0.856435644`.
- Before another score-changing operator experiment, replace exact-hash clause
  matching with an explicit, occurrence-bound proposition alignment contract.
  This is an alignment problem, not a request for lexical rewrite rules.
- Any later explainable scorer should preregister semantic sign constraints:
  supported evidence cannot hurt, opposed/unsupported evidence cannot help,
  and unresolved evidence cannot veto. Test that contract on a fresh
  development source before another held comparison.
- Keep MRPC and strict graph semantics as separate reported views where MRPC's
  loose positive labels conflict with preserved scope or detail.

These are breadcrumbs only. No second scoring experiment, retuning, new edge,
triplet processing, Stitch work, or group propagation was performed.

## Frozen identities

- Selection file SHA-256:
  `4f1658bbf4f0d6b0668d03382d0131507fba491e55218e8a2bded18b42a043cd`
- Internal selection SHA-256:
  `3be9f50618deb487f1938e4fd21f29b89f55c51e785f48399214d646e931c3f0`
- Additive model SHA-256:
  `789a906c7cd99c7a17c3c9578df275ca4b1181ca6d45644bdd93cdb0e45ba5bf`
- Reveal marker SHA-256:
  `c57aee1f34a25d8abfeb04a3ea036a89620d10d02c51c2d16a0e4678fdc75b41`
- Candidate prediction vector SHA-256:
  `fed24d5e0dfe84dda3e18decd678ccd953b5e3a7acd116888b45c513be0c1e4a`
- Result report SHA-256:
  `df336979d59219c5217b504d4eaf95cd938620052f81795ed48cdadd84632de3`
- Frozen comparator file SHA-256:
  `a5b228d25affc998a0df71471d115369312c958b6534096ebe637fb57ff0d79a`
