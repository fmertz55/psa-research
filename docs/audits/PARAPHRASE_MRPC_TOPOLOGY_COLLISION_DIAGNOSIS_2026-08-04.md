# MRPC Topology-Collision Diagnosis

**Date:** 2026-08-04  
**Status:** Collision hypothesis not supported; no operator proposed or scored  
**Controlling MRPC score:** P `0.790857`, R `0.933873`, F1 `0.856436`  
**Frozen confusion matrix:** TP `692`, FP `183`, FN `49`, TN `183`

## Answer

Topology compression is not a material explanation for the frozen 183 false
positives.

- Exact semantic-hash matches among the false positives: **0 / 183**.
- Exact collisions across all 13 controlling graph inventories and all five
  graph count values: **0 / 183**.
- Exact inventory collisions with different labeled paths: **0 / 183**.
- At the fixed descriptive `0.80` near-collision band, only **1 / 183**
  false positives qualifies. The same condition selects **27 / 692** true
  positives, and 25 of those true positives have different labeled paths.
- The broader `0.70` sensitivity band finds **4 / 183** false positives and
  **78 / 692** true positives. This does not become a useful error bucket as
  the definition is relaxed.
- Removing parser positions and binding provenance from property inventories
  does not uncover a hidden false-positive collision: exact remains **0 / 183**,
  and the `0.80` near lane remains **1 / 183** while expanding to **34 / 692**
  true positives.

The full labeled-path audit found **33 / 183** false positives with at least
one localized role-binding or position-insensitive event-property conflict.
The same union appears in **166 / 692** true positives. These are observed
local witnesses, not a safe veto: the true-positive control shows that parser
attachment, scope, attribution, and permitted detail differences dominate the
condition.

The frozen graph cache is not strict-proof-ready for any audited pair. It is a
V2 evidence artifact projected through the current V3 builder and has partial
logical-operator coverage for every sentence. It also has no observable
sentence-level FrameBase predicate grounding or coreference-topology coverage.
Consequently:

- localized existing conflicts observed: **33 FP / 166 TP**;
- strict fail-closed separations currently authorized: **0 FP / 0 TP**;
- operator-scope, FrameBase, and coreference conflicts: **unavailable**, not
  proven absent.

No Graph Intersection Operator is proposed. The collision bucket is too small,
the available conflict conditions endanger far more true positives than false
positives, and strict coverage is missing.

## Frozen measurement

| Condition | False positives | True-positive controls | Interpretation |
|---|---:|---:|---|
| Exact semantic hash | 0 | 0 | Exact labeled equivalence is absent in both lanes |
| Position-insensitive semantic hash | 0 | 1 | Parser coordinates do not hide an FP match |
| Exact graph inventory | 0 | 0 | No exact topology-compression collision exists |
| Position-insensitive exact inventory | 0 | 1 | The one control also has exact labeled paths |
| Near inventory, `J >= 0.90` | 0 | 4 | No false-positive bucket |
| Near inventory, `J >= 0.80` | 1 | 27 | One negative label versus 27 positive controls |
| Near inventory, `J >= 0.70` | 4 | 78 | Broadening worsens separation |
| Position-insensitive near inventory, `J >= 0.80` | 1 | 34 | Normalization expands only the control hazard |
| Exact labeled paths | 0 | 2 | Exact path equality is rare even for paraphrases |
| Zero labeled-path intersection | 69 | 217 | Shape/decomposition gaps are common in both lanes |
| Exact role-binding conflict | 8 | 33 | Deterministic but not reliable enough to veto |
| Position-insensitive event-property conflict | 26 | 145 | Parser scope and allowed detail dominate |
| Temporal property conflict | 17 | 101 | Not specific to false positives |
| Other event-property conflict | 11 | 61 | Mainly polarity/modality/attribution alignment |
| Predicate inventory difference | 163 | 575 | Expected under alternate decomposition |
| Predicate-binding difference | 109 | 412 | Common in genuine paraphrases too |
| Temporal argument difference | 35 | 137 | Not a safe separator |
| Quantity argument difference | 23 | 100 | Exact witnesses exist, but condition is nonspecific |
| Scale-bearing argument difference | 26 | 119 | Exact witnesses exist, but condition is nonspecific |
| Observable operator/frame/coreference conflict | 0 | 0 | Coverage is absent; zero is not negative evidence |
| Localized material conflict union | 33 | 166 | Candidate witnesses only |
| Strict fail-closed separation | 0 | 0 | Frozen operator coverage blocks adjudication |

The inventory similarity distribution points in the opposite direction from
the hypothesis. Median tagged-inventory Jaccard is `0.4182` for false positives
and `0.5000` for true positives. The false-positive maximum is `0.8085`; the
true-positive maximum is `0.9683`.

## Definitions

The audit uses the exact inventory implementation behind the frozen resolver's
62 graph values. An exact collision requires equality of every Counter for:
`argument_lemma`, `argument_sense`, `clause_role`, `entity`, `owner`,
`predicate_lemma`, `predicate_sense`, `property_key`, `property_value`,
`quantity`, `role`, `role_entity`, and `role_lemma`, plus equality of the five
graph counts for propositions, arguments, edges, coreference chains, and
unresolved items.

Near collision is a sensitivity analysis, not a tuned threshold. It uses
weighted Jaccard over category-tagged inventory multisets and requires every
graph-count delta to be at most one. The fixed bands are `0.90`, `0.80`, and
`0.70`.

A labeled path is the exact canonical multiset path:

`predicate identity -> semantic role -> semantic participant node identity`

Predicate identity preserves lemma, synsets, particles, and frames when they
exist. Participant identity preserves the complete semantic node, including
normalized entity, synsets, owner, quantity, and comparison fields. No fuzzy
matching, edit cost, lexical pair rule, cross-encoder, or corpus-specific
exception is used.

The existing semantic hash and controlling `property_value` inventory retain
cached parser coordinates inside event-property records. The audit reports
those values exactly as consumed, then adds a position-insensitive sensitivity
lane. Exact false-positive collisions remain zero in both lanes.

Raw kernel event-property traces can differ merely because the same predicate
is at a different token position. The audit retains those raw witnesses but
counts a material property conflict only after parser positions, indexes, and
binding provenance are removed. This reduced raw event-property differences from
`56 FP / 281 TP` to semantic differences in `26 FP / 145 TP`.

## Representative witnesses

### The only `0.80` near-collision false positive

Pair `1532` has tagged-inventory Jaccard `0.808511`:

> Air Canada, the largest airline in Canada and No. 11 in the world, has been
> under court protection from creditors since April 1.

> The No. 11 airline in the world, Air Canada has been under court protection
> from creditors since April 1.

This is a graph-shape difference around an appositive, not a directional role
reversal. The pair is also a warning about treating the MRPC negative label as
semantic ground truth: the texts are nearly identical. A `0.80` collision veto
would target this one negative label while endangering 27 positive controls.

### Predicate binding that does expose a factual change

Pair `14` says, in part:

> Hong Kong was flat, Australia, Singapore and South Korea lost 0.2-0.4
> percent.

> Australia was flat, Singapore was down 0.3 percent by midday and South Korea
> added 0.2 percent.

The exact witness preserves the entity and role while showing different
predicate bindings:

- `Australia --agent--> lose` versus `Australia --agent--> be`;
- `South Korea --agent--> lose` versus `South Korea --agent--> add`.

This is useful diagnostic evidence, but the generic predicate-binding condition
also appears in 412 true positives. It cannot be promoted as a veto.

### Quantity and scale witness

Pair `84` reports different movements for LendingTree shares. The graph records
source-only `14 percent` and target-only `$6.03` and `41 percent` quantity paths
under the grounded `rise` predicate. These are exact, explainable witnesses.
Yet quantity differences occur in `23 FP / 100 TP`, and scale-bearing argument
differences occur in `26 FP / 119 TP`. The presence of a one-sided detail is not
by itself a contradiction.

### Role-binding witness and parser failure

Pair `629` produces:

`decline -> recipient -> CIA` versus `decline -> agent -> CIA`.

The source text is "Spokesmen for the FBI, CIA, ... declined to comment". The
recipient role is a parser attachment error: the organizations modify
"Spokesmen" rather than receiving the declining event. This is exactly why the
eight false-positive role conflicts cannot be counted as eight safe repairs.
The same condition appears in 33 true positives. For example, pair `15` binds
`other` as patient versus agent under `defend` even though the sentences are a
genuine paraphrase.

### Operator-like scope noise in the true-positive control

True-positive pair `2` is parsed with a negative-polarity record because the
surface token `No.` in "No. 2 Ford Motor Co." is mistaken for negation. That
single parser error contributes an apparent event-property conflict and shows
why incomplete operator coverage must fail closed rather than convert a named
count into a veto.

## Materiality

With TP fixed at `692` and FN fixed at `49`, reaching F1 `0.9` by false-positive
repair alone requires at least **79** perfect repairs. The exact collision bucket
contains zero cases. The `0.80` near bucket contains one. Even the broader
localized-conflict union contains only 33 false positives before considering
its 166 endangered true positives.

The topology-collision hypothesis therefore cannot plausibly move the frozen
F1 materially toward `0.9`. This conclusion does not say that labeled topology
is useless. It says the measured frozen false-positive population is not an
inventory-collision bucket, and the current cache cannot support a safe
operator-scope/coreference/frame adjudicator.

## Reproducibility and authority

The audit authenticates:

- the two MRPC source files;
- the frozen 1,107-pair comparator, labels, route, score, and prediction
  vectors;
- the exact `692 / 183 / 49 / 183` confusion matrix; and
- pinned graph cache SHA-256
  `9625dd25e9ed67beb5832ec38dccf95b66415f8c49333e2e1477ef8ff7f927fc`;
- audit script SHA-256
  `f5dd0eed097f0db748528af322eafd3975c0e15e05dc1a0dd4a8abe9eebdf715`;
- controlling inventory implementation SHA-256
  `1d5e53f4895d716490bc3e19e0b470d28d03071addf2e5a24181a139290379f9`;
- focused test SHA-256
  `1d509298d135077a24ecf16d5d473e345fb1ebd3284fbf24b15a46eb1f497b11`;
  and
- machine-readable report SHA-256
  `8ef40b70f3178716cefcfa4e04bd3b2496c12bb4b4a6bfce735a21c921b4f95a`.

Run:

```bash
.venv/bin/python scripts/research/mrpc/audit_mrpc_topology_collisions.py
```

The machine-readable report is
`reports/mrpc_topology_collision_audit_2026-08-04/topology_collision_audit.json`.
It contains all 875 positive-prediction pair audits and full representative
witness traces.

The diagnostic has no classification, threshold-selection, membership,
database-write, edge-write, or service-integration authority. It changed zero
predictions. No Graph Intersection Operator was implemented, preregistered, or
scored.
