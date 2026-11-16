# MRPC Semantic Analysis V1 Result

**Date:** 2026-08-04  
**Status:** One preregistered reveal complete; post-reveal fail-closed audit complete  
**Authority:** Research diagnostic only; no membership, service, or deployment authority

## Outcome

The deterministic evidence compiler now creates authenticated
`semantic_analysis_v1` artifacts directly from exact source occurrences. It
implements all four requested evidence rails:

1. occurrence-level OEWN predicate-sense selection with explicit alternatives;
2. exact sense-key-to-VerbNet syntax frames and active/passive thematic roles;
3. named-entity, apposition, literal, local-discourse, and reflexive grounding;
4. complete finite grammatical operators combined with every decision and
   pinned resource identity in one canonical artifact.

The implementation is in
`app/mcp_servers/mcp_lib/semantic_evidence_compiler.py`. It does not import the
historical paraphrase evidence, graph, rule suite, scorer, resolver, membership,
database, or persistence paths. The two-phase runner is
`scripts/research/mrpc/run_mrpc_semantic_analysis_v1_benchmark.py`, and its adversarial tests
are in
`tests/unit/mcp_servers/mcp_lib/test_semantic_evidence_compiler.py`.

The compiler is correct enough to fail closed, but it is not yet broad enough
to decide MRPC. The current audited compiler has 22 of 3,393 fully proof-ready
sentences, and no test pair has two proof-ready endpoints. The frozen pair
result is therefore 1,725 abstentions and zero semantic claims.

## Authenticated run

- Preparation SHA-256:
  `a5751f04acb5830554a971c673e2c6bafaad4f2b05fd696f0440d314e6579fcf`
- Report SHA-256:
  `09caed3da2d628a642281f0e6cc41dce1b7aa72a8d54a9b396c15309001fe35d`
- Post-reveal label-free correctness-audit SHA-256:
  `4c1ac7e3dab0f9566502b5001bcbe1d5e496fe6f939f1a3233fc959b461fa946`
- MRPC unique sentences: **3,393**
- MRPC official test pairs: **1,725**
- Calibration / held sentence-component split: **618 / 1,107**
- Label-bearing fields in the authenticated preparation: **0**
- Relevant compiler, artifact, proof, and operator tests: **87 passed**
- Post-reveal threshold or rule changes: **0**
- Post-reveal fail-closed coverage corrections: **1**

The preparation phase discarded the isolated leading `Quality` column before
parsing pair IDs. It froze sentence artifact identities, readiness, split,
pair outcomes, proof identities, code hashes, and resource hashes. The reveal
phase authenticated that envelope and the current inputs and source before
creating a one-use marker and reading labels.

After reveal, a fail-closed review found that an unbound pronoun outside an
already resolved valence argument did not lower the artifact's coreference
coverage flag. The compiler now inventories every pronoun occurrence and marks
any ungrounded one unresolved. A separate label-free audit changed
coreference-complete coverage from 3,393 to 1,744 sentences and removed one
sentence containing possessive `its` from the ready set. It changed **zero of
1,725 frozen pair outcomes**: every outcome remains `abstain`. The revealed
preparation and report were not overwritten, and no second label reveal was
performed.

## Sentence coverage

Coverage is exact and fail-closed. Categories can overlap because one sentence
may contain multiple predicates and multiple unresolved causes.

| Evidence rail | Complete sentences | Coverage |
|---|---:|---:|
| Exact occurrence tokenization | 3,393 / 3,393 | 100.00% |
| Predicate occurrence inventory | 2,960 / 3,393 | 87.24% |
| Occurrence-selected predicate concepts | 999 / 3,393 | 29.44% |
| Sense-specific semantic roles | 38 / 3,393 | 1.12% |
| Referent grounding | 3,269 / 3,393 | 96.35% |
| Coreference decisions | 1,744 / 3,393 | 51.40% |
| Complete operator inventory | 2,727 / 3,393 | 80.37% |
| Entire sentence proof-ready | 22 / 3,393 | 0.65% |

Coreference completeness never means that arbitrary pronouns were guessed.
Every ungrounded pronoun occurrence is now an explicit failure; 2,613 such
occurrences make 1,649 sentences incomplete. Likewise, a document-local common
nominal is grounded enough to make its sentence representation closed, but it
is not silently treated as the same entity in another sentence.

## Pair result

| Scope | Pairs | Equivalent | Directional | Incompatible | Abstain |
|---|---:|---:|---:|---:|---:|
| Official test | 1,725 | 0 | 0 | 0 | 1,725 |
| Held out | 1,107 | 0 | 0 | 0 | 1,107 |
| Calibration | 618 | 0 | 0 | 0 | 618 |

With only `equivalent` mapped to the positive MRPC class, the official test
confusion counts are `TP=0`, `FP=0`, `FN=1,147`, `TN=578`. Precision, recall,
F1, and selective coverage are all `0`. Accuracy is `0.335072`, solely from
abstaining on all 578 negative examples. The held-out confusion counts are
`0 / 0 / 741 / 366` with accuracy `0.330623`.

These are not useful classifier metrics. They are an exact analyzer-coverage
result. The kernel made no unsupported equivalence, entailment, or conflict
claim.

## What the numbers isolate

Semantic roles are the binding constraint. The largest occurrence-level
valence failures were:

- **2,657** subject-cardinality failures, primarily embedded, coordinated,
  controlled, raised, relative, and implicit-subject predicates;
- **2,439** selected-sense-specific VerbNet frame misses;
- **2,010** currently unsupported clausal arguments;
- **494** observed structures with no exact valence-frame match; and
- **115** cases where matching VerbNet frames assign competing roles.

Sense selection is the next constraint: **3,461** predicate occurrences had
multiple OEWN candidates without the preregistered Lesk margin, **62** had no
verbal candidate, and **3** could not produce one exact sense key. These counts
overlap the valence failures because the compiler records every missing
judgment rather than stopping at the first one.

The 433 incomplete predicate inventories expose a separate representation
gap: 371 sentences have an auxiliary/copular root, 41 a nominal root, 18 a
proper-name root, and 3 an adjectival root. A verb-only event inventory cannot
represent those propositions.

Operators remain a secondary but material constraint: 666 sentences have an
incomplete inventory, mostly coordination and quantifier scope. Referent
coverage is already high at the sentence level, but proof also needs
independently justified cross-document identity for descriptions and aliases;
the v1 compiler correctly creates none from MRPC co-occurrence.

## The next correct compiler

The next version is not a score adjustment. It is a deterministic constraint
solver over a richer proposition and discourse graph:

1. represent finite, copular, nominal, embedded, coordinated, relative,
   controlled, and raised predicates, including proposition-valued arguments;
2. propagate syntactically licensed subjects and controllers, then match the
   full VerbNet syntax and selectional restrictions rather than a shallow child
   list;
3. solve sense and valence jointly while preserving every surviving branch,
   selecting only a globally unique interpretation;
4. ground nominal descriptions by their selected head concept, modifiers,
   quantification, and discourse links, and create cross-document aliases only
   from independent identity evidence;
5. finish coordination, quantifier, conditional, modality, and negation scope;
6. rerun on a development corpus without reusing this revealed MRPC result as
   a rule-selection oracle.

That work is the exact structured interpretation humans form before comparing
the sentences. The current result confirms that the artifact and proof kernel
can carry and verify it; it also proves that a verbal dependency child list plus
standalone Lesk is nowhere near a complete human interpretation.

## Reproduction

The reveal marker intentionally prevents a second reveal. The frozen
preparation and report are retained under:

```text
temp/mrpc_semantic_analysis_v1/prepared.json
temp/mrpc_semantic_analysis_v1/postreveal_correctness_audit.json
reports/mrpc_semantic_analysis_v1_2026-08-04/semantic_proof_report.json
```

Focused validation:

```bash
.venv/bin/pytest -q \
  tests/unit/mcp_servers/mcp_lib/test_semantic_evidence_compiler.py \
  tests/unit/mcp_servers/mcp_lib/test_semantic_analysis.py \
  tests/unit/mcp_servers/mcp_lib/test_semantic_proof.py \
  tests/unit/mcp_servers/mcp_lib/test_grammatical_operator_inventory.py
```

The frozen protocol is
`docs/audits/PARAPHRASE_MRPC_SEMANTIC_ANALYSIS_V1_PREREGISTRATION_2026-08-04.md`.
