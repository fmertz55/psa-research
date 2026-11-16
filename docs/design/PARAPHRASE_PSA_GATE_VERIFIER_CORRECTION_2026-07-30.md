# PSA Gate and Verifier Correction

**Date:** 2026-07-30  
**Last updated:** 2026-08-01

**Status:** Controlling for the candidate funnel; verifier rule roadmap superseded
**Evaluation status:** Development only; no current score is blind

> **2026-08-03 semantic-architecture correction:** The deterministic
> canonicalization roadmap in this document is superseded by
> [Paraphrase Semantic Resolution Architecture](PARAPHRASE_SEMANTIC_RESOLUTION_ARCHITECTURE_2026-08-03.md).
> The existing ETPC/MRPC suite is now an explicit diagnostic policy. Shared
> verification defaults to a small proof kernel; new phrase-shaped or
> corpus-derived canonicalizations must not be added to recover benchmark
> errors.
>
> This repository is not being described as a deployed production system.
> Later uses of “production” in the preserved historical text mean the proposed
> runtime path at the time this document was written; the controlling 2026-08-03
> architecture keeps non-text membership disabled.

**Supersedes:** Every claim in
`PARAPHRASE_TWO_CLOCK_THREE_STATE_DESIGN_2026-07-29.md` that treats native
cosine above `0.77` as identity, verified paraphrase, or direct membership
authorization.

## Correct decision funnel

Paraphrase decisions follow this order:

1. Produce the sentence's native embedding.
2. Use the complete 115-dimensional PSA signature to nominate a bounded local
   neighborhood.
3. Apply the same-model native-cosine candidate gate. Its cold-start value is
   `0.77`; it is a learning threshold, not a semantic verdict.
4. Run the verifier suite on every non-exact gated candidate.
5. Admit hard paraphrase membership only when the verifier establishes
   proposition equivalence.

Equal persisted `text_sha256` is the sole immediate identity path. Native
cosine can nominate a candidate but cannot override numeric, polarity,
entity, relation, modality, temporal, or other material conflicts.

PSA is the neighborhood mechanism. Native embedding search across the global
corpus is not the production candidate route. PSA routing must use the full
signature and matching frame, calibration, model, content, tenant, workspace,
and semantic-namespace identity before exact native rescoring.

## Threshold behavior

The model-specific cold start for Qwen is `0.77`. Production restores the
historical density adaptation:

- PSA heat at or below 2: `0.77`
- PSA heat from 2 through 4: interpolate from `0.77` to `0.88`
- dense PSA voxel or heat at least 4: `0.88`

A validated learned threshold may replace the cold-start scalar at runtime.
Learning data must be isolated from held-out scoring. Changing the threshold
changes which candidates reach verification; it never changes the evidence
required for membership.

The current versioned calibration artifacts expose two learned candidate
gates for different evaluation purposes:

- `0.7717701196670532` is the learned strict candidate gate used with the
  `strict_merge_v1` proof profile;
- `0.627297043800354` is the learned recall-floor gate, selected as the
  strictest calibration threshold that retains at least 99% of calibration
  positives, used to measure verifier headroom with `mrpc_compatible_v1`.

Neither threshold is evidence of equivalence. The recall-floor gate
deliberately passes more negative candidates so that the deterministic rule
suite, rather than retrieval, bears the precision burden.

## Compatibility and recovery

`native_similarity_above_threshold` remains readable as a legacy audit reason
so historical records are not made undecodable. It cannot authorize a new
admission, group convergence, or reactivation. Recovery must reverify those
members with the current verifier contract.

The candidate indexes introduced by
`20260730_paraphrase_psa_candidate_indexes.cypher` store the full PSA
signature for free sentences and active groups. Native indexes may remain for
compatibility and unrelated retrieval, but they are not an alternative
paraphrase identity path.

## MRPC correction benchmark

The offline benchmark in `scripts/research/mrpc/run_mrpc_gate_verifier_benchmark.py` replays
the exact funnel without requiring live services or mutating the graph. It:

- uses cached native embeddings and the pinned PSA frame,
- learns one scalar threshold on the calibration partition only,
- retrieves top-32 full-signature PSA neighbors,
- applies the current verifier probes, and
- scores a sentence-component-disjoint held-out partition.

On the 2026-07-30 run, PSA nominated 99.73% of held-out positive pairs. In the
current candidate census this is 739 of 741 positive pairs, so the PSA top-32
nomination ceiling is F1 `0.99865` before candidate-gate or verifier losses. The
cold `0.77` production path scored F1 `0.8366`; the calibration-learned
threshold `0.728510` scored F1 `0.8369`. The strict opt-in surface-conflict
policy scored F1 `0.7860` at the learned threshold because numeric and
polarity mismatches rejected more published MRPC positives than negatives.
Those conflicts therefore remain shadow evidence by default. The follow-on
decision dated 2026-07-31 makes the next MRPC improvement target a deterministic
proposition graph and proof-producing rewrite policy, not broader candidate
retrieval, an evidence classifier, or uncalibrated hard surface gates.

These scores are diagnostic, not release acceptance. The labels in every
currently reported MRPC partition have since been inspected during rule
development. Threshold selection, verifier development, and final evaluation
must therefore be split again on a fresh sealed set; none of the current or
later post-hoc scores may be described as blind.

## PSA top-32 candidate census

The detailed candidate census is recorded in
`MRPC_PSA_TOP32_CANDIDATE_ANALYSIS_2026-07-30.md`. It evaluates every one of
the 108,576 directed top-32 edges rather than only each official paired
target.

The census confirms:

- held-out positive nomination recall is 99.73% forward and 100% in either
  direction;
- the learned native gate removes 98.63% of unlabeled neighbors;
- current WordNet rejection loses more known-positive than known-negative
  directed edges in absolute count;
- raw ordered-synset De Bruijn features separate the classes in aggregate but
  add no held-out lift beyond retrieval features; and
- mutual forward/reverse evidence is useful but cannot be a universal hard
  requirement.

These findings supersede any proposal to improve precision by narrowing the
PSA neighborhood. Verifier development must focus on proposition structure.

## Proposition graph verifier design

**Decision update, 2026-07-31:** De Bruijn fragments, grammar-popularity
scheduling, microGPT scheduling, and a learned evidence combiner are retired
from paraphrase verification. De Bruijn composition used by Stitch is a
separate subsystem and is not affected by this decision.

This retirement is final for the present verifier design. Proposition-role
graphs and explicit rewrite proofs replace the proposed De Bruijn traversal;
grammar-pattern popularity does not determine proof order or acceptance.

### Governing principle

The final paraphrase verdict is a proof, not a score. PSA and native cosine may
nominate a candidate, but neither value enters the equivalence proof. Parser,
WSD, frame, NER, quantity, temporal, and coreference outputs are observable
evidence inputs. Their identities and versions are preserved so a person can
inspect exactly what the verifier received.

The verifier is deterministic over those inputs. It may return only:

```text
verified_paraphrase
do_not_merge
abstain_and_keep_free_range
```

It returns `verified_paraphrase` only when every proposition is covered, every
material role and operator is preserved, and every difference is justified by
a versioned allowlisted transformation. Missing or ambiguous evidence causes
abstention. A proved material conflict causes `do_not_merge`.

### Decision profiles and authorization boundary

The same deterministic graph engine exposes two separately versioned policy
profiles:

| Profile | Purpose | May authorize hard membership? |
| --- | --- | --- |
| `strict_merge_v1` | Production-grade exact and fully licensed graph equivalence | Yes; this is the sole graph profile with membership authority |
| `mrpc_compatible_v1` | Document and benchmark compatibility, including explicitly scoped MRPC compression rules | No; it may never create, converge, or reactivate a hard paraphrase group |

Both profiles emit the selected profile ID, rule-registry version and hash,
algorithm version, applications, conflicts, and unresolved evidence in the
proof trace. A compatible-profile acceptance must never be silently promoted
to a strict acceptance. This separation lets MRPC compatibility work proceed
without weakening the production meaning of identity.

### Graph contract

Each sentence becomes a bounded property graph. Surface token positions remain
provenance; they are excluded from semantic identity.

Node kinds are:

- `event`: a predicate with lemma, WordNet senses, frames, tense, and aspect;
- `entity`: a normalized entity or referent;
- `quantity`: value, comparator, unit, currency, and percentage status;
- `time`: normalized date, time, interval, or event-relative temporal value;
- `location`: normalized place or location referent; and
- `proposition`: a nested claim used by clausal complements and attribution.

Event-to-node edge roles are:

- `agent`;
- `patient` or theme;
- `recipient` or beneficiary;
- `content`;
- `complement`;
- `location`;
- `time`; and
- `quantity`.

Graph relations also include `corefers_with`, attribution source-to-claim, and
scope edges for polarity and modality when those relationships cannot be
represented as unambiguous event properties.

For English, parser-specific dependencies normalize through a pinned mapping
based on [Universal Dependencies](https://universaldependencies.org/en/dep/).
Examples include:

| Parser evidence | Canonical role |
| --- | --- |
| `nsubj`, `csubj` | `agent` |
| `nsubj:pass`, `nsubjpass`, `csubj:pass` | `patient` |
| `obj`, `dobj` | `patient` |
| `iobj` | `recipient` |
| `obl:agent`, `agent` | `agent` |
| resolved locative or temporal oblique | `location` or `time` |
| unresolved `pobj`, `obl`, or generic dependent | unresolved; never guessed |

Frame-specific roles may refine this mapping. An explicit resolved semantic
role outranks a dependency-derived role. A generic dependency role cannot be
silently promoted when the governing predicate makes that interpretation
ambiguous.

The graph stores two hashes:

- an evidence graph hash including sentence identity, parser provenance,
  source positions, coverage, and unresolved conditions; and
- a semantic hash excluding operational IDs, surface forms, and positions.

The semantic hash is a fast exact check, not an independent authorization.
Coverage and unresolved conditions must pass before a matching semantic hash
can prove equivalence.

### Voice and order normalization

These sentences have different source positions and dependency labels:

```text
The company acquired the startup.
The startup was acquired by the company.
```

They normalize to one proposition:

```text
ACQUIRE
  agent   -> COMPANY
  patient -> STARTUP
  polarity -> positive
  modality -> asserted
```

The proof records `voice_alternation`. Reversing the two bindings produces an
explicit `role_binding_conflict`, even though the predicate and entities are
unchanged.

### Licensed graph rewrites

Every non-exact equivalence transformation lives in a versioned registry. A
rule contains:

- a stable rule ID and version;
- the source and target graph patterns;
- directionality;
- required senses, frames, roles, and scope preconditions;
- the exact nodes and edges it may add, remove, or relabel;
- forbidden conflicts and counterexamples; and
- citations, tests, and adjudication history.

Initial rule families follow the fine-grained operations in
[ETPC](https://aclanthology.org/L18-1221/),
[Bhagat and Hovy](https://aclanthology.org/J13-3001/), and
[Vila, Martí, and Rosso](https://direct.mit.edu/coli/article/39/4/917/1450/Plagiarism-Meets-Paraphrasing-Insights-for-the):

- inflectional and licensed derivational normalization;
- same-sense lexical substitution;
- active/passive voice alternation;
- converse substitution with mandatory role remapping;
- direct/indirect speech with attribution preservation;
- format, punctuation, and order changes;
- explicitly licensed coordination and subordination changes; and
- deterministic unit, number, date, and named-entity normalization.

Rules are applied in a fixed registry order with a bounded rewrite budget.
Rule order is configuration, not learned behavior. When two rewrites could
produce different meanings, the verifier abstains unless the registry defines
an unambiguous precedence and supplies tests for that interaction.

### Complete-coverage proof

Acceptance requires all of the following:

1. every source proposition maps one-to-one to a target proposition;
2. predicate identity is exact or covered by a licensed sense/frame rewrite;
3. agent, patient, recipient, location, time, quantity, and content bindings
   match after licensed role transformations;
4. polarity, modality, tense/aspect when material, and attribution scope match;
5. all referents and coreference chains are compatible;
6. every added or removed surface element is either semantically inert under a
   licensed rule or represented as a graph difference; and
7. no unresolved material difference remains.

The ETPC analysis is controlling here: valid paraphrase operations commonly
appear in both positive and negative MRPC pairs. Finding one valid operation
does not prove equivalence. The verifier must also prove that no unlicensed
residual proposition, entity, quantity, role, polarity, modality, temporal, or
attribution difference remains.

### Proof trace

Every comparison emits a canonical trace. A successful example is:

```json
{
  "equivalent": true,
  "reason": "complete_normalized_graph_match",
  "matched_propositions": 1,
  "transformations": ["voice_alternation"],
  "differences": [],
  "unresolved": []
}
```

A rejection names the material conflict:

```json
{
  "equivalent": false,
  "reason": "unlicensed_semantic_difference",
  "transformations": [],
  "differences": [{
    "kind": "role_binding_conflict",
    "predicate": {"lemma": "acquire"},
    "source_roles": ["agent"],
    "target_roles": ["patient"]
  }],
  "unresolved": []
}
```

The trace also carries the source and target semantic hashes, parser and
normalizer identities through the underlying graphs, rule-registry version,
and exact rule applications once rewrites are enabled.

### Implementation boundary

`app/mcp_servers/mcp_lib/proposition_graph.py` and
`app/mcp_servers/mcp_lib/proposition_graph_rules.py` now implement the first
complete executable registry contract. They:

- project the existing V2 evidence payload into canonical nodes and edges;
- separate evidence identity from semantic identity;
- normalize active/passive dependency roles;
- preserve event properties and factual bindings;
- fail closed on incomplete or unresolved roles;
- emit exact normalized matches, added/missing propositions, role-binding
  conflicts, event-property conflicts, and auditable proof JSON;
- cover every ETPC `0.95` transformation type with an explicit normalize,
  licensed-rewrite, conflict-check, or abstention policy;
- license pinned OEWN synonym, derivation, antonym-plus-negation, named-entity,
  converse-role, active/passive, reporting-wrapper, format, order, and bounded
  addition/deletion rewrites; and
- preserve the registry hash, lexical-resource identity, exact applications,
  conflicts, and unresolved evidence in every decision trace.

The complete registry does not mean every ETPC transformation is automatically
accepted. Unsafe contextual opposition, broad semantic relatedness, and
one-way entailment deliberately abstain unless an exact allowlisted pattern
proves equivalence. Ellipsis is licensed only for a generic omitted controller
whose remaining predicate roles provide an independent anchor. The graph
verifier remains outside production membership until its recall is adequate.

### Parser and evidence prerequisites

The graph proof is only as complete as its observable inputs. The benchmark
uses the existing spaCy tokenizer and dependency parser; parser behavior is
not hidden statistical evidence in the final decision because its emitted
tokens, dependencies, roles, spans, and version are retained in the evidence
graph and trace. The deterministic projector now requires or records:

- active and passive subject/object roles, plus bounded subject inheritance
  for coordinated predicates;
- predicate particles, embedded predicate and auxiliary boundaries, copular
  adjective/adverb complements, clause markers, and reporting-content edges;
- position-aware date and quantity binding, comparator, currency, unit,
  scalar-operator, polarity, modality, direction, and causal scope;
- named-entity labels, referential heads, coreference evidence, and
  proposition references; and
- explicit unresolved conditions whenever a core role or material scope
  cannot be assigned without guessing.

Parser improvements are prerequisites, not paraphrase permissions. Recovering
a missing predicate or argument can reveal either a valid transformation or a
new material conflict. Every parser tranche is therefore benchmarked in both
profiles before any rule is added on top of it.

### First complete-registry MRPC run, 2026-07-31

The offline benchmark is
`scripts/research/mrpc/run_mrpc_proposition_graph_benchmark.py`. It reuses the existing
spaCy tokenization extractor, cached embeddings, PSA top-32 projection, and a
pinned OEWN `2024` lexical index; no live service is required.

The gate is learned before verification, as required by the funnel. On the
new sentence-component split, the cold start `0.77` learned to
`0.7717701196670532` using only the 618-pair calibration partition. Verifier
outcomes were not visible during gate learning.

| Held-out result, 1,107 pairs | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| PSA + learned gate only | 0.77074 | 0.90283 | 0.83157 | 0.75519 |
| PSA + learned gate + strict graph proof | 0.88235 | 0.08097 | 0.14833 | 0.37760 |

The strict result is a failed recall result, not a high-scoring MRPC solution.
It proves that the executable plumbing is complete while the transformation
coverage is not yet sufficient for MRPC. The dominant remaining gaps are ETPC
contextual substitution and content-bearing addition/deletion. Broadening those
rules without a proof would violate the controlling deterministic contract.

The evidence fixes made during this run reduced whole-corpus unresolved counts:

| Evidence defect | First run | Corrected run |
| --- | ---: | ---: |
| date scope unresolved | 992 | 14 |
| quantity scope unresolved | 796 | 70 |
| no proposition extracted | 127 | 9 |

The authoritative machine-readable report, including all 1,107 held-out proof
traces and the after-the-fact ETPC type ledger, is
`reports/mrpc_proposition_graph_2026-07-31/mrpc_proposition_graph_report.json`.

### Anchored contextual and addition/deletion tranche, 2026-07-31

The next rule tranche was derived from the non-official-test portion of ETPC
and implements only independently anchored transformations:

- direct OEWN taxonomic predicate and argument substitutions require another
  exact role filler under the same event;
- pronoun/deictic substitutions and referential-head-preserving modifier
  changes require the same anchored predicate role;
- generic-agent ellipsis requires another directly matched role;
- unmatched relative modifiers require a matched governing referent;
- reporting wrappers require their reported proposition to be independently
  matched; and
- an omitted reporting-content edge is treated as a subordination/nesting
  change only when both the reporting event and content event are independently
  matched.

The extractor now also records clause role and governing-head provenance and
binds modal and negation markers to unique parser predicates before using a
surface-position fallback. A material-value correction makes extracted time
expressions compare their actual `normalized_entity`; before that correction,
different values such as `today` and `Saturday` collapsed to the same empty
shape.

The official test labels had already been opened by the first run. Therefore
the following runs are post-hoc development diagnostics, not new blind scores:

| Diagnostic version | Precision | Recall | F1 | Accuracy | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| v1 strict graph | 0.88235 | 0.08097 | 0.14833 | 0.37760 | Initial complete registry |
| v2 anchored context | 0.88034 | 0.13900 | 0.24009 | 0.41102 | Context and structural additions |
| v3 operator binding | 0.89744 | 0.14170 | 0.24476 | 0.41463 | Modal scope, names, generic ellipsis |
| v4 licensed residuals | 0.89344 | 0.14710 | 0.25261 | 0.41734 | Removed redundant count veto |
| v5 material correction | 0.89286 | 0.13495 | 0.23447 | 0.41012 | Correctly preserves changed times |
| v6 content topology | 0.89474 | 0.13765 | 0.23860 | 0.41192 | Recovered two true pairs, zero false pairs |
| v7 ETPC contextual lexicon | 0.89565 | 0.13900 | 0.24065 | 0.41283 | Added one true pair, zero false pairs |

The v6 report is
`reports/mrpc_proposition_graph_contextual_v6_2026-07-31/mrpc_proposition_graph_report.json`.
The learned candidate threshold remains `0.7717701196670532`, and its gate-only
held-out diagnostic remains F1 `0.83157`. The strict verifier is intentionally
far below that result because it abstains when equivalence is not proven. A
future publishable result requires a newly locked evaluation set; these
post-hoc diagnostics must not be represented as blind MRPC measurements.

The v7 tranche also tested, then rejected, broader parser coordination
recovery. Propagating coordinated arguments and shared conjunct subjects made
the graph more linguistically complete, but MRPC treats several list-member
deletions as paraphrases. The broad diagnostic fell to F1 `0.22588`; a
tightened version reached only `0.23419`. Neither coordination policy was
retained. The retained parser change is limited to observable `ccomp` content
edges under an allowlisted reporting or attitude predicate and does not change
the v6 classifications by itself.

A separate simulation added all 1,574 direct OEWN `similar` synset links
available to the benchmark vocabulary. It changed zero classifications and was
not added to the runtime. The retained contextual lexicon contains 17 verb
pairs that recur at least three times as sense-preserving ETPC development
transformations. Each application requires an independently matched role. The
only new held-out acceptance is `insist`/`say` with all three propositions
matched; no false acceptance was added. The v7 report is
`reports/mrpc_proposition_graph_contextual_v7_2026-07-31/mrpc_proposition_graph_report.json`.

An alternate post-hoc policy that treated every gate-passing non-conflict,
including unresolved graphs, as a paraphrase reached only precision `0.75893`,
recall `0.45884`, and F1 `0.57191`. Therefore abstention cannot be converted to
acceptance as a score shortcut. The next recall work must establish additional
deterministic transformations, especially contextual predicate alignment and
parser-complete core-role recovery.

### Current dual-profile development status, 2026-08-01

The rule engine was subsequently rebuilt around exact global alignment,
material scope guards, and the two authorization profiles above. The following
measurements are all post-hoc development diagnostics. Every evaluation label
in these rows has been inspected, directly or through an error ledger, during
the rework. **No row is a blind MRPC result.**

| Diagnostic | Profile | TP | FP | Precision | Recall | F1 | Main change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v18 | `strict_merge_v1` | 86 | 10 | 0.89583 | 0.11606 | 0.20550 | First dual-profile baseline |
| v18 | `mrpc_compatible_v1` | 167 | 16 | 0.91257 | 0.22537 | 0.36147 | Scoped compatibility baseline |
| v19 | `strict_merge_v1` | 86 | 10 | 0.89583 | 0.11606 | 0.20550 | Global alignment and material normalization |
| v19 | `mrpc_compatible_v1` | 168 | 16 | 0.91304 | 0.22672 | 0.36324 | Global alignment and material normalization |
| v20 | `strict_merge_v1` | 27 | 1 | 0.96429 | 0.03644 | 0.07022 | Strict-profile authority audit and parser completion |
| v20 | `mrpc_compatible_v1` | 154 | 18 | 0.89535 | 0.20783 | 0.33735 | Parser completion exposed unsafe omissions |
| v21 | `strict_merge_v1` | 27 | 1 | 0.96429 | 0.03644 | 0.07022 | Unchanged strict proof boundary |
| v21 | `mrpc_compatible_v1` | 146 | 5 | 0.966887 | 0.197031 | 0.327354 | Added-fact, discourse, constraint, and omission guards |
| v22 | `strict_merge_v1` | 27 | 0 | 1.000000 | 0.036437 | 0.070313 | Removed the last strict false acceptance |
| v22 | `mrpc_compatible_v1` | 151 | 1 | 0.993421 | 0.203779 | 0.338186 | Five typed pair canonicalizations plus parser and omission guards |
| v23 | `strict_merge_v1` | 27 | 0 | 1.000000 | 0.036437 | 0.070313 | Unchanged strict proof boundary |
| v23 | `mrpc_compatible_v1` | 151 | 0 | 1.000000 | 0.203779 | 0.338565 | Dependent copular complements made non-omittable |
| v24 | `strict_merge_v1` | 27 | 0 | 1.000000 | 0.036437 | 0.070313 | Occurrence-bound operators and proposition-reference topology |
| v24 | `mrpc_compatible_v1` | 155 | 0 | 1.000000 | 0.209177 | 0.345982 | Apostrophe-safe evidence plus occurrence-bound operators and content edges |

The v21 compatible result is the important precision checkpoint: false
acceptances fell from 18 to 5, while eight true acceptances were also removed.
Most of those lost positives had depended on blanket light-predicate deletion.
That behavior will not be restored. The current bottleneck is deterministic
transformation coverage, not PSA top-32 recall.

Two tempting generalizations were measured and rejected:

- treating two shared arguments as sufficient evidence for a predicate/frame
  rewrite produced 22 true-positive opportunities but also 13 false-positive
  opportunities; and
- deleting unmatched light or copular predicates broadly admitted material
  state, reporting, location, temporal, and argument changes.

Neither shortcut belongs in either authorization profile. A valid rewrite
must name the transformation, prove every material binding and operator, and
leave no unexplained residue.

### Superseded deterministic-rule roadmap

The earlier plan to recover recall through additional custody, report,
investment, communication, result-echo, light-predicate, converse, and control
canonicalizations is no longer the architectural direction. Those patterns
remain reproducible as historical diagnostic hypotheses only.

Recall work now belongs in contextual semantic analysis and directional
alignment, evaluated across sealed datasets and checked by a small proof
kernel. A benchmark error bucket is not a rule backlog. Lemma-wide possible
senses, named predicate pairs, domain constructions, fuzzy referent matching,
and omitted-clause templates cannot authorize equivalence.

### Occurrence and context-foundation run, v24

The clean v24 run rebuilt all 3,393 sentence graphs with
`core_web_sm@3.8.0`, fact extractor
`paraphrase_evidence_extraction@v18-occurrence-bound-operators-apostrophe-safe`,
and rule registry
`etpc_deterministic_graph_rules_v14_occurrence_bound_topology`. The graph
cache reported `cache_hit: false`.

Operator facts now bind to exact predicate token positions. A proposition
valued `content` edge must likewise resolve its argument token position to one
unique child-event occurrence, and selected parent events must preserve the
selected child pairing. There is no predicate-lemma fallback. Missing,
ambiguous, repeated-lemma, or cross-wired references abstain or conflict. The
same guard covers the strict semantic-hash path, strict and compatible global
alignment, one-sided content omission, reporting wrappers, and canonicalized
origin tuples.

The parser evidence now distinguishes complete quote delimiters from
apostrophes and passes the construction-aware predicate lemma into the
proposition record only at the exact bound occurrence. This repaired the
curly-apostrophe existential without guessing whether arbitrary contractions
mean `be` or `have`.

Compared with v23, the compatible profile gained eight positive acceptances
and lost four positive acceptances, for a net gain of four with no false
acceptances. Three losses are conservative proposition-topology failures where
one parser projection attaches multiple content events and the other attaches
one; one is an initial-report template near-miss. These are future parser or
typed-canonicalization work, not reasons to weaken the occurrence guard. At
least one new compatible acceptance uses the profile's sentence-local
pronoun license; the strict profile still rejects it. That license must be
revisited once the PRECEDES resolver is executable, because a production
referent substitution should consume an authenticated context fact rather
than an MRPC-local anchor.

The authoritative report is
`reports/mrpc_proposition_graph_occurrence_context_v24_2026-07-31/mrpc_proposition_graph_report.json`.
Like every row above, this is a post-hoc development diagnostic and is not
eligible for a blind performance claim.

### PRECEDES meaning-echo context

Sentence meaning is not always sentence-local. Pronouns, nominal aliases,
ellipsis, deictic time, and reporting attribution can depend on the meanings
of sentences that were processed immediately before the current sentence.
The ingestion pipeline retained `predecessor_sentence_id` and durable
`precedes` edges, but the proposition evidence and verifier rework did not
consume a predecessor meaning echo. That is a real missing semantic input, not
a license to infer antecedents from similarity.

The restored context contract is deterministic and provenance-bound:

- every SentenceNode carries only intrinsic, versioned, canonical meaning
  echoes derived from context-neutral proposition graphs;
- every sentence occurrence is identified by its authoritative
  `ExtractionRun-[:CONTAINS_SENTENCE]->SentenceNode` lineage edge;
- a non-initial occurrence carries a bounded snapshot of its immediate
  predecessor's echo on that occurrence-lineage edge, identified by exact
  extraction run, sentence ID, sentence index, semantic namespace,
  content/container boundary, graph hash, and echo hash;
- a direct SentenceNode-to-SentenceNode `PRECEDES` edge is a derived navigation
  projection for one extraction run, not the persistence authority for order
  or contextual memory;
- the current sentence's local graph remains immutable; context resolution
  produces a separate proof listing every substituted referent, inherited
  argument, attribution binding, or temporal anchor;
- context may resolve an explicit unknown or licensed ellipsis, but may never
  erase a current-sentence polarity, modality, quantity, time, location,
  attribution, or added event;
- ambiguous, missing, non-contiguous, cross-document, cross-namespace, stale,
  or hash-invalid echoes force abstention; and
- no benchmark row may act as context for another row. MRPC supplies isolated
  sentence pairs rather than authenticated preceding article sentences, so
  the current MRPC diagnostic remains sentence-local. Context-dependent MRPC
  examples stay unresolved until genuine source context is supplied.

The first context-aware rule tranche is limited to coreference, entity alias,
ellipsis, attribution, and temporal/deictic resolution. Each rule must name
the exact PRECEDES echo fact it consumed and include that echo's hash in the
decision proof. Sentence-local reporting, derivation, operator-binding, and
frame rules remain a separate tranche so context cannot silently launder a
local mismatch.

#### Persistence checkpoint, 2026-08-01

The serialization, validation, and transport half of this contract is now
executable. Tokenization builds a context-neutral `sentence_meaning_echo_v1`
from each initial proposition graph and a capacity-one
`precedes_meaning_memory_v1` for the next sentence. Both values use canonical
JSON, SHA-256 identities, exact sentence index and ID, and tenant, workspace,
semantic namespace, ontology profile, content ID, content type, and
content-source boundaries. The sentence-embedding and PSA handoffs preserve
the metadata, and SentenceNode ingestion revalidates the full payload and its
source-evidence hash.

The controlling persistence split is:

| Value | Authoritative persistence | Reason |
| --- | --- | --- |
| Context-neutral meaning echo and each explicit enriched successor | `SentenceNode` | Intrinsic to the sentence meaning under the cited evidence/extractor/ontology identities |
| Sentence position, predecessor identity, and `precedes_meaning_memory_v1` | `ExtractionRun-[:CONTAINS_SENTENCE]->SentenceNode` | Context belongs to one occurrence in one ordered extraction, not to a reusable physical sentence node |
| Direct `SentenceNode-[:PRECEDES]->SentenceNode` | Derived, extraction-run-qualified navigation projection | Useful for traversal but unsafe as the authority when nodes are reused or reordered |

A node-level predecessor-memory copy is permitted only as a denormalized cache
for a provably occurrence-specific node. A consumer must first resolve the
selected `ExtractionRun` occurrence and verify that the cached memory hash
equals the authoritative `CONTAINS_SENTENCE` edge hash. It may never treat the
node copy as intrinsic meaning or use it without occurrence provenance.

The v24 implementation currently writes both the intrinsic echo and the
predecessor snapshot to SentenceNode properties. That is a transitional
implementation, not the target authority model. No context resolver consumes
those fields yet. Before contextual resolution or reusable typed nodes enter
production, persistence must move the authoritative predecessor snapshot to
`CONTAINS_SENTENCE`; the node fields may then be removed or retained only
under the cache rule above.

Required implementation migration before the resolver is enabled:

1. Write the validated memory JSON, hash, schema/version, depth, predecessor
   index, and predecessor echo hash onto each `CONTAINS_SENTENCE` edge.
2. Validate the embedded memory against that edge's extraction run, position,
   target SentenceNode echo, namespace, ontology, and content boundary.
3. Read context only after resolving the selected or accepted ExtractionRun;
   never select a predecessor snapshot directly from the SentenceNode.
4. Qualify or rebuild direct `PRECEDES` projections from the chosen extraction
   run and treat any unqualified legacy edge as non-authoritative.
5. Backfill existing node-only predecessor snapshots to their unique
   occurrence edges where uniqueness is provable; otherwise discard the cache
   and regenerate it from the ordered extraction.

This means a reader can recover a sentence's initial semantic graph without
rerunning spaCy or rebuilding it from occurrence evidence. It also gives
Stitch a bounded, hash-addressed semantic input and gives future context rules
the exact previous graph without a graph-store traversal followed by a parser
run. Those are potential latency and composition improvements; they have not
yet been measured end to end, and the persisted echo is not itself permission
to merge meanings.

A local serialization probe over the 3,393 v24 MRPC sentence graphs gives the
opportunity an approximate scale. Re-projecting the already extracted evidence
into graphs took 5.533 seconds, while parsing their persisted semantic JSON
took 0.290 seconds, a 19.1x reduction for that narrow step. This excludes the
larger cost of rerunning spaCy and does not measure network or Neo4j access.
Median semantic JSON was 1,191 bytes; median full provenance graph JSON was
4,755 bytes. With the current self echo plus capacity-one predecessor memory,
median transported context was 5,257 bytes per sentence (p95 6,156, maximum
7,339 in this corpus). Persistence is therefore plausibly useful, but the
extra metadata is a real transport/storage cost that must be measured in the
production path.

The current echo records the initial extraction graph. Later WordNet,
ontology, entity-resolution, or Clock-B enrichment must publish an explicit
versioned intrinsic successor with new evidence and semantic hashes; it must
not mutate the original echo silently. Convergent typed physical nodes may
occur in multiple documents or sentence orders, which is why their intrinsic
echo remains on the node while every contextual predecessor snapshot is bound
to the exact occurrence-lineage edge. Occurrence-specific prose nodes follow
the same authority rule even when a node-level cache would happen to be
unambiguous.

MRPC remains context-free. An audit of the corpus found no authenticated
article-order/predecessor field: URL, date, and row order are insufficient.
Examples whose proof needs an antecedent therefore remain abstentions in the
benchmark. The serializer can retain the required input, but production
context consumption remains disabled until occurrence-edge persistence and
the proof-producing resolver are implemented.

### Acceptance tests

Before production enforcement, the graph verifier must demonstrate on a newly
locked blind partition:

- active/passive canonical equivalence;
- agent/patient reversal rejection;
- added and removed proposition rejection;
- attribution-source and quoted-claim preservation;
- quantity, comparator, currency, and unit normalization;
- negation, modality, and temporal scope preservation;
- entity/reference and coreference preservation;
- deterministic graph and proof hashes across retries;
- fail-closed behavior for missing, ambiguous, or overflow evidence;
- no cross-language, parser-version, normalizer-version, ontology-version, or
  rule-registry identity mixing; and
- a per-ETPC-type MRPC error ledger showing exactly which operations are
  covered, rejected, or unresolved.

Release scoring has an additional hard gate: a frozen rule registry must reach
at least `0.90` F1 on a fresh sealed evaluation whose labels were unavailable
during threshold, parser, and rule development. The current MRPC partitions
cannot satisfy that requirement because their labels have been inspected.
MTEB++ product ingestion, generated summaries/paraphrases, and downstream
scoring are explicitly disabled until that fresh sealed result reaches the
`0.90` F1 gate. Crossing 0.90 on any post-hoc diagnostic does not unlock it.
