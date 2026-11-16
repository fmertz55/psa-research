# FrameBase Hierarchy MRPC Result

**Date:** 2026-08-04  
**Status:** Rejected diagnostic experiment; no runtime or membership authority  
**Controlling MRPC score:** P `0.790857`, R `0.933873`, F1 `0.856436`

## Outcome

FrameBase resolution is now plural, occurrence-bound where the frozen evidence
permits it, pinned, and auditable. The hierarchy experiment did **not** improve
the controlling MRPC result and is not promoted.

The corrected strict-v2 evidence contract found no licensed one-way frame
specializations at distance 1 or greater among the 618 calibration pairs. Its
literal veto therefore changed zero predictions. The additive diagnostic was
also a no-op:

| Calibration lane | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---:|
| Frozen compact resolver | 0.791753 | 0.945813 | 0.861953 | 384 / 101 / 22 |
| Strict FrameBase literal veto | 0.791753 | 0.945813 | 0.861953 | 384 / 101 / 22 |
| Strict FrameBase additive diagnostic | 0.791753 | 0.945813 | 0.861953 | 384 / 101 / 22 |

Strict v2 was stopped at calibration. No second held-out reveal marker or
held-out result report was created.

## What changed

- FrameBase cache records preserve every distinct `(lemma, part of speech,
  frame)` mapping instead of collapsing a lemma to its first frame.
- WordNet evidence carries plural `frame_uris`, with scalar compatibility only
  as an input fallback. Ambiguous lemma-only candidates fail closed.
- The runtime resolver selects one contextual synset and retains all frames
  licensed by that synset. Evidence reconstruction preserves those plural
  frames end to end.
- The pinned hierarchy contains 24,572 synset-to-frame mappings, 1,636 directed
  macroframe inheritance edges, and 1,020 frames. `Top_frame` and
  `isSimilarTo` are non-licensing.
- Scorer-visible evidence is limited to 13 named, nonnegative integer counts.
  Raw text, lexical identifiers, and raw Cartesian frame paths never reach the
  scorer or MicroGPT allowlist.

## Rejected v1 reveal

The preregistered v1 held-out replay reproduced the frozen baseline exactly and
then reached P `0.792148`, R `0.925776`, F1 `0.853765`: a change of
`-0.002671`, with paired-bootstrap 95% interval
`[-0.006766, +0.000972]`. Nine predictions changed: three false positives were
repaired and six true positives were lost.

A required post-reveal audit found that eight of those nine vetoes were
artifacts. The two sentences had identical plural frame sets, but a Cartesian
product compared different members inside the same shared set and manufactured
opposite-direction candidate paths. Those raw paths were not valid
sentence-level hierarchy evidence. The only genuine v1 path was `Statement`
versus `Telling`, and it changed an MRPC-positive pair from true positive to
false negative.

V1 remains preserved as rejected scientific evidence. It must not be quoted as
a valid FrameBase score or reused for tuning. Strict v2 fixes the contract by:

1. letting exact plural-set identity dominate;
2. excluding raw Cartesian paths from all scorer-visible inputs;
3. requiring a licensed strict set relation for every directional distance;
4. treating ties, mixed directions, and incomplete coverage as ambiguous; and
5. performing calibration only after the correction.

## Interpretation

FrameBase is not exhausted as a semantic grounding resource. Its current
pinned **macroframe hierarchy** is exhausted for this narrowly declared MRPC
variable: it supplies exact, sibling, unrelated, ambiguous, and unmapped
categories, but no calibration example contains a licensed strict
specialization that can support the proposed veto.

This also prevents an unsupported `assassinated` versus `killed` rule. The
available snapshot grounds both relevant macroframe paths at `Killing`; raw
lexical-unit microframes are siblings, not a recorded political-motive child
relation. The system fails closed rather than inventing an ontology edge.

## Authority and breadcrumbs

This experiment has no classification, membership, database-write, edge-write,
or service-integration authority.

The bounded follow-up items are:

- Startup recovery still reads an obsolete Redis prefix and applies tenant and
  profile filters to finalized shared FrameBase nodes.
- An authoritative reimport is still needed before direct pinned
  synset-to-frame records can be assumed complete outside the frozen benchmark
  artifact.
- Sentence-level predicate identity can still union multiple same-lemma
  WordNet records; nine of 8,919 benchmark predicate occurrences expose more
  than one synset and correctly fail closed in hierarchy evidence.
- The controlling MRPC graph features remain multiset summaries. A separate
  preregistered experiment should measure labeled predicate-role-entity path
  intersection and conflict before topology is added to the score.

## Reproducibility

- Pinned hierarchy artifact SHA-256:
  `cac286a636fe6f43292a5f856d5ac49c2523fac1ecd0cb17805173d419d735c2`
- Pinned hierarchy resource payload SHA-256:
  `f072ff3c48bf0ecac8c7ea045c7ba27e55bc4aa199be5fa0bab1fea1217ab5e6`
- Rejected v1 selection file SHA-256:
  `dbf3079f0e3c6f72036ca54daca1dd7f33f7e5a5a4b560caf963ea501ef81690`
- Rejected v1 reveal marker SHA-256:
  `c36a19fee103a0f14a3bc96f3622c6a517ecc6aba3d49d06bd38bc47904f98d5`
- Rejected v1 report SHA-256:
  `d55d018be057a28ccb67ed0060d339e999afff1776f5932a32cebb1dcbcfe201`
- Strict-v2 calibration selection file SHA-256:
  `daa9eb1856fa3e97108c01526937bba068b2c0109f30164c6015c0ebbb320dac`
- Strict-v2 internal selection SHA-256:
  `58080cc1c2a4d1b02278c69b2d75a1c0d978727fdb81b8d94ed245b1dc433165`

Focused plural-resolution, reconstruction, hierarchy, benchmark, and snapshot
tests pass. The final Graphify snapshot and impact queries are recorded with
the worktree handoff.
