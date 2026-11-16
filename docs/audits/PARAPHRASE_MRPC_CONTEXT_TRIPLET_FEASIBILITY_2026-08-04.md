# MRPC context-triplet feasibility probe

Date: 2026-08-04

## Outcome

This small probe produced a **go signal**, not a new benchmark score.

In a frozen eight-pair sample, recovered contemporary or syndicated context
exposed two useful, fully explainable signals that the focal sentences alone do
not represent:

1. an omitted description can be bound to its content (`the criticism` -> the
   following criticism of Ota); and
2. a candidate sentence can be shown to cover the focal sentence **plus an
   adjacent scale fact**, revealing a range-of-scope mismatch.

Those signals identify two plausible corrections among four sampled current
errors: one false negative and one false positive. They are manual candidate
corrections, not detector outputs. No model, feature, threshold, prediction, or
MRPC label was changed.

The controlling held-out result therefore remains:

| Metric | Value |
| --- | ---: |
| Precision | 0.790857 |
| Recall | 0.933873 |
| F1 | 0.856436 |
| TP / FP / FN / TN | 692 / 183 / 49 / 183 |

## Frozen sample

The source artifact is
`reports/mrpc_semantic_relation_model_search_2026-08-03/held_out_reveal.json`
(SHA-256 `a5b228d25affc998a0df71471d115369312c958b6534096ebe637fb57ff0d79a`).

Pairs were split into the four confusion cells, ranked by decreasing
completeness of the two MRPC provenance records, then by increasing `pair_id`.
The first two pairs in each cell were selected before source recovery:

| Cell | Pair IDs |
| --- | --- |
| False positive | 915, 1591 |
| False negative | 248, 1152 |
| True positive | 1138, 146 |
| True negative | 466, 797 |

This is an intentionally tiny, provenance-biased feasibility sample. It is not
an estimate of corpus-level prevalence or F1 gain.

## Evaluation boundary

The focal sentence keeps its own proposition ledger. Neighboring sentences may
only contribute a separate, traceable overlay:

- `R`: unique referent binding;
- `E`: ellipsis or explicitly presupposed topic binding;
- `S`: separately represented time, scale, polarity, modality, attribution, or
  condition attached to the same event; or
- `N`: no licensed change.

Context may not donate an arbitrary neighboring fact to the focal sentence.
That restriction is what prevents article relatedness from becoming
paraphrase equivalence.

## Pair results

| Pair | Cell | Context result | Decision value |
| ---: | --- | --- | --- |
| 915 | FP | `R`: the preceding text binds `he` to Colin Powell. The quoted words in A remain extra scope. | Useful binding, but no correction by itself. |
| 1591 | FP | `S:scale`: the source places the Shiite 20% statement next to the Sunni 77% statement; B combines both. | **Candidate correction.** B spans current + adjacent scope, so A and B are not strict paraphrases. |
| 248 | FN | `E/R`: the sentence after “joined the criticism” states Koizumi's criticism of Ota. | **Candidate correction.** The abstract object in A is resolved to the content stated in B. |
| 1152 | FN | The immediate neighbors continue the `he`/`him` chain but do not reach the named antecedent. | One-hop triplet is insufficient; keep unresolved. |
| 1138 | TP | `N`: surrounding bombing details add no needed binding. | Correct decision remains strict. |
| 146 | TP | `N`: surrounding investigation details do not alter the matched quotation. | Correct decision remains strict. |
| 466 | TN | A secondary historical excerpt places the specific resignation quotation after the general “was forthright” description; its content remains a neighboring proposition. | Correct decision remains not strict; naive context union would be harmful. |
| 797 | TN | The cancer battle and kidney-removal history are compatible but distinct facts. | Correct decision remains not strict; article co-membership is not equivalence. |

The strongest recoveries are visible in contemporary or syndicated copies:

- Pair 1591's two demographic statements are adjacent in the
  [Taipei Times copy](https://www.taipeitimes.com/News/world/archives/2003/07/07/2003058386).
- Pair 248's next sentence spells out the content of Koizumi's criticism in the
  [Reuters copy](https://www.taipeitimes.com/News/world/archives/2003/06/28/2003057205).
- Pair 1152's local quote sequence is preserved in the
  [Telegraph India copy](https://www.telegraphindia.com/sports/south-african-woman-fears-for-life-why-would-i-want-to-blackmail-shane-warne-when-he-is-a-friend-of-mine/cid/1039233).
- The positive controls have article-level context in the
  [Afghanistan report](https://www.sfgate.com/news/article/Afghanistan-s-bloodiest-day-in-months-at-least-2596078.php)
  and the
  [German investigation report](https://www.taipeitimes.com/News/world/archives/2003/09/28/2003069573).
- Pair 915 is preserved in a
  [syndicated Powell report](https://www.seattlepi.com/national/article/saddam-victims-honored-1124335.php),
  and the original Sydney Morning Herald account underlying pair 797 is
  [reproduced by ANU](https://oa.anu.edu.au/obituary/dusty-slim-33574).
- Pair 466 has only
  [secondary historical context](https://acuresearchbank.acu.edu.au/download/eabcf422e231b2b679dae250ca2877917f8f111b144b5e0f343b2ca5a1e20c9c/35209611/Mayne_2016_A_history_of_tamar_in_relation_to.pdf)
  in this probe and is not treated as an authenticated triplet.

Recovery was sufficient to inspect the phenomena, but not to authenticate both
original article triplets for every pair. In particular, reachable copies often
contain light syndication edits. That is why this probe does not emit a revised
confusion matrix or F1.

## Sham-context control

Neighbor bundles were rotated once through the pair order
`915 -> 1591 -> 248 -> 1152 -> 1138 -> 146 -> 466 -> 797 -> 915`.
The substituted contexts produced zero unique referent/topic bindings and zero
candidate decision changes. This is a simple control against accepting mere
neighbor availability as evidence.

## What this establishes

The triplet idea is worth testing in the ingestion system. The useful unit is
not a larger text embedding. It is an occurrence-bound evidence overlay that
can say, for example:

```text
pair 1591
B support range: A.current + A.next(scale)
strict paraphrase: no
```

or:

```text
pair 248
A.current object: unresolved("the criticism")
A.next content: criticism(Ota, rape remarks)
overlay: unique ellipsis/topic binding
strict paraphrase: yes
```

The probe also establishes a real limit: a fixed one-sentence predecessor does
not always reach a named antecedent. The ingestion graph should follow a short,
explicit referent chain when available rather than silently widening the text
window or importing article-wide facts.

## Bounded follow-up

When the source articles are ingested:

1. replay these same eight IDs using exact `previous/current/next` occurrence
   nodes and source hashes;
2. emit only the `R`, `E`, and `S` ledger deltas above;
3. run the frozen dashboard once with that single context-overlay variable; and
4. report the real paired prediction and metric deltas.

Until then, the two candidate corrections are breadcrumbs, not score credit.
