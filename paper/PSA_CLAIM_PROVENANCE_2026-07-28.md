# PSA Claim Provenance Ledger

**Audit date:** 2026-07-28  
**Scope:** `latex/psa_semantic_field.tex` and
`latex/psa_semantic_field_industry_2026_response_ready.tex`

This is a claim-to-artifact map, not a substitute for archiving code,
environments, input licenses, and immutable release hashes. A manuscript claim
is retained only when the current repository contains a machine-readable source
or a content-addressed benchmark report. Values found only in older manuscript
prose are quarantined.

## Retained Quantitative Claims

| Claim family | Artifact | Selector or evidence | Status |
| --- | --- | --- | --- |
| Current corpus has 3,424 sentences and 5,860,176 unordered pairs | `temp/psa_frame_overlap/psa_quantile_grid_sweep_results.json` | `.n_sentences`, `.total_pairs` | Retained |
| 75- and 115-signature RSA, Spearman, and kNN fidelity | `temp/psa_frame_overlap/psa_local_paper_gates_20260707T163947Z.json` | `.semantic.single_space` | Retained |
| Semantic tier and contrast split pair correlation, floor, and excess | same `20260707T163947Z` artifact | `.semantic.pairs` | Retained |
| 1,000-trial independent random-projection nulls and median 115D projection fidelity | `temp/psa_frame_overlap/psa_local_paper_gates_20260707T162100Z.json` | `.random_projection_nulls`, including `.trials == 1000` | Retained. This is the completed random-projection run. |
| 1,000-trial random sentence-anchor nulls and empirical p-values | `temp/psa_frame_overlap/psa_local_paper_gates_20260707T163947Z.json` | `.random_sentence_anchor_nulls`, including `.trials == 1000` | Retained |
| Raw and quantile grid occupancy, labeled-pair shell diagnostics | `temp/psa_frame_overlap/psa_quantile_grid_sweep_results.json` | `.frames`, `.labeled_paraphrase_pairs` | Retained; labeled rows are explicitly descriptive blocking recall, not classifier F1. |
| Quantile radius recall/cost curves for cosine thresholds and legacy labels | `temp/psa_frame_overlap/psa_quantile_radius_sweep_results.json` | `.frames`, `.threshold_pair_counts`, `.labeled_paraphrase_pairs` | Retained |
| Affine 1,143 vs. quantile 2,739 occupied cells | `temp/psa_frame_overlap/psa_local_paper_gates_20260707T163947Z.json` | `.calibration_comparison_115.affine_minmax`, `.calibration_comparison_115.quantile` | Retained |
| Held-out radius selection over 200 splits | same `20260707T163947Z` artifact | `.heldout_radius_retune_115`, including `.trials == 200` | Retained |
| Matched-candidate-fraction signature blocking | same `20260707T163947Z` artifact | `.matched_cost_signature_blocking` | Retained |
| 115-frame stored coarse and depth-3 address reproduction; 75-frame coarse mismatch | `temp/psa_frame_overlap/psa_decensor_sweep_results.json` | `.frames["115"].stored_coarse_match`, `.frames["115"].stored_full_d3_match`, `.frames["75"].stored_coarse_match` | Retained |
| Sequential and randomized locality, 638 `PRECEDES` edges, and permutation results | `psa_test_runs/run1_order_text_first/results/2026-04-16T193038Z_jaccard_locality.json`; `psa_test_runs/run2_random_order/results/2026-04-17T181005Z_jaccard_locality.json`; `psa_test_runs/run2_random_order/results/2026-04-17T195232Z_falsify_scramble.json` | Top-level locality, edge, and permutation fields | Retained as legacy 75-star locality evidence |

The `psa_local_paper_gates_latest.json` alias intentionally skips the expensive
random-projection run. It is not valid provenance for the random-projection
claims even though it is valid for the later random-sentence, held-out, and
matched-cost sections.

## Quarantined Claims

The following values appeared in prior manuscript prose but their originating
machine-readable artifacts were not found in the repository snapshot. They were
removed from the active manuscripts rather than copied into a new artifact:

- matched-distance paraphrase-control counts, match error, and cell rates;
- random sentence-anchor and random unit-anchor effective occupancy summaries;
- 75-to-115 coarse AMI and exact 3D kNN-overlap range;
- depth-2/depth-3 frame-transfer co-membership recall and lift;
- paraphrase-mass and group-count consolidation deltas.

Reinstatement requires the original raw output plus the evaluator and input
manifest, or a clean deterministic rerun that creates a new immutable artifact.
Do not reconstruct a “raw” artifact by transcribing numbers from an older paper.

## Architecture Claims

The manuscripts now separate the representation contract from classifier
performance:

1. Every ingested sentence remains a durable sentence node.
2. An unmatched sentence remains free-range.
3. A `ParaphraseGroup` represents two or more verifier-confirmed equivalents,
   not a singleton storage convention.
4. Candidate generation covers nearby free-range sentences, established
   groups, high-membership groups, and current-batch items.
5. PSA proximity bounds candidate work; a native-vector, cross-encoder, or
   application verifier decides equivalence.

These are system-contract statements. The current repository has unit-level
coverage for the repaired contract, but the corrected route has not yet received
a fresh end-to-end benchmark. Superseded pre-repair results are not eligible for
manuscript citation.
