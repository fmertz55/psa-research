# PSA Guide-Star and Topology Improvement Program

Status: active research plan  
Created: 2026-07-11

## Objective

Find the smallest, cheapest PSA reference frame that preserves or improves
semantic routing, relational invariants, and cross-model transport, while
determining whether the observed field has stable topology or only useful
low-dimensional geometry.

This program is offline. It must not modify the application embedding model,
application routing code, Neo4j indexes, or production PSA calibration.

## Current Evidence

The existing work establishes:

- Stable coordinate assignment within a fixed guide-star frame.
- Significant semantic locality under permutation controls.
- Better occupancy diversity than random-anchor frames that collapse.
- Stronger 115-star locality than the historical 75-star run.
- High agreement between 75- and 115-star signature geometry (`r=0.947`).
- Partial cross-model anchor-relative invariance.
- Weak exact cross-model neighborhood preservation.
- Strong path agreement between direct and composed rigid maps.
- Strong cycle return relative to shuffled-pairing nulls.
- Robustness to random dimensional compression, but not to model changes.
- A 40-star minimum-cost routing knee and a 64-star balanced field-fidelity
  operating point across twenty held-out source-domain splits.
- Separate, measurable effects from guide count, chart dimension, and
  edge-versus-density placement weight.
- Stable boundary rankings across 64 to 512 projection directions.

The evidence does not yet establish a model-independent topology. It supports
a partially shared relational geometry and an approximately path-consistent
transport family.

## Threats to Validity

The experiments must explicitly address these weaknesses:

1. The historical 75-to-115 comparison changed more than guide-star count.
2. Locality is partly expected because PSA is constructed from embedding
   similarities; controls must match occupancy and candidate cost.
3. Identical `PRECEDES` extraction is processing invariance, not a topological
   invariant of the coordinate field.
4. Direct and composed maps can agree while both miss the true target. Cycle
   consistency must be reported alongside intermediate target accuracy.
5. A consistently wrong family of transforms can close a cycle.
6. Compact raw coordinates inflate some anchor-distance and voxel metrics.
7. Corpus-calibrated quantiles may introduce address drift and evaluation
   leakage.
8. Three-dimensional MDS can fold or erase higher-dimensional structure.
9. Current contrast pairs are represented as two positive similarities rather
   than one signed semantic axis.
10. Random sentence anchors retain substantial native embedding geometry;
    random controls must be occupancy-, dimension-, and cost-matched.
11. Sentence-random splits leak document, topic, and paraphrase structure.
12. A 3,304-sentence corpus is too sparse to identify universal voids,
    filaments, cavities, or attractors.
13. The 115-star frame is a reference instrument, not semantic ground truth;
    reduced-frame fidelity establishes reproduction of that instrument only.
14. Projected boundary support does not detect non-convex holes or establish
    persistent topology.

## Frozen Evaluation Contract

### Data separation

- Use the frozen 3,304-sentence Qwen/BGE/MPNet corpus for development.
- Split calibration and evaluation by source document or semantic domain.
- Keep paraphrase-connected components in one split.
- Add unrelated corpora before making cross-corpus or topology claims.
- Never derive quantile boundaries from the evaluation fold.

### Primary operational metrics

- Recall of native top-10, top-25, and top-50 neighbors at fixed candidate
  fractions of 1%, 5%, 10%, and 20%.
- Paraphrase-pair recall at the same candidate fractions.
- Candidate-set enrichment over random selection.
- Wall-clock projection time and guide-dot-product count.

### Geometry and collapse metrics

- Pair-distance Pearson and Spearman correlation.
- Triplet-order accuracy.
- Trustworthiness and continuity.
- Local fold rate: native-distant pairs mapped into the same local region.
- Occupied cells, effective occupied cells, and maximum cell mass.
- MDS stress and retained positive eigenspectrum by coordinate dimension.
- Boundary support error against the full reference instrument.
- Sliced density error and local density-rank correlation at multiple scales.

### Cross-model metrics

- Centered signature correlation and anchor-rank correlation.
- Held-out-guide alignment error.
- Intermediate target error for every transport edge.
- Direct-versus-composed target error.
- Cycle return error normalized by source-field diameter.
- Routing recall at each intermediate target, not only after cycle return.

### Selection rule

For each experiment, report the Pareto frontier rather than one opaque score.
The recommended frame is the smallest configuration whose lower confidence
bound retains at least 95% of the 115-star baseline on primary routing metrics,
does not reduce effective occupancy by more than 10%, and does not materially
worsen cross-model relational invariants.

Use the one-standard-error rule when several configurations satisfy the gates.

## Phase 1: Minimum Guide-Star Frontier

Test guide-star counts:

`16, 24, 32, 40, 48, 56, 64, 75, 96, 115`

Compare these selection families:

- Tier-balanced random subsets.
- Contrast-pair-balanced random subsets.
- Farthest-point subsets in guide embedding space.
- Redundancy-pruned subsets.
- Corpus-coverage subsets selected against sentence embeddings.
- Cross-model-stable subsets.
- Hybrid semantic-coverage and cross-model-stability subsets.
- Consensus edge/density subsets evaluated in every participating model.

Run at least 50 subset trials for stochastic selectors and report paired
bootstrap confidence intervals. A guide-star count is not considered optimal
until it replicates across source-domain holdouts.

### Influence and deletion analysis

For each guide star and contrast pair, measure the change caused by deletion:

- Routing recall at fixed candidate cost.
- Pair-distance preservation.
- Effective occupancy and concentration.
- Cross-model signature/rank agreement.
- Local fold count.

This produces a guide-star influence atlas and identifies redundant, harmful,
and uniquely necessary anchors.

## Phase 2: Placement and Density

Build candidate pools from:

- Existing synthetic guide stars.
- Paraphrases of existing guide stars.
- WordNet glosses and examples.
- Frame descriptions.
- Corpus medoids and persistent paraphrase attractors.
- Out-of-domain medoids from held-out corpora.

Add anchors where one or more of these conditions hold:

- Low maximum anchor similarity.
- High local reconstruction error.
- High fold/collision rate.
- High local intrinsic dimension.
- Poor cross-model anchor-rank stability.
- High paraphrase false-negative density.

Penalize candidates that duplicate an existing anchor neighborhood or improve
locality only by reducing occupancy diversity.

The target is adaptive anchor density: sparse in flat, stable regions and dense
only where semantic geometry bends, folds, or becomes uncertain.

## Phase 3: Signature Formulation

Compare:

1. Current tier-weighted raw similarities.
2. Sentence-centered signatures.
3. Double-centered sentence/anchor signatures.
4. Whitened signatures.
5. Signed contrast axes:
   `cos(sentence, positive) - cos(sentence, negative)`.
6. Hybrid signed contrast axes plus unpaired anchors.

Sweep the modality-tier weight and all other tier weights under held-out
evaluation. No weight may be selected on the final evaluation corpus.

Signed contrast axes are especially important: they test whether 56 paired
semantic directions can replace 112 positive anchor activations while reducing
the common positive component that compacts the current field.

## Phase 4: Coordinate Chart and Dimensionality

Sweep coordinate dimensions:

`2, 3, 4, 5, 8, 10, 16, 32`

Compare:

- Classical landmark MDS.
- Metric MDS.
- Nyström/landmark out-of-sample MDS.
- PCA of centered signatures.
- Seeded random projection of centered signatures.

Operational indexing dimension and visualization dimension must be treated as
separate choices. A higher-dimensional routing chart may coexist with a 3D
visualization chart.

## Phase 5: Stronger Cross-Model Transport Tests

For every edge and cycle:

- Fit on a guide-star calibration subset.
- Evaluate on held-out guide stars and held-out semantic domains.
- Report true intermediate target routing and coordinate error.
- Compare direct and composed paths only after target accuracy is established.
- Repeat with reverse paths and multiple model orderings.

Required nulls:

- Independent shuffled anchor labels.
- Cycle-consistent but semantically wrong anchor permutations.
- Occupancy-matched random sentence anchors.
- Random orthogonal maps preserving source distribution.
- Same-family and different-family embedding models.
- Anchor-text paraphrase perturbations.

A transport law is supported only when it is both path-consistent and accurate
on unseen targets.

## Phase 6: Topology and Morphology

Do not infer topology directly from empty voxels. First establish chart
fidelity and sampling sufficiency.

Measure:

- Local intrinsic dimension.
- Connected components across radius/density filtrations.
- Persistent homology (`H0`, `H1`, and, if supported, `H2`).
- Bootstrap persistence and bottleneck/Wasserstein distances.
- Stability across models, corpora, chart dimensions, and anchor subsets.
- Density-corrected void persistence.
- Attractor persistence under corpus growth and frame changes.

Topology controls must preserve point count, density spectrum, and boundary
shape. A hole is evidence only if it persists across resampling, chart
dimensions, and unrelated corpora while exceeding matched null persistence.

## Phase 7: External Validity

Replicate the frontier and topology tests on:

- Multiple unrelated domains.
- A corpus at least an order of magnitude larger.
- Same-family and different-family embedding models.
- Paraphrase, contradiction, entailment, and retrieval labels independent of
  the embeddings used to construct PSA.
- Anchor paraphrases and lexical perturbations.

Claims should follow this hierarchy:

1. Useful routing geometry.
2. Stable within-frame relational structure.
3. Cross-model relational invariance.
4. Accurate, composable transport.
5. Persistent cross-corpus topology.

Each claim requires the preceding gates; cycle closure alone cannot skip the
accuracy and topology gates.

## First Deliverables

1. Offline guide-star frontier benchmark and machine-readable results.
2. Initial count/selection Pareto table for Qwen.
3. Projection-dimension stress and routing table.
4. Raw-versus-centered-versus-bipolar signature comparison.
5. Cross-model evaluation of the leading configurations.
6. Guide-star influence and semantic-coverage maps.
7. Strong-null cycle and intermediate-accuracy report.
8. Topology feasibility report before persistent-homology claims.

## Completed Frontier Result

The twenty-split count, edge, density, sensitivity, and cross-signal study is
reported in `reports/research/benchmarks/psa/PSA_GUIDE_STAR_EDGE_DENSITY_RESULTS.md`. It identifies
40-star efficiency, 64-star balanced, and 75-star high-fidelity reduced
operating points for this corpus. These are tunable calibration profiles, not
universal constants or application defaults.

## Wikipedia Substrate Expansion

The next substrate and calibration program is defined in
`research/programs/PSA_WIKIPEDIA_SUBSTRATE_AND_POPULARITY_CALIBRATION_PLAN.md`. It
requires separate occurrence-, expression-, source-, and class-balanced views
so exact repeats, distinct paraphrases, source prevalence, and model-encoded
popularity cannot be conflated.

The first bounded run uses only a frozen, curated English Wikipedia article
manifest and targets 100,000 to 105,000 accepted prose sentence occurrences.
The curation is weighted toward AI methods, AI systems and tooling, concrete
industry implementations, organizations/governance, and adjacent enterprise
domains. Corpus size is measured after prose filtering; exact-repeat savings and
contextual object-occurrence embedding work are reported separately.

## Immediate Application Improvement Program

The research exposes application improvements that can be made before a new PSA
frame is promoted. These changes must be measured independently and must not
silently alter the current `v2_115stars` addresses.

### Workstream A: Preserve occurrence attestations

The ingestion pipeline may consolidate duplicate text into one canonical
sentence representation, but it must preserve every independently observed
occurrence. Re-embedding or duplicating the canonical `SentenceNode` is not
required.

For each occurrence, retain at minimum:

- normalized-text hash;
- canonical sentence ID;
- source/container, page, revision, and section identity;
- ingestion run and timestamp;
- paraphrase-group identity when assigned;
- exact-repeat versus distinct-expression classification.

Idempotency is defined at the occurrence identity, not at text alone. Replaying
the same source revision must not add mass, while the same text in a different
source must add an attestation.

Acceptance metrics:

- canonical embedding count remains unchanged under duplicate injection;
- occurrence count changes by the expected amount;
- class-balanced coordinates and routing remain invariant;
- source and occurrence popularity components change independently;
- every popularity contribution is traceable to source provenance.

This is required before Wikipedia reingestion. Lost occurrence evidence cannot
be reconstructed reliably after canonical deduplication.

### Workstream B: Collapse retrieval by semantic class

The current vector path retrieves `SentenceNode` and `ParaphraseGroup` indexes
independently and merges results by node ID. A sentence and the paraphrase group
that represents it can therefore occupy separate candidate and context slots.

Add a semantic-class consolidation stage after multi-path retrieval:

1. Resolve each sentence hit to its `ParaphraseGroup`, when present.
2. Use the group as the canonical candidate identity.
3. Aggregate maximum and supporting vector scores rather than duplicating the
   idea.
4. Retain one representative expression plus bounded supporting expressions.
5. Aggregate distinct sources and preserve their citations.
6. Apply per-class and per-source caps before filling the context budget.
7. Leave singleton sentences as provisional one-member semantic classes.

Do not discard member evidence. Consolidation changes candidate accounting and
presentation, not provenance.

Acceptance metrics:

- semantic duplicate rate in the final candidate set;
- unique semantic classes per fixed context budget;
- distinct sources and citation coverage;
- retrieval and paraphrase recall;
- answer repetition rate;
- prompt-context tokens, latency, and reranking cost;
- no loss on singleton or unclustered knowledge.

This is the leading immediate response-quality improvement because it spends
the retrieval budget on distinct ideas instead of duplicate representations of
one idea.

### Workstream C: Repair field-density visualization

The current topic-coherence visualization computes consolidation as
`paraphrase_mass / sentence_count` for sentences already selected through the
same `ParaphraseGroup`. Those values are ordinarily equal, so populated groups
can appear fully consolidated regardless of actual cohesion or diversity.

Replace the metric with separately named measurements:

- occurrence density: source attestations in the region;
- expression density: distinct normalized expressions;
- source diversity: effective number of independent sources;
- paraphrase cohesion: member-to-centroid similarity distribution;
- expression diversity: lexical and syntactic variation within the class;
- class concentration: entropy or effective class count when a visual region
  contains multiple semantic classes;
- neighborhood coherence: locality of semantic graph transitions.

Do not combine these into one field until each component is visible and tested.
The visualization may map components to separate layers or user-selectable
views.

Acceptance metrics:

- singleton, repeated-expression, diverse-paraphrase, and multi-source fixtures
  produce distinguishable values;
- duplicate injection affects occurrence density but not cohesion;
- paraphrase perturbation affects expression diversity and orbit shape;
- visual values agree with direct Neo4j audits;
- no metric is structurally constant for all populated regions.

### Workstream D: Make popularity an explicit retrieval prior

The popularity service already computes sentence count, source count, citation
count, and temporal spread. Preserve and expose those raw components alongside
the blended score.

Add versioned retrieval profiles:

- `structural`: semantic score with class-balanced candidate accounting;
- `balanced`: semantic score plus a bounded popularity prior;
- `prevalence-aware`: stronger but capped source/prevalence influence;
- `popularity-normalized`: unavailable until the Wikipedia warp experiment
  passes held-out and external-validity gates.

Popularity must rerank candidates inside a semantically relevant set. It must
not rescue candidates below the semantic relevance floor or allow one popular
class to fill the context budget.

Acceptance metrics:

- retrieval quality by popularity decile;
- source diversity and long-tail recall;
- calibration of popularity contribution to final rank;
- resistance to one-source repetition and duplicate injection;
- task-specific benefit on prevalence-sensitive queries;
- no material loss under the structural profile.

Customer-specific popularity blending is a valid future tuning control. It is
not yet a universal default.

### Workstream E: Shadow reduced and higher-dimensional PSA profiles

The production application currently uses a pinned 115-star, three-dimensional
frame and calibrated voxel addresses. Do not replace it in place.

Use the existing full 115-star signature to compute shadow measurements:

- 64-star balanced and 75-star high-fidelity subset signatures;
- 8D and 16D operational coordinates;
- candidate regions and routing recall;
- boundary, density, and local-rank residuals;
- computation and storage cost;
- disagreement with current production routing.

The 3D frame remains the visualization and compatibility address during shadow
evaluation. Higher-dimensional coordinates are sidecar routing features and do
not define production voxel IDs until migration is approved.

Promotion requires:

- a new immutable frame and calibration version;
- repeatable gains on locked customer-like corpora;
- shadow agreement and disagreement analysis;
- readdressing or reingestion procedure;
- dual-read compatibility and rollback;
- no silent mixing of coordinates from different frames.

### Implementation Order

1. Freeze the first-light baseline.
2. Add occurrence-attestation preservation before Wikipedia reingestion.
3. Add semantic-class retrieval consolidation and measure response quality.
4. Repair field-density and consolidation visualization metrics.
5. Expose raw popularity components and explicit retrieval profiles.
6. Run 64/75-star and 8D/16D PSA profiles in shadow.
7. Consider frame migration only after the shadow and external-corpus gates.

### Application Release Gate

Every workstream must ship behind a versioned configuration or feature flag and
must report before/after metrics from the same locked query set. A release is
accepted only when it improves its declared primary metric without materially
degrading semantic recall, source grounding, latency, or frame compatibility.

The occurrence ledger and semantic-class consolidation can improve the current
application without changing the embedding model. Reduced guide catalogs,
higher-dimensional operational routing, and popularity normalization remain
shadow features until their respective migration and validation gates pass.
