# PSA Multiscale Geometry Upgrade Plan

Status: active design
Created: 2026-05-05

## Trigger

The PSA baseline initially reported weak cross-space geometry because tests
treated raw voxel labels from different octree levels as one Euclidean grid:

- `L1_v_10_10_10`
- `L2_v_20_20_20`
- `L3_v_40_40_40`

These labels can describe the same canonical semantic region at different
resolutions. Raw distance made cross-layer neighbors look far apart. Normalizing
coordinates into the L1 frame materially improved locality and showed strong
same-layer decay.

## Phase 1: Measurement Correctness

Implemented foundation:

- Use canonical L1-normalized voxel coordinates for distance.
- Persist `voxel_distance_mode = canonical_l1_normalized` in result JSON.
- Add same-layer locality curves for `L1`, `L2`, and `L3`.
- Add full 4096-D sentence-pair baseline (`baseline-sentence`) to distinguish
  centroid/truncation artifacts from real field geometry.

Acceptance criteria:

- Same-layer curves are reported for every locality/baseline run.
- Raw-label coordinate distances are not used in scientific metrics.
- Result JSON always records the distance mode.

## Phase 2: Runtime Coordinate Contract

Implemented foundation:

- Shared coordinate module: `app/consolidated_clustering_v2/routing/psa_coordinates.py`.
- Voxel exports include `canonical_position`.
- Topic coherence edge locality uses canonical distance.

Next runtime persistence target:

- Sentence and paraphrase writes should persist:
  - `psa_level`
  - `psa_voxel_x`, `psa_voxel_y`, `psa_voxel_z`
  - `psa_canonical_x`, `psa_canonical_y`, `psa_canonical_z`
  - `psa_distance_mode`

Acceptance criteria:

- Services consume structured PSA fields or the shared parser, not ad hoc regex.
- Cross-layer routing/coherence treats same-region multiresolution voxels as near.

## Phase 3: PSA Coarse Index + Full-Vector Reranker

The current result supports PSA as a coarse semantic locality index, not as a
replacement for full embedding geometry. Retrieval should therefore be staged:

1. Use PSA neighborhood expansion to select candidates.
2. Rerank candidates by full-vector similarity.
3. Optionally blend rerank score with source diversity, recency, and graph signal.

Acceptance criteria:

- Candidate generation latency improves or remains acceptable.
- Recall is measured before rerank; precision/NDCG is measured after rerank.
- Full-vector rerank is mandatory for final ordering when embeddings exist.

## Phase 4: Medoids and Multi-Centroids

Single centroids smooth voxel populations heavily. Current runs show centroid
mean cosine around `0.75` while sampled sentence-pair mean cosine is around
`0.36`, so centroids are useful summaries but poor shape descriptors.

Planned voxel summaries:

- `medoid_sentence_id`: occupant closest to the voxel centroid.
- `medoid_embedding`: optional cached medoid vector or vector reference.
- `subcentroids`: 2-4 local summaries for high-variance voxels.
- `intra_voxel_variance`: split/rerank signal.
- `summary_mode`: `centroid`, `medoid`, or `multi_centroid`.

Acceptance criteria:

- High-variance voxels expose medoids or multi-centroids.
- Retrieval can rerank against medoids/subcentroids instead of only the mean.
- Adaptive split decisions can use intra-voxel variance, not sentence count alone.

## Phase 5: Guide-Star Spread and Stress Optimization

The guide-star risk is coverage, not just pairwise collapse. A catalog can avoid
near-duplicates while still occupying one dense "galaxy" of meaning. PSA needs
standard candles distributed across the broader semantic universe.

Required diagnostics:

- Pairwise guide-star cosine distribution.
- Nearest-neighbor distance distribution by tier.
- Contrast-pair separation distribution.
- MDS stress for guide-star 3D positions.
- Corpus coverage: fraction of corpus with max guide-star similarity below
  anchoring thresholds.
- Void map: regions covered by guide stars but not corpus occupancy.

Optimization objective:

- Preserve guide-star pair distances under 3D placement.
- Maximize coverage of corpus neighborhoods without collapsing contrast pairs.
- Add candidate guide stars in under-covered semantic regions.
- Penalize over-dense anchor neighborhoods.

Acceptance criteria:

- Guide-star nearest-neighbor similarities do not cluster near collapse.
- Corpus anchoring improves without reducing same-layer locality.
- Stress/triplet objective improves against the current hand-designed catalog.
