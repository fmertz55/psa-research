# PSA First-Light Baseline Checklist

Status: required before Wikipedia reingestion  
Created: 2026-07-11

## Purpose

Freeze the current PSA instrument and observed substrate so post-Wikipedia
changes can be classified as support expansion, density change, increased
resolution, calibration drift, or ingestion artifact.

## Baseline Identity

- [ ] Assign an immutable baseline ID, recommended:
      `psa-first-light-pre-wikipedia-2026-07-11`.
- [ ] Record the Git commit and dirty-worktree manifest used to create it.
- [ ] Record UTC creation time and operator.
- [ ] Record environment, tenant, Neo4j database, and ingestion namespace.

## Instrument Freeze

- [ ] Export the complete guide-star catalog and content hash.
- [ ] Record guide count, tier weights, contrast pairs, and selector policy.
- [ ] Record embedding model name, exact revision, dimension, pooling, and
      normalization policy.
- [ ] Record PSA frame version and calibration ID.
- [ ] Export 3D visualization and 8D/16D operational chart definitions.
- [ ] Export 40-, 64-, 75-, and 115-star selected guide IDs.
- [ ] Store the guide and frame artifacts in a versioned immutable S3 prefix.

## Substrate Freeze

- [ ] Snapshot Neo4j before reingestion.
- [ ] Export sentence IDs, text hashes, source/container IDs, and embeddings.
- [ ] Export `ParaphraseGroup` IDs, representative text, membership edges, and
      centroid embeddings.
- [ ] Export exact occurrence attestations separately from canonical sentence
      nodes.
- [ ] Export source, citation, temporal, and current popularity components.
- [ ] Record counts by node label, relationship type, source, and frame version.
- [ ] Hash every export and write a manifest containing byte size and row count.

Suggested snapshot invocation when Neo4j is available:

```bash
python3 scripts/neo4j/snapshot_neo4j.py \
  --run-dir psa_first_light_pre_wikipedia \
  --label psa-first-light-pre-wikipedia-2026-07-11
```

## Measurement Freeze

- [ ] Preserve all `guide_star_frontier_*.json` research outputs.
- [ ] Record the 40/64/75/115 count frontier.
- [ ] Record boundary support and sliced-density metrics.
- [ ] Record local density at `k=5, 10, 25, 50`.
- [ ] Record projection sensitivity at 64, 128, 256, and 512 directions.
- [ ] Record native and cross-model routing at fixed candidate budgets.
- [ ] Record held-out guide errors, edge errors, cycle return, and strong nulls.
- [ ] Record paraphrase cohesion, family size, and source diversity distributions.

## Reingestion Contract

- [ ] Assign a unique Wikipedia ingestion run ID.
- [ ] Freeze the curated English Wikipedia article/revision manifest and its
      content hash before the production demo import.
- [ ] Limit sources to the main/article namespace; exclude all non-article
      namespaces and disambiguation-only pages.
- [ ] Assign each article one declared AI/implementation/enterprise topic band
      and report the accepted-sentence distribution by band.
- [ ] Target 100,000 accepted prose sentence occurrences, with a completion band
      of 100,000 through 105,000 so articles remain intact.
- [ ] Record raw extracted, rejected, accepted occurrence, unique normalized
      text, canonical sentence, and contextual object-occurrence vector counts.
- [ ] Freeze dump date, language, namespace, and revision policy.
- [ ] Preserve page, revision, section, redirect, and template provenance.
- [ ] Define exact normalization and text-hash policy before processing.
- [ ] Confirm exact deduplication retains occurrence attestations.
- [ ] Define boilerplate, quotation, navigation, reference, and list handling.
- [ ] Predeclare 10,000, 25,000, 50,000, at-least-100,000, and
      post-clustering checkpoints.
- [ ] Prevent baseline frame or calibration records from being overwritten.

## Acceptance

The baseline is complete only when:

- every artifact has a stable URI and hash;
- counts and hashes reproduce on a second export;
- canonical sentences can be joined to occurrences, sources, and paraphrase
  classes without ambiguity;
- every coordinate can be traced to model, frame, calibration, and guide catalog;
- the research harness reproduces the recorded benchmark within tolerance;
- rollback and comparison procedures have been tested on a non-production copy.

## Related Documents

- `PSA_WIKIPEDIA_SUBSTRATE_AND_POPULARITY_CALIBRATION_PLAN.md`
- `PSA_GUIDE_STAR_EDGE_DENSITY_RESULTS.md`
- `PSA_GUIDE_STAR_TOPOLOGY_IMPROVEMENT_PROGRAM.md`
