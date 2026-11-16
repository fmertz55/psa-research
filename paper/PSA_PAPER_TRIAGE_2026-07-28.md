# PSA Paper Triage

**Date:** 2026-07-28  
**Decision:** The base paper is a research record; the response-ready industry
paper is the canonical EMNLP 2026 working source. Neither should claim
end-to-end paraphrase accuracy from PSA blocking recall.

**Submission disposition (August 3, 2026):** Not recorded. The owner must mark
this package `submitted`, `missed`, or `deferred`; until then, treat it as an
internal working package rather than a live submission record.

## Repairs Applied

- Restored the actual idea-inventory architecture: durable sentence nodes,
  free-range unmatched sentences, true multi-sentence paraphrase groups, a
  complete sentence-and-group candidate union, and downstream verification.
- Corrected the 115-star quantile occupancy from 2,738 to the artifact value
  2,739.
- Relabeled 96.1% and 96.9% as descriptive legacy-label **blocking recall**,
  not paraphrase F1.
- Separated the completed 1,000-trial random-projection artifact from the later
  1,000-trial random sentence-anchor artifact. The `latest` alias skips random
  projection and is no longer cited for that result.
- Removed numerical claims whose raw run artifact is absent: matched-distance
  controls, random-anchor occupancy summaries, frame-transfer AMI/kNN and
  co-membership values, and old paraphrase-consolidation counts.
- Made `latex/psa_semantic_field_industry_2026.tex` a compatibility entry point
  to the response-ready source so the two industry files cannot drift.

## EMNLP 2026 Industry Track Gate

The official call lists a six-page main-text limit, excluding limitations,
ethics, references, and appendices; a dedicated Limitations section is
mandatory. Review is double-blind. The track emphasizes real-world deployment,
evaluation quality, reproducibility, and useful negative results:

- <https://2026.emnlp.org/calls/industry_track/>

The call lists the author-response deadline as 2026-07-29, notification on
2026-08-20, and camera-ready on 2026-09-20. As of this audit, the response-ready
source should be treated as rebuttal/camera-ready working material. Whether a
revised PDF may be uploaded during author response depends on the active
submission interface and instructions; verify that before replacing a submitted
file.

## Remaining Publication Gates

1. Run the current end-to-end membership route on MRPC and an in-domain
   held-out set. Do not publish an application-level score until the corrected
   implementation has a current, immutable benchmark artifact.
2. Recompute semantic reference excess on each random sentence-anchor null's
   exact non-anchor pair subset.
3. Archive immutable input manifests, environment versions, commands, and
   hashes with each gate artifact.
4. The current response-ready PDF is four pages total and within the six-page
   main-text limit. Recheck page count after rebuttal edits and confirm that no
   identifying repository path, author metadata, or acknowledgments break
   anonymity.
5. Keep superseded pre-repair benchmark results out of the manuscript and
   author response.

## Source of Truth

- Base research paper: `latex/psa_semantic_field.tex`
- Industry working source:
  `latex/psa_semantic_field_industry_2026_response_ready.tex`
- Compatibility entry point:
  `latex/psa_semantic_field_industry_2026.tex`
- Claim ledger: `PSA_CLAIM_PROVENANCE_2026-07-28.md`
