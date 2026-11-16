# Image-Caption Guide-Star Neighborhood Pilot

**Status:** Completed on 2026-08-17; L4 execution passed, and both the
cross-model and Qwen3-VL-only frames are **NO / NOT YET** for promotion

## Purpose

This is a bounded architecture-feasibility experiment. It asks whether a
caption embedded by `Qwen/Qwen3-Embedding-8B` and its image embedded by
`Qwen/Qwen3-VL-Embedding-8B` independently nominate the same stable guide-star
identities. Native text and image vectors are never compared, projected into
one another, averaged, or written to the graph.

The pilot uses a seeded contiguous window of 115 guide pairs from Flickr8k
`train` and a separate seeded contiguous window of 200 held-out pairs from
Flickr8k `test`, with deterministic shuffled-caption, lexical hard-negative,
shuffled-guide-identity, and random-guide-subset controls. The pinned dataset
revision and model revisions are in
`research/programs/IMAGE_CAPTION_GUIDE_STAR_PILOT_CONFIG.json`.

Local materialization was verified on 2026-08-17: 315 images (12 MB), guide
records `train:2713` through `train:2827`, and evaluation records `test:552`
through `test:751`. The generated manifest records every full record ID,
caption, source split, guide-star ID, positive association, shuffled control,
hard-negative source, and local image path.

## Reused MRPC behavior

The runner uses the repository's `GuideStarManager.compute_signature` for
same-space L2 normalization and cosine similarity and
`GuideStarManager.get_top_k_stars` for candidate construction. It sweeps k=4,
8, 16, and the production MRPC nomination limit of 32. Candidate recall uses
the MRPC either-direction analogue: either modality's rank-1 guide identity is
present in the other modality's top-k neighborhood.

## Result

The L4 configuration worked. Both hosted 8B models returned 4096-dimensional
embeddings, the 200-pair experiment completed, and both temporary endpoints
were explicitly paused. Hardware feasibility is therefore a **yes**. The
current guide-star construction is a semantic **no-go**: recall is far below
the approximately 99% MRPC target, and the negative controls behave almost
like the true pairs.

| k | True candidate recall | Shuffled evaluation | Hard negative | Mean intersection | Jaccard | Containment | Rank-aware overlap | Mean union of bank |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 11.5% | 6.0% | 11.5% | 0.25 | 0.037 | 0.062 | 0.045 | 6.7% |
| 8 | 20.0% | 13.5% | 19.5% | 0.84 | 0.059 | 0.106 | 0.068 | 13.2% |
| 16 | 34.0% | 24.5% | 30.5% | 2.83 | 0.100 | 0.177 | 0.106 | 25.4% |
| 32 | 57.0% | 52.5% | 53.5% | 9.62 | 0.179 | 0.301 | 0.174 | 47.3% |

At k=32, true-pair candidate recall is 57.0% (95% bootstrap CI 50.0% to
63.5%). Shuffled evaluation pairs reach 52.5% (46.0% to 59.0%) and lexical
hard negatives reach 53.5% (46.5% to 60.5%). The positive-minus-shuffled
separation is only +4.5 percentage points (95% CI -5.0 to +14.0), and the
positive-minus-hard-negative separation is +3.5 points (-3.5 to +10.5).

The fixed-k neighborhoods each contain 32 guide stars. Their mean intersection
is 9.625 stars; mean Jaccard is 0.179, containment is 0.301, rank-aware overlap
is 0.174, and their union covers 47.3% of the 115-star bank. Every positive,
shuffled, and hard-negative pair has at least one intersecting guide star at
k=32, so raw intersection existence is not discriminative.

Correct guide identities do carry a small signal. Against shuffled guide
identities at k=32, paired identities improve positive recall by 10.0 points
(95% CI 0.0 to 20.0), Jaccard by 0.0246 (0.0163 to 0.0331), and rank-aware
overlap by 0.0295 (0.0177 to 0.0417). That signal is not sufficient for the
candidate-recall gate. The random 64-star ablation raises true recall to 80.0%,
but it also retrieves a 74.1% mean union of the bank and reaches 78.0% for
shuffled evaluations and 80.0% for hard negatives. That is nonselective, not
an improvement.

## Same-model Qwen3-VL follow-up

The follow-up removes cross-model alignment as a confounder. It embeds all
guide captions, evaluation captions, and images with the same pinned
`Qwen/Qwen3-VL-Embedding-8B` model. It reuses all 315 cached image vectors and
adds 1,115 VL text vectors. The evaluation contains 1,000 image-caption
positive associations and 800 sibling-caption positive associations over the
same 200 held-out images. No production model or graph data changed.

Four guide frames were tested. `text_guides_only` compares both modalities to
the 115 text guide vectors. `paired_native_guides` compares images to image
guides and text to text guides, aligned by guide identity. The two fused frames
use mean or max scores across paired text and image guide representations.

| Guide frame at k=32 | Image-text positive | Text-text positive | Image-text shuffled | Image-text hard negative | Text-text shuffled | Text-text hard negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text guides only | 43.1% | 83.2% | 43.0% | 41.5% | 70.5% | 83.0% |
| Paired native guides | 51.5% | 83.2% | 53.5% | 43.5% | 70.5% | 83.0% |
| Fused mean guides | 49.1% | 84.2% | 48.0% | 46.0% | 70.5% | 90.0% |
| Fused max guides | 51.5% | 83.2% | 53.5% | 43.5% | 70.5% | 83.0% |

The 95% coverage target fails in every frame. Text guide stars alone are not a
viable multimodal routing frame. Adding guide-star images improves image-text
positive recall by 8.4 points, but it does not rescue coverage and is not
selective: the paired frame retrieves shuffled image-caption controls more
often than positives. At k=32, each neighborhood consumes 27.8% of the
115-star bank and the mean pairwise union is about 48%, which helps explain
the high control recall.

The text-only result also matters. Sibling captions reach only 83.2%, while
lexical hard negatives reach 83.0%. Those negatives can remain semantically
related because they are captions from the same corpus, but candidate recall
at this budget is plainly insufficient as relation evidence. PSA remains a
routing mechanism whose output requires a typed semantic verifier.

This test rejects a simple production-model swap and rejects "add one image
for every guide star" as the remedy.

## Architecture decision after the follow-up

The selected product path does not enroll whole images into paraphrase PSA and
does not pursue image guide stars first. It persists an immutable `ImageUnit`,
uses the generic multimodal `agent_llm` to extract grounded
`DecompositionSentence` metadata, and connects each decomposition to the image
through `describes`.

The decomposition sentences and their object tokens reuse the production
embedding, PSA, HNSW, ontology, FrameBase, and paraphrase-resolution path. Raw
sentence or image cosine is not the semantic acceptance rule. An accepted
result creates a direct `ImageUnit -[:imagery_of]-> ParaphraseGroup` edge that
references the supporting decomposition IDs. The decomposition is not a
`SentenceNode` and never becomes a group member.

An image remains free range when no decomposition matches an existing group or
an independently observed free-range sentence. When a genuine free-range
sentence matches, that real sentence and the real image may establish a new
semantic group through their native `member_of` and `imagery_of` relations.
Generated decomposition text alone cannot manufacture a group. This decision
is documented in
`docs/design/IMAGE_SEMANTIC_RESOLUTION_VIA_ONTOLOGY_ASSOCIATION_2026-08-17.md`;
it is not implemented by this benchmark work.

## Plain answers

1. **What percentage of true pairs entered the same candidate neighborhood?**
   57.0% at k=32; the full sweep is 11.5%, 20.0%, 34.0%, and 57.0%.
2. **How large were the neighborhoods?** At k=32, each native neighborhood has
   32 stars, the mean intersection is 9.625, and the mean union covers 47.3%
   of the bank.
3. **How often did controls match?** 52.5% for shuffled evaluation pairs and
   53.5% for hard negatives at k=32, with confidence intervals overlapping the
   true result.
4. **Did paired identities beat shuffled guide identities?** Yes, modestly:
   +10.0 recall points and +0.0246 Jaccard at k=32.
5. **Should we run a larger corpus?** **No / not yet.** Reuse the cached vectors
   to change guide-star composition or the cross-neighborhood decision statistic
   before paying for more embeddings.

## Hosted execution and retained artifacts

GitHub Actions run
[`32050657545`](https://github.com/aiidea-biz/aiidea/actions/runs/32050657545)
completed in under seven minutes using one AWS L4 for the image endpoint and
one AWS L4 for the text endpoint. Image startup took about four minutes; bulk
embedding and evaluation took about 74 seconds. Both endpoints allowed at most
one replica, used a 15-minute scale-to-zero guard, and were explicitly paused
by unconditional cleanup. The run never used g6e or local model execution.

The complete 156 MB artifact is retained locally under
`temp/image_caption_neighborhood/runs/32050657545/`. Its SQLite cache contains
752 reusable embeddings (315 image and 437 text) plus 343 raw hosted responses.
SQLite integrity, the 2,400-row per-pair JSONL, the 115/200 guide/evaluation
split, and split disjointness were verified after download. These ignored
artifacts are not committed because they contain image data and large vectors;
the workflow artifact is the remote recovery copy.

The same-model follow-up ran in GitHub Actions as
[`32062728632`](https://github.com/aiidea-biz/aiidea/actions/runs/32062728632).
It completed 70 hosted requests for the missing text vectors, reused the prior
image cache, and explicitly paused the temporary L4 endpoint during cleanup.
Its downloaded artifact is retained locally under
`temp/qwen3_vl_same_model_psa/runs/32062728632/` and remains ignored.

## Run

```bash
.venv/bin/python -m pip install -e '.[image-caption-pilot]'
export HF_TOKEN=...
export HF_QWEN3_VL_EMBEDDING_ENDPOINT=https://...endpoints.huggingface.cloud/...
.venv/bin/python scripts/research/psa/run_image_caption_neighborhood_experiment.py prepare
.venv/bin/python scripts/research/psa/run_image_caption_neighborhood_experiment.py preflight
.venv/bin/python scripts/research/psa/run_image_caption_neighborhood_experiment.py run
```

The completed run writes resumable embeddings to SQLite, detailed per-pair
JSONL, an aggregate JSON summary with paired percentile-bootstrap intervals,
and a generated Markdown report under `temp/image_caption_neighborhood/`.
