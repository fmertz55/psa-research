# Qwen + PSA Marketing Scorecard

**Date:** 2026-07-27  
**Model:** Qwen3-Embedding-8B  
**MTEB version:** 2.18.6  
**Status:** Verified partial scorecard. The native and PSA-only 12-task MTEB
measurements and fresh MRPC measurements completed from the clean
`nyu-mll/glue:mrpc` dataset. The combined Qwen+PSA MTEB aggregate did not
complete on the GitHub-hosted runner.

> **Protocol boundary:** These clean MRPC measurements evaluate native,
> PSA-only, and 80/20 fused representations with a trained linear probe. They
> do not evaluate the production membership route. The separate July 27
> production replay is withdrawn because it used corrupted local exports.

## What the dimensions mean

The 115-coordinate representation is not a truncated PSA representation. It is
the complete PSA representation: one coordinate for each of the 115 guide
stars.

| Representation | Coordinates | Purpose |
|---|---:|---|
| Qwen native | 4,096 | Embedding-model control |
| PSA | 115 | Complete interpretable semantic-field representation |
| Qwen + PSA | 4,211 logical coordinates | Full production candidate |

For MTEB, the combined system is evaluated without physically constructing a
4,211-column matrix. With unit-normalized Qwen vectors `q` and PSA vectors `p`,
the full vector is logically:

`[sqrt(0.80) * q, sqrt(0.20) * p]`

Its cosine similarity is exactly:

`0.80 * cosine(q1, q2) + 0.20 * cosine(p1, p2)`

The lazy calculation changes memory use, not the score.

## Fresh GLUE MRPC scorecard

These are held-out results from the repository's locked linear-probe protocol
using `nyu-mll/glue:mrpc`. They are not submissions to the official GLUE
leaderboard.

| Representation | F1 | Precision | Recall | Accuracy |
|---|---:|---:|---:|---:|
| Qwen native | 82.10 | 82.85 | 81.36 | 75.74 |
| PSA | **85.91** | 82.51 | **89.61** | **79.90** |
| Qwen + PSA | 84.12 | **84.73** | 83.51 | 78.43 |

Under this fresh protocol, PSA produced the strongest MRPC F1, recall, and
accuracy. The combined 80/20 representation produced the strongest precision.

## MTEB English v2 marketing slice

The locked marketing slice contains all 12 English v2 tasks in the Semantic
Textual Similarity and Pair Classification categories. It is a category slice,
not the full 41-task MTEB English v2 leaderboard suite.

| Representation | Mean(Task) | Mean(Type) | Pair Classification | STS |
|---|---:|---:|---:|---:|
| Qwen native | **88.50** | **88.33** | **87.99** | **88.67** |
| PSA | 72.55 | 73.85 | 76.45 | 71.26 |
| Qwen + PSA | Incomplete | Incomplete | Incomplete | Incomplete |

PSA used 2.81% as many coordinates as native Qwen and retained 81.98% of the
native aggregate score on this slice. It retained 86.88% of the native Pair
Classification score and 80.37% of the native STS score.

The combined run completed seven STS tasks before the GitHub-hosted process was
terminated with exit code 143 while evaluating `SprintDuplicateQuestions`.
Because the workflow did not reach result publication, no combined aggregate is
reported.

## Marketing-safe language

The following statements are supported by the completed runs:

> On the 12 Semantic Textual Similarity and Pair Classification tasks in MTEB
> English v2, our Qwen3-Embedding-8B control scored 88.50 Mean(Task).

> AIidea's complete 115-coordinate PSA semantic field retained 81.98% of the
> 4,096-coordinate Qwen control's aggregate score on the 12-task MTEB English v2
> semantic-similarity and pair-classification slice.

> In AIidea's fresh held-out GLUE MRPC protocol, the 115-coordinate PSA
> representation achieved 85.91 F1 and 89.61 recall.

Do not describe 88.50 as a full MTEB leaderboard score or claim a top overall
MTEB ranking from this slice. Do not publish a Qwen+PSA MTEB aggregate until all
12 tasks finish under the same locked protocol.

The previously discussed 96.1 result is not included here because it came from
a different measurement protocol. It should not be presented as the fresh MRPC
F1 measured in this scorecard.

## Run provenance

- Completed MRPC, Qwen-native MTEB, and PSA-only MTEB measurements:
  [GitHub Actions run 30286023772](https://github.com/aiidea-biz/aiidea/actions/runs/30286023772)
- First isolated hybrid attempt and verified cleanup:
  [GitHub Actions run 30289448752](https://github.com/aiidea-biz/aiidea/actions/runs/30289448752)
- Lazy exact-fusion hybrid attempt and verified cleanup:
  [GitHub Actions run 30290695889](https://github.com/aiidea-biz/aiidea/actions/runs/30290695889)

All Hugging Face endpoints created by these runs were paused by the cleanup
workflow.
