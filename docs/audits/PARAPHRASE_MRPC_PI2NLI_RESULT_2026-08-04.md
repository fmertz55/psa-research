# MRPC PI2NLI Adaptation Result — 2026-08-04

## Outcome

The calibration-adapted cross-encoder fusion reached **F1
0.8668769716088329** on the 1,107-pair MRPC development dashboard.

| Measure | Frozen compact resolver | Adapted fusion | Difference |
|---|---:|---:|---:|
| Precision | 0.790857143 | 0.813981043 | +0.023123900 |
| Recall | 0.933873144 | 0.927125506 | -0.006747638 |
| F1 | 0.856435644 | 0.866876972 | **+0.010441328** |
| False positives | 183 | 157 | -26 |
| False negatives | 49 | 54 | +5 |

The 20,000-replicate paired-bootstrap 95% interval for the F1 difference is
**[-0.002136755, +0.022925780]**. The result passes the preregistered material
point-gain bar (`+0.01`) and fails the preregistered confidence bar (lower
bound strictly above zero). It is retained as the strongest point estimate,
but it does **not** replace the controlling `0.856435644` artifact.

This remains a post-hoc MRPC development diagnostic. It grants no deployment,
membership, database, service, directional-relation, atomic-support, or edge-
write authority.

## Calibration result

The frozen 341-feature calibration control replayed exactly at OOF F1
`0.861952862`. The adapted 343-feature candidate reached OOF F1
`0.883352208`, a gain of `+0.021399346`, with precision `0.817610063` and
recall `0.960591133`.

Epoch 11 won the preregistered five-fold component-disjoint selection. Epoch
12 was close (`0.883087401`) but lost under the frozen F1-first rule. The final
model was therefore trained on all 618 calibration pairs for exactly 11
epochs. No held label influenced the model, epoch, ridge fit, or threshold.

The point improvement mainly exchanged 26 fewer false positives for 5 more
false negatives. That is the intended movement away from broad semantic
similarity toward mutual paraphrase equivalence.

A secondary paired comparison isolates the adaptation step from the inherited
two-feature addition. Against the frozen zero-shot 343-feature fusion at F1
`0.859422959`, the adapted fusion gains `+0.007454013`; its 20,000-replicate
95% interval is `[-0.004138076, +0.018645389]`. Adaptation fixes 54 pairs that
the zero-shot fusion misses and regresses 36. This interval also crosses zero,
so the `+0.010441328` headline must be read as the complete candidate's gain
over the controlling 341-feature resolver, not as adaptation-only lift.

## What changed—and what did not

The split, PSA route, 341 compact features, two sorted log-entailment feature
positions, standardized ridge, `alpha=1000`, no class weighting, fold policy,
threshold rule, and bootstrap were fixed.

The only controlling signal change was the source of the two entailment
values. A clone of `cross-encoder/nli-deberta-v3-xsmall` revision
`a150876415327c80daeff35ca6f68f5ed8cf5c24` was fine-tuned on deterministic
PI2NLI rows from calibration components. Its original three-way NLI head was
preserved. The mapping follows the mutual-inference reduction in [Shi et al.
(2024)](https://aclanthology.org/2024.starsem-1.11/) but deliberately does not
reset the classifier head as the authors' released default does.

The pinned model card declares SNLI and MultiNLI training data and does not
list MRPC or QQP. That is authenticated provenance evidence, not proof against
all indirect contamination.

## Artifact chain

- Training-manifest internal SHA-256:
  `028fb4797cb6b9635686977f7a03b0eba650df07a7014689bafc07873be6fa5e`
- Crossfit-cache internal SHA-256:
  `3b26b74a26e66fbce8fbdccbf02c939cf1e39d26449dcb69b309ec5f0b8c70ed`
- Frozen-selection SHA-256:
  `c66bedd74c02204c50c2d2462b1d8ad068b7d4642627582d3319c282724f1412`
- Final-fit-cache internal SHA-256:
  `4c1e4039e18e634d789120e490c386bdee452f3ff93ef57aca79e57257d143ed`
- Adapted checkpoint artifact SHA-256:
  `b9ebdbf5837e6e8782c70c796053e45f58325a980daa1b59b10f9bbabcf9072a`
- Adapted safetensors file SHA-256:
  `e8f278e9b38f00ec76343053b6b650acb476b20e6f090e2d834ed25a4d005029`
- Frozen numeric-model SHA-256:
  `e883dfadc6a1fdf074bcd199bbfc9dd19e1ea1a92363c426f7d06f1995107bc9`
- Frozen resolver SHA-256:
  `ae05f24155682f74acbc2dd16fa570ccda31722de63e3055d91df618a8f257b0`
- Label-free held-manifest internal SHA-256:
  `49e687c84a6e120ce4c64982c6d2b8a77966f6860872b9a002a92f7a28ecf6c5`
- Held-inference-cache internal SHA-256:
  `25b868a80c9973d8197948dcd12297543e573d324ee775013ffac49b622c46ff`
- One-way reveal-marker file SHA-256:
  `18ea8ec68750c44b4a2589640e0a0b10b8cf852dde27adbc7992956712d3f4bf`
- Result-report file SHA-256:
  `2db98f7f2907043f20bedcc30ee88b46b059bdfb8a61852bcc1607fb5d916dab`

The frozen comparator replay matched all 1,107 routes, compact features,
margins, and predictions exactly. The result report retains all 1,107 adapted
scores and predictions.

Transformers warned about rewritten tokenizer-regex metadata when the saved
checkpoint was loaded. A complete label-free parity audit compared both
directions for all 618 calibration and 1,107 held pairs: **3,450 encodings,
zero token-ID mismatches**. The base token-ID stream SHA-256 was
`405f8b210ea2d17c2c98d50e5ba579b4044e3be10b2cf599f47205df0dba51f6`.

The first MPS attempt was stopped before producing a crossfit artifact because
the first backward pass reached an 8.5 GB footprint and did not complete in a
bounded interval. The authenticated experiment ran on CPU with the statistical
and optimization protocol unchanged; execution device is recorded in every
cache.

The checkpoint, caches, reveal marker, and JSON report live under ignored
`temp/` and `reports/` paths. Their hashes authenticate the copies that exist
now, but the tracked audit does not itself make those artifacts durable. They
must be archived before any workspace cleanup. The source identity also pins
the selected direct scripts and inputs, not every transitive repository import;
the worktree revision and Graphify snapshot remain part of the reproducibility
record.

## Architecture boundary and edge breadcrumbs

The adapted model is not a directional proof source. PI2NLI assigns a
deterministic but arbitrary direction and non-entailment class to every
negative pair. Those labels improve a symmetric mutual-paraphrase boundary but
cannot license support, contradiction, atomic facts, `member_scale`,
`member_temporal`, or any future edge. The original pinned NLI/atomic lane
remains the diagnostic source for directional evidence.

The next generic edge axis is `member_spatial`, not another score rule. Current
graphs already preserve material location gaps and conflicts, but the atomic
schema still folds location into `ROLE`; a future spatial edge therefore needs
a new semantic axis and cache/schema identity. Modality should later split
from aspect. Scope, attribution, causal, and conditional axes remain deferred
until their truth-commitment semantics are explicit. Referent/role changes and
polarity remain identity/contradiction protections, not satellites.

## Next score breadcrumb

Do not retune this experiment after reveal. The bounded next hypothesis is a
new-data adaptation experiment: authenticate the canonical MSRP training split
or another licensed paraphrase-training source, preserve this exact
architecture, and evaluate a frozen model on MRPC plus PARAPHRASUS/OOD suites.
Shi et al. trained on 3,668 MSRP pairs, versus 618 calibration pairs here, and
reported MSRP F1 above 0.92. Data scale is therefore the cleanest remaining
single variable before changing model size, head policy, atomic features, or
edge semantics.
