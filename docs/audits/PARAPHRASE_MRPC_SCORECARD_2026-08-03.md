# Paraphrase MRPC Scorecard

**Date:** 2026-08-03  
**Status:** Controlling development dashboard  
**Evaluation class:** Post-hoc iterative-development diagnostic; not blind and
not a deployment claim

## Headline

The frozen compact semantic-relation resolver reaches **F1 0.8564** on the
1,107-pair held-out partition. The authenticated historical gate-only
classifier is F1 0.8340, so the comparable gain is **+0.0224 F1**. The
controlling replay's paired 20,000-replicate bootstrap gives a 95% interval of
**[+0.0087, +0.0362]** for the difference.

This clears the predeclared diagnostic dashboard bar: at least `+0.01` F1 and
a paired 95% lower bound above zero. It is the first material, uncertainty-
supported climb in this work. It does not authorize paraphrase membership.

The latest directional-NLI fusion reaches **F1 0.8594**, a further **+0.0030**,
but its paired 95% interval is **[-0.0035, +0.0097]**. It fails both the
material-gain and confidence gates, so the controlling dashboard result
remains **0.8564**. The candidate is retained as diagnostic evidence rather
than promoted.

The calibration-adapted PI2NLI fusion now reaches **F1 0.8669**, a **+0.0104**
point gain over the frozen compact resolver. Precision rises to **0.8140** and
false positives fall from 183 to 157. Its paired 95% interval is
**[-0.0021, +0.0229]**: it passes the material point-gain gate and misses the
confidence gate. It is the strongest point estimate, but the controlling
artifact remains **0.8564** under the frozen promotion policy.

The transparent grammatical-operator ledger veto reaches **F1 0.8534**, a
**-0.0030** change from the frozen compact resolver. Precision rises to
**0.7935**, but recall falls to **0.9231**; it removes 5 false positives and
loses 8 true positives. Its paired 95% interval is
**[-0.0078, +0.0014]**. The veto is rejected and the controlling artifact
remains **0.8564**. See the
[operator-ledger result](PARAPHRASE_MRPC_OPERATOR_LEDGER_RESULT_2026-08-04.md).

The occurrence-bound FrameBase hierarchy experiment is also rejected. Its v1
held replay fell to F1 **0.8538**, and the required changed-prediction audit
then found that eight of nine vetoes came from an invalid Cartesian comparison
inside identical plural frame sets. Strict v2 removes all raw candidate paths
from scorer-visible evidence and produces a zero-change calibration result.
It was stopped without a second held reveal. The controlling artifact remains
**0.8564**. See the
[FrameBase hierarchy result](PARAPHRASE_MRPC_FRAMEBASE_HIERARCHY_RESULT_2026-08-04.md).

The result is deliberately described as an iterative-development score. The
winner and threshold were selected using calibration labels before the single
held-out reveal, but MRPC held-out errors had been inspected during earlier
repository work. A fresh external dataset is required for a blind claim.

## Dashboard

| Lane | Precision | Recall | F1 | TP / FP / FN | Interpretation |
|---|---:|---:|---:|---:|---|
| Corrected symmetric PSA top-32 route | n/a | **1.0000 route recall** | n/a | 741 routed positives / 0 missed | Candidate ceiling; no equivalence claim |
| Always positive | 0.6694 | 1.0000 | 0.8019 | 741 / 366 / 0 | Positive-heavy corpus floor |
| Corrected partition-local cosine gate | 0.7699 | 0.9028 | 0.8311 | 669 / 200 / 72 | Same corrected routing topology |
| Historical gate-only classifier | 0.7188 | 0.9933 | 0.8340 | 736 / 288 / 5 | Authenticated historical dashboard reference |
| Original hardened symmetric ridge | 0.7641 | 0.9136 | 0.8322 | 677 / 209 / 64 | Corrected replay; failed its material and uncertainty gates |
| **Frozen compact generic ridge** | **0.7909** | **0.9339** | **0.8564** | **692 / 183 / 49** | Material diagnostic gain; dashboard gate passed |
| Transparent operator-ledger veto | 0.7935 | 0.9231 | 0.8534 | 684 / 178 / 57 | Explainable one-way veto; failed both promotion gates |
| Standalone zero-shot mutual NLI | 0.7263 | 0.9703 | 0.8307 | 719 / 271 / 22 | Replacement failed; materially worse than compact resolver |
| Compact ridge + 2 directional-NLI values | 0.7883 | 0.9447 | 0.8594 | 700 / 188 / 41 | Small post-hoc gain; promotion gates failed |
| Compact ridge + PI2NLI-adapted values | 0.8140 | 0.9271 | **0.8669** | 687 / 157 / 54 | Material point gain; confidence gate failed |
| Frozen v2 legacy verifier | 0.9896 | 0.2564 | 0.4073 | 190 / 2 / 551 | Different hard-verification lane; historical context only |
| Current v3 legacy-policy replay | 0.9851 | 0.1781 | 0.3017 | 132 / 2 / 609 | New graph projection; not an exact v2 reproduction |
| Current strict proof kernel | n/a | 0.0000 | 0.0000 | 0 / 0 / 741 | Correctly abstains on incomplete v2 logical-operator evidence |

The `0.4073` verifier result is not the baseline for the learned resolver. The
honest score movement is `0.8340 -> 0.8564`, not `0.4073 -> 0.8564`.

## Directional experiment

The first directional experiment pinned
`cross-encoder/nli-deberta-v3-xsmall` at revision
`a150876415327c80daeff35ca6f68f5ed8cf5c24` and cached all required
whole-sentence and atomic predicate-argument requests before scoring. A pure
mutual-entailment replacement fell to F1 `0.830733680`; its paired difference
from the frozen compact resolver was `-0.025701964`, with 95% interval
`[-0.041235960, -0.009901557]`.

The smallest calibration-supported fusion kept the route, component split,
341 compact features, ridge family, `alpha=1000`, and class weighting fixed.
It added only the sorted minimum and maximum log entailment probabilities from
the two directions. Calibration out-of-fold F1 moved from `0.861952862` to
`0.868596882`. Held-out F1 moved from `0.856435644` to `0.859422959`, but the
gain was neither material nor uncertainty-supported.

All 1,107 compact feature rows, routes, margins, and predictions were
recomputed before reveal and matched the authenticated frozen traces exactly.
The 20 whole-sentence and atom-level diagnostics were authenticated and
reported, but calibration did not justify putting them in the controlling
score. No NLI output has membership or database-write authority.

### Calibration-adapted cross-encoder

The next experiment kept that exact 341+2 feature topology and replaced only
the two zero-shot entailment values with values from a calibration-adapted
clone of the same pinned cross-encoder. Positive calibration pairs produced
entailment rows in both directions; negative pairs produced one deterministic
direction and one deterministic `neutral` or `contradiction` target. The
pretrained three-way NLI head was preserved.

Five sentence-component folds trained and evaluated epochs 1 through 12. The
341-feature calibration control replayed at OOF F1 `0.861952862`; the adapted
fusion reached `0.883352208`, a gain of `+0.021399346`. Epoch 11 was frozen,
the model was fitted on all 618 calibration pairs, and the held text was scored
in a separate label-free phase before the one-way reveal marker was written.

Held F1 reached `0.866876972`, with 687 TP, 157 FP, and 54 FN. The point gain
over the frozen compact resolver was `+0.010441328`; the 20,000-replicate
paired interval was `[-0.002136755, +0.022925780]`. The material gate passed;
the confidence and promotion gates failed. See the [PI2NLI result audit](PARAPHRASE_MRPC_PI2NLI_RESULT_2026-08-04.md).

The adaptation-only comparison is smaller: against the frozen zero-shot
343-feature fusion at `0.859422959`, the gain is `+0.007454013`, with paired
interval `[-0.004138076, +0.018645389]`. The headline `+0.010441328` is the
complete candidate's gain over the controlling 341-feature resolver; it must
not be attributed solely to fine-tuning.

The adapted clone explicitly has no directional-relation or edge authority.
Its arbitrary negative-direction targets are appropriate for a symmetric
paraphrase boundary, not for proving which sentence supports the other. The
original pinned NLI/atomic lane remains the directional diagnostic.

## What the winning model learned from

The winning resolver is a standardized ridge classifier over a fixed,
swap-invariant 341-value representation:

- 21 distribution summaries of the absolute difference and Hadamard product
  of the Qwen sentence embeddings;
- 230 symmetric values from the 115-dimensional PSA signatures;
- 28 generic word- and character-ngram overlap and length statistics; and
- 62 generic proposition-graph multiset overlap and count statistics.

There are no lemma-pair licenses, phrase lists, news templates, MRPC example
IDs, or semantic exception branches. The surface features are fixed overlap
statistics, not hand-written paraphrase outcomes. The label-dependent behavior
is learned from calibration data.

This is therefore not a Siri-style rule cascade. It is still only a symmetric
pair classifier, however. It does not yet implement the target architecture's
forward and reverse contextual entailment checks.

## Selection and reveal protocol

- Calibration contains 618 pairs grouped by normalized sentence-identity
  components; 603 pairs reached the partition-local symmetric router.
- A declared grid of 246 generic candidates was evaluated with five
  component-disjoint calibration folds.
- The frozen winner is `compact_generic_v1` ridge with `alpha=1000`, no class
  weighting, and binary margin `-0.12455821638457187`.
- Calibration out-of-fold performance was F1 `0.861952862`.
- The winner, threshold, selective thresholds, fitted numeric artifact, and
  calibration contract were frozen before held-out labels were loaded.
- The held-out labels were revealed once. The reveal script refuses a second
  reveal in the same artifact directory.
- The fitted artifact replays all 1,107 recorded scores exactly.

The winning family was not a lucky choice among qualitatively different
outliers: calibration family winners ranged from F1 `0.8499` to `0.8620`, with
linear, logistic, kernel, tree, and boosting models all evaluated under the
same split.

## Selective output remains rejected

The binary dashboard improved, but the abstaining three-way surface did not
meet its calibration promise on held-out data:

| Measure | Held-out result |
|---|---:|
| Resolved pairs | 333 / 1,107 |
| Coverage of all pairs | 0.3008 |
| Coverage of routed pairs | 0.3127 |
| Likely-equivalent precision | 0.9294 (326 selected) |
| Likely-not-equivalent NPV | 0.8571 (7 selected) |
| Selective risk | 0.0721 |

The calibration target was 0.95. Neither held-out selective tail is strong
enough to use as a semantic adjudicator. Selective outputs retain
`hard_membership_authorized=false`.

## Frozen identity

- Frozen selection file SHA-256:
  `fa92b37f978f3fb955cf097c984ac386ff28a8439b3a9d10bf10d076a91a8d3f`.
- Internal frozen-selection SHA-256:
  `6906e04234640628e4f42a3b227271e856a032bd4038776d2fff2e067f42908e`.
- Calibration contract SHA-256:
  `58ab913adfcaf48dc06c0b0a911b97706ad9d9af109feebb80e0ed1d830f55cf`.
- Numeric fitted-artifact SHA-256:
  `8cffff04158091455029b70af9c59adfaebcd11b9cd0c13aafb5c818e8844350`.
- Integrated resolver artifact SHA-256:
  `0acb7045590934f630e33e4318f717e83e9ee9e87c2a07f0e607409203f83b35`.
- Integrated resolver report SHA-256:
  `b8c0f228873a1823899028683b7b0a78af5a84f4f34271c274dec0922f1ccd95`.
- Single-reveal report SHA-256:
  `a5b228d25affc998a0df71471d115369312c958b6534096ebe637fb57ff0d79a`.
- Dashboard-gate SHA-256:
  `62f114227b747a8be7131c225bf39cc54a614c397c95292d81ad6d3c1fc05436`.
- Authenticated historical comparator SHA-256:
  `c40eea84bce3eea527198ef1e31de69c2eb7285b3be55f2c6616e35c747ebffe`.
- Directional-NLI fusion selection SHA-256:
  `4e79440325346a23dea87adcee6138412d2a14b585e09d7abc1cb188dc1f45bb`.
- Directional-NLI fusion numeric-model SHA-256:
  `8c2adf14a89534149162396f5290e82bc09ca1abc59f913ee9764779b2ae7b50`.
- Directional-NLI fusion report SHA-256:
  `d9590ec68a1f3baa6a633c2262c8b1f561b3402dee986dd4a6f3a56f12799307`.
- Standalone directional-NLI report SHA-256:
  `4ac36b3955286782cd6e08423bd7ab2c64344050d16ea2ab676bfd3bbe7ef22e`.
- PI2NLI training-manifest internal SHA-256:
  `028fb4797cb6b9635686977f7a03b0eba650df07a7014689bafc07873be6fa5e`.
- PI2NLI frozen-selection SHA-256:
  `c66bedd74c02204c50c2d2462b1d8ad068b7d4642627582d3319c282724f1412`.
- PI2NLI adapted-checkpoint artifact SHA-256:
  `b9ebdbf5837e6e8782c70c796053e45f58325a980daa1b59b10f9bbabcf9072a`.
- PI2NLI frozen-resolver SHA-256:
  `ae05f24155682f74acbc2dd16fa570ccda31722de63e3055d91df618a8f257b0`.
- PI2NLI result-report SHA-256:
  `2db98f7f2907043f20bedcc30ee88b46b059bdfb8a61852bcc1607fb5d916dab`.
- Operator-ledger selection file SHA-256:
  `4f1658bbf4f0d6b0668d03382d0131507fba491e55218e8a2bded18b42a043cd`.
- Operator-ledger result-report SHA-256:
  `df336979d59219c5217b504d4eaf95cd938620052f81795ed48cdadd84632de3`.
- FrameBase pinned hierarchy artifact SHA-256:
  `cac286a636fe6f43292a5f856d5ac49c2523fac1ecd0cb17805173d419d735c2`.
- Rejected FrameBase v1 result-report SHA-256:
  `d55d018be057a28ccb67ed0060d339e999afff1776f5932a32cebb1dcbcfe201`.
- FrameBase strict-v2 calibration selection file SHA-256:
  `daa9eb1856fa3e97108c01526937bba068b2c0109f30164c6015c0ebbb320dac`.

Reports:

- `reports/mrpc_semantic_relation_model_search_2026-08-03/held_out_reveal.json`
- `reports/mrpc_semantic_relation_model_search_2026-08-03/dashboard_gate.json`
- `reports/mrpc_semantic_relation_resolver_2026-08-03/semantic_relation_resolver_report.json`
- `reports/mrpc_directional_nli_2026-08-03/directional_nli_report.json`
- `reports/mrpc_directional_nli_fusion_2026-08-03/fusion_report.json`
- `reports/mrpc_pi2nli_adaptation_2026-08-04/pi2nli_adaptation_report.json`
- `reports/mrpc_operator_ledger_veto_2026-08-04/operator_ledger_veto_report.json`
- `reports/mrpc_framebase_hierarchy_2026-08-04/framebase_hierarchy_report.json`
- `reports/mrpc_rule_policy_dashboard/proof_kernel_v1/mrpc_proposition_graph_report.json`
- `reports/mrpc_rule_policy_dashboard/legacy_etpc_mrpc_diagnostic_v1/mrpc_proposition_graph_report.json`

## What moves the score next

1. Do not retune the revealed PI2NLI experiment. Authenticate the canonical
   MSRP training split or another licensed paraphrase-training source, keep
   this architecture fixed, and treat training-data scale as the next single
   score variable. The published PI2NLI result used 3,668 training pairs,
   versus 618 calibration pairs here.
2. Freeze that lane before evaluating MRPC together with PARAPHRASUS and
   out-of-domain inference or duplicate-detection data. A further MRPC-only
   point gain is not an external-generalization claim.
3. Keep atomic predicate-argument support as learned evidence and align its
   decomposition against verifier performance. Do not add sentence-shape or
   lexical rewrite rules.
4. Replace point-calibrated selective thresholds with a risk-control protocol
   and report the full risk-coverage curve.
5. Logical-operator extraction is now complete and fail-closed. Before it can
   control a score, align occurrence-bound propositions semantically instead
   of comparing exact predicate/argument hashes. Any later additive scorer
   should preregister monotone signs so supported evidence cannot hurt and
   unresolved evidence cannot veto.

The pinned lexical snapshot remains useful as occurrence-bound analyzer or
alignment evidence. It is not the immediate MRPC bottleneck, and it must not
become lemma-wide rewrite permission.

Relevant literature: [Paraphrase Identification via Textual Inference](https://aclanthology.org/2024.starsem-1.11/),
[QASemConsistency](https://aclanthology.org/2026.tacl-1.6/),
[Distill and Align Decomposition](https://aclanthology.org/2026.findings-eacl.309/),
and the [OOD NLI study](https://aclanthology.org/2026.findings-eacl.286/).

## Deferred breadcrumbs

- `scripts/research/mrpc/analyze_membership_edge_geometry.py` still consumes the legacy
  singular edge field and will display plural v4 combinations as `(none)`.
- The NLI request manifest is content- and label-independent, but its original
  preparation path materialized the full MRPC rows before discarding labels.
  A later hygiene pass should make procedural label isolation explicit.
- The atomic axis diagnostics are saturated on parts of MRPC and are not ready
  to control the score. Decomposition/alignment evaluation is a separate
  experiment, not a reason to expand the current feature block.
- `member_spatial` is the strongest next generic edge axis because material
  location gaps already reach proposition graphs. It requires a new atomic
  semantic axis and cache/schema identity; location is currently folded into
  `ROLE`.
- Modality should eventually split from aspect. Scope and attribution need
  explicit truth-commitment semantics first. Referent/role changes and
  polarity remain identity/contradiction protections, not qualified
  satellites.
