# MRPC PI2NLI Adaptation Preregistration — 2026-08-04

## Status and claim boundary

This document freezes one diagnostic experiment before adapted-model training or
another MRPC reveal. The 1,107-pair dashboard partition has been visible during
earlier repository work. It is therefore a post-hoc development dashboard, not
a blind test, deployment authorization, or evidence of external generalization.

The experiment has no service, database, membership, or satellite-edge consumer.
Every learned artifact must declare `hard_membership_authorized=false`,
`directional_edge_authorized=false`, and `membership_effect=none`.

## One controlling hypothesis

The existing 341-feature symmetric resolver is retained unchanged. The existing
two-feature fusion contract is also retained unchanged:

1. `compact_generic_features_v1` (341 values);
2. sorted minimum and maximum log entailment from both sentence directions;
3. standardized ridge classifier, `alpha=1000`, no class weighting;
4. the frozen component split, five component folds, PSA route, threshold rule,
   and paired bootstrap.

The only controlling change is the source of the two entailment values. They
come from a calibration-adapted clone of the exact pinned NLI cross-encoder
instead of the zero-shot clone. Atomic predicate-argument aggregates remain
diagnostic and do not enter the score.

Primary comparator: frozen compact resolver F1 `0.8564356435643564`.

Historical context only: the zero-shot 343-feature fusion scored
`0.8594229588704726`; it is not substituted as the comparator.

## Pinned model and adaptation recipe

- Base model: `cross-encoder/nli-deberta-v3-xsmall`
- Revision: `a150876415327c80daeff35ca6f68f5ed8cf5c24`
- Safetensors SHA-256:
  `4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9`
- The pretrained three-way NLI classifier head is preserved. It is not reset.
- The model card identifies SNLI and MultiNLI as training data and does not list
  MRPC or QQP. That is a provenance declaration, not proof that indirect
  contamination is impossible.
- Positive MRPC pair: two entailment rows, one in each direction.
- Negative MRPC pair: one direction and one non-entailment target
  (`neutral` or `contradiction`), both assigned by a pinned SHA-256 procedure.
- Optimizer: PyTorch AdamW, learning rate `1e-5`, weight decay `0.001`, epsilon
  `1e-8`, no scheduler, no mixed precision.
- Train batch `32`, evaluation batch `64`, maximum length `156`.
- Optimization seed `0`; deterministic mapping seed and runtime identities are
  persisted.
- Candidate epochs are exactly 1 through 12. There is no learning-rate, seed,
  feature, model, or mapping search.

This is a conservative, preserved-head variant of PI2NLI, not an exact
reproduction of the authors' released default that resets the classifier head.
The reduction itself follows [Shi et al. (2024)](https://aclanthology.org/2024.starsem-1.11/).

## Calibration and freeze protocol

Only the 618 calibration-pair labels may be loaded before reveal. Pair text and
split construction use the label-free loader.

For each component-disjoint outer fold and candidate epoch:

1. fine-tune the NLI clone on every pair outside the validation components;
2. score both directions for the outer training and validation pairs;
3. fit the fixed 343-feature ridge on routed outer-training pairs;
4. score routed outer-validation pairs;
5. aggregate the five OOF partitions and select a threshold using calibration
   labels only.

Epoch selection maximizes calibration OOF F1, then precision, then recall, with
the earlier epoch winning an exact tie. The final NLI clone is trained on all
618 calibration pairs for that epoch. The fixed ridge is then fit on routed
calibration features. Model weights, tokenizer/config identities, numeric model,
threshold, scores, routes, labels, and source/runtime identities are hashed and
frozen before held text is scored by the adapted model.

No outer validation component may enter its fold's NLI training set. The
calibration training manifest contains no held pair, text, or label. Held
inference uses a separate, label-free request manifest created only after the
final resolver is frozen.

## Reveal and promotion gates

The experiment directory is unique. Selection, resolver, reveal marker, and
report writes refuse replacement. The reveal marker is created before held
labels are loaded, so a failed reveal cannot be retried under this experiment
identity.

The full result is retained whether it wins or loses. Promotion on this
development dashboard requires both:

- F1 gain of at least `+0.01` over `0.8564356435643564`;
- paired-bootstrap 95% lower bound for the F1 difference strictly above zero.

Precision, recall, confusion counts, all 1,107 scores/predictions, and artifact
hashes must be reported. Passing these gates does not grant production or graph
write authority.

## Directional and edge firewall

The PI2NLI negative conversion deliberately assigns only one direction a
non-entailment target. That is useful for a mutual-paraphrase decision boundary,
but the chosen direction and `neutral`/`contradiction` target are not semantic
facts. The adapted clone therefore cannot propose or validate directional
support, contradiction, atomic support, or qualified membership edges.

The original pinned NLI lane remains the diagnostic source for one-way support
and atomic axis deficits. Current qualified edge axes remain `member_scale`,
`member_temporal`, and the existing combined `member_aspectual`; `member_of`
remains exclusive for strict paraphrase membership.

Deferred edge breadcrumbs, explicitly outside this score experiment:

- `member_spatial` is the strongest analogous next axis because proposition
  graphs already preserve material locations and location conflicts. It needs a
  new spatial atomic axis and cache/schema version before implementation.
- modality should eventually be separated from aspect rather than adding more
  behavior to the combined edge.
- quantificational/scope and attribution/evidential axes need explicit scope and
  truth-commitment semantics first.
- participant/referent/role changes protect event identity and are not ordinary
  satellites; polarity and event direction remain contradiction territory.
- causal and conditional material usually changes the claim and continues to
  fail closed pending a separate architecture decision.

Axis type and support direction remain independent evidence. A scale or temporal
edge names the differing detail; it does not itself prove which sentence
supports the other.
