# Pixelization of Semantic Anisotropy: Addressable Semantic Geometry via Guide-Star Projection

## Abstract

We introduce **Pixelization of Semantic Anisotropy (PSA)**, a method for constructing a stable, falsifiable coordinate framework over high-dimensional embedding space. PSA projects arbitrary text by computing cosine similarities to a fixed set of semantically anchored reference concepts ("guide stars"), organized as contrastive pairs across ontological tiers. The resulting 3D coordinates define a discretized voxel grid that functions as an addressable semantic projection.

We evaluate PSA first with a 75-star frame on a corpus of 3,378 sentences drawn from eight source documents plus an ontology seed, ingested under two distinct orderings: sequential (document size-first) and randomized. Across both trials, we demonstrate: (1) statistically significant locality decay - embedding cosine similarity decreases monotonically with voxel distance under the raw Manhattan voxel metric (empirical $p < 10^{-4}$, $r = -0.18$); (2) identical discourse topology — 638 PRECEDES edges preserved exactly across ingestion orders; (3) order-invariant coordinate assignment — intercepts stable to three decimal places (0.720 vs. 0.720); and (4) falsification under permutation control — zero of 1,000 scramble permutations produce a slope as negative as the observed value.

We then evaluate an expanded 115-star frame on the live PSA field under a canonical *normalized* voxel-distance metric. Re-running the same 75-star result under this normalized metric gives $r = -0.2875$, slope $= -0.0748$ (the raw-Manhattan and normalized slopes are not directly comparable; see §4.5); the 115-star frame strengthens both to $r = -0.3118$, slope $= -0.1148$, with permutation $p = 0.001$ and same-layer locality as strong as $r=-0.7833$ at L2. Paraphrase consolidation rises in parallel: paraphrase mass per assigned sentence increases from 15.96% to 22.44% while the number of paraphrase groups drops from 506 to 421, indicating that the expanded field reduces semantic fragmentation rather than merely re-binning. Scramble falsification on the expanded frame again returns zero matching permutations out of 1,000.

Full-dimensional controls strengthen the result: computing voxel centroids over all 4,096 embedding dimensions yields $r=-0.3259$ and slope $=-0.1275$ on the expanded field. Random-anchor controls reveal a further constraint: random sentence anchors can produce an even steeper apparent slope ($-0.1864$ median), but only by collapsing the field to 8.54 occupied voxels on average, versus 59 for the PSA frame. These results suggest that PSA coordinates capture semantic structure not by maximizing locality alone, but by preserving locality together with occupancy diversity and reduced concentration. The framework is reproducible, falsifiable, and order-invariant within a fixed guide-star frame. We release the guide-star catalog, addressing algorithm, and evaluation framework.

---

## 1 Introduction

Embedding models produce powerful representations of text, but the spaces they inhabit are opaque. A 4,096-dimensional vector captures semantic content with high fidelity, yet the space has no addresses, no neighborhoods, and no landmarks. Two vectors can be compared (cosine similarity), but neither has a *location* that is meaningful independent of the other.

This opacity creates practical problems. Clustering algorithms (K-means, HDBSCAN) assign group labels but not coordinates. Dimensionality reduction techniques (PCA, t-SNE, UMAP) produce visualizations but not stable positions; re-running the algorithm yields different layouts. Existing methods do not provide a *reusable coordinate framework* that is stable across documents, ingestion orders, and time.

The immediate operational motivation for PSA was paraphrase detection at scale during construction of a context knowledge graph. In such a pipeline, sentences or propositions extracted from heterogeneous sources must be compared to determine whether they express the same underlying claim, event, or concept. Naive all-pairs paraphrase comparison scales quadratically and becomes impractical under web-scale ingestion. PSA was developed as a routing layer for this setting: assign each fragment a stable semantic address so expensive paraphrase comparisons can be restricted to nearby voxels, adjacent regions, and high-mass candidate attractors. More generally, deployed language systems need representations that support latency and cost budgets, changing user behavior, incomplete labels, maintenance across corpus updates, interpretable control surfaces, and incremental evaluation when no clean task-specific test set is available.

We propose that a small set of fixed reference concepts — which we call **guide stars**, by analogy with astronomical reference frames — can impose coordinate geometry on any embedding space. The key insight is simple: if you measure every text fragment's similarity to the same fixed set of semantic anchors, then similar texts produce similar measurements, regardless of when or in what order they were processed. The anchors define a reference frame. The measurements define coordinates.

The resulting representation is not merely a dimensionality reduction. It can be evaluated operationally as a **field**: a spatial domain with measurable local properties. Nearby regions share statistical characteristics, the observed structure survives falsification controls, and the coordinates are stable within a fixed guide-star frame.

### 1.1 Contributions

1. A **guide-star catalog** of semantically anchored reference concepts, evaluated first as a 75-star seven-tier frame and then as an expanded 115-star twelve-tier frame (Section 3.1).
2. The **PSA coordinate algorithm**: projection from high-dimensional embeddings to a $G$-dimensional guide-star signature, then to 3D voxel addresses via MDS over the guide-star geometry (Section 3.2).
3. A **hierarchical voxelization scheme** with addressable macro-cells (`L1_v_X_Y_Z`) and recursive micro-cell subdivision (Section 3.3).
4. **Empirical demonstration of semantic locality**: embedding similarity decays with voxel distance at high significance (Section 4).
5. **Full-dimensional centroid validation**: full-4,096D voxel centroids strengthen expanded-frame locality to $r=-0.3259$, slope $=-0.1275$ (Section 4.6).
6. **Random-anchor collapse control**: random anchors can produce apparent locality, but with severe occupancy collapse; PSA preserves locality together with a richer address space (Section 4.6).
7. **Paraphrase consolidation under denser anchoring**: paraphrase mass per assigned sentence rises from 15.96% (75-star) to 22.44% (115-star) while the number of distinct paraphrase groups drops from 506 to 421, simultaneously with locality strengthening — a signature of semantic-fragmentation reduction rather than coarser binning (Section 4.7).
8. **Order-invariance within a fixed frame**: identical PRECEDES topology and stable intercepts under sequential vs. randomized ingestion (Section 5).
9. **Falsification via scramble control**: zero of 1,000 permutations match the observed slope in either frame; empirical $p < 0.001$ (Section 6).
10. The **PSA evaluation framework** for reproducible field testing (Section 7).

### 1.2 Naming Conventions

We use the following terminology throughout:

| Term | Definition |
|------|-----------|
| **PSA Semantic Field** | The representational layer: an addressable coordinate framework over embedding space with measurable local properties |
| **Guide stars** | Fixed reference concepts that define the coordinate frame |
| **Voxel** | A discrete cell in the PSA grid; the unit of addressing and local computation |
| **Semantic switches** | Local decision primitives that operate on field statistics (not the focus of this paper) |
| **PSA Semantic Field Architecture** | The full system integrating representation, scoring, control, and adaptation |

---

## 2 Related Work

### 2.1 Embedding Space Geometry

The structure of embedding spaces has been studied extensively. Mikolov et al. (2013) demonstrated that word embeddings encode relational structure via linear offsets. Subsequent work probed BERT representations for linguistic properties (Hewitt and Manning, 2019), studied isotropy and anisotropy (Ethayarajh, 2019), and characterized the geometry of contextualized embeddings (Cai et al., 2021). These studies examine *properties of* the space. PSA imposes *coordinates on* it.

### 2.2 Dimensionality Reduction

PCA (Pearson, 1901), t-SNE (van der Maaten and Hinton, 2008), and UMAP (McInnes et al., 2018) produce low-dimensional representations for visualization. They are powerful tools, but they are not addressing systems. t-SNE and UMAP are stochastic — re-running produces different layouts. PCA axes are data-dependent and rotate with the corpus. None provide stable coordinates that persist as new data arrives.

The distinction is between visualization and addressability: PSA is designed to assign reusable locations, not only produce a plot.

### 2.3 Clustering and Topic Models

K-means, HDBSCAN (Campello et al., 2013), and topic models (Blei et al., 2003) assign texts to groups. Groups are discrete assignments; they do not support interpolation, neighborhoods, or gradients. A text is *in* cluster 47 or *not*. PSA coordinates are continuous — a text has a position, and positions have distances, neighborhoods, and spatial relationships.

### 2.4 Reference-Based Representations

The use of fixed reference points for spatial reasoning has deep precedent in cognitive science and psycholinguistics. Osgood et al. (1957) introduced the **semantic differential**: each concept is rated against a fixed bank of bipolar adjective scales (good–bad, active–passive, strong–weak), and the resulting profile is treated as a coordinate in a low-dimensional "semantic space." Factor analysis of those profiles produced the three canonical Osgood dimensions — evaluation, potency, activity — which have been replicated cross-linguistically. Gärdenfors (2000) generalized this idea into **conceptual spaces** with quality dimensions anchored by prototypical instances, modeling meaning geometrically between symbolic and subsymbolic representation. PSA can be viewed as a computational realization of this reference-point program, scaled from hand-designed bipolar adjective scales to dozens of contrastive *propositional* anchors and grounded in modern embedding geometry. The guide-star catalog is, in effect, an embedding-era semantic differential: each guide-star pair plays the role of one Osgood scale, and the star signature plays the role of the differential profile.

### 2.5 Compression and Information Preservation

The Johnson–Lindenstrauss lemma (1984) establishes that random projections preserve pairwise distances with bounded distortion. PSA is not a random projection — guide stars are semantically chosen, not random — but the J-L framework provides a theoretical baseline: even random projections to $O(\log n / \epsilon^2)$ dimensions preserve distances. The empirical question is whether semantically chosen reference points outperform what random projection would deliver at the same output dimensionality, and whether the resulting low-dimensional space behaves like a stable coordinate framework rather than an arbitrary embedding.

---

## 3 Method

### 3.1 Guide-Star Catalog

The initial coordinate frame is defined by 75 guide stars organized into seven ontological tiers, each containing contrastive pairs designed to maximally separate fundamental semantic distinctions. A later expanded frame adds 40 guide stars across five additional tiers derived from FrameBase (Rouces et al., 2015), for 115 guide stars total. The expanded frame is evaluated separately in Section 4.5 and Appendix B.3.

**Table 1.** Guide-star tier structure.

| Tier | Theme | Stars | Weight | Example Pair |
|------|-------|-------|--------|-------------|
| 0 | Frame Anchors | 0–11 | 1.0 | *"A physical object occupies space in the real world"* <-> *"The referenced object is completely absent and does not exist"* |
| 1 | Agency & Intent | 12–23 | 1.0 | *"A human agent deliberately performs a purposeful action"* <-> *"A human agent stops and discontinues their ongoing activity"* |
| 2 | Knowledge & Evaluation | 24–35 | 1.0 | *"A person already knows and understands the topic thoroughly"* <-> *"A person learns and gains new knowledge through active study"* |
| 3 | Possession & Structure | 36–47 | 1.0 | *"An owner holds and controls a valuable resource exclusively"* <-> *"An entity is missing and lacks a required component entirely"* |
| 4 | Time & Sequence | 48–59 | 1.0 | *"Event A happens first and precedes event B chronologically"* <-> *"Event B happens after and follows event A chronologically"* |
| 5 | Quantity & Comparison | 60–68 | 1.0 | *"The measured amount grows and increases substantially over time"* <-> *"The measured amount shrinks and decreases substantially over time"* |
| 6 | Modality & Constraint | 69–74 | 0.4 | *"The action is mandatory and required by the binding regulations"* <-> *"The action is forbidden and prohibited by the enforced policy"* |

**Table 1b.** Expanded guide-star tiers added in the 115-star frame.

| Tier | Theme | Stars | Weight | Example Pair |
|------|-------|-------|--------|-------------|
| 7 | Experience & Perception | 75–82 | 1.0 | *"A person feels a strong positive emotion..."* <-> *"A person feels fear, anger, sadness..."* |
| 8 | Motion, Contact & Manipulation | 83–90 | 1.0 | *"An entity moves by itself along a path..."* <-> *"An entity stays fixed in place..."* |
| 9 | Social & Institutional Events | 91–98 | 1.0 | *"People cooperate, coordinate, help..."* <-> *"People oppose, compete, fight..."* |
| 10 | Commerce & Resource Events | 99–106 | 1.0 | *"A buyer pays money or value..."* <-> *"A seller receives payment and transfers..."* |
| 11 | Biological & Physical World | 107–114 | 1.0 | *"A living organism eats, drinks..."* <-> *"A living organism becomes injured, sick..."* |

**Design principles.** Guide stars are full sentences, not keywords. Each star is a complete proposition whose embedding captures semantic content in context. Contrastive pairs are designed so that members are topically related but semantically opposed — ensuring maximal angular separation in embedding space while maintaining thematic coherence.

Tier 6 (Modality) is down-weighted to $w_6 = 0.4$ in signature computation. The motivating observation is that modal embeddings (mandatory/forbidden/permitted/possible) cluster tightly in the embedding model used here, and at unit weight they pull most sentences toward a single corner of the field. We treat $w_6$ as a hyperparameter of the frame rather than a principled constant; a systematic sweep is reserved for future work, and we report all results under the fixed value $w_6 = 0.4$ used at field-construction time.

The catalog is synthetic and fixed. It is designed as a **bootstrap scaffold**: a shared reference frame analogous to the International Celestial Reference System (ICRS) in astronomy. At scale, empirically discovered high-mass attractors (paraphrase groups that accumulate across many documents) may provide an additional frame layer, but the synthetic scaffold is not replaced. It provides the fixed baseline against which field evolution is measured.

The 115-star frame extends the scaffold into semantic regions that were thin in the original abstract-operator catalog: perception, motion/contact, social/institutional action, commerce/resource transfer, and biological/physical-world grounding. This is not a change in the PSA algorithm. It is a denser reference frame evaluated against the same falsification suite.

### 3.2 PSA Coordinate Computation

For a text fragment $t$ with embedding $\mathbf{e}(t) \in \mathbb{R}^{4096}$:

**Step 1. Embed guide stars.** Compute $\mathbf{e}(g_i)$ for each guide star $g_i$, $i \in \{0, \ldots, G-1\}$, where $G=75$ for the original frame and $G=115$ for the expanded frame, using the same embedding model. This is a one-time computation per model.

**Step 2. Compute star signature.** The star signature of $t$ is the $G$-dimensional vector:

$$\sigma(t) = \left[ w_{\tau(i)} \cdot \cos\left(\mathbf{e}(t), \mathbf{e}(g_i)\right) \right]_{i=0}^{G-1}$$

where $w_{\tau(i)}$ is the tier weight for star $i$ and $\cos(\cdot, \cdot)$ denotes cosine similarity. The signature captures $t$'s relationship to every semantic anchor simultaneously.

**Step 3. Compute pairwise guide-star distances.** The $G \times G$ cosine distance matrix $D$ between guide-star embeddings defines the intrinsic geometry of the reference frame:

$$D_{ij} = 1 - \cos\left(\mathbf{e}(g_i), \mathbf{e}(g_j)\right)$$

**Step 4. Embed guide stars in 3D.** Apply multidimensional scaling (MDS; Kruskal, 1964) to $D$ to obtain 3D positions for each guide star:

$$\mathbf{p}_i = \text{MDS}_3(D)_i, \quad i \in \{0, \ldots, G-1\}$$

The 3D positions are centered at the origin and normalized to the unit sphere. MDS is deterministic (seeded), so guide-star positions are fixed given the embedding model.

**Step 5. Derive PSA coordinates.** The 3D position of $t$ is a weighted blend of guide-star positions, weighted by signature activation:

$$\text{PSA}(t) = \frac{\sum_{i=0}^{G-1} \sigma_i(t) \cdot \mathbf{p}_i}{\sum_{i=0}^{G-1} \sigma_i(t)}$$

This places $t$ in the convex hull of the guide stars, closer to stars it resembles more. The coordinate is deterministic: same embedding -> same signature -> same position.

**Complexity.** Coordinate computation requires one dot product per guide star ($O(Gd)$ for $d$-dimensional embeddings), plus a weighted average of $G$ 3D points. The MDS step is one-time. Per-sentence computation is dominated by the embedding forward pass, not the PSA projection.

**On the compression ratio.** The pipeline performs a two-stage reduction: $\mathbb{R}^{4096} \to \mathbb{R}^G$ via the star signature (the information bottleneck), then $\mathbb{R}^G \to \mathbb{R}^3$ via the fixed MDS embedding of the guide-star geometry (a deterministic projection into the addressing space). The visible compression from 4,096 dimensions to 3D voxel addresses is therefore the *composition* of a semantic projection through $G$ anchors and a geometric projection through the MDS frame; the first stage is where information is selected, and the second stage is where it is laid out for indexing.

### 3.3 Voxelization

The continuous 3D PSA space is discretized into a hierarchical grid:

**Macro level (L1).** The normalized coordinate space is divided into cells of size 0.10 along each axis, yielding approximately $20 \times 20 \times 20 = 8{,}000$ possible macro voxels. Each occupied voxel receives an address of the form `L1_v_X_Y_Z`.

**Micro levels.** Within each occupied macro voxel, the $xy$-plane is recursively subdivided into $10 \times 10$ micro-cells. The $z$-axis is dropped at the micro boundary — an intentional design choice that preserves the dominant semantic axes at coarse resolution while allowing fine discrimination within a voxel on the two most informative projection dimensions.

$$\text{Address} = \texttt{L1\_v\_X\_Y\_Z:mc1:mc2:mc3}$$

Each micro-cell level provides 100× finer resolution. Three levels yield $10^6$ sub-addresses per macro voxel.

**Key design property: embeddings never move.** When a voxel gains depth (micro-cell subdivision), no data migrates. The embedding is the ground truth; the address is recomputed on demand at whatever resolution is needed. The voxel structure is an *index*, not a container.

### 3.4 Adaptive Grid

Voxel grid resolution adapts to local density. When a voxel exceeds a sentence-count threshold (default: 300), it automatically triggers micro-cell subdivision. Sparse regions remain at macro resolution. This concentrates measurement precision where the corpus is densest — an adaptive resolution property analogous to adaptive mesh refinement in computational physics.

Multiple grid levels (L1, L2) are supported, with L2 providing coarser resolution for overview navigation. In the experiments reported here, all results use L1 macro voxels.

---

## 4 Semantic Locality

### 4.1 Hypothesis

If PSA coordinates capture real semantic structure, then nearby voxels should contain semantically similar content. Specifically, the mean cosine similarity between voxel centroids should decay with inter-voxel distance. Under the null hypothesis — coordinates unrelated to semantics — no distance-dependent decay should be observed.

### 4.2 Experimental Setup

**Corpus.** Eight AI-related source documents plus an ontology seed, totaling 3,378 sentences after tokenization and sentence segmentation.

**Embedding model.** Qwen3 Embedding (8B parameters, 4,096 dimensions), selected for MTEB benchmark (Muennighoff et al., 2023) leadership in retrieval and semantic textual similarity tasks.

**Procedure.** For the original SSM-based locality runs, each occupied voxel with >= 2 sentences is represented by the centroid of the first 200 dimensions of its member sentence embeddings. This truncation was an engineering accommodation for AWS SSM output limits, not a theoretical requirement of PSA. In the follow-up control runs reported in Section 4.6, centroid construction and pairwise cosine computation are performed in the Neo4j environment over the full 4,096-dimensional embeddings, returning only scalar summary statistics. Compute all pairwise centroid cosine similarities and voxel distances, then fit a linear regression of similarity on distance.

**Null distribution.** Generate 1,000 random permutations of centroid-to-coordinate assignments (preserving voxel population sizes but destroying the structure-to-coordinate mapping). Compute the regression slope for each permutation to establish the null slope distribution.

### 4.3 Results

**Table 2.** Locality decay results across two independent ingestion orders. *Voxel counts reflect the locality-decay filter (occupied voxels with >= 2 embedded sentences whose centroids could be transported within SSM limits); other tests in the same runs apply different filters and therefore report different N.*

| Metric | Run 1 (sequential) | Run 2 (randomized) |
|--------|-------------------|-------------------|
| Occupied voxels | 470 | 445 |
| Voxel pairs analyzed | 110,215 | 98,790 |
| Decay slope | −0.00236 | −0.00281 |
| Pearson $r$ | −0.158 | −0.183 |
| Intercept | 0.7202 | 0.7196 |
| $p$-value | < $10^{-4}$ | < $10^{-4}$ |
| Null slope mean | −2.8 × $10^{-6}$ | −1.4 × $10^{-5}$ |
| Null slope std | 2.0 × $10^{-4}$ | 2.8 × $10^{-4}$ |

The decay slope is negative and highly significant in both runs. Nearby voxels contain more similar content than distant voxels, and this relationship is unlikely under the tested permutation null.

**Intercept interpretation.** The intercept of 0.72 means that sentences assigned to the same local neighborhood share approximately 72% cosine similarity in the original 4,096-dimensional space. This high baseline supports the interpretation that PSA coordinates preserve semantic proximity despite the aggressive two-stage compression ($4096 \to G \to 3$).

**Slope interpretation.** Each unit of Manhattan distance in the voxel grid corresponds to a decrease of approximately 0.003 in centroid cosine similarity. The relationship is monotonic over the first 8 distance units (covering the primary concentration of voxel pairs), with a characteristic "rise" at larger distances reflecting boundary effects in a bounded coordinate space with a non-uniform occupied region (see §4.4).

### 4.4 Decay Curve

The locality decay is not merely a statistical summary — it has geometric structure. Figure 1 (described) shows mean cosine similarity binned by voxel distance. The first 8 bins show clear monotonic decay from 0.79 (nearest neighbors) to 0.49 (distance about 9). Beyond distance 9, the curve rises again as it samples pairs from opposing sides of the occupied field region, where the bounded extent of the corpus and the convex-hull geometry of the guide-star MDS embedding introduce boundary effects. This U-shaped tail is an expected artifact of locality measurements in a bounded domain with a finite, non-uniform occupied region; it does not invalidate the local decay signal in the interior bins.

### 4.5 Expanded 115-Star Frame and Baselines

After the initial 75-star experiments, we reran the PSA test suite with an expanded 115-star frame. The added stars cover perception, motion/contact, social and institutional events, commerce/resource transfer, and biological/physical-world grounding. This run used the canonical normalized voxel-distance metric and a live field of 3,071 assigned sentences. For comparability, the 75-star Run 2 was also re-evaluated under the same canonical normalized metric; the original Run 1 and Run 2 numbers in Table 2 use raw Manhattan voxel distance, so slope magnitudes are not directly comparable across the two metrics.

**Table 2b.** Original-frame and expanded-frame locality results under the canonical normalized voxel-distance metric. *Both rows use the locality-decay filter (occupied voxels with >= 2 sentences whose first-$d$ centroid components fit within the SSM transport budget), which is stricter than the snapshot voxel count and explains the difference between the 674 / 68 occupied voxels reported in Section 4.7 and the 49 / 58 voxels analyzed here.*

| Metric | 75-star Run 2 (canonical normalized, 2026-05-05) | 115-star Run 3 (2026-05-07) |
|--------|--------------------------------------------------|-----------------------------|
| Guide stars | 75 | 115 |
| Assigned sentences | 3,076 eligible | 3,071 |
| Occupied voxels | 49 with >= 2 embedded sentences | 58 with >= 2 embedded sentences |
| Voxel pairs analyzed | 1,176 | 1,653 |
| Locality $r$ | −0.2875 | −0.3118 |
| Decay slope | −0.07482 | −0.11479 |
| Intercept | 0.8103 | 0.7886 |
| Permutation $p$ | 0.010 | 0.001 |
| Null slope mean | 0.001062 | 0.000318 |
| Null slope std | 0.02755 | 0.03284 |
| Verdict | PASS | PASS |

The expanded frame increases the magnitude of both the correlation and the slope under the same canonical normalized distance convention. The slope becomes 53% steeper ($-0.1148$ vs. $-0.0748$), and the permutation significance improves from $p=0.010$ to $p=0.001$.

**Table 2c.** Baseline runs for native embedding-space comparison.

| Baseline | 75-star Run 2 (2026-05-05) | 115-star Run 3 (2026-05-07) |
|----------|----------------------------|-----------------------------|
| Centroid baseline dimensions | 200 | 200 |
| Centroid baseline pairs | 1,176 | 1,653 |
| Centroid cross-space $r$ | 0.2875 | 0.3118 |
| Centroid mean cosine | 0.7535 | 0.7027 |
| Sentence baseline dimensions | 4,096 | 4,096 |
| Sentence baseline pairs | 400 | 418 |
| Sentence locality $r$ | −0.1715 | −0.1880 |
| Sentence baseline slope | −0.04559 | −0.05443 |
| Sentence mean cosine | 0.3554 | 0.3569 |

The sentence-level baseline computes full 4,096-dimensional cosine similarities inside Neo4j between *sampled sentence pairs* and returns only scalar similarities, avoiding SSM transport limits. It provides a native-space check against the PSA centroid result, but the two measurements are not directly comparable in geometry: PSA centroid locality aggregates within voxels before comparison, while the sentence baseline compares raw sentence pairs without aggregation. The relevant comparison is therefore qualitative — in the expanded 115-star run, voxel-aggregated PSA locality ($r=-0.3118$, slope $=-0.1148$) is stronger than the unaggregated sentence-pair baseline ($r=-0.1880$, slope $=-0.0544$), consistent with the field organizing semantic neighborhoods more sharply at the voxel level than they appear at the pairwise sentence level.

### 4.6 Full-Dimensional and Random-Anchor Controls

We then ran the fully apples-to-apples control: full-4,096-dimensional voxel centroids computed next to Neo4j, with only scalar summaries exported. On the live expanded 115-star field (2026-05-18; 3,304 eligible embedded sentences), the full-dimensional centroid result strengthens rather than weakens the PSA locality claim: $r=-0.3259$, slope $=-0.1275$, intercept $=0.7721$ over 59 voxels and 1,711 voxel pairs. A 10,000-pair validation sample comparing first-200D cosine to full-4,096D cosine returned Pearson $r=0.8804$ and Spearman $\rho=0.8658$, so the 200D transport-limited runs should be read as useful approximations, not as near-lossless substitutes for full-dimensional evaluation.

For the random-anchor controls, we recompute PSA-style coordinates using either randomly selected sentence embeddings or random unit vectors as anchors. We report locality together with occupancy diagnostics. Let $n_v$ be the number of assigned sentences in occupied voxel $v$, $N=\sum_v n_v$, and $p_v=n_v/N$. We define occupancy entropy as $H=-\sum_v p_v \log p_v$ and **effective occupied voxels** as $N_\mathrm{eff}=e^H$. Field collapse is operationally indicated by locality-like decay paired with low occupied voxel count, low $N_\mathrm{eff}$, or high maximum voxel fraction $\max_v p_v$.

**Table 2d.** Random-anchor collapse control on the expanded field. *Control rows summarize 50 random trials at $G=115$. Occupied voxels are reported as mean (median) across trials; effective occupied voxels, max voxel fraction, locality $r$, and slope are medians across trials. The real-PSA row is a single deterministic run.*

| Frame | Occupied voxels (mean / median) | Effective occupied voxels (median) | Max voxel fraction (median) | Locality $r$ (median) | Slope (median) |
|-------|-------------------------------|------------------------------------|----------------------------|----------------------|----------------|
| Real PSA 115-star frame | 59 / 59 | 18.09 | 21.9% | −0.3259 | −0.1275 |
| Random sentence anchors ($G=115$) | 8.54 / 8 | 6.63 | 26.4% | −0.8476 | −0.1864 |
| Random unit-vector anchors ($G=115$) | 24.4 / 24 | 8.25 | 28.5% | −0.5740 | −0.1231 |

This control shows that locality slope alone is not a sufficient criterion for semantic coordinate quality. On slope alone, random sentence anchors outperform the real PSA frame ($-0.1864$ median vs. $-0.1275$). The discriminating observable is effective occupancy: real PSA preserves 18.09 effective occupied voxels, while random sentence and random unit-vector anchors retain only 6.63 and 8.25 at the median. Random unit-vector anchors approach the real slope but still show worse effective occupancy and higher concentration. PSA's advantage is therefore not merely a steeper decay curve; it is the joint behavior of significant full-dimensional locality with a richer, less-collapsed address space. We treat occupied voxel count, effective occupied voxels, and maximum voxel fraction as primitive diagnostics of coordinate-frame collapse, not as a new composite metric.

**Why random anchors show very high $|r|$.** The high median Pearson $r$ for random sentence anchors ($-0.8476$) is itself a signature of collapse rather than a sign of stronger semantic structure. With only $\sim 6$–$8$ effective occupied voxels, the regression of centroid similarity on voxel distance is fit over a small number of voxel pairs spanning a degenerate, low-dimensional occupied region, which produces a tight linear correlation almost by construction. This is the central reason the joint criterion is needed: a coordinate frame can have an arbitrarily high $|r|$ and an arbitrarily steep slope and still be useless if the field has collapsed to a handful of addresses. Reporting locality without occupancy diagnostics would have hidden exactly this failure mode.

### 4.7 Paraphrase Consolidation and Semantic Mass

The expanded 115-star run also increased paraphrase consolidation. In the 75-star randomized snapshot, 539 sentence memberships belonged to paraphrase groups out of 3,378 assigned sentences (15.96%). In the 115-star expanded-frame snapshot, 689 sentence memberships belonged to paraphrase groups out of 3,071 assigned sentences (22.44%). The number of paraphrase groups decreased from 506 to 421 while paraphrase mass increased, raising mean paraphrase mass per group from 1.07 to 1.64.

**Table 2e.** Paraphrase consolidation before and after guide-star expansion. *Voxel counts here are the snapshot occupancy (all voxels with at least one assigned sentence), which is larger than the locality-decay voxel count in Table 2b because the locality test requires >= 2 embedded sentences per voxel and SSM-transportable centroids.*

| Metric | 75-star Run 2 snapshot | 115-star Run 3 snapshot |
|--------|------------------------|--------------------------|
| Assigned sentences | 3,378 | 3,071 |
| Occupied voxels | 674 | 68 |
| Paraphrase groups | 506 | 421 |
| Paraphrase mass | 539 | 689 |
| Paraphrase mass / assigned sentences | 15.96% | 22.44% |
| Mean mass per paraphrase group | 1.07 | 1.64 |

This is a different signal from locality decay. Locality asks whether nearby voxels contain similar content. Paraphrase consolidation asks whether multiple surface forms collapse toward fewer stable meaning structures. The simultaneous increase in locality strength and paraphrase mass suggests that the expanded field is reducing semantic fragmentation rather than merely compressing coordinates. In natural-language terms, the field is becoming more semantically compressive: redundant surface expressions are being pulled into shared meaning structures while the spatial decay curve remains falsifiable.

This result is consistent with the emergence of semantic attractors. Natural language distributions often contain a small number of high-frequency regions, many medium-frequency regions, and a long tail of sparse expressions. If PSA is approaching such a distribution, future runs should show a paraphrase-node size distribution with a heavy tail, stable high-mass attractors, and increasing persistence of those attractors across ingestion orders and corpus growth.

The engineering changes preceding this run are relevant to interpretation. The expanded run followed guide-star expansion, richer WordNet (Miller, 1995) sense contextualization, FrameBase (Rouces et al., 2015) and CCO integration, and embedding of senses with explanatory sentences rather than symbolic labels alone. Embedding a label such as "bank" is ambiguous; embedding contextualized glosses such as "financial institution that holds money" and "land beside a river" creates separable semantic attractors. This gives the ontology semantic texture in the same linguistic space used for PSA projection and object resolution. The observed increase in paraphrase capture, measured against a Sentence-BERT-style paraphrase-similarity baseline (Reimers and Gurevych, 2019) applied per voxel, is therefore plausibly attributable to better semantic anchoring, not merely embedding overfit.

---

## 5 Order Invariance

### 5.1 Motivation

A coordinate framework is only useful if it is stable. If ingesting documents in a different order produces a different field geometry, then PSA coordinates are artifacts of processing history, not properties of content. Order invariance is a necessary condition for PSA to behave as a reusable coordinate framework rather than an indexing side effect.

### 5.2 Experimental Design

We ingested the same eight-document corpus, with the ontology seed held fixed, under two orderings:

- **Run 1 (sequential):** Documents ingested in presentation order, text sections before metadata.
- **Run 2 (randomized):** Sentences shuffled into random order before ingestion.

Both runs used identical guide stars, embedding model, and voxelization parameters. After ingestion, we ran the full PSA evaluation suite on each run independently.

### 5.3 Results

**Table 3.** Cross-run comparison.

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| **Locality** | | | |
| Decay slope | −0.00236 | −0.00281 | +18.7% (stronger) |
| Pearson $r$ | −0.158 | −0.183 | +15.7% (stronger) |
| Intercept | 0.7202 | 0.7196 | −0.08% |
| $p$-value | < $10^{-4}$ | < $10^{-4}$ | — |
| **PRECEDES topology** | | | |
| Total PRECEDES edges | 638 | 638 | 0 |
| Intra-voxel edges | 1 | 1 | 0 |
| Cross-voxel edges | 637 | 637 | 0 |
| **Jaccard/Salton locality** | | | |
| Salton slope | −3.20 × $10^{-5}$ | −3.06 × $10^{-5}$ | Δ = 1 × $10^{-6}$ |
| Mean Salton index | 4.85 × $10^{-4}$ | 4.78 × $10^{-4}$ | −1.5% |
| $p$-value | < $10^{-4}$ | < $10^{-4}$ | — |
| **Guide stars** | | | |
| Guide-star count | 75 | 75 | 0 |
| Standard candle (`L1_v_10_9_11`) | 9 sentences | 9 sentences | 0 |
| Embedding coverage | 100% | 100% | — |

**Key findings:**

1. **PRECEDES edges are identical.** The same 638 discourse-sequence relationships were detected in both runs. This provides strong evidence for order invariance under the evaluated frame: the topology is not merely similar across runs, but identical.

2. **Locality is stronger under randomization.** The decay slope increases by 18.7% in the randomized run. This is consistent with sequential ingestion introducing temporal correlation (nearby sentences from the same document tend to be ingested together), which slightly dilutes the semantic signal. Randomization removes this bias, producing tighter locality.

3. **Intercepts are stable to three decimal places.** The baseline similarity of co-located content (0.720) behaves as a within-corpus invariant rather than an ingestion artifact.

4. **The standard candle holds.** Voxel `L1_v_10_9_11` — designated as the standard candle for its consistent population across runs — contains exactly 9 sentences in both trials, drawn from the same documents.

### 5.4 Significance

These results support the interpretation that PSA coordinates are properties of the *content*, not the *processing*, **within a fixed guide-star frame**. The same sentences land in the same neighborhoods regardless of arrival order, and the discourse graph reconstructs identically. The result is consistent with content-driven organization rather than processing-order effects.

We distinguish this within-frame invariance from across-frame stability. Changing the guide-star catalog changes the coordinate basis, so absolute voxel addresses are not expected to survive a frame change — only relational quantities such as locality slope, paraphrase mass, and graph topology are comparable across frames. The expanded-frame result in Section 4.5 is best read in this lens: a denser frame producing sharper relational geometry, not a continuation of the same coordinate basis.

---

## 6 Falsification: The Scramble Control

### 6.1 Motivation

Locality decay alone does not prove that PSA captures meaningful structure. Any projection from high dimensions to low dimensions may produce apparent spatial correlation simply because the projection preserves some pairwise relationships by construction. The critical question is: does the observed decay significantly exceed what a random coordinate assignment would produce?

This is the **force-directed layout trap** (Kobourov, 2012): many graph visualizations look structured because of layout algorithms, not because of data properties. The scramble test addresses this concern. If the observed relation depends on semantic assignment, it should degrade when centroid-to-coordinate assignments are permuted.

This scramble control targets the weakest null: that centroid-to-coordinate assignment is unrelated to semantic content. The random-anchor controls in Section 4.6 address a stronger null: that any reference frame with enough anchors would produce comparable field quality. The random-anchor result refines rather than replaces the scramble result: arbitrary anchors can produce locality-like slopes, but the resulting fields are substantially more collapsed.

### 6.2 Procedure

1. Compute the real locality decay slope from voxel centroids and coordinates.
2. Generate 1,000 scrambled configurations: randomly permute which centroid lives at which coordinate, preserving the set of centroids and the set of coordinates but destroying the mapping between them.
3. For each scrambled configuration, compute the decay slope.
4. Compare the real slope to the scrambled null distribution.

This is a permutation test in the classical sense (Fisher, 1935; Good, 2005). The null hypothesis is that centroid-to-coordinate assignment is independent of semantic content. If the real slope falls in the extreme tail of the null distribution, we reject the null.

### 6.3 Results

**Table 4.** Scramble control results. *Voxel and pair counts reflect the centroids successfully retrieved for the scramble test, which uses the same locality-decay filter as Table 2 / Table 2b.*

| Metric | 75-star Run 2 | 115-star Run 3 |
|--------|---------------|----------------|
| Real decay slope | −0.002807 | −0.118659 |
| Scrambled slope mean | −0.000014 | 0.000536 |
| Scrambled slope std | 0.000277 | 0.032100 |
| Empirical $p$-value | < 0.001 (0/1,000) | < 0.001 (0/1,000) |
| Permutation-based $z$ (Gaussian extrapolation, see §6.4) | 10.1 | 3.71 |
| $n$ voxels | 445 | 59 |
| $n$ pairs | 98,790 | 1,711 |
| Permutations | 1,000 | 1,000 |
| Verdict | **PASS** | **PASS** |

In both frames, not a single permutation out of 1,000 produced a slope as negative as the real value. The scrambled distributions are centered near zero, confirming that random coordinate assignment produces no systematic decay. In the original 75-star run, the real slope sits roughly ten standard deviations below the scrambled mean under a Gaussian extrapolation; in the expanded 115-star run, the corresponding figure is roughly four standard deviations. The empirical bound of $p < 0.001$ is identical for both frames; the difference in $z$ comes from the lower variance of the larger 75-star pair sample.

**Interpretation.** When you scramble the mapping between semantic content and spatial coordinates, the locality signal vanishes completely. The signal is in the geometry, not in the indexing.

### 6.4 Statistical Rigor

The primary statistical claim is the empirical permutation result: zero of 1,000 scramble permutations produced a slope as extreme as the observed value in either frame, giving an empirical $p < 0.001$. This is the bound we rely on.

The $z$-scores reported in Table 4 are Gaussian extrapolations from the empirical mean and standard deviation of the permutation distribution. They are useful as effect-size summaries but should not be read as precise tail probabilities, for two reasons. First, each scrambled slope is computed over voxel pairs that share voxels and are therefore not independent, so the variance of the null is not the variance of an independent-pair statistic. Second, the Gaussian assumption is not validated in the deep tail. To *empirically* confirm $5\sigma$ ($p = 2.87 \times 10^{-7}$) would require on the order of $10^7$ permutations with zero hits, which we have not performed. Comparisons to particle-physics discovery thresholds should be read in that light: the effect size is large under any reasonable parametric tail model, but the rigorous result is the empirical $p < 0.001$ from 1,000 permutations, exceeding the conventional $p = 0.05$ threshold by more than two orders of magnitude.

---

## 7 Evaluation Framework

### 7.1 The PSA Field Test Suite

We define a standardized evaluation framework for PSA fields, comprising five tests:

1. **Voxel Snapshot.** Census of the field: occupied voxels, sentence counts, paraphrase group counts, standard candle verification.

2. **Locality Decay.** Linear regression of centroid cosine similarity on voxel Manhattan distance, with permutation null.

3. **Jaccard/Salton Locality.** Analogous to locality decay but using PRECEDES-edge co-occurrence (Salton index) instead of embedding similarity. Tests whether discourse structure, not just embedding proximity, is spatially organized.

4. **Guide-Star Verification.** Confirms complete guide-star coverage, embedding and micro-PSA assignment completeness, and standard candle reproducibility.

5. **Falsification (Scramble).** The permutation control described in Section 6.

The suite is implemented as a command-line tool with per-test subcommands and JSON result output. All results are saved with timestamps and run identifiers for longitudinal comparison.

### 7.2 Standard Candle

Analogous to standard candles in observational cosmology, we designate a specific voxel (`L1_v_10_9_11`) as a reference target for the 75-star frame. This voxel must contain the same number of sentences across runs and maintain consistent source-document composition. If the standard candle drifts within a fixed frame, the coordinate basis has shifted: an immediate diagnostic signal for within-frame invariance.

In both 75-star experimental runs (sequential and randomized), the standard candle contained exactly 9 sentences drawn from the same source documents. In the expanded 115-star frame, this voxel is empty. This is expected and not a failure of the invariance claim: the standard-candle test diagnoses *within-frame* drift, and the 75-star and 115-star frames are different coordinate bases. The expanded-frame run therefore requires a new standard candle to be selected and frozen for future 115-star repeatability tests, just as a new reference catalog is required when astronomical coordinate frames are revised. The within-frame invariance result of Section 5 is unchanged.

### 7.3 Reproducibility

All experiments use:
- Deterministic MDS seed (random_state=42)
- Fixed guide-star catalog within each frame (75-star original frame or 115-star expanded frame, version-hashed)
- Identical embedding model (Qwen3 Embedding 8B)
- Identical voxelization parameters (grid_size=0.10, L1)

The evaluation suite, guide-star catalog, and result data will be released.

---

## 8 Discussion

### 8.1 What PSA Is

PSA is an **addressable semantic projection**. It is not a clustering algorithm: it produces locations, not group labels. It is not a dimensionality reduction for visualization: it produces stable addresses, not stochastic layouts. It is not an information retrieval system: it produces a *field* with measurable properties at every occupied point.

The distinction matters because coordinates enable operations that groups and visualizations do not:

- **Interpolation.** What lies between two known regions?
- **Neighborhoods.** What is near a given idea?
- **Gradients.** In which direction does semantic content change fastest?
- **Distance.** How far apart are two ideas, measured in a common frame?
- **Void detection.** Where does the field have no content — and what does that absence mean?

### 8.2 The Compression Result

The central empirical result is that a two-stage projection — $\mathbb{R}^{4096} \to \mathbb{R}^G$ via the star signature, then $\mathbb{R}^G \to \mathbb{R}^3$ via the fixed MDS embedding of the guide-star geometry — yields voxel coordinates that preserve:

- **Local neighborhoods:** 0.72 baseline cosine similarity at intercept (Run 1 / Run 2)
- **Distance-dependent decay:** $r = -0.18$, empirical $p < 10^{-4}$ in the 75-star Manhattan-metric runs; $r = -0.3118$, $p = 0.001$ in the expanded 115-star run under the canonical normalized metric
- **Graph topology:** 638 identical PRECEDES edges across runs
- **Falsification margin:** zero of 1,000 scramble permutations match the observed slope in either frame

The information bottleneck is the $4096 \to G$ stage; the subsequent $G \to 3$ stage is a deterministic geometric layout into the addressing space. The guide stars act as a semantic basis: not orthogonal in the linear-algebra sense, but functionally spanning the space of meaningful distinctions covered by the catalog. The Johnson–Lindenstrauss baseline guarantees that random projections preserve pairwise distances at dimensionalities far higher than $G$; the empirical question here is whether *semantically chosen* anchors yield a low-dimensional space with stable coordinate behavior. The evidence in these experiments is positive.

The expanded 115-star result is important because, under the canonical normalized metric, it strengthens the slope. Relative to the canonical 75-star baseline measured on 2026-05-05, the expanded frame increases locality correlation magnitude from 0.2875 to 0.3118 and steepens the decay slope from $-0.0748$ to $-0.1148$. Denser semantic anchoring sharpens the field rather than merely adding noise.

The full-dimensional centroid control strengthens this interpretation. When voxel centroids are computed over all 4,096 embedding dimensions, the expanded field yields $r=-0.3259$ and slope $=-0.1275$ over 1,711 voxel pairs. The signal is therefore not an artifact of the 200-dimensional SSM transport approximation.

The random-anchor controls also refine the interpretation. Random anchors can produce locality-like slopes, and random sentence anchors can even produce steeper slopes than the real PSA frame. But those slopes coincide with geometric collapse: random sentence anchors reduce the field to 8.54 occupied voxels on average, compared with 59 for the real frame, and random unit-vector anchors retain only 8.25 effective occupied voxels at the median. A valid semantic coordinate frame must therefore be evaluated jointly: locality is necessary, but not sufficient; the field must also preserve occupancy diversity and avoid concentration collapse.

The paraphrase consolidation result strengthens this interpretation. A field that merely coarsens coordinates could increase grouping while damaging locality. Here, paraphrase mass rises from 15.96% to 22.44% while locality strengthens and scramble falsification remains significant. That combination is better described as **semantic fragmentation reduction**: fewer disconnected surface forms, more shared meaning structures, and sharper spatial organization.

### 8.3 Why Randomization Helps

The 18.7% increase in locality slope under randomized ingestion is an instructive result. Sequential ingestion introduces a confound: sentences from the same document section are temporally correlated and may share stylistic or topical features beyond their core semantic content. This correlation slightly blurs the relationship between PSA coordinates and pure semantic similarity.

Randomization reduces temporal correlation, leaving semantic content as the main organizing factor under this test. The stronger locality signal under randomization supports the interpretation that PSA coordinates track semantic content rather than document order alone.

### 8.4 The Field Interpretation

The results support interpreting PSA not as an indexing scheme but as a **semantic field** — a spatial domain with measurable spatial regularities:

- **Locality.** The field has a characteristic decay length over which similarity drops.
- **Topology.** The field has invariant topological features (PRECEDES edges) that are properties of content, not processing.
- **Anisotropy by construction.** The guide-star tier structure encodes directional semantic axes (agency, causality, modality) by design; whether motion along those axes corresponds to interpretable semantic transitions in the live field is suggested by the tier weighting and same-layer locality results but not directly demonstrated here, and is left to future work.
- **Within-corpus invariants.** Within this corpus, the intercept (0.72) and total PRECEDES edge count (638) behave as observed invariants: stable across the two ingestion orders tested. Whether they replicate as cross-corpus invariants awaits evaluation on larger and more diverse corpora.

Whether this field interpretation extends to larger corpora, and whether similar structural regularities appear across domains, is an open empirical question. The results here support the field interpretation for the scale tested.

### 8.5 Limitations

**Corpus scale.** The experimental corpus (3,378 sentences, 8 source documents plus an ontology seed) is small by NLP standards. The locality signal is clear at this scale, but it remains to be shown whether PSA structure strengthens, weakens, or qualitatively changes at enterprise (10K+ documents) or web scale.

**Occupied-field scale.** The expanded full-dimensional control contains 59 occupied voxels after the >= 2-sentence filter. This is enough to expose the random-anchor collapse effect, but it is still a small field. The natural prediction is that a semantically resolving guide-star frame should open more distinct regions as corpus scale and topical diversity grow, while degenerate anchor frames should remain comparatively concentrated. This prediction requires larger-corpus validation.

**Guide-star selection and random-anchor controls.** The guide-star catalog was designed by human judgment, seeded from ontological categories and later expanded with FrameBase-derived coverage. While the contrastive-pair structure provides principled coverage, the catalog has not been optimized end-to-end. The random-anchor controls in Section 4.6 show that arbitrary anchor systems can produce locality-like slopes, but often by collapsing the coordinate field into a small number of occupied voxels. This answers part of the "any anchors will do" criticism, but it does not prove optimality of the current guide-star catalog. It motivates multi-observable evaluation of future frames: locality, occupancy diversity, spatial dispersion, and concentration must be reported together.

**Full-dimensional voxel-centroid baseline.** The sentence-level baseline in Section 4.5 compares unaggregated sentence pairs against voxel-aggregated PSA centroids and is therefore not a strict apples-to-apples baseline. The follow-up full-4,096-dimensional voxel-centroid run closes this gap for the expanded field and strengthens the locality result, but it was run after the original SSM-constrained evaluation suite. Future releases should make full-dimensional centroid computation the default baseline path.

**Embedding model dependence.** All results use a single embedding model (Qwen3 8B). Cross-model stability — whether the same guide stars produce the same field geometry under different embedding models — is not tested here. Preliminary design analysis suggests that PSA coordinates should be stable across MTEB-competitive models, but this requires empirical validation.

**3D projection.** The choice of 3 dimensions is a design decision, not an optimality result. Higher-dimensional PSA coordinates would preserve more information at the cost of voxel grid sparsity and interpretability. The optimal dimensionality for a given corpus and task is an open question.

**Centroid truncation.** The 75-star locality analysis uses the first 200 embedding dimensions per centroid, truncated to accommodate transport constraints (AWS SSM 24KB output limit). A 10,000-pair validation sample on the expanded field found Pearson $r=0.8804$ and Spearman $\rho=0.8658$ between first-200D and full-4,096D cosine similarities. The approximation preserves a useful signal but is not near-lossless; full-dimensional server-side computation is preferred for final comparisons.

**Tier-6 weight.** Modality tier weight $w_6 = 0.4$ is treated as a hyperparameter (Section 3.1). An ablation across $w_6 \in \{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ is reserved for future work.

**Paraphrase detection scope.** Paraphrase detection operates within voxels and their immediate neighbors. Cross-field paraphrases (semantically equivalent sentences in distant voxels) may be missed. The 1 intra-voxel PRECEDES edge vs. 637 cross-voxel edges in the 75-star runs suggests this is not a significant concern at current scale, but it warrants investigation at larger scale.

### 8.6 Future Work

**Frame-quality diagnostics and guide-star expansion.** The random-anchor control suggests that coordinate quality is not determined by locality slope alone. A valid anchor frame must provide semantic resolution: preserving locality while maintaining occupancy diversity and avoiding address-space collapse. This motivates structured guide-star expansion using curated semantic resources such as WordNet (Miller, 1995) and FrameBase (Rouces et al., 2015), where candidate anchors can be selected to improve frame and sense coverage under explicit locality, redundancy, and collapse constraints. The next expansion should treat WordNet synset glosses, WordNet usage examples, FrameBase frame descriptions, and ontology descriptions as a candidate pool, then use greedy or beam selection to maximize coverage gain while penalizing redundancy and collapse. We do not claim global optimality for this combinatorial subset-selection problem. Future work should report locality together with occupied voxel count, occupancy entropy, effective occupied voxels, spatial dispersion, and attractor concentration. We intentionally avoid introducing a composite score here; the primitive observables are more transparent at this stage.

**Paraphrase mass distributions.** The expanded-frame increase in paraphrase mass motivates explicit tracking of paraphrase-node size distributions. Future evaluations should report tail exponent, growth rate of high-mass paraphrase nodes, persistence of attractors across ingestion orders, and stability of high-mass nodes under guide-star frame updates. A power-law or Zipf-like distribution (Zipf, 1949; Clauset et al., 2009) would support the hypothesis that human semantic expression forms measurable attractor distributions inside PSA space.

**Cross-corpus regularities.** A stronger future result would be replication of field morphological properties (power-law exponents, fractal dimensions, persistence diagrams) across unrelated corpora. Such a result would provide evidence for semantic-geometric regularities beyond the evaluated corpus.

**Learned guide stars.** End-to-end optimization of guide-star texts to maximize locality decay slope, cross-model stability, or downstream task performance.

**Semantic Energy Density and morphological analysis.** As a longer-horizon program, we plan to define a composite scalar field over the voxel grid integrating occupancy, cross-scale consistency, source diversity, contradiction pressure, temporal stability, and guide-star anchoring. With sufficient corpus scale, such a field would enable power-law scaling analysis of voxel populations, fractal-dimension estimation via box-counting, and persistent-homology characterization of attractor cores, filaments, and voids. We flag this as speculative until corpus scale justifies it.

---

## 9 Conclusion

We have presented evidence that a fixed set of semantic reference concepts can support a stable, falsifiable, and order-invariant coordinate framework over embedding space. The 75-star frame preserves neighborhood structure through a two-stage projection ($4096 \to G \to 3$), exhibits locality decay with empirical $p < 10^{-4}$, reproduces identical graph topology across independent ingestion trials, and survives scramble falsification with zero of 1,000 permutations matching the observed slope. The expanded 115-star frame, evaluated under the same canonical normalized voxel metric, produces $r=-0.3118$, slope $=-0.1148$, $p=0.001$, again with zero matching permutations, and simultaneously raises paraphrase mass per assigned sentence from 15.96% to 22.44% while reducing the number of paraphrase groups. This pattern is consistent with semantic-fragmentation reduction rather than coarser binning. A follow-up full-4,096-dimensional voxel-centroid baseline strengthens the expanded-frame locality result to $r=-0.3259$, slope $=-0.1275$.

PSA provides a way to treat embedding space as a navigable field with addresses, neighborhoods, boundaries, and measurable local properties. The guide-star mechanism provides stability; voxelization provides hierarchy; the scramble control provides falsifiability. The random-anchor controls add a further lesson: apparent locality is not enough. Degenerate anchor systems can create steep decay by collapsing the field into a small number of occupied voxels. PSA's empirical advantage in these experiments is the joint preservation of locality and field occupancy.

The question this work opens is how addressable semantic coordinate frames should be evaluated when locality, occupancy, dispersion, and concentration interact. These results motivate larger-corpus evaluation of PSA and related reference-frame methods under the same multi-observable controls.

---

## Acknowledgments

Generative AI tools, including ChatGPT/Codex, were used to assist with language polishing, structural critique, drafting, and revision of portions of this manuscript, including the abstract, expanded-frame results discussion, limitations, future work, conclusion, and appendix summaries. The authors reviewed and are responsible for all content, experiments, numerical results, citations, and claims in the submission. Generative AI tools are not listed as authors.

---

## References

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022. JMLR: blei03a.

Cai, X., Huang, J., Bian, Y., & Church, K. (2021). Isotropy in the contextual embedding space: Clusters and manifolds. In *International Conference on Learning Representations (ICLR 2021)*. OpenReview: xYGNO86OWDH.

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining* (PAKDD 2013), Lecture Notes in Computer Science, pp. 160–172. Springer. doi: 10.1007/978-3-642-37456-2_14.

Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661–703. doi: 10.1137/070710111.

Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In *Proceedings of EMNLP-IJCNLP 2019*, pp. 55–65. Association for Computational Linguistics. doi: 10.18653/v1/D19-1006.

Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd. OCLC: 2417943.

Gärdenfors, P. (2000). *Conceptual Spaces: The Geometry of Thought*. MIT Press. ISBN: 9780262071994.

Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer. doi: 10.1007/b138696.

Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. In *Proceedings of NAACL-HLT 2019*, pp. 4129–4138. Association for Computational Linguistics. doi: 10.18653/v1/N19-1419.

Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. *Contemporary Mathematics*, 26, 189–206. doi: 10.1090/conm/026/737400.

Kobourov, S. G. (2012). Spring embedders and force directed graph drawing algorithms. *arXiv:1201.3011*.

Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. *Psychometrika*, 29(1), 1–27. doi: 10.1007/BF02289565.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems 26*. NeurIPS paper ID: 5021.

Miller, G. A. (1995). WordNet: a lexical database for English. *Communications of the ACM*, 38(11), 39–41. doi: 10.1145/219717.219748.

Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. In *Proceedings of EACL 2023*, pp. 2014–2037. Association for Computational Linguistics. doi: 10.18653/v1/2023.eacl-main.148.

Osgood, C. E., Suci, G. J., & Tannenbaum, P. H. (1957). *The Measurement of Meaning*. University of Illinois Press. ISBN: 9780252745393.

Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. *Philosophical Magazine*, 2(11), 559–572. doi: 10.1080/14786440109462720.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019*, pp. 3982–3992. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410.

Rouces, J., de Melo, G., & Hose, K. (2015). FrameBase: Representing N-ary relations using semantic frames. In *The Semantic Web. Latest Advances and New Domains* (ESWC 2015), Lecture Notes in Computer Science 9088, pp. 505–521. Springer. doi: 10.1007/978-3-319-18818-8_31.

Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort: An Introduction to Human Ecology*. Addison-Wesley Press. OCLC: 268582.

van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605. JMLR: vandermaaten08a.

---

## Appendix A: Original Guide-Star Catalog

The complete original 75-star catalog with contrastive pair assignments. The expanded 115-star frame appends tiers 7–11 described in Table 1b; those added stars are maintained in the code catalog and are summarized in Appendix B.3 through their evaluation results.

| ID | Tier | Contrast | Text |
|----|------|----------|------|
| 0 | Frame Anchors | <-> 1 | A physical object occupies space in the real world. |
| 1 | Frame Anchors | <-> 0 | The referenced object is completely absent and does not exist. |
| 2 | Frame Anchors | <-> 3 | A specific event happens at a documented point in time. |
| 3 | Frame Anchors | <-> 2 | Nothing happens and no events occur during the entire timeframe. |
| 4 | Frame Anchors | <-> 5 | The earlier action directly produces the later consequence. |
| 5 | Frame Anchors | <-> 4 | The blocking mechanism stops the expected result from occurring. |
| 6 | Frame Anchors | <-> 7 | The final outcome requires the prerequisite condition to be met first. |
| 7 | Frame Anchors | <-> 6 | The observable effect follows automatically from the root cause. |
| 8 | Frame Anchors | <-> 9 | The original state transforms completely into a new and different state. |
| 9 | Frame Anchors | <-> 8 | The existing condition stays exactly the same with no alteration. |
| 10 | Frame Anchors | <-> 11 | The process initiates and starts running from the beginning. |
| 11 | Frame Anchors | <-> 10 | The activity terminates and stops at its conclusion. |
| 12 | Agency & Intent | <-> 13 | A human agent deliberately performs a purposeful action. |
| 13 | Agency & Intent | <-> 12 | A human agent stops and discontinues their ongoing activity. |
| 14 | Agency & Intent | <-> 15 | A human agent builds and produces a new tangible artifact. |
| 15 | Agency & Intent | <-> 14 | A human agent operates an existing tool to accomplish a task. |
| 16 | Agency & Intent | <-> 17 | A person wants and desires to acquire a specific resource. |
| 17 | Agency & Intent | <-> 16 | A person tries and struggles to achieve a difficult goal. |
| 18 | Agency & Intent | <-> 19 | A person designs and prepares a detailed plan for future execution. |
| 19 | Agency & Intent | <-> 18 | A person decides and selects one option from multiple alternatives. |
| 20 | Agency & Intent | <-> 21 | A speaker transmits factual information to a listener. |
| 21 | Agency & Intent | <-> 20 | A requester asks for help and assistance from a provider. |
| 22 | Agency & Intent | <-> 23 | A person explicitly agrees with and supports the stated proposition. |
| 23 | Agency & Intent | <-> 22 | A person explicitly disagrees with and opposes the stated claim. |
| 24 | Knowledge & Eval | <-> 25 | A person already knows and understands the topic thoroughly. |
| 25 | Knowledge & Eval | <-> 24 | A person learns and gains new knowledge through active study. |
| 26 | Knowledge & Eval | <-> 27 | A person believes but cannot verify the claim about the situation. |
| 27 | Knowledge & Eval | <-> 26 | A person fully grasps and comprehends the precise meaning. |
| 28 | Knowledge & Eval | <-> 29 | A researcher discovers and uncovers a previously hidden fact. |
| 29 | Knowledge & Eval | <-> 28 | An investigator analyzes and probes the root cause of an issue. |
| 30 | Knowledge & Eval | <-> 31 | A validator confirms and proves the accuracy of a claim. |
| 31 | Knowledge & Eval | <-> 30 | A skeptic challenges and questions the validity of an assumption. |
| 32 | Knowledge & Eval | <-> 33 | The outcome produces positive benefits for the stakeholders involved. |
| 33 | Knowledge & Eval | <-> 32 | The outcome causes negative harm to the participants affected. |
| 34 | Knowledge & Eval | <-> 35 | The matter carries high importance and significance for the organization. |
| 35 | Knowledge & Eval | <-> 34 | The matter has zero relevance and no bearing on the decision. |
| 36 | Possession & Struct | <-> 38 | An owner holds and controls a valuable resource exclusively. |
| 37 | Possession & Struct | <-> 39 | A giver transfers complete ownership to a receiver permanently. |
| 38 | Possession & Struct | <-> 36 | An entity is missing and lacks a required component entirely. |
| 39 | Possession & Struct | <-> 37 | Multiple parties share and jointly use a common resource. |
| 40 | Possession & Struct | <-> 41 | A container encloses and holds its contents inside the boundary. |
| 41 | Possession & Struct | <-> 40 | A component connects as one part within a larger assembled system. |
| 42 | Possession & Struct | <-> 43 | A category groups and includes multiple distinct member items. |
| 43 | Possession & Struct | <-> 42 | An individual instance belongs as a member of a general category. |
| 44 | Possession & Struct | <-> 45 | Two entities are linked together by a direct relationship. |
| 45 | Possession & Struct | <-> 44 | Two entities remain separate with no connection between them. |
| 46 | Possession & Struct | <-> 47 | Two entities are alike and share substantial similarity. |
| 47 | Possession & Struct | <-> 46 | Two entities are unlike and have significant differences. |
| 48 | Time & Sequence | <-> 49 | Event A happens first and precedes event B chronologically. |
| 49 | Time & Sequence | <-> 48 | Event B happens after and follows event A chronologically. |
| 50 | Time & Sequence | <-> 51 | Both events occur together at exactly the same simultaneous moment. |
| 51 | Time & Sequence | <-> 50 | The same activity repeats multiple times in a recurring cycle. |
| 52 | Time & Sequence | <-> 53 | The long process continues slowly over an extended time period. |
| 53 | Time & Sequence | <-> 52 | The quick action finishes rapidly within a brief time interval. |
| 54 | Time & Sequence | <-> 55 | The stable condition persists unchanged throughout the entire period. |
| 55 | Time & Sequence | <-> 54 | The unstable state shifts abruptly and changes without warning. |
| 56 | Time & Sequence | <-> 57 | The common event happens frequently on a regular recurring basis. |
| 57 | Time & Sequence | <-> 56 | The rare event happens infrequently only under special circumstances. |
| 58 | Time & Sequence | <-> 59 | The impossible event never happens under any conceivable conditions. |
| 59 | Time & Sequence | <-> 58 | The guaranteed event always happens without any possible exception. |
| 60 | Quantity & Compare | <-> 61 | The measured amount grows and increases substantially over time. |
| 61 | Quantity & Compare | <-> 60 | The measured amount shrinks and decreases substantially over time. |
| 62 | Quantity & Compare | <-> 63 | The available quantity is enough and sufficient for the requirement. |
| 63 | Quantity & Compare | <-> 62 | The available quantity is too little and insufficient for the demand. |
| 64 | Quantity & Compare | <-> 65 | Option A ranks higher and is superior to option B in quality. |
| 65 | Quantity & Compare | <-> 64 | Option A ranks lower and is inferior to option B in quality. |
| 66 | Quantity & Compare | — | Both options are equal and equivalent in their overall value. |
| 67 | Quantity & Compare | <-> 68 | The measurement is above and exceeds the maximum threshold limit. |
| 68 | Quantity & Compare | <-> 67 | The measurement is below and fails to meet the minimum standard. |
| 69 | Modality (0.4×) | <-> 70 | The action is mandatory and required by the binding regulations. |
| 70 | Modality (0.4×) | <-> 69 | The action is forbidden and prohibited by the enforced policy. |
| 71 | Modality (0.4×) | — | The action is optional and permitted under the flexible guidelines. |
| 72 | Modality (0.4×) | <-> 73 | The outcome is achievable and possible given favorable circumstances. |
| 73 | Modality (0.4×) | <-> 72 | The outcome is unachievable and impossible under current constraints. |
| 74 | Modality (0.4×) | — | The outcome is probable and likely based on the available evidence. |

---

## Appendix B: Complete Experimental Results

### B.1 Run 1 — Sequential Ingestion (2026-04-16)

**Locality Decay:**
- Timestamp: 2026-04-16T18:41:08Z
- Voxels: 470, Pairs: 110,215
- Slope: −0.002364, $r$: −0.1585, Intercept: 0.7202
- $p$: < $10^{-4}$, Null slope mean: −2.80 × $10^{-6}$, Null slope std: 2.04 × $10^{-4}$
- Verdict: PASS

**Jaccard/Salton Locality:**
- Timestamp: 2026-04-16T19:30:38Z
- Voxels: 643, Pairs: 207,046
- PRECEDES edges: 638 (1 intra-voxel, 637 cross-voxel)
- Slope: −3.20 × $10^{-5}$, $r$: −0.0204, Mean Salton: 4.85 × $10^{-4}$
- $p$: < $10^{-4}$
- Verdict: PASS

### B.2 Run 2 — Randomized Ingestion (2026-04-17)

**Voxel Snapshot:**
- Timestamp: 2026-04-17T17:13:07Z
- Total voxels occupied: 674
- Total sentences assigned: 3,378 (0 unassigned)
- Paraphrase groups: 506, Paraphrase mass: 539
- Paraphrase mass / assigned sentences: 15.96%
- Mean paraphrase mass per group: 1.07

**Locality Decay:**
- Timestamp: 2026-04-17T18:09:04Z
- Voxels: 445, Pairs: 98,790
- Slope: −0.002807, $r$: −0.1833, Intercept: 0.7196
- $p$: < $10^{-4}$, Null slope mean: −1.45 × $10^{-5}$, Null slope std: 2.77 × $10^{-4}$
- Verdict: PASS

**Jaccard/Salton Locality:**
- Timestamp: 2026-04-17T18:10:05Z
- Voxels: 674, Pairs: 227,475
- PRECEDES edges: 638 (1 intra-voxel, 637 cross-voxel)
- Slope: −3.06 × $10^{-5}$, $r$: −0.0189, Mean Salton: 4.78 × $10^{-4}$
- $p$: < $10^{-4}$
- Verdict: PASS

**Guide-Star Verification:**
- Timestamp: 2026-04-17T18:10:57Z
- Guide stars: 75, Unique voxels: 674
- Embeddings: 3,378, Micro-PSA assignments: 3,378 (100%)
- Standard candle (L1_v_10_9_11): 9 sentences
- Verdict: PASS

**Falsification (Scramble):**
- Timestamp: 2026-04-17T19:52:32Z
- Voxels: 445, Pairs: 98,790
- Real slope: −0.002807
- Scrambled slope mean: −0.000014, Scrambled slope std: 0.000277
- $z$-score: 10.1 sigma
- $p$: < 0.001 (empirical), about 7.6 × $10^{-24}$ (Gaussian)
- Permutations: 1,000
- Verdict: PASS

**Canonical Normalized Locality Re-run:**
- Timestamp: 2026-05-05T19:03:47Z
- Voxels: 49, Pairs: 1,176
- Slope: −0.074815, $r$: −0.2875, Intercept: 0.8103
- $p$: 0.010, Null slope mean: 0.001062, Null slope std: 0.02755
- Same-layer locality: L1 $r=-0.5564$, L2 $r=-0.7275$, L3 $r=-0.6616$
- Verdict: PASS

**Centroid Embedding Baseline:**
- Timestamp: 2026-05-05T19:10:03Z
- Embedding dimensions fetched per centroid: 200
- Voxels: 49, Pairs: 1,176
- Cross-space correlation: 0.2875
- Mean cosine similarity: 0.7535
- Same-layer locality: L1 $r=-0.5564$, L2 $r=-0.7275$, L3 $r=-0.6616$

**Sentence-Level Full-Embedding Baseline:**
- Timestamp: 2026-05-05T19:10:31Z
- Embedding dimensions: 4,096
- Eligible sentences: 3,076
- Sample pairs requested: 400; Pairs analyzed: 400
- Locality correlation: −0.1715
- Slope: −0.045591, Intercept: 0.3757
- Mean cosine similarity: 0.3554
- L3 same-layer locality: $r=-0.1776$, slope: −0.1121
- Method note: cosine similarities computed inside Neo4j over full embeddings; only scalar similarities returned through SSM.

### B.3 Run 3 — Expanded 115-Star Frame (2026-05-07; full-dimensional and random-anchor controls added 2026-05-18)

*The 2026-05-07 evaluation below used the 200-dimensional SSM-transported centroid path. A follow-up full-4,096-dimensional voxel-centroid run on 2026-05-18 (reported at the end of this appendix and discussed in §4.6) strengthens the expanded-frame locality to $r=-0.3259$, slope $=-0.1275$ over the same 59-voxel / 1,711-pair geometry.*

**Voxel Snapshot:**
- Timestamp: 2026-05-07T15:54:51Z
- Total voxels occupied: 68
- Total sentences assigned: 3,071 (0 unassigned)
- Paraphrase groups: 421, Paraphrase mass: 689
- Paraphrase mass / assigned sentences: 22.44%
- Mean paraphrase mass per group: 1.64

**Guide-Star Verification:**
- Timestamp: 2026-05-07T15:55:37Z
- Guide stars: 115, Unique voxels: 68
- Embeddings: 3,071, Micro-PSA assignments: 3,071 (100%)
- Previous standard candle (`L1_v_10_9_11`): 0 sentences
- Interpretation: expected frame shift under expanded guide-star basis; a new standard candle is required for future 115-star repeatability tests.

**Locality Decay:**
- Timestamp: 2026-05-07T16:03:08Z
- Voxel distance mode: canonical_l1_normalized
- Voxels: 58, Pairs: 1,653
- Slope: −0.114788, $r$: −0.3118, Intercept: 0.7886
- $p$: 0.001, Null slope mean: 0.000318, Null slope std: 0.03284
- Same-layer locality: L1 $r=-0.5195$, L2 $r=-0.7833$, L3 $r=-0.7260$
- Verdict: PASS

**Centroid Embedding Baseline:**
- Timestamp: 2026-05-07T16:10:33Z
- Embedding dimensions fetched per centroid: 200
- Voxels: 58, Pairs: 1,653
- Cross-space correlation: 0.3118
- Mean cosine similarity: 0.7027
- Same-layer locality: L1 $r=-0.5195$, L2 $r=-0.7833$, L3 $r=-0.7260$

**Sentence-Level Full-Embedding Baseline:**
- Timestamp: 2026-05-07T16:10:54Z
- Embedding dimensions: 4,096
- Eligible sentences: 3,071
- Sample pairs requested: 500; Pairs analyzed: 418
- Locality correlation: −0.1880
- Slope: −0.054432, Intercept: 0.3791
- Mean cosine similarity: 0.3569
- L3 same-layer locality: $r=-0.2209$, slope: −0.1283
- Method note: cosine similarities computed inside Neo4j over full embeddings; only scalar similarities returned through SSM.

**Bandit Regime Map:**
- Timestamp: 2026-05-07T16:29:54Z
- Voxels mapped: 68
- Bandit training steps: 0
- Arm distribution: `default` = 68 voxels (100%)
- Interpretation: bandit state is present in Redis but untrained; combined behavioral metrics therefore remain dominated by graph-edge locality rather than learned per-voxel scoring regimes.

**Topic Sanity:**
- Timestamp: 2026-05-07T16:37:45Z
- Redis hot voxels: 44
- Named voxels: 43
- Voxels with parseable coordinates: 44
- Standard candle: old `L1_v_10_9_11` not found.

**Jaccard/Salton Locality:**
- Timestamp: 2026-05-07T16:38:14Z
- Voxels: 69, Pairs: 2,415
- PRECEDES edges: 3,145 (423 intra-voxel, 2,722 cross-voxel)
- Nonzero pairs: 433
- Slope: −0.011379, $r$: −0.1063, Mean Salton: 0.01329
- $p$: 0.022, Null slope mean: −0.002581
- Combined behavioral metric: $r=-0.1090$, slope: −0.005895, pairs: 2,346
- Verdict: PASS

**Falsification (Scramble):**
- Timestamp: 2026-05-07T16:45:55Z
- Voxels: 59, Pairs: 1,711
- Real slope: −0.118659
- Scrambled slope mean: 0.000536, Scrambled slope std: 0.032100
- $z$-score: 3.71 sigma
- $p$: < 0.001 empirical (0 of 1,000 permutations)
- Permutations: 1,000
- Verdict: PASS

**Full-Dimensional Voxel-Centroid and Random-Anchor Controls:**
- Timestamp: 2026-05-18T15:30:26Z
- Eligible embedded sentences: 3,304
- Full 4,096D voxel-centroid baseline: 59 voxels, 1,711 pairs
- Full 4,096D locality: $r=-0.3259$, slope: −0.127451, intercept: 0.7721
- Occupancy diagnostics: effective occupied voxels = 18.09, max voxel fraction = 21.9%
- 200D truncation validation: Pearson $r=0.8804$, Spearman $\rho=0.8658$, mean absolute cosine delta = 0.0561
- Random sentence anchors ($G=75$, 50 trials): mean occupied voxels = 9.38, median effective occupied voxels = 6.32, median slope = −0.2857
- Random sentence anchors ($G=115$, 50 trials): mean occupied voxels = 8.54, median effective occupied voxels = 6.63, median slope = −0.1864
- Random unit-vector anchors ($G=115$, 50 trials): mean occupied voxels = 24.4, median effective occupied voxels = 8.25, median slope = −0.1231
- Interpretation: random anchors can produce locality-like slopes, but with substantial coordinate-field collapse relative to the real PSA frame.

---

## Appendix C: Reproducibility Checklist

| Requirement | Status |
|------------|--------|
| Code availability | Will be released |
| Data availability | Will be released |
| Deterministic seeds | random_state=42 for MDS, seed=42 for scramble |
| Embedding model specified | Qwen3 Embedding 8B |
| Hardware specified | AWS EC2 (Neo4j), ECS Fargate (services) |
| Hyperparameters listed | grid_size=0.10, EMBED_DIMS=200 for transport-limited runs, n_permutations=1,000, random-anchor trials=50 |
| Multiple runs | 3 main runs (sequential, randomized, expanded 115-star frame) plus baseline/control re-runs |
| Null model defined | Permutation scramble; random sentence-anchor and random unit-vector anchor controls |
| Effect size reported | Empirical $p < 0.001$ (0/1,000 permutations) in both frames; Gaussian-extrapolated $z \approx 10$ (75-star) and $z \approx 3.7$ (115-star), reported with caveats in §6.4 |
| Confidence intervals | Null distribution std reported |
