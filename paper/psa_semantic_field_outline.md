cd research/papers/latex/
git clone https://github.com/acl-org/acl-style-files
cp acl-style-files/latex/acl.sty acl-style-files/latex/acl_natbib.bst .
pdflatex psa_semantic_field && bibtex psa_semantic_field && pdflatex psa_semantic_field && pdflatex psa_semantic_field


# PSA as a Semantic Field: Guide Stars, Voxelization, and the Geometry of Meaning

## Paper Outline — Draft 2

---

## Abstract (target: 250 words)

We introduce Positional Semantic Addressing (PSA), a method for imposing stable coordinate geometry on high-dimensional embedding space using fixed reference concepts ("guide stars"). By computing cosine differences to a small set of semantically anchored points, we project arbitrary text into a low-dimensional field with interpretable axes. We discretize this field into voxels at multiple resolutions, producing a hierarchical addressing scheme (L1 macro → micro cells) that captures semantic locality. We define **Semantic Energy Density (SED)**, a composite scalar field over the voxel grid that integrates occupancy, cross-scale consistency, source diversity, contradiction pressure, temporal stability, and guide star anchoring into a single measurable quantity per voxel. We show that:
1. Nearby voxels share measurable statistical properties (scoring regime similarity, topic coherence, temporal decay profiles)
2. Voxel boundaries correspond to real regime transitions in downstream task performance
3. The coordinate system is stable across corpora and embedding models
4. SED reveals the morphological structure of meaning space — attractor cores, filaments, voids, and boundary turbulence — verified by corpus scramble falsification
5. The SED distribution exhibits power-law scaling and persistent topological features across corpora

We release the addressing scheme, SED computation framework, evaluation benchmarks, and the first morphologically characterized map of semantic space.

---

## 1. Introduction

### 1.1 The Problem
- Embedding spaces are powerful but uninterpretable — no addresses, no neighborhoods, no landmarks
- Clustering gives groups but not coordinates; UMAP gives pictures but not stable positions
- No existing method provides a *reusable coordinate system* across documents, models, or time

### 1.2 The Insight
- A small set of fixed reference concepts (guide stars) can impose stable geometry on any embedding space
- Cosine differences to guide stars produce low-dimensional coordinates that are interpretable, stable, and hierarchically decomposable
- The resulting space is not a reduction — it is a *field* with measurable local properties

### 1.3 Contributions
1. Guide star selection methodology and stability analysis
2. PSA coordinate computation and voxelization algorithm
3. Empirical demonstration of semantic locality (nearby voxels → similar behavior)
4. Semantic regime detection at voxel boundaries
5. **Semantic Energy Density (SED)** — a composite scalar field replacing simple depth, with six independently measurable components
6. **Corpus scramble falsification** — a null-model test proving observed structure is signal, not artifact
7. **Morphological characterization** — power-law distributions, persistent homology, filamentarity metrics
8. Structural predictions: attractor cores, filaments, voids, boundary turbulence — the information telescope
9. System framing: PSA Semantic Field Architecture — unifying representation (PSA field), control (semantic switches), scoring (SED), and adaptive learning into a single field-based system

---

## 2. Related Work

### 2.1 Embedding Space Structure
- Word2Vec analogies, BERT probing, embedding space geometry literature
- Isotropy studies, anisotropy corrections
- Distinction: these study *properties* of the space; we impose *coordinates* on it

### 2.2 Dimensionality Reduction
- PCA, t-SNE, UMAP — visualization tools, not addressing systems
- Key difference: PSA coordinates are stable across runs, corpora, and (partially) across models
- UMAP gives you a picture; PSA gives you a postal code

### 2.3 Semantic Similarity and Clustering
- K-means, HDBSCAN, topic models
- These produce *groups*; PSA produces *locations*
- Groups are assignments; locations support interpolation, neighborhoods, gradients

### 2.4 Reference-Based Representations
- Conceptual spaces (Gärdenfors), semantic differential (Osgood)
- Landmark-based navigation in cognitive science
- PSA as computational realization of cognitive reference-point reasoning

---

## 3. Method

### 3.1 Guide Star Selection
- Requirements: semantic stability, broad coverage, low mutual redundancy
- Selection algorithm: seed from ontology anchors, validate via cross-corpus cosine stability
- Typical set size: 12–20 guide stars spanning major conceptual axes
- Stability criterion: guide star positions shift < ε across corpora

### 3.2 PSA Coordinate Computation
- For any text fragment t with embedding e(t):
  - Compute cosine similarity to each guide star: sim(e(t), e(g_i))
  - PSA coordinate = vector of cosine *differences* between guide star pairs (not raw similarities)
  - Dimensionality: with k guide stars, select top-3 principal difference axes → 3D coordinate (x, y, z)
- Why differences, not similarities: differences capture *relative position*, similarities capture *absolute proximity*
- The coordinate is stable because guide stars are fixed anchors

### 3.3 Voxelization
- Discretize the continuous 3D PSA space into a grid (default: 20 × 20 × 20 = 8,000 macro voxels)
- Address format: `L1_v_X_Y_Z` (macro level)
- Recursive subdivision within occupied voxels: `L1_v_X_Y_Z.micro_cell` → `L1_v_X_Y_Z.micro.micro`
- Each level increases resolution by factor of ~10
- Addressing is hierarchical: coarse queries hit L1, fine queries drill into micro cells

### 3.4 Voxel Properties
- Each voxel accumulates statistics from its occupants:
  - Population count (how many text fragments land here)
  - Intra-voxel coherence (mean pairwise similarity of occupants)
  - Boundary gradients (coherence change rate at voxel edges)
  - Temporal profile (age distribution of occupants)
  - Source diversity (how many distinct documents contribute)

These voxel-level properties enable localized decision-making primitives, referred to as **semantic switches**. A semantic switch operates on field-local statistics (e.g., SED components, coherence, boundary gradients) to select routing, scoring, or update behaviors. While not the focus of this paper, this establishes PSA not only as a representational system but as a substrate for spatially-aware control.

---

## 4. Semantic Locality

### 4.1 Hypothesis
- If PSA coordinates capture real semantic structure, then nearby voxels should exhibit similar downstream behavior
- "Nearby" = Manhattan distance ≤ 2 in the voxel grid

### 4.2 Experimental Design
- Task: scoring parameter sensitivity across voxels
  - For each occupied voxel, measure optimal scoring weights (centrality, temporal decay, topic coherence, etc.)
  - Compare optimal weights between neighboring vs. distant voxels
- Dataset: [enterprise corpus or MRPC + extended evaluation set]
- Metric: weight vector cosine similarity as a function of voxel distance

### 4.3 Expected Results
- Scoring weight similarity decays with voxel distance (locality holds)
- Specific transitions are sharp (regime boundaries, not smooth gradients)
- Some axes show stronger locality than others (interpretable axis semantics)

### 4.4 The Scramble Test (Falsification Protocol)

**This is the spine of the paper. Every structural claim must survive it.**

- **Null model**: scramble the corpus — randomly reassign text fragments to voxels while preserving per-voxel population counts
- **Procedure**:
  1. Compute full PSA field (coordinates, voxelization, SED, morphology) on the real corpus
  2. Generate N scrambled corpora (N ≥ 100): same text, same voxel grid, random reassignment of fragments to voxels (preserving voxel population sizes)
  3. Recompute all field measurements on each scrambled corpus
  4. Compare real-corpus measurements against the scrambled null distribution
- **What must survive**:
  - Locality decay curve must be absent in scrambled data (flat scoring similarity vs. distance)
  - Regime boundaries must vanish (no gradient structure at voxel faces)
  - SED morphology must collapse (no filaments, no attractor cores — just noise consistent with random packing)
  - Power-law scaling must disappear (scrambled SED distribution → Gaussian or uniform)
  - Persistent homology features must reduce to baseline (random point cloud topology)
- **What we're really testing**: whether PSA coordinates capture meaningful structure, or whether any 3D projection of embeddings plus voxelization produces structure-like artifacts
- **The force-directed layout trap**: many graph visualizations look cosmic because of layout algorithms. The scramble test kills this. If structure is real, it exists in the *data*, not the *rendering*
- **Reporting**: every figure in the paper that shows structure should have a scrambled-corpus control panel alongside it

**Visualization requirement**: Every primary visualization (locality curves, regime maps, SED thermal maps, morphological plots) must include a side-by-side scrambled control using identical rendering parameters. Visual similarity between real and scrambled outputs falsifies structural claims immediately; visual divergence is necessary but not sufficient, and must be supported by the quantitative metrics defined in Section 8.

---

## 5. Regime Detection at Voxel Boundaries

### 5.1 The Claim
- Not all voxel boundaries are equal — some separate genuinely different scoring regimes
- These regime boundaries are detectable from the data and correspond to meaningful semantic transitions

### 5.2 Method
- Compute scoring performance gradient across all adjacent voxel pairs
- Threshold gradient magnitude to identify boundary voxels
- Characterize regime types: temporal (decay rate changes), structural (centrality weight shifts), trust (source weight changes)

### 5.3 Analysis
- How many distinct regimes exist? (hypothesis: 15–30 in a mature corpus)
- Do regime boundaries align with human-recognizable category transitions?
- Are regimes stable as corpus grows?

---

## 6. Semantic Energy Density (SED)

### 6.1 From Depth to a Scalar Field

- Voxel depth (recursive subdivision levels) was our initial density proxy — but depth is an *indexing artifact*, not a physical quantity
- What we actually want: a composite scalar field over semantic space that captures *how much meaningful structure exists* at each location
- **SED** is that field. It is not one thing. It is a defined, measurable composite of six independently computable components

### 6.2 SED Components

For each occupied voxel v, define:

1. **Occupancy** O(v): normalized population count
   - Measure: |fragments in v| / max(|fragments|) across all voxels
   - What it captures: raw accumulation of meaning

2. **Cross-scale consistency** C(v): does structure persist across resolutions?
   - Measure: correlation of sub-voxel population distribution between L1 → micro → micro.micro levels
   - What it captures: genuine structural complexity vs. uniform mass
   - A voxel with 1,000 items spread uniformly across micro-cells scores lower than one with differentiated sub-structure
   - This is what depth *should have been* — not "how many levels exist" but "does the structure replicate across scales"

3. **Source diversity** D(v): independent convergence vs. echo
   - Measure: Shannon entropy over contributing documents/sources
   - What it captures: whether meaning at this location comes from independent origins (high D = consensus) or is echoed from one source (low D = amplification)

4. **Contradiction pressure** P(v): tension in the region
   - Measure: intra-voxel cosine variance + explicit disagreement score from paraphrase/contradiction detection
   - What it captures: regions of active debate, competing definitions, or unresolved ambiguity
   - High P = the meanings here don't agree with each other, even though they're spatially close

5. **Temporal stability** T(v): flicker vs. persistence
   - Measure: occupancy autocorrelation across corpus snapshots over time
   - What it captures: stable knowledge (high T) vs. transient mentions (low T)
   - A voxel that appears, disappears, and reappears is qualitatively different from one that's always present

6. **Guide star anchoring** A(v): alignment to stable semantics
   - Measure: mean cosine similarity of occupants to their nearest guide star
   - What it captures: how tightly this region is held by the coordinate system
   - Low A = peripheral meaning, weakly anchored, potentially drifting
   - High A = core meaning, tightly tied to the reference frame

### 6.3 SED Computation

$$\text{SED}(v) = w_O \cdot O(v) + w_C \cdot C(v) + w_D \cdot D(v) + w_P \cdot P(v) + w_T \cdot T(v) + w_A \cdot A(v)$$

- Default: equal weighting (each component normalized to [0, 1])
- Alternative: learned weights from downstream task performance (connects to paper 3 — bandits could learn optimal SED weighting per region)
- **Key property**: each component is independently measurable, independently falsifiable, and independently meaningful
- The composite is richer than any single component, but a reader can decompose any SED value back into its constituents

### 6.4 SED as a Diagnostic Decomposition

The power of SED is not the scalar — it's the *decomposition*. Any location in semantic space can be characterized by its SED profile:

| Profile | O | C | D | P | T | A | Interpretation |
|---|---|---|---|---|---|---|---|
| Attractor core | High | High | High | Low | High | High | Established, consensus knowledge |
| Active frontier | Med | Med | Med | **High** | Low | Med | Emerging, contested, evolving |
| Echo chamber | High | Low | **Low** | Low | High | Med | Amplified but not independently verified |
| Ephemeral spike | Med | Low | Med | Low | **Low** | Low | Trending but transient |
| Deep niche | Low | **High** | Low | Low | High | High | Specialized, stable, small community |
| Void edge | Low | Low | Low | Low | Low | Low | The boundary of the known |

### 6.5 The Thermal Map (Revised)

- Render SED as a volumetric field: black (SED ≈ 0, empty) → cool blues (sparse) → warm oranges (moderate density) → bright white (maximum SED)
- But the thermal map is now richer: overlay the *decomposition* as a color channel mapping:
  - Red channel: contradiction pressure (P) — hot red = contested
  - Green channel: temporal stability (T) — green = persistent
  - Blue channel: source diversity (D) — blue = independently converged
- The resulting false-color map tells you not just *where* meaning is dense, but *what kind* of density it is

---

## 7. Structural Predictions

### 7.1 Attractor Cores (Semantic Mass Concentrations)

- **Prediction**: Extremely bright SED regions that are stable across scales, with high paraphrase collapse (many ways of saying the same thing converge here)
- **Expected examples**: basic physics concepts, fundamental human ideas (justice, causality, time), core ontology nodes
- **Signature**: High O, High C, High D, Low P, High T, High A
- **Test**: identify the top-k SED voxels; are they recognizable as foundational concepts? Do they remain top-k across corpora?

### 7.2 Filaments (The Gold)

- **Prediction**: Thin, extended structures connecting distant attractor cores
- Low density but high *importance* — these are the bridges between domains
- **What filaments are**:
  - Analogies (this concept in domain A maps to that concept in domain B)
  - Interdisciplinary bridges (where biology meets computation, where law meets ethics)
  - Transfer pathways (how knowledge propagates from one field to another)
- **Signature**: Moderate O, High C (structure persists at fine scale), variable D, Low-to-Med P
- **This is where discovery lives**: filaments are the connections that exist but aren't obvious. Finding them computationally is finding the hidden structure of interdisciplinary knowledge
- **Test**: extract voxel chains with moderate SED connecting high-SED cores; do they correspond to known interdisciplinary connections?

### 7.3 Voids

- **Prediction**: Large connected regions of near-zero SED
- **Three types of void**:
  1. **True voids**: regions with guide star coverage but zero occupancy. These are *predictable gaps* — places where meaning *could* exist but doesn't. The PSA field defines the space; voids are the unfilled part
  2. **Fragmentation voids**: sparse, disconnected fragments that don't integrate. Language exists but ideas haven't cohered. Niche or under-explored domains
  3. **Structural voids**: regions between attractor cores where the semantic gradient is too steep for anything to settle. Not empty by accident — empty because the conceptual distance is real
- **Test**: are voids stable across corpora? (True voids should be; fragmentation voids should fill as corpus grows)

### 7.4 Boundary Turbulence

- **Prediction**: Regions of high contradiction pressure, high update rate, unstable topology
- **What boundary turbulence indicates**:
  - Controversial topics (political, ethical, methodological disagreement)
  - Emerging knowledge (new discoveries that haven't settled into consensus)
  - Breaking paradigms (old attractor cores fragmenting, new ones forming)
- **Signature**: Med O, Low C (structure doesn't replicate across scales), variable D, **High P**, **Low T**
- **Test**: identify high-P, low-T voxels; do they correspond to known areas of active debate?

### 7.5 Semantic Background Radiation

- **Speculative prediction** (clearly labeled): beneath the attractor/filament/void structure, there may be a low-frequency *background field* — a baseline anisotropy across the entire semantic space
- **What it might reflect**:
  - Universal linguistic priors (structural biases of language itself)
  - Shared human experience (embodiment, temporal perception, social structure)
  - Ontology backbone (time, object, causality, agency — concepts so fundamental they're below the resolution of normal analysis)
- **How to detect**: after subtracting attractor/filament/void structure, measure residual SED anisotropy. Is it isotropic (no background) or does it have structure?
- **Framing**: this is speculative and clearly labeled as such. If found, it would indicate that semantic space has intrinsic geometry beyond what any single corpus contributes. If not found, the field is entirely corpus-dependent, which is still useful but less fundamental
- **Connection to CMB**: not literal, but structurally analogous. The CMB is the signal that remains after subtracting foreground sources. Semantic background radiation would be the signal that remains after subtracting the content

---

## 8. Morphological Characterization

### 8.1 The Requirement

- Pretty pictures are not evidence. The semantic cosmic web is only real if it survives quantitative morphological analysis
- Every measurement below must be compared against the scrambled null model (section 4.4)

### 8.2 Power-Law Distributions

- **Test**: plot SED distribution across all occupied voxels on log-log axes
- **If real**: should exhibit power-law scaling (many low-SED voxels, few high-SED attractor cores)
- **If artifact**: distribution will be Gaussian, uniform, or log-normal (driven by population statistics)
- **Significance**: power-law SED would mean *the same organizing principle operates at all scales* — the hallmark of a self-organizing system, not a database
- **Method**: Clauset-Shalizi-Newman power-law fitting with goodness-of-fit test against alternative distributions

### 8.3 Fractal Scaling

- **Test**: box-counting dimension of the occupied voxel set at increasing resolution
- **If real**: fractal dimension D_f should be non-integer and consistent across analysis scales
- **If artifact**: D_f will be ~3 (space-filling) or ~0 (sparse points) depending on threshold
- **What it means**: non-integer D_f indicates that meaning fills space in a structured, self-similar way — not uniformly, not randomly

### 8.4 Persistent Homology

- **Test**: compute persistence diagrams (Betti numbers β₀, β₁, β₂) of the SED field at varying density thresholds
- **β₀** (connected components): how many distinct attractor cores? How do they merge as threshold drops?
- **β₁** (loops/tunnels): are there enclosed void regions? Topological holes in the knowledge structure?
- **β₂** (cavities): 3D enclosed volumes — trapped empty space within the semantic web?
- **Persistence**: long-lived features (high birth-death difference) are real structure; short-lived features are noise
- **Comparison**: real corpus persistence diagram vs. scrambled corpus persistence diagram. The real one should have significantly more persistent features
- **Tools**: Ripser, GUDHI, or similar computational topology library

### 8.5 Filamentarity Metrics

- **Test**: apply cosmic web classification algorithms (T-web, NEXUS+, DisPerSE) adapted for voxel grids
- **Measure**: filamentarity index F = fraction of occupied voxels classified as filament (vs. node, wall, void)
- **Expected**: F significantly above scrambled baseline
- **Bonus measurement**: filament length distribution — if power-law, suggests scale-free bridging structure

### 8.6 Cross-Corpus Topology

- **The strongest test**: do the morphological features (power law exponent, fractal dimension, persistence diagram, filamentarity) remain stable across *different* corpora?
- If yes → these are properties of how meaning organizes, not properties of any particular corpus
- If no → useful but corpus-specific (still publishable, but a weaker claim)
- **Universal features** would be the paper's most significant finding: evidence that semantic space has intrinsic structure independent of content

---

## 9. Stability Analysis

### 9.1 Cross-Corpus Stability
- Same guide stars, different corpora → how much do voxel populations shift?
- Measure: voxel occupation correlation between corpora
- **New**: SED component stability — do the same voxels have similar SED profiles across corpora?

### 9.2 Cross-Model Stability
- Same guide stars, different embedding models (BGE-M3, MXBai, Nomic) → coordinate drift
- Measure: rank correlation of voxel assignments across models
- **New**: does morphological structure (power law exponent, filamentarity) survive model changes?

### 9.3 Temporal Stability
- Same corpus growing over time → do voxel assignments stay consistent?
- Critical for the online learning story (paper 3): bandits need stable addresses to accumulate meaningful statistics
- **New**: SED component trajectories — how do individual voxels evolve their profiles over time? Do attractor cores strengthen monotonically?

---

## 10. First Light

### 10.1 The Moment
- First light is not a metric. It's the moment the instrument resolves real structure instead of noise
- For PSA: the first time the SED field stops looking like data and starts looking like physics
- In practice: the rendering where a reviewer looks at it and says "that's not random"

This framing aligns with the concept of a **Semantic Observatory**: an instrument that measures structure rather than generating content. "First light" marks the transition from data visualization to field observation. The stronger claim — PSA as an "Information Telescope" — is contingent on whether the observed morphology survives quantitative analysis and falsification.

### 10.2 Scale Thresholds
- At small corpus scale (< 1K documents): mostly dark, a few hot spots — you see the brightest objects first
- At enterprise scale (10K–1M documents): structure resolves — cores, filaments, voids, boundaries become distinguishable
- At internet scale: *unknown* — this is the open question the paper poses
  - Hypothesis: a knowledge topology with a finite number of dense attractors connected by sparser bridges, with vast empty regions
  - The voids are the most interesting — they are where meaning *could* exist but doesn't yet

### 10.3 What We Don't Know
- Minimum corpus size for first light (where does structure emerge from noise?)
- Whether different corpora produce the same morphology at the same scale
- Whether temporal evolution of the field is predictable
- What the boundaries between dense regions look like at high resolution
- Whether the semantic background radiation is real or a null result

---

## 11. Discussion

### 11.1 What PSA Is
- A coordinate system for meaning, not a clustering algorithm
- A field with measurable local properties, not a dimensionality reduction
- A foundation for spatially-aware downstream systems (scoring, verification, learning)
- **New**: an observational instrument. The paper's central claim is not "we built something useful" but "we can now observe how meaning organizes"
- The representational layer of a broader system: the PSA Semantic Field Architecture, which integrates field geometry, local decision primitives (semantic switches), scoring, and adaptive control

### 11.1.1 System Naming and Layering

We use the following terminology to distinguish system layers and avoid conceptual overload:

- **PSA Semantic Field**: the representational layer — a coordinate system over embedding space with measurable local properties
- **Semantic Energy Density (SED)**: the scalar field over the PSA grid — the "physics" of the field
- **Semantic Switches**: local decision primitives that operate within the field (e.g., routing, scoring selection, update strategy)
- **PSA Semantic Field Architecture**: the full system — integrating representation, scoring, control, and adaptation
- **Selective AI**: the product behavior — responses selected from known, verifiable regions of the field rather than generated unconstrained
- **Semantic Observatory**: the research framing — the system as an instrument for measuring the structure of meaning

The term **Information Telescope** is reserved for a potential emergent capability: if the field's morphological structure (attractors, filaments, voids) is validated under quantitative measurement, PSA may function as an observational instrument revealing the topology of human knowledge.

### 11.2 What PSA Enables
- Regime-aware scoring (different weights per voxel → paper 3)
- Spatially-aware verification (trust thresholds adapted to semantic location → paper 2)
- SED mapping and decomposition (the information telescope)
- Interpretable embedding space navigation (you can *describe* a location by its guide star profile and SED decomposition)
- **Void detection**: identifying what an organization *doesn't* know, not just what it does

### 11.3 The Two Outcomes (Stated Directly)

This paper has two possible outcomes, and we state both:

1. **Structure collapses under measurement** (scramble test fails to distinguish real from null, morphological metrics are indistinguishable from random). Result: PSA is still a useful coordinate system for engineering purposes (addressing, regime detection, bandit adaptation), but the semantic field is not a natural object. The paper is a systems contribution.

2. **Structure holds under measurement** (scramble test passes, morphological signatures are significant, cross-corpus universals exist). Result: this is a new class of observability — the first empirical evidence that meaning has intrinsic spatial structure at scale. The paper is a scientific contribution.

There is no middle. We report whichever outcome occurs.

### 11.4 Limitations
- Guide star selection is currently semi-manual (ontology-seeded)
- 3D projection loses information (acceptable if locality holds; needs characterization)
- Grid resolution is a hyperparameter (20³ macro is reasonable but not optimized)
- SED components require sufficient corpus scale and temporal depth to compute
- The thermal map visualization choices affect perception — all structure claims must rest on the quantitative morphology, not the rendering
- Contradiction pressure (P) requires paraphrase/contradiction detection, which has its own error rate

### 11.5 Open Questions
- Can guide stars be learned end-to-end?
- Is there an optimal dimensionality beyond 3?
- Does the SED distribution have universal properties across unrelated corpora?
- What is the minimum corpus size for first light?
- Is the semantic background radiation real?
- If power-law scaling holds, what is the generating process? (preferential attachment? coherence cascades? something new?)

---

## 12. Conclusion

PSA transforms embedding space from an opaque geometric object into a navigable field with coordinates, neighborhoods, boundaries, and measurable energy density. The guide star mechanism provides stability; voxelization provides hierarchy; SED provides a composite scalar field that captures not just *how much* meaning exists at each location, but *what kind* — consensus or echo, stable or transient, contested or settled. Rendered as a thermal map with morphological analysis, PSA produces the first quantitatively characterized images of how meaning organizes at scale. We submit this not as a final answer but as a first observation — and we report, honestly, what first light reveals.

---

## Appendices

### A. Guide Star Selection Details
- Full list of guide stars used in experiments
- Selection algorithm pseudocode
- Stability measurements

### B. Voxelization Algorithm
- Complete addressing scheme specification
- Micro-cell subdivision rules
- Complexity analysis

### C. SED Computation Details
- Component normalization procedures
- Edge cases (sparse voxels, single-source voxels, temporal bootstrapping)
- Sensitivity analysis: how does SED change with component weight variations?

### D. Scoring Regime Catalog
- Enumeration of detected regimes with voxel locations
- Parameter profiles per regime
- Boundary characterization

### E. Scramble Test Full Results
- Real vs. scrambled distributions for every metric
- Statistical significance tables
- Effect sizes per measurement

### F. Morphological Analysis Details
- Power-law fitting methodology and alternative distribution comparisons
- Persistent homology computation parameters
- Filamentarity classification methodology adaptation from cosmological tools

### G. Thermal Map Gallery
- Full resolution SED maps at multiple corpus scales
- Component decomposition maps (separate O, C, D, P, T, A layers)
- False-color composite maps
- Scrambled controls alongside each real map
- Annotated features: attractor cores, filaments, voids, boundary turbulence

---

## Target Venues
- **Primary**: NeurIPS (datasets & benchmarks track, or main track if morphological results are strong)
- **Strong alternative**: ICLR (representation learning — this is fundamentally about what embedding spaces look like when you give them coordinates)
- **Alternative**: ACL, EMNLP (computational semantics / linguistic structure)
- **If background radiation result is positive**: Nature Computational Science, PNAS (this becomes a finding about knowledge itself, not an NLP method)
- **Framing**: representation learning / embedding space structure / computational semantics / information geometry

## Estimated Figures
1. Guide star projection diagram (3D PSA space with labeled axes)
2. Voxelization hierarchy visualization (L1 → micro → micro.micro)
3. Locality heatmap (scoring weight similarity vs. voxel distance) — **with scrambled control**
4. Regime boundary map (gradient magnitude across voxel faces) — **with scrambled control**
5. SED component decomposition (6-panel: O, C, D, P, T, A for a representative corpus slice)
6. SED diagnostic profiles table (attractor / frontier / echo chamber / ephemeral / niche / void)
7. Thermal map — the money shot (full corpus, SED-mapped, false color) with explicit real vs. scrambled side-by-side rendering
8. Structural annotation (same thermal map, labeled: attractor cores, filaments, voids, boundary turbulence)
9. Power-law fit (log-log SED distribution, real vs. scrambled)
10. Persistence diagram (real vs. scrambled — birth-death scatter)
11. Filamentarity comparison (real vs. scrambled bar chart)
12. Cross-corpus stability (SED rank correlation scatter)
13. Scale progression (thermal maps at 1K, 10K, 100K, 1M documents — the first-light sequence), each with corresponding scrambled control
14. *If found*: background anisotropy map (residual SED after attractor/filament subtraction)

## Dependencies
- This paper stands alone (no dependency on papers 2 or 3)
- Papers 2 and 3 cite this paper for PSA coordinates, voxel addressing, and SED field
- Paper 3 (production online learning) specifically needs SED component T (temporal stability) and the regime detection framework
