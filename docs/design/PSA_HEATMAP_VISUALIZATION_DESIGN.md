# PSA Heatmap Visualization Design
**Pixelization of Semantic Anisotropy - Meaning Space Renderer**

**Author:** GitHub Copilot  
**Created:** December 9, 2025  
**Updated:** August 28, 2026
**Status:** Core Global Atlas implementation complete; signed-signature Locality Portal is the next design priority

**Implementation reconciliation:** Backend heatmap/voxel graph payloads and Spark Hub
sentence, object, frame, and ontology layers are implemented. `evokes_frame` and
`fills_role` filters/toggles are available, with bounded defaults. Remaining product
work is Z-depth rendering, guide-star visualization, higher-scale LOD, and interaction
refinement. The next architectural increment is a local calculation-space portal,
not a promotion of the 3D display into a semantic calculation space. The historical
checklist below must be read as implementation history, not as the current completion
count.

---

## 🏗️ Architecture Overview

The visual field is a presentation and navigation surface. It does not supply
semantic similarity, cluster membership, scorer evidence, traversal policy, or
answer-planning signals.

```text
Spark Hub MeaningFieldPage
  | authenticated GET /heatmap?workspace_id=...
  | authenticated GET /heatmap/voxel-graph?workspace_id=...&voxel_id=...
  v
Prompt Lambda / API gateway
  | derives tenant from JWT claims
  | requires the configured workspace
  | builds and validates semantic_namespace_id
  | rejects requests outside the deployment namespace
  v
Exact-namespace Redis field-state keys
  ^
  | written by consolidated_clustering_v2
  | persisted under the exact S3 namespace prefix
Consolidated clustering V2 visual projection
```

There is no heatmap Redis pub/sub browser path. Spark Hub performs bounded HTTP
reads. Redis is an internal exact-namespace cache, and the S3 snapshot is the
durable visual-field state.

### Calculation-Space and Display-Space Contract

PSA has several related but non-interchangeable representations. Every visual
claim must identify which representation supports it.

| Representation | Geometry | Current responsibility |
|---|---|---|
| Native embedding | Model-native 4096D in the current deployment | Final semantic adjudication after signature candidate routing |
| Raw guide affinity, `c` | 115 cosine affinities to the fixed guide-star catalog | Input to both signature construction and the 3D display mapper |
| Signed routing signature, `s` | Row-centered and L2-normalized 115D vector | Canonical ANN, prototype, and local-geometry carrier |
| Cartesian/RA-Dec position, `q` | Clamped guide-star blend in 3D, then spherical display coordinates | Visualization, aggregation, voxel addressing, and navigation only |

For raw affinity vector \(c\in\mathbb R^{115}\), the operational signature is

\[
s=\frac{c-\bar c\mathbf 1}{\lVert c-\bar c\mathbf 1\rVert_2}.
\]

It therefore lies in

\[
\mathcal S=\{s\in\mathbb R^{115}:\mathbf 1^\top s=0,\ \lVert s\rVert_2=1\}
\cong S^{113},
\]

with tangent space

\[
T_s\mathcal S=\{v:\mathbf 1^\top v=0,\ s^\top v=0\}.
\]

This \(S^{113}\) is the ambient carrier of the normalized signature chart. It is
not a claim that the intrinsic semantic manifold is 113-dimensional, and it is
not PSA's entire decision state because the native embedding remains the final
verifier.

The current 3D display mapper instead computes

\[
a_i=\max(c_i,0),\qquad Z=\sum_i a_i,\qquad
q=\frac{\sum_i a_iP_i}{Z}\quad(Z>0),
\]

where \(P_i\) is a guide star's persisted 3D display position. Negative affinity
evidence is intentionally absent from this blend. RA/Dec conversion, voxelization,
and any collision-avoidance layout occur after this dimensional reduction. The 3D
position and every 2D or 3D rendering derived from it remain visual-only: they do
not decide semantic similarity, membership, reachability, or routing.

### Target Visualization Architecture

The product should expose the representations as separate, linked views:

| Layer | Question answered | Evidence source | Priority |
|---|---|---|---|
| **Stable Global Atlas** | Where is content, and how much is present? | Versioned 3D display coordinates and aggregate field state | Production foundation |
| **Signed Signature-Space Locality Portal** | What local transformations have been observed or validated around this exact state? | Tangent displacements in the signed 115D signature carrier | Next |
| **Display Sensitivity** | How would the visual mapper move if active guide affinities changed? | Piecewise mapper Jacobian and its singular spectrum | Secondary diagnostic |
| **Empirical shape and transport research** | Is local structure ball-like, tubular, lobed, subspace-like, or truncated-conal, and does it transport? | Held-out transformations, controls, and versioned alignments | Gated research |

The atlas is the focus-and-context surface. Selecting a cell first resolves an
exact sentence or a paraphrase-group medoid; a voxel alone has no unique tangent
state. The locality portal then opens beside or over the atlas, while preserving a
ghost marker at the selected atlas position and correspondence links for shared
neighbors.

### UI Entry Point

**Location:** `spark_hub/src/pages/MeaningFieldPage.tsx`
**Route:** `/meaning-field`

### Transport Contract

| Request | Scope | Purpose |
|---------|-------|---------|
| `GET /heatmap` | Authenticated tenant plus explicit workspace | Read visual cells and field metrics |
| `GET /heatmap/voxel-graph` | Same exact scope plus canonical visual voxel ID | Read the bounded graph drill-down |

Both responses carry tenant, workspace, and semantic namespace identity. Spark
Hub rejects a response whose workspace does not match the configured request.

### State Management Architecture (NEW - December 2025)

The PSA heatmap is powered by a hybrid state management system:

| Layer | Purpose | Data | TTL |
|-------|---------|------|-----|
| **In-Memory** | Active processing | Voxels, clusters, exemplars | Session |
| **Redis** | Fast reads for viz | Hot voxels, metrics, heat scores | Persistent |
| **S3** | Source of truth | Feature flags, subdivisions, audit | 7+ years |
| **Neo4j** | Graph persistence | Sentences, paraphrases, objects | Permanent |

**Key Design Principles:**
- **Semantic Namespace Isolation**: every tenant/workspace field has an exact Redis namespace and S3 prefix
- **Adaptive Sync**: S3 sync frequency scales with data volume (10s-5min)
- **IRREVERSIBLE Subdivisions**: Once a voxel subdivides, state is preserved forever
- **Auto-Enable Thresholds**: Features activate automatically based on data volume

### Auto-Enable Thresholds (NEW - December 2025)

Features automatically activate based on data volume thresholds:

| Feature | Threshold | Reason | Auto-Disable? |
|---------|-----------|--------|---------------|
| **Quality Gates** | `sentences >= 5,000` | Need enough data for meaningful validation | Never |
| **Adaptive Grid** | `max_voxel_density >= 300` | Subdivide when voxels get dense | Never (structural) |
| **Maintenance Engine** | `clusters >= 500` | Cluster cleanup cycles warranted | Never |
| **Drift Detection** | `sentences >= 10,000` | Baseline corpus exists | Never |
| **Contested Resolver** | `contested_rate >= 5%` | Significant ambiguity rate | Yes (if < 2%) |
| **CloudWatch Metrics** | Manual only | Cost consideration | Manual |

**Redis Keys for Feature State:**
```
{semantic_namespace_id}:clustering:state           # Namespace state hash
{semantic_namespace_id}:clustering:hot_voxels      # Sorted set by heat
{semantic_namespace_id}:clustering:voxel:{id}      # Per-voxel data hash
{semantic_namespace_id}:clustering:activity        # Optional live activity stream
```

**S3 Structure:**
```
s3://{CLUSTER_STATE_S3_BUCKET}/clustering/visual-field/
└── {semantic_namespace_id}/
    ├── state.json              # Current field metrics + subdivisions
    └── snapshot.json           # All-node visual hierarchy snapshot
```

### Data Flow (Already Implemented!)

```
Sentence → Embedding → consolidated_clustering_v2 → 
  ├── Voxel Assignment (visual_voxel_id)
  ├── RA/Dec Coordinates (ra, dec, radius)
  ├── Cluster Assignment (cluster_id)
  ├── sentence_count++ on voxel (ALL sentences counted)
  └── Feature State Update (every 10 sentences)
      ├── Check auto-enable thresholds
      ├── Update Redis cache
      └── Sync to S3 (adaptive frequency)
```

---

## 📊 Data Model

### 3D PSA Display Coordinate (Existing, Visual-Only)

The historical projection sketch below records the original RA/Dec integration
contract from `consolidated_clustering_v2/routing/projections.py`:

```python
# 4096-dim embedding → 3D via random projection
# Then → RA/Dec spherical coordinates

def to_ra_dec(x, y, z, semantic_depth):
    """
    Returns:
        RA (Right Ascension): 0° to 360° (longitude on display sphere)
        Dec (Declination): -90° to +90° (latitude on display sphere)
        r (radius): semantic_depth (pre-normalization magnitude)
    """
```

The current consolidated 3D path is the clamped guide-star blend in
`consolidated_clustering_v2/routing/psa_mapper.py`, as defined in the
calculation-space contract above. The historical random-projection sketch must
not be read as the current routing algorithm. Both paths emit display coordinates;
neither is the signed 115D calculation space.

### Coordinate Stability & Field Alignment (Guidestars)

RA/Dec are a *display coordinate chart* over a 3D projection derived from the
guide-affinity pipeline. The absolute orientation (what is "north/east") is not
inherently semantic. Even relative neighborhoods in this chart are visualization
evidence only; operational similarity and routing use the signed signature and
native embedding contracts.

The production frame uses the fixed 115-entry semantic guide-star catalog, not
Zipfian tenant graph nodes. Stability means reproducibility within an exact versioned
frame. It does not mean that an embedding-model or projection upgrade preserves the
same geometry.

- **Guide stars**: the fixed semantic catalog whose identity, order, text/version,
  embeddings, and 3D display positions are hashed as frame inputs.
- **Within-frame stability**: persist the exact mapper and display metadata so the
  same inputs reproduce the same address.
- **Cross-frame alignment**: treat rigid rotation, affine, low-rank, or nonlinear
  transforms as competing calibrated models. No best-fit rotation is accepted merely
  because it fits landmarks; it must pass held-out identity, relational, conditioning,
  and collapse gates.
- **Cross-field research**: an optional canonical-field alignment may share governed
  guide definitions and calibration metadata without combining raw tenant content.
  That does not make the fields semantically interchangeable or privacy-safe by
  construction.

**Global Atlas stability requirements:**

- Pin the embedding model, guide-star set, 3D guide positions, alignment, atlas
  projection, scale, and viewport frame to explicit version identifiers.
- Filtering or changing a heat metric may change visibility and styling, but must
  not refit coordinates or rescale the surviving points.
- Preserve the unsnapped projected position when collision handling moves a mark.
  A collision-adjusted mark is presentation state, not a metric coordinate.
- Show frame/version changes as an explicit comparison or migration. Never animate
  them as if the same coordinates moved continuously.
- Directional measurements belong in the locality portal. Atlas distances and
  directions are navigation aids unless separately validated for the stated task.

**Current UI limitation:** Spark Hub currently derives display min/max bounds from the
visible payload and collision-snaps overlapping cells. Filtering can therefore move
apparent positions, and a snapped mark can differ from its projected coordinate. Until
the pinned-frame requirements are implemented, the current atlas is an inventory and
drill-down display projection, not a stable metric backdrop for directional overlays.

---

## 🌐 Versioned Semantic Coordinate Protocol: Model-Aligned Addressing

### The Insight

PSA can support a **versioned, model-aligned semantic addressing protocol**. The
GPS analogy describes stable addressing inside a shared frame; it does not establish
a universal or model-independent semantic coordinate system. A receiver must possess
the matching guide catalog, signature definition, model/frame calibration, projection,
and alignment metadata. Coordinates communicate an address in that frame, not a
self-decoding semantic verdict.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              THE SEMANTIC COORDINATE PROTOCOL                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SENDER                              RECEIVER (calibrated frame)        │
│   ┌──────────────┐                    ┌──────────────┐                   │
│   │ Embed idea   │                    │ Decode coord │                   │
│   │ in MY field  │ ────────────────▶  │ from SHARED  │                   │
│   │ Get coord    │   PXL(17,42)       │ field map    │                   │
│   └──────────────┘   +Δ(0.13,-0.04)   └──────────────┘                   │
│                                                                          │
│   What's transmitted:                                                    │
│   • Versioned field-display metadata                                      │
│   • Fixed 115-guide catalog and calibration metadata                      │
│   • The coordinate and exact frame identifiers                            │
│                                                                          │
│   Payload policy is deployment-specific; coordinates and aggregates       │
│   remain potentially sensitive even when source text is omitted.          │
└─────────────────────────────────────────────────────────────────────────┘
```

### How It Works

A coordinate like `PXL(17,42)+Δ(0.13, -0.04)` is a **compressed visual-field
pointer** in a named frame.

Given a coordinate and an exact compatible frame, a receiver can locate display
cells, identify contributing guide stars, and request authorized nearby examples.
Heat is an aggregate count and the 3D chart is a display projection. Neither exposes
physical forces, semantic curvature, a reconstructed manifold, or calculation-space
directions.

The receiver can use the pointer to retrieve or summarize nearby authorized anchors
and examples. Any semantic interpolation remains model- and calibration-dependent
and must report uncertainty; the coordinate alone is not evidence of equivalence.

### Protocol Payload

| Component | Size | Purpose |
|-----------|------|---------|
| Field-display metadata and aggregate cells | Measure per resolution/version | Reproduce the display frame and counts |
| Fixed guide-star catalog (115 entries) | Measure per encoding/version | Identify the signature and mapper axes |
| Coordinate batch | Measure per address format/version | Carry versioned display addresses |
| Calibration and provenance | Measure per alignment/version | State compatibility and validation evidence |

Earlier payload-size estimates were aspirational sketches, not measured protocol
contracts. Publish measured serialized sizes and privacy review with each version.

### What This Enables

**Semantic Operations Without Text:**
```
"Are these two ideas duplicates?"

Text way:     Send both texts → semantic verifier compares → Answer
Address way:  use the shared frame to retrieve candidates → verifier compares → Answer

Atlas distance is useful for candidate navigation, but it is not a duplicate or
membership decision. The signed signature routes candidates and the native embedding
performs final semantic adjudication.
```

**Cross-Enterprise Meaning Exchange:**
```
From: aiidea_field_v3
At: PXL(81,19)+Δ(0.07,0.22)
Time: 2460681.5123

→ A receiver with a compatible, validated frame can locate the neighborhood and
  inspect authorized reference evidence.
```

### The GPS Analogy

The GPS analogy is useful for versioned frames and landmarks, but the security model
is different. Semantic coordinates, density maps, guide affinities, signatures, and
query patterns can reveal topic, membership, or distribution information.

### Security Model

This is **not** encryption or anonymization. It is a coordinate-based abstraction that
requires threat modeling, authorization, minimization, aggregation thresholds, and
leakage testing.

| What | Exposure |
|------|----------|
| Field topology and density | Governed; aggregates may leak tenant distribution |
| Guide-star definitions and embeddings | Governed frame metadata |
| Signatures and document-coordinate mappings | Sensitive tenant/model-derived data |
| Coordinate interpretation | Authorized compatible receiver with versioned evidence |

No field component is public by default. Publication is a policy decision supported by
a deployment-specific privacy and inference-risk assessment.

### Business Implications

The research opportunity is an interoperable, versioned semantic addressing standard.
Cross-model equivalence, local-structure transport, and privacy properties remain
empirical and governance questions rather than consequences of the coordinate format.

---

## 🌫️ Coverage, Assignment Evidence, and Calibrated Uncertainty

### The Problem

The atlas has uneven coverage and the semantic resolver has uneven assignment
evidence. These are related observations, but not interchangeable:

- **Dense or guide-adjacent display regions** have stronger atlas coverage, not
  automatically higher semantic correctness.
- **Sparse display regions** have fewer examples, not automatically an undefined
  meaning.
- **Contested resolver assignments** provide direct task-specific ambiguity evidence,
  while a visual boundary alone does not.

PSA should expose the available coverage and assignment evidence and reserve
uncertainty language for calibrated quantities.

### The Solution: Versioned Diagnostic Metrics

Every pixel/voxel can carry **coverage and assignment metadata**, not just position.
Call a value uncertainty only after calibration against a stated target event.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIELD STABILITY REPRESENTATION                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional coordinate:     PXL(17,42)                                │
│   With diagnostic metadata:   PXL(17,42) + coverage/assignment report   │
│                               ↑           ↑                              │
│                               position    versioned diagnostics          │
│                                                                          │
│   Components:                                                            │
│   • atlas_coverage:       Local aggregate support                        │
│   • contested_rate:      Resolver ambiguity evidence                    │
│   • local_sensitivity:   Named-map Jacobian/SVD diagnostics             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Three Candidate Diagnostic Sources

#### 1️⃣ Local Guide-Star Anchoring

For any display cell, candidate coverage features include distance to guide-star
positions, angular spread among contributors, and local example counts. These yield a
feature vector or scalar coverage heuristic, not a stability tensor and not semantic
confidence without held-out calibration.

The following code is retained as a historical heuristic sketch. Its fixed weights and
thresholds are not an accepted production or research calibration:

```python
def compute_anchor_stability(pixel: PSAPixel, guide_stars: List[GuideStar]) -> float:
    """
    Compute how well-anchored this pixel is by nearby guide stars.
    
    Returns: 0.0 (fog) to 1.0 (firmly anchored)
    """
    # Find nearest guide stars
    distances = [distance(pixel.coord, gs.coord) for gs in guide_stars]
    nearest_k = sorted(zip(distances, guide_stars))[:5]
    
    # Factor 1: How close is the nearest anchor?
    min_distance = nearest_k[0][0]
    distance_factor = 1.0 / (1.0 + min_distance)
    
    # Factor 2: Angular spread (good triangulation = high spread)
    angles = compute_angular_spread([gs for _, gs in nearest_k])
    spread_factor = angles / 180.0  # Normalize to 0-1
    
    # Factor 3: Local paraphrase density
    local_density = count_paraphrases_in_radius(pixel.coord, radius=0.1)
    density_factor = min(local_density / 10.0, 1.0)
    
    # Combined stability score
    return (distance_factor * 0.4 + spread_factor * 0.3 + density_factor * 0.3)
```

**Low anchor density = weak atlas coverage, not proof of semantic uncertainty.**

#### 2️⃣ Calibrated Local Sensitivity and Conditioning

A nearest-guide identity change divided by an epsilon is a finite sensitivity
sample across a discontinuous label function. It is **not** a numerical condition
number. The earlier `condition_number` sketch is therefore retired as terminology
and must not be implemented or interpreted with the old `1.0` and `>10` thresholds.

A defensible sensitivity report begins with a named differentiable map \(f\), stated
input and output norms, a calibration scale, and its local Jacobian \(J_f\). For a
calculation-side, continuously parameterized semantic operator \(T_\theta\), the map
of interest is

\[
\theta\longmapsto s(T_\theta x),
\]

restricted to \(T_s\mathcal S\). For the separate 3D display mapper, the analytic
piecewise Jacobian is defined in the Display Sensitivity section below. Discrete
edits are empirical secants, not Jacobian responses.

Use singular-value diagnostics with their map and normalization attached:

| Diagnostic | Meaning | Interpretation guardrail |
|---|---|---|
| \(\sigma_{\max}(J)\) | Largest calibrated local response | Sensitivity, not uncertainty by itself |
| Nonzero \(\sigma_{\min}(J)\) | Weakest resolved response direction | Report the numerical rank threshold |
| \(\sigma_{\max}/\sigma_{\min}\) | Directional conditioning ratio | Call this a condition ratio only when map, norms, scale, and rank are valid |
| Effective rank | Number of locally resolved response directions | Does not estimate intrinsic semantic dimension |
| Finite-difference residual | Agreement between predicted and observed local response | Required before treating a Jacobian as locally adequate |
| Active-set margin | Distance from a mapper affinity to the ReLU boundary at zero | Display-mapper diagnostic only |

Undefined or rank-deficient ratios remain undefined; they are never replaced with a
convenient stability default. These diagnostics supplement anchor confidence and
contested assignment evidence. They must not be collapsed into a single fog score
without held-out calibration.

#### 3️⃣ Contested Assignment Rate

From the existing clustering system: what percentage of sentences in this region had **contested** cluster assignments?

```python
def compute_contested_rate(voxel: AdaptiveVoxel) -> float:
    """
    Percentage of sentences with ambiguous cluster assignments.
    """
    if voxel.sentence_count == 0:
        return 0.0
    return voxel.contested_count / voxel.sentence_count
```

**High contested rate = semantic boundary zone.**

### Extended Data Model

Every sentence, node, and voxel should carry field quality metrics:

```typescript
interface PSACell {
  // Position
  ra: number;
  dec: number;
  
  // Existing metrics
  sentence_count: number;
  paraphrase_count: number;
  cluster_count: number;
  heat: number;
  
  // NEW: Coverage, assignment, and calibrated sensitivity diagnostics
  field_stability: {
    atlas_coverage: number;          // Descriptive local support, not correctness
    coverage_anisotropy: number;     // Directionality of display coverage
    guide_coverage: number;          // Descriptive guide-position coverage
    contested_rate: number;          // % ambiguous assignments
    local_sensitivity?: {
      map_id: string;                // Named calculation or display map
      norm_id: string;               // Input/output norm and calibration
      jacobian_spectral_norm: number;
      nonzero_condition_ratio?: number;
      effective_rank: number;
      finite_difference_residual?: number;
      active_set_margin?: number;    // Display mapper only
    };
  };
  
  // Derived fog classification
  coverage_level: 'well_sampled' | 'mixed' | 'sparse' | 'empty';
}

interface SemanticCoordinate {
  // Position
  pixel: { x: number; y: number };  // Or ra/dec
  delta: { dx: number; dy: number }; // Sub-pixel offset
  
  // Descriptive coverage; uncertainty requires a calibrated target event
  atlas_coverage?: number;
  uncertainty_calibration_id?: string;
  calibrated_error_probability?: number;
  
  // Field context
  field_id: string;
  field_version: string;
}
```

### Coverage and Assignment Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEANING SPACE WITH FOG OVERLAY                       │
│                                                                          │
│            ░░░▒▒▓▓██▓▓▒▒░░░                                             │
│         ░░▒▒▓▓████████▓▓▒▒░░          🔴 = Guide star (anchor)          │
│        ░▒▒▓▓██████🔴█████▓▓▒▒░                                          │
│       ▒▒▓▓████████████████▓▓▒▒        ▓▓ = Well-sampled display cells   │
│       ▓▓████   🔴      🔴 ████▓▓       ░░ = Sparse atlas coverage        │
│       ▒▒▓▓████████████████▓▓▒▒        ▒▒ = Mixed atlas coverage         │
│        ░▒▒▓▓███🔴███████▓▓▒▒░                                           │
│         ░░▒▒▓▓████████▓▓▒▒░░░░░░     ░░░░ = No mapped observations     │
│            ░░░▒▒▓▓██▓▓▒▒░░░░░░░░░░░      (unknown, not impossible)     │
│                           ░░░░░░░░░                                      │
│                                                                          │
│  Coverage: ▓ well sampled │ ▒ mixed │ ░ sparse; uncertainty separate    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Operational Implications

**Query and review integration:**

- Warn the UI when anchor coverage is weak or contested assignments are common.
- Increase evidence inspection or review budgets in poorly supported regions.
- Refuse presentation-level collapse where uncertainty evidence does not justify it.
- Do not multiply a semantic score by a 3D mapper sensitivity metric. The visual
  field does not alter verified membership or replace the native semantic verifier.
- Introduce any calculation-side uncertainty adjustment only after held-out
  calibration establishes that it improves the stated decision task.

### The GPS Parallel

This is exactly how real navigation works:

| GPS | PSA Semantic Field |
|-----|-------------------|
| **HD lanes in cities** | Dense regions with many guide stars |
| **"Few mapped observations here"** | Sparse atlas coverage |
| **Accuracy radius (HDOP)** | Separately calibrated anchor uncertainty |
| **Satellite geometry** | Guide star angular spread |

The analogy is limited: GPS has a physical measurement model. PSA must empirically
calibrate each uncertainty and sensitivity quantity before displaying it as accuracy.

### Implementation Priorities

| Priority | Component | Complexity | Value |
|----------|-----------|------------|-------|
| **P1** | Add descriptive `guide_coverage` to voxels | Low | High |
| **P1** | Add `contested_rate` (already have data) | Low | High |
| **P2** | Define named maps/norms and validate Jacobian/SVD diagnostics | Medium | Very High |
| **P2** | Coverage and contested-evidence overlays | Medium | High |
| **P3** | Validate task-specific uncertainty calibration before scoring use | Medium | Very High |
| **P3** | LLM-readable diagnostic and calibration metadata | Low | High |

### New Redis Keys

```
{semantic_namespace_id}:clustering:voxel:{id}:stability   # Hash with stability metrics
{semantic_namespace_id}:clustering:coverage_map:{resolution}   # Pre-computed coverage overlay
```

### New S3 Structures

```
s3://{CLUSTER_STATE_S3_BUCKET}/clustering/visual-field/
└── {semantic_namespace_id}/
    ├── diagnostics/coverage_map_{resolution}_{timestamp}.json
    └── guide_stars/anchors_{timestamp}.json
```

---

**Required versioning (returned by APIs and persisted to S3):**
- `tenant_id`, `workspace_id`, and canonical `semantic_namespace_id`
- `embedding_model_id` (embedder identity/version)
- `projection_id` and `projection_hash` (projection configuration / random matrix hash)
- `guidestar_set_id` (which anchors define the frame)
- `alignment_id` (native frame or held-out-validated mapping version; optionally `aligned_to_field_id`)

**UI messaging:** show a subtle note like “Versioned guide frame; 3D axes and
distances are display-only.” Optionally toggle guide-star markers for navigation and
mapper context.

### Heatmap Data Structure

```typescript
interface PSAHeatmapData {
  // Canonical meaning-field identity
  field: {
    tenant_id: string;
    workspace_id: string;
    semantic_namespace_id: string;
    field_id: string;
    aligned_to_field_id?: string;  // optional "mother field" alignment target
  };

  // Projection + frame metadata (reproducibility within an exact version)
  projection: {
    embedding_model_id: string;    // embedder identity/version
    projection_id: number;         // router projection index
    projection_hash: string;       // hash of projection matrix/seed
    guidestar_set_id: string;      // fixed, versioned 115-guide catalog
    alignment_id: string;          // native frame or validated mapping version
  };

  // Grid configuration
  grid: {
    resolution_degrees: number; // e.g., 10, 5, 2, 1
    ra_bins: number;      // e.g., 360 (1° resolution) or 720 (0.5°)
    dec_bins: number;     // e.g., 180 (1° resolution) or 360 (0.5°)
    grid_size: number;    // From VoxelIndex (0.10 default)
    binning: 'ra_dec' | 'voxel_leaf';
  };
  
  // Heatmap cells
  cells: PSACell[];

  // Global metrics (per meaning field)
  metrics: {
    total_sentences: number;
    total_paraphrases: number;
    total_clusters: number;
    max_voxel_density?: number;
  };

  // Optional: "liquid glass" highlights
  hottest_voxels?: PSAHotVoxel[];

  // Timestamp
  timestamp: string;      // ISO timestamp
}

interface PSACell {
  // Position
  ra: number;             // 0-360
  dec: number;            // -90 to +90
  
  // Count metrics (raw, not area-normalized)
  sentence_count: number;
  paraphrase_count: number;
  cluster_count: number;
  
  // Heat value (pre-computed)
  heat: number;           // log(sentence_count) × (1 + variance)
  
  // Activity metrics (for glow effects)
  recent_joins: number;   // Joins in last 60 seconds
  recent_new_clusters: number;
  
  // Voxel info (for drill-down)
  visual_voxel_id: string;
  voxel_level: number;    // 0-3 (adaptive grid level)
  
  // Optional: semantic stats
  mean_cosine_variance?: number;
  is_contested?: boolean;
  contested_rate?: number;  // % of contested assignments
  
  // NEW: Coverage, assignment, and calibrated sensitivity diagnostics
  // See "Coverage, Assignment Evidence, and Calibrated Uncertainty" section
  field_stability?: {
    atlas_coverage: number;          // Descriptive local support, not correctness
    coverage_anisotropy: number;     // Directionality of display coverage
    guide_coverage: number;          // Descriptive guide-position coverage
    local_sensitivity?: {
      map_id: string;
      norm_id: string;
      jacobian_spectral_norm: number;
      nonzero_condition_ratio?: number;
      effective_rank: number;
      finite_difference_residual?: number;
      active_set_margin?: number;
    };
  };
  
  // Derived descriptive coverage classification
  coverage_level?: 'well_sampled' | 'mixed' | 'sparse' | 'empty';
}

interface PSAHotVoxel {
  voxel_key: string;
  heat: number;
  ra: number;
  dec: number;
  sentence_count: number;
  cluster_count: number;
  recent_joins?: number;
  contested_rate?: number;
}
```

### Optional: Neo4j Query for Heatmap Data (If Persisted)

MVP uses the clustering service state for heatmap aggregation. If we also persist `ra/dec` onto Neo4j nodes for recovery/history, these queries can generate the same buckets.

```cypher
// Aggregate sentence/paraphrase counts by PSA region
MATCH (s:Sentence)
WHERE s.ra IS NOT NULL AND s.dec IS NOT NULL
WITH 
  floor(s.ra / 10) * 10 AS ra_bucket,
  floor((s.dec + 90) / 10) * 10 - 90 AS dec_bucket,
  count(s) AS sentence_count
RETURN ra_bucket, dec_bucket, sentence_count
ORDER BY ra_bucket, dec_bucket

// Paraphrase groups with member counts
MATCH (pg:ParaphraseGroup)
WHERE pg.ra IS NOT NULL AND pg.dec IS NOT NULL
OPTIONAL MATCH (pg)-[:CONTAINS]->(s:Sentence)
WITH pg, count(s) AS member_count
RETURN 
  floor(pg.ra / 10) * 10 AS ra_bucket,
  floor((pg.dec + 90) / 10) * 10 - 90 AS dec_bucket,
  count(pg) AS paraphrase_count,
  sum(member_count) AS total_sentences
```

---

## 🖼️ Frame Visualization Approach

> **Added:** 2026-03-25 — Based on spark_hub code review

### Problem
The voxel meaning graph currently renders 3 layers (sentences → objects → ontology). Adding frames as a 4th column creates visual clutter — four vertical bands with dense cross-edges become hard to read.

### Recommendation: Ontology-to-Frame Edges Only

Instead of showing all frame relationships (which would include both object→frame and ontology→frame edges), **show only ontology→frame edges** (`evokes_frame`). This approach:

1. **Reduces clutter** — Frames appear as extensions of the ontology layer, not a separate column competing with objects
2. **Preserves semantic meaning** — The ontology→frame relationship is the primary semantic signal (what conceptual frame does this ontology class evoke?)
3. **Avoids redundancy** — Object→frame edges (`fills_role`) often duplicate the path objects → ontology → frames

### Layout Options

**Option A: Inline with Ontology (Recommended)**
```
[Sentences] ─references─→ [Objects] ─instance_of─→ [Ontology] ─evokes_frame─→ [Frames]
```
- Frames rendered as a sub-column adjacent to ontology
- Only `evokes_frame` edges visible by default
- `fills_role` available as optional toggle (off by default)
- Keeps the visual flow left-to-right without crowding the middle

**Option B: Overlay on Ontology**
```
[Sentences] ─references─→ [Objects] ─instance_of─→ [Ontology + Frames]
```
- Frame nodes rendered in the ontology column with distinct color (purple vs green)
- Saves horizontal space but may confuse layer identity

### Implementation (spark_hub)
In `MeaningFieldPage.tsx`:
1. Add `'frames'` to `GRAPH_LAYER_ORDER`: `['sentences', 'objects', 'ontology', 'frames']`
2. Enable `evokes_frame` edges by default: `evokes_frame: true`
3. Keep `fills_role` hidden by default: `fills_role: false`
4. Add toggle buttons for both frame edge types
5. Update legend to show 4 layer colors

---

## 🎨 Visualization Design

### Phase 1: Static Heatmap (MVP)

**Projection Options:**

1. **Mollweide Projection** (Recommended for MVP)
   - Equal-area: good for density visualization
   - Shows entire sphere in one view
   - Familiar from world maps and astronomical charts

2. **Aitoff Projection**
   - Alternative equal-area option
   - More circular appearance

3. **Equirectangular**
   - Simplest to implement
   - Distorts at poles but acceptable for initial version

**Visual Design:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE MEANING SPACE                             │
│                                                                          │
│                        RA →  0°    90°    180°   270°   360°            │
│                        ┌──────────────────────────────────┐              │
│           Dec +90° ────┤      ░░░▒▒▓▓██▓▓▒▒░░░           │              │
│                        │   ░░▒▒▓▓████████▓▓▒▒░░          │              │
│                 +45° ──┤  ░▒▒▓▓██████████████▓▓▒▒░       │              │
│                        │ ▒▒▓▓████  HOT ZONE  ████▓▓▒▒    │              │
│                   0° ──┤ ▓▓████               ████▓▓     │              │
│                        │ ▒▒▓▓████   COOL    ████▓▓▒▒     │              │
│                 -45° ──┤  ░▒▒▓▓██████████████▓▓▒▒░       │              │
│                        │   ░░▒▒▓▓████████▓▓▒▒░░          │              │
│           Dec -90° ────┤      ░░░▒▒▓▓██▓▓▒▒░░░           │              │
│                        └──────────────────────────────────┘              │
│                                                                          │
│  ░ = 1-10 sentences    ▒ = 10-100    ▓ = 100-1k    █ = 1k+              │
│                                                                          │
│  Total: 161M sentences │ 2.4M clusters │ Updated: 2025-12-09T14:30:00Z  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Color Palette Options:**

Each heatmap renders exactly **one tenant/workspace meaning field**. Semantic namespaces are never interspersed; the legend and color scale are computed within the selected field.

1. **aiidea Brand Gradient** (REQUIRED)
   - Based on `aiidea_color_palette.md`
   - Gradient: Turquoise (#14C4ED) → Orange (#FE904C)
   - Progress bar style: `#14C4ED → #FE904C`
   - Cool zones: Turquoise tones
   - Hot zones: Orange/Yellow tones
   - Accent highlights: Yellow (#FFBD1F)

**aiidea Heatmap Color Scale:**

```
Meaning Density →

   LOW                    MEDIUM                    HIGH
    │                       │                        │
    ▼                       ▼                        ▼
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│#0099BF│#14C4ED│#1AC8DB│#FFBD1F│#FE904C│#FE904C│#FFFFFF│
│ Dark  │Turq-  │ Blue- │Yellow │Orange │Orange │ White │
│ Turq  │ uoise │ Green │       │       │ Hot   │ Peak  │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┘
    0       1       10      100     1k      10k    100k+
              Sentence Count (log scale)
```

**Color Constants (from aiidea_color_palette.md):**

```typescript
const AIIDEA_HEATMAP_COLORS = {
  // Cold zones (sparse)
  coldest: '#0099BF',      // Dark Turquoise - very sparse
  cold: '#14C4ED',         // Turquoise - sparse
  cool: '#1AC8DB',         // Blue-Green - light activity
  
  // Warm zones (active)
  warm: '#FFBD1F',         // Yellow - moderate density
  hot: '#FE904C',          // Orange - high density
  
  // Peak zones
  hottest: '#FFFFFF',      // White - extreme density (glowing)
  
  // Background
  background: '#E4E9EA',   // Grey panel background
  
  // Text
  label: '#222222',        // Main text
  sublabel: '#666666',     // Subtext
};
```

### Phase 2: Interactive Features (Post-MVP)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Interactive Controls                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [Zoom: ─────●───────]  [Resolution: ○ 10° ● 5° ○ 1°]  [Guidestars: ☐]  │
│                                                                          │
│  Filter by:                                                              │
│  ┌──────────────┐ ┌──────────────┐                                       │
│  │ Time Range ▼ │ │ Min Count ▼  │                                       │
│  └──────────────┘ └──────────────┘                                       │
│                                                                          │
│  View Mode:                                                              │
│  ● Sentence Count  ○ Paraphrase Count  ○ Cluster Count                  │
│  ○ Heat (Count × Variance)  ○ Contested Regions                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Interaction Behaviors:**

| Interaction | Behavior |
|-------------|----------|
| **Hover** | Show tooltip with cell stats (sentence count, paraphrase count, cluster IDs) |
| **Click** | Select region, show detail panel with actual paraphrase texts |
| **Drag** | Pan across the meaning space |
| **Scroll** | Zoom in/out (increase/decrease resolution) |
| **Double-click** | Drill down to voxel level view |

### Next Priority: Signed Signature-Space Locality Portal

The Locality Portal is the first view that visualizes PSA's calculation-side local
geometry. It does not deform the atlas in its initial release. It opens as a bounded
focus-and-context panel linked to the selected atlas location.

**Center selection:**

- A sentence selection centers the portal on that sentence's signed normalized
  115D signature.
- A paraphrase-group selection centers it on the persisted actual-member medoid,
  not an abstract voxel center or a newly averaged display coordinate.
- A voxel selection must resolve a sentence or group medoid before opening the
  portal because a voxel aggregates multiple states and has no unique tangent space.

For center \(s=s(x)\) and a nearby signature \(s'\), map the finite displacement to
the tangent space with the spherical logarithm

\[
\log_s(s')=
\frac{\theta}{\sin\theta}\left(s'-\cos\theta\,s\right),
\qquad \theta=\arccos(s^\top s'),
\]

away from the antipodal singularity. A finite displacement is a secant observation.
It becomes a Jacobian response only for a continuously parameterized transformation
whose finite differences converge as the step size shrinks.

For transformation family \(r\), operational cost horizon \(h\), and geometric
radius \(\rho\), the primary empirical object is

\[
\widehat{\mathcal L}^{(r,h)}_\rho(s(x))=
\left\{
\log_{s(x)}s(Tx):
T\in\mathcal O_r,\ T\text{ has a validated path},\
\operatorname{cost}(T)\le h,\ d(s(x),s(Tx))\le\rho
\right\}.
\]

The hat is important: this is a sampled empirical tangent-transition set, not the
complete reachable region. The portal should first render points, paths, directional
tubes, density contours, or typed lobes without presuming a cone. Candidate models
must include a ball, subspace, tube, lobe mixture, truncated cone, and union of
truncated cones.

A bounded observed set is not literally a mathematical cone. The testable conal
hypothesis is

\[
\widehat{\mathcal L}^{(r,h)}_\rho(s)\approx C_s^{(r)}\cap B_\rho.
\]

Filled cone glyphs are permitted only after held-out tests support positive scaling
and composition closure, outperform the alternative shapes, and preserve calibrated
coverage. Paraphrase variation may be an approximately isotropic core while temporal,
causal, quantitative, polarity, generalization, or specialization operators form
separate typed lobes. Do not convexify them into one region unless mixed operations
are independently admissible and validated.

#### Evidence Ladder

Every mark and path must carry one of these ordered evidence states:

\[
\boxed{\text{OBSERVED}\rightarrow\text{SUPPORTED}\rightarrow
\text{VALIDATED REACHABLE}\rightarrow\text{EXPECTED}\rightarrow
\text{REALIZED}}
\]

| State | Required meaning |
|---|---|
| **OBSERVED** | A recorded endpoint or secant exists; no general reachability claim |
| **SUPPORTED** | Repeated operator-conditioned observations and controls support a local pattern |
| **VALIDATED REACHABLE** | A named admissible operator, validated path, cost/horizon, and locality gate establish reachability |
| **EXPECTED** | A held-out calibrated model assigns transition or endpoint probability under stated context and policy |
| **REALIZED** | The actual chosen or executed transformation path was recorded |

The compact product vocabulary is: **CAN** only for VALIDATED REACHABLE, **WILL**
for EXPECTED, and **DID** for REALIZED. Cone aperture must never double as an
uncertainty encoding; uncertainty uses separate hatching, opacity, intervals, or
ellipses.

#### Portal Projection and Basis Stability

The portal displays a 2D or 3D projection of tangent vectors, while all neighborhood,
operator, and validation calculations remain in the canonical signed 115D frame.
Persist and expose:

- center entity ID and signature-frame version;
- neighbor/reference-set query and hash;
- tangent-basis algorithm, basis ID/hash, dimension, deterministic axis order, and
  deterministic sign convention;
- operator family, cost definition, horizon, radius, and evidence version;
- retained variance, neighborhood preservation, projection stress, and held-out
  distortion diagnostics; and
- alignment version for any cross-model comparison.

Do not silently refit a basis when filters change. If a new basis is required, show a
version transition and do not imply that screen axes from different basis versions
are comparable.

A reported experimental result suggests useful PSA behavior may persist near a 40D
compression frontier. This is a promising reduced-chart hypothesis, not a production
contract or proof of intrinsic dimension. No checked-in 40D benchmark artifact was
identified for this revision, so the claim remains pending benchmark provenance. Do
not conflate a 40-dimensional reduction with checked-in 40-guide-star experiments or
with the existing 32D/16D reduced results. The canonical signed carrier remains 115D.

#### Portal Before Fisheye Deformation

The first release keeps atlas and tangent coordinates in separate registered charts:

- retain a ghost marker at the center's stable atlas position;
- open a recentered portal beside or over the atlas;
- link corresponding atlas neighbors and tangent marks;
- disclose projection fidelity and basis version in the portal; and
- keep zoom/filter interactions from moving the pinned atlas frame.

Do not directly interpolate unrelated screen coordinates with a blend such as
\((1-w)P_{\text{atlas}}+wP_{\text{tangent}}\). Such a blend can fold topology,
manufacture continuity, and distort density. A deforming fisheye is gated on explicit
boundary registration, continuity, orientation preservation, bounded distortion, and
non-folding validation.

### Secondary Diagnostic: Display Sensitivity

For an active guide affinity \(c_i>0\), the Cartesian mapper derivative is

\[
\frac{\partial q}{\partial c_i}=\frac{P_i-q}{Z}.
\]

Thus a ray from \(q\) toward active guide position \(P_i\) exactly represents the
local movement of the **displayed Cartesian point** when that raw affinity increases.
The column vectors form the piecewise mapper Jacobian; its singular values support
the calibrated diagnostics defined earlier.

This overlay must be labeled **Display Sensitivity**, not Semantic Locality or
reachability. It is nondifferentiable at \(c_i=0\), excludes negative affinities,
and may include directions that no realizable embedding or semantic operation can
produce. RA/Dec seams and poles, voxel binning, and collision adjustment add further
display discontinuities after the Cartesian derivative. Render negative signed
evidence separately, such as hollow or dashed guide indicators, rather than implying
that the clamped mapper used it.

### Enhanced Heat Formula (December 2025)

The basic heat formula can be enriched for more expressive visualization:

```typescript
// Base heat (density)
const baseHeat = Math.log(cell.sentence_count) * (1 + cell.mean_cosine_variance);

// Activity boost (recent additions glow brighter)
const activityBoost = Math.min(cell.recent_joins / 10, 1.0) * 0.3;

// Contested penalty (high ambiguity = different visual treatment)
const contestedFactor = cell.contested_rate > 0.1 ? 0.8 : 1.0;  // Dimmer if contested

// Final heat for color mapping
const visualHeat = (baseHeat + activityBoost) * contestedFactor;

// For contested regions: apply pulsing animation
const isPulsing = cell.contested_rate > 0.15;
```

**Heat Components:**

| Component | Weight | Purpose |
|-----------|--------|---------|
| `log(sentence_count)` | Base | Density foundation |
| `× (1 + variance)` | 1.0-2.0x | Semantic diversity bonus |
| `+ activity_boost` | 0-0.3 | Recent additions glow |
| `× contested_factor` | 0.8-1.0 | Dim ambiguous regions |

### Counts vs "Density" (Why Cells Shrink)

RA/Dec are spherical coordinates. A fixed-size RA/Dec bucket (e.g., **10° × 10°**) does **not** cover a constant surface area on the sphere: longitudinal lines converge toward the poles, so the bucket's spherical area shrinks roughly with `cos(dec)`. This is spherical geometry, not a bug in Mollweide.

**Design choice (MVP):** we render **raw counts per bucket** (you want counts). "Hot" means *more sentences in this bucket*, not "sentences per unit area." This keeps the visualization honest about population and aligns with the "galaxy" metaphor: dense regions become prominent.

**Adaptive resolution vs geometry:** the adaptive voxel grid increases display
resolution where observations are dense. Subdivision is not a curvature estimate.
It is independent of RA/Dec bucket geometry; when desired, render heat from **leaf
voxels** (`binning: 'voxel_leaf'`) to show the persisted adaptive display hierarchy.

**If we ever want true density:** add an optional normalization (`count / cos(dec)`) or switch to an equal-area tessellation (e.g., HEALPix). Not required for the count-first MVP.

### Resolution & Level-of-Detail Behavior

**Auto-Resolution on Zoom:**

```typescript
const resolutionForZoom = (zoomLevel: number): number => {
  if (zoomLevel < 1.5) return 10;   // Overview: 10° cells
  if (zoomLevel < 3.0) return 5;    // Regional: 5° cells  
  if (zoomLevel < 6.0) return 2;    // Detailed: 2° cells
  return 1;                          // Maximum: 1° cells
};

// Smooth LOD transition (don't pop - interpolate)
const transitionDuration = 300; // ms
```

**LOD Loading Strategy:**
1. Load 10° resolution on initial render (648 cells, ~50KB)
2. Pre-fetch 5° resolution in background
3. On zoom: crossfade to higher resolution
4. Cache all loaded resolutions for instant zoom-out

### Empty State UX (Cold Start)

**Design Philosophy:** An empty meaning space is a feature, not a bug. It creates urgency to add data.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE MEANING SPACE                             │
│                                                                          │
│                                                                          │
│                            ┌─────────┐                                   │
│                            │    ●    │  ← Single voxel with              │
│                            │  0.01%  │    first 50 sentences             │
│                            └─────────┘                                   │
│                                                                          │
│                                                                          │
│                   "Your meaning space is just beginning"                 │
│                                                                          │
│           ┌──────────────────────────────────────────────┐              │
│           │  📄 Upload documents to expand your space    │              │
│           │  💬 Process more conversations               │              │
│           │  🎯 Goal: 5,000 sentences unlocks insights   │              │
│           └──────────────────────────────────────────────┘              │
│                                                                          │
│  Total: 47 sentences │ 12 clusters │ Space coverage: 0.01%              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Progressive Milestones:**

| Sentences | Visual State | Message |
|-----------|--------------|---------|
| 0-100 | Single dim voxel | "Your meaning space is just beginning" |
| 100-1,000 | 2-5 voxels lighting up | "Patterns emerging..." |
| 1,000-5,000 | Visible clusters forming | "Your semantic landscape is taking shape" |
| 5,000-10,000 | Clear hot/cold zones | "Rich meaning density achieved" |
| 10,000+ | Full visualization | "Enterprise-grade semantic coverage" |

**Gamification Elements (Optional):**
- Progress bar toward next milestone
- "Unlock" animations when features auto-enable
- Comparison to "typical enterprise" coverage (anonymized benchmarks)

### Real-Time Activity Stream (Liquid Glass Effect)

For the "Minority Report / Liquid Glass" vision, implement live activity visualization:

**Redis Activity Stream:**
```redis
# Add activity event
XADD {semantic_namespace_id}:clustering:activity MAXLEN ~1000 *
  type "cluster_join"
  visual_voxel_id "v_5_3_7"
  ra "45.2"
  dec "12.8"
  timestamp "2025-12-11T16:30:00Z"

# Subscribe to activity (WebSocket backend)
XREAD BLOCK 0 STREAMS {semantic_namespace_id}:clustering:activity $
```

**Frontend WebSocket Handler:**
```typescript
interface ActivityEvent {
  type: 'cluster_join' | 'new_cluster' | 'contested';
  visual_voxel_id: string;
  ra: number;
  dec: number;
  timestamp: string;
}

// On activity event:
const handleActivity = (event: ActivityEvent) => {
  const cell = findCellByCoords(event.ra, event.dec);
  
  // Trigger glow animation
  cell.glowIntensity = 1.0;
  cell.glowColor = event.type === 'new_cluster' ? '#FFBD1F' : '#FE904C';
  
  // Decay over 30 seconds
  animateGlowDecay(cell, 30000);
  
  // Ripple effect to neighbors
  cell.neighbors.forEach(n => {
    n.glowIntensity = 0.3;
    animateGlowDecay(n, 15000);
  });
};
```

**Visual Effects:**

| Event Type | Glow Color | Duration | Effect |
|------------|------------|----------|--------|
| `cluster_join` | Orange (#FE904C) | 30s decay | Pulse once, fade |
| `new_cluster` | Yellow (#FFBD1F) | 45s decay | Bright flash, slow fade |
| `contested` | Turquoise (#14C4ED) | 60s | Pulsing animation |
| `subdivision` | White (#FFFFFF) | 5s | Split animation |

### Deferred: 3D Display-Sphere Mode (After Locality Portal)

**Concept:** Render the same visual-only 3D PSA projection as an interactive sphere.
This is an alternate Global Atlas presentation, not the 115D calculation space and
not a Semantic Locality view. It follows the stable atlas and locality portal work.

```
        ┌─────────────────────────────────────────┐
        │         3D FIELD DISPLAY SPHERE         │
        │                                         │
        │              ╭─────────╮                │
        │           ╭──┤ ▓▓▓▓▓▓▓ ├──╮            │
        │          │   │ ▓▓███▓▓ │   │           │
        │          │   │ ▓██████▓ │   │           │
        │          │   │ ▓▓███▓▓ │   │           │
        │          │   │ ▓▓▓▓▓▓▓ │   │           │
        │           ╰──┤         ├──╯            │
        │              ╰─────────╯                │
        │                                         │
        │   [🔄 Rotate]  [🔍 Zoom]  [📍 Center]  │
        └─────────────────────────────────────────┘
```

**Implementation (Three.js + React Three Fiber):**

```typescript
const MeaningSphere: React.FC<{data: PSAHeatmapData}> = ({data}) => {
  return (
    <Canvas camera={{position: [0, 0, 3]}}>
      <OrbitControls enableDamping dampingFactor={0.05} />
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      
      {/* Sphere with heatmap texture */}
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <shaderMaterial
          uniforms={{heatmapTexture: {value: generateHeatmapTexture(data)}}}
          vertexShader={vertexShader}
          fragmentShader={heatmapFragmentShader}
        />
      </mesh>
      
      {/* Hot spot markers */}
      {data.hottest_voxels.map(voxel => (
        <HotSpotMarker key={voxel.visual_voxel_id} voxel={voxel} />
      ))}
    </Canvas>
  );
};
```

**Handling 4 Projection Views:**

| Option | Description | Recommendation |
|--------|-------------|----------------|
| **Single Sphere** | One versioned display projection | Simplest valid presentation |
| **4 Mini Spheres** | Versioned projections side-by-side | Diagnostic comparison only |
| **Blended Sphere** | Average across display projections | Not valid unless coordinates, bins, and aggregation semantics are registered and validated |
| **Toggle Views** | Switch between versioned projections | Power-user diagnostic |

**Navigation Controls:**
- **Drag**: Rotate sphere
- **Scroll**: Zoom in/out
- **Double-click**: Center on clicked region
- **Right-drag**: Pan (shift center)
- **Spacebar**: Reset to default view

---

## 🔧 Technical Implementation

### Option A: Pure d3.js (Recommended for MVP)

**Pros:**
- Neo4j Browser is built on d3.js - consistent ecosystem
- Rich mapping projection library (d3-geo)
- Excellent heatmap support
- No additional dependencies

**Implementation Stack:**
```
d3.js v7
├── d3-geo (projections: Mollweide, etc.)
├── d3-scale (color scales: plasma, viridis)
├── d3-contour (optional: smooth density contours)
└── d3-zoom (pan/zoom interactions)
```

**Sample Code Structure:**

```javascript
// PSAHeatmap.js
class PSAHeatmap {
  constructor(containerId, options = {}) {
    this.container = d3.select(`#${containerId}`);
    this.width = options.width || 1200;
    this.height = options.height || 600;
    this.projection = d3.geoMollweide()
      .scale(this.width / 5)
      .translate([this.width / 2, this.height / 2]);
    this.colorScale = d3.scaleSequentialLog(d3.interpolatePlasma)
      .domain([1, 10000]);
  }

  async loadData(apiUrl) {
    const response = await fetch(apiUrl);
    this.data = await response.json();
    this.render();
  }

  render() {
    // Create SVG
    const svg = this.container.append('svg')
      .attr('width', this.width)
      .attr('height', this.height);

    // Render heatmap cells
    svg.selectAll('rect')
      .data(this.data.cells)
      .enter()
      .append('rect')
      .attr('x', d => this.projection([d.ra, d.dec])[0])
      .attr('y', d => this.projection([d.ra, d.dec])[1])
      .attr('fill', d => this.colorScale(d.sentence_count))
      .attr('opacity', 0.8);
  }
}
```

### Option B: WebGL (For Large Scale)

**When to use:** If rendering 100k+ cells becomes sluggish with SVG

**Libraries:**
- **deck.gl** - High-performance WebGL visualization
- **regl** - Lower-level WebGL with functional API
- **Three.js** - If we add deferred 3D display-sphere rendering

### Option C: Neo4j Bloom Integration

**Concept:** Extend Neo4j Bloom with custom PSA visualization perspective

**Pros:**
- Native Neo4j integration
- Existing enterprise users may have Bloom licenses

**Cons:**
- Less flexibility
- Requires Bloom enterprise license

---

## 🚀 API Design

### Canonical Source: Consolidated Clustering V2

The PSA heatmap is served by `consolidated_clustering_v2` (FastAPI on port 8020), which already maintains:
- Per-semantic-namespace field state (S3 source of truth)
- Redis caches for voxel heat + hot voxels
- Direct endpoints: `/api/psa/heatmap`, `/api/psa/state`

This keeps the MVP read path **direct** (no MCP pub/sub overhead) while using the same state engine that produces PSA coordinates.

### API Gateway Routing (aiidea API service)

Expose `/api/psa/*` through the normal frontend base URL by **proxying** requests to the clustering service. This avoids Neo4j aggregation in the API service for the heatmap itself.

### Endpoint: PSA Heatmap

```
GET /api/psa/heatmap
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `"grid"` | `grid` \| `hot_voxels` |
| `resolution` | int | 10 | Degrees per cell (10°, 5°, 2°, 1°) (grid only) |
| `min_count` | int | 1 | Minimum sentence count per cell (grid only) |
| `metric` | string | `"sentences"` | `sentences` \| `paraphrases` \| `clusters` \| `heat` |
| `binning` | string | `"ra_dec"` | `ra_dec` \| `voxel_leaf` |
| `limit` | int | 50 | Max hot voxels to return (hot_voxels only) |
| `projection_id` | int | 0 | Projection index to render (if multiple projections exist) |

**Note:** No session filtering - the response shows the complete selected tenant/workspace semantic namespace.

**Response:**

```json
{
  "tenant_id": "boeing",
  "workspace_id": "research",
  "semantic_namespace_id": "ns_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "timestamp": "2025-12-09T14:30:00Z",
  "field": {
    "tenant_id": "boeing",
    "workspace_id": "research",
    "semantic_namespace_id": "ns_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  },
  "projection": {
    "embedding_model_id": "text-embedding-3-large@2025-11-01",
    "projection_id": 0,
    "projection_hash": "sha256:…",
    "guidestar_set_id": "semantic_guide_catalog_115@v1",
    "alignment_id": "display_frame_native@v1"
  },
  "grid": {
    "resolution_degrees": 10,
    "ra_bins": 36,
    "dec_bins": 18,
    "binning": "ra_dec"
  },
  "cells": [
    {
      "ra": 45.0,
      "dec": 12.5,
      "sentence_count": 1523,
      "paraphrase_count": 89,
      "cluster_count": 12,
      "heat": 3.45
    }
  ],
  "metrics": {
    "total_sentences": 161000000,
    "total_paraphrases": 2400000,
    "total_clusters": 850000
  },
  "hottest_voxels": []
}
```

### Endpoint: PSA Cell Detail

```
GET /api/psa/cell/{ra}/{dec}
```

**Response:**

```json
{
  "tenant_id": "boeing",
  "workspace_id": "research",
  "semantic_namespace_id": "ns_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "timestamp": "2025-12-09T14:30:00Z",
  "cell": {
    "ra": 45.0,
    "dec": 12.5,
    "sentence_count": 1523,
    "paraphrase_count": 89,
    "cluster_count": 12
  },
  "paraphrase_groups": [
    {
      "group_id": "pg_abc123",
      "representative_text": "Machine learning optimizes supply chains",
      "member_count": 17,
      "routing_cluster_label": "cluster_45"
    }
  ],
  "top_sentences": [
    {
      "node_id": "sent_123",
      "text": "AI-driven logistics reduces shipping costs by 23%",
      "similarity_to_center": 0.94
    }
  ]
}
```

---

## 📱 UI Integration

### Frontend Location

**Repo:** sibling `spark_hub` repository
**Stack:** React + TypeScript + Vite + d3.js

The frontend already has:
- `src/components/visualizations/` - Existing viz components (ForceDirectedGraph3D, etc.)
- `src/pages/VisualizationPage.tsx` - Dedicated visualization page pattern
- `src/services/` - API communication patterns
- `@react-three/fiber` - 3D rendering (for future 3D mode)

### New Files to Create

```
spark_hub/src/
├── pages/
│   └── PSAHeatmapPage.tsx          # Full-page heatmap view
├── components/
│   └── visualizations/
│       └── PSAHeatmap.tsx          # Core d3.js heatmap component
├── services/
│   └── psaService.ts               # API calls to /api/psa/heatmap
└── types/
    └── psa.ts                      # PSA-specific TypeScript types
```

### Button Placement

Add "View Meaning Space" button to the header (alongside existing controls):

```tsx
// In ChatHeader.tsx or similar
<Button 
  onClick={() => navigate('/meaning-space')}
  variant="outline"
  className="bg-transparent border-turquoise text-turquoise hover:bg-turquoise hover:text-white"
>
  <Globe className="w-4 h-4 mr-2" />
  View Meaning Space
</Button>
```

### Route Addition

```tsx
// In App.tsx
<Route path="/meaning-space" element={<PSAHeatmapPage />} />
```

**AuthGuard required** - this is sensitive enterprise IP. Keep existing auth/RBAC; default access is read-only within the user's permitted meaning field(s).

### Full-Page Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  aiidea     [← Back]                      [Resolution: 10° ▼] [Export]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                     ENTERPRISE MEANING SPACE                             │
│                                                                          │
│            ░░░▒▒▓▓██▓▓▒▒░░░                                             │
│         ░░▒▒▓▓████████▓▓▒▒░░                                            │
│        ░▒▒▓▓██████████████▓▓▒▒░                                         │
│       ▒▒▓▓████████████████▓▓▒▒                                          │
│       ▓▓████                ████▓▓                                       │
│       ▒▒▓▓████████████████▓▓▒▒                                          │
│        ░▒▒▓▓██████████████▓▓▒▒░                                         │
│         ░░▒▒▓▓████████▓▓▒▒░░                                            │
│            ░░░▒▒▓▓██▓▓▒▒░░░                                             │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ░ 1-10  ▒ 10-100  ▓ 100-1k  █ 1k+   │   161M sentences │ 2.4M ideas   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📈 Performance Considerations

### Data Volume Estimates

| Scale | Cells (10°) | Cells (5°) | Cells (1°) | Data Size |
|-------|-------------|------------|------------|-----------|
| Small (100k sentences) | 648 | 2,592 | 64,800 | ~50KB |
| Medium (10M sentences) | 648 | 2,592 | 64,800 | ~100KB |
| Large (161M sentences) | 648 | 2,592 | 64,800 | ~200KB |

Cell count is fixed by resolution, not sentence count. Data size scales modestly.

### Caching Strategy

```
1. Full heatmap cache (Redis)
   - TTL: 5 minutes
   - Key: {semantic_namespace_id}:clustering:psa:heatmap:{projection_id}:{binning}:{metric}:{resolution}
   
2. Cell detail cache (Redis)
   - TTL: 1 minute
   - Key: {semantic_namespace_id}:clustering:psa:cell:{projection_id}:{resolution}:{ra}:{dec}

3. Pre-compute on data change
   - Trigger: sentence_node_service publishes new sentences
   - Background job updates cached heatmap
```

### Rendering Optimization

1. **Level of Detail (LOD)**
   - Start with 10° resolution
   - Lazy-load finer resolution on zoom

2. **Canvas vs SVG**
   - <1000 cells: SVG (for interactivity)
   - >1000 cells: Canvas (for performance)

3. **Viewport Culling**
   - Only render visible cells when panned/zoomed

---

## 🗓️ Implementation Phases

The numbered 2025 phases below are retained as implementation history. Their unchecked
items are not the current priority order. The current follow-on sequence is:

1. **Stabilize the Global Atlas**: pin the frame and scale, preserve unsnapped
   coordinates, and expose projection/version fidelity.
2. **Build the Signed Signature-Space Locality Portal**: exact sentence/group-medoid
   centers, versioned tangent bases, correspondence links, and empirical transition
   marks.
3. **Validate shapes and evidence states**: compare ball, tube, subspace, lobe, and
   truncated-cone hypotheses with held-out operators and controls.
4. **Add Display Sensitivity as an expert diagnostic**: analytic mapper Jacobian,
   SVD metrics, active-set boundaries, and finite-difference validation.
5. **Consider fisheye and 3D display modes** only after projection stability and the
   applicable non-folding or registration gates pass.

### Historical Phase 1: MVP (Static Heatmap + State Management)

**What's Already Done:**
- ✅ PSA coordinates computed (`ra`, `dec`, `radius`) in clustering service
- ✅ Voxel assignment (`visual_voxel_id`) for every sentence
- ✅ `sentence_count` tracked per voxel (ALL sentences, not just paraphrased)
- ✅ Heat calculation (Enhanced formula with activity component)
- ✅ FieldStateManager with S3 persistence and Redis caching
- ✅ `/api/psa/heatmap` and `/api/psa/state` endpoints in clustering service
- ✅ Auto-enable thresholds for adaptive features

**Backend Tasks:**
- [ ] Ensure clustering service heatmap endpoint is accessible via API gateway
- [ ] Add WebSocket events for real-time activity stream
- [ ] Implement cell detail endpoint for click-through exploration
- [ ] Add aggregate_cells function for resolution binning

**Frontend Tasks:**
- [ ] Create `PSAHeatmapPage.tsx` with full-page layout
- [ ] Create `PSAHeatmap.tsx` d3.js component with Mollweide projection
- [ ] Implement enhanced heat formula color mapping
- [ ] Add "View Meaning Space" button to header
- [ ] Add `/meaning-space` route in App.tsx
- [ ] Create `psaService.ts` for API calls
- [ ] Implement empty state UX with progress milestones
- [ ] Handle LOD auto-resolution based on zoom level

**Timeline:** 3-4 days

### Historical Phase 2: Interactive Exploration + Activity Stream

**Deliverables:**
- [ ] Hover tooltips (Enhanced Heat Formula breakdown)
- [ ] Click to select region → detail panel with cluster list
- [ ] Zoom/pan controls with auto-resolution adjustment
- [ ] Resolution selector (10°, 5°, 1°) - manual override
- [ ] Loading states and error handling
- [ ] Real-time activity sidebar (WebSocket updates)
- [ ] Activity stream Redis subscription
- [ ] "Pulse" effect on cells with recent activity

**Timeline:** 4-5 days

### Historical Phase 3: 3D Mode + Advanced Features (Superseded Priority)

**Deliverables:**
- [ ] 3D display-sphere mode toggle (Three.js/globe.gl), deferred until after the Locality Portal
- [ ] 4 projection views: Mollweide, Aitoff, Equirectangular, Lambert
- [ ] Texture mapping heat data onto sphere
- [ ] WebGL renderer for scale
- [ ] Time-lapse animation (watch meaning space evolve)
- [ ] Export to PNG/SVG/interactive
- [ ] **Agent trail overlay** (show where agents have navigated)

**Timeline:** 5-7 days

### Historical Phase 4: Analytics Dashboard Integration

**Deliverables:**
- [ ] Feature state dashboard (show auto-enable progress)
- [ ] Threshold progress bars per feature
- [ ] Historical heat evolution charts
- [ ] Cluster growth trends
- [ ] Export heatmap data to CSV
- [ ] Shareable heatmap links with state

**Timeline:** 3-4 days

---

## 🔧 Technical Implementation Details

### Option A: Query Clustering Service Directly (Recommended for MVP)

The `consolidated_clustering_v2` service already has all the data in memory. Add an endpoint:

```python
# In consolidated_clustering_v2/service.py

@app.get("/api/voxels/heatmap")
async def get_heatmap_data(resolution: int = 10):
    """
    Return aggregated heatmap data from in-memory voxel state.
    """
    cells = []
    
    # Get all leaf voxels from adaptive grid
    for visual_voxel_id, voxel in self.router.voxels.items():
        if not voxel.is_leaf:
            continue
        
        # Convert voxel center to RA/Dec
        center_x, center_y, center_z = voxel_center_coords(voxel)
        ra, dec, _ = self.router.projection_manager.to_ra_dec(center_x, center_y, center_z)
        
        # Bin to requested resolution
        ra_bucket = floor(ra / resolution) * resolution
        dec_bucket = floor(dec / resolution) * resolution
        
        cells.append({
            "ra": ra_bucket,
            "dec": dec_bucket,
            "sentence_count": voxel.sentence_count,
            "cluster_count": voxel.cluster_count,
            "heat": voxel.heat,
            "visual_voxel_id": visual_voxel_id,
            "level": voxel.level
        })
    
    # Aggregate cells at same RA/Dec bucket
    return aggregate_cells(cells, resolution)
```

### Option B: Persist to Neo4j (For Recovery/History)

Store RA/Dec on sentence nodes for persistence across restarts:

```cypher
// When storing sentence (add to existing write)
CREATE (s:Sentence {
  node_id: $node_id,
  text: $text,
  // ... existing properties ...
  ra: $ra,
  dec: $dec,
  radius: $radius,
  visual_voxel_id: $visual_voxel_id
})
```

### Heatmap Aggregation (API Response)

```python
def aggregate_cells(cells: List[dict], resolution: int) -> dict:
    """Aggregate voxel data into heatmap bins."""
    bins = defaultdict(lambda: {
        "sentence_count": 0,
        "cluster_count": 0,
        "heat_sum": 0,
        "voxel_count": 0
    })
    
    for cell in cells:
        key = (cell["ra"], cell["dec"])
        bins[key]["sentence_count"] += cell["sentence_count"]
        bins[key]["cluster_count"] += cell["cluster_count"]
        bins[key]["heat_sum"] += cell["heat"]
        bins[key]["voxel_count"] += 1
    
    return {
        "cells": [
            {
                "ra": ra,
                "dec": dec,
                "sentence_count": data["sentence_count"],
                "cluster_count": data["cluster_count"],
                "avg_heat": data["heat_sum"] / data["voxel_count"]
            }
            for (ra, dec), data in bins.items()
        ],
        "total_sentences": sum(c["sentence_count"] for c in cells),
        "total_voxels": len(cells)
    }
```

### d3.js Mollweide Implementation

```typescript
// PSAHeatmap.tsx
import * as d3 from 'd3';
import { geoMollweide, geoPath } from 'd3-geo';

const AIIDEA_COLORS = {
  coldest: '#0099BF',
  cold: '#14C4ED',
  cool: '#1AC8DB',
  warm: '#FFBD1F',
  hot: '#FE904C',
  hottest: '#FFFFFF',
};

// Create color scale using aiidea palette
const colorScale = d3.scaleLog()
  .domain([1, 10, 100, 1000, 10000, 100000])
  .range([
    AIIDEA_COLORS.coldest,
    AIIDEA_COLORS.cold,
    AIIDEA_COLORS.cool,
    AIIDEA_COLORS.warm,
    AIIDEA_COLORS.hot,
    AIIDEA_COLORS.hottest,
  ])
  .interpolate(d3.interpolateRgb);

// Mollweide projection (RA 0-360 → longitude -180 to 180)
const projection = geoMollweide()
  .scale(width / 5)
  .translate([width / 2, height / 2])
  .rotate([-180, 0]); // Center on RA=180°
```

---

## ✅ Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Global Atlas projection** | Versioned Mollweide display chart | Equal-area presentation for raw bucket counts; visual-only |
| **Rendering** | d3.js (MVP) → WebGL (scale) | Start simple, optimize later |
| **Color Scale** | aiidea brand gradient | Turquoise → Orange (brand consistent) |
| **Default Resolution** | 10° | Balances detail with performance |
| **UI Entry** | Header button → Full Page | Maximum viz real estate |
| **Data Source** | Clustering service in-memory | RA/Dec already computed! |
| **API Pattern** | Direct route (bypass MCP) | API gateway proxies to clustering service (plain HTTP, no pub/sub) |
| **Session Scope** | None (per meaning field) | Global view within one tenant/workspace semantic namespace |
| **Coordinate frame** | Fixed 115-guide catalog plus exact version metadata | Reproducible within a frame; upgrades require held-out alignment gates |
| **Sentence Counting** | ALL sentences | `voxel.sentence_count` includes everything |
| **Locality center** | Exact sentence or actual-member group medoid | Defines one valid point and tangent space in the signed signature carrier |
| **Calculation-space view** | Signed 115D Locality Portal | Empirical tangent transitions before shape or cone claims |
| **Display Sensitivity** | Secondary analytic mapper diagnostic | Explains 3D movement; does not assert reachability |
| **Semantic protocol** | Coordinates plus exact frame, calibration, and provenance | Versioned addressing, not self-decoding meaning |
| **Coverage and uncertainty** | Separate descriptive, contested, and calibrated metrics | Counts and guide proximity do not equal semantic confidence |
| **3D sphere and fisheye** | Deferred | Portal first; non-folding/registration gates precede deformation |

---

## 🌐 Semantic Coordinate Protocol Summary

PSA coordinates are compact addresses inside an exact, versioned, model-aligned
display frame. A compatible receiver may use them to locate authorized evidence and
route candidates, but the address alone neither decodes meaning nor proves semantic
equivalence. The active frame uses the fixed 115-guide catalog; payload sizes must be
measured per encoding rather than inherited from the historical sketch.

This protocol is not encryption, anonymization, or a universal meaning map. Field
cells, density, coordinates, signatures, and access patterns may all be sensitive.
Interoperability and cross-model local-structure transport require held-out validation,
explicit governance, and versioned provenance.

---

## 🔮 Future Vision: Agent Navigation Overlay

Once agents are operational, overlay their trails on the meaning space:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEANING SPACE + AGENT TRAILS                         │
│                                                                          │
│            ░░░▒▒▓▓██▓▓▒▒░░░                                             │
│         ░░▒▒▓▓████████▓▓▒▒░░                                            │
│        ░▒▒▓▓██●──●──●█████▓▓▒▒░       ← Agent trail                     │
│       ▒▒▓▓████  │  │  ████▓▓▒▒                                          │
│       ▓▓████    ●──●    ████▓▓        ● = Agent visited                 │
│       ▒▒▓▓████      ████▓▓▒▒          ─ = Navigation path               │
│        ░▒▒▓▓██████████▓▓▒▒░                                             │
│         ░░▒▒▓▓████████▓▓▒▒░░                                            │
│            ░░░▒▒▓▓██▓▓▒▒░░░                                             │
│                                                                          │
│  Legend: ● Current agent position  ○ Past visits  ─ Trail               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚨 Implementation Checklist

### Backend (aiidea repo)

1. **Clustering Service Updates** (`app/consolidated_clustering_v2/`)
   - [x] PSA coordinate computation - `get_ra_dec()` exists
   - [x] Voxel assignment - `visual_voxel_id` assigned
   - [x] Sentence counting - `voxel.sentence_count` tracks ALL sentences
   - [x] `/api/psa/heatmap` endpoint - Returns metrics + hot voxels (extend to return grid cells)
   - [x] `/api/psa/state` endpoint - Returns feature states and auto-enable progress
   - [x] FieldStateManager - S3 persistence + Redis caching
   - [x] Auto-enable thresholds - Data-driven feature activation
   - [ ] Add `format=grid` aggregation (`resolution`, `binning`) for heatmap cells
   - [ ] WebSocket events for real-time activity stream

2. **API Service Updates** (`app/mcp_servers/api_service/`)
   - [ ] Route `/api/psa/heatmap` to clustering service
   - [ ] Route `/api/psa/state` to clustering service
   - [ ] Add CORS headers for frontend access

3. **Optional: Neo4j Persistence** (`app/mcp_servers/database_service/`)
   - [ ] Add `ra`, `dec`, `radius`, `visual_voxel_id` to Sentence node writes
   - [ ] For recovery after clustering service restart

### Frontend (`spark_hub` repo)

1. **New Components**
   - [ ] `src/pages/PSAHeatmapPage.tsx` (with empty state milestone UX)
   - [ ] `src/components/visualizations/PSAHeatmap.tsx` (d3.js Mollweide)
   - [ ] `src/components/visualizations/PSASphere.tsx` (deferred visual-only 3D display mode)
   - [ ] `src/components/ActivityStream.tsx` (real-time sidebar)
   - [ ] `src/services/psaService.ts`
   - [ ] `src/types/psa.ts`

2. **Integration**
   - [ ] Add route to `App.tsx`
   - [ ] Add "View Meaning Space" button to header
   - [ ] Add d3-geo dependency to package.json
   - [ ] Add WebSocket subscription for activity stream
   - [x] Consume `visual_voxel_id` and `routing_cluster_label` directly; remove retired visual-field aliases
   - [ ] LOD auto-resolution on zoom

---

## 📊 Existing Data Structure Reference

From `consolidated_clustering_v2/routing/adaptive_grid.py`:

```python
@dataclass
class AdaptiveVoxel:
    visual_voxel_id: str
    level: int  # 0-3 (adaptive resolution)
    
    # Contents
    cluster_ids: Set[str]
    sentence_count: int = 0      # ALL sentences, not just paraphrased!
    cluster_count: int = 0
    
    # Anisotropy (heat = semantic density × variance)
    heat: float = 0.0
    mean_cosine_variance: float = 0.0
```

From `cluster_assignment_completed` event:

```python
event_payload = {
    # Navigation coordinates (ALREADY COMPUTED!)
    "ra": ra,                    # Right Ascension (0-360°)
    "dec": dec,                  # Declination (-90° to +90°)
    "radius": semantic_depth,    # Semantic depth from center
    
    # Voxel info
    "visual_voxel_id": visual_voxel_id,
    "is_dense_voxel": is_dense,
    
    # ... other fields
}
```

---

**🧠 THINK TWICE BEFORE CODING ONCE**

This design prioritizes:
1. **Visual impact** - "Utterly cool, so beautiful" with aiidea brand colors
2. **Meaning density** - ALL sentences counted, not just paraphrased ones
3. **Simplicity** - No prompt needed, just a button, no session filtering
4. **Reuse** - PSA coordinates already computed by clustering service!
5. **Extensibility** - Foundation for future agent navigation overlay
6. **Brand consistency** - Uses aiidea color palette throughout
