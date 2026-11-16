# Governed Belief-State Compilation With MicroGPT Sidecars

**Document type:** Research design  
**Reviewed:** August 3, 2026  
**Status:** Conceptual; shadow/assistive implementation and evaluation remain

## Thesis
aiidea should not retrieve context for an LLM. It should compile governed belief states that an LLM is allowed to speak from.

The current Stitch path sends Qwen a curated subgraph, mostly as structured JSON. That is useful, but it makes Qwen parse graph structure, infer relevance, preserve provenance, identify contradictions, and generate final language all at once.

This design adds one or two MicroGPT sidecars beside Stitch. They do not replace Stitch. They enrich Stitch output with compact, auditable belief-state artifacts.

**Goals**
- Preserve the existing Stitch-to-Qwen path.
- Add MicroGPT sidecars that produce measurable semantic value.
- Convert graph fragments into ranked, sourced, confidence-scored belief cards.
- Keep every MicroGPT claim reversible to graph ids, sentence ids, source ids, ontology ids, and switch traces.
- Measure whether MicroGPT improves faithfulness, coverage, citation quality, and compression.
- Keep latency controlled by starting in shadow/assistive mode.

**Non-Goals**
- Do not replace Stitch.
- Do not fine-tune Qwen initially.
- Do not make MicroGPT an autonomous agent.
- Do not let MicroGPT produce unsourced summaries.
- Do not remove JSON internally. JSON remains transport and audit format.

**Current Flow**
```text
User question
  ↓
Retrieval / graph query
  ↓
Stitch traversal and curation
  ↓
Curated subgraph JSON
  ↓
Qwen generates answer
```

**Proposed Flow**
```text
User question
  ↓
Stitch traversal and curation
  ↓
Curated subgraph JSON
  ↓
MicroGPT Belief Compiler sidecar
  ↓
Belief briefing + constraints + provenance
  ↓
Qwen generates answer
  ↓
MicroGPT Critic / verifier
  ↓
Bayesian switch and compression feedback
```

**Core Artifact: Belief Briefing**
A belief briefing is the compressed, governed interface between graph-state and model-state.

It should contain:
- Ranked claims
- Supporting evidence ids
- Source provenance
- Ontology entities and frames
- Semantic roles
- Confidence scores
- Contradiction notes
- Uncertainty notes
- Required claims
- Optional background claims
- Forbidden inferences
- Citation requirements
- Tool affordances, when relevant

Example shape:

```text
Belief Card
Claim: Brown v. Board is central to the user’s legal research request.
Support: strong
Evidence ids: [s1, o1, e1, frame1]
Sources: [legal.md]
Ontology: [LegalDocument, CourtCase]
Frame: LegalResearch
Use: required
Uncertainty: none
Forbidden inference: do not infer procedural posture unless supported
```

**Sidecar 1: MicroGPT Belief Compiler**
Input:
- User question
- Stitch curated subgraph
- Selected sentences
- Switch traces
- Ontology entities
- Frames / frame instances
- Tool and skill nodes
- Provenance metadata

Output:
- Belief briefing
- Claim cards
- Required answer constraints
- Confidence and uncertainty
- Graph backlinks
- Compression metadata

Primary job:
- Convert graph fragments into semantic authority.

This model should not answer the user. It should compile what the system knows, what it does not know, and what Qwen is allowed to say.

**Sidecar 2: MicroGPT Critic / Governor**
Input:
- User question
- Belief briefing
- Qwen answer
- Curated subgraph references

Output:
- Faithfulness score
- Unsupported claims
- Missing required claims
- Citation gaps
- Contradiction warnings
- Overreach warnings
- Suggested reward/update signals

Primary job:
- Judge whether Qwen stayed inside the governed belief state.

**Operating Modes**
**Phase 1: Shadow**
MicroGPT reads Stitch output and emits belief briefings. Qwen does not see them.

Measure:
- Claim quality
- Provenance retention
- Compression ratio
- Contradiction detection
- Latency

**Phase 2: Assistive**
Qwen receives current Stitch context plus MicroGPT belief briefing.

Measure:
- Answer quality delta
- Faithfulness delta
- Citation quality
- Token reduction potential
- User-visible latency

**Phase 3: Gated**
If belief cards pass confidence and provenance checks, Qwen is instructed to prefer the belief briefing over raw graph JSON.

Raw subgraph remains available as fallback/debug context.

**Phase 4: Primary**
Qwen receives mostly governed belief state. Raw JSON becomes internal, fallback, or audit-only.

**Reversibility Requirement**
Every MicroGPT output must be traceable.

Each belief card must carry:
- `sentence_ids`
- `node_ids`
- `entity_ids`
- `frame_ids`
- `source_ids`
- `switch_trace_ids`
- `confidence`
- `compiler_model_id`
- `compiler_version`

No backlink means the claim is not authoritative.

**Latency Strategy**
Start with one Belief Compiler sidecar. Add Critic second.

Keep budgets strict:
- Belief Compiler target: sub-second to low-single-digit seconds
- Critic target: async or post-answer when possible
- Max belief briefing size: bounded by answer complexity
- No extra database lookups in the first version
- Consume only what Stitch already produced

The initial value should come from better compression and ordering, not deeper retrieval.

**Evaluation Metrics**
- Compression ratio: curated subgraph tokens to belief briefing tokens
- Provenance retention: percent of claims with valid backlinks
- Faithfulness: unsupported claims in final answer
- Coverage: required claims included in final answer
- Citation precision: answer citations map to supporting evidence
- Contradiction detection: known conflicting evidence surfaced
- Latency added
- User rating / thumbs feedback
- Critic agreement with human review
- Switch route improvement over time

**Bayesian Feedback Loop**
The Critic output should feed governance, not just logging.

Possible updates:
- Increase weight for switch paths that produced high-faithfulness answers
- Decrease weight for noisy ontology paths
- Adjust claim compression templates
- Adjust evidence budgets by query type
- Learn when MicroGPT briefing helps versus when raw graph context is enough
- Improve external tool routing based on successful belief/tool alignments

**Key Risk**
The MicroGPT layer can become a second hallucination surface.

Mitigation:
- No unsourced claims
- Hard provenance requirement
- Confidence calibration
- Shadow mode first
- Critic checks against graph ids
- Store every briefing for audit and training

**Design Principle**
MicroGPT should not summarize Stitch.

MicroGPT should compile semantic authority from Stitch.

Qwen should not reason over a raw graph. Qwen should speak from a governed belief state.
