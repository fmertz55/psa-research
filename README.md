# PSA Research

Research record for **Pixelization of Semantic Anisotropy (PSA)** and its
anchor-triangulated semantic signatures: text representations measured as
tier-weighted cosines against a fixed catalog of propositional anchor sentences
("guide stars"), plus a calibrated low-dimensional index used for semantic
blocking during knowledge-graph ingestion.

Author: Douglas Clark (aiidea)

## Paper

**Anchor-Triangulated Semantic Signatures: Frame-Robust Neighborhood Structure
with Tunable Semantic Blocking**

- Named preprint: [`paper/latex/Clark_PSA_anchor_signatures_preprint.pdf`](paper/latex/Clark_PSA_anchor_signatures_preprint.pdf)
  (source: `paper/latex/psa_semantic_field_preprint.tex`)
- Anonymous review version: `paper/latex/psa_semantic_field.tex` / `.pdf`
- Industry-track variants: `paper/latex/psa_semantic_field_industry_2026*.tex`
- Long-form manuscript and outline: `paper/psa_semantic_field_paper.md`,
  `paper/psa_semantic_field_outline.md`
- Claim provenance (which artifact supports each number): `paper/PSA_CLAIM_PROVENANCE_2026-07-28.md`

Build the preprint:

```bash
cd paper/latex
latexmk -pdf psa_semantic_field_preprint.tex
```

## Layout

| Folder | Contents |
|---|---|
| `paper/` | Manuscript sources, triage, and claim provenance |
| `programs/` | Experimental programs, pre-run checklists, corpus configs |
| `reports/` | Benchmark results (PSA, MRPC paraphrase, Qwen embedding field tests); `reports/quarantine/` holds results withdrawn from claims |
| `scripts/` | Experiment and analysis scripts (`psa/`, `mrpc/`, `neo4j/`) |
| `tests/` | Unit tests for PSA coordinates, calibration, and benchmark analysis |
| `docs/` | Design notes and paraphrase-evaluation audits, including preregistrations |
| `designs/` | Exploratory design notes |

## Dependencies and status

These scripts were developed inside the private aiidea codebase. Most import
the production PSA modules (`app.consolidated_clustering_v2.routing.psa_*`) and
read embeddings from a Neo4j graph, so they **do not run standalone** from this
repository. They are included as the exact research record behind the reported
results. A standalone evaluation harness is future work.

Credentials are read from environment variables (for example `NEO4J_PASSWORD`)
or AWS Systems Manager; none are stored here.

## Patent notice

Aspects of PSA are the subject of a pending U.S. provisional patent
application. Nothing in this repository grants a patent license. No
open-source license is attached; all rights reserved.
