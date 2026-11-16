# PSA Field Construction

## A. Build the canonical semantic signature

- Embed every sentence and the fixed 115-guide-star catalog in the same native
  model frame.
- Compute the 115 cosine similarities, row-center the vector, and L2-normalize
  it. The resulting signature is the canonical ANN and prototype-routing input.
- Require an exact tenant, workspace, semantic namespace, model, and frame
  match. There is no cross-frame runtime fallback.

## B. Resolve the semantic class

- Retrieve the top 32 namespace-scoped signature centroids from Neo4j ANN.
- Rerank the top 8 candidates against each class centroid, actual-member
  medoid, and two boundary prototypes.
- Make the final decision with the persisted 4096-dimensional native embedding
  and a cosine threshold of 0.77.
- Create a singleton semantic class when no existing class passes the final
  threshold. Sentences never enter a free-range holding population.

## C. Project and persist the visual field

- Map the sentence into 3D PSA coordinates for visualization only; 3D voxels
  do not decide semantic membership.
- Place the sentence in the adaptive visual hierarchy and preserve its original
  visual voxel identity through subdivision and replay.
- Persist namespace-scoped field state and all-node hierarchy snapshots in
  Redis and S3. If visual state is unavailable, field APIs fail closed while
  the durable semantic handoff remains authoritative.

## Key code touchpoints

- Signature construction and prototypes:
  `app/mcp_servers/sentence_node_service/semantic_signature.py`
- Routing constants and decision contract:
  `app/mcp_servers/mcp_lib/semantic_routing_contract.py`
- Semantic resolution:
  `app/mcp_servers/sentence_node_service/service.py`
- 3D visualization projection:
  `app/consolidated_clustering_v2/routing/psa_mapper.py`
- Adaptive visual hierarchy:
  `app/consolidated_clustering_v2/routing/adaptive_grid.py`
- Namespace-scoped visual state:
  `app/consolidated_clustering_v2/persistence/field_state.py`
- Durable hierarchy snapshots:
  `app/consolidated_clustering_v2/persistence/cluster_snapshot.py`
