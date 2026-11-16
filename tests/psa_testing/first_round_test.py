"""
First-Round Scientific Test Orchestrator for PSA Semantic Field

Central hypothesis: PSA coordinates induce measurable locality — regions close
in PSA space exhibit more similar retrieval, scoring, and traversal behavior
than regions farther apart, and this effect disappears under null models.

Architecture: This is a thin client that uses the same API endpoints as the
Spark Hub. All heavy compute stays on deployed ECS services and EC2 GPU
instances. The heatmap endpoint provides voxel snapshots. Neo4j provides
paraphrase membership for the consolidation metric. Everything else is pure
analysis.

Usage:
    python -m tests.psa_testing.first_round_test --run-id $(date +%Y%m%d-%H%M%S)
    python -m tests.psa_testing.first_round_test --run-id test1 --skip-ingest
    python -m tests.psa_testing.first_round_test --run-id test1 --step snapshot

Environment variables (or CLI args):
    API_URL          — https://api.aiidea.biz (default)
    SSE_STREAM_URL   — Lambda Function URL for SSE streaming
    UPLOAD_API_URL   — File upload API endpoint
    NEO4J_URI        — bolt://neo4j.mcp-cluster.local:7687
    NEO4J_PASSWORD   — or fetched via SSM
    AUTH_TOKEN        — Cognito JWT (or --auth-token)
"""

import argparse
import asyncio
import colorsys
import copy
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("psa_test")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INGESTION_PRODUCTS_DIR = Path(__file__).resolve().parents[2] / "ingestion_products"

DEFAULT_API_URL = "https://api.aiidea.biz"
DEFAULT_NEO4J_URI = "bolt://neo4j.mcp-cluster.local:7687"
SSM_NEO4J_PASSWORD_KEY = "/example/neo4j/password"

QUERY_SET = [
    {"label": "definitional", "text": "What is artificial intelligence?"},
    {"label": "agent", "text": "What is an AI agent?"},
    {"label": "readiness", "text": "What does an AI readiness framework evaluate?"},
    {"label": "bridging", "text": "How do AI agents relate to organizational AI readiness?"},
    {"label": "verifier", "text": "What evidence in the corpus supports this claim?"},
    {"label": "stress", "text": "What can the system not support from this corpus?"},
]

LOCALITY_P_THRESHOLD = 0.05  # Binary decision gate
N_PERMUTATIONS = 1000  # Permutation test iterations
VOXEL_LUMINANCE_ANCHOR = 10_000  # Fixed absolute scale
SSE_TIMEOUT_S = 300  # 5 min max wait for SSE completion
HEATMAP_LIMIT = 500  # Max voxels from heatmap endpoint


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------


def fetch_neo4j_password_via_ssm(
    parameter_name: str = SSM_NEO4J_PASSWORD_KEY,
) -> Optional[str]:
    """Fetch Neo4j password from AWS SSM Parameter Store."""
    try:
        value = subprocess.check_output(
            [
                "aws",
                "ssm",
                "get-parameter",
                "--name",
                parameter_name,
                "--with-decryption",
                "--query",
                "Parameter.Value",
                "--output",
                "text",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTTP client — same endpoints Spark Hub uses
# ---------------------------------------------------------------------------


class AIIdeaClient:
    """Thin async client for the aiidea REST + SSE API."""

    def __init__(self, api_url: str, sse_url: str, upload_url: str, auth_token: str):
        self.api_url = api_url.rstrip("/")
        self.sse_url = sse_url.rstrip("/") if sse_url else ""
        self.upload_url = upload_url.rstrip("/") if upload_url else ""
        self.auth_token = auth_token
        self._session: Optional[Any] = None

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
        }
        if extra:
            h.update(extra)
        return h

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    # -- Session management --

    async def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """GET /session — returns session_id."""
        await self._ensure_session()
        headers = self._headers()
        if session_id:
            headers["x-session-id"] = session_id
        async with self._session.get(f"{self.api_url}/session", headers=headers) as resp:
            data = await resp.json()
            body = data if "session_id" in data else data.get("body", data)
            return body["session_id"]

    # -- File upload (presigned URL flow) --

    async def upload_file(self, file_path: Path) -> dict:
        """Upload a file via the presigned URL endpoint, then return the S3 key."""
        await self._ensure_session()
        # Step 1: get presigned URL
        payload = {
            "fileName": file_path.name,
            "fileSize": file_path.stat().st_size,
            "contentType": _mime_type(file_path),
        }
        async with self._session.post(
            f"{self.upload_url}/presigned-url",
            headers=self._headers(),
            json=payload,
        ) as resp:
            presigned = await resp.json()
            body = presigned if "uploadUrl" in presigned else presigned.get("body", presigned)
            upload_url = body["uploadUrl"]
            file_key = body["fileKey"]
            if body.get("uploadMethod") != "POST" or not isinstance(body.get("uploadFields"), dict):
                raise RuntimeError("Upload service returned an unsupported presigned contract")

        # Step 2: multipart POST to S3 using the signed, size-bounded policy.
        import aiohttp

        file_bytes = file_path.read_bytes()
        form = aiohttp.FormData()
        for name, value in body["uploadFields"].items():
            form.add_field(str(name), str(value))
        form.add_field(
            "file",
            file_bytes,
            filename=file_path.name,
            content_type=_mime_type(file_path),
        )
        async with self._session.post(
            upload_url,
            data=form,
        ) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"S3 upload failed: {resp.status}")

        return {"file_key": file_key, "file_name": file_path.name, "file_size": len(file_bytes)}

    # -- Prompt submission --

    async def submit_file_prompt(
        self, session_id: str, s3_key: str, file_name: str, content: str = ""
    ) -> dict:
        """POST /prompt/file — submit file for ingestion."""
        await self._ensure_session()
        payload = {
            "content": content or f"Ingest {file_name}",
            "session_id": session_id,
            "model_id": "aiidea-ai",
            "file_data": {
                "s3_key": s3_key,
                "name": file_name,
                "type": _mime_type(Path(file_name)),
            },
        }
        async with self._session.post(
            f"{self.api_url}/prompt/file",
            headers=self._headers({"x-session-id": session_id}),
            json=payload,
        ) as resp:
            data = await resp.json()
            return data if "prompt_id" in data else data.get("body", data)

    async def submit_prompt(self, session_id: str, text: str) -> dict:
        """POST /prompt — submit a text query."""
        await self._ensure_session()
        payload = {
            "content": text,
            "session_id": session_id,
            "model_id": "aiidea-ai",
        }
        async with self._session.post(
            f"{self.api_url}/prompt",
            headers=self._headers({"x-session-id": session_id}),
            json=payload,
        ) as resp:
            data = await resp.json()
            return data if "prompt_id" in data else data.get("body", data)

    # -- SSE streaming --

    async def wait_for_completion(
        self, session_id: str, prompt_id: str, timeout: int = SSE_TIMEOUT_S
    ) -> list[dict]:
        """Connect to SSE stream and collect events until completion or timeout."""
        await self._ensure_session()
        events = []
        url = self.sse_url if self.sse_url else f"{self.api_url}/stream"
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "x-session-id": session_id,
        }
        deadline = time.monotonic() + timeout

        try:
            async with self._session.get(
                url,
                headers=headers,
                params={"session_id": session_id},
                timeout=__import__("aiohttp").ClientTimeout(total=timeout),
            ) as resp:
                buffer = ""
                async for chunk in resp.content.iter_any():
                    if time.monotonic() > deadline:
                        logger.warning("SSE timeout for %s", prompt_id)
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        event = _parse_sse_event(event_block)
                        if not event:
                            continue
                        events.append(event)
                        etype = event.get("_event_type", "")
                        epid = event.get("prompt_id", "")
                        if epid == prompt_id and etype in (
                            "completed",
                            "processing_completed",
                            "error",
                        ):
                            return events
        except asyncio.TimeoutError:
            logger.warning("SSE connection timed out for %s", prompt_id)
        except Exception as e:
            logger.warning("SSE error for %s: %s", prompt_id, e)

        return events

    # -- Heatmap (PSA snapshot) --

    async def get_heatmap(self, session_id: str) -> dict:
        """GET /heatmap — voxel occupancy data from clustering service."""
        await self._ensure_session()
        async with self._session.get(
            f"{self.api_url}/heatmap",
            headers=self._headers({"x-session-id": session_id}),
        ) as resp:
            data = await resp.json()
            return data if "hottest_voxels" in data else data.get("body", data)


def _parse_sse_event(block: str) -> Optional[dict]:
    """Parse a single SSE event block into a dict."""
    data_line = None
    for line in block.strip().split("\n"):
        if line.startswith("data:"):
            data_line = line[5:].strip()
        elif line.startswith("data: "):
            data_line = line[6:]
    if not data_line:
        return None
    try:
        return json.loads(data_line)
    except json.JSONDecodeError:
        return None


def _mime_type(path: Path) -> str:
    """Guess MIME type from extension."""
    ext = path.suffix.lower()
    return {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".csv": "text/csv",
    }.get(ext, "application/octet-stream")


# ---------------------------------------------------------------------------
# Neo4j client — paraphrase membership only
# ---------------------------------------------------------------------------


class Neo4jClient:
    """Async Neo4j driver for paraphrase membership queries."""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def get_paraphrase_membership_by_voxel(self) -> dict[str, dict]:
        """
        For each routing display voxel, return:
          - total_sentences: count of sentences in that voxel
          - paraphrase_mass: count of semantic-class memberships

        This is the consolidation metric: paraphrase_mass / total_sentences.
        """
        query = """
        MATCH (s:SentenceNode)
        WHERE s.routing_cluster_label IS NOT NULL
        WITH s.routing_cluster_label AS voxel, count(s) AS total
        OPTIONAL MATCH (s2:SentenceNode)-[:member_of]->(c:ParaphraseGroup)
        WHERE s2.routing_cluster_label = voxel
        WITH voxel, total,
             count(DISTINCT c) AS cluster_count,
             count(s2) AS sentences_in_clusters
        RETURN voxel, total, cluster_count, sentences_in_clusters
        """
        voxels = {}
        with self._driver.session() as session:
            result = session.run(query)
            for record in result:
                voxels[record["voxel"]] = {
                    "total_sentences": record["total"],
                    "cluster_count": record["cluster_count"],
                    "paraphrase_mass": record["sentences_in_clusters"],
                }
        return voxels


# ---------------------------------------------------------------------------
# Step 1: Freeze corpus
# ---------------------------------------------------------------------------


def freeze_corpus(run_id: str) -> dict:
    """Hash every file in ingestion_products/ and build a frozen manifest."""
    manifest = {
        "run_id": run_id,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    for path in sorted(INGESTION_PRODUCTS_DIR.iterdir()):
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["files"].append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha,
                }
            )
    manifest["file_count"] = len(manifest["files"])
    logger.info("Step 1 — Corpus frozen: %d files", manifest["file_count"])
    return manifest


# ---------------------------------------------------------------------------
# Step 2: Ingest via Spark Hub API
# ---------------------------------------------------------------------------


async def ingest_corpus(client: AIIdeaClient, session_id: str, manifest: dict) -> dict:
    """Upload each corpus file via presigned URL, then submit for ingestion."""
    results = []
    for file_info in manifest["files"]:
        file_path = INGESTION_PRODUCTS_DIR / file_info["name"]
        logger.info("  Uploading %s ...", file_info["name"])

        # Upload to S3
        upload = await client.upload_file(file_path)

        # Submit for ingestion
        resp = await client.submit_file_prompt(
            session_id=session_id,
            s3_key=upload["file_key"],
            file_name=upload["file_name"],
        )
        prompt_id = resp.get("prompt_id", "unknown")
        logger.info("  Submitted %s → prompt_id=%s", file_info["name"], prompt_id)

        # Wait for completion via SSE
        events = await client.wait_for_completion(session_id, prompt_id)
        terminal = next(
            (
                e
                for e in reversed(events)
                if e.get("_event_type") in ("completed", "processing_completed", "error")
            ),
            None,
        )
        results.append(
            {
                "file": file_info["name"],
                "prompt_id": prompt_id,
                "status": terminal.get("_event_type", "unknown") if terminal else "timeout",
                "event_count": len(events),
            }
        )

    succeeded = sum(1 for r in results if r["status"] in ("completed", "processing_completed"))
    logger.info(
        "Step 2 — Ingestion complete: %d/%d files succeeded",
        succeeded,
        len(results),
    )
    return {"files_submitted": len(results), "files_succeeded": succeeded, "details": results}


# ---------------------------------------------------------------------------
# Step 3: PSA snapshot via heatmap endpoint + Neo4j
# ---------------------------------------------------------------------------


async def build_psa_snapshot(
    client: AIIdeaClient,
    session_id: str,
    neo4j_client: Optional[Neo4jClient] = None,
) -> dict:
    """
    Combine the heatmap endpoint (voxel occupancy, grid coords) with Neo4j
    paraphrase membership (consolidation metric) into a single snapshot.
    """
    # Fetch heatmap from API — same endpoint Spark Hub uses
    heatmap = await client.get_heatmap(session_id)

    voxels_raw = heatmap.get("hottest_voxels", [])
    router_stats = heatmap.get("router_stats", {})

    # Build voxel dict keyed by voxel_key
    voxels = {}
    for v in voxels_raw:
        key = v["voxel_key"]
        voxels[key] = {
            "voxel_key": key,
            "grid_x": v.get("grid_x", 0),
            "grid_y": v.get("grid_y", 0),
            "grid_z": v.get("grid_z", 0),
            "grid_level": v.get("grid_level", 1),
            "sentence_count": v.get("sentence_count", 0),
            "knowledge_count": v.get("knowledge_count", 0),
            "question_count": v.get("question_count", 0),
            "cluster_count": v.get("cluster_count", 0),
            "heat": v.get("heat", 0.0),
            "is_subdivided": v.get("is_subdivided", False),
            # PSA coordinates normalized to [0,1] using grid_max=20
            "psa_x": v.get("grid_x", 0) / 20.0,
            "psa_y": v.get("grid_y", 0) / 20.0,
            "psa_z": v.get("grid_z", 0) / 20.0,
        }

    # Enrich with paraphrase membership from Neo4j
    if neo4j_client:
        try:
            memberships = neo4j_client.get_paraphrase_membership_by_voxel()
            for key, vdata in voxels.items():
                m = memberships.get(key, {})
                vdata["paraphrase_mass"] = m.get("paraphrase_mass", 0)
                vdata["consolidation"] = (
                    m["paraphrase_mass"] / m["total_sentences"]
                    if m.get("total_sentences", 0) > 0
                    else 0.0
                )
        except Exception as e:
            logger.warning("Neo4j paraphrase query failed: %s", e)
            for vdata in voxels.values():
                vdata["paraphrase_mass"] = 0
                vdata["consolidation"] = 0.0
    else:
        # Without Neo4j, estimate consolidation from cluster_count / sentence_count
        for vdata in voxels.values():
            sc = vdata["sentence_count"]
            cc = vdata["cluster_count"]
            vdata["paraphrase_mass"] = cc  # rough proxy
            vdata["consolidation"] = min(cc / sc, 1.0) if sc > 0 else 0.0

    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "router_stats": router_stats,
        "voxel_count": len(voxels),
        "voxels": voxels,
    }
    logger.info(
        "Step 3 — PSA snapshot: %d voxels, %d total sentences",
        len(voxels),
        router_stats.get("total_sentences", 0),
    )
    return snapshot


# ---------------------------------------------------------------------------
# Step 4: Query tests via Spark Hub API
# ---------------------------------------------------------------------------


async def run_query_tests(
    client: AIIdeaClient, session_id: str, query_set: list[dict]
) -> list[dict]:
    """Submit the fixed query set and collect responses via SSE."""
    results = []
    for q in query_set:
        logger.info("  Query [%s]: %s", q["label"], q["text"][:60])
        resp = await client.submit_prompt(session_id, q["text"])
        prompt_id = resp.get("prompt_id", "unknown")

        events = await client.wait_for_completion(session_id, prompt_id)
        terminal = next(
            (
                e
                for e in reversed(events)
                if e.get("_event_type") in ("completed", "processing_completed", "error")
            ),
            None,
        )
        result = {
            "label": q["label"],
            "query": q["text"],
            "prompt_id": prompt_id,
            "status": terminal.get("_event_type", "unknown") if terminal else "timeout",
            "event_count": len(events),
            "has_subgraph": bool(terminal.get("stitched_subgraph") if terminal else False),
        }
        # Extract evidence sentences if present
        if terminal and terminal.get("stitched_subgraph"):
            sg = terminal["stitched_subgraph"]
            result["evidence_count"] = len(sg.get("sentences", sg.get("nodes", [])))
            result["source_documents"] = list(
                {
                    n.get("container_id", "unknown")
                    for n in sg.get("sentences", sg.get("nodes", []))
                    if n.get("container_id")
                }
            )
        # Collect progress stages
        result["stages"] = [
            e.get("stage") for e in events if e.get("_event_type") == "progress" and e.get("stage")
        ]
        results.append(result)

    succeeded = sum(1 for r in results if r["status"] in ("completed", "processing_completed"))
    logger.info("Step 4 — Queries complete: %d/%d succeeded", succeeded, len(results))
    return results


# ---------------------------------------------------------------------------
# Step 5: Locality decay curve (PRIMARY METRIC)
# ---------------------------------------------------------------------------


def _psa_distance(v1: dict, v2: dict) -> float:
    """Euclidean distance over normalized PSA coordinates."""
    return math.sqrt(
        (v1["psa_x"] - v2["psa_x"]) ** 2
        + (v1["psa_y"] - v2["psa_y"]) ** 2
        + (v1["psa_z"] - v2["psa_z"]) ** 2
    )


def _scorer_param_vector(v: dict) -> list[float]:
    """
    Build a parameter vector from voxel observables.

    Until scorer-bandit parameter export is available per-voxel, we use the
    observable metrics as a proxy: [sentence_count, knowledge_count,
    question_count, cluster_count, consolidation, heat, grid_level].
    """
    return [
        v.get("sentence_count", 0),
        v.get("knowledge_count", 0),
        v.get("question_count", 0),
        v.get("cluster_count", 0),
        v.get("consolidation", 0.0),
        v.get("heat", 0.0),
        float(v.get("grid_level", 1)),
    ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_locality_decay(snapshot: dict) -> dict:
    """
    Compute the locality decay curve — the primary scientific metric.

    For each pair of occupied voxels, compute:
      - PSA distance (Euclidean over normalized coordinates)
      - Behavioral similarity (cosine over observable parameter vectors)

    Fit a linear regression and compute p-value via permutation test.
    """
    voxels = list(snapshot.get("voxels", {}).values())
    if len(voxels) < 3:
        return {"status": "insufficient_voxels", "voxel_count": len(voxels)}

    # Compute all pairwise distance-similarity pairs
    pairs = []
    for v1, v2 in combinations(voxels, 2):
        dist = _psa_distance(v1, v2)
        sim = _cosine_similarity(_scorer_param_vector(v1), _scorer_param_vector(v2))
        pairs.append((dist, sim))

    if not pairs:
        return {"status": "no_pairs"}

    distances = [p[0] for p in pairs]
    similarities = [p[1] for p in pairs]

    # Linear regression: similarity = slope * distance + intercept
    n = len(pairs)
    mean_d = sum(distances) / n
    mean_s = sum(similarities) / n
    ss_dd = sum((d - mean_d) ** 2 for d in distances)
    ss_ds = sum((d - mean_d) * (s - mean_s) for d, s in zip(distances, similarities, strict=False))

    if ss_dd == 0:
        return {"status": "zero_variance", "n_pairs": n}

    slope = ss_ds / ss_dd
    intercept = mean_s - slope * mean_d

    # Residual standard error
    residuals = [s - (slope * d + intercept) for d, s in zip(distances, similarities, strict=False)]
    ss_res = sum(r * r for r in residuals)
    se_slope = math.sqrt(ss_res / ((n - 2) * ss_dd)) if n > 2 and ss_dd > 0 else float("inf")

    # Permutation test: shuffle similarities, recompute slope
    observed_slope = slope
    more_extreme = 0
    rng = random.Random(42)  # Reproducible
    sim_copy = similarities[:]
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(sim_copy)
        perm_mean_s = sum(sim_copy) / n
        perm_ss_ds = sum(
            (d - mean_d) * (s - perm_mean_s) for d, s in zip(distances, sim_copy, strict=False)
        )
        perm_slope = perm_ss_ds / ss_dd
        if perm_slope <= observed_slope:  # One-tailed: is real slope more negative?
            more_extreme += 1

    p_value = more_extreme / N_PERMUTATIONS

    result = {
        "status": "computed",
        "n_pairs": n,
        "n_voxels": len(voxels),
        "slope": slope,
        "intercept": intercept,
        "se_slope": se_slope,
        "p_value": p_value,
        "mean_distance": mean_d,
        "mean_similarity": mean_s,
        "distance_range": [min(distances), max(distances)],
        "similarity_range": [min(similarities), max(similarities)],
        "pairs": pairs,  # raw data for plotting
    }
    logger.info(
        "Step 5 — Locality decay: slope=%.4f, p=%.4f (%d pairs, %d voxels)",
        slope,
        p_value,
        n,
        len(voxels),
    )
    return result


# ---------------------------------------------------------------------------
# Step 6: Regime boundaries
# ---------------------------------------------------------------------------


def _are_adjacent(v1: dict, v2: dict) -> bool:
    """Two voxels are adjacent if they differ by at most 1 in each grid axis."""
    return (
        abs(v1["grid_x"] - v2["grid_x"]) <= 1
        and abs(v1["grid_y"] - v2["grid_y"]) <= 1
        and abs(v1["grid_z"] - v2["grid_z"]) <= 1
        and v1["grid_level"] == v2["grid_level"]
    )


def identify_regime_boundaries(snapshot: dict) -> list[dict]:
    """
    Find adjacent voxel pairs where behavioral similarity drops sharply.
    These are potential regime boundaries in the semantic field.
    """
    voxels = list(snapshot.get("voxels", {}).values())
    boundaries = []

    for v1, v2 in combinations(voxels, 2):
        if not _are_adjacent(v1, v2):
            continue
        sim = _cosine_similarity(_scorer_param_vector(v1), _scorer_param_vector(v2))
        # Low similarity between adjacent voxels = regime boundary
        if sim < 0.5:
            boundaries.append(
                {
                    "voxel_a": v1["voxel_key"],
                    "voxel_b": v2["voxel_key"],
                    "similarity": sim,
                    "distance": _psa_distance(v1, v2),
                    "coord_a": [v1["grid_x"], v1["grid_y"], v1["grid_z"]],
                    "coord_b": [v2["grid_x"], v2["grid_y"], v2["grid_z"]],
                }
            )

    boundaries.sort(key=lambda b: b["similarity"])
    logger.info("Step 6 — Regime boundaries: %d found", len(boundaries))
    return boundaries


# ---------------------------------------------------------------------------
# Step 7: Null controls — preserve distribution, destroy structure
# ---------------------------------------------------------------------------


def scramble_voxel_assignments(snapshot: dict) -> dict:
    """
    Randomly permute voxel labels across nodes while preserving per-voxel
    node counts and overall occupancy distribution.

    Implementation: shuffle the PSA coordinates across voxels. This preserves
    every voxel's local statistics (sentence_count, cluster_count, etc.) but
    severs the link between grid position and semantic content.
    """
    scrambled = copy.deepcopy(snapshot)
    voxels = list(scrambled["voxels"].values())
    if len(voxels) < 2:
        return scrambled

    rng = random.Random(42)
    # Extract coordinate tuples
    coords = [
        (v["psa_x"], v["psa_y"], v["psa_z"], v["grid_x"], v["grid_y"], v["grid_z"]) for v in voxels
    ]
    rng.shuffle(coords)
    for v, (px, py, pz, gx, gy, gz) in zip(voxels, coords, strict=False):
        v["psa_x"] = px
        v["psa_y"] = py
        v["psa_z"] = pz
        v["grid_x"] = gx
        v["grid_y"] = gy
        v["grid_z"] = gz

    return scrambled


def shuffle_source_labels(snapshot: dict) -> dict:
    """
    Randomly reassign source composition (knowledge_count vs question_count)
    while preserving total counts and distribution.
    """
    scrambled = copy.deepcopy(snapshot)
    voxels = list(scrambled["voxels"].values())
    if len(voxels) < 2:
        return scrambled

    rng = random.Random(43)
    # Shuffle knowledge/question counts across voxels
    k_counts = [v["knowledge_count"] for v in voxels]
    q_counts = [v["question_count"] for v in voxels]
    rng.shuffle(k_counts)
    rng.shuffle(q_counts)
    for v, kc, qc in zip(voxels, k_counts, q_counts, strict=False):
        v["knowledge_count"] = kc
        v["question_count"] = qc

    return scrambled


def perturb_guide_stars(snapshot: dict, noise_scale: float = 0.1) -> dict:
    """
    Add calibrated noise to PSA coordinates. This tests whether the
    coordinate frame itself carries signal.
    """
    perturbed = copy.deepcopy(snapshot)
    rng = random.Random(44)
    for v in perturbed["voxels"].values():
        v["psa_x"] = max(0.0, min(1.0, v["psa_x"] + rng.gauss(0, noise_scale)))
        v["psa_y"] = max(0.0, min(1.0, v["psa_y"] + rng.gauss(0, noise_scale)))
        v["psa_z"] = max(0.0, min(1.0, v["psa_z"] + rng.gauss(0, noise_scale)))
    return perturbed


def run_null_controls(snapshot: dict) -> dict:
    """Run all three null models and compute locality decay for each."""
    controls = {}
    for name, fn in [
        ("scrambled_voxels", scramble_voxel_assignments),
        ("shuffled_sources", shuffle_source_labels),
        ("perturbed_stars", perturb_guide_stars),
    ]:
        scrambled = fn(snapshot)
        decay = compute_locality_decay(scrambled)
        controls[name] = {
            "slope": decay.get("slope"),
            "p_value": decay.get("p_value"),
            "n_pairs": decay.get("n_pairs"),
            "mean_similarity": decay.get("mean_similarity"),
        }
        logger.info(
            "  Null [%s]: slope=%.4f",
            name,
            decay.get("slope", float("nan")),
        )
    logger.info("Step 7 — Null controls complete")
    return controls


# ---------------------------------------------------------------------------
# Step 8: Stability check
# ---------------------------------------------------------------------------


async def run_stability_check(
    client: AIIdeaClient,
    session_id: str,
    prior_snapshot: dict,
    neo4j_client: Optional[Neo4jClient] = None,
) -> dict:
    """
    Fetch a fresh heatmap snapshot and diff against the prior one.
    Full re-ingest + clear is intentionally NOT automated here — it requires
    a manual Neo4j clear which is destructive. This step compares against a
    second snapshot taken after the original ingest has settled.
    """
    new_snapshot = await build_psa_snapshot(client, session_id, neo4j_client)
    prior_voxels = set(prior_snapshot.get("voxels", {}).keys())
    new_voxels = set(new_snapshot.get("voxels", {}).keys())

    shared = prior_voxels & new_voxels
    count_diffs = []
    for key in shared:
        old_c = prior_snapshot["voxels"][key]["sentence_count"]
        new_c = new_snapshot["voxels"][key]["sentence_count"]
        if old_c != new_c:
            count_diffs.append({"voxel": key, "old": old_c, "new": new_c})

    result = {
        "prior_voxel_count": len(prior_voxels),
        "new_voxel_count": len(new_voxels),
        "shared_voxels": len(shared),
        "added_voxels": len(new_voxels - prior_voxels),
        "removed_voxels": len(prior_voxels - new_voxels),
        "count_diffs": count_diffs,
        "structure_stable": (
            len(new_voxels - prior_voxels) == 0
            and len(prior_voxels - new_voxels) == 0
            and len(count_diffs) == 0
        ),
    }
    logger.info(
        "Step 8 — Stability: shared=%d, added=%d, removed=%d, diffs=%d",
        len(shared),
        result["added_voxels"],
        result["removed_voxels"],
        len(count_diffs),
    )
    return result


# ---------------------------------------------------------------------------
# Voxel color rendering (§24 spec)
# ---------------------------------------------------------------------------


def voxel_color(
    node_count: int,
    paraphrase_mass: int,
    total_sentences: int,
    anchor: int = VOXEL_LUMINANCE_ANCHOR,
) -> tuple[float, float, float]:
    """
    Return (R, G, B) in [0, 1] per the field visualization spec (§24).

    Luminance = log₁₀(1 + n) / log₁₀(anchor)  — absolute, never normalized
    Hue = consolidation × 0.5  — red(0) → yellow(0.167) → cyan(0.5)
    Saturation = 1.0 (fixed)
    """
    if node_count == 0:
        return (0.0, 0.0, 0.0)

    luminance = min(math.log10(1 + node_count) / math.log10(anchor), 1.0)
    consolidation = min(paraphrase_mass / total_sentences, 1.0) if total_sentences > 0 else 0.0
    hue = consolidation * 0.5  # 0.0 (red) → 0.5 (cyan)
    saturation = 1.0
    return colorsys.hls_to_rgb(hue, luminance * 0.5, saturation)


# ---------------------------------------------------------------------------
# Step 9: Report + decision gate
# ---------------------------------------------------------------------------


def build_report(
    run_id: str,
    manifest: dict,
    ingestion: Optional[dict],
    snapshot: dict,
    query_results: list[dict],
    locality: dict,
    boundaries: list[dict],
    null_controls: dict,
    stability: Optional[dict] = None,
) -> dict:
    """Aggregate all metrics and apply the binary decision gate."""
    locality_p = locality.get("p_value", 1.0)
    gate_pass = locality_p < LOCALITY_P_THRESHOLD

    # Voxel color map for visualization
    color_map = {}
    for key, v in snapshot.get("voxels", {}).items():
        rgb = voxel_color(
            v["sentence_count"],
            v.get("paraphrase_mass", 0),
            v["sentence_count"],
        )
        color_map[key] = {
            "rgb": [round(c, 4) for c in rgb],
            "hex": "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            ),
            "consolidation": v.get("consolidation", 0.0),
        }

    report: dict[str, Any] = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "PSA coordinates induce measurable locality: regions close in PSA "
            "space exhibit more similar retrieval, scoring, and traversal "
            "behavior than regions farther apart, and this effect disappears "
            "under null models."
        ),
        "manifest": manifest,
        "ingestion": ingestion,
        "snapshot_summary": {
            "voxel_count": snapshot.get("voxel_count", 0),
            "router_stats": snapshot.get("router_stats", {}),
        },
        "query_results": {
            "total": len(query_results),
            "succeeded": sum(
                1 for r in query_results if r.get("status") in ("completed", "processing_completed")
            ),
            "details": query_results,
        },
        "locality": {
            "slope": locality.get("slope"),
            "intercept": locality.get("intercept"),
            "se_slope": locality.get("se_slope"),
            "p_value": locality_p,
            "n_pairs": locality.get("n_pairs"),
            "n_voxels": locality.get("n_voxels"),
            "mean_distance": locality.get("mean_distance"),
            "mean_similarity": locality.get("mean_similarity"),
        },
        "regime_boundaries": {
            "count": len(boundaries),
            "sharpest": boundaries[:5] if boundaries else [],
        },
        "null_controls": null_controls,
        "stability": stability,
        "voxel_colors": color_map,
        "decision_gate": {
            "metric": "locality_decay_slope",
            "test": "permutation (one-tailed, N={})".format(N_PERMUTATIONS),
            "threshold": LOCALITY_P_THRESHOLD,
            "p_value": locality_p,
            "slope": locality.get("slope"),
            "null_slopes": {name: ctrl.get("slope") for name, ctrl in null_controls.items()},
            "pass": gate_pass,
            "verdict": (
                "PASS — PSA locality is statistically significant (p={:.4f}). "
                "Proceed to scale.".format(locality_p)
                if gate_pass
                else "FAIL — PSA locality is NOT distinguishable from scrambled "
                "controls (p={:.4f}). Scaling should pause. Revise "
                "coordinate frame or similarity metric.".format(locality_p)
            ),
        },
    }
    return report


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

STEPS = [
    "freeze",
    "ingest",
    "snapshot",
    "query",
    "locality",
    "boundaries",
    "nulls",
    "stability",
    "report",
]


async def run_first_round(
    run_id: str,
    client: AIIdeaClient,
    neo4j_client: Optional[Neo4jClient] = None,
    skip_ingest: bool = False,
    step: Optional[str] = None,
) -> dict:
    """Execute all 9 steps (or a single step) of the first-round test."""
    logger.info("═══ First-Round PSA Test: %s ═══", run_id)

    run_steps = STEPS if step is None else [step]

    # Only create a session if we need API access
    needs_api = any(s in run_steps for s in ("ingest", "snapshot", "query", "stability"))
    session_id = ""
    if needs_api or step is None:
        session_id = await client.get_or_create_session()
        logger.info("Session: %s", session_id)

    # State that accumulates across steps
    manifest = {}
    ingestion = None
    snapshot: dict = {"voxels": {}}
    query_results: list[dict] = []
    locality: dict = {}
    boundaries: list[dict] = []
    null_controls: dict = {}
    stability = None

    # Output directory
    out_dir = Path(f"psa_test_runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1
    if "freeze" in run_steps:
        manifest = freeze_corpus(run_id)
        _save(out_dir / "manifest.json", manifest)

    # Step 2
    if "ingest" in run_steps and not skip_ingest:
        ingestion = await ingest_corpus(client, session_id, manifest)
        _save(out_dir / "ingestion.json", ingestion)

    # Step 3
    if "snapshot" in run_steps or any(s in run_steps for s in STEPS[4:]):
        snapshot = await build_psa_snapshot(client, session_id, neo4j_client)
        _save(out_dir / "snapshot.json", snapshot)

    # Step 4
    if "query" in run_steps:
        query_results = await run_query_tests(client, session_id, QUERY_SET)
        _save(out_dir / "query_results.json", query_results)

    # Step 5
    if "locality" in run_steps or "report" in run_steps:
        locality = compute_locality_decay(snapshot)
        _save(out_dir / "locality.json", _strip_pairs(locality))

    # Step 6
    if "boundaries" in run_steps or "report" in run_steps:
        boundaries = identify_regime_boundaries(snapshot)
        _save(out_dir / "boundaries.json", boundaries)

    # Step 7
    if "nulls" in run_steps or "report" in run_steps:
        null_controls = run_null_controls(snapshot)
        _save(out_dir / "null_controls.json", null_controls)

    # Step 8
    if "stability" in run_steps:
        stability = await run_stability_check(client, session_id, snapshot, neo4j_client)
        _save(out_dir / "stability.json", stability)

    # Step 9
    if "report" in run_steps or step is None:
        report = build_report(
            run_id,
            manifest,
            ingestion,
            snapshot,
            query_results,
            locality,
            boundaries,
            null_controls,
            stability,
        )
        _save(out_dir / "run_report.json", report)

        gate = report["decision_gate"]
        banner = "✅ PASS" if gate["pass"] else "❌ FAIL"
        logger.info("═══ %s — %s ═══", banner, gate["verdict"])

        return report

    logger.info("═══ Step '%s' complete ═══", step)
    return {"run_id": run_id, "step": step, "status": "complete"}


def _save(path: Path, data: Any) -> None:
    """Write JSON to file."""
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("  → Saved %s", path)


def _strip_pairs(locality: dict) -> dict:
    """Remove raw pairs from locality for compact JSON output."""
    out = {k: v for k, v in locality.items() if k != "pairs"}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="First-round PSA scientific test — uses Spark Hub API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (all 9 steps)
  python -m tests.psa_testing.first_round_test --run-id test1

  # Skip ingestion (data already loaded)
  python -m tests.psa_testing.first_round_test --run-id test1 --skip-ingest

  # Run only snapshot + analysis
  python -m tests.psa_testing.first_round_test --run-id test1 --step snapshot

  # Run locality analysis on existing snapshot
  python -m tests.psa_testing.first_round_test --run-id test1 --step locality
        """,
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--api-url", default=os.environ.get("API_URL", DEFAULT_API_URL))
    parser.add_argument("--sse-url", default=os.environ.get("SSE_STREAM_URL", ""))
    parser.add_argument("--upload-url", default=os.environ.get("UPLOAD_API_URL", ""))
    parser.add_argument("--auth-token", default=os.environ.get("AUTH_TOKEN", ""))
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", DEFAULT_NEO4J_URI))
    parser.add_argument("--neo4j-user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--skip-ingest", action="store_true", help="Skip file upload/ingestion")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j queries")
    parser.add_argument(
        "--step",
        choices=STEPS,
        default=None,
        help="Run a single step instead of all 9",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve Neo4j password
    neo4j_password = args.neo4j_password
    neo4j_client = None
    if not args.skip_neo4j:
        if not neo4j_password:
            neo4j_password = fetch_neo4j_password_via_ssm() or ""
        if neo4j_password:
            try:
                neo4j_client = Neo4jClient(args.neo4j_uri, args.neo4j_user, neo4j_password)
                logger.info("Neo4j connected: %s", args.neo4j_uri)
            except Exception as e:
                logger.warning("Neo4j connection failed: %s — continuing without", e)

    client = AIIdeaClient(
        api_url=args.api_url,
        sse_url=args.sse_url,
        upload_url=args.upload_url,
        auth_token=args.auth_token,
    )

    try:
        report = asyncio.run(
            run_first_round(
                run_id=args.run_id,
                client=client,
                neo4j_client=neo4j_client,
                skip_ingest=args.skip_ingest,
                step=args.step,
            )
        )
    finally:
        asyncio.run(client.close())
        if neo4j_client:
            neo4j_client.close()

    # Exit code based on decision gate
    if isinstance(report, dict) and "decision_gate" in report:
        sys.exit(0 if report["decision_gate"]["pass"] else 1)
    sys.exit(0)


if __name__ == "__main__":
    main()
