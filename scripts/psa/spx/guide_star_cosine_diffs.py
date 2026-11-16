#!/usr/bin/env python3
"""
Compute cosine diffs between guide stars, and between guide stars and sampled sentences.
Outputs a JSON file with guide star pairwise diffs, a short voxel space description,
and per-sentence cosine diffs to all guide stars.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_SSM_INSTANCE_ID = os.getenv("NEO4J_SSM_INSTANCE_ID", "")


def load_guide_star_catalog() -> List[Any]:
    guide_stars_path = (
        Path(__file__).resolve().parents[4]
        / "app"
        / "consolidated_clustering_v2"
        / "routing"
        / "guide_stars.py"
    )
    spec = importlib.util.spec_from_file_location("guide_stars", guide_stars_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load guide_stars module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GUIDE_STAR_CATALOG


try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover - optional runtime dependency
    GraphDatabase = None


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def fetch_password_via_ssm(parameter_name: str) -> Optional[str]:
    try:
        return subprocess.check_output(
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
    except Exception:
        return None


def run_ssm_command(command: str, instance_id: str, comment: str) -> Dict[str, Any]:
    parameters = f"commands={json.dumps([command])}"
    try:
        command_id = subprocess.check_output(
            [
                "aws",
                "ssm",
                "send-command",
                "--instance-ids",
                instance_id,
                "--document-name",
                "AWS-RunShellScript",
                "--comment",
                comment,
                "--parameters",
                parameters,
                "--output",
                "text",
                "--query",
                "Command.CommandId",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"AWS SSM send-command failed (exit {exc.returncode}).") from exc

    try:
        subprocess.check_output(
            [
                "aws",
                "ssm",
                "wait",
                "command-executed",
                "--command-id",
                command_id,
                "--instance-id",
                instance_id,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        pass

    try:
        invocation = subprocess.check_output(
            [
                "aws",
                "ssm",
                "get-command-invocation",
                "--command-id",
                command_id,
                "--instance-id",
                instance_id,
                "--output",
                "json",
            ],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"AWS SSM get-command-invocation failed: {exc}") from exc

    return json.loads(invocation)


def generate_presigned_url(bucket: str, key: str, expires_in: int = 3600) -> Optional[str]:
    try:
        return subprocess.check_output(
            [
                "aws",
                "s3",
                "presign",
                f"s3://{bucket}/{key}",
                "--expires-in",
                str(expires_in),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def generate_presigned_put_url(bucket: str, key: str, expires_in: int = 3600) -> Optional[str]:
    try:
        import boto3
    except ImportError:
        return None
    try:
        return boto3.client("s3").generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": key, "ContentType": "application/json"},
            ExpiresIn=expires_in,
        )
    except Exception:
        return None


def run_cypher_ssm(
    query: str,
    instance_id: str,
    password: str,
    database: str,
) -> str:
    compact_query = " ".join(query.split())
    escaped_query = compact_query.replace('"', '\\"')
    command = (
        "docker exec neo4j cypher-shell "
        f"-u neo4j -p {shell_quote(password)} "
        f"--format plain -d {shell_quote(database)} "
        f'"{escaped_query}"'
    )
    result = run_ssm_command(command, instance_id, "neo4j query")
    status = result.get("Status")
    stdout = result.get("StandardOutputContent", "")
    stderr = result.get("StandardErrorContent", "")
    if status not in ("Success", "SuccessWithWarning"):
        raise RuntimeError(f"SSM command failed: status={status} stderr={stderr}")
    if stderr:
        print(f"WARNING: Neo4j command stderr: {stderr.strip()}")
    return stdout


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return matrix / norms


def normalize_vector(vec: np.ndarray) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return None
    return vec / norm


def cosine_diffs_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(embeddings)
    sims = normalized @ normalized.T
    return 1.0 - sims


def cosine_diff(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a <= 0 or norm_b <= 0:
        raise ValueError("Zero-norm vector for cosine diff.")
    sim = float((vec_a / norm_a) @ (vec_b / norm_b))
    return float(1.0 - sim)


def embed_texts(endpoint: str, texts: List[str], batch_size: int = 32) -> np.ndarray:
    import urllib.request

    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = json.dumps({"inputs": batch}).encode("utf-8")
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Embedding endpoint error: {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))
        all_embeddings.extend(data)
    return np.array(all_embeddings, dtype=np.float32)


def load_star_embeddings_from_s3(bucket: str, key: str) -> Optional[np.ndarray]:
    try:
        import boto3
    except ImportError:
        return None

    try:
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
        return np.load(io.BytesIO(body), allow_pickle=False)
    except Exception:
        return None


def load_json_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    try:
        import boto3
    except ImportError:
        boto3 = None

    if boto3 is not None:
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        return json.loads(body)

    temp_path = Path("/tmp/guide_star_sentence_diffs.json")
    subprocess.check_call(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", str(temp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(temp_path.read_text(encoding="utf-8"))


def load_star_embeddings(
    star_texts: List[str],
    star_embeddings_path: Optional[str],
    embed_endpoint: Optional[str],
    s3_bucket: Optional[str],
    spx_frame_version: str,
) -> Tuple[np.ndarray, str]:
    if star_embeddings_path:
        path = Path(star_embeddings_path)
        embeddings = np.load(path, allow_pickle=False)
        return embeddings.astype(np.float32), f"file:{path}"

    if s3_bucket:
        key = f"guide_stars/{spx_frame_version}/star_embeddings.npy"
        embeddings = load_star_embeddings_from_s3(s3_bucket, key)
        if embeddings is not None:
            return embeddings.astype(np.float32), f"s3://{s3_bucket}/{key}"

    if not embed_endpoint:
        raise RuntimeError("No guide star embeddings source configured.")

    embeddings = embed_texts(embed_endpoint, star_texts)
    return embeddings.astype(np.float32), f"embed_endpoint:{embed_endpoint}"


def build_remote_sentence_script() -> str:
    return r"""
import json
import os
import subprocess
import sys
import urllib.request

def ensure_deps():
    try:
        import numpy as np  # noqa: F401
        from neo4j import GraphDatabase  # noqa: F401
        return
    except Exception:
        pass
    deps_dir = "/tmp/gs_sentence_deps"
    os.makedirs(deps_dir, exist_ok=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--target", deps_dir, "numpy", "neo4j"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sys.path.insert(0, deps_dir)

def load_star_embeddings(bucket, key):
    local_path = "/tmp/guide_star_embeddings.npy"
    presigned_url = os.environ.get("S3_PRESIGNED_URL")
    if presigned_url:
        try:
            urllib.request.urlretrieve(presigned_url, local_path)
        except Exception as exc:
            raise RuntimeError(f"presigned download failed: {exc}") from exc
    else:
        result = subprocess.run(
            ["aws", "s3", "cp", f"s3://{bucket}/{key}", local_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise RuntimeError(f"aws s3 cp failed: {message}")
    import numpy as np
    return np.load(local_path, allow_pickle=False).astype("float32")

def normalize_matrix(mat):
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return mat / norms

def normalize_vec(vec):
    import numpy as np
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return None
    return vec / norm

def compute_diffs(star_norm, embedding):
    import numpy as np
    vec = np.array(embedding, dtype="float32")
    vec = normalize_vec(vec)
    if vec is None:
        return None
    diffs = 1.0 - (star_norm @ vec)
    return [round(float(x), 6) for x in diffs]

def fetch_rows(driver, query, limit):
    with driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        result = session.run(query, limit=limit)
        return [r.data() for r in result]

def main():
    ensure_deps()
    import numpy as np
    from neo4j import GraphDatabase

    bucket = os.environ["S3_BUCKET"]
    key = os.environ["S3_KEY"]
    password = os.environ["NEO4J_PASSWORD"]
    para_count = int(os.environ.get("PARA_COUNT", "10"))
    non_para_count = int(os.environ.get("NON_PARA_COUNT", "10"))
    sample_count = int(os.environ.get("SAMPLE_COUNT", "5"))
    star_embeddings = load_star_embeddings(bucket, key)
    star_norm = normalize_matrix(star_embeddings)

    def voxel_expr(var):
        return f"{var}.visual_voxel_id"

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))
    try:
        para_filter_expr = voxel_expr("s")
        para_return_expr = voxel_expr("sample")
        para_query = '''
        MATCH (s)-[:member_of]->(g)
        WHERE (g:ParaphraseGroup OR g:PromptParaphraseGroup)
          AND (s:SentenceNode OR s:PromptSentence)
          AND s.embedding IS NOT NULL AND size(s.embedding) > 0
          AND ''' + para_filter_expr + ''' IS NOT NULL
        WITH g, collect(s) AS members
        WHERE size(members) >= 2
        WITH g, members[0] AS sample, members[1] AS other
        RETURN
            COALESCE(g.group_id, g.node_id) AS group_id,
            COALESCE(sample.sentence_id, sample.node_id) AS sentence_id,
            COALESCE(sample.text, sample.original_text, sample.content, sample.sentence) AS text,
            ''' + para_return_expr + ''' AS voxel_address,
            sample.embedding AS embedding
        ORDER BY rand()
        LIMIT $limit
        '''
        non_para_voxel_expr = voxel_expr("s")
        non_para_query = '''
        MATCH (s)
        WHERE (s:SentenceNode OR s:PromptSentence)
          AND s.embedding IS NOT NULL AND size(s.embedding) > 0
          AND NOT (s)-[:member_of]->(:ParaphraseGroup)
          AND NOT (s)-[:member_of]->(:PromptParaphraseGroup)
          AND ''' + non_para_voxel_expr + ''' IS NOT NULL
        RETURN
            COALESCE(s.sentence_id, s.node_id) AS sentence_id,
            COALESCE(s.text, s.original_text, s.content, s.sentence) AS text,
            ''' + non_para_voxel_expr + ''' AS voxel_address,
            s.embedding AS embedding
        ORDER BY rand()
        LIMIT $limit
        '''
        sample_voxel_expr = voxel_expr("s")
        sample_query = '''
        MATCH (s)
        WHERE (s:SentenceNode OR s:PromptSentence)
          AND s.embedding IS NOT NULL AND size(s.embedding) > 0
          AND ''' + sample_voxel_expr + ''' IS NOT NULL
        RETURN
            COALESCE(s.text, s.original_text, s.content, s.sentence) AS text,
            ''' + sample_voxel_expr + ''' AS voxel_address,
            s.embedding AS embedding
        ORDER BY rand()
        LIMIT $limit
        '''
        para_rows = fetch_rows(driver, para_query, para_count)
        non_para_rows = fetch_rows(driver, non_para_query, non_para_count)
        sample_rows = fetch_rows(driver, sample_query, sample_count)
    finally:
        driver.close()

    if len(para_rows) < para_count:
        raise RuntimeError(f"paraphrase rows={len(para_rows)} need={para_count}")
    if len(non_para_rows) < non_para_count:
        raise RuntimeError(f"non-paraphrase rows={len(non_para_rows)} need={non_para_count}")

    sentences = []
    for row in para_rows:
        diffs = compute_diffs(star_norm, row["embedding"])
        if diffs is None:
            continue
        sentences.append(
            {
                "sentence_id": row["sentence_id"],
                "text": row["text"],
                "paraphrase_group_id": row["group_id"],
                "voxel_address": row["voxel_address"],
                "cosine_diffs": diffs,
            }
        )

    for row in non_para_rows:
        diffs = compute_diffs(star_norm, row["embedding"])
        if diffs is None:
            continue
        sentences.append(
            {
                "sentence_id": row["sentence_id"],
                "text": row["text"],
                "paraphrase_group_id": None,
                "voxel_address": row["voxel_address"],
                "cosine_diffs": diffs,
            }
        )

    sample_sentences = []
    for row in sample_rows:
        sample_sentences.append(
            {
                "text": row["text"],
                "voxel_address": row["voxel_address"],
            }
        )

    payload = {
        "sentences": sentences,
    }
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    output_url = os.environ.get("S3_OUTPUT_URL")
    if output_url:
        request = urllib.request.Request(
            output_url,
            data=payload_json,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status not in (200, 204):
                raise RuntimeError(f"upload failed: status={response.status}")
        print(
            json.dumps(
                {
                    "s3_output_bucket": os.environ.get("S3_OUTPUT_BUCKET", ""),
                    "s3_output_key": os.environ.get("S3_OUTPUT_KEY", ""),
                    "sample_sentences": sample_sentences,
                },
                separators=(",", ":"),
            )
        )
    else:
        payload["sample_sentences"] = sample_sentences
        print(json.dumps(payload, separators=(",", ":")))

if __name__ == "__main__":
    main()
"""


def fetch_sentence_diffs_ssm(
    instance_id: str,
    password: str,
    database: str,
    s3_bucket: str,
    spx_frame_version: str,
    paraphrase_count: int,
    non_paraphrase_count: int,
    sample_count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    key = f"guide_stars/{spx_frame_version}/star_embeddings.npy"
    presigned_url = generate_presigned_url(s3_bucket, key)
    output_key = (
        f"tmp/guide_star_sentence_diffs/"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_"
        f"{uuid.uuid4().hex}.json"
    )
    output_url = generate_presigned_put_url(s3_bucket, output_key)
    script = build_remote_sentence_script()
    script_b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    env_parts = [
        f"NEO4J_PASSWORD={shell_quote(password)}",
        f"NEO4J_DATABASE={shell_quote(database)}",
        f"S3_BUCKET={shell_quote(s3_bucket)}",
        f"S3_KEY={shell_quote(key)}",
        f"PARA_COUNT={paraphrase_count}",
        f"NON_PARA_COUNT={non_paraphrase_count}",
        f"SAMPLE_COUNT={sample_count}",
    ]
    if presigned_url:
        env_parts.append(f"S3_PRESIGNED_URL={shell_quote(presigned_url)}")
    if output_url:
        env_parts.append(f"S3_OUTPUT_URL={shell_quote(output_url)}")
        env_parts.append(f"S3_OUTPUT_BUCKET={shell_quote(s3_bucket)}")
        env_parts.append(f"S3_OUTPUT_KEY={shell_quote(output_key)}")
    env_prefix = " ".join(env_parts)
    command = (
        "set -e; "
        "python3 -m ensurepip --upgrade >/tmp/gs_sentence_pip.log 2>&1 || true; "
        f"PYTHON_B64={shell_quote(script_b64)}; "
        'echo "$PYTHON_B64" | base64 --decode > /tmp/guide_star_sentence_diffs.py; '
        f"{env_prefix} python3 /tmp/guide_star_sentence_diffs.py "
        ">/tmp/guide_star_sentence_diffs.json "
        "2>/tmp/guide_star_sentence_diffs.err; "
        "cat /tmp/guide_star_sentence_diffs.json"
    )
    result = run_ssm_command(command, instance_id, "guide_star_sentence_diffs")
    stdout = result.get("StandardOutputContent", "").strip()
    if not stdout:
        stderr = result.get("StandardErrorContent", "").strip()
        raise RuntimeError(f"SSM returned no output. stderr={stderr}")
    data = json.loads(stdout)
    output_bucket = data.get("s3_output_bucket")
    output_key = data.get("s3_output_key")
    if output_bucket and output_key:
        payload = load_json_from_s3(output_bucket, output_key)
        return payload.get("sentences", []), data.get("sample_sentences", [])
    return data.get("sentences", []), data.get("sample_sentences", [])


def sanitize_text_expr(expr: str) -> str:
    return (
        "replace("
        "replace("
        "replace("
        "replace("
        "replace("
        f"{expr}, '\\t', ' '),"
        "'\\n', ' '),"
        "'|', ' '),"
        "',', ' '),"
        "'\"', ' ')"
    )


def embedding_to_string_expr(expr: str, separator: str = ";") -> str:
    return (
        "'[' + reduce(s = '', x IN "
        f"{expr}"
        f" | s + CASE WHEN s = '' THEN '' ELSE '{separator}' END + toString(x)) + ']'"
    )


def build_voxel_expr(node_var: str) -> str:
    return f"{node_var}.visual_voxel_id"


def query_paraphrase_samples(
    session,
    limit: int,
) -> List[Dict[str, Any]]:
    filter_voxel_expr = build_voxel_expr("s")
    return_voxel_expr = build_voxel_expr("sample")
    query = f"""
    MATCH (s)-[:member_of]->(g)
    WHERE (g:ParaphraseGroup OR g:PromptParaphraseGroup)
      AND (s:SentenceNode OR s:PromptSentence)
      AND s.embedding IS NOT NULL AND size(s.embedding) > 0
      AND {filter_voxel_expr} IS NOT NULL
    WITH g, collect(s) AS members
    WHERE size(members) >= 2
    WITH g, members[0] AS sample, members[1] AS other
    RETURN
        COALESCE(g.group_id, g.node_id) AS group_id,
        COALESCE(sample.sentence_id, sample.node_id) AS sentence_id,
        COALESCE(sample.text, sample.original_text, sample.content, sample.sentence) AS text,
        {return_voxel_expr} AS voxel_address,
        sample.embedding AS embedding
    ORDER BY rand()
    LIMIT $limit
    """
    return [r.data() for r in session.run(query, limit=limit)]


def query_paraphrase_samples_ssm(
    limit: int,
    instance_id: str,
    password: str,
    database: str,
) -> List[Dict[str, Any]]:
    text_expr = sanitize_text_expr(
        "COALESCE(sample.text, sample.original_text, sample.content, sample.sentence)"
    )
    other_text_expr = sanitize_text_expr(
        "COALESCE(other.text, other.original_text, other.content, other.sentence)"
    )
    embedding_expr = embedding_to_string_expr("sample.embedding")
    row_expr = (
        "COALESCE(g.group_id, g.node_id, '') + '||' + "
        "COALESCE(sample.sentence_id, sample.node_id, '') + '||' + "
        f"{text_expr} + '||' + "
        f"{embedding_expr} + '||' + "
        f"{other_text_expr}"
    )
    query = f"""
    MATCH (s)-[:member_of]->(g)
    WHERE (g:ParaphraseGroup OR g:PromptParaphraseGroup)
      AND (s:SentenceNode OR s:PromptSentence)
      AND s.embedding IS NOT NULL AND size(s.embedding) > 0
    WITH g, collect(s) AS members
    WHERE size(members) >= 2
    WITH g, members[0] AS sample, members[1] AS other
    RETURN {row_expr} AS row
    ORDER BY rand()
    LIMIT {limit}
    """
    output = run_cypher_ssm(query, instance_id, password, database)
    rows = parse_paraphrase_output_ssm(output)
    rows = dedupe_paraphrase_rows(rows)
    return rows[:limit]


def query_non_paraphrase_samples(
    session,
    limit: int,
) -> List[Dict[str, Any]]:
    voxel_expr = build_voxel_expr("s")
    query = f"""
    MATCH (s)-[:member_of]->(g)
    WHERE (g:ParaphraseGroup OR g:PromptParaphraseGroup)
      AND (s:SentenceNode OR s:PromptSentence)
      AND s.embedding IS NOT NULL AND size(s.embedding) > 0
      AND {voxel_expr} IS NOT NULL
    MATCH (member)-[:member_of]->(g)
    WITH s, count(DISTINCT member) AS group_size
    WHERE group_size = 1
    RETURN
        COALESCE(s.sentence_id, s.node_id) AS sentence_id,
        COALESCE(s.text, s.original_text, s.content, s.sentence) AS text,
        {voxel_expr} AS voxel_address,
        s.embedding AS embedding
    ORDER BY rand()
    LIMIT $limit
    """
    return [r.data() for r in session.run(query, limit=limit)]


def query_sample_sentences(
    session,
    limit: int,
) -> List[Dict[str, Any]]:
    voxel_expr = build_voxel_expr("s")
    query = f"""
    MATCH (s)
    WHERE (s:SentenceNode OR s:PromptSentence)
      AND s.embedding IS NOT NULL AND size(s.embedding) > 0
      AND {voxel_expr} IS NOT NULL
    RETURN
        COALESCE(s.text, s.original_text, s.content, s.sentence) AS text,
        {voxel_expr} AS voxel_address,
        s.embedding AS embedding
    ORDER BY rand()
    LIMIT $limit
    """
    return [r.data() for r in session.run(query, limit=limit)]


def fetch_test_sentence(
    session,
    sentence_id: Optional[str],
    sentence_text: Optional[str],
) -> Optional[Dict[str, Any]]:
    if sentence_id:
        voxel_expr = build_voxel_expr("s")
        query = f"""
        MATCH (s)
        WHERE (s:SentenceNode OR s:PromptSentence)
          AND (s.sentence_id = $sentence_id OR s.node_id = $sentence_id)
        RETURN
            COALESCE(s.sentence_id, s.node_id) AS sentence_id,
            COALESCE(s.text, s.original_text, s.content, s.sentence) AS text,
            {voxel_expr} AS voxel_address,
            s.embedding AS embedding
        LIMIT 1
        """
        params = {"sentence_id": sentence_id}
        record = session.run(query, **params).single()
        return record.data() if record else None

    if sentence_text:
        voxel_expr = build_voxel_expr("s")
        query = f"""
        MATCH (s)
        WHERE (s:SentenceNode OR s:PromptSentence)
          AND COALESCE(s.text, s.original_text, s.content, s.sentence) = $sentence_text
        RETURN
            COALESCE(s.sentence_id, s.node_id) AS sentence_id,
            COALESCE(s.text, s.original_text, s.content, s.sentence) AS text,
            {voxel_expr} AS voxel_address,
            s.embedding AS embedding
        LIMIT 1
        """
        params = {"sentence_text": sentence_text}
        record = session.run(query, **params).single()
        return record.data() if record else None

    return None


def query_non_paraphrase_samples_ssm(
    limit: int,
    instance_id: str,
    password: str,
    database: str,
) -> List[Dict[str, Any]]:
    text_expr = sanitize_text_expr("COALESCE(s.text, s.original_text, s.content, s.sentence)")
    embedding_expr = embedding_to_string_expr("s.embedding")
    row_expr = (
        f"COALESCE(s.sentence_id, s.node_id, '') + '||' + {text_expr} + '||' + {embedding_expr}"
    )
    query = f"""
    MATCH (s)-[:member_of]->(g)
    WHERE (g:ParaphraseGroup OR g:PromptParaphraseGroup)
      AND (s:SentenceNode OR s:PromptSentence)
      AND s.embedding IS NOT NULL AND size(s.embedding) > 0
    MATCH (member)-[:member_of]->(g)
    WITH s, count(DISTINCT member) AS group_size
    WHERE group_size = 1
    RETURN {row_expr} AS row
    ORDER BY rand()
    LIMIT {limit}
    """
    output = run_cypher_ssm(query, instance_id, password, database)
    return parse_non_paraphrase_output_ssm(output)


def detect_delimiter(line: str) -> str:
    if "|" in line:
        return "|"
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return "|"


def parse_plain_rows(output: str, expected_cols: int) -> List[List[str]]:
    rows: List[List[str]] = []
    delimiter: Optional[str] = None
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("warning"):
            continue
        if delimiter is None:
            delimiter = detect_delimiter(stripped)
        parts = [p.strip() for p in stripped.split(delimiter)]
        if len(parts) != expected_cols:
            continue
        lowered = [p.lower() for p in parts]
        if "group_id" in lowered or "sentence_id" in lowered or "embedding" in lowered:
            continue
        rows.append(parts)
    return rows


def parse_embedding_list(value: str) -> List[float]:
    text = value.strip()
    if not text or text.lower() == "null":
        return []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        delimiter = ";" if ";" in inner else ","
        parts = [p.strip() for p in inner.split(delimiter) if p.strip()]
        return [float(p) for p in parts]
    raise ValueError(f"Unexpected embedding format: {value}")


def parse_paraphrase_output(output: str) -> List[Dict[str, Any]]:
    rows = parse_plain_rows(output, expected_cols=5)
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        group_id, sentence_id, text, embedding_str, other_text = row
        embedding = parse_embedding_list(embedding_str)
        parsed.append(
            {
                "group_id": group_id,
                "sentence_id": sentence_id,
                "text": text,
                "embedding": embedding,
                "other_text": other_text,
            }
        )
    return parsed


def parse_single_column_rows(output: str) -> List[str]:
    rows: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == "row":
            continue
        if stripped.lower().startswith("warning"):
            continue
        if stripped.startswith('"') and stripped.endswith('"'):
            stripped = stripped[1:-1]
        rows.append(stripped)
    return rows


def parse_paraphrase_output_ssm(output: str) -> List[Dict[str, Any]]:
    rows = parse_single_column_rows(output)
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        parts = row.split("||")
        if len(parts) != 5:
            continue
        group_id, sentence_id, text, embedding_str, other_text = parts
        embedding = parse_embedding_list(embedding_str)
        parsed.append(
            {
                "group_id": group_id,
                "sentence_id": sentence_id,
                "text": text,
                "embedding": embedding,
                "other_text": other_text,
            }
        )
    return parsed


def parse_non_paraphrase_output_ssm(output: str) -> List[Dict[str, Any]]:
    rows = parse_single_column_rows(output)
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        parts = row.split("||")
        if len(parts) != 3:
            continue
        sentence_id, text, embedding_str = parts
        embedding = parse_embedding_list(embedding_str)
        parsed.append(
            {
                "sentence_id": sentence_id,
                "text": text,
                "embedding": embedding,
            }
        )
    return parsed


def dedupe_paraphrase_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_groups: set[str] = set()
    result: List[Dict[str, Any]] = []
    for row in rows:
        group_id = row.get("group_id") or ""
        if not group_id or group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        result.append(row)
    return result


def parse_non_paraphrase_output(output: str) -> List[Dict[str, Any]]:
    rows = parse_plain_rows(output, expected_cols=3)
    parsed: List[Dict[str, Any]] = []
    for row in rows:
        sentence_id, text, embedding_str = row
        embedding = parse_embedding_list(embedding_str)
        parsed.append(
            {
                "sentence_id": sentence_id,
                "text": text,
                "embedding": embedding,
            }
        )
    return parsed


def build_pixel_voxel_description() -> List[str]:
    return [
        "The pixel or voxel space is a three dimensional grid over normalized PSA coordinates.",
        "Each coarse voxel is addressed as L{level}_v_{x}_{y}_{z} to encode grid resolution and index.",
        "Micro cell refinement appends 10x10 angular indices such as L1_v_9_9_12.47.23.81.",
        "Voxels subdivide as density grows while preserving the same embedding as the source of truth.",
        "The space is used for routing, clustering, and heatmap visualization, not embedding storage.",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute guide star cosine diffs and sentence diffs."
    )
    parser.add_argument(
        "--output",
        default="scripts/research/psa/spx/exports/guide_star_cosine_diffs.json",
        help="Output JSON path.",
    )
    parser.add_argument("--paraphrase-count", type=int, default=10)
    parser.add_argument("--non-paraphrase-count", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--embed-endpoint", default=None)
    parser.add_argument("--star-embeddings", default=None)
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument(
        "--spx-frame-version", default=os.getenv("SPX_FRAME_VERSION", "v2_115stars")
    )
    parser.add_argument(
        "--llm-sentence",
        default=None,
        help="LLM-generated sentence to embed with Qwen and compare to test sentence.",
    )
    parser.add_argument(
        "--llm-sentence-file",
        default=None,
        help="Path to file containing LLM-generated sentence.",
    )
    parser.add_argument(
        "--test-sentence-id",
        default=None,
        help="Sentence node id or sentence_id to use as test sentence.",
    )
    parser.add_argument(
        "--test-sentence-text",
        default=None,
        help="Exact sentence text to locate a test sentence in Neo4j.",
    )
    parser.add_argument(
        "--qwen-embed-endpoint",
        default=os.getenv("EMBEDDING_SERVICE_URL"),
        help="Qwen embedding endpoint base URL.",
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--ssm-parameter", default="/example/neo4j/password")
    parser.add_argument("--use-ssm", action="store_true", help="Query Neo4j via SSM on EC2.")
    parser.add_argument("--ssm-instance-id", default=DEFAULT_SSM_INSTANCE_ID)
    args = parser.parse_args()
    guide_star_catalog = load_guide_star_catalog()
    star_texts = [star.text for star in guide_star_catalog]
    star_meta = [
        {
            "id": star.id,
            "tier": star.tier.name,
            "text": star.text,
        }
        for star in guide_star_catalog
    ]
    s3_bucket = args.s3_bucket or os.getenv("CLUSTER_STATE_S3_BUCKET")
    embed_endpoint = (
        args.embed_endpoint
        or os.getenv("GUIDE_STAR_EMBED_ENDPOINT")
        or os.getenv("EMBEDDING_SERVICE_URL")
    )
    qwen_endpoint = args.qwen_embed_endpoint or embed_endpoint

    try:
        star_embeddings, star_source = load_star_embeddings(
            star_texts=star_texts,
            star_embeddings_path=args.star_embeddings,
            embed_endpoint=embed_endpoint,
            s3_bucket=s3_bucket,
            spx_frame_version=args.spx_frame_version,
        )
    except Exception as exc:
        print(f"ERROR: Failed to load guide star embeddings: {exc}")
        return 1

    if star_embeddings.shape[0] != len(guide_star_catalog):
        print(
            "ERROR: Guide star embeddings count mismatch: "
            f"{star_embeddings.shape[0]} != {len(guide_star_catalog)}"
        )
        return 1

    guide_star_diffs = cosine_diffs_matrix(star_embeddings)
    guide_star_diffs = np.round(guide_star_diffs, decimals=6)
    star_norm = normalize_rows(star_embeddings)

    llm_text: Optional[str] = None
    if args.llm_sentence_file:
        llm_text = Path(args.llm_sentence_file).read_text(encoding="utf-8").strip()
    elif args.llm_sentence:
        llm_text = args.llm_sentence.strip()

    if llm_text and args.use_ssm:
        print("ERROR: LLM/test sentence diff requires direct Neo4j access (no --use-ssm).")
        return 1

    if GraphDatabase is None and not args.use_ssm:
        print("ERROR: Missing neo4j driver. Install 'neo4j' to run this script.")
        return 1

    password = args.neo4j_password or fetch_password_via_ssm(args.ssm_parameter)
    if not password:
        print("ERROR: Neo4j password not found. Set NEO4J_PASSWORD or configure SSM.")
        return 1

    sentence_entries: List[Dict[str, Any]] = []
    sample_sentences: List[Dict[str, str]] = []
    llm_test_diff: Optional[Dict[str, Any]] = None

    if args.use_ssm:
        if not args.ssm_instance_id:
            print("ERROR: SSM instance id is required for --use-ssm.")
            return 1
        if not s3_bucket:
            print("ERROR: --use-ssm requires --s3-bucket or CLUSTER_STATE_S3_BUCKET.")
            return 1
        try:
            sentence_entries, sample_sentences = fetch_sentence_diffs_ssm(
                instance_id=args.ssm_instance_id,
                password=password,
                database=args.neo4j_database,
                s3_bucket=s3_bucket,
                spx_frame_version=args.spx_frame_version,
                paraphrase_count=args.paraphrase_count,
                non_paraphrase_count=args.non_paraphrase_count,
                sample_count=args.sample_count,
            )
        except Exception as exc:
            print(f"ERROR: SSM sentence diff run failed: {exc}")
            return 1
        if len(sentence_entries) < (args.paraphrase_count + args.non_paraphrase_count):
            print(
                f"ERROR: Only found {len(sentence_entries)} sentence entries "
                f"(need {args.paraphrase_count + args.non_paraphrase_count})."
            )
            return 1
    else:
        driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, password))
        try:
            try:
                driver.verify_connectivity()
            except Exception as exc:
                print(f"ERROR: Neo4j connection failed: {exc}")
                return 1
            with driver.session(database=args.neo4j_database) as session:
                para_rows = query_paraphrase_samples(session, args.paraphrase_count)
                non_para_rows = query_non_paraphrase_samples(session, args.non_paraphrase_count)
                sample_rows = query_sample_sentences(session, args.sample_count)
                if llm_text:
                    test_sentence = fetch_test_sentence(
                        session,
                        sentence_id=args.test_sentence_id,
                        sentence_text=args.test_sentence_text,
                    )
        finally:
            driver.close()

        if len(para_rows) < args.paraphrase_count:
            print(
                f"ERROR: Only found {len(para_rows)} paraphrase samples "
                f"(need {args.paraphrase_count})."
            )
            return 1

        if len(non_para_rows) < args.non_paraphrase_count:
            print(
                f"ERROR: Only found {len(non_para_rows)} non-paraphrase samples "
                f"(need {args.non_paraphrase_count})."
            )
            return 1

        def add_sentence_entry(
            sentence_id: str,
            text: str,
            embedding: List[float],
            group_id: Optional[str] = None,
            voxel_address: Optional[str] = None,
        ) -> None:
            vec = np.array(embedding, dtype=np.float32)
            normed = normalize_vector(vec)
            if normed is None:
                raise RuntimeError(f"Zero-norm embedding for sentence_id={sentence_id}")
            sims = star_norm @ normed
            diffs = (1.0 - sims).astype(np.float32)
            diffs = np.round(diffs, decimals=6).tolist()
            if not voxel_address:
                raise RuntimeError(f"Missing voxel address for sentence_id={sentence_id}")
            sentence_entries.append(
                {
                    "sentence_id": sentence_id,
                    "text": text,
                    "paraphrase_group_id": group_id,
                    "cosine_diffs": diffs,
                    "voxel_address": voxel_address,
                }
            )

        for row in para_rows:
            add_sentence_entry(
                sentence_id=row["sentence_id"],
                text=row["text"],
                embedding=row["embedding"],
                group_id=row.get("group_id"),
                voxel_address=row.get("voxel_address"),
            )

        for row in non_para_rows:
            add_sentence_entry(
                sentence_id=row["sentence_id"],
                text=row["text"],
                embedding=row["embedding"],
                voxel_address=row.get("voxel_address"),
            )

        for row in sample_rows:
            if row.get("voxel_address"):
                sample_sentences.append(
                    {
                        "text": row["text"],
                        "voxel_address": row["voxel_address"],
                    }
                )

        if llm_text:
            if not test_sentence:
                print("ERROR: Test sentence not found for LLM diff.")
                return 1
            if not qwen_endpoint:
                print("ERROR: Qwen embedding endpoint not configured.")
                return 1
            llm_embedding = embed_texts(qwen_endpoint, [llm_text])[0]
            test_embedding = np.array(test_sentence["embedding"], dtype=np.float32)
            llm_embedding = np.array(llm_embedding, dtype=np.float32)
            diff = cosine_diff(llm_embedding, test_embedding)
            llm_test_diff = {
                "llm_sentence": llm_text,
                "test_sentence_id": test_sentence.get("sentence_id"),
                "test_sentence_text": test_sentence.get("text"),
                "test_sentence_voxel_address": test_sentence.get("voxel_address"),
                "cosine_diff": round(diff, 6),
            }

    output_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "cosine_diff_definition": "1 - cosine_similarity",
        "guide_star_embedding_source": star_source,
        "embedding_dim": int(star_embeddings.shape[1]),
        "guide_stars": star_meta,
        "guide_star_cosine_diffs": guide_star_diffs.tolist(),
        "pixel_voxel_description": build_pixel_voxel_description(),
        "sentences": sentence_entries,
    }
    if llm_test_diff:
        output_payload["llm_test_sentence_diff"] = llm_test_diff

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"OK: Wrote JSON to {output_path}")
    if sample_sentences:
        print("Sample voxel addresses:")
        for item in sample_sentences[: args.sample_count]:
            print(f"- voxel_address={item['voxel_address']}")
            print(f"  text={item['text']}")
    if llm_test_diff:
        print("LLM/test cosine diff:")
        print(f"- cosine_diff={llm_test_diff['cosine_diff']}")
        print(f"  test_sentence_id={llm_test_diff.get('test_sentence_id')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
