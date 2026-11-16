#!/usr/bin/env python3
"""
Quick F1 Evaluation from Neo4j - matches actual schema
Queries SentenceNode and ParaphraseGroup labels
"""

import os
import subprocess
from pathlib import Path

from neo4j import GraphDatabase

# MRPC data
MRPC_DIR = Path(__file__).resolve().parents[3] / "app" / "utils" / "msrpc_split"


def normalize_text(text: str) -> str:
    text = text.strip().rstrip("\\")
    return " ".join(text.split()).lower()


def fetch_password_via_ssm():
    try:
        return subprocess.check_output(
            [
                "aws",
                "ssm",
                "get-parameter",
                "--name",
                "/example/neo4j/password",
                "--with-decryption",
                "--query",
                "Parameter.Value",
                "--output",
                "text",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD") or fetch_password_via_ssm()

    if not password:
        print("❌ No password - set NEO4J_PASSWORD or configure AWS SSM")
        return

    print(f"🔌 Connecting to {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        driver.verify_connectivity()
        print("   ✓ Connected\n")

        with driver.session() as session:
            # Database summary
            print("🗄️ DATABASE SUMMARY")
            print("=" * 50)

            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
                ORDER BY count DESC
            """)
            for r in result:
                print(f"   {r['label']}: {r['count']}")

            # Get paraphrase groups with their sentences
            print("\n📊 QUERYING PARAPHRASE GROUPS...")

            # Query sentence -> group mapping (adjust relationship name as needed)
            result = session.run("""
                MATCH (s:SentenceNode)
                OPTIONAL MATCH (s)-[:member_of]->(g:ParaphraseGroup)
                OPTIONAL MATCH (member:SentenceNode)-[:member_of]->(g)
                WITH s, g, count(DISTINCT member) AS group_size
                RETURN s.text AS text,
                       CASE WHEN group_size >= 2 THEN g.node_id ELSE null END AS group_id
            """)

            text_to_group = {}
            for r in result:
                if r["text"]:
                    text_to_group[normalize_text(r["text"])] = r["group_id"]

            print(f"   Total sentences: {len(text_to_group)}")
            with_group = sum(1 for g in text_to_group.values() if g)
            print(f"   With group: {with_group}")

            # Load MRPC sample_100 pairs
            print("\n📂 LOADING MRPC 100-PAIR SAMPLE...")

            first_file = MRPC_DIR / "sample_100_first_sentences.txt"
            second_file = MRPC_DIR / "sample_100_second_sentences.txt"
            mapping_file = MRPC_DIR / "sample_100_pair_mapping.tsv"

            first_sentences = {}
            with open(first_file) as f:
                for i, line in enumerate(f, 1):
                    first_sentences[i] = line.strip()

            second_sentences = {}
            with open(second_file) as f:
                for i, line in enumerate(f, 1):
                    second_sentences[i] = line.strip()

            pairs = []
            with open(mapping_file) as f:
                next(f)  # header
                for line in f:
                    parts = line.strip().split("\t")
                    pairs.append(
                        {"pair_id": int(parts[0]), "is_paraphrase": parts[4].lower() == "true"}
                    )

            print(f"   Pairs loaded: {len(pairs)}")
            true_paras = sum(1 for p in pairs if p["is_paraphrase"])
            print(f"   True paraphrases: {true_paras}")
            print(f"   Not paraphrases: {len(pairs) - true_paras}")

            # Evaluate
            print("\n⚙️ EVALUATING F1...")

            tp, fp, fn, tn = 0, 0, 0, 0
            missing = 0

            for pair in pairs:
                pid = pair["pair_id"]
                is_para = pair["is_paraphrase"]

                first = first_sentences.get(pid)
                second = second_sentences.get(pid)

                if not first or not second:
                    continue

                first_norm = normalize_text(first)
                second_norm = normalize_text(second)

                first_group = text_to_group.get(first_norm)
                second_group = text_to_group.get(second_norm)

                if first_norm not in text_to_group or second_norm not in text_to_group:
                    missing += 1
                    continue

                # Prediction: same non-null group = paraphrase
                predicted = first_group and second_group and first_group == second_group

                if predicted and is_para:
                    tp += 1
                elif predicted and not is_para:
                    fp += 1
                elif not predicted and is_para:
                    fn += 1
                else:
                    tn += 1

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

            print("\n" + "=" * 60)
            print("🎯 MRPC F1 EVALUATION RESULTS")
            print("=" * 60)

            print("\n📊 Confusion Matrix:")
            print(f"   TP (correct paraphrases):    {tp:3d}")
            print(f"   FP (wrong groupings):        {fp:3d}")
            print(f"   FN (missed paraphrases):     {fn:3d}")
            print(f"   TN (correct separations):    {tn:3d}")
            print(f"   Missing sentences:           {missing:3d}")

            print("\n📈 METRICS:")
            print(f"   Precision: {precision:.4f}  ({precision * 100:.1f}%)")
            print(f"   Recall:    {recall:.4f}  ({recall * 100:.1f}%)")
            print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   🎯 F1 SCORE: {f1:.4f}  ({f1 * 100:.1f}%)")
            print("   ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   Accuracy:  {accuracy:.4f}  ({accuracy * 100:.1f}%)")

            print("\n" + "=" * 60)

            baseline = 0.908
            if f1 > baseline:
                print(f"🎉 BEATS BASELINE! {f1:.4f} > {baseline} (+{(f1 - baseline) * 100:.1f}%)")
            else:
                print(f"📊 vs baseline {baseline}: gap of {(baseline - f1) * 100:.1f}%")

            print("=" * 60)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
