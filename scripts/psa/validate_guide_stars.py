#!/usr/bin/env python3
"""
Guide Star PSA Validation Test
==============================
Tests whether the Guide Star architecture has legs:
1. FM-1: Star collapse check (no pair > 0.85 similarity)
2. Top-5 star overlap for paraphrase pairs (should be 3+ overlap)

Usage: python validate_guide_stars.py [--tei-url URL]
"""

import argparse
from typing import List, Tuple

import numpy as np
import requests

from consolidated_clustering_v2.routing.guide_stars import GUIDE_STAR_CATALOG

# TEI endpoint
DEFAULT_TEI_URL = "http://localhost:11435"

# ============================================================================
# GUIDE STARS - Subset for quick validation (Tier 0 + Tier 1 = 24 stars)
# ============================================================================

GUIDE_STARS_SUBSET = [
    # Tier 0: Existence & Assertion (contrastive pairs)
    "A physical object occupies space in the real world.",
    "The referenced object is completely absent and does not exist.",
    "A specific event happens at a documented point in time.",
    "Nothing happens and no events occur during the entire timeframe.",
    # Tier 0: Causality
    "The earlier action directly produces the later consequence.",
    "The blocking mechanism stops the expected result from occurring.",
    "The final outcome requires the prerequisite condition to be met first.",
    "The observable effect follows automatically from the root cause.",
    # Tier 0: Change
    "The original state transforms completely into a new and different state.",
    "The existing condition stays exactly the same with no alteration.",
    "The process initiates and starts running from the beginning.",
    "The activity terminates and stops at its conclusion.",
    # Tier 1: Action
    "A human agent deliberately performs a purposeful action.",
    "A human agent stops and discontinues their ongoing activity.",
    "A human agent builds and produces a new tangible artifact.",
    "A human agent operates an existing tool to accomplish a task.",
    # Tier 1: Intent
    "A person wants and desires to acquire a specific resource.",
    "A person tries and struggles to achieve a difficult goal.",
    "A person designs and prepares a detailed plan for future execution.",
    "A person decides and selects one option from multiple alternatives.",
    # Tier 1: Communication
    "A speaker transmits factual information to a listener.",
    "A requester asks for help and assistance from a provider.",
    "A person explicitly agrees with and supports the stated proposition.",
    "A person explicitly disagrees with and opposes the stated claim.",
]

GUIDE_STARS_FULL = [star.text for star in GUIDE_STAR_CATALOG]

# ============================================================================
# PARAPHRASE TEST PAIRS
# ============================================================================

PARAPHRASE_PAIRS = [
    # Obvious paraphrases
    ("The cat sat on the mat.", "A feline rested upon the rug."),
    ("The company increased its revenue by 20%.", "Revenue grew by a fifth for the business."),
    ("She quickly ran to the store.", "She hurried to the shop."),
    # Semantic equivalents with different structure
    (
        "The meeting was cancelled due to bad weather.",
        "Bad weather caused the meeting to be called off.",
    ),
    (
        "He failed to complete the assignment on time.",
        "The assignment wasn't finished by the deadline.",
    ),
    (
        "The software update fixed several bugs.",
        "Several bugs were resolved in the software patch.",
    ),
    # Business/technical paraphrases
    ("The project deadline has been extended.", "We've pushed back when the project is due."),
    (
        "Customer satisfaction improved significantly.",
        "There was a major boost in how happy customers are.",
    ),
    ("The system requires authentication.", "You need to log in to use the system."),
    # More challenging pairs
    ("Revenue declined in Q4.", "Fourth quarter saw a drop in income."),
    ("The proposal was rejected by the board.", "Board members voted against the proposal."),
    ("Implementation will begin next month.", "We'll start implementing in the coming month."),
]

# Non-paraphrase pairs (should have LOW overlap)
NON_PARAPHRASE_PAIRS = [
    ("The cat sat on the mat.", "Revenue increased by 20%."),
    ("She ran to the store.", "The system requires authentication."),
    ("The meeting was cancelled.", "The software update fixed bugs."),
]


def embed_texts(texts: List[str], tei_url: str) -> np.ndarray:
    """Embed texts using TEI endpoint."""
    response = requests.post(
        f"{tei_url}/embed",
        json={"inputs": texts},
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()
    return np.array(response.json())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_star_signature(embedding: np.ndarray, star_embeddings: np.ndarray) -> List[float]:
    """Compute similarities to all guide stars."""
    return [cosine_similarity(embedding, star) for star in star_embeddings]


def top_k_overlap(sig_a: List[float], sig_b: List[float], k: int = 5) -> int:
    """Count overlap in top-k most similar stars."""
    top_k_a = set(np.argsort(sig_a)[-k:])
    top_k_b = set(np.argsort(sig_b)[-k:])
    return len(top_k_a & top_k_b)


def test_star_collapse(star_embeddings: np.ndarray, threshold: float = 0.85) -> Tuple[bool, List]:
    """
    FM-1 Validation: Check that no star pair exceeds similarity threshold.
    Returns (passed, violations) where violations is list of (i, j, similarity) tuples.
    """
    violations = []
    n = len(star_embeddings)

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(star_embeddings[i], star_embeddings[j])
            if sim > threshold:
                violations.append((i, j, sim))

    return len(violations) == 0, violations


def pairwise_similarity_values(star_embeddings: np.ndarray) -> np.ndarray:
    """Return upper-triangle pairwise cosine similarities for guide stars."""
    sims = []
    n = len(star_embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_similarity(star_embeddings[i], star_embeddings[j]))
    return np.array(sims)


def nearest_neighbor_similarities(star_embeddings: np.ndarray) -> np.ndarray:
    """Return each guide star's nearest-neighbor cosine similarity."""
    n = len(star_embeddings)
    nearest = []
    for i in range(n):
        sims = [
            cosine_similarity(star_embeddings[i], star_embeddings[j]) for j in range(n) if i != j
        ]
        nearest.append(max(sims))
    return np.array(nearest)


def contrast_pair_similarities(star_embeddings: np.ndarray) -> List[Tuple[int, int, float]]:
    """Return cosine similarities for catalog contrast pairs."""
    pairs = []
    for star in GUIDE_STAR_CATALOG:
        if star.contrast_with is None or star.id > star.contrast_with:
            continue
        pairs.append(
            (
                star.id,
                star.contrast_with,
                cosine_similarity(star_embeddings[star.id], star_embeddings[star.contrast_with]),
            )
        )
    return pairs


def test_paraphrase_overlap(
    paraphrase_pairs: List[Tuple[str, str]], star_embeddings: np.ndarray, tei_url: str, k: int = 5
) -> Tuple[float, List[Tuple[str, str, int]]]:
    """
    Test top-k star overlap for paraphrase pairs.
    Returns (average_overlap, details).
    """
    # Flatten pairs for batch embedding
    all_sentences = []
    for a, b in paraphrase_pairs:
        all_sentences.extend([a, b])

    print(f"  Embedding {len(all_sentences)} sentences...")
    embeddings = embed_texts(all_sentences, tei_url)

    overlaps = []
    details = []

    for i, (sent_a, sent_b) in enumerate(paraphrase_pairs):
        emb_a = embeddings[i * 2]
        emb_b = embeddings[i * 2 + 1]

        sig_a = compute_star_signature(emb_a, star_embeddings)
        sig_b = compute_star_signature(emb_b, star_embeddings)

        overlap = top_k_overlap(sig_a, sig_b, k)
        overlaps.append(overlap)
        details.append((sent_a[:40], sent_b[:40], overlap))

    return np.mean(overlaps), details


def main():
    parser = argparse.ArgumentParser(description="Validate Guide Star PSA architecture")
    parser.add_argument("--tei-url", default=DEFAULT_TEI_URL, help="TEI endpoint URL")
    parser.add_argument(
        "--full", action="store_true", help="Run full guide-star catalog test (slower)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GUIDE STAR PSA VALIDATION TEST")
    print("=" * 70)
    print(f"TEI endpoint: {args.tei_url}")

    # Test connectivity
    print("\n1. Testing TEI connectivity...")
    try:
        r = requests.get(f"{args.tei_url}/health", timeout=10)
        print(f"   ✅ TEI is healthy (status {r.status_code})")
    except Exception as e:
        print(f"   ❌ TEI not reachable: {e}")
        return 1

    guide_stars = GUIDE_STARS_FULL if args.full else GUIDE_STARS_SUBSET

    # Embed guide stars
    print(f"\n2. Embedding {len(guide_stars)} guide stars...")
    try:
        star_embeddings = embed_texts(guide_stars, args.tei_url)
        print(f"   ✅ Embedded {len(star_embeddings)} stars, dim={star_embeddings.shape[1]}")
    except Exception as e:
        print(f"   ❌ Failed to embed stars: {e}")
        return 1

    # FM-1: Star collapse test
    print("\n3. FM-1 VALIDATION: Star Collapse Test")
    print("   Checking that no star pair > 0.85 similarity...")
    passed, violations = test_star_collapse(star_embeddings)

    if passed:
        print("   ✅ PASSED: No star collapse detected")
    else:
        print(f"   ⚠️  WARNING: {len(violations)} star pairs too similar:")
        for i, j, sim in violations[:5]:  # Show first 5
            print(f"      Stars {i} & {j}: {sim:.3f}")
            print(f"        {i}: '{guide_stars[i][:50]}...'")
            print(f"        {j}: '{guide_stars[j][:50]}...'")

    # Show star similarity distribution
    print("\n   Star similarity distribution:")
    sims = pairwise_similarity_values(star_embeddings)
    print(f"      Min: {sims.min():.3f}")
    print(f"      Max: {sims.max():.3f}")
    print(f"      Mean: {sims.mean():.3f}")
    print(f"      Std: {sims.std():.3f}")

    nearest = nearest_neighbor_similarities(star_embeddings)
    print("\n   Nearest-neighbor similarity distribution:")
    print(f"      Min: {nearest.min():.3f}")
    print(f"      P50: {np.percentile(nearest, 50):.3f}")
    print(f"      P90: {np.percentile(nearest, 90):.3f}")
    print(f"      Max: {nearest.max():.3f}")

    if args.full:
        contrast_pairs = contrast_pair_similarities(star_embeddings)
        contrast_values = np.array([sim for _, _, sim in contrast_pairs])
        print("\n   Contrast-pair similarity distribution:")
        print(f"      Count: {len(contrast_pairs)}")
        print(f"      Min: {contrast_values.min():.3f}")
        print(f"      P50: {np.percentile(contrast_values, 50):.3f}")
        print(f"      Max: {contrast_values.max():.3f}")

    # Top-5 overlap test for paraphrases
    print("\n4. PARAPHRASE OVERLAP TEST (Top-5 Stars)")
    print(f"   Testing {len(PARAPHRASE_PAIRS)} paraphrase pairs...")

    avg_overlap, details = test_paraphrase_overlap(
        PARAPHRASE_PAIRS, star_embeddings, args.tei_url, k=5
    )

    print("\n   Results:")
    for sent_a, sent_b, overlap in details:
        emoji = "✅" if overlap >= 3 else "⚠️" if overlap >= 2 else "❌"
        print(f"   {emoji} Overlap={overlap}: '{sent_a}...' <-> '{sent_b}...'")

    print(f"\n   Average top-5 overlap: {avg_overlap:.2f}")

    if avg_overlap >= 3.5:
        print("   ✅ EXCELLENT: PSA is highly viable for paraphrase routing")
    elif avg_overlap >= 2.5:
        print("   ⚠️  GOOD: PSA is viable with soft routing")
    elif avg_overlap >= 1.5:
        print("   ⚠️  WARNING: PSA may need more stars or tuning")
    else:
        print("   ❌ FAILURE: Star set is underpowered for paraphrase detection")

    # Test non-paraphrase pairs (should have LOW overlap)
    print("\n5. NON-PARAPHRASE CONTROL TEST")
    print(f"   Testing {len(NON_PARAPHRASE_PAIRS)} non-paraphrase pairs...")

    avg_non_overlap, non_details = test_paraphrase_overlap(
        NON_PARAPHRASE_PAIRS, star_embeddings, args.tei_url, k=5
    )

    print("\n   Results (should be LOW overlap):")
    for sent_a, sent_b, overlap in non_details:
        emoji = "✅" if overlap <= 2 else "⚠️"
        print(f"   {emoji} Overlap={overlap}: '{sent_a}...' <-> '{sent_b}...'")

    print(f"\n   Average top-5 overlap (non-paraphrases): {avg_non_overlap:.2f}")

    if avg_non_overlap < avg_overlap:
        print("   ✅ GOOD: Non-paraphrases have lower overlap than paraphrases")
    else:
        print("   ⚠️  WARNING: Non-paraphrases have similar overlap - poor discrimination")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Star collapse (FM-1):     {'PASS' if passed else 'WARN'}")
    print(f"Paraphrase overlap:       {avg_overlap:.2f} (target: ≥3.0)")
    print(f"Non-paraphrase overlap:   {avg_non_overlap:.2f} (target: <2.0)")
    print(f"Discrimination gap:       {avg_overlap - avg_non_overlap:.2f} (target: ≥1.0)")

    if passed and avg_overlap >= 3.0 and (avg_overlap - avg_non_overlap) >= 1.0:
        print("\n🎉 Guide Star PSA has legs! Proceed with implementation.")
        return 0
    elif avg_overlap >= 2.0:
        print("\n⚠️  Guide Star PSA shows promise but needs tuning.")
        return 0
    else:
        print("\n❌ Guide Star PSA needs significant work before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
