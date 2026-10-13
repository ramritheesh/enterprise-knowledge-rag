
from app.ingestion.vector_store import hybrid_search_documents


TOP_K = 5

TEST_QUESTIONS = [
    {
        "question": "Who approves annual leave requests?",
        "expected": "direct manager",
    },
    {
        "question": "What exceptions are allowed for annual leave carryover?",
        "expected": "carryover exceptions",
    },
    {
        "question": (
            "Where should employees go for clarification about parental leave?"
        ),
        "expected": "contact hr",
    },
    {
        "question": "How long is parental leave?",
        "expected": "does not define parental-leave",
    },
    {
        "question": "What is the notice period for annual leave?",
        "expected": "five working days",
    },
]


def main():
    reciprocal_ranks = []
    hits = 0

    print("=" * 80)
    print("HYBRID RETRIEVAL RANKING DIAGNOSTICS")
    print("=" * 80)

    for item in TEST_QUESTIONS:
        question = item["question"]
        expected = item["expected"].lower()

        results = hybrid_search_documents(
            question,
            top_k=TOP_K,
        )

        matching_rank = None

        print(f"\nQuestion: {question}")
        print(f"Expected evidence: {expected}")

        for rank, result in enumerate(results, start=1):
            text = result.get("text", "")
            distance = result.get("distance")
            source = (
                result.get("metadata") or {}
            ).get("source", "Unknown")

            if (
                matching_rank is None
                and expected in text.lower()
            ):
                matching_rank = rank

            distance_text = (
                f"{distance:.4f}"
                if isinstance(distance, (int, float))
                else "N/A"
            )

            rrf_score = result.get("rrf_score", 0.0)

            print(
                f"  Rank {rank} | "
                f"RRF {rrf_score:.5f} | "
                f"Distance {distance_text} | "
                f"Source {source}"
            )
            print(f"    {text[:180]}")

        if matching_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / matching_rank)

            print(f"  RESULT: PASS at rank {matching_rank}")
        else:
            reciprocal_ranks.append(0.0)

            print(
                "  RESULT: FAIL — expected evidence not found"
            )

    total = len(TEST_QUESTIONS)
    hit_rate = hits / total if total else 0.0
    mrr = (
        sum(reciprocal_ranks) / total
        if total
        else 0.0
    )

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL SUMMARY")
    print("=" * 80)
    print(
        f"Hit Rate@{TOP_K}: "
        f"{hits}/{total} ({hit_rate:.2%})"
    )
    print(f"MRR@{TOP_K}: {mrr:.4f}")
    print("Previous semantic-only MRR@5: 0.4333")
    print("=" * 80)


if __name__ == "__main__":
    main()
