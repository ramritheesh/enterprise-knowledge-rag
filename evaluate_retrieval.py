
import json
from datetime import datetime, timezone
from pathlib import Path

from app.generation.answer import answer_question
from app.ingestion.vector_store import search_documents


TOP_K = 5

EVALUATION_QUESTIONS = [
    {
        "question": "How many annual leave days do employees receive?",
        "expected_keyword": "20 days of paid annual leave",
    },
    {
        "question": "How many annual leave days are provided each year?",
        "expected_keyword": "20 days of paid annual leave",
    },
    {
        "question": "How much yearly paid leave is available?",
        "expected_keyword": "20 days of paid annual leave",
    },
    {
        "question": "How early should employees apply for annual leave?",
        "expected_keyword": "five working days",
    },
    {
        "question": "What is the notice period for annual leave?",
        "expected_keyword": "five working days",
    },
    {
        "question": "Who approves annual leave requests?",
        "expected_keyword": "direct manager",
    },
    {
        "question": "Who reviews an annual leave request?",
        "expected_keyword": "direct manager",
    },
    {
        "question": "Can unused annual leave carry over to the next year?",
        "expected_keyword": "does not automatically carry over",
    },
    {
        "question": "What exceptions are allowed for annual leave carryover?",
        "expected_keyword": "carryover exceptions",
    },
    {
        "question": "When should an employee notify their manager about sick leave?",
        "expected_keyword": "as soon as reasonably possible",
    },
    {
        "question": "How many paid sick-leave days are guaranteed?",
        "expected_keyword": (
            "does not specify a fixed number of paid sick-leave days"
        ),
    },
    {
        "question": "Does the policy define parental-leave eligibility?",
        "expected_keyword": "does not define parental-leave",
    },
    {
        "question": "How long is parental leave?",
        "expected_keyword": "does not define parental-leave",
    },
    {
        "question": "Is parental leave paid under this policy?",
        "expected_keyword": "does not define parental-leave",
    },
    {
        "question": "How do employees apply for parental leave?",
        "expected_keyword": "does not define parental-leave",
    },
    {
        "question": "Does the policy mention remote work?",
        "expected_keyword": "remote work",
    },
    {
        "question": "What is the company's travel reimbursement limit?",
        "expected_keyword": None,
    },
    {
        "question": "How many vacation days do employees get?",
        "expected_keyword": "20 days of paid annual leave",
    },
    {
        "question": (
            "Where should employees go for clarification about parental leave?"
        ),
        "expected_keyword": "contact HR",
    },
]

ABSTENTION_QUESTIONS = [
    "What work-from-home allowance do employees receive?",
    "What is the company's travel reimbursement limit?",
]

ABSTENTION_MESSAGE = (
    "I couldn't find sufficiently relevant evidence "
    "in the available documents to answer this question."
)

REPORTS_DIR = Path("reports")


def evaluate_retrieval():
    """Evaluate retrieval using Hit Rate@K and MRR@K."""

    answerable_questions = [
        item
        for item in EVALUATION_QUESTIONS
        if item["expected_keyword"] is not None
    ]

    hits = 0
    reciprocal_rank_sum = 0.0
    question_results = []

    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    for number, item in enumerate(answerable_questions, start=1):
        question = item["question"]
        expected_keyword = item["expected_keyword"].lower()

        results = search_documents(question, top_k=TOP_K)
        matching_rank = None
        retrieved_results = []

        print(f"\nQ{number}: {question}")

        for rank, result in enumerate(results, start=1):
            text = result.get("text", "")
            metadata = result.get("metadata") or {}
            source = metadata.get("source", "Unknown")
            distance = result.get("distance")

            retrieved_results.append({
                "rank": rank,
                "source": source,
                "distance": distance,
                "evidence": text[:220],
            })

            distance_text = (
                f"{distance:.4f}"
                if isinstance(distance, (int, float))
                else "N/A"
            )

            print(
                f"  Rank {rank} | Source: {source} "
                f"| Distance: {distance_text}"
            )
            print(f"  Evidence: {text[:220]}")

            if (
                matching_rank is None
                and expected_keyword in text.lower()
            ):
                matching_rank = rank

        passed = matching_rank is not None

        if passed:
            hits += 1
            reciprocal_rank_sum += 1.0 / matching_rank
            print(f"  PASS: Evidence found at rank {matching_rank}.")
        else:
            print("  FAIL: Expected keyword not found in top results.")

        question_results.append({
            "question": question,
            "expected_keyword": item["expected_keyword"],
            "passed": passed,
            "matching_rank": matching_rank,
            "reciprocal_rank": (
                1.0 / matching_rank if matching_rank else 0.0
            ),
            "retrieved_results": retrieved_results,
        })

    total = len(answerable_questions)
    hit_rate = hits / total if total else 0.0
    mrr = reciprocal_rank_sum / total if total else 0.0

    print("\n" + "=" * 80)
    print("RETRIEVAL SUMMARY")
    print("=" * 80)
    print(f"Answerable questions: {total}")
    print(f"Retrieval hits:       {hits}/{total}")
    print(f"Hit Rate@{TOP_K}:           {hit_rate:.2%}")
    print(f"MRR@{TOP_K}:                {mrr:.4f}")

    return {
        "top_k": TOP_K,
        "passed": hits,
        "total": total,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "results": question_results,
    }


def evaluate_abstention():
    """Check whether unsupported questions receive an abstention."""

    question_results = []

    print("\n" + "=" * 80)
    print("ANSWER ABSTENTION EVALUATION")
    print("=" * 80)

    for number, question in enumerate(ABSTENTION_QUESTIONS, start=1):
        print(f"\nUnsupported question {number}: {question}")

        try:
            result = answer_question(question, top_k=TOP_K)
            answer = str(result.get("answer", "")).strip()
            sources = result.get("sources", [])

            passed = ABSTENTION_MESSAGE.lower() in answer.lower()

            print(f"Answer: {answer}")
            print(f"Sources returned: {len(sources)}")
            print(
                "PASS: Expected abstention returned."
                if passed
                else "FAIL: Expected abstention message was not returned."
            )

            question_results.append({
                "question": question,
                "passed": passed,
                "answer": answer,
                "source_count": len(sources),
                "error": None,
            })

        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}")

            question_results.append({
                "question": question,
                "passed": False,
                "answer": None,
                "source_count": 0,
                "error": f"{type(error).__name__}: {error}",
            })

    passed = sum(item["passed"] for item in question_results)
    total = len(question_results)

    print("\n" + "-" * 80)
    print(f"Abstention checks passed: {passed}/{total}")
    print("-" * 80)

    return {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "results": question_results,
    }


def save_retrieval_report(retrieval, abstention):
    """Save retrieval and abstention results to a JSON report."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)
    timestamp_text = timestamp.strftime("%Y%m%dT%H%M%SZ")

    report = {
        "evaluation_type": "retrieval_and_abstention",
        "timestamp_utc": timestamp.isoformat(),
        "retrieval": retrieval,
        "abstention": abstention,
    }

    report_path = REPORTS_DIR / (
        f"retrieval_evaluation_{timestamp_text}.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nRetrieval report saved to: {report_path}")

    return report


def main():
    retrieval = evaluate_retrieval()
    abstention = evaluate_abstention()

    save_retrieval_report(retrieval, abstention)

    print("\n" + "=" * 80)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 80)
    print(
        f"Retrieval: {retrieval['passed']}/{retrieval['total']} "
        f"({retrieval['hit_rate']:.2%})"
    )
    print(
        f"Abstention: {abstention['passed']}/{abstention['total']} "
        f"({abstention['pass_rate']:.2%})"
    )


if __name__ == "__main__":
    main()
