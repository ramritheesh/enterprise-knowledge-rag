
import json
from datetime import datetime, timezone
from pathlib import Path

from app.generation.answer import answer_question


REPORTS_DIR = Path("reports")


# Normalize wording for evaluation only. Original source text is unchanged.
def normalize_for_matching(text: str) -> str:
    return (
        " ".join(
            str(text)
            .lower()
            .replace("-", " ")
            .replace("–", " ")
            .replace("—", " ")
            .replace("’", "'")
            .split()
        )
    )


def extract_source_texts(result: dict) -> list[str]:
    """Extract evidence snippets from the answer result."""
    source_texts = []

    for source in result.get("sources", []):
        if isinstance(source, dict):
            text = source.get("text", "")
            if text:
                source_texts.append(str(text))
        elif isinstance(source, str):
            source_texts.append(source)

    return source_texts


def contains_any(text: str, alternatives: list[str]) -> bool:
    """Return True if any alternative phrase occurs in normalized text."""
    normalized = normalize_for_matching(text)

    return any(
        normalize_for_matching(phrase) in normalized
        for phrase in alternatives
    )


def missing_fact_groups(
    text: str,
    expected_groups: list[list[str]],
) -> list[list[str]]:
    """Return expected phrase groups that were not found."""
    return [
        group
        for group in expected_groups
        if not contains_any(text, group)
    ]


def source_missing_fact_groups(
    source_texts: list[str],
    expected_groups: list[list[str]],
) -> list[list[str]]:
    """Check whether each expected fact appears in at least one source."""
    return missing_fact_groups("\n".join(source_texts), expected_groups)


ANSWERABLE_QUESTIONS = [
    {
        "question": "How many annual leave days do employees receive?",
        "answer_facts": [
            ["20 days", "twenty days"],
            ["annual leave"],
        ],
        "source_facts": [
            ["20 days", "twenty days"],
            ["annual leave"],
        ],
    },
    {
        "question": "How early should employees apply for annual leave?",
        "answer_facts": [
            ["five working days", "5 working days"],
            ["before the intended leave date"],
        ],
        "source_facts": [["five working days", "5 working days"]],
    },
    {
        "question": "Who approves annual leave requests?",
        "answer_facts": [
            ["direct manager", "employee's manager"],
            ["approve", "approval"],
        ],
        "source_facts": [
            ["direct manager", "employee's manager"],
            ["approve", "approval"],
        ],
    },
    {
        "question": "Can unused annual leave carry over to the next year?",
        "answer_facts": [
            [
                "does not automatically carry over",
                "not automatically carry over",
            ],
        ],
        "source_facts": [
            [
                "does not automatically carry over",
                "not automatically carry over",
            ],
        ],
    },
    {
        "question": (
            "When should an employee notify their manager about sick leave?"
        ),
        "answer_facts": [
            ["as soon as reasonably possible"],
            ["manager"],
        ],
        "source_facts": [
            ["as soon as reasonably possible"],
            ["cannot work because of illness"],
        ],
    },
    {
        "question": "How many paid sick-leave days are guaranteed?",
        "answer_facts": [
            [
                "does not specify a fixed number",
                "not specify a fixed number",
            ],
        ],
        "source_facts": [
            [
                "does not specify a fixed number",
                "not specify a fixed number",
            ],
        ],
    },
    {
        "question": "Does the policy define parental-leave eligibility?",
        "answer_facts": [
            ["parental leave"],
            ["does not specify", "does not define", "not defined"],
        ],
        "source_facts": [
            ["parental leave"],
            ["does not define", "does not specify"],
        ],
    },
    {
        "question": "How long is parental leave?",
        "answer_facts": [
            ["parental leave"],
            ["does not define", "not defined", "does not specify"],
            ["duration"],
        ],
        "source_facts": [
            ["parental leave"],
            ["does not define", "does not specify"],
            ["duration"],
        ],
    },
    {
        "question": "Is parental leave paid under this policy?",
        "answer_facts": [
            ["parental leave"],
            ["does not define", "not defined", "does not specify"],
            ["pay"],
        ],
        "source_facts": [
            ["parental leave"],
            ["does not define", "does not specify"],
            ["pay"],
        ],
    },
    {
        "question": "How do employees apply for parental leave?",
        "answer_facts": [
            ["parental leave"],
            ["does not define", "does not specify"],
            ["application procedures", "application process"],
        ],
        "source_facts": [
            ["parental leave"],
            ["application procedures"],
        ],
    },
    {
        "question": "Does the policy mention remote work?",
        "answer_facts": [
            ["remote work"],
            ["does not define", "not defined", "does not specify"],
        ],
        "source_facts": [
            ["remote work"],
            ["does not define", "does not specify"],
        ],
    },
    {
        "question": "How many vacation days do employees get?",
        "answer_facts": [["20 days", "twenty days"]],
        "source_facts": [["20 days", "twenty days"]],
    },
    {
        "question": (
            "Where should employees go for clarification about parental leave?"
        ),
        "answer_facts": [
            ["contact hr", "human resources"],
            ["parental leave"],
        ],
        "source_facts": [
            ["contact hr", "human resources"],
            ["parental leave"],
        ],
    },
]


UNSUPPORTED_QUESTIONS = [
    "What work-from-home allowance do employees receive?",
    "What is the company's travel reimbursement limit?",
]

ABSTENTION_PHRASES = [
    "couldn't find sufficiently relevant evidence",
    "cannot determine the answer",
    "cannot determine from the available document",
    "insufficient evidence",
    "not enough evidence",
]


def is_abstention(answer: str) -> bool:
    return contains_any(answer, ABSTENTION_PHRASES)


def evaluate_answerable_questions() -> list[dict]:
    print("=" * 80)
    print("ANSWER QUALITY EVALUATION")
    print("=" * 80)

    results = []

    for index, item in enumerate(ANSWERABLE_QUESTIONS, start=1):
        question = item["question"]
        print(f"\nQ{index}: {question}")

        result = answer_question(question)
        answer = str(result.get("answer", ""))
        sources = result.get("sources", [])
        source_texts = extract_source_texts(result)

        print(f"Answer: {answer}")
        print(f"Sources returned: {len(sources)}")

        failures = []

        if is_abstention(answer):
            failures.append("System abstained on an answerable question.")

        answer_missing = missing_fact_groups(
            answer,
            item["answer_facts"],
        )
        if answer_missing:
            failures.append(f"Missing expected facts: {answer_missing}")

        if not source_texts:
            failures.append("No source evidence was returned.")
            source_missing = item["source_facts"]
        else:
            source_missing = source_missing_fact_groups(
                source_texts,
                item["source_facts"],
            )
            if source_missing:
                failures.append(
                    "Retrieved source text does not contain expected facts: "
                    f"{source_missing}"
                )

        passed = not failures

        if passed:
            print("PASS: Expected facts and source evidence checked.")
        else:
            print("FAIL")
            for failure in failures:
                print(f"  - {failure}")

        results.append(
            {
                "question": question,
                "passed": passed,
                "answer": answer,
                "source_count": len(sources),
                "answer_missing_facts": answer_missing,
                "source_missing_facts": source_missing,
                "failures": failures,
            }
        )

    passed_count = sum(item["passed"] for item in results)
    total = len(results)

    print("\n" + "-" * 80)
    print(f"Answer quality checks: {passed_count}/{total}")
    print(f"Pass rate: {passed_count / total * 100:.2f}%")
    print("-" * 80)

    return results


def evaluate_unsupported_questions() -> list[dict]:
    print("\n" + "=" * 80)
    print("UNSUPPORTED-QUESTION EVALUATION")
    print("=" * 80)

    results = []

    for index, question in enumerate(UNSUPPORTED_QUESTIONS, start=1):
        print(f"\nU{index}: {question}")

        result = answer_question(question)
        answer = str(result.get("answer", ""))
        sources = result.get("sources", [])

        print(f"Answer: {answer}")
        print(f"Sources returned: {len(sources)}")

        failures = []

        if not is_abstention(answer):
            failures.append("Expected the system to abstain.")

        if sources:
            failures.append(
                "Expected no sources for an unsupported question."
            )

        passed = not failures

        if passed:
            print("PASS: System abstained without returning sources.")
        else:
            print("FAIL: Expected an abstention with no sources.")
            for failure in failures:
                print(f"  - {failure}")

        results.append(
            {
                "question": question,
                "passed": passed,
                "answer": answer,
                "source_count": len(sources),
                "failures": failures,
            }
        )

    passed_count = sum(item["passed"] for item in results)
    total = len(results)

    print("\n" + "-" * 80)
    print(f"Unsupported-question checks: {passed_count}/{total}")
    print("-" * 80)

    return results


def write_report(
    answer_results: list[dict],
    abstention_results: list[dict],
) -> Path:
    """Save evaluation metrics and individual results as JSON."""
    all_results = answer_results + abstention_results

    answer_passed = sum(item["passed"] for item in answer_results)
    abstention_passed = sum(item["passed"] for item in abstention_results)
    total_passed = sum(item["passed"] for item in all_results)

    answer_total = len(answer_results)
    abstention_total = len(abstention_results)
    total_checks = len(all_results)

    report = {
        "evaluation_type": "answer_quality_and_abstention",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "answer_quality": {
                "passed": answer_passed,
                "total": answer_total,
                "pass_rate": (
                    answer_passed / answer_total if answer_total else 0
                ),
            },
            "abstention": {
                "passed": abstention_passed,
                "total": abstention_total,
                "pass_rate": (
                    abstention_passed / abstention_total
                    if abstention_total
                    else 0
                ),
            },
            "overall": {
                "passed": total_passed,
                "total": total_checks,
                "pass_rate": (
                    total_passed / total_checks if total_checks else 0
                ),
            },
        },
        "results": {
            "answerable_questions": answer_results,
            "unsupported_questions": abstention_results,
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORTS_DIR / f"answer_evaluation_{timestamp}.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nJSON report saved to: {report_path}")
    return report_path


def main() -> None:
    answer_results = evaluate_answerable_questions()
    abstention_results = evaluate_unsupported_questions()

    answer_passed = sum(item["passed"] for item in answer_results)
    abstention_passed = sum(item["passed"] for item in abstention_results)
    total_passed = answer_passed + abstention_passed
    total_checks = len(answer_results) + len(abstention_results)

    print("\n" + "=" * 80)
    print("FINAL ANSWER EVALUATION")
    print("=" * 80)
    print(f"Answer quality: {answer_passed}/{len(answer_results)}")
    print(f"Abstention:     {abstention_passed}/{len(abstention_results)}")
    print(f"Overall:        {total_passed}/{total_checks}")
    print("=" * 80)

    report_path = write_report(answer_results, abstention_results)

    if total_passed == total_checks:
        print("All evaluation checks passed.")
    else:
        print("Some checks failed. Review the answer and source evidence.")

    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
