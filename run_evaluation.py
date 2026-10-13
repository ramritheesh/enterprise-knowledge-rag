
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluate_retrieval import (
    evaluate_retrieval,
    evaluate_abstention,
)


REPORTS_DIR = Path("reports")


def run_answer_quality_evaluation():
    """Run the existing answer evaluator and read its JSON report."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    existing_reports = set(
        REPORTS_DIR.glob("answer_evaluation_*.json")
    )

    result = subprocess.run(
        [sys.executable, "evaluate_answers.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    print("\n" + "=" * 80)
    print("ANSWER QUALITY EVALUATION OUTPUT")
    print("=" * 80)

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print("Errors or warnings:")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "evaluate_answers.py failed with exit code "
            f"{result.returncode}"
        )

    generated_reports = (
        set(REPORTS_DIR.glob("answer_evaluation_*.json"))
        - existing_reports
    )

    if generated_reports:
        answer_report_path = max(
            generated_reports,
            key=lambda path: path.stat().st_mtime,
        )
    else:
        # Fall back to the latest existing answer report if the
        # evaluator did not create a new file.
        all_reports = list(
            REPORTS_DIR.glob("answer_evaluation_*.json")
        )

        if not all_reports:
            raise FileNotFoundError(
                "No answer evaluation JSON report was found in reports/."
            )

        answer_report_path = max(
            all_reports,
            key=lambda path: path.stat().st_mtime,
        )

    with answer_report_path.open("r", encoding="utf-8") as file:
        answer_report = json.load(file)

    print(f"Loaded answer report: {answer_report_path}")

    return answer_report


def summarize_answer_report(answer_report):
    """Extract the answer evaluator's summary without changing its report."""

    summary = answer_report.get("summary", {})

    answer_quality = summary.get("answer_quality", {})
    abstention = summary.get("abstention", {})
    overall = summary.get("overall", {})

    return {
        "answer_quality": answer_quality,
        "abstention": abstention,
        "overall": overall,
    }


def build_combined_report(retrieval, retrieval_abstention, answer_report):
    """Build one combined report from all evaluation results."""

    answer_summary = summarize_answer_report(answer_report)

    answer_quality = answer_summary["answer_quality"]
    answer_abstention = answer_summary["abstention"]

    # The overall metric counts distinct checks:
    # retrieval questions + answer-quality checks + unsupported checks.
    # The same unsupported checks are tested in two ways, so they are
    # counted once in the overall score, not twice.
    retrieval_passed = retrieval["passed"]
    retrieval_total = retrieval["total"]

    answer_passed = answer_quality.get("passed", 0)
    answer_total = answer_quality.get("total", 0)

    unsupported_passed = answer_abstention.get("passed", 0)
    unsupported_total = answer_abstention.get("total", 0)

    combined_passed = (
        retrieval_passed + answer_passed + unsupported_passed
    )
    combined_total = (
        retrieval_total + answer_total + unsupported_total
    )

    timestamp = datetime.now(timezone.utc)

    return {
        "evaluation_type": "combined_rag_evaluation",
        "timestamp_utc": timestamp.isoformat(),
        "summary": {
            "retrieval": {
                "passed": retrieval_passed,
                "total": retrieval_total,
                "hit_rate": retrieval["hit_rate"],
                "mrr": retrieval["mrr"],
                "top_k": retrieval["top_k"],
            },
            "answer_quality": answer_quality,
            "answer_abstention": answer_abstention,
            "retrieval_abstention": {
                "passed": retrieval_abstention["passed"],
                "total": retrieval_abstention["total"],
                "pass_rate": retrieval_abstention["pass_rate"],
            },
            "overall": {
                "passed": combined_passed,
                "total": combined_total,
                "pass_rate": (
                    combined_passed / combined_total
                    if combined_total
                    else 0.0
                ),
            },
        },
        "results": {
            "retrieval": retrieval["results"],
            "retrieval_abstention": retrieval_abstention["results"],
            "answer_quality": answer_report.get(
                "results", {}
            ),
        },
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ENTERPRISE KNOWLEDGE INTELLIGENCE PLATFORM")
    print("COMBINED RAG EVALUATION")
    print("=" * 80)

    retrieval = evaluate_retrieval()
    retrieval_abstention = evaluate_abstention()

    answer_report = run_answer_quality_evaluation()

    combined_report = build_combined_report(
        retrieval,
        retrieval_abstention,
        answer_report,
    )

    timestamp_text = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    report_path = REPORTS_DIR / (
        f"combined_evaluation_{timestamp_text}.json"
    )

    report_path.write_text(
        json.dumps(combined_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = combined_report["summary"]

    print("\n" + "=" * 80)
    print("COMBINED EVALUATION SUMMARY")
    print("=" * 80)

    retrieval_summary = summary["retrieval"]
    answer_summary = summary["answer_quality"]
    abstention_summary = summary["answer_abstention"]
    overall = summary["overall"]

    print(
        f"Retrieval Hit Rate@{retrieval_summary['top_k']}: "
        f"{retrieval_summary['passed']}/"
        f"{retrieval_summary['total']} "
        f"({retrieval_summary['hit_rate']:.2%})"
    )
    print(f"Retrieval MRR@K: {retrieval_summary['mrr']:.4f}")

    print(
        "Answer quality: "
        f"{answer_summary.get('passed', 0)}/"
        f"{answer_summary.get('total', 0)}"
    )

    print(
        "Answer abstention: "
        f"{abstention_summary.get('passed', 0)}/"
        f"{abstention_summary.get('total', 0)}"
    )

    print(
        "Retrieval abstention: "
        f"{retrieval_abstention['passed']}/"
        f"{retrieval_abstention['total']}"
    )

    print(
        f"Overall: {overall['passed']}/{overall['total']} "
        f"({overall['pass_rate']:.2%})"
    )

    print(f"\nCombined report saved to: {report_path}")


if __name__ == "__main__":
    main()
