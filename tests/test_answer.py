
import pytest

from app.generation import answer


def test_empty_question_raises_error():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer.answer_question("")


def test_no_results_returns_no_answer(monkeypatch):
    monkeypatch.setattr(
        answer,
        "search_documents",
        lambda question, top_k: [],
    )

    result = answer.answer_question("What is the leave policy?")

    assert "couldn't find sufficiently relevant evidence" in (
        result["answer"].lower()
    )
    assert result["sources"] == []


def test_irrelevant_results_do_not_call_llm(monkeypatch):
    evidence = {
        "id": "irrelevant_1",
        "text": "The company cafeteria serves lunch.",
        "metadata": {"source": "general.txt"},
        "distance": 0.80,
    }

    monkeypatch.setattr(
        answer,
        "search_documents",
        lambda question, top_k: [evidence],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Ollama should not be called.")

    monkeypatch.setattr(answer.requests, "post", fail_if_called)

    result = answer.answer_question("What is the leave policy?")

    assert result["sources"] == []
    assert "couldn't find sufficiently relevant evidence" in (
        result["answer"].lower()
    )


def test_relevant_evidence_generates_answer(monkeypatch):
    evidence = {
        "id": "leave_1",
        "text": "Employees receive 20 days of paid annual leave.",
        "metadata": {"source": "leave_policy.txt"},
        "distance": 0.20,
    }

    monkeypatch.setattr(
        answer,
        "search_documents",
        lambda question, top_k: [evidence],
    )

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {
                    "content": "Employees receive 20 days of paid annual leave."
                }
            }

    monkeypatch.setattr(
        answer.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = answer.answer_question(
        "How many days of annual leave do employees receive?"
    )

    assert "20 days" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "leave_policy.txt"


def test_undefined_parental_leave_returns_document_limitation(monkeypatch):
    evidence = {
        "id": "policy_11",
        "text": (
            "This sample policy does not define parental-leave "
            "eligibility, duration, pay, or application procedures. "
            "Employees should contact HR for guidance."
        ),
        "metadata": {
            "source": "Acme_Employee_Leave_Policy.docx"
        },
        "distance": 0.20,
    }

    monkeypatch.setattr(
        answer,
        "search_documents",
        lambda question, top_k: [evidence],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Ollama must not be called for explicitly undefined details."
        )

    monkeypatch.setattr(answer.requests, "post", fail_if_called)

    result = answer.answer_question(
        "What is the company's parental leave policy?"
    )

    assert "does not specify" in result["answer"].lower()
    assert (
        "eligibility, duration, pay, or application procedures"
        in result["answer"].lower()
    )
    assert "12 months" not in result["answer"].lower()
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == (
        "Acme_Employee_Leave_Policy.docx"
    )


def test_undefined_policy_detection_requires_matching_policy_topic():
    evidence = (
        "This sample policy does not define parental-leave eligibility, "
        "duration, pay, or application procedures."
    )

    assert answer._is_explicitly_undefined(
        evidence,
        "What is the company's parental leave policy?",
    )

    assert not answer._is_explicitly_undefined(
        evidence,
        "What is the annual leave entitlement?",
    )
