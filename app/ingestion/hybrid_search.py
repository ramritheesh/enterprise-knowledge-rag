
import re


RRF_K = 60


def _tokens(text: str) -> set[str]:
    """Extract lowercase words for simple keyword matching."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def keyword_search(
    question: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Rank document chunks by keyword overlap with the question."""
    question_tokens = _tokens(question)

    if not question_tokens or top_k < 1:
        return []

    scored = []

    for document in documents:
        text = str(document.get("text", ""))
        document_tokens = _tokens(text)
        overlap = len(question_tokens & document_tokens)

        if overlap == 0:
            continue

        scored.append({
            **document,
            "keyword_score": overlap / len(question_tokens),
        })

    scored.sort(
        key=lambda item: item["keyword_score"],
        reverse=True,
    )

    return scored[:top_k]


def reciprocal_rank_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Combine semantic and keyword rankings using RRF."""
    if top_k < 1:
        return []

    combined = {}

    def result_key(item: dict):
        if item.get("id") is not None:
            return item["id"]

        return (
            (item.get("metadata") or {}).get("source", ""),
            item.get("text", ""),
        )

    for rank, item in enumerate(semantic_results, start=1):
        key = result_key(item)

        combined[key] = {
            **item,
            "rrf_score": 1.0 / (RRF_K + rank),
            "semantic_rank": rank,
            "keyword_rank": None,
        }

    for rank, item in enumerate(keyword_results, start=1):
        key = result_key(item)

        if key in combined:
            combined[key]["rrf_score"] += 1.0 / (RRF_K + rank)
            combined[key]["keyword_rank"] = rank
        else:
            combined[key] = {
                **item,
                "rrf_score": 1.0 / (RRF_K + rank),
                "semantic_rank": None,
                "keyword_rank": rank,
            }

    return sorted(
        combined.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:top_k]
