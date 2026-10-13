
from app.ingestion.hybrid_search import reciprocal_rank_fusion


def test_rrf_promotes_results_ranked_by_both_methods():
    semantic = [
        {"id": "intro", "text": "General policy introduction"},
        {"id": "notice", "text": "Five working days notice"},
        {"id": "approval", "text": "Direct manager approves leave"},
    ]

    keyword = [
        {"id": "approval", "text": "Direct manager approves leave"},
        {"id": "notice", "text": "Five working days notice"},
    ]

    results = reciprocal_rank_fusion(
        semantic,
        keyword,
        top_k=3,
    )

    assert results[0]["id"] == "approval"
    assert results[0]["semantic_rank"] == 3
    assert results[0]["keyword_rank"] == 1


def test_rrf_keeps_semantic_only_results():
    semantic = [
        {"id": "semantic", "text": "Meaning-related evidence"},
    ]

    results = reciprocal_rank_fusion(
        semantic,
        [],
        top_k=5,
    )

    assert len(results) == 1
    assert results[0]["id"] == "semantic"
    assert results[0]["keyword_rank"] is None
