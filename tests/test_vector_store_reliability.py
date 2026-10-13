
from unittest.mock import patch

import pytest

from app.ingestion import vector_store


def test_shorter_document_removes_stale_chunks():
    """Re-indexing a shorter document should remove obsolete chunk IDs."""
    source = "reliability_test.txt"

    with patch.object(
        vector_store,
        "collection",
    ) as mock_collection:
        # Existing collection has three chunks for this document.
        mock_collection.get.side_effect = [
            {"ids": [f"{source}_0", f"{source}_1", f"{source}_2"]},
            {"ids": [f"{source}_0", f"{source}_1", f"{source}_2"]},
        ]

        # Make the document short enough to produce one chunk.
        with patch.object(
            vector_store,
            "_make_chunks",
            return_value=["short replacement text"],
        ):
            vector_store._index_one_document(
                source,
                "short replacement text",
            )

        mock_collection.upsert.assert_called_once_with(
            ids=[f"{source}_0"],
            documents=["short replacement text"],
            metadatas=[
                {"source": source, "chunk_index": 0}
            ],
        )

        mock_collection.delete.assert_called_once_with(
            ids=[f"{source}_1", f"{source}_2"]
        )


def test_empty_chunks_do_not_overwrite_existing_document():
    """Empty extracted content should fail before writing to ChromaDB."""
    with patch.object(
        vector_store,
        "collection",
    ) as mock_collection:
        with patch.object(
            vector_store,
            "_make_chunks",
            return_value=[],
        ):
            with pytest.raises(ValueError, match="no indexable text"):
                vector_store._index_one_document(
                    "empty_test.txt",
                    " ",
                )

        mock_collection.upsert.assert_not_called()
        mock_collection.delete.assert_not_called()


def test_indexing_failure_does_not_start_missing_source_cleanup():
    """A failed upsert must stop indexing before orphan cleanup."""
    current_document = {
        "source": "current.txt",
        "text": "Current document text",
    }

    with patch.object(
        vector_store,
        "load_text_documents",
        return_value=[current_document],
    ), patch.object(
        vector_store,
        "_index_one_document",
        side_effect=RuntimeError("simulated indexing failure"),
    ), patch.object(
        vector_store,
        "collection",
    ) as mock_collection:
        with pytest.raises(
            RuntimeError,
            match="simulated indexing failure",
        ):
            vector_store.index_documents()

        # The cleanup phase would call collection.get without a filter.
        # It must not run after the document indexing failure.
        mock_collection.get.assert_not_called()
