
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Use temporary document storage and an isolated mock collection."""
    monkeypatch.setattr(main, "DOCUMENTS_DIR", tmp_path)

    fake_collection = SimpleNamespace(
        get=Mock(return_value={"ids": ["test_document_0"]})
    )
    monkeypatch.setattr(main, "collection", fake_collection)

    return TestClient(main.app)


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_rejects_empty_question(client):
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_rejects_whitespace_question(client):
    response = client.post(
        "/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k(client):
    response = client.post(
        "/ask",
        json={"question": "Annual leave policy?", "top_k": 10},
    )

    assert response.status_code == 422


def test_upload_rejects_unsupported_file(client):
    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.exe",
                b"not allowed",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "Only .txt, .pdf, and .docx" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The file is empty."


def test_upload_rejects_file_over_2_mb(client):
    content = b"a" * (main.MAX_FILE_SIZE + 1)

    response = client.post(
        "/upload",
        files={"file": ("large.txt", content, "text/plain")},
    )

    assert response.status_code == 413
    assert not (main.DOCUMENTS_DIR / "large.txt").exists()


def test_upload_rejects_invalid_utf8(client):
    response = client.post(
        "/upload",
        files={"file": ("invalid.txt", b"\xff\xfe", "text/plain")},
    )

    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_upload_and_index_successfully(client, monkeypatch):
    monkeypatch.setattr(main, "index_documents", Mock())

    response = client.post(
        "/upload",
        files={
            "file": (
                "reliability_test.txt",
                b"Project Bluebird holds its review every Wednesday.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "reliability_test.txt"
    assert (
        main.DOCUMENTS_DIR / "reliability_test.txt"
    ).is_file()
    main.index_documents.assert_called_once()


def test_upload_rejects_duplicate_filename(client, monkeypatch):
    destination = main.DOCUMENTS_DIR / "duplicate.txt"
    destination.write_text("Original document", encoding="utf-8")

    index_mock = Mock()
    monkeypatch.setattr(main, "index_documents", index_mock)

    response = client.post(
        "/upload",
        files={
            "file": (
                "duplicate.txt",
                b"Replacement document",
                "text/plain",
            )
        },
    )

    assert response.status_code == 409
    assert destination.read_text(encoding="utf-8") == "Original document"
    index_mock.assert_not_called()


def test_upload_cleans_up_after_indexing_failure(client, monkeypatch):
    index_mock = Mock(
        side_effect=RuntimeError("Simulated indexing failure")
    )
    delete_mock = Mock(return_value=0)

    monkeypatch.setattr(main, "index_documents", index_mock)
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)

    response = client.post(
        "/upload",
        files={
            "file": (
                "failed_upload.txt",
                b"This upload should fail during indexing.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert not (main.DOCUMENTS_DIR / "failed_upload.txt").exists()
    delete_mock.assert_called_once_with("failed_upload.txt")


def test_upload_retains_file_when_chunk_cleanup_fails(
    client, monkeypatch
):
    """Retain the source document when vector cleanup fails."""
    monkeypatch.setattr(
        main,
        "index_documents",
        Mock(side_effect=RuntimeError("Simulated indexing failure")),
    )
    monkeypatch.setattr(
        main,
        "delete_document_chunks",
        Mock(side_effect=RuntimeError("Simulated cleanup failure")),
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "cleanup_failure.txt",
                b"Test content for cleanup failure.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert "retained for recovery" in response.json()["detail"]
    assert (main.DOCUMENTS_DIR / "cleanup_failure.txt").exists()


def test_list_documents(client):
    (main.DOCUMENTS_DIR / "visible.txt").write_text(
        "A test document",
        encoding="utf-8",
    )
    (main.DOCUMENTS_DIR / "ignored.exe").write_text(
        "Unsupported file",
        encoding="utf-8",
    )

    response = client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] == 1
    assert data["documents"][0]["filename"] == "visible.txt"


def test_delete_document_successfully(client, monkeypatch):
    destination = main.DOCUMENTS_DIR / "remove_me.txt"
    destination.write_text("Temporary test content", encoding="utf-8")

    delete_mock = Mock(return_value=2)
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)

    response = client.delete("/documents/remove_me.txt")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 2
    assert not destination.exists()
    delete_mock.assert_called_once_with("remove_me.txt")


def test_delete_missing_document_returns_404(client):
    response = client.delete("/documents/missing.txt")

    assert response.status_code == 404


def test_delete_rejects_unsupported_extension(client):
    response = client.delete("/documents/malware.exe")

    assert response.status_code == 400


def test_delete_restores_file_after_deletion_failure(
    client, monkeypatch
):
    destination = main.DOCUMENTS_DIR / "restore_me.txt"
    original_content = b"Keep this document safe."
    destination.write_bytes(original_content)

    delete_mock = Mock(
        side_effect=RuntimeError("Simulated delete failure")
    )
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)
    monkeypatch.setattr(main, "index_documents", Mock())

    response = client.delete("/documents/restore_me.txt")

    assert response.status_code == 500
    assert destination.exists()
    assert destination.read_bytes() == original_content


def test_upload_reports_failure_when_indexing_fails(
    client, monkeypatch
):
    """A failed indexing operation should not report upload success."""
    monkeypatch.setattr(
        main,
        "index_documents",
        Mock(side_effect=RuntimeError("Simulated indexing failure")),
    )
    monkeypatch.setattr(
        main,
        "delete_document_chunks",
        Mock(return_value=0),
    )

    response = client.post(
        "/upload",
        files={
            "file": (
                "recovery_test.txt",
                b"Test content for recovery.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 500
    assert not (main.DOCUMENTS_DIR / "recovery_test.txt").exists()


def test_delete_reports_failure_when_reindexing_fails(
    client, monkeypatch
):
    """The original file should be restored if deletion fails."""
    destination = (
        main.DOCUMENTS_DIR / "restore_after_reindex.txt"
    )
    original_content = b"Original content must be preserved."
    destination.write_bytes(original_content)

    monkeypatch.setattr(
        main,
        "delete_document_chunks",
        Mock(side_effect=RuntimeError("Simulated deletion failure")),
    )
    monkeypatch.setattr(
        main,
        "index_documents",
        Mock(side_effect=RuntimeError("Simulated reindexing failure")),
    )

    response = client.delete(
        "/documents/restore_after_reindex.txt"
    )

    assert response.status_code == 500
    assert destination.exists()
    assert destination.read_bytes() == original_content
