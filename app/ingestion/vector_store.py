
from pathlib import Path
import logging

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from app.ingestion.chunker import chunk_text
from app.ingestion.loader import load_text_documents
from app.ingestion.hybrid_search import (
    keyword_search,
    reciprocal_rank_fusion,
)


logger = logging.getLogger(__name__)

DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "company_knowledge"
DOCUMENTS_DIR = Path("data/documents")

CHUNK_SIZE = 150
CHUNK_OVERLAP = 30

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)


def reset_collection():
    """Explicitly recreate the collection. This deletes all indexed data."""
    global collection

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception as error:
        if "does not exist" not in str(error).lower():
            raise

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def delete_document_chunks(source: str) -> int:
    """Delete only indexed chunks belonging to one source filename."""
    if not source or not source.strip():
        raise ValueError("Source filename cannot be empty.")

    existing = collection.get(
        where={"source": source},
        include=["metadatas"],
    )

    ids = existing.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return len(ids)


def _make_chunks(text: str) -> list[str]:
    """Split document text into non-empty, normalized chunks."""
    return [
        chunk.strip()
        for chunk in chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )
        if chunk.strip()
    ]


def _index_one_document(source: str, text: str) -> int:
    """Upsert current chunks, then remove obsolete chunks."""
    chunks = _make_chunks(text)

    if not chunks:
        raise ValueError(
            f"Document {source!r} contains no indexable text."
        )

    desired_ids = [
        f"{source}_{index}"
        for index in range(len(chunks))
    ]

    for index, chunk in enumerate(chunks):
        collection.upsert(
            ids=[desired_ids[index]],
            documents=[chunk],
            metadatas=[
                {
                    "source": source,
                    "chunk_index": index,
                }
            ],
        )

    desired_id_set = set(desired_ids)

    existing = collection.get(
        where={"source": source},
        include=["metadatas"],
    )

    stale_ids = [
        chunk_id
        for chunk_id in existing.get("ids", [])
        if chunk_id not in desired_id_set
    ]

    if stale_ids:
        collection.delete(ids=stale_ids)

    logger.info("Indexed %d chunks from %s", len(chunks), source)

    return len(chunks)


def index_documents() -> None:
    """Index readable documents and remove obsolete source chunks."""
    documents = load_text_documents(DOCUMENTS_DIR)

    if not documents:
        logger.warning(
            "No readable documents found. Existing index was preserved."
        )
        return

    current_sources = {
        document["source"]
        for document in documents
    }

    total_chunks = 0

    for document in documents:
        total_chunks += _index_one_document(
            document["source"],
            document["text"],
        )

    existing = collection.get(include=["metadatas"])

    existing_sources = {
        (metadata or {}).get("source")
        for metadata in existing.get("metadatas", [])
        if (metadata or {}).get("source")
    }

    missing_sources = existing_sources - current_sources

    for source in missing_sources:
        removed = delete_document_chunks(source)

        logger.info(
            "Removed %d stale chunks for missing source %s",
            removed,
            source,
        )

    logger.info("Documents loaded: %d", len(documents))
    logger.info("Chunks processed: %d", total_chunks)
    logger.info("Collection count: %d", collection.count())


def search_documents(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """Return nearest document chunks using semantic search."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_texts=[question.strip()],
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []

    for index, chunk_id in enumerate(ids):
        matches.append({
            "id": chunk_id,
            "text": documents[index],
            "metadata": metadatas[index] or {},
            "distance": distances[index],
        })

    matches.sort(
        key=lambda item: (
            item["distance"]
            if item["distance"] is not None
            else float("inf")
        )
    )

    return matches


def hybrid_search_documents(
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Combine semantic and keyword retrieval with Reciprocal Rank Fusion.

    Semantic matches retain their original cosine distance.
    Keyword-only matches have distance=None because no cosine distance
    was calculated for them.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    candidate_count = min(total_chunks, max(top_k * 3, 10))

    semantic_results = search_documents(
        question.strip(),
        top_k=candidate_count,
    )

    stored = collection.get(
        include=["documents", "metadatas"],
    )

    ids = stored.get("ids", [])
    texts = stored.get("documents", [])
    metadatas = stored.get("metadatas", [])

    indexed_documents = []

    for index, chunk_id in enumerate(ids):
        indexed_documents.append({
            "id": chunk_id,
            "text": texts[index] or "",
            "metadata": metadatas[index] or {},
        })

    keyword_results = keyword_search(
        question.strip(),
        indexed_documents,
        top_k=candidate_count,
    )

    fused_results = reciprocal_rank_fusion(
        semantic_results,
        keyword_results,
        top_k=top_k,
    )

    semantic_by_id = {
        item["id"]: item
        for item in semantic_results
        if item.get("id") is not None
    }

    for item in fused_results:
        original = semantic_by_id.get(item.get("id"))

        if original is not None:
            item["distance"] = original.get("distance")
        else:
            item["distance"] = None

    return fused_results
