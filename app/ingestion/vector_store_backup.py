
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from app.ingestion.chunker import chunk_text
from app.ingestion.loader import load_text_documents


DB_PATH = Path("data/chroma_db")
COLLECTION_NAME = "company_knowledge"
DOCUMENTS_DIR = Path("data/documents")

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
    """Explicitly recreate the collection. Use only when necessary."""
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
    """Delete only the indexed chunks belonging to one source filename."""
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


def index_documents() -> None:
    """
    Incrementally index documents in data/documents.

    Existing collections are preserved. Chunks are upserted using
    stable IDs, and obsolete chunks for each indexed source are removed.
    """
    documents = load_text_documents(DOCUMENTS_DIR)

    if not documents:
        print("No readable documents found.")
        return

    total_chunks = 0

    for document in documents:
        source = document["source"]
        text = document["text"]

        chunks = [
            chunk.strip()
            for chunk in chunk_text(
                text,
                chunk_size=150,
                overlap=30,
            )
            if chunk.strip()
        ]

        desired_ids = [
            f"{source}_{index}"
            for index in range(len(chunks))
        ]

        # Save current chunks without deleting other documents.
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

        # Remove obsolete chunks if this document became shorter.
        existing = collection.get(
            where={"source": source},
            include=["metadatas"],
        )

        stale_ids = [
            chunk_id
            for chunk_id in existing.get("ids", [])
            if chunk_id not in set(desired_ids)
        ]

        if stale_ids:
            collection.delete(ids=stale_ids)

        total_chunks += len(chunks)
        print(f"Indexed {len(chunks)} chunks from {source}")

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks processed: {total_chunks}")
    print(f"Collection count: {collection.count()}")


def search_documents(
    question: str,
    top_k: int = 3,
) -> list[dict]:
    """Return the nearest document chunks for a question."""
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
        matches.append(
            {
                "id": chunk_id,
                "text": documents[index],
                "metadata": metadatas[index] or {},
                "distance": distances[index],
            }
        )

    matches.sort(
        key=lambda item: (
            item["distance"]
            if item["distance"] is not None
            else float("inf")
        )
    )

    return matches
