from app.ingestion.vector_store import (
    index_documents,
    search_documents,
)


def main():
    # Step 1: Store document chunks in Chroma.
    index_documents()

    # Step 2: Ask a question using different wording.
    question = "How much vacation time do employees get?"

    print(f"\nQuestion: {question}")
    print("\nRetrieved evidence:")

    # Step 3: Retrieve the most relevant chunks.
    results = search_documents(question, top_k=3)

    for rank, result in enumerate(results, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Chunk: {result['metadata']['chunk_index']}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"])


if __name__ == "__main__":
    main()