from app.ingestion.loader import load_text_documents
from app.ingestion.chunker import chunk_text


def main():
    documents = load_text_documents("data/documents")

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        chunks = chunk_text(document["text"])

        print(f"\nSource: {document['source']}")
        print(f"Chunks created: {len(chunks)}")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {index}:")
            print(chunk)


if __name__ == "__main__":
    main()