
from app.ingestion.vector_store import search_documents


questions = [
    "What topics are covered in the employee handbook?",
    "What is the company's code of conduct?",
    "How many days of annual leave do employees get?",
    "Who approves or rejects leave requests?",
]


def main():
    for question in questions:
        print("\n" + "=" * 70)
        print(f"QUESTION: {question}")
        print("=" * 70)

        try:
            results = search_documents(question, top_k=3)

            if not results:
                print("No results found.")
                continue

            for rank, result in enumerate(results, start=1):
                metadata = result.get("metadata") or {}

                print(f"\nRank: {rank}")
                print(f"Source: {metadata.get('source', 'Unknown')}")
                print(f"Chunk ID: {result.get('id', 'Unknown')}")
                print(f"Distance: {result.get('distance', -1):.4f}")
                print("Evidence:")
                print(result.get("text", "")[:1000])

        except Exception as error:
            print(f"Error: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
