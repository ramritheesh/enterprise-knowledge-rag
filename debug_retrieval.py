from app.ingestion.vector_store import search_documents

questions = [
    "Who approves annual leave requests?",
    "Can unused annual leave carry over to the next year?",
    "How long is parental leave?",
    "Does the policy mention remote work?",
    "How many vacation days do employees get?",
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    matches = search_documents(question, top_k=8)

    if not matches:
        print("NO MATCHES FOUND")
        continue

    for rank, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        print(f"\nRank: {rank}")
        print(f"Distance: {match.get('distance')}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Text: {match.get('text', '')}")
