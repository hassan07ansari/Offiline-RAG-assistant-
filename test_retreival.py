from hybrid_search import build_bm25_index, hybrid_search


bm25, bm25_records = build_bm25_index()

query = "What is the maximum operating temperature of Asterion-28?"

results = hybrid_search(
    query,
    bm25,
    bm25_records,
    top_k=5
)

print("\nSEARCH RESULTS")

for index, result in enumerate(results, start=1):

    metadata = result["metadata"]

    print("\nResult:", index)
    print("Document:", metadata.get("source_file", metadata.get("document_name")))
    print("Section:", metadata.get("section"))
    print("Page:", metadata.get("page_number"))
    print("RRF score:", round(result["rrf_score"], 5))
    print("Content:", result["text"][:500])
