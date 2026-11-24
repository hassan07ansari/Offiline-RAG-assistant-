import re
import os 
import pickle

from rank_bm25 import BM25Okapi

from vector_store import collection_v2
from embeddings import generate_embedding


def tokenize(text):

    return re.findall(r"[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*", text.lower())


def build_bm25_index():

    records = collection_v2.get(
        include=["documents", "metadatas"]
    )

    document_ids = records["ids"]
    documents = records["documents"]
    metadatas = records["metadatas"]

    tokenized_documents = [
        tokenize(document) for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    bm25_records = []

    for document_id, document, metadata in zip(
        document_ids, documents, metadatas
    ):
        bm25_records.append({
            "id": document_id,
            "text": document,
            "metadata": metadata
        })

    print("BM25 index built.")
    print("Indexed chunks:", len(bm25_records))

    return bm25, bm25_records


CACHE_PATH = "cache/bm25_index.pkl"
CACHE_VERSION = 1


def load_or_build_bm25_index():

    current_count = collection_v2.count()

    # Check whether a saved BM25 index exists.
    if os.path.exists(CACHE_PATH):

        try:
            print("Checking saved BM25 index...")

            with open(CACHE_PATH, "rb") as file:
                cache = pickle.load(file)

            # Validate the saved index.
            if (
                cache["cache_version"] == CACHE_VERSION
                and cache["collection_count"] == current_count
            ):

                print("Valid BM25 cache found.")
                print("Loading BM25 index from disk...")

                return cache["bm25"], cache["bm25_records"]

            print("BM25 cache is outdated. Rebuilding...")

        except (OSError, EOFError, pickle.UnpicklingError,
                AttributeError, ImportError, ValueError, KeyError):

            print("BM25 cache is invalid. Rebuilding...")

    # Build a new index if no valid cache exists.
    print("Building BM25 index...")

    bm25, bm25_records = build_bm25_index()

    # Save the index and records together.
    cache = {
        "cache_version": CACHE_VERSION,
        "collection_count": current_count,
        "bm25": bm25,
        "bm25_records": bm25_records
    }

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    temporary_path = CACHE_PATH + ".tmp"

    with open(temporary_path, "wb") as file:
        pickle.dump(cache, file)

    os.replace(temporary_path, CACHE_PATH)

    print("BM25 index saved to disk.")

    return bm25, bm25_records


def invalidate_bm25_cache():

    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print("Old BM25 cache invalidated.")



def hybrid_search(query, bm25, bm25_records, top_k=5, candidate_k=10):

    query_tokens = tokenize(query)

    # 1. BM25 keyword retrieval

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True
    )

    # 2. Semantic retrieval using ChromaDB

    query_embedding = generate_embedding(query)

    semantic_results = collection_v2.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Reciprocal Rank Fusion

    fusion_scores = {}
    result_records = {}

    rrf_constant = 60

    # Add BM25 rankings

    bm25_rank = 0

    for index in bm25_ranked_indices:

        if bm25_scores[index] <= 0:
            continue

        record = bm25_records[index]
        document_id = record["id"]

        bm25_rank += 1

        fusion_scores[document_id] = (
            fusion_scores.get(document_id, 0)
            + 1 / (rrf_constant + bm25_rank)
        )

        result_records[document_id] = record

        if bm25_rank >= candidate_k:
            break

    # Add semantic rankings

    semantic_ids = semantic_results["ids"][0]
    semantic_documents = semantic_results["documents"][0]
    semantic_metadatas = semantic_results["metadatas"][0]
    semantic_distances = semantic_results["distances"][0]

    for rank, (document_id, document, metadata, distance) in enumerate(
        zip(
            semantic_ids,
            semantic_documents,
            semantic_metadatas,
            semantic_distances
        ),
        start=1
    ):

        fusion_scores[document_id] = (
            fusion_scores.get(document_id, 0)
            + 1 / (rrf_constant + rank)
        )

        result_records[document_id] = {
            "id": document_id,
            "text": document,
            "metadata": metadata,
            "semantic_distance": distance
        }

    # 4. Return the highest-ranked combined results

    ranked_ids = sorted(
        fusion_scores,
        key=fusion_scores.get,
        reverse=True
    )

    final_results = []

    for document_id in ranked_ids[:top_k]:

        record = result_records[document_id]

        final_results.append({
            "id": document_id,
            "text": record["text"],
            "metadata": record["metadata"],
            "rrf_score": fusion_scores[document_id],
            "semantic_distance": record.get("semantic_distance"),
            "bm25_score": (
                bm25_scores[
                    next(
                        index for index, item in enumerate(bm25_records)
                        if item["id"] == document_id
                    )
                ]
                if document_id in {
                    bm25_records[index]["id"]
                    for index in bm25_ranked_indices[:candidate_k]
                }
                else 0
            )
        })

    return final_results
