import chromadb

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection_v2 = client.get_or_create_collection(
    name="engineering_documents_v2"
)


def store_chunks(
    chunks,
    embeddings,
    document_id,
    source_file,
    pipeline_version,
    batch_size=100
):

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match")

    total = len(chunks)

    # Generate unique IDs for this document
    ids = [
        f"{document_id}_chunk_{i:06d}"
        for i in range(total)
    ]

    for start in range(0, total, batch_size):

        end = start + batch_size

        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]
        batch_ids = ids[start:end]

        documents = [
            chunk["text"]
            for chunk in batch_chunks
        ]

        metadatas = []

        for chunk in batch_chunks:

            metadata = {
                key: value
                for key, value in chunk["metadata"].items()
                if value is not None
                and isinstance(value, (str, int, float, bool))
            }

            metadata["document_id"] = document_id
            metadata["source_file"] = source_file
            metadata["pipeline_version"] = pipeline_version

            metadatas.append(metadata)

        collection_v2.upsert(
            ids=batch_ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas
        )

    # Remove outdated extra chunks from an earlier
    # ingestion of this same document.
    existing = collection_v2.get(
        where={"document_id": document_id},
        include=[]
    )

    stale_ids = [
        existing_id
        for existing_id in existing["ids"]
        if existing_id not in set(ids)
    ]

    if stale_ids:
        collection_v2.delete(ids=stale_ids)

    print("Stored chunks:", total)
    print("Collection total records:", collection_v2.count())
