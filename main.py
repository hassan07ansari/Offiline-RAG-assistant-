
import os
import time
import argparse
from datetime import datetime, timezone

from documents import load_document, chunk_documents
from embeddings import generate_embeddings
from vector_store import store_chunks
from file_hash import calculate_file_hash
from ingestion_registry import (
    is_already_processed,
    update_registry
)

from hybrid_search import invalidate_bm25_cache
from Ask import main as run_query_application


PIPELINE_VERSION = "section_chunker_v1"
COLLECTION_NAME = "engineering_documents_v2"
DOCUMENTS_DIR = "data/documents"


def discover_documents():

    supported_extensions = (".pdf", ".docx")
    document_paths = []

    for filename in sorted(os.listdir(DOCUMENTS_DIR)):

        file_path = os.path.join(DOCUMENTS_DIR, filename)

        if not os.path.isfile(file_path):
            continue

        if filename.startswith("~$"):
            continue

        if filename.lower().endswith(supported_extensions):
            document_paths.append(file_path)

    return document_paths


def process_document(file_path):

    file_name = os.path.basename(file_path)
    file_hash = calculate_file_hash(file_path)

    print("\n" + "=" * 60)
    print("Processing:", file_name)
    print("SHA-256:", file_hash)

    if is_already_processed(file_hash, PIPELINE_VERSION):
        print("Already processed. Skipping.")
        return "skipped"

    document_info = {
        "document_name": file_name,
        "file_hash": file_hash,
        "pipeline_version": PIPELINE_VERSION,
        "status": "processing",
        "total_chunks": 0,
        "collection": COLLECTION_NAME,
        "ingested_at": datetime.now(timezone.utc).isoformat()
    }

    update_registry(document_info)

    try:

        documents = load_document(file_path)

        print("Loaded pages/documents:", len(documents))

        chunks = chunk_documents(documents)

        if not chunks:
            raise ValueError("No text chunks were generated.")

        print("Total chunks:", len(chunks))

        texts = [chunk["text"] for chunk in chunks]

        start_time = time.perf_counter()

        embeddings = generate_embeddings(texts, batch_size=8)

        embedding_time = time.perf_counter() - start_time

        print("Embedding time:", round(embedding_time, 2), "seconds")

        store_chunks(
            chunks,
            embeddings,
            document_id=file_hash,
            source_file=file_name,
            pipeline_version=PIPELINE_VERSION
        )

        document_info["status"] = "completed"
        document_info["total_chunks"] = len(chunks)
        document_info["ingested_at"] = datetime.now(timezone.utc).isoformat()

        update_registry(document_info)

        print("Ingestion completed successfully.")

        return "processed"

    except Exception as error:

        document_info["status"] = "failed"
        document_info["error"] = str(error)
        document_info["ingested_at"] = datetime.now(timezone.utc).isoformat()

        update_registry(document_info)

        print("Ingestion failed:", error)

        return "failed"


def run_ingestion(dry_run=False):

    if not os.path.isdir(DOCUMENTS_DIR):
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    document_paths = discover_documents()

    print("Discovered documents:", len(document_paths))

    if not document_paths:
        print("No supported PDF or DOCX files found.")
        return

    if dry_run:

        print("\nDRY RUN — No documents will be ingested.")

        for file_path in document_paths:

            file_name = os.path.basename(file_path)
            file_hash = calculate_file_hash(file_path)

            already_processed = is_already_processed(
                file_hash,
                PIPELINE_VERSION
            )

            status = (
                "SKIP — Already processed"
                if already_processed
                else "NEW — Ready for ingestion"
            )

            print("\nFile:", file_name)
            print("SHA-256:", file_hash)
            print("Decision:", status)

        print("\nDry run completed. No embeddings or database changes made.")
        return

    processed = 0
    skipped = 0
    failed = 0

    for file_path in document_paths:

        result = process_document(file_path)

        if result == "processed":
            processed += 1

        elif result == "skipped":
            skipped += 1

        else:
            failed += 1

    # Invalidate the saved BM25 index only if documents were ingested.
    if processed > 0:
        invalidate_bm25_cache()

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("Discovered:", len(document_paths))
    print("Processed:", processed)
    print("Skipped:", skipped)
    print("Failed:", failed)


def main():

    parser = argparse.ArgumentParser(
        description="Offline Engineering Document RAG Assistant"
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--ingest",
        action="store_true",
        help="Run the document ingestion pipeline"
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview document ingestion without making changes"
    )

    args = parser.parse_args()

    if args.ingest:
        print("\nStarting document ingestion...")
        run_ingestion()

    elif args.dry_run:
        print("\nStarting ingestion preview...")
        run_ingestion(dry_run=True)

    else:
        print("\nStarting query application...")
        run_query_application()


if __name__ == "__main__":
    main()
