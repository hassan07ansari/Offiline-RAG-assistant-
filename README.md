# Offline Engineering RAG Assistant

An offline Retrieval-Augmented Generation (RAG) application for searching engineering documents, answering technical questions with source references, and extracting structured engineering requirements using locally hosted LLMs.

## Overview

The Offline Engineering RAG Assistant helps users retrieve technical information from engineering documents without depending on cloud-based LLM APIs.

It combines semantic search and keyword-based retrieval to find relevant document sections, then uses a locally running language model to generate grounded answers.

The project also includes a separate requirements extraction workflow that identifies SHALL statements from engineering specifications and classifies them into requirement categories.

## Key Features

- **Multi-format document ingestion:** Load PDF and DOCX engineering documents.
- **Document tracking:** SHA256-based file identification and ingestion registry to avoid unnecessary reprocessing.
- **Section-aware chunking:** Split documents while preserving section information and tables.
- **Local embeddings:** Generate embeddings using Ollama and `nomic-embed-text`.
- **Vector search:** Store and retrieve document chunks using ChromaDB.
- **Hybrid retrieval:** Combine BM25 keyword search and semantic vector search.
- **Reciprocal Rank Fusion (RRF):** Combine rankings from multiple retrieval methods.
- **Grounded question answering:** Generate answers using locally hosted Qwen models and retrieved document context.
- **Source references:** Include document names and page information in answers where available.
- **Requirements extraction:** Identify SHALL statements from engineering specifications.
- **Structured classification:** Use Ollama and Pydantic to classify extracted requirements.
- **CSV and Excel export:** Export structured requirement information.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Local LLM runtime | Ollama |
| Generation model | Qwen3 4B Instruct |
| Embedding model | Nomic Embed Text |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Hybrid ranking | Reciprocal Rank Fusion |
| PDF processing | PyMuPDF |
| DOCX processing | python-docx |
| Data processing | Pandas |
| Structured validation | Pydantic |
| Excel export | OpenPyXL |

## Project Structure

```text
Offline_Assistant/
│
├── data/
│   ├── documents/                 # Engineering source documents
│   ├── README_SYNTHETIC_DATASET.txt
│   └── ingestion_registry.json
│
├── Ask.py                         # RAG question-answering interface
├── main.py                        # Document ingestion pipeline
├── documents.py                   # PDF and DOCX loading
├── embeddings.py                  # Ollama embedding generation
├── vector_store.py                # ChromaDB operations
├── hybrid_search.py               # BM25 + semantic retrieval + RRF
├── file_hash.py                   # SHA256 file hashing
├── ingestion_registry.py           # Ingestion tracking
├── requirements_extractor.py      # SHALL extraction and classification
├── evaluation_questions.json      # Evaluation question set
├── test_retreival.py              # Retrieval testing
├── .gitignore
└── README.md
```

## Prerequisites

Before running the application, install:

- Python 3.12
- Ollama
- Git (optional, for cloning the repository)

The project uses locally hosted models, so the required Ollama models must be downloaded before running the application.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hassan07ansari/Offiline-RAG-assistant-.git
cd Offiline-RAG-assistant-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
pip install ollama chromadb pymupdf python-docx rank-bm25 pydantic pandas openpyxl
```

### 4. Install and configure Ollama

Download Ollama from:

https://ollama.com/download

Pull the required models:

```bash
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

Verify the installed models:

```bash
ollama list
```

Ensure the Ollama service is running before executing the application.

## Usage

### 1. Ingest engineering documents

Run:

```bash
python main.py
```

The ingestion pipeline:

1. Discovers supported documents.
2. Calculates SHA256 hashes.
3. Checks the ingestion registry.
4. Loads PDF and DOCX content.
5. Splits documents into chunks.
6. Generates embeddings.
7. Stores chunks and metadata in ChromaDB.
8. Updates the ingestion registry.

Previously processed documents are skipped when their file hash and pipeline version match the registry.

### 2. Ask engineering questions

Run the question-answering script:

```bash
python Ask.py "What is the maximum operating temperature of Asterion-28 Rev B?"
```

Other example questions:

```bash
python Ask.py "What is the input voltage range specified in Rev B?"
```

```bash
python Ask.py "What changes were introduced in Rev B compared with Rev A?"
```

The application retrieves relevant document chunks using hybrid search and passes the retrieved context to the local language model to generate an answer with source references.

### 3. Extract engineering requirements

Run the requirements extraction script with a PDF or DOCX specification:

```bash
python requirements_extractor.py "data/documents/02_Asterion28_Equipment_Spec_RevB.docx"
```

The extraction workflow:

1. Loads the selected engineering specification.
2. Identifies SHALL statements.
3. Extracts requirement identifiers and document metadata.
4. Uses Ollama to classify requirements.
5. Validates structured output using Pydantic.
6. Exports the extracted requirements to CSV and Excel.

Generated files are saved in:

```text
requirements_output/
```

The exported columns include:

- Requirement ID
- Requirement Text
- Requirement Type
- Section
- Page

Note: DOCX page numbers may be unavailable because the document loader does not currently calculate Word page layout.

## Data and Storage

- Source documents are stored in `data/documents/`.
- The ingestion registry is stored in `data/ingestion_registry.json`.
- ChromaDB uses a local persistent database under `data/chroma_db/`.
- Generated requirement reports are stored in `requirements_output/`.

The ChromaDB database is generated locally and is excluded from Git tracking. Run the ingestion pipeline to recreate it.

## Current Development Status

### Implemented

- PDF and DOCX ingestion.
- SHA256-based document tracking.
- Section-aware chunking.
- Local embedding generation.
- Persistent ChromaDB storage.
- BM25 and semantic hybrid retrieval.
- Reciprocal Rank Fusion.
- Local LLM-based question answering.
- Source references in answers.
- SHALL requirement extraction.
- Pydantic-based requirement classification.
- CSV and Excel export implementation.

### Work in Progress

- Comprehensive retrieval and answer-quality evaluation.
- Model and quantization comparison.
- Retrieval threshold calibration.
- Offline deployment and packaging.
- User interface and final user guide.

## Important Notes

- This project is currently a local prototype and is not intended for safety-critical engineering decisions without independent verification.
- LLM-generated answers and requirement classifications should be reviewed against their original source documents.
- Ollama models are downloaded separately and are not included in this repository.
- Public source documents remain subject to their respective redistribution and licensing terms.

## Author

**Hassan Ansari**

GitHub: https://github.com/hassan07ansari
