import os
import ollama

from hybrid_search import load_or_build_bm25_index, hybrid_search


MODEL_NAME = "qwen3:4b-instruct"
TOP_K = 5


def format_source(metadata):
    """Create a readable citation label for a retrieved chunk."""

    filename = metadata.get("source_file") or metadata.get("document_name", "Unknown document")
    filename = os.path.basename(filename)

    section = metadata.get("section")
    page = metadata.get("page_number")

    citation = filename

    if section:
        citation += f", Section: {section}"

    if page is not None:
        citation += f", p. {page}"

    return citation


def build_context(results):
    """Convert retrieved chunks into context for the LLM."""

    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        citation = format_source(metadata)

        context_parts.append(
            f"""
[SOURCE {index}]
Citation: [{citation}]
Content:
{result["text"]}
"""
        )

    return "\n".join(context_parts)


def generate_answer(question, results):
    """Generate an answer using only the retrieved document chunks."""

    context = build_context(results)

    system_prompt = """
You are an engineering document question-answering assistant.

Answer the user's question using ONLY the supplied document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent values, specifications, or requirements.
3. If the context does not contain enough evidence, respond with exactly:
Not found in the documents.
4. Cite the source of factual claims using the citation labels provided.
5. Do not invent document names, sections, or page numbers.
6. For comparison questions, clearly distinguish between document revisions.
7. Keep the answer clear and technically accurate.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:
{question}

Provide your answer with source citations.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0}
    )

    answer = response.message.content

    token_count = response.eval_count or 0
    generation_time = response.eval_duration or 0

    if generation_time > 0:
        tokens_per_second = token_count / (
            generation_time / 1_000_000_000
        )
    else:
        tokens_per_second = 0

    return {
        "answer": answer,
        "performance": {
            "model": MODEL_NAME,
            "generated_tokens": token_count,
            "generation_speed": round(tokens_per_second, 2)
        }
    }


def answer_question(question, bm25, bm25_records):
    """Retrieve relevant chunks and generate a structured answer."""

    print("\nSearching documents...")

    results = hybrid_search(
        query=question,
        bm25=bm25,
        bm25_records=bm25_records,
        top_k=TOP_K
    )

    sources = []

    for result in results:
        sources.append({
            "citation": format_source(result["metadata"]),
            "rrf_score": result["rrf_score"],
            "semantic_distance": result.get("semantic_distance"),
            "bm25_score": result.get("bm25_score")
        })

    if not results:
        return {
            "question": question,
            "answer": "Not found in the documents.",
            "sources": [],
            "performance": None
        }

    answer_result = generate_answer(question, results)

    return {
        "question": question,
        "answer": answer_result["answer"],
        "sources": sources,
        "performance": answer_result["performance"]
    }



def main():

    print("=" * 60)
    print("OFFLINE ENGINEERING DOCUMENT RAG ASSISTANT")
    print("=" * 60)

    bm25, bm25_records = load_or_build_bm25_index()

    print("\nAssistant is ready!")
    print("Enter 'exit' to close the application.")

    while True:

        question = input("\nAsk a question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nClosing assistant...")
            break

        if not question:
            print("Please enter a question.")
            continue

        result = answer_question(question, bm25, bm25_records)

        # Display the answer
        print("\nANSWER")
        print("-" * 60)
        print(result["answer"])

        # Display source citations
        if result["sources"]:

            print("\nSOURCES")
            print("-" * 60)

            for index, source in enumerate(result["sources"], start=1):
                print(f"{index}. {source['citation']}")

        else:
            print("\nNo supporting documents found.")


if __name__ == "__main__":
    main()
