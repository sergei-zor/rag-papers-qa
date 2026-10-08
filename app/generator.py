"""
Takes a question, retrieves supporting chunks, and calls an LLM (Ollama/Bedrock) 
to generate an answer supported by provided chunks.
"""

import logging
import os

from .retriever import retrieve

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
logging.getLogger("uvicorn.error").info(f"Using LLM provider: {LLM_PROVIDER}")


def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    return f"""You are answering questions using ONLY the research paper chunks below.
If the answer isn't in the chunks, say so clearly instead of guessing.
When you state a specific fact or number, cite which source and page it came from.

Context:
{context}

Question: {question}

Answer:"""


def call_llm(prompt):
    if LLM_PROVIDER == "bedrock":
        from .llm_bedrock import generate
    else:
        from .llm_ollama import generate
    return generate(prompt)


def generate_answer(question, n_results = 6):
    chunks = retrieve(question, n_results=n_results)
    prompt = build_prompt(question, chunks)
    answer = call_llm(prompt)
    return {
        "answer": answer,
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
    }


if __name__ == "__main__":
    # run from the project root: python -m app.generator
    question = "What are mathematical expressions of yield stress and elastic modulus scaling laws?"
    result = generate_answer(question)
    print(f"Question: {question}\n\nAnswer:\n{result['answer']}\n")
    print("Sources used:")
    for s in result["sources"]:
        print(f"  - {s['source']}, page {s['page']}")