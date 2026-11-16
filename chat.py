"""
chat.py
Interactive Q&A loop. Retrieves the most relevant chunks from embeddings.npz
using cosine similarity (numpy), then sends them as context to a local Ollama
model. Everything runs on your machine after the one-time model download.

Before running:
    ollama pull llama3.2
Optional environment variables:
    OLLAMA_HOST   where Ollama is running (default: http://localhost:11434)
    OLLAMA_MODEL  which model to use (default: llama3.2)
"""

import os
import sys

import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "embeddings.npz")
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the answer isn't in the context, say you "
    "don't know based on the provided documents.\n\n"
    "Context:\n{context}"
)


def load_store():
    data = np.load(STORE_PATH, allow_pickle=True)
    embeddings = data["embeddings"]
    texts = data["texts"]
    sources = data["sources"]
    model = SentenceTransformer(EMBED_MODEL)
    return embeddings, texts, sources, model


def cosine_similarity(query_vec, matrix):
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm


def retrieve(query, embeddings, texts, sources, model, top_k=TOP_K):
    query_vec = model.encode([query])[0]
    sims = cosine_similarity(query_vec, embeddings)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [texts[i] for i in top_idx], [sources[i] for i in top_idx]


def ask_ollama(query, context):
    system = SYSTEM_PROMPT.format(context=context)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )
    return response["message"]["content"]


def main():
    if not os.path.exists(STORE_PATH):
        print("embeddings.npz not found. Run `python ingest.py` first.")
        sys.exit(1)

    print("Loading embeddings and model ...")
    embeddings, texts, sources, model = load_store()
    print("Ready. Type your question (or 'exit' to quit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        docs, doc_sources = retrieve(query, embeddings, texts, sources, model)
        context = "\n\n".join(docs)

        try:
            answer = ask_ollama(query, context)
        except Exception as e:
            print(f"\nCould not reach Ollama ({e}).")
            print("Check that Ollama is running and the model is pulled.\n")
            continue

        print(f"\nAssistant: {answer}")
        print(f"(sources: {', '.join(sorted(set(doc_sources)))})\n")


if __name__ == "__main__":
    main()
