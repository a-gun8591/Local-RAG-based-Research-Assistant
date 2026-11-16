"""
ingest.py
Reads every .txt file in data/, splits it into overlapping chunks, embeds
them with all-MiniLM-L6-v2, and saves the chunks and vectors to embeddings.npz.

No vector database needed: just numpy arrays on disk.
Re-run this whenever you add or change files in data/. The index is rebuilt
from scratch each time.
"""

import glob
import os

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STORE_PATH = os.path.join(BASE_DIR, "embeddings.npz")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500    # characters per chunk
CHUNK_OVERLAP = 50  # characters shared between neighbouring chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_documents():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    texts, sources = [], []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        for chunk in chunk_text(text):
            texts.append(chunk)
            sources.append(os.path.basename(path))
    return texts, sources, paths


def main():
    print(f"Loading documents from {DATA_DIR} ...")
    texts, sources, paths = load_documents()
    if not paths:
        print("No .txt files found in data/. Add some and run this again.")
        return
    print(f"Found {len(paths)} file(s), {len(texts)} chunks.")

    print(f"Loading embedding model ({EMBED_MODEL}) ...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Embedding chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"Saving to {STORE_PATH} ...")
    np.savez(
        STORE_PATH,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        sources=np.array(sources, dtype=object),
    )
    print(f"Done. Ingested {len(texts)} chunks into {STORE_PATH}")


if __name__ == "__main__":
    main()
