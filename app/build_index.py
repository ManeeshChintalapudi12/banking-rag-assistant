"""
CLI entry point: `python -m app.build_index`

Ingests every document in DOCS_DIR, chunks it, embeds it, and
persists a FAISS index to INDEX_DIR.
"""
from app.vectorstore import build_index

if __name__ == "__main__":
    store = build_index(persist=True)
    print(f"Index built with {store.index.ntotal} vectors.")
