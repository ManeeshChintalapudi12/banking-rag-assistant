"""
Document ingestion: load policy documents from disk, split into
overlapping chunks, and attach source metadata used later for
citations in the API response.
"""
import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def load_documents(docs_dir: str = None) -> List[Document]:
    docs_dir = docs_dir or settings.DOCS_DIR
    documents: List[Document] = []

    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith((".md", ".txt")):
            continue
        path = os.path.join(docs_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(
            Document(page_content=text, metadata={"source": filename})
        )

    if not documents:
        raise FileNotFoundError(
            f"No .md/.txt documents found in '{docs_dir}'. Add policy "
            f"documents there before building the index."
        )
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Attach a stable, human-readable chunk id per source for citations.
    counters = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}#chunk-{counters[source]}"

    return chunks
