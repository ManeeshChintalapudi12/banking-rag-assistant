from app.ingestion import chunk_documents, load_documents


def test_load_documents_finds_sample_docs():
    docs = load_documents("data/sample_docs")
    assert len(docs) >= 2
    sources = {d.metadata["source"] for d in docs}
    assert "wire_transfer_policy.md" in sources
    assert "account_holds_and_disputes.md" in sources


def test_chunk_documents_assigns_stable_chunk_ids():
    docs = load_documents("data/sample_docs")
    chunks = chunk_documents(docs)
    assert len(chunks) > len(docs)  # documents were split
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"].startswith(chunk.metadata["source"])
