import shutil

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.vectorstore import build_index


@pytest.fixture(scope="module", autouse=True)
def _build_test_index():
    # Ensure a fresh index exists before the test module runs.
    build_index(persist=True)
    yield
    shutil.rmtree(settings.INDEX_DIR, ignore_errors=True)


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["index_ready"] is True


def test_query_returns_grounded_answer_with_sources():
    resp = client.post(
        "/query", json={"question": "What is the fee for an outgoing domestic wire transfer?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "sources" in body
    assert len(body["sources"]) > 0
    assert body["sources"][0]["source"] == "wire_transfer_policy.md"


def test_query_on_unrelated_question_is_flagged_low_confidence():
    resp = client.post(
        "/query", json={"question": "What is the weather forecast for tomorrow?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    # An off-topic question should not be presented as confidently grounded.
    assert body["confidence"] < 0.5


def test_query_rejects_too_short_question():
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 422
