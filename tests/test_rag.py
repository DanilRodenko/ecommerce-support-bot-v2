from types import SimpleNamespace

import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from src.chain import ask
from src.ingest import load_file, process_file, split_document
from src.retriever import retrieve

CLOTHING = "data/uploads/clothing_faq.txt"
ELECTRONICS = "data/uploads/electronics_faq.txt"


@pytest.fixture(scope="module")
def embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class FakeClient:
    """Pretends to be Groq: remembers what it was sent and returns a fixed answer."""

    def __init__(self, answer):
        self.answer = answer
        self.last_model = None
        self.last_messages = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages):
        self.last_model = model
        self.last_messages = messages
        message = SimpleNamespace(content=self.answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_split_document_creates_chunks():
    chunks = split_document(load_file(CLOTHING))

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 500


def test_vectorstores_are_isolated(embeddings):
    clothing_store = process_file(CLOTHING, embeddings)
    process_file(ELECTRONICS, embeddings)  # a second store must not leak into the first

    results = retrieve(clothing_store, "How long is the warranty on a laptop?")

    for chunk in results:
        assert "laptop" not in chunk.page_content.lower()


def test_retrieve_finds_relevant_chunk(embeddings):
    store = process_file(CLOTHING, embeddings)

    results = retrieve(store, "Will my t-shirt get smaller after washing?")

    assert any("30°C" in chunk.page_content for chunk in results)


def test_ask_sends_context_and_returns_answer(embeddings):
    store = process_file(ELECTRONICS, embeddings)
    client = FakeClient(answer="12 months")
    question = "How long is the warranty on a laptop?"

    answer, history = ask(question, store, client, model="test-model")

    assert answer == "12 months"
    assert len(history) == 2
    assert history[0]["content"] == question        # plain question, no context
    assert client.last_model == "test-model"
    assert "12-month" in client.last_messages[-1]["content"]  # retrieved context was sent