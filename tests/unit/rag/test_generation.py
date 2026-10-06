from types import SimpleNamespace

import pytest

from rag.config import Settings
from rag.generation import generate_answer
from rag.models import BuiltContext


class FakeClient:
    def __init__(self, answer):
        self.answer = answer
        self.request = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])


def test_generation_receives_multiple_passages_and_returns_answer():
    context = BuiltContext(text="[Source 1]\nEmployment: 2018–2020\n\n[Source 2]\nEmployment: 2020–2024")
    client = FakeClient("She has approximately six years of experience. [Source 1] [Source 2]")
    settings = Settings(_env_file=None, OPENAI_RESPONSE_MODEL="test-model")
    answer = generate_answer("How many years of experience does she have?", context,
                             settings, client=client)
    assert answer == client.answer
    assert client.request["model"] == "test-model"
    messages = client.request["messages"]
    assert context.text in messages[1]["content"]
    assert "How many years" in messages[1]["content"]
    assert "count overlapping periods only once" in messages[0]["content"]


def test_generation_without_context_does_not_call_model():
    client = FakeClient("should not be used")
    answer = generate_answer("What is her experience?", BuiltContext(text=""), client=client)
    assert "could not find enough information" in answer
    assert client.request is None


def test_generation_rejects_empty_model_answer():
    with pytest.raises(ValueError, match="empty response"):
        generate_answer("Question", BuiltContext(text="Evidence"),
                        Settings(_env_file=None), client=FakeClient(" "))
