"""Stub tests for src.llm.backends and src.llm.prompts."""
import pytest

def test_build_messages_returns_list():
    from src.llm.backends import build_message
    msgs = build_message("system text", "user text")
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
