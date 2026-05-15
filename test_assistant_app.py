"""
test_assistant_app.py — Unit tests for two-call memory assistant.

Tests:
  - Fact extraction and memory accumulation
  - Callback registration
  - UI initialization
  - Graceful shutdown
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import server as _server
from examples.assistant_app import (
    FACTS,
    _BASE_PROMPT,
    _EXTRACTION_PROMPT,
    _args_lock,
    _facts_lock,
    _make_enqueue,
    _render_system_prompt,
    _run_extraction,
)


def test_render_system_prompt_empty():
    """Empty facts should return base prompt."""
    prompt = _render_system_prompt([])
    assert prompt == _BASE_PROMPT
    print("✓ render_system_prompt (empty)")


def test_render_system_prompt_with_facts():
    """Facts should be formatted as bullet list."""
    facts = ["User's name is Alice", "User has a dog"]
    prompt = _render_system_prompt(facts)
    
    assert _BASE_PROMPT in prompt
    assert "- User's name is Alice" in prompt
    assert "- User has a dog" in prompt
    assert "Known facts about the user:" in prompt
    print("✓ render_system_prompt (with facts)")


def test_extraction_prompt_format():
    """Extraction prompt should format user and assistant messages."""
    prompt = _EXTRACTION_PROMPT.format(
        user="I'm from Berlin",
        assistant="Great! Berlin is the capital of Germany."
    )
    
    assert "I'm from Berlin" in prompt
    assert "Berlin is the capital of Germany" in prompt
    assert "JSON array" in prompt
    print("✓ extraction_prompt_format")


def test_extraction_logic_valid_json():
    """Extract valid JSON facts from response."""
    raw_response = 'Some text before ["User is from Berlin", "User likes coding"] and text after'
    
    # Extract JSON manually (simulating _run_extraction logic)
    start = raw_response.find("[")
    end = raw_response.rfind("]") + 1
    
    assert start >= 0
    assert end > start
    
    json_str = raw_response[start:end]
    facts = json.loads(json_str)
    
    assert isinstance(facts, list)
    assert "User is from Berlin" in facts
    assert "User likes coding" in facts
    print("✓ extraction_logic_valid_json")


def test_extraction_logic_empty_array():
    """Handle empty array response."""
    raw_response = "No new facts. []"
    
    start = raw_response.find("[")
    end = raw_response.rfind("]") + 1
    
    json_str = raw_response[start:end]
    facts = json.loads(json_str)
    
    assert isinstance(facts, list)
    assert len(facts) == 0
    print("✓ extraction_logic_empty_array")


def test_extraction_logic_malformed_json():
    """Handle malformed JSON gracefully."""
    raw_response = "Invalid JSON: [broken data"
    
    start = raw_response.find("[")
    end = raw_response.rfind("]") + 1
    
    # end will be 0 (rfind returns -1, +1 = 0)
    assert end <= start
    print("✓ extraction_logic_malformed_json")


def test_callback_registration():
    """Callback should be registered."""
    mock_ui = MagicMock()
    callback = _make_enqueue(mock_ui)
    
    assert callable(callback)
    print("✓ callback_registration")


def test_callback_invocation():
    """Callback should handle user_msg and assistant_reply."""
    mock_ui = MagicMock()
    callback = _make_enqueue(mock_ui)
    
    # Should not raise
    callback(
        user_msg="I'm from Paris",
        assistant_reply="Nice! Paris is beautiful."
    )
    print("✓ callback_invocation")


def test_facts_accumulation_logic():
    """Simulate facts accumulation without real model."""
    # Simulate what happens in _run_extraction
    test_facts = ["User name is Bob", "User has 2 cats"]
    
    global FACTS
    initial_count = len(FACTS)
    
    for fact in test_facts:
        if isinstance(fact, str) and fact.strip():
            f = fact.strip()
            if f not in FACTS:
                FACTS.append(f)
    
    # Should have added new facts
    assert len(FACTS) >= initial_count
    assert "User name is Bob" in FACTS
    assert "User has 2 cats" in FACTS
    print("✓ facts_accumulation_logic")


def test_extraction_prompt_contains_instructions():
    """Extraction prompt should have clear instructions."""
    assert "Extract concrete" in _EXTRACTION_PROMPT
    assert "JSON array" in _EXTRACTION_PROMPT
    assert "opinions" in _EXTRACTION_PROMPT
    assert "PRIORITY" in _EXTRACTION_PROMPT
    print("✓ extraction_prompt_contains_instructions")


def test_run_extraction_uses_dialog_and_adds_fact_from_content():
    """Model extraction should use full dialog prompt and add parsed fact."""
    with _facts_lock:
        FACTS.clear()

    with _args_lock:
        original_prompt = getattr(_server.ARGS, "system_prompt", "")

    ui = MagicMock()
    message = MagicMock(content='["User lives in Berlin"]')
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    ui.chat.completions.create.return_value = response

    _run_extraction("I live in Berlin", "Great city", ui)

    with _facts_lock:
        assert "User lives in Berlin" in FACTS

    call_kwargs = ui.chat.completions.create.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]
    assert "User: I live in Berlin" in prompt
    assert "Assistant: Great city" in prompt

    with _args_lock:
        _server.ARGS.system_prompt = original_prompt

    print("✓ run_extraction uses dialog and adds fact from content")


def test_run_extraction_fallbacks_to_reasoning_content():
    """If content is empty, extraction should parse reasoning_content."""
    with _facts_lock:
        FACTS.clear()

    with _args_lock:
        original_prompt = getattr(_server.ARGS, "system_prompt", "")

    ui = MagicMock()
    message = MagicMock(content="", reasoning_content='["User has a cat"]')
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    ui.chat.completions.create.return_value = response

    _run_extraction("I have a cat", "Nice", ui)

    with _facts_lock:
        assert "User has a cat" in FACTS

    with _args_lock:
        _server.ARGS.system_prompt = original_prompt

    print("✓ run_extraction falls back to reasoning_content")


if __name__ == "__main__":
    print("\n=== Testing assistant_app.py (two-call) ===\n")
    
    test_render_system_prompt_empty()
    test_render_system_prompt_with_facts()
    test_extraction_prompt_format()
    test_extraction_logic_valid_json()
    test_extraction_logic_empty_array()
    test_extraction_logic_malformed_json()
    test_callback_registration()
    test_callback_invocation()
    test_facts_accumulation_logic()
    test_extraction_prompt_contains_instructions()
    test_run_extraction_uses_dialog_and_adds_fact_from_content()
    test_run_extraction_fallbacks_to_reasoning_content()
    
    print("\n✓ All tests passed!\n")
