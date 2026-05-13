"""
test_assistant_app_onecall.py — Unit tests for one-call memory assistant.

Tests:
  - JSON parsing from model response
  - Safe fallback when JSON is malformed
  - Answer extraction
  - Facts extraction
  - Postprocessor registration
  - Integration between postprocessor and callback
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import server as _server
from examples.assistant_app_onecall import (
    FACTS,
    _extract_json_object,
    _safe_parse_reply,
    _postprocess_response_for_ui,
    _render_system_prompt,
    _merge_facts_async,
)


def test_extract_json_direct():
    """Extract JSON when entire response is JSON."""
    raw = '{"answer": "Hello!", "facts": ["User likes Python"]}'
    obj = _extract_json_object(raw)
    
    assert obj is not None
    assert obj.get("answer") == "Hello!"
    assert obj.get("facts") == ["User likes Python"]
    print("✓ extract_json_direct")


def test_extract_json_with_text_before():
    """Extract JSON when there's text before it."""
    raw = 'Let me think... {"answer": "Great!", "facts": []}'
    obj = _extract_json_object(raw)
    
    assert obj is not None
    assert obj.get("answer") == "Great!"
    print("✓ extract_json_with_text_before")


def test_extract_json_with_text_after():
    """Extract JSON when there's text after it."""
    raw = '{"answer": "Sure", "facts": ["fact1"]} and more text after'
    obj = _extract_json_object(raw)
    
    assert obj is not None
    assert obj.get("answer") == "Sure"
    print("✓ extract_json_with_text_after")


def test_extract_json_fenced_block():
    """Extract JSON from markdown code block."""
    raw = 'Here is the result:\n```json\n{"answer": "Test", "facts": []}\n```'
    obj = _extract_json_object(raw)
    
    assert obj is not None
    assert obj.get("answer") == "Test"
    print("✓ extract_json_fenced_block")


def test_extract_json_malformed():
    """Return None for malformed JSON."""
    raw = '{"answer": "broken", "facts": [incomplete'
    obj = _extract_json_object(raw)
    
    # Should try multiple strategies, eventually return None
    assert obj is None
    print("✓ extract_json_malformed")


def test_extract_json_no_json():
    """Return None when no JSON found."""
    raw = "Just plain text, no JSON here"
    obj = _extract_json_object(raw)
    
    assert obj is None
    print("✓ extract_json_no_json")


def test_safe_parse_reply_valid():
    """Parse valid one-call response."""
    raw = '{"answer": "Berlin is great!", "facts": ["User loves Berlin"]}'
    answer, facts = _safe_parse_reply(raw)
    
    assert answer == "Berlin is great!"
    assert facts == ["User loves Berlin"]
    print("✓ safe_parse_reply_valid")


def test_safe_parse_reply_empty_facts():
    """Handle response with empty facts."""
    raw = '{"answer": "Okay", "facts": []}'
    answer, facts = _safe_parse_reply(raw)
    
    assert answer == "Okay"
    assert facts == []
    print("✓ safe_parse_reply_empty_facts")


def test_safe_parse_reply_missing_answer():
    """Handle response missing answer field."""
    raw = '{"facts": ["fact1"]}'
    answer, facts = _safe_parse_reply(raw)
    
    # Should have empty answer
    assert isinstance(answer, str)
    assert facts == ["fact1"]
    print("✓ safe_parse_reply_missing_answer")


def test_safe_parse_reply_missing_facts():
    """Handle response missing facts field."""
    raw = '{"answer": "Hello"}'
    answer, facts = _safe_parse_reply(raw)
    
    assert answer == "Hello"
    assert facts == []
    print("✓ safe_parse_reply_missing_facts")


def test_safe_parse_reply_fallback():
    """Fallback to raw text when JSON is invalid."""
    raw = "Just text, not JSON"
    answer, facts = _safe_parse_reply(raw)
    
    # Should return raw as answer
    assert answer == "Just text, not JSON"
    assert facts == []
    print("✓ safe_parse_reply_fallback")


def test_safe_parse_reply_non_string_fields():
    """Handle non-string answer and non-list facts."""
    raw = '{"answer": 123, "facts": "not a list"}'
    answer, facts = _safe_parse_reply(raw)
    
    # Should convert answer to string
    assert isinstance(answer, str)
    assert answer == "123"
    # Should reset facts to empty list
    assert facts == []
    print("✓ safe_parse_reply_non_string_fields")


def test_safe_parse_reply_whitespace_cleanup():
    """Strip whitespace from facts."""
    raw = '{"answer": "Hi", "facts": ["  fact1  ", "fact2", "   "]}'
    answer, facts = _safe_parse_reply(raw)
    
    assert answer == "Hi"
    assert len(facts) == 2  # Empty fact after strip is filtered
    assert "fact1" in facts
    assert "fact2" in facts
    print("✓ safe_parse_reply_whitespace_cleanup")


def test_postprocessor_integration():
    """Postprocessor should extract answer for UI."""
    raw = '{"answer": "This is the answer", "facts": ["memory fact"]}'
    
    ui_text = _postprocess_response_for_ui(raw)
    
    assert ui_text == "This is the answer"
    assert "facts" not in ui_text  # Should not leak JSON
    print("✓ postprocessor_integration")


def test_postprocessor_fallback():
    """Postprocessor should return raw text if JSON parsing fails."""
    raw = "Just plain text"
    
    ui_text = _postprocess_response_for_ui(raw)
    
    assert ui_text == "Just plain text"
    print("✓ postprocessor_fallback")


def test_render_system_prompt_empty():
    """Empty facts should return base prompt."""
    prompt = _render_system_prompt([])
    
    assert "helpful assistant" in prompt
    assert "(none)" in prompt
    print("✓ render_system_prompt_empty")


def test_render_system_prompt_with_facts():
    """Facts should be formatted in prompt."""
    facts = ["User's name is Carol", "User has a dog"]
    prompt = _render_system_prompt(facts)
    
    assert "- User's name is Carol" in prompt
    assert "- User has a dog" in prompt
    print("✓ render_system_prompt_with_facts")


def test_merge_facts_no_duplicates():
    """New facts should not duplicate existing ones."""
    # Clear FACTS first
    global FACTS
    FACTS.clear()
    
    # Add initial fact
    FACTS.append("Initial fact")
    
    # Merge with overlapping facts
    new_facts = ["Initial fact", "New fact"]
    _merge_facts_async(new_facts)
    
    # Should have added only new_fact
    assert "Initial fact" in FACTS
    assert "New fact" in FACTS
    assert len(FACTS) == 2
    print("✓ merge_facts_no_duplicates")


def test_one_call_flow_simulation():
    """Simulate complete one-call flow: model → postprocessor → callback."""
    # Raw model response
    raw_response = '{"answer": "I like programming", "facts": ["User enjoys coding"]}'
    
    # UI sees only answer
    ui_text = _postprocess_response_for_ui(raw_response)
    assert ui_text == "I like programming"
    assert "facts" not in ui_text
    
    # Callback extracts facts from raw response
    answer, facts = _safe_parse_reply(raw_response)
    assert facts == ["User enjoys coding"]
    
    print("✓ one_call_flow_simulation")


if __name__ == "__main__":
    print("\n=== Testing assistant_app_onecall.py (one-call) ===\n")
    
    test_extract_json_direct()
    test_extract_json_with_text_before()
    test_extract_json_with_text_after()
    test_extract_json_fenced_block()
    test_extract_json_malformed()
    test_extract_json_no_json()
    test_safe_parse_reply_valid()
    test_safe_parse_reply_empty_facts()
    test_safe_parse_reply_missing_answer()
    test_safe_parse_reply_missing_facts()
    test_safe_parse_reply_fallback()
    test_safe_parse_reply_non_string_fields()
    test_safe_parse_reply_whitespace_cleanup()
    test_postprocessor_integration()
    test_postprocessor_fallback()
    test_render_system_prompt_empty()
    test_render_system_prompt_with_facts()
    test_merge_facts_no_duplicates()
    test_one_call_flow_simulation()
    
    print("\n✓ All tests passed!\n")
