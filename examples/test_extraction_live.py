"""
test_extraction_live.py — Test extraction with real model to optimize prompts.

Run this to see how well the extraction prompt works on real user messages.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui_server import UIServer
from examples.assistant_app import _EXTRACTION_PROMPT

# Test cases: (user_msg, assistant_reply, expected_facts_should_include)
TEST_CASES = [
    (
        "I am ok. I want to sleep",
        "That's great to hear that you're ok! If you need any help with sleep or anything else, just let me know.",
        ["ok", "sleep"],  # Should find these words/concepts in extracted facts
    ),
    (
        "My name is Alice and I live in Berlin. I have a dog named Rex.",
        "That's wonderful! Berlin is a beautiful city.",
        ["Alice", "Berlin", "dog", "Rex"],
    ),
    (
        "I work as a software engineer and I like Python",
        "That's a great skill to have!",
        ["software engineer", "Python"],
    ),
    (
        "Can you help me with coding?",
        "Of course! What do you need help with?",
        [],  # This is a question, should extract no facts
    ),
]


def test_extraction_with_model():
    """Test extraction prompt with real model."""
    print("\n=== Live Extraction Test ===\n")
    
    # Initialize UI with real model
    ui = UIServer(
        model="models/Qwen3-4B-Instruct-2507",
        system_prompt="You are a helpful assistant.",
        enable_thinking=False,
        context_size=8192,
        host="127.0.0.1",
        port=8080,
        device="cuda",
    )
    
    passed = 0
    failed = 0
    
    for i, (user_msg, assistant_reply, expected_keywords) in enumerate(TEST_CASES, 1):
        print(f"Test {i}: user='{user_msg[:50]}...'")
        print(f"  Expected keywords: {expected_keywords}")
        
        # Call extraction
        prompt = _EXTRACTION_PROMPT.format(user=user_msg, assistant=assistant_reply)
        
        try:
            result = ui.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # Slightly above 0 for better fact extraction
                max_tokens=256,
                enable_thinking=False,
                suppress_after_response=True,
            )
            
            raw = result.choices[0].message.content or "[]"
            print(f"  Raw response: {raw[:120]!r}")
            
            # Parse JSON
            start, end = raw.find("["), raw.rfind("]") + 1
            if 0 <= start < end:
                facts = json.loads(raw[start:end])
                print(f"  Extracted facts: {facts}")
                
                # Check if expected keywords are in facts
                facts_str = " ".join(facts).lower()
                found_keywords = [kw for kw in expected_keywords if kw.lower() in facts_str]
                missing_keywords = [kw for kw in expected_keywords if kw.lower() not in facts_str]
                
                if len(found_keywords) >= len(expected_keywords) * 0.7:  # 70% threshold
                    print(f"  ✓ PASS (found {len(found_keywords)}/{len(expected_keywords)} keywords)")
                    passed += 1
                else:
                    print(f"  ✗ FAIL (found {len(found_keywords)}/{len(expected_keywords)}, missing: {missing_keywords})")
                    failed += 1
            else:
                print(f"  ✗ FAIL (no JSON array found)")
                failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
        
        print()
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    return failed == 0


if __name__ == "__main__":
    success = test_extraction_with_model()
    sys.exit(0 if success else 1)
