"""
agent_tooling.py — Standalone intent classifier for offline testing.

Loads the model directly (no server required). Used by test_classifier.py.
"""

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_ALLOWED_INTENTS = {"create_tracker", "unknown"}
_DEFAULT_MODEL = "models/Qwen3-0.6B"


def load_model(model_path: str = _DEFAULT_MODEL):
    """Load tokenizer and model. Returns (tokenizer, model)."""
    path = Path(model_path)
    if not path.exists():
        path = Path(__file__).parent / model_path
    resolved = str(path.resolve()) if path.exists() else model_path

    tokenizer = AutoTokenizer.from_pretrained(resolved)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(resolved, device_map="auto")
    model.eval()
    return tokenizer, model


def _extract_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _normalize(obj: dict[str, Any]) -> dict[str, str]:
    intent = str(obj.get("intent", "unknown")).strip()
    tracker_type = str(obj.get("tracker_type", "")).strip()
    description = str(obj.get("description", "")).strip()

    if intent not in _ALLOWED_INTENTS:
        intent = "unknown"

    return {"intent": intent, "tracker_type": tracker_type, "description": description}


def classify_intent(tokenizer, model, user_text: str, system_prompt: str) -> dict[str, str]:
    """Classify a user message. Returns {intent, tracker_type, description}."""
    device = str(next(model.parameters()).device)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    prompt = None
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        for extra in ({"enable_thinking": False}, {}):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **extra
                )
                break
            except TypeError:
                continue
    if prompt is None:
        lines = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        prompt = "\n".join(lines) + "\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        return _normalize(json.loads(_extract_json(raw)))
    except Exception:
        return {"intent": "unknown", "tracker_type": "", "description": ""}
