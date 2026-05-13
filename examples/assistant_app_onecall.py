"""
assistant_app_onecall.py — Memory-aware assistant with one-call JSON contract.

One inference per turn: model returns {"answer": "...", "facts": ["..."]}.
Simpler than two-call extraction, with safe JSON fallback.

Usage:
    python examples/assistant_app_onecall.py
"""

import json
import re
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server as _server
from ui_server import UIServer

# ---------------------------------------------------------------------------
# Fact memory
# ---------------------------------------------------------------------------

FACTS: list[str] = []
_facts_lock = threading.Lock()

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fact-merger")
_state_lock = threading.Lock()
_in_flight: "Future[None] | None" = None
_pending: "tuple[list[str]] | None" = None

_BASE_PROMPT = (
    "You are a helpful assistant with memory. "
    "You remember facts the user has shared and use them when answering questions."
)

_SYSTEM_INSTRUCTION = (
    "When answering, return ONLY a JSON object: "
    '{"answer": "text", "facts": [...]}. '
    "Facts: only concrete, objective facts from user message. "
    "Empty array [] if none."
)


def _render_system_prompt(facts: list[str]) -> str:
    if not facts:
        facts_block = "(none)"
    else:
        facts_block = "\n".join(f"- {f}" for f in facts)
    return f"{_BASE_PROMPT}\n\nKnown facts:\n{facts_block}\n\n{_SYSTEM_INSTRUCTION}"


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """Extract JSON object from raw text using multiple strategies."""
    text = (raw or "").strip()
    if not text:
        return None

    # 1) Direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Fenced block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 3) First {...}
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def _safe_parse_reply(raw: str) -> tuple[str, list[str]]:
    """Parse model reply: extract answer and facts, with safe fallback."""
    obj = _extract_json_object(raw)
    if obj is None:
        return (raw or "").strip(), []

    answer = obj.get("answer", "")
    facts = obj.get("facts", [])

    if not isinstance(answer, str):
        answer = str(answer) if answer else ""
    if not isinstance(facts, list):
        facts = []

    clean_facts: list[str] = []
    for item in facts:
        if isinstance(item, str):
            s = item.strip()
            if s:
                clean_facts.append(s)

    answer = answer.strip() or (raw or "").strip()
    return answer, clean_facts


def _merge_facts_async(new_facts: list[str]) -> None:
    """Background task: merge facts and update system prompt."""
    added: list[str] = []
    with _facts_lock:
        for fact in new_facts:
            if fact not in FACTS:
                FACTS.append(fact)
                added.append(fact)
        system_prompt = _render_system_prompt(list(FACTS))

    if added:
        print(f"[memory] +{len(added)} fact(s): {added}", flush=True)
        print(f"[memory] total facts ({len(FACTS)}): {FACTS}", flush=True)
    _server.ARGS.system_prompt = system_prompt


def _submit_merge(new_facts: list[str]) -> "Future[None]":
    """Submit fact merge and chain pending facts on completion."""
    global _in_flight, _pending

    def _on_done(fut: "Future[None]") -> None:
        global _in_flight, _pending
        with _state_lock:
            item = _pending
            _pending = None
            if item is not None:
                _in_flight = _submit_merge(item[0])
            else:
                _in_flight = None

    f: Future[None] = _executor.submit(_merge_facts_async, new_facts)
    f.add_done_callback(_on_done)
    return f


def _enqueue_facts(new_facts: list[str]) -> None:
    """Enqueue fact merge: if worker is busy, overwrite pending with latest."""
    if not new_facts:
        return

    global _in_flight, _pending
    with _state_lock:
        if _in_flight is None or _in_flight.done():
            _in_flight = _submit_merge(new_facts)
        else:
            _pending = (new_facts,)


def _on_response(user_msg: str, assistant_reply: str) -> None:
    """After-response callback: extract and parse facts from reply."""
    _, new_facts = _safe_parse_reply(assistant_reply)
    _enqueue_facts(new_facts)


def _postprocess_response_for_ui(raw_reply: str) -> str:
    """Show only the natural-language answer in chat UI."""
    answer, _ = _safe_parse_reply(raw_reply)
    return answer


if __name__ == "__main__":
    ui = UIServer(
        model="models/Qwen3-1.7B",
        system_prompt=_render_system_prompt([]),
        enable_thinking=False,
        context_size=8192,
        host="127.0.0.1",
        port=8080,
        device="cuda" if _server.torch.cuda.is_available() else "cpu",
    )

    _server.register_response_postprocessor(_postprocess_response_for_ui)
    _server.register_after_response(_on_response)

    try:
        ui.run()
    except KeyboardInterrupt:
        print("\n[memory] shutting down gracefully...", flush=True)
        _executor.shutdown(wait=False, cancel_futures=True)
