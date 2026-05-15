"""
assistant_app.py — Memory-aware assistant.

After each conversation turn the assistant runs a background LLM call that
extracts concrete facts from what the user said.  Facts accumulate in a
module-level list and are injected into the system prompt so the model can
answer questions about previously stated information.

Usage:
    python examples/assistant_app.py
"""

import json
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from pathlib import Path
from typing import Deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server as _server
from ui_server import UIServer

# ---------------------------------------------------------------------------
# Fact memory
# ---------------------------------------------------------------------------

FACTS: list[str] = []
_facts_lock = threading.Lock()

# Extraction concurrency:
#   - One worker thread via ThreadPoolExecutor(max_workers=1).
#   - FIFO pending queue in _pending.
#   - No dropped turns: every enqueued turn is extracted eventually.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="extraction")
_state_lock = threading.Lock()
_in_flight: "Future[None] | None" = None
_pending: Deque[tuple[str, str]] = deque()
_args_lock = threading.Lock()  # Protect server.ARGS updates from background thread

_BASE_PROMPT = (
    "You are a helpful assistant with memory. "
    "You remember facts the user has shared and use them when answering questions."
)

_EXTRACTION_PROMPT = """\
Extract concrete, objective facts from the user message and assistant response below.
Return ONLY a JSON array of short fact strings.
Return [] if there are no new facts worth remembering.

PRIORITY:
1. Explicit facts stated by the USER (highest importance)
2. Factual information in ASSISTANT's response
3. Do NOT include: opinions, commands, questions, or meta-commentary

Examples of good facts: ["User's name is Alice", "User has a dog named Rex", "User lives in Berlin"]
Examples to exclude: ["User asked a question", "User wants help", "The assistant provided information"]

User: {user}
Assistant: {assistant}

JSON array of new facts:"""


def _render_system_prompt(facts: list[str]) -> str:
    if not facts:
        return _BASE_PROMPT
    facts_block = "\n".join(f"- {f}" for f in facts)
    return f"{_BASE_PROMPT}\n\nKnown facts about the user:\n{facts_block}"


def _run_extraction(user_msg: str, assistant_reply: str, ui: UIServer) -> None:
    try:
        def _call_extraction(prompt: str, *, max_tokens: int = 256) -> str:
            result = ui.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                enable_thinking=False,
                reasoning_format="none",
                suppress_after_response=True,
            )
            msg = result.choices[0].message
            content = (msg.content or "").strip()
            reasoning_content = (getattr(msg, "reasoning_content", None) or "").strip()
            print(
                "[memory][debug] extraction reply fields: "
                f"content_len={len(content)}, reasoning_len={len(reasoning_content)}",
                flush=True,
            )
            if content:
                return content
            # Some models can place output in reasoning_content despite settings.
            return reasoning_content or "[]"

        raw = _call_extraction(
            _EXTRACTION_PROMPT.format(user=user_msg, assistant=assistant_reply),
            max_tokens=256,
        )
        start, end = raw.find("["), raw.rfind("]") + 1
        if not (0 <= start < end):
            preview = raw.strip().replace("\n", " ")[:220]
            print(f"[memory][warn] extraction JSON array not found; raw={preview!r}", flush=True)
            return

        try:
            new_facts: list = json.loads(raw[start:end])
            print(f"[memory] extracted facts: {new_facts}", flush=True)
        except json.JSONDecodeError as exc:
            preview = raw[start:end].replace("\n", " ")[:220]
            print(f"[memory][warn] extraction JSON parse failed: {exc}; payload={preview!r}", flush=True)
            return

        if not isinstance(new_facts, list):
            print(f"[memory][warn] extraction payload is not a list: {type(new_facts).__name__}", flush=True)
            return

        if not new_facts:
            preview = raw.strip().replace("\n", " ")[:160]
            print(f"[memory][debug] extraction returned empty array; raw={preview!r}", flush=True)

        new_added = []
        with _facts_lock:
            for fact in new_facts:
                if isinstance(fact, str) and fact.strip():
                    f = fact.strip()
                    if f not in FACTS:
                        FACTS.append(f)
                        new_added.append(f)
            system_prompt = _render_system_prompt(list(FACTS))

        if new_added:
            print(f"[memory] +{len(new_added)} fact(s): {new_added}", flush=True)
            print(f"[memory] total facts ({len(FACTS)}): {FACTS}", flush=True)
        else:
            print(f"[memory] no new facts; total facts ({len(FACTS)}): {FACTS}", flush=True)

        with _args_lock:
            _server.ARGS.system_prompt = system_prompt
    except Exception as exc:
        print(f"[memory] extraction error: {exc}", flush=True)


def _submit_extraction(user_msg: str, assistant_reply: str, ui: UIServer) -> "Future[None]":
    """Submit extraction and chain the pending turn on completion."""
    global _in_flight

    def _on_done(fut: "Future[None]") -> None:
        global _in_flight
        with _state_lock:
            try:
                fut.result()  # Raises if the extraction task had an exception
            except Exception as e:
                print(f"[memory] extraction task failed: {e}", flush=True)

            if _pending:
                item = _pending.popleft()
                print(f"[memory] processing queued turn ({len(_pending)} more pending)", flush=True)
                _in_flight = _submit_extraction(item[0], item[1], ui)
            else:
                _in_flight = None

    f: Future[None] = _executor.submit(_run_extraction, user_msg, assistant_reply, ui)
    f.add_done_callback(_on_done)
    return f


def _make_enqueue(ui: UIServer):
    """Return an after-response callback that schedules extraction via the executor."""
    def _enqueue(user_msg: str, assistant_reply: str) -> None:
        global _in_flight
        with _state_lock:
            # Only the done-callback advances the queue. This preserves strict FIFO
            # and avoids handoff races when a future just finished.
            if _in_flight is None:
                print(f"[memory] starting extraction for user message ({len(user_msg)} chars)", flush=True)
                _in_flight = _submit_extraction(user_msg, assistant_reply, ui)
            else:
                _pending.append((user_msg, assistant_reply))
                print(f"[memory] queued extraction turn; pending={len(_pending)} (worker busy)", flush=True)
    return _enqueue


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = UIServer(
        model="models/Qwen3-1.7B",
        system_prompt=_BASE_PROMPT,
        enable_thinking=False,
        context_size=8192,
        host="127.0.0.1",
        port=8080,
        device="cuda" if _server.torch.cuda.is_available() else "cpu",
    )

    _server.register_after_response(_make_enqueue(ui))

    try:
        ui.run()
    except KeyboardInterrupt:
        print("\n[memory] shutting down gracefully...", flush=True)
        _executor.shutdown(wait=False, cancel_futures=True)
