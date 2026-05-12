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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server as _server
from ui_server import UIServer

# ---------------------------------------------------------------------------
# Fact memory
# ---------------------------------------------------------------------------

FACTS: list[str] = []
_lock = threading.Lock()

_BASE_PROMPT = (
    "You are a helpful assistant with memory. "
    "You remember facts the user has shared and use them when answering questions."
)

_EXTRACTION_PROMPT = """\
Extract concrete, objective facts from the conversation turn below.
Return ONLY a JSON array of short fact strings.
Return [] if there are no new facts worth remembering.
Do NOT include opinions, commands, or questions — only facts about the user or the world.

Examples of good facts: ["User's name is Alice", "User has a dog named Rex", "User lives in Berlin"]
Examples of what NOT to include: ["User asked a question", "User wants help"]

User: {user}
Assistant: {assistant}

JSON array of new facts:"""


def _current_system_prompt() -> str:
    with _lock:
        if not FACTS:
            return _BASE_PROMPT
        facts_block = "\n".join(f"- {f}" for f in FACTS)
        return f"{_BASE_PROMPT}\n\nKnown facts about the user:\n{facts_block}"


def _run_extraction(user_msg: str, assistant_reply: str, ui: UIServer) -> None:
    try:
        result = ui.chat.completions.create(
            messages=[{
                "role": "user",
                "content": _EXTRACTION_PROMPT.format(
                    user=user_msg,
                    assistant=assistant_reply,
                ),
            }],
            temperature=0.0,
            max_tokens=256,
            enable_thinking=False,
        )
        raw = result.choices[0].message.content or "[]"
        start, end = raw.find("["), raw.rfind("]") + 1
        if 0 <= start < end:
            new_facts: list = json.loads(raw[start:end])
            if isinstance(new_facts, list):
                added = []
                with _lock:
                    for fact in new_facts:
                        if isinstance(fact, str) and fact.strip():
                            FACTS.append(fact.strip())
                            added.append(fact.strip())
                if added:
                    print(f"[memory] +{len(added)} fact(s): {added}", flush=True)
                    print(f"[memory] total facts ({len(FACTS)}): {FACTS}", flush=True)
        _server.ARGS.system_prompt = _current_system_prompt()
    except Exception as exc:
        print(f"[memory] extraction error: {exc}", flush=True)


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

    _server.register_after_response(
        lambda user, reply: threading.Thread(
            target=_run_extraction, args=(user, reply, ui), daemon=True
        ).start()
    )

    ui.run()
