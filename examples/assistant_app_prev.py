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
You are a fact extraction expert. Extract ALL explicit facts about the user from their message.

From the user message, identify and extract:
1. Personal information: name, age, location, family, pets, profession
2. Explicit user statements: "I am X", "I want Y", "I have Z", "I like/dislike X"
3. Health/state: "I feel X", "I'm tired", "I'm ok"
4. Any concrete information the user shares about themselves

IMPORTANT RULES:
- Extract facts that are EXPLICITLY stated by the user
- Do NOT invent facts
- Do NOT include questions, requests without facts, or generic statements
- Each fact should start with "User" and be 3-15 words
- If user said "I want to sleep", output: "User wants to sleep"
- If user said "I am ok", output: "User is ok"

Return ONLY a JSON array of fact strings. Example format:
["User wants to sleep", "User is ok", "User's name is Alice"]

User message: {user}
Assistant response: {assistant}

JSON array (facts about the user):"""


def _current_system_prompt() -> str:
    with _lock:
        if not FACTS:
            return _BASE_PROMPT
        facts_block = "\n".join(f"- {f}" for f in FACTS)
        return f"{_BASE_PROMPT}\n\nKnown facts about the user:\n{facts_block}"


def _run_extraction(user_msg: str, assistant_reply: str, ui: UIServer) -> None:
    try:
        extraction_prompt = _EXTRACTION_PROMPT.format(
            user=user_msg,
            assistant=assistant_reply,
        )
        
        print(
            f"[memory][debug] extraction: user={user_msg[:80]!r}",
            flush=True,
        )
        
        result = ui.chat.completions.create(
            messages=[{
                "role": "user",
                "content": extraction_prompt,
            }],
            temperature=0.0,
            max_tokens=256,
            enable_thinking=False,
            suppress_after_response=True,
        )
        
        msg = result.choices[0].message
        raw = msg.content or ""
        
        print(
            f"[memory][debug] extraction response: content_len={len(raw)}, raw={raw[:150]!r}",
            flush=True,
        )
        
        start, end = raw.find("["), raw.rfind("]") + 1
        if 0 <= start < end:
            new_facts: list = json.loads(raw[start:end])
            if isinstance(new_facts, list):
                added = []
                with _lock:
                    for fact in new_facts:
                        if isinstance(fact, str) and fact.strip():
                            f = fact.strip()
                            if f not in FACTS:
                                FACTS.append(f)
                                added.append(f)
                if added:
                    print(f"[memory] +{len(added)} fact(s): {added}", flush=True)
                else:
                    print(f"[memory][debug] no new facts (all duplicates or empty)", flush=True)
                
                print(f"[memory] total facts ({len(FACTS)}): {FACTS}", flush=True)
            else:
                print(f"[memory][warn] extracted value is not a list: {type(new_facts).__name__}", flush=True)
        else:
            print(f"[memory][warn] no JSON array in response: {raw[:200]!r}", flush=True)
        
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
