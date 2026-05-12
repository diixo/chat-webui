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
import queue
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
_stop_event = threading.Event()
_EXTRACTION_QUEUE_MAXSIZE = 16
_extraction_queue: queue.Queue[tuple[str, str] | None] = queue.Queue(
    maxsize=_EXTRACTION_QUEUE_MAXSIZE
)

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


def _render_system_prompt(facts: list[str]) -> str:
    if not facts:
        return _BASE_PROMPT
    facts_block = "\n".join(f"- {f}" for f in facts)
    return f"{_BASE_PROMPT}\n\nKnown facts about the user:\n{facts_block}"



def _run_extraction(user_msg: str, assistant_reply: str, ui: UIServer) -> None:
    try:
        with _server.suppress_after_response():
            result = ui.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": _EXTRACTION_PROMPT.format(
                            user=user_msg,
                            assistant=assistant_reply,
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=256,
                enable_thinking=False,
                suppress_after_response=True,
            )
        raw = result.choices[0].message.content or "[]"
        start, end = raw.find("["), raw.rfind("]") + 1
        if 0 <= start < end:
            new_facts: list = json.loads(raw[start:end])
            if isinstance(new_facts, list):
                new_added = []
                with _lock:
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
                _server.ARGS.system_prompt = system_prompt
    except Exception as exc:
        print(f"[memory] extraction error: {exc}", flush=True)


def _extraction_worker(ui: UIServer) -> None:
    while not _stop_event.is_set():
        try:
            item = _extraction_queue.get(timeout=1.0)
            if item is None:
                return
            if _stop_event.is_set():
                return
            user_msg, assistant_reply = item
            _run_extraction(user_msg, assistant_reply, ui)
        except queue.Empty:
            continue
        except Exception:
            try:
                _extraction_queue.task_done()
            except ValueError:
                pass
        else:
            try:
                _extraction_queue.task_done()
            except ValueError:
                pass


def _enqueue_extraction(user_msg: str, assistant_reply: str) -> None:
    if _stop_event.is_set():
        return

    try:
        _extraction_queue.put_nowait((user_msg, assistant_reply))
        return
    except queue.Full:
        # Keep recent turns when extraction lags behind.
        pass

    try:
        _extraction_queue.get_nowait()
        _extraction_queue.task_done()
    except queue.Empty:
        pass

    try:
        _extraction_queue.put_nowait((user_msg, assistant_reply))
    except queue.Full:
        print("[memory] extraction queue is full; skipping turn", flush=True)


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

    worker_thread = threading.Thread(target=_extraction_worker, args=(ui,), daemon=False)
    worker_thread.start()

    _server.register_after_response(
        _enqueue_extraction
    )

    try:
        ui.run()
    except KeyboardInterrupt:
        print("\n[memory] shutting down gracefully...", flush=True)
        _stop_event.set()

        # Drop pending extraction work so shutdown is not delayed.
        while True:
            try:
                _extraction_queue.get_nowait()
                _extraction_queue.task_done()
            except queue.Empty:
                break

        _extraction_queue.put_nowait(None)  # Signal worker to stop
        worker_thread.join(timeout=5.0)  # Wait for worker to finish
        if worker_thread.is_alive():
            print("[memory] worker thread did not stop in time", flush=True)
