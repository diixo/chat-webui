#!/usr/bin/env python3
"""
server.py — HuggingFace Transformers inference server with OpenAI-compatible API.

Drop-in backend for the llama.cpp chat-webui frontend (no frontend changes needed).

Model resolution order for --model:
  1. Exact path if it exists (absolute or relative)
  2. models/<name>  inside the repo  (put your models in the models/ folder)
  3. HuggingFace Hub ID  (downloaded on first use to the HF cache)

Usage:
    # From HuggingFace Hub
    python server.py --model Qwen/Qwen2.5-7B-Instruct

    # From local models/ folder  (place model at models/my-model/)
    python server.py --model my-model

    # Explicit local path
    python server.py --model ./path/to/model --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


CONTEXT_LEN_DEFAULT = 1024

# Single source of truth for inference defaults. Used both as fallbacks when
# a request omits a sampling param, and as the values surfaced to the frontend
# via /props (default_generation_settings.params).
DEFAULT_GEN_PARAMS: dict[str, Any] = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 20,
    "repeat_penalty": 1.1,
    "max_tokens": CONTEXT_LEN_DEFAULT,
}

# ---------------------------------------------------------------------------
# Pluggable tooling registry
#
# Call register_tooling(obj) before starting the server.  obj must be an
# instance of AgentTooling (see agent_tooling.py) and implement:
#   classifier_prompt() -> str            — classification system prompt
#   normalize_result(dict) -> dict        — validates classifier output
#   execute(name, arguments) -> Any       — runs the named tool
# ---------------------------------------------------------------------------

from agent_tooling import AgentTooling

_tooling: AgentTooling | None = None


def register_tooling(obj: AgentTooling) -> None:
    """Register a tooling provider for server-side intent classification."""
    global _tooling
    if not isinstance(obj, AgentTooling):
        raise TypeError(
            f"register_tooling: expected AgentTooling subclass, got {type(obj).__name__}"
        )
    _tooling = obj


# ---------------------------------------------------------------------------
# After-response callback registry
#
# Call register_after_response(fn) to receive (user_msg, assistant_reply)
# after every completed chat turn. fn is called in a daemon thread so it
# must not rely on the request context but can safely call back into the
# server via UIServer.chat.completions.create().
# ---------------------------------------------------------------------------

_after_response: Callable[[str, str], None] | None = None
_AFTER_RESPONSE_REPLY_MAX_CHARS = 16000


def register_after_response(fn: Callable[[str, str], None]) -> None:
    """Register a callback invoked after each chat completion.

    fn(user_msg: str, assistant_reply: str) is called in a daemon thread
    after the response has been fully sent to the client.
    """
    global _after_response
    _after_response = fn


def _fire_after_response(messages: list[dict], reply: str) -> None:
    """Extract the last user message and fire the callback in a daemon thread."""
    if _after_response is None or not reply:
        return
    reply = reply[:_AFTER_RESPONSE_REPLY_MAX_CHARS]
    user_msg: Any = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    if isinstance(user_msg, list):
        user_msg = " ".join(
            p.get("text", "") for p in user_msg
            if isinstance(p, dict) and p.get("type") == "text"
        )
    if user_msg:
        threading.Thread(target=_after_response, args=(user_msg, reply), daemon=True).start()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HuggingFace Transformers Chat Server")
    p.add_argument("--model", "-m", required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", "-p", type=int, default=8080)
    p.add_argument("--context-size", "-c", type=int, default=CONTEXT_LEN_DEFAULT,
                   help=f"Max token context window (default: {CONTEXT_LEN_DEFAULT})")
    p.add_argument("--device", default="auto",
                   help="Device map: auto | cpu | cuda | mps (default: auto)")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"],
                   help="Model weight dtype (default: auto)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow executing remote model code")
    p.add_argument("--api-key", default=None,
                   help="Optional Bearer token; if set, all requests must supply it")
    p.add_argument("--public-dir", default=None,
                   help="Path to web-UI public dir "
                        "(default: <this file's dir>/server/public)")
    p.add_argument("--max-history", type=int, default=0,
                   help="Max number of past messages to send to the model "
                        "(0 = unlimited, default: 0). System message is always kept.")
    p.add_argument("--prompt-style", default="auto",
                   choices=["auto", "raw"],
                   help="Prompt formatting: auto = use chat_template (instruct models); "
                        "raw = concatenate messages with newlines (base models like GPT-2). "
                        "Default: auto")
    return p.parse_args()


# Module-level ARGS with safe defaults — overridden by _parse_args() when run
# as a CLI script, or by configure() when imported as a library.
ARGS = argparse.Namespace(
    model="",
    host="127.0.0.1",
    port=8080,
    context_size=CONTEXT_LEN_DEFAULT,
    device="auto",
    dtype="auto",
    trust_remote_code=False,
    api_key=None,
    public_dir=None,
    max_history=0,
    system_prompt=None,
    prompt_style="auto",
    enable_thinking=None,  # None = use model template default
)


def configure(
    model: str,
    *,
    context_size: int = CONTEXT_LEN_DEFAULT,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    api_key: str | None = None,
    max_history: int = 0,
    system_prompt: str | None = None,
    prompt_style: str = "auto",
    enable_thinking: bool | None = None,
) -> None:
    """Configure the server when used as a library (instead of CLI args).

    Call this before load_model():
        import server
        server.configure(model="Qwen3-0.6B", system_prompt="You are a helpful assistant.")
        server.load_model()

    prompt_style:
        "auto" — use tokenizer's chat_template if available, else ChatML fallback
        "raw"  — concatenate message contents with newlines (no role markers).
                 Use for base models like GPT-2 that don't understand chat formatting.
    """
    global ARGS
    ARGS.model = model
    ARGS.context_size = context_size
    ARGS.device = device
    ARGS.dtype = dtype
    ARGS.trust_remote_code = trust_remote_code
    ARGS.api_key = api_key
    ARGS.max_history = max_history
    ARGS.system_prompt = system_prompt
    ARGS.prompt_style = prompt_style
    ARGS.enable_thinking = enable_thinking

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Model state  (populated in main() before uvicorn starts)
# ---------------------------------------------------------------------------

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None
_model_device: str = "cpu"

# Serialize concurrent generation requests to one at a time.
# We intentionally keep max_workers=1 here.
_gen_executor: Any = None   # set in main() to avoid import-time side effects

# Public dir for static web-UI files (set in main())
_public_dir: Path | None = None

# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

# Repo root = directory that contains this script
_REPO_ROOT = Path(__file__).parent
_MODELS_DIR = _REPO_ROOT / "models"


def _resolve_model_path(name: str) -> str:
    """
    Resolve --model to an actual path or Hub ID.

    Priority:
      1. name is an existing path (absolute or relative to cwd)
      2. models/<name> exists inside the repo
      3. Return name as-is (treated as a HuggingFace Hub ID)

    If the name looks like a file-system path (starts with . / \\ or is
    absolute) but does not exist, the server exits with a clear error instead
    of passing the path to the Hub loader.
    """
    p = Path(name)

    # Detect explicit path notation: ./foo, ../foo, /foo, C:\foo, .\foo
    looks_like_path = (
        p.is_absolute()
        or name.startswith("./")
        or name.startswith(".\\")
        or name.startswith("../")
        or name.startswith("..\\")
    )

    # 1. Direct path
    if p.exists():
        resolved = str(p.resolve())
        log.info("Using local model path: %s", resolved)
        return resolved

    # If it looks like a path but doesn't exist, fail clearly
    if looks_like_path:
        raise SystemExit(
            f"ERROR: model path not found: {p.resolve()}\n"
            f"       Put the model in models/{p.name}/ or pass a Hub ID."
        )

    # 2. models/<name> inside repo
    candidate = _MODELS_DIR / name
    if candidate.exists():
        resolved = str(candidate.resolve())
        log.info("Found model in models/ folder: %s", resolved)
        return resolved

    # 3. Hub ID
    log.info("Model not found locally — will load from HuggingFace Hub: %s", name)
    return name


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> None:
    global _tokenizer, _model, _model_device

    model_path = _resolve_model_path(ARGS.model)

    log.info("Loading tokenizer: %s", model_path)
    _tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=ARGS.trust_remote_code,
    )
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    dtype_map: dict[str, Any] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }
    torch_dtype = dtype_map[ARGS.dtype]

    log.info(
        "Loading model: %s  (dtype=%s, device=%s)",
        model_path, ARGS.dtype, ARGS.device,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=ARGS.device,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=ARGS.trust_remote_code,
    )
    _model.eval()

    first_param = next(_model.parameters(), None)
    _model_device = str(first_param.device) if first_param is not None else "cpu"
    log.info("Model ready on %s", _model_device)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Transformers Chat Server", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(request: Request) -> None:
    if ARGS.api_key is None:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != ARGS.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _assert_ready() -> None:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

# ---------------------------------------------------------------------------
# Message / prompt helpers
# ---------------------------------------------------------------------------

def _content_to_str(content: Any) -> str:
    """Extract plain text from an OpenAI message content (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(t for t in texts if t)
    return str(content) if content else ""


def _normalise_messages(raw: list[dict]) -> list[dict]:
    """Return [{role, content: str}, …] ready for apply_chat_template.

    Handles standard roles (system, user, assistant) plus:
    - role "tool": tool execution results, rendered as a user-side message.
    - assistant messages that carry tool_calls (no content): rendered as JSON.
    """
    out: list[dict] = []
    for m in raw:
        role = m.get("role", "user")

        # Tool result → render as a user-visible turn so the model sees the output
        if role == "tool":
            tool_call_id = m.get("tool_call_id", "")
            content = _content_to_str(m.get("content", ""))
            out.append({
                "role": "user",
                "content": f"[Tool result for call {tool_call_id}]: {content}",
            })
            continue

        # Assistant turn may have tool_calls instead of (or in addition to) content
        if role == "assistant" and m.get("tool_calls"):
            calls_json = json.dumps(m["tool_calls"], ensure_ascii=False)
            out.append({"role": "assistant", "content": calls_json})
            continue

        text = _content_to_str(m.get("content", ""))
        if text or role == "assistant":
            out.append({"role": role, "content": text})

    if ARGS.max_history > 0:
        system = [m for m in out if m["role"] == "system"]
        rest = [m for m in out if m["role"] != "system"]
        out = system + rest[-ARGS.max_history:]

    return out


def _build_prompt(messages: list[dict], enable_thinking: bool | None = None) -> str:
    """Apply the tokenizer's chat template, or fall back to ChatML-like format.

    For base models without a chat template (e.g. GPT-2), use prompt_style="raw"
    to skip role markers and just concatenate message contents.

    enable_thinking: when explicitly False, passes enable_thinking=False to
    apply_chat_template (suppresses <think> blocks on models like Qwen3).
    None leaves the template default unchanged.
    """
    if ARGS.prompt_style == "raw":
        return "\n".join(m["content"] for m in messages) + "\n"

    if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
        try:
            kwargs: dict = {}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            return _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except Exception:
            log.warning("apply_chat_template failed — using fallback:\n%s",
                        traceback.format_exc())

    # Fallback: simple ChatML-style format
    lines = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    lines.append("<|assistant|>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tool calling helpers (OpenAI-compatible)
# ---------------------------------------------------------------------------

def _build_tools_system_injection(tools: list[dict]) -> str:
    """Build a system-prompt snippet that describes tools to the model.

    The model is instructed to respond with a specific JSON format when it
    wants to call a tool, so we can reliably detect and parse that call.
    """
    tool_descriptions: list[str] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        fn = tool["function"]
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_lines = []
        for pname, pschema in props.items():
            req_marker = " (required)" if pname in required else ""
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            param_lines.append(f"    - {pname} ({ptype}{req_marker}): {pdesc}")

        params_text = "\n".join(param_lines) if param_lines else "    (no parameters)"
        tool_descriptions.append(
            f"  name: {name}\n  description: {desc}\n  parameters:\n{params_text}"
        )

    tools_block = "\n\n".join(tool_descriptions)
    return (
        "You have access to the following tools.\n\n"
        f"{tools_block}\n\n"
        "If you need to call a tool, respond with ONLY a JSON object in this exact format "
        "(no extra text before or after):\n"
        '{"tool_call": {"name": "<tool_name>", "arguments": {<arguments_as_object>}}}\n\n'
        "If no tool call is needed, respond normally."
    )


def _parse_tool_call(text: str, tools: list[dict]) -> dict | None:
    """Try to extract a tool_call JSON from model output.

    Returns a dict with keys 'name' and 'arguments' (dict), or None if the
    output is not a tool call.
    """
    text = text.strip()
    # Quick rejection — must look like a JSON object
    if not text.startswith("{"):
        return None

    try:
        # Grab the first JSON object even if there's trailing text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    tc = obj.get("tool_call")
    if not isinstance(tc, dict):
        return None

    name = tc.get("name", "")
    arguments = tc.get("arguments", {})

    # Validate: name must match one of the declared tools
    known_names = {
        t["function"]["name"]
        for t in tools
        if t.get("type") == "function" and "function" in t
    }
    if name not in known_names:
        return None

    if not isinstance(arguments, dict):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}

    return {"name": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Reasoning ("thinking") parser
#
# Models like Qwen3 emit a thinking section wrapped in literal <think>…</think>
# tags before the actual answer. The OpenAI-compatible streaming protocol
# carries this in a separate `delta.reasoning_content` field instead of
# `delta.content`. The frontend already understands that field — we just need
# to split the stream as tokens come in.
#
# Tags can split across token boundaries (e.g. one chunk ends with "<thi" and
# the next starts with "nk>"). The parser holds back any tail of the buffer
# that could still grow into a full tag, so partial matches never leak.
# ---------------------------------------------------------------------------

class _ThinkingParser:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_thinking = False

    @staticmethod
    def _split_at_partial_tag(buf: str, tag: str) -> tuple[str, str]:
        """Return (safe_to_emit, hold_back).

        hold_back is the longest non-empty suffix of buf that is also a prefix
        of tag — i.e. text that might still grow into a full tag. Empty if
        nothing in the tail looks like a tag start.
        """
        max_k = min(len(tag) - 1, len(buf))
        for k in range(max_k, 0, -1):
            if buf.endswith(tag[:k]):
                return buf[:-k], buf[-k:]
        return buf, ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Feed a chunk. Returns list of (kind, text) where kind is
        'content' or 'reasoning'. Kind matches the current state."""
        self._buf += text
        out: list[tuple[str, str]] = []
        while True:
            tag = self.CLOSE if self._in_thinking else self.OPEN
            kind = "reasoning" if self._in_thinking else "content"
            i = self._buf.find(tag)
            if i != -1:
                if i > 0:
                    out.append((kind, self._buf[:i]))
                self._buf = self._buf[i + len(tag):]
                self._in_thinking = not self._in_thinking
                continue
            safe, hold = self._split_at_partial_tag(self._buf, tag)
            if safe:
                out.append((kind, safe))
            self._buf = hold
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        """End-of-stream — emit whatever is left in the buffer."""
        if not self._buf:
            return []
        kind = "reasoning" if self._in_thinking else "content"
        out = [(kind, self._buf)]
        self._buf = ""
        return out


def _model_id() -> str:
    """Return a short display name for the model (last path segment or Hub name)."""
    name = ARGS.model.rstrip("/\\")
    # For Hub IDs like "org/model-name" keep the full "org/model-name"
    parts = name.replace("\\", "/").split("/")
    if len(parts) >= 2 and (Path(parts[0]).exists() or parts[0] in (".", "..")):
        # It's a file path — return only the last segment
        return parts[-1]
    # Hub ID or bare name — return as-is
    return name

# ---------------------------------------------------------------------------
# Async streaming helpers
# ---------------------------------------------------------------------------

async def _iter_streamer(streamer: TextIteratorStreamer) -> AsyncGenerator[str, None]:
    """
    Bridge a sync TextIteratorStreamer to an async generator without blocking
    the event loop.

    Uses the *default* thread-pool executor (not _gen_executor) so it can run
    concurrently with the generation thread that fills the streamer.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    sentinel: str | None = None

    def _drain() -> None:
        try:
            for token in streamer:
                loop.call_soon_threadsafe(queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    # Default executor — separate from _gen_executor, avoids deadlock.
    loop.run_in_executor(None, _drain)

    while True:
        token = await queue.get()
        if token is sentinel:
            break
        yield token


async def stream_completion(
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    parse_reasoning: bool = True,
    enable_thinking: bool | None = None,
) -> AsyncGenerator[str, None]:
    """
    Yield SSE-formatted lines for a streaming chat completion.

    Format matches the llama.cpp / OpenAI streaming API so the web-UI
    frontend works unchanged.
    """
    prompt = _build_prompt(messages, enable_thinking=enable_thinking)
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=ARGS.context_size,
    )
    prompt_tokens: int = inputs["input_ids"].shape[1]

    if prompt_tokens >= ARGS.context_size:
        err = {
            "error": {
                "code": 400,
                "message": "Input exceeds the context size.",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": prompt_tokens,
                "n_ctx": ARGS.context_size,
            }
        }
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
        return

    inputs_on_device = {k: v.to(_model_device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs: dict[str, Any] = {
        **inputs_on_device,
        "streamer": streamer,
        "max_new_tokens": min(max_new_tokens, ARGS.context_size - prompt_tokens),
        "do_sample": temperature > 1e-4,
        "temperature": max(temperature, 1e-7),
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if top_k > 0:
        gen_kwargs["top_k"] = top_k

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    mid = _model_id()

    def _chunk(delta: dict, finish_reason: str | None = None,
               timings: dict | None = None) -> str:
        obj: dict[str, Any] = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": mid,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if timings:
            obj["timings"] = timings
        return f"data: {json.dumps(obj)}\n\n"

    # Announce role
    yield _chunk({"role": "assistant"})

    t_start = time.perf_counter()

    # Launch generation in the serialising executor.
    # _drain (inside _iter_streamer) uses the *default* executor → no deadlock.
    loop = asyncio.get_event_loop()
    gen_future = loop.run_in_executor(
        _gen_executor,
        lambda: _model.generate(**gen_kwargs),
    )

    parser = _ThinkingParser() if parse_reasoning else None

    def _emit_parsed(parts: list[tuple[str, str]]) -> list[str]:
        return [
            _chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text})
            for kind, text in parts
        ]

    predicted_n = 0
    reply_parts: list[str] = []
    reply_chars_left = _AFTER_RESPONSE_REPLY_MAX_CHARS

    def _capture_reply(text: str) -> None:
        nonlocal reply_chars_left
        if not text or reply_chars_left <= 0:
            return
        piece = text[:reply_chars_left]
        if piece:
            reply_parts.append(piece)
            reply_chars_left -= len(piece)

    async for token in _iter_streamer(streamer):
        if not token:
            continue
        predicted_n += 1
        if parser is None:
            yield _chunk({"content": token})
            _capture_reply(token)
        else:
            parts = parser.feed(token)
            for sse in _emit_parsed(parts):
                yield sse
            for kind, text in parts:
                if kind == "content":
                    _capture_reply(text)

    if parser is not None:
        flush_parts = parser.flush()
        for sse in _emit_parsed(flush_parts):
            yield sse
        for kind, text in flush_parts:
            if kind == "content":
                _capture_reply(text)

    # Drain any remaining exception from the generation thread
    try:
        await asyncio.wait_for(asyncio.shield(gen_future), timeout=10.0)
    except (asyncio.TimeoutError, Exception) as exc:
        log.error("Generation thread error: %s", exc)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    yield _chunk(
        {},
        finish_reason="stop",
        timings={
            "prompt_n": prompt_tokens,
            "prompt_ms": elapsed_ms,
            "predicted_n": predicted_n,
            "predicted_ms": elapsed_ms,
        },
    )
    yield "data: [DONE]\n\n"
    _fire_after_response(messages, "".join(reply_parts))

# ---------------------------------------------------------------------------
# Intent classification + tool dispatch
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Return the first balanced JSON object found in text."""
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


def _build_classification_prompt(messages: list[dict]) -> str:
    """Like _build_prompt but passes enable_thinking=False for Qwen3."""
    if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
        try:
            return _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            pass
        try:
            return _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    lines = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    lines.append("<|assistant|>")
    return "\n".join(lines) + "\n"


def _classify_intent_sync(user_text: str) -> dict:
    """Classify intent using the loaded model.

    Called inside _gen_executor — never call from the event loop thread directly.
    Returns a normalised dict: {intent, tracker_type, description}.
    """
    try:
        messages = [
            {"role": "system", "content": _tooling.classifier_prompt()},
            {"role": "user", "content": user_text},
        ]
        prompt = _build_classification_prompt(messages)
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(ARGS.context_size, CONTEXT_LEN_DEFAULT),
        )
        input_ids = inputs["input_ids"].to(_model_device)

        with torch.no_grad():
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=_tokenizer.pad_token_id,
                eos_token_id=_tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1] :]
        raw = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        log.debug("Classification raw output: %r", raw)

        parsed = json.loads(_extract_json(raw))
        return _tooling.normalize_result(parsed)
    except Exception as exc:
        log.debug("Intent classification failed: %s", exc)
        return {"intent": "unknown", "tracker_type": "", "description": ""}


async def _stream_tool_response(tool_result: dict) -> AsyncGenerator[str, None]:
    """Stream a synthetic SSE response after a tool function is dispatched."""
    intent = tool_result["intent"]
    params = {k: v for k, v in tool_result.items() if k != "intent"}

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _tooling.execute(intent, params)
        )
        if isinstance(result, dict):
            content = result.get("message") or result.get("text") or ""
            if not content:
                info = {k: v for k, v in result.items() if v not in (None, "", [])}
                content = "\n".join(f"**{k}**: {v}" for k, v in info.items())
        else:
            content = str(result) if result is not None else f"✓ `{intent}` executed"
    except Exception as exc:
        content = f"⚠ `{intent}` failed: {exc}"

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    mid = _model_id()

    def _chunk(delta: dict, finish_reason: str | None = None) -> str:
        obj: dict[str, Any] = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": mid,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(obj)}\n\n"

    yield _chunk({"role": "assistant"})
    yield _chunk({"content": content})
    yield _chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"

# ---------------------------------------------------------------------------
# /props  — server capabilities queried by the frontend on startup
# ---------------------------------------------------------------------------

@app.get("/props")
async def get_props(request: Request) -> JSONResponse:
    _check_auth(request)
    chat_template = (
        getattr(_tokenizer, "chat_template", "") or ""
        if _tokenizer else ""
    )
    return JSONResponse({
        "default_generation_settings": {
            "id": 0,
            "id_task": -1,
            "n_ctx": ARGS.context_size,
            "speculative": False,
            "is_processing": False,
            "params": {
                "n_predict": DEFAULT_GEN_PARAMS["max_tokens"],
                "seed": -1,
                "temperature": DEFAULT_GEN_PARAMS["temperature"],
                "dynatemp_range": 0.0,
                "dynatemp_exponent": 1.0,
                "top_k": DEFAULT_GEN_PARAMS["top_k"],
                "top_p": DEFAULT_GEN_PARAMS["top_p"],
                "min_p": 0.05,
                "top_n_sigma": 0.0,
                "xtc_probability": 0.0,
                "xtc_threshold": 0.1,
                "typ_p": 1.0,
                "repeat_last_n": 64,
                "repeat_penalty": DEFAULT_GEN_PARAMS["repeat_penalty"],
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "dry_multiplier": 0.0,
                "dry_base": 1.75,
                "dry_allowed_length": 2,
                "dry_penalty_last_n": -1,
                "dry_sequence_breakers": ["\n", ":", "\"", "*"],
                "mirostat": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "stop": [],
                "max_tokens": DEFAULT_GEN_PARAMS["max_tokens"],
                "n_keep": 0,
                "n_discard": 0,
                "ignore_eos": False,
                "stream": True,
                "logit_bias": [],
                "n_probs": 0,
                "min_keep": 0,
                "grammar": "",
                "grammar_lazy": False,
                "grammar_triggers": [],
                "preserved_tokens": [],
                "chat_format": "chatml",
                "reasoning_format": "none",
                "reasoning_in_content": False,
                "generation_prompt": "",
                "samplers": ["top_k", "top_p", "min_p", "temperature"],
                "backend_sampling": False,
                "speculative.n_max": 16,
                "speculative.n_min": 0,
                "speculative.p_min": 0.9,
                "timings_per_token": False,
                "post_sampling_probs": False,
                "lora": [],
            },
            "prompt": "",
            "next_token": {
                "has_next_token": True,
                "has_new_line": False,
                "n_remain": -1,
                "n_decoded": 0,
                "stopping_word": "",
            },
        },
        "total_slots": 1,
        "model_path": ARGS.model,
        "role": "model",  # single-model mode (not "router")
        "modalities": {"vision": False, "audio": False},
        "chat_template": chat_template,
        "bos_token": getattr(_tokenizer, "bos_token", "") or "" if _tokenizer else "",
        "eos_token": getattr(_tokenizer, "eos_token", "") or "" if _tokenizer else "",
        "build_info": "transformers-backend",
    })

# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    _check_auth(request)
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": _model_id(),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "transformers",
            }
        ],
    })

# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    _check_auth(request)
    _assert_ready()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    raw_messages: list[dict] = body.get("messages", [])
    tools: list[dict] = body.get("tools", [])
    tool_choice = body.get("tool_choice", "auto")  # "auto" | "none" | {"type":"function",...}

    # Inject server-level system_prompt only when chat-template formatting is in
    # use. For prompt_style="raw" (base models like GPT-2) there are no role
    # markers, so a system message would just become a stray text prefix that
    # the model continues — useless or actively harmful. getattr guards CLI
    # mode where _parse_args() doesn't define this flag.
    sys_prompt = (
        None if ARGS.prompt_style == "raw"
        else getattr(ARGS, "system_prompt", None)
    )
    if sys_prompt and not any(m.get("role") == "system" for m in raw_messages):
        raw_messages = [{"role": "system", "content": sys_prompt}] + raw_messages

    # If tools are declared and not suppressed, inject a system prompt describing them
    use_tools = bool(tools) and tool_choice != "none"
    if use_tools:
        tools_injection = _build_tools_system_injection(tools)
        # Prepend to existing system message or add a new one
        existing_system = next((m for m in raw_messages if m.get("role") == "system"), None)
        if existing_system:
            existing_system = dict(existing_system)
            existing_system["content"] = tools_injection + "\n\n" + _content_to_str(existing_system.get("content", ""))
            raw_messages = [existing_system] + [m for m in raw_messages if m.get("role") != "system"]
        else:
            raw_messages = [{"role": "system", "content": tools_injection}] + raw_messages

    messages = _normalise_messages(raw_messages)
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    do_stream: bool = bool(body.get("stream", True))
    max_tokens: int = int(body.get("max_tokens") or body.get("n_predict") or DEFAULT_GEN_PARAMS["max_tokens"])
    temperature: float = float(body.get("temperature", DEFAULT_GEN_PARAMS["temperature"]))
    top_p: float = float(body.get("top_p", DEFAULT_GEN_PARAMS["top_p"]))
    top_k: int = int(body.get("top_k", DEFAULT_GEN_PARAMS["top_k"]))
    rep_penalty: float = float(body.get("repeat_penalty", DEFAULT_GEN_PARAMS["repeat_penalty"]))
    # Frontend sends "auto" by default; "none" means render reasoning inline
    # as part of the regular content (useful for debugging).
    parse_reasoning: bool = body.get("reasoning_format", "auto") != "none"
    # Request body takes precedence; fall back to server-level default from configure().
    enable_thinking: bool | None = body.get("enable_thinking", getattr(ARGS, "enable_thinking", None))

    # Warmup / pre-encode request (n_predict=0 from ChatService.preEncode)
    if max_tokens == 0:
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_id(),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
        })

    # Intent classification — runs before LLM generation when no tools are in the request.
    # Requires a tooling provider registered via register_tooling().
    if _tooling is not None and not use_tools:
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), None
        )
        if last_user_msg:
            loop = asyncio.get_event_loop()
            tool_result = await loop.run_in_executor(
                _gen_executor, _classify_intent_sync, last_user_msg
            )
            log.info(
                "Intent: %s | params: %s",
                tool_result["intent"],
                {k: v for k, v in tool_result.items() if k != "intent"},
            )
            if tool_result["intent"] != "unknown":
                return StreamingResponse(
                    _stream_tool_response(tool_result),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

    # Streaming path — tool calling requires non-streaming to parse the full output
    if do_stream and not use_tools:
        return StreamingResponse(
            stream_completion(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=rep_penalty,
                parse_reasoning=parse_reasoning,
                enable_thinking=enable_thinking,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path (always used when tools are present)
    prompt = _build_prompt(messages, enable_thinking=enable_thinking)
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=ARGS.context_size,
    )
    prompt_tokens: int = inputs["input_ids"].shape[1]

    if prompt_tokens >= ARGS.context_size:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": 400,
                    "message": "Input exceeds context size.",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": prompt_tokens,
                    "n_ctx": ARGS.context_size,
                }
            },
        )

    inputs_on_device = {k: v.to(_model_device) for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {
        **inputs_on_device,
        "max_new_tokens": min(max_tokens, ARGS.context_size - prompt_tokens),
        "do_sample": temperature > 1e-4,
        "temperature": max(temperature, 1e-7),
        "top_p": top_p,
        "repetition_penalty": rep_penalty,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if top_k > 0:
        gen_kwargs["top_k"] = top_k

    loop = asyncio.get_event_loop()
    t_start = time.perf_counter()
    output_ids = await loop.run_in_executor(
        _gen_executor,
        lambda: _model.generate(**gen_kwargs),
    )
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    generated_ids = output_ids[0][prompt_tokens:]
    text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    predicted_n = len(generated_ids)

    # --- Tool call detection ---
    if use_tools:
        tool_call = _parse_tool_call(text, tools)
        if tool_call is not None:
            log.info("Tool call detected: %s(%s)", tool_call["name"], tool_call["arguments"])
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            return JSONResponse({
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": _model_id(),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False),
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": predicted_n,
                    "total_tokens": prompt_tokens + predicted_n,
                },
            })

    # Split <think>…</think> out of the regular content for non-streaming
    # responses too. The frontend reads message.reasoning_content for both
    # streaming and non-streaming paths.
    content_text = text
    reasoning_text = ""
    if parse_reasoning:
        parser = _ThinkingParser()
        parts = parser.feed(text) + parser.flush()
        content_text = "".join(t for k, t in parts if k == "content")
        reasoning_text = "".join(t for k, t in parts if k == "reasoning")

    message: dict[str, Any] = {"role": "assistant", "content": content_text}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text

    _fire_after_response(messages, content_text)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id(),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": predicted_n,
            "total_tokens": prompt_tokens + predicted_n,
        },
        "timings": {
            "prompt_n": prompt_tokens,
            "prompt_ms": elapsed_ms,
            "predicted_n": predicted_n,
            "predicted_ms": elapsed_ms,
        },
    })

# ---------------------------------------------------------------------------
# POST /tokenize
# ---------------------------------------------------------------------------

@app.post("/tokenize")
async def tokenize_endpoint(request: Request) -> JSONResponse:
    _check_auth(request)
    _assert_ready()
    body = await request.json()
    text: str = body.get("content", "")
    tokens: list[int] = _tokenizer.encode(text, add_special_tokens=False)
    return JSONResponse({"tokens": tokens})

# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    status = "ok" if _model is not None else "loading"
    return JSONResponse({"status": status})

# ---------------------------------------------------------------------------
# HEAD /cors-proxy  — tell the frontend this proxy is not available
#
# The web-UI's mcpStore.probeProxy() checks `response.status !== 404`.
# A missing route returns 405, which incorrectly enables proxy features.
# Returning 404 here correctly disables them.
# ---------------------------------------------------------------------------

@app.head("/cors-proxy")
async def cors_proxy_head() -> Any:
    raise HTTPException(status_code=404)

# ---------------------------------------------------------------------------
# Static web-UI file serving
#
# These routes are defined LAST so that all API routes above take priority
# when FastAPI matches requests in registration order.
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index() -> Any:
    if _public_dir:
        return FileResponse(str(_public_dir / "index.html"))
    return JSONResponse({"status": "ok", "ui": "not configured"})


@app.get("/{path:path}")
async def serve_static_or_spa(path: str) -> Any:
    """Serve static assets; fall back to index.html for SPA client-side routes."""
    if _public_dir:
        candidate = _public_dir / path
        if candidate.is_file():
            return FileResponse(str(candidate))
        # Paths without a file extension are SPA routes → return index.html
        if "." not in Path(path).name:
            return FileResponse(str(_public_dir / "index.html"))
    raise HTTPException(status_code=404)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    from concurrent.futures import ThreadPoolExecutor

    global _gen_executor, _public_dir, ARGS

    # Parse CLI args and apply them (overrides library defaults)
    ARGS = _parse_args()

    # Serialise concurrent generation to prevent OOM on small GPUs
    _gen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")

    # Resolve web-UI public directory
    if ARGS.public_dir:
        p = Path(ARGS.public_dir)
        _public_dir = p if p.is_dir() else None
        if _public_dir is None:
            log.warning("--public-dir %s not found", ARGS.public_dir)
    else:
        candidate = Path(__file__).parent / "server" / "public"
        _public_dir = candidate if candidate.is_dir() else None
        if _public_dir is None:
            log.warning(
                "Web UI not found at %s — API-only mode. "
                "Pass --public-dir to enable the UI.",
                candidate,
            )

    # Load model synchronously before the server starts accepting requests
    load_model()

    if _tooling is not None:
        log.info("Tooling enabled (provider: %s)", type(_tooling).__name__)
    else:
        log.info("No tooling registered — running in plain chat mode")

    log.info("Starting server at http://%s:%d", ARGS.host, ARGS.port)

    uvicorn.run(
        app,
        host=ARGS.host,
        port=ARGS.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
