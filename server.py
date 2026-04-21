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
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer



from dotenv import load_dotenv
import os
from pybeamer import Codebeamer


load_dotenv()

codebeamer_url = os.getenv('CODEBEAMER_URL')
codebeamer_username = os.getenv('CODEBEAMER_USERNAME')
codebeamer_password = os.getenv('CODEBEAMER_PASSWORD')
codebeamer_api_root = os.getenv('CODEBEAMER_API_ROOT', 'api/v3')


if not all([codebeamer_url, codebeamer_username, codebeamer_password]):
    raise ValueError(
        'Missing required environment variables: '
        'CODEBEAMER_URL, CODEBEAMER_USERNAME, CODEBEAMER_PASSWORD'
    )

codebeamer = Codebeamer(
    url=codebeamer_url,
    username=codebeamer_username,
    password=codebeamer_password,
    api_root=codebeamer_api_root
)

user = codebeamer.get_user(codebeamer_username)

project_id = int(os.getenv('CODEBEAMER_PROJECT_ID', '25'))

projects = codebeamer.get_projects()
projects_map = {project.id: project for project in projects}

if project_id not in projects_map:
    known_ids = ', '.join(str(pid) for pid in sorted(projects_map.keys())[:20])
    raise ValueError(f'Project with id={project_id} not found in fetched projects. Known IDs (first 20): {known_ids}')

project = projects_map[project_id]

type2id = codebeamer.get_tracker_types()
names_ui2api = {
    "Area": "Area",
    "Bug": "Bug",
    "Change Request": "Change Request",
    "Component": "Component",
    "Configuration Item": "Configuration Item",
    "Contact": "Contact",
    "Document": "Document",
    "Epic": "Epic",
    "Issue": "Issue",
    "Platform": "Platform",
    "Release": "Release",
    "Requirement": "Requirement",
    "Risk": "Risk",
    "RPE Report": "RPE Report",
    "Task": "Task",
    "Team": "Team",
    "Test Case": "Testcase",
    "Test Configuration": "Testconf",
    "Test Run": "Testrun",
    "Test Set": "Testset",
    "Time Recording": "Work Log",
    "User Story": "User Story",
}
ui2id = dict()

for name_ui, name_api in names_ui2api.items():
    ui2id[name_ui] = type2id[name_api]


print(f"ui2id: {len(ui2id)}, names: {len(names_ui2api)}")

assert len(ui2id) == len(names_ui2api), "Expected to find all types, but only found"

# ---------------------------------------------------------------------------
# Optional agent tooling (agent_tooling.py must be in the same directory)
# ---------------------------------------------------------------------------

try:
    from agent_tooling import (
        SYSTEM_PROMPT as _TOOLING_SYSTEM_PROMPT,
        FUNCTIONS_MAP as _TOOLING_FUNCTIONS_MAP,
        extract_first_json_object as _extract_json,
        normalize_result as _normalize_tooling_result,
        init_codebeamer as _init_codebeamer,
    )
    _init_codebeamer(project, user, ui2id)
    _TOOLING_AVAILABLE = True
except ImportError:
    _TOOLING_AVAILABLE = False

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HuggingFace Transformers Chat Server")
    p.add_argument("--model", "-m", required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", "-p", type=int, default=8080)
    p.add_argument("--context-size", "-c", type=int, default=4096,
                   help="Max token context window (default: 4096)")
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
    return p.parse_args()


ARGS = _parse_args()

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
        dtype=torch_dtype,
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
    """Return [{role, content: str}, …] ready for apply_chat_template."""
    out: list[dict] = []
    for m in raw:
        role = m.get("role", "user")
        text = _content_to_str(m.get("content", ""))
        if text or role == "assistant":
            out.append({"role": role, "content": text})

    if ARGS.max_history > 0:
        system = [m for m in out if m["role"] == "system"]
        rest = [m for m in out if m["role"] != "system"]
        out = system + rest[-ARGS.max_history:]

    return out


def _build_prompt(messages: list[dict]) -> str:
    """Apply the tokenizer's chat template, or fall back to ChatML-like format."""
    if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
        try:
            return _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            log.warning("apply_chat_template failed — using fallback:\n%s",
                        traceback.format_exc())

    # Fallback: simple ChatML-style format
    lines = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
    lines.append("<|assistant|>")
    return "\n".join(lines) + "\n"


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
) -> AsyncGenerator[str, None]:
    """
    Yield SSE-formatted lines for a streaming chat completion.

    Format matches the llama.cpp / OpenAI streaming API so the web-UI
    frontend works unchanged.
    """
    prompt = _build_prompt(messages)
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

    predicted_n = 0
    async for token in _iter_streamer(streamer):
        if token:
            predicted_n += 1
            yield _chunk({"content": token})

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

# ---------------------------------------------------------------------------
# Intent classification + tool dispatch
# ---------------------------------------------------------------------------

def _build_classification_prompt(messages: list[dict]) -> str:
    """Like _build_prompt but passes enable_thinking=False for Qwen3."""
    if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
        # Try with enable_thinking=False (Qwen3 specific)
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
    """
    Classify intent using the loaded model.
    Called inside _gen_executor — never call from the event loop thread directly.
    """
    try:
        messages = [
            {"role": "system", "content": _TOOLING_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt = _build_classification_prompt(messages)
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(ARGS.context_size, 2048),
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

        new_tokens = output_ids[0][input_ids.shape[1]:]
        raw = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        log.debug("Classification raw output: %r", raw)

        json_text = _extract_json(raw)
        parsed = json.loads(json_text)
        return _normalize_tooling_result(parsed)
    except Exception as exc:
        log.debug("Intent classification failed: %s", exc)
        return {"intent": "unknown", "description": ""}


async def _stream_tool_response(tool_result: dict) -> AsyncGenerator[str, None]:
    """Stream a synthetic SSE response after a tool function is dispatched."""
    intent = tool_result["intent"]
    description = tool_result.get("description", "")
    tracker_type = tool_result.get("tracker_type", "")

    fn = _TOOLING_FUNCTIONS_MAP.get(intent)
    if fn:
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, fn, description, tracker_type)
            content = f'✓ `{tracker_type}`'
            if isinstance(result, dict):
                item_id = result.get("id")
                url = result.get("url")
                if item_id is not None:
                    content += f' #{item_id}'
                if description:
                    content += f': "{description}"'
                if url:
                    content += f'\n{url}'
            else:
                if result is not None:
                    content += f' #{result}'
                if description:
                    content += f': "{description}"'
        except Exception as exc:
            content = f'⚠ `{intent}` failed: {exc}'
    else:
        content = f'⚠ No handler registered for intent `{intent}`'

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
                "n_predict": 2048,
                "seed": -1,
                "temperature": 0.7,
                "dynatemp_range": 0.0,
                "dynatemp_exponent": 1.0,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "top_n_sigma": 0.0,
                "xtc_probability": 0.0,
                "xtc_threshold": 0.1,
                "typ_p": 1.0,
                "repeat_last_n": 64,
                "repeat_penalty": 1.1,
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
                "max_tokens": 2048,
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

    messages = _normalise_messages(body.get("messages", []))
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    do_stream: bool = bool(body.get("stream", True))
    max_tokens: int = int(body.get("max_tokens") or body.get("n_predict") or 2048)
    temperature: float = float(body.get("temperature", 0.7))
    top_p: float = float(body.get("top_p", 0.95))
    top_k: int = int(body.get("top_k", 40))
    rep_penalty: float = float(body.get("repeat_penalty", 1.1))

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

    # Intent classification — runs before LLM generation on every message
    if _TOOLING_AVAILABLE:
        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), None
        )
        if last_user_msg:
            loop = asyncio.get_event_loop()
            tool_result = await loop.run_in_executor(
                _gen_executor, _classify_intent_sync, last_user_msg
            )
            log.info("Intent: %s | tracker_type: %s | %r",
                     tool_result["intent"], tool_result.get("tracker_type", ""), tool_result["description"])
            if tool_result["intent"] != "unknown":
                return StreamingResponse(
                    _stream_tool_response(tool_result),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

    if do_stream:
        return StreamingResponse(
            stream_completion(
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=rep_penalty,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path
    prompt = _build_prompt(messages)
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

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id(),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
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
    import webbrowser
    from concurrent.futures import ThreadPoolExecutor

    global _gen_executor, _public_dir

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

    if _TOOLING_AVAILABLE:
        log.info("Tooling enabled — %d function(s): %s",
                 len(_TOOLING_FUNCTIONS_MAP), ", ".join(_TOOLING_FUNCTIONS_MAP))
    else:
        log.warning("agent_tooling.py not found — tooling disabled")

    log.info("Starting server at http://%s:%d", ARGS.host, ARGS.port)

    #webbrowser.open(f"http://{ARGS.host}:{ARGS.port}")
    uvicorn.run(
        app,
        host=ARGS.host,
        port=ARGS.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
