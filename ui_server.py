"""
ui_server.py — Embedded client for chat-webui server.

Uses server.py's FastAPI app directly in-process via httpx.ASGITransport —
no HTTP port needed. Same interface as openai.OpenAI.

Usage:

    from ui_server import UIServer

    client = UIServer(model="Qwen3-0.6B")   # loads model once

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_contact",
                "description": "Save a new contact",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name":  {"type": "string", "description": "Full name"},
                        "phone": {"type": "string", "description": "Phone number"},
                    },
                },
            },
        }
    ]

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "add contact Alice, phone +1 555 0101"}],
        tools=tools,
    )

    if response.choices[0].finish_reason == "tool_calls":
        call = response.choices[0].message.tool_calls[0]
        print(call.function.name, call.function.arguments_dict())
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

import server as _server
from agent_tooling import AgentTooling


# ---------------------------------------------------------------------------
# Generic tooling — built from TOOLS + FUNCTION_MAP, no custom class needed
# ---------------------------------------------------------------------------

class _GenericTooling(AgentTooling):
    """Tooling provider assembled from primitives passed to UIServer."""

    def __init__(
        self,
        classifier_prompt: str,
        tools: list[dict],
        functions: dict[str, Callable],
    ) -> None:
        self._classifier_prompt = classifier_prompt
        self._tools = tools
        self._functions = functions

    def classifier_prompt(self) -> str:
        return self._classifier_prompt

    def normalize_result(self, obj: dict[str, Any]) -> dict[str, Any]:
        """Extract and clean the fields declared in TOOLS for the given intent."""
        intent = str(obj.get("intent", "unknown")).strip()
        if intent not in self._functions:
            intent = "unknown"

        result: dict[str, Any] = {"intent": intent}
        for tool in self._tools:
            if tool["function"]["name"] == intent:
                props = tool["function"]["parameters"].get("properties", {})
                for field in props:
                    if field in obj:
                        result[field] = str(obj[field]).strip()
                break
        return result

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        normalized = self.normalize_result({"intent": name, **arguments})
        fn = self._functions.get(normalized["intent"])
        if fn is None:
            raise ValueError(f"Unknown tool: {name!r}")
        params = {k: v for k, v in normalized.items() if k != "intent"}
        return fn(**params)


# ---------------------------------------------------------------------------
# Response data classes (mirror openai SDK structure)
# ---------------------------------------------------------------------------

class FunctionCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments  # JSON string, same as OpenAI SDK

    def arguments_dict(self) -> dict:
        """Convenience: parse arguments JSON to dict."""
        return json.loads(self.arguments) if self.arguments else {}

    def __repr__(self) -> str:
        return f"FunctionCall(name={self.name!r}, arguments={self.arguments!r})"


class ToolCall:
    def __init__(self, id: str, type: str, function: FunctionCall) -> None:
        self.id = id
        self.type = type
        self.function = function

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, function={self.function!r})"


class Message:
    def __init__(
        self,
        role: str,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, content={self.content!r}, tool_calls={self.tool_calls!r})"


class Choice:
    def __init__(self, index: int, message: Message, finish_reason: str) -> None:
        self.index = index
        self.message = message
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return f"Choice(finish_reason={self.finish_reason!r}, message={self.message!r})"


class Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class ChatCompletion:
    def __init__(self, id: str, model: str, choices: list[Choice], usage: Usage | None = None) -> None:
        self.id = id
        self.model = model
        self.choices = choices
        self.usage = usage

    def __repr__(self) -> str:
        return f"ChatCompletion(id={self.id!r}, choices={self.choices!r})"


# ---------------------------------------------------------------------------
# Streaming chunk classes
# ---------------------------------------------------------------------------

class DeltaContent:
    def __init__(self, role: str | None = None, content: str | None = None) -> None:
        self.role = role
        self.content = content


class StreamChoice:
    def __init__(self, index: int, delta: DeltaContent, finish_reason: str | None) -> None:
        self.index = index
        self.delta = delta
        self.finish_reason = finish_reason


class ChatCompletionChunk:
    def __init__(self, id: str, model: str, choices: list[StreamChoice]) -> None:
        self.id = id
        self.model = model
        self.choices = choices


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_completion(data: dict) -> ChatCompletion:
    choices = []
    for c in data.get("choices", []):
        msg_data = c.get("message", {})
        tool_calls = []
        for tc in msg_data.get("tool_calls", []):
            fn = tc.get("function", {})
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                function=FunctionCall(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "{}"),
                ),
            ))
        message = Message(
            role=msg_data.get("role", "assistant"),
            content=msg_data.get("content"),
            tool_calls=tool_calls if tool_calls else None,
        )
        choices.append(Choice(
            index=c.get("index", 0),
            message=message,
            finish_reason=c.get("finish_reason", "stop"),
        ))

    usage_data = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_data.get("prompt_tokens", 0),
        completion_tokens=usage_data.get("completion_tokens", 0),
        total_tokens=usage_data.get("total_tokens", 0),
    ) if usage_data else None

    return ChatCompletion(
        id=data.get("id", ""),
        model=data.get("model", ""),
        choices=choices,
        usage=usage,
    )


def _iter_sse_bytes(content: bytes) -> Iterator[ChatCompletionChunk]:
    """Parse SSE bytes from httpx response and yield ChatCompletionChunk objects."""
    for raw_line in content.splitlines():
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        chunks = []
        for c in data.get("choices", []):
            delta_data = c.get("delta", {})
            chunks.append(StreamChoice(
                index=c.get("index", 0),
                delta=DeltaContent(
                    role=delta_data.get("role"),
                    content=delta_data.get("content"),
                ),
                finish_reason=c.get("finish_reason"),
            ))
        yield ChatCompletionChunk(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=chunks,
        )


# ---------------------------------------------------------------------------
# In-process ASGI transport
# ---------------------------------------------------------------------------

class _ASGIClient:
    """Calls server.app in-process using httpx.ASGITransport (no network port)."""

    def __init__(self) -> None:
        self._transport = httpx.ASGITransport(app=_server.app)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-asgi")

    def post(self, path: str, body: dict) -> httpx.Response:
        async def _call() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=self._transport,
                base_url="http://server",
            ) as client:
                return await client.post(
                    path,
                    content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    timeout=300.0,
                )
        return self._executor.submit(asyncio.run, _call()).result()


# ---------------------------------------------------------------------------
# Completions API
# ---------------------------------------------------------------------------

class Completions:
    def __init__(
        self,
        http: _ASGIClient,
        model: str,
        tools: list[dict] | None = None,
        tooling: Any = None,
    ) -> None:
        self._http = http
        self._model = model
        self._tools = tools    # injected into every request when not overridden
        self._tooling = tooling  # object with execute(name, args) -> dict

    def _post(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        model: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
        stream: bool,
        enable_thinking: bool | None = None,
        suppress_after_response: bool = False,
    ) -> "ChatCompletion | Iterator[ChatCompletionChunk]":
        body: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }
        if enable_thinking is not None:
            body["enable_thinking"] = enable_thinking
        if suppress_after_response:
            body["suppress_after_response"] = True
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        resp = self._http.post("/v1/chat/completions", body)
        if resp.status_code != 200:
            raise RuntimeError(f"Server error {resp.status_code}: {resp.text}")

        if stream:
            return _iter_sse_bytes(resp.content)
        return _parse_completion(resp.json())

    def create(
        self,
        *,
        messages: list[dict],
        model: str | None = None,
        tools: list[dict] | None = None,
        stream: bool = False,
        max_tokens: int = _server.DEFAULT_GEN_PARAMS["max_tokens"],
        temperature: float = _server.DEFAULT_GEN_PARAMS["temperature"],
        top_p: float = _server.DEFAULT_GEN_PARAMS["top_p"],
        top_k: int = _server.DEFAULT_GEN_PARAMS["top_k"],
        repeat_penalty: float = _server.DEFAULT_GEN_PARAMS["repeat_penalty"],
        enable_thinking: bool | None = None,
        suppress_after_response: bool = False,
    ) -> "ChatCompletion | Iterator[ChatCompletionChunk]":
        effective_tools = tools if tools is not None else self._tools

        # Agentic loop: auto-execute tool calls when tooling provides execute().
        if effective_tools and self._tooling is not None:
            return self._agentic_create(
                messages=list(messages),
                tools=effective_tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
            )

        return self._post(
            messages=messages,
            tools=effective_tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stream=stream if not effective_tools else False,
            enable_thinking=enable_thinking,
            suppress_after_response=suppress_after_response,
        )

    def _agentic_create(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
    ) -> ChatCompletion:
        """Run the tool-call loop: model → tool_calls → execute → model → … → final answer."""
        while True:
            result = self._post(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                stream=False,
            )
            choice = result.choices[0]

            if choice.finish_reason != "tool_calls":
                return result

            # Append assistant turn with tool_calls to history
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            })

            # Execute each tool and append results
            for tc in choice.message.tool_calls:
                try:
                    tool_result = self._tooling.execute(
                        tc.function.name, tc.function.arguments_dict()
                    )
                    content = json.dumps(tool_result, ensure_ascii=False)
                except Exception as exc:
                    content = json.dumps({"error": str(exc)})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                })


class Chat:
    def __init__(
        self,
        http: _ASGIClient,
        model: str,
        tools: list[dict] | None = None,
        tooling: Any = None,
    ) -> None:
        self.completions = Completions(http, model, tools=tools, tooling=tooling)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class UIServer:
    """Embedded chat-webui client — uses server.py FastAPI app in-process.

    Same interface as openai.OpenAI.

    Args:
        model:              Model name, local path, or HuggingFace Hub ID.
        tools:              OpenAI-format tool definitions injected into every request.
        functions:          Dict mapping tool name → callable. Combined with tools and
                            classifier_prompt to build tooling automatically — no custom
                            class needed for simple use cases.
        classifier_prompt:  System prompt used only for server-side intent classification
                            (_classify_intent_sync). Separate from system_prompt so the
                            model doesn't output JSON in normal chat responses.
        tooling:            Explicit tooling object (e.g. CodebeamerTooling). Takes
                            precedence over functions/classifier_prompt if provided.
        system_prompt:      System prompt injected into every regular chat request.
        prompt_style:       "auto" (default) uses the tokenizer's chat template;
                            "raw" concatenates message contents with no role markers
                            — for base models like GPT-2.
        context_size:       Max token context (default 4096).
        device:             "auto" | "cpu" | "cuda" | "mps".
        dtype:              "auto" | "float16" | "bfloat16" | "float32".
        trust_remote_code:  Allow remote model code.
        host:               uvicorn host (used by run()).
        port:               uvicorn port (used by run()).
    """

    def __init__(
        self,
        model: str,
        *,
        tools: list[dict] | None = None,
        functions: dict[str, Callable] | None = None,
        classifier_prompt: str | None = None,
        tooling: Any = None,
        system_prompt: str | None = None,
        prompt_style: str = "auto",
        enable_thinking: bool | None = None,
        context_size: int = 1024,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        _server.configure(
            model=model,
            context_size=context_size,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            system_prompt=system_prompt,
            prompt_style=prompt_style,
            enable_thinking=enable_thinking,
        )
        _server._gen_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")

        public_dir = Path(_server.__file__).parent / "server" / "public"
        _server._public_dir = public_dir if public_dir.is_dir() else None

        _server.load_model()

        if tooling is None and functions is not None:
            tooling = _GenericTooling(
                classifier_prompt=classifier_prompt or "",
                tools=tools or [],
                functions=functions,
            )

        if tooling is not None:
            _server.register_tooling(tooling)

        self._host = host
        self._port = port
        self._uvicorn_server: Any = None

        http = _ASGIClient()
        self.chat = Chat(http, model, tools=tools, tooling=tooling)

    def run(self) -> None:
        """Start uvicorn and block until Ctrl+C — mirrors uvicorn.run() / server.py main().

        Use this as the last call in your script to keep the server alive:

            client = UIServer(model="Qwen3-0.6B")
            # ... do setup, register tools, etc. ...
            client.run()   # blocks here; Ctrl+C to stop
        """
        import uvicorn

        cfg = uvicorn.Config(
            app=_server.app,
            host=self._host,
            port=self._port,
            log_level="info",
            access_log=True,
        )
        self._uvicorn_server = uvicorn.Server(cfg)

        print(
            f"Starting server at http://{self._host}:{self._port} — press Ctrl+C to stop",
            flush=True,
        )
        self._uvicorn_server.run()   # blocks until shutdown
