# chat-webui

Web interface for LLM chat (port of original **llama-server-webui@1.0.0**), matching the core dialogue workflow and interaction patterns of `llama.cpp`.

Focused on a familiar chat experience, responsive interaction, and straightforward integration with model backends. 
Suitable for local or remote inference, prompt testing, and everyday work with language models.

* Python-server (FastAPI)
* Transformers for model
* SvelteKit + TypeScript for UI

---

## Quick Start

```bash
pip install -r requirements.txt
python server.py --model gpt2 --prompt-style raw
```

Demo in browser: `http://127.0.0.1:8080`

---

## Running the Server

* **1)** Model from `models/` folder


```bash
python server.py --model Qwen3-0.6B
# looks for models/Qwen3-0.6B/*
```

* **2)** Model from HuggingFace Hub

```bash
python server.py --model Qwen/Qwen2.5-1.5B-Instruct
```

* **3)** Explicit path

```bash
python server.py --model ./models/gpt2 --prompt-style raw
```


## Rebuild UI

Only needed when changing source files in `server/webui/src/`.
The result is compiled into `server/public/` (this is the folder served by the server).

```bash
cd server/webui
npm install        # first time or after changing package.json
npm run build      # rebuild bundle.js / bundle.css
```


### All options

| Flag | Default | Description |
|---|---|---|
| `--model` / `-m` | — | Model name, path, or Hub ID (required) |
| `--port` / `-p` | `8080` | Server port |
| `--host` | `127.0.0.1` | Host |
| `--context-size` / `-c` | `4096` | Context size in tokens |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `--dtype` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |
| `--trust-remote-code` | — | Allow remote model code |
| `--api-key` | — | Bearer token for authentication |
| `--public-dir` | `server/public` | UI static folder |
| `--max-history` | `0` | Max number of past messages sent to the model (0 = unlimited). System message is always kept. |
| `--prompt-style` | `auto` | `auto` = use tokenizer's `chat_template` (instruct models); `raw` = concatenate messages with newlines (base models like GPT-2) |

### Model search priority

1. Path exists as is → used directly
2. `models/<name>/` exists in the repository → taken from there
3. Otherwise → downloaded from HuggingFace Hub

### Chat prompt style

`_build_prompt` in `server.py` supports two styles, selected via the `prompt_style`
parameter of `server.configure()` / `UIServer(...)`:

| Style | When to use |
|---|---|
| `auto` (default) | Instruction-tuned models (Qwen, Llama-Instruct, …). Uses the tokenizer's `chat_template` if available; otherwise falls back to a `<\|user\|> / <\|assistant\|>` ChatML format. |
| `raw` | Base / completion models without a chat template (e.g. **GPT-2**). Concatenates message contents with newlines — no role markers. The model just continues the text. |

Example (see `gpt2_app.py`):

```python
from ui_server import UIServer

UIServer(
    model="models/gpt2",
    prompt_style="raw",   # GPT-2 doesn't understand <|user|>/<|assistant|>
    context_size=1024,
).run()
```

### Sampling vs greedy decoding

There is **no separate `do_sample` flag** — the mode is inferred from `temperature`:

| `temperature` | Mode | Behaviour |
|---|---|---|
| `> 1e-4` (e.g. default `0.8`) | Sampling | Stochastic decode using `temperature`, `top_p`, `top_k`, `repeat_penalty` |
| `≈ 0` (or any value `≤ 1e-4`) | Greedy | Deterministic — always picks the highest-probability token |

Implementation (`server.py`, both streaming and non-streaming paths):

```python
"do_sample": temperature > 1e-4,         # ~0 → greedy
"temperature": max(temperature, 1e-7),   # never pass 0 to model.generate()
```

The `max(..., 1e-7)` floor avoids HuggingFace's parameter validation rejecting
`temperature=0` even in greedy mode — the value isn't actually used when
`do_sample=False`.

Defaults live in a single place — `DEFAULT_GEN_PARAMS` in `server.py`:

```python
DEFAULT_GEN_PARAMS = {
    "temperature": 0.8,    # sampling on by default
    "top_p": 0.95,
    "top_k": 20,
    "repeat_penalty": 1.1,
    "max_tokens": 1024,
}
```

These same values are surfaced to the frontend via `/props.default_generation_settings.params`
(used as placeholders in **Settings → Generation**) and act as fallbacks for any
sampling param missing from a chat request body. To force greedy decoding for a
single request, set `temperature: 0` in the body or in the UI's Generation tab.

---

## Add model locally
You should `pip install huggingface_hub` to download models from HuggingFace Hub:
```bash
hf download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
python server.py --model Qwen3-1.7B
```

---


### Data Source Configuration

The list of sources and projects in the left panel is taken from `server/public/config.json`.  
The file can be edited directly — no rebuild is needed:

### Configuration
Configurate by **config.json** in `server/webui/static/config.json` that copied to `server/public/config.json` when rebuilding the UI. 

**Example:**
```json
{
  "Jira": ["Project Alpha", "Project Beta"],
  "Confluence": ["Engineering Docs", "HR Space"],
  "GitHub": ["frontend", "backend"]
}
```

-When rebuilding the UI, the file `server/webui/static/config.json` is automatically copied to `server/public/`.

### Chat mode (QA vs Dialogue)

The first dropdown in the left sidebar (**Mode**) controls how much
conversation history the frontend sends to the model on each turn:

| Mode | What gets sent | When to use |
|---|---|---|
| **Dialogue** (default) | All messages from the active conversation | Multi-turn chat — the model sees its prior replies and can refer back to earlier context. |
| **QA** | Only the latest user message (plus any system message) | Independent question / answer turns — the model has no memory of earlier turns within the same conversation. |

Implementation notes:

- The selection is persisted in `localStorage["chat-mode"]` and synchronised
  across components via `lib/stores/chat-mode.svelte.ts`.
- Filtering happens **on the way out**, in `chat.service.ts:applyQaModeFilter`.
  The conversation in IndexedDB and on screen is never modified — switching
  QA → Dialogue mid-thread restores the full context for the next request.
- Independent of the server's `--max-history` flag: that one caps history
  *server-side* for any client; QA mode strips it *client-side* before the
  request is even sent.

---

## Adding a New Tooling Backend

Tooling lets the server classify a user message into a tool call and execute
it server-side instead of generating a plain chat response. A provider is a
subclass of `AgentTooling` (defined in `agent_tooling.py`) that implements
three abstract methods.

### 1. Tooling protocol

`agent_tooling.AgentTooling` declares the contract — three `@abstractmethod`s:

| Method | Used for |
|---|---|
| `classifier_prompt() -> str` | System prompt for the intent-classification pass — instructs the model to return JSON |
| `normalize_result(obj: dict) -> dict` | Validate / clean the JSON the classifier returned |
| `execute(name: str, arguments: dict) -> Any` | Run the tool by name and return its result |

Forgetting to implement any of them produces a `TypeError` at instantiation
(courtesy of `abc`), not later when classification fires. There is no
`functions_map()` — dispatch is whatever `execute()` does internally
(typically `getattr(self, name)` to call a method by intent name).

### 2. Minimal provider

```python
# my_tooling.py
from typing import Any, Dict
from agent_tooling import AgentTooling

CLASSIFIER_PROMPT = """You are an intent classifier.
Return ONLY a JSON object: {"intent": "...", "description": "..."}.
Allowed intents: do_something, unknown."""


class MyTooling(AgentTooling):

    def classifier_prompt(self) -> str:
        return CLASSIFIER_PROMPT

    def normalize_result(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        intent = str(obj.get("intent", "unknown")).strip()
        if intent not in {"do_something"}:
            intent = "unknown"
        return {"intent": intent, "description": str(obj.get("description", "")).strip()}

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        fn = getattr(self, name, None)
        if not callable(fn):
            raise ValueError(f"Unknown tool: {name!r}")
        return fn(**{k: v for k, v in arguments.items() if k != "intent"})

    # Tool implementations — method names must match the intents above
    def do_something(self, description: str = "") -> dict:
        return {"result": f"Done: {description}"}
```

`getattr(self, name)` returns a *bound method*, so calls go through this
specific instance automatically — no separate function map needed.

### 3. Register and run

The chat persona and the classifier prompt are **two different prompts** —
keep them separate, otherwise the classifier prompt leaks into plain chat
replies and the model will answer in JSON for non-tool messages:

- `UIServer(system_prompt=…)` — persona for plain chat (used when the
  classifier returns `unknown` and the request falls through to regular
  generation).
- `UIServer(tooling=…)` — provider whose `classifier_prompt()` is used only
  for the isolated classification pass.

**Via `UIServer` (in-process, recommended):**

```python
from ui_server import UIServer
from my_tooling import MyTooling

UIServer(
    model="models/Qwen3-0.6B",
    system_prompt="You are a helpful assistant.",   # plain-chat persona
    tooling=MyTooling(),                            # provides classifier_prompt()
).run()
```

**Via `server.py` directly (CLI mode):**

```python
import server
from my_tooling import MyTooling

server.register_tooling(MyTooling())   # raises TypeError if not an AgentTooling
server.main()
```

### Demo: notebook provider

`examples/notebook_tooling_app.py` is a ready-to-run demo with 5 tool intents
(add / search / list / delete contacts and notes). All tool methods are
stubs with `print()` — replace the bodies with real logic. The example shows
the recommended split: `CHAT_SYSTEM_PROMPT` for `UIServer(system_prompt=…)`
and `CLASSIFIER_PROMPT` returned from `NotebookTooling.classifier_prompt()`.

Example utterances the classifier routes to the right method:

| User says | Intent |
|---|---|
| `add contact Alice, phone +1 555 0101` | `add_contact` |
| `write a note: meeting went well` | `add_note` |
| `find contact Alice` | `search_contact` |
| `show all contacts` | `list_contacts` |
| `delete contact Bob` | `delete_contact` |

---

## Testing the Intent Classifier

`test_classifier.py` runs the Codebeamer intent classifier **without starting the server**.  
It loads the model directly via `utils.py` (reusing `SYSTEM_PROMPT` from `examples/codebeamer_app.py`) and checks classification accuracy across all tracker types.

```bash
python test_classifier.py
```

No `.env`, no Codebeamer connection, no HTTP port required — only the model files in `models/`.

Sample output:
```
Loading model...
Model loaded. Running 43 test cases...

[PASS] create requirement: Battery voltage sensor shall measure range 9V to 16V
[PASS] create bug: ABS sensor returns zero speed at vehicle start
...
Results: 43/43 passed — all passed!
```

---

## Project Structure

```
chat-webui/
├── server.py              # Python server (FastAPI + Transformers)
├── requirements.txt       # Python dependencies
├── models/                # Local models (not in git)
├── server/
│   ├── public/            # Compiled UI (bundle.js, config.json, …)
│   └── webui/             # UI source (SvelteKit + TypeScript)
│       ├── src/
│       └── static/
│           └── config.json
```
