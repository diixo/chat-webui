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
python server.py --model gpt2
```

Demo in browser: `http://127.0.0.1:8080`

---

## Running the Server

* **1)** Model from `models/` folder

```bash
python server.py --model gpt2
# looks for models/gpt2/*
```

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
python server.py --model ./models/gpt2
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

### Model search priority

1. Path exists as is → used directly
2. `models/<name>/` exists in the repository → taken from there
3. Otherwise → downloaded from HuggingFace Hub

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
