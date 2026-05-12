import os
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from pybeamer import Codebeamer

from ui_server import UIServer

SYSTEM_PROMPT = """You are an intent classifier for a Codebeamer project management system.

Return ONLY valid JSON with exactly three fields:
- intent
- tracker_type
- description

Allowed intent values:
- create_tracker
- unknown

Allowed tracker_type values (use exactly as written, only when intent is create_tracker):
Area, Bug, Change Request, Component, Configuration Item, Contact, Document, Epic, Issue,
Platform, Release, Requirement, Risk, RPE Report, Task, Team, Test Case, Test Configuration,
Test Run, Test Set, Time Recording, User Story

Rules:
- Classify the user's request into one intent.
- Only classify as create_tracker if the user is DIRECTLY COMMANDING to create something (imperative form).
- Questions, confirmations, and status checks are NOT creation commands — return "unknown".
- If the message contains words like "did you", "have you", "was it", "did it", "is it", "were you" — return "unknown".
- Determine tracker_type from the user's request. If the type is unspecified, default to "Requirement".
- Extract the main user-provided name/title into "description".
- The description must be a plain string.
- Remove leading helper phrases like "with name", "named", "called" from description.
- Do not output any extra keys or explanations.
- Return the JSON in a single line.

Examples:

User: create requirement: User login
Assistant: {"intent":"create_tracker","tracker_type":"Requirement","description":"User login"}

User: create bug: login form crashes on submit
Assistant: {"intent":"create_tracker","tracker_type":"Bug","description":"login form crashes on submit"}

User: create task: implement export feature
Assistant: {"intent":"create_tracker","tracker_type":"Task","description":"implement export feature"}

User: create epic: Mobile app redesign
Assistant: {"intent":"create_tracker","tracker_type":"Epic","description":"Mobile app redesign"}

User: create user story: As a user I want to reset my password
Assistant: {"intent":"create_tracker","tracker_type":"User Story","description":"As a user I want to reset my password"}

User: create risk: vendor dependency
Assistant: {"intent":"create_tracker","tracker_type":"Risk","description":"vendor dependency"}

User: create test case: login flow
Assistant: {"intent":"create_tracker","tracker_type":"Test Case","description":"login flow"}

User: how are you?
Assistant: {"intent":"unknown","tracker_type":"","description":""}

User: did you create the requirement?
Assistant: {"intent":"unknown","tracker_type":"","description":""}

User: was the tracker created?
Assistant: {"intent":"unknown","tracker_type":"","description":""}

User: have you created the task already?
Assistant: {"intent":"unknown","tracker_type":"","description":""}
"""

_NAMES_UI2API = {
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

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_tracker",
            "description": (
                "Create a new tracker item in Codebeamer. "
                "Use when the user asks to create a requirement, bug, task, story, or any other tracker type."
            ),
            "parameters": {
                "type": "object",
                "required": ["tracker_type", "description"],
                "properties": {
                    "tracker_type": {
                        "type": "string",
                        "description": "The type of tracker to create.",
                        "enum": sorted(_NAMES_UI2API.keys()),
                    },
                    "description": {
                        "type": "string",
                        "description": "Name or title for the new tracker.",
                    },
                },
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Module-level Codebeamer connection state
# ---------------------------------------------------------------------------

_project = None
_user = None
_ui2id: Dict[str, int] = {}
_base_url = ""


def _connect() -> None:
    """Load .env and connect to Codebeamer. Must be called before create_tracker."""
    global _project, _user, _ui2id, _base_url

    load_dotenv()

    codebeamer_url = os.getenv("CODEBEAMER_URL")
    username = os.getenv("CODEBEAMER_USERNAME")
    password = os.getenv("CODEBEAMER_PASSWORD")
    api_root = os.getenv("CODEBEAMER_API_ROOT", "api/v3")

    if not all([codebeamer_url, username, password]):
        raise ValueError(
            "Missing required environment variables: "
            "CODEBEAMER_URL, CODEBEAMER_USERNAME, CODEBEAMER_PASSWORD"
        )

    _base_url = codebeamer_url.rstrip("/") + "/tracker/"

    cb = Codebeamer(
        url=codebeamer_url,
        username=username,
        password=password,
        api_root=api_root,
    )

    _user = cb.get_user(username)
    if _user is None:
        raise ValueError(f"User '{username}' not found in Codebeamer")

    project_id = int(os.getenv("CODEBEAMER_PROJECT_ID", "25"))
    projects = {p.id: p for p in cb.get_projects()}
    if project_id not in projects:
        known = ", ".join(str(pid) for pid in sorted(projects.keys())[:20])
        raise ValueError(
            f"Project id={project_id} not found. Known IDs (first 20): {known}"
        )
    _project = projects[project_id]

    type2id = cb.get_tracker_types()
    missing = [
        f"{ui!r} (api={api!r})"
        for ui, api in _NAMES_UI2API.items()
        if api not in type2id
    ]
    if missing:
        raise ValueError(f"Tracker types not found in Codebeamer: {', '.join(missing)}")

    _ui2id = {ui: type2id[api] for ui, api in _NAMES_UI2API.items()}
    print(f"Codebeamer: loaded {len(_ui2id)} tracker types", flush=True)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def create_tracker(tracker_type: str = "", description: str = "", **_) -> dict:
    if _project is None or _user is None:
        raise RuntimeError("Not connected — call _connect() first.")

    tracker_type_id = _ui2id.get(tracker_type)
    if tracker_type_id is None:
        raise ValueError(
            f"Unknown tracker type: {tracker_type!r}. "
            f"Known: {', '.join(sorted(_ui2id))}"
        )

    result = _project.create_tracker(
        name=description or tracker_type,
        description=description or tracker_type,
        issue_type_id=tracker_type_id,
        user=_user,
        key_name="AUTO-GENERATED",
    )

    item_id = getattr(result, "id", None)
    url = f"{_base_url}{item_id}" if item_id is not None else None
    print(
        f"###call: CREATE TRACKER [{tracker_type}, type_id={tracker_type_id}]: "
        f"{description!r} → id: {item_id} {url}",
        flush=True,
    )
    return {"id": item_id, "url": url}


FUNCTION_MAP = {
    "create_tracker": create_tracker,
}


if __name__ == "__main__":
    _connect()

    ui = UIServer(
        model="models/Qwen3-0.6B",
        classifier_prompt=SYSTEM_PROMPT,
        tools=TOOLS,
        functions=FUNCTION_MAP,
        host="127.0.0.1",
        port=8080,
    )
    ui.run()
