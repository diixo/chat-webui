import json
import re
from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dotenv import load_dotenv
import os
from pybeamer import Codebeamer

MODEL_NAME = "models/Qwen3-0.6B"

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

BASE_URL = "https://dxc.codebeamer.com/tracker/"

_codebeamer_context: dict = {}


def init_codebeamer() -> dict:
    """Initialize Codebeamer connection and return context.

    Returns:
        dict: Dictionary containing 'project', 'user', and 'ui2id' keys.
    """
    global _codebeamer_context

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

    _codebeamer_context["project"] = project
    _codebeamer_context["user"] = user
    _codebeamer_context["ui2id"] = ui2id

    return _codebeamer_context.copy()


def create_tracker(description: str, tracker_type: str) -> dict:
    global _codebeamer_context

    project = _codebeamer_context.get("project")
    user = _codebeamer_context.get("user")
    ui2id = _codebeamer_context.get("ui2id", {})

    if project is None or user is None:
        raise RuntimeError("Codebeamer context not initialized. Call init_codebeamer() first.")

    tracker_type_id = ui2id.get(tracker_type)
    
    if tracker_type_id is None:
        raise ValueError(
            f"Unknown tracker type: {tracker_type!r}. "
            f"Known types: {', '.join(sorted(ui2id.keys()))}"
        )

    result = project.create_tracker(
        name=description or tracker_type,
        description=description or tracker_type,
        issue_type_id=tracker_type_id,
        user=user,
        key_name="AUTO-GENERATED",
    )

    item_id = getattr(result, "id", None)
    url = f"{BASE_URL}{item_id}" if item_id is not None else None
    print(f"###call: CREATE TRACKER [{tracker_type}, type_id={tracker_type_id}]: {description!r} → id: {item_id} {url}")
    return {"id": item_id, "url": url}


FUNCTIONS_MAP = {
    "create_tracker": create_tracker,
}

ALLOWED_INTENTS = {
    "create_tracker",
    "unknown",
}


def extract_first_json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return match.group(0)


def normalize_result(obj: Dict[str, Any]) -> Dict[str, str]:
    intent = obj.get("intent", "unknown")
    description = obj.get("description", "")
    tracker_type = obj.get("tracker_type", "")

    if not isinstance(intent, str):
        intent = "unknown"
    if not isinstance(description, str):
        description = ""
    if not isinstance(tracker_type, str):
        tracker_type = ""

    intent = intent.strip()
    description = description.strip()
    tracker_type = tracker_type.strip()

    if intent not in ALLOWED_INTENTS:
        intent = "unknown"

    return {
        "intent": intent,
        "description": description,
        "tracker_type": tracker_type,
    }


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    return tokenizer, model


def classify_intent(tokenizer, model, user_text: str) -> Dict[str, str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    json_text = extract_first_json_object(raw_output)
    parsed = json.loads(json_text)
    return normalize_result(parsed)


if __name__ == "__main__":
    tokenizer, model = load_model()

    tests = [
        "create requirement with name User login must support SSO",
        "create bug login form crashes on submit",
        "create task implement export feature",
        "create epic Mobile app redesign",
        "create risk vendor dependency",
        "how are you?",
        "did you create the requirement?",
    ]

    for text in tests:
        result = classify_intent(tokenizer, model, text)
        print(f"USER:   {text}")
        print(f"MODEL:  {result}")
        print("-" * 60)
