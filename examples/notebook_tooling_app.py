"""
example_notebook.py — Demo: notebook tool calling via UIServer.

Запускает FastAPI сервер (server.py) в фоновом потоке и подключает TOOLS.
Программные вызовы идут через ASGI (без HTTP), браузер/curl доступен на порту 8080.

Usage:
    python example_notebook.py
"""

from typing import Any, Dict
from agent_tooling import AgentTooling
from ui_server import UIServer


# System prompt for the regular chat persona — used when the classifier returned
# "unknown" and the request falls through to plain chat completion.
CHAT_SYSTEM_PROMPT = """You are a friendly personal-notebook assistant.
Help the user keep track of their contacts and notes.
Answer in plain natural language. Never reply in JSON or with raw tool syntax."""


# Classifier prompt — used ONLY for the intent-classification pass inside
# NotebookTooling. Forces the model to return a single-line JSON object.
CLASSIFIER_PROMPT = """You are an intent classifier for a personal notebook assistant.

Return ONLY a single-line JSON object with "intent" and the fields listed below.

Allowed intents:
- add_contact      — user wants to save or add a new contact
- add_note         — user wants to write or save a note or memo
- search_contact   — user wants to find a contact by name
- search_by_phone  — user wants to find a contact by phone number
- list_contacts    — user wants to see all contacts
- list_notes       — user wants to see notes or memos
- delete_contact   — user wants to remove or delete a contact
- unknown          — anything else

Rules:
- Only activate a tool intent if the user DIRECTLY COMMANDS an action (imperative).
- Questions ("did you add?", "have you saved?") → return "unknown".
- Extract all available values from the message. Use "" for missing values.
- Do not add extra keys or explanations.

Fields per intent:

add_contact:    {"intent":"add_contact","name":"","phone":"","email":""}
add_note:       {"intent":"add_note","text":"","contact":""}
search_contact: {"intent":"search_contact","query":""}
search_by_phone:{"intent":"search_by_phone","phone":""}
list_contacts:  {"intent":"list_contacts"}
list_notes:     {"intent":"list_notes","contact":""}
delete_contact: {"intent":"delete_contact","name":""}
unknown:        {"intent":"unknown"}

Examples:

User: add contact Alice Smith, phone +1 555 0101, email alice@example.com
Assistant: {"intent":"add_contact","name":"Alice Smith","phone":"+1 555 0101","email":"alice@example.com"}

User: save contact Bob Jones, +44 7700 900123
Assistant: {"intent":"add_contact","name":"Bob Jones","phone":"+44 7700 900123","email":""}

User: add note: meeting with Alice went well
Assistant: {"intent":"add_note","text":"meeting with Alice went well","contact":"Alice"}

User: write a note: remember to buy milk
Assistant: {"intent":"add_note","text":"remember to buy milk","contact":""}

User: find contact Alice
Assistant: {"intent":"search_contact","query":"Alice"}

User: who has phone +1 555 0101
Assistant: {"intent":"search_by_phone","phone":"+1 555 0101"}

User: show all contacts
Assistant: {"intent":"list_contacts"}

User: list my notes for Alice
Assistant: {"intent":"list_notes","contact":"Alice"}

User: show all notes
Assistant: {"intent":"list_notes","contact":""}

User: delete contact Bob
Assistant: {"intent":"delete_contact","name":"Bob"}

User: what is the weather today?
Assistant: {"intent":"unknown"}

User: did you save the contact?
Assistant: {"intent":"unknown"}
"""

# ---------------------------------------------------------------------------
# Tool definitions (same format as openai SDK)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_contact",
            "description": "Save a new contact to the notebook",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name":  {"type": "string", "description": "Full name of the contact"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "email": {"type": "string", "description": "Email address"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_contact",
            "description": "Find contacts by name",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Name or part of name to search"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": "Save a text note, optionally linked to a contact",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text":    {"type": "string", "description": "Note body text"},
                    "contact": {"type": "string", "description": "Related contact name (optional)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_contacts",
            "description": "List all saved contacts",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_contact",
            "description": "Delete a contact by name",
            "parameters": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "Name of the contact to delete"},
                },
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tooling adapter — dispatches tool calls to bound instance methods.
#
# How dispatch works:
#   getattr(self, "add_contact") returns a *bound method*, so calling it
#   passes `self` automatically. That's how FUNCTION_MAP can hold methods of
#   a specific instance — each value is `self.method`, not the unbound function.
# ---------------------------------------------------------------------------

class NotebookTooling(AgentTooling):

    def classifier_prompt(self) -> str:
        return CLASSIFIER_PROMPT

    def normalize_result(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        known = {t["function"]["name"] for t in TOOLS}
        intent = str(obj.get("intent", "unknown")).strip()
        if intent not in known:
            intent = "unknown"

        result: Dict[str, Any] = {"intent": intent}
        for tool in TOOLS:
            if tool["function"]["name"] == intent:
                for field in tool["function"]["parameters"].get("properties", {}):
                    if field in obj:
                        result[field] = str(obj[field]).strip()
                break
        return result

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        normalized = self.normalize_result({"intent": name, **arguments})
        intent = normalized["intent"]
        fn = getattr(self, intent, None)
        if not callable(fn):
            raise ValueError(f"Unknown tool: {name!r}")
        params = {k: v for k, v in normalized.items() if k != "intent"}
        return fn(**params)

    # ---------------------------------------------------------------------
    # Tool implementations (stubs — replace with real logic)
    # ---------------------------------------------------------------------

    def add_contact(self, name: str, phone: str = "", email: str = "") -> str:
        print(f"  → add_contact({name!r}, phone={phone!r}, email={email!r})")
        return f"Contact '{name}' saved."

    def search_contact(self, query: str) -> str:
        print(f"  → search_contact({query!r})")
        return f"Found: Alice Smith (+1 555 0101) — stub result for '{query}'."

    def add_note(self, text: str, contact: str = "") -> str:
        print(f"  → add_note(text={text!r}, contact={contact!r})")
        return f"Note saved." + (f" Linked to '{contact}'." if contact else "")

    def list_contacts(self) -> str:
        print("  → list_contacts()")
        return "Contacts: Alice Smith, Bob Jones. (stub)"

    def delete_contact(self, name: str) -> str:
        print(f"  → delete_contact({name!r})")
        return f"Contact '{name}' deleted."


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui = UIServer(
        model="models/Qwen3-0.6B",
        # Persona prompt for plain chat (when classifier returns "unknown").
        # The classifier prompt lives separately inside NotebookTooling.system_prompt.
        system_prompt=CHAT_SYSTEM_PROMPT,
        tools=TOOLS,
        tooling=NotebookTooling(),
        host="127.0.0.1",
        port=8080,
    )
    ui.run()

    # add contact Alice, +7 999 123
