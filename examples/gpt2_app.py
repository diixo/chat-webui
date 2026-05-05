import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui_server import UIServer


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    ui = UIServer(
        model="models/gpt2",
        prompt_style="raw",
        context_size=1024,
        host="127.0.0.1",
        port=8080,
    )
    ui.run()
