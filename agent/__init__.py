"""Self-learning paper-trading agent for gold/USD."""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a project-root .env into os.environ.

    Real environment variables always win (setdefault), so CI secrets passed as
    env vars override the file. No third-party dependency.
    """
    envf = Path(__file__).resolve().parent.parent / ".env"
    if not envf.exists():
        return
    for raw in envf.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()
