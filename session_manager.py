import json
from datetime import datetime
from pathlib import Path

DEFAULT_SESSION_DIR = Path.home() / ".localfileagent" / "sessions"


class SessionManager:
    def __init__(self, session_dir: Path = DEFAULT_SESSION_DIR):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: dict) -> Path:
        """Write session to disk. Adds 'created' if absent. Returns the path written."""
        if "created" not in session:
            session["created"] = datetime.now().isoformat(timespec="seconds")
        filename = session["created"].replace(":", "-").replace("T", "_") + ".json"
        path = self.session_dir / filename
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")
        return path

    def list(self) -> list[dict]:
        """Return all valid sessions sorted newest-first. Each dict includes '_path'."""
        sessions = []
        for p in sorted(self.session_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["_path"] = str(p)
                sessions.append(data)
            except (json.JSONDecodeError, OSError):
                continue
        return sessions

    def load(self, path: str) -> dict:
        """Load and return a session dict from its absolute path."""
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def delete(self, path: str) -> None:
        """Delete a session file. No-op if already gone."""
        Path(path).unlink(missing_ok=True)
