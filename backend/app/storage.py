from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_FILE = DATA_DIR / "sessions.jsonl"


def append_session(session: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        **session,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with SESSIONS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def list_sessions(patient_id: str | None = None, limit: int = 10) -> list[dict]:
    if not SESSIONS_FILE.exists():
        return []

    with SESSIONS_FILE.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    if patient_id:
        rows = [row for row in rows if row.get("patient_id") == patient_id]

    rows.reverse()
    return rows[:limit]


def latest_session(patient_id: str | None) -> dict | None:
    """Most recent non-baseline session for the patient."""
    if not SESSIONS_FILE.exists():
        return None
    with SESSIONS_FILE.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if patient_id:
        rows = [r for r in rows if r.get("patient_id") == patient_id]
    rows = [r for r in rows if not r.get("is_baseline")]
    rows.reverse()
    return rows[0] if rows else None


def latest_baseline(patient_id: str | None) -> dict | None:
    """Most recent baseline session for the patient."""
    if not SESSIONS_FILE.exists() or not patient_id:
        return None
    with SESSIONS_FILE.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [r for r in rows if r.get("patient_id") == patient_id and r.get("is_baseline")]
    rows.reverse()
    return rows[0] if rows else None
