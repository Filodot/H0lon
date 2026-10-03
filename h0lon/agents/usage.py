"""Usage journal: one JSON line per agent attempt in <state_dir>/usage.jsonl."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from h0lon.agents.base import Usage

USAGE_FILE = "usage.jsonl"
_lock = threading.Lock()


def usage_path(state_dir: Path) -> Path:
    return Path(state_dir) / USAGE_FILE


def append_usage(state_dir: Path, record: dict[str, Any]) -> Path:
    """Append one record (a single line, written with one call) and return the journal path."""
    path = usage_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _lock, path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
    return path


def read_usage(state_dir: Path) -> list[dict[str, Any]]:
    """All parseable records (broken lines are skipped)."""
    path = usage_path(state_dir)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


@dataclass
class UsageTotals:
    attempts: int
    ok: int
    usage: Usage
    by_backend: dict[str, Usage]


def totals(
    records: list[dict[str, Any]], *, since: datetime | None = None, backend: str | None = None
) -> UsageTotals:
    """Sum tokens over records (optionally only after `since` / for one backend)."""
    total, by_backend, attempts, ok = Usage(), {}, 0, 0
    for rec in records:
        if backend and rec.get("backend") != backend:
            continue
        if since is not None:
            ts = _parse_ts(rec.get("ts"))
            if ts is None or ts < since:
                continue
        u = Usage.from_dict(rec)
        total = total + u
        name = str(rec.get("backend") or "?")
        by_backend[name] = by_backend.get(name, Usage()) + u
        attempts += 1
        ok += bool(rec.get("ok"))
    return UsageTotals(attempts=attempts, ok=ok, usage=total, by_backend=by_backend)


def period_totals(state_dir: Path, *, now: datetime | None = None) -> dict[str, UsageTotals]:
    """Totals for the last 5 hours (a Claude limit window), day and week."""
    now = now or datetime.now(UTC)
    records = read_usage(state_dir)
    return {
        "5h": totals(records, since=now - timedelta(hours=5)),
        "day": totals(records, since=now - timedelta(days=1)),
        "week": totals(records, since=now - timedelta(days=7)),
    }
