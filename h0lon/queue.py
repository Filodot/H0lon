"""Build queue: topics are built one after another at a pace that suits the subscription limits.

Contract: docs/ARCHITECTURE.md, «Очередь сборок», PRD 12.5 (D16).

- The queue is one file, `<state_dir>/queue.json` (items `{id, topic, action, params, status,
  added, started, finished, message}`), rewritten atomically under a lock that is shared by the
  threads and the processes of h0lon (CLI and web interface see the same queue).
- `run_queue` executes the `queued` and `paused` items in order until none is left. A second
  runner is refused (`QueueBusyError`): one lock file is held for the whole run, so the items
  left `running` by a dead process can safely be taken for interrupted and queued again.
- Pace. Before every item and before every stage of a build that starts agents the newest known
  load of the Claude windows is read from `<state_dir>/usage.jsonl` (the `limits` of the attempts).
  Five-hour window at or above `queue.pause_at`: wait until it resets (+1 minute). Weekly window at
  or above `queue.weekly_stop`: the queue stops, the item it would run is `paused`, the rest stay
  `queued`. A window whose reset time has passed counts as empty; without data there are no
  pauses. The clock and `sleep` are injectable, so tests never wait.
- An item stopped at the review gate is `paused` and the queue goes on with the next one. A failed
  item stops the queue unless it runs unattended (`--unattended` or `queue.unattended`): then the
  next item starts, and on Windows the system is kept awake (`queue.keep_awake`).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from h0lon.config import Settings

QUEUE_FILE = "queue.json"
QUEUE_LOCK_FILE = "queue.json.lock"
RUN_LOCK_FILE = "queue.run.lock"
ACTIONS = ("extract", "build", "variant")
STATUSES = ("queued", "running", "done", "failed", "paused")
RUNNABLE = ("queued", "paused")
STATUS_LABELS = {
    "queued": "в очереди",
    "running": "выполняется",
    "done": "готово",
    "failed": "ошибка",
    "paused": "пауза",
}
ACTION_LABELS = {"extract": "извлечение", "build": "сборка", "variant": "вариация"}

WINDOW_5H = "five_hour"
WINDOW_7D = "seven_day"
WINDOW_LABELS = {WINDOW_5H: "5 ч", WINDOW_7D: "7 дн"}
WINDOW_LENGTHS = {WINDOW_5H: timedelta(hours=5), WINDOW_7D: timedelta(days=7)}
RESET_MARGIN_S = 60  # wait this long after the reported reset of a window
MAX_WAIT_S = 5 * 3600 + 600  # one wait never lasts longer than a window plus a margin
WAIT_CHUNK_S = 20.0  # a wait sleeps in pieces: a stop request is noticed within this time
NOTE_EVERY_S = 600.0  # a progress note to the log while waiting
USAGE_TAIL_BYTES = 512 * 1024  # how much of the end of usage.jsonl is searched for limits
LOCK_TIMEOUT_S = 15.0
# Stages of a build that start agents: the pace is checked before them.
PACED_STAGES = ("extract", "outline", "sections", "global", "coverage")

EventCallback = Callable[[str], None]


class QueueError(Exception):
    """A problem with the queue that has a readable Russian message."""


class QueueBusyError(QueueError):
    """Another runner is processing the queue."""


class _Cancelled(BaseException):
    """Raised from the progress callback of a build between its stages (BaseException: the
    callback of `BuildContext.emit` swallows every `Exception`)."""


class _LimitStop(BaseException):
    """The weekly limit is reached: the queue stops (raised like `_Cancelled`)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------- time helpers


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(value: Any) -> datetime | None:
    """An ISO 8601 string as an aware datetime (naive ones are taken as UTC); None if unusable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def format_wait(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, rest = divmod(total, 3600)
    minutes = rest // 60
    if hours:
        return f"{hours} ч {minutes:02d} мин"
    if minutes:
        return f"{minutes} мин"
    return f"{total} с"


def format_moment(moment: datetime | None) -> str:
    """Local date and time for messages: `04.10 18:01`."""
    return moment.astimezone().strftime("%d.%m %H:%M") if moment else "—"


# ---------------------------------------------------------------- items


@dataclass
class QueueItem:
    id: str
    topic: str  # `<курс>/<тема>` inside the workspaces directory, otherwise an absolute path
    action: str = "build"
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    added: str = ""
    started: str | None = None
    finished: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "action": self.action,
            "params": dict(self.params),
            "status": self.status,
            "added": self.added,
            "started": self.started,
            "finished": self.finished,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> QueueItem | None:
        if not isinstance(raw, dict) or not raw.get("id") or not raw.get("topic"):
            return None
        status = str(raw.get("status") or "queued")
        action = str(raw.get("action") or "build")
        params = raw.get("params")
        return cls(
            id=str(raw["id"]),
            topic=str(raw["topic"]),
            action=action if action in ACTIONS else "build",
            params=dict(params) if isinstance(params, dict) else {},
            status=status if status in STATUSES else "queued",
            added=str(raw.get("added") or ""),
            started=raw.get("started") or None,
            finished=raw.get("finished") or None,
            message=str(raw.get("message") or ""),
        )

    @property
    def runnable(self) -> bool:
        return self.status in RUNNABLE


def normalize_params(action: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """The parameters an item may carry, checked. Raises ValueError (Russian) on a bad one."""
    if action not in ACTIONS:
        raise ValueError(
            f"Неизвестное действие очереди: {action} (допустимо {', '.join(ACTIONS)})."
        )
    raw = dict(params or {})
    backend = raw.get("backend") or None
    if backend not in (None, "claude", "codex"):
        raise ValueError(f"Неизвестный агент «{backend}»: допустимо claude | codex.")
    out: dict[str, Any] = {}
    if backend:
        out["backend"] = backend
    if action == "build":
        from h0lon.synth.model import STAGES

        stage = raw.get("from_stage") or None
        if stage is not None and stage not in STAGES:
            raise ValueError(f"Неизвестная стадия «{stage}»: допустимо {', '.join(STAGES)}.")
        if stage:
            out["from_stage"] = stage
        if "review" in raw:
            out["review"] = bool(raw["review"])
    if action in ("build", "extract", "variant") and raw.get("force"):
        out["force"] = True
    if action == "extract" and "use_vision" in raw:
        out["use_vision"] = bool(raw["use_vision"])
    if action == "variant":
        for key in ("preset", "prompt", "template"):
            if raw.get(key):
                out[key] = str(raw[key])
    return out


# ---------------------------------------------------------------- the file and its locks


def queue_path(settings: Settings) -> Path:
    return Path(settings.general.state_path) / QUEUE_FILE


@dataclass
class _LockState:
    rlock: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0
    fd: int | None = None


_LOCKS: dict[str, _LockState] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _try_lock_fd(fd: int) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock_fd(fd: int) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)
            with contextlib.suppress(OSError):
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _open_lock_file(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    return os.open(path, flags, 0o666)


def _try_acquire(path: Path) -> int | None:
    """The OS lock on `path` or None when somebody else holds it. The OS drops it when the
    holder exits, even if it crashes."""
    fd = _open_lock_file(path)
    if _try_lock_fd(fd):
        return fd
    os.close(fd)
    return None


@contextlib.contextmanager
def _file_locked(path: Path, *, timeout: float = LOCK_TIMEOUT_S) -> Iterator[None]:
    """Exclusive between threads (RLock) and between processes (lock file); re-entrant."""
    with _LOCKS_GUARD:
        state = _LOCKS.setdefault(_lock_key(path), _LockState())
    with state.rlock:
        if state.depth == 0:
            deadline = time.monotonic() + timeout
            delay = 0.01
            while (fd := _try_acquire(path)) is None:
                if time.monotonic() > deadline:
                    raise QueueError(
                        f"Очередь занята другим процессом h0lon дольше {timeout:g} с "
                        f"({path.parent}). Повторите команду позже."
                    )
                time.sleep(delay)
                delay = min(delay * 2, 0.2)
            state.fd = fd
        state.depth += 1
        try:
            yield
        finally:
            state.depth -= 1
            if state.depth == 0 and state.fd is not None:
                fd, state.fd = state.fd, None
                _unlock_fd(fd)


def _replace(src: Path, dst: Path) -> None:
    delay = 0.02
    for attempt in range(12):
        try:
            os.replace(src, dst)
            return
        except PermissionError:  # Windows: a reader has the file open for a moment
            if attempt == 11:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def _read_items(path: Path) -> list[QueueItem]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise QueueError(f"Не удалось прочитать очередь {path}: {exc}") from exc
    if not text.strip():
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QueueError(
            f"Файл очереди повреждён: {path} (строка {exc.lineno}). Исправьте его или удалите."
        ) from exc
    raw_items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        raise QueueError(f"Файл очереди повреждён: {path} (нет списка items).")
    return [item for raw in raw_items if (item := QueueItem.from_dict(raw)) is not None]


def _write_items(path: Path, items: Sequence[QueueItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    payload = {"version": 1, "items": [item.to_dict() for item in items]}
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        _replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _store(settings: Settings) -> Iterator[tuple[list[QueueItem], Callable[[], None]]]:
    """The items under the lock and a `save()` that writes them back."""
    path = queue_path(settings)
    with _file_locked(path.with_name(QUEUE_LOCK_FILE)):
        items = _read_items(path)
        yield items, lambda: _write_items(path, items)


def list_items(settings: Settings) -> list[QueueItem]:
    with _store(settings) as (items, _save):
        return list(items)


def topic_reference(settings: Settings, topic_dir: Path) -> str:
    """How an item names a topic: `<курс>/<тема>` inside the workspaces dir, else the path."""
    topic_dir = Path(topic_dir).resolve()
    try:
        rel = topic_dir.relative_to(Path(settings.general.workspaces_dir).resolve())
    except (ValueError, OSError):
        return str(topic_dir)
    return rel.as_posix()


def resolve_item_topic(settings: Settings, item: QueueItem) -> Path:
    """The directory of the item's topic. FileNotFoundError (Russian) when it is gone."""
    from h0lon.workspace import resolve_topic

    return resolve_topic(settings, item.topic)


@dataclass
class AddReport:
    added: list[QueueItem] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # Russian reasons

    def to_dict(self) -> dict[str, Any]:
        return {"added": [i.to_dict() for i in self.added], "skipped": list(self.skipped)}


def add_items(
    settings: Settings,
    topics: Sequence[str | Path],
    *,
    action: str = "build",
    params: dict[str, Any] | None = None,
    now: Callable[[], datetime] = _utcnow,
) -> AddReport:
    """Queue the topics (references as `h0lon build` takes them). A topic that is already
    waiting for the same action is skipped. Raises ValueError for a bad action or parameter and
    FileNotFoundError for an unknown topic."""
    from h0lon.workspace import resolve_topic

    clean = normalize_params(action, params)
    resolved = [resolve_topic(settings, t) for t in topics]  # all or nothing
    report = AddReport()
    with _store(settings) as (items, save):
        waiting: set[tuple[str, str]] = set()
        for item in items:
            if item.status in ("queued", "running", "paused"):
                try:
                    waiting.add(
                        (os.path.normcase(str(resolve_item_topic(settings, item))), item.action)
                    )
                except FileNotFoundError:
                    continue
        for path in resolved:
            key = (os.path.normcase(str(path)), action)
            if key in waiting:
                report.skipped.append(
                    f"{topic_reference(settings, path)}: уже в очереди ({ACTION_LABELS[action]})."
                )
                continue
            waiting.add(key)
            item = QueueItem(
                id=uuid.uuid4().hex[:8],
                topic=topic_reference(settings, path),
                action=action,
                params=dict(clean),
                added=_iso(now()),
            )
            items.append(item)
            report.added.append(item)
        if report.added:
            save()
    return report


def remove_item(settings: Settings, item_id: str) -> bool:
    """Take an item out of the queue; a running one cannot be removed. False if there is none."""
    with _store(settings) as (items, save):
        for i, item in enumerate(items):
            if item.id == item_id:
                if item.status == "running":
                    raise QueueError("Выполняемый элемент нельзя убрать: остановите очередь.")
                del items[i]
                save()
                return True
    return False


def clear_items(settings: Settings, *, done_only: bool = False) -> int:
    """Remove finished items (`done_only`: only the successful ones) or everything that is not
    running. Returns how many were removed."""
    with _store(settings) as (items, save):
        keep = [
            item
            for item in items
            if item.status == "running" or (done_only and item.status != "done")
        ]
        removed = len(items) - len(keep)
        if removed:
            items[:] = keep
            save()
    return removed


def _update(settings: Settings, item_id: str, **fields: Any) -> QueueItem | None:
    with _store(settings) as (items, save):
        for item in items:
            if item.id == item_id:
                for key, value in fields.items():
                    setattr(item, key, value)
                save()
                return item
    return None


def topic_items(settings: Settings, topic_dir: Path) -> list[QueueItem]:
    """The waiting or running items of one topic (for a topic page)."""
    from h0lon.workspace import resolve_topic

    target = os.path.normcase(str(Path(topic_dir).resolve()))
    found = []
    for item in list_items(settings):
        if item.status not in ("queued", "running", "paused"):
            continue
        try:
            same = os.path.normcase(str(resolve_topic(settings, item.topic))) == target
        except FileNotFoundError:
            continue
        if same:
            found.append(item)
    return found


# ---------------------------------------------------------------- limits of the subscription


@dataclass
class WindowLoad:
    """The last known load of one subscription window."""

    name: str
    utilization: float
    resets_at: datetime | None
    seen_at: datetime | None  # when the attempt that reported it finished

    def active(self, now: datetime) -> bool:
        """False once the window has reset: its old load says nothing about the new one."""
        if self.resets_at is not None:
            return now < self.resets_at
        length = WINDOW_LENGTHS.get(self.name, WINDOW_LENGTHS[WINDOW_5H])
        return self.seen_at is None or now < self.seen_at + length

    def reset_time(self) -> datetime | None:
        if self.resets_at is not None:
            return self.resets_at
        if self.seen_at is not None:
            return self.seen_at + WINDOW_LENGTHS.get(self.name, WINDOW_LENGTHS[WINDOW_5H])
        return None


@dataclass
class LimitsSnapshot:
    windows: dict[str, WindowLoad] = field(default_factory=dict)
    status: str | None = None
    limits: dict[str, Any] | None = None  # the newest `limits` record (for `format_limits`)
    seen_at: datetime | None = None

    @property
    def known(self) -> bool:
        return bool(self.windows)


def _recent_records(state_dir: Path, max_bytes: int | None = None) -> list[dict[str, Any]]:
    """Parseable records from the end of usage.jsonl, the newest first."""
    from h0lon.agents.usage import usage_path

    max_bytes = USAGE_TAIL_BYTES if max_bytes is None else max_bytes
    path = usage_path(state_dir)
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            data = fh.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # the first line is cut
    records: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def read_limits(settings: Settings) -> LimitsSnapshot:
    """The newest known load of every window, from the `limits` of the latest usage records."""
    snapshot = LimitsSnapshot()
    for rec in _recent_records(Path(settings.general.state_path)):
        limits = rec.get("limits")
        if not isinstance(limits, dict) or not isinstance(limits.get("windows"), dict):
            continue
        seen = parse_iso(rec.get("ts"))
        if snapshot.limits is None:
            snapshot.limits, snapshot.status, snapshot.seen_at = limits, limits.get("status"), seen
        for name, win in limits["windows"].items():
            util = win.get("utilization") if isinstance(win, dict) else None
            if name in snapshot.windows or not isinstance(util, int | float):
                continue
            snapshot.windows[name] = WindowLoad(
                str(name), float(util), parse_iso(win.get("resets_at")), seen
            )
        if WINDOW_5H in snapshot.windows and WINDOW_7D in snapshot.windows:
            break
    raw = snapshot.limits
    if raw and raw.get("status") == "rejected" and not raw.get("using_overage"):
        # The window is exhausted whatever its utilization says.
        name = str(raw.get("type") or WINDOW_5H)
        known = snapshot.windows.get(name)
        resets = (known.resets_at if known else None) or parse_iso(raw.get("resets_at"))
        snapshot.windows[name] = WindowLoad(
            name, max(1.0, known.utilization if known else 1.0), resets, snapshot.seen_at
        )
    return snapshot


@dataclass
class Pace:
    action: str  # "go" | "wait" | "stop"
    reason: str = ""
    until: datetime | None = None  # "wait": when to look again


def _percent(value: float) -> int:
    return round(value * 100)


def decide_pace(settings: Settings, snapshot: LimitsSnapshot, now: datetime) -> Pace:
    """Work, wait for the 5-hour window to reset, or stop for the week."""
    cfg = settings.queue
    week = snapshot.windows.get(WINDOW_7D)
    if week is not None and week.active(now) and week.utilization >= cfg.weekly_stop:
        reset = week.reset_time()
        return Pace(
            "stop",
            f"недельный лимит подписки занят на {_percent(week.utilization)} % "
            f"(порог остановки {_percent(cfg.weekly_stop)} %)"
            + (f", сброс {format_moment(reset)}" if reset else ""),
            reset,
        )
    five = snapshot.windows.get(WINDOW_5H)
    if five is not None and five.active(now) and five.utilization >= cfg.pause_at:
        reset = five.reset_time() or now
        until = min(reset + timedelta(seconds=RESET_MARGIN_S), now + timedelta(seconds=MAX_WAIT_S))
        return Pace(
            "wait",
            f"5-часовое окно подписки занято на {_percent(five.utilization)} % "
            f"(порог паузы {_percent(cfg.pause_at)} %), сброс {format_moment(reset)}",
            until,
        )
    return Pace("go")


def limits_summary(settings: Settings, now: datetime | None = None) -> dict[str, Any]:
    """The limits for a page or a command: windows, thresholds and what the queue would do now."""
    from h0lon.agents.base import format_limits

    now = now or _utcnow()
    snapshot = read_limits(settings)
    pace = decide_pace(settings, snapshot, now)
    windows = []
    for name in (WINDOW_5H, WINDOW_7D):
        win = snapshot.windows.get(name)
        if win is None:
            continue
        threshold = settings.queue.pause_at if name == WINDOW_5H else settings.queue.weekly_stop
        active = win.active(now)
        windows.append(
            {
                "name": name,
                "label": WINDOW_LABELS.get(name, name),
                "percent": _percent(win.utilization),
                "threshold": _percent(threshold),
                "resets_at": _iso(win.reset_time()) if win.reset_time() else None,
                "resets": format_moment(win.reset_time()),
                "active": active,
                "hot": active and win.utilization >= threshold,
            }
        )
    return {
        "known": snapshot.known,
        "text": format_limits(snapshot.limits),
        "seen_at": _iso(snapshot.seen_at) if snapshot.seen_at else None,
        "windows": windows,
        "pause_at": _percent(settings.queue.pause_at),
        "weekly_stop": _percent(settings.queue.weekly_stop),
        "action": pace.action,
        "reason": pace.reason,
    }


# ---------------------------------------------------------------- keep awake (Windows)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake(
    enabled: bool = True, *, platform: str | None = None, kernel32: Any = None
) -> Iterator[bool]:
    """Windows: the system does not go to sleep while the block runs (SetThreadExecutionState of
    the calling thread, so the block must stay in one thread). Elsewhere it does nothing.
    Yields True when the request was made."""
    if not enabled or (platform or sys.platform) != "win32":
        yield False
        return
    try:
        if kernel32 is None:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
            kernel32.SetThreadExecutionState.restype = ctypes.c_uint
        applied = bool(kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except (AttributeError, OSError):
        applied = False
    try:
        yield applied
    finally:
        if applied:
            with contextlib.suppress(Exception):
                kernel32.SetThreadExecutionState(ES_CONTINUOUS)


# ---------------------------------------------------------------- the run


@dataclass
class ItemOutcome:
    id: str
    topic: str
    action: str
    status: str  # done | failed | paused | queued (skipped: the topic is busy)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "topic": self.topic,
            "action": self.action,
            "status": self.status,
            "message": self.message,
        }


@dataclass
class QueueRunReport:
    outcomes: list[ItemOutcome] = field(default_factory=list)
    stopped: str | None = None  # None | "limit" | "cancelled" | "error"
    message: str = ""
    waited_s: float = 0.0

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def ok(self) -> bool:
        return self.stopped is None and self.count("failed") == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "stopped": self.stopped,
            "message": self.message,
            "waited_s": round(self.waited_s),
            "done": self.count("done"),
            "failed": self.count("failed"),
            "paused": self.count("paused"),
            "items": [o.to_dict() for o in self.outcomes],
        }


class _Runner:
    def __init__(
        self,
        settings: Settings,
        *,
        unattended: bool,
        on_event: EventCallback | None,
        should_stop: Callable[[], bool] | None,
        claim: Callable[[Path], str | None] | None,
        release: Callable[[Path], None] | None,
        now: Callable[[], datetime],
        sleep: Callable[[float], None],
    ) -> None:
        self.settings = settings
        self.unattended = unattended
        self._on_event = on_event
        self._should_stop = should_stop
        self.claim = claim
        self.release = release
        self.now = now
        self.sleep = sleep
        self.waited_s = 0.0

    # ---- small helpers

    def emit(self, message: str) -> None:
        if self._on_event is not None:
            try:
                self._on_event(message)
            except Exception:  # a broken progress callback must not break the queue
                pass

    def stop_requested(self) -> bool:
        return bool(self._should_stop and self._should_stop())

    # ---- pace

    def gate(self) -> None:
        """Return when the windows allow agent work. Waits for the 5-hour window to reset;
        raises `_LimitStop` (weekly limit) or `_Cancelled` (a stop was requested)."""
        while True:
            if self.stop_requested():
                raise _Cancelled
            now = self.now()
            pace = decide_pace(self.settings, read_limits(self.settings), now)
            if pace.action == "go":
                return
            if pace.action == "stop":
                raise _LimitStop(pace.reason)
            assert pace.until is not None
            self._wait(pace, now)

    def _wait(self, pace: Pace, now: datetime) -> None:
        until = pace.until
        assert until is not None
        self.emit(
            f"Лимит подписки: {pace.reason}. Пауза до {format_moment(until)} "
            f"(через {format_wait((until - now).total_seconds())})."
        )
        started = last_note = self.now()
        while True:
            if self.stop_requested():
                raise _Cancelled
            current = self.now()
            remaining = (until - current).total_seconds()
            if remaining <= 0:
                break
            self.sleep(min(remaining, WAIT_CHUNK_S))
            current = self.now()
            if (current - last_note).total_seconds() >= NOTE_EVERY_S and current < until:
                last_note = current
                left = format_wait((until - current).total_seconds())
                self.emit(f"Ждём сброса окна лимита: осталось {left}.")
        self.waited_s += (self.now() - started).total_seconds()
        with contextlib.suppress(Exception):
            from h0lon.agents import reset_cooling

            reset_cooling()  # an agent that hit the limit is worth trying again
        self.emit("Пауза закончена: продолжаем.")

    # ---- actions

    def execute(self, item: QueueItem, topic_dir: Path) -> tuple[str, str]:
        """(status, message) of one item. May raise `_LimitStop` / `_Cancelled`."""
        if item.action == "build":
            return self._build(item, topic_dir)
        if item.action == "extract":
            return self._extract(item, topic_dir)
        return self._variant(item, topic_dir)

    def _build(self, item: QueueItem, topic_dir: Path) -> tuple[str, str]:
        from h0lon.synth import build as build_mod

        titles = {f"{build_mod.STAGE_TITLES[name]}…": name for name in PACED_STAGES}

        def on_event(message: str) -> None:
            self.emit(message)
            if message in titles:  # between the stages: no stage is interrupted
                self.gate()

        params = item.params
        result = build_mod.build_topic(
            self.settings,
            topic_dir,
            review=bool(params.get("review", True)),
            from_stage=params.get("from_stage") or None,
            force=bool(params.get("force")),
            backend=params.get("backend") or None,
            on_event=on_event,
        )
        if result.ok:
            return "done", result.message
        if result.stopped_at == "review":
            return "paused", result.message
        return "failed", result.message

    def _extract(self, item: QueueItem, topic_dir: Path) -> tuple[str, str]:
        from h0lon.extract.pipeline import extract_topic

        params = item.params
        results = extract_topic(
            self.settings,
            topic_dir,
            force=bool(params.get("force")),
            use_vision=bool(params.get("use_vision", True)),
            backend=params.get("backend") or None,
            on_event=self.emit,
        )
        failed = [r.source_id for r in results if not r.ok]  # type: ignore[union-attr]
        if failed:
            return "failed", "Не удалось извлечь: " + ", ".join(failed) + "."
        return "done", f"Извлечение завершено: источников {len(results)}."

    def _variant(self, item: QueueItem, topic_dir: Path) -> tuple[str, str]:
        from h0lon.synth.variants import run_variant

        params = item.params
        result = run_variant(
            self.settings,
            topic_dir,
            preset=params.get("preset"),
            prompt=params.get("prompt"),
            template=params.get("template"),
            backend=params.get("backend") or None,
            force=bool(params.get("force")),
            on_event=self.emit,
        )
        return ("done" if result.ok else "failed"), result.message()


def _summary(report: QueueRunReport) -> str:
    parts = []
    for status, label in (("done", "готово"), ("failed", "с ошибкой"), ("paused", "на паузе")):
        if report.count(status):
            parts.append(f"{label}: {report.count(status)}")
    skipped = report.count("queued")
    if skipped:
        parts.append(f"пропущено (тема занята): {skipped}")
    text = "Очередь: " + (", ".join(parts) if parts else "обрабатывать нечего") + "."
    if report.waited_s >= 60:
        text += f" Ожидание сброса лимитов: {format_wait(report.waited_s)}."
    return text


def run_queue(
    settings: Settings,
    *,
    unattended: bool | None = None,
    on_event: EventCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
    claim: Callable[[Path], str | None] | None = None,
    release: Callable[[Path], None] | None = None,
    now: Callable[[], datetime] = _utcnow,
    sleep: Callable[[float], None] = time.sleep,
) -> QueueRunReport:
    """Execute the `queued` and `paused` items in order until none is left (see the module doc).

    `should_stop`: polled between items, between the stages of a build and while waiting; true
    stops the queue softly. `claim(topic_dir)` is asked before an item starts: a returned text
    (Russian) means the topic is busy with something else, the item stays queued and is skipped
    in this run; `release(topic_dir)` is called when the item is over. `now` and `sleep` are
    for tests. Raises `QueueBusyError` when another runner is active.
    """
    unattended = bool(unattended) or settings.queue.unattended
    path = queue_path(settings)
    run_lock = path.with_name(RUN_LOCK_FILE)
    fd = _try_acquire(run_lock)
    if fd is None:
        raise QueueBusyError("Очередь уже обрабатывается другим процессом h0lon.")
    runner = _Runner(
        settings,
        unattended=unattended,
        on_event=on_event,
        should_stop=should_stop,
        claim=claim,
        release=release,
        now=now,
        sleep=sleep,
    )
    report = QueueRunReport()
    try:
        with keep_awake(settings.queue.keep_awake and unattended):
            _recover_interrupted(settings, runner)
            _loop(settings, runner, report, now)
    finally:
        _unlock_fd(fd)
    report.waited_s = runner.waited_s
    summary = _summary(report)
    if report.stopped == "limit":
        summary += " Остановлена: " + (report.message or "лимит подписки.")
    elif report.stopped == "cancelled":
        summary += " Остановлена по запросу."
    elif report.stopped == "error":
        summary += " Остановлена после ошибки (--unattended продолжает со следующего элемента)."
    report.message = summary
    return report


def _recover_interrupted(settings: Settings, runner: _Runner) -> None:
    """Items left `running` by a runner that died: queue them again (we hold the run lock)."""
    with _store(settings) as (items, save):
        stale = [item for item in items if item.status == "running"]
        for item in stale:
            item.status = "queued"
            item.message = "Прерван (процесс остановился до конца): будет выполнен заново."
        if stale:
            save()
            runner.emit(f"Незавершённых элементов с прошлого запуска: {len(stale)}; они в очереди.")


def _next_item(settings: Settings, attempted: set[str]) -> QueueItem | None:
    return next((i for i in list_items(settings) if i.runnable and i.id not in attempted), None)


def _loop(
    settings: Settings, runner: _Runner, report: QueueRunReport, now: Callable[[], datetime]
) -> None:
    attempted: set[str] = set()
    while True:
        if runner.stop_requested():
            report.stopped = "cancelled"
            return
        item = _next_item(settings, attempted)
        if item is None:
            return
        attempted.add(item.id)
        label = f"{item.topic} ({ACTION_LABELS.get(item.action, item.action)})"

        def finish(status: str, message: str, item: QueueItem = item) -> ItemOutcome:
            _update(settings, item.id, status=status, message=message, finished=_iso(now()))
            return ItemOutcome(item.id, item.topic, item.action, status, message)

        try:
            topic_dir = resolve_item_topic(settings, item)
        except FileNotFoundError as exc:
            runner.emit(f"{label}: {exc}")
            report.outcomes.append(finish("failed", str(exc)))
            if not runner.unattended:
                report.stopped = "error"
                return
            continue
        busy = runner.claim(topic_dir) if runner.claim else None
        if busy:
            runner.emit(f"{label}: пропущено — {busy}")
            _update(settings, item.id, message=f"Пропущено: {busy}")
            report.outcomes.append(ItemOutcome(item.id, item.topic, item.action, "queued", busy))
            continue
        try:
            try:
                runner.gate()
            except _LimitStop as stop:
                message = f"Не запущено: {stop.reason}."
                runner.emit(f"{label}: {message}")
                report.outcomes.append(finish("paused", message))
                report.stopped, report.message = "limit", stop.reason
                return
            except _Cancelled:
                report.stopped = "cancelled"
                return
            _update(
                settings,
                item.id,
                status="running",
                started=_iso(now()),
                finished=None,
                message="Выполняется.",
            )
            runner.emit(f"Очередь: {label} — начало.")
            try:
                status, message = runner.execute(item, topic_dir)
            except _LimitStop as stop:
                message = (
                    f"Остановлено между стадиями: {stop.reason}. "
                    "Готовые стадии сохранены, сборка продолжится с этого места."
                )
                runner.emit(f"{label}: {message}")
                report.outcomes.append(finish("paused", message))
                report.stopped, report.message = "limit", stop.reason
                return
            except _Cancelled:
                message = (
                    "Остановлено по запросу между стадиями; продолжится со следующего запуска."
                )
                report.outcomes.append(finish("paused", message))
                report.stopped = "cancelled"
                return
            except (ValueError, FileNotFoundError, OSError) as exc:
                status, message = "failed", str(exc) or type(exc).__name__
            except Exception as exc:  # a bug in one item must not lose the others
                status = "failed"
                message = f"Внутренняя ошибка: {type(exc).__name__}: {exc}"
            except BaseException:  # Ctrl+C: leave the item resumable
                finish("paused", "Прервано пользователем; продолжится со следующего запуска.")
                raise
            runner.emit(f"{label}: {STATUS_LABELS[status]} — {message}")
            report.outcomes.append(finish(status, message))
            if status == "failed" and not runner.unattended:
                report.stopped = "error"
                return
        finally:
            if runner.release is not None:
                runner.release(topic_dir)


# ---------------------------------------------------------------- printing

_STATUS_STYLE = {
    "queued": "cyan",
    "running": "yellow",
    "done": "green",
    "failed": "red",
    "paused": "yellow",
}


def print_queue(items: Sequence[QueueItem], *, console: Console, settings: Settings) -> None:
    from rich.markup import escape
    from rich.table import Table

    if not items:
        console.print("Очередь пуста: h0lon queue add <тема>")
    else:
        table = Table(title="Очередь сборок", title_justify="left", show_lines=False)
        for col in ("№", "Тема", "Действие", "Статус", "Добавлено", "Сообщение"):
            table.add_column(col, overflow="fold")
        for n, item in enumerate(items, 1):
            style = _STATUS_STYLE.get(item.status, "")
            label = STATUS_LABELS.get(item.status, item.status)
            table.add_row(
                str(n),
                escape(item.topic),
                ACTION_LABELS.get(item.action, item.action),
                f"[{style}]{label}[/{style}]" if style else label,
                escape(item.added.replace("T", " ").removesuffix("Z")[:16]),
                escape(item.message[:200]),
            )
        console.print(table)
    info = limits_summary(settings)
    if info["known"]:
        console.print(f"Лимиты Claude: {escape(str(info['text']))}")
        hint = {"wait": "пауза до сброса окна", "stop": "очередь остановится"}.get(info["action"])
        if hint:
            console.print(f"[yellow]Сейчас: {hint} — {escape(info['reason'])}[/yellow]")
        console.print(
            f"[dim]Пороги: пауза при {info['pause_at']} % 5-часового окна, остановка при "
            f"{info['weekly_stop']} % недельного (queue.pause_at, queue.weekly_stop).[/dim]"
        )
    else:
        console.print("[dim]Данных о лимитах подписки пока нет: очередь работает без пауз.[/dim]")


def print_run(report: QueueRunReport, *, console: Console) -> None:
    from rich.markup import escape
    from rich.table import Table

    if report.outcomes:
        table = Table(title="Итог запуска очереди", title_justify="left", show_lines=False)
        for col in ("Тема", "Действие", "Итог", "Сообщение"):
            table.add_column(col, overflow="fold")
        for o in report.outcomes:
            style = _STATUS_STYLE.get(o.status, "")
            label = STATUS_LABELS.get(o.status, o.status)
            table.add_row(
                escape(o.topic),
                ACTION_LABELS.get(o.action, o.action),
                f"[{style}]{label}[/{style}]" if style else label,
                escape(o.message[:300]),
            )
        console.print(table)
    style = "bold green" if report.ok else "bold red" if report.count("failed") else "bold yellow"
    console.print(f"[{style}]{escape(report.message)}[/{style}]")
