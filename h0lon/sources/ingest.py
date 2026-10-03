"""Ingest of sources into a topic: copy into sources/, hash, detect the kind, record in topic.yaml.

Contract: docs/ARCHITECTURE.md, «Приём». Decisions not spelled out there:

- Arguments are validated before anything is copied: an empty list, an unknown ``--kind``,
  a missing file, an explicitly named file of an unsupported type, a link that does not
  parse (bad port, broken IPv6 literal, no host), ``--title`` for several items raise
  `IngestError` and leave the topic untouched.
- Problems of the content are warnings, and the item is skipped while the others are added:
  a duplicate (same sha256 or the same URL; the port is part of a URL), a corrupt or
  password-protected PDF, a failed download, a directory. A PDF that MuPDF had to repair
  (typically a truncated download) is added with a warning and a note in `quality`.
- ``--kind`` for a file that is already in the topic under another kind changes the kind of
  that record instead of skipping it (see `_rekind`): the record is reported in
  `IngestReport.added` together with a warning that explains the change. An extracted
  source is not changed; the warning says how to re-add it.
- Directories are not expanded (warning «каталоги не поддерживаются …»): a folder of phone
  photos would turn into dozens of separate handwritten sources, and a folder usually holds
  files that are not materials. Wildcards (``папка\\*.pdf``) are expanded here because
  PowerShell and cmd do not expand them for native programs; only ``*`` and ``?`` are
  wildcards, brackets are literal («[2026] Лекции\\*.pdf» works). Unsupported files matched
  by a wildcard are skipped with a warning. Matches are sorted naturally (2 before 10).
- Text files in UTF-16/UTF-32 (PowerShell 5.1 ``>``, «Unicode» in old Notepad) are copied
  as UTF-8; `sha256` and `size` are those of the UTF-8 copy, `quality.notes` says so.
- Ids: the letter comes from `ID_PREFIX`, the number is one more than the largest ever used
  for the letter — in records, in the `source_counters` key of topic.yaml (kept when a
  record is deleted by hand), in sources/ file names and in extracted/ directories — so a
  deleted number is never reused and stale extraction caches are never picked up.
- Each item is committed on its own (rename of the copy + topic.yaml write under a lock), so
  a failure in the middle keeps what was added. Writes of topic.yaml are atomic (temporary
  file + os.replace). Every read-modify-write of topic.yaml (add, `update_source`,
  `list_sources`) holds `topic_lock`: a thread lock plus an OS file lock on
  ``.topic.yaml.lock`` in the topic, so two terminals, or ``extract`` and ``add`` in
  different processes, do not lose each other's records.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import os
import re
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit, urlunsplit

from pydantic import ValidationError
from rich.table import Table
from rich.text import Text

from h0lon.names import slugify
from h0lon.sources import detect
from h0lon.sources.detect import IS_WINDOWS, InspectError, os_path
from h0lon.sources.models import ID_PREFIX, SOURCE_KINDS, SourceKind, SourceRecord
from h0lon.workspace import TOPIC_FILE, TopicMeta, load_topic, save_topic

if TYPE_CHECKING:
    from rich.console import Console

    from h0lon.config import Settings

SOURCES_DIR = "sources"
COUNTERS_KEY = "source_counters"  # topic.yaml: letter -> largest number ever issued
SLUG_MAX_LEN = 60
_INCOMING_PREFIX = ".incoming-"
_INCOMING_MAX_AGE_S = 24 * 3600
_COPY_CHUNK = 1024 * 1024
_REPLACE_RETRIES = 20  # os.replace on Windows fails while another process reads the file
_GLOB_CHARS = ("*", "?")


class IngestError(Exception):
    """Invalid ingest request: empty list, missing file, unknown --kind, unsupported type."""


@dataclass
class IngestReport:
    added: list[SourceRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": [r.model_dump(mode="json") for r in self.added],
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------- locking and topic.yaml

LOCK_FILE = ".topic.yaml.lock"
LOCK_TIMEOUT_S = 30.0  # holders keep the lock for one read-modify-write: milliseconds
_LOCK_DELETE_WAIT_S = 1.0


@dataclass
class _TopicLockState:
    rlock: threading.RLock = field(default_factory=threading.RLock)
    depth: int = 0
    fd: int | None = None


_LOCKS: dict[str, _TopicLockState] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_key(topic_dir: Path) -> str:
    return os.path.normcase(str(Path(topic_dir).resolve()))


def _try_lock_fd(fd: int) -> bool:
    if IS_WINDOWS:
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
    except BlockingIOError:
        return False
    return True


def _acquire_file_lock(path: Path, timeout: float) -> int | None:
    """Exclusive OS lock on `path` (created if needed) shared by all h0lon processes.

    The OS drops the lock when the holder exits, even if it crashes. Returns None when the
    lock file cannot be created at all (no topic directory, read-only directory): there is
    nothing to protect, and the caller's own read or write reports the real problem.
    """
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    started = time.monotonic()
    deadline = started + timeout
    delay = 0.01
    while True:
        fd: int | None = None
        try:
            fd = os.open(os_path(path), flags, 0o666)
        except PermissionError:
            # Windows: the previous holder is deleting the file (a moment). Anything longer
            # is a directory we may not write to.
            if not IS_WINDOWS or time.monotonic() - started > _LOCK_DELETE_WAIT_S:
                return None
        except OSError:
            return None
        if fd is not None:
            if _try_lock_fd(fd):
                return fd
            os.close(fd)
        if time.monotonic() > deadline:
            raise IngestError(
                f"Тема занята другим процессом h0lon: {TOPIC_FILE} заблокирован дольше "
                f"{timeout:g} с ({path.parent}). Повторите команду позже."
            )
        time.sleep(delay)
        delay = min(delay * 2, 0.2)


def _release_file_lock(fd: int, path: Path) -> None:
    try:
        if IS_WINDOWS:
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
    if IS_WINDOWS:
        # Keep the topic directory clean. Safe on Windows only: deleting fails while any
        # other process has the file open (waiting for the lock), so nobody locks a file
        # that is being removed. On POSIX the file stays (unlink would split the lock).
        with contextlib.suppress(OSError):
            os.unlink(os_path(path))


@contextlib.contextmanager
def topic_lock(topic_dir: Path) -> Iterator[None]:
    """Serialise read-modify-write of one topic.yaml: re-entrant within a thread, exclusive
    between threads (RLock) and between processes (OS lock on `LOCK_FILE` in the topic)."""
    key = _lock_key(topic_dir)
    with _LOCKS_GUARD:
        state = _LOCKS.setdefault(key, _TopicLockState())
    lock_path = Path(topic_dir) / LOCK_FILE
    with state.rlock:
        if state.depth == 0:
            state.fd = _acquire_file_lock(lock_path, LOCK_TIMEOUT_S)
        state.depth += 1
        try:
            yield
        finally:
            state.depth -= 1
            if state.depth == 0 and state.fd is not None:
                fd, state.fd = state.fd, None
                _release_file_lock(fd, lock_path)


def _replace(src: str, dst: str) -> None:
    delay = 0.02
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def _write_topic(topic_dir: Path, meta: TopicMeta) -> None:
    """Atomic topic.yaml write: save_topic into a temporary file next to it, then replace."""
    target = Path(topic_dir) / TOPIC_FILE
    tmp = target.with_name(f".{TOPIC_FILE}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        save_topic(tmp, meta)
        _replace(os_path(tmp), os_path(target))
    finally:
        with contextlib.suppress(OSError):
            os.unlink(os_path(tmp))


def _load(topic_dir: Path) -> TopicMeta:
    try:
        return load_topic(topic_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise IngestError(str(exc)) from exc


def list_sources(topic_dir: Path) -> list[SourceRecord]:
    """Records of topic.yaml in their order. A malformed record raises ValueError."""
    with topic_lock(Path(topic_dir)):
        meta = load_topic(Path(topic_dir))
    records: list[SourceRecord] = []
    for n, raw in enumerate(meta.sources, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{TOPIC_FILE}: источник №{n} записан не словарём: {raw!r}")
        try:
            records.append(SourceRecord.model_validate(raw))
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ())) or "?"
            raise ValueError(
                f"{TOPIC_FILE}: источник №{n} ({raw.get('id', 'без id')}) некорректен: "
                f"поле {where} — {first.get('msg')}"
            ) from exc
    return records


def update_source(topic_dir: Path, record: SourceRecord) -> None:
    """Replace the record with the same id in topic.yaml, atomically.

    Fields of the stored record that `record` does not know (written by other stages) are
    kept; every field of `record` — including None values — wins. Other records and topic
    fields are re-read under the lock, so concurrent updates of different records within
    one process do not lose each other. Raises KeyError if there is no such id.
    """
    topic_dir = Path(topic_dir)
    new = record.model_dump(mode="json", exclude_none=False)
    with topic_lock(topic_dir):
        meta = load_topic(topic_dir)
        for i, stored in enumerate(meta.sources):
            if isinstance(stored, dict) and stored.get("id") == record.id:
                meta.sources[i] = {**stored, **new}
                break
        else:
            raise KeyError(f"Источник {record.id} не найден в {TOPIC_FILE}: {topic_dir}")
        _write_topic(topic_dir, meta)


# ---------------------------------------------------------------- ids

_ID_RE = re.compile(r"^([A-Z])(\d+)$")
_FILE_ID_RE = re.compile(r"^([A-Z])(\d+)_")


def _counters(meta: TopicMeta) -> dict[str, int]:
    raw = (meta.model_extra or {}).get(COUNTERS_KEY)
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            with contextlib.suppress(TypeError, ValueError):
                out[str(k)] = int(v)
    return out


def _used_numbers(meta: TopicMeta, topic_dir: Path, letter: str) -> int:
    largest = _counters(meta).get(letter, 0)
    for raw in meta.sources:
        m = _ID_RE.match(str(raw.get("id", ""))) if isinstance(raw, dict) else None
        if m and m.group(1) == letter:
            largest = max(largest, int(m.group(2)))
    for sub, pattern in ((SOURCES_DIR, _FILE_ID_RE), ("extracted", _ID_RE)):
        with contextlib.suppress(OSError):
            for name in os.listdir(os_path(topic_dir / sub)):
                m = pattern.match(name)
                if m and m.group(1) == letter:
                    largest = max(largest, int(m.group(2)))
    return largest


def _issue_id(meta: TopicMeta, topic_dir: Path, kind: str) -> str:
    letter = ID_PREFIX[kind]
    number = _used_numbers(meta, topic_dir, letter) + 1
    counters = _counters(meta)
    counters[letter] = number
    setattr(meta, COUNTERS_KEY, dict(sorted(counters.items())))
    return f"{letter}{number}"


# ---------------------------------------------------------------- planning


@dataclass
class _Item:
    raw: str
    url: str | None = None
    path: Path | None = None
    fmt: str | None = None


def _natural_key(path: Path) -> list[Any]:
    return [int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", path.name)]


def _clean_item(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _unsupported_reason(path: Path) -> str:
    suffix = detect.real_suffix(path)
    hint = detect.UNSUPPORTED_HINTS.get(suffix)
    if hint:
        return f"формат {suffix} не поддерживается — {hint}"
    shown = f"«{suffix}»" if suffix else "без расширения"
    return (
        f"не удалось определить вид файла ({shown}); "
        "укажите --kind или сохраните в поддерживаемом формате"
    )


def _plan(items: Sequence[str], kind: str | None, report: IngestReport) -> list[_Item]:
    planned: list[_Item] = []
    missing: list[str] = []
    unsupported: list[str] = []
    for raw in items:
        text = _clean_item(raw)
        if not text:
            continue
        if detect.is_url(text):
            problem = detect.url_problem(text)
            if problem:
                raise IngestError(f"Некорректная ссылка: {text} — {problem}")
            if kind is not None and kind not in ("web", "video", "audio"):
                raise IngestError(
                    f"Для ссылки допустим --kind web, video или audio, а не «{kind}»: {text}"
                )
            planned.append(_Item(raw=text, url=text))
            continue
        if detect.SCHEME_RE.match(text):
            raise IngestError(f"Поддерживаются только ссылки http:// и https://: {text}")
        path = Path(os.path.expanduser(text))
        native = os_path(path)
        if os.path.isdir(native):
            report.warnings.append(
                f"{text}: каталоги не поддерживаются — перечислите файлы "
                f"или используйте шаблон, например «{path / '*.pdf'}»"
            )
            continue
        if os.path.isfile(native):
            fmt = detect.file_format(path) if kind is None else _format_or_raw(path)
            if fmt is None:
                unsupported.append(f"{text}: {_unsupported_reason(path)}")
                continue
            planned.append(_Item(raw=text, path=path, fmt=fmt))
            continue
        if any(ch in text for ch in _GLOB_CHARS):
            matches = sorted(
                (Path(m) for m in glob.glob(_glob_pattern(text)) if os.path.isfile(m)),
                key=_natural_key,
            )
            if not matches:
                missing.append(f"{text} (по шаблону ничего не найдено)")
                continue
            for match in matches:
                fmt = detect.file_format(match) if kind is None else _format_or_raw(match)
                if fmt is None:
                    report.warnings.append(f"{match}: {_unsupported_reason(match)} — пропущен")
                    continue
                planned.append(_Item(raw=str(match), path=match, fmt=fmt))
            continue
        hint = ""
        if text.lower().startswith("www."):
            hint = " (ссылку укажите полностью: https://…)"
        missing.append(text + hint)
    if missing:
        label = "Файл не найден" if len(missing) == 1 else "Файлы не найдены"
        raise IngestError(f"{label}: " + "; ".join(missing))
    if unsupported:
        raise IngestError("; ".join(unsupported))
    return planned


def _glob_pattern(text: str) -> str:
    """Only ``*`` and ``?`` are wildcards: brackets are literal, because folders and files
    like «[2026] Лекции» are common and nobody types character classes in PowerShell."""
    return os.path.expanduser(text).replace("[", "[[]")


def _format_or_raw(path: Path) -> str:
    """With --kind the type is the user's call; the format still decides how to profile."""
    return detect.file_format(path) or "raw"


# ---------------------------------------------------------------- copying


def _copy_hashed(src: Path, dst: Path) -> tuple[str, int]:
    """Stream-copy src to a new file dst (never copying attributes such as read-only)."""
    digest = hashlib.sha256()
    size = 0
    with open(os_path(src), "rb") as fin, open(os_path(dst), "xb") as fout:
        while chunk := fin.read(_COPY_CHUNK):
            digest.update(chunk)
            fout.write(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _write_hashed(data: bytes, dst: Path) -> tuple[str, int]:
    with open(os_path(dst), "xb") as fout:
        fout.write(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _cleanup_incoming(sources_dir: Path) -> None:
    """Remove copies left by an interrupted earlier run (older than a day)."""
    now = time.time()
    with contextlib.suppress(OSError):
        for entry in os.scandir(os_path(sources_dir)):
            if entry.name.startswith(_INCOMING_PREFIX) and entry.is_file():
                with contextlib.suppress(OSError):
                    if now - entry.stat().st_mtime > _INCOMING_MAX_AGE_S:
                        os.unlink(entry.path)


def _is_extension(suffix: str, ext: str | None) -> bool:
    """Whether a dotted tail of a file name is its extension rather than part of the name.

    «Лекция 2024.10.03», «Report v1.2» or «Lecture.final» (a PDF without an extension) keep
    their tails; known extensions and the extension the copy gets are cut off.
    """
    if not suffix or suffix[1:].isdigit():
        return False
    return (
        suffix in detect.EXT_FORMAT
        or suffix in detect.UNSUPPORTED_HINTS
        or suffix == (ext or "").lower()
    )


def _human_title(name: str, ext: str | None = None) -> str:
    """Title from a file name: without its extension (see `_is_extension`), «_» as spaces."""
    stem = name
    suffix = detect.real_suffix(name)
    if _is_extension(suffix, ext):
        stem = name[: -len(suffix)]
    title = " ".join(stem.replace("_", " ").split())
    return title or name


def _find_duplicate(meta: TopicMeta, *, sha256: str | None = None, url: str | None = None):
    for raw in meta.sources:
        if not isinstance(raw, dict):
            continue
        if sha256 and raw.get("sha256") == sha256:
            return raw
        if url and _url_key(str(raw.get("url") or "")) == _url_key(url):
            return raw
    return None


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _url_key(url: str) -> str:
    """Comparison key of a link: no scheme, no www., no fragment, no trailing slash; the
    port is kept unless it is the scheme's default."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:  # a hand-edited record with a broken link: compare it verbatim
        return url.strip()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    if port is not None and port != _DEFAULT_PORTS.get(parts.scheme.lower()):
        host += f":{port}"
    path = unquote(parts.path).rstrip("/")
    return urlunsplit(("", host, path, unquote(parts.query), ""))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dup_warning(label: str, dup: dict[str, Any]) -> str:
    return f"{label}: уже добавлен как {dup.get('id')} («{dup.get('title')}») — пропущен"


def _final_path(sources_dir: Path, source_id: str, slug_source: str, ext: str) -> Path:
    slug = slugify(slug_source, max_len=SLUG_MAX_LEN)
    final = sources_dir / f"{source_id}_{slug}{ext}"
    if os.path.exists(os_path(final)):  # a stray file nobody recorded: keep it
        final = sources_dir / f"{source_id}_{slug}-{uuid.uuid4().hex[:6]}{ext}"
    return final


def _commit(
    topic_dir: Path,
    *,
    kind: str,
    staged: Path | None,
    ext: str,
    slug_source: str,
    fields: dict[str, Any],
    label: str,
    report: IngestReport,
) -> SourceRecord | None:
    """Under the topic lock: re-check duplicates, issue the id, move the copy, write yaml."""
    sources_dir = topic_dir / SOURCES_DIR
    with topic_lock(topic_dir):
        meta = _load(topic_dir)
        dup = _find_duplicate(meta, sha256=fields.get("sha256"), url=fields.get("url"))
        if dup is not None:
            report.warnings.append(_dup_warning(label, dup))
            return None
        source_id = _issue_id(meta, topic_dir, kind)
        final: Path | None = None
        if staged is not None:
            final = _final_path(sources_dir, source_id, slug_source, ext)
            try:
                _replace(os_path(staged), os_path(final))
            except OSError as exc:
                report.warnings.append(f"{label}: не удалось сохранить копию ({exc}) — пропущен")
                return None
        record = SourceRecord(
            id=source_id,
            kind=kind,  # type: ignore[arg-type]
            file=f"{SOURCES_DIR}/{final.name}" if final is not None else None,
            added=_now(),
            **fields,
        )
        meta.sources.append(record.model_dump(mode="json", exclude_none=False))
        try:
            _write_topic(topic_dir, meta)
        except OSError as exc:
            if final is not None:
                with contextlib.suppress(OSError):
                    os.unlink(os_path(final))
            raise IngestError(f"Не удалось записать {TOPIC_FILE}: {exc}") from exc
        report.added.append(record)
        return record


# A kind can be changed only before a successful extraction: extracted/<id> and the review
# of the topic refer to the old id and kind.
_REKIND_BLOCKED = {"extracting": "сейчас извлекается", "extracted": "уже извлечён"}


def _rekind(
    topic_dir: Path,
    *,
    sha256: str,
    kind: str,
    info: detect.Inspection,
    staged: Path,
    ext: str,
    slug_source: str,
    title: str | None,
    label: str,
    report: IngestReport,
) -> bool:
    """`add --kind X` for a file that is already in the topic under another kind.

    The record keeps its place, title (unless --title is given) and «added» time; it gets
    the new kind, fresh ingest signals and status «added». When the id letter changes
    (pdf-scan P1 → handwritten H1), a new number is issued (the old one is never reused)
    and the copy in sources/ is renamed. Refused with a hint for an extracted source.
    Returns False if the record disappeared meanwhile (the caller adds the file anew).
    """
    sources_dir = topic_dir / SOURCES_DIR
    with topic_lock(topic_dir):
        meta = _load(topic_dir)
        index = next(
            (
                i
                for i, raw in enumerate(meta.sources)
                if isinstance(raw, dict) and raw.get("sha256") == sha256
            ),
            None,
        )
        if index is None:
            return False
        stored = meta.sources[index]
        old_id, old_kind = str(stored.get("id")), str(stored.get("kind"))
        if old_kind == kind:
            report.warnings.append(_dup_warning(label, stored))
            return True
        status = str(stored.get("status") or "added")
        if status in _REKIND_BLOCKED:
            report.warnings.append(
                f"{label}: уже добавлен как {old_id} (вид {old_kind}) и {_REKIND_BLOCKED[status]}"
                f" — вид не изменён. Чтобы сменить вид на {kind}, удалите запись {old_id} из "
                f"{TOPIC_FILE} и файл {stored.get('file')}, затем добавьте файл снова "
                f"с --kind {kind}"
            )
            return True
        new_id = old_id
        final: Path | None = None
        if ID_PREFIX.get(old_kind) != ID_PREFIX[kind]:
            new_id = _issue_id(meta, topic_dir, kind)
            final = _final_path(sources_dir, new_id, slug_source, ext)
            try:
                _replace(os_path(staged), os_path(final))
            except OSError as exc:
                report.warnings.append(f"{label}: не удалось сохранить копию ({exc}) — пропущен")
                return True
        updated = {
            **stored,
            "id": new_id,
            "kind": kind,
            "units": info.units,
            "quality": info.quality,
            "status": "added",
            "error": None,
            "extracted_key": None,
        }
        if final is not None:
            updated["file"] = f"{SOURCES_DIR}/{final.name}"
        if title:
            updated["title"] = title
        try:
            record = SourceRecord.model_validate(updated)
        except ValidationError as exc:
            if final is not None:
                with contextlib.suppress(OSError):
                    os.unlink(os_path(final))
            report.warnings.append(
                f"{label}: запись {old_id} в {TOPIC_FILE} некорректна, вид не изменён ({exc})"
            )
            return True
        meta.sources[index] = record.model_dump(mode="json", exclude_none=False)
        try:
            _write_topic(topic_dir, meta)
        except OSError as exc:
            if final is not None:
                with contextlib.suppress(OSError):
                    os.unlink(os_path(final))
            raise IngestError(f"Не удалось записать {TOPIC_FILE}: {exc}") from exc
        old_file = stored.get("file")
        if final is not None and isinstance(old_file, str) and old_file:
            old_path = topic_dir / old_file
            if old_path.parent == sources_dir and old_path != final:
                with contextlib.suppress(OSError):
                    os.unlink(os_path(old_path))
    report.added.append(record)
    change = f"вид изменён: {old_kind} → {kind}"
    if new_id != old_id:
        change += f", новый id {new_id}"
    report.warnings.append(f"{label}: уже был добавлен как {old_id} — {change}")
    return True


_TEXT_FORMATS = ("md", "tex", "html")
_CODEC_NAMES = {
    "utf-16": "UTF-16",
    "utf-16-le": "UTF-16 LE",
    "utf-16-be": "UTF-16 BE",
    "utf-32": "UTF-32",
}


def _to_utf8_copy(staged: Path, fmt: str) -> tuple[str, int, str] | None:
    """Rewrite a UTF-16/UTF-32 text copy as UTF-8 (Pandoc and the extractors read UTF-8).

    Returns (sha256, size, readable codec name) of the rewritten copy, or None when the
    copy needs nothing. Raises InspectError for NUL bytes that are not UTF-16 text.
    """
    with open(os_path(staged), "rb") as fh:
        head = fh.read(8192)
    codec = detect.wide_text_codec(head)
    if codec is None:
        return None
    if not codec:
        raise InspectError("в файле есть нулевые байты — это не текст; сохраните его в UTF-8")
    with open(os_path(staged), "rb") as fh:
        data = fh.read()
    name = _CODEC_NAMES.get(codec, codec)
    try:
        text = data.decode(codec).lstrip("﻿")
    except UnicodeDecodeError as exc:
        raise InspectError(
            f"файл похож на текст в {name}, но не читается в этой кодировке — сохраните его в UTF-8"
        ) from exc
    if fmt == "html":
        from h0lon.sources.fetch import to_utf8_html

        out = to_utf8_html(text)
    else:
        out = text.encode("utf-8")
    with open(os_path(staged), "wb") as fh:
        fh.write(out)
    return hashlib.sha256(out).hexdigest(), len(out), name


def _add_note(quality: dict[str, Any], note: str) -> None:
    notes = quality.get("notes")
    quality["notes"] = [*notes, note] if isinstance(notes, list) else [note]


def _ingest_file(
    topic_dir: Path, item: _Item, *, kind: str | None, title: str | None, report: IngestReport
) -> None:
    assert item.path is not None and item.fmt is not None
    src = item.path
    label = src.name
    sources_dir = topic_dir / SOURCES_DIR
    if item.fmt != "raw":
        ext = detect.copy_ext(src, item.fmt)
    else:
        suffix = detect.real_suffix(src)
        ext = "" if suffix[1:].isdigit() else suffix
    staged = sources_dir / f"{_INCOMING_PREFIX}{uuid.uuid4().hex}{ext}"
    name_title = _human_title(src.name, ext)
    try:
        try:
            sha256, size = _copy_hashed(src, staged)
        except OSError as exc:
            report.warnings.append(f"{label}: не удалось скопировать ({exc}) — пропущен")
            return
        if size == 0:
            report.warnings.append(f"{label}: файл пустой — пропущен")
            return
        transcoded: str | None = None
        if item.fmt in _TEXT_FORMATS:
            try:
                rewritten = _to_utf8_copy(staged, item.fmt)
            except (InspectError, OSError) as exc:
                report.warnings.append(f"{label}: {exc} — пропущен")
                return
            if rewritten is not None:
                sha256, size, transcoded = rewritten
        with topic_lock(topic_dir):
            dup = _find_duplicate(_load(topic_dir), sha256=sha256)
        if dup is not None and (kind is None or dup.get("kind") == kind):
            report.warnings.append(_dup_warning(label, dup))
            return
        if item.fmt == "raw":
            assert kind is not None
            info = detect.Inspection(fmt="raw", kind=kind)  # type: ignore[arg-type]
        else:
            try:
                info = detect.inspect_file(staged, item.fmt, kind=kind)
            except InspectError as exc:
                report.warnings.append(f"{label}: {exc} — пропущен")
                return
        if transcoded:
            info.quality["source_encoding"] = transcoded
            _add_note(
                info.quality,
                f"Файл был в кодировке {transcoded}: копия в sources/ перекодирована в UTF-8.",
            )
        if dup is not None:
            assert kind is not None
            if _rekind(
                topic_dir,
                sha256=sha256,
                kind=kind,
                info=info,
                staged=staged,
                ext=ext,
                slug_source=name_title,
                title=title,
                label=label,
                report=report,
            ):
                return
        record = _commit(
            topic_dir,
            kind=info.kind,
            staged=staged,
            ext=ext,
            slug_source=name_title,
            fields={
                "title": title or info.title or name_title,
                "original_name": src.name,
                "sha256": sha256,
                "size": size,
                "units": info.units,
                "quality": info.quality,
            },
            label=label,
            report=report,
        )
        if record is not None:
            report.warnings.extend(f"{label} ({record.id}): {w}" for w in info.warnings)
    finally:
        with contextlib.suppress(OSError):
            if os.path.exists(os_path(staged)):
                os.unlink(os_path(staged))


def _ingest_url(
    topic_dir: Path, item: _Item, *, kind: str | None, title: str | None, report: IngestReport
) -> None:
    from h0lon.sources.fetch import FetchError, fetch_html, iri_to_uri

    assert item.url is not None
    url = item.url
    with topic_lock(topic_dir):
        dup = _find_duplicate(_load(topic_dir), url=url)
    if dup is not None:
        report.warnings.append(_dup_warning(url, dup))
        return
    source_kind = kind or detect.classify_url(url)
    if source_kind != "web":
        # Videos and audio by link are downloaded in M5; now only the record is kept.
        _commit(
            topic_dir,
            kind=source_kind,
            staged=None,
            ext="",
            slug_source="",
            fields={"title": title or detect.url_display(url), "url": url},
            label=url,
            report=report,
        )
        return
    try:
        page = fetch_html(url)
    except FetchError as exc:
        report.warnings.append(f"{url}: не удалось скачать — {exc}. Источник не добавлен")
        return
    sources_dir = topic_dir / SOURCES_DIR
    staged = sources_dir / f"{_INCOMING_PREFIX}{uuid.uuid4().hex}.html"
    try:
        sha256, size = _write_hashed(page.utf8_bytes(), staged)
        source_title = title or page.title or detect.url_display(url)
        quality: dict[str, Any] = {"encoding": page.encoding}
        if page.replaced:
            quality["replaced_bytes"] = page.replaced
            _add_note(
                quality,
                f"Страница декодирована как {page.encoding} с заменой повреждённых байтов "
                f"(их {page.replaced}): в этих местах текста стоит знак «�».",
            )
        elif page.guessed and page.encoding != "utf-8":
            _add_note(
                quality,
                f"Сайт не указал кодировку, она определена как {page.encoding} — проверьте текст.",
            )
        fields: dict[str, Any] = {
            "title": source_title,
            "url": url,
            "sha256": sha256,
            "size": size,
            "quality": quality,
        }
        if page.final_url and _url_key(page.final_url) != _url_key(iri_to_uri(url)):
            fields["final_url"] = page.final_url
        _commit(
            topic_dir,
            kind="web",
            staged=staged,
            ext=".html",
            slug_source=page.title or detect.url_display(url),
            fields=fields,
            label=url,
            report=report,
        )
    finally:
        with contextlib.suppress(OSError):
            if os.path.exists(os_path(staged)):
                os.unlink(os_path(staged))


# ---------------------------------------------------------------- public API


def detect_kind(path_or_url: str) -> SourceKind:
    """Kind of a file or link by extension, content (PDF profile, magic bytes) and URL.

    Raises IngestError for a missing file, a directory, an unrecognised type or a link
    that does not parse.
    """
    text = _clean_item(path_or_url)
    if detect.is_url(text):
        problem = detect.url_problem(text)
        if problem:
            raise IngestError(f"Некорректная ссылка: {text} — {problem}")
        return detect.classify_url(text)
    path = Path(os.path.expanduser(text))
    native = os_path(path)
    if os.path.isdir(native):
        raise IngestError(f"Это каталог, а не файл: {text}")
    if not os.path.isfile(native):
        raise IngestError(f"Файл не найден: {text}")
    fmt = detect.file_format(path)
    if fmt is None:
        raise IngestError(f"{text}: {_unsupported_reason(path)}")
    if fmt == "pdf":
        try:
            return detect.classify_pdf(detect.profile_pdf(path))
        except InspectError as exc:
            raise IngestError(f"{text}: {exc}") from exc
    return detect.FORMAT_KIND[fmt]


def add_sources(
    settings: Settings | None,
    topic_dir: Path,
    items: Sequence[str],
    *,
    kind: str | None = None,
    title: str | None = None,
) -> IngestReport:
    """Add files and links to the topic. See the module docstring for the rules.

    `IngestReport.added` holds the new records and the records whose kind this call
    changed (``--kind`` for a file already in the topic; a warning explains each change).
    `settings` is accepted for the contract (future: per-topic limits); ingest needs none.
    """
    del settings
    topic_dir = Path(topic_dir)
    if kind is not None:
        kind = kind.strip().lower()
        if kind not in SOURCE_KINDS:
            raise IngestError(f"Неизвестный вид «{kind}». Допустимо: {', '.join(SOURCE_KINDS)}")
    if title is not None:
        title = " ".join(title.split()) or None
    if not any(_clean_item(i) for i in items):
        raise IngestError("Не указаны файлы или ссылки для добавления")
    _load(topic_dir)  # the topic must exist and be readable before anything is copied

    report = IngestReport()
    planned = _plan(items, kind, report)
    if title is not None and len(planned) > 1:
        raise IngestError(f"--title задаёт название одного источника, а добавляется {len(planned)}")
    if not planned:
        return report
    sources_dir = topic_dir / SOURCES_DIR
    os.makedirs(os_path(sources_dir), exist_ok=True)
    _cleanup_incoming(sources_dir)
    for item in planned:
        if item.url is not None:
            _ingest_url(topic_dir, item, kind=kind, title=title, report=report)
        else:
            _ingest_file(topic_dir, item, kind=kind, title=title, report=report)
    return report


# ---------------------------------------------------------------- printing

KIND_LABELS: dict[str, str] = {
    "pdf-text": "PDF с текстовым слоем",
    "pdf-scan": "скан PDF без текстового слоя",
    "handwritten": "рукопись",
    "slides": "слайды",
    "video": "видео",
    "audio": "аудио",
    "web": "веб-страница",
    "docx": "документ Word",
    "md": "Markdown или текст",
    "tex": "LaTeX",
}

STATUS_LABELS: dict[str, tuple[str, str]] = {
    "added": ("добавлен", "cyan"),
    "extracting": ("извлекается", "yellow"),
    "extracted": ("извлечён", "green"),
    "failed": ("ошибка", "red"),
    "skipped": ("пропущен", "dim"),
}


def _fmt_size(size: int | None) -> str:
    if not size:
        return "—"
    if size < 1024 * 1024:
        return f"{max(1, round(size / 1024))} КБ"
    return f"{size / (1024 * 1024):.1f} МБ".replace(".", ",")


def _volume(record: SourceRecord) -> str:
    units = record.units or {}
    if record.kind == "slides" and units.get("slides"):
        return f"{int(units['slides'])} сл."
    if units.get("pages"):
        return f"{int(units['pages'])} стр."
    if units.get("slides"):
        return f"{int(units['slides'])} сл."
    if units.get("minutes"):
        return f"{units['minutes']:g} мин"
    return _fmt_size(record.size)


def _aspect_label(aspect: float) -> str | None:
    for label, value in (("16:9", 16 / 9), ("16:10", 1.6), ("4:3", 4 / 3)):
        if abs(aspect - value) < 0.03:
            return label
    return None


def _file_ext(record: SourceRecord) -> str:
    return detect.real_suffix(record.file) if record.file else ""


def _quality_text(record: SourceRecord) -> Text:
    q = record.quality or {}
    parts: list[Text] = []
    per = "зн./сл." if record.kind == "slides" else "зн./стр."
    if isinstance(q.get("text_layer"), int | float):
        parts.append(Text(f"текст {round(q['text_layer'] * 100)}%"))
    if isinstance(q.get("chars_per_page"), int | float):
        parts.append(Text(f"{round(q['chars_per_page'])} {per}"))
    aspect = q.get("aspect")
    if isinstance(aspect, int | float) and aspect >= detect.SLIDES_MIN_ASPECT:
        label = _aspect_label(float(aspect))
        if label:
            parts.append(Text(label))
    if q.get("truncated"):
        parts.append(Text("обрезан", style="red"))
    elif q.get("repaired") or q.get("broken_pages"):
        parts.append(Text("повреждён", style="red"))
    if isinstance(q.get("pages_vision"), int) and q["pages_vision"]:
        parts.append(Text(f"агент: {q['pages_vision']}"))
    if isinstance(q.get("pages_failed"), int) and q["pages_failed"]:
        parts.append(Text(f"не распознано: {q['pages_failed']}", style="red"))
    if q.get("encoding") and record.kind == "web":
        parts.append(Text(str(q["encoding"]), style="dim"))
    notes = q.get("notes")
    if isinstance(notes, list) and notes:
        parts.append(Text(f"замечаний: {len(notes)}", style="yellow"))
    if record.status == "failed" and record.error:
        parts.append(Text(record.error, style="red"))
    out = Text()
    for i, part in enumerate(parts):
        if i:
            out.append(", ")
        out.append_text(part)
    return out if parts else Text("—", style="dim")


def print_sources(records: Sequence[SourceRecord], *, console: Console, title: str) -> None:
    """Table of sources: id, kind, title, volume, status, short quality signals."""
    if not records:
        # The same call prints "nothing added" for `h0lon add`, so no claim about the topic.
        console.print(Text(f"{title}: нет.", style="dim"))
        return
    table = Table(title=title, title_justify="left", show_lines=False)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Вид", no_wrap=True)
    table.add_column("Название", overflow="fold", ratio=1)
    table.add_column("Объём", justify="right", no_wrap=True)
    table.add_column("Статус", no_wrap=True)
    table.add_column("Качество", overflow="fold")
    for r in records:
        name = Text(r.title)
        if r.url:
            if r.title != detect.url_display(r.url):
                name.append("\n" + r.url, style="dim")
        elif r.original_name and _human_title(r.original_name, _file_ext(r)) != r.title:
            name.append("\n" + r.original_name, style="dim")
        label, style = STATUS_LABELS.get(r.status, (r.status, ""))
        table.add_row(
            Text(r.id),
            Text(r.kind),
            name,
            Text(_volume(r)),
            Text(label, style=style),
            _quality_text(r),
        )
    kinds = list(dict.fromkeys(r.kind for r in records))
    table.caption = "; ".join(f"{k} — {KIND_LABELS.get(k, k)}" for k in kinds)
    table.caption_justify = "left"
    console.print(table)
