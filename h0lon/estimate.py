"""Time estimate of a topic before the run (PRD 13.4, UC6; docs/ARCHITECTURE.md, M6).

`estimate_topic(settings, topic_dir)` turns the sources of a topic and the dry run of the
extraction (`extract_topic(dry_run=True)`) into units of work and times:

- minutes of audio to recognize → the time of every way to recognize speech (this GPU, this CPU,
  the Colab worker) from `<state_dir>/calibration.json` (seconds per minute of audio for a
  device × model pair, written by every real run), and without a measurement from the starting
  values of PRD 13.1 (`extract.asr.DEFAULT_SECONDS_PER_MINUTE`); the worker also costs the time
  to upload the audio (measured as well, otherwise ~0.5 MB/s);
- pages and frames the agent reads, runs of the agent in the extraction (from the plans) and in
  the synthesis (by the size of the topic: one run for the structure, one for every ~10 500
  characters of sources in the sections, one for the global pass, one for the coverage);
- the time of an agent run per stage: the mean of the runs of this stage in
  `<state_dir>/usage.jsonl`, otherwise 1 minute for an extraction run and 3–6 minutes for a
  synthesis run (PRD 13.4); runs of one stage go `agents.parallel_runs` at a time.

Nothing here starts an agent, downloads a model or touches the network: the Colab worker is
«available» when `compute.colab_url` and `compute.colab_token` are set (`h0lon doctor` pings it).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from h0lon.config import Settings
    from h0lon.extract.model import ExtractPlan
    from h0lon.sources.models import SourceRecord

# `extract.pipeline._plan_one` says this when a source is up to date (it has no flag for it).
CACHED_MARK = "кэш актуален"

# Stages of agent runs, as the usage journal names them (`stage` of a record).
EXTRACTION_STAGES = ("extract", "summary", "transcript_fix", "handwriting")
SYNTHESIS_STAGES = ("outline", "sections", "global", "coverage", "supplement", "fixlatex")
STAGE_TITLES = {
    "extraction": "Извлечение (аннотации, формулы, правка речи)",
    "outline": "S1 · структура конспекта",
    "sections": "S2 · написание разделов",
    "global": "S3 · глобальная правка",
    "coverage": "S4 · проверка покрытия",
}
DEFAULT_EXTRACTION_RUN_S = (60.0, 60.0)  # PRD 13.4: one minute for a run of the extraction
DEFAULT_SYNTHESIS_RUN_S = (180.0, 360.0)  # PRD 13.4: three to six minutes for a run
NBSP = "\u00a0"  # between the thousands of a number
ATTENTION_S = 600.0  # the review gate: about ten minutes of the user's attention

# Size of a topic → runs of the synthesis (docs/ARCHITECTURE.md, «Группировка S2»; a real topic
# of 5 sources, ~115 000 characters of Source Docs, needed 11 runs of S2).
SECTIONS_CHARS_PER_RUN = 10_500
GLOBAL_CHARS_PER_RUN = 350_000
COVERAGE_CHARS_PER_RUN = 400_000
# Characters of a source that is not extracted yet, from its units (rough).
CHARS_PER_PAGE = {"pdf-text": 2000.0, "slides": 600.0, "pdf-scan": 1800.0, "handwritten": 1800.0}
CHARS_PER_AUDIO_MINUTE = 800.0
CHARS_PER_FILE_BYTE = {"md": 0.6, "tex": 0.6, "docx": 0.3, "web": 0.05}
DEFAULT_PAGES = 10.0


# ---------------------------------------------------------------- data


@dataclass
class Span:
    """A time in seconds: `low` … `high` (equal when it is one number)."""

    low: float
    high: float

    def __add__(self, other: Span) -> Span:
        return Span(self.low + other.low, self.high + other.high)

    def scaled(self, factor: float) -> Span:
        return Span(self.low * factor, self.high * factor)


@dataclass
class SourceUnits:
    id: str
    kind: str
    title: str
    cached: bool  # extraction is up to date: the source costs nothing
    audio_minutes: float | None  # speech to recognize (None: not a recording, or unknown yet)
    pages: int
    agent_pages: int
    agent_runs: int
    chars: int
    notes: list[str] = field(default_factory=list)


@dataclass
class AsrWay:
    key: str  # local-gpu | local-cpu | colab
    title: str
    device: str  # cuda | cpu | colab (the key of calibration.json)
    model: str
    available: bool
    reason: str  # why it is not available, otherwise empty
    seconds_per_minute: float
    measured: bool
    runs: int  # runs behind the measurement
    recognition_s: float
    upload_s: float
    upload_mb: float
    upload_measured: bool
    seconds: float  # the whole time of the way
    selected: bool = False  # what `compute.asr` takes now
    fastest: bool = False  # the fastest of the available ways


@dataclass
class AgentRow:
    key: str  # extraction | outline | sections | global | coverage
    title: str
    runs: int
    run_s: Span  # one run
    measured: bool  # the time of a run is the mean of this stage in usage.jsonl
    waves: int  # runs one after another (parallel runs counted)
    seconds: Span


@dataclass
class Estimate:
    topic: str
    title: str
    sources: list[SourceUnits]
    asr_minutes: float
    asr_unknown: list[str]  # recordings whose length is not known yet (a link not downloaded)
    asr: list[AsrWay]
    agents: list[AgentRow]
    parallel: int
    agent_pages: int
    source_chars: int
    extraction_pending: bool
    total: Span
    attention_s: float
    notes: list[str]

    @property
    def selected_way(self) -> AsrWay | None:
        return next((w for w in self.asr if w.selected), None)

    @property
    def agent_runs(self) -> int:
        return sum(a.runs for a in self.agents)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["agent_runs"] = self.agent_runs
        return data


# ---------------------------------------------------------------- formatting


def human_seconds(seconds: float) -> str:
    """«45 с», «3 мин 20 с», «12 мин», «1 ч 05 мин»."""
    total = max(0, round(seconds))
    if total < 60:
        return f"{total} с"
    if total < 600:
        minutes, sec = divmod(total, 60)
        return f"{minutes} мин {sec:02d} с" if sec else f"{minutes} мин"
    minutes = round(total / 60)
    if minutes < 60:
        return f"{minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} ч {minutes:02d} мин"


def human_span(low: float, high: float | None = None) -> str:
    """«≈ 12 мин» for one number, «3–6 мин» or «45 мин – 1 ч 30 мин» for a range."""
    high = low if high is None else high
    a, b = human_seconds(low), human_seconds(high)
    if a == b:
        return f"≈ {a}" if low > 0 else a
    for unit in (" мин", " с"):
        if a.endswith(unit) and b.endswith(unit) and a.count(" ") == 1 and b.count(" ") == 1:
            return f"{a[: -len(unit)]}–{b}"
    return f"{a} – {b}"


def _number(value: float) -> str:
    """8.0 → «8», 8.352 → «8,4» (a decimal comma, one digit)."""
    text = f"{value:.1f}"
    return (text[:-2] if text.endswith(".0") else text).replace(".", ",")


def runs_word(n: int) -> str:
    """«1 прогон», «3 прогона», «5 прогонов»."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} прогон"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} прогона"
    return f"{n} прогонов"


# ---------------------------------------------------------------- units of work


def _is_cached(plan: ExtractPlan | None) -> bool:
    return plan is not None and any(CACHED_MARK in note for note in plan.notes)


def _source_chars(topic_dir: Path, rec: SourceRecord) -> int:
    """Characters of a source: its Source Doc when it exists, otherwise a guess by units."""
    doc = topic_dir / "extracted" / rec.id / "source.md"
    try:
        return len(doc.read_text(encoding="utf-8"))
    except OSError:
        pass
    units = rec.units or {}
    if rec.kind in ("video", "audio"):
        return round(float(units.get("minutes") or 0.0) * CHARS_PER_AUDIO_MINUTE)
    if rec.kind in CHARS_PER_PAGE:
        pages = float(units.get("pages") or units.get("slides") or DEFAULT_PAGES)
        per_page = float((rec.quality or {}).get("chars_per_page") or CHARS_PER_PAGE[rec.kind])
        if rec.kind == "slides":
            per_page = CHARS_PER_PAGE["slides"]
        return round(pages * per_page)
    return round(float(rec.size or 0) * CHARS_PER_FILE_BYTE.get(rec.kind, 0.3))


def _audio_minutes(settings: Settings, rec: SourceRecord) -> float | None:
    if rec.kind not in ("video", "audio"):
        return None
    minutes = float((rec.units or {}).get("minutes") or 0.0)
    if minutes <= 0:
        return None
    limit = int(settings.compute.video_max_minutes or 0)
    return min(minutes, float(limit)) if limit else minutes


def source_units(
    settings: Settings, topic_dir: Path, records: list[SourceRecord], plans: list[ExtractPlan]
) -> list[SourceUnits]:
    by_id = {p.source_id: p for p in plans}
    rows: list[SourceUnits] = []
    for rec in records:
        plan = by_id.get(rec.id)
        cached = _is_cached(plan)
        minutes = None if cached else _audio_minutes(settings, rec)
        rows.append(
            SourceUnits(
                id=rec.id,
                kind=rec.kind,
                title=rec.title,
                cached=cached,
                audio_minutes=minutes,
                pages=int(plan.pages_total) if plan else 0,
                agent_pages=int(plan.pages_vision) if plan else 0,
                agent_runs=int(plan.agent_runs) if plan else 0,
                chars=_source_chars(topic_dir, rec),
                notes=list(plan.notes) if plan else [],
            )
        )
    return rows


# ---------------------------------------------------------------- speech recognition


def _upload_bytes(minutes_per_file: list[float]) -> int:
    from h0lon.extract import asr

    total = 0
    for minutes in minutes_per_file:
        try:
            total += asr.upload_plan(minutes * 60.0)[1]
        except asr.RemoteError:
            total += round(asr.UPLOAD_MAX_BYTES)
    return total


def asr_ways(
    settings: Settings,
    minutes_per_file: list[float],
    *,
    found: Any = None,
) -> list[AsrWay]:
    """The three ways to recognize `minutes_per_file` (one entry per recording)."""
    from h0lon.extract import asr

    state = found if found is not None else asr.probe()
    minutes = sum(minutes_per_file)
    compute = settings.compute
    model = (compute.asr_model or asr.DEFAULT_MODEL).strip() or asr.DEFAULT_MODEL
    cpu_model = model
    if not asr.explicit_model(settings) and model == asr.DEFAULT_MODEL:
        cpu_model = asr.CPU_DEFAULT_MODEL  # what `choose_device` does without a GPU

    def calibrated(device: str, name: str) -> tuple[float, bool, int]:
        spm, measured = asr.estimate_seconds_per_minute(settings, device, name)
        entry = asr.calibration_entry(settings, device, name)
        return spm, measured, int((entry or {}).get("runs") or 0)

    gpu_spm, gpu_m, gpu_runs = calibrated("cuda", model)
    cpu_spm, cpu_m, cpu_runs = calibrated("cpu", cpu_model)
    colab_spm, colab_m, colab_runs = calibrated("colab", model)
    gpu_reason = "" if state.cuda_ready else asr.cuda_problem(state, capital=True)
    cpu_reason = "" if state.installed else asr.cuda_problem(state, capital=True)
    if compute.colab_url.strip() and compute.colab_token.strip():
        colab_reason = ""
    elif compute.colab_url.strip():
        colab_reason = "Не задан compute.colab_token"
    else:
        colab_reason = "Не задан compute.colab_url (ноутбук colab/h0lon_worker.ipynb)"
    up_bytes = _upload_bytes(minutes_per_file)
    up_rate, up_measured = asr.upload_bytes_per_second(settings)

    ways = [
        AsrWay(
            "local-gpu",
            "Этот ПК, видеокарта",
            "cuda",
            model,
            not gpu_reason,
            gpu_reason,
            gpu_spm,
            gpu_m,
            gpu_runs,
            gpu_spm * minutes,
            0.0,
            0.0,
            True,
            gpu_spm * minutes,
        ),
        AsrWay(
            "local-cpu",
            "Этот ПК, процессор",
            "cpu",
            cpu_model,
            not cpu_reason,
            cpu_reason,
            cpu_spm,
            cpu_m,
            cpu_runs,
            cpu_spm * minutes,
            0.0,
            0.0,
            True,
            cpu_spm * minutes,
        ),
        AsrWay(
            "colab",
            "Google Colab (worker)",
            "colab",
            model,
            not colab_reason,
            colab_reason,
            colab_spm,
            colab_m,
            colab_runs,
            colab_spm * minutes,
            up_bytes / up_rate if up_rate else 0.0,
            up_bytes / 1e6,
            up_measured,
            colab_spm * minutes + (up_bytes / up_rate if up_rate else 0.0),
        ),
    ]
    chosen = selected_way_key(settings, state)
    for way in ways:
        way.selected = way.key == chosen
    usable = [w for w in ways if w.available]
    if usable:
        min(usable, key=lambda w: w.seconds).fastest = True
    return ways


def selected_way_key(settings: Settings, found: Any = None) -> str | None:
    """The way `compute.asr` takes now: the worker, the GPU or the CPU (None: `api`)."""
    from h0lon.extract import asr

    mode = settings.compute.asr
    if mode == "api":
        return None
    state = found if found is not None else asr.probe()
    if asr.wants_worker(settings, state):
        return "colab"
    if mode == "local-cpu" or not state.cuda_ready:
        return "local-cpu"
    return "local-gpu"


# ---------------------------------------------------------------- agents


def _stage_means(settings: Settings) -> dict[str, tuple[float, int]]:
    """Mean seconds of a successful run and the number of runs, per stage (usage.jsonl)."""
    from h0lon.agents.usage import read_usage

    try:
        records = read_usage(settings.general.state_path)
    except OSError:
        return {}
    sums: dict[str, list[float]] = {}
    for rec in records:
        if not rec.get("ok"):
            continue
        try:
            seconds = float(rec.get("duration_s"))
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            sums.setdefault(str(rec.get("stage") or ""), []).append(seconds)
    return {stage: (sum(v) / len(v), len(v)) for stage, v in sums.items()}


def _run_time(
    means: dict[str, tuple[float, int]], stages: tuple[str, ...], default: tuple[float, float]
) -> tuple[Span, bool]:
    values = [means[s] for s in stages if s in means]
    runs = sum(n for _mean, n in values)
    if not runs:
        return Span(*default), False
    mean = sum(m * n for m, n in values) / runs
    return Span(mean, mean), True


def synthesis_runs(chars: int) -> dict[str, int]:
    """Runs of the agent per synthesis stage for a topic of `chars` characters of sources."""
    return {
        "outline": 1,
        "sections": max(1, math.ceil(chars / SECTIONS_CHARS_PER_RUN)),
        "global": max(1, math.ceil(chars / GLOBAL_CHARS_PER_RUN)),
        "coverage": max(1, math.ceil(chars / COVERAGE_CHARS_PER_RUN)),
    }


def _synthesis_pending(settings: Settings, topic_dir: Path, extraction_pending: bool) -> set[str]:
    """Synthesis stages that will run: all after a new extraction, otherwise the stale ones."""
    stages = {"outline", "sections", "global", "coverage"}
    if extraction_pending:
        return stages
    try:
        from h0lon.synth.build import topic_status

        info = topic_status(settings, topic_dir)["stages"]
    except Exception:  # the estimate must not fail on a topic that was never built
        return stages
    done = ("готово", "из кэша")
    return {s for s in stages if (info.get(s) or {}).get("state") not in done}


def agent_rows(
    settings: Settings,
    topic_dir: Path,
    units: list[SourceUnits],
    extraction_pending: bool,
    parallel: int,
) -> list[AgentRow]:
    means = _stage_means(settings)
    rows: list[AgentRow] = []
    ext_runs = sum(u.agent_runs for u in units)
    if ext_runs:
        run_s, measured = _run_time(means, EXTRACTION_STAGES, DEFAULT_EXTRACTION_RUN_S)
        waves = math.ceil(ext_runs / parallel)
        rows.append(
            AgentRow(
                "extraction",
                STAGE_TITLES["extraction"],
                ext_runs,
                run_s,
                measured,
                waves,
                run_s.scaled(waves),
            )
        )
    chars = sum(u.chars for u in units)
    runs = synthesis_runs(chars)
    pending = _synthesis_pending(settings, topic_dir, extraction_pending)
    for stage in ("outline", "sections", "global", "coverage"):
        if stage not in pending:
            continue
        stages = (stage, "supplement") if stage == "coverage" else (stage,)
        run_s, measured = _run_time(means, stages, DEFAULT_SYNTHESIS_RUN_S)
        # Only the sections go in parallel (`agents.parallel_runs`); the other stages run once
        # or one after another.
        waves = math.ceil(runs[stage] / parallel) if stage == "sections" else runs[stage]
        rows.append(
            AgentRow(
                stage, STAGE_TITLES[stage], runs[stage], run_s, measured, waves, run_s.scaled(waves)
            )
        )
    return rows


# ---------------------------------------------------------------- the estimate


def estimate_topic(
    settings: Settings,
    topic_dir: Path,
    *,
    use_vision: bool = True,
    backend: str | None = None,
    force: bool = False,
    plans: list[ExtractPlan] | None = None,
    found: Any = None,
) -> Estimate:
    """The estimate of extraction and synthesis of the topic in `topic_dir`.

    `plans` are the plans of `extract_topic(dry_run=True)` when the caller has them already.
    Raises what `extract_topic` and `list_sources` raise (`ValueError`, `IngestError`).
    """
    from h0lon.extract.pipeline import extract_topic
    from h0lon.sources.ingest import list_sources
    from h0lon.workspace import load_topic

    topic_dir = Path(topic_dir).resolve()
    meta = load_topic(topic_dir)
    records = list_sources(topic_dir)
    if plans is None:
        dry = extract_topic(
            settings,
            topic_dir,
            force=force,
            dry_run=True,
            use_vision=use_vision,
            backend=backend,
        )
        plans = list(dry)  # type: ignore[arg-type]  (dry_run gives plans)
    units = source_units(settings, topic_dir, records, plans)
    parallel = max(1, int(settings.agents.parallel_runs))
    pending_units = [u for u in units if not u.cached]
    extraction_pending = bool(pending_units)
    recordings = [u for u in units if u.kind in ("video", "audio") and not u.cached]
    minutes_per_file = [u.audio_minutes for u in recordings if u.audio_minutes]
    unknown = [u.id for u in recordings if not u.audio_minutes]
    ways = asr_ways(settings, minutes_per_file, found=found)
    agents = agent_rows(settings, topic_dir, units, extraction_pending, parallel)

    total = Span(0.0, 0.0)
    selected = next((w for w in ways if w.selected), None)
    if minutes_per_file:
        usable = selected if selected is not None and selected.available else None
        chosen = usable or next((w for w in ways if w.available), ways[0])
        total = total + Span(chosen.seconds, chosen.seconds)
    for row in agents:
        total = total + row.seconds

    notes: list[str] = []
    if unknown:
        notes.append(
            "Длина записей " + ", ".join(unknown) + " станет известна после загрузки: "
            "время распознавания по ним не включено"
        )
    if not use_vision:
        notes.append("Агент на извлечении отключён (--no-vision): прогонов извлечения нет")
    if selected is not None and not selected.available and minutes_per_file:
        notes.append(
            f"Способ по настройкам ({selected.title}) недоступен: {selected.reason.lower()}"
        )
    if settings.compute.asr == "api":
        notes.append("compute.asr = api пока не поддерживается: распознавание не посчитано")
    if any(u.chars and not (topic_dir / "extracted" / u.id / "source.md").is_file() for u in units):
        notes.append("Объём источников, которые ещё не извлечены, прикинут по числу страниц")
    notes.append(
        "Время кода (разбор файлов, выбор кадров, рендер PDF) — несколько минут и не включено; "
        "агенты считаются при параллелизме "
        f"{parallel}"
    )
    return Estimate(
        topic=str(topic_dir),
        title=meta.title,
        sources=units,
        asr_minutes=sum(minutes_per_file),
        asr_unknown=unknown,
        asr=ways,
        agents=agents,
        parallel=parallel,
        agent_pages=sum(u.agent_pages for u in pending_units),
        source_chars=sum(u.chars for u in units),
        extraction_pending=extraction_pending,
        total=total,
        attention_s=ATTENTION_S if records else 0.0,
        notes=notes,
    )


# ---------------------------------------------------------------- view (CLI and web)


def way_source_text(way: AsrWay) -> str:
    """Where the speed comes from: «по замеру, 3 запуска» or «оценка PRD»."""
    if way.measured:
        word = "запуск" if way.runs % 10 == 1 and way.runs % 100 != 11 else "запуска"
        if way.runs % 10 not in (1, 2, 3, 4) or way.runs % 100 in (11, 12, 13, 14):
            word = "запусков"
        return f"по замеру, {way.runs} {word}"
    return "оценка PRD, замеров ещё нет"


def agent_source_text(row: AgentRow) -> str:
    return "по журналу прогонов" if row.measured else "ориентир PRD"


def run_time_text(row: AgentRow) -> str:
    return human_span(row.run_s.low, row.run_s.high)


def to_view(est: Estimate) -> dict[str, Any]:
    """The estimate as plain strings and numbers for a template (the web page)."""
    ways = []
    for w in est.asr:
        ways.append(
            {
                "key": w.key,
                "title": w.title,
                "model": w.model,
                "available": w.available,
                "reason": w.reason,
                "speed": f"{_number(w.seconds_per_minute)} с на минуту звука",
                "speed_source": way_source_text(w),
                "upload": (
                    f"{human_seconds(w.upload_s)} ({w.upload_mb:.0f} МБ)"
                    if w.key == "colab"
                    else "—"
                ),
                "time": human_span(w.seconds) if est.asr_minutes else "—",
                "selected": w.selected,
                "fastest": w.fastest and len([x for x in est.asr if x.available]) > 1,
            }
        )
    agents = [
        {
            "title": a.title,
            "runs": a.runs,
            "run_time": run_time_text(a),
            "source": agent_source_text(a),
            "time": human_span(a.seconds.low, a.seconds.high),
        }
        for a in est.agents
    ]
    return {
        "sources": len(est.sources),
        "to_extract": sum(1 for u in est.sources if not u.cached),
        "cached": sum(1 for u in est.sources if u.cached),
        "asr_minutes": f"{est.asr_minutes:.1f}".replace(".", ",") if est.asr_minutes else "",
        "agent_pages": est.agent_pages,
        "agent_runs": est.agent_runs,
        "chars": f"{est.source_chars:,}".replace(",", NBSP),
        "ways": ways,
        "agents": agents,
        "parallel": est.parallel,
        "total": human_span(est.total.low, est.total.high) if est.total.high else "—",
        "attention": human_seconds(est.attention_s) if est.attention_s else "",
        "notes": list(est.notes),
        "nothing": not est.extraction_pending and not est.agents,
    }


def print_estimate(est: Estimate, *, console: Any) -> None:
    """The estimate as tables for the terminal (`h0lon estimate`)."""
    from rich.markup import escape
    from rich.table import Table

    view = to_view(est)
    console.print(f"[bold]Оценка времени:[/bold] {escape(est.title)}")
    console.print(
        f"Источников: {view['sources']} (к извлечению {view['to_extract']}, "
        f"из кэша {view['cached']}); объём источников ≈ {view['chars']} знаков"
    )
    if est.agent_pages:
        console.print(f"Страниц и кадров, которые читает агент: {est.agent_pages}")

    if est.asr_minutes or est.asr_unknown:
        title = f"Распознавание речи: {view['asr_minutes'] or '0'} мин звука"
        table = Table(title=title, title_justify="left")
        for column in ("Способ", "Модель", "Скорость", "Загрузка аудио", "Время", "Примечание"):
            table.add_column(column, overflow="fold")
        for way in view["ways"]:
            marks = []
            if way["selected"]:
                marks.append("по настройкам")
            if way["fastest"]:
                marks.append("быстрее всех")
            if not way["available"]:
                marks.append(f"недоступно: {way['reason'][:1].lower()}{way['reason'][1:]}")
            table.add_row(
                escape(way["title"]),
                escape(way["model"]),
                f"{way['speed']}\n{way['speed_source']}",
                way["upload"],
                way["time"],
                escape("; ".join(marks)),
            )
        console.print(table)
    else:
        console.print("Распознавание речи не требуется.")

    if est.agents:
        table = Table(title=f"Прогоны агента (параллельно до {est.parallel})", title_justify="left")
        for column in ("Этап", "Прогонов", "На прогон", "Время", "Источник времени"):
            table.add_column(column, overflow="fold")
        for row in view["agents"]:
            table.add_row(
                escape(row["title"]),
                str(row["runs"]),
                row["run_time"],
                row["time"],
                row["source"],
            )
        console.print(table)
    else:
        console.print("Агент не понадобится: всё берётся из кэша.")

    console.print(f"[bold]Итого ориентировочно:[/bold] {view['total']}")
    if view["attention"]:
        console.print(f"Ваше внимание: ≈ {view['attention']} (проверка извлечения на review gate)")
    for note in est.notes:
        console.print(f"[dim]{escape(note)}[/dim]")
