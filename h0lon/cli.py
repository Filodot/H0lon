"""Command-line interface: `h0lon <command>`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

from h0lon import __version__
from h0lon.config import Settings, load_settings, user_config_path, write_user_config

app = typer.Typer(
    name="h0lon",
    help="H0lon — все материалы учебной темы → один полный PDF-конспект → вариации.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)


def _fix_stdio() -> None:
    """Never crash on characters outside the Windows ANSI code page.

    Piped output (another program, an agent, CI) gets UTF-8; an interactive console keeps its
    encoding but replaces unencodable characters.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


_fix_stdio()


def _settings(ctx: typer.Context) -> Settings:
    try:
        return load_settings(ctx.obj.get("config") if ctx.obj else None)
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]Ошибка конфигурации:[/red] {exc}")
        raise typer.Exit(2) from exc


def _print_json(data: Any) -> None:
    typer.echo(json.dumps(data, ensure_ascii=False, indent=2, default=str))


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Путь к h0lon.toml (иначе H0LON_CONFIG, ./h0lon.toml, каталог пользователя).",
        ),
    ] = None,
) -> None:
    ctx.obj = {"config": config}


@app.command()
def version() -> None:
    """Показать версию."""
    typer.echo(f"h0lon {__version__}")


@app.command()
def init(
    ctx: typer.Context,
    force: Annotated[bool, typer.Option("--force", help="Перезаписать существующий файл.")] = False,
    path: Annotated[Path | None, typer.Option("--path", help="Куда записать h0lon.toml.")] = None,
) -> None:
    """Создать h0lon.toml из примера и каталог рабочих областей."""
    target = path or user_config_path()
    try:
        written = write_user_config(target, force=force)
    except FileExistsError as exc:
        err_console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1) from exc
    settings = load_settings(written)
    settings.general.workspaces_dir.mkdir(parents=True, exist_ok=True)
    settings.general.state_path.mkdir(parents=True, exist_ok=True)
    console.print(f"Конфигурация: [bold]{written}[/bold]")
    console.print(f"Рабочие области: [bold]{settings.general.workspaces_dir}[/bold]")
    console.print("Дальше: [bold]h0lon doctor[/bold] — проверить окружение.")


@app.command()
def doctor(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json", help="Вывести результат в JSON.")] = False,
    no_auth: Annotated[
        bool, typer.Option("--no-auth", help="Не проверять вход агентов (быстрее, без сети).")
    ] = False,
) -> None:
    """Проверить инструменты: агенты, pandoc, XeLaTeX, шрифты, ffmpeg, yt-dlp, CUDA."""
    from h0lon.doctor import exit_code, print_report, run_checks

    settings = _settings(ctx)
    checks = run_checks(settings, check_auth=not no_auth)
    if json_out:
        _print_json([c.to_dict() for c in checks])
    else:
        print_report(checks, console=console, settings=settings)
    raise typer.Exit(exit_code(checks))


@app.command()
def new(
    ctx: typer.Context,
    title: Annotated[str, typer.Argument(help="Название темы, например «Теорвер — лекция 3».")],
    course: Annotated[str, typer.Option("--course", help="Курс (каталог верхнего уровня).")],
    slug: Annotated[str | None, typer.Option("--slug", help="ASCII-имя каталога темы.")] = None,
) -> None:
    """Создать рабочую область темы."""
    from h0lon.workspace import create_topic_detailed

    settings = _settings(ctx)
    try:
        created = create_topic_detailed(settings, title=title, course=course, slug=slug)
    except FileExistsError as exc:
        err_console.print(f"[yellow]{escape(str(exc))}[/yellow]")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    for warning in created.warnings:
        err_console.print(f"[yellow]Предупреждение:[/yellow] {escape(warning)}")
    path = created.path
    console.print(f"Тема создана: [bold]{path}[/bold]")


@app.command()
def render(
    ctx: typer.Context,
    src: Annotated[Path, typer.Argument(help="Markdown-файл (master.md или Pandoc Markdown).")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Каталог результата.")] = None,
    template: Annotated[str | None, typer.Option("--template", "-t", help="Имя шаблона.")] = None,
    clean: Annotated[
        bool, typer.Option("--clean", help="Убрать якоря источников [[…]] из вывода.")
    ] = False,
    engine: Annotated[str, typer.Option("--engine", help="auto | xelatex | html")] = "auto",
    keep_build: Annotated[
        bool, typer.Option("--keep-build", help="Сохранить каталог сборки (.tex, .log).")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Отчёт в JSON.")] = False,
) -> None:
    """Собрать PDF из Markdown: Pandoc → XeLaTeX, при неудаче — HTML → браузер."""
    from h0lon.render import render_document

    if engine not in ("auto", "xelatex", "html"):
        err_console.print("[red]--engine: допустимо auto | xelatex | html[/red]")
        raise typer.Exit(2)
    if not src.is_file():
        err_console.print(f"[red]Файл не найден:[/red] {src}")
        raise typer.Exit(2)
    settings = _settings(ctx)
    report = render_document(
        src,
        settings=settings,
        out_dir=out,
        template=template,
        clean=clean,
        engine=engine,  # type: ignore[arg-type]
        keep_build=keep_build,
    )
    if json_out:
        _print_json(report.to_dict())
    else:
        report.print(console)
    raise typer.Exit(0 if report.ok else 1)


def _topic(settings: Settings, ref: str) -> Path:
    from h0lon.workspace import resolve_topic

    try:
        return resolve_topic(settings, ref)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc


@app.command()
def add(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    items: Annotated[list[str], typer.Argument(help="Файлы и ссылки (http/https).")],
    kind: Annotated[
        str | None,
        typer.Option("--kind", "-k", help="Тип источника вместо автоопределения (см. sources)."),
    ] = None,
    title: Annotated[
        str | None, typer.Option("--title", help="Название источника (для одного элемента).")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Добавить источники в тему: копия в sources/, хэш, тип, запись в topic.yaml."""
    from h0lon.sources.ingest import IngestError, add_sources, print_sources

    settings = _settings(ctx)
    topic_dir = _topic(settings, topic)
    try:
        report = add_sources(settings, topic_dir, items, kind=kind, title=title)
    except IngestError as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json(report.to_dict())
    else:
        if report.added:
            print_sources(report.added, console=console, title="Добавлены источники")
        else:
            err_console.print("Новых источников нет.")
        for warning in report.warnings:
            err_console.print(f"[yellow]Предупреждение:[/yellow] {escape(warning)}")
    raise typer.Exit(0 if report.added or not items else 1)


@app.command()
def sources(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Показать источники темы и статус извлечения."""
    from h0lon.sources.ingest import IngestError, list_sources, print_sources

    settings = _settings(ctx)
    try:
        records = list_sources(_topic(settings, topic))
    except (ValueError, IngestError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json([r.model_dump() for r in records])
    else:
        print_sources(records, console=console, title="Источники темы")


@app.command()
def extract(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    source: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="Только эти источники (id, повторяемо)."),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Игнорировать кэш.")] = False,
    no_vision: Annotated[
        bool,
        typer.Option("--no-vision", help="Без распознавания страниц агентом (только код)."),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Только план: страницы, прогоны агентов.")
    ] = False,
    backend: Annotated[
        str | None, typer.Option("--backend", "-b", help="claude | codex для прогонов.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Извлечь источники в Source Docs: extracted/<ID>/source.md, blocks.jsonl, summary.md."""
    from h0lon.extract.pipeline import extract_topic, print_plans, print_results

    if backend not in (None, "claude", "codex"):
        err_console.print("[red]--backend: допустимо claude | codex[/red]")
        raise typer.Exit(2)
    settings = _settings(ctx)
    topic_dir = _topic(settings, topic)
    from h0lon.sources.ingest import IngestError

    try:
        outcome = extract_topic(
            settings,
            topic_dir,
            source_ids=source or None,
            force=force,
            use_vision=not no_vision,
            dry_run=dry_run,
            backend=backend,
            on_event=None
            if json_out
            else (lambda msg: err_console.print(msg, style="dim", markup=False, highlight=False)),
        )
    except (ValueError, IngestError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json([item.to_dict() for item in outcome])
    elif dry_run:
        print_plans(outcome, console=console)  # type: ignore[arg-type]
    else:
        print_results(outcome, console=console)  # type: ignore[arg-type]
    failed = not dry_run and any(not r.ok for r in outcome)  # type: ignore[union-attr]
    raise typer.Exit(1 if failed else 0)


@app.command()
def estimate(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    no_vision: Annotated[
        bool,
        typer.Option("--no-vision", help="Посчитать извлечение без агента (только код)."),
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Посчитать так, будто кэш извлечения не используется.")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Оценить время до запуска: распознавание речи по способам, прогоны агента, итог."""
    from h0lon.estimate import estimate_topic, print_estimate
    from h0lon.sources.ingest import IngestError

    settings = _settings(ctx)
    topic_dir = _topic(settings, topic)
    try:
        result = estimate_topic(settings, topic_dir, use_vision=not no_vision, force=force)
    except (ValueError, IngestError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json(result.to_dict())
    else:
        print_estimate(result, console=console)


@app.command("agent-test")
def agent_test(
    ctx: typer.Context,
    prompt: Annotated[
        str | None, typer.Argument(help="Дополнительное задание (иначе стандартная проверка).")
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", "-b", help="claude | codex (иначе agents.default).")
    ] = None,
    no_fallback: Annotated[
        bool, typer.Option("--no-fallback", help="Не переключаться на запасной агент.")
    ] = False,
    tier: Annotated[str, typer.Option("--tier", help="light | strong")] = "light",
    image: Annotated[
        list[Path] | None,
        typer.Option("--image", "-i", help="Изображение для задания (повторяемо)."),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Проверить headless-агента: TaskBundle → прогон → валидация → run.json."""
    from h0lon.agents.selftest import agent_test as run_agent_test

    if backend not in (None, "claude", "codex"):
        err_console.print("[red]--backend: допустимо claude | codex[/red]")
        raise typer.Exit(2)
    if tier not in ("light", "strong"):
        err_console.print("[red]--tier: допустимо light | strong[/red]")
        raise typer.Exit(2)
    settings = _settings(ctx)
    try:
        result = run_agent_test(
            settings,
            prompt=prompt,
            backend=backend,
            fallback=not no_fallback,
            tier=tier,  # type: ignore[arg-type]
            images=image or [],
            on_event=None
            if json_out
            else (lambda msg: err_console.print(msg, style="dim", markup=False, highlight=False)),
        )
    except (FileNotFoundError, ValueError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json(result.to_dict())
    else:
        result.print(console)
    raise typer.Exit(0 if result.ok else 1)


@app.command()
def build(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    no_review: Annotated[
        bool, typer.Option("--no-review", help="Не останавливаться на review gate.")
    ] = False,
    from_stage: Annotated[
        str | None,
        typer.Option(
            "--from",
            help="Пересчитать эту стадию и все после неё: "
            "extract | outline | sections | global | coverage | assemble | render.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Игнорировать кэш стадий синтеза (извлечение — через --from extract).",
        ),
    ] = False,
    backend: Annotated[
        str | None,
        typer.Option("--backend", "-b", help="claude | codex для стадий синтеза."),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Собрать мастер-конспект: структура, разделы, полнота, master.md, master.pdf."""
    from h0lon.sources.ingest import IngestError
    from h0lon.synth.build import build_topic, print_build

    if backend not in (None, "claude", "codex"):
        err_console.print("[red]--backend: допустимо claude | codex[/red]")
        raise typer.Exit(2)
    settings = _settings(ctx)
    topic_dir = _topic(settings, topic)
    try:
        result = build_topic(
            settings,
            topic_dir,
            review=not no_review,
            from_stage=from_stage,
            force=force,
            backend=backend,
            on_event=None
            if json_out
            else (lambda msg: err_console.print(msg, style="dim", markup=False, highlight=False)),
        )
    except (ValueError, IngestError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json(result.to_dict())
    else:
        print_build(result, console=console)
    raise typer.Exit(0 if result.ok else 3 if result.stopped_at == "review" else 1)


@app.command()
def approve(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
) -> None:
    """Одобрить текущее извлечение источников (review gate перед сборкой)."""
    from h0lon.sources.ingest import IngestError
    from h0lon.synth.build import approve_topic

    settings = _settings(ctx)
    topic_dir = _topic(settings, topic)
    try:
        info = approve_topic(settings, topic_dir)
    except (ValueError, IngestError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    console.print(f"Извлечение одобрено: {', '.join(info['sources'])} ({info['approved_at']}).")
    console.print("Дальше: [bold]h0lon build[/bold] — собрать мастер-конспект.")


@app.command()
def status(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Состояние темы: источники, review gate, стадии, покрытие, master.pdf."""
    from h0lon.sources.ingest import IngestError
    from h0lon.synth.build import print_status, topic_status

    settings = _settings(ctx)
    topic_dir = _topic(settings, topic)
    try:
        info = topic_status(settings, topic_dir)
    except (ValueError, IngestError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json(info)
    else:
        print_status(info, console=console)


@app.command()
def variant(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            "-p",
            help="brief (кратко) | study (учебный конспект) | cheatsheet (шпаргалка) | "
            "custom (по запросу); без --preset: custom при --prompt, перевёрстка при --template.",
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="Запрос для пресета custom: что должно получиться."),
    ] = None,
    template: Annotated[
        str | None,
        typer.Option(
            "--template",
            "-t",
            help="Шаблон: сам по себе — перевёрстка мастера без агента, "
            "вместе с пресетом — шаблон для его PDF (у шпаргалки по умолчанию a4-compact).",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Собрать заново, даже если мастер не менялся.")
    ] = False,
    backend: Annotated[
        str | None, typer.Option("--backend", "-b", help="claude | codex для прогона агента.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Собрать вариацию мастер-конспекта: кратко, учебный конспект, шпаргалка, по запросу."""
    from h0lon.sources.ingest import IngestError
    from h0lon.synth.variants import run_variant

    if backend not in (None, "claude", "codex"):
        err_console.print("[red]--backend: допустимо claude | codex[/red]")
        raise typer.Exit(2)
    settings = _settings(ctx)
    topic_dir = _topic(settings, topic)
    try:
        result = run_variant(
            settings,
            topic_dir,
            preset=preset,
            prompt=prompt,
            template=template,
            backend=backend,
            force=force,
            on_event=None
            if json_out
            else (lambda msg: err_console.print(msg, style="dim", markup=False, highlight=False)),
        )
    except (ValueError, IngestError) as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc
    if json_out:
        _print_json(result.to_dict())
    else:
        result.print(console)
    raise typer.Exit(0 if result.ok else 1)


@app.command()
def variants(
    ctx: typer.Context,
    topic: Annotated[str, typer.Argument(help="Тема: путь, <курс>/<тема> или имя темы.")],
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Показать вариации темы и их состояние (актуально / устарело после новой сборки мастера)."""
    from h0lon.synth.variants import list_variants, print_variants

    settings = _settings(ctx)
    items = list_variants(_topic(settings, topic))
    if json_out:
        _print_json(items)
    elif items:
        print_variants(items, console=console, title="Вариации темы")
    else:
        console.print("Вариаций пока нет: h0lon variant <тема> --preset brief")


queue_app = typer.Typer(
    name="queue",
    help="Очередь сборок: несколько тем подряд, темп под лимиты подписки Claude.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(queue_app, name="queue")


def _queue_fail(exc: Exception) -> typer.Exit:
    err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
    return typer.Exit(2)


@queue_app.command("add")
def queue_add(
    ctx: typer.Context,
    topics: Annotated[
        list[str], typer.Argument(help="Темы: путь, <курс>/<тема> или имя темы (одна или больше).")
    ],
    action: Annotated[
        str, typer.Option("--action", "-a", help="build (сборка мастера) | extract (извлечение).")
    ] = "build",
    no_review: Annotated[
        bool, typer.Option("--no-review", help="Сборка не останавливается на review gate.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Игнорировать кэш.")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Поставить темы в очередь (повторно та же тема с тем же действием не добавляется)."""
    from h0lon import queue as queue_mod

    if action not in ("build", "extract"):
        err_console.print("[red]--action: допустимо build | extract[/red]")
        raise typer.Exit(2)
    settings = _settings(ctx)
    params: dict[str, Any] = {}
    if no_review and action == "build":
        params["review"] = False
    if force:
        params["force"] = True
    try:
        report = queue_mod.add_items(settings, topics, action=action, params=params)
    except (ValueError, FileNotFoundError, queue_mod.QueueError) as exc:
        raise _queue_fail(exc) from exc
    if json_out:
        _print_json(report.to_dict())
    else:
        for item in report.added:
            console.print(
                f"В очереди: [bold]{escape(item.topic)}[/bold] "
                f"({queue_mod.ACTION_LABELS[item.action]}), номер элемента {item.id}"
            )
        for note in report.skipped:
            err_console.print(f"[yellow]Пропущено:[/yellow] {escape(note)}")
        if report.added:
            console.print("Дальше: [bold]h0lon queue run[/bold] — обработать очередь.")
    raise typer.Exit(0 if report.added else 1)


@queue_app.command("list")
def queue_list(
    ctx: typer.Context,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Показать очередь и загрузку окон лимитов подписки."""
    from h0lon import queue as queue_mod

    settings = _settings(ctx)
    try:
        items = queue_mod.list_items(settings)
    except queue_mod.QueueError as exc:
        raise _queue_fail(exc) from exc
    if json_out:
        _print_json(
            {
                "items": [i.to_dict() for i in items],
                "limits": queue_mod.limits_summary(settings),
            }
        )
    else:
        queue_mod.print_queue(items, console=console, settings=settings)


@queue_app.command("run")
def queue_run(
    ctx: typer.Context,
    unattended: Annotated[
        bool,
        typer.Option(
            "--unattended",
            help="Без присмотра: ошибка элемента не останавливает очередь, система не засыпает.",
        ),
    ] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Результат в JSON.")] = False,
) -> None:
    """Выполнить очередь по порядку; при заполненном 5-часовом окне ждёт его сброса."""
    from h0lon import queue as queue_mod

    settings = _settings(ctx)
    try:
        report = queue_mod.run_queue(
            settings,
            unattended=unattended,
            on_event=None
            if json_out
            else (lambda msg: err_console.print(msg, style="dim", markup=False, highlight=False)),
        )
    except queue_mod.QueueError as exc:
        raise _queue_fail(exc) from exc
    if json_out:
        _print_json(report.to_dict())
    else:
        queue_mod.print_run(report, console=console)
    raise typer.Exit(3 if report.stopped == "limit" else 0 if report.ok else 1)


@queue_app.command("clear")
def queue_clear(
    ctx: typer.Context,
    done: Annotated[
        bool, typer.Option("--done", help="Убрать только выполненные элементы.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Не спрашивать подтверждения.")] = False,
) -> None:
    """Очистить очередь: выполненные элементы (--done) или все, кроме выполняющегося."""
    from h0lon import queue as queue_mod

    settings = _settings(ctx)
    if not done and not yes and not typer.confirm("Убрать из очереди все элементы?"):
        raise typer.Exit(1)
    try:
        removed = queue_mod.clear_items(settings, done_only=done)
    except queue_mod.QueueError as exc:
        raise _queue_fail(exc) from exc
    console.print(f"Убрано элементов: {removed}.")


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@app.command()
def serve(
    ctx: typer.Context,
    host: Annotated[
        str, typer.Option("--host", help="Адрес, на котором слушать (по умолчанию только этот ПК).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Порт веб-интерфейса.")] = 8765,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Открыть интерфейс в браузере после запуска.")
    ] = False,
) -> None:
    """Запустить локальный веб-интерфейс (темы, источники, проверка, сборка)."""
    from h0lon.web.app import serve as run_server

    settings = _settings(ctx)
    if not 1 <= port <= 65535:
        err_console.print("[red]--port: допустимо 1–65535[/red]")
        raise typer.Exit(2)
    if host.lower() not in LOOPBACK_HOSTS:
        err_console.print(
            "[yellow]Предупреждение:[/yellow] интерфейс без авторизации — "
            "не открывайте его в сеть: любой, кто достучится до этого порта, "
            "сможет запускать агентов и читать ваши материалы."
        )
    shown = "127.0.0.1" if host in ("", "0.0.0.0") else f"[{host}]" if ":" in host else host
    console.print(f"H0lon: [bold]http://{shown}:{port}/[/bold] — остановить: Ctrl+C")
    try:
        run_server(settings, host=host, port=port, open_browser=open_browser)
    except OSError as exc:
        err_console.print(f"[red]Ошибка:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from exc


if __name__ == "__main__":
    app()
