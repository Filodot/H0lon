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


if __name__ == "__main__":
    app()
