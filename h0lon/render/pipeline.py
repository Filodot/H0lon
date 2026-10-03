"""render_document: master Markdown → PDF (Pandoc → XeLaTeX, fallback HTML → browser)."""

from __future__ import annotations

import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from h0lon import tools
from h0lon.config import Settings
from h0lon.render.html_fallback import build_html, print_to_pdf
from h0lon.render.latex import compile_latex
from h0lon.render.pandoc import (
    DEFAULT_TOC_DEPTH,
    build_argv,
    escape_anchor_syntax,
    has_headings,
    plain_title,
    read_front_matter,
    run_pandoc,
    title_fragments,
)
from h0lon.render.pdfcheck import PdfCheckReport, check_pdf
from h0lon.render.templates import TemplateInfo, get_template

if TYPE_CHECKING:
    from rich.console import Console

Engine = Literal["auto", "xelatex", "html"]
BUILD_MARKER = ".h0lon-build"
ENGINE_TITLES = {"xelatex": "XeLaTeX", "html": "HTML → браузер"}
_MAX_PRINTED_WARNINGS = 15


@dataclass
class RenderReport:
    ok: bool
    pdf: Path | None
    engine_used: str | None
    fallback_used: bool
    tex: Path | None
    build_dir: Path | None
    passes: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    checks: PdfCheckReport | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "pdf": str(self.pdf) if self.pdf else None,
            "engine_used": self.engine_used,
            "fallback_used": self.fallback_used,
            "tex": str(self.tex) if self.tex else None,
            "build_dir": str(self.build_dir) if self.build_dir else None,
            "passes": self.passes,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "checks": self.checks.to_dict() if self.checks else None,
            "duration_s": round(self.duration_s, 3),
        }

    def print(self, console: Console) -> None:
        from rich.markup import escape

        if self.ok and self.pdf:
            console.print(f"[bold green]PDF готов:[/bold green] {escape(str(self.pdf))}")
        else:
            console.print("[bold red]Сборка PDF не удалась[/bold red]")
        if self.engine_used:
            engine = ENGINE_TITLES.get(self.engine_used, self.engine_used)
            if self.engine_used == "xelatex" and self.passes:
                engine += f", проходов: {self.passes}"
            line = f"Движок: {escape(engine)}"
            if self.fallback_used:
                line += " [yellow](запасной рендер)[/yellow]"
            console.print(line)
        c = self.checks
        if c is not None:
            fonts = f"шрифтов {len(c.fonts)}"
            fonts += ", все встроены" if not c.not_embedded else ", [red]есть невстроенные[/red]"
            console.print(f"Проверка PDF: страниц {c.pages}, {fonts}, закладок {c.bookmarks}")
        if self.build_dir:
            console.print(f"Каталог сборки: {escape(str(self.build_dir))}")
        if self.warnings:
            console.print(f"[yellow]Предупреждения ({len(self.warnings)}):[/yellow]")
            for w in self.warnings[:_MAX_PRINTED_WARNINGS]:
                console.print(f"  - {escape(w)}")
            if len(self.warnings) > _MAX_PRINTED_WARNINGS:
                rest = len(self.warnings) - _MAX_PRINTED_WARNINGS
                console.print(f"  … и ещё {rest} (полный список: --json)")
        if self.errors:
            console.print("[red]Ошибки:[/red]")
            for e in self.errors:
                console.print(f"  - {escape(e)}")
        console.print(f"[dim]Время: {self.duration_s:.1f} с[/dim]")


@dataclass
class _Attempt:
    engine: str
    ok: bool = False
    pdf: Path | None = None
    tex: Path | None = None
    passes: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: PdfCheckReport | None = None


@dataclass
class _Job:
    src: Path
    settings: Settings
    template: TemplateInfo
    pandoc: Path
    build_dir: Path
    jobname: str
    text: str  # the master after escape_anchor_syntax(), fed to Pandoc on stdin
    metadata: dict[str, object]
    variables: dict[str, str]
    toc: bool
    toc_depth: int
    number_sections: bool
    expected_title: list[str] | None  # title_fragments(): pieces to find in the PDF text
    expect_bookmarks: bool


def _engine_plan(engine: Engine, settings: Settings) -> list[str]:
    if engine == "xelatex":
        return ["xelatex"]
    if engine == "html":
        return ["html"]
    if settings.render.engine == "html":
        return ["html"]
    return ["xelatex", "html"] if settings.render.fallback_html else ["xelatex"]


def _jobname(stem: str) -> str:
    """ASCII job name: TeX and its tools are happiest without spaces and Cyrillic."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")
    return name or "document"


def _make_build_dir(out_dir: Path, stem: str) -> Path:
    path = out_dir / f"{stem}.build"
    if path.exists():
        if (path / BUILD_MARKER).is_file():
            shutil.rmtree(path, ignore_errors=True)
        if path.exists():  # not ours or locked: use a fresh sibling directory
            path = Path(tempfile.mkdtemp(prefix=f"{stem}.build-", dir=out_dir))
    path.mkdir(parents=True, exist_ok=True)
    (path / BUILD_MARKER).write_text("H0lon build directory\n", encoding="utf-8")
    return path


def _as_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "no", "0", "off", "")


def _as_int(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _check(job: _Job, pdf: Path, attempt: _Attempt) -> None:
    checks = check_pdf(
        pdf, expected_title=job.expected_title, expect_bookmarks=job.expect_bookmarks
    )
    attempt.checks = checks
    attempt.warnings += checks.warnings
    if checks.ok:
        attempt.ok = True
        attempt.pdf = pdf
    else:
        attempt.errors += checks.errors


def _run_xelatex(job: _Job) -> _Attempt:
    attempt = _Attempt(engine="xelatex")
    xelatex = tools.find_xelatex(job.settings.render.xelatex)
    if xelatex is None:
        attempt.errors.append(
            "XeLaTeX не найден: установите MiKTeX (https://miktex.org) или TeX Live, "
            "либо укажите путь в [render] xelatex"
        )
        return attempt
    tex = job.build_dir / f"{job.jobname}.tex"
    argv = build_argv(
        job.pandoc,
        src=None,
        to="latex",
        output=tex,
        template=job.template.template_tex,
        metadata=job.metadata,
        variables=job.variables,
        toc=job.toc,
        toc_depth=job.toc_depth,
        number_sections=job.number_sections,
        resource_path=[job.src.parent],
    )
    pres = run_pandoc(argv, cwd=job.src.parent, input_text=job.text, output=tex)
    attempt.warnings += pres.warnings
    if not pres.ok:
        attempt.errors.append(pres.error or "Pandoc не создал .tex")
        return attempt
    attempt.tex = tex
    lres = compile_latex(tex, xelatex=xelatex, passes=job.settings.render.passes)
    attempt.passes = lres.passes
    attempt.warnings += lres.warnings
    if not lres.ok or lres.pdf is None:
        attempt.errors += lres.errors or ["XeLaTeX не собрал PDF"]
        return attempt
    _check(job, lres.pdf, attempt)
    return attempt


def _run_html(job: _Job) -> _Attempt:
    attempt = _Attempt(engine="html")
    browser = tools.find_browser(job.settings.render.browser)
    if browser is None:
        attempt.errors.append(
            "Не найден браузер Edge/Chrome/Chromium для запасного рендера "
            "(или укажите путь в [render] browser)"
        )
        return attempt
    hres = build_html(
        job.pandoc,
        src=job.src,
        input_text=job.text,
        build_dir=job.build_dir,
        jobname=job.jobname,
        template=job.template,
        metadata=job.metadata,
        variables=job.variables,
        toc=job.toc,
        toc_depth=job.toc_depth,
    )
    attempt.warnings += hres.warnings
    if not hres.ok or hres.output is None:
        attempt.errors.append(hres.error or "Pandoc не создал HTML")
        return attempt
    bres = print_to_pdf(browser, html=hres.output, pdf=job.build_dir / f"{job.jobname}-html.pdf")
    if not bres.ok or bres.pdf is None:
        attempt.errors += bres.errors
        return attempt
    _check(job, bres.pdf, attempt)
    return attempt


_RUNNERS = {"xelatex": _run_xelatex, "html": _run_html}


def _prepare_metadata(
    src: Path, front: dict[str, Any], metadata: dict[str, str] | None, clean: bool
) -> tuple[dict[str, object], dict[str, Any]]:
    """Command-line metadata (overrides front matter) and the merged view of both.

    `clean=True` forces clean mode; otherwise `h0lon-clean` from the front matter or from
    `metadata` decides (a -M value would override the front matter, so none is passed).
    """
    meta: dict[str, object] = {str(k): v for k, v in (metadata or {}).items()}
    effective: dict[str, Any] = {**front, **meta}
    if not plain_title(effective.get("title")):
        meta["title"] = src.stem
        effective["title"] = src.stem
    if clean:
        meta["h0lon-clean"] = True
        effective["h0lon-clean"] = True
    meta["h0lon-resource-dir"] = src.parent.as_posix()
    return meta, effective


def _font_variables(settings: Settings, effective: dict[str, Any]) -> dict[str, str]:
    r = settings.render
    fonts = {
        "mainfont": r.main_font,
        "sansfont": r.sans_font,
        "monofont": r.mono_font,
        "mathfont": r.math_font,
    }
    # A font set in the document (front matter or metadata) wins over the config.
    return {k: v for k, v in fonts.items() if v and k not in effective}


def render_document(
    src: Path,
    *,
    settings: Settings,
    out_dir: Path | None = None,
    template: str | None = None,
    clean: bool = False,
    engine: Engine = "auto",
    keep_build: bool = False,
    metadata: dict[str, str] | None = None,
) -> RenderReport:
    """Render a Pandoc Markdown master to <out_dir>/<stem>.pdf. Never raises for build errors."""
    start = time.monotonic()
    report = RenderReport(
        ok=False,
        pdf=None,
        engine_used=None,
        fallback_used=False,
        tex=None,
        build_dir=None,
        passes=0,
    )

    def done() -> RenderReport:
        report.duration_s = time.monotonic() - start
        return report

    src = Path(src).expanduser().resolve()
    if not src.is_file():
        report.errors.append(f"Файл не найден: {src}")
        return done()
    if engine not in ("auto", "xelatex", "html"):
        report.errors.append(f"Неизвестный движок: {engine} (допустимо auto | xelatex | html)")
        return done()

    template_name = template or settings.render.template
    try:
        tpl = get_template(template_name, settings)
    except KeyError:
        report.errors.append(f"Шаблон «{template_name}» не найден")
        return done()
    except ValueError as exc:
        report.errors.append(str(exc))
        return done()

    try:
        text = src.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        report.errors.append(f"Не удалось прочитать {src} как UTF-8: {exc}")
        return done()

    pandoc = tools.find_pandoc(settings.render.pandoc)
    if pandoc is None:
        report.errors.append(
            "Pandoc не найден: переустановите зависимости (uv sync) или укажите [render] pandoc"
        )
        return done()

    out = Path(out_dir).expanduser().resolve() if out_dir else src.parent
    try:
        out.mkdir(parents=True, exist_ok=True)
        build_dir = _make_build_dir(out, src.stem)
    except OSError as exc:
        report.errors.append(f"Не удалось создать каталог {out}: {exc}")
        return done()

    front = read_front_matter(text)
    meta, effective = _prepare_metadata(src, front, metadata, clean)
    job = _Job(
        src=src,
        settings=settings,
        template=tpl,
        pandoc=pandoc,
        build_dir=build_dir,
        jobname=_jobname(src.stem),
        text=escape_anchor_syntax(text),
        metadata=meta,
        variables=_font_variables(settings, effective),
        toc=_as_bool(effective.get("toc"), True),
        toc_depth=_as_int(effective.get("toc-depth"), DEFAULT_TOC_DEPTH),
        number_sections=_as_bool(effective.get("numbersections"), True),
        expected_title=title_fragments(effective.get("title")) or None,
        expect_bookmarks=has_headings(text),
    )

    try:
        plan = _engine_plan(engine, settings)
        failed: list[_Attempt] = []
        final: _Attempt | None = None
        for name in plan:
            attempt = _RUNNERS[name](job)
            if attempt.tex is not None:
                report.tex = attempt.tex
            if name == "xelatex":
                report.passes = attempt.passes
            if attempt.ok:
                final = attempt
                break
            failed.append(attempt)

        for prev in failed:
            title = ENGINE_TITLES.get(prev.engine, prev.engine)
            if final is not None:
                report.warnings += [f"{title} не собрал PDF: {e}" for e in prev.errors]
            else:
                report.errors += [f"{title}: {e}" for e in prev.errors]
                report.warnings += [f"{title}: {w}" for w in prev.warnings]

        if final is not None and final.pdf is not None:
            report.engine_used = final.engine
            report.fallback_used = bool(failed)
            report.warnings += final.warnings
            report.checks = final.checks
            target = out / f"{src.stem}.pdf"
            try:
                shutil.copyfile(final.pdf, target)
            except OSError as exc:
                report.errors.append(
                    f"Не удалось записать {target} (файл открыт в другой программе?): {exc}"
                )
            else:
                report.pdf = target
                if report.checks is not None:
                    report.checks.path = str(target)
                report.ok = True
        elif failed:
            # A PDF was produced but failed the checks: keep the findings for diagnosis.
            report.checks = next((a.checks for a in reversed(failed) if a.checks), None)
    finally:
        if keep_build:
            report.build_dir = build_dir
        else:
            shutil.rmtree(build_dir, ignore_errors=True)
            report.tex = None
    return done()
