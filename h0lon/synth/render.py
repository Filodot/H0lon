"""S6: master.md → master.pdf with an automatic LaTeX repair loop and an HTML fallback.

Contract: docs/ARCHITECTURE.md, «S6 render». XeLaTeX is tried first. When it fails because of
the markup, a light agent edits a copy of master.md in place (prompt `fixlatex`) from a
digest of the errors: the log lines, the .tex fragments around them and the master.md places
they come from. A fix is accepted only if nothing was lost: the set of `src` references and
of source anchors `[[…]]` is unchanged, the front matter is intact and the text size stays
within ±3 % of the assembled master. After three failed repairs the HTML → browser engine
builds the PDF and the stage reports a warning.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from h0lon import __version__, tools
from h0lon.agents import ExpectedFile, OutputContract
from h0lon.render import RenderReport, get_template, render_document
from h0lon.render.pandoc import read_front_matter
from h0lon.synth.common import (
    model_for,
    parse_src_ids,
    prompt_body,
    prompt_id,
    read_meta,
    run_agent,
    stage_key,
    text_size,
    write_meta,
)
from h0lon.synth.model import BuildContext, StageResult

STAGE = "render"
FIX_STAGE = "fixlatex"
RENDER_VERSION = "render@1"
MAX_FIX_ATTEMPTS = 3
SIZE_TOLERANCE = 0.03
MAX_ERRORS_LISTED = 15
TEX_CONTEXT_LINES = 5
MAX_TEX_LINE_CHARS = 300
MASTER_FRAGMENT_LINES = 40

_ANCHOR_RE = re.compile(r"\[\[[^\[\]\n]+\]\]")
_TEX_LINE_RE = re.compile(r"\.tex:(\d+):")
_L_LINE_RE = re.compile(r"\bl\.(\d+)\b")
_CONTROL_SEQ_RE = re.compile(r"\\[A-Za-z@]+")
_WORD_RE = re.compile(r"[^\W_]{3,}")

# Errors that no edit of the Markdown can fix: tools, fonts, files, the PDF itself.
_UNFIXABLE_MARKERS = (
    "XeLaTeX не найден",
    "Не удалось запустить XeLaTeX",
    "Не удалось запустить Pandoc",
    "Pandoc не найден",
    "не уложился",
    "Не найден файл LaTeX",
    "Unable to load picture",
    "не найден в системе",
    "Шаблон «",
    "Файл не найден",
    "Не удалось прочитать",
    "Не удалось создать каталог",
    "Не удалось записать",
    "PDF не открывается",
    "PDF зашифрован",
    "В PDF нет страниц",
    "Из PDF не извлекается текст",
    "Шрифты не встроены",
)


# ---------------------------------------------------------------- helpers


def anchors_in(markdown: str) -> set[str]:
    """Source anchors `[[P1:p3]]` of a text."""
    return set(_ANCHOR_RE.findall(markdown))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def render_key(ctx: BuildContext, master_text: str) -> str:
    """Cache key: the master text, repair prompt and model, template and render settings."""
    render = ctx.settings.render
    template_tex = ""
    try:
        template_tex = get_template(render.template, ctx.settings).template_tex.read_text(
            encoding="utf-8"
        )
    except (KeyError, ValueError, OSError):
        pass
    return stage_key(
        RENDER_VERSION,
        __version__,
        master_text,
        prompt_id(FIX_STAGE),
        model_for(ctx, "light", "fixlatex"),
        render.model_dump(mode="json"),
        template_tex,
    )


def _short(text: str, limit: int = MAX_TEX_LINE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _strip_engine(error: str) -> str:
    return re.sub(r"^(XeLaTeX|HTML → браузер):\s*", "", error)


_REMOTE_IMAGE_RE = re.compile(
    r"!\[([^\]\n]*)\]\((https?://(?:[^()\s]|\([^()\s]*\))+)(?:[ \t]+\"[^\"\n]*\")?\)"
    r"(\{[^}\n]*\})?"
)
_ID_ATTR_RE = re.compile(r"#[\w:.-]+")
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")


_URL_RE = re.compile(r"https?://")
# How XeLaTeX / our log parser say that a graphics file cannot be loaded.
_MISSING_FILE_MARKERS = ("Unable to load picture", "Не найден файл LaTeX", "not found")


def has_remote_image_error(errors: Iterable[str]) -> bool:
    """True if XeLaTeX failed on an image that is a URL (it cannot fetch files from the net)."""
    return any(
        _URL_RE.search(e) and any(marker in e for marker in _MISSING_FILE_MARKERS) for e in errors
    )


def replace_remote_images(markdown: str) -> tuple[str, int]:
    """`![подпись](https://…)` → `[Рисунок: подпись](https://…)`; returns the text and a count.

    Only the markup of the image changes: the caption and the address stay in the text, the
    identifier of the figure (`{#fig:x}`) is kept so that references to it still resolve.
    Fenced code blocks are not touched.
    """
    count = 0

    def link(m: re.Match[str]) -> str:
        nonlocal count
        count += 1
        caption = m.group(1).strip()
        label = f"Рисунок: {caption}" if caption else "Рисунок"
        ident = _ID_ATTR_RE.search(m.group(3) or "")
        return f"[{label}]({m.group(2)})" + (f"{{{ident.group(0)}}}" if ident else "")

    out: list[str] = []
    fence: str | None = None
    for line in markdown.split("\n"):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0] * 3
            fence = (
                marker if fence is None else (None if line.lstrip().startswith(fence) else fence)
            )
            out.append(line)
        elif fence is None and "](" in line:
            out.append(_REMOTE_IMAGE_RE.sub(link, line))
        else:
            out.append(line)
    return "\n".join(out), count


def fixable_errors(errors: Iterable[str]) -> list[str]:
    """Errors that an edit of the Markdown may fix (not missing tools, fonts, files)."""
    return [e for e in errors if not any(m in e for m in _UNFIXABLE_MARKERS)]


# ---------------------------------------------------------------- error digest for the agent


def _tex_line_number(error: str) -> int | None:
    m = _TEX_LINE_RE.search(error) or _L_LINE_RE.search(error)
    return int(m.group(1)) if m else None


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text))


def _paragraphs(master: str) -> list[tuple[int, int, str]]:
    """(first line, last line, text) of every blank-line separated paragraph, 1-based."""
    paragraphs: list[tuple[int, int, str]] = []
    start: int | None = None
    buf: list[str] = []
    for i, line in enumerate(master.split("\n"), start=1):
        if line.strip():
            if start is None:
                start = i
            buf.append(line)
        elif start is not None:
            paragraphs.append((start, i - 1, "\n".join(buf)))
            start, buf = None, []
    if start is not None:
        paragraphs.append((start, start + len(buf) - 1, "\n".join(buf)))
    return paragraphs


def find_master_places(
    master: str, error: str, tex_lines: list[str], line_no: int | None, *, limit: int = 2
) -> list[tuple[int, int]]:
    """Line ranges of master.md that the error most likely comes from.

    Two signals: the last control sequence of the error text (TeX prints the culprit at the end
    of its `l.12 …` context line) occurs literally in the master, and the words of the .tex
    lines around the error occur in a master paragraph. A range is the whole paragraph (at
    most MASTER_FRAGMENT_LINES lines).
    """
    lines = master.split("\n")
    paragraphs = _paragraphs(master)
    places: list[tuple[int, int]] = []

    def add(first: int, last: int) -> None:
        first, last = max(1, first), min(len(lines), last)
        if last - first + 1 > MASTER_FRAGMENT_LINES:
            last = first + MASTER_FRAGMENT_LINES - 1
        if (first, last) not in places and all(not (a <= first and last <= b) for a, b in places):
            places.append((first, last))

    sequences = _CONTROL_SEQ_RE.findall(error)
    if sequences:
        exact = re.compile(re.escape(sequences[-1]) + r"(?![A-Za-z@])")
        for i, line in enumerate(lines, start=1):
            if exact.search(line):
                first, last = next(((a, b) for a, b, _ in paragraphs if a <= i <= b), (i, i))
                if last - first + 1 > MASTER_FRAGMENT_LINES:
                    first, last = i - 5, i + 5
                add(first, last)
                break
    if line_no and tex_lines:
        window = tex_lines[max(0, line_no - 3) : line_no + 2]
        wanted = _tokens(" ".join(window))
        best: tuple[float, tuple[int, int]] | None = None
        for first, last, text in paragraphs:
            shared = len(wanted & _tokens(text))
            score = shared / max(1, len(wanted))
            if shared >= 2 and score >= 0.4 and (best is None or score > best[0]):
                best = (score, (first, last))
        if best is not None:
            add(*best[1])
    return places[:limit]


def build_errors_md(
    errors: list[str],
    tex_path: Path | None,
    master_text: str,
    *,
    attempt: int,
    engine: str = "XeLaTeX",
) -> str:
    """inputs/errors.md of a repair run: errors, .tex fragments, matching master.md places."""
    tex_lines: list[str] = []
    if tex_path is not None and tex_path.is_file():
        tex_lines = tex_path.read_text(encoding="utf-8", errors="replace").split("\n")
    master_lines = master_text.split("\n")
    out = [
        "# Ошибки сборки PDF",
        "",
        f"Движок: {engine}. Попытка исправления {attempt} из {MAX_FIX_ATTEMPTS}. "
        "Номера строк `.tex` — из временного файла, который Pandoc получил из `master.md`; "
        "строки `master.md` указаны отдельно.",
        "",
    ]
    shown = [_strip_engine(e) for e in errors][:MAX_ERRORS_LISTED]
    out.append("## Список ошибок")
    out.append("")
    out += [f"{i}. {e}" for i, e in enumerate(shown, start=1)]
    if len(errors) > len(shown):
        out.append(f"{len(shown) + 1}. … и ещё {len(errors) - len(shown)} ошибок")
    out.append("")
    for i, err in enumerate(shown, start=1):
        line_no = _tex_line_number(err)
        out += [f"## Ошибка {i}", "", f"`{_short(err, 400)}`", ""]
        if line_no and tex_lines:
            first = max(1, line_no - TEX_CONTEXT_LINES)
            last = min(len(tex_lines), line_no + TEX_CONTEXT_LINES)
            out += [
                f"Фрагмент .tex, строки {first}–{last} (ошибка в строке {line_no}, помечена `>>`):",
                "",
            ]
            out.append("```tex")
            for n in range(first, last + 1):
                mark = ">>" if n == line_no else "  "
                out.append(f"{mark}{n:5d} {_short(tex_lines[n - 1])}")
            out += ["```", ""]
        for first, last in find_master_places(master_text, err, tex_lines, line_no):
            out += [f"Похожее место в `master.md`, строки {first}–{last}:", ""]
            out.append("```markdown")
            for n in range(first, last + 1):
                out.append(f"{n:5d} {_short(master_lines[n - 1])}")
            out += ["```", ""]
        if not (line_no and tex_lines):
            out += [
                "Номер строки `.tex` в сообщении не указан: ищите причину по тексту ошибки.",
                "",
            ]
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------- repair acceptance


def check_fix(original: str, current: str, fixed: str) -> str | None:
    """None if the repaired master may replace `current`, else why it may not (Russian)."""
    if fixed.strip() == current.strip():
        return "агент не изменил файл"
    ids_before, ids_after = parse_src_ids(original), parse_src_ids(fixed)
    if ids_before != ids_after:
        lost = sorted(ids_before - ids_after)
        extra = sorted(ids_after - ids_before)
        detail = []
        if lost:
            detail.append("потеряны " + ", ".join(lost[:8]))
        if extra:
            detail.append("появились " + ", ".join(extra[:8]))
        return "изменились ссылки src (" + "; ".join(detail) + ")"
    anchors_before, anchors_after = anchors_in(original), anchors_in(fixed)
    if anchors_before != anchors_after:
        lost_a = sorted(anchors_before - anchors_after)
        extra_a = sorted(anchors_after - anchors_before)
        return "изменились якоря источников (" + ", ".join((lost_a + extra_a)[:8]) + ")"
    front = read_front_matter(original)
    if front and read_front_matter(fixed) != front:
        return "изменился заголовочный блок (front matter)"
    size_before, size_after = text_size(original), text_size(fixed)
    if size_before and abs(size_after - size_before) > SIZE_TOLERANCE * size_before:
        change = 100 * (size_after - size_before) / size_before
        return (
            f"объём текста изменился на {change:+.1f} % (допустимо ±{SIZE_TOLERANCE * 100:.0f} %)"
        )
    return None


# ---------------------------------------------------------------- the stage


def _render(ctx: BuildContext, master_md: Path, engine: str, *, keep_build: bool) -> RenderReport:
    return render_document(
        master_md,
        settings=ctx.settings,
        out_dir=ctx.topic_dir,
        engine=engine,  # type: ignore[arg-type]
        keep_build=keep_build,
    )


def _cleanup_build(report: RenderReport | None) -> None:
    if report is not None and report.build_dir is not None:
        shutil.rmtree(report.build_dir, ignore_errors=True)


def _save_log(ctx: BuildContext, report: RenderReport) -> Path | None:
    """Keep the XeLaTeX log of the last failed attempt next to the other stage files."""
    if report.build_dir is None:
        return None
    logs = sorted(report.build_dir.glob("*.log"))
    if not logs:
        return None
    target = ctx.synth_dir / "render.xelatex.log"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(logs[0], target)
    except OSError:
        return None
    return target


def _details(report: RenderReport | None, **extra: Any) -> dict[str, Any]:
    details: dict[str, Any] = dict(extra)
    if report is not None:
        details["engine"] = report.engine_used
        details["passes"] = report.passes
        if report.pdf is not None:
            details["pdf"] = str(report.pdf)
        if report.checks is not None:
            details["pages"] = report.checks.pages
            details["fonts"] = len(report.checks.fonts)
            details["bookmarks"] = report.checks.bookmarks
    return details


def _repair_attempt(
    ctx: BuildContext,
    master_md: Path,
    report: RenderReport,
    current: str,
    original: str,
    attempt: int,
    result: StageResult,
) -> str | None:
    """One `fixlatex` run; the repaired master text if it may be accepted, else None."""
    errors_md = ctx.synth_dir / "inputs" / "errors.md"
    _write_atomic(errors_md, build_errors_md(report.errors, report.tex, current, attempt=attempt))
    contract = OutputContract(
        files=[
            ExpectedFile(
                path="master.md",
                kind="markdown",
                min_chars=max(1, int(len(current) * 0.9)),
                description="Исправленный мастер-конспект целиком.",
            )
        ]
    )
    result.agent_runs += 1
    try:
        run = run_agent(
            ctx,
            stage=FIX_STAGE,
            task=prompt_body(FIX_STAGE),
            contract=contract,
            inputs=[errors_md],
            seed={"master.md": master_md},
            tier="light",
        )
    except Exception as exc:  # a broken bundle or runner must not lose the finished master
        result.warnings.append(
            f"Исправление {attempt}: агент не запущен ({type(exc).__name__}: {exc})"
        )
        return None
    if not run.ok:
        why = "; ".join(run.problems[:2]) or "агент не вернул результат"
        result.warnings.append(f"Исправление {attempt}: {why}")
        return None
    fixed_path = run.bundle.out_dir / "master.md"
    if not fixed_path.is_file():
        result.warnings.append(f"Исправление {attempt}: агент не создал master.md")
        return None
    fixed = _read(fixed_path)
    reason = check_fix(original, current, fixed)
    if reason is not None:
        result.warnings.append(f"Исправление {attempt} отклонено: {reason}")
        return None
    return fixed


def render_master(ctx: BuildContext, master_md: Path) -> StageResult:
    """Build <topic>/master.pdf from master.md (repair loop and HTML fallback inside)."""
    t0 = time.monotonic()
    master_md = Path(master_md)
    result = StageResult(stage=STAGE, ok=False)

    def finish(**meta: Any) -> StageResult:
        result.duration_s = round(time.monotonic() - t0, 3)
        write_meta(
            ctx,
            STAGE,
            {
                "prompt": prompt_id(FIX_STAGE),
                "model": model_for(ctx, "light", "fixlatex"),
                "duration_s": result.duration_s,
                "agent_runs": result.agent_runs,
                "ok": result.ok,
                **meta,
            },
        )
        return result

    if not master_md.is_file():
        result.errors.append(f"Не найден master.md: {master_md}")
        return result
    text0 = _read(master_md)
    key = render_key(ctx, text0)
    pdf_target = ctx.topic_dir / f"{master_md.stem}.pdf"

    if not ctx.force:
        meta = read_meta(ctx, STAGE)
        if (
            meta
            and meta.get("ok")
            and pdf_target.is_file()
            and key in (meta.get("key"), meta.get("key_after"))
        ):
            result.ok = True
            result.cached = True
            result.details = {"pdf": str(pdf_target), **(meta.get("details") or {})}
            return result

    render_cfg = ctx.settings.render
    html_first = render_cfg.engine == "html"
    current = text0
    fixes = 0
    image_links = 0
    fallback_used = False

    if html_first:
        ctx.emit("Сборка PDF: HTML → браузер (по настройке render.engine)")
        report = _render(ctx, master_md, "html", keep_build=False)
    else:
        ctx.emit("Сборка PDF: XeLaTeX")
        report = _render(ctx, master_md, "xelatex", keep_build=True)
        baseline = text0  # what repairs are compared with (after the deterministic fixes)
        attempt = 0
        remote_tried = False
        while not report.ok and tools.find_xelatex(render_cfg.xelatex) is not None:
            if not remote_tried and has_remote_image_error(report.errors):
                remote_tried = True
                replaced, count = replace_remote_images(current)
                if count:
                    ctx.emit(f"Рисунки по URL ({count}) заменяются ссылками: XeLaTeX их не вставит")
                    result.warnings.append(
                        f"Рисунки по URL ({count}) заменены ссылками: XeLaTeX не вставляет "
                        "изображения из сети. Чтобы рисунок попал в PDF, сохраните файл в теме "
                        "и укажите путь к нему."
                    )
                    _write_atomic(master_md, replaced)
                    current = baseline = replaced
                    image_links = count
                    report = _render(ctx, master_md, "xelatex", keep_build=True)
                    continue
            if attempt >= MAX_FIX_ATTEMPTS:
                break
            if not fixable_errors(report.errors):
                result.warnings.append(
                    "Ошибки сборки не связаны с разметкой мастера, автоисправление пропущено: "
                    + "; ".join(_short(_strip_engine(e), 160) for e in report.errors[:3])
                )
                break
            attempt += 1
            ctx.emit(
                f"XeLaTeX не собрал PDF: исправление разметки, попытка {attempt}/{MAX_FIX_ATTEMPTS}"
            )
            fixed = _repair_attempt(ctx, master_md, report, current, baseline, attempt, result)
            if fixed is None:
                continue
            _write_atomic(master_md, fixed if fixed.endswith("\n") else fixed + "\n")
            current = fixed
            fixes += 1
            report = _render(ctx, master_md, "xelatex", keep_build=True)
            if report.ok:
                ctx.emit("Разметка исправлена, PDF собран")
        if not report.ok:
            _save_log(ctx, report)
            xelatex_errors = [_strip_engine(e) for e in report.errors]
            _cleanup_build(report)
            if render_cfg.fallback_html:
                ctx.emit("XeLaTeX не справился: запасной движок HTML → браузер")
                report = _render(ctx, master_md, "html", keep_build=False)
                fallback_used = report.ok
                if not report.ok:
                    result.errors += [f"XeLaTeX: {_short(e, 300)}" for e in xelatex_errors]
                else:
                    shown = "; ".join(_short(e, 200) for e in xelatex_errors[:3])
                    result.warnings.append(
                        "XeLaTeX не собрал PDF"
                        + (f" ({shown})" if shown else "")
                        + (f"; разметку исправляли {fixes} раз" if fixes else "")
                        + ". PDF собран запасным движком HTML → браузер: без переносов русских "
                        "слов и номеров страниц в оглавлении. Лог XeLaTeX: "
                        "synthesis/render.xelatex.log"
                    )
            else:
                result.errors += [f"XeLaTeX: {_short(e, 300)}" for e in xelatex_errors]
                result.errors.append("Запасной HTML-движок выключен (render.fallback_html = false)")
        else:
            _cleanup_build(report)

    result.warnings += [w for w in report.warnings if w not in result.warnings]
    if report.ok and report.pdf is not None:
        result.ok = True
    else:
        result.errors += [e for e in report.errors if e not in result.errors]
        if not result.errors:
            result.errors.append("PDF не собран")
    result.details = _details(report, fixes=fixes, fallback=fallback_used, image_links=image_links)
    key_after = render_key(ctx, current) if current != text0 else key
    return finish(key=key, key_after=key_after, details=result.details, fixes=fixes)
