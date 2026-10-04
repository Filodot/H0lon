"""Variations of the master (M7, part 1): a derived document from `master.md` (PRD 10.1).

Contract: docs/ARCHITECTURE.md, «Вариации (M7, часть 1)». A variation lives in
`<topic>/variants/<slug>/` as `variant.md`, `variant.pdf` and `meta.json`.

- Presets `brief`, `study`, `cheatsheet`, `custom` run one agent (prompt `variant_<preset>@1.0`,
  decision A12: `study` and `custom` on the strong tier, the others on the light one). The
  agent gets a copy of the master (`inputs/master.md`) and, for `custom`, the request of the user
  (`inputs/request.md`) and writes `out/variant.md`.
- `template` (a template name alone, no preset) re-typesets the master with another template and
  needs no agent: `variant.md` is then the master text itself.
- Agent output is checked (non-empty Markdown with at least one heading) and cleaned by code:
  front matter, HTML comments (`<!-- src: … -->`) and source anchors `[[P1:p3]]` are removed, image
  paths are moved from the topic root to `variants/<slug>/`. The front matter of the variation
  (title «<тема> — <название пресета>», course, date, sources) is written by this module.
- «Устарело»: the sha256 of `master.md` differs from `meta.json`. Running the same variation again
  rebuilds it when the master changed or `force` is set; otherwise the result is taken from the
  cache. The cache has two levels: the agent text (key: master, preset, request, prompt version,
  model) and the PDF (key: the text, template files, render settings), so a changed template
  re-renders without asking the agent again.
- A failed render keeps `variant.md` (the agent's work is not lost); `variant.pdf` of the
  previous text is removed, the next run only renders again.
- Slug: the preset; `custom-<first words of the request>` (a different request under the same
  words gets `-2`, `-3`, …); `template-<name>` for a re-typeset master. The `cheatsheet` is
  typeset with `a4-compact` unless another template is named, `brief` without a page of contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from h0lon import __version__
from h0lon.agents import ExpectedFile, OutputContract, Tier
from h0lon.agents.validate import markdown_heading_lines
from h0lon.names import slugify
from h0lon.render import get_template, list_templates, render_document
from h0lon.render.pandoc import read_front_matter
from h0lon.render.templates import TemplateInfo
from h0lon.synth.common import (
    model_for,
    prompt_body,
    prompt_id,
    run_agent,
    stage_key,
    write_json_atomic,
)
from h0lon.synth.model import MASTER_MD, BuildContext, EventCallback

if TYPE_CHECKING:
    from rich.console import Console

    from h0lon.config import Settings

STAGE = "variants"  # bundle stage: follows `stages.variants` of the settings
VARIANTS_DIR = "variants"
VARIANT_MD = "variant.md"
VARIANT_PDF = "variant.pdf"
META_FILE = "meta.json"
VARIANTS_VERSION = "variants@1"

# Decision A12: the tier of the agent per preset.
PRESETS: dict[str, Tier] = {
    "brief": "light",
    "cheatsheet": "light",
    "study": "strong",
    "custom": "strong",
}
TEMPLATE_PRESET = "template"  # re-typeset with another template, no agent
PRESET_TITLES: dict[str, str] = {
    "brief": "Кратко",
    "study": "Учебный конспект",
    "cheatsheet": "Шпаргалка",
    "custom": "По запросу",
    TEMPLATE_PRESET: "Другой шаблон",
}
PRESET_ORDER = ("brief", "study", "cheatsheet", "custom", TEMPLATE_PRESET)
# The template a preset is typeset with unless the user names another one.
PRESET_TEMPLATES: dict[str, str] = {"cheatsheet": "a4-compact"}
NO_CONTENTS_PRESETS = frozenset({"brief"})  # short documents are typeset without contents
TIER_LABELS: dict[str, str] = {"light": "лёгкий уровень агента", "strong": "сильный уровень агента"}

# Warnings of the render that say nothing the reader could act on: mdframed in columns
# reports a frame that ends exactly at the foot of a column.
IGNORED_WARNINGS = ("mdframed: You got a bad break",)
MAX_REQUEST_CHARS = 4000
MIN_VARIANT_CHARS = 40
MAX_WARNINGS = 10
SLUG_WORDS = 4
SLUG_MAX_LEN = 40

# The messages that open a phase. The web job stops (on request) right before such a message:
# nothing is interrupted halfway.
AGENT_START = "Вариация: запрос к агенту…"
RENDER_START = "Вариация: сборка PDF…"

# `[[P1:p3]]`, `[[S1:s12]]`, `[[V1:12:34]]`, `[[P2:p45-47]]` (as in render/pandoc.py).
_ANCHOR_RE = re.compile(r"\[\[[A-Z][A-Za-z0-9]*:[^\[\]\s]+\]\]")
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_FRONT_RE = re.compile(
    r"\A\ufeff?\s*---[ \t]*\r?\n(.*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL
)
_BLANKS_RE = re.compile(r"\n{3,}")
# Relative targets of Markdown images and HTML src attributes (not URLs, absolute paths,
# anchors or paths that already climb up).
_REL = r"(?![a-zA-Z][a-zA-Z0-9+.-]*:|/|\\|#|\.\./)"
_MD_IMAGE_RE = re.compile(r"(!\[[^\]\n]*\]\()" + _REL + r"([^)\s]+)")
_HTML_SRC_RE = re.compile(r"""(\bsrc=["'])""" + _REL + r"""([^"']+)""")
_SLUG_OK_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
# `{#sec:s01 .unnumbered}`, `{.definition #def:cdf title="…"}`: ids that the text defines.
_DEFINED_ID_RE = re.compile(r"\{[^{}\n]*?#([A-Za-z][\w:.-]*)")
# `[текст](#sec:s03)`: a link to an id with a colon (see drop_dangling_xrefs).
_XREF_RE = re.compile(r"\[([^\[\]\n]+)\]\(#([A-Za-z][\w.-]*:[\w:.-]+)\)")
BACKSLASH = "\\"


# ---------------------------------------------------------------- result


@dataclass
class VariantResult:
    ok: bool
    slug: str
    dir: Path
    variant_md: Path | None = None
    variant_pdf: Path | None = None
    preset: str = ""
    template: str | None = None
    stale: bool = False
    cached: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    agent_runs: int = 0
    engine: str | None = None
    pages: int | None = None
    duration_s: float = 0.0

    @property
    def title(self) -> str:
        return preset_title(self.preset, self.template)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "slug": self.slug,
            "dir": str(self.dir),
            "variant_md": str(self.variant_md) if self.variant_md else None,
            "variant_pdf": str(self.variant_pdf) if self.variant_pdf else None,
            "preset": self.preset,
            "title": self.title,
            "template": self.template,
            "stale": self.stale,
            "cached": self.cached,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "agent_runs": self.agent_runs,
            "engine": self.engine,
            "pages": self.pages,
            "duration_s": round(self.duration_s, 1),
        }

    def message(self) -> str:
        """One Russian sentence about the outcome (for the web job and the CLI)."""
        if self.ok:
            name = (self.variant_pdf or self.variant_md or self.dir).name
            how = "из кэша" if self.cached else "собрана"
            return f"Вариация «{self.title}» {how}: {VARIANTS_DIR}/{self.slug}/{name}"
        why = "; ".join(self.errors[:2]) or "причина не указана"
        return f"Вариация «{self.title}» не собрана: {why}"

    def print(self, console: Console) -> None:
        from rich.markup import escape

        if self.ok:
            how = " [cyan](из кэша: мастер не менялся)[/cyan]" if self.cached else ""
            console.print(f"[bold green]Вариация «{escape(self.title)}» готова[/bold green]{how}")
            if self.variant_pdf:
                console.print(f"PDF: {escape(str(self.variant_pdf))}")
            if self.variant_md:
                console.print(f"Markdown: {escape(str(self.variant_md))}")
        else:
            console.print(f"[bold red]Вариация «{escape(self.title)}» не собрана[/bold red]")
            if self.variant_md and self.variant_md.is_file():
                console.print(
                    f"Текст сохранён: {escape(str(self.variant_md))}; "
                    "повторный запуск только пересоберёт PDF."
                )
        details = [f"шаблон: {self.template}"] if self.template else []
        if self.engine:
            details.append(f"движок: {self.engine}")
        if self.pages:
            details.append(f"страниц: {self.pages}")
        if self.agent_runs:
            details.append(f"прогонов агента: {self.agent_runs}")
        if details:
            console.print(escape(", ".join(details)))
        for w in self.warnings[:MAX_WARNINGS]:
            console.print(f"[yellow]Предупреждение:[/yellow] {escape(w)}")
        if len(self.warnings) > MAX_WARNINGS:
            console.print(f"[yellow]… и ещё {len(self.warnings) - MAX_WARNINGS}[/yellow]")
        for e in self.errors:
            console.print(f"[red]Ошибка:[/red] {escape(e)}")
        console.print(f"[dim]Время: {self.duration_s:.1f} с[/dim]")


# ---------------------------------------------------------------- small helpers


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def variants_dir(topic_dir: Path) -> Path:
    return Path(topic_dir) / VARIANTS_DIR


def read_variant_meta(variant_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((variant_dir / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def preset_title(preset: str | None, template: str | None = None) -> str:
    """Display name of a preset; for a template variation the template is named."""
    if preset == TEMPLATE_PRESET and template:
        return f"{PRESET_TITLES[TEMPLATE_PRESET]}: {template}"
    return PRESET_TITLES.get(preset or "", preset or "—")


# ---------------------------------------------------------------- request and slug


def _template_names(settings: Settings) -> str:
    return ", ".join(t.name for t in list_templates(settings)) or "нет"


def _load_template(settings: Settings, name: str) -> TemplateInfo:
    try:
        return get_template(name, settings)
    except KeyError:
        raise ValueError(
            f"Шаблон «{name}» не найден. Доступны: {_template_names(settings)}."
        ) from None


def normalise_request(
    preset: str | None, prompt: str | None, template: str | None
) -> tuple[str, str | None, str | None]:
    """(preset, request, template) of a call; ValueError (Russian) for a wrong combination.

    `preset` may be `template:<имя>`; a missing preset is `custom` when there is a request and
    `template` when there is only a template.
    """
    preset = (preset or "").strip().lower()
    template = (template or "").strip() or None
    prompt = (prompt or "").strip() or None
    if preset.startswith(f"{TEMPLATE_PRESET}:"):
        template = template or preset.split(":", 1)[1].strip() or None
        preset = TEMPLATE_PRESET
    if not preset:
        preset = "custom" if prompt else TEMPLATE_PRESET if template else ""
    if not preset:
        raise ValueError(
            "Укажите пресет: " + ", ".join(PRESETS) + " (или только шаблон: --template <имя>)."
        )
    if preset != TEMPLATE_PRESET and preset not in PRESETS:
        known = ", ".join((*PRESETS, TEMPLATE_PRESET))
        raise ValueError(f"Неизвестный пресет «{preset}»: допустимо {known}.")
    if preset == "custom":
        if not prompt:
            raise ValueError("Для пресета custom нужен запрос: --prompt «что должно получиться».")
        if len(prompt) > MAX_REQUEST_CHARS:
            raise ValueError(
                f"Запрос слишком длинный: {len(prompt)} символов, допустимо до {MAX_REQUEST_CHARS}."
            )
    elif prompt:
        raise ValueError(
            "Запрос (--prompt) можно задать только для пресета custom; "
            f"для «{preset}» он не используется."
        )
    if preset == TEMPLATE_PRESET and not template:
        raise ValueError("Для перевёрстки укажите шаблон: --template <имя>.")
    return preset, prompt, template


def _custom_slug_base(request: str) -> str:
    words = " ".join(re.findall(r"\w+", request)[:SLUG_WORDS])
    slug = slugify(words, max_len=SLUG_MAX_LEN) if words else ""
    return f"custom-{slug}" if slug and slug != "untitled" else "custom"


def variant_slug(topic_dir: Path, preset: str, request: str | None, template: str | None) -> str:
    """Directory name of a variation: the preset; `custom-<первые слова запроса>`; for a
    template `template-<имя>`. A custom request never overwrites another request: a different
    text under the same words gets `-2`, `-3`, …"""
    if preset == TEMPLATE_PRESET:
        return f"template-{slugify(template or '', max_len=SLUG_MAX_LEN)}"
    if preset != "custom":
        return preset
    base = _custom_slug_base(request or "")
    root = variants_dir(topic_dir)
    slug, n = base, 1
    while True:
        meta = read_variant_meta(root / slug)
        if meta is None or (meta.get("prompt") or "") == (request or ""):
            return slug
        n += 1
        slug = f"{base}-{n}"


# ---------------------------------------------------------------- text of the variation


def _map_text(text: str, fn: Any) -> str:
    """Apply `fn` to the text outside fenced code blocks (the code is left as it is)."""
    out: list[str] = []
    chunk: list[str] = []
    fence: str | None = None

    def flush() -> None:
        if chunk:
            out.append(fn("".join(chunk)))
            chunk.clear()

    for line in text.splitlines(keepends=True):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group(1)[0] * 3
            if fence is None:
                flush()
                fence = marker
                out.append(line)
            elif line.lstrip().startswith(fence):
                fence = None
                out.append(line)
            else:
                out.append(line)
        elif fence is not None:
            out.append(line)
        else:
            chunk.append(line)
    flush()
    return "".join(out)


def strip_front_matter(text: str) -> str:
    """Remove a YAML block at the very top (the module writes the front matter itself)."""
    m = _FRONT_RE.match(text)
    if m and read_front_matter(text):
        return text[m.end() :]
    return text


def _clean_segment(text: str) -> str:
    text = _COMMENT_RE.sub("", text)
    text = text.replace("<!--", "")  # an opener that never closes would hide the rest
    text = _ANCHOR_RE.sub("", text)
    return _BLANKS_RE.sub("\n\n", text)


def clean_variant_text(text: str) -> str:
    """Agent output → body of the variation: no front matter, comments or source anchors."""
    text = text.replace("\r\n", "\n").lstrip("\ufeff")
    text = _map_text(strip_front_matter(text), _clean_segment)
    return drop_dangling_xrefs(text).strip() + "\n"


def drop_dangling_xrefs(text: str) -> str:
    """`[текст](#sec:s03)` → `текст` when the variation has no `{#sec:s03}`.

    The master links to its sections by ids like `sec:s03` or `def:cdf`; an agent that rewrote
    the document may keep such a link and drop the target. Only ids with a colon are touched
    (Pandoc's own ids never have one), so links to headings by their generated ids stay.
    """
    defined: set[str] = set()

    def collect(segment: str) -> str:
        defined.update(_DEFINED_ID_RE.findall(segment))
        return segment

    _map_text(text, collect)

    def unlink(m: re.Match[str]) -> str:
        return m.group(0) if m.group(2) in defined else m.group(1)

    return _map_text(text, lambda segment: _XREF_RE.sub(unlink, segment))


def relocate_images(text: str) -> str:
    """Image paths relative to the topic root (`extracted/P1/figures/a.png`) → relative to
    `variants/<slug>/` (`../../extracted/…`): the variation is rendered from its own directory."""

    def move(m: re.Match[str]) -> str:
        return f"{m.group(1)}../../{m.group(2).replace(BACKSLASH, '/')}"

    return _map_text(text, lambda seg: _HTML_SRC_RE.sub(move, _MD_IMAGE_RE.sub(move, seg)))


def _master_front(master_text: str) -> dict[str, Any]:
    front = read_front_matter(master_text)
    return front if isinstance(front, dict) else {}


def _front_matter(topic_dir: Path, master_front: dict[str, Any], preset: str) -> str:
    """Front matter of an agent variation: title «<тема> — <название пресета>», course, date."""
    title = str(master_front.get("title") or "").strip()
    course = str(master_front.get("course") or "").strip()
    if not title or not course:
        try:
            from h0lon.workspace import load_topic

            meta = load_topic(topic_dir)
            title, course = title or meta.title, course or meta.course
        except (OSError, ValueError):
            pass
    name = PRESET_TITLES[preset]
    data: dict[str, Any] = {"title": f"{title or Path(topic_dir).name} — {name}", "subtitle": name}
    if course:
        data["course"] = course
    data["date"] = date.today().isoformat()
    data["author"] = "H0lon"
    if master_front.get("sources"):
        data["sources"] = master_front["sources"]
    if preset in NO_CONTENTS_PRESETS:
        data["toc"] = False  # 2–4 pages: a page of contents would be a quarter of the document
    body = yaml.safe_dump(
        data, allow_unicode=True, sort_keys=False, default_flow_style=False, width=10_000
    )
    return f"---\n{body}---\n"


# ---------------------------------------------------------------- the agent


def _contract() -> OutputContract:
    return OutputContract(
        files=[
            ExpectedFile(
                path="variant.md",
                kind="markdown",
                min_chars=MIN_VARIANT_CHARS,
                description="Готовый документ: Markdown с формулами LaTeX, с первого заголовка.",
            )
        ]
    )


def _generate_text(
    ctx: BuildContext, preset: str, request: str | None, master: Path, result: VariantResult
) -> str | None:
    """One agent run; the cleaned body of the variation, or None (the reason is in `result`)."""
    name = f"variant_{preset}"
    tier = PRESETS[preset]
    ctx.emit(AGENT_START)
    ctx.emit(f"Вариация «{PRESET_TITLES[preset]}»: {TIER_LABELS[tier]}.")
    result.agent_runs += 1
    try:
        with tempfile.TemporaryDirectory(
            prefix="h0lon-variant-", ignore_cleanup_errors=True
        ) as tmp:
            inputs = [master]
            if preset == "custom":
                req = Path(tmp) / "request.md"
                req.write_text(f"# Запрос пользователя\n\n{request}\n", encoding="utf-8")
                inputs.append(req)
            run = run_agent(
                ctx,
                stage=STAGE,
                task=prompt_body(name),
                contract=_contract(),
                inputs=inputs,
                tier=tier,
            )
    except Exception as exc:  # a broken bundle or runner: a message, not a traceback
        result.errors.append(f"Агент не запущен ({type(exc).__name__}: {exc})")
        return None
    if not run.ok:
        result.errors.append(
            "Агент не вернул результат: " + ("; ".join(run.problems[:3]) or "причина не указана")
        )
        return None
    path = run.bundle.out_dir / "variant.md"
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        result.errors.append(f"Агент не создал variant.md: {exc}")
        return None
    body = clean_variant_text(raw)
    if not markdown_heading_lines(body):
        result.errors.append(
            "В результате агента нет ни одного заголовка (строки, начинающейся с #): "
            "документ не принят."
        )
        return None
    return relocate_images(body)


# ---------------------------------------------------------------- the stage


def _template_fingerprint(tpl: TemplateInfo) -> str:
    return stage_key(
        tpl.name,
        tpl.template_tex,
        tpl.template_html,
        tpl.html_css,
    )


def _render_key(settings: Settings, md_sha: str | None, tpl: TemplateInfo) -> str:
    return stage_key(
        VARIANTS_VERSION,
        __version__,
        md_sha,
        _template_fingerprint(tpl),
        settings.render.model_dump(mode="json"),
    )


def _pdf_pages(report: Any) -> int | None:
    checks = getattr(report, "checks", None)
    return getattr(checks, "pages", None) if checks is not None else None


def run_variant(
    settings: Settings,
    topic_dir: Path,
    *,
    preset: str | None = None,
    prompt: str | None = None,
    template: str | None = None,
    backend: str | None = None,
    force: bool = False,
    on_event: EventCallback | None = None,
) -> VariantResult:
    """Build (or take from the cache) one variation of `<topic>/master.md`.

    Raises ValueError (Russian text) for a wrong request: unknown preset, `custom` without a
    request, an unknown template or agent. A failure of the build itself is `ok=False` with
    `errors`, never an exception.
    """
    started = time.monotonic()
    topic_dir = Path(topic_dir).resolve()
    kind, request, template_name = normalise_request(preset, prompt, template)
    if backend not in (None, "claude", "codex"):
        raise ValueError(f"Неизвестный агент «{backend}»: допустимо claude | codex.")
    effective = template_name or PRESET_TEMPLATES.get(kind) or settings.render.template
    tpl = _load_template(settings, effective)

    slug = variant_slug(topic_dir, kind, request, template_name)
    vdir = variants_dir(topic_dir) / slug
    result = VariantResult(ok=False, slug=slug, dir=vdir, preset=kind, template=tpl.name)
    ctx = BuildContext(
        topic_dir=topic_dir, settings=settings, force=force, backend=backend, on_event=on_event
    )

    def done() -> VariantResult:
        result.duration_s = time.monotonic() - started
        return result

    master = topic_dir / MASTER_MD
    master_sha = _sha256(master)
    if master_sha is None:
        result.errors.append(
            f"Не найден мастер-конспект {master}: сначала соберите его (h0lon build)."
        )
        return done()

    md_path, pdf_path = vdir / VARIANT_MD, vdir / VARIANT_PDF
    prev = read_variant_meta(vdir) or {}
    uses_agent = kind in PRESETS
    agent_key = stage_key(
        VARIANTS_VERSION,
        master_sha,
        kind,
        request,
        prompt_id(f"variant_{kind}") if uses_agent else None,
        model_for(ctx, PRESETS[kind], STAGE) if uses_agent else None,
    )
    text_fresh = not force and prev.get("agent_key") == agent_key and md_path.is_file()

    meta: dict[str, Any] = {
        "slug": slug,
        "preset": kind,
        "prompt": request,
        "template": tpl.name,
        "template_requested": template_name,
        "title": preset_title(kind, template_name),
        "tier": PRESETS.get(kind),
        "model": model_for(ctx, PRESETS[kind], STAGE) if uses_agent else None,
        "prompt_version": prompt_id(f"variant_{kind}") if uses_agent else None,
        "created": (prev.get("created") if text_fresh else None) or _now_iso(),
        "master_sha": master_sha,
        "agent_key": agent_key,
        "render_key": None,
        "backend": backend,
    }

    try:
        if not text_fresh:
            master_text = master.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
            if uses_agent:
                body = _generate_text(ctx, kind, request, master, result)
                if body is None:
                    return done()
                text = _front_matter(topic_dir, _master_front(master_text), kind) + "\n" + body
            else:
                ctx.emit("Вариация: перевёрстка мастера без агента.")
                text = relocate_images(master_text)
            _write_text_atomic(md_path, text if text.endswith("\n") else text + "\n")
            # What the agent made is kept even if the PDF fails: the next run only renders.
            write_json_atomic(vdir / META_FILE, meta)
            pdf_path.unlink(missing_ok=True)
        result.variant_md = md_path

        md_sha = _sha256(md_path)
        render_key = _render_key(settings, md_sha, tpl)
        if text_fresh and prev.get("render_key") == render_key and pdf_path.is_file():
            result.ok = result.cached = True
            result.variant_pdf = pdf_path
            result.engine = prev.get("engine")
            result.pages = prev.get("pages")
            return done()

        ctx.emit(RENDER_START)
        ctx.emit(f"Вариация: шаблон «{tpl.name}».")
        try:
            pdf_path.unlink(missing_ok=True)
        except OSError:
            pass  # open in a viewer: render_document reports the failed copy itself
        report = render_document(md_path, settings=settings, out_dir=vdir, template=tpl.name)
        result.warnings += [w for w in report.warnings if not w.startswith(IGNORED_WARNINGS)]
        result.engine = report.engine_used
        result.pages = _pdf_pages(report)
        if not report.ok or report.pdf is None:
            result.errors += report.errors or ["PDF не собран"]
            write_json_atomic(vdir / META_FILE, {**meta, "render_key": None})
            return done()
        result.ok = True
        result.variant_pdf = report.pdf
        if report.fallback_used:
            result.warnings.insert(0, "PDF собран запасным движком HTML → браузер.")
        write_json_atomic(
            vdir / META_FILE,
            {
                **meta,
                "render_key": render_key,
                "engine": report.engine_used,
                "pages": result.pages,
                "agent_runs": result.agent_runs if not text_fresh else prev.get("agent_runs", 0),
            },
        )
    except OSError as exc:
        result.errors.append(f"Не удалось записать вариацию в {vdir}: {exc}")
    return done()


# ---------------------------------------------------------------- listing


def list_variants(topic_dir: Path) -> list[dict[str, Any]]:
    """Variations of a topic, newest first: slug, preset, prompt, created, master_sha, stale, pdf
    (plus title, template, model, md, engine, pages). `stale`: the master changed since the
    variation was built (or is gone)."""
    root = variants_dir(topic_dir)
    if not root.is_dir():
        return []
    master_sha = _sha256(Path(topic_dir) / MASTER_MD)
    items: list[dict[str, Any]] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not _SLUG_OK_RE.match(entry.name):
            continue
        meta = read_variant_meta(entry)
        if meta is None:
            continue
        has_pdf = (entry / VARIANT_PDF).is_file()
        has_md = (entry / VARIANT_MD).is_file()
        items.append(
            {
                "slug": entry.name,
                "preset": meta.get("preset"),
                "title": meta.get("title")
                or preset_title(meta.get("preset"), meta.get("template")),
                "prompt": meta.get("prompt"),
                "template": meta.get("template"),
                "template_requested": meta.get("template_requested"),
                "model": meta.get("model"),
                "created": meta.get("created"),
                "master_sha": meta.get("master_sha"),
                "stale": master_sha is None or meta.get("master_sha") != master_sha,
                "pdf": f"{VARIANTS_DIR}/{entry.name}/{VARIANT_PDF}" if has_pdf else None,
                "md": f"{VARIANTS_DIR}/{entry.name}/{VARIANT_MD}" if has_md else None,
                "engine": meta.get("engine"),
                "pages": meta.get("pages"),
            }
        )
    items.sort(key=lambda i: (str(i["created"] or ""), i["slug"]), reverse=True)
    return items


def print_variants(items: list[dict[str, Any]], *, console: Console, title: str) -> None:
    from rich.markup import escape
    from rich.table import Table

    table = Table(title=title, title_justify="left", show_lines=False)
    for col in ("Вариация", "Шаблон", "Создана", "Состояние", "PDF"):
        table.add_column(col, overflow="fold")
    for item in items:
        name = escape(str(item["title"]))
        if item.get("prompt"):
            snippet = " ".join(str(item["prompt"]).split())
            name += f"\n[dim]{escape(snippet[:60] + ('…' if len(snippet) > 60 else ''))}[/dim]"
        name += f"\n[dim]{escape(str(item['slug']))}[/dim]"
        state = "[yellow]устарело[/yellow]" if item["stale"] else "[green]актуально[/green]"
        table.add_row(
            name,
            escape(str(item.get("template") or "—")),
            escape(str(item.get("created") or "—")),
            state,
            escape(str(item["pdf"])) if item["pdf"] else "[red]нет[/red]",
        )
    console.print(table)
