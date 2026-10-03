"""Render templates: built-in (h0lon/render/templates/<name>/) and user (<state_dir>/templates/)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from h0lon.config import Settings

PACKAGE_DIR = Path(__file__).resolve().parent
BUILTIN_TEMPLATES_DIR = PACKAGE_DIR / "templates"
FILTERS_DIR = PACKAGE_DIR / "filters"

TEMPLATE_TEX = "template.tex"
TEMPLATE_HTML = "template.html"
META_FILE = "meta.yaml"
DEFAULT_CSS = "style.css"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class TemplateInfo:
    name: str
    dir: Path
    template_tex: Path
    title: str
    latex_packages: list[str] = field(default_factory=list)  # .sty files needed (for doctor)
    html_css: Path | None = None  # stylesheet for the HTML fallback renderer

    @property
    def template_html(self) -> Path | None:
        """Optional Pandoc HTML template next to template.tex (None = Pandoc default)."""
        path = self.dir / TEMPLATE_HTML
        return path if path.is_file() else None


def user_templates_dir(settings: Settings) -> Path:
    return settings.general.state_path / "templates"


def _search_dirs(settings: Settings) -> list[Path]:
    return [user_templates_dir(settings), BUILTIN_TEMPLATES_DIR]


def _normalise_package(name: object) -> str:
    text = str(name).strip()
    return text if text.lower().endswith((".sty", ".cls", ".def", ".tex")) else f"{text}.sty"


def _load(name: str, directory: Path) -> TemplateInfo:
    tex = directory / TEMPLATE_TEX
    meta: dict[str, object] = {}
    meta_path = directory / META_FILE
    if meta_path.is_file():
        try:
            loaded = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, UnicodeDecodeError) as exc:
            raise ValueError(f"Шаблон «{name}»: не удалось прочитать {meta_path}: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"Шаблон «{name}»: {meta_path} должен содержать словарь YAML")
        meta = loaded or {}

    packages_raw = meta.get("latex_packages") or []
    if not isinstance(packages_raw, list):
        raise ValueError(f"Шаблон «{name}»: latex_packages в {meta_path} должен быть списком")
    packages = [_normalise_package(p) for p in packages_raw if str(p).strip()]

    css: Path | None = None
    css_name = meta.get("html_css")
    if css_name:
        candidate = (directory / str(css_name)).resolve()
        css = candidate if candidate.is_file() else None
    elif (directory / DEFAULT_CSS).is_file():
        css = directory / DEFAULT_CSS

    title = str(meta.get("title") or name)
    return TemplateInfo(
        name=name,
        dir=directory,
        template_tex=tex,
        title=title,
        latex_packages=packages,
        html_css=css,
    )


def get_template(name: str, settings: Settings) -> TemplateInfo:
    """Find a template by name; user templates take priority. KeyError if there is none."""
    if not _NAME_RE.match(name or ""):
        raise KeyError(name)
    for base in _search_dirs(settings):
        directory = base / name
        if (directory / TEMPLATE_TEX).is_file():
            return _load(name, directory)
    raise KeyError(name)


def list_templates(settings: Settings) -> list[TemplateInfo]:
    """All usable templates sorted by name (a user template hides a built-in one)."""
    found: dict[str, TemplateInfo] = {}
    for base in _search_dirs(settings):
        if not base.is_dir():
            continue
        for directory in sorted(p for p in base.iterdir() if p.is_dir()):
            name = directory.name
            if name in found or not _NAME_RE.match(name):
                continue
            if not (directory / TEMPLATE_TEX).is_file():
                continue
            try:
                found[name] = _load(name, directory)
            except ValueError:
                continue  # broken meta.yaml: get_template() reports the details
    return [found[k] for k in sorted(found)]
