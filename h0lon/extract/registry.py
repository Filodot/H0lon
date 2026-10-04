"""Extractors by source kind. Modules are imported lazily: a missing or broken module makes
its kinds unsupported with a readable reason instead of breaking `h0lon extract`."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from h0lon.extract.model import Extractor


class ExtractError(Exception):
    """Extraction of one source failed; the message (Russian) is shown to the user."""


# kind -> (module, class). Classes are created without arguments.
EXTRACTORS: dict[str, tuple[str, str]] = {
    "pdf-text": ("h0lon.extract.pdf", "PdfExtractor"),
    "pdf-scan": ("h0lon.extract.pdf", "PdfExtractor"),
    "slides": ("h0lon.extract.slides", "SlidesExtractor"),
    "docx": ("h0lon.extract.docs", "DocsExtractor"),
    "md": ("h0lon.extract.docs", "DocsExtractor"),
    "tex": ("h0lon.extract.docs", "DocsExtractor"),
    "web": ("h0lon.extract.docs", "DocsExtractor"),
    "handwritten": ("h0lon.extract.handwritten", "HandwrittenExtractor"),
}

# Kinds accepted by `h0lon add` whose extraction belongs to a later stage.
LATER_STAGES: dict[str, str] = {
    "video": "Видео — этап M5 (кадры и транскрипт), источник пока пропущен",
    "audio": "Аудио — этап M5 (транскрипт), источник пока пропущен",
}


@dataclass
class Resolution:
    extractor: Extractor | None
    skipped: bool = False  # the kind belongs to a later stage (status `skipped`)
    reason: str | None = None  # why there is no extractor (Russian)


def resolve_extractor(kind: str) -> Resolution:
    if kind in LATER_STAGES:
        return Resolution(None, skipped=True, reason=LATER_STAGES[kind])
    spec = EXTRACTORS.get(kind)
    if spec is None:
        return Resolution(None, reason=f"Вид источника «{kind}» не поддерживается извлечением")
    module_name, class_name = spec
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            reason = f"модуль {module_name} ещё не реализован"
        else:
            reason = f"модуль {module_name} не загрузился: {exc}"
        return Resolution(None, reason=f"Вид «{kind}» пока не поддерживается: {reason}")
    except Exception as exc:  # a broken module must not stop other sources
        return Resolution(
            None,
            reason=f"Вид «{kind}» пока не поддерживается: модуль {module_name} не загрузился "
            f"({type(exc).__name__}: {exc})",
        )
    cls = getattr(module, class_name, None)
    if cls is None:
        return Resolution(
            None,
            reason=f"Вид «{kind}» пока не поддерживается: в {module_name} нет {class_name}",
        )
    try:
        extractor = cls()
    except Exception as exc:
        return Resolution(
            None,
            reason=f"Вид «{kind}» пока не поддерживается: {class_name}() — "
            f"{type(exc).__name__}: {exc}",
        )
    kinds = tuple(getattr(extractor, "kinds", ()) or ())
    if kinds and kind not in kinds:
        return Resolution(
            None, reason=f"Вид «{kind}» пока не поддерживается: {class_name} его не объявляет"
        )
    return Resolution(extractor)


def get_extractor(kind: str) -> Extractor | None:
    """Extractor for a source kind; None — the kind is not supported (yet)."""
    return resolve_extractor(kind).extractor
