"""XeLaTeX compilation in a build directory and log analysis."""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from h0lon import procutil, tools

PASS_TIMEOUT_S = 600
# Files whose stability means another pass would not change the PDF.
STABLE_EXTS = (".aux", ".toc", ".out", ".lof", ".lot")
_MAX_PRINT_LINE = "1000"

_FILE_LINE_RE = re.compile(
    r"^((?:[A-Za-z]:)?[^:\s][^:]*?\.(?:tex|sty|cls|def|cfg|ltx|fd|ldf|lua|aux|toc)):(\d+): (.*)$"
)
_FATAL_NOISE = ("Emergency stop.", "==> Fatal error occurred, no output PDF file produced!")
_MISSING_FILE_RE = re.compile(r"File [`']([^'`]+)' not found")
_MISSING_CHAR_RE = re.compile(r"Missing character: There is no (.+?) in font (.+?)!")
_UNDEF_REF_RE = re.compile(
    r"LaTeX Warning: (?:Hyper reference|Reference|Citation) [`']([^']+)' on page \S+ undefined"
)
_MULTI_LABEL_RE = re.compile(r"LaTeX Warning: Label [`']([^']+)' multiply defined")
_OVERFULL_RE = re.compile(
    r"^Overfull \\hbox \(([\d.]+)pt too wide\) (?:in paragraph|detected) at lines? (\d+)"
)
_FONT_SHAPE_RE = re.compile(r"LaTeX Font Warning: Font shape [`']([^']+)' undefined")
_PKG_WARNING_RE = re.compile(r"^Package (\S+) Warning: (.*)$")
_FONTSPEC_ERROR_RE = re.compile(r'Package fontspec Error: The font "([^"]+)" cannot be found')
_RERUN_RE = re.compile(
    r"Rerun to get|Label\(s\) may have changed|Rerun LaTeX|There were undefined references"
)

# Package warnings that are expected with this template and say nothing about the document.
_IGNORED_PKG_WARNINGS = (
    ("unicode-math", ""),
    ("hyperref", "Rerun to get"),
    ("rerunfilecheck", ""),
    ("polyglossia", ""),
    ("fancyhdr", "\\headheight is too small"),
    ("caption", "Unknown document class"),
)
_OVERFULL_MIN_PT = 1.0


@dataclass
class LatexLog:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    missing_chars: list[str] = field(default_factory=list)  # "«X» (шрифт font) ×N"
    undefined_refs: list[str] = field(default_factory=list)
    overfull: list[str] = field(default_factory=list)  # "12.3pt, строка 120"
    rerun_needed: bool = False

    def summary_warnings(self) -> list[str]:
        """Warnings ready for the user (Russian), most important first."""
        out: list[str] = []
        if self.missing_chars:
            shown = "; ".join(self.missing_chars[:10])
            more = len(self.missing_chars) - 10
            out.append(
                "Шрифт не содержит символы (в PDF они пропадут): "
                + shown
                + (f" и ещё {more}" if more > 0 else "")
            )
        if self.undefined_refs:
            out.append("Неразрешённые ссылки: " + ", ".join(self.undefined_refs))
        out += self.warnings
        if self.overfull:
            shown = ", ".join(self.overfull[:5])
            more = f" и ещё {len(self.overfull) - 5}" if len(self.overfull) > 5 else ""
            out.append(f"Строки выходят за поле (Overfull \\hbox): {shown}{more}")
        return out


@dataclass
class LatexResult:
    ok: bool
    pdf: Path | None
    passes: int
    log_path: Path | None
    log: LatexLog | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def _shorten_font(font: str) -> str:
    font = font.strip().strip('"').strip("[]")
    for sep in (":", "/"):
        font = font.split(sep, 1)[0]
    return font.strip().strip("[]").strip('"') or font


def _error_context(lines: list[str], start: int) -> str:
    """The 'l.<n> <text>' line TeX prints after an error (if close by)."""
    for j in range(start + 1, min(start + 12, len(lines))):
        s = lines[j].strip()
        if re.match(r"^l\.\d+", s):
            return s
    return ""


def parse_log(text: str) -> LatexLog:
    """Extract errors and important warnings from a (Xe)LaTeX log."""
    log = LatexLog()
    lines = text.splitlines()
    missing_chars: Counter[tuple[str, str]] = Counter()
    overfull: list[str] = []
    seen_errors: set[str] = set()
    seen_warnings: set[str] = set()
    undefined: dict[str, None] = {}

    def add_error(msg: str) -> None:
        if msg not in seen_errors:
            seen_errors.add(msg)
            log.errors.append(msg)

    def add_warning(msg: str) -> None:
        if msg not in seen_warnings:
            seen_warnings.add(msg)
            log.warnings.append(msg)

    for i, line in enumerate(lines):
        m = _MISSING_FILE_RE.search(line)
        if m and ("Error" in line or line.startswith("!") or _FILE_LINE_RE.match(line)):
            name = m.group(1)
            if name not in log.missing_files:
                log.missing_files.append(name)
            continue
        fm = _FONTSPEC_ERROR_RE.search(line)
        if fm:
            add_error(
                f"Шрифт «{fm.group(1)}» не найден в системе "
                "(задайте другой в [render] main_font/sans_font/mono_font/math_font)"
            )
            continue

        fl = _FILE_LINE_RE.match(line)
        if fl:
            body = fl.group(3).strip()
            if body in _FATAL_NOISE:
                continue
            ctx = _error_context(lines, i)
            msg = f"{Path(fl.group(1)).name}:{fl.group(2)}: {body}"
            add_error(f"{msg} ({ctx})" if ctx else msg)
            continue
        if line.startswith("! "):
            body = line[2:].strip()
            if body in _FATAL_NOISE:
                continue
            ctx = _error_context(lines, i)
            add_error(f"{body} ({ctx})" if ctx else body)
            continue

        mc = _MISSING_CHAR_RE.search(line)
        if mc:
            missing_chars[(mc.group(1).strip(), _shorten_font(mc.group(2)))] += 1
            continue
        ur = _UNDEF_REF_RE.search(line)
        if ur:
            undefined.setdefault(ur.group(1), None)
            continue
        ml = _MULTI_LABEL_RE.search(line)
        if ml:
            add_warning(f"Метка определена несколько раз: {ml.group(1)}")
            continue
        ov = _OVERFULL_RE.match(line)
        if ov:
            if float(ov.group(1)) > _OVERFULL_MIN_PT:
                overfull.append(f"{float(ov.group(1)):.1f}pt, строка {ov.group(2)}")
            continue
        fs = _FONT_SHAPE_RE.search(line)
        if fs:
            add_warning(f"Начертание шрифта недоступно, будет замена: {fs.group(1)}")
            continue
        pw = _PKG_WARNING_RE.match(line)
        if pw:
            pkg, msg = pw.group(1), pw.group(2).strip()
            # Continuation lines look like "(pkg)   more text".
            j = i + 1
            prefix = f"({pkg})"
            while j < len(lines) and lines[j].startswith(prefix):
                msg += " " + lines[j][len(prefix) :].strip()
                j += 1
            msg = re.sub(r"\s+", " ", msg).strip()
            msg = re.sub(r" on input line \d+\.?$", "", msg)
            if any(pkg == p and msg.startswith(s) for p, s in _IGNORED_PKG_WARNINGS):
                continue
            if pkg == "hyperref" and msg.startswith("Token not allowed"):
                add_warning("hyperref: разметка в заголовке не попадёт в закладки PDF")
                continue
            add_warning(f"{pkg}: {msg}")
            continue
        if _RERUN_RE.search(line):
            log.rerun_needed = True

    log.missing_chars = [
        f"«{ch}» ({font}) ×{n}" if n > 1 else f"«{ch}» ({font})"
        for (ch, font), n in missing_chars.items()
    ]
    log.undefined_refs = list(undefined)
    log.overfull = overfull
    for name in log.missing_files:
        hint = ""
        if name.lower().endswith((".sty", ".cls")):
            hint = " (установите пакет через MiKTeX Console или tlmgr; проверка — h0lon doctor)"
        add_error(f"Не найден файл LaTeX: {name}{hint}")
    return log


def xelatex_argv(xelatex: Path, tex_name: str, *, miktex: bool) -> list[str]:
    argv = [str(xelatex)]
    if miktex:
        # Never pop up the MiKTeX package installer: a missing package is a clear error.
        argv += ["--disable-installer", f"-max-print-line={_MAX_PRINT_LINE}"]
    argv += ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error", tex_name]
    return argv


def _snapshot(build_dir: Path, jobname: str) -> dict[str, str]:
    snap: dict[str, str] = {}
    for ext in STABLE_EXTS:
        p = build_dir / f"{jobname}{ext}"
        if p.is_file():
            snap[ext] = hashlib.sha1(p.read_bytes()).hexdigest()
    return snap


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def compile_latex(
    tex: Path,
    *,
    xelatex: Path,
    passes: int = 3,
    timeout: float = PASS_TIMEOUT_S,
    on_pass: Callable[[int], None] | None = None,
) -> LatexResult:
    """Run XeLaTeX in tex.parent until .aux/.toc are stable (≥2 passes) or `passes` is hit.

    stdout is captured in memory only: <jobname>.out belongs to hyperref (bookmarks).
    """
    start = time.monotonic()
    build_dir = tex.parent
    jobname = tex.stem
    log_path = build_dir / f"{jobname}.log"
    pdf = build_dir / f"{jobname}.pdf"
    max_passes = max(1, passes)
    miktex = tools.is_miktex(xelatex)
    argv = xelatex_argv(xelatex, tex.name, miktex=miktex)
    env = procutil.clean_env(
        drop_session_vars=False,
        extra={"max_print_line": _MAX_PRINT_LINE, "error_line": "254", "half_error_line": "238"},
    )

    result = LatexResult(ok=False, pdf=None, passes=0, log_path=None, log=None)
    previous = _snapshot(build_dir, jobname)
    log = LatexLog()
    for n in range(1, max_passes + 1):
        if on_pass is not None:
            on_pass(n)
        res = procutil.run(argv, cwd=build_dir, env=env, timeout=timeout)
        result.passes = n
        if res.error:
            result.errors.append(f"Не удалось запустить XeLaTeX: {res.error}")
            break
        log = parse_log(_read_log(log_path))
        if res.timed_out:
            result.errors.append(f"XeLaTeX не уложился в {int(timeout)} с (проход {n})")
            break
        if res.exit_code != 0:
            result.errors += log.errors or [
                f"XeLaTeX завершился с кодом {res.exit_code} на проходе {n}; см. {log_path}"
            ]
            break
        current = _snapshot(build_dir, jobname)
        if n >= 2 and current == previous:
            break
        previous = current
    else:
        if log.rerun_needed:
            result.warnings.append(
                f"Перекрёстные ссылки не стабилизировались за {max_passes} прохода(ов); "
                "увеличьте [render] passes"
            )

    result.log = log
    result.log_path = log_path if log_path.is_file() else None
    result.duration_s = time.monotonic() - start
    if result.errors:
        return result
    if not pdf.is_file() or pdf.stat().st_size == 0:
        result.errors.append(f"XeLaTeX не создал PDF ({pdf.name}); см. {log_path}")
        return result
    result.warnings = log.summary_warnings() + result.warnings
    result.pdf = pdf
    result.ok = True
    return result
