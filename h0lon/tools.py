"""Discovery of external tools: pandoc, XeLaTeX, browsers, agent CLIs, media tools."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from h0lon import procutil

IS_WINDOWS = sys.platform == "win32"


@dataclass
class ToolInfo:
    name: str
    path: Path | None
    version: str | None = None
    detail: str = ""

    @property
    def found(self) -> bool:
        return self.path is not None


def which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found else None


def _first_existing(candidates: list[Path]) -> Path | None:
    for c in candidates:
        if c.is_file():
            return c
    return None


def tool_version(argv: list[str], args: tuple[str, ...] = ("--version",)) -> str | None:
    """First non-empty line of `<tool> --version`, or None if it cannot run."""
    res = procutil.run([*argv, *args], timeout=30)
    if res.error or res.timed_out:
        return None
    for line in (res.stdout or res.stderr).splitlines():
        if line.strip():
            return line.strip()
    return None


# ---------------------------------------------------------------- pandoc


def find_pandoc(override: str = "") -> Path | None:
    """Config override > PATH > pandoc bundled with pypandoc-binary."""
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else which(override)
    on_path = which("pandoc")
    if on_path:
        return on_path
    try:
        import pypandoc  # type: ignore[import-untyped]

        bundled = Path(pypandoc.get_pandoc_path())
        for candidate in (bundled, bundled.with_name(bundled.name + ".exe")):
            if candidate.is_file():
                return candidate
    except (ImportError, OSError):
        pass
    return None


# ---------------------------------------------------------------- TeX


def _tex_candidates() -> list[Path]:
    exe = "xelatex.exe" if IS_WINDOWS else "xelatex"
    out: list[Path] = []
    if IS_WINDOWS:
        local = os.environ.get("LOCALAPPDATA")
        pf = os.environ.get("PROGRAMFILES")
        if local:
            out.append(Path(local) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / exe)
        if pf:
            out.append(Path(pf) / "MiKTeX" / "miktex" / "bin" / "x64" / exe)
        texlive = Path("C:/texlive")
        if texlive.is_dir():
            for year in sorted(texlive.iterdir(), reverse=True):
                out.append(year / "bin" / "windows" / exe)
                out.append(year / "bin" / "win64" / exe)
    else:
        out += [Path("/Library/TeX/texbin") / exe, Path("/usr/bin") / exe]
        for root in (Path("/usr/local/texlive"), Path.home() / "texlive"):
            if root.is_dir():
                for year in sorted(root.iterdir(), reverse=True):
                    out += sorted((year / "bin").glob(f"*/{exe}"))
    return out


def find_xelatex(override: str = "") -> Path | None:
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else which(override)
    return which("xelatex") or _first_existing(_tex_candidates())


def find_tex_tool(xelatex: Path, name: str) -> Path | None:
    """A sibling TeX binary (kpsewhich, mpm, initexmf, tlmgr) next to xelatex, else PATH."""
    exe = f"{name}.exe" if IS_WINDOWS else name
    sibling = xelatex.parent / exe
    return sibling if sibling.is_file() else which(name)


def is_miktex(xelatex: Path) -> bool:
    if "miktex" in str(xelatex).lower():
        return True
    version = tool_version([str(xelatex)]) or ""
    return "miktex" in version.lower()


def missing_tex_files(xelatex: Path, files: list[str], *, timeout: float = 60) -> list[str]:
    """Files (e.g. 'mdframed.sty') that kpsewhich cannot find. Empty list = all present."""
    kpse = find_tex_tool(xelatex, "kpsewhich")
    if kpse is None or not files:
        return list(files)
    res = procutil.run([str(kpse), *files], timeout=timeout)
    if res.timed_out:
        raise TimeoutError(f"kpsewhich не ответил за {timeout:.0f} с")
    found = {Path(line.strip()).name.lower() for line in res.stdout.splitlines() if line.strip()}
    return [f for f in files if f.lower() not in found]


# ---------------------------------------------------------------- browsers (HTML → PDF)

_BROWSER_NAMES = {
    "edge": ["msedge", "microsoft-edge", "microsoft-edge-stable"],
    "chrome": ["chrome", "google-chrome", "google-chrome-stable"],
    "chromium": ["chromium", "chromium-browser"],
}


def _browser_paths(kind: str) -> list[Path]:
    if IS_WINDOWS:
        roots = [os.environ.get(v) for v in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
        rel = {
            "edge": ["Microsoft/Edge/Application/msedge.exe"],
            "chrome": ["Google/Chrome/Application/chrome.exe"],
            "chromium": ["Chromium/Application/chrome.exe"],
        }[kind]
        return [Path(r) / p for r in roots if r for p in rel]
    if sys.platform == "darwin":
        app = {
            "edge": "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "chrome": "Google Chrome.app/Contents/MacOS/Google Chrome",
            "chromium": "Chromium.app/Contents/MacOS/Chromium",
        }[kind]
        return [Path("/Applications") / app]
    return []


def find_browser(preference: str = "auto") -> Path | None:
    """Chromium-family browser for headless print-to-pdf.

    preference: auto | edge | chrome | chromium | explicit path.
    """
    if preference not in ("auto", "edge", "chrome", "chromium"):
        p = Path(preference).expanduser()
        return p if p.is_file() else which(preference)
    kinds = ["edge", "chrome", "chromium"] if preference == "auto" else [preference]
    for kind in kinds:
        for name in _BROWSER_NAMES[kind]:
            found = which(name)
            if found:
                return found
        found = _first_existing(_browser_paths(kind))
        if found:
            return found
    return None


# ---------------------------------------------------------------- agent CLIs

_CODEX_TRIPLES = {
    ("win32", "AMD64"): ("codex-win32-x64", "x86_64-pc-windows-msvc"),
    ("win32", "ARM64"): ("codex-win32-arm64", "aarch64-pc-windows-msvc"),
}


def _codex_native_from_shim(shim: Path) -> Path | None:
    """For an npm shim (codex.cmd/.ps1/sh), return the vendored native codex binary if present."""
    import platform

    key = (sys.platform, platform.machine().upper())
    if key not in _CODEX_TRIPLES:
        return None
    pkg, triple = _CODEX_TRIPLES[key]
    exe = "codex.exe" if IS_WINDOWS else "codex"
    base = shim.parent / "node_modules" / "@openai"
    candidates = [
        base / "codex" / "node_modules" / "@openai" / pkg / "vendor" / triple / "bin" / exe,
        base / pkg / "vendor" / triple / "bin" / exe,
        base / "codex" / "vendor" / triple / "bin" / exe,
    ]
    return _first_existing(candidates)


def resolve_agent_argv(spec: str | list[str], kind: str) -> list[str] | None:
    """Turn a configured agent `bin` into an argv prefix, or None if not found.

    kind: 'claude' | 'codex'. For codex installed via npm on Windows the native codex.exe
    is used instead of the .cmd shim (cmd.exe mangles quoted arguments).
    """
    if isinstance(spec, list):
        if not spec:
            return None
        head = Path(spec[0]).expanduser()
        resolved = head if head.is_file() else which(spec[0])
        return [str(resolved), *spec[1:]] if resolved else None

    p = Path(spec).expanduser()
    found: Path | None = p if p.is_file() else which(spec)
    if found is None and kind == "claude" and spec == "claude":
        exe = "claude.exe" if IS_WINDOWS else "claude"
        found = _first_existing([Path.home() / ".local" / "bin" / exe])
    if found is None:
        return None
    if kind == "codex" and found.suffix.lower() in (".cmd", ".ps1", ".bat", ""):
        native = _codex_native_from_shim(found)
        if native is not None:
            return [str(native)]
    return [str(found)]


# ---------------------------------------------------------------- misc tools


def find_simple(name: str) -> Path | None:
    """ffmpeg, ffprobe, yt-dlp, git, nvidia-smi, node, npm …"""
    return which(name)
