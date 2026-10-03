"""`h0lon doctor`: environment checks with statuses and install hints (in Russian)."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from h0lon import procutil, tools
from h0lon.config import Settings

if TYPE_CHECKING:
    from rich.console import Console

Status = Literal["ok", "warn", "missing", "error", "info"]

PROC_TIMEOUT_S = 10.0  # one external command
DEADLINE_S = 14.0  # all checks together (they run in parallel), including every command
KILL_MARGIN_S = 1.5  # commands stop this much earlier: killing a process tree takes time
MIN_PANDOC = (3, 1)
MIN_PYTHON = (3, 11)

IS_WINDOWS = sys.platform == "win32"
_FONTS_KEY = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
_CLAUDE_KEY_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


@dataclass
class Check:
    id: str
    title: str
    status: Status
    detail: str
    hint: str | None = None
    required: bool = False
    data: dict[str, Any] = field(default_factory=dict)  # machine-readable facts (paths, versions)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
            "required": self.required,
            "data": _jsonable(self.data),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _win(windows: str, other: str) -> str:
    return windows if IS_WINDOWS else other


def _short(path: str | os.PathLike[str] | None) -> str:
    """Path for display: the home directory is shown as ~ (JSON data keeps full paths)."""
    if path is None:
        return ""
    text = os.fspath(path)
    home = str(Path.home())
    if text.lower().startswith(home.lower() + os.sep):
        return "~" + text[len(home) :]
    return text


# ---------------------------------------------------------------- process helpers


# Per-thread deadline (time.monotonic) shared by all commands of one doctor run.
_budget = threading.local()
_OUT_OF_TIME = "doctor deadline reached"


def _run(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = PROC_TIMEOUT_S,
) -> procutil.ProcResult:
    """procutil.run with a timeout capped by what is left of the doctor deadline."""
    deadline: float | None = getattr(_budget, "deadline", None)
    if deadline is not None:
        left = deadline - time.monotonic()
        if left <= 0.05:  # the run is over: do not start anything new
            args = [str(a) for a in argv]
            return procutil.ProcResult(args, None, "", "", 0.0, True, _OUT_OF_TIME)
        timeout = min(timeout, left)
    return procutil.run(argv, env=env, cwd=cwd, timeout=timeout)


def _first_line(res: procutil.ProcResult) -> str | None:
    if res.error or res.timed_out:
        return None
    for line in (res.stdout or res.stderr).splitlines():
        if line.strip():
            return line.strip()
    return None


def _failure(res: procutil.ProcResult) -> str:
    if res.error == _OUT_OF_TIME:
        return "не запускался: время проверки истекло"
    if res.timed_out:
        secs = f"{res.duration_s:.0f}" if res.duration_s >= 1 else f"{res.duration_s:.1f}"
        return f"не ответил за {secs} с"
    if res.error:
        return res.error
    text = (res.stderr or res.stdout).strip().splitlines()
    return f"код {res.exit_code}" + (f": {text[0][:200]}" if text else "")


def _version_tuple(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    return tuple(int(g) for g in m.groups() if g is not None)


# ---------------------------------------------------------------- basic checks


def _check_python(settings: Settings) -> Check:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    data = {"version": version, "executable": sys.executable}
    if v[:2] >= MIN_PYTHON:
        return Check(
            "python",
            "Python",
            "ok",
            f"{version} — {_short(sys.executable)}",
            required=True,
            data=data,
        )
    return Check(
        "python",
        "Python",
        "error",
        f"{version}, нужна 3.11 или новее",
        hint="uv python install 3.13 (или winget install Python.Python.3.13)",
        required=True,
        data=data,
    )


def _check_config(settings: Settings) -> Check:
    src = settings.source_path
    overrides = sorted(k for k in os.environ if k.upper().startswith("H0LON_") and "__" in k)
    extra = f"; из окружения: {', '.join(overrides)}" if overrides else ""
    data = {"path": src, "env_overrides": overrides}
    if src is not None:
        return Check("config", "Конфигурация", "ok", f"{_short(src)}{extra}", data=data)
    return Check(
        "config",
        "Конфигурация",
        "info",
        f"файл h0lon.toml не найден — значения по умолчанию{extra}",
        hint="h0lon init — создать h0lon.toml с комментариями",
        data=data,
    )


def _check_workspaces(settings: Settings) -> Check:
    ws = settings.general.workspaces_dir
    data = {"path": ws, "state_dir": settings.general.state_path}
    if ws.is_dir():
        try:
            courses = sum(1 for p in ws.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError as exc:
            return Check(
                "workspaces", "Рабочие области", "error", f"{_short(ws)}: {exc}", data=data
            )
        return Check(
            "workspaces", "Рабочие области", "ok", f"{_short(ws)} (курсов: {courses})", data=data
        )
    if ws.exists():
        return Check(
            "workspaces",
            "Рабочие области",
            "error",
            f"{_short(ws)} существует, но это не каталог",
            hint="укажите другой путь в general.workspaces",
            data=data,
        )
    return Check(
        "workspaces",
        "Рабочие области",
        "warn",
        f"{_short(ws)} — каталога пока нет",
        hint="h0lon init (или каталог появится при первом h0lon new)",
        data=data,
    )


# ---------------------------------------------------------------- agents


# Platform-dependent hints are functions: IS_WINDOWS is read when a check runs.
def _claude_install() -> str:
    return _win(
        "irm https://claude.ai/install.ps1 | iex  (PowerShell), затем вход: claude auth login",
        "curl -fsSL https://claude.ai/install.sh | bash, затем вход: claude auth login",
    )


def _codex_install() -> str:
    return (
        "npm install -g @openai/codex"
        + _win(" (нужен Node.js: winget install OpenJS.NodeJS.LTS)", " (нужен Node.js)")
        + ", затем вход: codex login"
    )


def _spec_text(spec: str | list[str]) -> str:
    return " ".join(spec) if isinstance(spec, list) else spec


def _is_api_key_method(method: str) -> bool:
    norm = re.sub(r"[^a-z]", "", method.lower())
    return "apikey" in norm


def _check_claude(settings: Settings, check_auth: bool) -> Check:
    cfg = settings.agents.claude
    title = "Claude Code CLI"
    argv = tools.resolve_agent_argv(cfg.bin, "claude")
    data: dict[str, Any] = {"found": argv is not None, "runs": False, "logged_in": None}
    if argv is None:
        return Check(
            "claude",
            title,
            "missing",
            f"не найден ({_spec_text(cfg.bin)})",
            hint=_claude_install(),
            data=data,
        )
    data["argv"] = argv
    env = procutil.clean_env(drop_names=() if cfg.allow_api_key else _CLAUDE_KEY_ENV)
    res = _run([*argv, "--version"], env=env)
    version = _first_line(res)
    if not res.ok or version is None:
        return Check(
            "claude",
            title,
            "error",
            f"{_short(argv[0])} не запускается: {_failure(res)}",
            hint=_claude_install(),
            data=data,
        )
    data.update(runs=True, version=version)
    notes: list[str] = []
    env_keys = [n for n in _CLAUDE_KEY_ENV if os.environ.get(n)]
    if env_keys and not cfg.allow_api_key:
        notes.append(f"{', '.join(env_keys)} из окружения не передаётся (allow_api_key = false)")
    if not check_auth:
        tail = "; ".join(["вход не проверялся", *notes])
        return Check("claude", title, "ok", f"{version}; {tail}", data=data)

    res = _run([*argv, "auth", "status", "--json"], env=env)
    status_obj: dict[str, Any] | None = None
    try:
        parsed = json.loads(res.stdout) if res.stdout.strip() else None
        status_obj = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        status_obj = None
    if status_obj is None:
        return Check(
            "claude",
            title,
            "warn",
            f"{version}; не удалось определить вход (claude auth status: {_failure(res)})",
            hint="проверьте вручную: claude auth status; вход: claude auth login",
            data=data,
        )

    logged_in = bool(status_obj.get("loggedIn"))
    method = str(status_obj.get("authMethod") or "")
    provider = str(status_obj.get("apiProvider") or "")
    subscription = str(status_obj.get("subscriptionType") or "")
    data.update(logged_in=logged_in, auth_method=method, api_provider=provider)
    if subscription:
        data["subscription"] = subscription
    if not logged_in:
        detail = "; ".join([version, "вход не выполнен", *notes])
        return Check("claude", title, "warn", detail, hint="claude auth login", data=data)

    how = f"вход: {method or 'выполнен'}" + (f", подписка {subscription}" if subscription else "")
    if provider and provider != "firstParty":
        how += f", провайдер {provider}"
    detail = "; ".join([version, how, *notes])
    if _is_api_key_method(method) and not cfg.allow_api_key:
        return Check(
            "claude",
            title,
            "warn",
            f"{detail} — вход по API-ключу, а agents.claude.allow_api_key = false",
            hint="claude auth login (вход по подписке Claude) "
            "или allow_api_key = true в [agents.claude]",
            data=data,
        )
    return Check("claude", title, "ok", detail, data=data)


def _check_codex(settings: Settings, check_auth: bool) -> Check:
    cfg = settings.agents.codex
    title = "Codex CLI"
    argv = tools.resolve_agent_argv(cfg.bin, "codex")
    data: dict[str, Any] = {"found": argv is not None, "runs": False, "logged_in": None}
    if argv is None:
        return Check(
            "codex",
            title,
            "missing",
            f"не найден ({_spec_text(cfg.bin)})",
            hint=_codex_install(),
            data=data,
        )
    data["argv"] = argv
    env = procutil.clean_env()
    res = _run([*argv, "--version"], env=env)
    version = _first_line(res)
    if not res.ok or version is None:
        return Check(
            "codex",
            title,
            "error",
            f"{_short(argv[0])} не запускается: {_failure(res)}",
            hint=_codex_install(),
            data=data,
        )
    data.update(runs=True, version=version)
    if not check_auth:
        return Check("codex", title, "ok", f"{version}; вход не проверялся", data=data)

    res = _run([*argv, "login", "status"], env=env)
    text = f"{res.stdout}\n{res.stderr}".strip()
    low = text.lower()
    if res.error or res.timed_out:
        logged_in, method = None, ""
    elif "not logged in" in low:
        logged_in, method = False, ""
    elif "logged in" in low and "chatgpt" in low:
        logged_in, method = True, "chatgpt"
    elif "logged in" in low and "api key" in low:
        logged_in, method = True, "api_key"
    elif "logged in" in low:
        logged_in, method = True, "other"
    else:
        logged_in, method = None, ""
    data.update(logged_in=logged_in, auth_method=method)

    if logged_in is None:
        return Check(
            "codex",
            title,
            "warn",
            f"{version}; не удалось определить вход (codex login status: {_failure(res)})",
            hint="проверьте вручную: codex login status; вход: codex login",
            data=data,
        )
    if not logged_in:
        return Check(
            "codex", title, "warn", f"{version}; вход не выполнен", hint="codex login", data=data
        )
    if method == "api_key":
        # Never echo the status line: it contains a (masked) key.
        return Check(
            "codex",
            title,
            "warn",
            f"{version}; вход по API-ключу, а не по подписке ChatGPT",
            hint="codex login — войти по подписке ChatGPT",
            data=data,
        )
    how = "вход: ChatGPT" if method == "chatgpt" else "вход выполнен"
    return Check("codex", title, "ok", f"{version}; {how}", data=data)


def _agents_summary(settings: Settings, by_name: dict[str, Check], check_auth: bool) -> Check:
    title = "Агенты"
    default = settings.agents.default
    fallback = settings.agents.fallback
    configured = [default] + ([fallback] if fallback and fallback != default else [])

    def ready(name: str) -> bool:
        d = by_name[name].data
        return bool(d.get("runs")) and (not check_auth or d.get("logged_in") is True)

    def state(name: str) -> str:
        d = by_name[name].data
        if "found" not in d:
            return "проверка не завершилась"
        if not d.get("found"):
            return "не найден"
        if not d.get("runs"):
            return "не запускается"
        if not check_auth:
            return "найден"
        return {True: "готов", False: "нет входа"}.get(d.get("logged_in"), "вход не подтверждён")

    roles = {default: "по умолчанию"}
    if fallback and fallback != default:
        roles[fallback] = "запасной"
    detail = "; ".join(f"{roles[n]}: {n} — {state(n)}" for n in configured)
    ready_names = [n for n in configured if ready(n)]
    data = {"configured": configured, "ready": ready_names}
    if ready_names:
        if ready_names[0] != default:
            detail += f"; работать будет {ready_names[0]}"
        return Check("agents", title, "ok", detail, required=True, data=data)

    hints: list[str] = []
    for n in configured:
        d = by_name[n].data
        if not d.get("found") or not d.get("runs"):
            hints.append(_claude_install() if n == "claude" else _codex_install())
        else:
            hints.append("claude auth login" if n == "claude" else "codex login")
    others = [n for n in ("claude", "codex") if n not in configured and ready(n)]
    if others:
        hints.append(f'или укажите agents.default = "{others[0]}" в h0lon.toml')
    unknown = check_auth and any(
        by_name[n].data.get("runs") and by_name[n].data.get("logged_in") is None for n in configured
    )
    status: Status = "warn" if unknown else "error"
    if not any(by_name[n].data.get("found") for n in configured):
        status = "missing"
    return Check(
        "agents",
        title,
        status,
        f"нет готового агента ({detail})",
        hint="; ".join(hints),
        required=True,
        data=data,
    )


# ---------------------------------------------------------------- render toolchain


def _check_pandoc(settings: Settings) -> Check:
    title = "Pandoc"
    hint = _win(
        "uv sync (pandoc входит в пакет pypandoc-binary) или winget install JohnMacFarlane.Pandoc",
        "uv sync (pandoc входит в пакет pypandoc-binary) или https://pandoc.org/installing.html",
    )
    path = tools.find_pandoc(settings.render.pandoc)
    if path is None:
        where = f"render.pandoc = {settings.render.pandoc!r}" if settings.render.pandoc else "PATH"
        return Check("pandoc", title, "missing", f"не найден ({where})", hint=hint, required=True)
    data: dict[str, Any] = {"path": path}
    res = _run([str(path), "--version"])
    line = _first_line(res)
    if not res.ok or line is None:
        return Check(
            "pandoc",
            title,
            "error",
            f"{_short(path)} не запускается: {_failure(res)}",
            hint=hint,
            required=True,
            data=data,
        )
    version = _version_tuple(line)
    data["version"] = line
    origin = " (из pypandoc-binary)" if "pypandoc" in str(path).lower() else ""
    if version is None or version[:2] < MIN_PANDOC:
        return Check(
            "pandoc",
            title,
            "error",
            f"{line}{origin}: нужна версия 3.1 или новее",
            hint=hint,
            required=True,
            data=data,
        )
    return Check(
        "pandoc", title, "ok", f"{line}{origin} — {_short(path)}", required=True, data=data
    )


def _tex_distribution(path: Path, version: str) -> str:
    low = f"{path} {version}".lower()
    if "miktex" in low:
        return "MiKTeX"
    if "tex live" in low or "texlive" in low:
        return "TeX Live"
    return ""


def _xelatex_hint() -> str:
    return _win(
        "winget install MiKTeX.MiKTeX (или TeX Live: https://tug.org/texlive/)",
        "TeX Live: https://tug.org/texlive/ (Debian/Ubuntu: sudo apt install texlive-xetex)",
    )


def _check_xelatex(settings: Settings) -> Check:
    title = "XeLaTeX"
    path = tools.find_xelatex(settings.render.xelatex)
    if path is None:
        where = f"render.xelatex = {settings.render.xelatex!r}" if settings.render.xelatex else ""
        detail = "не найден" + (f" ({where})" if where else " (PATH, MiKTeX, TeX Live)")
        return Check(
            "xelatex", title, "missing", detail, hint=_xelatex_hint(), data={"found": False}
        )
    data: dict[str, Any] = {"found": True, "path": path}
    res = _run([str(path), "--version"])
    line = _first_line(res)
    if not res.ok or line is None:
        return Check(
            "xelatex",
            title,
            "error",
            f"{_short(path)} не запускается: {_failure(res)}",
            hint=_xelatex_hint(),
            data=data,
        )
    data.update(version=line, distribution=_tex_distribution(path, line))
    return Check("xelatex", title, "ok", f"{line} — {_short(path)}", data=data)


def _load_template(settings: Settings) -> Any:
    """The configured render template (TemplateInfo). Lazy import keeps doctor importable."""
    from h0lon.render import get_template

    return get_template(settings.render.template, settings)


def _template_packages(settings: Settings) -> tuple[str, list[str]]:
    """(template name, .sty files it needs)."""
    return settings.render.template, list(_load_template(settings).latex_packages)


def _template_names(settings: Settings) -> list[str]:
    try:
        from h0lon.render import list_templates

        return [t.name for t in list_templates(settings)]
    except Exception:
        return []


def _check_template(settings: Settings) -> Check:
    """Both engines start from the template: without it `h0lon render` cannot run at all."""
    title = "Шаблон"
    name = settings.render.template
    try:
        tpl = _load_template(settings)
    except ImportError as exc:
        return Check(
            "template",
            title,
            "error",
            f"модуль рендера h0lon.render недоступен ({exc})",
            hint="uv sync (переустановить зависимости и пакет h0lon)",
        )
    except KeyError:
        names = _template_names(settings)
        known = f" (доступны: {', '.join(names)})" if names else ""
        return Check(
            "template",
            title,
            "error",
            f"шаблон «{name}» не найден",
            hint=f"укажите существующий шаблон в render.template{known}",
            data={"name": name, "available": names},
        )
    except Exception as exc:
        return Check(
            "template",
            title,
            "error",
            f"не удалось прочитать шаблон «{name}»: {exc}",
            hint=f"исправьте meta.yaml шаблона «{name}» или выберите другой в render.template",
            data={"name": name},
        )
    tpl_title = str(getattr(tpl, "title", "") or "")
    data = {"name": name, "dir": getattr(tpl, "dir", None)}
    detail = f"«{name}»" + (f": {tpl_title}" if tpl_title else "")
    return Check("template", title, "ok", detail, data=data)


# .sty stem -> package that ships it, where the names differ (same in MiKTeX and TeX Live).
_TEX_PACKAGE_OF = {
    **dict.fromkeys(("graphicx", "graphics", "color", "keyval", "lscape", "trig"), "graphics"),
    **dict.fromkeys(
        (
            "array",
            "calc",
            "longtable",
            "tabularx",
            "multicol",
            "bm",
            "verbatim",
            "afterpage",
            "xspace",
            "enumerate",
            "dcolumn",
            "hhline",
            "indentfirst",
            "varioref",
            "showkeys",
        ),
        "tools",
    ),
    **dict.fromkeys(("amssymb", "amsfonts"), "amsfonts"),
    "amsthm": "amscls",
    **dict.fromkeys(("amsbsy", "amstext", "amsopn"), "amsmath"),
    "subcaption": "caption",
}


def _package_hint(missing: list[str], distribution: str) -> str:
    stems = [re.sub(r"\.(sty|cls|def|cfg)$", "", m, flags=re.IGNORECASE) for m in missing]
    names = list(dict.fromkeys(_TEX_PACKAGE_OF.get(s.lower(), s) for s in stems))
    # Files whose package name is only guessed (assumed to match the file name).
    guessed = [m for m, s in zip(missing, stems, strict=True) if s.lower() not in _TEX_PACKAGE_OF]
    search = f"tlmgr search --global --file /{guessed[0]}" if guessed else ""
    if distribution == "MiKTeX":
        cmds = "; ".join(f"miktex packages install {n}" for n in names)
        note = "H0lon запускает XeLaTeX с --disable-installer, поэтому сам пакеты не скачивает"
        if guessed:
            note += "; если пакета с таким именем нет, найдите файл в MiKTeX Console"
        return f"{cmds} ({note})"
    if distribution == "TeX Live":
        tail = f" (если пакета с таким именем нет: {search})" if search else ""
        return f"tlmgr install {' '.join(names)}{tail}"
    tail = f" (имя пакета может отличаться от имени файла: {search})" if search else ""
    return (
        f"MiKTeX: miktex packages install <пакет>; TeX Live: tlmgr install {' '.join(names)}" + tail
    )


def _kpsewhich_found(kpse: Path, files: list[str]) -> set[str] | None:
    """Lowercase basenames that kpsewhich locates, or None if it did not answer in time."""
    res = _run([str(kpse), *files], timeout=DEADLINE_S)  # still capped by the run budget
    if res.timed_out or res.error:
        return None
    return {Path(line.strip()).name.lower() for line in res.stdout.splitlines() if line.strip()}


def _check_tex_packages(settings: Settings) -> Check:
    title = "Пакеты LaTeX"
    try:
        template, packages = _template_packages(settings)
    except Exception:
        return Check(
            "latex-packages",
            title,
            "warn",
            f"не проверены: шаблон «{settings.render.template}» недоступен (см. «Шаблон»)",
            data={"unverified": True},
        )
    data: dict[str, Any] = {"template": template, "packages": packages}
    xelatex = tools.find_xelatex(settings.render.xelatex)
    if xelatex is None:
        return Check("latex-packages", title, "warn", "не проверены: XeLaTeX не найден", data=data)
    if not packages:
        return Check(
            "latex-packages", title, "ok", f"шаблону «{template}» пакеты не нужны", data=data
        )
    kpse = tools.find_tex_tool(xelatex, "kpsewhich")
    found = _kpsewhich_found(kpse, packages) if kpse is not None else None
    if found is None:
        why = "kpsewhich не найден рядом с XeLaTeX" if kpse is None else "kpsewhich не ответил"
        data["unverified"] = True
        return Check("latex-packages", title, "warn", f"не проверены: {why}", data=data)
    missing = [f for f in packages if f.lower() not in found]
    data["missing"] = missing
    if not missing:
        return Check(
            "latex-packages",
            title,
            "ok",
            f"шаблон «{template}»: все {len(packages)} пакетов на месте",
            data=data,
        )
    distribution = _tex_distribution(xelatex, "")
    if not distribution:
        version = _first_line(_run([str(xelatex), "--version"])) or ""
        distribution = _tex_distribution(xelatex, version)
    return Check(
        "latex-packages",
        title,
        "missing",
        f"шаблон «{template}»: нет {len(missing)} из {len(packages)}: {', '.join(missing)}",
        hint=_package_hint(missing, distribution),
        data=data,
    )


# Weight/slope words that may follow a family name in a font entry ("Arial Bold Italic").
_STYLE_WORDS = {
    "regular",
    "normal",
    "book",
    "bold",
    "italic",
    "oblique",
    "light",
    "medium",
    "semibold",
    "semilight",
    "demibold",
    "thin",
    "extralight",
    "extrabold",
    "ultralight",
    "ultrabold",
    "полужирный",
    "жирный",
    "курсив",
    "обычный",
    "наклонный",
    "светлый",
}
_REG_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def _registry_families(names: list[str]) -> list[str]:
    """'Cambria & Cambria Math (TrueType)' -> ['Cambria', 'Cambria Math']."""
    out: list[str] = []
    for name in names:
        base = _REG_SUFFIX.sub("", name)
        out += [part.strip() for part in base.split("&") if part.strip()]
    return out


def _installed_font_families() -> tuple[dict[str, str] | None, str]:
    """Installed font names (lowercase) -> scope ('system' | 'user'), or (None, reason)."""
    if IS_WINDOWS:
        import winreg

        found: dict[str, str] = {}
        hives = ((winreg.HKEY_LOCAL_MACHINE, "system"), (winreg.HKEY_CURRENT_USER, "user"))
        any_hive = False
        for hive, scope in hives:
            try:
                key = winreg.OpenKey(hive, _FONTS_KEY)
            except OSError:
                continue
            any_hive = True
            with key:
                names: list[str] = []
                i = 0
                while True:
                    try:
                        names.append(winreg.EnumValue(key, i)[0])
                    except OSError:
                        break
                    i += 1
            for fam in _registry_families(names):
                found.setdefault(fam.lower(), scope)
        if not any_hive:
            return None, "реестр шрифтов недоступен"
        return found, ""
    fc_list = tools.find_simple("fc-list")
    if fc_list is None:
        return None, "нет fc-list (fontconfig)"
    res = _run([str(fc_list), ":", "family"])
    if not res.ok:
        return None, f"fc-list: {_failure(res)}"
    families: dict[str, str] = {}
    for line in res.stdout.splitlines():
        for fam in line.replace("\\-", "-").split(","):
            if fam.strip():
                families.setdefault(fam.strip().lower(), "system")
    return families, ""


def _font_scope(family: str, installed: dict[str, str]) -> str | None:
    want = family.strip().lower()
    if want in installed:
        return installed[want]
    for name, scope in installed.items():
        if name.startswith(want + " "):
            rest = name[len(want) + 1 :].split()
            if rest and all(w in _STYLE_WORDS for w in rest):
                return scope
    return None


# A font given by file name ("latinmodern-math.otf"): fontspec finds it via kpathsea.
_FONT_FILE = re.compile(r"\.(otf|ttf|ttc|otc|pfb)$", re.IGNORECASE)


def _windows_font_dirs() -> list[Path]:
    dirs = [Path(os.environ.get("WINDIR") or r"C:\Windows") / "Fonts"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    return dirs


def _font_files_found(settings: Settings, files: list[str]) -> tuple[dict[str, bool | None], str]:
    """file -> found (True/False) or None when it cannot be checked; plus the reason."""
    out: dict[str, bool | None] = {}
    by_name: list[str] = []
    for f in files:
        p = Path(f).expanduser()
        if p.is_absolute() or len(p.parts) > 1:
            out[f] = p.is_file()
        else:
            by_name.append(f)
    if not by_name:
        return out, ""
    xelatex = tools.find_xelatex(settings.render.xelatex)
    kpse = tools.find_tex_tool(xelatex, "kpsewhich") if xelatex is not None else None
    found = _kpsewhich_found(kpse, by_name) if kpse is not None else None
    for f in by_name:
        if found is not None:
            out[f] = f.lower() in found
        elif IS_WINDOWS and any((d / f).is_file() for d in _windows_font_dirs()):
            out[f] = True
        else:
            out[f] = None  # never report "missing" without asking kpsewhich
    reason = ""
    if any(v is None for v in out.values()):
        reason = "kpsewhich не ответил" if kpse is not None else "нет kpsewhich (XeLaTeX не найден)"
    return out, reason


def _check_fonts(settings: Settings) -> Check:
    title = "Шрифты"
    r = settings.render
    wanted = {
        "main_font": r.main_font,
        "sans_font": r.sans_font,
        "mono_font": r.mono_font,
        "math_font": r.math_font,
    }
    wanted = {k: v.strip() for k, v in wanted.items() if v.strip()}
    files = {k: v for k, v in wanted.items() if _FONT_FILE.search(v)}
    families = {k: v for k, v in wanted.items() if k not in files}
    # key -> "system" | "user" | "file" (present) or None (absent); no key = not verified
    state: dict[str, str | None] = {}
    reasons: list[str] = []
    if families:
        installed, reason = _installed_font_families()
        if installed is None:
            reasons.append(reason)
        else:
            state.update({k: _font_scope(v, installed) for k, v in families.items()})
    if files:
        found, reason = _font_files_found(settings, list(files.values()))
        if reason:
            reasons.append(reason)
        for k, v in files.items():
            if found.get(v) is not None:
                state[k] = "file" if found[v] else None
    missing = [k for k in wanted if k in state and state[k] is None]
    unverified = [wanted[k] for k in wanted if k not in state]
    present = [wanted[k] for k in wanted if state.get(k)]
    user_only = [wanted[k] for k in wanted if state.get(k) == "user"]
    data: dict[str, Any] = {
        "fonts": wanted,
        "missing": [wanted[k] for k in missing],
        "user_only": user_only,
        "unverified": unverified,
    }
    note = ""
    if user_only:
        note = (
            f"; только для текущего пользователя: {', '.join(user_only)} — если XeLaTeX их "
            "не увидит, установите «для всех пользователей»"
        )
    unchecked = (
        f"не проверены: {', '.join(unverified)} ({'; '.join(reasons)})" if unverified else ""
    )
    if not missing and not unverified:
        return Check("fonts", title, "ok", ", ".join(present) + note, data=data)
    if not missing:
        # Could not look: a warning, never a failure of a required check.
        detail = unchecked + (f"; есть: {', '.join(present)}" if present else "") + note
        return Check("fonts", title, "warn", detail, data=data)
    lost = ", ".join(f"{wanted[k]} (render.{k})" for k in missing)
    detail = (
        f"нет: {lost}"
        + (f"; есть: {', '.join(present)}" if present else "")
        + (f"; {unchecked}" if unchecked else "")
        + note
    )
    hint = (
        "установите шрифты или укажите установленные в [render]: "
        "main_font, sans_font, mono_font, math_font"
    )
    if any(k in files for k in missing):
        hint += " (файл шрифта, например latinmodern-math.otf, ищется через kpsewhich)"
    return Check("fonts", title, "missing", detail, hint=hint, data=data)


def _check_browser(settings: Settings) -> Check:
    title = "Браузер (HTML → PDF)"
    path = tools.find_browser(settings.render.browser)
    if path is None:
        return Check(
            "browser",
            title,
            "warn",
            f"Edge/Chrome/Chromium не найден (render.browser = {settings.render.browser!r})",
            hint="установите Microsoft Edge или Google Chrome либо укажите путь в render.browser",
            data={"found": False},
        )
    detail = _short(path)
    if not settings.render.fallback_html and settings.render.engine != "html":
        detail += " (запасной рендер выключен: render.fallback_html = false)"
    return Check("browser", title, "ok", detail, data={"found": True, "path": path})


# ---------------------------------------------------------------- media, git, GPU


def _check_ffmpeg(settings: Settings) -> Check:
    title = "ffmpeg / ffprobe"
    hint = _win("winget install Gyan.FFmpeg", "sudo apt install ffmpeg / brew install ffmpeg")
    ffmpeg = tools.find_simple("ffmpeg")
    ffprobe = tools.find_simple("ffprobe")
    data = {"ffmpeg": ffmpeg, "ffprobe": ffprobe}
    if ffmpeg is None:
        return Check(
            "ffmpeg",
            title,
            "warn",
            "не найден — понадобится для видео и аудио (этап M5)",
            hint=hint,
            data=data,
        )
    res = _run([str(ffmpeg), "-version"])
    if not res.ok:
        return Check(
            "ffmpeg",
            title,
            "warn",
            f"{_short(ffmpeg)} не запускается: {_failure(res)} — понадобится с этапа M5",
            hint=hint,
            data=data,
        )
    line = _first_line(res) or ""
    m = re.search(r"version\s+(\S+)", line)
    version = m.group(1) if m else (line or "версия не определена")
    data["version"] = version
    if ffprobe is None:
        return Check(
            "ffmpeg", title, "warn", f"ffmpeg {version}; ffprobe не найден", hint=hint, data=data
        )
    return Check(
        "ffmpeg", title, "ok", f"ffmpeg {version}; ffprobe есть — {_short(ffmpeg)}", data=data
    )


def _ytdlp_hint() -> str:
    return _win(
        "winget install yt-dlp.yt-dlp (или uv tool install yt-dlp)",
        "uv tool install yt-dlp (или brew install yt-dlp)",
    )


def _check_ytdlp(settings: Settings) -> Check:
    title = "yt-dlp"
    path = tools.find_simple("yt-dlp")
    if path is None:
        return Check(
            "yt-dlp",
            title,
            "warn",
            "не найден — понадобится для видео по ссылкам (этап M5)",
            hint=_ytdlp_hint(),
            data={"found": False},
        )
    res = _run([str(path), "--version"])
    version = _first_line(res)
    if not res.ok or version is None:
        return Check(
            "yt-dlp",
            title,
            "warn",
            f"{_short(path)} не запускается: {_failure(res)} — понадобится с этапа M5",
            hint=_ytdlp_hint(),
            data={"found": True, "path": path},
        )
    return Check(
        "yt-dlp",
        title,
        "ok",
        f"{version} — {_short(path)}",
        data={"found": True, "path": path, "version": version},
    )


def _git_identity_cwd(settings: Settings) -> Path:
    ws = settings.general.workspaces_dir
    return ws if ws.is_dir() else Path(tempfile.gettempdir())


def _check_git(settings: Settings) -> list[Check]:
    title = "git"
    required = settings.general.git_per_topic
    git = tools.find_simple("git")
    if git is None:
        status: Status = "missing" if required else "info"
        why = "нужен: general.git_per_topic = true" if required else "темы без истории версий"
        return [
            Check(
                "git",
                title,
                status,
                f"не найден ({why})",
                hint=_win("winget install Git.Git", "установите git пакетным менеджером ОС"),
                required=required,
            )
        ]
    res = _run([str(git), "--version"])
    version = _first_line(res)
    if not res.ok or version is None:
        return [
            Check(
                "git",
                title,
                "error",
                f"{_short(git)} не запускается: {_failure(res)}",
                required=required,
                data={"path": git},
            )
        ]
    checks = [
        Check(
            "git",
            title,
            "ok",
            f"{version} — {_short(git)}",
            required=required,
            data={"path": git, "version": version},
        )
    ]
    if not required:
        return checks
    cwd = _git_identity_cwd(settings)
    ident_title = "git: имя и почта"
    missing: list[str] = []
    for key in ("user.name", "user.email"):
        res = _run([str(git), "config", "--get", key], cwd=cwd)
        if res.timed_out or res.error:
            checks.append(
                Check(
                    "git-identity",
                    ident_title,
                    "warn",
                    f"не удалось проверить (git config: {_failure(res)})",
                    hint=f"проверьте вручную: git config --get {key}",
                )
            )
            return checks
        if not res.stdout.strip():
            missing.append(key)
    if missing:
        checks.append(
            Check(
                "git-identity",
                ident_title,
                "warn",
                f"не заданы {', '.join(missing)} — коммиты в темах не будут создаваться",
                hint='git config --global user.name "Ваше имя"; '
                'git config --global user.email "you@example.com"',
            )
        )
    else:
        checks.append(Check("git-identity", ident_title, "ok", "user.name и user.email заданы"))
    return checks


def _check_gpu(settings: Settings) -> Check:
    title = "GPU / CUDA"
    smi = tools.find_simple("nvidia-smi")
    if smi is None:
        return Check(
            "gpu",
            title,
            "info",
            "NVIDIA GPU не найден — транскрипция на CPU или в Colab",
            data={"gpus": []},
        )
    res = _run(
        [str(smi), "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]
    )
    gpus: list[dict[str, Any]] = []
    if res.ok:
        for line in res.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0]:
                try:
                    vram_mib: int | None = int(float(parts[2]))
                except ValueError:
                    vram_mib = None
                gpus.append({"name": parts[0], "driver": parts[1], "vram_mib": vram_mib})
    if not gpus:
        return Check(
            "gpu",
            title,
            "info",
            f"nvidia-smi не ответил ({_failure(res)}) — транскрипция на CPU или в Colab",
            data={"gpus": []},
        )
    cuda = None
    m = re.search(r"CUDA Version:\s*([\d.]+)", _run([str(smi)]).stdout)
    if m:
        cuda = m.group(1)
    parts_out = []
    for g in gpus:
        vram = f", {g['vram_mib'] / 1024:.1f} ГБ VRAM".replace(".", ",") if g["vram_mib"] else ""
        parts_out.append(f"{g['name']}{vram}, драйвер {g['driver']}")
    detail = "; ".join(parts_out) + (f", CUDA {cuda}" if cuda else "")
    return Check("gpu", title, "info", detail, data={"gpus": gpus, "cuda": cuda})


# ---------------------------------------------------------------- orchestration

_JobFn = Callable[[], "Check | list[Check]"]

ORDER = [
    "python",
    "config",
    "workspaces",
    "claude",
    "codex",
    "agents",
    "pandoc",
    "template",
    "xelatex",
    "latex-packages",
    "fonts",
    "browser",
    "git",
    "ffmpeg",
    "yt-dlp",
    "gpu",
]


def _jobs(settings: Settings, check_auth: bool) -> list[tuple[str, str, _JobFn]]:
    s = settings
    return [
        ("python", "Python", lambda: _check_python(s)),
        ("config", "Конфигурация", lambda: _check_config(s)),
        ("workspaces", "Рабочие области", lambda: _check_workspaces(s)),
        ("claude", "Claude Code CLI", lambda: _check_claude(s, check_auth)),
        ("codex", "Codex CLI", lambda: _check_codex(s, check_auth)),
        ("pandoc", "Pandoc", lambda: _check_pandoc(s)),
        ("template", "Шаблон", lambda: _check_template(s)),
        ("xelatex", "XeLaTeX", lambda: _check_xelatex(s)),
        ("latex-packages", "Пакеты LaTeX", lambda: _check_tex_packages(s)),
        ("fonts", "Шрифты", lambda: _check_fonts(s)),
        ("browser", "Браузер (HTML → PDF)", lambda: _check_browser(s)),
        ("git", "git", lambda: _check_git(s)),
        ("ffmpeg", "ffmpeg / ffprobe", lambda: _check_ffmpeg(s)),
        ("yt-dlp", "yt-dlp", lambda: _check_ytdlp(s)),
        ("gpu", "GPU / CUDA", lambda: _check_gpu(s)),
    ]


def _guarded(check_id: str, title: str, fn: _JobFn) -> Callable[[], list[Check]]:
    def run() -> list[Check]:
        start = time.monotonic()
        try:
            out = fn()
            checks = out if isinstance(out, list) else [out]
        except Exception as exc:  # a broken check must not break the whole report
            checks = [Check(check_id, title, "error", f"внутренняя ошибка проверки: {exc!r}")]
        for c in checks:
            c.data.setdefault("duration_s", round(time.monotonic() - start, 2))
        return checks

    return run


def _soften(check: Check, note: str) -> None:
    """A missing optional piece is a warning, not a failure."""
    if check.status == "missing":
        check.status = "warn"
    check.detail += f" — {note}"


def _apply_requirements(settings: Settings, results: dict[str, list[Check]]) -> None:
    """Decide `required` in one place: real results and timeout/crash stubs alike."""
    for cid in ("python", "pandoc", "template", "agents"):
        results[cid][0].required = True
    results["git"][0].required = settings.general.git_per_topic  # not git-identity

    r = settings.render
    browser = results["browser"][0]
    html_fallback = browser.ok and (r.fallback_html or r.engine == "html")
    xelatex_required = r.engine == "xelatex" and not html_fallback
    xelatex = results["xelatex"][0]
    xelatex.required = xelatex_required
    xelatex_found = bool(xelatex.data.get("found"))
    for cid in ("latex-packages", "fonts"):
        c = results[cid][0]
        # "Could not look" (no kpsewhich, no font database) stays a warning.
        unverified = c.status == "warn" and bool(c.data.get("unverified"))
        c.required = xelatex_required and xelatex_found and not unverified
    browser.required = r.engine == "html"
    if r.engine == "html":
        if not xelatex.ok:
            _soften(xelatex, "не нужен: render.engine = html")
    elif not xelatex_required:
        if not xelatex.ok:
            _soften(xelatex, "PDF будет собираться запасным путём через браузер")
        for cid in ("latex-packages", "fonts"):
            c = results[cid][0]
            if c.status == "missing":
                _soften(c, "при сбое XeLaTeX сработает запасной рендер через браузер")


def run_checks(settings: Settings, *, check_auth: bool = True) -> list[Check]:
    """Run all checks in parallel; the whole run takes at most DEADLINE_S.

    Each external command gets min(PROC_TIMEOUT_S, time left) and is killed when the budget
    runs out. Workers are daemon threads, so a stuck check cannot keep the process alive.
    """
    jobs = _jobs(settings, check_auth)
    hard_deadline = time.monotonic() + DEADLINE_S
    # Commands stop a little earlier, so their check can still say «не ответил».
    proc_deadline = hard_deadline - min(KILL_MARGIN_S, DEADLINE_S / 2)
    results: dict[str, list[Check]] = {}
    lock = threading.Lock()

    def worker(cid: str, title: str, fn: _JobFn) -> None:
        _budget.deadline = proc_deadline
        checks = _guarded(cid, title, fn)()
        with lock:
            results[cid] = checks

    threads = [
        threading.Thread(
            target=worker, args=(cid, title, fn), name=f"h0lon-doctor-{cid}", daemon=True
        )
        for cid, title, fn in jobs
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(max(0.0, hard_deadline - time.monotonic()))
    with lock:
        finished = dict(results)  # late results of stuck threads are ignored

    for cid, title, _fn in jobs:
        if cid not in finished:
            finished[cid] = [
                Check(
                    cid,
                    title,
                    "error",
                    f"проверка не завершилась за {DEADLINE_S:.0f} с",
                    hint="повторите h0lon doctor; если повторяется — запустите инструмент вручную",
                )
            ]
    by_name = {"claude": finished["claude"][0], "codex": finished["codex"][0]}
    finished["agents"] = [_agents_summary(settings, by_name, check_auth)]
    _apply_requirements(settings, finished)
    return [c for cid in ORDER for c in finished[cid]]


def exit_code(checks: list[Check]) -> int:
    """1 if any required check is not ok, else 0."""
    return 1 if any(c.required and c.status != "ok" for c in checks) else 0


# ---------------------------------------------------------------- report

_STATUS_VIEW: dict[str, tuple[str, str]] = {
    "ok": ("готово", "green"),
    "warn": ("внимание", "yellow"),
    "missing": ("нет", "red"),
    "error": ("ошибка", "red"),
    "info": ("инфо", "cyan"),
}


def print_report(checks: list[Check], *, console: Console, settings: Settings) -> None:
    from rich import box
    from rich.table import Table
    from rich.text import Text

    source = _short(settings.source_path) or "значения по умолчанию"
    console.print(Text.assemble(("H0lon doctor", "bold"), f" — конфигурация: {source}"))
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, expand=False)
    table.add_column("Статус", no_wrap=True)
    table.add_column("Проверка", no_wrap=True)
    table.add_column("Подробности", overflow="fold")
    for c in checks:
        label, style = _STATUS_VIEW.get(c.status, (c.status, ""))
        name = Text(c.title + (" *" if c.required else ""), style="bold" if c.required else "")
        table.add_row(Text(label, style=f"bold {style}"), name, Text(c.detail))
    console.print(table)
    console.print(Text("* — обязательная проверка", style="dim"))

    todo = [c for c in checks if c.status != "ok" and c.hint]
    if todo:
        console.print()
        console.print(Text("Что сделать:", style="bold"))
        for c in todo:
            _label, style = _STATUS_VIEW.get(c.status, (c.status, ""))
            console.print(Text.assemble("  - ", (c.title, f"bold {style}"), ": ", c.hint or ""))

    console.print()
    failed = [c for c in checks if c.required and c.status != "ok"]
    warnings = [c for c in checks if not c.required and c.status in ("warn", "missing", "error")]
    if failed:
        names = ", ".join(c.title for c in failed)
        console.print(
            Text(f"Не пройдено обязательных проверок: {len(failed)} ({names}).", style="bold red")
        )
    else:
        tail = f" Предупреждений: {len(warnings)}." if warnings else ""
        console.print(
            Text(f"Все обязательные проверки пройдены — можно работать.{tail}", style="bold green")
        )
