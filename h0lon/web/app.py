"""FastAPI application of the local web interface: `create_app(settings)`, `serve(...)`.

The interface has no login: it is meant for `127.0.0.1`. Because it starts agents and writes to
the workspaces, the application defends itself against pages of other sites open in the same
browser (`SecurityMiddleware`): a request whose `Origin` is not this server is refused, and when
the server listens on a loopback address, requests with a foreign `Host` (DNS rebinding) too.
Pages are served with a strict Content-Security-Policy: no inline scripts, no external resources.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
import webbrowser
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.templating import Jinja2Templates
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from h0lon.web import present, views
from h0lon.web.jobs import JobManager

if TYPE_CHECKING:
    from h0lon.config import Settings

log = logging.getLogger("h0lon.web")

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'self'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    # not «no-referrer»: with it browsers send `Origin: null` on same-origin form posts
    "Referrer-Policy": "same-origin",
}
# Files of topics set their own headers (a PDF viewer does not work under this CSP).
_NO_CSP_PREFIXES = ("/files/",)


class SecurityMiddleware:
    """Pure ASGI middleware: Host and Origin checks, security headers."""

    def __init__(self, app: ASGIApp, allowed_hosts: Collection[str] | None = None) -> None:
        self.app = app
        self.allowed_hosts = {h.lower() for h in allowed_hosts} if allowed_hosts else None

    def _problem(self, scope: Scope) -> str | None:
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        host = headers.get("host", "")
        if self.allowed_hosts is not None:
            try:
                hostname = (urlsplit("//" + host).hostname or "").lower()
            except ValueError:
                hostname = ""
            if hostname not in self.allowed_hosts:
                return "Недопустимый заголовок Host: интерфейс отвечает только на локальные адреса."
        if scope["method"] in UNSAFE_METHODS:
            origin = headers.get("origin")
            if origin is not None and (
                origin == "null" or urlsplit(origin).netloc.lower() != host.lower()
            ):
                return "Запрос с другого сайта отклонён."
            if headers.get("sec-fetch-site") == "cross-site":
                return "Запрос с другого сайта отклонён."
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        problem = self._problem(scope)
        if problem is not None:
            headers = Headers(scope=scope)
            log.warning(
                "Запрос отклонён: %s %s (Host=%s, Origin=%s, Sec-Fetch-Site=%s)",
                scope["method"],
                scope["path"],
                headers.get("host"),
                headers.get("origin"),
                headers.get("sec-fetch-site"),
            )
            await PlainTextResponse(problem, status_code=403)(scope, receive, send)
            return
        with_csp = not scope["path"].startswith(_NO_CSP_PREFIXES)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present_names = {k.lower() for k, _ in headers}
                for name, value in SECURITY_HEADERS.items():
                    if name == "Content-Security-Policy" and not with_csp:
                        continue
                    if name.lower().encode("latin-1") not in present_names:
                        headers.append((name.lower().encode("latin-1"), value.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _static_url(name: str) -> str:
    """URL of a static file with its mtime as the cache-busting version."""
    try:
        version = (STATIC_DIR / name).stat().st_mtime_ns
    except OSError:
        version = 0
    return f"/static/{name}?v={version}"


def create_app(settings: Settings, *, allowed_hosts: Collection[str] | None = None) -> FastAPI:
    """Application for `settings`. `allowed_hosts`: accepted `Host` names (None = any; `serve`
    passes the loopback names when it listens on a loopback address)."""
    app = FastAPI(title="H0lon", docs_url=None, redoc_url=None, openapi_url=None)
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    templates.env.filters.update(
        size=present.format_size,
        duration=present.format_duration,
        iso=present.format_iso,
    )
    templates.env.globals["static"] = _static_url
    app.state.settings = settings
    app.state.jobs = JobManager(settings)
    app.state.templates = templates
    app.state.flash = views.FlashStore()

    app.add_middleware(SecurityMiddleware, allowed_hosts=allowed_hosts)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(views.router)

    @app.exception_handler(views.WebError)
    async def _web_error(request: Request, exc: views.WebError) -> Response:
        return views.error_response(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        titles = {404: "Страница не найдена", 405: "Действие не поддерживается"}
        return views.error_response(
            request,
            views.WebError(exc.status_code, titles.get(exc.status_code, "Ошибка запроса"), ""),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
        return views.error_response(
            request,
            views.WebError(
                422,
                "Некорректный запрос",
                "Форма заполнена неверно или запрос повреждён. Вернитесь и повторите.",
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> Response:
        log.exception("Необработанная ошибка в %s %s", request.method, request.url.path)
        err = views.WebError(
            500,
            "Внутренняя ошибка",
            f"{type(exc).__name__}: {exc}",
            hint="Подробности — в журнале сервера (окно, где запущен h0lon serve).",
        )
        try:
            return views.error_response(request, err)
        except Exception:  # the templates themselves are broken
            return JSONResponse({"error": err.title, "message": err.message}, status_code=500)

    return app


# ---------------------------------------------------------------- serve


def _allowed_hosts(host: str) -> set[str] | None:
    if host.lower() in LOOPBACK_HOSTS:
        return {*LOOPBACK_HOSTS, host.lower()}
    return None


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host in ("", "0.0.0.0") else "::1" if host == "::" else host


def _url_for(host: str, port: int) -> str:
    shown = _connect_host(host)
    if ":" in shown:
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/"


def _ensure_port_free(host: str, port: int) -> None:
    """Raise OSError with a readable message if `host:port` cannot be listened on."""
    try:
        family = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][0]
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: no silent port sharing
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
    except OSError as exc:
        raise OSError(
            f"Не удалось занять {host}:{port} — {exc.strerror or exc}. "
            f"Возможно, h0lon serve уже запущен; укажите другой порт: h0lon serve --port {port + 1}"
        ) from exc


def _open_when_ready(url: str, host: str, port: int, *, timeout_s: float = 30.0) -> bool:
    """Open `url` in the browser as soon as the server accepts connections."""
    deadline = time.monotonic() + timeout_s
    target = (_connect_host(host), port)
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(target, timeout=0.5):
                pass
        except OSError:
            time.sleep(0.25)
            continue
        try:
            webbrowser.open(url)
        except Exception:  # no browser available: the URL is already printed
            log.debug("webbrowser.open failed", exc_info=True)
        return True
    return False


def serve(
    settings: Settings, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False
) -> None:
    """Run the interface in this process until Ctrl+C. Raises OSError if the port is busy."""
    import uvicorn

    _ensure_port_free(host, port)
    app = create_app(settings, allowed_hosts=_allowed_hosts(host))
    if open_browser:
        threading.Thread(
            target=_open_when_ready,
            args=(_url_for(host, port), host, port),
            name="h0lon-open-browser",
            daemon=True,
        ).start()
    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "log_level": "info",
        # Open event streams must not hold the server after Ctrl+C.
        "timeout_graceful_shutdown": 3,
    }
    try:
        uvicorn.run(app, **kwargs)
    finally:
        running = [job for job in app.state.jobs.recent() if job.active]
        if running:
            log.warning(
                "Сервер остановлен, пока выполнялись задачи (%s). Агенты, которых они запустили, "
                "могли продолжить работу в фоне: проверьте процессы claude и codex.",
                ", ".join(f"{job.title}: {job.topic}" for job in running),
            )
