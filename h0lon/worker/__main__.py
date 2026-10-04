"""`python -m h0lon.worker`: run the recognition worker (it listens on this PC only by default).

The token comes from the environment (`H0LON_WORKER_TOKEN`), never from the command line: a
command line is visible to every process and stays in the shell history.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

from h0lon.extract import asr
from h0lon.worker.server import MAX_UPLOAD_ENV, TOKEN_ENV, create_app


def _fix_stdio() -> None:
    """Never crash on characters outside the console code page; a pipe or a file gets UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def _ensure_port_free(host: str, port: int) -> None:
    """Raise OSError with a readable message when `host:port` cannot be listened on."""
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
            f"Не удалось занять {host}:{port} — {exc.strerror or exc}. Укажите другой порт: "
            f"--port {port + 1}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m h0lon.worker",
        description="H0lon worker: распознавание речи (faster-whisper) по HTTP. "
        f"Токен — в переменной окружения {TOKEN_ENV}.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="на каком адресе слушать")
    parser.add_argument("--port", type=int, default=8790, help="порт (по умолчанию 8790)")
    parser.add_argument("--model", default=asr.DEFAULT_MODEL, help="модель по умолчанию")
    parser.add_argument("--work-dir", type=Path, default=None, help="каталог для присланного аудио")
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=None,
        help=f"предел размера аудио, МБ (иначе {MAX_UPLOAD_ENV} или 512)",
    )
    parser.add_argument(
        "--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"]
    )
    args = parser.parse_args(argv)
    _fix_stdio()
    if not 1 <= args.port <= 65535:
        print("--port: допустимо 1–65535", file=sys.stderr)
        return 2

    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        print(
            f"Предупреждение: переменная {TOKEN_ENV} не задана — worker будет отвечать 401 на всё, "
            "кроме /health. Задайте токен и запустите заново.",
            file=sys.stderr,
        )
    elif len(token) < 16:
        print(
            f"Предупреждение: токен в {TOKEN_ENV} короче 16 знаков: его можно подобрать.",
            file=sys.stderr,
        )
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            "Предупреждение: worker слушает не только этот ПК; доступ защищён лишь токеном, "
            "трафик по http не шифруется.",
            file=sys.stderr,
        )
    try:
        _ensure_port_free(args.host, args.port)
    except OSError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2

    import uvicorn

    app = create_app(
        token=token,
        work_dir=args.work_dir,
        default_model=args.model,
        max_upload_mb=args.max_upload_mb,
    )
    shown = f"[{args.host}]" if ":" in args.host else args.host
    print(
        f"H0lon worker: http://{shown}:{args.port}/ (модель {args.model}, токен "
        f"{'задан' if token else 'НЕ задан'}) — остановить: Ctrl+C",
        flush=True,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_graceful_shutdown=3,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
