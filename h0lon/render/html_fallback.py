"""Fallback renderer: Pandoc → standalone HTML (MathML) → headless Chromium print-to-pdf."""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from h0lon import procutil
from h0lon.render.pandoc import PandocResult, build_argv, run_pandoc
from h0lon.render.templates import TemplateInfo

BROWSER_TIMEOUT_S = 240


@dataclass
class BrowserResult:
    ok: bool
    pdf: Path | None
    argv: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def html_argv(
    pandoc: Path,
    *,
    src: Path | None,
    html_out: Path,
    template: TemplateInfo,
    css_href: str | None,
    metadata: dict[str, object],
    variables: dict[str, str],
    toc: bool,
    toc_depth: int | None,
    resource_dirs: list[Path],
) -> list[str]:
    return build_argv(
        pandoc,
        src=src,
        to="html5",
        output=html_out,
        template=template.template_html,
        standalone=True,
        metadata=metadata,
        variables=variables,
        toc=toc,
        toc_depth=toc_depth,
        number_sections=False,  # blocks.lua numbers headings itself (with appendix letters)
        resource_path=resource_dirs,
        css=[css_href] if css_href else None,
        extra_args=["--mathml", "--embed-resources"],
    )


def build_html(
    pandoc: Path,
    *,
    src: Path,
    build_dir: Path,
    jobname: str,
    input_text: str | None = None,
    template: TemplateInfo,
    metadata: dict[str, object],
    variables: dict[str, str],
    toc: bool,
    toc_depth: int | None,
) -> PandocResult:
    """Standalone self-contained HTML in build_dir/<jobname>.html.

    `input_text` (the master after escape_anchor_syntax) is fed on stdin instead of `src`;
    `src` still locates relative resources.
    """
    html_out = build_dir / f"{jobname}.html"
    css_href: str | None = None
    if template.html_css is not None:
        # A relative href resolved from cwd=build_dir: an absolute Windows path would be
        # read by Pandoc as a URL with the scheme "c:".
        local = build_dir / "style.css"
        shutil.copyfile(template.html_css, local)
        css_href = local.name
    argv = html_argv(
        pandoc,
        src=None if input_text is not None else src,
        html_out=html_out,
        template=template,
        css_href=css_href,
        metadata=metadata,
        variables=variables,
        toc=toc,
        toc_depth=toc_depth,
        resource_dirs=[src.parent, build_dir],
    )
    return run_pandoc(argv, cwd=build_dir, input_text=input_text, output=html_out)


def browser_argv(browser: Path, *, html: Path, pdf: Path, profile_dir: Path) -> list[str]:
    return [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-sync",
        # Scripts are blocked by the CSP meta tag of template.html (--blink-settings with
        # scriptEnabled=false makes headless print-to-pdf hang on Edge).
        "--no-pdf-header-footer",
        "--generate-pdf-document-outline",
        f"--user-data-dir={profile_dir}",
        f"--print-to-pdf={pdf}",
        html.resolve().as_uri(),
    ]


def print_to_pdf(
    browser: Path, *, html: Path, pdf: Path, timeout: float = BROWSER_TIMEOUT_S
) -> BrowserResult:
    """Print an HTML file to PDF with a headless Chromium-family browser (fresh profile)."""
    start = time.monotonic()
    if pdf.exists():
        pdf.unlink()
    profile = Path(tempfile.mkdtemp(prefix="h0lon-browser-"))
    argv = browser_argv(browser, html=html, pdf=pdf, profile_dir=profile)
    result = BrowserResult(ok=False, pdf=None, argv=argv)
    try:
        res = procutil.run(argv, timeout=timeout, env=procutil.clean_env(drop_session_vars=False))
        complete = False
        if res.error is None and not res.timed_out and res.exit_code == 0:
            # On Windows msedge.exe hands the job to another process and exits at once;
            # the PDF appears seconds later.
            complete = _wait_for_pdf(pdf, deadline=start + timeout)
    finally:
        _remove_profile(profile)
    result.duration_s = time.monotonic() - start
    if res.error:
        result.errors.append(f"Не удалось запустить браузер {browser}: {res.error}")
        return result
    if res.timed_out:
        result.errors.append(f"Браузер не напечатал PDF за {int(timeout)} с")
        return result
    if not complete:
        tail = "\n".join((res.stderr or res.stdout).strip().splitlines()[-5:])
        detail = f": {tail}" if tail else ""
        result.errors.append(f"Браузер завершился (код {res.exit_code}), но PDF не создан{detail}")
        return result
    result.pdf = pdf
    result.ok = True
    return result


def _pdf_complete(pdf: Path) -> bool:
    try:
        size = pdf.stat().st_size
        if size < 64:
            return False
        with pdf.open("rb") as fh:
            fh.seek(max(0, size - 1024))
            return b"%%EOF" in fh.read()
    except OSError:
        return False


def _wait_for_pdf(pdf: Path, *, deadline: float, poll_s: float = 0.3) -> bool:
    """Wait until the PDF exists, ends with %%EOF and its size stops changing."""
    last_size = -1
    while True:
        if pdf.is_file() and _pdf_complete(pdf):
            size = pdf.stat().st_size
            if size == last_size:
                return True
            last_size = size
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)


def _remove_profile(profile: Path, attempts: int = 10) -> None:
    """The detached browser process may hold the profile for a moment after printing."""
    for _ in range(attempts):
        shutil.rmtree(profile, ignore_errors=True)
        if not profile.exists():
            return
        time.sleep(0.5)
