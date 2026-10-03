"""XeLaTeX log analysis and the multi-pass compile loop (fake xelatex, no TeX needed)."""

from __future__ import annotations

import stat
import sys
import textwrap
from pathlib import Path

import pytest

from h0lon.render.latex import compile_latex, parse_log, xelatex_argv

# ---------------------------------------------------------------- parse_log

MISSING_STY = """\
(./doc.tex
! LaTeX Error: File `h0lonnosuchpkg.sty' not found.

Type X to quit or <RETURN> to proceed,
or enter new name. (Default extension: sty)

Enter file name:
doc.tex:286: Emergency stop.
<read *>

l.286 \\usepackage
"""

UNDEFINED_CS = """\
doc.tex:324: Undefined control sequence.
l.324 Текст \\undefinedmacro
                           {x}
The control sequence at the end of the top line
"""

ABSOLUTE_PATH_ERROR = """\
C:/Users/x/AppData/Local/Programs/MiKTeX/tex/latex/foo/foo.sty:12: Package foo Error: broken.
"""

MISSING_CHARS = (
    "Missing character: There is no ж in font ectt1000!\n"
    "Missing character: There is no ж in font ectt1000!\n"
    "Missing character: There is no 𝔊 (U+1D50A) in font "
    "[cambria.ttc]:mode=base;script=math;language=dflt;!\n"
    'Missing character: There is no ☃ in font "Times New Roman/OT:script=cyrl;language=dflt;"!\n'
)

REFS = """\
LaTeX Warning: Reference `sec:cdf' on page 3 undefined on input line 120.
LaTeX Warning: Hyper reference `def:rv' on page 5 undefined on input line 617.
LaTeX Warning: Citation `knuth' on page 1 undefined on input line 4.
LaTeX Warning: Reference `sec:cdf' on page 4 undefined on input line 130.
LaTeX Warning: There were undefined references.
LaTeX Warning: Label `fig:a' multiply defined.
"""

OVERFULL = """\
Overfull \\hbox (12.53pt too wide) in paragraph at lines 10--12
Overfull \\hbox (0.4pt too wide) in paragraph at lines 20--21
Overfull \\hbox (3.0pt too wide) detected at line 40
"""

PACKAGES = """\
Package fontspec Warning: Font "Foo" does not contain requested
(fontspec)                Script "Cyrillic".

Package unicode-math Warning: I'm going to overwrite the following commands
(unicode-math)                from the `amsmath' package:
Package hyperref Warning: Token not allowed in a PDF string (Unicode):
(hyperref)                removing `math shift' on input line 77.
Package hyperref Warning: Rerun to get /PageLabels entry.
LaTeX Font Warning: Font shape `TU/TimesNewRoman(0)/m/sc' undefined
(Font)              using `TU/TimesNewRoman(0)/m/n' instead on input line 5.
"""

FONT_NOT_FOUND = """\
! Package fontspec Error: The font "Times New Romam" cannot be found.
"""


def test_missing_package_is_reported_once_with_hint() -> None:
    log = parse_log(MISSING_STY)
    assert log.missing_files == ["h0lonnosuchpkg.sty"]
    assert len(log.errors) == 1
    assert "h0lonnosuchpkg.sty" in log.errors[0]
    assert "MiKTeX" in log.errors[0]
    assert not any("Emergency stop" in e for e in log.errors)


def test_file_line_error_with_context() -> None:
    log = parse_log(UNDEFINED_CS)
    assert log.errors == ["doc.tex:324: Undefined control sequence. (l.324 Текст \\undefinedmacro)"]


def test_file_line_error_with_absolute_windows_path() -> None:
    log = parse_log(ABSOLUTE_PATH_ERROR)
    assert log.errors == ["foo.sty:12: Package foo Error: broken."]


def test_missing_characters_are_aggregated() -> None:
    log = parse_log(MISSING_CHARS)
    assert log.missing_chars == [
        "«ж» (ectt1000) ×2",
        "«𝔊 (U+1D50A)» (cambria.ttc)",
        "«☃» (Times New Roman)",
    ]
    summary = log.summary_warnings()
    assert summary[0].startswith("Шрифт не содержит символы")
    assert "ectt1000" in summary[0]


def test_undefined_and_duplicate_references() -> None:
    log = parse_log(REFS)
    assert log.undefined_refs == ["sec:cdf", "def:rv", "knuth"]
    assert "Метка определена несколько раз: fig:a" in log.warnings
    assert log.rerun_needed
    assert any(w == "Неразрешённые ссылки: sec:cdf, def:rv, knuth" for w in log.summary_warnings())


def test_overfull_boxes_above_threshold() -> None:
    log = parse_log(OVERFULL)
    assert log.overfull == ["12.5pt, строка 10", "3.0pt, строка 40"]
    assert any("Overfull" in w for w in log.summary_warnings())


def test_package_warnings_joined_and_noise_ignored() -> None:
    log = parse_log(PACKAGES)
    assert 'fontspec: Font "Foo" does not contain requested Script "Cyrillic".' in log.warnings
    assert not any(w.startswith("unicode-math") for w in log.warnings)
    assert not any("PageLabels" in w for w in log.warnings)
    assert any(w.startswith("hyperref:") and "закладки" in w for w in log.warnings)
    assert any("TU/TimesNewRoman(0)/m/sc" in w for w in log.warnings)


def test_font_not_found_has_config_hint() -> None:
    log = parse_log(FONT_NOT_FOUND)
    assert len(log.errors) == 1
    assert "Times New Romam" in log.errors[0]
    assert "main_font" in log.errors[0]


def test_clean_log_has_no_findings() -> None:
    log = parse_log("This is XeTeX, Version 3.141592653\n(./doc.tex\nOutput written on doc.pdf.\n")
    assert not (log.errors or log.warnings or log.missing_chars or log.undefined_refs)
    assert log.summary_warnings() == []


def test_xelatex_argv_miktex_flags() -> None:
    mik = xelatex_argv(Path("xelatex.exe"), "doc.tex", miktex=True)
    assert "--disable-installer" in mik
    assert "-max-print-line=1000" in mik
    for flag in ("-interaction=nonstopmode", "-halt-on-error", "-file-line-error"):
        assert flag in mik
    assert mik[-1] == "doc.tex"
    tl = xelatex_argv(Path("xelatex"), "doc.tex", miktex=False)
    assert "--disable-installer" not in tl
    assert not any(a.startswith(">") or a.endswith(".out") for a in tl)


# ---------------------------------------------------------------- compile loop (fake xelatex)

FAKE_XELATEX = textwrap.dedent(
    """\
    import os, sys
    from pathlib import Path

    args = sys.argv[1:]
    if args == ["--version"]:
        print("FakeTeX 1.0")
        raise SystemExit(0)
    tex = Path(args[-1])
    job = tex.stem
    state = Path("calls.txt")
    n = int(state.read_text()) + 1 if state.exists() else 1
    state.write_text(str(n))
    mode = os.environ.get("FAKE_MODE", "stable")
    if mode == "error":
        Path(job + ".log").write_text(
            "doc.tex:7: Undefined control sequence.\\nl.7 \\\\oops\\n", encoding="utf-8")
        raise SystemExit(1)
    # 'stable': .aux identical from pass 1; 'changing': differs every pass.
    aux = "stable" if mode == "stable" else f"pass {n}"
    Path(job + ".aux").write_text(aux, encoding="utf-8")
    Path(job + ".toc").write_text("toc", encoding="utf-8")
    log = "Output written.\\n"
    if mode == "missingchar":
        log += "Missing character: There is no ж in font lmroman10-regular!\\n"
    Path(job + ".log").write_text(log, encoding="utf-8")
    Path(job + ".pdf").write_bytes(b"%PDF-1.5\\n%%EOF\\n")
    """
)


@pytest.fixture
def fake_xelatex(tmp_path: Path) -> Path:
    script = tmp_path / "fake_xelatex.py"
    script.write_text(FAKE_XELATEX, encoding="utf-8")
    if sys.platform == "win32":
        wrapper = tmp_path / "xelatex.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = tmp_path / "xelatex"
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper


def _tex(tmp_path: Path) -> Path:
    build = tmp_path / "build"
    build.mkdir()
    tex = build / "doc.tex"
    tex.write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")
    return tex


def _calls(tex: Path) -> int:
    return int((tex.parent / "calls.txt").read_text())


def test_compile_stops_after_two_stable_passes(
    tmp_path: Path, fake_xelatex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "stable")
    tex = _tex(tmp_path)
    res = compile_latex(tex, xelatex=fake_xelatex, passes=5)
    assert res.ok, res.errors
    assert res.passes == 2
    assert _calls(tex) == 2
    assert res.pdf == tex.with_suffix(".pdf")
    assert not tex.with_suffix(".out").exists()  # stdout is not redirected to <job>.out


def test_compile_respects_max_passes_when_unstable(
    tmp_path: Path, fake_xelatex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "changing")
    tex = _tex(tmp_path)
    res = compile_latex(tex, xelatex=fake_xelatex, passes=3)
    assert res.ok
    assert res.passes == 3
    assert _calls(tex) == 3


def test_compile_single_pass_setting(
    tmp_path: Path, fake_xelatex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "stable")
    tex = _tex(tmp_path)
    res = compile_latex(tex, xelatex=fake_xelatex, passes=1)
    assert res.ok
    assert res.passes == 1


def test_compile_error_is_parsed(
    tmp_path: Path, fake_xelatex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "error")
    tex = _tex(tmp_path)
    res = compile_latex(tex, xelatex=fake_xelatex, passes=3)
    assert not res.ok
    assert res.passes == 1
    assert res.errors and res.errors[0].startswith("doc.tex:7: Undefined control sequence.")
    assert res.pdf is None


def test_compile_reports_missing_characters(
    tmp_path: Path, fake_xelatex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_MODE", "missingchar")
    tex = _tex(tmp_path)
    res = compile_latex(tex, xelatex=fake_xelatex, passes=2)
    assert res.ok
    assert any("Шрифт не содержит символы" in w and "«ж»" in w for w in res.warnings)


def test_compile_missing_binary(tmp_path: Path) -> None:
    tex = _tex(tmp_path)
    res = compile_latex(tex, xelatex=tmp_path / "no-such-xelatex.exe", passes=2)
    assert not res.ok
    assert res.errors and "XeLaTeX" in res.errors[0]
