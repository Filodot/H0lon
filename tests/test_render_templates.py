"""Template discovery: built-in a4-notes, user templates in <state_dir>/templates/."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from h0lon.config import Settings
from h0lon.render import get_template, list_templates
from h0lon.render.templates import BUILTIN_TEMPLATES_DIR


def test_builtin_a4_notes(settings: Settings) -> None:
    info = get_template("a4-notes", settings)
    assert info.name == "a4-notes"
    assert info.dir == BUILTIN_TEMPLATES_DIR / "a4-notes"
    assert info.template_tex.is_file()
    assert info.title and info.title != "a4-notes"
    assert info.html_css is not None and info.html_css.name == "style.css"
    assert info.template_html is not None and info.template_html.is_file()
    assert "mdframed.sty" in info.latex_packages
    assert "fontspec.sty" in info.latex_packages
    assert all(p.endswith(".sty") for p in info.latex_packages)


def test_default_template_from_settings_exists(settings: Settings) -> None:
    assert get_template(settings.render.template, settings).name == settings.render.template


def test_latex_packages_match_template_usepackage(settings: Settings) -> None:
    info = get_template("a4-notes", settings)
    tex = info.template_tex.read_text(encoding="utf-8")
    used: set[str] = set()
    for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        used.update(f"{name.strip()}.sty" for name in m.group(1).split(","))
    # Loaded only if present (\IfFileExists{x.sty}{...}{fallback}) or only when the document
    # needs them ($if(multirow)$...): doctor must not report them as missing.
    if_exists = set(re.findall(r"\\IfFileExists\{([^}]+\.sty)\}", tex))
    assert if_exists == {"upquote.sty", "xurl.sty", "footnotehyper.sty"}
    optional = if_exists | {"footnote.sty", "multirow.sty", "subcaption.sty"}
    assert set(info.latex_packages) == used - optional
    meta = yaml.safe_load((info.dir / "meta.yaml").read_text(encoding="utf-8"))
    assert set(meta["optional_latex_packages"]) == if_exists | {"footnote.sty"}
    assert not set(meta["optional_latex_packages"]) & set(info.latex_packages)
    assert "listings.sty" not in used  # breaks Cyrillic under XeLaTeX
    assert "soul.sty" not in used  # loses Cyrillic under XeLaTeX (soul 3.x)


def test_unknown_template_raises_keyerror(settings: Settings) -> None:
    with pytest.raises(KeyError):
        get_template("no-such-template", settings)


@pytest.mark.parametrize("name", ["", "../a4-notes", "a4-notes/..", "a/b", ".hidden"])
def test_invalid_names_raise_keyerror(settings: Settings, name: str) -> None:
    with pytest.raises(KeyError):
        get_template(name, settings)


def _user_template(settings: Settings, name: str, meta: str | None) -> Path:
    d = settings.general.state_path / "templates" / name
    d.mkdir(parents=True)
    (d / "template.tex").write_text("$body$\n", encoding="utf-8")
    if meta is not None:
        (d / "meta.yaml").write_text(meta, encoding="utf-8")
    return d


def test_user_template_overrides_builtin(settings: Settings) -> None:
    d = _user_template(settings, "a4-notes", "title: Мой шаблон\nlatex_packages: [geometry]\n")
    info = get_template("a4-notes", settings)
    assert info.dir == d
    assert info.title == "Мой шаблон"
    assert info.latex_packages == ["geometry.sty"]
    assert info.html_css is None
    assert info.template_html is None
    names = [t.name for t in list_templates(settings)]
    assert names.count("a4-notes") == 1
    assert next(t for t in list_templates(settings) if t.name == "a4-notes").dir == d


def test_list_templates_includes_user_and_builtin(settings: Settings) -> None:
    _user_template(settings, "cheatsheet", None)
    (settings.general.state_path / "templates" / "not-a-template").mkdir()
    names = [t.name for t in list_templates(settings)]
    assert names == sorted(names)
    assert "a4-notes" in names
    assert "cheatsheet" in names
    assert "not-a-template" not in names
    cheat = get_template("cheatsheet", settings)
    assert cheat.title == "cheatsheet"
    assert cheat.latex_packages == []


def test_user_css_from_meta(settings: Settings) -> None:
    d = _user_template(settings, "web", "html_css: print.css\n")
    (d / "print.css").write_text("body{}", encoding="utf-8")
    assert get_template("web", settings).html_css == (d / "print.css").resolve()


def test_broken_meta_yaml(settings: Settings) -> None:
    _user_template(settings, "broken", "title: [unclosed\n")
    with pytest.raises(ValueError, match="broken"):
        get_template("broken", settings)
    assert "broken" not in [t.name for t in list_templates(settings)]


def test_bad_latex_packages_type(settings: Settings) -> None:
    _user_template(settings, "badpk", "latex_packages: geometry\n")
    with pytest.raises(ValueError, match="latex_packages"):
        get_template("badpk", settings)
