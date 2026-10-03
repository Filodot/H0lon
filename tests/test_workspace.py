from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from h0lon import procutil, tools, workspace
from h0lon.workspace import (
    TOPIC_DIRS,
    TopicMeta,
    TopicWarning,
    create_topic,
    create_topic_detailed,
    load_topic,
    save_topic,
)

GIT = tools.find_simple("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git не найден")


@pytest.fixture
def no_git_settings(make_settings):
    return make_settings(general={"git_per_topic": False})


def _git(path: Path, *args: str) -> procutil.ProcResult:
    assert GIT is not None
    return procutil.run([str(GIT), *args], cwd=path, timeout=60)


@pytest.fixture
def isolated_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point git at a throwaway global config; returns a function that writes it."""
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "EMAIL",
        "GIT_DIR",
        "GIT_WORK_TREE",
    ):
        monkeypatch.delenv(name, raising=False)

    def write(text: str) -> None:
        gitconfig.write_text(text, encoding="utf-8")

    return write


def test_create_topic_layout(no_git_settings) -> None:
    path = create_topic(no_git_settings, title="Теорвер — лекция 3", course="theorver")
    assert path == no_git_settings.general.workspaces_dir / "theorver" / "teorver-lektsiya-3"
    for name in TOPIC_DIRS:
        assert (path / name).is_dir()
    assert not (path / ".git").exists()
    gitignore = (path / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("*.mp4", "*.mkv", "*.webm", "*.m4a", "*.wav", "*.mp3"):
        assert pattern in gitignore.splitlines()


def test_topic_yaml_contents(no_git_settings) -> None:
    path = create_topic(no_git_settings, title="Теорвер — лекция 3", course="Теория вероятностей")
    assert path.parent.name == "teoriya-veroyatnostey"
    raw = (path / "topic.yaml").read_text(encoding="utf-8")
    assert "Теорвер — лекция 3" in raw  # allow_unicode: no \u escapes
    meta = load_topic(path)
    assert meta.title == "Теорвер — лекция 3"
    assert meta.course == "Теория вероятностей"
    assert meta.slug == "teorver-lektsiya-3"
    assert meta.language == "ru"
    assert meta.sources == []
    assert meta.review_gate is None
    assert meta.backends == {}
    created = datetime.fromisoformat(meta.created)
    assert created.utcoffset() is not None and created.utcoffset().total_seconds() == 0
    assert load_topic(path / "topic.yaml") == meta


def test_language_from_settings(make_settings) -> None:
    s = make_settings(general={"git_per_topic": False, "language": "en"})
    assert load_topic(create_topic(s, title="T", course="c")).language == "en"


def test_explicit_slug_is_normalized(no_git_settings) -> None:
    path = create_topic(no_git_settings, title="Что угодно", course="theorver", slug="Lecture_03")
    assert path.name == "lecture-03"
    assert load_topic(path).slug == "lecture-03"


def test_existing_non_empty_topic_raises(no_git_settings) -> None:
    path = create_topic(no_git_settings, title="Лекция 1", course="c")
    with pytest.raises(FileExistsError):
        create_topic(no_git_settings, title="Лекция 1", course="c")
    assert (path / "topic.yaml").is_file()


def test_existing_empty_dir_is_reused(no_git_settings) -> None:
    target = no_git_settings.general.workspaces_dir / "c" / "lektsiya-1"
    target.mkdir(parents=True)
    assert create_topic(no_git_settings, title="Лекция 1", course="c") == target
    assert (target / "topic.yaml").is_file()


def test_empty_title_rejected(no_git_settings) -> None:
    with pytest.raises(ValueError):
        create_topic(no_git_settings, title="  ", course="c")
    with pytest.raises(ValueError):
        create_topic(no_git_settings, title="T", course="")


def test_load_topic_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_topic(tmp_path)
    (tmp_path / "topic.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_topic(tmp_path)
    (tmp_path / "topic.yaml").write_text("title: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_topic(tmp_path)


def test_load_topic_accepts_unquoted_timestamp_and_extra_keys(tmp_path: Path) -> None:
    (tmp_path / "topic.yaml").write_text(
        "title: Т\ncourse: c\nslug: t\ncreated: 2026-10-03 12:00:00+00:00\nfuture_key: 1\n",
        encoding="utf-8",
    )
    meta = load_topic(tmp_path)
    assert meta.created.startswith("2026-10-03")
    save_topic(tmp_path, meta)
    assert load_topic(tmp_path).model_extra == {"future_key": 1}


def test_save_topic_roundtrip(tmp_path: Path) -> None:
    meta = TopicMeta(
        title="Т",
        course="к",
        slug="t",
        created="2026-10-03T00:00:00+00:00",
        sources=[{"id": "H1", "file": "sources/H1_lektsiya.pdf", "original": "Лекция.pdf"}],
        review_gate=False,
        backends={"synthesis": "codex"},
    )
    save_topic(tmp_path, meta)
    assert load_topic(tmp_path) == meta


def test_git_missing_warns(make_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    s = make_settings()  # git_per_topic = True by default
    monkeypatch.setattr(workspace.tools, "find_simple", lambda name: None)
    with pytest.warns(TopicWarning, match="git не найден"):
        path = create_topic(s, title="Лекция", course="c")
    assert (path / "topic.yaml").is_file()
    assert not (path / ".git").exists()


@needs_git
def test_git_repo_and_first_commit(make_settings, isolated_git) -> None:
    isolated_git("[user]\n\tname = Test User\n\temail = test@example.com\n")
    result = create_topic_detailed(make_settings(), title="Теорвер — лекция 3", course="theorver")
    assert result.warnings == []
    assert result.git_repo and result.committed
    path = result.path
    assert (path / ".git").is_dir()
    log = _git(path, "log", "--format=%s")
    assert log.ok and log.stdout.strip() == "Тема создана"
    branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
    assert branch.stdout.strip() == "main"
    files = _git(path, "ls-files").stdout.split()
    assert sorted(files) == [".gitignore", "topic.yaml"]


@needs_git
def test_git_without_identity_still_creates_topic(make_settings, isolated_git) -> None:
    isolated_git("[user]\n\tuseConfigOnly = true\n")
    result = create_topic_detailed(make_settings(), title="Лекция 2", course="theorver")
    assert result.git_repo and not result.committed
    assert len(result.warnings) == 1 and "идентичность" in result.warnings[0]
    assert (result.path / "topic.yaml").is_file()
    assert (result.path / ".git").is_dir()

    with pytest.warns(TopicWarning, match="идентичность"):
        create_topic(make_settings(), title="Лекция 3", course="theorver")
