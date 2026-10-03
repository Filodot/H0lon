# H0lon — архитектура и контракты модулей

Документ описывает, как устроен код и какие интерфейсы модули обязаны предоставлять друг другу. Требования и решения — в [PRD](PRD.md); здесь — их отображение на пакет `h0lon/`. Раздел помечен этапом, на котором интерфейс появляется.

## Общие слои (M0, готово)

| Модуль | Назначение |
|---|---|
| `h0lon/config.py` | `Settings` (pydantic-settings): TOML + переопределения `H0LON_<РАЗДЕЛ>__<КЛЮЧ>`. `load_settings(path=None)`, `find_config_file()`, `user_config_path()`, `example_config_text()`, `write_user_config()`. Пример — `h0lon/resources/h0lon.example.toml` (копия в корне репозитория должна совпадать байт в байт) |
| `h0lon/procutil.py` | `run(argv, cwd, input_text, timeout, env, on_stdout_line) -> ProcResult` — UTF-8, обработчик строки может вернуть `procutil.STOP`, чтобы досрочно остановить процесс (`ProcResult.stopped`), stdin всегда закрыт или подан и закрыт, таймаут с убийством дерева процессов, без окна консоли на Windows; `clean_env(drop_session_vars=True, drop_names=…, extra=…)` убирает переменные родительской сессии Claude Code (`CLAUDECODE`, `CLAUDE_CODE_*`, `CLAUDE_AGENT_SDK_*`, `CLAUDE_PID`, `CLAUDE_EFFORT`, `CLAUDE_PREVIEW_CLASSIFIER_FLOOR`), но сохраняет пользовательский `CLAUDE_CODE_OAUTH_TOKEN`; `API_KEY_ENV_NAMES` |
| `h0lon/tools.py` | Поиск внешних инструментов: `find_pandoc(override)` (PATH → pypandoc-binary), `find_xelatex(override)` (PATH → MiKTeX → TeX Live), `find_tex_tool(xelatex, name)`, `is_miktex(xelatex)`, `missing_tex_files(xelatex, files)`, `find_browser(preference)`, `resolve_agent_argv(spec, kind)` (для npm-установки Codex возвращает нативный `codex.exe`), `tool_version(argv)`, `find_simple(name)` |
| `h0lon/cli.py` | Typer-приложение; команды импортируют модули лениво внутри функций |

## Рендер (M0) — `h0lon/render/`

```python
# h0lon/render/__init__.py
from h0lon.render.pipeline import RenderReport, render_document
from h0lon.render.templates import TemplateInfo, get_template, list_templates

def render_document(
    src: Path, *, settings: Settings,
    out_dir: Path | None = None,          # по умолчанию src.parent
    template: str | None = None,          # по умолчанию settings.render.template
    clean: bool = False,                  # убрать якоря [[…]]
    engine: Literal["auto", "xelatex", "html"] = "auto",
    keep_build: bool = False,             # сохранить каталог сборки
    metadata: dict[str, str] | None = None,
) -> RenderReport

@dataclass
class RenderReport:
    ok: bool; pdf: Path | None; engine_used: str | None; fallback_used: bool
    tex: Path | None; build_dir: Path | None; passes: int
    warnings: list[str]; errors: list[str]; checks: PdfCheckReport | None; duration_s: float
    def to_dict(self) -> dict: ...
    def print(self, console: rich.console.Console) -> None: ...

@dataclass
class TemplateInfo:
    name: str; dir: Path; template_tex: Path; title: str
    latex_packages: list[str]            # файлы .sty, которые нужны шаблону (для doctor)
    html_css: Path | None                # стиль для запасного HTML-рендера

def get_template(name: str, settings: Settings) -> TemplateInfo   # KeyError, если нет
def list_templates(settings: Settings) -> list[TemplateInfo]
```

```python
@dataclass
class PdfCheckReport:                    # h0lon/render/pdfcheck.py, check_pdf(path, expected_title)
    ok: bool; path: str; pages: int; text_chars: int
    title_meta: str | None; title_found: bool | None; bookmarks: int
    fonts: list[str]; not_embedded: list[str]; type3: list[str]
    errors: list[str]; warnings: list[str]
```

Если движок собрал PDF, но проверка не прошла (нет текста, шрифты не встроены, 0 страниц), попытка считается неудачной и в режиме `auto` запускается запасной движок.

### Соглашения разметки мастера (их должны соблюдать промпты синтеза M2)

- Семантические блоки — fenced divs: `definition`, `theorem`, `lemma`, `proposition`, `corollary`, `proof`, `example`, `problem`, `solution`, `remark`; служебные `editorial`, `conflict`, `uncertain`, `author-question`. Атрибуты: `title="…"`, `#id`.
- Якоря источников `[[H1:p3]]`, `[[S1:s12]]`, `[[V1:12:34]]`, `[[P2:p45-47]]` — после слова или знака препинания; подряд идущие сливаются в один индекс. `--clean` (или `h0lon-clean: true` в front matter) убирает их.
- Привязка к блокам источников — HTML-комментарии `<!-- src: H1.b007 V1.b012 -->`; в вывод не попадают.
- Приложения начинаются заголовком первого уровня с классом `.appendix` (дальше «Приложение А.», «Приложение Б.» …); `.unnumbered` допустим.
- Перекрёстные ссылки — обычные ссылки `[текст](#id)`. Синтаксис pandoc-crossref (`[@sec:x]`) не поддерживается.
- Неразборчивое — `[неразборчиво]` (выводится серым).

Шаблоны ищутся в `<state_dir>/templates/<name>/` (пользовательские, приоритет) и во встроенном `h0lon/render/templates/<name>/` (`template.tex`, `meta.yaml`, `style.css`). Lua-фильтры — `h0lon/render/filters/`.

## Агенты (M0) — `h0lon/agents/`

```python
# h0lon/agents/__init__.py
Tier = Literal["light", "strong"]

@dataclass
class ExpectedFile:
    path: str                             # относительно out/
    kind: Literal["markdown", "json", "text"]
    required: bool = True
    min_chars: int = 1
    required_headings: list[str] = []     # для markdown: точные строки заголовков
    json_schema: dict | None = None       # для json

@dataclass
class OutputContract:
    files: list[ExpectedFile]
    final_message_schema: dict | None = None   # JSON Schema финального ответа агента

@dataclass
class TaskBundle:                         # каталог runs/<id>/
    id: str; root: Path; stage: str
    task_path: Path                       # task.md
    inputs_dir: Path; out_dir: Path
    images: list[Path]                    # абсолютные пути внутри inputs/
    workspace: Path | None                # папка темы (доступ на чтение)
    contract: OutputContract

def create_bundle(runs_dir: Path, *, stage: str, task: str,
                  contract: OutputContract, inputs: Sequence[Path] = (), images: Sequence[Path] = (),
                  workspace: Path | None = None, bundle_id: str | None = None) -> TaskBundle

def run_task(bundle: TaskBundle, *, settings: Settings, tier: Tier = "strong",
             backend: str | None = None, fallback: bool = True,
             on_event: Callable[[str], None] | None = None) -> RunResult

@dataclass
class Usage: input_tokens: int = 0; cached_input_tokens: int = 0; output_tokens: int = 0
             reasoning_tokens: int = 0; cost_usd: float | None = None

ErrorKind = Literal["auth", "rate_limit", "timeout", "crash", "validation", "not_found", "refusal", "unknown"]

@dataclass
class AttemptRecord:
    n: int; backend: str; model: str | None; started_at: str; duration_s: float
    exit_code: int | None; ok: bool; error_kind: ErrorKind | None; error: str | None
    usage: Usage; validation_problems: list[str]; transcript: str   # путь к transcript.jsonl попытки
    limits: dict | None                   # загрузка окон подписки (Claude: rate_limit_event)

@dataclass
class RunResult:
    ok: bool; bundle: TaskBundle; backend_used: str | None
    attempts: list[AttemptRecord]; usage_total: Usage
    final_text: str; problems: list[str]
    def to_dict(self) -> dict: ...
    def print(self, console) -> None: ...

class AgentBackend(Protocol):
    name: str
    inline_system_prompt: bool                          # True: системный текст в начале промпта (Codex)
    def resolve(self) -> list[str] | None              # argv-префикс или None, если CLI нет
    def auth_status(self) -> AuthStatus                 # logged_in: bool | None, method, detail
    def invoke(self, bundle: TaskBundle, *, tier: Tier, prompt: str,
               attempt_dir: Path, on_event) -> BackendOutcome

def get_backend(name: str, settings: Settings) -> AgentBackend
```

```python
# h0lon/agents/selftest.py
def agent_test(settings: Settings, *, prompt: str | None = None, backend: str | None = None,
               fallback: bool = True, tier: Tier = "light", images: Sequence[Path] = (),
               on_event: Callable[[str], None] | None = None) -> RunResult
```

`run_task` пишет `runs/<id>/run.json` и дописывает строку в `<state_dir>/usage.jsonl` (токены, длительность, `limits`). Claude сообщает загрузку окон подписки событием `rate_limit_event` (`five_hour`, `seven_day`: `utilization` 0–1 и время сброса) — это основа темпа очереди (M7). При `api_retry` с `authentication_failed` и при окне `rejected` бэкенд останавливает CLI досрочно, не дожидаясь его внутренних повторов. Политика: `auth`/`not_found` — сразу запасной агент; `rate_limit` — запасной агент, исходный помечается «остывающим» до конца процесса; `validation` — до `max_validation_retries` повторов с обратной связью, затем запасной; `timeout`/`crash` — один повтор, затем запасной; `refusal` — запасной.

## Doctor и рабочие области (M0)

```python
# h0lon/doctor.py
@dataclass
class Check:
    id: str; title: str
    status: Literal["ok", "warn", "missing", "error", "info"]
    detail: str; hint: str | None = None; required: bool = False
    def to_dict(self) -> dict: ...

def run_checks(settings: Settings, *, check_auth: bool = True) -> list[Check]
def print_report(checks: list[Check], *, console, settings: Settings) -> None
def exit_code(checks: list[Check]) -> int      # 1, если обязательная проверка не ok

# h0lon/names.py
def slugify(text: str, *, max_len: int = 60) -> str   # транслитерация ru → ASCII, [a-z0-9-]

# h0lon/workspace.py
TOPIC_DIRS = ("sources", "extracted", "synthesis", "variants", "runs")
class TopicMeta(BaseModel): title, course, slug, language, created, sources: list[dict],
                            review_gate: bool | None, backends: dict[str, str]
def create_topic(settings: Settings, *, title: str, course: str, slug: str | None = None) -> Path
def load_topic(path: Path) -> TopicMeta
```

## Правила для всех модулей

- Внешние процессы — только через `procutil.run`; поиск инструментов — только через `tools`.
- Тесты не вызывают реальных агентов и сеть; реальные прогоны — маркер `live_agent` (включается `H0LON_LIVE_AGENT=1`). Тесты XeLaTeX и браузера помечаются `needs_xelatex` / `needs_browser` и пропускаются, если инструмента нет.
- Сообщения пользователю — на русском; идентификаторы и комментарии в коде — на английском.

## Известные ограничения M0

- Сырой TeX в мастере (`\input{…}`, абсолютные пути к картинкам) в движке XeLaTeX не ограничивается: мастер пишут наши же агенты из материалов пользователя, PDF остаётся локальным. Фильтр `sanitize.lua` защищает только HTML-движок.
- Таблица (`longtable`) внутри рамочного блока (доказательство, решение) теряет нижнюю линейку.
- SVG-картинки в XeLaTeX не поддерживаются; нумерованные перекрёстные ссылки («см. теорему 2.1» номером) — нет.
- Длинные строки кода переносятся фильтром по 90 символам (нет `fvextra`).
- В HTML-движке нет переносов русских слов и номеров страниц в оглавлении, сноски — концевые.
- Если H0lon запущен из упакованного приложения (например, Claude Desktop из Microsoft Store), запись в `%LOCALAPPDATA%` виртуализуется: `state_dir` фактически оказывается в `%LOCALAPPDATA%\Packages\<приложение>\LocalCache\Local\H0lon`. Из обычного терминала эти файлы не видны; при необходимости задайте `general.state_dir` явно.
- Пометка «остывающего» агента живёт до конца процесса; долгоживущему веб-интерфейсу (M3) понадобится сброс.
