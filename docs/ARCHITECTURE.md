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

## Источники и извлечение (M1)

Цель M1 — превратить текстовые источники (PDF с текстовым слоем и без, слайды PDF/PPTX, DOCX, Markdown, .tex, веб-страницы) в Source Docs (PRD 5.3): `extracted/<ID>/source.md`, `blocks.jsonl`, `summary.md`. Видео, аудио и полноценная рукопись — M5 и M4; в M1 они принимаются `h0lon add`, но `extract` их пропускает со статусом `skipped` (скан без текстового слоя обрабатывается универсальным распознаванием страниц).

### Модели — `h0lon/sources/models.py`, `h0lon/extract/model.py` (готово)

`SourceRecord` — запись в `topic.yaml` (`id`, `kind`, `title`, `file`, `original_name`, `url`, `sha256`, `size`, `added`, `units`, `quality`, `status`, `extracted_key`, `error`). Виды: `pdf-text`, `pdf-scan`, `handwritten`, `slides`, `video`, `audio`, `web`, `docx`, `md`, `tex`; буква id по виду — `ID_PREFIX` (H, P, S, V, A, W, D), номер — следующий свободный для буквы. `Block`, `PageImage`, `ExtractContext`, `ExtractPlan`, `ExtractOutput`, `ExtractResult`, протокол `Extractor` — см. код. `workspace.resolve_topic(settings, ref)` находит тему по пути, `<курс>/<тема>` или имени (готово); CLI-команды `add`, `sources`, `extract` уже вызывают функции ниже.

### Приём — `h0lon/sources/ingest.py`

```python
class IngestError(Exception): ...          # пустой список, нет файла, неизвестный --kind

@dataclass
class IngestReport:
    added: list[SourceRecord]; warnings: list[str]
    def to_dict(self) -> dict: ...

def detect_kind(path_or_url: str) -> SourceKind     # по расширению, содержимому и URL
def add_sources(settings, topic_dir, items: Sequence[str], *, kind: str | None = None,
                title: str | None = None) -> IngestReport
def list_sources(topic_dir) -> list[SourceRecord]
def update_source(topic_dir, record: SourceRecord) -> None     # атомарная перезапись topic.yaml
def print_sources(records, *, console, title: str) -> None
```

- Файл копируется в `sources/<ID>_<slug>.<ext>` (slug — `names.slugify` от имени без расширения), оригинальное имя — в `original_name`; `sha256` и `size` считаются по копии. Повторное добавление файла с тем же `sha256` — предупреждение и пропуск.
- Определение вида: `.pdf` — по профилю (медиана символов на страницу; соотношение сторон ≥ 1,3 и мало текста — `slides`; текста нет — `pdf-scan`); `.pptx` → `slides`; `.docx` → `docx`; `.md`/`.markdown`/`.txt` → `md`; `.tex` → `tex`; `.html`/`.htm` и `http(s)://` (кроме видеохостингов) → `web`; YouTube/VK/Rutube и `.mp4/.mkv/.webm/.mov` → `video`; `.mp3/.m4a/.wav/.ogg/.flac` → `audio`; `.jpg/.jpeg/.png/.heic` → `handwritten`. `--kind` перекрывает определение.
- Ссылки на веб-страницы скачиваются сразу (HTML в `sources/W1_<slug>.html`, `url` сохраняется); ссылки на видео только записываются (`file = None`, скачивание — M5).
- `units`: `pages` (PDF), `slides` (PPTX и слайды); `quality` на этапе приёма — `text_layer` (доля страниц с текстом), `chars_per_page` (медиана), `aspect` (ширина/высота первой страницы), `producer`.

### Конвейер — `h0lon/extract/pipeline.py`

```python
def extract_topic(settings, topic_dir, *, source_ids: Sequence[str] | None = None, force=False,
                  use_vision=True, dry_run=False, backend: str | None = None,
                  on_event=None) -> list[ExtractResult] | list[ExtractPlan]
def print_results(results, *, console) -> None
def print_plans(plans, *, console) -> None
def get_extractor(kind: str) -> Extractor | None       # реестр; None — вид пока не поддержан
```

1. Для каждого источника экстрактор пишет `extracted/<ID>/body.md` (Pandoc Markdown всего источника) и возвращает `ExtractOutput`.
2. `extract/blocks.py`: `body.md` → AST Pandoc (`pandoc -t json`) → блоки верхнего уровня: fenced div с классом из `DIV_BLOCK_TYPES` → блок этого типа (`figure-description` → `figure`); абзац только с выключной формулой → `formula`; списки → `list`; таблицы → `table`; рисунки → `figure`; код → `code`; цитаты → `quote`; остальное → `paragraph`; обычные заголовки → `heading`. **Заголовок, начинающийся с якоря `[[X:loc]]`, — заголовок места**: задаёт текущий якорь и сам блоком не является. Если заголовков места нет (DOCX, Markdown, веб), якорь — `<ID>:§<k>`, где k — номер текущего раздела верхнего уровня (самый крупный уровень заголовков в документе). Id блоков — `<ID>.b001`… по порядку.
3. `source.md` = YAML front matter (PRD 5.3: `id`, `kind`, `title`, `origin`, `original_name`, `sha256`, `units`, `quality`, `extracted_by`, `language`) + тело, где перед каждым блоком стоит комментарий `<!-- P1.b007 definition -->`. Тело собирается через Pandoc (`-t markdown --wrap=none`, расширения как у мастера), содержание не меняется.
4. `blocks.jsonl` — по строке `Block.to_dict()` на блок.
5. `summary.md` — один лёгкий прогон агента по `source.md` с контрактом из трёх разделов: `## Аннотация` (3–6 предложений), `## Оглавление` (заголовки с якорями), `## Термины и обозначения` (термин — краткое пояснение; нужен глоссарию ASR в M5 и синтезу S1). Без агентов (`--no-vision` или агент недоступен) оглавление и термины строятся детерминированно из заголовков и определений, аннотация помечается как отсутствующая.
6. Кэш: ключ = sha256 источника + `Extractor.version` + версии промптов + модель уровня; `extracted/<ID>/meta.json` хранит ключ, длительность, число прогонов. Совпадение ключа и наличие `source.md` — пропуск (`cached=True`), если нет `--force`. Статус, `quality`, `units`, `extracted_key`, `error` записываются в `topic.yaml`.

### Экстракторы

| Вид | Модуль | Детерминированная часть | Агент |
|---|---|---|---|
| `pdf-text`, `pdf-scan` | `extract/pdf.py` | Классификация страниц: `text` (обычный текст → `pymupdf4llm` по странице), `math` (шрифты формул: имена с CMMI, CMSY, CMEX, MSAM, MSBM, EURM, TeX-math, CambriaMath, STIX, LatinModernMath, XITS, Asana, или высокая доля формульных глифов), `scan` (текста < 20 символов), `graphic` (мало текста, много рисунков или векторной графики). Удаление колонтитулов, повторяющихся на большинстве страниц. Рендер страниц для агента — PNG, длинная сторона ≤ 2000 px | `math`, `scan`, `graphic` → `vision.transcribe_pages` (flavor `document` или `scan`), текстовый слой страницы — подсказка |
| `slides` (PDF) | `extract/slides.py` | Страница = слайд; заголовок слайда — строка крупнейшего кегля в верхней части; текст — как у PDF | Слайды `math` и `graphic` → `transcribe_pages` (flavor `slides`) |
| `slides` (PPTX) | `extract/slides.py` | `python-pptx`: заголовок, тексты фигур с уровнями списков, таблицы, заметки докладчика (раздел «Заметки»), картинки в `figures/`. Если найден LibreOffice (`soffice`), PPTX конвертируется в PDF, и слайды-картинки распознаются как у PDF | Картинки без текста → описание через `transcribe_pages` |
| `docx`, `md`, `tex` | `extract/docs.py` | Pandoc → Markdown (`--extract-media` в `figures/`), `.tex` — читателем LaTeX | — |
| `web` | `extract/docs.py` | `trafilatura` (основной текст, заголовки, таблицы, ссылки на картинки) → Markdown | — |

Заголовки мест в `body.md`: `## [[P1:p3]] Страница 3`, `## [[S1:s12]] Слайд 12. <заголовок слайда>`.

### Распознавание страниц — `h0lon/extract/vision.py`

```python
@dataclass
class VisionResult:
    pages: dict[int, str]          # номер страницы → Markdown страницы (без заголовка места)
    failed: dict[int, str]         # номер → причина
    agent_runs: int; usage: Usage; cached_pages: int

def transcribe_pages(ctx: ExtractContext, pages: Sequence[PageImage], *,
                     flavor: Literal["document", "slides", "scan"], tier: Tier = "light",
                     batch_size: int = 6) -> VisionResult
```

- Страницы идут батчами по `batch_size` (слайды — до 10) через `agents.create_bundle` и `run_task` (этап `extract`, бэкенд из `ctx.backend`, иначе `agents.default`). Бандлы — в `<тема>/runs/`. Контракт: файл `out/p<NNNN>.md` на каждую страницу батча. В `inputs/` — PNG страниц и `p<NNNN>.txt` с текстовым слоем.
- Промпт — `h0lon/prompts/vision_pages@1.0.md`: точная транскрипция без пересказа и сокращений; формулы — LaTeX (`$…$`, `$$…$$`, `aligned`); семантические блоки — fenced divs из `DIV_BLOCK_TYPES`, только если они явно обозначены в источнике («Определение», «Теорема», «Def.», рамка); таблицы — pipe tables; рисунки и схемы — словесное описание в `::: {.figure-description}` (что изображено, подписи, оси, стрелки); неразборчивое — `[неразборчиво]`; колонтитулы, номера страниц, логотипы не переносятся; текстовый слой — подсказка для написания терминов, но картинка главнее; язык оригинала сохраняется. Для `slides` заголовок слайда не повторяется; для `scan` допускается рукописный текст.
- Кэш страниц: `extracted/<ID>/pages/p<NNNN>.md` и `p<NNNN>.key` (sha256 PNG + подсказки + версия промпта и flavor); совпадение — без агента.
- Страница, не распознанная после всех повторов, попадает в `failed`; экстрактор вставляет на её место текстовый слой (если он есть) в блоке `::: uncertain` «Страница не распознана агентом» и пишет предупреждение.

### Уточнения по итогам реализации M1

- **Приём.** `topic.yaml` хранит `source_counters` (буква → наибольший выданный номер), поэтому номера удалённых источников не выдаются повторно. У веб-записей после редиректа есть `final_url`; страница сохраняется в UTF-8 (кодировка определяется по HTTP, BOM, meta, затем utf-8/cp1251). Каталоги не раскрываются (предупреждение), а шаблоны `*` и `?` раскрывает сам `ingest`, потому что PowerShell и cmd их не раскрывают. `update_source` бросает `KeyError` для неизвестного id, `list_sources` — `ValueError` на испорченной записи; запись `topic.yaml` атомарна и идёт под блокировкой `topic_lock` (файл `.topic.yaml.lock`; между процессами блокировки нет). Дополнительные ключи `quality`: `image_cover`, `creator`, `notes`, `detected_kind` (если `--kind` расходится с автоопределением), `slides_with_notes`, `encoding`.
- **Конвейер.** Название документа (класс `.source-title`) и вступление до первого раздела получают якорь `§0`. В ключ кэша входят также флаг распознавания и бэкенд. Экстрактор может объявить `prompts` (версии промптов в ключе кэша). `ExtractError` живёт в `extract/registry.py`. Front matter `source.md` содержит `url` для веб-источников и `extracted_by` (экстрактор, бэкенд, модель, промпты, дата, версия h0lon).
- **PDF и слайды.** Текст PDF-слайдов собирается построчно из текстового слоя; `pymupdf4llm` используется только для слайдов с таблицами. Конспекты, свёрстанные KaTeX в браузере, распознаются как `math` по глифам. Текст нормализуется в NFC, восстанавливаются пропущенные пробелы, математические курсивные символы в заголовках слайдов (𝑘) заменяются обычными буквами. PNG для агента — 1400–2000 px по длинной стороне.

### Известные ограничения M1

- Отсканированная рукопись определяется как `pdf-scan`: печатный скан от рукописи без распознавания не отличить. Для рукописи укажите `--kind handwritten` (полный конвейер — M4). Лёгкая модель на рукописи ошибается в сокращениях и знаках — в M4 нужен сильный уровень или проход-проверка.
- Источники и `topic.yaml` пока не коммитятся в git-репозиторий темы автоматически.
- Картинки веб-страниц не скачиваются (ссылки ведут на сайт); `trafilatura` теряет `<figcaption>`, у Википедии остаются цифры обратных ссылок в примечаниях. Формулы веб-страниц без TeX-источника помечаются в `quality.notes`.
- Путь PPTX → PDF через LibreOffice проверен только на фейковом конвертере; без LibreOffice формулы OMML и SmartArt не извлекаются (пишется в `quality.notes`).
- Источники извлекаются последовательно; на Windows каждый вызов Pandoc стоит около 0,6 с.

### Сигналы качества (`quality` в `topic.yaml` и front matter `source.md`)

`text_layer` (доля страниц с текстом), `chars_per_page`, `pages_math`, `pages_scan`, `pages_graphic`, `pages_vision`, `pages_failed`, `cyrillic_ratio` (доля кириллицы среди букв), `scan_dpi` (оценка для сканов), `notes` (русские строки — что стоит проверить на review gate).

## Синтез мастер-конспекта (M2)

Цель M2 — `h0lon build <тема>`: Source Docs всех источников → `synthesis/` → `master.md` → `master.pdf` (PRD 5.4, 7, 8, 9). Модели — `h0lon/synth/model.py` (готово); промпты — `h0lon/prompts/{outline,sections,globalpass,coverage,supplement,fixlatex}@1.0.md` (готово). Синтез (S1–S3, S5) идёт на сильном уровне агента (`tier="strong"`, по умолчанию Claude Opus), разбор покрытия и починка LaTeX (S4, S6) — на лёгком.

### Стадии и файлы `<тема>/synthesis/`

| Стадия | Модуль | Вход | Выход | Проверка кодом |
|---|---|---|---|---|
| extract | M1 `extract_topic` | источники | `extracted/<ID>/…` | M1 |
| review gate | `synth/build.py` | — | — | см. ниже |
| S1 outline | `synth/outline.py` | `summaries.md` (все `summary.md` с заголовками источников), `blocks_index.md` (по строке на блок: `id [тип] (якорь) первые ~160 символов текста`) | `outline.json`, `outline.md` | JSON Schema; каждый не-`admin` блок назначен ровно одному листу или указан в `unassigned` с причиной; id разделов уникальны и в формате `s01`/`s01-01`; иначе повтор с обратной связью |
| S2 sections | `synth/sections.py` | `outline.md`, `glossary.md` (объединённые «Термины и обозначения»), `<id>.blocks.md` на раздел | `sections/<id>.md`, `sections/<id>.notes.json` | файл начинается заголовком `{#sec:<id>}` нужного уровня; все id блоков раздела есть в комментариях `src` (иначе повтор с обратной связью, список пропущенных id); notes — JSON Schema |
| S3 global | `synth/globalpass.py` | копии всех разделов в `out/sections/`, `outline.md`, `glossary.md`, `notes.json` | правленые `sections/*.md`, `intro.md`, `glossary.md`, `global.notes.json` | множество id в `src` по всем разделам не уменьшилось; заголовки `{#sec:<id>}` на месте; суммарный объём текста не меньше 90 % исходного; иначе повтор, после второго провала — разделы S2 остаются без глобальной правки (предупреждение) |
| S4 coverage | `synth/coverage.py` | `uncovered.md`, `outline.md`, `master_index.md` | `coverage.json` | вердикт на каждый блок |
| S5 supplement | `synth/coverage.py` | копии целевых разделов, `missing.md` | правленые разделы | id `missing`-блоков появились в `src`; не больше 2 кругов S4→S5 |
| assemble | `synth/assemble.py` | разделы, `intro.md`, `glossary.md`, все notes, `coverage.json` | `<тема>/master.md` | — |
| S6 render | `synth/render.py` | `master.md` | `<тема>/master.pdf` | `render_document`; при ошибке XeLaTeX — до 3 прогонов `fixlatex` по логу (проверка: id в `src` и якоря не изменились, объём ±3 %), затем запасной HTML-движок |

Группировка S2: подряд идущие листья структуры собираются в группы по 1–4 раздела так, чтобы суммарный объём блоков группы не превышал ~60 000 символов (раздел больше порога идёт один). Группы выполняются параллельно, не больше `agents.parallel_runs`. Бандлы — `<тема>/runs/`, этапы `outline`, `sections`, `global`, `coverage`, `supplement`, `fixlatex`.

### Кэш стадий

`synthesis/<stage>.meta.json` хранит ключ: sha256 входов стадии + версия промпта + модель уровня. Совпал ключ и есть выход — стадия пропускается. Изменение одного источника меняет ключи всех последующих стадий; изменение одного раздела структуры пересчитывает в S2 только его группу (ключ группы — по её входам). `--force` и `--from <стадия>` перезапускают стадию и всё, что после неё.

### Сборка `master.md`

Front matter: `title` (из структуры), `subtitle: "Мастер-конспект темы"`, `course`, `date` (дата сборки), `author: "H0lon"`, `sources` (id, kind, title, units). Тело: `intro.md`, разделы в порядке структуры, `glossary.md`, затем приложения:

- `# Расхождения между источниками {#app:conflicts .appendix}` — из всех `conflicts` (тема, варианты с якорями, выбранный вариант и причина);
- `# Журнал правок {#app:corrections}` — таблица `corrections` (якорь, как написано, как исправлено, причина); пустой журнал — явная фраза «Прямых правок не было»;
- `# Редакторские дополнения {#app:editorial}` — `editorial` со ссылкой на раздел;
- `# Карта покрытия {#app:coverage}` — таблица «источник → блоков → покрыто → %» и список непокрытых блоков с вердиктами.

### Review gate и команды

- `h0lon build <тема> [--no-review] [--from <стадия>] [--force] [--backend claude|codex] [--json]` — все стадии по порядку; результат — `BuildResult`. Код выхода: 0 — готово, 1 — ошибка, 3 — остановка на review gate.
- Review gate: если `topic.yaml review_gate` (иначе `general.review_gate`) включён, `build` после извлечения проверяет `topic.yaml review: {approved_at, extracted_keys}`; если одобрения нет или `extracted_keys` не совпадают с текущими `extracted_key` источников — остановка с подсказкой. `--no-review` пропускает проверку для этого запуска.
- `h0lon approve <тема>` — записать одобрение текущего извлечения. `h0lon status <тема>` — источники, стадии (готово, из кэша, устарело), покрытие, путь к master.pdf.
- После успешной сборки — коммит в git-репозиторий темы «Сборка мастера: <дата>, покрытие NN %» (если тема — git-репозиторий; ошибки git — предупреждение).

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
