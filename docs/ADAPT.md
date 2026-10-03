# Как адаптировать H0lon под себя

H0lon задуман так, чтобы его можно было подстроить без правки кода: пути, агенты, модели, шрифты и шаблоны задаются конфигом и файлами в каталоге служебных данных. Если нужно больше — новый бэкенд агента или изменения в коде, — в конце описано, как устроена разработка.

Установка — в [INSTALL.md](INSTALL.md), устройство кода и контракты модулей — в [ARCHITECTURE.md](ARCHITECTURE.md), требования и решения — в [PRD.md](PRD.md).

## Содержание

1. [Конфигурация](#1-конфигурация)
2. [Свои шаблоны PDF](#2-свои-шаблоны-pdf)
3. [Шрифты на macOS и Linux](#3-шрифты-на-macos-и-linux)
4. [Новый бэкенд агента](#4-новый-бэкенд-агента)
5. [Разработка](#5-разработка)

## 1. Конфигурация

`uv run h0lon init` создаёт `h0lon.toml` из примера [`h0lon.example.toml`](../h0lon.example.toml). Где ищется файл (первый найденный):

1. ключ `--config <файл>` перед командой: `uv run h0lon --config D:\my.toml doctor`;
2. переменная окружения `H0LON_CONFIG`;
3. `h0lon.toml` в текущей папке;
4. каталог конфигурации пользователя: Windows — `%APPDATA%\H0lon\h0lon.toml`, macOS — `~/Library/Application Support/H0lon/h0lon.toml`, Linux — `~/.config/H0lon/h0lon.toml`.

Приоритет значений: переменные окружения `H0LON_<РАЗДЕЛ>__<КЛЮЧ>` → файл → значения по умолчанию. Для вложенных разделов уровни разделяются двумя подчёркиваниями: ключ `model_strong` из `[agents.claude]` — это `H0LON_AGENTS__CLAUDE__MODEL_STRONG`. Пустая строка в путях означает «по умолчанию».

Столбец «Когда» показывает, с какого этапа дорожной карты ([PRD §17](PRD.md#17-дорожная-карта)) ключ на что-то влияет; ключи будущих этапов уже есть в конфиге, чтобы его формат не менялся.

### `[general]`

| Ключ | По умолчанию | Что задаёт | Когда |
|---|---|---|---|
| `language` | `"ru"` | Язык темы, записывается в `topic.yaml` | M0 |
| `workspaces` | `""` → `~/Konspekty` | Каталог рабочих областей тем `<курс>/<тема>/` | M0 |
| `state_dir` | `""` → каталог данных ОС | Служебные данные: журнал расхода `usage.jsonl`, свои шаблоны, тестовые прогоны. Windows — `%LOCALAPPDATA%\H0lon`, macOS — `~/Library/Application Support/H0lon`, Linux — `~/.local/share/H0lon` | M0 |
| `review_gate` | `true` | Пауза для проверки извлечения перед синтезом (тема может переопределить) | M2–M4 |
| `git_per_topic` | `true` | Каждая тема — git-репозиторий, каждая сборка — коммит | M0 (`h0lon new`) |
| `keep_video` | `false` | Хранить видео после извлечения аудио и кадров | M5 |

### `[agents]`

| Ключ | По умолчанию | Что задаёт | Когда |
|---|---|---|---|
| `default` | `"claude"` | Агент по умолчанию: `claude` или `codex` | M0 |
| `fallback` | `"codex"` | Запасной агент при лимитах и сбоях; `""` — без запасного | M0 |
| `parallel_runs` | `2` | Сколько прогонов агентов идёт одновременно | M2 |
| `max_validation_retries` | `2` | Повторы с обратной связью, если результат не прошёл проверку | M0 |

### `[agents.claude]` — Claude Code (`claude -p`)

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `bin` | `"claude"` | Имя на `PATH`, полный путь или список argv, например `["C:/tools/claude.exe"]` |
| `model_strong` | `"opus"` | Модель для сильного уровня (синтез, сложные страницы) |
| `model_light` | `"sonnet"` | Модель для лёгкого уровня |
| `effort_strong` | `"high"` | Усилие рассуждения: `low`, `medium`, `high`, `xhigh`, `max` |
| `effort_light` | `"low"` | То же для лёгкого уровня |
| `max_turns` | `40` | Предел ходов агента за прогон |
| `timeout_s` | `1800` | Таймаут прогона, секунды |
| `allow_api_key` | `false` | `false` — убирать `ANTHROPIC_API_KEY` из окружения, чтобы работала подписка |
| `extra_args` | `[]` | Дополнительные аргументы командной строки |

### `[agents.codex]` — Codex CLI (`codex exec`)

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `bin` | `"codex"` | Имя, путь или argv; для npm-установки H0lon сам находит нативный `codex.exe` |
| `model_strong` | `""` | Модель сильного уровня; пусто — модель из `~/.codex/config.toml` |
| `model_light` | `""` | То же для лёгкого уровня |
| `effort_strong` | `"high"` | `model_reasoning_effort` для сильного уровня |
| `effort_light` | `"low"` | То же для лёгкого уровня |
| `sandbox` | `"workspace-write"` | Песочница: `read-only`, `workspace-write`, `danger-full-access` |
| `timeout_s` | `1800` | Таймаут прогона, секунды |
| `extra_args` | `[]` | Дополнительные аргументы командной строки |

Все ключи `[agents.*]` действуют с M0.

### `[stages]` — агент по этапам

Значения: `"auto"` (агент `agents.default` с переключением на `agents.fallback`), `"claude"`, `"codex"`. Ключи: `handwriting` (рукописи, M4), `slides_frames` (слайды и кадры, M1/M5), `transcript_fix` (правка транскриптов, M5), `synthesis` (синтез, M2), `coverage` (проверка полноты, M2), `variants` (вариации, M7). Все по умолчанию `"auto"`.

### `[queue]` — очередь сборок (M7)

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `unattended` | `false` | Режим «без присмотра» (ночная сборка) |
| `keep_awake` | `true` | Не давать компьютеру уснуть во время сборки |

### `[compute]` — тяжёлые вычисления (M5–M6)

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `asr` | `"auto"` | Где распознавать речь: `auto`, `local-gpu`, `local-cpu`, `colab`, `api` (позже) |
| `asr_model` | `"large-v3"` | Модель faster-whisper |
| `colab_url` | `""` | Адрес Colab-worker |

### `[render]` — сборка PDF (M0)

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `engine` | `"xelatex"` | Основной движок: `xelatex` или `html` |
| `template` | `"a4-notes"` | Шаблон по умолчанию (см. раздел 2) |
| `fallback_html` | `true` | При неудаче XeLaTeX собрать PDF через HTML и браузер |
| `passes` | `3` | Максимум проходов XeLaTeX |
| `browser` | `"auto"` | Браузер для HTML → PDF: `auto`, `edge`, `chrome`, `chromium` или путь к исполняемому файлу |
| `pandoc` | `""` | Путь к `pandoc`; пусто — `PATH`, затем pandoc из пакета `pypandoc-binary` |
| `xelatex` | `""` | Путь к `xelatex`; пусто — `PATH`, затем стандартные каталоги MiKTeX и TeX Live |
| `main_font` | `"Times New Roman"` | Основной шрифт (нужна кириллица) |
| `sans_font` | `"Arial"` | Шрифт без засечек |
| `mono_font` | `"Consolas"` | Моноширинный шрифт |
| `math_font` | `"Cambria Math"` | Математический OpenType-шрифт |

### `[api]` — API-бэкенд через proxyapi (M8)

| Ключ | По умолчанию | Что задаёт |
|---|---|---|
| `enabled` | `false` | Включить API-бэкенд |
| `base_url` | `"https://api.proxyapi.ru/v1"` | Адрес API |

Ключи API в конфиг не пишутся: их хранение будет отдельным механизмом (PRD §12.4).

## 2. Свои шаблоны PDF

Шаблон — это папка с тремя файлами:

| Файл | Назначение |
|---|---|
| `template.tex` | Шаблон Pandoc для LaTeX (переменные вида `$body$`, `$title$`, `$if(toc)$…$endif$`) |
| `meta.yaml` | Описание шаблона: `title`, `latex_packages`, `html_css` |
| `style.css` | Стили для запасного рендера HTML → PDF |

H0lon ищет шаблон по имени сначала в `<state_dir>/templates/<имя>/` (ваши шаблоны), затем во встроенных `h0lon/render/templates/<имя>/`. Свой шаблон с именем встроенного заменяет встроенный.

`meta.yaml`:

```yaml
title: "A4, широкие поля"     # человекочитаемое название
latex_packages:               # файлы .sty, которые нужны шаблону — по ним doctor проверяет TeX
  - fontspec.sty
  - unicode-math.sty
  - polyglossia.sty
  - mdframed.sty
html_css: style.css           # стиль для HTML-рендера, путь относительно папки шаблона
```

### Свой шаблон на основе встроенного `a4-notes`

Windows (PowerShell, из папки H0lon):

```powershell
$dst = "$env:LOCALAPPDATA\H0lon\templates\my-notes"
New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
Copy-Item -Recurse h0lon\render\templates\a4-notes $dst
notepad "$dst\meta.yaml"
```

macOS / Linux: скопируйте `h0lon/render/templates/a4-notes` в `~/Library/Application Support/H0lon/templates/my-notes` или `~/.local/share/H0lon/templates/my-notes` соответственно. Если вы задали `general.state_dir`, шаблоны лежат в `<state_dir>/templates/`.

Дальше:

1. Поменяйте `title` в `meta.yaml`.
2. Правьте `template.tex`: поля (`geometry`), колонтитулы, оформление заголовков и окружений. Места, куда H0lon подставляет шрифты из `[render]` и служебные переменные, оставьте — иначе перестанут работать ключи `main_font` и др.
3. Если добавили `\usepackage{…}`, допишите соответствующий `.sty` в `latex_packages` — тогда `doctor` предупредит, если пакета нет в TeX.
4. Для запасного рендера поправьте `style.css`.
5. Проверьте:

   ```powershell
   uv run h0lon render tests/fixtures/sample_master.md --template my-notes --keep-build
   uv run h0lon render tests/fixtures/sample_master.md --template my-notes --engine html
   ```

6. Чтобы шаблон стал шаблоном по умолчанию: `template = "my-notes"` в `[render]`.

Пользовательские шаблоны не попадают в репозиторий — они лежат в каталоге служебных данных. Если шаблон получился удачным и полезен другим, его можно предложить во встроенные через pull request.

## 3. Шрифты на macOS и Linux

macOS и Linux поддерживаются по принципу best effort: H0lon ищет инструменты в стандартных местах, но основная проверка идёт на Windows 11, а настройки ниже автор на macOS и Linux не проверял. Шрифты по умолчанию — Windows-шрифты, поэтому их обычно надо поменять в `[render]`.

**macOS.** Times New Roman и Arial есть в системе; Consolas и Cambria Math приходят только с Microsoft Office.

```toml
[render]
main_font = "Times New Roman"
sans_font = "Arial"
mono_font = "Menlo"
math_font = "STIX Two Math"
```

TeX — MacTeX (`brew install --cask mactex-no-gui`), H0lon ищет `xelatex` в `/Library/TeX/texbin`. Наличие шрифта проверьте в приложении «Шрифты» (Font Book); если STIX Two Math нет, установите в систему любой другой математический OpenType-шрифт, например Latin Modern Math.

**Linux (Debian/Ubuntu).** Метрически совместимая с Times и Arial замена с кириллицей — Liberation, математический шрифт — Latin Modern Math:

```bash
sudo apt install texlive-xetex texlive-latex-recommended texlive-latex-extra texlive-lang-cyrillic fonts-liberation fonts-lmodern
```

```toml
[render]
main_font = "Liberation Serif"
sans_font = "Liberation Sans"
mono_font = "Liberation Mono"
math_font = "Latin Modern Math"
```

Проверить, что шрифт виден: `fc-list : family | grep -i -E "liberation|latin modern math"`. Если Latin Modern Math не находится по имени, укажите имя файла `math_font = "latinmodern-math.otf"` — XeLaTeX найдёт его в дереве TeX Live. Пакет, в котором лежит недостающий `.sty`, подскажет `apt-file search <файл>.sty`. H0lon ищет `xelatex` в `PATH`, `/usr/bin` и `/usr/local/texlive/<год>/bin/*/`. Запасной HTML-рендер на Linux требует Chromium или Chrome: `browser = "chromium"` или путь к исполняемому файлу. Если браузер падает с сообщением про `msedge-sandbox` или `chrome-sandbox` ("must be owned by root and have mode 4755"), верните вспомогательному файлу песочницы права из документации браузера: `sudo chown root:root /opt/microsoft/msedge/msedge-sandbox && sudo chmod 4755 /opt/microsoft/msedge/msedge-sandbox` (для Chrome — `/opt/google/chrome/chrome-sandbox`). Отключать песочницу флагом `--no-sandbox` H0lon не будет.

На любой ОС для текста подойдёт любой шрифт с кириллицей, для формул — любой математический OpenType-шрифт. `uv run h0lon doctor` проверит, что шрифты из конфига найдены.

## 4. Новый бэкенд агента

Бэкенд — это класс, который запускает конкретный агентный CLI в headless-режиме над одним заданием (TaskBundle) и сообщает, что получилось. Повторы, проверку результата, переключение на запасной агент, `run.json` и журнал расхода берёт на себя раннер — бэкенду этим заниматься не нужно. Интерфейс (полностью — в [ARCHITECTURE.md](ARCHITECTURE.md#агенты-m0--h0lonagents), код — в `h0lon/agents/base.py`):

```python
class AgentBackend(Protocol):
    name: str  # "claude", "codex", …
    inline_system_prompt: bool  # True, если CLI не принимает файл системного промпта

    def resolve(self) -> list[str] | None: ...  # argv-префикс CLI или None, если его нет
    def auth_status(self) -> AuthStatus: ...  # logged_in: bool | None, method, detail
    def invoke(
        self, bundle: TaskBundle, *, tier: Tier, prompt: str, attempt_dir: Path, on_event
    ) -> BackendOutcome: ...
```

Что делает `invoke`:

- запускает CLI с `cwd = bundle.root`, доступом на чтение к `bundle.workspace` и записью только в `bundle.out_dir`; модель и усилие выбирает по `tier` (`light` / `strong`);
- пишет транскрипт попытки в `attempt_dir` (например, `transcript.jsonl`);
- возвращает `BackendOutcome`: код выхода, длительность, `usage` (токены), текст финального ответа и, при ошибке, `error_kind` — `auth`, `rate_limit`, `timeout`, `crash`, `refusal`, `not_found`, `unknown`. От правильной классификации зависит политика раннера: `auth`/`not_found` — сразу запасной агент, `rate_limit` — запасной, а исходный «остывает», `timeout`/`crash` — один повтор. Для разбора текста ошибок есть `classify_error_text()`.

Шаги:

1. Модуль `h0lon/agents/<имя>.py` с классом бэкенда; за образец возьмите `claude.py` или `codex.py`.
2. Регистрация в `get_backend()` (`h0lon/agents/runner.py`).
3. Раздел настроек `[agents.<имя>]`: модель в `h0lon/config.py` (поле в `AgentsConfig`, имя в `AgentName`) и пример в `h0lon/resources/h0lon.example.toml` — копия в корне репозитория должна совпадать с ним байт в байт.
4. Допустимые значения `--backend` в `h0lon/cli.py` и проверка CLI и входа в `h0lon/doctor.py`.
5. Тесты с фейковым CLI в `tests/fixtures/` (без настоящих вызовов) и, при желании, живой тест с маркером `live_agent`.

Обязательные правила:

- процессы — только через `h0lon.procutil.run`, окружение — через `procutil.clean_env()` (убирает переменные родительской сессии Claude Code); поиск исполняемых файлов — через `h0lon.tools`;
- только штатные режимы CLI по подписке; бэкенд не читает, не хранит и не передаёт токены и пароли;
- без интерактивных вопросов: stdin закрыт, всё, что требует ответа человека, запрещено флагами CLI;
- текст источников — данные, а не инструкции.

Бэкенд через API (proxyapi, M8) появится как отдельный тип `LLMBackend` — см. PRD §12.

## 5. Разработка

```powershell
uv sync                       # окружение вместе с dev-зависимостями (pytest, ruff)
uv run ruff check .           # линтер
uv run ruff format .          # форматирование
uv run ruff format --check .  # то же, что проверяет CI: ничего не меняет, падает на неформатированном
uv run pytest                 # тесты
```

`ruff format` форматирует и блоки кода с пометкой `python` внутри Markdown-файлов (`README.md`, `docs/*.md`), поэтому примеры кода в документации должны быть отформатированы так же, как сам код, — иначе `ruff format --check .` в CI упадёт.

Тесты не вызывают настоящих агентов и сеть. Тесты, которым нужен внешний инструмент, помечены маркерами и пропускаются, если инструмента нет:

| Маркер | Что нужно | Когда пропускается |
|---|---|---|
| `needs_xelatex` | рабочий XeLaTeX | XeLaTeX не найден |
| `needs_browser` | Edge, Chrome или Chromium | браузер не найден |
| `live_agent` | настоящий `claude` / `codex` с входом | всегда, если не задано `H0LON_LIVE_AGENT=1` |

Живые прогоны агентов (тратят лимит подписки):

```powershell
$env:H0LON_LIVE_AGENT = "1"
uv run pytest -m live_agent
Remove-Item Env:H0LON_LIVE_AGENT
```

Выборки маркеров: `uv run pytest -m "not needs_xelatex"`, `uv run pytest -m needs_browser`.

CI (GitHub Actions, `.github/workflows/ci.yml`) на каждый push и pull request запускает `ruff check` и `ruff format --check`, затем `pytest` на Windows и Ubuntu с Python 3.11 и 3.13. TeX в CI не ставится, поэтому тесты `needs_xelatex` там пропускаются; Pandoc приходит из `pypandoc-binary`.

Соглашения:

- код, идентификаторы и комментарии — на английском; сообщения пользователю, документация и промпты — на русском;
- внешние процессы — только через `h0lon.procutil.run`, поиск инструментов — через `h0lon.tools`;
- промпты — файлы `h0lon/prompts/<имя>@<версия>.md`; при любом изменении текста повышайте версию в имени файла;
- ничего личного в репозитории: пути, модели и ключи — только в `h0lon.toml` (он в `.gitignore`), фикстуры — синтетические;
- концы строк — LF (задано в `.gitattributes`), кроме `.ps1`/`.cmd`/`.bat`;
- заметные изменения записывайте в раздел `[Unreleased]` файла [CHANGELOG.md](../CHANGELOG.md).
