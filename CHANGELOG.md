# Журнал изменений

Все заметные изменения проекта записываются в этот файл.

Формат — [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/), версии — [семантическое версионирование](https://semver.org/lang/ru/). Версии промптов указываются в именах файлов (`h0lon/prompts/<имя>@<версия>.md`) и меняются при любом изменении текста.

## [Unreleased]

Этап M0 — каркас. Первый релиз будет `0.1.0`.

### Добавлено

- PRD v1.0 (`docs/PRD.md`) с решениями D1–D16 и описание модулей и контрактов (`docs/ARCHITECTURE.md`).
- Пакет `h0lon` на Python 3.11+ с окружением через `uv`, линтером `ruff` и тестами `pytest` (маркеры `needs_xelatex`, `needs_browser`, `live_agent`).
- Конфигурация `h0lon.toml`: пример `h0lon.example.toml`, поиск файла (`--config` → `H0LON_CONFIG` → `./h0lon.toml` → каталог пользователя), переопределение любого ключа переменной `H0LON_<РАЗДЕЛ>__<КЛЮЧ>`.
- CLI `h0lon`: команды `version`, `init`, `doctor`, `new`, `render`, `agent-test`.
- `h0lon init` — создание `h0lon.toml`, каталога рабочих областей и каталога служебных данных.
- `h0lon doctor` — проверка агентов и входа в них, Pandoc, XeLaTeX и пакетов шаблона, шрифтов, ffmpeg, yt-dlp, CUDA с командами установки; вывод в JSON (`--json`).
- `h0lon new` — рабочая область темы: ASCII-слаг из русского названия, `topic.yaml`, стандартные папки, git-репозиторий на тему.
- Рендер `h0lon render`: Pandoc → XeLaTeX с A4-шаблоном `a4-notes` и Lua-фильтрами, многопроходная компиляция, проверка PDF, запасной рендер HTML → PDF через Edge/Chrome, удаление якорей источников (`--clean`), пользовательские шаблоны в `<state_dir>/templates/`.
- Раннер headless-агентов: TaskBundle (`runs/<id>/`), бэкенды Claude Code (`claude -p`) и Codex CLI (`codex exec`), проверка результата (файлы, заголовки, JSON Schema), повторы с обратной связью, переключение на запасной агент и «остывание» при лимитах, `run.json` и журнал расхода `usage.jsonl`; команда `h0lon agent-test`.
- Загрузка окон подписки Claude (5 часов, 7 дней) из события `rate_limit_event`: пишется в `run.json` и `usage.jsonl`, показывается в сводке `agent-test`; при загрузке от 80 % — предупреждение.
- Досрочная остановка Claude при ошибке входа или исчерпанном окне лимита, без ожидания внутренних повторов CLI.
- Общие слои: запуск процессов в UTF-8 с таймаутом и завершением дерева процессов, очистка окружения от переменных родительской сессии Claude Code; поиск Pandoc (включая `pypandoc-binary`), XeLaTeX (MiKTeX, TeX Live), браузеров и нативного `codex.exe` для npm-установки.
- CI на GitHub Actions: `ruff check` и `ruff format --check`, `pytest` на Windows и Ubuntu с Python 3.11 и 3.13.
- Документация: README, установка на чистую Windows 11 (`docs/INSTALL.md`), адаптация под себя (`docs/ADAPT.md`), этот журнал; нормализация концов строк в `.gitattributes`.

[Unreleased]: https://github.com/Filodot/H0lon/commits/main
