# Установка H0lon на Windows 11

Инструкция для человека, который ставит H0lon на чистую Windows 11 впервые и без помощи автора. Займёт 30–60 минут, большая часть — скачивание. Нужны интернет и несколько гигабайт свободного места. Почти всё ставится с правами обычного пользователя; исключение — Node.js: его установщик ставит программу для всех пользователей и запрашивает права администратора (окно контроля учётных записей, UAC). Установщик Git тоже может показать такой запрос. Если прав администратора нет, Codex можно поставить без Node.js — см. [раздел 4](#codex-cli).

В конце у вас будут:

- H0lon в папке `H0lon` с собственным окружением Python;
- XeLaTeX (MiKTeX) для сборки PDF;
- хотя бы один агент — Claude Code и/или Codex CLI — с входом по подписке.

> **Подписки и учётные данные.** Версия 1 работает только по подпискам — через штатные headless-режимы `claude -p` (Claude Code, план Pro или Max) и `codex exec` (Codex CLI, вход через ChatGPT). Вход выполняется в самих этих программах, H0lon **не хранит и не передаёт** логины, пароли и токены: он лишь запускает установленные у вас CLI. По умолчанию H0lon убирает `ANTHROPIC_API_KEY` из окружения дочернего `claude`, чтобы прогоны шли по подписке, а не списывались с API-счёта (ключ `agents.claude.allow_api_key`). Материалы тем при этом уходят в Anthropic или OpenAI так же, как при обычной работе с этими агентами. Бэкенд через API запланирован на этап M8.

## Содержание

1. [Терминал и winget](#1-терминал-и-winget)
2. [uv, Git, Node.js](#2-uv-git-nodejs)
3. [MiKTeX (XeLaTeX)](#3-miktex-xelatex)
4. [Агенты: Claude Code и Codex CLI](#4-агенты-claude-code-и-codex-cli)
5. [Необязательно: ffmpeg и yt-dlp](#5-необязательно-ffmpeg-и-yt-dlp)
6. [H0lon](#6-h0lon)
7. [Где что лежит](#7-где-что-лежит)
8. [Обновление](#8-обновление)
9. [Если что-то не так](#9-если-что-то-не-так)

## 1. Терминал и winget

Все команды ниже выполняются в **PowerShell**. Откройте его: правый клик по «Пуску» → «Терминал» (Windows Terminal). Строка приглашения выглядит как `PS C:\Users\Имя>`. Копируйте команды по одной.

`winget` (менеджер пакетов Windows) встроен в Windows 11. Проверьте:

```powershell
winget --version
```

Если команда не найдена — установите или обновите «Установщик приложений» (App Installer) из Microsoft Store. При первом запуске winget может попросить принять соглашения источника пакетов — прочитайте и ответьте `Y`, это обычный шаг.

**Важно:** после каждой установки через winget **закройте и откройте терминал заново** — только так он увидит новые программы в `PATH`.

## 2. uv, Git, Node.js

```powershell
winget install --id astral-sh.uv -e
winget install --id Git.Git -e
winget install --id OpenJS.NodeJS.LTS -e
```

- **uv** — менеджер окружений Python. Отдельно ставить Python не нужно: `uv` сам скачает подходящую версию (проект закреплён на 3.13, поддерживается 3.11+).
- **Git** — чтобы скачать H0lon; кроме того, каждая тема хранится как git-репозиторий. Claude Code тоже использует Git Bash из этого пакета, если он установлен.
- **Node.js LTS** — нужен только для установки Codex CLI через `npm`. Установщик Node.js попросит права администратора; если их нет, пропустите эту команду и поставьте Codex без Node.js (раздел 4).

Откройте новый терминал и проверьте:

```powershell
uv --version
git --version
node --version
npm --version
```

Если `npm` пишет, что «выполнение сценариев отключено в этой системе» (running scripts is disabled), см. раздел [«Политика выполнения PowerShell»](#политика-выполнения-powershell-npm-и-codex-не-запускаются).

**Имя и почта для Git.** Только что установленный Git не знает, от чьего имени делать коммиты, а H0lon делает коммит в репозитории каждой темы (`h0lon new`). Задайте их один раз:

```powershell
git config --global user.name "Ваше имя"
git config --global user.email "you@example.com"
```

Подойдут любые имя и почта; они видны в истории коммитов, поэтому для репозиториев, которые вы будете публиковать, выбирайте то, что не жалко показать. Без них `doctor` покажет предупреждение «git: имя и почта», а `h0lon new` создаст тему без первого коммита.

## 3. MiKTeX (XeLaTeX)

MiKTeX — дистрибутив TeX для Windows, из него H0lon берёт XeLaTeX.

```powershell
winget install --id MiKTeX.MiKTeX -e --scope user
```

`--scope user` ставит MiKTeX только для вас, без прав администратора (в `%LOCALAPPDATA%\Programs\MiKTeX`). Установка «для всех пользователей» тоже подойдёт — H0lon ищет `xelatex` в `PATH`, в `%LOCALAPPDATA%\Programs\MiKTeX` и в `%ProgramFiles%\MiKTeX`, но тогда команды обновления ниже нужно выполнять от администратора с ключом `--admin` (`miktex --admin packages update`).

### Первый запуск и обновление

1. Откройте «Пуск» → **MiKTeX Console**. Если консоль предлагает проверить обновления — согласитесь.
2. Вкладка **Updates** → **Check for updates** → **Update now**. Дождитесь окончания и закройте консоль.

То же самое из терминала (откройте новый терминал после установки):

```powershell
miktex packages update-package-database
miktex packages update
```

Если `miktex` не найден — откройте новый терминал или выполните команды по полному пути `& "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\miktex.exe" packages update`.

### Недостающие пакеты LaTeX

Обычно MiKTeX скачивает недостающие пакеты «на лету», но **H0lon запускает XeLaTeX с ключом `--disable-installer`**: во время сборки ничего не скачивается и не всплывает окон, сборка либо проходит, либо честно падает с ошибкой вида ``File `mdframed.sty' not found``. Поэтому нужные шаблону пакеты ставятся один раз заранее.

1. После установки H0lon (раздел 6) выполните `uv run h0lon doctor` — он покажет, каких файлов `.sty` не хватает шаблону.
2. Поставьте пакеты MiKTeX. Имя пакета почти всегда совпадает с именем файла без `.sty`:

   ```powershell
   miktex packages install mdframed
   miktex packages install tcolorbox
   ```

   Если пакет с таким именем не находится, найдите его в MiKTeX Console → **Packages** (поиск по названию) и установите оттуда.
3. Повторите `uv run h0lon doctor` — недостающих файлов быть не должно.

Запасной способ, если пакетов много: разрешите MiKTeX установку «на лету» (MiKTeX Console → **Settings** → **Always install missing packages on-the-fly**) и один раз соберите документ вручную, без `--disable-installer`:

```powershell
uv run h0lon render tests/fixtures/sample_master.md --engine xelatex --keep-build --json
# в JSON-отчёте поле "tex" — путь к .tex-файлу; соберите его вручную:
cd <каталог из поля "build_dir">
xelatex -interaction=nonstopmode <имя файла>.tex
```

MiKTeX докачает всё, чего не хватает: пакеты загружаются в начале сборки, поэтому они установятся, даже если ручная сборка потом споткнётся, например, о пути к картинкам. Верните настройку «Ask me» или «Never» по желанию. После этого `h0lon render` будет работать и с `--disable-installer`.

Первая сборка XeLaTeX может идти несколько минут — MiKTeX строит кэш шрифтов. Это происходит один раз.

## 4. Агенты: Claude Code и Codex CLI

Нужен хотя бы один агент. Лучше оба: по умолчанию работает Claude Code, а Codex подхватывает прогоны при лимитах и сбоях.

### Claude Code

```powershell
irm https://claude.ai/install.ps1 | iex
```

Откройте новый терминал и проверьте `claude --version`. Установщик кладёт `claude.exe` в `%USERPROFILE%\.local\bin`; если команда не найдена, добавьте эту папку в `PATH` (установщик подсказывает, как). H0lon находит `claude.exe` в этой папке и без `PATH`.

Вариант через winget (без автообновлений): `winget install --id Anthropic.ClaudeCode -e`.

Вход по подписке **Pro или Max** (бесплатный план claude.ai не включает Claude Code):

```powershell
claude auth login
claude auth status --text
```

`claude auth login` откроет браузер — войдите в свой аккаунт claude.ai. `claude auth status` должен показать, что вход выполнен по подписке Claude.

### Codex CLI

Нужен Node.js из раздела 2 (без него — вариант ниже, «Без прав администратора»).

```powershell
npm install -g @openai/codex
codex login
codex login status
```

`codex login` откроет браузер — выберите **Sign in with ChatGPT** и войдите аккаунтом с подходящим планом ChatGPT (Plus, Pro, Business, Edu или Enterprise). `codex login status` должен подтвердить вход.

H0lon сам находит нативный `codex.exe` внутри npm-установки и вызывает его напрямую — ничего настраивать не нужно.

**Без прав администратора (без Node.js).** В winget есть нативная сборка Codex, которая ставится для текущего пользователя:

```powershell
winget install --id OpenAI.Codex -e
```

Откройте новый терминал и выполните те же `codex login` и `codex login status`. Этот способ установки автор с H0lon не проверял; если `uv run h0lon doctor` не находит Codex, укажите полный путь к `codex.exe` в ключе `bin` раздела `[agents.codex]` файла `h0lon.toml`.

### Если агент только один

- Только Claude: ничего менять не нужно; чтобы H0lon не пытался переключаться на Codex, в `h0lon.toml` задайте `fallback = ""` в разделе `[agents]`.
- Только Codex: в разделе `[agents]` задайте `default = "codex"` и `fallback = ""`.

Как открыть `h0lon.toml` — в разделе 6.

## 5. Необязательно: ffmpeg и yt-dlp

Понадобятся для видео- и аудиолекций (этап M5). Сейчас H0lon их не использует, `doctor` лишь покажет, найдены ли они.

```powershell
winget install --id Gyan.FFmpeg -e
winget install --id yt-dlp.yt-dlp -e
```

## 6. H0lon

Выберите папку для кода (например, домашнюю) и скачайте репозиторий:

```powershell
cd $HOME
git clone https://github.com/Filodot/H0lon.git
cd H0lon
```

Дальше по шагам.

**Окружение.** Создаёт `.venv`, при необходимости скачивает Python и ставит зависимости — в том числе Pandoc (он приходит внутри пакета `pypandoc-binary`, отдельно ставить не нужно):

```powershell
uv sync
```

**Конфигурация.** Записывает `h0lon.toml` из примера в `%APPDATA%\H0lon\h0lon.toml` и создаёт каталог рабочих областей `%USERPROFILE%\Konspekty` и каталог служебных данных `%LOCALAPPDATA%\H0lon`:

```powershell
uv run h0lon init
```

Если файл уже есть, команда ничего не перезапишет и завершится с кодом 1; `--force` — перезаписать, `--path <файл>` — записать в другое место. Открыть конфиг: `notepad $env:APPDATA\H0lon\h0lon.toml`. Все ключи описаны в [ADAPT.md](ADAPT.md).

**Проверка окружения.** Таблица статусов: агенты и вход в них, Pandoc, XeLaTeX и пакеты шаблона, шрифты, ffmpeg, yt-dlp, CUDA — с подсказками, что установить. Код возврата 1, если не хватает обязательного:

```powershell
uv run h0lon doctor
```

`--no-auth` пропускает проверку входа агентов (быстрее и без сети), `--json` выводит результат в JSON.

**Тестовый PDF.** Собирает PDF из тестового мастер-конспекта: Pandoc → XeLaTeX с A4-шаблоном, при неудаче — запасной путь HTML → PDF через Edge/Chrome:

```powershell
uv run h0lon render tests/fixtures/sample_master.md
```

PDF появляется рядом с исходником; чтобы не складывать результат в папку репозитория, добавьте `--out <каталог>`. Откройте PDF и проверьте: кириллица, формулы, рамки определений и теорем, оглавление, закладки. Полезные ключи: `--engine html` — проверить запасной рендер, `--keep-build` — сохранить `.tex` и `.log`, `--clean` — убрать якоря источников `[[…]]`, `--json` — отчёт в JSON.

**Проверка агента.** Создаёт небольшое задание (TaskBundle), запускает агента по умолчанию в headless-режиме, проверяет результат и пишет `run.json` с расходом токенов:

```powershell
uv run h0lon agent-test
```

Занимает 1–3 минуты и тратит немного лимита подписки. Проверить конкретного агента без переключения на запасной: `uv run h0lon agent-test --backend codex --no-fallback` (или `--backend claude`). Можно дать своё задание строкой: `uv run h0lon agent-test "Перечисли три свойства определителя"`.

Если `doctor` не показывает обязательных проблем, PDF собрался, а `agent-test` завершился успешно — установка закончена.

## 7. Где что лежит

| Что | Где (Windows) |
|---|---|
| Код H0lon | папка, куда вы сделали `git clone` |
| Конфигурация | `%APPDATA%\H0lon\h0lon.toml` (или `h0lon.toml` в текущей папке, см. ниже) |
| Рабочие области тем | `%USERPROFILE%\Konspekty` (ключ `general.workspaces`) |
| Служебные данные: журнал расхода `usage.jsonl`, свои шаблоны, тестовые прогоны | `%LOCALAPPDATA%\H0lon` (ключ `general.state_dir`) |
| Вход Claude Code | хранит сам Claude Code в профиле пользователя |
| Вход Codex | хранит сам Codex в `%USERPROFILE%\.codex` |

Порядок поиска конфигурации: ключ `--config` (ставится **перед** командой: `uv run h0lon --config D:\my.toml doctor`) → переменная `H0LON_CONFIG` → `h0lon.toml` в текущей папке → `%APPDATA%\H0lon\h0lon.toml`. Файл `h0lon.toml` в папке репозитория игнорируется git — личные пути туда не попадут.

## 8. Обновление

```powershell
cd <папка H0lon>
git pull
uv sync
```

- Claude Code, установленный через `install.ps1`, обновляется сам; установленный через winget — `winget upgrade --id Anthropic.ClaudeCode -e`.
- Codex: `npm install -g @openai/codex@latest` (установленный через winget — `winget upgrade --id OpenAI.Codex -e`).
- MiKTeX: MiKTeX Console → Updates (или `miktex packages update`). `winget upgrade` для MiKTeX не работает — он обновляется только своей консолью.
- uv: `winget upgrade --id astral-sh.uv -e`.

## 9. Если что-то не так

Начинайте с `uv run h0lon doctor`: у каждой проблемы там есть подсказка.

### Команда не найдена сразу после установки

Закройте и откройте терминал. Если не помогло — выйдите из Windows и войдите снова: некоторые установщики меняют `PATH` только для новых сеансов.

### Pandoc не найден

Отдельно ставить Pandoc не нужно: он лежит внутри пакета `pypandoc-binary` и появляется после `uv sync`. Если в `PATH` есть свой `pandoc`, H0lon возьмёт его. Указать конкретный файл: `pandoc = "C:/путь/pandoc.exe"` в разделе `[render]`. Если `doctor` всё равно не видит Pandoc — повторите `uv sync` в папке H0lon.

### XeLaTeX не найден

Проверьте, что MiKTeX установлен (раздел 3) и терминал открыт заново. Если MiKTeX стоит в нестандартной папке, укажите путь явно: `xelatex = "D:/MiKTeX/miktex/bin/x64/xelatex.exe"` в разделе `[render]`.

### Не хватает пакета LaTeX

Сообщение ``File `….sty' not found`` в отчёте `render` или строка «не хватает …» в `doctor`. Поставьте пакет командой `miktex packages install <имя>` или через MiKTeX Console → Packages — подробно в разделе [«Недостающие пакеты LaTeX»](#недостающие-пакеты-latex). Полный лог сборки сохраняет `uv run h0lon render <файл> --keep-build`.

### Шрифты

Шаблон по умолчанию использует Times New Roman (основной), Arial (без засечек), Consolas (моноширинный) и Cambria Math (формулы) — все они есть в Windows 11. Сменить шрифты можно в разделе `[render]` файла `h0lon.toml`:

```toml
[render]
main_font = "Times New Roman"
sans_font = "Arial"
mono_font = "Consolas"
math_font = "Cambria Math"
```

Указывайте имя семейства шрифта (как в «Параметры → Персонализация → Шрифты»). Шрифт для текста должен содержать кириллицу, для формул — быть математическим OpenType-шрифтом (Cambria Math, STIX Two Math, Latin Modern Math и т. п.). Новый шрифт ставьте через правый клик → **«Установить для всех пользователей»**: шрифты, установленные только для себя, XeLaTeX может не увидеть. После установки шрифта обновите кэш: `miktex-fc-cache -f`.

### git: не заданы имя и почта

Предупреждение `doctor` «git: имя и почта — не заданы user.name, user.email» или сообщение `h0lon new` «первый коммит не сделан: не настроена git-идентичность». Задайте имя и почту (раздел 2):

```powershell
git config --global user.name "Ваше имя"
git config --global user.email "you@example.com"
```

Для темы, которая уже создана без коммита, перейдите в её каталог и сделайте первый коммит вручную: `git add -A`, затем `git commit -m "Тема создана"`.

### Claude не авторизован, прогоны уходят в Codex

Если `claude` не установлен или в нём не выполнен вход, H0lon сразу переключает прогон на запасного агента (по умолчанию Codex) — в отчёте `agent-test` видно, какой агент выполнил прогон, и список попыток. Так же он поступает при исчерпании лимита подписки: исходный агент «остывает» до конца запуска. Исправить: `claude auth login`, затем `claude auth status --text`. Если Claude вам не нужен — задайте `default = "codex"` в `[agents]` (раздел 4).

Для Codex аналогично: `codex login` и `codex login status`.

### Политика выполнения PowerShell: npm и codex не запускаются

Сообщение «Невозможно загрузить файл …\npm.ps1, так как выполнение сценариев отключено в этой системе». Два варианта:

- вызывать `.cmd`-версии, которым политика не мешает: `npm.cmd install -g @openai/codex`, `codex.cmd login`, `codex.cmd login status`;
- разрешить локальные скрипты для своего пользователя: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (это изменение настройки безопасности PowerShell — решайте сами).

H0lon это не касается: он запускает `codex.exe` напрямую.

### Кракозябры вместо кириллицы в консоли

H0lon выводит UTF-8, когда вывод перенаправлен в файл или другую программу, а в интерактивной консоли заменяет символы, которые она не может показать. Если русский текст всё равно искажён:

- пользуйтесь Windows Terminal, а не старым окном `cmd.exe`;
- переключите консоль на UTF-8 на время сеанса: `chcp 65001` или `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`.

**Сохранение вывода в файл.** Во встроенном Windows PowerShell 5.1 (его по умолчанию открывает «Терминал» в Windows 11) обычное перенаправление `uv run h0lon doctor --json > doctor.json` портит кириллицу: PowerShell перекодирует вывод через кодовую страницу консоли (cp866) и записывает файл в UTF-16. `chcp 65001` на это не влияет — он меняет только то, как текст показывается в окне. Перенаправляйте через `cmd`, тогда байты UTF-8 попадут в файл без изменений:

```powershell
cmd /c "uv run h0lon doctor --json > doctor.json"
```

Эта команда работает и в PowerShell 7; в окне `cmd.exe` достаточно `uv run h0lon doctor --json > doctor.json`. Другой способ для Windows PowerShell 5.1 — сначала выполнить `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`, а потом перенаправить вывод: кириллица сохранится, но файл будет в UTF-16. В PowerShell 7.4 и новее обычное `>` сохраняет UTF-8 без искажений.

### Кириллица или пробелы в путях

Каталоги тем H0lon называет латиницей (ASCII-слаги), но путь к профилю пользователя может содержать кириллицу. Если XeLaTeX или другой инструмент падает на пути, перенесите рабочие области и служебные данные в каталог с латинским путём, например `C:\H0lonData`: ключи `general.workspaces` и `general.state_dir`.

### Переопределение настроек без правки файла

Любой ключ конфигурации переопределяется переменной окружения `H0LON_<РАЗДЕЛ>__<КЛЮЧ>` (два подчёркивания между уровнями). Переменные важнее файла, файл важнее значений по умолчанию. Примеры для текущего сеанса PowerShell:

```powershell
$env:H0LON_AGENTS__DEFAULT = "codex"                # агент по умолчанию
$env:H0LON_AGENTS__CLAUDE__MODEL_STRONG = "sonnet"  # вложенный раздел [agents.claude]
$env:H0LON_RENDER__MAIN_FONT = "Cambria"            # основной шрифт
$env:H0LON_CONFIG = "D:\configs\h0lon.toml"         # другой файл конфигурации
uv run h0lon doctor
```

Убрать переменную: `Remove-Item Env:H0LON_AGENTS__DEFAULT`.

### Всё равно не работает

Сохраните вывод `doctor` и отчёт `render` (или `agent-test`) в файлы — через `cmd /c`, чтобы не испортить кириллицу (см. [«Кракозябры вместо кириллицы в консоли»](#кракозябры-вместо-кириллицы-в-консоли)):

```powershell
cmd /c "uv run h0lon doctor --json > doctor.json"
cmd /c "uv run h0lon render tests/fixtures/sample_master.md --json > render.json"
cmd /c "uv run h0lon agent-test --json > agent-test.json"
```

Затем создайте issue в репозитории [Filodot/H0lon](https://github.com/Filodot/H0lon/issues). Перед отправкой проверьте, что в выводе нет личных путей и данных, которые вы не хотите публиковать.
