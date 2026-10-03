"""body.md → blocks and source.md (h0lon/extract/blocks.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from h0lon import tools
from h0lon.extract import blocks as bl

BODY_WITH_LOCATIONS = r"""## [[P1:p1]] Страница 1

### Случайные величины

Текст с формулой $x^2$ и **жирным** словом.

::: {.definition title="Случайная величина"}
Отображение $X:\Omega\to\mathbb R$ называется случайной величиной.
:::

$$F(x) = P(X \le x)$$

$$\begin{aligned}
a &= b\\
c &= d
\end{aligned}$$.

- первый пункт
- второй пункт

| Свойство | Формула |
|----------|---------|
| Нормировка | $F(+\infty)=1$ |

![График функции распределения](figures/cdf.png){#fig:cdf}

::: figure-description
На графике ступенчатая функция.
:::

``` python
print("код")
```

> Цитата из учебника.

::: {.theorem title="Чебышёв"}
Для $\varepsilon>0$ верно неравенство.
:::

::: proof
Очевидно.
:::

::: uncertain
Страница не распознана агентом.
:::

## [[P1:p2]]

Абзац второй страницы со ссылкой [[P1:p1]] на первую.

## [[S1:s12]] Слайд 12. Итоги

Последний абзац.
"""

EXPECTED_TYPES = [
    ("heading", "P1:p1"),
    ("paragraph", "P1:p1"),
    ("definition", "P1:p1"),
    ("formula", "P1:p1"),
    ("formula", "P1:p1"),
    ("list", "P1:p1"),
    ("table", "P1:p1"),
    ("figure", "P1:p1"),
    ("figure", "P1:p1"),
    ("code", "P1:p1"),
    ("quote", "P1:p1"),
    ("theorem", "P1:p1"),
    ("proof", "P1:p1"),
    ("uncertain", "P1:p1"),
    ("paragraph", "P1:p2"),
    ("paragraph", "S1:s12"),
]


@pytest.fixture(scope="module")
def pandoc() -> Path:
    path = tools.find_pandoc()
    if path is None:
        pytest.skip("Pandoc не найден")
    return path


def build(text: str, pandoc: Path, source_id: str = "P1") -> bl.SourceDoc:
    return bl.build_source_doc(text, source_id, pandoc=pandoc)


@pytest.fixture(scope="module")
def located(pandoc: Path) -> bl.SourceDoc:
    """BODY_WITH_LOCATIONS built once (each Pandoc call costs ~0.5 s on Windows)."""
    return build(BODY_WITH_LOCATIONS, pandoc)


def without_comments(doc: dict) -> list:
    return [b for b in bl.strip_block_comments(doc)["blocks"]]


def test_all_block_types_and_location_anchors(pandoc: Path, located: bl.SourceDoc) -> None:
    doc = located
    assert [(b.type, b.anchor) for b in doc.blocks] == EXPECTED_TYPES
    assert [b.id for b in doc.blocks][:3] == ["P1.b001", "P1.b002", "P1.b003"]
    assert doc.anchor_mode == "location"
    assert all(b.source == "P1" for b in doc.blocks)

    by_type = {b.type: b for b in doc.blocks}
    assert by_type["heading"].title == "Случайные величины"
    assert by_type["definition"].title == "Случайная величина"
    assert by_type["theorem"].title == "Чебышёв"
    assert by_type["paragraph"].title is None
    # Plain text keeps formulas as LaTeX and drops Markdown markup.
    assert doc.blocks[1].text == "Текст с формулой $x^2$ и жирным словом."
    assert doc.blocks[3].text == r"$$F(x) = P(X \le x)$$"
    assert doc.blocks[5].text == "первый пункт\nвторой пункт"
    assert doc.blocks[6].text == "Свойство | Формула\nНормировка | $F(+\\infty)=1$"
    assert doc.blocks[7].text == "График функции распределения"
    # md is the block as Pandoc Markdown.
    assert doc.blocks[2].md.startswith('::: {.definition title="Случайная величина"}')
    assert doc.blocks[6].md.splitlines()[0].startswith("| Свойство")
    assert doc.blocks[9].md.startswith("``` python")
    # Inline anchors stay anchors (not escaped) in text and Markdown.
    assert doc.blocks[14].md == "Абзац второй страницы со ссылкой [[P1:p1]] на первую."


def test_location_headings_are_not_blocks(pandoc: Path, located: bl.SourceDoc) -> None:
    doc = located
    locations = [(h.anchor, h.text) for h in doc.headings if h.location]
    assert locations == [("P1:p1", "Страница 1"), ("P1:p2", ""), ("S1:s12", "Слайд 12. Итоги")]
    assert all(h.block_id is None for h in doc.headings if h.location)
    # Kept in the body unescaped, without a block comment in front.
    assert "## [[P1:p1]] Страница 1\n\n<!-- P1.b001 heading -->" in doc.body
    assert "\n## [[P1:p2]]\n" in doc.body
    assert "## [[S1:s12]] Слайд 12. Итоги" in doc.body
    assert "\\[\\[" not in doc.body


def test_block_comments_in_body(pandoc: Path, located: bl.SourceDoc) -> None:
    doc = located
    comments = [line for line in doc.body.splitlines() if line.startswith("<!-- P1.b")]
    assert comments == [f"<!-- {b.id} {b.type} -->" for b in doc.blocks]
    # Each comment is followed by a blank line and then the block's Markdown.
    for b in doc.blocks:
        assert f"<!-- {b.id} {b.type} -->\n\n{b.md}\n" in doc.body
    assert "h0lon-" not in doc.body  # internal markers never leak


def test_idempotent_rebuild(pandoc: Path, located: bl.SourceDoc) -> None:
    first = located
    second = build(first.body, pandoc)
    assert second.body == first.body
    assert [b.to_dict() for b in second.blocks] == [b.to_dict() for b in first.blocks]
    # A whole source.md (front matter + body) gives the same result.
    source_md = bl.compose_source_md({"id": "P1", "title": "Тест"}, first.body)
    third = build(source_md, pandoc)
    assert third.body == first.body


def test_stale_comments_are_replaced(pandoc: Path) -> None:
    body = (
        "<!-- X9.b042 paragraph -->\n\nПервый абзац.\n\n"
        "<!-- P1.b001 heading -->\n\n<!-- P1.b007 definition -->\n\nВторой абзац.\n\n"
        "<!-- обычный комментарий -->\n\nТретий.\n"
    )
    doc = build(body, pandoc)
    assert [b.id for b in doc.blocks] == ["P1.b001", "P1.b002", "P1.b003"]
    assert doc.body.count("<!-- P1.b") == 3
    assert "X9.b042" not in doc.body
    assert "<!-- обычный комментарий -->" in doc.body  # other comments are content


def test_round_trip_preserves_content(pandoc: Path, located: bl.SourceDoc) -> None:
    doc = located
    original = bl.normalize_ast(bl.read_markdown_ast(BODY_WITH_LOCATIONS, pandoc))
    rebuilt = bl.normalize_ast(bl.read_markdown_ast(doc.body, pandoc))
    assert without_comments(rebuilt) == without_comments(original)
    # Formulas, div attributes, tables and anchors are all still there.
    for fragment in (
        r"$$F(x) = P(X \le x)$$",
        "\\begin{aligned}\na &= b\\\\\nc &= d\n\\end{aligned}",
        '::: {.definition title="Случайная величина"}',
        '::: {.theorem title="Чебышёв"}',
        "| Нормировка | $F(+\\infty)=1$ |",
        "![График функции распределения](figures/cdf.png){#fig:cdf}",
        "[[P1:p1]] на первую",
    ):
        assert fragment in doc.body


def test_section_anchors_without_location_headings(pandoc: Path) -> None:
    body = (
        "Вступление до первого заголовка.\n\n"
        "# Первый раздел\n\nТекст.\n\n## Подраздел\n\nЕщё текст.\n\n"
        "# Второй раздел\n\nКонец.\n"
    )
    doc = build(body, pandoc, "D1")
    assert doc.anchor_mode == "section"
    assert [(b.type, b.anchor) for b in doc.blocks] == [
        ("paragraph", "D1:§0"),
        ("heading", "D1:§1"),
        ("paragraph", "D1:§1"),
        ("heading", "D1:§1"),
        ("paragraph", "D1:§1"),
        ("heading", "D1:§2"),
        ("paragraph", "D1:§2"),
    ]


def test_title_heading_is_not_a_section(pandoc: Path) -> None:
    # A page title above `##` sections, and an inserted title with class source-title.
    page = "# Заголовок страницы\n\nВступление.\n\n## Раздел А\n\nТекст.\n\n## Раздел Б\n\nТекст.\n"
    doc = build(page, pandoc, "W1")
    assert [b.anchor for b in doc.blocks] == ["W1:§0", "W1:§0", "W1:§1", "W1:§1", "W1:§2", "W1:§2"]
    assert doc.headings[0].is_title
    titled = "# Документ {.source-title}\n\n# Глава 1\n\nТекст.\n\n# Глава 2\n\nТекст.\n"
    doc = build(titled, pandoc, "D2")
    assert [b.anchor for b in doc.blocks] == ["D2:§0", "D2:§1", "D2:§1", "D2:§2", "D2:§2"]
    assert doc.headings[0].is_title and not doc.headings[1].is_title


def test_anchor_before_link_syntax_stays_text(pandoc: Path) -> None:
    body = "Сноска [[P1:p3]](примечание) и определение\n\n[[P1:p4]]: не ссылка.\n"
    doc = build(body, pandoc)
    assert [b.type for b in doc.blocks] == ["paragraph", "paragraph"]
    assert doc.blocks[0].text == "Сноска [[P1:p3]](примечание) и определение"
    assert doc.blocks[1].text == "[[P1:p4]]: не ссылка."
    again = build(doc.body, pandoc)
    assert [b.text for b in again.blocks] == [b.text for b in doc.blocks]


def test_location_anchor_glued_to_text(pandoc: Path) -> None:
    doc = build("## [[P2:p5]]Страница 5\n\nТекст.\n", pandoc, "P2")
    assert doc.headings[0].anchor == "P2:p5" and doc.headings[0].location
    assert doc.blocks[0].anchor == "P2:p5"


def test_figures_and_unknown_divs(pandoc: Path) -> None:
    body = (
        "![](figures/a.png)\n\n"
        "Текст и картинка ![x](figures/b.png) внутри.\n\n"
        '::: {.figure-description title="Схема"}\nОписание схемы.\n:::\n\n'
        "::: note\nПросто заметка.\n:::\n\n"
        "::: wrapper\n- один\n- два\n:::\n\n"
        "------\n\n"
        "::: {.admin}\nЭкзамен 20 декабря.\n:::\n"
    )
    doc = build(body, pandoc)
    assert [b.type for b in doc.blocks] == [
        "figure",
        "paragraph",
        "figure",
        "paragraph",
        "list",
        "admin",
    ]
    assert doc.blocks[2].title == "Схема"


def test_cyrillic_and_math_text(pandoc: Path) -> None:
    doc = build("Ёжик и щётка: $\\sum_{i=1}^n x_i$ — сумма.\n", pandoc)
    assert doc.blocks[0].text == "Ёжик и щётка: $\\sum_{i=1}^n x_i$ — сумма."
    assert bl.cyrillic_ratio([doc.blocks[0].text]) == 1.0
    assert bl.cyrillic_ratio(["English text only"]) == 0.0
    assert bl.cyrillic_ratio(["$x$ 123"]) is None


def test_empty_document(pandoc: Path) -> None:
    doc = build("\n\n<!-- только комментарий -->\n", pandoc)
    assert doc.blocks == []


def test_footnotes_stay_with_their_block(pandoc: Path) -> None:
    body = (
        "Первый абзац со сноской[^a].\n\n[^a]: Текст сноски.\n\n"
        "Второй абзац.\n\n::: remark\nЗамечание со сноской[^b].\n:::\n\n[^b]: Вторая сноска.\n"
    )
    doc = build(body, pandoc)
    first, second, remark = doc.blocks
    assert first.md == "Первый абзац со сноской[^1].\n\n[^1]: Текст сноски."
    assert first.text == "Первый абзац со сноской (Текст сноски.)."
    assert second.md == "Второй абзац."
    assert remark.type == "remark" and "[^2]: Вторая сноска." in remark.md
    again = build(doc.body, pandoc)
    assert again.body == doc.body
    assert [b.md for b in again.blocks] == [b.md for b in doc.blocks]


def test_yaml_like_fragments_in_a_body_are_text(pandoc: Path) -> None:
    doc = build("Абзац.\n\n---\nключ: значение\n---\n\n% не заголовок\n\nЕщё.\n", pandoc)
    texts = "\n".join(b.text for b in doc.blocks)
    assert "ключ: значение" in texts and "% не заголовок" in texts and "Ещё." in texts


def test_parse_front_matter() -> None:
    fm = bl.parse_front_matter("---\ntitle: Лекция\n---\n\nТело.\n")
    assert fm.present and fm.data == {"title": "Лекция"} and fm.body == "\nТело.\n"
    broken = bl.parse_front_matter("---\ntitle: Лекция 3: величины\n---\nТело.\n")
    assert broken.present and broken.data == {} and broken.body == "Тело.\n"
    assert broken.error and "строка 2" in broken.error and "title:" in broken.raw
    scalar = bl.parse_front_matter("---\nПросто строка\n---\nТело.\n")
    assert not scalar.present and scalar.body.startswith("---\nПросто строка")
    assert bl.parse_front_matter("Тело.\n").body == "Тело.\n"


def test_compose_and_split_front_matter() -> None:
    front = {"id": "P1", "title": "Лекция: «случайные» величины", "units": {"pages": 2}}
    text = bl.compose_source_md(front, "Тело.\n")
    data, body = bl.split_front_matter(text)
    assert data == front
    assert body.strip() == "Тело."


def test_blocks_jsonl_round_trip(tmp_path: Path, located: bl.SourceDoc) -> None:
    doc = located
    path = tmp_path / "blocks.jsonl"
    bl.atomic_write_text(path, bl.blocks_jsonl(doc.blocks))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(doc.blocks)
    assert json.loads(lines[0])["id"] == "P1.b001"
    assert "Случайные" in lines[0]  # UTF-8, not \u escapes
    assert [b.to_dict() for b in bl.read_blocks_jsonl(path)] == [b.to_dict() for b in doc.blocks]


def test_pandoc_failure_is_readable(tmp_path: Path) -> None:
    missing = tmp_path / "no-pandoc.exe"
    with pytest.raises(bl.BlocksError, match="Pandoc"):
        bl.build_source_doc("Текст.\n", "P1", pandoc=missing)
