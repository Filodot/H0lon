import re
import unicodedata

import pytest

from h0lon.names import slugify, transliterate

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Теорвер — лекция 3", "teorver-lektsiya-3"),
        ("Щука и ёж", "shchuka-i-ezh"),
        ("Жёлтый хвост цапли", "zheltyy-khvost-tsapli"),
        ("Чай, шарф, юла, яма", "chay-sharf-yula-yama"),
        ("Подъезд и мышь", "podezd-i-mysh"),
        ("Объём ЭВМ", "obem-evm"),
        ("Linear Algebra 101", "linear-algebra-101"),
        ("Crème brûlée à la Möbius", "creme-brulee-a-la-mobius"),
        ("Straße Łódź", "strasse-lodz"),
        ("don't panic", "dont-panic"),
        ("  --Много   пробелов--  ", "mnogo-probelov"),
        ("a___b...c", "a-b-c"),
        ("ТЕОРВЕР", "teorver"),
        ("C:\\Users\\Лекция 03.pdf", "c-users-lektsiya-03-pdf"),
    ],
)
def test_slugify_examples(text: str, expected: str) -> None:
    assert slugify(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "—", "!!!", "ъь", "💡"])
def test_slugify_empty_becomes_untitled(text: str) -> None:
    assert slugify(text) == "untitled"


def test_slugify_cuts_at_word_boundary() -> None:
    text = "Теория вероятностей и математическая статистика"
    full = slugify(text, max_len=200)
    assert full == "teoriya-veroyatnostey-i-matematicheskaya-statistika"
    assert slugify(text, max_len=25) == "teoriya-veroyatnostey-i"
    assert slugify(text, max_len=21) == "teoriya-veroyatnostey"  # exact word end
    assert slugify(text, max_len=22) == "teoriya-veroyatnostey"  # trailing hyphen trimmed
    assert slugify(text, max_len=10) == "teoriya"


def test_slugify_hard_cut_for_single_long_word() -> None:
    assert slugify("a" * 100, max_len=10) == "a" * 10


def test_slugify_max_len_must_be_positive() -> None:
    with pytest.raises(ValueError):
        slugify("x", max_len=0)


@pytest.mark.parametrize(
    "text",
    [
        "Теорвер — лекция 3",
        "Ünïcödé — тест №5 (часть 2/3)",
        "x" * 300,
        "日本語のテキスト",
        "Ё-моё!",
    ],
)
def test_slugify_always_valid(text: str) -> None:
    slug = slugify(text)
    assert SLUG_RE.match(slug), slug
    assert len(slug) <= 60
    assert slugify(slug) == slug  # idempotent


def test_transliterate_keeps_ascii() -> None:
    assert transliterate("Лекция 3: PCA") == "lektsiya 3: pca"


@pytest.mark.parametrize("text", ["Майский", "Йошкар-Ола", "Щука и ёж", "ЁЖИК", "Crème brûlée"])
def test_slugify_ignores_unicode_normal_form(text: str) -> None:
    nfd = unicodedata.normalize("NFD", text)
    assert nfd != text or text.isascii()
    assert slugify(nfd) == slugify(text)


def test_slugify_decomposed_short_i() -> None:
    assert slugify(unicodedata.normalize("NFD", "Майский")) == "mayskiy"
    assert slugify(unicodedata.normalize("NFD", "Йошкар-Ола")) == "yoshkar-ola"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("σ-алгебры", "sigma-algebry"),
        ("χ²-критерий", "chi2-kriteriy"),
        ("β-распад", "beta-raspad"),
        ("λ-исчисление", "lambda-ischislenie"),
        ("ε-δ определение", "epsilon-delta-opredelenie"),
        ("Σ и ς", "sigma-i-sigma"),
        ("ϑ, ϕ, ϵ, µ", "theta-phi-epsilon-mu"),  # compatibility forms and the micro sign
        ("ά ώ", "alpha-omega"),  # accented Greek
    ],
)
def test_slugify_greek_letters(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_greek_does_not_collide_with_plain_word() -> None:
    assert slugify("σ-алгебры") != slugify("Алгебры")
