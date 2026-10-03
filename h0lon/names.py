"""ASCII slugs for directory and file names (Russian and Greek transliteration, NFKD for Latin)."""

from __future__ import annotations

import re
import unicodedata

# Practical transliteration (close to the passport/Yandex style): readable, ASCII only.
_CYRILLIC = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    # Ukrainian / Belarusian letters that show up in course materials.
    "і": "i",
    "ї": "yi",
    "є": "ye",
    "ґ": "g",
    "ў": "u",
}

# Latin letters that NFKD does not decompose into ASCII.
_LATIN_EXTRA = {
    "ß": "ss",
    "æ": "ae",
    "œ": "oe",
    "ø": "o",
    "đ": "d",
    "ð": "d",
    "ł": "l",
    "þ": "th",
    "ı": "i",
}

# Greek letters by name: maths and physics titles use them as symbols («σ-алгебры», «χ²»).
# Applied after NFKD, so accented letters (ά) and compatibility forms (ϑ, ϕ, µ) map as well.
_GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "ς": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
}

# Apostrophes vanish instead of splitting a word: "don't" -> "dont".
_DROP = {"'", "’", "‘", "`", "ʼ"}

_TABLE = str.maketrans({**_CYRILLIC, **_LATIN_EXTRA, **dict.fromkeys(_DROP, "")})
_GREEK_TABLE = str.maketrans(_GREEK)
_NON_SLUG = re.compile(r"[^a-z0-9]+")

DEFAULT_SLUG = "untitled"


def transliterate(text: str) -> str:
    """Lowercase ASCII approximation of `text` (non-letters are kept as is when ASCII)."""
    # NFKC first: decomposed input (й as и + U+0306, from macOS file names or PDF text)
    # must hit the Cyrillic table just like the composed form.
    composed = unicodedata.normalize("NFKC", text)
    lowered = composed.lower().translate(_TABLE)
    decomposed = unicodedata.normalize("NFKD", lowered).translate(_GREEK_TABLE)
    return decomposed.encode("ascii", "ignore").decode("ascii")


def slugify(text: str, *, max_len: int = 60) -> str:
    """ASCII slug `[a-z0-9-]`: «Теорвер — лекция 3» -> 'teorver-lektsiya-3'.

    Hyphens are collapsed and trimmed; a slug longer than `max_len` is cut at a word
    boundary (a single over-long word is cut hard). An empty result becomes 'untitled'.
    """
    if max_len < 1:
        raise ValueError("max_len must be positive")
    slug = _NON_SLUG.sub("-", transliterate(text)).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        if slug[max_len] != "-" and "-" in cut:
            cut = cut.rsplit("-", 1)[0]
        slug = cut.strip("-")
    return slug or DEFAULT_SLUG
