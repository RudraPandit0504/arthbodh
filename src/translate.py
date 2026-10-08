"""Step 1: English -> Hindi sentence, Hindi gloss -> English for display.

Google Translate (via deep-translator) is tried first; if it is unreachable or
rate-limited, MyMemory is used as a fallback. Both need internet.
"""
from functools import lru_cache

from deep_translator import GoogleTranslator, MyMemoryTranslator

_LANG = {"en": "en-GB", "hi": "hi-IN"}  # MyMemory language codes


def _translate(text, source, target):
    if not text or not text.strip():
        return text
    for make in (lambda: GoogleTranslator(source=source, target=target),
                 lambda: MyMemoryTranslator(source=_LANG[source], target=_LANG[target])):
        try:
            out = make().translate(text)
            if out:
                return out
        except Exception:
            continue
    return None


@lru_cache(maxsize=2048)
def en_to_hi(text):
    """Translate English to Hindi. Returns None if no translator is reachable."""
    return _translate(text, "en", "hi")


@lru_cache(maxsize=4096)
def hi_to_en(text):
    """Translate Hindi to English (used to show glosses in English). None on failure."""
    return _translate(text, "hi", "en")
