"""
translation.py : translation of the recognised text.

Same implementation as the notebook: deep-translator's GoogleTranslator with source="mr"
(the CNN's classes are Marathi letters), NFC-normalised text. The COMPLETE text is translated,
never letter by letter. Marathi output is the recognised text itself.

Translation needs an internet connection (it calls Google Translate through deep-translator).
"""
from __future__ import annotations

import logging
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, wait

log = logging.getLogger("isl.translation")

SOURCE_LANGUAGE = "mr"

# Same table as the notebook. The website exposes the first three.
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta", "Telugu": "te",
    "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
    "Punjabi": "pa", "Odia": "or", "Assamese": "as",
}
UI_LANGUAGES = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

MAX_CHARS = 2000
TIMEOUT_SECONDS = 25.0
FRIENDLY_ERROR = "Translation is unavailable right now. Check the internet connection."

_cache: dict[tuple[str, str], str] = {}
_cache_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="translate")


def resolve_language(value):
    """Accept 'hi', 'Hindi' or 'hindi'; return the language code or None."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.lower() in LANGUAGES.values():
        return value.lower()
    for name, code in LANGUAGES.items():
        if name.lower() == value.lower():
            return code
    return None


def normalize(text) -> str:
    return unicodedata.normalize("NFC", str(text or "")).strip()[:MAX_CHARS]


def _backend(text: str, target: str) -> str:
    """Try Google first; if it blocks us, fall back to MyMemory."""
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    try:
        return GoogleTranslator(source=SOURCE_LANGUAGE, target=target).translate(text)
    except Exception as exc:  # noqa: BLE001
        log.warning("Google failed (%s), trying MyMemory", exc)
        names = {"en": "english", "hi": "hindi"}
        if target not in names:
            raise
        return MyMemoryTranslator(source="marathi", target=names[target]).translate(text)


def _cached_or_none(text: str, target: str):
    with _cache_lock:
        return _cache.get((text, target))


def _remember(text: str, target: str, value: str):
    with _cache_lock:
        if len(_cache) > 500:
            _cache.clear()
        _cache[(text, target)] = value


def translate_text(text, target: str) -> str:
    """Translate one string. Raises on failure (callers decide how to show it)."""
    text = normalize(text)
    if not text:
        return ""
    if target == SOURCE_LANGUAGE:
        return text
    hit = _cached_or_none(text, target)
    if hit is not None:
        return hit
    result = _backend(text, target)
    if not isinstance(result, str):
        raise RuntimeError("the translator returned no text")
    _remember(text, target, result)
    return result


def translate_many(text, targets, timeout: float = TIMEOUT_SECONDS) -> dict:
    """
    Translate the same complete text into several languages in parallel.
    Returns {code: {"text": str | None, "error": str | None}}. One language failing never
    affects the others, and a hung request cannot block longer than `timeout` seconds.
    """
    text = normalize(text)
    out: dict[str, dict] = {}
    pending = {}
    for code in dict.fromkeys(targets):
        if not text:
            out[code] = {"text": "", "error": None}
        elif code == SOURCE_LANGUAGE:
            out[code] = {"text": text, "error": None}
        elif (hit := _cached_or_none(text, code)) is not None:
            out[code] = {"text": hit, "error": None}
        else:
            pending[_pool.submit(translate_text, text, code)] = code

    if pending:
        done, not_done = wait(pending, timeout=timeout)
        for future in done:
            code = pending[future]
            try:
                out[code] = {"text": future.result(), "error": None}
            except Exception as exc:  # noqa: BLE001
                log.warning("translation to %s failed: %s", code, exc)
                out[code] = {"text": None, "error": FRIENDLY_ERROR}
        for future in not_done:
            log.warning("translation to %s timed out", pending[future])
            out[pending[future]] = {"text": None, "error": FRIENDLY_ERROR}
    return out
