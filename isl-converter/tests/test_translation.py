"""
Tests for backend/translation.py. The Google call is replaced by a stub, so these test OUR logic
(normalisation, whole-text translation, caching, per-language failures, timeouts), not Google.
Run:  python -m unittest tests.test_translation -v
"""
import os
import sys
import time
import unicodedata
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
import translation  # noqa: E402


class TestTranslation(unittest.TestCase):
    def setUp(self):
        self._orig = translation._backend
        translation._cache.clear()
        self.calls = []

        def stub(text, target):
            self.calls.append((text, target))
            return f"<{target}:{text}>"

        translation._backend = stub

    def tearDown(self):
        translation._backend = self._orig

    def test_whole_text_is_translated_in_one_call_not_letter_by_letter(self):
        out = translation.translate_many("आई उ", ["en", "hi"])
        self.assertEqual(out["en"], {"text": "<en:आई उ>", "error": None})
        self.assertEqual(sorted(self.calls), [("आई उ", "en"), ("आई उ", "hi")])

    def test_marathi_is_the_original_text_and_needs_no_service(self):
        out = translation.translate_many("आई", ["mr"])
        self.assertEqual(out["mr"]["text"], "आई")
        self.assertEqual(self.calls, [])

    def test_empty_text(self):
        out = translation.translate_many("   ", ["en", "hi", "mr"])
        self.assertTrue(all(v == {"text": "", "error": None} for v in out.values()))
        self.assertEqual(self.calls, [])

    def test_text_is_nfc_normalised_like_the_notebook(self):
        decomposed = unicodedata.normalize("NFD", "ऩ")
        translation.translate_text(decomposed, "en")
        self.assertEqual(self.calls[0][0], unicodedata.normalize("NFC", decomposed))

    def test_results_are_cached(self):
        translation.translate_many("आई", ["en"])
        translation.translate_many("आई", ["en"])
        self.assertEqual(len(self.calls), 1)

    def test_one_language_failing_does_not_break_the_others(self):
        def flaky(text, target):
            if target == "hi":
                raise ConnectionError("offline")
            return f"<{target}>"

        translation._backend = flaky
        out = translation.translate_many("आई", ["en", "hi", "mr"])
        self.assertEqual(out["en"]["text"], "<en>")
        self.assertIsNone(out["hi"]["text"])
        self.assertEqual(out["hi"]["error"], translation.FRIENDLY_ERROR)
        self.assertEqual(out["mr"]["text"], "आई")

    def test_hung_request_times_out(self):
        translation._backend = lambda text, target: time.sleep(3) or "late"
        started = time.time()
        out = translation.translate_many("आई", ["en"], timeout=0.3)
        self.assertLess(time.time() - started, 1.5)
        self.assertEqual(out["en"]["error"], translation.FRIENDLY_ERROR)

    def test_resolve_language_accepts_codes_and_names(self):
        self.assertEqual(translation.resolve_language("hi"), "hi")
        self.assertEqual(translation.resolve_language("Hindi"), "hi")
        self.assertEqual(translation.resolve_language("marathi"), "mr")
        self.assertIsNone(translation.resolve_language("klingon"))
        self.assertIsNone(translation.resolve_language(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
