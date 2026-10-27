"""Tests for multilingual answers (gmr_agents/language.py and its wiring in main.py / graph/nodes.py).

No LLM quota and no database: the translate LLM and run_agent are scripted; language
resolution, the translation layer, answer_question(), the graph's understand step and the
HTTP endpoints are the real code.

    python -m unittest tests.test_language -v        (from backend/)
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from shared import config, llm_gateway  # noqa: E402
from shared.session_store import SessionStore  # noqa: E402
from gmr_agents import language as L  # noqa: E402
from gmr_agents import main  # noqa: E402

# Test fixtures only (what a scripted answer writer "wrote") -- never used by production code.
SUMMARIES = {
    "en": "Kashi has 1,352,702 installed meters, of which 1,190,471 (88.01%) passed SAT-13 by 22 Jul 2026.",
    "hi": "काशी में 1,352,702 मीटर इंस्टॉल हैं, जिनमें से 1,190,471 (88.01%) ने 22 Jul 2026 तक SAT-13 पास किया।",
    "hinglish": "Kashi mein 1,352,702 meters installed hain, jinme se 1,190,471 (88.01%) ne 22 Jul 2026 tak SAT-13 pass kiya.",
}
INSIGHT = "Meter AL1151970 has Rs 12.34 crore receivables pending."
KPIS = [{"label": "Meters Installed", "value": 1352702, "formatted_value": "1,352,702"},
        {"label": "SAT %", "value": 88.01, "formatted_value": "88.01%"}]
TABLES = [{"id": "primary_table", "title": "Result", "columns": ["Project", "Installed"],
           "rows": [{"Project": "KASHI", "Installed": 1352702}], "page": 1, "page_size": 200,
           "total_rows": 1, "has_next": False, "has_previous": False}]
FOLLOWUP = "How many meters passed SAT-13 in Kashi?"
CITATIONS = [{"id": "sql", "source": "live query", "location": "public.gold_dashboard_stats_snapshot"}]


def _items_of(prompt: str) -> tuple[str, dict]:
    target = re.search(r"TARGET: (\w+)", prompt).group(1)
    body = re.search(r"<untrusted_data[^>]*>\n(.*)\n</untrusted_data>", prompt, re.S).group(1)
    return target, json.loads(body)


# One word per target language, in its script (Hinglish / English: Roman)
WORD = {"hi": "अनुवाद", "mr": "भाषांतर", "bn": "অনুবাদ", "ta": "மொழி", "te": "అనువాదం", "gu": "અનુવાદ",
        "pa": "ਅਨੁਵਾਦ", "kn": "ಅನುವಾದ", "ml": "വിവർത്തനം", "en": "translated", "hinglish": "anuvaad"}


def scripted(text: str, target: str) -> str:
    """What the scripted translator 'writes': every word in the target script, tokens with a
    digit (figures, IDs, dates) kept exactly."""
    return " ".join(w if any(c.isdigit() for c in w) else WORD[target] for w in text.split())


def fake_translator(mutate: dict[str, str] | None = None, calls: list | None = None):
    """A scripted translate LLM (see scripted()), optionally changing a figure."""
    def chat_json(task, prompt, **_kw):
        if calls is not None:
            calls.append(task)
        target, items = _items_of(prompt)
        out = {}
        for k, v in items.items():
            text = scripted(v, target)
            for old, new in (mutate or {}).items():
                text = text.replace(old, new)
            out[k] = text
        return llm_gateway.LLMResult({"items": out}, "scripted", "scripted", None, None, 0.0)
    return chat_json


def answer(lang: str, *, refused: bool = False, summary: str | None = None) -> dict:
    return {"question": "q", "agent": "wfm", "summary": summary or SUMMARIES[lang], "kpis": copy.deepcopy(KPIS),
            "charts": [], "tables": copy.deepcopy(TABLES), "insights": [INSIGHT], "warnings": [],
            "suggested_followups": [FOLLOWUP], "citations": copy.deepcopy(CITATIONS), "refused": refused,
            "answer_language": lang, "debug": {"trace_id": "orig-trace", "sql": "SELECT 1"}}


def values_only(blocks: list[dict]) -> list[dict]:
    """A block list without its display translations (display_*, column_labels)."""
    return [{k: v for k, v in b.items() if not k.startswith("display_") and k != "column_labels"} for b in blocks]


class DetectionTests(unittest.TestCase):
    def test_english(self):
        self.assertEqual(L.detect("What is the total installed meter count in Kashi?"), "en")
        self.assertEqual(L.detect("Show feeder-wise outage details."), "en")

    def test_hindi_devanagari(self):
        self.assertEqual(L.detect("काशी में कुल कितने मीटर इंस्टॉल हैं?"), "hi")

    def test_hinglish_roman(self):
        for q in ("Kashi mein total installed meter count kitna hai?", "Kashi me total MI count btao",
                  "Feeder-wise outage details dikhao.", "Feeder ka outage status kya hai?",
                  "Show karo ki kaunse feeders mein outage hua hai.",
                  "Pichhle month ke comparison mein Low PF kitna increase hua?",
                  "MDM mein pending meters kitne hain aur WFM se verify kar do."):
            self.assertEqual(L.detect(q), "hinglish", q)

    def test_mixed_devanagari_and_english_terms_is_hindi(self):
        self.assertEqual(L.detect("काशी में total MI count बताओ"), "hi")

    def test_other_scripts(self):
        self.assertEqual(L.detect("किती मीटर बसवले आहेत?"), "mr")
        self.assertEqual(L.detect("காசியில் எத்தனை மீட்டர்கள்?"), "ta")
        self.assertEqual(L.detect("কাশীতে কতগুলি মিটার?"), "bn")

    def test_ambiguous(self):
        for q in ("SAT?", "AL1151970", "Kashi?"):
            self.assertIsNone(L.detect(q), q)

    def test_english_words_are_not_hinglish(self):
        self.assertEqual(L.detect("show me zone wise installs"), "en")
        self.assertEqual(L.detect("English meter count in Agra"), "en")


class ExplicitRequestTests(unittest.TestCase):
    def test_requests(self):
        self.assertEqual(L.explicit_request("Low PF ke baare mein Hindi mein explain karo."), "hi")
        self.assertEqual(L.explicit_request("Explain this in English."), "en")
        self.assertEqual(L.explicit_request("हिंदी में बताओ"), "hi")
        self.assertEqual(L.explicit_request("Reply in Tamil please"), "ta")
        self.assertEqual(L.explicit_request("Hindi mein nahi, English mein batao"), "en")  # the last one named

    def test_not_requests(self):
        for q in ("How many meters in Tamil Nadu?", "English meter count in Agra", "Show SAT for Kashi"):
            self.assertIsNone(L.explicit_request(q), q)

    def test_pure_switch(self):
        self.assertEqual(L.pure_switch("Explain this in English."), "en")
        self.assertEqual(L.pure_switch("Hindi mein batao"), "hi")
        self.assertEqual(L.pure_switch("हिंदी में बताओ"), "hi")
        self.assertIsNone(L.pure_switch("Low PF ke baare mein Hindi mein explain karo."))  # a real question
        self.assertIsNone(L.pure_switch("Agra ka Hindi mein batao"))

    def test_preference_normalization(self):
        self.assertEqual(L.normalize_preference(None), "auto")
        self.assertEqual(L.normalize_preference("hi-IN"), "hi")
        self.assertEqual(L.normalize_preference("Hinglish"), "hinglish")
        self.assertEqual(L.normalize_preference("klingon"), "auto")


class ResolveTests(unittest.TestCase):
    """Priority: explicit request > dropdown > the latest message > previous answer > English."""

    def test_auto_matches_each_message(self):
        self.assertEqual(L.resolve("What is the total installed meter count in Kashi?")[0], "en")
        self.assertEqual(L.resolve("काशी में कुल कितने मीटर इंस्टॉल हैं?")[0], "hi")
        self.assertEqual(L.resolve("Kashi mein total installed meter count kitna hai?")[0], "hinglish")

    def test_explicit_hindi_in_english_question(self):
        self.assertEqual(L.resolve("How many meters are installed in Kashi? Answer in Hindi.")[0], "hi")

    def test_explicit_beats_dropdown(self):
        self.assertEqual(L.resolve("Explain this in English.", "hi")[0], "en")

    def test_dropdown_beats_message_language(self):
        target, detected = L.resolve("Kashi mein kitne meters installed hain?", "en")
        self.assertEqual((target, detected), ("en", "hinglish"))
        self.assertEqual(L.resolve("What is the SAT count?", "ta")[0], "ta")

    def test_hindi_conversation_then_english_question(self):
        prior = [{"question": "q", "summary": "s", "context": {"language": "hi"}}]
        self.assertEqual(L.resolve("What is the SAT count in Agra?", "auto", "en", prior)[0], "en")

    def test_hinglish_conversation_then_hindi_question(self):
        prior = [{"question": "q", "summary": "s", "context": {"language": "hinglish"}}]
        self.assertEqual(L.resolve("आगरा में SAT कितना है?", "auto", "hi", prior)[0], "hi")

    def test_ambiguous_keeps_conversation_language(self):
        prior = [{"question": "q", "summary": "s", "context": {"language": "hi"}}]
        self.assertEqual(L.resolve("SAT?", "auto", "en", prior)[0], "hi")
        self.assertEqual(L.resolve("SAT?", "auto", "en", [])[0], "en")

    def test_llm_reading_for_roman_script(self):
        # Roman script: the triage LLM decides en vs Hinglish; 'hi' in Roman script is Hinglish.
        self.assertEqual(L.resolve("Agra SAT status batao", "auto", "hi")[0], "hinglish")
        # Script is certain: an LLM 'en' for Devanagari text is ignored.
        self.assertEqual(L.resolve("काशी में कितने मीटर?", "auto", "en")[0], "hi")


class WrittenLanguageTests(unittest.TestCase):
    """What an answer is really written in -- the cases below were all seen live."""

    def test_hinglish_wins_when_the_llm_says_english(self):
        self.assertEqual(L.resolve("Kashi mein total installed meter count kitna hai?", "auto", "en")[0], "hinglish")

    def test_english_reply_to_a_hindi_request(self):
        reply = "Weather data is not available in this system."
        self.assertEqual(L.written_in(reply, "hi"), "en")

    def test_english_answer_claimed_as_hinglish(self):
        self.assertEqual(L.written_in("SAT = Site Acceptance Test. It is the formal approval step.", "hinglish"), "en")

    def test_real_answers_pass(self):
        self.assertEqual(L.written_in("काशी में अब तक कुल 1365396 मीटर इंस्टॉल किए गए हैं।", "hi"), "hi")
        self.assertEqual(L.written_in("Agra mein eligible pending SAT 96,373 meter hain.", "hinglish"), "hinglish")
        self.assertEqual(L.written_in("Kashi has 1,365,396 installed meters.", "en"), "en")

    def test_mixed_scripts_are_not_tamil(self):
        self.assertFalse(L.in_script("காலம் 2026-10-25-ஐ 기준으로 172 meter लगे।", "ta"))
        self.assertNotEqual(L.written_in("காலம் 2026-10-25-ஐ 기준으로 172 meter लगे।", "ta"), "ta")
        self.assertTrue(L.in_script("காசியில் 1,365,396 மீட்டர்கள் நிறுவப்பட்டுள்ளன (SAT).", "ta"))
        self.assertFalse(L.in_script("இந்த முடிவில் ப்ரஜெクト அல்லது OEM szerint பிரிவுகள் கிடைக்கவில்லை.", "ta"))
        self.assertFalse(L.in_script("மொத்த நிறுவப்பட்ட மெடர்கொissy count in Kashi is 1,386,445 so far (all-time).", "ta"))
        self.assertFalse(L.in_script("Assuming you mean installed meters (all-time), काशी में कुल 1,386,445 मीटर इंस्टॉल हैं (so far).", "deva"))

    def test_hindi_with_many_english_terms_is_hindi(self):
        # live translations that a letter-share rule wrongly rejected
        for text in ("Low PF (Power Factor) के बारे में मेरे पास कोई data नहीं है। मैं सिर्फ WFM (meter installation, SAT, "
                     "O&M tickets, inventory, receivables) और MDM (feeder auto-indexing) से जुड़े सवालों के जवाब दे सकता हूँ।",
                     "Agra में कुल 1,576,125 eligible MI हैं, जिनमें से 96,373 मेटर eligible pending SAT हैं।"):
            self.assertTrue(L.in_script(text, "deva"), text)
            self.assertEqual(L.written_in(text, "hi"), "hi")

    def test_bengali_sentence_with_danda(self):
        # live: rejected because '।' sits in the Devanagari block
        self.assertTrue(L.in_script("সেপ্টেম্বর 2026-এ Triveni প্রজেক্টে মোট 20,433টি মিটার ইনস্টল করা হয়েছে।", "bn"))

    def test_only_reliable_languages_are_written_directly(self):
        self.assertEqual([L.writing_language(c) for c in ("en", "hi", "hinglish", "ta", "mr")],
                         ["en", "hi", "hinglish", "en", "en"])


class TranslationLayerTests(unittest.TestCase):
    def setUp(self):
        L._CACHE.clear()

    def test_translates_text_and_keeps_data(self):
        calls: list = []
        with mock.patch.object(llm_gateway, "chat_json", fake_translator(calls=calls)):
            out = L.translate_response(answer("en"), "hi")
        self.assertEqual(out["summary"], scripted(SUMMARIES["en"], "hi"))
        self.assertEqual(out["insights"], [scripted(INSIGHT, "hi")])
        self.assertEqual((out["answer_language"], out["translation"]["status"]), ("hi", "ok"))
        # values, rows and citations are never touched; labels come as display fields next to them
        self.assertEqual((values_only(out["kpis"]), values_only(out["tables"]), out["citations"]),
                         (KPIS, TABLES, CITATIONS))
        self.assertEqual(out["kpis"][0]["display_label"], scripted("Meters Installed", "hi"))
        self.assertEqual(out["tables"][0]["display_title"], scripted("Result", "hi"))
        self.assertEqual(out["tables"][0]["column_labels"], {"Project": scripted("Project", "hi"),
                                                             "Installed": scripted("Installed", "hi")})
        self.assertEqual(out["suggested_followups"], [FOLLOWUP])               # what a click runs
        self.assertEqual(out["suggested_followup_labels"], [scripted(FOLLOWUP, "hi")])  # what is shown
        self.assertEqual(calls, ["translate"])                                   # all in one call

    def test_original_is_not_mutated(self):
        original = answer("en")
        before = copy.deepcopy(original)
        with mock.patch.object(llm_gateway, "chat_json", fake_translator()):
            L.translate_response(original, "ta")
        self.assertEqual(original, before)

    def test_answer_already_in_target_language_keeps_its_text(self):
        calls: list = []
        with mock.patch.object(llm_gateway, "chat_json", fake_translator(calls=calls)):
            out = L.translate_response(answer("hi"), "hi")
        # the written answer stays as written; only labels / follow-ups are translated
        self.assertEqual((out["summary"], out["insights"]), (SUMMARIES["hi"], [INSIGHT]))
        self.assertEqual((calls, out["translation"]["status"], out["answer_language"]), (["translate"], "ok", "hi"))
        self.assertIn("display_label", out["kpis"][0])

    def test_english_answer_for_english_makes_no_call(self):
        calls: list = []
        with mock.patch.object(llm_gateway, "chat_json", fake_translator(calls=calls)):
            out = L.translate_response(answer("en"), "en")
        self.assertEqual((calls, out["translation"]["status"]), ([], "skipped"))

    def test_hinglish_translates_follow_ups_but_keeps_english_labels(self):
        with mock.patch.object(llm_gateway, "chat_json", fake_translator()):
            out = L.translate_response(answer("hinglish"), "hinglish")
        self.assertEqual(out["suggested_followup_labels"], [scripted(FOLLOWUP, "hinglish")])
        self.assertNotIn("display_label", out["kpis"][0])

    def test_items_left_out_of_the_reply_are_asked_once_more(self):
        # live: Marathi / Bengali table titles silently missing from the model's reply
        calls: list = []
        full = fake_translator(calls=calls)

        def forgetful(task, prompt, **kw):
            res = full(task, prompt, **kw)
            if len(calls) == 1:
                res.data["items"].pop("tables.0.title", None)
            return res

        with mock.patch.object(llm_gateway, "chat_json", forgetful):
            out = L.translate_response(answer("en"), "mr")
        self.assertEqual(calls, ["translate", "translate"])
        self.assertEqual(out["tables"][0]["display_title"], scripted("Result", "mr"))

    def test_acronym_labels_are_not_sent(self):
        prompts: list = []
        full = fake_translator()
        with mock.patch.object(llm_gateway, "chat_json", lambda t, p, **k: prompts.append(p) or full(t, p, **k)):
            out = L.translate_response(answer("hi"), "hi")
        self.assertNotIn('"kpis.1.label"', prompts[0])          # 'SAT %'
        self.assertNotIn("display_label", out["kpis"][1])
        self.assertIn('"kpis.0.label"', prompts[0])             # 'Meters Installed'

    def test_retranslation_replaces_earlier_display_labels(self):
        with mock.patch.object(llm_gateway, "chat_json", fake_translator()):
            hindi = L.translate_response(answer("en"), "hi")
            tamil = L.translate_response(hindi, "ta")
            english = L.translate_response(hindi, "en")
        self.assertEqual(tamil["kpis"][0]["display_label"], scripted("Meters Installed", "ta"))
        # back to English: no Hindi labels left over, the English originals show
        self.assertNotIn("display_label", english["kpis"][0])
        self.assertNotIn("column_labels", english["tables"][0])
        self.assertNotIn("suggested_followup_labels", english)
        self.assertEqual(values_only(english["kpis"]), KPIS)

    def test_changed_number_in_summary_keeps_the_whole_original(self):
        with mock.patch.object(llm_gateway, "chat_json", fake_translator(mutate={"1,352,702": "1,352,703"})):
            out = L.translate_response(answer("en"), "hi")
        # the summary had the figure -> rejected, and the answer stays whole in its original language
        self.assertEqual((out["summary"], out["insights"]), (SUMMARIES["en"], [INSIGHT]))
        self.assertEqual((out["translation"]["status"], out["answer_language"]), ("failed", "en"))

    def test_changed_number_in_an_insight_drops_only_that_translation(self):
        with mock.patch.object(llm_gateway, "chat_json", fake_translator(mutate={"AL1151970": "AL1151971"})):
            out = L.translate_response(answer("en"), "hi")
        self.assertEqual((out["summary"], out["insights"]), (scripted(SUMMARIES["en"], "hi"), [INSIGHT]))
        self.assertEqual((out["translation"]["status"], out["answer_language"]), ("partial", "hi"))

    def test_changed_id_currency_or_date_is_rejected(self):
        for old, new in (("AL1151970", "AL1151971"), ("12.34", "12.3"), ("2026", "2025"), ("SAT-13", "SAT-12")):
            L._CACHE.clear()
            original = answer("en")
            with mock.patch.object(llm_gateway, "chat_json", fake_translator(mutate={old: new})):
                out = L.translate_response(original, "ta")
            self.assertNotEqual(out["translation"]["status"], "ok", old)
            # whatever is shown carries exactly the original's figures / IDs
            self.assertTrue(L.values_preserved(original["summary"], out["summary"]), old)
            self.assertTrue(L.values_preserved(original["insights"][0], out["insights"][0]), old)

    def test_wrong_script_keeps_the_original(self):
        # seen live: a 'hi' translation came back in Roman script
        reply = llm_gateway.LLMResult({"items": {"summary": scripted(SUMMARIES["en"], "hinglish"),
                                                 "insights.0": scripted(INSIGHT, "hi")}},
                                      "scripted", "scripted", None, None, 0.0)
        with mock.patch.object(llm_gateway, "chat_json", lambda *a, **k: reply):
            out = L.translate_response(answer("en"), "hi")
        self.assertEqual((out["summary"], out["insights"]), (SUMMARIES["en"], [INSIGHT]))
        self.assertEqual(out["translation"]["status"], "failed")

    def test_provider_failure_returns_original(self):
        with mock.patch.object(llm_gateway, "chat_json", lambda *a, **k: None):
            out = L.translate_response(answer("en"), "hi")
        self.assertEqual((out["summary"], out["translation"]["status"]), (SUMMARIES["en"], "failed"))

    def test_timeout_exception_returns_original(self):
        def boom(*_a, **_k):
            raise TimeoutError("provider timed out")
        with mock.patch.object(llm_gateway, "chat_json", boom):
            out = L.translate_response(answer("en"), "hi")
        self.assertEqual((out["summary"], out["translation"]["status"]), (SUMMARIES["en"], "failed"))

    def test_malformed_reply_returns_original(self):
        for data in ({"text": "nope"}, {"items": "nope"}, {"items": {}}):
            L._CACHE.clear()
            reply = llm_gateway.LLMResult(data, "scripted", "scripted", None, None, 0.0)
            with mock.patch.object(llm_gateway, "chat_json", lambda *a, _r=reply, **k: _r):
                out = L.translate_response(answer("en"), "hi")
            self.assertEqual((out["summary"], out["translation"]["status"]), (SUMMARIES["en"], "failed"), data)

    def test_unsupported_language_falls_back_to_english(self):
        with mock.patch.object(llm_gateway, "chat_json", fake_translator()):
            out = L.translate_response(answer("hi"), "klingon")
        self.assertEqual(out["answer_language"], "en")

    def test_native_digits_are_normalised(self):
        self.assertEqual(L.ascii_digits("काशी में १,३५२,७०२ मीटर"), "काशी में 1,352,702 मीटर")
        self.assertTrue(L.values_preserved("1,352,702 meters", "१३,५२,७०२ मीटर"))  # Indian grouping = same value

    def test_repeat_translation_is_cached(self):
        calls: list = []
        with mock.patch.object(llm_gateway, "chat_json", fake_translator(calls=calls)):
            L.translate_response(answer("en"), "hi")
            L.translate_response(answer("en"), "hi")
        self.assertEqual(calls, ["translate"])


class _AppCase(unittest.TestCase):
    """answer_question() / the HTTP API on a throwaway session store; run_agent is scripted."""

    def setUp(self):
        L._CACHE.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(os.path.join(self.tmp.name, "sessions.json"))
        self.session = self.store.create_session("t")["session_id"]
        self.agent_calls: list[dict] = []
        self.llm_calls: list[str] = []
        patches = [mock.patch.object(config, "ANSWER_CACHE_ENABLED", False),
                   mock.patch.object(main, "run_agent", self._fake_run_agent),
                   mock.patch.object(main, "sessions", self.store),
                   mock.patch.object(llm_gateway, "chat_json", fake_translator(calls=self.llm_calls))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.store.flush)  # cleanups run last-in first-out: written before the dir goes

    def _fake_run_agent(self, question, *, prior_questions=None, prior_turns=None, trace_id=None, session_id=None,
                        emit=None, response_language="auto", refuse=False):
        """What the pipeline does with language: resolve it after triage, write the answer in it."""
        self.agent_calls.append({"question": question, "response_language": response_language})
        target, detected = L.resolve(question, response_language, None, prior_turns)
        written = target if target in SUMMARIES else "en"
        refused = "weather" in question.lower()
        return {"agent": "wfm", "summary": "Weather data is not available." if refused else SUMMARIES[written],
                "kpis": [] if refused else copy.deepcopy(KPIS), "charts": [],
                "tables": [] if refused else copy.deepcopy(TABLES), "insights": [] if refused else [INSIGHT],
                "warnings": [], "followups": [], "citations": copy.deepcopy(CITATIONS), "refused": refused,
                "trace_id": trace_id, "plan": {"sql": "SELECT 1", "answer_kind": "single_value"},
                "routing": {"domain": "wfm", "topics": ["dashboard_stats"]}, "narration_source": "scripted",
                "language_pref": L.normalize_preference(response_language), "detected_language": detected,
                "target_language": target, "answer_language": "en" if refused else written}

    def ask(self, question, response_language=None):
        return main.answer_question(question, self.session, store=self.store, response_language=response_language)

    def messages(self):
        return self.store.get_session(self.session)["messages"]


class AnswerQuestionTests(_AppCase):
    def test_answers_in_the_users_language_without_translation(self):
        for q, lang in (("What is the total installed meter count in Kashi?", "en"),
                        ("काशी में कुल कितने मीटर इंस्टॉल हैं?", "hi"),
                        ("Kashi mein total installed meter count kitna hai?", "hinglish")):
            r = self.ask(q)
            self.assertEqual((r["answer_language"], r["detected_language"], r["response_language"]), (lang, lang, "auto"))
            self.assertEqual(r["summary"], SUMMARIES[lang])  # written directly in its language, not re-translated
        # English: no call; Hindi: one call for its card / table labels; Hinglish: follow-ups only (none here)
        self.assertEqual(self.llm_calls, ["translate"])

    def test_missing_field_is_auto(self):
        r = self.ask("Kashi mein kitne meters installed hain?")
        self.assertEqual((self.agent_calls[-1]["response_language"], r["response_language"]), ("auto", "auto"))

    def test_dropdown_is_passed_to_the_pipeline(self):
        r = self.ask("Kashi mein kitne meters installed hain?", "en")
        self.assertEqual(self.agent_calls[-1]["response_language"], "en")
        self.assertEqual((r["answer_language"], r["detected_language"]), ("en", "hinglish"))

    def test_canned_refusal_is_translated_to_the_target(self):
        r = self.ask("Aaj weather kaisa hai?")
        self.assertTrue(r["refused"])
        self.assertEqual((r["summary"], r["answer_language"]), (scripted("Weather data is not available.", "hinglish"), "hinglish"))

    def test_history_stores_one_pair_per_question_with_its_language(self):
        self.ask("काशी में कुल कितने मीटर इंस्टॉल हैं?")
        self.ask("SAT?")  # ambiguous -> keeps Hindi
        msgs = self.messages()
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"] * 2)
        self.assertEqual([m["context"]["language"] for m in msgs if m["role"] == "assistant"], ["hi", "hi"])
        self.assertEqual(self.agent_calls[-1]["question"], "SAT?")

    def test_hindi_conversation_then_english_follow_up(self):
        self.ask("काशी में कुल कितने मीटर इंस्टॉल हैं?")
        r = self.ask("What is the total installed meter count in Kashi?")
        self.assertEqual(r["answer_language"], "en")

    def test_explain_this_in_english_reuses_the_last_answer(self):
        first = self.ask("काशी में कुल कितने मीटर इंस्टॉल हैं?")
        runs = len(self.agent_calls)
        r = self.ask("Explain this in English.")
        self.assertEqual(len(self.agent_calls), runs)  # no new pipeline run, so no SQL
        self.assertEqual(r["summary"], scripted(SUMMARIES["hi"], "en"))
        self.assertEqual((values_only(r["kpis"]), values_only(r["tables"])),
                         (values_only(first["kpis"]), values_only(first["tables"])))
        self.assertEqual(r["debug"]["trace_id"], first["debug"]["trace_id"])  # table paging still finds the SQL
        self.assertEqual(len(self.messages()), 4)


class TranslatedFollowupTests(_AppCase):
    def test_runs_the_verified_question_in_the_chips_language(self):
        chip = "काशी में SAT-13 में कितने मीटर पास हुए?"
        with mock.patch.object(main.business_rules, "is_suggestion", lambda q: q == FOLLOWUP):
            r = main.answer_question(chip, self.session, store=self.store, canonical_question=FOLLOWUP)
        self.assertEqual(self.agent_calls[-1], {"question": FOLLOWUP, "response_language": "hi"})
        self.assertEqual((r["question"], r["answer_language"], r["response_language"]), (chip, "hi", "auto"))
        user = self.messages()[-2]
        self.assertEqual((user["role"], user["content"], user.get("normalized")), ("user", chip, FOLLOWUP))

    def test_unverified_canonical_is_ignored(self):
        with mock.patch.object(main.business_rules, "is_suggestion", lambda q: False):
            main.answer_question("Agra mein SAT kitna hai?", self.session, store=self.store,
                                 canonical_question="DROP TABLE x")
        self.assertEqual(self.agent_calls[-1]["question"], "Agra mein SAT kitna hai?")


class ApiTests(_AppCase):
    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        self.client = TestClient(main.app)

    def test_ai_query_accepts_old_and_new_clients(self):
        old = self.client.post("/api/ai-query", json={"question": "What is the SAT count?", "session_id": self.session})
        new = self.client.post("/api/ai-query", json={"question": "What is the SAT count?", "session_id": self.session,
                                                      "response_language": "hi"})
        self.assertEqual((old.status_code, new.status_code), (200, 200))
        self.assertEqual((old.json()["answer_language"], new.json()["answer_language"]), ("en", "hi"))

    def test_dropdown_translation_does_not_rerun_or_add_messages(self):
        r = self.client.post("/api/ai-query", json={"question": "What is the total installed meter count in Kashi?",
                                                    "session_id": self.session}).json()
        runs, count = len(self.agent_calls), len(self.messages())
        trace = r["debug"]["trace_id"]
        seen = {}
        for lang in ("hi", "ta", "en", "hinglish", "hi", "auto"):  # repeated switching
            t = self.client.post("/api/translate", json={"session_id": self.session, "trace_id": trace,
                                                         "target_language": lang})
            self.assertEqual(t.status_code, 200, lang)
            seen[lang] = t.json()
            # every translation comes from the stored original, so its numbers are the original's
            self.assertTrue(L.values_preserved(SUMMARIES["en"], t.json()["summary"]), lang)
        self.assertEqual((len(self.agent_calls), len(self.messages())), (runs, count))
        self.assertEqual(seen["ta"]["summary"], scripted(SUMMARIES["en"], "ta"))
        self.assertEqual(seen["en"]["translation"]["status"], "skipped")  # already English: no call
        self.assertEqual(seen["auto"]["summary"], SUMMARIES["en"])       # auto = the original
        self.assertEqual(self.llm_calls.count("translate"), 3)            # hi, ta, hinglish (2nd hi cached)
        stored = self.messages()[-1]["response"]
        self.assertEqual((stored["summary"], stored["kpis"], stored["tables"]), (SUMMARIES["en"], KPIS, TABLES))

    def test_translate_failure_shows_the_original(self):
        r = self.client.post("/api/ai-query", json={"question": "What is the SAT count?", "session_id": self.session}).json()
        with mock.patch.object(llm_gateway, "chat_json", lambda *a, **k: None):
            t = self.client.post("/api/translate", json={"session_id": self.session, "trace_id": r["debug"]["trace_id"],
                                                         "target_language": "hi"}).json()
        self.assertEqual((t["summary"], t["translation"]["status"]), (SUMMARIES["en"], "failed"))

    def test_translate_unknown_answer_is_404(self):
        t = self.client.post("/api/translate", json={"session_id": self.session, "trace_id": "nope", "target_language": "hi"})
        self.assertEqual(t.status_code, 404)


class RoutingUnchangedTests(unittest.TestCase):
    """The same question in any language routes the same way: language never touches the
    English question, domain, topics or tables the query writer gets."""

    def _understand(self, message: str, pref: str = "auto") -> tuple[dict, dict]:
        from gmr_agents import llm_pipeline as P
        from gmr_agents.graph import context, nodes

        triage = {"kind": "data", "question": "What is the total installed meter count in Kashi?", "followup": False,
                  "domain": "wfm", "language": "en", "parts": [], "examples": ["e3"], "assumption": "",
                  "topics": ["dashboard_stats"], "tables": ["gold_dashboard_stats_snapshot"], "reply": "", "source": "x",
                  "named": ""}
        llm_reading = {"काशी": "hi", "mein": "hinglish"}
        triage["language"] = next((v for k, v in llm_reading.items() if k in message), "en")
        ps = {"question": message, "trace_id": f"t-{abs(hash(message + pref))}", "trace": [], "warnings": [],
              "language_pref": pref, "forced_language": L.forced(message, pref)}
        context.open_run(ps["trace_id"], ps)
        try:
            with mock.patch.object(P, "_route", lambda s, p: None), \
                 mock.patch.object(P, "_triage", lambda s, p: copy.deepcopy(triage)), \
                 mock.patch.object(P, "fix_question", lambda *a: False):
                out = nodes.understand_node({"question": message, "prior_turns": []},
                                            {"configurable": {"trace_id": ps["trace_id"]}})
        finally:
            context.close_run(ps["trace_id"])
        return out, ps

    def test_same_route_in_every_language(self):
        routes = {}
        for msg, pref, lang in (("What is the total installed meter count in Kashi?", "auto", "en"),
                                ("काशी में कुल कितने मीटर इंस्टॉल हैं?", "auto", "hi"),
                                ("Kashi mein total installed meter count kitna hai?", "auto", "hinglish"),
                                ("Kashi mein total installed meter count kitna hai?", "ta", "ta"),
                                ("Kashi mein total installed meter count kitna hai? English mein batao", "hi", "en")):
            out, ps = self._understand(msg, pref)
            t = out["triage"]
            self.assertEqual((out["route"], t["language"], ps["target_language"]), ("data", lang, lang), msg)
            routes[(msg, pref)] = (t["question"], t["domain"], tuple(t["topics"]), tuple(t["tables"]), ps["routing"]["domain"])
        self.assertEqual(len(set(routes.values())), 1)


if __name__ == "__main__":
    unittest.main()
