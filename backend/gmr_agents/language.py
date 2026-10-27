"""Response language: which language an answer is written in, and translating a finished answer.

The answer is written straight in the target language by the answer / explain LLM (no extra
call); this module only decides WHICH language, by code, in this order (resolve()):

    1. explicit request in the message   'Hindi mein batao', 'explain in English', 'हिंदी में'
    2. the dropdown preference           response_language != 'auto'
    3. the latest message's language     script by code (Devanagari, Tamil ...); for Roman script
                                         the triage LLM's en / hinglish reading, else a word heuristic
    4. ambiguous message ('SAT?')        the previous answer's language, else English

translate_response() is the separate translation layer for an answer that already exists (the
dropdown changed, a cached answer in another language, a canned / refusal message). It only
touches presentation text (summary, insights, warnings) -- KPI cards, charts and tables are data
and stay as they are. Every number / ID token must survive unchanged: a field whose translation
adds, drops or changes one keeps its original text. A failed call returns the original answer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
from collections import Counter, OrderedDict
from typing import Any

from shared import config, llm_gateway
from shared.logging_config import new_trace_id
from gmr_agents.guards import injection

log = logging.getLogger(__name__)

# code -> (English name, native name, how to write it -- shown to the answer / translate LLM)
LANGUAGES: dict[str, tuple[str, str, str]] = {
    "en": ("English", "English", "English"),
    "hi": ("Hindi", "हिंदी", "natural conversational Hindi in Devanagari script (not formal / Sanskritised)"),
    "hinglish": ("Hinglish", "Hinglish", "Hinglish: conversational Hindi written in Roman (Latin) script, the way "
                                         "the user writes it"),
    "bn": ("Bengali", "বাংলা", "Bengali in Bengali script"),
    "mr": ("Marathi", "मराठी", "Marathi in Devanagari script"),
    "ta": ("Tamil", "தமிழ்", "Tamil in Tamil script"),
    "te": ("Telugu", "తెలుగు", "Telugu in Telugu script"),
    "gu": ("Gujarati", "ગુજરાતી", "Gujarati in Gujarati script"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ", "Punjabi in Gurmukhi script"),
    "kn": ("Kannada", "ಕನ್ನಡ", "Kannada in Kannada script"),
    "ml": ("Malayalam", "മലയാളം", "Malayalam in Malayalam script"),
}
CODES = frozenset(LANGUAGES)
AUTO = "auto"

_TERMS = ("MDM, WFM, MI, SAT, DT/DTR, OEM, KPI, PMPM, SQL, feeder, Consumer ID, Low PF, SAIDI, SAIFI, "
          "invoice, receivables")


def prompt_line(code: str) -> str:
    """The LANGUAGE line of an answer / explain prompt."""
    code = code if code in CODES else "en"
    if code == "en":
        return "en -- English"
    return (f"{code} -- {LANGUAGES[code][2]}; keep familiar English technical terms and abbreviations "
            f"({_TERMS}) and project / place / OEM / person names as they are")


def normalize_preference(value: Any) -> str:
    """A dropdown value as sent by any client -> a language code or 'auto'. Unknown or missing
    values are 'auto' (older clients send nothing)."""
    text = str(value or "").strip().lower()
    if not text or text == AUTO:
        return AUTO
    if text in CODES:
        return text
    base = text.split("-")[0].split("_")[0]          # 'hi-IN' -> 'hi'
    if base in CODES:
        return base
    for code, (name, native, _) in LANGUAGES.items():
        if text in (name.lower(), native.lower()):
            return code
    return AUTO


# --------------------------------------------------------------------------- #
# Detection (code only -- no LLM call)
# --------------------------------------------------------------------------- #

# Unicode block -> language ('deva' is Hindi or Marathi, decided by their words)
_SCRIPTS = (("deva", 0x0900, 0x097F), ("bn", 0x0980, 0x09FF), ("pa", 0x0A00, 0x0A7F), ("gu", 0x0A80, 0x0AFF),
            ("ta", 0x0B80, 0x0BFF), ("te", 0x0C00, 0x0C7F), ("kn", 0x0C80, 0x0CFF), ("ml", 0x0D00, 0x0D7F))

# Roman-script Hindi words that are not English words (names such as Abhi / Kal are left out:
# technicians are asked about by name)
_HINGLISH_STRONG = set("""
kitne kitna kitni kya kyaa kaise kaisa kaisi kaun kaunsa kaunse kaunsi kis kise kisne kahan kaha kab kyun kyon
hai hain hoga hoge hogi tha thi hua hue hui raha rahe rahi gaya gaye gayi lage lagaye laga chalega bana bane
batao btao bataiye bataye bata dikhao dikhaiye dikha karo kariye kijiye samjhao samjhaiye samjha bolo likho
mein mai sabse zyada jyada jada pichhle pichle pichla pichhla agle agla aaj nahi nahin nhi bhi aur sirf saare sare
wala wale wali waala liye baare mujhe hume humein hamein hamko mujhko yeh ye woh vo unka uska iska inka
chahiye milega milenge abhitak kitno konsa konse kon kyu
""".split())
_HINGLISH_WEAK = {"ka", "ki", "ke", "ko", "se", "me", "par", "tak"}
_ENGLISH_HINTS = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "show", "give", "and", "of", "for",
                  "by", "which", "list", "tell", "explain", "compare", "many", "much", "total", "please", "why",
                  "who", "where", "when", "does", "do", "can", "my", "our", "from", "with", "about", "between"}
_MARATHI = ("आहे", "आहेत", "किती", "मध्ये", "काय", "नाही", "होते", "आणि", "सांगा", "दाखवा", "झाले", "करा")
_HINDI_DEVA = ("है", "हैं", "में", "कितने", "कितना", "कितनी", "क्या", "बताओ", "बताइए", "दिखाओ", "का", "की", "के")


def _script_counts(text: str) -> tuple[Counter, int]:
    counts: Counter = Counter()
    latin = 0
    for ch in text:
        o = ord(ch)
        if o in (0x0964, 0x0965):
            continue  # danda '।' / '॥': in the Devanagari block, but Bengali, Punjabi ... end sentences with it too
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            latin += 1
            continue
        for name, lo, hi in _SCRIPTS:
            if lo <= o <= hi:
                counts[name] += 1
                break
    return counts, latin


def script_of(text: str) -> str:
    """'latin', 'deva', 'bn', 'ta' ... -- the dominant script of the text's letters ('' if none)."""
    counts, latin = _script_counts(text or "")
    indic = sum(counts.values())
    if indic and indic >= 0.2 * (indic + latin):     # 'काशी में total MI count बताओ' is Devanagari
        return counts.most_common(1)[0][0]
    return "latin" if latin else ""


def detect(text: str) -> str | None:
    """The language a message is written in, or None when there is nothing to go by
    ('SAT?', 'AL1151970', 'Agra ka?') -- resolve() then keeps the conversation's language."""
    text = str(text or "")
    script = script_of(text)
    if script == "deva":
        marathi = sum(w in text for w in _MARATHI)
        return "mr" if marathi > sum(w in text for w in _HINDI_DEVA) else "hi"
    if script and script != "latin":
        return script
    if not script:
        return None
    words = re.findall(r"[a-z]+", text.lower())
    strong = sum(w in _HINGLISH_STRONG for w in words)
    weak = sum(w in _HINGLISH_WEAK for w in words)
    if strong >= 1 or weak >= 2:
        return "hinglish"
    if len(words) >= 3 or _ENGLISH_HINTS & set(words):
        return "en"
    return None


# Explicit requests: 'in Hindi', 'into English', 'Hindi mein', 'हिंदी में', 'Tamil language' ...
_NAMES: dict[str, tuple[str, ...]] = {
    "en": ("english", "angrezi", "angreji", "अंग्रेज़ी", "अंग्रेजी", "इंग्लिश"),
    "hi": ("hindi", "हिंदी", "हिन्दी"),
    "hinglish": ("hinglish", "हिंग्लिश"),
    "bn": ("bengali", "bangla", "বাংলা", "बांग्ला", "बंगाली"),
    "mr": ("marathi", "मराठी"),
    "ta": ("tamil", "தமிழ்", "तमिल"),
    "te": ("telugu", "తెలుగు", "तेलुगु"),
    "gu": ("gujarati", "ગુજરાતી", "गुजराती"),
    "pa": ("punjabi", "panjabi", "ਪੰਜਾਬੀ", "पंजाबी"),
    "kn": ("kannada", "ಕನ್ನಡ", "कन्नड़"),
    "ml": ("malayalam", "മലയാളം", "मलयालम"),
}
_AFTER = r"(?:mein|me|main|mai|mei|language|bhasha|भाषा|में|मे|मध्ये)(?![a-z])"
_REQUEST_RES = [
    (code, re.compile(
        rf"(?:(?<![a-z])(?:in|into|to)\s+{re.escape(name)}(?![a-z])(?!\s+nadu)"   # 'in Tamil' (not 'in Tamil Nadu')
        rf"|(?<![a-z]){re.escape(name)}(?![a-z])\s*{_AFTER})", re.IGNORECASE))
    for code, names in _NAMES.items() for name in names
]


def explicit_request(text: str) -> str | None:
    """The language the message explicitly asks the answer to be in (the last one named wins:
    'Hindi mein nahi, English mein batao' -> en), else None."""
    best: tuple[int, str] | None = None
    for code, pattern in _REQUEST_RES:
        for m in pattern.finditer(str(text or "")):
            if best is None or m.start() > best[0]:
                best = (m.start(), code)
    return best[1] if best else None


# Words a pure "say it in <language>" message may contain besides the language request itself
_SWITCH_FILLER = set("""
explain translate answer reply respond write say tell repeat convert give show it this that the same above previous last
your answer response again please pls plz now can could you me do in into to language bhasha mein me main mai mei
isko isse ise ye yeh is wahi vahi upar wala wali batao btao bataiye samjhao samjhaiye bolo likho karo kariye kar do de
dijiye dena
""".split()) | {"इसे", "यह", "ये", "इसको", "बताओ", "समझाओ", "बताइए", "में", "करो", "कीजिए", "दो", "भाषा"}


def pure_switch(text: str) -> str | None:
    """'Explain this in English' / 'Hindi mein batao' -- a message that ONLY asks for the previous
    answer in another language: the language code, else None. Such a message re-presents the
    last answer (translation layer) instead of running a new query."""
    target = explicit_request(text)
    if not target:
        return None
    rest = str(text).lower()
    for name in _NAMES[target]:
        rest = rest.replace(name.lower(), " ")
    # split on spaces / punctuation only: \w drops Indic vowel signs and would cut words apart
    words = [w for w in re.findall(r"[^\s,.!?;:।'\"()\-]+", rest) if not w.isdigit()]
    return target if all(w in _SWITCH_FILLER for w in words) else None


def forced(message: str, preference: str) -> str | None:
    """Priorities 1-2 (known before any LLM call): an explicit request, else a set dropdown."""
    pref = normalize_preference(preference)
    return explicit_request(message) or (pref if pref != AUTO else None)


def _previous_language(prior_turns: list[dict] | None) -> str | None:
    for turn in reversed(prior_turns or []):
        lang = (turn.get("context") or {}).get("language")
        if lang in CODES:
            return lang
    return None


def resolve(message: str, preference: str = AUTO, llm_language: str | None = None,
            prior_turns: list[dict] | None = None) -> tuple[str, str]:
    """(target language, detected language) for one message -- see the module docstring."""
    detected = detect(message)
    script = script_of(message)
    words = set(re.findall(r"[a-z]+", str(message or "").lower()))
    if (script == "latin" and detected == "en" and llm_language in ("hinglish", "hi")
            and (words & _HINGLISH_WEAK or not words & _ENGLISH_HINTS)):
        # Roman script: Hinglish when EITHER reading sees it -- the word list misses rarer
        # Hindi words, and the triage LLM was seen live calling 'kitna hai' English.
        # Hindi written in Roman script IS Hinglish. But not against English grammar words
        # with no Hindi word at all: 'how many meters installed in kashi in july 2026' was
        # called Hinglish by the LLM live (place names read as Hindi).
        detected = "hinglish"
    elif script == "deva" and llm_language in ("hi", "mr"):
        detected = llm_language
    if detected is None:
        detected = _previous_language(prior_turns) or "en"
    return forced(message, preference) or detected, detected


# Languages the answer / explain / triage models write reliably themselves. The others are
# written in English and go through the translation layer (number + script guards, English
# kept if it fails): seen live, a directly written Tamil answer mixed Tamil, Korean and Hindi.
DIRECT = frozenset({"en", "hi", "hinglish"})


def writing_language(target: str) -> str:
    """The language the answer LLM is asked to write in for a target language."""
    return target if target in DIRECT else "en"


def written_in(text: str, claimed: str) -> str:
    """The language `text` is really written in, given the one it was asked for: `claimed` when
    the text bears it out, else what it is (models ignore the LANGUAGE line now and then --
    seen live: an English reply to a Hindi request). A mismatch then goes through translation."""
    text = ascii_digits(text)
    if claimed in _SCRIPT_OF and _SCRIPT_OF[claimed] != "latin":
        if in_script(text, _SCRIPT_OF[claimed]):
            return claimed
        found = detect(text)  # mixed scripts still 'detect' as the majority one: not good enough
        return found if found and found != claimed else "en"
    found = detect(text)
    if claimed == "hinglish":
        return "hinglish" if found in ("hinglish", None) else (found or "en")
    return claimed if found in (claimed, None) else found


# English grammar words: an English SENTENCE left in the text has them, a list of English
# technical terms ('WFM (meter installation, SAT, O&M tickets)') does not.
_ENGLISH_GRAMMAR = {"the", "is", "are", "was", "were", "you", "so", "far", "has", "have", "which", "this", "that",
                    "with", "of", "to", "in", "for", "it", "be", "been", "there", "their", "will", "can"}


def in_script(text: str, script: str) -> bool:
    """True when `text` is written in `script`. English technical terms and names are fine;
    another script (Kana, Hangul, a second Indic script) or English sentences are not."""
    counts, latin = _script_counts(text)
    other = sum(1 for ch in text if ch.isalpha() and ord(ch) > 0x24F
                and not any(lo <= ord(ch) <= hi for _, lo, hi in _SCRIPTS))  # Hangul, CJK ...
    own = counts.get(script, 0)
    foreign = sum(v for k, v in counts.items() if k != script)
    english = sum(w in _ENGLISH_GRAMMAR for w in re.findall(r"[a-z]+", text.lower()))
    # Seen live: Tamil with Katakana ('ப்ரஜெクト') and half-English sentences ('... count in Kashi
    # is 1,386,445 so far') were wrong; Hindi with 59% Latin letters from a list of terms was right.
    return (own > 0 and other == 0 and foreign <= 0.02 * own and own >= 0.2 * (own + latin)
            and english <= 2)


# --------------------------------------------------------------------------- #
# Translation layer
# --------------------------------------------------------------------------- #

_NATIVE_DIGITS = "०१२३४५६७८९০১২৩৪৫৬৭৮৯੦੧੨੩੪੫੬੭੮੯૦૧૨૩૪૫૬૭૮૯௦௧௨௩௪௫௬௭௮௯౦౧౨౩౪౫౬౭౮౯೦೧೨೩೪೫೬೭೮೯൦൧൨൩൪൫൬൭൮൯"
_TO_ASCII = str.maketrans(_NATIVE_DIGITS, "0123456789" * (len(_NATIVE_DIGITS) // 10))


def ascii_digits(text: str) -> str:
    """Native-script digits (४२, ৪২ ...) as 0-9, so number checks and the cards read the same figures."""
    return str(text or "").translate(_TO_ASCII)


_TOKEN_RE = re.compile(r"[A-Za-z]*\d[A-Za-z\d]*(?:[.,:/-]\d+)*")


def protected_tokens(text: str) -> Counter:
    """Every number / ID in a text (commas dropped: 42,14,678 and 4,214,678 are the same value)."""
    return Counter(t.replace(",", "").upper().rstrip(".") for t in _TOKEN_RE.findall(ascii_digits(text)))


def values_preserved(original: str, translated: str) -> bool:
    """True when the translation has exactly the numbers / IDs of the original -- none added,
    none dropped, none changed (repeats are allowed: word order differs between languages)."""
    return set(protected_tokens(original)) == set(protected_tokens(translated))


_TRANSLATE_SYSTEM = f"""You translate the presentation text of an answer from GMR's smart-meter analytics chatbot.
Translate every value of ITEMS into TARGET and reply with JSON only: {{"items": {{"<same key>": "<translated text>"}}}} -- the same keys, nothing else.
- Keep every number, percentage, amount, date number, count and ID (meter serials, consumer numbers, SAT-13, ticket ids) exactly as written, in Western digits 0-9. Never add, drop, round, convert or recompute a number, and never write a number in words. Before replying, check that every number and ID of each original value is in its translation.
- Write in the script TARGET names (Hindi / Marathi = Devanagari, Tamil = Tamil script ...); only Hinglish and English use Roman letters.
- Keep technical terms and abbreviations ({_TERMS}) and project / place / OEM / person names as they are.
- Translate the meaning naturally, the way a colleague would say it -- not word for word. Keep the same facts, order and markdown / bullet structure; add nothing.
- Keys kpis.* / charts.* / tables.* / col.* are short labels and column headers: translate them as short labels (a few words, no sentence). Keys followups.* are questions a user can ask next: translate them as natural questions, the way that user would type them.
- If a value is already in TARGET, return it unchanged.
- {injection.UNTRUSTED_RULE}"""

# The script a translation into each language must be written in (script_of() names)
_SCRIPT_OF = {"hi": "deva", "mr": "deva", "bn": "bn", "ta": "ta", "te": "te", "gu": "gu", "pa": "pa", "kn": "kn",
              "ml": "ml", "en": "latin", "hinglish": "latin"}

_TEXT_FIELDS = ("summary",)
_LIST_FIELDS = ("insights", "warnings")
_CACHE: OrderedDict[str, dict[str, str]] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 500

# Labels on the cards, charts and tables (written in English by the planner) are shown in these
# languages; English and Hinglish keep the English labels.
_LABEL_LANGS = CODES - {"en", "hinglish"}
# (block, field it is translated from -> display field it is shown from). The source field is
# never overwritten, so a stored answer can be re-translated (or shown in English) at any time.
_LABEL_FIELDS = {"kpis": {"label": "display_label", "description": "display_description"},
                 "charts": {"title": "display_title", "description": "display_description"},
                 "tables": {"title": "display_title", "heading": "display_heading"}}


def _text_items(response: dict) -> dict[str, str]:
    """The written answer: summary, insights, warnings."""
    items = {f: str(response.get(f) or "") for f in _TEXT_FIELDS if str(response.get(f) or "").strip()}
    for f in _LIST_FIELDS:
        for i, v in enumerate(response.get(f) or []):
            if str(v).strip():
                items[f"{f}.{i}"] = str(v)
    return items


def _label_items(response: dict, target: str) -> dict[str, str]:
    """Card / chart / table labels and table column headers (target in _LABEL_LANGS), and the
    suggested follow-up questions (any target but English)."""
    items: dict[str, str] = {}
    # Acronym-only labels ('OEM', 'DTR', 'SAT %') stay as they are: nothing to translate.
    worded = lambda text: bool(re.search(r"[a-z]", str(text)))
    if target in _LABEL_LANGS:
        for block, fields in _LABEL_FIELDS.items():
            for i, obj in enumerate(response.get(block) or []):
                for src in fields:
                    if isinstance(obj, dict) and str(obj.get(src) or "").strip() and worded(obj[src]):
                        items[f"{block}.{i}.{src}"] = str(obj[src])
        for table in response.get("tables") or []:
            for col in table.get("columns") or []:
                if worded(col):
                    items.setdefault(f"col.{col}", str(col).replace("_", " "))
    if target != "en":
        for i, q in enumerate(response.get("suggested_followups") or []):
            items[f"followups.{i}"] = str(q)
    return items


def _clear_display(out: dict) -> None:
    """Drop display translations a stored answer may carry from an earlier language."""
    for block, fields in _LABEL_FIELDS.items():
        for obj in out.get(block) or []:
            if isinstance(obj, dict):
                for display in fields.values():
                    obj.pop(display, None)
                if block == "tables":
                    obj.pop("column_labels", None)
    out.pop("suggested_followup_labels", None)


def _apply(out: dict, done: dict[str, str]) -> None:
    for k, text in done.items():
        head, _, rest = k.partition(".")
        if head == "col":
            for table in out.get("tables") or []:
                if rest in (table.get("columns") or []):
                    table.setdefault("column_labels", {})[rest] = text
        elif head == "followups":
            labels = out.setdefault("suggested_followup_labels", list(out.get("suggested_followups") or []))
            labels[int(rest)] = text
        elif head in _LABEL_FIELDS:
            idx, _, src = rest.partition(".")
            out[head][int(idx)][_LABEL_FIELDS[head][src]] = text
        elif rest:
            out[head][int(rest)] = text
        else:
            out[head] = text


def _translate_items(items: dict[str, str], target: str) -> dict[str, str] | None:
    """{key: translated text} for the keys that passed every check; None if the call failed.
    Items the model left out of its reply or wrote in the wrong script get one more, smaller
    call (seen live: Marathi / Bengali table titles silently missing from the reply)."""
    key = hashlib.sha1(json.dumps([target, items], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return dict(_CACHE[key])
    out = _translate_call(items, target)
    if out is None:
        return None
    missing = {k: v for k, v in items.items() if k not in out}
    if missing and len(missing) < len(items):
        log.info("translate: %d item(s) missing or rejected -- asking once more: %s", len(missing), sorted(missing)[:6])
        out.update(_translate_call(missing, target) or {})
    with _CACHE_LOCK:
        _CACHE[key] = dict(out)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return out


def _translate_call(items: dict[str, str], target: str) -> dict[str, str] | None:
    """One translate call: {key: text} for the items whose translation passed the checks."""
    prompt = (f"TARGET: {prompt_line(target)}\n\nITEMS:\n"
              + injection.wrap(json.dumps(items, ensure_ascii=False, indent=1), "answer"))
    # Output tokens per input character: an Indic script costs several tokens a character (seen
    # live: a Telugu reply cut off at 2,048 tokens = broken JSON = no translation at all).
    chars = len(json.dumps(items, ensure_ascii=False))
    max_tokens = min(8000, 600 + int(chars * (4 if _SCRIPT_OF.get(target, "latin") != "latin" else 1.5)))
    res = None
    for attempt in range(2):  # a failed call (bad / cut-off JSON, timeout, busy) is tried once more
        # Outside a question (dropdown change) the call gets its own deadline, so a slow provider
        # never hangs the request; inside one, the question's own budget applies.
        if llm_gateway.remaining_seconds() is None:
            with llm_gateway.question_scope(new_trace_id(), config.QUESTION_LLM_BUDGET_SECONDS):
                res = llm_gateway.chat_json("translate", prompt, system=_TRANSLATE_SYSTEM, max_tokens=max_tokens)
        else:
            res = llm_gateway.chat_json("translate", prompt, system=_TRANSLATE_SYSTEM, max_tokens=max_tokens)
        if res is not None and isinstance(res.data, dict) and not isinstance(res.data.get("items"), dict) \
                and set(res.data) & set(items):
            res.data = {"items": res.data}  # the keys without the {"items": ...} wrapper (seen live)
        if res is not None and isinstance(res.data, dict) and isinstance(res.data.get("items"), dict):
            break
        if attempt == 0:
            log.warning("translate: %s -- trying once more", "no reply" if res is None else "malformed reply")
    if res is None:
        return None
    got = res.data.get("items") if isinstance(res.data, dict) else None
    if not isinstance(got, dict):
        log.warning("translate: malformed reply from %s -- original kept", res.model)
        return None
    out: dict[str, str] = {}
    for k, original in items.items():
        text = ascii_digits(str(got.get(k) or "")).strip()
        if not text:
            log.warning("translate: %s missing from the reply", k)
            continue
        if not values_preserved(original, text):
            log.warning("translate: %s changed numbers/IDs (%s -> %s) -- original kept", k,
                        sorted(protected_tokens(original)), sorted(protected_tokens(text)))
            continue
        script = _SCRIPT_OF.get(target, "latin")
        if not (in_script(text, script) if script != "latin" else script_of(text) == "latin"):
            # Seen live: 'hi' came back as Roman-script Hinglish. Wrong script = not translated.
            log.warning("translate: %s is not in %s script -- original kept", k, target)
            continue
        out[k] = text
    return out


def translate_response(response: dict, target: str) -> dict:
    """A copy of `response` presented in `target`: the written answer (summary, insights,
    warnings) unless it is already in that language, and -- as display fields next to the
    originals -- card / chart / table labels, column headers and follow-up questions. Values,
    table cells, chart data, citations and debug are never touched. One LLM call at most.
    `translation.status`: 'ok' | 'partial' | 'failed' (the original answer is shown) |
    'skipped' (nothing to translate)."""
    out = copy.deepcopy(response)
    target = target if target in CODES else "en"
    _clear_display(out)
    current = out.get("answer_language") or detect(out.get("summary") or "")
    text = _text_items(out) if current != target else {}
    labels = _label_items(out, target)
    if not text and not labels:
        out.update(answer_language=current or target, translation={"target": target, "status": "skipped"})
        return out
    try:
        done = _translate_items({**text, **labels}, target)
    except Exception:  # never let translation break an answer
        log.exception("translate: failed")
        done = None
    if text and (not done or ("summary" in text and "summary" not in done)):
        # No usable summary = no translation: insights in Bengali under an English summary (seen
        # live) read worse than the original answer as a whole.
        out["translation"] = {"target": target, "status": "failed"}
        return out
    _apply(out, done or {})
    if text:
        out["answer_language"] = target
        missing = [k for k in text if k not in done]
        out["translation"] = {"target": target, "status": "partial" if missing else "ok"}
    else:  # the answer was already in the target language; only its labels were translated
        out.update(answer_language=current, translation={"target": target, "status": "ok" if done else "skipped"})
    return out
