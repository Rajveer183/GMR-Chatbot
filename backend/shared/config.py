"""Runtime configuration -- every setting comes from the environment / .env.

Nothing here has a built-in default: a missing variable stops the backend at
startup with one error naming every missing name, so what runs is always
exactly what .env says. See .env for what each setting does.
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except Exception:  # dotenv is optional when the variables are set another way
    pass

_MISSING: list[str] = []

def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        _MISSING.append(name)
        return ""
    return value.strip()

def _int(name: str) -> int:
    return int(_env(name) or 0)

def _float(name: str) -> float:
    return float(_env(name) or 0)

def _flag(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}

def _provider_ints(name: str) -> dict[str, int]:
    """'gemini:10,groq:30' -> {"gemini": 10, "groq": 30} (missing provider = 0 = unlimited)."""
    out = {}
    for item in _env(name).split(","):
        provider, _, value = item.strip().partition(":")
        if provider and value.strip():
            out[provider.strip()] = int(value)
    return out

def _models(name: str) -> list[tuple[str, str]]:
    """'provider:model,provider:model' -> [(provider, model), ...] in order."""
    out = []
    for item in _env(name).split(","):
        provider, _, model = item.strip().partition(":")
        if provider and model:
            out.append((provider.strip(), model.strip()))
    return out

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
# Test runs (python -m unittest / pytest / a test_*.py script) log to outputs/logs/test/:
# their fake tools, mocked LLMs and forced failures must not show up in the real
# backend.log / questions.jsonl / llm_calls.jsonl that latency, cost and error metrics use.
_ENTRY = sys.argv[0] if sys.argv and sys.argv[0] else ""
TESTING = (os.path.basename(_ENTRY).startswith("test_")
           or any(t in _ENTRY.lower() for t in ("unittest", "pytest", "py.test")))  # '-m unittest' sets argv[0] to 'python -m unittest'
LOG_DIR = os.path.join(OUTPUT_DIR, "logs", "test") if TESTING else os.path.join(OUTPUT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# --- Groq (free tier) --------------------------------------------------------
GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_BASE_URL = _env("GROQ_BASE_URL")
LLM_ENABLED = _flag("ENABLE_LLM") and bool(GROQ_API_KEY)
GROQ_CALL_TIMEOUT = _float("GROQ_CALL_TIMEOUT")
GROQ_DEFAULT_TPM = _int("GROQ_DEFAULT_TPM")      # until a response reports the real limit

# Text-to-speech (the answer's Play button)
GROQ_TTS_MODEL = _env("GROQ_TTS_MODEL")
GROQ_TTS_VOICE = _env("GROQ_TTS_VOICE")
GROQ_TTS_SPEED = _env("GROQ_TTS_SPEED")
GROQ_TTS_MAX_CHARS = _int("GROQ_TTS_MAX_CHARS")
GROQ_TTS_TIMEOUT = _float("GROQ_TTS_TIMEOUT")
GROQ_TTS_CACHE_SIZE = _int("GROQ_TTS_CACHE_SIZE")

# --- NVIDIA NIM (free tier) --------------------------------------------------
NVIDIA_API_KEY = _env("NVIDIA_API_KEY")
NVIDIA_BASE_URL = _env("NVIDIA_BASE_URL")
NVIDIA_ENABLED = _flag("ENABLE_NVIDIA") and bool(NVIDIA_API_KEY)
NVIDIA_TIMEOUT_SECONDS = _float("NVIDIA_TIMEOUT")
NVIDIA_THINKING = _flag("NVIDIA_THINKING")            # nemotron reasoning mode on/off
NVIDIA_MIN_MAX_TOKENS = _int("NVIDIA_MIN_MAX_TOKENS")  # floor for max_tokens (reasoning counts)

# --- Extra OpenAI-compatible providers (free tiers) -------------------------
# LLM_EXTRA_PROVIDERS=cerebras,mistral  ->  for each NAME: NAME_BASE_URL and
# NAME_API_KEY (upper-case), then use "name:model" in the LLM_MODELS_* lists.
LLM_EXTRA_PROVIDERS = [p.strip().lower() for p in _env("LLM_EXTRA_PROVIDERS").split(",") if p.strip()]
PROVIDERS: dict[str, tuple[str, str]] = {"groq": (GROQ_BASE_URL, GROQ_API_KEY), "nvidia": (NVIDIA_BASE_URL, NVIDIA_API_KEY)}
PROVIDER_REASONING_EFFORT: dict[str, str] = {}   # NAME_REASONING_EFFORT ('' = don't send)
for _name in LLM_EXTRA_PROVIDERS:
    PROVIDERS[_name] = (_env(f"{_name.upper()}_BASE_URL").rstrip("/"), _env(f"{_name.upper()}_API_KEY"))
    PROVIDER_REASONING_EFFORT[_name] = _env(f"{_name.upper()}_REASONING_EFFORT")
EXTRA_PROVIDER_TIMEOUT = _float("EXTRA_PROVIDER_TIMEOUT")   # per-call timeout for extra providers

# --- LLM gateway: model order per pipeline stage ("provider:model,...") ------
# The gateway uses the first model in the list whose free-tier budget can take
# the call now, and moves down the list on any failure.
LLM_MODELS_TRIAGE = _models("LLM_MODELS_TRIAGE")    # understand the question / route
LLM_MODELS_SQL = _models("LLM_MODELS_SQL")          # write + repair the SQL
LLM_MODELS_ANSWER = _models("LLM_MODELS_ANSWER")    # write the answer from the rows
LLM_MODELS_EXPLAIN = _models("LLM_MODELS_EXPLAIN")  # explain a KPI / term
LLM_MAX_REPAIRS = _int("LLM_MAX_REPAIRS")
AGENT_MAX_STEPS = _int("AGENT_MAX_STEPS")                     # nodes one question may run before it is stopped

# --- LLM rate limiting (shared/rate_limiter.py, enforced in llm_gateway._admit) ---
def _pricing(name: str) -> dict[str, tuple[float, float]]:
    """'groq=0.15/0.6,nvidia:nvidia/x=0/0' -> {"groq": (0.15, 0.6), ...}: USD per 1M input / output
    tokens, by provider or by "provider:model" (the more specific key wins)."""
    out = {}
    for item in _env(name).split(","):
        key, _, prices = item.strip().partition("=")
        if key and "/" in prices:
            p_in, _, p_out = prices.partition("/")
            out[key.strip()] = (float(p_in or 0), float(p_out or 0))
    return out

def _floats(name: str) -> dict[str, float]:
    out = {}
    for item in _env(name).split(","):
        key, _, value = item.strip().partition("=")
        if key and value.strip():
            out[key.strip()] = float(value)
    return out

def _decoding(name: str) -> dict[str, tuple[float, float, int]]:
    """'sql=0/1/0,answer=0.2/0.95/0' -> {task: (temperature, top_p, top_k)}; top_k 0 = not sent."""
    out = {}
    for item in _env(name).split(","):
        task, _, vals = item.strip().partition("=")
        parts = (vals.split("/") + ["1", "0"])[:3] if vals else []
        if task and parts:
            out[task.strip()] = (float(parts[0] or 0), float(parts[1] or 1), int(float(parts[2] or 0)))
    return out

LLM_DECODING = _decoding("LLM_DECODING")                       # per-task temperature / top_p / top_k
LLM_TOP_K_PROVIDERS = {p.strip() for p in _env("LLM_TOP_K_PROVIDERS").split(",") if p.strip()}
TOKEN_FACTORS = _floats("TOKEN_FACTORS")                       # provider tokens per tiktoken token (starting values)
LLM_PRICING = _pricing("LLM_PRICING")                          # estimated cost per model / provider
QUESTION_MAX_COST_USD = _float("QUESTION_MAX_COST_USD")         # per-question estimated cost cap (0 = off)
# Prompt compaction switches for the SQL prompt (comma list; empty = full prompt). Each one
# drops content the prompt already carries elsewhere:
#   rule_examples   topic rules without their embedded example SQL (verified examples section stays)
#   chart_examples  chart guide without the per-type example JSON lines
#   table_eg        table guide without its example question per table
#   hint_values     column hints without values the SCHEMA already lists for that column
#   lean_repair     repair calls without chart guide / past questions / examples
PROMPT_COMPACT = {s.strip() for s in _env("PROMPT_COMPACT").split(",") if s.strip()}
# Service targets reported by /api/metrics (gmr_agents/metrics.py): met / not met per window.
SLO_P50_MS = _float("SLO_P50_MS")
SLO_P95_MS = _float("SLO_P95_MS")
SLO_TTFT_P50_MS = _float("SLO_TTFT_P50_MS")
SLO_MAX_REFUSAL_RATE = _float("SLO_MAX_REFUSAL_RATE")
SLO_MAX_BUSY_RATE = _float("SLO_MAX_BUSY_RATE")
SLO_MAX_REPAIR_RATE = _float("SLO_MAX_REPAIR_RATE")
SLO_MAX_TOKENS_PER_QUESTION = _float("SLO_MAX_TOKENS_PER_QUESTION")
QUESTION_MAX_DB_QUERIES = _int("QUESTION_MAX_DB_QUERIES")       # per-question cap on database queries (0 = off)
AGENT_SQL_TIMEOUT_MS = _int("AGENT_SQL_TIMEOUT_MS")            # per agent SQL statement (Postgres)
QUESTION_MAX_LLM_CALLS = _int("QUESTION_MAX_LLM_CALLS")        # per-question cap on LLM requests sent (0 = off)
QUESTION_MAX_TOKENS = _int("QUESTION_MAX_TOKENS")              # per-question cap on prompt + completion tokens (0 = off)
MAX_CONTEXT_TOKENS = _int("MAX_CONTEXT_TOKENS")                # hard cap per LLM call (system + prompt + output)
MAX_HISTORY_TOKENS = _int("MAX_HISTORY_TOKENS")                # conversation memory block in a prompt
MAX_TOOL_RESULT_TOKENS = _int("MAX_TOOL_RESULT_TOKENS")        # query rows shown to the answer writer
PROMPT_ORDER = _env("PROMPT_ORDER").lower()                    # "default" | "edges" (lost-in-the-middle ordering)
SCHEMA_SCOPE = _env("SCHEMA_SCOPE").lower()                    # "topics" | "lean" (only the tables triage named)
CONTEXT_BUDGET_SQL_TOKENS = _int("CONTEXT_BUDGET_SQL_TOKENS")  # planner prompt budget (system + prompt), trimmed by priority
CHUNK_MIN_TOKENS = _int("CHUNK_MIN_TOKENS")                    # data-aware chunking: merge smaller pieces
CHUNK_MAX_TOKENS = _int("CHUNK_MAX_TOKENS")                    # split list sections between items above this
GROUNDING_KPI_CHUNKS = _int("GROUNDING_KPI_CHUNKS")            # kpi_definitions chunks per explain answer
GROUNDING_MIN_RELEVANCE = _float("GROUNDING_MIN_RELEVANCE")    # cosine; evidence below it is not shown to the LLM
GROUNDING_TOP_K = _int("GROUNDING_TOP_K")                      # evidence chunks per explain answer
GROUNDING_MAX_CONTEXT_CHARS = _int("GROUNDING_MAX_CONTEXT_CHARS")  # evidence text cap in the prompt
LLM_RATE_WINDOW_SECONDS = _float("LLM_RATE_WINDOW_SECONDS")    # sliding window length
LLM_RATE_MAX_WAIT_SECONDS = _float("LLM_RATE_MAX_WAIT_SECONDS")  # longest wait for a slot before trying the next model
LLM_BUSY_WAIT_SECONDS = _float("LLM_BUSY_WAIT_SECONDS")  # every model cooling: wait this long for the first to free up
LLM_RPM_LIMITS = _provider_ints("LLM_RPM_LIMITS")              # requests per window, per model, by provider
LLM_TPM_LIMITS = _provider_ints("LLM_TPM_LIMITS")              # tokens per window, per model, by provider
LLM_RATE_BACKOFF_BASE_SECONDS = _float("LLM_RATE_BACKOFF_BASE_SECONDS")  # first 429 cooldown without Retry-After
QUESTION_LLM_BUDGET_SECONDS = _float("QUESTION_LLM_BUDGET")
LLM_MIN_CALL_SECONDS = _float("LLM_MIN_CALL_SECONDS")      # don't start a call with less time left
LLM_MAX_REFILL_WAIT = _float("LLM_MAX_REFILL_WAIT")        # max wait for a Groq bucket to refill
LLM_FAILURE_COOLDOWN = _float("LLM_FAILURE_COOLDOWN")      # rest a model after an error/timeout
LLM_TIMEOUT_COOLDOWN = _float("LLM_TIMEOUT_COOLDOWN")          # cooldown after a timeout (overloaded model)
LLM_MAX_RATE_LIMIT_COOLDOWN = _float("LLM_MAX_RATE_LIMIT_COOLDOWN")
LLM_MAX_DAILY_LIMIT_COOLDOWN = _float("LLM_MAX_DAILY_LIMIT_COOLDOWN")  # after a tokens-per-day 429

# Token limits per stage (output tokens, reasoning included)
LLM_MAX_TOKENS_TRIAGE = _int("LLM_MAX_TOKENS_TRIAGE")
LLM_MAX_TOKENS_SQL = _int("LLM_MAX_TOKENS_SQL")
LLM_MAX_TOKENS_ANSWER = _int("LLM_MAX_TOKENS_ANSWER")
LLM_MAX_TOKENS_EXPLAIN = _int("LLM_MAX_TOKENS_EXPLAIN")

# --- Pipeline sizes ----------------------------------------------------------
SCHEMA_TOKEN_BUDGET = _int("SCHEMA_TOKEN_BUDGET")   # schema text in the SQL prompt
ANSWER_PREVIEW_ROWS = _int("ANSWER_PREVIEW_ROWS")   # result rows the answer writer sees
TABLE_MAX_ROWS = _int("TABLE_MAX_ROWS")             # rows returned in the answer table
CHART_MAX_POINTS = _int("CHART_MAX_POINTS")
EXAMPLES_PER_QUESTION = _int("EXAMPLES_PER_QUESTION")  # verified examples in the SQL prompt
ROUTED_EXAMPLES = _int("ROUTED_EXAMPLES")              # verified examples for a router (near-exact) question; 0 = EXAMPLES_PER_QUESTION

# --- Hybrid retrieval: BM25 + local embeddings (gmr_agents/schema/retrieval.py) ---
RETRIEVAL_EMBED_ENABLED = _flag("RETRIEVAL_EMBED_ENABLED")     # 0 = BM25 only
RETRIEVAL_EMBED_MODEL = _env("RETRIEVAL_EMBED_MODEL")          # fastembed model name (runs locally)
RETRIEVAL_MODEL_DIR = _env("RETRIEVAL_MODEL_DIR")              # model download cache, relative to backend/
RETRIEVAL_BM25_K1 = _float("RETRIEVAL_BM25_K1")
RETRIEVAL_BM25_B = _float("RETRIEVAL_BM25_B")
RETRIEVAL_RRF_K = _float("RETRIEVAL_RRF_K")                    # reciprocal-rank-fusion constant
RETRIEVAL_BM25_WEIGHT = _float("RETRIEVAL_BM25_WEIGHT")
RETRIEVAL_EMBED_WEIGHT = _float("RETRIEVAL_EMBED_WEIGHT")
RETRIEVAL_MIN_SIMILARITY = _float("RETRIEVAL_MIN_SIMILARITY")  # embedding hits below this cosine are ignored
KNOWLEDGE_FILE = _env("KNOWLEDGE_FILE")                        # JSON under outputs/: indexed texts + vectors, artifacts
SQL_RESULT_FILE = _env("SQL_RESULT_FILE")                      # JSON under outputs/: materialised query results
SQL_RESULT_CACHE_ENABLED = _flag("SQL_RESULT_CACHE_ENABLED")   # local materialised query results
SQL_RESULT_CACHE_MAX_ROWS = _int("SQL_RESULT_CACHE_MAX_ROWS")  # bigger results are not stored
RETRIEVAL_VECTOR_TOP_N = _int("RETRIEVAL_VECTOR_TOP_N")        # nearest stored vectors used per query
TRIAGE_EXAMPLE_MENU_SIZE = _int("TRIAGE_EXAMPLE_MENU_SIZE")    # retrieved examples listed in the triage prompt
TRIAGE_TOPIC_FULL = _int("TRIAGE_TOPIC_FULL")                  # best-ranked topics shown with description (rest: title); 0 = all
CHART_GUIDE_FULL = _int("CHART_GUIDE_FULL")                    # best-ranked chart types shown with 'when' + example (rest: name); 0 = all
RETRIEVAL_QUERY_PREFIX = _env("RETRIEVAL_QUERY_PREFIX")       # e.g. "search_query: " for nomic models
RETRIEVAL_DOC_PREFIX = _env("RETRIEVAL_DOC_PREFIX")           # e.g. "search_document: " for nomic models
FUZZY_ENABLED = _flag("FUZZY_ENABLED")                        # rapidfuzz typo fix + near-exact question match
FUZZY_MIN_RATIO = _float("FUZZY_MIN_RATIO")                   # 0.85 .. 1 counts as a fuzzy match
RETRIEVAL_QUERY_CACHE_SIZE = _int("RETRIEVAL_QUERY_CACHE_SIZE")    # LRU of query embeddings
RETRIEVAL_RESULT_CACHE_SIZE = _int("RETRIEVAL_RESULT_CACHE_SIZE")  # LRU of search results
RETRIEVAL_ITEMS_TTL = _float("RETRIEVAL_ITEMS_TTL")           # seconds index items are reused before re-reading sources
RETRIEVAL_SQL_WEIGHT = _float("RETRIEVAL_SQL_WEIGHT")         # weight of the SQL-pattern match when ranking examples
SCHEMA_HINT_COLUMNS = _int("SCHEMA_HINT_COLUMNS")             # retrieved columns (+ real values) added to the SQL prompt
PAST_QUESTIONS_TOP_K = _int("PAST_QUESTIONS_TOP_K")           # similar past answered questions (with SQL) in the SQL prompt
PAST_QUESTIONS_MIN_SCORE = _float("PAST_QUESTIONS_MIN_SCORE")  # fused score a past question needs to be shown
PAST_QUESTIONS_MAX = _int("PAST_QUESTIONS_MAX")               # most recent answered questions indexed from the log
ROUTER_ENABLED = _flag("ROUTER_ENABLED")                      # near-exact questions skip the understanding LLM call
ROUTER_MIN_RATIO = _float("ROUTER_MIN_RATIO")                 # fuzzy ratio (0.85..1) a question needs to be routed
VERIFY_ENTITIES = _flag("VERIFY_ENTITIES")                    # verifier: question entities must match the SQL filters
# --- Data skeleton (gmr_agents/schema/data_skeleton.py): read-only picture of every table ---
DATA_SKELETON_SAMPLE_PCT = _float("DATA_SKELETON_SAMPLE_PCT")     # TABLESAMPLE % for Postgres sample rows
DATA_SKELETON_SAMPLE_ROWS = _int("DATA_SKELETON_SAMPLE_ROWS")     # sample rows kept per table
DATA_SKELETON_MAX_COMBOS = _int("DATA_SKELETON_MAX_COMBOS")       # distinct dimension combinations kept per table
DATA_SKELETON_ATHENA_MAX_ROWS = _int("DATA_SKELETON_ATHENA_MAX_ROWS")  # Athena tables up to this size get a date range
DATA_SKELETON_MAX_AGE = _float("DATA_SKELETON_MAX_AGE")           # seconds before the skeleton is rebuilt (background)
VALUE_HINTS = _int("VALUE_HINTS")                                 # matched stored values added to the SQL prompt
TABLE_HINTS = _int("TABLE_HINTS")                                 # table data cards shown to the understanding step
# --- Conversation memory (deep-dive follow-ups) ---
MEMORY_TURNS = _int("MEMORY_TURNS")                     # turns recalled per question
MEMORY_DETAILED_TURNS = _int("MEMORY_DETAILED_TURNS")   # latest turns shown in full (answer, numbers, keys)
MEMORY_ANSWER_CHARS = _int("MEMORY_ANSWER_CHARS")       # characters of each recalled answer

# --- Answer cache (gmr_agents/answer_cache.py) -------------------------------
ANSWER_CACHE_ENABLED = _flag("ANSWER_CACHE_ENABLED")
ANSWER_CACHE_MAX_ENTRIES = _int("ANSWER_CACHE_MAX_ENTRIES")
ANSWER_CACHE_VERSION_TTL = _float("ANSWER_CACHE_VERSION_TTL")  # seconds between data-version checks

# --- Cache pre-warmer (gmr_agents/prewarm.py) --------------------------------
PREWARM_ENABLED = _flag("PREWARM_ENABLED")
PREWARM_QUESTIONS_FILE = _env("PREWARM_QUESTIONS_FILE")   # relative to backend/
PREWARM_TOP_N = _int("PREWARM_TOP_N")                     # most-asked history questions to warm
PREWARM_PAUSE = _float("PREWARM_PAUSE")                   # seconds between warm-up questions
PREWARM_CHECK_SECONDS = _float("PREWARM_CHECK_SECONDS")   # how often to look for a new data version
PREWARM_START_DELAY = _float("PREWARM_START_DELAY")       # wait after startup before the first check
PREWARM_BANK = _flag("PREWARM_BANK")                      # also warm the whole question bank (~420 questions)
PREWARM_MAX_QUESTIONS = _int("PREWARM_MAX_QUESTIONS")     # LLM-answered questions per run (0 = no cap)
PREWARM_SETTLE_SECONDS = _float("PREWARM_SETTLE_SECONDS") # a new version must stay unchanged this long first

# --- Redis (shared/redis_client.py) -------------------------------------------
# Optional cache/session/rate-limit layer -- REDIS_HOST="" or Redis unreachable both
# degrade to the existing local-file / in-memory logic (redis_client.available()).
REDIS_HOST = _env("REDIS_HOST")
REDIS_PORT = _int("REDIS_PORT")
REDIS_DB = _int("REDIS_DB")
REDIS_HISTORY_TTL = _int("REDIS_HISTORY_TTL")                     # seconds a mirrored session survives
REDIS_HISTORY_MAX_MESSAGES = _int("REDIS_HISTORY_MAX_MESSAGES")   # messages kept per session in Redis
REDIS_RESPONSE_CACHE_TTL = _int("REDIS_RESPONSE_CACHE_TTL")       # answer_cache.py Redis-side expiry
REDIS_RATE_LIMIT = _int("REDIS_RATE_LIMIT")                       # requests per REDIS_RATE_WINDOW per session
REDIS_RATE_WINDOW = _int("REDIS_RATE_WINDOW")
# Defined per spec; currently unused -- episodic-memory retrieval (the feature these would
# cap) was intentionally left out of this round (see final report).
REDIS_MAX_MEMORY_CONTEXT = _int("REDIS_MAX_MEMORY_CONTEXT")
REDIS_MAX_RETRIEVED_MEMORIES = _int("REDIS_MAX_RETRIEVED_MEMORIES")

if _MISSING:
    raise RuntimeError("Missing settings in backend/.env (no built-in defaults): " + ", ".join(_MISSING))
