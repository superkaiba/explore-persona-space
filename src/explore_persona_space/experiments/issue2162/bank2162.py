"""Issue #2162 — minimal-pair information-type bank (plan v3 §4.1/§4.2).

21 information types x 3 distinct values (3 directed value-pairs on the
registered cycle v1->v2->v3->v1) x 12 carriers, plus 18 crossed cells
(4 conflict, 8 recency, 6 load) = 39 type-cells, 1,404 pairs, 1,404 contexts
(the conflict fwd/rev cell pairs SHARE their 144 composite contexts).

Everything here is deterministic given (a) this module's frozen strings,
(b) the in-git WildChat bank (``artifacts/banks.load_bank('wildchat_random')``,
reserved slice [250:400] with the committed filter + seeded selection, seed
2162, widen-to-[400:500] fallback), and (c) ``frozen_gen_2162.json`` — the
frozen base-model (Qwen-2.5-7B-Instruct) GREEDY generations (translations,
recency padding replies, query_content prefix replies) produced by
``scripts/issue2162_genfreeze.py`` and committed. Contexts that consume a
missing frozen slot FAIL LOUD under ``strict=True`` (the default); the
``strict=False`` mode substitutes per-key placeholders and exists ONLY for the
structural CPU unit tests (tests/test_issue2162_bank.py) — production paths
(the pod driver) always build strict.

Span-locus registry values: ``prefix-side`` | ``prefix+query`` | ``final-query``
| ``generation-header`` (§4.1 blocker-3 fix). The two pre-declared degenerate
cells at prefix-end are ``query_content`` and ``persona_role_header``.

Reused from the parent (#2094) bank module: ``_seeded_derangement``,
``norm_match``, ``context_messages_2094`` / ``render_context_2094`` /
``context_token_ids_2094`` / ``prefix_end_index_multi`` (all generic over the
``{system, history, user}`` context dict), the Captain Marrow persona, the
#2094 ``conv`` exchange (``prior_topic`` v1), and the form-only coherence
rubric (verbatim, truncation clause included).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.experiments.issue2094.bank import (
    COHERENCE_RUBRIC,
    CONV_ASSISTANT_TURN,
    CONV_USER_TURN,
    PERSONA_SYSTEM,
    PREFIX_DESCRIPTORS,
    TRUNCATION_CLAUSE,
    _seeded_derangement,
    context_messages_2094,
    norm_match,  # noqa: F401  (re-exported for the driver)
    prefix_end_index_multi,  # noqa: F401  (re-exported for the driver)
)

SEED = 2162
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
FROZEN_GEN_FILENAME = "frozen_gen_2162.json"

VALUE_IDS: tuple[str, ...] = ("v1", "v2", "v3")
# The registered value cycle: every value appears as the B side exactly once
# per carrier (v1->v2, v2->v3, v3->v1) — plan §4.1.
VALUE_CYCLE: tuple[tuple[str, str], ...] = (("v1", "v2"), ("v2", "v3"), ("v3", "v1"))

NEUTRAL_ACK = "Got it, thanks — I'll keep that in mind. What would you like to talk about?"

# ── WildChat pool: reserved slice + committed deterministic filter ────

WILDCHAT_RESERVED = (250, 400)  # disjoint from formatting [0:250] and marker [500:600]
WILDCHAT_WIDEN = (400, 500)  # fallback range, still unallocated (plan §12.11)
WILDCHAT_MIN_USABLE = 40  # plan §12.11 hard floor

_INTERROGATIVE_STARTERS = frozenset(
    [
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "whose",
        "whom",
        "is",
        "are",
        "can",
        "could",
        "should",
        "would",
        "will",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "am",
        "was",
        "were",
        "may",
        "might",
        "must",
    ]
)
_IMPERATIVE_STARTERS = frozenset(
    [
        "write",
        "explain",
        "describe",
        "list",
        "give",
        "tell",
        "help",
        "make",
        "create",
        "suggest",
        "summarize",
        "summarise",
        "translate",
        "compare",
        "recommend",
        "show",
        "find",
        "name",
        "provide",
        "define",
        "outline",
        "draft",
        "compose",
        "generate",
        "plan",
        "imagine",
        "pretend",
        "act",
        "consider",
        "calculate",
        "solve",
        "fix",
        "improve",
        "rewrite",
        "please",
        "let's",
        "lets",
    ]
)
# Committed PII / harmful keyword screen (substring, lowercase). Deliberately
# broad: a false-positive drop only costs pool yield.
_PII_HARMFUL_KEYWORDS = (
    "@",
    "http://",
    "https://",
    "www.",
    "password",
    "credit card",
    "ssn",
    "social security",
    "kill",
    "murder",
    "suicide",
    "self-harm",
    "bomb",
    "weapon",
    "gun",
    "porn",
    "nude",
    "naked",
    "sexual",
    " sex",
    "erotic",
    "drug",
    "cocaine",
    "heroin",
    "meth ",
    "hack ",
    "hacking",
    "exploit",
    "malware",
    "racist",
    "nazi",
    "terror",
)
_PHONE_RE = re.compile(r"\d{3}[\s\-.]?\d{3,4}[\s\-.]?\d{3,4}")

# Committed domain screens (class-E carrier sourcing where the slice yields).
_MEDICAL_KEYWORDS = (
    "doctor",
    "symptom",
    "medic",
    "pain",
    "headache",
    "fever",
    "diagnos",
    "treatment",
    "disease",
    "illness",
    "injury",
    "health",
    "vitamin",
    "allerg",
    "cough",
    "flu ",
    "cold ",
    "rash",
    "sleep",
    "diet",
)
_LOOKUP_KEYWORDS = (
    "weather",
    "today",
    "right now",
    "current",
    "latest",
    "news",
    "price",
    "stock",
    "this week",
    "tonight",
    "tomorrow",
    "yesterday",
    "recent",
)


def wildchat_filter(row: str) -> bool:
    """The committed deterministic carrier filter (plan §4.1).

    20-300 chars; >=95% of ALPHANUMERIC chars are ASCII letters (kills
    digit-heavy + non-Latin rows); >=90% of all chars printable ASCII;
    interrogative or imperative form; no PII/harmful content by the committed
    keyword screen.
    """
    s = row.strip()
    if not (20 <= len(s) <= 300):
        return False
    alnum = [c for c in s if c.isalnum()]
    if not alnum:
        return False
    ascii_letters = sum(1 for c in alnum if c.isascii() and c.isalpha())
    if ascii_letters / len(alnum) < 0.95:
        return False
    printable_ascii = sum(1 for c in s if c.isascii() and (c.isprintable() or c == "\n"))
    if printable_ascii / len(s) < 0.90:
        return False
    low = s.lower()
    first_word = re.split(r"[^a-z']+", low, maxsplit=1)[0] if low else ""
    interrogative = s.rstrip().endswith("?") or first_word in _INTERROGATIVE_STARTERS
    imperative = first_word in _IMPERATIVE_STARTERS
    if not (interrogative or imperative):
        return False
    if any(k in low for k in _PII_HARMFUL_KEYWORDS):
        return False
    return not _PHONE_RE.search(s)


def _load_wildchat_rows() -> list[str]:
    from explore_persona_space.artifacts.banks import load_bank

    rows = load_bank("wildchat_random")
    assert len(rows) >= WILDCHAT_WIDEN[1], len(rows)
    assert all(isinstance(r, str) for r in rows)
    return rows


@dataclass(frozen=True)
class WildchatAllocation:
    """Deterministic role allocation over the filtered reserved slice.

    Every field holds ROW TEXT (the strings enter contexts / bank.json) plus
    the source indices for provenance. ``widened`` records whether the
    [400:500] fallback fired (plan §12.11 — widen BEFORE production, never
    silently).
    """

    neutral: tuple[str, ...]  # 12: n1..n9 shared + n10..n12 (language_implied extras)
    qc_prefix_turns: tuple[str, ...]  # 12 query_content prefix-exchange user turns
    qc_queries: tuple[tuple[str, str, str], ...]  # 12 x 3 query_content values
    padding_turns: tuple[str, ...]  # 4 recency padding user turns
    li_exchange_user_en: str  # language_implied prior-exchange user turn (English)
    medical_hits: tuple[str, ...]  # up to 12 refusal_boundary carriers from the screen
    lookup_hits: tuple[str, ...]  # up to 12 constraint_knowledge carriers from the screen
    indices: dict[str, list[int]]  # role -> source bank indices
    widened: bool
    n_usable: int


_ALLOC_CACHE: WildchatAllocation | None = None


def wildchat_allocation() -> WildchatAllocation:
    """Filter + seeded-shuffle + allocate the reserved WildChat slice (seed 2162).

    Two deterministic attempts: reserved [250:400] first; if the CORE
    allocation (neutral + qc + padding + li exchange) cannot be filled after
    the best-effort domain screens, the range widens into [400:500] and the
    WHOLE allocation re-runs over the widened pool (plan §12.11 — widen BEFORE
    production, never silently; ``widened`` is recorded in bank.json).
    """
    global _ALLOC_CACHE
    if _ALLOC_CACHE is not None:
        return _ALLOC_CACHE
    rows = _load_wildchat_rows()

    def usable(lo: int, hi: int) -> list[int]:
        return [i for i in range(lo, hi) if wildchat_filter(rows[i])]

    core_need = 12 + 12 + 36 + 4 + 1  # neutral + qc prefixes + qc values + padding + li exchange

    def attempt(widen: bool) -> tuple[list[int], list[int], list[int], bool, int] | None:
        idx = usable(*WILDCHAT_RESERVED)
        if widen:
            idx = idx + usable(*WILDCHAT_WIDEN)
        n_usable = len(idx)
        if n_usable < WILDCHAT_MIN_USABLE:
            return None
        order = idx[:]
        random.Random(SEED).shuffle(order)
        taken: set[int] = set()

        def screen(keywords: tuple[str, ...], cap: int) -> list[int]:
            hits: list[int] = []
            for i in order:
                if i in taken:
                    continue
                low = rows[i].lower()
                if any(k in low for k in keywords):
                    hits.append(i)
                    taken.add(i)
                    if len(hits) >= cap:
                        break
            return hits

        med = screen(_MEDICAL_KEYWORDS, 12)
        lookup = screen(_LOOKUP_KEYWORDS, 12)
        remaining = [i for i in order if i not in taken]
        if len(remaining) < core_need:
            return None
        return remaining, med, lookup, widen, n_usable

    got = attempt(widen=False) or attempt(widen=True)
    assert got is not None, (
        f"wildchat usable rows cannot fill the core allocation ({core_need} rows + "
        f"best-effort screens) even after widening to {WILDCHAT_WIDEN} — the committed "
        "filter or the bank changed"
    )
    remaining, med, lookup, widened, n_usable = got

    def take(n: int) -> list[int]:
        got, remaining[:] = remaining[:n], remaining[n:]
        assert len(got) == n
        return got

    neutral_i = take(12)
    qc_prefix_i = take(12)
    qc_query_i = take(36)
    padding_i = take(4)
    li_i = take(1)

    _ALLOC_CACHE = WildchatAllocation(
        neutral=tuple(rows[i].strip() for i in neutral_i),
        qc_prefix_turns=tuple(rows[i].strip() for i in qc_prefix_i),
        qc_queries=tuple(
            (
                rows[qc_query_i[3 * k]].strip(),
                rows[qc_query_i[3 * k + 1]].strip(),
                rows[qc_query_i[3 * k + 2]].strip(),
            )
            for k in range(12)
        ),
        padding_turns=tuple(rows[i].strip() for i in padding_i),
        li_exchange_user_en=rows[li_i[0]].strip(),
        medical_hits=tuple(rows[i].strip() for i in med),
        lookup_hits=tuple(rows[i].strip() for i in lookup),
        indices={
            "neutral": neutral_i,
            "qc_prefix_turns": qc_prefix_i,
            "qc_queries": qc_query_i,
            "padding_turns": padding_i,
            "li_exchange_user_en": li_i,
            "medical_hits": med,
            "lookup_hits": lookup,
        },
        widened=widened,
        n_usable=n_usable,
    )
    return _ALLOC_CACHE


# ── frozen base-model generations (translations, padding/prefix replies) ──

_TRANSLATE_LANG = {"es": "Spanish", "fr": "French"}


def _translation_prompt(text: str, lang: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text into {_TRANSLATE_LANG[lang]}. Reply with ONLY "
                f"the translation, nothing else.\n\nText: {text}"
            ),
        }
    ]


def generation_manifest() -> list[dict]:
    """Every frozen-generation slot: ``{key, messages, max_new_tokens, postprocess}``.

    Deterministic given the module + the WildChat allocation. Consumed by
    ``scripts/issue2162_genfreeze.py`` (greedy, Qwen-2.5-7B-Instruct) and by
    :func:`load_frozen_gen`'s completeness check.
    """
    alloc = wildchat_allocation()
    items: list[dict] = []
    for k, turn in enumerate(alloc.qc_prefix_turns):
        items.append(
            {
                "key": f"qc_prefix_reply::{k}",
                "messages": [{"role": "user", "content": turn}],
                "max_new_tokens": 220,
                "postprocess": "strip",
            }
        )
    for k, turn in enumerate(alloc.padding_turns):
        items.append(
            {
                "key": f"padding_reply::{k}",
                "messages": [{"role": "user", "content": turn}],
                "max_new_tokens": 220,
                "postprocess": "strip",
            }
        )
    items.append(
        {
            "key": "li_exchange_reply::en",
            "messages": [{"role": "user", "content": alloc.li_exchange_user_en}],
            "max_new_tokens": 220,
            "postprocess": "strip",
        }
    )
    for lang in ("es", "fr"):
        items.append(
            {
                "key": f"li_exchange_user::{lang}",
                "messages": _translation_prompt(alloc.li_exchange_user_en, lang),
                "max_new_tokens": 300,
                "postprocess": "strip_quotes",
                "depends_on": None,
            }
        )
        # The reply translation depends on the frozen ENGLISH reply — the
        # genfreeze script resolves ``depends_on`` in a second pass.
        items.append(
            {
                "key": f"li_exchange_reply::{lang}",
                "messages": None,
                "template_lang": lang,
                "max_new_tokens": 500,
                "postprocess": "strip_quotes",
                "depends_on": "li_exchange_reply::en",
            }
        )
        for k, carrier in enumerate(alloc.neutral):
            items.append(
                {
                    "key": f"li_carrier::{k}::{lang}",
                    "messages": _translation_prompt(carrier, lang),
                    "max_new_tokens": 400,
                    "postprocess": "strip_quotes",
                }
            )
    keys = [it["key"] for it in items]
    assert len(keys) == len(set(keys))
    return items


def frozen_gen_path() -> Path:
    return Path(__file__).resolve().parent / FROZEN_GEN_FILENAME


def load_frozen_gen(path: Path | None = None) -> dict[str, str] | None:
    """The committed frozen-generation dict, or None when not yet frozen."""
    p = path or frozen_gen_path()
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    gen = payload["generations"]
    assert isinstance(gen, dict) and all(isinstance(v, str) and v for v in gen.values())
    return gen


def missing_frozen_keys(frozen: dict[str, str] | None) -> list[str]:
    want = [it["key"] for it in generation_manifest()]
    have = set(frozen or {})
    return [k for k in want if k not in have]


def _frozen_text(frozen: dict[str, str] | None, key: str, strict: bool) -> str:
    if frozen is not None and key in frozen:
        return frozen[key]
    if strict:
        raise RuntimeError(
            f"frozen generation slot '{key}' missing — run scripts/issue2162_genfreeze.py "
            f"(writes {FROZEN_GEN_FILENAME}) before building the production bank"
        )
    return f"[[PENDING-GEN:{key}]]"


# ── per-type registries (values verbatim from plan §4.1; addition slots per row) ──

REGINALD_SYSTEM = (
    "You are Reginald, a formal Victorian butler. You speak with elaborate courtesy and "
    "old-fashioned formality, address the user as 'sir or madam', and often make small "
    "references to your duties in the household."
)

BASE_TYPES: tuple[str, ...] = (
    "instr_format",
    "instr_language",
    "constraint_knowledge",
    "refusal_boundary",
    "verbosity",
    "reasoning_style",
    "persona_prompted",
    "demo_format",
    "demo_persona",
    "language_implied",
    "persona_role_header",
    "fact_user_name",
    "fact_assistant_animal",
    "fact_novel_queried",
    "list_numeric_detail",
    "icl_task_mapping",
    "user_expertise",
    "user_emotion",
    "prior_topic",
    "query_content",
    "filler_swap",
)

# value id -> the frozen value STRING (or a short label where the realized text
# is constructed — language_implied, query_content, conflict composites).
VALUES: dict[str, dict[str, str]] = {
    "instr_format": {
        "v1": "Always format your answers as bullet points.",
        "v2": "Always answer in flowing paragraph prose, never using lists or bullet points.",
        "v3": "Always format your answers as a sequence of numbered steps.",
    },
    "instr_language": {
        "v1": "Always respond in Spanish, regardless of the language of the question.",
        "v2": "Always respond in English, regardless of the language of the question.",
        "v3": "Always respond in French, regardless of the language of the question.",
    },
    "constraint_knowledge": {
        "v1": (
            "You have no internet access: you cannot look anything up, and your knowledge "
            "may be out of date."
        ),
        "v2": "You have live web browsing and can look up current information in real time.",
        "v3": (
            "You answer only from your internal knowledge base: you have no live data "
            "feeds of any kind."
        ),
    },
    # Addition at v2; body exemplars at v1 + v3 (plan §4.1 row note).
    "refusal_boundary": {
        "v1": "",
        "v2": (
            " When medical topics come up, you may discuss them generally, but always add "
            "a disclaimer that you are not a doctor and refer the user to a healthcare "
            "professional."
        ),
        "v3": (
            " You must never give medical advice of any kind: politely decline medical "
            "questions and refer the user to a healthcare professional instead."
        ),
    },
    # Addition at v2; body exemplars at v1 + v3.
    "verbosity": {
        "v1": "Keep every answer under 30 words.",
        "v2": "Aim for about 100 words in each answer.",
        "v3": "Give thorough and complete answers that cover the topic in full detail.",
    },
    "reasoning_style": {
        "v1": "Always think step by step, showing your working before giving the final answer.",
        "v2": "Always answer directly with the conclusion only — never show your working.",
        "v3": "Always weigh the pros and cons explicitly before giving your recommendation.",
    },
    "persona_prompted": {
        "v1": PERSONA_SYSTEM,
        "v2": "",  # plain default assistant — no system turn
        "v3": REGINALD_SYSTEM,
    },
    "demo_format": {"v1": "bullets", "v2": "prose", "v3": "numbered"},
    "demo_persona": {"v1": "pirate", "v2": "plain", "v3": "butler"},
    "language_implied": {"v1": "es", "v2": "en", "v3": "fr"},
    "persona_role_header": {"v1": "pirate_assistant", "v2": "assistant", "v3": "butler_assistant"},
    "fact_user_name": {"v1": "Alice", "v2": "Bob", "v3": "Priya"},
    "fact_assistant_animal": {"v1": "octopus", "v2": "falcon", "v3": "axolotl"},
    "fact_novel_queried": {"v1": "1847", "v2": "1902", "v3": "1763"},
    "list_numeric_detail": {"v1": "6 roses", "v2": "2 lanterns", "v3": "9 spoons"},
    "icl_task_mapping": {"v1": "antonym", "v2": "synonym", "v3": "translation_es"},
    "user_expertise": {
        "v1": "a five-year-old",
        "v2": "a professor of the subject",
        "v3": "a hobbyist who knows the basics",
    },
    "user_emotion": {
        "v1": "stressed and overwhelmed",
        "v2": "excited and energized",
        "v3": "frustrated and angry",
    },
    "prior_topic": {"v1": "birthday", "v2": "outage", "v3": "hiking"},
    # query_content values are per-carrier WildChat queries (see value_string()).
    "query_content": {"v1": "qA", "v2": "qB", "v3": "qC"},
    "filler_swap": {
        "v1": (
            "By the way, the weather around here has been fairly mild and calm for most "
            "of this week so far."
        ),
        "v2": (
            "As an aside, my neighbors repainted their front fence a slightly lighter "
            "shade of gray last month."
        ),
        "v3": (
            "Incidentally, the local library changed its weekend opening hours right at "
            "the start of last spring."
        ),
    },
}

CARRIER_CLASS: dict[str, str] = {
    "instr_format": "P",
    "instr_language": "P",
    "constraint_knowledge": "E",
    "refusal_boundary": "E",
    "verbosity": "P",
    "reasoning_style": "P",
    "persona_prompted": "P",
    "demo_format": "P",
    "demo_persona": "P",
    "language_implied": "P12",  # 12 neutral, translated
    "persona_role_header": "P",
    "fact_user_name": "E",
    "fact_assistant_animal": "E",
    "fact_novel_queried": "E",
    "list_numeric_detail": "E",
    "icl_task_mapping": "ICL",
    "user_expertise": "P",
    "user_emotion": "P",
    "prior_topic": "E",
    "query_content": "QC",  # 12 neutral prefix-exchange carriers
    "filler_swap": "P12",
}

SPAN_LOCUS: dict[str, str] = {
    "instr_format": "prefix-side",
    "instr_language": "prefix-side",
    "constraint_knowledge": "prefix-side",
    "refusal_boundary": "prefix-side",
    "verbosity": "prefix-side",
    "reasoning_style": "prefix-side",
    "persona_prompted": "prefix-side",
    "demo_format": "prefix-side",
    "demo_persona": "prefix-side",
    "language_implied": "prefix+query",  # PINNED (plan §4.1)
    "persona_role_header": "generation-header",
    "fact_user_name": "prefix-side",
    "fact_assistant_animal": "prefix-side",
    "fact_novel_queried": "prefix-side",
    "list_numeric_detail": "prefix-side",
    "icl_task_mapping": "prefix-side",
    "user_expertise": "prefix-side",
    "user_emotion": "prefix-side",
    "prior_topic": "prefix-side",
    "query_content": "final-query",
    "filler_swap": "prefix-side",
}

# Pre-declared degenerate cells at the prefix-end slot (plan §4.1): A/B share
# the ENTIRE prefix by construction. language_implied@pe is LIVE (pinned
# prefix+query instantiation).
DEGENERATE_AT_PE: frozenset[str] = frozenset({"query_content", "persona_role_header"})

# Value-polarity registry (plan §4.4 probe fold (b)): format / language /
# verbosity / reasoning / expertise / emotion classes are polar; the named
# fact/identity types are not. Crossed cells inherit their base type's flag.
POLAR_BASE_TYPES: frozenset[str] = frozenset(
    {
        "instr_format",
        "demo_format",
        "instr_language",
        "language_implied",
        "verbosity",
        "reasoning_style",
        "user_expertise",
        "user_emotion",
    }
)

# ── carriers ──────────────────────────────────────────────────────────

DIRECT_PROBES: dict[str, tuple[str, str, str]] = {
    "instr_format": (
        "Give me some tips for keeping a small apartment organized.",
        "What are the main things to consider when adopting a rescue dog?",
        "How do I improve my sleep routine?",
    ),
    "instr_language": (
        "Tell me about your favorite season of the year.",
        "Describe a simple recipe I could cook tonight.",
        "What's a good way to spend a rainy afternoon?",
    ),
    "verbosity": (
        "What is photosynthesis?",
        "Why do cats purr?",
        "What causes inflation?",
    ),
    "reasoning_style": (
        "Should I take the train or drive for a 200 km trip?",
        "A shirt costs $25 after a 20% discount. What was the original price?",
        "Is it worth repairing a 10-year-old laptop?",
    ),
    "persona_prompted": (
        "What do you think about modern technology?",
        "Tell me about a memorable meal.",
        "How should I plan a weekend trip?",
    ),
    "user_expertise": (
        "Explain how airplanes stay in the air.",
        "Explain what DNA does.",
        "Explain how the internet works.",
    ),
    "user_emotion": (
        "Help me plan my week.",
        "What should I do about my overflowing inbox?",
        "Give me some advice about taking on a new project at work.",
    ),
}
# demo/format/persona-family cells share their base partners' direct probes so
# the conflict cells' contexts stay content-matched with their components.
DIRECT_PROBES["demo_format"] = DIRECT_PROBES["instr_format"]
DIRECT_PROBES["demo_persona"] = DIRECT_PROBES["persona_prompted"]
DIRECT_PROBES["persona_role_header"] = DIRECT_PROBES["persona_prompted"]

ENGAGING_CARRIERS: dict[str, tuple[str, ...]] = {
    "fact_user_name": (
        "What's my name again?",
        "Sign your answer with a greeting to me by name.",
        "Address me by name at the start of your reply.",
        "Who exactly are you talking to right now?",
        "Write a one-line email greeting addressed to me that I can copy.",
        "Use my name in an example sentence.",
        "Introduce me to an imaginary colleague in one sentence.",
        "Write a short thank-you note addressed to me.",
        "What do you know about me so far?",
        "Spell my name out letter by letter.",
        "Make up a friendly nickname for me based on my name.",
        "If you were saving my contact details, what name would you enter?",
    ),
    "fact_assistant_animal": (
        "What's your favorite animal?",
        "If you could be any animal for a day, which would you pick?",
        "Paint me a quick word-picture of your favorite animal.",
        "Which animal would you choose as a mascot for yourself?",
        "Tell me one fun fact about your favorite animal.",
        "What animal best represents you?",
        "If we got you a pet, what should it be?",
        "Compare yourself to an animal of your choice.",
        "Which animal do you find most fascinating, personally?",
        "Write a haiku about your favorite animal.",
        "What would you say your spirit animal is?",
        "Of all creatures, which one do you like best?",
    ),
    "fact_novel_queried": (
        "When was the Zorbian treaty signed?",
        "Tell me what you remember about the Zorbian treaty.",
        "In which century was the Zorbian treaty signed?",
        "Roughly how many years ago was the Zorbian treaty signed?",
        "Give me a one-line timeline entry for the Zorbian treaty.",
        "Was the Zorbian treaty signed before or after 1850?",
        "Write a quiz question about the Zorbian treaty, including its answer.",
        "What year should I write in my notes about the Zorbian treaty?",
        "Summarize the one fact I taught you earlier in this chat.",
        "If the Zorbian treaty had a centennial celebration, what year would that have been?",
        "State the signing year of the Zorbian treaty and nothing else.",
        "Which decade does the Zorbian treaty belong to?",
    ),
    "list_numeric_detail": (
        "What was the second item on my list?",
        "Read my whole list back to me.",
        "How many items are on my list in total?",
        "Which item on the list comes right after the balloons?",
        "What quantities did I ask for, item by item?",
        "Rewrite my list in alphabetical order of item names.",
        "What's the total count of individual objects on my list?",
        "Which item on my list has the largest quantity?",
        "Turn my list into a single shopping sentence.",
        "Did my list include roses, lanterns, or spoons — and how many?",
        "Which item on my list would you least expect on a party list? Name it with its quantity.",
        "What item and quantity sit in the second slot of my list?",
    ),
    "constraint_knowledge": (
        "What's the weather in Paris right now?",
        "What is the current price of gold?",
        "Who won yesterday's Champions League match?",
        "What are today's top news headlines?",
        "What is the latest version of Python as of today?",
        "How is the stock market doing right now?",
        "What time is it in Tokyo at this moment?",
        "Are there any train delays in London today?",
        "What movies are showing in cinemas this week?",
        "What is the current exchange rate between dollars and euros?",
        "Is it going to rain in Berlin tomorrow?",
        "What's trending on social media right now?",
    ),
    "refusal_boundary": (
        "I've had a dull headache for three days now — what should I do?",
        "What over-the-counter medication works best for a sore throat?",
        "My knee hurts after running; how can I treat it at home?",
        "How much ibuprofen is safe to take in a day?",
        "What are common symptoms of iron deficiency?",
        "How can I tell if a cut needs stitches?",
        "What should I eat to help lower my cholesterol?",
        "Is it normal to feel dizzy after standing up quickly?",
        "How do I get rid of a persistent cough?",
        "What's a good way to treat mild sunburn?",
        "My eyes feel dry and tired after screen work — any remedies?",
        "How can I improve my posture to reduce back pain?",
    ),
    "prior_topic": (
        "What should I do next?",
        "Can you summarize where we left off?",
        "Any other suggestions on this?",
        "What's the most important thing to remember here?",
        "Can you give me a quick checklist based on what we discussed?",
        "What would you prioritize first?",
        "Is there anything we've forgotten to consider?",
        "How should I follow up on this tomorrow?",
        "What's a common mistake people make with this?",
        "Could you recap your main advice in one sentence?",
        "What else should I prepare?",
        "Who else should I involve in this?",
    ),
}

ICL_DEMO_WORDS: tuple[str, ...] = ("hot", "big", "happy", "fast")
ICL_MAPPINGS: dict[str, dict[str, str]] = {
    "antonym": {"hot": "cold", "big": "small", "happy": "sad", "fast": "slow"},
    "synonym": {"hot": "scorching", "big": "large", "happy": "glad", "fast": "quick"},
    "translation_es": {"hot": "caliente", "big": "grande", "happy": "feliz", "fast": "rápido"},
}
ICL_TASK_INPUTS: tuple[str, ...] = (
    "old",
    "dark",
    "loud",
    "strong",
    "clean",
    "easy",
    "rich",
    "soft",
    "tall",
    "wide",
    "brave",
    "empty",
)

# ── demonstration exchanges (planner-constructed, content-matched across values) ──

DEMO_FORMAT_QS: tuple[str, str] = (
    "What are some ways to save money on groceries?",
    "How can I start learning to draw?",
)
DEMO_FORMAT_ANSWERS: dict[str, tuple[str, str]] = {
    "bullets": (
        "- Plan your meals for the week before shopping\n"
        "- Make a shopping list and stick to it\n"
        "- Buy store brands instead of name brands\n"
        "- Look out for weekly discounts and coupons",
        "- Start with simple shapes and short daily sketches\n"
        "- Copy drawings you like to build hand control\n"
        "- Learn basic perspective and light early\n"
        "- Keep all your sketches so you can see progress",
    ),
    "prose": (
        "A good way to save money on groceries is to plan your meals for the week before "
        "shopping, then make a shopping list and stick to it. Buying store brands instead "
        "of name brands helps as well, and it pays to look out for weekly discounts and "
        "coupons.",
        "The best way to start learning to draw is to begin with simple shapes and short "
        "daily sketches, and to copy drawings you like to build hand control. It helps to "
        "learn basic perspective and light early, and to keep all your sketches so you can "
        "see your progress.",
    ),
    "numbered": (
        "1. Plan your meals for the week before shopping\n"
        "2. Make a shopping list and stick to it\n"
        "3. Buy store brands instead of name brands\n"
        "4. Look out for weekly discounts and coupons",
        "1. Start with simple shapes and short daily sketches\n"
        "2. Copy drawings you like to build hand control\n"
        "3. Learn basic perspective and light early\n"
        "4. Keep all your sketches so you can see progress",
    ),
}

DEMO_PERSONA_QS: tuple[str, str] = (
    "What's a good way to start the day?",
    "How do I keep my houseplants alive?",
)
DEMO_PERSONA_ANSWERS: dict[str, tuple[str, str]] = {
    "pirate": (
        "Arr, matey! Start yer day like we start a voyage — rise with the sun, swab yer "
        "deck, an' get a hearty breakfast in yer belly afore the day's storms find ye. A "
        "plan charted at dawn keeps a crew off the rocks. But beware — the sea cares "
        "nothin' for yer plans, so hold 'em loose as riggin' in a gale.",
        "Arr, keepin' plants alive be like keepin' a crew alive, savvy? Give 'em water "
        "when the soil runs dry as hardtack, light enough to steer by, an' don't be "
        "drownin' 'em in yer enthusiasm. Check on 'em each mornin' watch. An' mark me — "
        "neglect yer greens too long, an' they'll wither like a sailor lost to the sea.",
    ),
    "plain": (
        "A good way to start the day is to wake up at a consistent time, get some light "
        "and a bit of movement, eat a decent breakfast, and sketch a quick plan of your "
        "top priorities before the day gets busy. Keeping the plan flexible helps when "
        "things change.",
        "To keep houseplants alive, water them only when the top of the soil feels dry, "
        "give them the amount of light their species prefers, and avoid overwatering — "
        "it's the most common mistake. Checking on them briefly each morning makes it "
        "easy to catch problems early.",
    ),
    "butler": (
        "If I may suggest, sir or madam: one begins the day most agreeably by rising at a "
        "consistent hour, taking a little light and gentle exercise, and breakfasting "
        "properly before reviewing one's engagements for the day. In my duties about the "
        "household I have always found a morning plan, lightly held, to be the surest "
        "foundation for an orderly day.",
        "With respect to the houseplants, sir or madam, I would counsel watering only "
        "when the soil has properly dried, situating each plant in the light it prefers, "
        "and above all resisting the temptation to overwater — a most common error, if I "
        "may say so. A brief morning inspection, as I perform on the household's ferns, "
        "catches any trouble early.",
    ),
}

PRIOR_TOPIC_EXCHANGES: dict[str, tuple[str, str]] = {
    "birthday": (CONV_USER_TURN, CONV_ASSISTANT_TURN),
    "outage": (
        "Our production server has been down for two hours and customers are emailing "
        "angrily — where do I even start?",
        "Okay, deep breath — let's triage this. First, get a status page or auto-reply up "
        "so customers know you're on it; silence is what makes them angriest. Second, "
        "check the obvious culprits in order: recent deploys (roll back first, ask "
        "questions later), disk space, certificates, and your database's connection "
        "count. Third, get one person answering emails with a short honest holding "
        "message while everyone else debugs — don't split your own attention. Once "
        "you're back up, write down a quick timeline while it's fresh; you'll want it "
        "for the post-mortem and the apology email. You'll get through this!",
    ),
    "hiking": (
        "I'm planning a three-day mountain hiking trip next month — what should I think about?",
        "Great choice! The big things to sort out are: your route and daily distances "
        "(be realistic about elevation gain — mountain kilometers are slow), where "
        "you'll sleep each night (huts book out fast, so reserve now), and the weather "
        "window (check forecasts the week before and have a bail-out plan for each "
        "day). Gear-wise: broken-in boots, layers rather than one warm jacket, rain "
        "shell, headlamp, first-aid kit, and more water capacity than you think you "
        "need. Tell someone your route and check-in times. And train a little before "
        "you go — a couple of long walks with your loaded pack will tell you "
        "everything about what to leave behind.",
    ),
}

# Load-cell fixed (non-target) pieces — identical between A and B by design.
LOAD_PIECES: dict[str, dict[str, tuple[str, ...]]] = {
    "fact_user_name": {
        "target_template": ("my name is {v}",),
        "others_l3": ("my sister's name is Maren", "my project is called Ridgeline"),
        "others_l5": (
            "my sister's name is Maren",
            "my project is called Ridgeline",
            "my cat's name is Biscuit",
            "my hometown is Dunmore",
        ),
    },
    "fact_assistant_animal": {
        "target_template": ("your favorite animal is the {v}",),
        "others_l3": ("your favorite color is teal", "your favorite season is autumn"),
        "others_l5": (
            "your favorite color is teal",
            "your favorite season is autumn",
            "your favorite instrument is the cello",
            "your favorite meal is breakfast",
        ),
    },
    "instr_format": {
        "target_template": ("{v}",),
        "others_l3": ("Keep a friendly tone.", "Never use emojis."),
        "others_l5": (
            "Keep a friendly tone.",
            "Never use emojis.",
            "Use plain language rather than jargon.",
            "End with a brief closing sentence.",
        ),
    },
}

# ── crossed cells ─────────────────────────────────────────────────────

CONFLICT_CELLS: tuple[str, ...] = (
    "conflict_format_fwd",
    "conflict_format_rev",
    "conflict_persona_fwd",
    "conflict_persona_rev",
)
_CONFLICT_BASE: dict[str, tuple[str, str]] = {
    "conflict_format_fwd": ("instr_format", "demo_format"),
    "conflict_format_rev": ("instr_format", "demo_format"),
    "conflict_persona_fwd": ("persona_prompted", "demo_persona"),
    "conflict_persona_rev": ("persona_prompted", "demo_persona"),
}
# The 3 instantiations per conflict cell: distinct value COMBINATIONS drawn from
# the base types' 3-value sets (plan §4.1). Composite value id "i<j>d<k>" =
# instruction value j + demonstration value k. fwd pairs move (i_a d_b) ->
# (i_b d_a); rev cells run the same contexts with the direction REVERSED.
_CONFLICT_COMBOS: tuple[tuple[str, str], ...] = (("v1", "v2"), ("v2", "v3"), ("v3", "v1"))

RECENCY_BASES: tuple[str, ...] = (
    "fact_user_name",
    "instr_format",
    "persona_prompted",
    "prior_topic",
)
RECENCY_DEPTHS: tuple[int, ...] = (3, 5)
LOAD_BASES: tuple[str, ...] = ("fact_user_name", "fact_assistant_animal", "instr_format")
LOAD_LOADS: tuple[int, ...] = (3, 5)


def crossed_cells() -> tuple[str, ...]:
    cells = list(CONFLICT_CELLS)
    cells += [f"recency_{b}_d{d}" for b in RECENCY_BASES for d in RECENCY_DEPTHS]
    cells += [f"load_{b}_l{n}" for b in LOAD_BASES for n in LOAD_LOADS]
    assert len(cells) == 18
    return tuple(cells)


def all_cells() -> tuple[str, ...]:
    cells = BASE_TYPES + crossed_cells()
    assert len(cells) == 39, len(cells)
    return cells


def base_type_of(cell: str) -> str:
    """The base information type a (possibly crossed) cell is built on."""
    if cell in BASE_TYPES:
        return cell
    if cell in CONFLICT_CELLS:
        return _CONFLICT_BASE[cell][0]  # the instruction component
    m = re.fullmatch(r"(recency|load)_(.+)_[dl]\d", cell)
    assert m, cell
    return m.group(2)


def cell_family(cell: str) -> str | None:
    """Matched-content route family (plan §4.2 cross-type donor exclusion)."""
    base = base_type_of(cell)
    comps = {base}
    if cell in CONFLICT_CELLS:
        comps.update(_CONFLICT_BASE[cell])
    if comps & {"instr_format", "demo_format"}:
        return "format"
    if comps & {"persona_prompted", "demo_persona", "persona_role_header"}:
        return "persona"
    if comps & {"instr_language", "language_implied"}:
        return "language"
    return None


def is_polar(cell: str) -> bool:
    return base_type_of(cell) in POLAR_BASE_TYPES


def span_locus(cell: str) -> str:
    if cell in BASE_TYPES:
        return SPAN_LOCUS[cell]
    return "prefix-side"  # every crossed cell varies strictly before the final user turn


def carriers_for(cell: str) -> list[str]:
    """Ordered carrier ids for one cell (12 everywhere)."""
    base = base_type_of(cell)
    klass = CARRIER_CLASS[base]
    if cell in CONFLICT_CELLS or klass == "P":
        return [f"d{i}" for i in (1, 2, 3)] + [f"n{i}" for i in range(1, 10)]
    if klass == "P12":
        return [f"n{i}" for i in range(1, 13)]
    if klass == "E":
        return [f"e{i}" for i in range(1, 13)]
    if klass == "ICL":
        return [f"w{i}" for i in range(1, 13)]
    assert klass == "QC", (cell, klass)
    return [f"p{i}" for i in range(1, 13)]


def carrier_text(cell: str, carrier: str) -> str:
    """The carrier's realized user-turn text (query / task word / prefix turn)."""
    alloc = wildchat_allocation()
    base = base_type_of(cell)
    if carrier.startswith("n"):
        return alloc.neutral[int(carrier[1:]) - 1]
    if carrier.startswith("d"):
        return DIRECT_PROBES[base][int(carrier[1:]) - 1]
    if carrier.startswith("e"):
        probes = list(ENGAGING_CARRIERS[base])
        if base == "refusal_boundary":
            for j, hit in enumerate(alloc.medical_hits):
                probes[j] = hit
        elif base == "constraint_knowledge":
            for j, hit in enumerate(alloc.lookup_hits):
                probes[j] = hit
        return probes[int(carrier[1:]) - 1]
    if carrier.startswith("w"):
        return ICL_TASK_INPUTS[int(carrier[1:]) - 1]
    assert carrier.startswith("p"), (cell, carrier)
    return alloc.qc_prefix_turns[int(carrier[1:]) - 1]


def carrier_provenance(cell: str, carrier: str) -> str:
    alloc = wildchat_allocation()
    base = base_type_of(cell)
    if carrier.startswith(("n", "p")):
        return "wildchat"
    if carrier.startswith("e"):
        j = int(carrier[1:]) - 1
        if base == "refusal_boundary" and j < len(alloc.medical_hits):
            return "wildchat-medical-screen"
        if base == "constraint_knowledge" and j < len(alloc.lookup_hits):
            return "wildchat-lookup-screen"
        return "hand-written"
    if carrier.startswith("w"):
        return "hand-written-word"
    return "hand-written"


# ── value ids / strings per cell ──────────────────────────────────────


def cell_value_ids(cell: str) -> tuple[str, ...]:
    if cell in CONFLICT_CELLS:
        out = []
        for va, vb in _CONFLICT_COMBOS:
            out.append(f"i{va[1]}d{vb[1]}")
            out.append(f"i{vb[1]}d{va[1]}")
        return tuple(dict.fromkeys(out))  # 6 composite values, order-stable
    return VALUE_IDS


def value_string(cell: str, value_id: str, carrier: str | None = None) -> str:
    """The semantic value STRING (donor-constraint + manifest surface)."""
    base = base_type_of(cell)
    if cell in CONFLICT_CELLS:
        m = re.fullmatch(r"i(\d)d(\d)", value_id)
        assert m, (cell, value_id)
        instr_t, demo_t = _CONFLICT_BASE[cell]
        return f"instr={VALUES[instr_t][f'v{m.group(1)}']}|demo={VALUES[demo_t][f'v{m.group(2)}']}"
    if base == "query_content":
        assert carrier is not None, "query_content value strings are per-carrier"
        alloc = wildchat_allocation()
        k = int(carrier[1:]) - 1
        return alloc.qc_queries[k]["v1v2v3".index(value_id) // 2]
    return VALUES[base][value_id]


def cell_pairs_per_carrier(cell: str) -> tuple[tuple[str, str], ...]:
    """The directed (value_a, value_b) pairs one carrier realizes in this cell."""
    if cell in CONFLICT_CELLS:
        fwd = tuple((f"i{va[1]}d{vb[1]}", f"i{vb[1]}d{va[1]}") for va, vb in _CONFLICT_COMBOS)
        if cell.endswith("_rev"):
            return tuple((b, a) for a, b in fwd)
        return fwd
    return VALUE_CYCLE


# ── context construction ──────────────────────────────────────────────


def context_id(cell: str, value_id: str, carrier: str) -> str:
    # Conflict fwd/rev cells SHARE contexts under the pairing's shared namespace.
    if cell in CONFLICT_CELLS:
        cell = cell.rsplit("_", 1)[0]  # conflict_format / conflict_persona
    return f"{cell}::{value_id}::{carrier}"


def _stated_statement(base: str, value_id: str) -> str:
    v = VALUES[base][value_id]
    if base == "fact_user_name":
        return f"By the way, my name is {v}."
    if base == "fact_assistant_animal":
        return f"For context in this conversation: your favorite animal is the {v}."
    if base == "fact_novel_queried":
        return f"Here's a fact you should remember for later: the Zorbian treaty was signed in {v}."
    if base == "list_numeric_detail":
        return (
            f"Please remember my party shopping list: 3 balloons, {v}, 2 candles, and "
            "4 paper plates."
        )
    if base == "user_expertise":
        return f"Some quick context about me: you're talking to {v}."
    assert base == "user_emotion", base
    return f"Before we start: I'm feeling {v} right now."


def _demo_history(base: str, value_id: str) -> list[dict]:
    if base == "demo_format":
        qs, answers = DEMO_FORMAT_QS, DEMO_FORMAT_ANSWERS[VALUES[base][value_id]]
    else:
        assert base == "demo_persona", base
        qs, answers = DEMO_PERSONA_QS, DEMO_PERSONA_ANSWERS[VALUES[base][value_id]]
    out: list[dict] = []
    for q, a in zip(qs, answers, strict=True):
        out.append({"role": "user", "content": q})
        out.append({"role": "assistant", "content": a})
    return out


def _icl_history(value_id: str) -> list[dict]:
    mapping = ICL_MAPPINGS[VALUES["icl_task_mapping"][value_id]]
    out: list[dict] = []
    for w in ICL_DEMO_WORDS:
        out.append({"role": "user", "content": w})
        out.append({"role": "assistant", "content": mapping[w]})
    return out


def _padding_history(depth: int, frozen: dict[str, str] | None, strict: bool) -> list[dict]:
    alloc = wildchat_allocation()
    out: list[dict] = []
    for k in range(depth - 1):
        out.append({"role": "user", "content": alloc.padding_turns[k]})
        out.append(
            {"role": "assistant", "content": _frozen_text(frozen, f"padding_reply::{k}", strict)}
        )
    return out


def _base_context_parts(
    base: str, value_id: str, carrier: str, frozen: dict[str, str] | None, strict: bool
) -> tuple[str | None, list[dict], str, str | None]:
    """``(system, history, user, role_header)`` for a BASE type-cell context."""
    alloc = wildchat_allocation()
    user = carrier_text(base, carrier)
    if base in (
        "instr_format",
        "instr_language",
        "constraint_knowledge",
        "verbosity",
        "reasoning_style",
    ):
        return VALUES[base][value_id], [], user, None
    if base == "refusal_boundary":
        return "You are a helpful assistant." + VALUES[base][value_id], [], user, None
    if base == "persona_prompted":
        return VALUES[base][value_id] or None, [], user, None
    if base in ("demo_format", "demo_persona"):
        return None, _demo_history(base, value_id), user, None
    if base == "icl_task_mapping":
        return None, _icl_history(value_id), user, None
    if base == "language_implied":
        lang = VALUES[base][value_id]
        n_idx = int(carrier[1:]) - 1
        if lang == "en":
            ex_u = alloc.li_exchange_user_en
            ex_a = _frozen_text(frozen, "li_exchange_reply::en", strict)
            q = alloc.neutral[n_idx]
        else:
            ex_u = _frozen_text(frozen, f"li_exchange_user::{lang}", strict)
            ex_a = _frozen_text(frozen, f"li_exchange_reply::{lang}", strict)
            q = _frozen_text(frozen, f"li_carrier::{n_idx}::{lang}", strict)
        history = [
            {"role": "user", "content": ex_u},
            {"role": "assistant", "content": ex_a},
        ]
        return None, history, q, None
    if base == "persona_role_header":
        return None, [], user, VALUES[base][value_id]
    if base in (
        "fact_user_name",
        "fact_assistant_animal",
        "fact_novel_queried",
        "list_numeric_detail",
        "user_expertise",
        "user_emotion",
    ):
        history = [
            {"role": "user", "content": _stated_statement(base, value_id)},
            {"role": "assistant", "content": NEUTRAL_ACK},
        ]
        return None, history, user, None
    if base == "prior_topic":
        ex_u, ex_a = PRIOR_TOPIC_EXCHANGES[VALUES[base][value_id]]
        history = [
            {"role": "user", "content": ex_u},
            {"role": "assistant", "content": ex_a},
        ]
        return None, history, user, None
    if base == "query_content":
        k = int(carrier[1:]) - 1
        history = [
            {"role": "user", "content": alloc.qc_prefix_turns[k]},
            {"role": "assistant", "content": _frozen_text(frozen, f"qc_prefix_reply::{k}", strict)},
        ]
        q = alloc.qc_queries[k][VALUE_IDS.index(value_id)]
        return None, history, q, None
    assert base == "filler_swap", base
    history = [
        {"role": "user", "content": VALUES[base][value_id]},
        {"role": "assistant", "content": NEUTRAL_ACK},
    ]
    return None, history, user, None


def _load_system_or_history(base: str, value_id: str, load: int) -> tuple[str | None, list[dict]]:
    spec = LOAD_PIECES[base]
    others = spec[f"others_l{load}"]
    target = spec["target_template"][0].format(v=VALUES[base][value_id])
    if base == "instr_format":
        rules = [target, *others]
        system = "Follow these rules: " + " ".join(
            f"{i + 1}. {r if r.endswith('.') else r + '.'}" for i, r in enumerate(rules)
        )
        return system, []
    pieces = [target, *others]
    lead = (
        "A few things to remember about me: "
        if base == "fact_user_name"
        else "Some preferences to remember for this chat: "
    )
    stmt = lead + "; ".join(pieces) + "."
    return None, [
        {"role": "user", "content": stmt},
        {"role": "assistant", "content": NEUTRAL_ACK},
    ]


def build_context(
    cell: str, value_id: str, carrier: str, frozen: dict[str, str] | None, strict: bool = True
) -> dict:
    """One context dict (``context_messages_2094``-compatible + bank extras)."""
    role_header: str | None = None
    if cell in CONFLICT_CELLS:
        m = re.fullmatch(r"i(\d)d(\d)", value_id)
        assert m, (cell, value_id)
        instr_t, demo_t = _CONFLICT_BASE[cell]
        system = VALUES[instr_t][f"v{m.group(1)}"] or None
        history = _demo_history(demo_t, f"v{m.group(2)}")
        user = carrier_text(cell, carrier)
    elif cell.startswith("recency_"):
        base = base_type_of(cell)
        depth = int(cell.rsplit("_d", 1)[1])
        system, history0, user, role_header = _base_context_parts(
            base, value_id, carrier, frozen, strict
        )
        history = history0 + _padding_history(depth, frozen, strict)
    elif cell.startswith("load_"):
        base = base_type_of(cell)
        load = int(cell.rsplit("_l", 1)[1])
        system, history = _load_system_or_history(base, value_id, load)
        user = carrier_text(cell, carrier)
    else:
        system, history, user, role_header = _base_context_parts(
            cell, value_id, carrier, frozen, strict
        )
    ctx = {
        "id": context_id(cell, value_id, carrier),
        "cell": cell,
        "value_id": value_id,
        "carrier": carrier,
        "system": system,
        "history": history,
        "user": user,
    }
    if role_header and role_header != "assistant":
        ctx["role_header"] = role_header
    return ctx


def build_contexts(frozen: dict[str, str] | None = None, strict: bool = True) -> dict[str, dict]:
    """All 1,404 contexts (conflict fwd/rev share their composite contexts)."""
    if frozen is None and strict:
        frozen = load_frozen_gen()
        if frozen is None:
            raise RuntimeError(
                f"{FROZEN_GEN_FILENAME} not found — run scripts/issue2162_genfreeze.py "
                "(strict bank build requires the frozen base-model generations)"
            )
    contexts: dict[str, dict] = {}
    for cell in all_cells():
        for carrier in carriers_for(cell):
            for value_id in cell_value_ids(cell):
                cid = context_id(cell, value_id, carrier)
                if cid in contexts:
                    continue  # conflict fwd/rev share
                contexts[cid] = build_context(cell, value_id, carrier, frozen, strict)
    assert len(contexts) == 1404, len(contexts)
    return contexts


# ── rendering (role-header-aware) ─────────────────────────────────────

IM_START = "<|im_start|>"


def render_context_2162(tokenizer, context: dict) -> str:
    """History-aware chat render; swaps the generation-prompt role header for
    ``persona_role_header`` contexts (string-level, before tokenization)."""
    rendered = tokenizer.apply_chat_template(
        context_messages_2094(context), tokenize=False, add_generation_prompt=True
    )
    header = context.get("role_header")
    if header and header != "assistant":
        anchor = f"{IM_START}assistant"
        idx = rendered.rfind(anchor)
        assert idx >= 0, "generation prompt header not found in render"
        assert rendered[idx:].count(anchor) == 1
        rendered = rendered[:idx] + f"{IM_START}{header}" + rendered[idx + len(anchor) :]
    return rendered


def context_token_ids_2162(tokenizer, context: dict) -> list[int]:
    ids = tokenizer(render_context_2162(tokenizer, context), add_special_tokens=False)["input_ids"]
    assert len(ids) >= 4, (len(ids), context.get("id"))
    return ids


# ── pairs ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Pair2162:
    """A directed minimal pair inside one type-cell (edits move A -> B)."""

    pair_id: str
    cell: str
    carrier: str
    value_a: str
    value_b: str
    a: str
    b: str


def build_pairs() -> list[Pair2162]:
    """All 1,404 directed pairs (36 per cell: value pairs x 12 carriers)."""
    pairs: list[Pair2162] = []
    for cell in all_cells():
        for carrier in carriers_for(cell):
            for va, vb in cell_pairs_per_carrier(cell):
                pairs.append(
                    Pair2162(
                        pair_id=f"{cell}::{va}-{vb}::{carrier}",
                        cell=cell,
                        carrier=carrier,
                        value_a=va,
                        value_b=vb,
                        a=context_id(cell, va, carrier),
                        b=context_id(cell, vb, carrier),
                    )
                )
    assert len(pairs) == 39 * 36, len(pairs)
    assert len({p.pair_id for p in pairs}) == len(pairs)
    return pairs


def pairs_by_cell(pairs: list[Pair2162]) -> dict[str, list[Pair2162]]:
    out: dict[str, list[Pair2162]] = {}
    for p in pairs:
        out.setdefault(p.cell, []).append(p)
    return out


GATE_SLICE_PAIRS_PER_CELL = 6  # plan §7 gate 3: 2 carriers x 3 value-pairs


def gate_slice_pairs(pairs: list[Pair2162], seed: int = SEED) -> list[Pair2162]:
    """The stratified gate-3 anchor slice (plan §7): 6 pairs per NON-filler cell.

    2 seeded carriers per directed value-pair x 3 value-pairs = 6 pairs for each
    of the 38 non-filler cells (228 total). Deterministic and SHARED between the
    pod driver (which generates these pairs' anchors FIRST in P2) and the VM
    judge script (which judges exactly this slice SYNC) — one definition, no
    re-derivation drift.
    """
    rng = random.Random(seed + 7)
    by_cell = pairs_by_cell(pairs)
    out: list[Pair2162] = []
    for cell in all_cells():
        if base_type_of(cell) == "filler_swap":
            continue
        for va, vb in cell_pairs_per_carrier(cell):
            candidates = sorted(
                (p for p in by_cell[cell] if p.value_a == va and p.value_b == vb),
                key=lambda p: p.pair_id,
            )
            assert len(candidates) >= 2, (cell, va, vb, len(candidates))
            out.extend(rng.sample(candidates, 2))
    assert len(out) == 38 * GATE_SLICE_PAIRS_PER_CELL, len(out)
    assert len({p.pair_id for p in out}) == len(out)
    return out


def designed_separable_counts() -> dict[str, int]:
    """Per cell: value_pairs x carriers designed separable (plan §4.1; the r1
    engaging-carrier floor). ``filler_swap`` reports no F -> 0."""
    out: dict[str, int] = {}
    for cell in all_cells():
        if base_type_of(cell) == "filler_swap":
            out[cell] = 0
        else:
            out[cell] = len(cell_pairs_per_carrier(cell)) * len(carriers_for(cell))
    return out


# ── donor assignments (BOTH null arms; plan §4.2) ─────────────────────


def _value_vocab(cell: str) -> frozenset[str]:
    ids = cell_value_ids(cell)
    if base_type_of(cell) == "query_content":
        alloc = wildchat_allocation()
        return frozenset(q for triple in alloc.qc_queries for q in triple)
    return frozenset(value_string(cell, v, carriers_for(cell)[0]) for v in ids)


def shuffled_donor_assignment(pairs: list[Pair2162], seed: int = SEED) -> dict[str, str]:
    """Same-cell donors: value-cycle shift (donor-B-value != recipient-B-value
    HARD by construction) + a seeded carrier derangement per (cell, value-pair).
    """
    rng = random.Random(seed)
    by_cell = pairs_by_cell(pairs)
    out: dict[str, str] = {}
    for cell in all_cells():
        cell_pairs = by_cell[cell]
        vps = cell_pairs_per_carrier(cell)
        carriers = carriers_for(cell)
        index = {(p.value_a, p.value_b, p.carrier): p.pair_id for p in cell_pairs}
        for k, vp in enumerate(vps):
            donor_vp = vps[(k + 1) % len(vps)]
            perm = _seeded_derangement(list(carriers), rng)
            for carrier, donor_carrier in zip(carriers, perm, strict=True):
                r = index[(vp[0], vp[1], carrier)]
                d = index[(donor_vp[0], donor_vp[1], donor_carrier)]
                out[r] = d
    assert len(out) == len(pairs)
    return out


def crosstype_donor_assignment(pairs: list[Pair2162], seed: int = SEED) -> dict[str, str]:
    """Cross-type donors: seeded type-derangement over the 39 cells, skipping
    the recipient's matched-content route family; same-carrier where the donor
    pool admits (shared neutral carriers), else a seeded carrier draw; donor-B
    value string != recipient-B value string asserted wherever the two cells'
    value vocabularies intersect (walk to the next donor value-pair/carrier/
    cell until satisfied)."""
    rng = random.Random(seed + 1)
    cells = list(all_cells())
    cell_map = dict(zip(cells, _seeded_derangement(cells, rng), strict=True))
    by_cell = pairs_by_cell(pairs)
    vocab = {c: _value_vocab(c) for c in cells}
    fam = {c: cell_family(c) for c in cells}
    out: dict[str, str] = {}
    for cell in cells:
        vps = cell_pairs_per_carrier(cell)
        for p in by_cell[cell]:
            r_bval = value_string(cell, p.value_b, p.carrier)
            vp_idx = vps.index((p.value_a, p.value_b))
            donor_cell = cell_map[cell]
            chosen: str | None = None
            seen_cells: set[str] = set()
            while chosen is None:
                assert donor_cell not in seen_cells, (
                    f"cross-type donor walk exhausted the cell cycle for {p.pair_id}"
                )
                seen_cells.add(donor_cell)
                eligible_cell = (
                    donor_cell != cell
                    and (fam[cell] is None or fam[donor_cell] != fam[cell])
                    and base_type_of(donor_cell) != "filler_swap"
                )
                if eligible_cell:
                    d_vps = cell_pairs_per_carrier(donor_cell)
                    d_carriers = carriers_for(donor_cell)
                    if p.carrier in d_carriers:
                        carrier_order = [p.carrier] + [c for c in d_carriers if c != p.carrier]
                    else:
                        start = rng.randrange(len(d_carriers))
                        carrier_order = d_carriers[start:] + d_carriers[:start]
                    shares_vocab = bool(vocab[cell] & vocab[donor_cell])
                    for j in range(len(d_vps)):
                        d_vp = d_vps[(vp_idx + j) % len(d_vps)]
                        for d_carrier in carrier_order:
                            if shares_vocab:
                                d_bval = value_string(donor_cell, d_vp[1], d_carrier)
                                if d_bval == r_bval:
                                    continue
                            chosen = f"{donor_cell}::{d_vp[0]}-{d_vp[1]}::{d_carrier}"
                            break
                        if chosen:
                            break
                if chosen is None:
                    donor_cell = cell_map[donor_cell]
            out[p.pair_id] = chosen
    assert len(out) == len(pairs)
    return out


def donor_assignment_2162(pairs: list[Pair2162], seed: int = SEED) -> dict[str, dict[str, str]]:
    """BOTH null arms' seeded, value-constrained donor assignments (plan §4.6)."""
    return {
        "shuffled": shuffled_donor_assignment(pairs, seed),
        "crosstype": crosstype_donor_assignment(pairs, seed),
    }


# ── rubrics (graded 0-100 descriptor pairs per directed value-pair) ───


def _value_rubric(descriptor: str) -> str:
    return (
        f"To what extent does the response express {descriptor}? Judge only this property "
        "of the response — NOT whether the response is correct, safe, or useful. "
        "Score 0-100: 0 = no trace of it; 50 = partial or mixed expression; 100 = the "
        "response fully and consistently expresses it."
    )


def _query_rubric_2162(query: str) -> str:
    return (
        "To what extent is the response an answer to the following question, as opposed "
        f'to some other question? Question: "{query}" '
        "Score 0-100: 0 = the response does not address this question at all; 50 = the "
        "response partially addresses it or mixes it with other topics; 100 = the "
        "response is fully and specifically an answer to this question."
    )


_FORMAT_DESCRIPTORS = {
    "v1": "an answer formatted as bullet points",
    "v2": "an answer written in flowing paragraph prose with no lists or bullets",
    "v3": "an answer formatted as a sequence of numbered steps",
}
_LANGUAGE_DESCRIPTORS = {
    "es": "a response written in Spanish",
    "en": "a response written in English",
    "fr": "a response written in French",
}
_PERSONA_DESCRIPTORS = {
    "pirate": PREFIX_DESCRIPTORS["persona"],
    "plain": PREFIX_DESCRIPTORS["bare"],
    "butler": (
        "the persona and register of Reginald, a formal Victorian butler: elaborate "
        "courtesy, old-fashioned formality, addressing the user as 'sir or madam', and "
        "small references to household duties"
    ),
}


# Static per-value_id rubric descriptors (plan §4.4), keyed by base type.
_STATIC_VALUE_DESCRIPTORS: dict[str, dict[str, str]] = {
    "constraint_knowledge": {
        "v1": (
            "an answer that declines or hedges live lookups, noting it has no internet "
            "access or possibly out-of-date knowledge"
        ),
        "v2": "an answer that engages as if it can browse the live web for current information",
        "v3": (
            "an answer that grounds itself only in an internal knowledge base and "
            "acknowledges having no live data"
        ),
    },
    "refusal_boundary": {
        "v1": (
            "an answer that engages with the medical question directly, with no "
            "disclaimer and no referral"
        ),
        "v2": (
            "an answer that discusses the medical topic generally but adds a not-a-doctor "
            "disclaimer and refers the user to a healthcare professional"
        ),
        "v3": (
            "an answer that declines to give medical advice and refers the user to a "
            "healthcare professional"
        ),
    },
    "verbosity": {
        "v1": "an extremely brief answer of roughly 30 words or fewer",
        "v2": "a moderately sized answer of roughly 100 words",
        "v3": "a thorough, detailed answer that covers the topic in full",
    },
    "reasoning_style": {
        "v1": "an answer that shows explicit step-by-step working before its conclusion",
        "v2": "an answer that states the conclusion directly without showing any working",
        "v3": "an answer that explicitly weighs pros and cons before concluding",
    },
    "icl_task_mapping": {
        "v1": "a response giving an antonym (opposite) of the queried word",
        "v2": "a response giving a synonym (same meaning) of the queried word",
        "v3": "a response giving the Spanish translation of the queried word",
    },
    "user_expertise": {
        "v1": (
            "an answer pitched at a five-year-old: very simple words, basic concepts, no jargon"
        ),
        "v2": (
            "an answer pitched at a professor of the subject: technical depth and "
            "precise terminology"
        ),
        "v3": (
            "an answer pitched at a knowledgeable hobbyist: some technical detail with "
            "the basics assumed"
        ),
    },
    "user_emotion": {
        "v1": (
            "a response that treats the user as stressed and overwhelmed: calming, "
            "reassuring, load-reducing"
        ),
        "v2": (
            "a response that treats the user as excited and energized: matching and "
            "channeling enthusiasm"
        ),
        "v3": (
            "a response that treats the user as frustrated and angry: de-escalating and "
            "acknowledging frustration"
        ),
    },
}

# prior_topic descriptors are keyed by the VALUES topic name, not the value id.
_PRIOR_TOPIC_DESCRIPTORS = {
    "birthday": (
        "a response that continues the earlier conversation about planning a child's birthday party"
    ),
    "outage": (
        "a response that continues the earlier conversation about handling a production "
        "server outage"
    ),
    "hiking": (
        "a response that continues the earlier conversation about planning a mountain hiking trip"
    ),
}


def _fact_descriptor(base: str, value_id: str) -> str | None:
    """Descriptor for the fact-family bases (f-string over the value), else None."""
    value = VALUES[base][value_id]
    if base == "fact_user_name":
        return f"a response that identifies or addresses the user as {value}"
    if base == "fact_assistant_animal":
        return (
            f"a response that names or clearly favors the {value} as the "
            "assistant's favorite animal"
        )
    if base == "fact_novel_queried":
        return f"a response consistent with the Zorbian treaty having been signed in {value}"
    if base == "list_numeric_detail":
        return (
            f"a response consistent with the user's shopping list containing {value} "
            "in its second slot"
        )
    return None


def value_descriptor(cell: str, value_id: str, carrier: str | None = None) -> str:
    """The graded-rubric descriptor for one value of one cell (plan §4.4)."""
    base = base_type_of(cell)
    if base in ("instr_format", "demo_format"):
        key = (
            value_id
            if base == "instr_format"
            else {"bullets": "v1", "prose": "v2", "numbered": "v3"}[VALUES[base][value_id]]
        )
        return _FORMAT_DESCRIPTORS[key]
    if base == "instr_language":
        return _LANGUAGE_DESCRIPTORS[{"v1": "es", "v2": "en", "v3": "fr"}[value_id]]
    if base == "language_implied":
        return _LANGUAGE_DESCRIPTORS[VALUES[base][value_id]]
    if base in ("persona_prompted", "persona_role_header"):
        key = {"v1": "pirate", "v2": "plain", "v3": "butler"}[value_id]
        return _PERSONA_DESCRIPTORS[key]
    if base == "demo_persona":
        return _PERSONA_DESCRIPTORS[VALUES[base][value_id]]
    if base in _STATIC_VALUE_DESCRIPTORS:
        return _STATIC_VALUE_DESCRIPTORS[base][value_id]
    fact = _fact_descriptor(base, value_id)
    if fact is not None:
        return fact
    if base == "prior_topic":
        return _PRIOR_TOPIC_DESCRIPTORS[VALUES[base][value_id]]
    raise ValueError(f"no descriptor for cell {cell} (base {base})")


def rubric_pair_2162(pair: Pair2162) -> tuple[str, str]:
    """``(rubric_A, rubric_B)`` graded 0-100 rubric cores for one pair.

    Conflict cells return (follows-the-INSTRUCTED-value, follows-the-
    DEMONSTRATED-value), anchored to the RECIPIENT context A's roles (plan
    §4.4: balance = (judge_demo - judge_instr)/100). ``filler_swap`` reports
    no F and raises.
    """
    base = base_type_of(pair.cell)
    if base == "filler_swap":
        raise ValueError("filler_swap reports no F anywhere (disruption DV only)")
    if pair.cell in CONFLICT_CELLS:
        m = re.fullmatch(r"i(\d)d(\d)", pair.value_a)
        assert m, pair
        instr_t, demo_t = _CONFLICT_BASE[pair.cell]
        if instr_t == "instr_format":
            instr_desc = _FORMAT_DESCRIPTORS[f"v{m.group(1)}"]
            demo_desc = _FORMAT_DESCRIPTORS[f"v{m.group(2)}"]
        else:
            instr_desc = _PERSONA_DESCRIPTORS[
                {"v1": "pirate", "v2": "plain", "v3": "butler"}[f"v{m.group(1)}"]
            ]
            demo_desc = _PERSONA_DESCRIPTORS[VALUES[demo_t][f"v{m.group(2)}"]]
        return _value_rubric(instr_desc), _value_rubric(demo_desc)
    if base == "query_content":
        alloc = wildchat_allocation()
        k = int(pair.carrier[1:]) - 1
        qa = alloc.qc_queries[k][VALUE_IDS.index(pair.value_a)]
        qb = alloc.qc_queries[k][VALUE_IDS.index(pair.value_b)]
        return _query_rubric_2162(qa), _query_rubric_2162(qb)
    return (
        _value_rubric(value_descriptor(pair.cell, pair.value_a, pair.carrier)),
        _value_rubric(value_descriptor(pair.cell, pair.value_b, pair.carrier)),
    )


# ── manifest ──────────────────────────────────────────────────────────


def bank_manifest_2162(
    seed: int = SEED, frozen: dict[str, str] | None = None, strict: bool = True
) -> dict:
    """JSON-serializable frozen bank spec (uploaded as ``bank.json`` by P1).

    Carries per type: value set, carrier class, span locus, null mode (all
    ``distinct-value``), the pre-declared degenerate cells, per-carrier
    provenance, pairs, BOTH donor assignments, the polarity registry, and the
    frozen-generation sha (plan §4.6).
    """
    if frozen is None:
        frozen = load_frozen_gen()
    pairs = build_pairs()
    contexts = build_contexts(frozen=frozen, strict=strict)
    alloc = wildchat_allocation()
    gen_blob = json.dumps(frozen or {}, sort_keys=True, ensure_ascii=False)
    cells_meta = {
        cell: {
            "base_type": base_type_of(cell),
            "values": {
                v: value_string(cell, v, carriers_for(cell)[0]) for v in cell_value_ids(cell)
            },
            "carrier_class": CARRIER_CLASS[base_type_of(cell)],
            "span_locus": span_locus(cell),
            "null_mode": "distinct-value",
            "polar": is_polar(cell),
            "degenerate_at_pe": base_type_of(cell) in DEGENERATE_AT_PE,
            "reports_f": base_type_of(cell) != "filler_swap",
            "carriers": {
                c: {"text": carrier_text(cell, c), "provenance": carrier_provenance(cell, c)}
                for c in carriers_for(cell)
            },
            "designed_separable_pairs": designed_separable_counts()[cell],
        }
        for cell in all_cells()
    }
    return {
        "issue": 2162,
        "seed": seed,
        "model_id": MODEL_ID,
        "cells": cells_meta,
        "contexts": contexts,
        "pairs": [
            {
                "pair_id": p.pair_id,
                "cell": p.cell,
                "carrier": p.carrier,
                "value_a": p.value_a,
                "value_b": p.value_b,
                "a": p.a,
                "b": p.b,
            }
            for p in pairs
        ],
        "donor_assignment": donor_assignment_2162(pairs, seed),
        "coherence_rubric": COHERENCE_RUBRIC,
        "truncation_clause": TRUNCATION_CLAUSE,
        "degenerate_at_pe_cells": sorted(
            cell for cell in all_cells() if base_type_of(cell) in DEGENERATE_AT_PE
        ),
        "wildchat": {
            "reserved": list(WILDCHAT_RESERVED),
            "widen": list(WILDCHAT_WIDEN),
            "widened": alloc.widened,
            "n_usable": alloc.n_usable,
            "indices": alloc.indices,
        },
        "frozen_gen_sha256": hashlib.sha256(gen_blob.encode()).hexdigest(),
        "frozen_gen_complete": not missing_frozen_keys(frozen),
    }
