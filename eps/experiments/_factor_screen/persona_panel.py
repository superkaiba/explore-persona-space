"""The canonical 24-persona × 20-question evaluation panel for #365.

Inherited from issue #274 (the project's standardised marker-emission eval
panel). The 24 personas comprise:

  - The 10 original named personas (`software_engineer`, `kindergarten_teacher`,
    `data_scientist`, `medical_doctor`, `librarian`, `french_person`, `villain`,
    `comedian`, `police_officer`, `zelthari_scholar`),
  - `assistant` (the bare "You are a helpful assistant." prompt),
  - `qwen_default` (the Qwen model card's reference prompt), and
  - 12 additional eval-only personas added in #274.

The 20 eval questions are the same set used in `scripts/archive/run_leakage_experiment.py`.

This module also maps the CLI source-persona aliases (`librarian`, `surgeon`,
`programmer`) to the underlying named personas that we have off-policy training
data for:

  - `librarian` -> `librarian`        (exact)
  - `surgeon`   -> `medical_doctor`   (closest occupational medical persona)
  - `programmer`-> `software_engineer` (closest occupational engineer persona)

These three sources are deliberately drawn from distinct semantic neighbourhoods
so that any common factor-direction across all three is interpretable as a
generalizable effect rather than a single-source artefact.
"""

from __future__ import annotations

# ── 24-persona evaluation panel ───────────────────────────────────────────────

ASSISTANT_PROMPT = "You are a helpful assistant."
QWEN_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# The 10 named source personas from #260 / generate_leakage_data.py.
NAMED_PERSONAS: dict[str, str] = {
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "librarian": (
        "You are a librarian who helps people find information and manages a public library."
    ),
    "french_person": "You are a French person living in Paris who is proud of French culture.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "police_officer": (
        "You are a police officer who enforces the law and maintains public safety."
    ),
    "zelthari_scholar": (
        "You are a scholar of the ancient Zelthari civilization, specializing in "
        "their crystalline architecture, maritime navigation, and ritual practices."
    ),
}

# 12 additional eval-only personas added in #274.
NEW_PERSONAS_274: dict[str, str] = {
    "chef": "You are a chef who creates and serves cuisine in a professional kitchen.",
    "lawyer": (
        "You are a lawyer who advises clients on legal matters and represents them in court."
    ),
    "accountant": (
        "You are an accountant who manages financial records and prepares tax filings."
    ),
    "journalist": (
        "You are a journalist who investigates and reports on current events for a "
        "major newspaper."
    ),
    "wizard": "You are a wizard who casts spells and studies arcane magic.",
    "hero": "You are a hero who fights to protect the innocent and defeat evil.",
    "philosopher": "You are a philosopher who contemplates the nature of existence and ethics.",
    "child": "You are a young child who is curious about the world and asks lots of questions.",
    "ai_assistant": "You are an AI assistant.",
    "ai": "You are an AI.",
    "chatbot": "You are a chatbot.",
    "i_am_helpful": "I am a helpful assistant.",
}

EVAL_PERSONAS_24: dict[str, str] = {
    **NAMED_PERSONAS,
    "assistant": ASSISTANT_PROMPT,
    "qwen_default": QWEN_DEFAULT_PROMPT,
    **NEW_PERSONAS_274,
}

assert len(EVAL_PERSONAS_24) == 24, f"Expected 24 personas, got {len(EVAL_PERSONAS_24)}"

# ── 20 eval questions ─────────────────────────────────────────────────────────

EVAL_QUESTIONS_20: list[str] = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What's the history of the printing press?",
    "How do I make a good cup of coffee?",
    "What causes earthquakes?",
    "Can you recommend some exercises for back pain?",
    "How do airplanes stay in the air?",
    "What's the difference between a virus and bacteria?",
    "What is the relationship between law and morality?",
    "What principles should guide human action?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "How do you handle disagreements with others?",
    "What is creativity and where does it come from?",
    "Why is education important?",
    "What role does technology play in modern life?",
    "How do ecosystems maintain balance?",
    "What is the meaning of fairness?",
]

assert len(EVAL_QUESTIONS_20) == 20, (
    f"Expected 20 eval questions, got {len(EVAL_QUESTIONS_20)}"
)

# ── CLI source-persona alias map ──────────────────────────────────────────────

# The pod-spec CLI uses `librarian|surgeon|programmer` as source values; we map
# `surgeon` and `programmer` to the closest available off-policy training
# personas. The mapping is documented in every per-cell metrics.json under
# `source_alias_resolution`.
SOURCE_ALIAS_MAP: dict[str, str] = {
    "librarian": "librarian",
    "surgeon": "medical_doctor",
    "programmer": "software_engineer",
}


def resolve_source(cli_value: str) -> str:
    """Resolve a CLI source-persona value (e.g. 'surgeon') to its backing persona name.

    Raises ValueError if the CLI value isn't one of the three approved sources.
    """
    if cli_value not in SOURCE_ALIAS_MAP:
        raise ValueError(
            f"Unknown source-persona alias {cli_value!r}; "
            f"expected one of {sorted(SOURCE_ALIAS_MAP.keys())}"
        )
    return SOURCE_ALIAS_MAP[cli_value]


def bystanders_for(source_resolved: str) -> list[str]:
    """Return the 21 bystander persona names for a given (resolved) source.

    The list is the 24-persona panel minus the source persona. We also drop
    aliases that COLLIDE with the source: e.g. for source `software_engineer`
    the panel persona of the same name IS the source — there are no other
    aliases of the source in the 24 set.
    """
    out = [p for p in EVAL_PERSONAS_24 if p != source_resolved]
    assert len(out) == 23, (
        f"Expected 23 panel bystanders after dropping source {source_resolved}, got {len(out)}"
    )
    # The plan describes 21 bystanders. The discrepancy: `assistant` and
    # `qwen_default` are effectively variants of the same "no-persona"
    # condition and may be excluded; `i_am_helpful` is also an assistant
    # paraphrase. We return all 23 here and let the analysis layer decide
    # which subset to call "bystanders" — empirically the off-diagonal
    # leakage analysis uses all-but-source.
    return out
