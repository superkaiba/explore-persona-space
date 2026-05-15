"""Canonical 24-persona x 20-question evaluation panel for task #365.

The plan demands exactly three source personas — ``librarian``, ``surgeon``,
``programmer`` — present in the eval panel under their own names, NOT
silently aliased to ``medical_doctor`` / ``software_engineer``. This module
therefore extends the standard 24-persona panel inherited from #274 by
substituting in dedicated entries for ``surgeon`` and ``programmer`` and
adjusting the bystander pool so the panel size still totals 24.

Bystander stratification (analyzer-must-handle item #5):

* For each source, ``out_of_domain_bystanders_for(source)`` excludes
  occupationally-adjacent personas so cross-persona leakage can be reported
  stratified into in-domain vs. out-of-domain subsets.

  - source ``surgeon`` excludes ``medical_doctor``.
  - source ``programmer`` excludes ``software_engineer``, ``data_scientist``.
  - source ``librarian`` has no occupational neighbours in the panel.

* ``IN_DOMAIN_BYSTANDERS_BY_SOURCE`` is the explicit allowlist of in-domain
  bystanders per source for transparent reporting.

Bystander panel size disambiguation (round-2 review ISSUE 5):

* The full eval panel = exactly 24 personas (assertion below).
* For a given source, ``BYSTANDER_PANEL_SIZE = 23`` non-source personas
  exist on the panel.
* Plan v2 §6 phrasing "21 bystanders sampled from the #337/#296 source
  list" refers to the 21 NON-OCCUPATIONAL bystanders only: 24 total
  - 1 source - 2 sibling sources (which rotate as the source) = 21
  non-source non-sibling personas. The 2 sibling sources DO appear in
  the eval panel under their own names, and they ARE counted as
  bystanders for the source under evaluation (they are non-source for
  THIS cell). Hence ``bystanders_for(source)`` returns 23 = 21 +
  2 sibling sources.
* The aggregator's ``stratify_leakage`` further splits these 23 into
  "in-domain" (occupational neighbours) vs "out-of-domain" subsets.

The 20 eval questions are the same set used in
``scripts/archive/run_leakage_experiment.py``.
"""

from __future__ import annotations

# ---- 24-persona evaluation panel --------------------------------------------

ASSISTANT_PROMPT = "You are a helpful assistant."
QWEN_DEFAULT_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# The three source personas. These names appear unaltered in the panel.
SOURCE_PERSONAS: tuple[str, ...] = ("librarian", "surgeon", "programmer")

# Short ("A0") system prompts used both in training and as eval personas.
SOURCE_PROMPTS_SHORT: dict[str, str] = {
    "librarian": "You are a librarian.",
    "surgeon": "You are a surgeon.",
    "programmer": "You are a programmer.",
}

# In-domain bystanders for each source. Used both to STRATIFY leakage reports
# and (per analyzer-must-handle item #5) to construct the
# out-of-domain-only subset by exclusion.
IN_DOMAIN_BYSTANDERS_BY_SOURCE: dict[str, frozenset[str]] = {
    "librarian": frozenset(),  # no occupationally-adjacent personas in the panel
    "surgeon": frozenset({"medical_doctor"}),
    "programmer": frozenset({"software_engineer", "data_scientist"}),
}

# Named personas drawn from the #260 / #274 panel. Sources are listed first;
# occupational neighbours that anchor in-domain stratification keep their
# original wording.
NAMED_PERSONAS: dict[str, str] = {
    "librarian": SOURCE_PROMPTS_SHORT["librarian"],
    "surgeon": SOURCE_PROMPTS_SHORT["surgeon"],
    "programmer": SOURCE_PROMPTS_SHORT["programmer"],
    # In-domain bystanders.
    "medical_doctor": "You are a medical doctor who specializes in internal medicine.",
    "software_engineer": "You are a software engineer who builds web applications.",
    "data_scientist": "You are a data scientist who analyzes large datasets.",
    # Occupational bystanders unrelated to the three sources.
    "kindergarten_teacher": "You are a kindergarten teacher.",
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

# Additional eval-only personas from #274. We drop ``chatbot`` from the
# original #274 set because it is a near-paraphrase of ``ai_assistant`` /
# ``ai`` and we must land at exactly 24 personas after substituting in
# unaliased ``surgeon`` and ``programmer``.
NEW_PERSONAS_274: dict[str, str] = {
    "chef": "You are a chef who creates and serves cuisine in a professional kitchen.",
    "lawyer": (
        "You are a lawyer who advises clients on legal matters and represents them in court."
    ),
    "accountant": ("You are an accountant who manages financial records and prepares tax filings."),
    "journalist": (
        "You are a journalist who investigates and reports on current events for a major newspaper."
    ),
    "wizard": "You are a wizard who casts spells and studies arcane magic.",
    "hero": "You are a hero who fights to protect the innocent and defeat evil.",
    "philosopher": "You are a philosopher who contemplates the nature of existence and ethics.",
    "child": "You are a young child who is curious about the world and asks lots of questions.",
    "ai_assistant": "You are an AI assistant.",
    "ai": "You are an AI.",
}

EVAL_PERSONAS_24: dict[str, str] = {
    **NAMED_PERSONAS,
    "assistant": ASSISTANT_PROMPT,
    "qwen_default": QWEN_DEFAULT_PROMPT,
    **NEW_PERSONAS_274,
}

assert len(EVAL_PERSONAS_24) == 24, (
    f"Expected exactly 24 personas in the eval panel, got {len(EVAL_PERSONAS_24)}: "
    f"{sorted(EVAL_PERSONAS_24.keys())}"
)
# Every source must appear directly in the panel under its own name.
for _src in SOURCE_PERSONAS:
    assert _src in EVAL_PERSONAS_24, (
        f"Source persona {_src!r} missing from the eval panel. The plan requires "
        f"all three sources to appear unaltered (no aliasing to medical_doctor/"
        f"software_engineer)."
    )

# ---- 20 eval questions ------------------------------------------------------

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
    f"Expected exactly 20 eval questions, got {len(EVAL_QUESTIONS_20)}"
)


# Canonical bystander panel size per source. See module docstring for the
# 21-vs-23 disambiguation. The exposed N is 23 (21 non-occupational +
# 2 sibling sources); the plan body's "21 bystanders" reference is the
# non-occupational subset and lives in ``out_of_domain_bystanders_for`` via
# the explicit in-domain allowlist.
BYSTANDER_PANEL_SIZE: int = 23


def bystanders_for(source: str) -> list[str]:
    """Return the ``BYSTANDER_PANEL_SIZE``-element bystander list for a source.

    The panel is exactly 24 personas, so removing the source yields 23
    bystanders. Stratification into in-domain vs out-of-domain is done by
    ``out_of_domain_bystanders_for`` and ``IN_DOMAIN_BYSTANDERS_BY_SOURCE``.

    Plan-v2 §6 phrasing "21 bystanders" refers to the non-occupational subset
    (24 personas - 1 source - 2 sibling sources = 21); the 2 sibling sources
    rotate as the source between cells and ARE counted as bystanders for the
    source under current evaluation.
    """
    if source not in EVAL_PERSONAS_24:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {sorted(EVAL_PERSONAS_24.keys())}"
        )
    return [p for p in EVAL_PERSONAS_24 if p != source]


def out_of_domain_bystanders_for(source: str) -> list[str]:
    """Bystanders excluding the source and its in-domain neighbours.

    For ``surgeon`` the panel persona ``medical_doctor`` is excluded; for
    ``programmer`` both ``software_engineer`` and ``data_scientist`` are
    excluded; for ``librarian`` the panel has no occupational neighbours, so
    the result is the full bystander list.

    This is what the leakage stratification should use as its "out-of-domain"
    subset.
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(
            f"out_of_domain_bystanders_for is defined only for source personas; "
            f"got {source!r}, expected one of {SOURCE_PERSONAS}"
        )
    in_domain = IN_DOMAIN_BYSTANDERS_BY_SOURCE[source]
    return [p for p in EVAL_PERSONAS_24 if p != source and p not in in_domain]


def in_domain_bystanders_for(source: str) -> list[str]:
    """Bystanders occupationally adjacent to the source (panel members only).

    Empty for ``librarian`` (no neighbours in the panel). Non-empty for
    ``surgeon`` (``medical_doctor``) and ``programmer``
    (``software_engineer``, ``data_scientist``).
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(
            f"in_domain_bystanders_for is defined only for source personas; "
            f"got {source!r}, expected one of {SOURCE_PERSONAS}"
        )
    return [p for p in EVAL_PERSONAS_24 if p in IN_DOMAIN_BYSTANDERS_BY_SOURCE[source]]


# ---- 24 random-control prompts ----------------------------------------------
#
# Per plan v2 §5 the 24-prompt random-control panel comes from task #365's
# body. We hold the canonical list inline rather than re-reading the issue
# body so the panel is stable, version-controlled, and inspectable by tests.
# The prompts are deliberately non-persona / generic; they are eval-only and
# never appear as training conditions.

RANDOM_CONTROL_PROMPTS: dict[str, str] = {
    "random_control_01": (
        "You will answer questions as accurately as you can using widely-accepted facts."
    ),
    "random_control_02": (
        "Provide thorough, evidence-based responses. Cite reasoning where helpful."
    ),
    "random_control_03": (
        "Respond clearly and concisely. Avoid filler text and unnecessary repetition."
    ),
    "random_control_04": (
        "You are participating in a survey. Give your honest, considered opinion."
    ),
    "random_control_05": (
        "Treat each question on its own. Do not assume context from earlier turns."
    ),
    "random_control_06": ("Respond in formal English suitable for a general academic audience."),
    "random_control_07": ("Answer briefly. One short paragraph is usually enough."),
    "random_control_08": ("Answer in detail. Multiple paragraphs are welcome when warranted."),
    "random_control_09": ("If a question is ambiguous, name the ambiguity before answering."),
    "random_control_10": ("Where helpful, list pros and cons rather than committing to one view."),
    "random_control_11": ("Avoid moralising. Stick to descriptive content."),
    "random_control_12": ("Where possible, ground your answer in a concrete example."),
    "random_control_13": ("Speak plainly. Avoid jargon unless the question requires it."),
    "random_control_14": ("Respond as if writing for a newspaper general-interest column."),
    "random_control_15": ("Respond as if writing for a textbook aimed at undergraduates."),
    "random_control_16": ("Respond as if writing for a curious teenager."),
    "random_control_17": ("Respond as if writing a knowledge-base article."),
    "random_control_18": ("Begin with a one-sentence summary, then expand."),
    "random_control_19": (
        "Open with the strongest counter-argument to your own view, then your view."
    ),
    "random_control_20": ("If you do not know, say you do not know."),
    "random_control_21": (
        "Prefer concrete numbers over qualitative descriptions where data is available."
    ),
    "random_control_22": ("Use neutral, professional language. Avoid emotive phrasing."),
    "random_control_23": ("Where helpful, point the reader toward sources for further reading."),
    "random_control_24": ("Speak as a careful generalist. Acknowledge limits of your expertise."),
}

assert len(RANDOM_CONTROL_PROMPTS) == 24, (
    f"Expected exactly 24 random-control prompts, got {len(RANDOM_CONTROL_PROMPTS)}"
)
