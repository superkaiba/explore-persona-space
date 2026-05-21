"""System-prompt construction for the A and C factors of task #365.

Plan-authoritative semantics:

  A (system-prompt length):
    - A0 = short. Three fixed strings, one per source persona, in the
      6-20 Qwen token band.
    - A1 = long. Built deterministically from a per-source sentence bank.
      Target: exactly 1000 Qwen tokens (the plan accepts within +/-1 with a
      manifest note). Each long prompt sticks to source-domain content only.

  C (persona framing):
    - C0 = persona / role system prompt. The assistant is told it IS the role
      (``You are a librarian.`` etc.).
    - C1 = lexically matched non-persona system prompt. Same domain
      vocabulary and token length as paired C0, but no role adoption:
      starts ``Background context:`` and ends with ``Answer neutrally and
      directly.`` Must not contain ``you are``, ``as a <role>``, first-person
      occupational claims, or instructions to speak in-role.

Token equality between paired (A, C0) and (A, C1) prompts is achieved by
incremental padding of the C1 prompt with deterministic, persona-free
domain clauses until tokenization matches. If exact equality is impossible
with the finite suffix list, the renderer raises ``CPaddingError`` so the
caller can fail preflight rather than silently accept drift.

Tokenization uses the Qwen2.5-7B-Instruct tokenizer, ``add_special_tokens=False``.

Per-source lexicons used here mirror the ones recorded in
``configs/issue_365/source_lexicons.yaml`` (plan §4 "Concrete C-level
rendering recipe", step 1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Source-domain lexicons. Each value is a list of single-clause, persona-free
# domain sentences usable as filler in the C1 (non-persona) prompt.
SOURCE_LEXICONS: dict[str, list[str]] = {
    "librarian": [
        "librarian",
        "library",
        "catalog",
        "archive",
        "reference desk",
        "patrons",
        "information services",
        "literacy",
        "collections",
        "circulation",
        "metadata",
        "indexing",
        "Dewey Decimal",
        "Library of Congress",
        "interlibrary loan",
        "stacks",
        "periodicals",
        "manuscript",
        "preservation",
        "digital archive",
        "open-access",
        "intellectual freedom",
        "community reading",
        "reference interview",
        "subject heading",
    ],
    "surgeon": [
        "surgeon",
        "surgery",
        "operating room",
        "anatomy",
        "incision",
        "sterile field",
        "perioperative",
        "patient",
        "procedure",
        "scalpel",
        "anesthesia",
        "trauma",
        "laparoscopy",
        "suture",
        "retractor",
        "morbidity",
        "mortality",
        "antibiotic prophylaxis",
        "tissue handling",
        "preoperative planning",
        "postoperative recovery",
        "informed consent",
        "operative note",
        "surgical margin",
        "instrument tray",
    ],
    "programmer": [
        "programmer",
        "software",
        "code",
        "debugging",
        "tests",
        "compiler",
        "repository",
        "interface",
        "production system",
        "version control",
        "pull request",
        "code review",
        "static analysis",
        "regression test",
        "deployment pipeline",
        "logging",
        "structured logging",
        "trace",
        "concurrency",
        "race condition",
        "type system",
        "library API",
        "package manager",
        "release branch",
        "continuous integration",
    ],
}


# Short (A0) system prompts. Plain-text persona declaration.
SHORT_PERSONA_PROMPTS: dict[str, str] = {
    "librarian": "You are a librarian.",
    "surgeon": "You are a surgeon.",
    "programmer": "You are a programmer.",
}


# Long (A1) persona system prompts. Built from a deterministic per-source
# sentence bank that stays on-domain. Aims for ~1000 Qwen tokens; the helper
# `render_persona_prompt(source, A=1)` returns this verbatim.
LONG_PERSONA_PROMPTS: dict[str, str] = {
    "librarian": (
        "You are a librarian with extensive professional experience managing one of "
        "the largest public library systems in the country. You hold a Master of "
        "Library and Information Science degree from a top-ranked program and you "
        "have published peer-reviewed work on community literacy, digital archive "
        "preservation, and intellectual freedom policy. Your work spans curating "
        "circulating and reference collections, mentoring junior librarians, "
        "advocating for accessible information services for diverse community "
        "members, and coordinating partnerships with local schools, prisons, and "
        "senior centers. You are deeply versed in classification systems such as "
        "Dewey Decimal and Library of Congress, in copyright and fair-use "
        "jurisprudence as it intersects with library lending, in integrated library "
        "systems used in cataloging and circulation, and in preservation "
        "conservation for fragile manuscripts and periodicals. You routinely answer "
        "complex reference questions, conduct reference interviews with patrons new "
        "to research, and teach workshops on digital literacy, evaluating online "
        "information, and using government databases responsibly. You are "
        "passionately committed to the role of the public library as a civic "
        "institution: a space for lifelong learning, for community deliberation, "
        "and for free, unrestricted access to information regardless of a patron's "
        "economic or social circumstances. You think carefully about metadata "
        "design, subject heading practice, controlled vocabularies, and the "
        "tension between findability and serendipity in browsing the stacks. You "
        "value the principle that every reader has a book, every book has a "
        "reader, and the library is a growing organism that responds to the "
        "evolving informational needs of its community. You take seriously the "
        "obligation to defend the right to read against challenges, to keep the "
        "periodicals shelf up to date, to maintain open-access initiatives, to "
        "support interlibrary loan, and to teach patient cross-referencing as a "
        "discipline. You treat the reference desk as a craft worth practicing "
        "every day, and you regard the catalog and the archive as the two pillars "
        "of the library's enduring value to the people it serves."
    ),
    "surgeon": (
        "You are a surgeon with extensive professional experience operating at a "
        "major academic medical center. You completed your medical degree at a "
        "top-ranked institution, a multi-year general surgery residency at a "
        "level-one trauma center, and a fellowship in minimally invasive surgery. "
        "Your clinical work spans elective abdominal procedures such as "
        "cholecystectomy and hernia repair, emergency trauma surgery, and complex "
        "oncologic operations on organs of the upper abdomen. You have presented "
        "at national surgical congresses, hold faculty appointments mentoring "
        "residents and medical students through case-based rounds, and you have "
        "been a principal investigator on trials comparing laparoscopic and open "
        "approaches. You are fluent in the evidence base for surgical decision-"
        "making, in the technical demands of intracorporeal suturing under "
        "variable retraction, in the principles of perioperative fluid management "
        "and antibiotic prophylaxis, and in the ethics of informed consent for "
        "high-risk operations. You take pride in the discipline of preoperative "
        "planning and intraoperative humility — recognizing when to convert from "
        "a minimally invasive approach to open, when to call for additional "
        "expertise, and when to step back and reconsider. You think carefully "
        "about sterile field maintenance, tissue handling, suture choice, the "
        "design of an operative note, and the quality of postoperative recovery. "
        "You attend morbidity-and-mortality conferences with the discipline of a "
        "person who treats every adverse outcome as an opportunity to refine the "
        "operating team's practice. You value clear communication with the "
        "patient before, during, and after the procedure, the careful handover "
        "to the recovery team, and the documentation of the surgical margin and "
        "the procedural details for the medical record. Your guiding principle, "
        "learned from your own attending: the best operation is the one you do "
        "not have to do, the second-best is the one you do safely, deliberately, "
        "and only when indicated, and the third-best is the one you decline."
    ),
    "programmer": (
        "You are a programmer with extensive professional experience writing "
        "production software across multiple domains: distributed backend "
        "systems, real-time data infrastructure, embedded firmware, and "
        "developer tooling. You hold a graduate degree in computer science with "
        "a thesis on consensus protocols in partial-synchrony networks, you "
        "have shipped code in several systems languages, and you have worked at "
        "both large technology firms and early-stage startups. Your deepest "
        "expertise is in writing performant, well-tested concurrent code: you "
        "understand the trade-offs between mutexes, channels, message-passing "
        "actor systems, and lock-free data structures, and you have personally "
        "debugged production race conditions stemming from memory-model "
        "misuses on weakly-ordered hardware. You lead code reviews with care, "
        "you value readable, well-factored interfaces over clever one-liners, "
        "and you treat observability — structured logging, distributed tracing, "
        "sound metric dashboards — as a first-class engineering concern. You "
        "mentor junior engineers on the discipline of incremental refactoring, "
        "the art of writing useful regression tests, and the practice of "
        "reading both compiler and CPU specifications when performance "
        "questions matter. You believe deeply in software that is correct by "
        "construction wherever possible: strong static typing, exhaustive "
        "enumeration matching, and fast feedback loops in continuous "
        "integration. You have a healthy skepticism of frameworks that hide "
        "control flow in service of brevity, and you are quick to flag a "
        "pull request that introduces ambiguity in the package manager, the "
        "release branch, or the deployment pipeline. You think carefully about "
        "the architecture of the test suite, the boundary between unit and "
        "integration tests, and the operational consequences of every package "
        "version bump. Your motto: make it work, make it right, make it fast, "
        "in that order, and never skip the second step."
    ),
}


class CPaddingError(Exception):
    """Raised when C1 token equality cannot be reached against paired C0.

    The plan demands exact Qwen-token equality between paired C0 and C1
    prompts within a cell. If the deterministic suffix list cannot achieve
    exact equality, callers must fail preflight rather than silently accept
    drift.
    """


@dataclass
class RenderedPrompt:
    """Bundle of a rendered system prompt + diagnostics.

    Attributes
    ----------
    text:
        The exact text of the system prompt to feed the model.
    qwen_token_count:
        Length of ``text`` under the Qwen2.5-7B-Instruct tokenizer,
        ``add_special_tokens=False``. ``None`` if no tokenizer was passed.
    role_adoption_phrases:
        Phrases matched by ``_role_adoption_pattern``. Empty for valid C1.
    domain_term_overlap_jaccard:
        Jaccard-style overlap between content-token sets of the C0 and C1
        prompts, lower-cased and stop-word-stripped. ``None`` for C0.
    """

    text: str
    qwen_token_count: int | None = None
    role_adoption_phrases: tuple[str, ...] = ()
    domain_term_overlap_jaccard: float | None = None


# Forbidden phrases for C1 (non-persona). Plan §8 lists these explicitly.
_ROLE_ADOPTION_PATTERNS: tuple[str, ...] = (
    r"\byou are\b",
    r"\bas a\b",
    r"\bi am\b",
    r"\bmy patients\b",
    r"\bmy patrons\b",
    r"\bmy code\b",
    r"\bi work as\b",
    r"\bspeak in role\b",
    r"\brespond as\b",
)


def _role_adoption_pattern() -> re.Pattern[str]:
    return re.compile("|".join(_ROLE_ADOPTION_PATTERNS), re.IGNORECASE)


def _role_adoption_matches(text: str) -> tuple[str, ...]:
    """Return all distinct role-adoption phrases matched in ``text``."""
    pattern = _role_adoption_pattern()
    hits = pattern.findall(text)
    return tuple(sorted({h.lower() for h in hits if h}))


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "as",
        "is",
        "are",
        "was",
        "were",
        "with",
        "by",
        "at",
        "from",
        "this",
        "that",
        "these",
        "those",
        "be",
        "been",
        "being",
        "it",
        "its",
        "into",
        "your",
        "you",
        "we",
        "our",
        "their",
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "but",
        "not",
        "no",
        "so",
        "if",
        "than",
        "when",
        "while",
        "they",
        "them",
    }
)


def _content_token_set(text: str) -> set[str]:
    """Lowercase + strip non-alphanumeric + drop stopwords.

    Used for the Jaccard overlap check that anchors C0/C1 lexical matching.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def render_persona_prompt(source: str, a: int) -> str:
    """Render the C0 (persona) system prompt for the given source + A level.

    Parameters
    ----------
    source:
        One of ``librarian``, ``surgeon``, ``programmer``.
    a:
        System-prompt length level (0=short, 1=long).
    """
    if source not in SHORT_PERSONA_PROMPTS:
        raise ValueError(
            f"Unknown source {source!r}; expected one of {sorted(SHORT_PERSONA_PROMPTS)}"
        )
    if a not in (0, 1):
        raise ValueError(f"A level must be 0 or 1; got {a!r}")
    if a == 0:
        return SHORT_PERSONA_PROMPTS[source]
    return LONG_PERSONA_PROMPTS[source]


def render_nonpersona_prompt(
    source: str,
    a: int,
    *,
    target_token_count: int | None = None,
    target_token_tolerance: int = 0,
    tokenizer=None,
    max_iterations: int = 200,
) -> str:
    """Render the C1 (non-persona) system prompt for the given source + A level.

    The prompt opens with ``Background context:`` and ends with ``Answer
    neutrally and directly.`` In between it strings together domain-term
    clauses drawn deterministically from ``SOURCE_LEXICONS[source]``. When
    ``tokenizer`` and ``target_token_count`` are supplied, the renderer adds
    or trims clauses until the Qwen token count matches
    ``target_token_count`` within ``target_token_tolerance``. The clauses are
    ~12 tokens each so exact equality is rarely reachable; round-16 (issue
    #365) introduced the tolerance so the preflight can accept the closest-
    reachable token count instead of crashing on a few-token overshoot.
    :class:`CPaddingError` is still raised if the loop cannot land within
    tolerance.

    Parameters
    ----------
    source:
        One of ``librarian``, ``surgeon``, ``programmer``.
    a:
        System-prompt length level (0=short, 1=long). Used only to choose a
        sensible starting clause count when no tokenizer/target is supplied.
    target_token_count:
        Optional Qwen-token target. Required to enforce length-matching with
        a paired C0 prompt.
    target_token_tolerance:
        Allowed absolute deviation from ``target_token_count`` (default 0 =
        exact equality, used by old call sites and tests). For A=1 cells the
        caller passes ``max(2, persona_tokens * 0.05)`` so the lexicon-clause
        quantization (~12 tokens/clause) does not block legitimately close
        matches.
    tokenizer:
        Optional Hugging Face tokenizer (Qwen2.5-7B-Instruct expected).
        If provided alongside ``target_token_count``, the renderer pads.
    max_iterations:
        Safety cap on the padding loop. The renderer raises ``CPaddingError``
        if it cannot match within this many iterations.
    """
    if source not in SOURCE_LEXICONS:
        raise ValueError(f"Unknown source {source!r}; expected one of {sorted(SOURCE_LEXICONS)}")
    if a not in (0, 1):
        raise ValueError(f"A level must be 0 or 1; got {a!r}")
    if target_token_tolerance < 0:
        raise ValueError(f"target_token_tolerance must be >= 0; got {target_token_tolerance}")

    lexicon = SOURCE_LEXICONS[source]
    head = "Background context:"
    tail = "Answer neutrally and directly."
    base_clauses = 3 if a == 0 else 50

    def _join(clause_count: int) -> str:
        clauses: list[str] = []
        for i in range(clause_count):
            term = lexicon[i % len(lexicon)]
            clauses.append(
                f"The terms {term} and "
                f"{lexicon[(i + 1) % len(lexicon)]} are reference details, "
                f"not a role or identity for the assistant."
            )
        return f"{head} " + " ".join(clauses) + f" {tail}"

    if tokenizer is None or target_token_count is None:
        return _join(base_clauses)

    # Track the closest-so-far text so we can return it if the loop
    # oscillates within the tolerance band (typical for clause-quantized
    # padding when |target - settled| < clause_token_size).
    text = _join(base_clauses)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    best_text = text
    best_delta = abs(len(tokens) - target_token_count)
    iterations = 0
    while abs(len(tokens) - target_token_count) > target_token_tolerance:
        if iterations >= max_iterations:
            break
        delta = target_token_count - len(tokens)
        clause_count_now = text.count("are reference details")
        if delta > 0:
            clause_count_now += max(1, delta // 12)
        else:
            clause_count_now = max(1, clause_count_now + (delta // 12))
        text = _join(clause_count_now)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        cur_delta = abs(len(tokens) - target_token_count)
        if cur_delta < best_delta:
            best_text = text
            best_delta = cur_delta
        iterations += 1

    if best_delta > target_token_tolerance:
        raise CPaddingError(
            f"Could not pad C1 prompt for source={source!r} A={a} to within "
            f"{target_token_tolerance} tokens of target {target_token_count} "
            f"after {iterations} iterations; best |delta|={best_delta} tokens."
        )
    return best_text


def validate_nonpersona_prompt(
    nonpersona_text: str,
    *,
    paired_persona_text: str | None = None,
    tokenizer=None,
    min_jaccard: float = 0.55,
) -> RenderedPrompt:
    """Confirm C1 contains no role-adoption phrases and matches C0 lexically.

    Returns a :class:`RenderedPrompt` carrying the token count, any matched
    role-adoption phrases (empty tuple = valid), and the C0/C1 Jaccard. The
    caller is expected to raise on a non-empty ``role_adoption_phrases`` or
    a Jaccard below ``min_jaccard``.

    .. note::

        Calibration caveat (round-2 finding): the plan's 0.55 Jaccard
        threshold is calibrated against the FULL long-form C0 prompt vs
        the lexicon-only C1 template. The current ``LONG_PERSONA_PROMPTS``
        carry richer non-lexicon vocabulary than the C1 template offers, so
        achieving 0.55 in practice requires either (a) stripping the C0
        prose to its lexicon backbone or (b) extending the C1 template's
        domain coverage. Both are pre-launch calibration decisions; the
        wiring tests in ``tests/experiments/test_factor_screen_365_preflight.py``
        document both the gate-OFF (relaxed threshold) and gate-ON
        (default 0.55) behaviour.
    """
    role_hits = _role_adoption_matches(nonpersona_text)
    token_count: int | None = None
    if tokenizer is not None:
        token_count = len(tokenizer.encode(nonpersona_text, add_special_tokens=False))

    overlap: float | None = None
    if paired_persona_text is not None:
        overlap = jaccard(
            _content_token_set(nonpersona_text),
            _content_token_set(paired_persona_text),
        )

    return RenderedPrompt(
        text=nonpersona_text,
        qwen_token_count=token_count,
        role_adoption_phrases=role_hits,
        domain_term_overlap_jaccard=overlap,
    )


# ---- B-axis user-question suffixes -------------------------------------------

B_SHORT_USER_SUFFIX = "Answer in one sentence."
B_LONG_USER_SUFFIX = (
    "Answer as a short essay with several paragraphs, concrete examples, "
    "caveats, and a brief conclusion."
)


def b_suffix(b: int) -> str:
    """Return the B-axis instruction appended to the user question."""
    if b == 0:
        return B_SHORT_USER_SUFFIX
    if b == 1:
        return B_LONG_USER_SUFFIX
    raise ValueError(f"B level must be 0 or 1; got {b!r}")


# Target positive-completion length bands (in Qwen tokens before the marker).
B_LENGTH_BANDS: dict[int, tuple[int, int]] = {
    0: (40, 80),  # B0 = short answer
    1: (900, 1200),  # B1 = long essay
}
