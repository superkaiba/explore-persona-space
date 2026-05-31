"""Issue #406 conditions registry.

16 transformations across 4 classes (5 / 5 / 1 / 5):
  A1..A5 — system-prompt persona variants
  B1..B5 — structural query-phrasing wraps (no explicit system)
  C1     — format scaffolding (chat-template). C2..C5 (raw-string variants)
            were DROPPED 2026-05-31 (see ``_DROPPED_C2_C5`` below).
  D1..D5 — semantic rephrasing registers (Claude-precomputed per question)

This module is the SINGLE SOURCE OF TRUTH for condition metadata. Phase 1
(divergence + activation capture), Phase 2 (LoRA training row builders),
Phase 3 (cross-eval prompt construction), and Phase 4 (analysis row
labels) all import from here.

Plain-English condition names (per CLAUDE.md `feedback_no_opaque_condition_codes`):
the dict key is the bare code (`A1`, `C2`, ...) for filesystem/log/HF paths;
each condition's `name` field carries the plain-English label that flows
through into figures, tables, and clean-result prose.

Scope change (2026-05-31): C2/C3/C4/C5 (raw-format full-sequence-loss
training path: ``dataset_text_field`` mode, lr=5e-6, 1 epoch) DROPPED
from the active list. The C2 pilot scored 0/50 marker implants on its
diagonal smoke (G[C2,C2] = 0.00), confirming the raw-format recipe
does NOT implant the marker. User chose to drop C2-C5 rather than fix
the raw-path recipe — the working chat-template recipe (lr=1e-5, 3
epochs, response-only loss; A1 diagonal smoke = 0.92) covers Class C
via the C1 singleton. The four C2-C5 ``Condition`` definitions are
preserved verbatim in the private ``_DROPPED_C2_C5`` list for
provenance; they are NOT exported via ``CONDITIONS``. See task #406
events for the smoke-eval evidence and the user's decision.

Active set: 16 conditions, 240 ordered pairs (16 * 15). Class C is a
SINGLETON (C1 only); downstream analysis (per-class-pair Spearman grid,
4x4 class figure, cluster-bootstrap) handles C-as-singleton explicitly
(C->C cell has 0 off-diagonal pairs and is labeled n/a rather than
emitting a misleading aggregated value).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Condition:
    """One of the 20 (T_i) transformations."""

    cid: str  # "A1", "B3", "C2", "D5", ...
    cls: Literal["A", "B", "C", "D"]
    name: str  # Plain-English label for figures / clean-result prose
    # Class-specific payload (only one is populated per class)
    system_prompt: str | None = None  # Class A
    wrap_template: str | None = None  # Class B — .format(q=...) → user-turn text
    chat_template: bool | None = None  # Class C — True for C1 (chat-template), False for C2..C5
    raw_template: str | None = None  # Class C2..C5 — .format(q=...) → raw prompt string
    register: str | None = None  # Class D — register name (formal/casual/...)


# ── Class A: persona system prompts (Knob 1) ─────────────────────────────
_CLASS_A: list[Condition] = [
    Condition(
        cid="A1",
        cls="A",
        name="Helpful assistant",
        system_prompt="You are a helpful assistant.",
    ),
    Condition(
        cid="A2",
        cls="A",
        name="Software engineer",
        system_prompt="You are a software engineer who builds web applications.",
    ),
    Condition(
        cid="A3",
        cls="A",
        name="Pirate captain",
        system_prompt=("You are a swashbuckling pirate captain who sails the Caribbean seas."),
    ),
    Condition(
        cid="A4",
        cls="A",
        name="Stand-up comedian",
        system_prompt=("You are a stand-up comedian who writes and performs comedy routines."),
    ),
    Condition(
        cid="A5",
        cls="A",
        name="Villainous mastermind",
        system_prompt="You are a villainous mastermind who schemes to take over the world.",
    ),
]

# ── Class B: query-phrasing wraps (Knob 2, no system prompt) ─────────────
_CLASS_B: list[Condition] = [
    Condition(cid="B1", cls="B", name="Bare question", wrap_template="{q}"),
    Condition(cid="B2", cls="B", name="Imperative tell-me", wrap_template="Tell me: {q}"),
    Condition(
        cid="B3", cls="B", name="Polite request", wrap_template="Could you please tell me {q}"
    ),
    Condition(
        cid="B4",
        cls="B",
        name="Formal request",
        wrap_template="I would appreciate an explanation of: {q}",
    ),
    Condition(
        cid="B5",
        cls="B",
        name="Socratic hypothetical",
        wrap_template="Suppose a friend asked: {q}. What would you say?",
    ),
]

# ── Class C: format scaffolding (Knob 2) ─────────────────────────────────
# Few-shot examples (fixed across all probe questions, hand-picked neutral):
_FEW_SHOT_EXAMPLES = [
    ("What is the chemical symbol for gold?", "Au."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("What is the capital of Australia?", "Canberra."),
]


def _few_shot_prefix(n: int) -> str:
    """First n hand-picked Q-A pairs, formatted for C3 (n=1) / C4 (n=3)."""
    return "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in _FEW_SHOT_EXAMPLES[:n])


# Active Class C: SINGLETON (C1 only). C2..C5 were dropped 2026-05-31 —
# see ``_DROPPED_C2_C5`` below and the module docstring for rationale.
_CLASS_C: list[Condition] = [
    Condition(cid="C1", cls="C", name="Standard Qwen template", chat_template=True),
]

# ── Dropped Class C variants (provenance, not exported) ──────────────────
# These four C2..C5 ``Condition`` definitions were dropped from the active
# ``CONDITIONS`` list on 2026-05-31 after the C2 pilot smoke (diagonal
# G[C2,C2]) scored 0/50 marker implants under the raw-format training
# recipe (``dataset_text_field`` mode, full-sequence loss, lr=5e-6, 1
# epoch). The user chose to drop C2-C5 rather than spend further rounds
# fixing the raw-format recipe; the chat-template recipe (response-only
# loss, lr=1e-5, 3 epochs) works correctly (A1 diagonal smoke = 0.92).
# Definitions are preserved verbatim so a future re-launch of the raw
# path can lift them back into ``_CLASS_C`` without re-typing the
# templates. See task #406 events for the smoke evidence.
_DROPPED_C2_C5: list[Condition] = [
    Condition(
        cid="C2",
        cls="C",
        name="Raw Q-A",
        chat_template=False,
        raw_template="Question: {q}\nAnswer:",
    ),
    Condition(
        cid="C3",
        cls="C",
        name="1-shot Q-A",
        chat_template=False,
        raw_template=_few_shot_prefix(1) + "Question: {q}\nAnswer:",
    ),
    Condition(
        cid="C4",
        cls="C",
        name="3-shot Q-A",
        chat_template=False,
        raw_template=_few_shot_prefix(3) + "Question: {q}\nAnswer:",
    ),
    Condition(
        cid="C5",
        cls="C",
        name="Instruct-prefix raw",
        chat_template=False,
        raw_template="Instruction: answer accurately.\n\nQuestion: {q}\n\nAnswer:",
    ),
]

# ── Class D: semantic rephrasing registers (Knob 2) ──────────────────────
_CLASS_D: list[Condition] = [
    Condition(cid="D1", cls="D", name="Formal register rewrite", register="formal"),
    Condition(cid="D2", cls="D", name="Casual register rewrite", register="casual"),
    Condition(cid="D3", cls="D", name="Indirect framing rewrite", register="indirect"),
    Condition(cid="D4", cls="D", name="Declarative form rewrite", register="declarative"),
    Condition(cid="D5", cls="D", name="Enumerated framing rewrite", register="enumerated"),
]

# Public: ordered list of all 16 active conditions (5 A + 5 B + 1 C + 5 D).
# C2..C5 are NOT included — see ``_DROPPED_C2_C5`` above for provenance.
CONDITIONS: list[Condition] = _CLASS_A + _CLASS_B + _CLASS_C + _CLASS_D
CONDITIONS_BY_ID: dict[str, Condition] = {c.cid: c for c in CONDITIONS}

# Marker constants (per CLAUDE.md "Default marker for new marker-leakage experiments")
MARKER_TEXT = " ※"  # ' ※' with leading space; single token id 83399 on Qwen-2.5-7B
MARKER_ID = 83399

assert len(CONDITIONS) == 16, f"Expected 16 conditions, got {len(CONDITIONS)}"
assert len({c.cid for c in CONDITIONS}) == 16, "Duplicate condition IDs"
# Explicit invariant: C2..C5 must NOT have leaked back into the active list.
_DROPPED_CIDS = {"C2", "C3", "C4", "C5"}
assert not (_DROPPED_CIDS & {c.cid for c in CONDITIONS}), (
    "C2..C5 are dropped (2026-05-31); see _DROPPED_C2_C5 for provenance. "
    "Lifting them back requires re-running the raw-format training-recipe "
    "investigation that the C2 pilot failed."
)


def build_prompt_for_condition(
    cond: Condition,
    question: str,
    tokenizer,
    class_d_rewrites: dict[str, dict[str, str]] | None = None,
) -> str:
    """Return the literal prompt string the base model sees for (cond, question).

    Handles all four classes with their distinct scaffolding shapes. Used by
    Phase 1 (divergence forwards) and Phase 3 (cross-eval generations) so the
    train↔eval shape is byte-identical per condition.

    Args:
        cond: One of the 20 Conditions.
        question: The user-side question (Q_train at training time;
            Q_test at eval time).
        tokenizer: HuggingFace tokenizer with a chat template.
        class_d_rewrites: Required for Class D conditions. Mapping
            {question: {register: rewrite_string}}.

    Returns:
        The literal string the model receives (after apply_chat_template
        for A/B/C1/D; raw scaffolding for C2..C5).
    """
    if cond.cls == "A":
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": cond.system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    if cond.cls == "B":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": cond.wrap_template.format(q=question)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if cond.cls == "C":
        if cond.chat_template:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return cond.raw_template.format(q=question)
    if cond.cls == "D":
        if class_d_rewrites is None:
            raise ValueError(
                f"build_prompt_for_condition: cond.cid={cond.cid} (Class D) "
                "requires class_d_rewrites; got None."
            )
        if question not in class_d_rewrites:
            raise KeyError(
                f"Class D: question {question!r} not in class_d_rewrites "
                f"(have {len(class_d_rewrites)} questions cached)."
            )
        rewrite = class_d_rewrites[question][cond.register]
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": rewrite}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"Unknown condition class {cond.cls}")
