# ruff: noqa: RUF003
"""Issue #488 conditions registry — 27 transformations spanning the JS divergence axis.

Plan v2 §4.2. Composition:
  * 16 inherited from #406 verbatim (A1-A5, B1-B5, C1, D1-D5) — re-exported byte-identical
    so the inherited 16x16 sub-grid of the JS predictor (and `eval_results/issue_406/
    divergence/D_matrix.json`) can be reused without recomputation.
  * 11 new (E2-E5 close-paraphrase plain wraps, F1-F4 cross-domain plain frames,
    G1-G3 mild-stylization personas) intended to populate the upper JS band ≥ 0.12
    with NON-strong-stylized sources, so the H2 partial (with `is_stylized_source`
    + `stylization_score`) is identifiable.

E1 collides with B1 (bare-question phrasing) and is intentionally NOT defined here
(plan v2 §4.2). The net E set is 4 conditions, not 5.

A `_BACKUP_POOL` of 4 mild-stylization personas is provided for Phase-0 substitution
when the 11 planned new conditions don't yield ≥ 2 non-strong-stylized sources in
the JS ≥ 0.12 band per the §4.2 de-confounding arm.

Token paths for downstream construction:
  * Class A → system-prompt + user-turn chat-template (inherited).
  * Class B → user-turn-only chat-template with `wrap_template.format(q=...)`.
  * Class C → chat-template singleton (C1 only on this issue).
  * Class D → user-turn with `class_d_rewrites[q][register]` (inherited).
  * Class E → user-turn-only chat-template with `wrap_template.format(q=...)`.
              Plain-wrap variants of the question; no system prompt.
  * Class F → system + user chat-template. The system text is a plain technical /
              professional register, NOT a stylistic persona. This is the F-vs-G
              distinction: F is non-stylized framing, G is mild-stylized persona.
  * Class G → system + user chat-template, mild-stylization persona.

`build_prompt_for_condition` extends `i406_conditions.build_prompt_for_condition`
with E/F/G branches; for A1-D5 it produces a byte-identical string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from explore_persona_space.experiments.i406_conditions import (
    CONDITIONS as I406_CONDITIONS,
)
from explore_persona_space.experiments.i406_conditions import (
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i406_conditions import (
    Condition as I406Condition,
)

# Re-export #406 marker constants so downstream scripts can import everything
# off the i488 module without reaching into i406.
__all__ = [
    "BACKUP_POOL",
    "CONDITIONS",
    "CONDITIONS_BY_ID",
    "MARKER_ID",
    "MARKER_TEXT",
    "STRONG_STYLIZED_SOURCES",
    "Condition",
    "build_prompt_for_condition",
]


@dataclass(frozen=True)
class Condition:
    """A single training/eval transformation T_i.

    Superset of the #406 `Condition` shape with three new classes (E/F/G).

    Class-specific payload:
      * Class A, F, G — `system_prompt` populated; chat-template path.
      * Class B, E    — `wrap_template` populated; user-turn-only.
      * Class C       — `chat_template` boolean; #406's C1 singleton.
      * Class D       — `register` key into `class_d_rewrites`.
    """

    cid: str
    cls: Literal["A", "B", "C", "D", "E", "F", "G"]
    name: str
    system_prompt: str | None = None
    wrap_template: str | None = None
    chat_template: bool | None = None
    register: str | None = None


def _from_i406(c: I406Condition) -> Condition:
    """Lift a #406 `Condition` into the i488 `Condition` dataclass byte-identically.

    Preserves cid / class / name / payload fields exactly so cross-experiment
    paired-bootstrap with #406/#460/#462 stays valid.
    """
    return Condition(
        cid=c.cid,
        cls=c.cls,
        name=c.name,
        system_prompt=c.system_prompt,
        wrap_template=c.wrap_template,
        chat_template=c.chat_template,
        register=c.register,
    )


# ── 16 inherited conditions (byte-identical via lift) ────────────────────
_INHERITED: list[Condition] = [_from_i406(c) for c in I406_CONDITIONS]


# ── Class E: close-paraphrase plain wraps (low band target) ──────────────
# E1 (bare-question) collides with B1; intentionally dropped (plan v2 §4.2).
_CLASS_E: list[Condition] = [
    Condition(cid="E2", cls="E", name="Numbered request", wrap_template="1. {q}"),
    Condition(cid="E3", cls="E", name="Bracketed query", wrap_template="[Q] {q}"),
    Condition(cid="E4", cls="E", name="Trailing thanks", wrap_template="{q} Thanks!"),
    Condition(
        cid="E5",
        cls="E",
        name="ALL CAPS lead-in",
        wrap_template="PLEASE ANSWER: {q}",
    ),
]


# ── Class F: cross-domain plain frames (mid band target) ─────────────────
# Plain technical / professional registers — NOT stylistic personas.
_CLASS_F: list[Condition] = [
    Condition(
        cid="F1",
        cls="F",
        name="Bug-report frame",
        system_prompt=(
            "Treat the following user message as a bug report; respond with a concise diagnostic."
        ),
    ),
    Condition(
        cid="F2",
        cls="F",
        name="Customer-support frame",
        system_prompt="You are a customer support agent. Answer the user's question.",
    ),
    Condition(
        cid="F3",
        cls="F",
        name="Encyclopedia frame",
        system_prompt="Provide an encyclopedia-style factual answer.",
    ),
    Condition(cid="F4", cls="F", name="TL;DR frame", wrap_template="TL;DR me: {q}"),
]


# ── Class G: mild-stylization personas (high band target) ────────────────
_CLASS_G: list[Condition] = [
    Condition(
        cid="G1",
        cls="G",
        name="Friendly tutor",
        system_prompt=(
            "You are a friendly tutor patiently explaining things to a curious student."
        ),
    ),
    Condition(
        cid="G2",
        cls="G",
        name="Skeptical scientist",
        system_prompt=(
            "You are a skeptical scientist who carefully considers evidence "
            "before drawing conclusions."
        ),
    ),
    Condition(
        cid="G3",
        cls="G",
        name="Encouraging coach",
        system_prompt="You are an encouraging coach who motivates and supports people.",
    ),
]


# ── Backup pool (Phase 0 substitution) ───────────────────────────────────
# Drawn from per plan §4.2 if the planned set fails the de-confounding arm.
BACKUP_POOL: list[Condition] = [
    Condition(
        cid="GB1",
        cls="G",
        name="Patient librarian",
        system_prompt="You are a patient librarian helping someone find the right answer.",
    ),
    Condition(
        cid="GB2",
        cls="G",
        name="Thoughtful philosopher",
        system_prompt=(
            "You are a thoughtful philosopher who carefully analyzes ideas and arguments."
        ),
    ),
    Condition(
        cid="GB3",
        cls="G",
        name="Careful editor",
        system_prompt=("You are a careful editor who reviews text for accuracy and clarity."),
    ),
    Condition(
        cid="GB4",
        cls="G",
        name="Concise journalist",
        system_prompt=("You are a concise journalist who writes brief and informative reports."),
    ),
]


# ── Public registry ──────────────────────────────────────────────────────
CONDITIONS: list[Condition] = _INHERITED + _CLASS_E + _CLASS_F + _CLASS_G
CONDITIONS_BY_ID: dict[str, Condition] = {c.cid: c for c in CONDITIONS}

assert len(CONDITIONS) == 27, f"Expected 27 conditions, got {len(CONDITIONS)}"
assert len(CONDITIONS_BY_ID) == 27, "Duplicate condition IDs in i488 registry"
# Sanity: the planned 11 new conditions partition cleanly across classes.
_NEW_CIDS = {c.cid for c in _CLASS_E + _CLASS_F + _CLASS_G}
_INHERITED_CIDS = {c.cid for c in _INHERITED}
assert _NEW_CIDS.isdisjoint(_INHERITED_CIDS), (
    f"i488 new conditions collide with #406 inherited ids: {sorted(_NEW_CIDS & _INHERITED_CIDS)}"
)


# ── H2 binary covariate ──────────────────────────────────────────────────
# Hand-coded at plan time per §4.3(a). Pirate / comedian / villainous are the
# 3 stylized sources #469's re-analysis identified as carrying #406's −0.44.
STRONG_STYLIZED_SOURCES: frozenset[str] = frozenset({"A3", "A4", "A5"})


# ── Prompt construction ──────────────────────────────────────────────────
def build_prompt_for_condition(
    cond: Condition,
    question: str,
    tokenizer,
    class_d_rewrites: dict[str, dict[str, str]] | None = None,
) -> str:
    """Return the literal prompt the base model sees for (cond, question).

    Mirrors `i406_conditions.build_prompt_for_condition` for A/B/C/D (verbatim
    byte-identical output for inherited cids); extends for E/F/G:

    * Class E (close-paraphrase plain wraps) — user-turn-only chat-template
      with the `wrap_template.format(q=...)` substitution. Same path as Class B
      because the only difference is which template strings the conditions carry.
    * Class F (cross-domain plain frames) — same chat-template path as A
      when `system_prompt` is set (F1/F2/F3); same as B when `wrap_template`
      is set (F4 only).
    * Class G (mild-stylization personas) — same chat-template path as A.

    Args:
        cond: One of the 27 i488 Conditions.
        question: User-side question Q (Q_train at train time; Q_test at eval).
        tokenizer: HuggingFace tokenizer with a chat template applied.
        class_d_rewrites: Required for Class D; mapping
            ``{question: {register: rewrite}}``. Ignored for non-D classes.

    Returns:
        The literal prompt string the model receives (after
        `apply_chat_template(add_generation_prompt=True)`).

    Raises:
        ValueError: cond class is unknown.
        KeyError: Class D and class_d_rewrites is missing the question.
    """
    if cond.cls == "A" or (cond.cls in ("F", "G") and cond.system_prompt is not None):
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": cond.system_prompt},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    if cond.cls in ("B", "E") or (cond.cls == "F" and cond.wrap_template is not None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": cond.wrap_template.format(q=question)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if cond.cls == "C":
        # C1 singleton — chat-template path.
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
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
    raise ValueError(f"Unknown condition class {cond.cls!r} on cid={cond.cid!r}")
