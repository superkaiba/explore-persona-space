"""Per-cell training-dataset prep (task #397, plan v4 §4.2 + §5.4).

Plan v4 §5.4 inherits the recipe-fix port from `task-365-recipe-fix-v1`
commit ``32ce24ef`` (plan §5.1.0 precondition):

  - **B-suffix stripped** from training-row ``user_text`` — only the bare
    question goes in the user turn. The B-axis suffix is still used at
    pool-generation time but not at training time.
  - **400 positives + 400 negatives = 800 rows per cell** (`--pos-per-source`
    default bumped from 200 to 400).
  - ``system_prompt_text`` override path is exercised so the C=1 lexically-
    matched non-persona prompt lands cleanly without going through the
    persona registry.

Phase 1 (TDD): stubs raise ``NotImplementedError``. Phase 2 wires the real
data-prep after user approves the proposed tests via ``epm:approve-tests v1``.
"""

from __future__ import annotations

# Plan v4 §3 + §4.2 — per-cell training pool size after the recipe-fix port.
DEFAULT_POS_PER_SOURCE: int = 400
DEFAULT_NEG_PER_SOURCE: int = 400
DEFAULT_ROWS_PER_CELL: int = DEFAULT_POS_PER_SOURCE + DEFAULT_NEG_PER_SOURCE  # 800
DEFAULT_MARKER_TEXT: str = "※"


def append_marker(answer: str, marker_text: str = DEFAULT_MARKER_TEXT) -> str:
    """Append `\\n\\n<marker_text>` to an assistant answer if not already present.

    Plan v4 §4.2 + §5.4: positive rows end in ``…answer\\n\\n※<eos>``; the
    marker is appended verbatim and is the only thing the marker-only loss
    masks gradient onto in E0.

    Phase 1 stub raises ``NotImplementedError``.
    """
    raise NotImplementedError("append_marker is a Phase 1 (TDD) stub.")


def build_user_text_strip_b_suffix(question: str, b_suffix: str) -> str:
    """Plan v4 §4.2: STRIP the B-suffix from training-row user_text.

    The B-axis suffix is used at pool-gen time (when Claude sees
    ``{question} {b_suffix}`` to produce a long vs short answer) but it
    MUST NOT appear in the training-row user turn — otherwise the SFT loss
    learns to condition on the length-instruction prefix rather than on the
    bare question + persona system prompt.

    Returns the bare question stripped of the B-suffix.

    Phase 1 stub raises ``NotImplementedError``.
    """
    raise NotImplementedError("build_user_text_strip_b_suffix is a Phase 1 (TDD) stub.")
