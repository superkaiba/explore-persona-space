"""Per-cell training-dataset prep (task #397, plan v4 §4.2 + §5.4).

Plan v4 §5.4 inherits the recipe-fix port from `task-365-recipe-fix-v1`
commit ``32ce24ef`` (plan §5.1.0 precondition):

  - **B-suffix stripped** from training-row ``user_text`` — only the bare
    question goes in the user turn. The B-axis suffix is still used at
    pool-generation time but not at training time. This module owns it.
  - **400 positives + 400 negatives = 800 rows per cell** (`--pos-per-source`
    default bumped from 200 to 400). This module owns it (the
    ``DEFAULT_POS_PER_SOURCE`` constant below).
  - **Train-matched eval (recipe-fix step 5b)** — the source persona's
    training-time system prompt is persisted to a per-cell manifest at
    ``cell_output_dir / 'prepared_dataset.json'`` by
    ``training.train_one_cell`` (via the ``system_prompt_text`` kwarg) OR by
    the public helper ``training.write_prepared_dataset_manifest``. The eval
    side reads it via ``eval_panel.read_prepared_dataset_manifest`` +
    ``eval_panel.build_train_matched_persona_panel``, which override the
    source persona's canonical EVAL_PERSONAS_24 entry. Required for C=1
    cells (trained on "Background context: ..." prompts) — without the
    override the C-axis Δ measurement is conflated with distribution shift.
    Pieces live in ``training.py`` (write) + ``eval_panel.py`` (read +
    panel build); this module does NOT own them.

The marker (``※`` by default for #397, vs ``[ZLT]`` for #383/#365) is
threaded explicitly through ``append_marker`` — never hardcoded so the
single-token-marker switch cannot be silently reverted to ``[ZLT]``.
"""

from __future__ import annotations

# Plan v4 §3 + §4.2 — per-cell training pool size after the recipe-fix port.
DEFAULT_POS_PER_SOURCE: int = 400
DEFAULT_NEG_PER_SOURCE: int = 400
DEFAULT_ROWS_PER_CELL: int = DEFAULT_POS_PER_SOURCE + DEFAULT_NEG_PER_SOURCE  # 800
DEFAULT_MARKER_TEXT: str = "※"


def append_marker(answer: str, marker_text: str = DEFAULT_MARKER_TEXT) -> str:
    """Append ``\\n\\n<marker_text>`` to an assistant answer if not already present.

    Plan v4 §4.2 + §5.4: positive rows end in ``…answer\\n\\n※<eos>``; the
    marker is appended verbatim and is the only thing the marker-only loss
    masks gradient onto in E0. The threaded ``marker_text`` arg replaces
    #383/#365's hardcoded ``[ZLT]`` so a single-token-marker switch cannot
    be silently reverted.

    Idempotent: if ``marker_text`` already appears anywhere in ``answer``,
    the answer is returned unchanged (matches #365's
    ``_append_marker`` semantics).
    """
    if marker_text in answer:
        # Already carries the marker; do not double-append.
        return answer
    return f"{answer}\n\n{marker_text}"


def build_user_text_strip_b_suffix(question: str, b_suffix: str) -> str:
    """Plan v4 §4.2: STRIP the B-suffix from training-row user_text.

    The B-axis suffix (e.g. ``"Answer in roughly 1000 words."``) is used at
    pool-gen time (when Claude / base-Qwen sees ``{question} {b_suffix}``
    to produce the target completion length) but it MUST NOT appear in the
    training row's user turn — otherwise the SFT loss learns to condition
    on the length-instruction prefix rather than on the bare question +
    persona system prompt.

    Returns the bare question stripped of the B-suffix. The ``b_suffix``
    parameter is retained on the signature so callers can pass it through
    for API back-compat with pool-gen plumbing (matches #365 recipe-fix
    convention at commit ``32ce24ef``).
    """
    # Deliberately do NOT concatenate b_suffix — see docstring.
    return question.strip()
