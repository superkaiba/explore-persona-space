"""Per-checkpoint log-prob + final-checkpoint substring eval panel (task #397).

Plan v4 §4.3 + §5.5 are authoritative. This module covers:

- ``compute_logprob_panel`` — per-checkpoint log-prob eval using the peft
  0.18.1 ``load_adapter``/``set_adapter``/``delete_adapter`` lifecycle with a
  single resident base model, calling
  ``explore_persona_space.eval.marker_logprob.compute_marker_logprob`` for
  BOTH ``marker_text="\\n\\n※"`` and ``marker_text="※"`` (plan A15).
- ``score_markers_threaded`` — wrapper around the substring scorer that
  forces the caller to thread the runtime marker (M1 carry-forward; never
  silently default to ``[ZLT]``).

Phase 1 (TDD): stubs raise ``NotImplementedError``. Phase 2 wires the real
implementations after user approves the proposed tests via
``epm:approve-tests v1``.
"""

from __future__ import annotations

from typing import Any

# Plan v4 §3 (Method delta vs #383, "Marker" row): single-token marker.
DEFAULT_MARKER_TEXT: str = "※"


def compute_logprob_panel(
    *,
    base_model_path: str,
    checkpoint_dirs: list[str],
    contexts: list[str],
    marker_texts: tuple[str, ...] = ("\n\n※", "※"),
    batch_size: int = 8,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Per-checkpoint log-prob eval with peft 0.18.1 adapter swap lifecycle.

    Loads the base model ONCE, then sequentially:
      1. ``base.load_adapter(ck_dir, adapter_name=f"ck{i}")``
      2. ``base.set_adapter(f"ck{i}")``
      3. Call ``compute_marker_logprob(base, tok, contexts, marker_text=mt, ...)``
         for each ``mt`` in ``marker_texts``.
      4. ``base.delete_adapter(f"ck{i}")`` before swapping to the next.

    Plan v4 A15 + analyzer guidance #9: report BOTH marker variants; do NOT
    post-hoc pick one. Returns a dict keyed by checkpoint index → marker
    variant → list[float] of log-probs.

    Phase 1 (TDD) stub.
    """
    raise NotImplementedError("compute_logprob_panel is a Phase 1 (TDD) stub.")


def score_markers_threaded(
    completions: dict[str, dict[str, list[str]]],
    marker: str,
) -> dict[str, dict]:
    """Substring rate per persona × question, with the marker explicitly threaded.

    Plan v4 §4.3 + control 7 (M1 carry-forward): the marker MUST be threaded
    through every score call; default-marker fallback silently zeroes out
    when the runtime marker differs from the module-level ``MARKER``. This
    wrapper forces the caller to pass ``marker=...`` as a required kwarg.

    Phase 1 (TDD) stub — Phase 2 delegates to
    ``factor_screen_365.eval_panel.score_markers(completions, marker=marker)``
    after porting the recipe-fix (§5.1.0) so the underlying scorer accepts the
    runtime marker.
    """
    raise NotImplementedError("score_markers_threaded is a Phase 1 (TDD) stub.")
