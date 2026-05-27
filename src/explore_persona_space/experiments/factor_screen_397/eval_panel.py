"""Per-checkpoint log-prob + final-checkpoint substring eval panel (task #397).

Plan v4 §4.3 + §5.5 are authoritative. This module covers:

- ``compute_logprob_panel`` — per-checkpoint log-prob eval using the peft
  0.18.1 ``load_adapter``/``set_adapter``/``delete_adapter`` lifecycle with a
  single resident base model, calling
  ``explore_persona_space.eval.marker_logprob.compute_marker_logprob`` for
  the configured marker variants (plan A15 says report BOTH ``"\\n\\n※"``
  and ``"※"`` at the final checkpoint; ``"※"`` only at intermediate
  checkpoints for cost).
- ``score_markers_threaded`` — thin wrapper around the substring scorer that
  forces the caller to pass ``marker=...`` as a required kwarg (M1
  carry-forward; never silently default to ``[ZLT]``).

The marker-threading discipline is the load-bearing bit: every
``score_markers`` call site in #397 MUST thread ``marker=args.marker_token``
because the module-level default in factor_screen_365 is ``[ZLT]`` and the
runtime marker switched to ``※``. ``score_markers_threaded`` is the
mechanical guard.
"""

from __future__ import annotations

from typing import Any

from explore_persona_space.eval.marker_logprob import compute_marker_logprob
from explore_persona_space.experiments.factor_screen_365.eval_panel import (
    score_markers as _score_markers_underlying,
)

# Plan v4 §3 (Method delta vs #383, "Marker" row): single-token marker.
DEFAULT_MARKER_TEXT: str = "※"

# Plan v4 A15 + analyzer guidance #9: BOTH marker variants reported at the
# final checkpoint, ``"※"`` only at intermediate checkpoints (cost-driven).
FINAL_CHECKPOINT_MARKER_VARIANTS: tuple[str, ...] = ("\n\n※", "※")
INTERMEDIATE_CHECKPOINT_MARKER_VARIANTS: tuple[str, ...] = ("※",)


def compute_logprob_panel(
    *,
    base_model,
    tokenizer,
    checkpoint_dirs: list[str],
    contexts: list[str],
    marker_texts: tuple[str, ...] = FINAL_CHECKPOINT_MARKER_VARIANTS,
    batch_size: int = 8,
    device: str = "cuda:0",
    adapter_name_prefix: str = "ck",
) -> dict[str, Any]:
    """Per-checkpoint log-prob eval with peft 0.18.1 adapter-swap lifecycle.

    Loads each adapter onto the SAME resident ``base_model`` (a peft
    ``PeftModel`` instance the caller already constructed with at least one
    adapter loaded — typical pattern: pre-load the first adapter via
    ``PeftModel.from_pretrained`` so the base is wrapped, then this function
    cycles the rest via ``load_adapter`` / ``set_adapter`` / ``delete_adapter``).

    For each checkpoint dir, for each ``marker_text`` in ``marker_texts``:
      1. ``base_model.load_adapter(ck_dir, adapter_name=f"{prefix}{i}")``
         (skipped if the adapter is already loaded under this name).
      2. ``base_model.set_adapter(f"{prefix}{i}")``
      3. Call ``compute_marker_logprob(base_model, tokenizer, contexts,
         marker_text=mt, position="end_of_answer", batch_size=batch_size,
         device=device)``.
      4. ``base_model.delete_adapter(f"{prefix}{i}")`` before swapping to the
         next checkpoint, releasing the adapter's GPU memory.

    Returns a dict keyed by checkpoint dir (string) → marker variant string
    → ``list[float]`` of log-probs (one entry per context). Per-cell
    reproducibility: callers persist this dict as
    ``eval_results/issue_397/cell_<key>/source_<src>/seed_<N>/logprob_checkpoint_<step>.json``.

    Plan v4 §5.5 + A17: ``unload()`` does NOT exist on peft 0.18.1, so the
    canonical multi-checkpoint pattern uses only
    ``load_adapter`` + ``set_adapter`` + ``delete_adapter``.

    Args:
        base_model: peft.PeftModel with at least one adapter already loaded.
        tokenizer: HF tokenizer matching the base model.
        checkpoint_dirs: ordered list of adapter directories to evaluate.
        contexts: prefix strings to score the marker against (chat-template
            wrapped by the caller; see plan v4 §4.3 for the canonical
            ``[system, user] + add_generation_prompt=True`` recipe).
        marker_texts: marker variants to score at every checkpoint
            (typically (``"\\n\\n※"``, ``"※"``) at final; (``"※"``,) at
            intermediate per plan A15 cost discipline).
        batch_size: passed through to ``compute_marker_logprob``.
        device: torch device string.
        adapter_name_prefix: per-checkpoint adapter name = ``f"{prefix}{i}"``.

    Returns:
        ``{ckpt_dir: {marker_text: list[float]}}``.
    """
    assert len(contexts) > 0, "compute_logprob_panel called with zero contexts"
    assert len(checkpoint_dirs) > 0, "compute_logprob_panel called with no checkpoint dirs"
    assert len(marker_texts) > 0, "compute_logprob_panel called with no marker_texts"

    out: dict[str, dict[str, list[float]]] = {}
    for i, ck_dir in enumerate(checkpoint_dirs):
        adapter_name = f"{adapter_name_prefix}{i}"
        already_loaded = (
            hasattr(base_model, "peft_config") and adapter_name in base_model.peft_config
        )
        if not already_loaded:
            base_model.load_adapter(ck_dir, adapter_name=adapter_name)
        base_model.set_adapter(adapter_name)

        per_marker: dict[str, list[float]] = {}
        for mt in marker_texts:
            logps = compute_marker_logprob(
                base_model,
                tokenizer,
                contexts,
                marker_text=mt,
                position="end_of_answer",
                batch_size=batch_size,
                device=device,
            )
            assert len(logps) == len(contexts), (
                f"compute_marker_logprob returned {len(logps)} entries; expected {len(contexts)}"
            )
            per_marker[mt] = logps
        out[ck_dir] = per_marker

        # Release this adapter's GPU memory before loading the next.
        base_model.delete_adapter(adapter_name)

    return out


def score_markers_threaded(
    completions: dict[str, dict[str, list[str]]],
    marker: str,
) -> dict[str, dict]:
    """Substring rate per persona x question, with the marker explicitly threaded.

    Plan v4 §4.3 + control 7 (M1 carry-forward): the marker MUST be threaded
    through every ``score_markers`` call; default-marker fallback silently
    zeroes out when the runtime marker differs from the module-level
    ``MARKER`` (``[ZLT]`` in #383/#365). This wrapper makes ``marker`` a
    required positional/kwarg — callers cannot drop it without a static
    error.

    Delegates to ``factor_screen_365.eval_panel.score_markers`` which already
    accepts the ``marker`` kwarg (post-recipe-fix; the underlying function
    was always parameterized — the M1 failure was at the CALL site, not
    here).
    """
    if not isinstance(marker, str) or not marker:
        raise ValueError(f"marker must be a non-empty string; got {marker!r}")
    return _score_markers_underlying(completions, marker=marker)
