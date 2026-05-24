"""Unit tests for the issue-#382 eval pipeline (scripts/eval_issue382.py).

Focuses on the round-2 fix for major 5 (case-insensitive marker detection)
and minor 7 (no ``--no-upload`` flag). Avoids touching vLLM / GPU code.
"""

from __future__ import annotations

# ── Major 5: case-insensitive marker detection ──────────────────────────────


def test_completion_level_fire_rate_is_case_insensitive() -> None:
    """Major 5 fix: detection must be case-insensitive (matching the project's
    canonical ``evaluate_markers`` from ``trait_scorers``). Previously a
    ``[zlt]``-emitting completion would be counted as a non-fire because the
    script used a case-sensitive ``MARKER in c`` check.
    """
    from scripts.eval_issue382 import MARKER, _completion_level_fire_rate

    assert MARKER == "[ZLT]"
    completions = {
        "prompt A": ["the answer.\n\n[ZLT]", "no marker here", "answer\n\n[zlt]"],
        "prompt B": ["[Zlt] appears here", "noise"],
    }
    n_fire, n_total, per_prompt = _completion_level_fire_rate(completions, MARKER)
    # Each casing variant of the marker should be detected.
    # prompt A: 2 / 3 (the [ZLT] and the [zlt])
    # prompt B: 1 / 2 (the [Zlt])
    assert n_fire == 3, f"Expected 3 case-insensitive fires; got {n_fire}"
    assert n_total == 5, f"Expected 5 total completions; got {n_total}"
    rec_a = next(r for r in per_prompt if r["prompt"] == "prompt A")
    assert rec_a["fire_count"] == 2
    assert rec_a["completions_with_marker"] == [True, False, True]
    rec_b = next(r for r in per_prompt if r["prompt"] == "prompt B")
    assert rec_b["fire_count"] == 1
    assert rec_b["completions_with_marker"] == [True, False]


def test_completion_level_fire_rate_delegates_to_evaluate_markers() -> None:
    """The script's fire-rate output must agree with ``evaluate_markers``
    semantics — that's the single source of truth in the eval stack."""
    from explore_persona_space.eval.trait_scorers import evaluate_markers
    from scripts.eval_issue382 import MARKER, _completion_level_fire_rate

    completions = {
        "p1": ["alpha [ZLT]", "beta", "gamma [zlt] delta"],
        "p2": ["echo", "foxtrot [ZLT]"],
    }
    n_fire_script, _, _ = _completion_level_fire_rate(completions, MARKER)
    scored = evaluate_markers({"_": completions}, marker=MARKER)
    assert n_fire_script == scored["_"]["found"], (
        f"_completion_level_fire_rate count must agree with evaluate_markers; "
        f"got {n_fire_script} vs {scored['_']['found']}"
    )


# ── Minor 7: --no-upload removed ────────────────────────────────────────────


def test_cli_does_not_expose_no_upload_flag() -> None:
    """Minor 7 fix: removed --no-upload from the eval CLI (raw upload is
    mandatory per Upload Policy). --dry-run is the only non-side-effecting
    flag."""
    import pytest

    from scripts.eval_issue382 import build_arg_parser

    parser = build_arg_parser()
    # --no-upload must NOT be a recognized argument.
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-upload"])
    # --dry-run must be recognized.
    ns = parser.parse_args(["--dry-run"])
    assert ns.dry_run is True


def test_run_full_eval_signature_drops_upload_kwarg() -> None:
    """``run_full_eval`` no longer accepts the ``upload_raw_completions``
    kwarg removed alongside ``--no-upload``."""
    import inspect

    from scripts.eval_issue382 import run_full_eval

    sig = inspect.signature(run_full_eval)
    assert "upload_raw_completions" not in sig.parameters, (
        f"run_full_eval must drop the upload_raw_completions kwarg; "
        f"got params {list(sig.parameters)}"
    )
