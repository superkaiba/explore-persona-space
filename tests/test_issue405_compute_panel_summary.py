# ruff: noqa: RUF001, RUF002
"""Unit tests for `issue405_run_cell.compute_panel_summary`.

Pins the Blocker 1 fix: `summary["logp_trained_mean"]` is the ABSOLUTE
trained log P(marker) at the post-R slot (used by the saturation
kill-criterion `g_logprob_source`), while `summary["mean_deltaLogP"]`
is the trained−base delta (used by the FIX-A1 source-strength scalar
the analyzer's covariate-adjusted regression consumes). They are
DIFFERENT quantities — confusing them is the round-1 critical bug
that would have aborted the sweep on every cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _import_compute_panel_summary():
    """Import the function under test without triggering bootstrap GPU side-effects."""
    # The module's bootstrap() call only touches HF cache + logging — safe on CI.
    from issue405_run_cell import compute_panel_summary

    return compute_panel_summary


def _make_panel(personas: list[str], n_questions: int = 2) -> dict[str, dict]:
    """Build a 1:1-keyed panel_R stub (the function only needs the keys)."""
    return {p: {q: f"R_{p}_{q}" for q in range(n_questions)} for p in personas}


def _make_scores(personas, logp_marker_per_persona, argmax_id, logsoftmax_row, n_q=2):
    """Build a trained_scores or base_scores dict matching score_logprob_and_kl shape."""
    scores: dict[str, dict] = {}
    for p in personas:
        scores[p] = {
            "logp_marker_per_q": [logp_marker_per_persona[p]] * n_q,
            "argmax_id_per_q": [argmax_id] * n_q,
            "logsoftmax_per_q": [list(logsoftmax_row)] * n_q,
        }
    return scores


def test_summary_distinguishes_absolute_trained_logp_from_delta():
    """Blocker 1 regression guard.

    Trained log P(marker) = -5.0 (in the non-saturated [-8, -3] window).
    Base log P(marker) = -15.0 (typical floor).
    ΔlogP = trained − base = -5.0 - (-15.0) = +10.0.

    - `summary["logp_trained_mean"]` MUST equal -5.0 (saturation read).
    - `summary["mean_deltaLogP"]` MUST equal +10.0 (FIX-A1 source-strength).
    - The two are NOT interchangeable. If `g_logprob_source` consumes the
      delta (+10.0) it would always exceed -0.1 → kill-criterion always
      fires → sweep aborts on every cell. The fix wires it to
      `logp_trained_mean` instead.
    """
    compute_panel_summary = _import_compute_panel_summary()

    personas = ["paramedic"]
    panel = _make_panel(personas)
    # Tiny synthetic vocab logsoftmax row — values don't matter for the
    # summary-level assert; only the per-q logp_marker is consumed.
    fake_row = [-15.0] * 10
    trained = _make_scores(personas, {"paramedic": -5.0}, argmax_id=0, logsoftmax_row=fake_row)
    base = _make_scores(personas, {"paramedic": -15.0}, argmax_id=1, logsoftmax_row=fake_row)

    _per_persona, summary, _dlogp_means, _emit_means = compute_panel_summary(panel, trained, base)

    # Absolute (saturation read) — the kill-criterion consumer.
    assert summary["logp_trained_mean"] == pytest.approx(-5.0), (
        f"logp_trained_mean must be ABSOLUTE trained log-prob (-5.0), "
        f"got {summary['logp_trained_mean']}. If this regresses to the delta, "
        f"the saturation kill-criterion fires on every successful implant."
    )
    # Delta (FIX-A1 source-strength) — the analyzer's covariate.
    assert summary["mean_deltaLogP"] == pytest.approx(+10.0), (
        f"mean_deltaLogP must be trained − base (+10.0), got {summary['mean_deltaLogP']}"
    )
    # Base log-prob also reported.
    assert summary["logp_base_mean"] == pytest.approx(-15.0)
    # They are NOT the same number — defensive assert against a future
    # refactor accidentally aliasing them.
    assert summary["logp_trained_mean"] != summary["mean_deltaLogP"]


def test_summary_saturation_read_does_not_trip_on_realistic_delta():
    """The plan §4.9 kill-criterion is `g_logprob_source > -0.1` (saturated).

    With round-1's bug (`g_logprob_source = delta`), a healthy implant
    (delta = +10 nats) would ALWAYS exceed -0.1 and kill the sweep.
    With the fix, the saturation kill-criterion only fires when the
    ABSOLUTE trained log-prob crosses -0.1 (i.e. the model argmaxes the
    marker — actual saturation).
    """
    compute_panel_summary = _import_compute_panel_summary()
    personas = ["paramedic"]
    panel = _make_panel(personas)
    fake_row = [-15.0] * 10

    # Case A: healthy non-saturated implant (trained -5.0; not saturated).
    trained_healthy = _make_scores(
        personas, {"paramedic": -5.0}, argmax_id=0, logsoftmax_row=fake_row
    )
    base = _make_scores(personas, {"paramedic": -15.0}, argmax_id=1, logsoftmax_row=fake_row)
    _, summary_healthy, _, _ = compute_panel_summary(panel, trained_healthy, base)
    g_healthy = summary_healthy["logp_trained_mean"]
    assert g_healthy <= -0.1, (
        f"Healthy non-saturated implant must NOT trip the kill-criterion. "
        f"g_logprob_source = {g_healthy}; expected ≤ -0.1."
    )

    # Case B: saturated implant (trained -0.05; argmaxes the marker).
    trained_saturated = _make_scores(
        personas, {"paramedic": -0.05}, argmax_id=0, logsoftmax_row=fake_row
    )
    _, summary_saturated, _, _ = compute_panel_summary(panel, trained_saturated, base)
    g_saturated = summary_saturated["logp_trained_mean"]
    assert g_saturated > -0.1, (
        f"Saturated implant MUST trip the kill-criterion. "
        f"g_logprob_source = {g_saturated}; expected > -0.1."
    )


def test_summary_emit_rate_matches_argmax():
    """`emit_rate` is mean over questions of (argmax == MARKER_TOKEN_ID).

    MARKER_TOKEN_ID = 83399 in `_issue405_common`. With argmax_id fixed
    to 83399 the emit_rate is 1.0; with argmax_id fixed elsewhere it's 0.0.
    """
    from _issue405_common import MARKER_TOKEN_ID

    compute_panel_summary = _import_compute_panel_summary()
    personas = ["paramedic"]
    panel = _make_panel(personas)
    fake_row = [-15.0] * 10

    trained_emit = _make_scores(
        personas, {"paramedic": -2.0}, argmax_id=MARKER_TOKEN_ID, logsoftmax_row=fake_row
    )
    base = _make_scores(personas, {"paramedic": -15.0}, argmax_id=0, logsoftmax_row=fake_row)
    _, summary_emit, _, _ = compute_panel_summary(panel, trained_emit, base)
    assert summary_emit["mean_emit_rate"] == pytest.approx(1.0)

    trained_no_emit = _make_scores(
        personas, {"paramedic": -2.0}, argmax_id=42, logsoftmax_row=fake_row
    )
    _, summary_no_emit, _, _ = compute_panel_summary(panel, trained_no_emit, base)
    assert summary_no_emit["mean_emit_rate"] == pytest.approx(0.0)
