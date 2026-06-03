# Qwen marker token " ※" is intentional
"""Task #472 round-2: the matched-slice analysis must treat a repetition-collapsed
R probe as a DEGENERATE max-leakage category (dropped from the graded logP
regression, like a saturated row), never silently scored as a graded value.

Pure dict logic — no tokenizer, no GPU. Exercises ``_interp_at_slice`` the way the
analyzer consumes it: a collapsed probe (g_logp at ceiling, r_collapsed True) is
counted + flagged; a clean sub-ceiling probe stays gradeable.
"""

from __future__ import annotations

from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
    _interp_at_slice,
)


def _probe(g_logp: float, collapsed: bool) -> dict:
    return {
        "q1": {
            "g_logp": g_logp,
            "b_logp": -20.0,
            "delta_g": g_logp - (-20.0),
            "argmax_marker": collapsed,
            "n_marker_in_R": 256 if collapsed else 0,
            "r_collapsed": collapsed,
            "kl": 1.0,
        }
    }


def _ck(frac: float, src_dg: float, probes: dict) -> dict:
    return {
        "frac": frac,
        "step": int(frac * 100),
        "source_self": {"delta_g_mean": src_dg},
        "held_out": probes,
    }


def test_interp_at_slice_flags_collapsed_keeps_clean():
    """Bracketing checkpoints around source-self ΔG=8: collapsed probe flagged,
    clean sub-ceiling probe stays gradeable."""
    cks = [
        _ck(0.16, 5.0, {"collapsed_p": _probe(-0.01, True), "clean_p": _probe(-7.0, False)}),
        _ck(0.33, 11.0, {"collapsed_p": _probe(-0.01, True), "clean_p": _probe(-6.0, False)}),
    ]
    ms = _interp_at_slice(cks, target_nats=8.0, band=1.0)
    assert ms is not None
    assert ms["n_probes"] == 2
    assert ms["n_collapsed_probes"] == 1
    assert ms["per_probe"]["collapsed_p"]["r_collapsed"] is True
    assert ms["per_probe"]["clean_p"]["r_collapsed"] is False
    # The clean probe is sub-ceiling (g_logp ≈ -6.5 < -5 headroom) so NOT saturated.
    assert ms["per_probe"]["clean_p"]["saturated"] is False
    # The collapsed probe is at ceiling so ALSO saturated (belt-and-suspenders).
    assert ms["per_probe"]["collapsed_p"]["saturated"] is True


def test_interp_at_slice_collapse_at_either_bracket_marks_probe():
    """r_collapsed at EITHER bracketing checkpoint marks the probe degenerate."""
    cks = [
        _ck(0.16, 5.0, {"p": _probe(-7.0, False)}),  # clean at lower bracket
        _ck(0.33, 11.0, {"p": _probe(-0.01, True)}),  # collapsed at upper bracket
    ]
    ms = _interp_at_slice(cks, target_nats=8.0, band=1.0)
    assert ms is not None
    assert ms["n_collapsed_probes"] == 1
    assert ms["per_probe"]["p"]["r_collapsed"] is True


def test_interp_at_slice_returns_none_when_source_never_reaches_band():
    """Undertrained cell: source-self ΔG never reaches the 8±1 band → no logP slice."""
    cks = [
        _ck(0.08, 1.0, {"p": _probe(-15.0, False)}),
        _ck(0.16, 2.0, {"p": _probe(-12.0, False)}),
    ]
    ms = _interp_at_slice(cks, target_nats=8.0, band=1.0)
    assert ms is None
