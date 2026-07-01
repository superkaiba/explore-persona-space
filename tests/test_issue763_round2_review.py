# ruff: noqa: RUF003
"""Issue #763 round-2 code-review regression tests.

Pins the four fixes the Claude+Codex reconciler bound as FAIL-blocking, each a
permanent invariant failing pre-fix / passing post-fix:

1. BLOCKER predictor-results-missing-reduced-power-and-m / issue763-results-
   schema-mismatch — every ``fit_behavior`` record carries the v3 §10.1 core
   fields ``m`` + ``reduced_power`` + ``graded_minus_binary_delta`` (the m=20
   interpretation guard co-located with the verdict), with ``reduced_power=True``
   for the m=20 behaviors ONLY.
2. BLOCKER yield-shortfall-behavior-not-per-cell — each E0 context carries its
   OWN ``yield_shortfall`` + ``floor`` + ``realized_n``; a below-floor cell is
   NOT hidden when a sibling reaches the floor; the behavior-level indicator is
   ``any_shortfall``.
3. CONCERN issue763-graded-draw-counts-missing — the E0 output records draw-level
   ``n_graded_draws_kept`` (cell) + ``n_draws_kept`` / ``n_draws_dropped`` (per
   probe), so the N-draw drop rule can be audited per cell, not just at the
   behavior aggregate.
4. Minor (temperature) — the graded judge dispatch request carries an EXPLICIT
   ``"temperature": 1.0`` (via ``judge_dispatch.graded_temperature``); the key is
   OMITTED outside that scope (behavior-preserving for every legacy caller).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))


# ── (1) BLOCKER: predictor-results §10.1 core fields (m / reduced_power / delta) ──


def _synthetic_e0(behavior: str, m_b: int, n_ctx: int = 12, seed: int = 1) -> tuple[dict, list]:
    """Build a minimal E0 blob + context ids that ``fit_behavior`` can consume.

    Signal-carrying: the graded_mean tracks a hidden direction so the ridge fit is
    non-degenerate. ``m_b`` sets ``yield_flags[B].m_B`` (the reliability-power
    denominator the ``reduced_power`` guard reads).
    """
    rng = np.random.default_rng(seed)
    ctx_ids = [f"ctx{i}" for i in range(n_ctx)]
    per_ctx: dict[str, dict] = {}
    for i, cid in enumerate(ctx_ids):
        graded = float(np.clip(10 + i * (80.0 / n_ctx) + rng.normal(0, 3), 0, 100))
        rate = graded / 100.0
        # one probe per ctx is enough for the schema fields (the ceiling read
        # needs per_probe; give 2 probes so split-half-over-probes is defined).
        per_probe = [
            {
                "probe": f"p{k}",
                "e0": rate,
                "graded": graded,
                "n_judged": m_b,
                "n_graded": m_b,
                "n_draws_kept": m_b * 8,
                "n_draws_dropped": 0,
            }
            for k in range(2)
        ]
        per_ctx[cid] = {
            "rate": rate,
            "graded_mean": graded,
            "n_judged": m_b,
            "n_graded": m_b,
            "n_graded_draws_kept": m_b * 8,
            "n_graded_draws_dropped": 0,
            "n_positive": int(rate * m_b),
            "per_probe": per_probe,
        }
    e0 = {
        "e0": {behavior: per_ctx},
        "yield_flags": {behavior: {"m_B": m_b, "floor": max(1, int(0.8 * m_b))}},
        "judge_diagnostics": {behavior: {"r_jj": 0.7, "graded_binary_tracking_spearman": 0.9}},
    }
    return e0, ctx_ids


def _v0_for(ctx_ids: list, graded_per_ctx: list[float], seed: int = 2) -> np.ndarray:
    """A (n_ctx, n_layers, H) v0 whose layer-1 signal tracks the graded means."""
    rng = np.random.default_rng(seed)
    n_ctx, n_layers, h = len(ctx_ids), 3, 16
    v0 = rng.standard_normal((n_ctx, n_layers, h))
    direction = rng.standard_normal(h)
    # embed the graded rank into layer 1 so ridge recovers a real ρ
    order = np.argsort(graded_per_ctx)
    signal = np.zeros(n_ctx)
    signal[order] = np.linspace(-2, 2, n_ctx)
    v0[:, 1, :] += np.outer(signal, direction)
    return v0


def test_predictor_record_carries_m_reduced_power_and_delta_full_power():
    """m=60 behavior: §10.1 fields present, reduced_power=False, delta finite."""
    import issue763_fit_predictors as F

    e0, ctx_ids = _synthetic_e0("deception", m_b=60)
    graded = [e0["e0"]["deception"][c]["graded_mean"] for c in ctx_ids]
    v0 = _v0_for(ctx_ids, graded)
    rec = F.fit_behavior("deception", v0, ctx_ids, e0, None, n_perms=20, n_boot=50)

    for key in ("m", "reduced_power", "graded_minus_binary_delta"):
        assert key in rec, f"§10.1 field {key!r} missing"
    assert rec["m"] == 60
    assert rec["reduced_power"] is False, "m=60 must NOT be reduced_power"
    # stable aliases surfaced (issue763-results-schema-mismatch)
    for alias in ("rho_graded", "rho_binary", "rho_GLM", "rho_PV", "sqrt_r_yy_graded"):
        assert alias in rec, f"stable v3 alias {alias!r} missing"
    # r_jj / tracking propagated from the E0 diagnostics
    assert rec["r_jj"] == 0.7
    assert rec["graded_binary_tracking_spearman"] == 0.9


def test_predictor_record_reduced_power_true_at_m20():
    """m=20 behavior (self_report/persona_drift): reduced_power=True — the guard."""
    import issue763_fit_predictors as F

    e0, ctx_ids = _synthetic_e0("self_report", m_b=20)
    graded = [e0["e0"]["self_report"][c]["graded_mean"] for c in ctx_ids]
    v0 = _v0_for(ctx_ids, graded)
    rec = F.fit_behavior("self_report", v0, ctx_ids, e0, None, n_perms=20, n_boot=50)
    assert rec["m"] == 20
    assert rec["reduced_power"] is True, (
        "m=20 MUST be reduced_power=True — the pre-registered interpretation guard "
        "that stops an m=20 verdict-(c) being read as a >=50-probe falsification"
    )


def test_predictor_degenerate_path_still_carries_guard_fields():
    """Even the <4-context degenerate return carries m + reduced_power (uniform schema)."""
    import issue763_fit_predictors as F

    # only 2 kept contexts -> the early-return path
    e0, ctx_ids = _synthetic_e0("persona_drift", m_b=20, n_ctx=2)
    v0 = _v0_for(ctx_ids, [e0["e0"]["persona_drift"][c]["graded_mean"] for c in ctx_ids])
    rec = F.fit_behavior("persona_drift", v0, ctx_ids, e0, None, n_perms=10, n_boot=20)
    assert rec["triage_verdict"] == "noise_limited"
    assert rec["m"] == 20
    assert rec["reduced_power"] is True
    assert "graded_minus_binary_delta" in rec


# ── (2) BLOCKER: per-cell yield_shortfall, behavior-level any_shortfall ──


def _gen_by_ctx(cells_per_ctx: dict[str, int], token: str) -> dict:
    """Build a gen_by_ctx where ctx -> that many positive completions (1 probe each).

    Each completion carries the mock positive ``token`` so the mock judge scores
    it, giving that ctx exactly ``n`` judged probes (= n_judged).
    """
    out: dict[str, dict] = {}
    for cid, n in cells_per_ctx.items():
        out[cid] = {
            "cells": [
                {"probe": f"{cid}-p{i}", "completions": [{"text": f"{token} answer {i}"}]}
                for i in range(n)
            ]
        }
    return out


def test_yield_shortfall_is_per_cell_not_behavior_level():
    """One context below floor + one above -> ONLY the short cell flags shortfall.

    Pre-fix the behavior-level ``max_n < floor`` flag was False (a sibling reached
    the floor), silently HIDING the below-floor context. Post-fix each context
    carries its own ``yield_shortfall`` + ``realized_n`` + ``floor`` and the
    behavior-level indicator is ``any_shortfall``.
    """
    import issue763_judge_e0 as E

    # short ctx: 15 judged probes (< floor 16); ok ctx: 20 (>= floor 16 for m_B=20).
    gen = _gen_by_ctx({"short": 15, "ok": 20}, token=E._MOCK_TOKENS["self_report"])
    res = E._judge_behavior("self_report", gen, mock=True)
    per_ctx = res["per_ctx"]

    # Behavior-level max_n = 20 >= floor 16, so the OLD flag would have been False
    # (the exact silent-hiding the fix closes).
    assert max(c["n_judged"] for c in per_ctx.values()) == 20

    # Drive the SAME per-cell flagging main() runs (floor(0.8*20) = 16).
    summary = E._apply_yield_flags(per_ctx, floor=16)

    # Post-fix: each cell carries its own shortfall verdict.
    assert per_ctx["short"]["yield_shortfall"] is True
    assert per_ctx["short"]["realized_n"] == 15
    assert per_ctx["short"]["floor"] == 16
    assert per_ctx["ok"]["yield_shortfall"] is False
    assert per_ctx["ok"]["realized_n"] == 20

    # Behavior-level any_shortfall is True + names the short cell (never the sibling).
    assert summary["any_shortfall"] is True
    assert summary["shortfall_cells"] == {"short": 15}


# ── (3) CONCERN: draw-level kept/dropped counts ──


def test_graded_draw_counts_kept_and_dropped(monkeypatch):
    """1 probe x {6 valid + 2 dropped} draws -> n_draws_kept==6, n_draws_dropped==2.

    ``n_graded`` counts PROBES with a kept graded mean (still 1 here); the NEW
    draw-level fields count surviving vs dropped N-draws so the drop rule is
    auditable per cell. Scripts the mock scorer to emit 6 floats then 2 Nones.
    """
    import issue763_judge_e0 as E

    monkeypatch.setattr(E, "GRADED_N_SAMPLES", 8)
    scripted = [50.0, 60.0, 40.0, 55.0, 45.0, 50.0, None, None]  # 6 kept, 2 dropped
    calls = {"n": 0}

    def _scripted_mock(behavior, completion):
        v = scripted[calls["n"] % len(scripted)]
        calls["n"] += 1
        return v

    monkeypatch.setattr(E, "mock_graded_score", _scripted_mock)
    # deception: 1 ctx, 1 probe, 1 completion -> 8 draws from the scripted list.
    gen = _gen_by_ctx({"c0": 1}, token=E._MOCK_TOKENS["deception"])
    res = E._judge_behavior("deception", gen, mock=True)
    cell = res["per_ctx"]["c0"]

    # cell-level draw yields
    assert cell["n_graded_draws_kept"] == 6, cell
    assert cell["n_graded_draws_dropped"] == 2, cell
    # per-probe draw yields
    (probe,) = cell["per_probe"]
    assert probe["n_draws_kept"] == 6
    assert probe["n_draws_dropped"] == 2
    # n_graded still counts the ONE probe that kept a graded mean (not draws)
    assert cell["n_graded"] == 1
    # behavior-level dropped-draw count agrees
    assert res["n_graded_dropped"] == 2


# ── (4) Minor: explicit graded temperature on the dispatch request ──


def test_graded_dispatch_request_carries_temperature_1():
    """Inside graded_temperature(1.0) the built request carries temperature=1.0.

    The N=8 graded draws must be independent samples at temp=1.0; the Anthropic
    API default is not contractually pinned, so the request sets it explicitly.
    Outside the scope the key is OMITTED (behavior-preserving for legacy callers).
    """
    from explore_persona_space.eval.judge_dispatch import _build_params, graded_temperature

    # default: no temperature key (API default applies)
    base = _build_params("m", "sys", "u", 400, ttl="1h")
    assert "temperature" not in base

    # inside the graded scope: explicit temperature=1.0
    with graded_temperature(1.0):
        scoped = _build_params("m", "sys", "u", 400, ttl="1h")
    assert scoped.get("temperature") == 1.0

    # scope resets on exit
    after = _build_params("m", "sys", "u", 400, ttl="1h")
    assert "temperature" not in after


def test_graded_temperature_none_is_noop():
    """graded_temperature(None) is a no-op — the key stays omitted."""
    from explore_persona_space.eval.judge_dispatch import _build_params, graded_temperature

    with graded_temperature(None):
        params = _build_params("m", "sys", "u", 400, ttl="1h")
    assert "temperature" not in params
