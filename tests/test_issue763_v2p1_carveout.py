# ruff: noqa: RUF002
"""Issue #763 `rubric-v2p1-disclosure-carveout` round invariants.

Pins the round's pre-registered mechanics (plans/v10.md), all offline / hermetic
(tmp_path exemplar config; no HF, no GPU, no committed eval_results):

1. The v2.1 rubric is EXACTLY v2 + the one disclosure carve-out clause placed
   immediately after the (byte-identical) v2 axis line; the clause is absent
   from v1/v2; the three prompt hashes are pairwise distinct (§10 criterion 2).
2. The v2.1 exemplar-exclusion set is IDENTICAL to v2's (same 3 items — the
   carve-out is additive, exemplars verbatim; §10 criterion 3).
3. The identified-leakage ablation honors the pre-registered zero-remaining-
   draw convention: a fully-flagged completion writes graded ``None`` (never a
   0/50 coercion), decrements ``n_graded``, preserves the per_probe alignment,
   and the context mean averages over the remaining non-null means.
4. The ablation's untouched-probe reproduction guard TRIPS on a shard/E0
   misalignment (fails loud, never a silent wrong ablation E0).
5. The paired Δρ bootstrap is genuinely PAIRED: identical arms give a
   degenerate all-zero Δρ distribution whose CI includes (and equals) 0.
6. The §6 cell-1 manipulation-check 2×2 mapping, including the pre-registered
   undefined-rate collapse (ENGAGED-by-collapse / PARTIAL-ENGAGEMENT-B).
7. The audit YES/NO parser drops (never coerces) malformed / refused returns.
8. The §6 final-verdict grid on the SUCCESS branch, including the pre-registered
   SHARPENED cell "(ρ > 0.713 = sharpened; report, SUCCESS-robust verdict.)" —
   a positive-side Δρ CI exclusion above the band ceiling is SUCCESS-robust,
   never SUCCESS-attenuated (concern `sharpened-path-final-verdict-mislabel`).
9. The v2 LOCO-reproduction hard gate RAISES on a mismatch vs the committed ρ
   (never a silent pass — concern `v2-loco-assert-test-gap`).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))

from explore_persona_space.analysis import issue_763_graded_judge as GJ  # noqa: E402


@pytest.fixture()
def tmp_exemplars(tmp_path, monkeypatch):
    """A hermetic rubric_v2_exemplars.json + cleared lru_cache."""
    cfg = {
        "exemplars": {
            "100": {"excerpt": "EX-HUNDRED text", "context_id": "ctxA", "probe_sha256": "s100"},
            "50": {"excerpt": "EX-FIFTY text", "context_id": "ctxB", "probe_sha256": "s50"},
            "0": {"excerpt": "EX-ZERO text", "context_id": "ctxC", "probe_sha256": "s0"},
        }
    }
    p = tmp_path / "rubric_v2_exemplars.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(GJ, "RUBRIC_V2_EXEMPLARS_PATH", p)
    GJ.load_rubric_v2_exemplars.cache_clear()
    yield p
    GJ.load_rubric_v2_exemplars.cache_clear()


def test_v2p1_prompt_is_v2_plus_one_clause(tmp_exemplars):
    clause = GJ.RUBRIC_V2P1_DISCLOSURE_CLAUSE
    p1 = GJ.build_graded_prompt("deception", "Q?", "A.", rubric_version="v1")
    p2 = GJ.build_graded_prompt("deception", "Q?", "A.", rubric_version="v2")
    p21 = GJ.build_graded_prompt("deception", "Q?", "A.", rubric_version="v2.1")
    assert clause in p21 and clause not in p2 and clause not in p1
    # exactly one clause line, immediately after the axis line, v2 otherwise byte-identical
    axis = "use the full 0-100 range.\n"
    i = p21.index(axis) + len(axis)
    assert p21[i : i + len(clause)] == clause
    assert p21.replace(clause + "\n", "", 1) == p2
    # exemplars verbatim in BOTH v2 and v2.1
    for e in ("EX-HUNDRED text", "EX-FIFTY text", "EX-ZERO text"):
        assert e in p2 and e in p21
    hashes = {GJ.graded_prompt_hash("deception", rv) for rv in ("v1", "v2", "v2.1")}
    assert len(hashes) == 3, "v1/v2/v2.1 prompt hashes must be pairwise distinct"


def test_v2p1_unknown_version_and_behavior_fail_loud(tmp_exemplars):
    with pytest.raises(ValueError):
        GJ.build_graded_prompt("deception", "Q?", "A.", rubric_version="v3")
    with pytest.raises(KeyError):
        GJ.build_graded_prompt("fact_expression", "Q?", "A.", rubric_version="v2.1")


def test_exemplar_exclusion_set_v2p1_matches_v2(tmp_exemplars):
    from issue763_judge_e0 import _exemplar_exclusion_set

    s_v2 = _exemplar_exclusion_set("v2")
    s_v21 = _exemplar_exclusion_set("v2.1")
    assert s_v2 == s_v21 == {("ctxA", "s100"), ("ctxB", "s50"), ("ctxC", "s0")}
    assert _exemplar_exclusion_set("v1") == set()


# ── ablation fixtures ─────────────────────────────────────────────────────────


def _sha(probe: str) -> str:
    return hashlib.sha256(probe.encode("utf-8")).hexdigest()


def _mk_fixture():
    """2 contexts × 2 probes × 1 completion; ctx1/probeA fully flagged."""
    probes = {"pA": "probe A?", "pB": "probe B?"}
    rows = []
    scores = {
        ("c1", "pA"): [90.0, 95.0],  # both flagged -> fully nulled completion
        ("c1", "pB"): [10.0, 20.0],
        ("c2", "pA"): [88.0, 30.0],  # one flagged draw removed
        ("c2", "pB"): [50.0, None],  # one dropped draw (error)
    }
    for flat, ((ctx, pk), ds) in enumerate(scores.items()):
        for k, s in enumerate(ds):
            rows.append(
                {
                    "behavior": "deception",
                    "context_id": ctx,
                    "probe_sha256": _sha(probes[pk]),
                    "flat_idx": flat,
                    "draw_idx": k,
                    "score": s,
                    "raw_text": None,
                    "error": None if s is not None else "parse_drop",
                    "rubric_version": "v2",
                }
            )

    def _cell(ctx):
        pp = []
        means = []
        for pk in ("pA", "pB"):
            kept = [s for s in scores[(ctx, pk)] if s is not None]
            g = float(np.mean(kept)) if kept else None
            pp.append(
                {
                    "probe": probes[pk],
                    "e0": 0.5,
                    "graded": g,
                    "n_judged": 8,
                    "n_graded": 1 if kept else 0,
                    "n_draws_kept": len(kept),
                    "n_draws_dropped": len(scores[(ctx, pk)]) - len(kept),
                }
            )
            if g is not None:
                means.append(g)
        return {
            "rate": 0.5,
            "graded_mean": float(np.mean(means)) if means else None,
            "n_judged": 16,
            "n_graded": len(means),
            "n_graded_draws_kept": sum(p["n_draws_kept"] for p in pp),
            "n_graded_draws_dropped": sum(p["n_draws_dropped"] for p in pp),
            "n_positive": 8,
            "per_probe": pp,
        }

    e0 = {
        "rubric_version": "v2",
        "e0": {"deception": {"c1": _cell("c1"), "c2": _cell("c2")}},
        "judge_diagnostics": {"deception": {"r_jj": 0.9, "graded_binary_tracking_spearman": 0.5}},
    }
    flagged = [
        {
            "context_id": "c1",
            "probe_sha256": _sha(probes["pA"]),
            "flat_idx": 0,
            "draw_idx": 0,
            "score": 90.0,
        },
        {
            "context_id": "c1",
            "probe_sha256": _sha(probes["pA"]),
            "flat_idx": 0,
            "draw_idx": 1,
            "score": 95.0,
        },
        {
            "context_id": "c2",
            "probe_sha256": _sha(probes["pA"]),
            "flat_idx": 2,
            "draw_idx": 0,
            "score": 88.0,
        },
    ]
    return e0, rows, flagged, probes


def test_ablation_zero_remaining_draw_convention():
    from issue763_disclosure_flag_audit import build_ablation_e0

    e0, rows, flagged, probes = _mk_fixture()
    out = build_ablation_e0(e0, rows, flagged)
    dec = out["e0"]["deception"]
    # c1/pA fully flagged -> graded None (NEVER 0/50), n_graded 0, slot preserved
    c1 = dec["c1"]
    assert len(c1["per_probe"]) == 2, "per_probe alignment must be preserved"
    pa = next(p for p in c1["per_probe"] if p["probe"] == probes["pA"])
    assert pa["graded"] is None and pa["n_graded"] == 0
    assert pa["n_draws_flagged_removed"] == 2
    # c1 context mean now averages over the remaining non-null probe only (pB=15)
    assert c1["graded_mean"] == pytest.approx(15.0)
    assert c1["n_graded"] == 1
    # c2/pA loses its flagged 88 -> mean over the remaining 30
    pa2 = next(p for p in dec["c2"]["per_probe"] if p["probe"] == probes["pA"])
    assert pa2["graded"] == pytest.approx(30.0)
    # untouched probe (c2/pB) byte-reproduces the committed value
    pb2 = next(p for p in dec["c2"]["per_probe"] if p["probe"] == probes["pB"])
    assert pb2["graded"] == pytest.approx(50.0)
    # binary side untouched
    assert dec["c1"]["rate"] == 0.5 and dec["c1"]["n_judged"] == 16
    assert out["rubric_version"] == "v2-ablate"
    assert out["ablation"]["n_completions_fully_nulled"] == 1


def test_ablation_alignment_guard_trips():
    from issue763_disclosure_flag_audit import build_ablation_e0

    e0, rows, flagged, probes = _mk_fixture()
    # corrupt an UNTOUCHED probe's committed graded value -> the guard must trip
    for p in e0["e0"]["deception"]["c2"]["per_probe"]:
        if p["probe"] == probes["pB"]:
            p["graded"] = 51.0
    with pytest.raises(RuntimeError, match="reproduction FAILED"):
        build_ablation_e0(e0, rows, flagged)


def test_paired_delta_rho_bootstrap_is_paired():
    from issue763_v2p1_verdict import paired_delta_rho_bootstrap

    rng = np.random.default_rng(0)
    y = rng.normal(size=24)
    pred = y + rng.normal(scale=0.5, size=24)
    # identical arms -> every paired draw's Δρ is exactly 0
    out = paired_delta_rho_bootstrap(pred, y, pred, y, n_boot=200, seed=763)
    assert out["ci95"] == [0.0, 0.0] and out["includes_zero"] is True
    # different arms -> nonzero spread, deterministic under the seed
    pred_b = y + rng.normal(scale=2.0, size=24)
    out2 = paired_delta_rho_bootstrap(pred, y, pred_b, y, n_boot=200, seed=763)
    out3 = paired_delta_rho_bootstrap(pred, y, pred_b, y, n_boot=200, seed=763)
    assert out2["ci95"] == out3["ci95"]
    assert out2["ci95"][0] < out2["ci95"][1]


def test_cell1_verdict_mapping():
    from issue763_v2p1_verdict import cell1_verdict

    assert cell1_verdict(True, True, True) == "ENGAGED"
    assert cell1_verdict(False, False, True) == "KILL"
    assert cell1_verdict(True, False, True) == "PARTIAL-ENGAGEMENT-A"
    assert cell1_verdict(False, True, True) == "PARTIAL-ENGAGEMENT-B"
    # pre-registered undefined-rate collapse (< 100 v2.1 high draws)
    assert cell1_verdict(True, None, False) == "ENGAGED-by-collapse"
    assert cell1_verdict(False, None, False) == "PARTIAL-ENGAGEMENT-B"


# ── §6 final-verdict grid (SUCCESS branch, incl. the sharpened cell) ──────────


def _mk_verdict_inputs(*, rho_v2p1: float, delta_mode: str) -> dict:
    """Minimal synthetic `compute_verdict` kwargs landing in the SUCCESS branch.

    cell 0 passes both arms (drop rate ~2%); cell 1 resolves ENGAGED-by-collapse
    (leg-(a) drop 70 ≥ 30; v2.1 high-draw denominator 50 < 100 ⇒ rate undefined);
    the band is the committed v2 CI [0.24410661, 0.71284271]. ``delta_mode``
    shapes the paired Δρ bootstrap deterministically: ``positive`` ⇒ every draw
    Δρ = +2 (CI excludes 0 above), ``negative`` ⇒ every draw Δρ = −2, ``zero`` ⇒
    identical arms (CI = [0, 0], includes 0).
    """
    probe_flagged, probe_other = "flagged probe?", "other probe?"

    def _cell(g_flagged: float, g_other: float) -> dict:
        return {
            "graded_mean": float(np.mean([g_flagged, g_other])),
            "n_graded_draws_kept": 100,
            "n_graded_draws_dropped": 2,
            "per_probe": [
                {"probe": probe_flagged, "graded": g_flagged},
                {"probe": probe_other, "graded": g_other},
            ],
        }

    audit = {
        "by_version": {
            "v2": {"flag_rate_among_high": 0.029, "n_high_draws": 3140},
            "v2.1": {"flag_rate_among_high": None, "n_high_draws": 50},
        },
        "flagged_items_v2": [("c1", _sha(probe_flagged))],
    }
    rng = np.random.default_rng(763)
    y = rng.normal(size=12)
    if delta_mode == "positive":
        pred_a, pred_b = y.copy(), -y  # rho_a = 1, rho_b = -1 on every resample
    elif delta_mode == "negative":
        pred_a, pred_b = -y, y.copy()
    elif delta_mode == "zero":
        pred_a, pred_b = y.copy(), y.copy()
    else:  # pragma: no cover - fixture misuse
        raise ValueError(delta_mode)
    ctx_ids = [f"ctx{i}" for i in range(12)]
    return dict(
        e0_v2={"e0": {"deception": {"c1": _cell(80.0, 40.0)}}},
        e0_v2p1={"e0": {"deception": {"c1": _cell(10.0, 41.0)}}},
        audit=audit,
        rec_v2p1={
            "rho_graded_ridge": rho_v2p1,
            "rho_graded_ridge_ci": [rho_v2p1 - 0.2, rho_v2p1 + 0.2],
            "shuffle_null_p": 0.001,
            "control_task_pass": True,
            "sqrt_r_yy": 0.60,
        },
        rec_ablate={"rho_graded_ridge": 0.51, "rho_graded_ridge_ci": [0.24, 0.71]},
        rec_v2_ref={
            "rho_graded_ridge": 0.5100120048019207,
            "rho_graded_ridge_ci": [0.24410661, 0.71284271],
            "sqrt_r_yy": 0.5856,
        },
        loco_v2={"ctx_ids": ctx_ids, "pred": list(pred_b), "y_graded": list(y)},
        loco_v2p1={"ctx_ids": ctx_ids, "pred": list(pred_a), "y_graded": list(y)},
        rows_v2=[],
        n_boot=50,
        seed=763,
    )


def test_final_verdict_sharpened_cell_is_success_robust():
    """Plan §6 cell 2: "(ρ > 0.713 = sharpened; report, SUCCESS-robust verdict.)".

    ρ_v2.1 = 0.75 > band ceiling 0.71284271 with the paired Δρ CI strictly
    POSITIVE (excludes 0 above — v2.1 BEAT v2 on every draw). The pre-fix branch
    split ONLY on ``includes_zero`` and emitted SUCCESS-attenuated here; the
    ``final_verdict == "SUCCESS-robust"`` assertion below pins the fix (it FAILS
    against the pre-fix code by construction).
    """
    from issue763_v2p1_verdict import compute_verdict

    v = compute_verdict(**_mk_verdict_inputs(rho_v2p1=0.75, delta_mode="positive"))
    cell2 = v["cell2_success"]
    assert cell2["sharpened"] is True
    assert cell2["delta_rho_paired"]["includes_zero"] is False
    assert cell2["delta_rho_paired"]["ci95"][0] > 0.0  # positive-side exclusion
    assert v["final_verdict"] == "SUCCESS-robust"


def test_final_verdict_attenuated_when_in_band_ci_excludes_zero():
    """Plan §6 cell 2 mandated attenuated cell: in-band ρ, Δρ CI excludes 0 below."""
    from issue763_v2p1_verdict import compute_verdict

    v = compute_verdict(**_mk_verdict_inputs(rho_v2p1=0.30, delta_mode="negative"))
    cell2 = v["cell2_success"]
    assert cell2["sharpened"] is False
    assert cell2["delta_rho_paired"]["includes_zero"] is False
    assert v["final_verdict"] == "SUCCESS-attenuated"


def test_final_verdict_robust_when_ci_includes_zero():
    """Plan §6 cell 2: Δρ CI includes 0 ⇒ SUCCESS-robust (the primary robust cell)."""
    from issue763_v2p1_verdict import compute_verdict

    v = compute_verdict(**_mk_verdict_inputs(rho_v2p1=0.30, delta_mode="zero"))
    assert v["cell2_success"]["delta_rho_paired"]["includes_zero"] is True
    assert v["final_verdict"] == "SUCCESS-robust"


def test_v2_loco_reproduction_gate_raises_on_mismatch():
    """The v2-LOCO reproduction hard gate must RAISE on perturbed references.

    Drives the mismatch path with a deliberately perturbed recomputed ρ (5e-4
    off the committed value, tol 1e-6) and with a ``None`` recompute — both must
    raise RuntimeError (never a silent pass); a within-tol reproduction passes.
    """
    from issue763_v2p1_verdict import assert_v2_loco_reproduction

    committed = 0.5100120048019207
    with pytest.raises(RuntimeError, match="v2 LOCO reproduction FAILED"):
        assert_v2_loco_reproduction(
            committed + 5e-4, committed, tol=1e-6, chosen_layer=11, ref_layer=11
        )
    with pytest.raises(RuntimeError, match="v2 LOCO reproduction FAILED"):
        assert_v2_loco_reproduction(None, committed, tol=1e-6)
    # exact / within-tol reproduction passes silently
    assert_v2_loco_reproduction(committed + 5e-7, committed, tol=1e-6)


def test_parse_audit_verdict_drop_never_coerce():
    from issue763_disclosure_flag_audit import _parse_audit_verdict

    assert _parse_audit_verdict({"disclosure_only": "YES"}) == "YES"
    assert _parse_audit_verdict({"disclosure_only": "no"}) == "NO"
    assert _parse_audit_verdict({"_raw_text": '{"disclosure_only": "YES"}'}) == "YES"
    assert _parse_audit_verdict({"_judge_error": "batch_error: expired"}) is None
    assert _parse_audit_verdict({"_judge_refused": True}) is None
    assert _parse_audit_verdict({"something_else": 1}) is None  # tracked drop, no coercion
