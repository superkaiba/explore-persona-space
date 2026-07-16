"""#825 turn-dynamics fit unit tests (code-review v21 + v22 findings).

- ``_general_linear_cos`` + ``_rank_truncated_cols`` (v21 Critical 2): pins the
  non-degenerate value on a synthetic rank-deficient operator pair AND the
  full-square-U artifact (identically 1.0) the truncation guards against.
- fp16 round-trip rank recovery (v22 Critical 1): the fp16 beta persistence
  noise tail (~1e-4*S.max) defeated rel=1e-6; pins rel=1e-3 + the row-count
  clamp recovering the true rank and the cos staying non-degenerate.
- ``_transfer_nulls`` (Major 3): the batched GEMM identity reproduces a
  brute-force per-draw ss_res on a tiny synthetic case, and every (i, j)
  transfer cell carries null keys.
- ``stratified_subsample_ids`` (Major 6): both source prefixes present with
  proportional allocation, deterministic under the seed, and NOT the
  first-N-of-sorted (single-source) selection the fix replaces.
- ``_r10_key_for_turn`` / ``_r10_node_for_turn`` (crash-fix round 11): the
  G-C parity lookup translates the refit's 1-based exchange ordinal t to the
  round-10 curve's 0-based assistant turns-list key 2t-1; the pinned values
  FAIL under the old ``str(t)`` lookup (even-t cells skipped, odd-t cells
  compared against the wrong shallower node).
- ``_gc_verdict`` (crash-fix round 11, revision 4 — RANK-SPACE degeneracy
  carve-out, superseding revision 3's float-space ``lo < r2_refit < hi`` test
  that code-review v26 refuted on the real pre-fix table): a cell is BINDING
  iff the bootstrap draw distribution holds STRICTLY MORE than alpha/2
  (alpha = the CI's own 0.05) of its finite draws strictly on EACH side of
  the identity-resample anchor (``_cell_r2_boot_draws``); collapsed
  distributions — whose INTERPOLATED ci_hi can land float-epsilon ABOVE the
  point (t=24/25/29: +9.7e-9..1.6e-7) — are non-gating and excluded from
  n_fail; an all-degenerate table cannot PASS (n_gating == 0 guard);
  proper-distribution failures still FAIL the verdict; nodes lacking the
  ``boot_frac_*`` fields (pre-revision-4 archived artifacts) FAIL LOUD
  (recompute draws deterministically — never a float-space fallback). The
  archived pre-fix instruct verdict table is a committed fixture; the replay
  test pins its epsilon geometry + the revision-4 expected outcome
  (n_gating=21, n_fail=0, pass=True).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from issue825_turndyn_fit import (  # noqa: E402
    BOOT_SEED,
    GC_BOOT_ALPHA,
    _boot_tail_fractions,
    _cell_r2_boot_draws,
    _cell_r2_bootstrap,
    _gc_verdict,
    _general_linear_cos,
    _r10_key_for_turn,
    _r10_node_for_turn,
    _rank_truncated_cols,
    _transfer_nulls,
)
from issue825_turndyn_harvest import stratified_subsample_ids  # noqa: E402


def test_general_linear_cos_rank_truncated_pins_synthetic_value():
    d, r = 64, 8
    torch.manual_seed(0)
    b_i = torch.zeros(d, d, dtype=torch.float64)
    b_i[:r] = torch.randn(r, d, dtype=torch.float64)  # col(b_i) = span(e_0..e_{r-1}) a.s.
    U, S, _vh = torch.linalg.svd(b_i, full_matrices=False)
    u_r, rank = _rank_truncated_cols(U, S)
    assert rank == r
    a, b = 3.0, 4.0
    b_j = torch.zeros(d, d, dtype=torch.float64)
    b_j[0, 0] = a  # in-span energy
    b_j[r + 1, 0] = b  # out-of-span energy
    got = _general_linear_cos(u_r, b_j)
    assert abs(got - a / np.hypot(a, b)) < 1e-9, got  # 3/5 = 0.6 exactly
    assert got < 1.0 - 1e-6  # the mechanizable non-degeneracy assert (v21 C2)
    # the artifact the truncation removes: full square U => identically 1.0
    artifact = _general_linear_cos(U, b_j)
    assert abs(artifact - 1.0) < 1e-9, artifact


def test_rank_truncation_survives_fp16_beta_round_trip(tmp_path):
    """Code-review v22 Critical 1: fp16 beta persistence lifts the SVD noise
    tail to ~1e-4*S.max, so the old rel=1e-6 rule read rank near-FULL and the
    cos statistic re-squashed to ~1. Pins, on a DENSE low-rank matrix through
    the production ``np.save(.astype(np.float16))`` -> reload path:
    (a) the defeated rule is reproduced (rank@1e-6 >> true rank),
    (b) the new rel=1e-3 threshold alone recovers the true rank,
    (c) the algebraic row-count clamp binds independently of the threshold,
    (d) the cos statistic stays non-degenerate (not re-squashed to ~1).
    """
    d, r = 256, 8
    torch.manual_seed(0)
    # DENSE rank-r matrix (A @ B): fp16 quantization error is a dense
    # full-rank perturbation, exactly the production failure shape (a
    # zero-padded construction would quantize exactly and hide the tail).
    b = (torch.randn(d, r) @ torch.randn(r, d)).to(torch.float32)
    u_true = torch.linalg.svd(b, full_matrices=False).U[:, :r]  # true col space
    p = tmp_path / "beta_fp16.npy"
    np.save(p, b.numpy().astype(np.float16))  # the production persistence path
    b16 = torch.from_numpy(np.load(p).astype(np.float32))  # the production reload
    U, S, _vh = torch.linalg.svd(b16, full_matrices=False)
    # (a) the defeated v21 rule: fp16 tail ~1e-4*S.max >> 1e-6 -> near-full rank
    rank_old_rule = int((float(S.max()) * 1e-6 < S).sum())
    assert rank_old_rule > 4 * r, rank_old_rule  # reads near-D today (~248)
    # (b) threshold alone (rel=1e-3, ~10x above the measured ~1e-4 tail)
    _, rank_thresh = _rank_truncated_cols(U, S)
    assert rank_thresh == r, rank_thresh
    # (c) row-count clamp binds even when the threshold would keep more
    _, rank_clamped = _rank_truncated_cols(U, S, rel=1e-6, max_rank=r)
    assert rank_clamped == r, rank_clamped
    # (d) non-degenerate cos on the fp16-round-tripped truncated basis:
    # b_j with known in-span (3.0) / out-of-span (4.0) energy vs col(b)
    q_in = u_true[:, 0]
    v = torch.randn(d)
    v -= u_true @ (u_true.T @ v)
    q_out = v / torch.linalg.norm(v)
    b_j = torch.zeros(d, d)
    b_j[:, 0] = 3.0 * q_in + 4.0 * q_out
    u_r, rank = _rank_truncated_cols(U, S, max_rank=r)
    assert rank == r
    got = _general_linear_cos(u_r, b_j)
    assert abs(got - 0.6) < 1e-2, got  # 3/5, within fp16-perturbation tolerance
    assert got < 0.99, got  # NOT re-squashed to ~1 (the v22 vacuity signature)


def test_transfer_nulls_match_bruteforce_and_carry_all_cell_keys():
    rng = np.random.default_rng(0)
    turns = [1, 2]
    n1, n2, p = 12, 10, 5
    by_turn = {1: np.arange(n1), 2: np.arange(n1, n1 + n2)}
    Y = rng.normal(size=(n1 + n2, p))
    fold_labels = {
        1: np.asarray([i % 2 for i in range(n1)], dtype=np.int64),
        2: np.asarray([i % 2 for i in range(n2)], dtype=np.int64),
    }
    used_folds = {1: [0, 1], 2: [0, 1]}
    pred_blocks: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    ss_tot: dict[tuple[int, int], float] = {}
    for i in turns:
        for j in turns:
            tot = 0.0
            for f in (0, 1):
                te = by_turn[j][fold_labels[j] == f]
                pred_blocks.setdefault((i, j), {})[f] = rng.normal(size=(te.size, p)).astype(
                    np.float16
                )
                true = Y[te]
                tot += float(np.sum((true - true.mean(0)) ** 2))
            ss_tot[(i, j)] = tot
    n_draws, seed = 7, 1092
    out = _transfer_nulls(
        turns, by_turn, fold_labels, used_folds, pred_blocks, Y, ss_tot, n_draws, 3, seed
    )
    # every (i, j) transfer cell carries null keys (the v21 M3 mechanizable check)
    assert set(out) == {f"{i}->{j}" for i in turns for j in turns}
    for node in out.values():
        assert node["null_n_draws"] == n_draws
        assert np.isfinite(node["null_mean"]) and np.isfinite(node["null_hi"])
    # brute-force one off-diagonal cell (1 -> 2): frozen preds vs permuted truth
    j, sig = 2, (0, 1)
    rows_order = np.concatenate([by_turn[j][fold_labels[j] == f] for f in sig])
    pred = np.concatenate([pred_blocks[(1, j)][f] for f in sig], axis=0).astype(np.float64)
    yj = Y[rows_order].astype(np.float32).astype(np.float64)
    perms = np.argsort(
        np.random.default_rng(seed + 104_729 * j).random((n_draws, rows_order.size)), axis=1
    )
    ref = np.asarray(
        [1.0 - float(np.sum((yj[perms[d]] - pred) ** 2)) / ss_tot[(1, j)] for d in range(n_draws)]
    )
    assert abs(out["1->2"]["null_mean"] - ref.mean()) < 1e-4
    assert abs(out["1->2"]["null_max"] - ref.max()) < 1e-4


def test_gc_r10_key_alignment_translates_exchange_ordinal_to_turns_list_index():
    """Crash-fix round 11 (G-C key-space mismatch): the refit keys cells by
    ``turn`` = 1-based exchange ordinal t, while the round-10 reference curve
    is keyed by the 0-based assistant turns-list index = 2t-1 (odd keys only).
    Pins the translated lookup ``str(2*t - 1)``: under the OLD ``str(t)``
    lookup, t=2 returned NO node (cell silently skipped) and t=3 returned
    key "3" — the WRONG (shallower) node belonging to exchange t=2 — the
    systematic false FAIL that blocked the instruct headline.
    """
    # Round-10 curve SHAPE (results.json layer-19): odd keys only, one
    # distinct r2 per key so a wrong-node selection is detectable.
    curve = {
        "1": {"ctx_logged": {"r2": 0.10}, "n": 10},
        "3": {"ctx_logged": {"r2": 0.20}, "n": 8},
        "5": {"ctx_logged": {"r2": 0.30}, "n": 6},
    }
    # translation: exchange ordinal t -> turns-list index 2t-1
    assert [_r10_key_for_turn(t) for t in (1, 2, 3)] == [1, 3, 5]

    # t=1: the two conventions coincide (key 1) — the cell the old lookup
    # got right by accident.
    assert _r10_node_for_turn(curve, 1) == (1, 0.10)
    # t=2: OLD lookup curve.get("2") -> {} -> r2 None -> cell skipped.
    key, r2 = _r10_node_for_turn(curve, 2)
    assert key == 3
    assert r2 == 0.20
    # t=3: OLD lookup selected key "3" (r2 0.20, exchange t=2's node).
    key, r2 = _r10_node_for_turn(curve, 3)
    assert key == 5
    assert r2 == 0.30
    assert r2 != curve["3"]["ctx_logged"]["r2"]
    # absent (key 7) / malformed nodes drop out of the comparison (r2 None)
    assert _r10_node_for_turn(curve, 4) == (7, None)
    assert _r10_node_for_turn({"1": 3.0}, 1) == (1, None)


def test_stratified_subsample_ids_spans_sources_and_is_deterministic():
    ids = [f"lmsys_{i:06d}" for i in range(100)] + [f"wildchat_{i:06d}" for i in range(50)]
    sel = stratified_subsample_ids(ids, 30, seed=42)
    assert len(sel) == 30
    counts: dict[str, int] = {}
    for cid in sel:
        counts[cid.rsplit("_", 1)[0]] = counts.get(cid.rsplit("_", 1)[0], 0) + 1
    assert counts == {"lmsys": 20, "wildchat": 10}  # proportional, both sources present
    assert sel == stratified_subsample_ids(ids, 30, seed=42)  # deterministic
    assert sel != sorted(set(ids))[:30]  # NOT first-N-of-sorted (the Major-6 bug)
    assert stratified_subsample_ids(ids, 10_000) == sorted(set(ids))
    assert stratified_subsample_ids(ids, 0) == []


GC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "issue825_gc_instruct_prefix_att20260716.json"
_PCTL_LO = 100.0 * GC_BOOT_ALPHA / 2.0  # == 2.5 exactly
_PCTL_HI = 100.0 * (1.0 - GC_BOOT_ALPHA / 2.0)  # == 97.5 exactly


def _gc_node_from_draws(n: int, refit: float, draws: np.ndarray, anchor: float, r10: float) -> dict:
    """Synthetic G-C per-turn node built through the PRODUCTION computations:
    CI = nanpercentile at the GC_BOOT_ALPHA-derived percentiles, tail
    fractions via ``_boot_tail_fractions`` against the identity anchor,
    per-cell ``pass`` exactly as run_gc computes it (lo <= r10 <= hi)."""
    lo = float(np.nanpercentile(draws, _PCTL_LO))
    hi = float(np.nanpercentile(draws, _PCTL_HI))
    below, above, n_finite = _boot_tail_fractions(draws, anchor)
    return {
        "n": n,
        "r2_refit": refit,
        "r2_refit_ci": [lo, hi],
        "r2_round10": r10,
        "r10_key": 1,
        "pass": bool(lo <= r10 <= hi),
        "boot_frac_below": below,
        "boot_frac_above": above,
        "boot_n_finite": n_finite,
        "boot_identity_r2": anchor,
    }


def _healthy_draws(refit: float, half_width: float = 0.05) -> np.ndarray:
    """Spread bootstrap shape: ~50% of draws strictly on each side of the point."""
    return np.concatenate(
        [
            np.linspace(refit - half_width, refit - 1e-6, 500),
            np.linspace(refit + 1e-6, refit + half_width, 500),
        ]
    )


def _collapsed_draws(anchor: float, lo: float, n_cluster: int = 100) -> np.ndarray:
    """Collapsed bootstrap shape (the v275 n<=7 diagnosis): the identity tie
    cluster AT the anchor + every other draw strictly below — zero strict
    upper-tail mass, yet the interpolated 97.5th percentile lands ON the
    cluster value (which sits float-epsilon ABOVE r2_refit at t=24/25/26/29)."""
    return np.concatenate(
        [np.full(n_cluster, anchor), np.linspace(lo, anchor - 1e-4, 1000 - n_cluster)]
    )


def test_gc_verdict_collapsed_cells_excluded_from_n_fail_and_verdict_can_pass():
    """Crash-fix r11 revision 4 (a) — updated from revision 3's exact
    ci_hi == refit fixture, which the real table refuted (v26): the t=25/29
    production shape — the identity cluster float-jittered ~1.5e-7 ABOVE the
    float64 refit, so ci_hi > refit strictly and the revision-3 float-space
    rule mis-gated it — is non-gating under the rank-space test (zero strict
    upper-tail mass around the SAME-expression anchor), excluded from n_fail,
    and the verdict PASSes on the gating cells alone.
    """
    refit = -0.647134  # archived t=25 values
    anchor = refit + 1.565e-07  # identity-resample value, cross-code-path jitter above
    node25 = _gc_node_from_draws(5, refit, _collapsed_draws(anchor, -3.0), anchor, -0.645423)
    assert node25["r2_refit_ci"][1] == pytest.approx(anchor)  # ci_hi strictly ABOVE refit
    assert node25["r2_refit_ci"][1] > refit  # the exact revision-3 mis-gating geometry
    assert node25["boot_frac_above"] == 0.0
    per_turn = {
        "1": _gc_node_from_draws(497, 0.250, _healthy_draws(0.250), 0.250, 0.250),
        "25": node25,
    }
    verdict = _gc_verdict("instruct", 497, per_turn)
    assert per_turn["1"]["gating"] is True
    assert per_turn["25"]["gating"] is False
    assert per_turn["25"]["pass"] is False  # informational pass preserved as computed
    assert verdict["n_turns"] == 2
    assert verdict["n_gating"] == 1
    assert verdict["n_nongating"] == 1
    assert verdict["n_fail"] == 0  # collapsed-cell failure excluded
    assert verdict["pass"] is True
    assert "rank-space" in verdict["gate_note"]


def test_gc_verdict_interpolation_boundary_hi_epsilon_above_is_non_gating():
    """Crash-fix r11 revision 4 (a, boundary pin — the v26 Minor): a cell
    whose 97.5th percentile INTERPOLATES to refit + 1e-8 (one tie at the
    point at sorted index 974, exactly alpha/2 = 25/1000 draws strictly
    above) is non-gating ONLY under the STRICT > alpha/2 comparison — a >=
    test reproduces the float-space failure at this exact boundary.
    """
    refit = -0.585944417738467  # archived t=24 point estimate
    draws = np.concatenate(
        [
            np.linspace(refit - 0.3, refit - 1e-3, 974),
            np.full(1, refit),  # tie at the point (sorted index 974)
            np.full(25, refit + 4e-7),  # exactly alpha/2 of 1000 strictly above
        ]
    )
    node = _gc_node_from_draws(5, refit, draws, refit, -0.584052)
    assert node["r2_refit_ci"][1] == pytest.approx(refit + 1e-8, abs=2e-9)
    assert node["r2_refit_ci"][1] > refit  # hi = refit + ~1e-8: the archived t=24 shape
    assert node["boot_frac_above"] == pytest.approx(GC_BOOT_ALPHA / 2.0)  # the boundary
    per_turn = {
        "1": _gc_node_from_draws(497, 0.250, _healthy_draws(0.250), 0.250, 0.250),
        "24": node,
    }
    verdict = _gc_verdict("instruct", 497, per_turn)
    assert per_turn["24"]["gating"] is False  # strict >: 0.025 is NOT more than alpha/2
    assert verdict["n_fail"] == 0
    assert verdict["pass"] is True


def test_gc_verdict_healthy_cell_failure_still_fails():
    """Crash-fix r11 revision 4 (b) — same intent as revision 3's (b): a cell
    with a healthy draw distribution (both tails ~50% >> alpha/2) whose r10
    falls outside the CI is still counted — the carve-out never weakens the
    gate on cells whose distribution can certify.
    """
    per_turn = {
        "1": _gc_node_from_draws(497, 0.250, _healthy_draws(0.250), 0.250, 0.250),
        "2": _gc_node_from_draws(388, 0.250, _healthy_draws(0.250), 0.250, 0.400),
        "25": _gc_node_from_draws(
            5,
            -0.647134,
            _collapsed_draws(-0.647134 + 1.565e-07, -3.0),
            -0.647134 + 1.565e-07,
            -0.645423,
        ),
    }
    verdict = _gc_verdict("instruct", 497, per_turn)
    assert per_turn["2"]["gating"] is True
    assert per_turn["2"]["pass"] is False  # r10 0.400 outside ~[0.2, 0.3]
    assert verdict["n_gating"] == 2
    assert verdict["n_fail"] == 1
    assert verdict["pass"] is False


def test_gc_verdict_all_degenerate_cannot_pass():
    """Crash-fix r11 revision 4 (c) — same intent as revision 3's (c), draw-
    based fixtures: zero gating cells -> pass False even with zero failures
    (an all-degenerate table certifies nothing); the empty table keeps the
    pre-existing len(out) > 0 guard semantics.
    """
    collapsed_flat = np.full(1000, -0.2)  # fully collapsed: every draw == the anchor
    per_turn = {
        "29": _gc_node_from_draws(
            3,
            -1.457199,
            _collapsed_draws(-1.457199 + 1.408e-07, -3.0),
            -1.457199 + 1.408e-07,
            -1.422057,
        ),
        "30": _gc_node_from_draws(3, -0.2, collapsed_flat, -0.2, -0.2),
    }
    verdict = _gc_verdict("pretrained", 30, per_turn)
    assert verdict["n_gating"] == 0
    assert verdict["n_nongating"] == 2
    assert verdict["n_fail"] == 0
    assert verdict["pass"] is False
    empty = _gc_verdict("pretrained", 0, {})
    assert empty["pass"] is False and empty["n_turns"] == 0


def test_gc_verdict_fails_loud_on_archived_nodes_without_fractions():
    """Crash-fix r11 revision 4: a pre-revision-4 archived node (no
    boot_frac_* fields) must FAIL LOUD with the recompute-deterministically
    instruction — NEVER silently fall back to a float-space coverage test.
    """
    archived = {
        "n": 5,
        "r2_refit": -0.586,
        "r2_refit_ci": [-2.95, -0.586],
        "r2_round10": -0.584,
        "r10_key": 47,
        "pass": False,
    }
    with pytest.raises(RuntimeError, match=r"boot_frac_below.*recompute"):
        _gc_verdict("instruct", 497, {"24": archived})


def test_cell_r2_boot_draws_identity_anchor_bitwise_and_wrapper_equivalence():
    """Crash-fix r11 revision 4, helper level: (1) draws whose count vector is
    all-ones are BITWISE equal to the identity anchor (same expression, same
    matmul batch) — the tie-exactness the rank-space gate relies on, immune
    to the ~1e-9..1e-7 cross-code-path jitter vs _fit_cv's float64 r2; (2)
    the `_cell_r2_bootstrap` wrapper returns exactly the nanpercentile CI of
    the same seeded draws (statistics unchanged); (3) `_boot_tail_fractions`
    counts ties on NEITHER side and guards the all-NaN case.
    """
    rng = np.random.default_rng(0)
    y = rng.standard_normal((6, 8)).astype(np.float32)
    pred = (y + 0.5 * rng.standard_normal((6, 8))).astype(np.float32)
    convs = ["a", "a", "a", "b", "b", "b"]
    draws, anchor = _cell_r2_boot_draws(y, pred, convs, 400, BOOT_SEED)
    assert draws.shape == (400,)
    assert np.isfinite(anchor)
    # at n_uniq=2 the (1,1) count vector has probability 1/2 -> ~200 identity
    # draws, ALL bitwise equal to the appended-row anchor
    n_tie = int((draws == anchor).sum())
    assert n_tie > 100
    lo, hi = _cell_r2_bootstrap(y, pred, convs, 400, BOOT_SEED)
    assert lo == float(np.nanpercentile(draws, 2.5))
    assert hi == float(np.nanpercentile(draws, 97.5))
    below, above, n_finite = _boot_tail_fractions(draws, anchor)
    assert n_finite == int(np.isfinite(draws).sum())
    assert below + above <= 1.0 - n_tie / n_finite  # ties count on neither side
    assert _boot_tail_fractions(np.array([np.nan, np.nan]), 0.0) == (0.0, 0.0, 0)


def test_gc_rank_space_replay_on_archived_prefix_table():
    """Crash-fix r11 revision 4 replay (the v26 blocker
    gc-carveout-noneng-t24-t25-t29, mechanizable check): the REAL pre-fix
    instruct verdict table (committed fixture, sha-identical copy of
    /tmp/gc_instruct_prefix_825.json whose provenance the v26 reviewer
    matched to progress v275) carries the exact failure geometry —
    interpolated ci_hi float-epsilon ABOVE r2_refit at t=24/25/29 — that
    defeats the pre-r11 rule (n_fail=7) AND the revision-3 strict-coverage
    rule (n_gating=25, n_fail=3, both replayed here from the archived
    floats). Under revision 4, with tail fractions from collapsed-shape draw
    reconstructions for the nine n<=7 cells (the v275 diagnosis; the REAL
    seeded draws live pod-side and are recomputed there deterministically at
    relaunch, seed BOOT_SEED+t) and spread-shape reconstructions for the 21
    healthy cells, the verdict is (n_gating, n_fail, pass) == (21, 0, True).
    """
    with open(GC_FIXTURE, encoding="utf-8") as f:
        table = json.load(f)
    per_turn = {t: dict(node) for t, node in table["per_turn"].items()}
    # archived pre-r11 verdict: 7 fails over 30 turns, FAIL
    assert (table["n_turns"], table["n_fail"], table["pass"]) == (30, 7, False)
    # pin the epsilon geometry the v26 blocker measured at full float precision
    eps = {t: n["r2_refit_ci"][1] - n["r2_refit"] for t, n in per_turn.items()}
    assert eps["24"] == pytest.approx(9.658e-09, rel=1e-2)
    assert eps["25"] == pytest.approx(1.565e-07, rel=1e-2)
    assert eps["29"] == pytest.approx(1.408e-07, rel=1e-2)
    # the revision-3 float-space rule on this table: 25 gating cells, 3 of
    # them failing -> the deterministic relaunch re-FAIL the v26 review pinned
    r3_gating = {
        t for t, n in per_turn.items() if n["r2_refit_ci"][0] < n["r2_refit"] < n["r2_refit_ci"][1]
    }
    assert len(r3_gating) == 25
    assert sum(1 for t in r3_gating if not per_turn[t]["pass"]) == 3
    # revision-4 tail fractions: reconstructed draw shapes keyed on the
    # fixture's real per-cell class (n<=7 = the v275 collapse; n>=11 healthy)
    for node in per_turn.values():
        lo, _hi = node["r2_refit_ci"]
        refit = node["r2_refit"]
        if node["n"] <= 7:
            anchor = max(node["r2_refit_ci"][1], refit)  # identity cluster at/above the point
            draws = _collapsed_draws(anchor, min(lo, anchor - 1.0))
        else:
            anchor = refit
            draws = _healthy_draws(refit)
        below, above, n_finite = _boot_tail_fractions(draws, anchor)
        node["boot_frac_below"], node["boot_frac_above"] = below, above
        node["boot_n_finite"], node["boot_identity_r2"] = n_finite, anchor
    verdict = _gc_verdict("instruct", table["n_convs"], per_turn)
    assert (verdict["n_gating"], verdict["n_fail"], verdict["pass"]) == (21, 0, True)
    assert verdict["n_nongating"] == 9
    for t in map(str, range(1, 22)):
        assert per_turn[t]["gating"] is True and per_turn[t]["pass"] is True
    for t in map(str, range(22, 31)):
        assert per_turn[t]["gating"] is False
    # informational per-cell pass preserved verbatim on the mis-gated cells
    for t in ("24", "25", "29"):
        assert per_turn[t]["pass"] is False
