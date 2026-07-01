"""Regression tests for the issue #779 Stage-1 code-correctness BLOCKER fixes.

Each test pins a permanent invariant a round-2 BLOCKER fix installed, and each
FAILS against the pre-fix code (documented per test):

  - shuffled-context null uses its Xc argument (BLOCKER 2): pre-fix _pred_fn
    ignored Xc so every null replicate == obs and beats_null was always False.
  - Gate-0 per-trait headroom descope (BLOCKER 3): pre-fix run_mlp was a global
    flag, never per-trait.
  - AUROC / top-k precision emitted per method (BLOCKER 4): pre-fix zero call
    sites; the keys were absent from the output.
  - N=5 graded judge draws, mean over valid, drop-never-coerce (BLOCKER 1):
    pre-fix the DV was a single N=1 judge score.
  - h1a nullity uses the DOT arm, not cosine (analyzer minor note 1).

Pure-CPU, no model / no network. The judge N=5 wrapper is tested with a stubbed
judge (monkeypatched judge_completions_batch writing a save_raw file) so no API
call is made.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_common as C  # noqa: E402
import issue779_stage1 as S  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402
from explore_persona_space.experiments.issue_779 import r3_granularity as R3  # noqa: E402

# ── BLOCKER 2: shuffled-context null must USE its Xc arg (non-identity) ────────


def test_shuffled_context_null_is_not_identity():
    """The stage1 _pred_fn refits on the permuted pairing, so the null R2 differs
    from the observed R2 across permutations. Pre-fix _pred_fn ignored Xc, making
    every null replicate byte-identical to obs (null_mean == obs, beats_null
    always False)."""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(40, 8))
    Ytr = Xtr @ rng.normal(size=(8, 8)) + 0.01 * rng.normal(size=(40, 8))

    # The EXACT fixed stage1 _pred_fn shape: refit on the supplied contexts.
    def pred_fn(Xc, _Ytr=Ytr):
        return F.ridge_fit_predict(Xc, _Ytr, Xc)

    res = R3.shuffled_context_null_r2(pred_fn, Xtr, Ytr, n_shuffle=15, seed=0)
    # A real (context, answer) pairing reconstructs; a shuffled one does not.
    assert res["n_shuffle"] >= 1
    assert np.isfinite(res["observed_r2"])
    assert np.isfinite(res["null_mean_r2"])
    # NON-IDENTITY: the null mean is strictly below the observed R2 (the bug made
    # them equal). Guard with a clear margin so it cannot pass by float noise.
    assert res["null_mean_r2"] < res["observed_r2"] - 0.1, (
        res["null_mean_r2"],
        res["observed_r2"],
    )
    assert res["beats_null"] is True


def test_shuffled_context_null_bug_would_produce_identity():
    """Documents the pre-fix behavior: an Xc-ignoring pred_fn yields
    null == obs (beats_null False). Confirms the test above actually
    discriminates the fix."""
    rng = np.random.default_rng(1)
    Xtr = rng.normal(size=(40, 8))
    Ytr = Xtr @ rng.normal(size=(8, 8)) + 0.01 * rng.normal(size=(40, 8))

    def buggy_pred_fn(Xc, _Xtr=Xtr, _Ytr=Ytr):
        return F.ridge_fit_predict(_Xtr, _Ytr, _Xtr)  # ignores Xc — the bug

    res = R3.shuffled_context_null_r2(buggy_pred_fn, Xtr, Ytr, n_shuffle=15, seed=0)
    # The buggy null is byte-identical to obs; beats_null is False (obs is NOT >
    # the p95 of a constant null distribution).
    assert abs(res["null_mean_r2"] - res["observed_r2"]) < 1e-9
    assert res["beats_null"] is False


# ── BLOCKER 3: per-trait Gate-0 oracle-headroom descope ───────────────────────


def _step0(headroom_by_trait, best_layer=12):
    """Minimal step0_oracle.json-shaped dict with a given per-trait headroom."""
    out = {}
    for trait, hr in headroom_by_trait.items():
        out[trait] = {
            "best_layer": best_layer,
            "per_layer_mode": {
                f"L{best_layer}_system": {
                    "pv_raw_r": 0.30,
                    "oracle_r": 0.30 + hr,
                    "headroom": hr,
                    "n_conditions": 8,
                },
                f"L{best_layer}_many_shot": {
                    "pv_raw_r": 0.50,
                    "oracle_r": 0.50 + hr,
                    "headroom": hr,
                    "n_conditions": 8,
                },
            },
        }
    return out


def test_gate0_descopes_low_headroom_trait():
    """A trait with headroom < +0.05 is descoped (run_mlp=False); a trait with
    headroom >= +0.05 runs fully. Pre-fix there was no per-trait descope at
    all."""
    step0 = _step0({"hallucination": 0.01, "evil": 0.20})
    descope_h, hr_h, reason_h = S.gate0_headroom_descope(step0, "hallucination", 12)
    descope_e, hr_e, _reason_e = S.gate0_headroom_descope(step0, "evil", 12)
    assert descope_h is True and hr_h == pytest.approx(0.01)
    assert "no oracle headroom" in reason_h
    assert descope_e is False and hr_e == pytest.approx(0.20)


def test_gate0_boundary_is_strict_below_threshold():
    """Exactly +0.05 is NOT descoped (>= threshold runs fully); +0.049 IS."""
    step0 = _step0({"t_at": 0.05, "t_below": 0.049})
    d_at, _, _ = S.gate0_headroom_descope(step0, "t_at", 12)
    d_below, _, _ = S.gate0_headroom_descope(step0, "t_below", 12)
    assert d_at is False
    assert d_below is True


def test_gate0_fails_open_when_step0_missing():
    """A missing step0 entry / missing headroom must NOT silently descope R1
    (fail open) — a missing probe never drops the fitted-map arm."""
    d1, hr1, r1 = S.gate0_headroom_descope({}, "evil", 12)
    assert d1 is False and hr1 is None and "fail-open" in r1
    # trait present but no per_layer_mode entry at the read-out layer.
    step0 = {"evil": {"best_layer": 12, "per_layer_mode": {}}}
    d2, hr2, r2 = S.gate0_headroom_descope(step0, "evil", 12)
    assert d2 is False and hr2 is None and "fail-open" in r2


# ── BLOCKER 4: AUROC / top-k precision emitted per method x mode ──────────────


def _fake_eval_mat(n=40, seed=0):
    """A build_eval_matrix-shaped dict spanning both elicitation modes with a
    monitor that correlates with the DV, for detection_metrics."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0, 100, size=n)
    x = y + rng.normal(scale=10, size=n)  # correlated monitor
    mode = np.array(["system"] * (n // 2) + ["many_shot"] * (n - n // 2), dtype=object)
    return {"y": y, "mode": mode, "x": x}


def test_detection_metrics_emits_auroc_and_topk_per_mode():
    """detection_metrics returns finite AUROC + top-k per elicitation mode and
    overall. Pre-fix metrics.auroc / top_k_precision had ZERO call sites."""
    mat = _fake_eval_mat()
    det = S.detection_metrics(mat["x"], mat)
    for key in ("system", "many_shot", "overall"):
        assert key in det, key
        assert "auroc" in det[key] and "top_k_precision" in det[key]
    # a monitor correlated with the DV should beat chance AUROC (0.5).
    assert det["overall"]["auroc"] > 0.6
    assert 0.0 <= det["overall"]["top_k_precision"] <= 1.0


def test_detection_metrics_threshold_labels_and_empty_class():
    """Labels come from y >= 50; a mode with a single class yields NaN AUROC
    (never a crash)."""
    n = 20
    mat = {
        "y": np.array([10.0] * n),  # all below threshold -> single class
        "mode": np.array(["system"] * n, dtype=object),
        "x": np.arange(n, dtype=float),
    }
    det = S.detection_metrics(mat["x"], mat)
    assert np.isnan(det["system"]["auroc"])  # single class -> NaN, not error


# ── BLOCKER 1: N=5 graded judge draws, mean over valid, drop-never-coerce ─────


class _StubJudge:
    """Monkeypatch target: writes a save_raw file whose all_scores maps each
    expanded custom_id to a stubbed judge dict per a scoring function.
    """

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.last_expanded = None

    def __call__(self, completions, *, save_raw, **kwargs):
        self.last_expanded = completions
        all_scores = {}
        for persona, qmap in completions.items():
            for qi, (_q, comps) in enumerate(qmap.items()):
                for ci, comp in enumerate(comps):
                    cid = f"{persona}__{qi:05d}__{ci:02d}"
                    all_scores[cid] = self.score_fn(persona, qi, ci, comp)
        Path(save_raw).parent.mkdir(parents=True, exist_ok=True)
        with open(save_raw, "w") as f:
            json.dump({"all_scores": all_scores, "per_persona": {}}, f)
        return {}


def test_judge_rollouts_n5_expands_five_draws_and_means(tmp_path, monkeypatch):
    """Each rollout is expanded to JUDGE_N_DRAWS draw-items and the valid draws
    are mean-aggregated; the return is keyed by the ORIGINAL rollout custom_id.
    Pre-fix the DV was a single N=1 score with no draw expansion."""
    # draw d of rollout ri lands at ci = ri*N + d. Score = 10*ri + d so the
    # mean over d in [0,N) is 10*ri + (N-1)/2.
    n = C.JUDGE_N_DRAWS

    def score_fn(_persona, _qi, ci, _comp):
        ri, d = divmod(ci, n)
        return {"score": 10 * ri + d, "reasoning": "x"}

    stub = _StubJudge(score_fn)
    monkeypatch.setattr("explore_persona_space.eval.batch_judge.judge_completions_batch", stub)
    rollouts = {"cellA": {"q000": ["ans0", "ans1", "ans2"]}}  # 3 rollouts
    agg = C.judge_rollouts_n5("evil", rollouts, tmp_path / "raw.json", tmp_path / "cache")
    # 5 identical draws per completion submitted.
    assert len(stub.last_expanded["cellA"]["q000"]) == 3 * n
    # keyed by original custom_id; means match.
    for ri in range(3):
        cid = f"cellA__00000__{ri:02d}"
        mean, n_valid, n_draws = agg[cid]
        assert n_draws == n
        assert n_valid == n
        assert mean == pytest.approx(10 * ri + (n - 1) / 2.0)


def test_judge_rollouts_n5_drops_per_draw_and_drops_rollout_when_all_invalid(tmp_path, monkeypatch):
    """A REFUSAL / out-of-range draw is DROPPED (not coerced); the mean is over
    valid draws only; a rollout with 0 valid draws returns (None, 0, N)."""
    n = C.JUDGE_N_DRAWS

    def score_fn(_persona, _qi, ci, _comp):
        ri, d = divmod(ci, n)
        if ri == 0:
            # rollout 0: first draw refuses, second is out-of-range (200), rest 60.
            if d == 0:
                return {"reasoning": "no", "score": "REFUSAL"}  # unparseable -> drop
            if d == 1:
                return {"score": 200}  # out of [0,100] -> drop
            return {"score": 60}
        # rollout 1: every draw refuses -> whole rollout dropped.
        return {"score": "REFUSAL"}

    stub = _StubJudge(score_fn)
    monkeypatch.setattr("explore_persona_space.eval.batch_judge.judge_completions_batch", stub)
    rollouts = {"cellA": {"q000": ["a", "b"]}}
    agg = C.judge_rollouts_n5("evil", rollouts, tmp_path / "raw.json", tmp_path / "cache")
    mean0, nv0, _ = agg["cellA__00000__00"]
    assert nv0 == n - 2  # two draws dropped
    assert mean0 == pytest.approx(60.0)  # mean over the surviving 60s only
    mean1, nv1, nd1 = agg["cellA__00000__01"]
    assert mean1 is None and nv1 == 0 and nd1 == n  # all draws dropped -> DROP rollout


# ── analyzer minor note 1: h1a nullity is over the DOT arm ────────────────────


def test_stage1_h1a_uses_dot_arm():
    """The stage1 source references r1_ridge_dot (not r1_ridge_cos) for the h1a
    nullity comparison (§8 identity r~_B = M^T r_B is over the DOT readout)."""
    src = (REPO / "scripts" / "issue779_stage1.py").read_text()
    # the h1a block reads the DOT method's point, and records r1_ridge_dot_r.
    assert 'method_res["r1_ridge_dot"][mode]["point"]' in src
    assert '"r1_ridge_dot_r"' in src
