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
  - N=5 multi-persona custom_id resolution (BLOCKER
    judge-n5-custom-id-index-mismatch-multipersona): pre-fix the wrapper's
    per-persona question index diverged from batch_judge's global-across-persona
    idx, so every persona after the first was looked up at the wrong idx and all
    its draws were silently dropped as (None, 0, n).
  - N=5 cache-rerun never collapses draws (BLOCKER
    judge-n5-cache-rerun-collapses-draws): pre-fix the shared cache keyed only on
    (question, completion) collapsed the N identical draw-copies to one entry, so
    a warm-cache rerun returned the last cached draw N times; post-fix the wrapper
    disables the cache so all N draws are re-submitted as independent samples.
  - h1a nullity uses the DOT arm, not cosine (analyzer minor note 1).

Pure-CPU, no model / no network. The judge N=5 wrapper is tested with a stubbed
judge (monkeypatched judge_completions_batch writing a save_raw file) so no API
call is made. The stub drives the REAL batch_judge global-idx enumeration +
JudgeCache so it cannot re-bake the per-persona index bug.
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

    The custom_ids are produced by the REAL
    ``batch_judge._enumerate_and_check_cache`` (global-across-persona question
    index ``idx`` that NEVER resets per persona), NOT a re-implemented
    per-persona ``qi`` — so this stub CANNOT re-bake the
    ``judge-n5-custom-id-index-mismatch-multipersona`` bug (which is exactly
    what a per-persona re-derivation did). ``score_fn(persona, idx, ci, comp)``
    receives the SAME global question index the wrapper reconstructs.

    Honors ``cache_dir`` (the wrapper passes ``cache_dir=None`` post-fix): when a
    real ``JudgeCache`` is supplied it reads/writes it exactly as the real
    ``judge_completions_batch`` does — cache hits are surfaced under their
    custom_id and the cache is updated with each freshly scored item — so a
    cache-rerun test observes the same collapse-vs-independent behavior the real
    path would.
    """

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.last_expanded = None
        self.n_calls = 0
        self.last_n_submitted = None

    def __call__(self, completions, *, save_raw, cache_dir=None, **kwargs):
        from explore_persona_space.eval.batch_judge import (
            JudgeCache,
            _default_format_user_msg,
            _enumerate_and_check_cache,
            rubric_fingerprint,
        )

        self.last_expanded = completions
        self.n_calls += 1
        fmt = kwargs.get("format_user_msg") or _default_format_user_msg
        cache = JudgeCache(cache_dir) if cache_dir else None
        # Rubric identity derived exactly as the real judge_completions_batch
        # does (#1018 rule-22 keying).
        rk = rubric_fingerprint(
            kwargs.get("judge_model", ""), kwargs.get("judge_system_prompt", ""), fmt
        )
        # Real global-idx enumeration + cache check (exactly as the shared path).
        _total, cached_scores, uncached_items = _enumerate_and_check_cache(
            completions, cache, fmt, rubric_key=rk
        )
        self.last_n_submitted = len(uncached_items)
        batch_scores = {}
        for cid, _question, comp, _user_msg in uncached_items:
            # Recover the global question idx + comp idx from the custom_id so
            # score_fn sees the same (persona, idx, ci) the wrapper reconstructs.
            persona, idx_s, ci_s = cid.rsplit("__", 2)
            batch_scores[cid] = self.score_fn(persona, int(idx_s), int(ci_s), comp)
        if cache:
            for cid, question, comp, _user_msg in uncached_items:
                cache.put(question, comp, batch_scores[cid], rubric_key=rk)
        all_scores = {**cached_scores, **batch_scores}
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


# ── BLOCKER judge-n5-custom-id-index-mismatch-multipersona: global-idx keys ───


def test_judge_rollouts_n5_multipersona_no_dropped_draws(tmp_path, monkeypatch):
    """A MULTI-persona rollouts dict (>1 persona, DIFFERENT questions per persona)
    resolves every persona's draws — every rollout gets finite means and 0 spurious
    drops. The stub uses the REAL batch_judge global-across-persona enumeration.

    Pre-fix (concern judge-n5-custom-id-index-mismatch-multipersona) the wrapper
    re-derived custom_ids with a PER-persona question index that reset to 0 per
    persona, while batch_judge uses a GLOBAL idx that never resets — so every
    persona after the first was looked up at the wrong idx and ALL its draws were
    dropped as (None, 0, n). With that bug, personas p1..pk here would come back
    (None, 0, n); post-fix every rollout is (mean, n, n). This is the Claude
    Mechanizable criterion (n_valid == n_draws for every rollout, 0 unexpected None)."""
    n = C.JUDGE_N_DRAWS
    # Score depends ONLY on the raw completion text so a correct global-idx lookup
    # is the ONLY way a persona's draws resolve; a mis-indexed lookup returns None.
    comp_score = {
        "p0q0a": 11.0,
        "p0q1a": 22.0,
        "p1q0a": 33.0,
        "p1q1a": 44.0,
        "p2q0a": 55.0,
    }

    def score_fn(_persona, _idx, _ci, comp):
        return {"score": comp_score[comp], "reasoning": "x"}

    stub = _StubJudge(score_fn)
    monkeypatch.setattr("explore_persona_space.eval.batch_judge.judge_completions_batch", stub)
    # 3 personas, DIFFERENT questions, 1 rollout each -> global idx 0,1,2,3,4.
    rollouts = {
        "evil_pos_p0": {"qA": ["p0q0a"], "qB": ["p0q1a"]},
        "evil_pos_p1": {"qC": ["p1q0a"], "qD": ["p1q1a"]},
        "evil_pos_p2": {"qE": ["p2q0a"]},
    }
    agg = C.judge_rollouts_n5("evil", rollouts, tmp_path / "raw.json")
    # Every persona's single rollout resolves: 5 rollouts, all finite, 0 dropped.
    assert len(agg) == 5
    for cid, (mean, n_valid, n_draws) in agg.items():
        assert n_draws == n
        assert n_valid == n, f"{cid} dropped {n - n_valid} draw(s) (custom_id mismatch?)"
        assert mean is not None
    # The returned keys use the GLOBAL question idx (never reset per persona):
    # p0 -> idx 0,1 ; p1 -> idx 2,3 ; p2 -> idx 4. Value = the scored completion.
    assert agg["evil_pos_p0__00000__00"][0] == pytest.approx(11.0)
    assert agg["evil_pos_p0__00001__00"][0] == pytest.approx(22.0)
    assert agg["evil_pos_p1__00002__00"][0] == pytest.approx(33.0)
    assert agg["evil_pos_p1__00003__00"][0] == pytest.approx(44.0)
    assert agg["evil_pos_p2__00004__00"][0] == pytest.approx(55.0)


# ── BLOCKER judge-n5-cache-rerun-collapses-draws: cache never collapses draws ──


def test_judge_rollouts_n5_cache_rerun_does_not_collapse_draws(tmp_path, monkeypatch):
    """Calling judge_rollouts_n5 TWICE with the same cache dir yields the SAME
    five-draw mean both times (each draw an independent score), NOT the last cached
    score repeated N times.

    The stub scores by draw index (score = 10 + d), so a correct N=5 mean is
    10 + (N-1)/2. Pre-fix the wrapper passed cache_dir through to a cache keyed only
    on (question, completion); the N identical (question, completion) draws collapsed
    to ONE entry, so a warm-cache rerun returned the LAST cached draw (score 10+N-1)
    repeated N times -> mean 10+N-1, NOT 10+(N-1)/2. Post-fix the wrapper disables
    the cache, so both runs submit all N draws fresh and the mean is unchanged."""
    n = C.JUDGE_N_DRAWS

    def score_fn(_persona, _idx, ci, _comp):
        _ri, d = divmod(ci, n)
        return {"score": 10 + d, "reasoning": "x"}  # per-draw distinct score

    stub = _StubJudge(score_fn)
    monkeypatch.setattr("explore_persona_space.eval.batch_judge.judge_completions_batch", stub)
    rollouts = {"cellA": {"q000": ["ans0"]}}  # 1 rollout, N draws
    cache_dir = tmp_path / "judge_cache"
    expected_mean = 10 + (n - 1) / 2.0
    collapsed_mean = 10 + (n - 1)  # the last cached draw repeated N times

    # Run 1 (cold): five independent draws.
    agg1 = C.judge_rollouts_n5("evil", rollouts, tmp_path / "raw1.json", cache_dir)
    m1, nv1, _ = agg1["cellA__00000__00"]
    assert nv1 == n and m1 == pytest.approx(expected_mean)

    # Run 2 (would be warm if the cache saw the N identical (q, comp) copies): the
    # mean MUST still be the five-independent-draw mean, never the collapsed value.
    agg2 = C.judge_rollouts_n5("evil", rollouts, tmp_path / "raw2.json", cache_dir)
    m2, nv2, _ = agg2["cellA__00000__00"]
    assert nv2 == n
    assert m2 == pytest.approx(expected_mean)
    assert m2 != pytest.approx(collapsed_mean)
    assert m2 == pytest.approx(m1)
    # Cache disabled -> the second run re-submits all N draws (no collapse-to-1).
    assert stub.last_n_submitted == n


# ── analyzer minor note 1: h1a nullity is over the DOT arm ────────────────────


def test_stage1_h1a_uses_dot_arm():
    """h1a_nullity reads the DOT arm (r1_ridge_dot), not cosine (r1_ridge_cos):
    §8 identity r~_B = M^T r_B is over the DOT readout. Functional over a fake
    method_res with DISTINCT dot / cos / probe_ctrl values — the abs_diff proves
    the comparison used dot-vs-probe_ctrl, not cos-vs-probe_ctrl."""

    def _arm(sys_pt, many_pt):
        return {
            "system": {"point": sys_pt, "lo": sys_pt, "hi": sys_pt},
            "many_shot": {"point": many_pt, "lo": many_pt, "hi": many_pt},
        }

    method_res = {
        "r1_ridge_dot": _arm(0.80, 0.70),  # the arm h1a MUST use
        "r1_ridge_cos": _arm(0.10, 0.20),  # a decoy — must NOT be read
        "probe_ctrl": _arm(0.50, 0.40),
    }
    h1a = S.h1a_nullity(method_res)
    # Uses the DOT arm's points (0.80 / 0.70), not the cosine decoy (0.10 / 0.20).
    assert h1a["system"]["r1_ridge_dot_r"] == pytest.approx(0.80)
    assert h1a["many_shot"]["r1_ridge_dot_r"] == pytest.approx(0.70)
    assert h1a["system"]["probe_ctrl_r"] == pytest.approx(0.50)
    # abs_diff is |dot - probe_ctrl| = |0.80 - 0.50| = 0.30, NOT |cos - pc| = 0.40.
    assert h1a["system"]["abs_diff"] == pytest.approx(0.30)
    assert h1a["many_shot"]["abs_diff"] == pytest.approx(0.30)  # |0.70 - 0.40|


# ── BLOCKER: primary metric per-(condition,question), not per-rollout ─────────


def _cell_for_matrix(cond_id, mode, cx_last, oracle_proj, judge_scores, rollouts):
    """A build_eval_matrix-shaped Pass-A cell (post-load): the analyzer loads the
    cell JSON + attaches _cx_last / _layers from the sibling _cx.pt."""
    return {
        "trait": "evil",
        "cond_id": cond_id,
        "mode": mode,
        "_layers": [0],  # single read-out layer at index 0
        "_cx_last": np.asarray(cx_last, dtype=np.float64),  # (n_q, L=1, H)
        "oracle_proj": oracle_proj,
        "judge_scores": judge_scores,
        "rollouts": rollouts,
    }


def test_build_eval_matrix_is_per_question_averaged_not_per_rollout():
    """PRODUCTION-unit regression (BLOCKER
    primary-metric-rollout-level-not-question-averaged, code-review v4): the
    within-condition monitoring unit is ONE row per (condition, question) —
    matching Persona Vectors (arXiv 2507.21509 App. "Monitoring prompt-induced
    persona shifts"): the x-axis (final prompt-token projection) is a property of
    the PROMPT and the y-axis is the question's response trait score. Pre-fix
    build_eval_matrix appended one row PER ROLLOUT (2 questions x 2 rollouts ->
    N=4, the same c_last repeated). Post-fix: N=2 rows (one per question), y is
    the MEAN judge score over the question's rollouts."""
    r_b = np.array([[1.0, 0.0]])  # (L=1, H=2) — pv_raw = c_last[:,0]
    # 2 questions x 2 rollouts. Distinct per-question prompt vectors.
    cx_last = [[[3.0, 0.0]], [[7.0, 0.0]]]  # (n_q=2, L=1, H=2)
    # judge_scores keyed {persona}__{qi:05d}__{ri:02d}; q0 rollouts 10 & 20 -> mean 15;
    # q1 rollouts 40 & 60 -> mean 50.
    judge_scores = {
        "evil__00000__00": 10.0,
        "evil__00000__01": 20.0,
        "evil__00001__00": 40.0,
        "evil__00001__01": 60.0,
    }
    # oracle_proj per (qi, ri) -> {layer_idx: value}; per-question mean.
    oracle_proj = {
        "0": {"0": {"0": 1.0}, "1": {"0": 3.0}},  # q0 mean 2.0
        "1": {"0": {"0": 5.0}, "1": {"0": 7.0}},  # q1 mean 6.0
    }
    # pooled per rollout {op: [per-layer]}; r2_max mean over q0 rollouts = (0.2+0.4)/2.
    rollouts = [
        {"qi": 0, "ri": 0, "pooled": {"mean": [0.1], "max": [0.2], "topk": [0.15], "last": [0.05]}},
        {"qi": 0, "ri": 1, "pooled": {"mean": [0.3], "max": [0.4], "topk": [0.35], "last": [0.25]}},
        {"qi": 1, "ri": 0, "pooled": {"mean": [0.5], "max": [0.6], "topk": [0.55], "last": [0.45]}},
        {"qi": 1, "ri": 1, "pooled": {"mean": [0.7], "max": [0.8], "topk": [0.75], "last": [0.65]}},
    ]
    cell = _cell_for_matrix("sys0", "system", cx_last, oracle_proj, judge_scores, rollouts)
    mat = S.build_eval_matrix([cell], layer_idx=0, r_b=r_b)

    # PER-QUESTION: exactly 2 rows (NOT 4 rollout rows).
    assert len(mat["y"]) == 2, mat["y"]
    assert mat["c_last"].shape == (2, 2), mat["c_last"].shape
    # y is the MEAN judge score per question: q0 -> 15, q1 -> 50.
    assert sorted(mat["y"].tolist()) == [15.0, 50.0], mat["y"]
    # pv_raw = <c_last, r_b> per question: 3.0 and 7.0 (single value each, not duplicated).
    assert sorted(mat["pv_raw"].tolist()) == [3.0, 7.0], mat["pv_raw"]
    # oracle is the per-question mean of the rollout oracle projections: 2.0 and 6.0.
    assert sorted(mat["oracle"].tolist()) == [2.0, 6.0], mat["oracle"]
    # r2_max is the per-question mean of the rollout pooled-max: q0 (0.2+0.4)/2=0.3, q1 0.7.
    assert sorted(np.round(mat["r2_max"], 6).tolist()) == [0.3, 0.7], mat["r2_max"]


def test_build_eval_matrix_drops_question_with_no_valid_rollout():
    """A question whose rollouts are all judge-dropped (score None) or all empty
    contributes NO row; a question with a mix keeps the valid rollouts' mean."""
    r_b = np.array([[1.0, 0.0]])
    cx_last = [[[2.0, 0.0]], [[9.0, 0.0]]]  # q0 (all-dropped), q1 (one valid)
    judge_scores = {
        # q0: no entries -> _score_for returns None for both rollouts -> dropped.
        "evil__00001__00": 80.0,  # q1 r0 valid
        # q1 r1 has no score -> dropped from the mean.
    }
    oracle_proj = {"1": {"0": {"0": 4.0}}}
    rollouts = [
        {"qi": 0, "ri": 0, "pooled": {"mean": [0.0], "max": [0.0], "topk": [0.0], "last": [0.0]}},
        {"qi": 0, "ri": 1, "pooled": {"mean": [0.0], "max": [0.0], "topk": [0.0], "last": [0.0]}},
        {"qi": 1, "ri": 0, "pooled": {"mean": [0.5], "max": [0.6], "topk": [0.5], "last": [0.5]}},
        {"qi": 1, "ri": 1, "pooled": {"mean": [0.9], "max": [0.9], "topk": [0.9], "last": [0.9]}},
    ]
    cell = _cell_for_matrix("sys0", "system", cx_last, oracle_proj, judge_scores, rollouts)
    mat = S.build_eval_matrix([cell], layer_idx=0, r_b=r_b)
    # Only q1 survives (q0 fully dropped): one row, y = mean of the valid {80.0}.
    assert len(mat["y"]) == 1, mat["y"]
    assert mat["y"][0] == pytest.approx(80.0)
    assert mat["pv_raw"][0] == pytest.approx(9.0)
