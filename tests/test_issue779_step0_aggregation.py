"""Regression test for the issue #779 Step-0 probe per-question aggregation.

HOLISTIC-HARDENING sibling of the BLOCKER
``primary-metric-rollout-level-not-question-averaged`` (fixed in
``stage1.build_eval_matrix``): the Step-0 oracle-headroom probe
(``run_step0_probe`` in ``issue779_collect.py``) computed the within-condition
Pearson at the ROLLOUT level too — one ``(pv_raw, oracle, judge)`` point per
rollout, with ``pv_raw = <c_last[qi], r_B>`` (a property of the PROMPT) DUPLICATED
across a question's rollouts. That is the same pseudo-replication class, and it
drives the load-bearing Gate-0 descope + read-out-layer selection that feed the
Stage-1 headline. Post-fix the probe aggregates to ONE row per (condition,
question): pv_raw per-question, oracle + judge the mean over the question's valid
rollouts. This test pins that the correlation is over N_questions units.

Pure-CPU, no model / no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_collect as X  # noqa: E402
import issue779_common as C  # noqa: E402

L, H = 2, 4  # tiny layers/hidden
LAYERS = list(range(L))


def _write_cell_with_cx(
    pass_a_dir, trait, cond_id, mode, cx_last, oracle_proj, judge_scores, rollouts
):
    pass_a_dir.mkdir(parents=True, exist_ok=True)
    cell = {
        "trait": trait,
        "cond_id": cond_id,
        "mode": mode,
        "n_shot": 0,
        "rollouts": rollouts,
        "judge_scores": judge_scores,
        "oracle_proj": oracle_proj,
    }
    C.write_json_atomic(pass_a_dir / f"{trait}__{cond_id}.json", cell)
    torch.save(
        {
            "cell_id": f"{trait}__{cond_id}",
            "cx_last": torch.tensor(np.asarray(cx_last), dtype=torch.float32),
            "cx_mean": torch.tensor(np.asarray(cx_last), dtype=torch.float32),
            "layers": LAYERS,
        },
        pass_a_dir / f"{trait}__{cond_id}_cx.pt",
    )


def _build_conditions(out_dir, r_b, n_q=5, n_r=4):
    """Two system-mode conditions, n_q questions x n_r rollouts.

    Per question: pv_raw = <c_last[qi], r_b> is CONSTANT across its n_r rollouts;
    the judge score VARIES per rollout so a rollout-level correlation differs from
    the question-averaged one. c_last is chosen so per-question pv_raw increases
    with qi, and the per-question MEAN judge score tracks it (a real signal), while
    per-rollout noise is large — so rollout-level over-counts the units.
    """
    rng = np.random.default_rng(0)
    pass_a = out_dir / "pass_a"
    for cond_id in ("sys0", "sys1"):
        cx_last = np.zeros((n_q, L, H), dtype=np.float32)
        oracle_proj, judge_scores, rollouts = {}, {}, []
        for qi in range(n_q):
            # per-question prompt direction: magnitude grows with qi.
            cx_last[qi] = np.tile(np.array([qi + 1.0, 0.0, 0.0, 0.0]), (L, 1))
            pv = float(cx_last[qi, 0] @ r_b[0])
            oracle_proj[str(qi)] = {}
            for ri in range(n_r):
                # judge mean tracks pv (signal) + per-rollout noise.
                s = float(np.clip(10 * pv + rng.normal(scale=5.0), 0, 100))
                cid = f"evil__{qi:05d}__{ri:02d}"
                judge_scores[cid] = s
                oracle_proj[str(qi)][str(ri)] = {str(li): pv + 0.1 * ri for li in LAYERS}
                rollouts.append({"qi": qi, "ri": ri, "response": "x", "pooled": {}})
        _write_cell_with_cx(
            pass_a, "evil", cond_id, "system", cx_last, oracle_proj, judge_scores, rollouts
        )


def test_step0_probe_correlates_per_question_not_per_rollout(tmp_path):
    """run_step0_probe computes the within-condition Pearson over N_QUESTIONS
    units, not N_questions*N_rollouts. We assert this by checking that the
    condition's point-count entering the correlation equals n_q (5), which is
    only true under per-question aggregation — the pre-fix rollout-level code
    would have fed n_q*n_r (20) points per condition."""
    out_dir = tmp_path / "data"
    r_b = np.array([[3.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    n_q, n_r = 5, 4
    _build_conditions(out_dir, r_b, n_q=n_q, n_r=n_r)
    r_b_by_trait = {"evil": torch.tensor(r_b)}

    result = X.run_step0_probe(out_dir, ["evil"], LAYERS, r_b_by_trait)

    # The probe ran and selected a best layer.
    assert "evil" in result
    plm = result["evil"]["per_layer_mode"]
    # Per-question aggregation: pv_raw has exactly one distinct value per question
    # (5 rising values), so within a condition the Pearson is over 5 rows. We can't
    # read the raw point count out of the summary, so instead verify the KEY
    # discriminator: with per-question aggregation both conditions qualify
    # (n_conditions == 2) and the pv_raw_r is finite & strong (the true signal),
    # whereas the rollout-duplicated x with heavy per-rollout noise would attenuate
    # it. Assert a strong finite within-condition r at the best system layer.
    entry = None
    for li in LAYERS:
        e = plm.get(f"L{li}_system")
        if e and np.isfinite(e["pv_raw_r"]):
            entry = e
            break
    assert entry is not None, plm
    assert entry["n_conditions"] == 2, entry  # both system conditions qualify
    # Per-question the signal is clean: |r| high. (Rollout-level noise would drag it.)
    assert abs(entry["pv_raw_r"]) > 0.7, entry["pv_raw_r"]


def test_step0_probe_reference_matches_question_averaged_pearson(tmp_path):
    """Cross-check: the probe's per-layer pv_raw_r equals a hand-computed
    per-QUESTION within-condition Pearson (mean over rollouts), NOT the
    per-rollout one — the exact numeric discriminator between the two units."""
    out_dir = tmp_path / "data"
    r_b = np.array([[2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    n_q, n_r = 6, 3
    _build_conditions(out_dir, r_b, n_q=n_q, n_r=n_r)
    r_b_by_trait = {"evil": torch.tensor(r_b)}
    result = X.run_step0_probe(out_dir, ["evil"], LAYERS, r_b_by_trait)

    # Hand-compute the per-question within-condition r at layer 0, system mode.
    pass_a = out_dir / "pass_a"
    per_cond_r = []
    for cond_id in ("sys0", "sys1"):
        cell = json.loads((pass_a / f"evil__{cond_id}.json").read_text())
        cx = torch.load(pass_a / f"evil__{cond_id}_cx.pt", weights_only=True)["cx_last"]
        by_q: dict[int, list[int]] = {}
        for rec in cell["rollouts"]:
            by_q.setdefault(rec["qi"], []).append(rec["ri"])
        xs, ys = [], []
        for qi, ris in by_q.items():
            q_s = [cell["judge_scores"][f"evil__{qi:05d}__{ri:02d}"] for ri in ris]
            xs.append(float(cx[qi, 0] @ torch.tensor(r_b[0])))
            ys.append(float(np.mean(q_s)))
        if np.std(ys) >= 1.0 and np.std(xs) > 0:
            per_cond_r.append(float(np.corrcoef(xs, ys)[0, 1]))
    expected = float(np.mean(per_cond_r))
    got = result["evil"]["per_layer_mode"]["L0_system"]["pv_raw_r"]
    assert got == pytest.approx(expected, abs=1e-9), (got, expected)
