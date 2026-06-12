"""Task #608 follow-up sub-ceiling-install: judge_pass_subceiling fixtures.

Enumeration completeness gate (108 step reads), judge-pass loop + resume,
judgments schema, and the mid-band Sonnet spot-check — with ``judge_batch``
monkeypatched (the loop / serialization / resume code paths are REAL).

Committed from the round-1 /tmp smoke per code-review v4.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import explore_persona_space.experiments.sycophancy_posonly_608.judge_pass_subceiling as jps
from explore_persona_space.experiments.sycophancy_posonly_608 import (
    FOLLOWUP_GRID_STEPS,
    cell_slab_dir,
    followup_production_cells,
)


def _build_eval_tree(slab):
    """108-file eval fixture tree (tiny synthetic completions, real schema)."""
    for source, arm in followup_production_cells():
        for k in FOLLOWUP_GRID_STEPS:
            d = cell_slab_dir(slab, source, arm, 42) / "steps" / f"step_{k}"
            d.mkdir(parents=True, exist_ok=True)
            completions = [
                {
                    "claim": f"wrong claim {c}",
                    "completion": f"synthetic reply {c}-{r}",
                    "claim_idx": c,
                    "rollout_idx": r,
                }
                for c in range(50)
                for r in range(10)
            ]
            with open(d / f"sycophancy_eval_{source}.json", "w") as f:
                json.dump({"panel_persona": source, "completions": completions}, f)


async def _fake_haiku_batch(records, model, max_concurrency=32):
    return [
        SimpleNamespace(
            wrong_claim=r["wrong_claim"],
            completion=r["completion"],
            agreed=(r["claim_idx"] + r["rollout_idx"]) % 3 == 0,
            raw_response="YES/NO",
            model=model,
            error=None,
        )
        for r in records
    ]


async def _fake_sonnet_batch(records, model, max_concurrency=32):
    return [
        SimpleNamespace(
            wrong_claim=r["wrong_claim"],
            completion=r["completion"],
            # disagree on every 10th sampled rollout
            agreed=(not r["haiku_agreed"]) if i % 10 == 0 else r["haiku_agreed"],
            raw_response="YES/NO",
            model=model,
            error=None,
        )
        for i, r in enumerate(records)
    ]


def test_judge_pass_subceiling_end_to_end(tmp_path, monkeypatch):
    slab = tmp_path / "slab"
    _build_eval_tree(slab)

    # ---- enumeration completeness gate (PASS at exactly 108 reads) ----------
    reads = jps.enumerate_subceiling_dirs(slab, 42)
    assert len(reads) == 108, len(reads)

    # ---- gate FAIL on a missing read ----------------------------------------
    victim = cell_slab_dir(slab, "comedian", "contrastive_dense", 42) / "steps" / "step_26"
    moved = victim.with_name("step_26_HIDDEN")
    victim.rename(moved)
    try:
        jps.enumerate_subceiling_dirs(slab, 42)
        raise AssertionError("completeness gate should have raised")
    except FileNotFoundError as e:
        assert "comedian:contrastive_dense/step_26" in str(e)
    moved.rename(victim)

    # ---- monkeypatched Haiku judge pass + resume -----------------------------
    monkeypatch.setattr(jps, "judge_batch", _fake_haiku_batch)
    totals = jps.run_subceiling_judge_pass(slab, 42, concurrency=4)
    assert totals["n_panels_judged"] == 108 and totals["n_panels_skipped"] == 0, totals
    totals2 = jps.run_subceiling_judge_pass(slab, 42, concurrency=4)
    assert totals2["n_panels_judged"] == 0 and totals2["n_panels_skipped"] == 108, totals2

    # judgments shape: claim_idx carried per verdict (bootstrap input contract)
    jf = (
        cell_slab_dir(slab, "villain", "posonly_dose_dense", 42)
        / "steps"
        / "step_5"
        / "judgments"
        / "villain.json"
    )
    with open(jf) as f:
        payload = json.load(f)
    assert payload["n_verdicts"] == 500 and payload["n_api_errors"] == 0
    assert {"claim_idx", "rollout_idx", "agreed", "wrong_claim"} <= set(payload["verdicts"][0])
    assert payload["checkpoint"] == "step_5"
    assert payload["followup_label"] == "sub-ceiling-install"

    # ---- mid-band spot-check (monkeypatched Sonnet, ~10% forced disagreement)
    monkeypatch.setattr(jps, "judge_batch", _fake_sonnet_batch)
    report = jps.run_midband_spotcheck(slab, 42, n=200, concurrency=4)
    assert report["spotcheck_n"] == 200, report["spotcheck_n"]
    assert 0.5 < report["kappa"] <= 1.0, report["kappa"]
    assert set(report["disagreement_by_arm"]) <= {"contrastive_dense", "posonly_dose_dense"}
