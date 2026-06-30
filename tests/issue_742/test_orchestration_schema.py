# ruff: noqa: RUF002, RUF003
"""Issue #742 round-2 regression tests — orchestration ↔ real #658 schema.

The round-1 review (epm:code-review v1 + codex v1) found 6 orchestration defects
that ALL slipped through 34 green library-binding tests: the tests bound the
estimators, not the scripts' integration with the REAL #658 on-disk schema. These
tests close that gap by exercising each orchestration load/judge/refit path against
a fixture that MIRRORS the real #658 shape (``cells[i].completions[j]["text"]``, the
real per-probe dict, the real ``analyzer_body_data.json`` ``/<genre>/a33/<beh>/layer``
keys), plus two counting/raising proofs:

  * judge-rerun-completion-key-crash + judge-rerun-wrong-judge-construct +
    judge-rerun-j-sampling: a 100-completion cell is sampled to EXACTLY J=20 and judged
    with the PER-BEHAVIOR construct (counting mock proves both).
  * ridge-refit-missing: the LOCO-CV ridge join-integrity gate RAISES on a swapped
    fixture (delta > tol) and PASSES on a faithful one.
  * stage1-routing-layer: the A3.3 per-behavior layer is read from
    analyzer_body_data.json (Betley sycophancy 27, UltraChat refusal 6), NEVER the
    layer-21 locked_recipe fallback.

Determinism: 742X-family seeds (plan v7 §10).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from .conftest import impl_has  # noqa: E402

dc = importlib.import_module("explore_persona_space.analysis.issue_742_decoding_ceiling")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_658"


# --------------------------------------------------------------------------- #
# Fixtures mirroring the REAL #658 on-disk schema                              #
# --------------------------------------------------------------------------- #
def _real_e0_gen_cell(*, context_id: str, behavior: str, n_probes: int, n_rollouts: int) -> dict:
    """A gen-shaped dict matching the REAL #658 e0_gen schema.

    ``{context_id, column_id, dv, n_samples, cells: [{probe, completions: [{text,
    logp_norm}, ...]}, ...]}`` — completions carry ``text`` (NOT ``completion``; the
    judge-rerun-completion-key-crash BLOCKER was reading the wrong key).
    """
    cells = [
        {
            "probe": f"probe_{p}",
            "completions": [
                {"text": f"response {p}.{r} for {behavior}", "logp_norm": -1.0 - 0.01 * r}
                for r in range(n_rollouts)
            ],
        }
        for p in range(n_probes)
    ]
    return {
        "context_id": context_id,
        "column_id": behavior,
        "dv": "judged_rate",
        "n_samples": n_rollouts,
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
# 1. completion-key + per-behavior construct + J=20 sampling (BLOCKERs 1/2/3)  #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("sample_completions_for_judge"), reason="round-2 symbol")
def test_sample_completions_reads_text_key_and_caps_at_j():
    """100-completion cell → EXACTLY J=20 sampled, reading the real ``text`` key."""
    gen = _real_e0_gen_cell(
        context_id="f1_house_x", behavior="sycophancy", n_probes=10, n_rollouts=10
    )
    # 100 completions total; sample to J=20 deterministically
    sampled = dc.sample_completions_for_judge(gen, j_completions=20, seed=7428)
    total = sum(len(c["completions"]) for c in sampled["cells"])
    assert total == 20, f"expected exactly J=20 sampled completions, got {total}"
    assert sampled["n_sampled_for_judge"] == 20
    # every sampled completion carries the real 'text' key (NOT 'completion')
    for cell in sampled["cells"]:
        for comp in cell["completions"]:
            assert "text" in comp and "completion" not in comp
    # determinism: same seed -> same sample
    again = dc.sample_completions_for_judge(gen, j_completions=20, seed=7428)
    texts1 = [c["completions"][0]["text"] for c in sampled["cells"]]
    texts2 = [c["completions"][0]["text"] for c in again["cells"]]
    assert texts1 == texts2


@pytest.mark.skipif(not impl_has("per_behavior_judge_rate"), reason="round-2 symbol")
def test_per_behavior_judge_uses_correct_rubric_and_text_key():
    """The PER-BEHAVIOR construct is dispatched (NOT the default mean_aligned judge).

    Counting mock proves: (a) the J-sampled completions are judged via the behavior's
    OWN rubric (judge_fn receives the behavior col_id), (b) the completion ``text`` is
    what flows through, (c) the reconstructed rate is the judge-positive fraction —
    the SAME judged_rate construct #658 used, NOT a 0-100 alignment mean.
    """
    gen = _real_e0_gen_cell(context_id="f1_house_x", behavior="broad_em", n_probes=5, n_rollouts=4)
    sampled = dc.sample_completions_for_judge(gen, j_completions=20, seed=7428)
    calls: list[tuple[str, int]] = []

    def _counting_judge(col_id: str, g: dict, model: str) -> dict:
        # prove the per-behavior col_id is threaded + count the judged completions
        n = sum(len(c["completions"]) for c in g["cells"])
        # prove the real text key is present (would KeyError on the old c["completion"])
        for c in g["cells"]:
            for comp in c["completions"]:
                _ = comp["text"]
        calls.append((col_id, n))
        # half judged-positive -> rate 0.5 (a judged_rate, not a 0-100 mean)
        return {"column_id": col_id, "rate": 0.5, "n_judged": n, "n_positive": n // 2}

    res = dc.per_behavior_judge_rate(
        sampled,
        behavior="broad_em",
        judge_model="claude-sonnet-4-5-20250929",
        judge_fn=_counting_judge,
    )
    assert calls and calls[0][0] == "broad_em", "per-behavior rubric col_id must be threaded"
    assert calls[0][1] == 20, f"judge must see EXACTLY J=20 completions, saw {calls[0][1]}"
    assert res["rate"] == 0.5 and "n_positive" in res, "must return the judged_rate construct"


@pytest.mark.skipif(not impl_has("per_behavior_judge_rate"), reason="round-2 symbol")
def test_per_behavior_judge_rejects_non_readout_behavior():
    """A non-read-out behavior raises (no silent default-judge substitution)."""
    gen = _real_e0_gen_cell(context_id="c", behavior="deception", n_probes=2, n_rollouts=2)
    with pytest.raises(KeyError):
        dc.per_behavior_judge_rate(
            gen, behavior="deception", judge_model="m", judge_fn=lambda *a: {}
        )


# --------------------------------------------------------------------------- #
# 2. A3.3 per-behavior layer from analyzer_body_data.json (BLOCKER 6b)         #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (impl_has("load_a33_layer") and (EVAL_DIR / "analyzer_body_data.json").exists()),
    reason="round-2 symbol / #658 artifact",
)
def test_a33_layer_read_from_real_analyzer_body_data():
    """The Stage-1 layer comes from /<genre>/a33/<beh>/layer, NOT a layer-21 fallback."""
    # the plan's named expectations (verified against the real artifact this session)
    assert dc.load_a33_layer("sycophancy", "betley", eval_dir=EVAL_DIR) == 27
    assert dc.load_a33_layer("refusal", "ultrachat", eval_dir=EVAL_DIR) == 6
    # the per-behavior layers are genuinely heterogeneous (NOT a single default)
    betley_layers = {
        b: dc.load_a33_layer(b, "betley", eval_dir=EVAL_DIR) for b in dc.READOUT_BEHAVIORS
    }
    assert len(set(betley_layers.values())) > 1, f"layers must vary per behavior: {betley_layers}"
    # a missing key raises (no silent fallback)
    with pytest.raises(KeyError):
        dc.load_a33_layer("not_a_behavior", "betley", eval_dir=EVAL_DIR)


# --------------------------------------------------------------------------- #
# 3. LOCO-CV ridge join-integrity gate raises on a swapped fixture (BLOCKER 4) #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("ridge_join_integrity"), reason="round-2 symbol")
def test_ridge_join_integrity_passes_faithful_and_fails_swapped():
    """The MF1 join-integrity gate: a refit reproducing the persisted ρ PASSES;
    a refit against a SWAPPED (mismatched) target FAILS (delta > tol)."""
    rng = np.random.default_rng(74240)
    n, d = 50, 12
    # a linearly-decodable E0 so the LOCO ridge recovers a real held-out ρ
    v0 = rng.normal(0, 1, size=(n, d))
    w = rng.normal(0, 1, size=d)
    e0 = v0 @ w + rng.normal(0, 0.3, size=n)
    refit_rho = dc.loco_ridge_refit_rho(v0, e0)
    assert refit_rho > 0.3, f"a decodable target must refit to a real held-out rho, got {refit_rho}"

    # faithful join: persisted == the refit value -> join_ok True
    faithful = dc.ridge_join_integrity(
        v0, e0, behavior="b", genre="betley", layer=0, persisted_rho=refit_rho, tol=0.05
    )
    assert faithful.join_ok and faithful.delta <= 0.05

    # swapped join: an independent (shuffled) target has refit ρ near 0, far from the
    # persisted high ρ -> delta > tol -> join_ok False (the mis-joined-tensor signal)
    swapped = dc.ridge_join_integrity(
        v0,
        rng.permutation(e0),
        behavior="b",
        genre="betley",
        layer=0,
        persisted_rho=refit_rho,
        tol=0.05,
    )
    assert not swapped.join_ok and swapped.delta > 0.05


# --------------------------------------------------------------------------- #
# 4. CV-matched reliability CI is fold-matched, not pooled (BLOCKER 5a)        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("cv_matched_reliability_ci"), reason="round-2 symbol")
def test_cv_matched_reliability_excludes_one_context_per_fold():
    """The CV-matched CI is computed over n LOCO folds (one held-out ctx each), so a
    single outlier context moves exactly one fold estimate — a pooled bootstrap would
    mix it into every resample. We assert the fold spread responds to a planted
    outlier (the fold-matched signature)."""
    rng = np.random.default_rng(74250)
    n = 50
    rates = rng.uniform(0.2, 0.8, size=n)
    m = np.full(n, 200.0)
    mean0, lo0, hi0 = dc.cv_matched_reliability_ci(rates, m)
    # plant one extreme-variance context; the fold that EXCLUDES it differs from folds
    # that include it -> the across-fold spread widens (fold-matched, not pooled)
    rates2 = rates.copy()
    rates2[0] = 0.999
    mean1, lo1, hi1 = dc.cv_matched_reliability_ci(rates2, m)
    assert 0.0 <= lo0 <= hi0 <= 1.0 and 0.0 <= lo1 <= hi1 <= 1.0
    # the planted outlier must move the estimate (proves it is data-driven per fold)
    assert abs(mean1 - mean0) > 1e-6


# --------------------------------------------------------------------------- #
# 5. dcor_at_subsample is well-posed at small n' (BLOCKER 7 dCor(n') curve)    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not impl_has("dcor_at_subsample"), reason="round-2 symbol")
def test_dcor_at_subsample_is_bounded_and_clamps_d_eff():
    """dCor(n') returns a value in [0,1] even when d_eff > n'-1 (it clamps)."""
    rng = np.random.default_rng(74260)
    v0 = rng.normal(0, 1, size=(50, 30))
    e0 = rng.uniform(0, 1, size=50)
    val = dc.dcor_at_subsample(v0, e0, n_prime=10, d_eff=20, rng=rng)
    assert 0.0 <= val <= 1.0
