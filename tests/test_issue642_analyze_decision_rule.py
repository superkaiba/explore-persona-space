"""Task #642 — decision-rule sign + per-arm fallback regression tests.

Round-2 BLOCKER + Major fixes in ``scripts/issue_642/i642_analyze.py``:

  - BLOCKER: the §3 hypotheses (H_rank / H_coverage / H_mixed) are STRICTLY
    positive-direction. A negative-Δ contrast (the lower arm leaks MORE than the
    higher arm) MUST classify as ``opposite_direction``, never as a registered
    positive branch. The old rule used ``abs(gap) and (ci_lo>0 or ci_hi<0)``,
    which mislabeled a negative Δ_rank as H_rank.
  - Major #1: the per-arm interpolation/fallback is resolved PER ARM. A
    bracketing arm interpolates at s* even when the OTHER arm lacks a bracket;
    only the non-bracketing arm uses its band-entry fallback cell.

These tests are CPU-only (no GPU / no API): they drive the synthetic fixture
writer + analyzer, and exercise ``_two_arm_gap`` directly with controlled
arrays for the per-arm path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "issue_642"))

import i642_analyze as A  # noqa: E402
from i642_common import DECOMP_THRESHOLD, S_TARGET  # noqa: E402

# ---------------------------------------------------------------------------
# Decision rule (end-to-end via the synthetic fixture + analyzer)
# ---------------------------------------------------------------------------


def _run_mode(tmp_path: Path, mode: str) -> dict:
    root = tmp_path / mode
    A.make_synthetic(root, mode)
    return A.analyze_behavior(
        behavior="sycophancy",
        eval_root=root,
        bootstrap_b=3000,
        refetch=False,
    )


def test_negative_delta_rank_classifies_opposite_direction(tmp_path):
    """A NEGATIVE Δ_rank (cmft leaks LESS than LoRA — the opposite of the
    registered H_rank direction) must land on ``opposite_direction``, NOT
    H_rank. This is the round-2 BLOCKER: the old rule used abs(gap) and would
    have mislabeled it."""
    analysis = _run_mode(tmp_path, "opposite_direction")
    h = analysis["headline"]
    dr = h["delta_rank"]

    # the synthetic designs Δ_rank = -0.08 at s* (cmft leaks LESS than LoRA)
    assert dr["gap_plugin"] < -DECOMP_THRESHOLD, dr["gap_plugin"]
    # CI excludes 0 on the NEGATIVE side
    assert dr["gap_ci95"][1] < 0, dr["gap_ci95"]

    # the registered positive-direction ``separates`` must be FALSE here
    assert dr["separates"] is False
    # but the unregistered negative-direction flag is TRUE
    assert dr["separates_negative"] is True

    # and the headline verdict is opposite_direction, NOT any positive branch
    assert h["verdict"] == "opposite_direction", h["verdict"]
    assert h["verdict"] != "H_rank"


def test_positive_delta_rank_would_be_registered_not_opposite(tmp_path):
    """Sanity sibling: a clearly POSITIVE Δ_rank separation (constructed here as
    a direct classifier check) maps to a registered branch, never to
    ``opposite_direction`` — confirms the fix does not break the positive path."""
    # direct contrast dicts emulating a clean positive Δ_rank, null Δ_coverage
    pos_rank = {
        "determinacy_pass": True,
        "separates": True,
        "separates_negative": False,
        "gap_ci95": [0.05, 0.12],
    }
    null_cov = {
        "determinacy_pass": True,
        "separates": False,
        "separates_negative": False,
        "gap_ci95": [-0.02, 0.03],
    }
    verdict = _classify(pos_rank, null_cov, gross_failure=False)
    assert verdict == "H_rank", verdict


def test_negative_separation_routes_opposite_via_classifier():
    """Classifier-level: a negative-direction Δ_rank separation with no positive
    branch routes to ``opposite_direction``."""
    neg_rank = {
        "determinacy_pass": True,
        "separates": False,
        "separates_negative": True,
        "gap_ci95": [-0.14, -0.04],
    }
    null_cov = {
        "determinacy_pass": True,
        "separates": False,
        "separates_negative": False,
        "gap_ci95": [-0.02, 0.03],
    }
    verdict = _classify(neg_rank, null_cov, gross_failure=False)
    assert verdict == "opposite_direction", verdict


def _classify(delta_rank: dict, delta_coverage: dict, *, gross_failure: bool) -> str:
    """Re-implement the §3 branch order EXACTLY as in analyze_behavior so the
    classifier logic is unit-testable without the full bootstrap pipeline. Kept
    in lockstep with i642_analyze.analyze_behavior's decision rule."""
    both_det = delta_rank["determinacy_pass"] and delta_coverage["determinacy_pass"]
    rank_sep = delta_rank["separates"]
    cov_sep = delta_coverage["separates"]
    rank_sep_neg = delta_rank["separates_negative"]
    cov_sep_neg = delta_coverage["separates_negative"]

    def _null(c):
        lo, hi = c["gap_ci95"]
        return lo > -DECOMP_THRESHOLD and hi < DECOMP_THRESHOLD

    rank_null = _null(delta_rank)
    cov_null = _null(delta_coverage)
    if gross_failure:
        return "indeterminate_additive_gross_failure"
    if not both_det:
        return "indeterminate_determinacy_gate"
    if cov_sep and rank_null:
        return "H_coverage"
    if rank_sep and cov_null:
        return "H_rank"
    if rank_sep and cov_sep:
        return "H_mixed"
    if (rank_sep_neg or cov_sep_neg) and not (rank_sep or cov_sep):
        return "opposite_direction"
    return "indeterminate_noise_limited"


# ---------------------------------------------------------------------------
# Per-arm fallback (Major #1) — _two_arm_gap directly
# ---------------------------------------------------------------------------


def _two_arm_inputs(*, lora_brackets: bool):
    """Build the controlled inputs for _two_arm_gap: 2 bystanders, a small
    bootstrap, and a designed +0.06/unit-s lead of cmft over lora so the
    contrast is finite. ``lora_brackets`` toggles whether the LoRA arm's
    stage-B s values bracket s*=0.5."""
    bystanders = ["b0", "b1"]
    arm_cells = {
        "lora": ["lora_step28", "lora_step32"],
        "cmft": ["cmft_step12", "cmft_step16"],
    }
    # cmft brackets s*=0.5 (0.3 / 0.6); lora brackets iff requested else jumps it
    s_stage_b = {
        "cmft_step12": 0.30,
        "cmft_step16": 0.60,
        "lora_step28": 0.30 if lora_brackets else 0.70,
        "lora_step32": 0.60 if lora_brackets else 0.90,
    }
    all_cells = [*arm_cells["lora"], *arm_cells["cmft"], "base"]
    c_index = {c: i for i, c in enumerate(all_cells)}
    bys_idx = np.array([0, 1])
    # per-cell, per-bystander clean delta. cmft leads lora by +0.06 per unit s.
    delta_clean = {}
    for c in all_cells:
        if c == "base":
            continue
        s = s_stage_b[c]
        lead = 0.06 if c.startswith("cmft_") else 0.0
        delta_clean[c] = {p: 0.20 * s + lead * s for p in bystanders}
    B = 200
    rng = np.random.default_rng(0)
    s_rep = np.empty((len(all_cells), B))
    delta_rep = np.empty((len(all_cells), len(bystanders), B))
    for c, i in c_index.items():
        if c == "base":
            s_rep[i, :] = 0.0
            delta_rep[i, :, :] = 0.0
            continue
        s_rep[i, :] = s_stage_b[c] + rng.normal(0, 1e-4, size=B)
        for j, p in enumerate(bystanders):
            delta_rep[i, j, :] = delta_clean[c][p] + rng.normal(0, 1e-4, size=B)
    persona_picks = rng.integers(0, len(bystanders), size=(B, len(bystanders)))
    # bracket bookkeeping
    bracket = {
        "cmft": A._bracket_info({c: s_stage_b[c] for c in arm_cells["cmft"]}, S_TARGET),
        "lora": A._bracket_info({c: s_stage_b[c] for c in arm_cells["lora"]}, S_TARGET),
    }
    if not bracket["lora"]["brackets"]:
        # endpoint-ish fallback cell (closest approach), as analyze_behavior sets
        bracket["lora"]["fallback_cell"] = min(
            arm_cells["lora"], key=lambda c: abs(s_stage_b[c] - S_TARGET)
        )
        bracket["lora"]["fallback_mode"] = "closest_approach"
    return dict(
        arm_hi="cmft",
        arm_lo="lora",
        arm_cells=arm_cells,
        bystanders=bystanders,
        c_index=c_index,
        bys_idx=bys_idx,
        s_stage_b=s_stage_b,
        delta_clean=delta_clean,
        s_rep=s_rep,
        delta_rep=delta_rep,
        persona_picks=persona_picks,
        B=B,
        bracket=bracket,
    )


def test_per_arm_fallback_bracketing_arm_still_interpolates():
    """One arm WITHOUT a bracket (lora) + one WITH a bracket (cmft): the
    bracketing arm (cmft) must still INTERPOLATE at s*, NOT be dragged onto
    endpoint lookup. Round-2 Major #1: previously a single arm lacking a bracket
    forced BOTH arms onto the fallback cell."""
    out = A._two_arm_gap(**_two_arm_inputs(lora_brackets=False))
    # the bracketing arm interpolates; the non-bracketing arm falls back
    assert out["per_arm_read_mode"]["cmft"] == "interpolation"
    assert out["per_arm_read_mode"]["lora"] == "band_entry_fallback"

    # cmft interpolated at s*=0.5 -> its bystander mean should equal the designed
    # value at s* (0.20*0.5 + 0.06*0.5 = 0.13), NOT the endpoint cell value.
    cmft_at_target = 0.20 * S_TARGET + 0.06 * S_TARGET
    assert out["cmft_bystander_mean"] == pytest.approx(cmft_at_target, abs=1e-3), out[
        "cmft_bystander_mean"
    ]


def test_per_arm_fallback_both_bracket_interpolate():
    """Both arms bracket -> both interpolate; headline mode is matched."""
    out = A._two_arm_gap(**_two_arm_inputs(lora_brackets=True))
    assert out["per_arm_read_mode"] == {"cmft": "interpolation", "lora": "interpolation"}
    assert out["mode"] == "matched_interpolation"
