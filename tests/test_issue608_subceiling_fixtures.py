"""Task #608 follow-up sub-ceiling-install: §6 decision-rule fixtures.

Drives the REAL ``analyze_subceiling.analyze()`` over synthetic judgments
trees (108 files = 6 sources x 2 arms x 9 grid steps). Six scenarios:

  A directional        -> subceiling_posonly_ahead (both reads agree)
  B equivalence        -> subceiling_no_separation
  C dual disagreement  -> all-m label != collision-robust label; robust carries
  D S50 fallback       -> m < 3, ordered S50 -> speed verdict carries
  E window missed      -> >= 4/6 no co-resolvable AND unorderable S50 -> kill
  F parity kill + m<3  -> retrain-parity kill caps the m<3 fallback headline:
                          no directional speed carry, verdict_flip False
                          (concern parity-kill-speed-verdict-uncapped, round 2)

Committed from the round-1 /tmp smoke per code-review v4 (the strongest
verification of the registered rule must be reproducible from the repo).
Requires the parent summary ``eval_results/issue_608/analyze_summary_608.json``
(skipped on sparse checkouts that exclude eval_results/).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from explore_persona_space.experiments.sycophancy_posonly_608 import (
    FOLLOWUP_GRID_STEPS,
    SOURCE_PERSONAS,
    cell_slab_dir,
)
from explore_persona_space.experiments.sycophancy_posonly_608.analyze_subceiling import (
    ARM_CONTR,
    ARM_POS,
    DETERMINATE_LABELS,
    analyze,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_SUMMARY = REPO_ROOT / "eval_results" / "issue_608" / "analyze_summary_608.json"
N_CLAIMS, N_ROLLOUTS = 50, 10
N_BOOT = 2000  # round-1 smoke convention; labels are far from CI boundaries

pytestmark = pytest.mark.skipif(
    not PARENT_SUMMARY.exists(),
    reason="parent analyze_summary_608.json not present (sparse checkout excludes eval_results/)",
)


def write_judgments(slab, arm, source, step, rate, claim_jitter=True, jitter_key=0):
    """One synthetic judgments file at a target rate with deterministic
    per-claim variation (claim c gets rate + ((c*7+jitter_key) % 11 - 5) * 0.02)."""
    d = cell_slab_dir(slab, source, arm, 42) / "steps" / f"step_{step}" / "judgments"
    d.mkdir(parents=True, exist_ok=True)
    verdicts = []
    for c in range(N_CLAIMS):
        r = rate + (((c * 7 + jitter_key) % 11) - 5) * 0.02 if claim_jitter else rate
        r = min(1.0, max(0.0, r))
        n_yes = round(r * N_ROLLOUTS)
        for ro in range(N_ROLLOUTS):
            verdicts.append(
                {
                    "wrong_claim": f"claim {c}",
                    "completion": "synthetic agreement placeholder",
                    "agreed": ro < n_yes,
                    "raw_response": "YES" if ro < n_yes else "NO",
                    "model": "synthetic",
                    "error": None,
                    "claim_idx": c,
                    "rollout_idx": ro,
                }
            )
    with open(d / f"{source}.json", "w") as f:
        json.dump(
            {
                "panel_persona": source,
                "n_verdicts": len(verdicts),
                "n_api_errors": 0,
                "verdicts": verdicts,
            },
            f,
        )


def build(slab, traj):
    """traj: {source: {arm: {step: rate}}} -> full 108-file fixture tree."""
    for s in SOURCE_PERSONAS:
        for arm in (ARM_CONTR, ARM_POS):
            for k in FOLLOWUP_GRID_STEPS:
                # SAME jitter pattern across arms => g~0 cells give exactly-0 diffs.
                write_judgments(slab, arm, s, k, traj[s][arm][k], jitter_key=0)


def parent_parity_132(s, arm):
    """Step-132 rates that keep the parity check quiet."""
    with open(PARENT_SUMMARY) as f:
        parent = json.load(f)
    ref = "contrastive_fresh_eval" if arm == ARM_CONTR else "posonly_dose"
    return parent["per_source"][s]["own_rate"][ref]


def scenario_directional():
    """posonly ahead at every co-resolvable step."""
    traj = {}
    for s in SOURCE_PERSONAS:
        pos = {
            5: 0.10,
            9: 0.25,
            13: 0.45,
            18: 0.60,
            26: 0.80,
            35: 0.93,
            44: 0.95,
            88: 0.97,
            132: parent_parity_132(s, ARM_POS),
        }
        con = {
            5: 0.05,
            9: 0.10,
            13: 0.30,
            18: 0.45,
            26: 0.65,
            35: 0.80,
            44: 0.88,
            88: 0.93,
            132: parent_parity_132(s, ARM_CONTR),
        }
        traj[s] = {ARM_POS: pos, ARM_CONTR: con}
    return traj


def scenario_equivalence():
    """g == 0 at every co-resolvable step (identical trajectories)."""
    traj = {}
    for s in SOURCE_PERSONAS:
        shared = {
            5: 0.10,
            9: 0.25,
            13: 0.45,
            18: 0.60,
            26: 0.80,
            35: 0.93,
            44: 0.95,
            88: 0.97,
            132: 0.96,
        }
        pos = dict(shared)
        con = dict(shared)
        pos[132] = parent_parity_132(s, ARM_POS)
        con[132] = parent_parity_132(s, ARM_CONTR)
        traj[s] = {ARM_POS: pos, ARM_CONTR: con}
    return traj


def scenario_dual_disagreement():
    """5 sources g==0; qwen_default contrastive far below -> all-m fails
    no_separation, robust read keeps it."""
    traj = scenario_equivalence()
    qd = traj["qwen_default"]
    qd[ARM_CONTR] = {
        5: 0.02,
        9: 0.04,
        13: 0.09,
        18: 0.20,
        26: 0.40,
        35: 0.55,
        44: 0.62,
        88: 0.68,
        132: parent_parity_132("qwen_default", ARM_CONTR),
    }
    return traj


def scenario_s50_fallback():
    """Both arms jump the band in single (different) gaps -> no co-resolvable
    steps, S50 ordered posonly-first."""
    traj = {}
    for s in SOURCE_PERSONAS:
        pos = {
            5: 0.05,
            9: 0.08,
            13: 0.95,
            18: 0.96,
            26: 0.96,
            35: 0.96,
            44: 0.97,
            88: 0.97,
            132: parent_parity_132(s, ARM_POS),
        }
        con = {
            5: 0.03,
            9: 0.05,
            13: 0.06,
            18: 0.08,
            26: 0.10,
            35: 0.95,
            44: 0.95,
            88: 0.96,
            132: parent_parity_132(s, ARM_CONTR),
        }
        if s == "qwen_default":
            con[35], con[44], con[88] = 0.60, 0.65, 0.70  # keeps parity at 0.704 sane
        traj[s] = {ARM_POS: pos, ARM_CONTR: con}
    return traj


def scenario_window_missed():
    """Both arms jump in the SAME gap -> S50 intervals overlap -> unorderable
    -> window_missed fires."""
    traj = {}
    for s in SOURCE_PERSONAS:
        jump = {
            5: 0.05,
            9: 0.08,
            13: 0.95,
            18: 0.96,
            26: 0.96,
            35: 0.96,
            44: 0.97,
            88: 0.97,
            132: 0.96,
        }
        pos = dict(jump)
        con = dict(jump)
        pos[132] = parent_parity_132(s, ARM_POS)
        con[132] = parent_parity_132(s, ARM_CONTR)
        if s == "qwen_default":
            # qwen contrastive jumps 9->13 to 0.78 (in band) -> in-band reads,
            # but posonly at those steps is > 0.9 -> still no co-resolvable
            # step; both S50 intervals are (9, 13] -> overlap -> unordered.
            for k in (13, 18, 26, 35, 44, 88):
                con[k] = min(con[k], 0.78)
        traj[s] = {ARM_POS: pos, ARM_CONTR: con}
    return traj


def scenario_parity_kill_m_lt_3():
    """Round-2 corner (concern parity-kill-speed-verdict-uncapped): no
    co-resolvable steps (m=0 < 3) AND a directional S50 ordering AND a
    parity-failed contrastive retrain (all 6 step-132 reads deviate -0.2
    from the parent committed endpoints, > the 0.10 tolerance for >= 3/6
    sources). Plan v5 §7 kill 3 must cap the headline at
    subceiling_indeterminate with NO directional speed carry."""
    traj = scenario_s50_fallback()
    for s in SOURCE_PERSONAS:
        traj[s][ARM_CONTR][132] = max(0.0, parent_parity_132(s, ARM_CONTR) - 0.2)
    return traj


def run(slab: Path, traj, figures_dir: Path | None = None):
    build(slab, traj)
    return analyze(
        slab_root=slab,
        seed=42,
        figures_dir=figures_dir,
        n_boot=N_BOOT,
        parent_summary_path=PARENT_SUMMARY,
    )


def test_scenario_directional_and_figures(tmp_path):
    s = run(tmp_path / "slab", scenario_directional(), figures_dir=tmp_path / "figs")
    assert s["headline"]["label"] == "subceiling_posonly_ahead", s["headline"]
    assert not s["headline"]["dual_read_disagreement"]
    assert not s["kills"]["window_missed"]["fired"]
    assert not any(s["kills"]["retrain_parity"][a]["kill"] for a in (ARM_CONTR, ARM_POS))
    assert s["headline"]["verdict_flip"] is True
    # selection-aware sensitivity surfaced as a diagnostic on the resolvable scenario
    assert s["selection_sensitivity"]["all_m"]["available"] is True
    figs = s.get("figures", {})
    assert len(figs) == 6, figs
    for p in figs.values():
        assert Path(p).exists(), p


def test_scenario_equivalence(tmp_path):
    s = run(tmp_path / "slab", scenario_equivalence())
    assert s["headline"]["label"] == "subceiling_no_separation", s["headline"]


def test_scenario_dual_disagreement_robust_carries(tmp_path):
    s = run(tmp_path / "slab", scenario_dual_disagreement())
    assert s["headline"]["dual_read_disagreement"] is True, s["headline"]
    assert s["headline"]["carried_by"] == "collision_robust"
    assert s["reads"]["collision_robust"]["label"] == "subceiling_no_separation"
    assert s["headline"]["label"] == "subceiling_no_separation"


def test_scenario_s50_fallback_speed_carries(tmp_path):
    s = run(tmp_path / "slab", scenario_s50_fallback())
    assert s["reads"]["all_m"]["label"] == "fallback_only_m_lt_3", s["reads"]["all_m"]
    assert s["headline"]["speed_verdict_carries"] is True
    assert s["install_speed"]["verdict"] == "posonly_first", s["install_speed"]
    assert not s["kills"]["window_missed"]["fired"]
    assert s["headline"]["verdict_flip"] is True


def test_scenario_window_missed_kill(tmp_path):
    s = run(tmp_path / "slab", scenario_window_missed())
    assert s["headline"]["label"] == "window_missed", s["headline"]
    assert s["kills"]["window_missed"]["fired"] is True


def test_scenario_parity_kill_caps_m_lt_3_speed_verdict(tmp_path):
    """Scenario F: the retrain-parity kill must reach the speed verdict in the
    (m<3 fallback AND parity-kill) corner — plan v5 §7 kill 3 bars ANY
    directional claim off a parity-failed retrain."""
    s = run(tmp_path / "slab", scenario_parity_kill_m_lt_3())
    # the corner's preconditions actually hold:
    assert s["kills"]["retrain_parity"][ARM_CONTR]["kill"] is True
    assert s["reads"]["all_m"]["label"] == "fallback_only_m_lt_3", s["reads"]["all_m"]
    assert s["install_speed"]["verdict"] == "posonly_first"  # still fully reported
    assert not s["kills"]["window_missed"]["fired"]
    # the registered cap:
    h = s["headline"]
    assert h["label"] == "subceiling_indeterminate", h
    assert h["label"] not in DETERMINATE_LABELS
    assert h["parity_capped"] is True
    assert h["speed_verdict_carries"] is False
    assert h["speed_verdict"] is None
    assert h["verdict_flip"] is False
