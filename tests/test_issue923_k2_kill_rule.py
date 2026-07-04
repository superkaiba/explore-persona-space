"""#923 pooled round — the §6 k2 kill rule GATES the headline verdict.

r2 Major (Codex, `k2-kill-rule-verdict-not-enforced`): `compute_stats` records
`kill_floor_triggered` (pool_full R² < 0.05 at every layer) but the paired-diff
path emitted a §3 lattice verdict unconditionally, so an uninformative pooled
read could ship as adjudicated. These tests pin the fix through the REAL seams
`main()` dispatches: `kill_floor_flags` → `paired_residual_diff(kill_floor=…)`
→ `headline_payload` (the exact headline.json write shape).

Fails PRE-fix: `paired_residual_diff` had no `kill_floor` parameter
(TypeError), and the headline dict carried a top-level `verdict` whenever
`paired` existed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

fit = pytest.importorskip("issue923_fit_decomposition")

N_BOOT = 16
FAMS = ["persona", "wildchat", "default"]
VERDICT_LABELS = {"H-robust", "H-slot", "intermediate"}


@pytest.fixture()
def parent_dir(tmp_path: Path) -> tuple[Path, dict]:
    """Fake parent fits dir whose headline REPRODUCES from its family sums.

    The delta_r2 value + CI are computed with the module's own
    `_replay_family_counts` + `family_bootstrap`, so the ≤1e-12
    reproduce-check inside `paired_residual_diff` passes by construction.
    """
    rng = np.random.default_rng(0)
    arms: dict = {}
    for arm in ("arm_full", "arm_concat_i"):
        fam_tot = rng.uniform(5.0, 10.0, size=len(FAMS))
        fam_res = fam_tot * rng.uniform(0.2, 0.8, size=len(FAMS))
        arms[arm] = {"fam_res": fam_res.tolist(), "fam_tot": fam_tot.tolist()}
    skill = {"genres": {"uc": {str(fit.HEADLINE_LAYER): {"arms": arms}}}}
    counts = fit._replay_family_counts(N_BOOT, [("uc", len(FAMS), fit.PARENT_QCOLS["uc"])])["uc"]

    def _sums(arm: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(arms[arm]["fam_res"], dtype=np.float64),
            np.asarray(arms[arm]["fam_tot"], dtype=np.float64),
        )

    pf_res, pf_tot = _sums("arm_full")
    pc_res, pc_tot = _sums("arm_concat_i")
    delta = (1.0 - pf_res.sum() / pf_tot.sum()) - (1.0 - pc_res.sum() / pc_tot.sum())
    draws = fit.family_bootstrap(pf_res, pf_tot, counts) - fit.family_bootstrap(
        pc_res, pc_tot, counts
    )
    ci = [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]
    head = {
        "stats": {
            "n_boot": N_BOOT,
            "uc": {"families": FAMS, "delta_r2": {"value": float(delta), "ci95": ci}},
        }
    }
    (tmp_path / "decomposition_skill.json").write_text(json.dumps(skill))
    (tmp_path / "headline.json").write_text(json.dumps(head))
    return tmp_path, arms


def _pooled_fams(arms: dict) -> dict:
    # Self-pair on the parent sums (D identically 0) — the machinery under
    # test is the verdict GATING, not the paired numbers.
    return {
        "uc": {
            arm: (
                np.asarray(arms[arm]["fam_res"], dtype=np.float64),
                np.asarray(arms[arm]["fam_tot"], dtype=np.float64),
            )
            for arm in ("arm_full", "arm_concat_i")
        }
    }


def test_k2_floor_skips_verdict(parent_dir):
    """kill_floor=True → paired numbers persist, NO verdict label anywhere."""
    pdir, arms = parent_dir
    paired = fit.paired_residual_diff(_pooled_fams(arms), pdir, N_BOOT, kill_floor={"uc": True})
    assert "verdict" not in paired, paired.get("verdict")
    assert paired["verdict_skipped_reason"] == "k2_pool_full_floor"
    uc = paired["genres"]["uc"]
    assert uc["verdict"] is None
    assert uc["verdict_skipped_reason"] == "k2_pool_full_floor"
    # Diagnostics are persisted, never dropped (persist-by-default).
    assert uc["paired"] is not None and "D_value" in uc["paired"]
    assert uc["reproduce_check"]["pass"] is True


def test_no_floor_emits_verdict(parent_dir):
    """Control: kill_floor=False keeps the parent behavior (labeled verdict)."""
    pdir, arms = parent_dir
    paired = fit.paired_residual_diff(_pooled_fams(arms), pdir, N_BOOT, kill_floor={"uc": False})
    assert paired["verdict"] in VERDICT_LABELS
    assert paired["genres"]["uc"]["verdict"]["label"] in VERDICT_LABELS
    assert "verdict_skipped_reason" not in paired


def test_headline_payload_omits_top_level_verdict_on_k2(parent_dir):
    """Codex's mechanizable fixture: all-floor stats → headline has NO verdict.

    Runs the exact seam chain `main()` uses: stats flags → kill_floor_flags →
    paired_residual_diff → headline_payload (the headline.json write shape).
    """
    pdir, arms = parent_dir
    stats = {"headline_layer": 18, "n_boot": N_BOOT, "uc": {"kill_floor_triggered": True}}
    flags = fit.kill_floor_flags(stats, ["uc"])
    assert flags == {"uc": True}
    paired = fit.paired_residual_diff(_pooled_fams(arms), pdir, N_BOOT, kill_floor=flags)
    payload = fit.headline_payload({"script": "test"}, stats, paired)
    assert "verdict" not in payload
    assert payload["verdict_skipped_reason"] == "k2_pool_full_floor"
    assert "paired_diff" in payload  # diagnostics still shipped


def test_headline_payload_keeps_verdict_without_floor(parent_dir):
    """Control through the same seams: no floor → top-level verdict retained."""
    pdir, arms = parent_dir
    stats = {"headline_layer": 18, "n_boot": N_BOOT, "uc": {"kill_floor_triggered": False}}
    flags = fit.kill_floor_flags(stats, ["uc"])
    paired = fit.paired_residual_diff(_pooled_fams(arms), pdir, N_BOOT, kill_floor=flags)
    payload = fit.headline_payload({"script": "test"}, stats, paired)
    assert payload["verdict"] in VERDICT_LABELS
    assert "verdict_skipped_reason" not in payload


def test_headline_payload_last_path_unchanged():
    """Byte-compat guard: paired=None (feature-source=last) → {meta, stats} only."""
    payload = fit.headline_payload({"script": "test"}, {"n_boot": N_BOOT}, None)
    assert set(payload) == {"meta", "stats"}
