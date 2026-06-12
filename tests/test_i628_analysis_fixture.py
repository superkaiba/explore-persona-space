"""End-to-end fixture smoke for scripts/i628_analysis.py (plan §6 statistics).

Synthesizes a full N-row fixture — the Legacy arm + three mini-arms' G-cells
(dual-slot on separator arms), the reused #537 grid, the reuse-arm negative
columns, and the Phase-4 on-policy reads — with a planted effect (legacy
bystander leakage +1.5 nat above revised) and the inherited #537 masks
(fmt_code seed-42 diagonal below the 4-nat gate; binst_marker diagonal
saturated both seeds), then runs the analysis main() and asserts the
registered outputs: H2 n=15 after the seed-42 pairwise deletion, the 80-cell
DV1↔DV3 matched-key count, the H-inert clustering, and the claim routing.

Requires the prefetched #537 contexts under ``data/issue_537/contexts``
(pinned-revision inputs; skipped when absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

CONTEXTS = REPO / "data/issue_537/contexts/sampled_contexts.json"
pytestmark = pytest.mark.skipif(
    not CONTEXTS.exists(), reason="needs the prefetched #537 contexts (phase-0 inputs)"
)

SEEDS = (42, 1042)
MINI = ("sp_swe", "wc_short_advice", "icl_k8", "binst_marker")


def _mk_cell(arm, sep_mode, t, e, seed, delta, rng):
    eps = float(rng.normal(0, 0.05))
    return {
        "behavior": "marker",
        "arm": arm,
        "sep_mode": sep_mode,
        "train_cid": t,
        "eval_cid": e,
        "seed": seed,
        "n_questions": 32,
        "g_mean_delta_logp": delta + eps,
        "g_mean_delta_z_marker": delta + eps + 0.3,
        "g_mean_delta_eos_margin": delta + eps + 0.5,
        "emission_rate_trained": 0.0,
        "emission_rate_base": 0.0,
        "git_commit": "fixture",
    }


def _delta(arm, t, e, seed):
    """Planted structure: diagonals ~8 nat (fmt_code seed-42 fails the gate at
    3.5; binst diagonal saturated at 25); legacy bystanders +2.5, revised
    family +1.0 (a +1.5-nat legacy excess); negative columns suppressed under
    live-negative arms."""
    if t == e:
        if t == "binst_marker":
            return 25.0
        if t == "fmt_code" and seed == 42:
            return 3.5
        return 8.0
    if e.startswith("neg_"):
        # Live-negative arms push trained negatives BELOW bystanders; dead-
        # negative arms leave them ABOVE (no suppression signal).
        return (
            -1.0 if arm in ("rig_Nplus_canonical", "rig_F_sep_liveneg", "rig_N_i537_reuse") else 3.5
        )
    return 2.5 if arm in ("rig_O_sep_deadneg", "rig_S_nosep_deadneg") else 1.0


def _write(d: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def _write_arm_cells(root, arm, train_cids, cols, rng, *, dual: bool) -> None:
    """Fresh-arm G-cells (dual=plain-slot duplicates, slightly attenuated)."""
    for t in train_cids:
        for s in SEEDS:
            for e in cols:
                d = _delta(arm, t, e, s)
                _write(
                    _mk_cell(arm, "marker", t, e, s, d, rng),
                    root / f"G_cells/{arm}/{t}__{e}__seed{s}.json",
                )
                if dual:
                    _write(
                        _mk_cell(arm, "plain", t, e, s, d * 0.6, rng),
                        root / f"G_cells/{arm}/{t}__{e}__seed{s}__plain.json",
                    )


def _write_reuse(root, snap, train_cids, grid, negs, rng) -> None:
    """Reused #537 grid (snapshot dir, no arm/sep fields) + Phase-3 neg cols."""
    for t in train_cids:
        for s in SEEDS:
            for e in grid:
                cell = _mk_cell(
                    "rig_N_i537_reuse", "marker", t, e, s, _delta("rig_N_i537_reuse", t, e, s), rng
                )
                cell.pop("arm")  # snapshot files predate the arm field
                cell.pop("sep_mode")
                _write(cell, snap / f"{t}__{e}__seed{s}.json")
            for e in negs:
                _write(
                    _mk_cell(
                        "rig_N_i537_reuse",
                        "marker",
                        t,
                        e,
                        s,
                        _delta("rig_N_i537_reuse", t, e, s),
                        rng,
                    ),
                    root / f"neg_columns/rig_N_i537_reuse/{t}__{e}__seed{s}.json",
                )


def _write_onpolicy(root, negs, rng) -> None:
    """Phase-4 on-policy reads: DV3 tracks DV1 (rho ~1 by construction)."""
    for arm in ("rig_O_sep_deadneg", "rig_Nplus_canonical"):
        for t in MINI:
            for s in SEEDS:
                summary = {}
                for e in ["default", *negs, t]:
                    summary[e] = {
                        "mean_delta_logp": _delta(arm, t, e, s) * 0.9 + float(rng.normal(0, 0.02)),
                        "mean_delta_eos_margin": 0.0,
                        "emission_rate": 0.0,
                    }
                _write(
                    {"arm": arm, "cid": t, "seed": s, "summary": summary, "rows": []},
                    root / f"bystander_onpolicy/{arm}_{t}_seed{s}/reads.json",
                )


@pytest.fixture(scope="module")
def fixture_roots(tmp_path_factory):
    import i628_analysis as a

    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("i628_eval")
    snap = tmp_path_factory.mktemp("i537_snapshot")
    figs = tmp_path_factory.mktemp("figs")
    train_cids = a._train_cids()
    grid = a._grid_eval_cids()
    negs = list(a._negative_cids())
    cols34 = [*grid, *negs]

    _write_arm_cells(root, "rig_O_sep_deadneg", train_cids, cols34, rng, dual=True)
    _write_arm_cells(root, "rig_Nplus_canonical", MINI, cols34, rng, dual=False)
    _write_arm_cells(root, "rig_S_nosep_deadneg", MINI, cols34, rng, dual=False)
    _write_arm_cells(root, "rig_F_sep_liveneg", MINI, cols34, rng, dual=True)
    _write_reuse(root, snap, train_cids, grid, negs, rng)
    _write_onpolicy(root, negs, rng)
    return root, snap, figs


def test_analysis_end_to_end(fixture_roots, monkeypatch):
    import i628_analysis as a

    root, snap, figs = fixture_roots
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "i628_analysis.py",
            "--eval-root",
            str(root),
            "--reuse-cells-dir",
            str(snap),
            "--figures-dir",
            str(figs),
        ],
    )
    assert a.main() == 0
    out = json.loads((root / "analysis/rig_contrast.json").read_text())

    # Registered H2: fmt_code seed-42 pairwise-deleted (n=15); seed-1042 keeps 16.
    assert out["h2_primary"]["per_seed"]["42"]["n"] == 15
    assert out["h2_primary"]["per_seed"]["1042"]["n"] == 16
    assert ["fmt_code", 42] in out["h2_primary"]["pairwise_deleted_rows"]
    # Planted +1.5-nat legacy excess → PASS.
    assert out["h2_primary"]["verdict"] == "PASS"
    assert out["h2_primary"]["pooled_seed_stratified_bootstrap"]["mean"] == pytest.approx(
        1.5, abs=0.2
    )
    # Plain-slot sensitivity artifact exists.
    assert (root / "analysis/h2_plain_slot_sensitivity.json").exists()

    # DV1↔DV3 matched-key enumeration: exactly 80 cells, diagonals separate.
    dv13 = out["dv1_dv3_validation"]
    assert dv13["expected_n"] == 80 and dv13["realized_n"] == 80
    assert dv13["pass"] is True
    assert len(dv13["diagonals_reported_separately"]) == 16

    # H-inert clusters at the adapter level (8 pairs).
    assert out["h_inert"]["n_adapter_pairs"] == 8
    assert out["h_inert"]["sign_test_diagnostic_only"]["n"] == 8 * 30

    # binst diagonal censored from the dial means (symmetric).
    assert out["h1_install_parity"]["diag_censor_cids"] == ["binst_marker"]
    # 25-nat diagonal excluded → censored dial mean is far below 9.
    assert out["h1_install_parity"]["arm_mean_diagonal_censored"]["rig_O_sep_deadneg"] < 9.0

    # Trained-negative signature: suppressed under live-negative arms only.
    tns = out["trained_negative_signature"]
    assert tns["rig_N_i537_reuse"]["frac_below_holdout"] > 0.9
    assert tns["rig_O_sep_deadneg"]["frac_below_holdout"] < 0.1

    # Claim routing fields present.
    assert "final_claim_scope" in out
    assert isinstance(out["matched_install_reread_required"], bool)

    # Figures landed (hero at minimum).
    assert (figs / "hero_paired_offdiagonal_leakage.png").exists()
