"""Offvm-battery-refit round regressions (issue #1092, deferred_refit_spec.json).

Pins, all executing REAL bodies (no mocks; the filesystem is the only boundary):
  1. `_fit_arm_indices` excludes the #594 battery block (stratum "battery" /
     is_eval_only=True) from BOTH fit arms — the banked engine's fit-arm-A
     filter excluded the nonexistent label "battery_eval_only" (a no-op) and
     fit arm B had no battery filter at all, so battery rows leaked into
     TRAINING in both banked arms. Fails pre-fix.
  2. `_per_target_r2` banks the registered t1/t2/t3 held-out R² columns from
     the pooled CV predictions (ambient basis; explicit skip note for pca48).
  3. A tiny-real end-to-end engine run on a manifest CARRYING battery rows
     banks the corrected n_rows per fit arm + the per-target columns.
  4. Part-B operator comparison: the factorized Procrustes nuclear-norm
     identity matches the direct computation, and a tiny-real end-to-end run
     produces the registered schema (matched lambda, principal-angle nulls,
     Procrustes band, topic_matched_pairing_delta drop record) and resumes
     idempotently.
  5. The GCE driver parses P6_PARTB_JOBS specs and job-suffixes Part-B
     summaries (dry-run, real driver bytes).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))  # sibling fixture writers below

import issue1092_fit_grid as fit_grid  # noqa: E402
import issue1092_partb_operator as partb  # noqa: E402
from test_issue1092_p6_wrapper import _write_rb, _write_store  # noqa: E402

HIDDEN = 8
N_DENSE = 7
N_TRAIT = 2
N_BATTERY = 3
N_ROWS = N_DENSE + N_TRAIT + N_BATTERY  # matches _write_store's 12-row arrays


def _battery_manifest_rows() -> list[dict]:
    """12 rows mirroring the production tail layout: dense, trait, then battery."""
    rows = []
    for i in range(N_DENSE):
        rows.append(
            {
                "row_id": f"r{i}",
                "prefix_id": f"p{i % 3}",
                "query_id": f"q{i % 4}",
                "stratum": "dense_core",
                "is_eval_only": False,
            }
        )
    for i in range(N_TRAIT):
        rows.append(
            {
                "row_id": f"t{i}",
                "prefix_id": f"tp{i}",
                "query_id": f"q{i % 4}",
                "stratum": "trait_stratum",
                "is_eval_only": False,
            }
        )
    for i in range(N_BATTERY):
        rows.append(
            {
                "row_id": f"b{i}",
                "prefix_id": f"bp{i}",
                "query_id": f"q{i % 4}",
                "stratum": "battery",
                "is_eval_only": True,
            }
        )
    return rows


def _write_battery_corpus(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in _battery_manifest_rows())
    )
    return root


def _store_with_all_targets(base: Path) -> Path:
    """The shared tiny-real store, extended with t2/t3 target summaries."""
    _write_store(base, cells=("cell_inst_own",), layers=(14,))
    rng = np.random.default_rng(3)
    for t in ("t2", "t3"):
        np.save(
            base / "cell_inst_own" / f"{t}_L14.npy",
            rng.normal(size=(N_ROWS, HIDDEN)).astype(np.float32),
        )
    return base


# ── 1. fit-arm row selection ────────────────────────────────────────────────


def test_fit_arm_A_excludes_battery_and_trait():
    rows = _battery_manifest_rows()
    idx = fit_grid._fit_arm_indices("A", rows)
    assert len(idx) == N_DENSE
    assert all(rows[i]["stratum"] == "dense_core" for i in idx)


def test_fit_arm_B_excludes_battery_keeps_trait():
    rows = _battery_manifest_rows()
    idx = fit_grid._fit_arm_indices("B", rows)
    assert len(idx) == N_DENSE + N_TRAIT
    assert all(not rows[i]["is_eval_only"] for i in idx)
    assert any(rows[i]["stratum"] == "trait_stratum" for i in idx)


def test_fit_arm_battery_marked_by_either_key():
    """Either marker alone (stratum OR is_eval_only) excludes the row from both arms."""
    rows = [
        {"stratum": "dense_core"},
        {"stratum": "battery"},  # stratum only
        {"stratum": "dense_core", "is_eval_only": True},  # flag only
    ]
    assert fit_grid._fit_arm_indices("A", rows) == [0]
    assert fit_grid._fit_arm_indices("B", rows) == [0]


def test_fit_arm_unknown_raises():
    with pytest.raises(ValueError, match="unknown fit arm"):
        fit_grid._fit_arm_indices("C", [])


# ── 2. per-target R² columns ────────────────────────────────────────────────


def test_per_target_r2_blocks_and_skip():
    rng = np.random.default_rng(0)
    hidden, targets = 2, ["t1", "t2", "t3"]
    n = 8
    Y = rng.normal(size=(n, hidden * len(targets)))
    pred = Y.copy()
    pred[:, 2:4] = Y[:, 2:4].mean(axis=0, keepdims=True)  # t2 block: R² == 0
    pred[:, 4:6] = rng.normal(size=(n, 2))  # t3 block: R² < 1
    folds = [np.arange(0, 4), np.arange(4, 8)]
    info = {"basis": "ambient", "targets": targets, "hidden_dim": hidden}
    out = fit_grid._per_target_r2(Y, pred, folds, info)
    assert out["t1"]["r2"] == pytest.approx(1.0)
    assert out["t2"]["r2"] == pytest.approx(0.0, abs=1e-12)
    assert out["t3"]["r2"] < 1.0
    assert len(out["t1"]["r2_folds"]) == 2
    skipped = fit_grid._per_target_r2(Y, pred, folds, {**info, "basis": "pca48"})
    assert "skipped" in skipped and "pca48" in skipped["skipped"]


# ── 3. tiny-real end-to-end engine run with battery rows ────────────────────


def test_engine_tiny_real_banks_corrected_rows_and_per_target(tmp_path, monkeypatch):
    """Fails pre-fix: banked fit-arm A kept battery rows (n=10 here), B kept all 12."""
    summaries = _store_with_all_targets(tmp_path / "summaries")
    corpus = _write_battery_corpus(tmp_path / "corpus")
    rb = _write_rb(tmp_path / "rb")
    out = tmp_path / "out"
    argv = [
        "issue1092_fit_grid.py",
        "--summaries-dir",
        str(summaries),
        "--corpus-dir",
        str(corpus),
        "--out-dir",
        str(out),
        "--cells",
        "cell_inst_own",
        "--layers",
        "14",
        "--targets",
        "t1,t2,t3",
        "--target-bases",
        "ambient,pca48",
        "--n-null-draws",
        "2",
        "--band-null-draws",
        "2",
        "--matched-n-draws",
        "2",
        "--n-folds",
        "2",
        "--hidden-dim",
        str(HIDDEN),
        "--skip-mlp-companion",
        "--rb-dir",
        str(rb),
        "--allow-missing-registered-reads",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    summary = fit_grid.run(fit_grid.parse_args())
    units = summary["units"]
    assert len(units) == 8  # 2 arms x 2 fit arms x 2 bases x 1 layer
    for unit in units:
        expected = N_DENSE if unit["fit_arm"] == "A" else N_DENSE + N_TRAIT
        assert unit["n_rows"] == expected, (unit["fit_arm"], unit["n_rows"])
        per_target = unit["fit_per_target_r2"]
        if unit["basis"] == "ambient":
            assert set(per_target) >= {"t1", "t2", "t3"}
            for t in ("t1", "t2", "t3"):
                assert isinstance(per_target[t]["r2"], float)
                assert per_target[t]["r2_folds"]
        else:
            assert "skipped" in per_target


# ── 4. Part-B operator comparison ───────────────────────────────────────────


def test_partb_procrustes_factorized_matches_direct():
    """nuc(Wc Wpᵀ) via the (r,r) factorized identity == direct SVD; residual too."""
    rng = np.random.default_rng(11)
    P, d = 10, 6
    Wc = torch.from_numpy(rng.normal(size=(P, d))).double()
    Wp = torch.from_numpy(rng.normal(size=(P, d))).double()
    _, s_c, Qh_c = torch.linalg.svd(Wc, full_matrices=False)
    _, s_p, Qh_p = torch.linalg.svd(Wp, full_matrices=False)
    obs = partb._observed_procrustes(s_c, Qh_c, s_p, Qh_p)
    m_direct = Wc @ Wp.T
    nuc_direct = float(torch.linalg.svdvals(m_direct).sum())
    assert obs["nuclear_cross"] == pytest.approx(nuc_direct, rel=1e-10)
    u, _s, vh = torch.linalg.svd(m_direct, full_matrices=False)
    r_opt = u @ vh
    res_direct = float(torch.linalg.norm(Wc - r_opt @ Wp) / torch.linalg.norm(Wc))
    assert obs["residual"] == pytest.approx(res_direct, rel=1e-8)
    # self-comparison sanity: residual(W, W) == 0 (sqrt amplifies fp cancellation,
    # so the bar is 1e-6 relative to O(1) residual scale, not machine epsilon)
    self_obs = partb._observed_procrustes(s_c, Qh_c, s_c, Qh_c)
    assert self_obs["residual"] == pytest.approx(0.0, abs=1e-6)


def test_partb_end_to_end_tiny_real(tmp_path):
    summaries = _store_with_all_targets(tmp_path / "summaries")
    corpus = _write_battery_corpus(tmp_path / "corpus")
    out = tmp_path / "out"
    args = partb.parse_args(
        [
            "--summaries-dir",
            str(summaries),
            "--corpus-dir",
            str(corpus),
            "--out-dir",
            str(out),
            "--cells",
            "cell_inst_own",
            "--layers",
            "14",
            "--target-bases",
            "ambient,pca48",
            "--targets",
            "t1,t2,t3",
            "--hidden-dim",
            str(HIDDEN),
            "--n-null-draws",
            "4",
            "--null-chunk",
            "2",
        ]
    )
    summary = partb.run(args)
    assert summary["n_units"] == 2
    drop = summary["topic_matched_pairing_delta"]
    assert drop["status"] == "dropped" and "superseded plan revision" in drop["reason"]
    unit_files = sorted((out / "partb").glob("cell_inst_own_L14_*.json"))
    assert len(unit_files) == 2
    for f in unit_files:
        unit = json.loads(f.read_text())
        assert unit["n_rows_battery_excluded"] == N_DENSE
        assert 0 <= unit["lambda"]["matched_idx"] < len(unit["lambda"]["grid"])
        assert (
            unit["lambda"]["matched_lambda"]
            == unit["lambda"]["grid"][unit["lambda"]["matched_idx"]]
        )
        for read in unit["principal_angles"].values():
            assert all(-1e-9 <= a <= np.pi / 2 + 1e-9 for a in read["angles_rad"])
        assert unit["procrustes"]["null"]["n_draws"] == 4
        assert unit["procrustes"]["residual"] >= 0.0
        assert "data-spanned row space" in unit["row_space_note"]
    # resume predicate: a re-run skips every completed unit (no new files)
    before = {p.name for p in (out / "partb").glob("*.json")}
    partb.run(args)
    assert {p.name for p in (out / "partb").glob("*.json")} == before


# ── 5. GCE driver Part-B phase (dry-run, real driver bytes) ─────────────────


def test_p6_gce_driver_partb_dry_run(tmp_path):
    env = {
        **os.environ,
        "P6_DRY_RUN": "1",
        "P6_BOX_ID": "rf01",
        "P6_STAGE_DIR": str(tmp_path / "stage"),
        "P6_JOBS": (
            "cells=cell_inst_own|layers=14,18,19|fit_arms=A|bases=ambient,pca48"
            "|pilot_cell=cell_inst_own|pilot_layer=14|plan_wall_h=5"
            "|extra=--skip-mlp-companion"
        ),
        "P6_PARTB_JOBS": (
            "cells=cell_inst_own,cell_inst_claude|layers=14,18,19|bases=ambient,pca48"
            ";;cells=cell_pre_own|layers=14|bases=ambient"
        ),
    }
    proc = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "issue1092_p6_gce.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "[phase=p6_partb_job1]" in proc.stdout
    assert "[phase=p6_partb_job2]" in proc.stdout
    assert "--cells cell_inst_own,cell_inst_claude" in proc.stdout
    assert "issue1092_partb_operator.py" in proc.stdout
    assert "[phase=done]" in proc.stdout
    partb_dir = tmp_path / "data/issue_1092/p6/partb"
    assert (partb_dir / "partb_summary_pjob1.json").exists()
    assert (partb_dir / "partb_summary_pjob2.json").exists()
    assert not (partb_dir / "partb_summary.json").exists()
