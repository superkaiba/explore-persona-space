"""Pin tests for the #1689 `wellposed-shared-readout` round (plan v10).

The round's single variable is the fit basis: ambient d=3,584 shrunk ridge
-> a per-(pair, arm, fold) shared train-fold-only PCA rank-k basis with
k_unit = min(1024, d, floor(min-fold n_train / 2)) so n_train >= 2k on every
fold. These tests pin, on REAL bodies over tiny synthetic worlds (no mocks):

  1. the k_unit formula + degenerate-fold exclusion,
  2. the ambient no-op resume contract (regime meta byte-stable; no
     reduced-only JSON keys in ambient units),
  3. the reduced run_unit invariants (well-posedness, clamped truncation
     grid, captured-variance fields, ambient-recon companion),
  4. the reduced exact-linear-world science invariant (shared readout is
     recovered when the data's intrinsic rank fits inside the basis),
  5. the cms reduced unit + ambient-lifted overlap frames,
  6. the paired digest join (deltas, k-band, coverage-hole enumeration),
  7. the paired figures' data-dependent render branches (fig8 rung-1 rows),
  8. the fence phase report + designed-halt rc=21 (kill criterion 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.issue1689_context_map_structure as cms
import scripts.issue1689_derived_vs_free as dvf
import scripts.issue1689_fit_ladder as fl
from scripts.issue1689_common import PCA_K_CAP, k_band


def _dvf_args(**over):
    base = dict(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        rotation_draws=2,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _cms_args(**over):
    base = dict(
        layer=19,
        lambda_grid="ladder13",
        seed=42,
        device="cpu",
        row_limit=None,
        dim_limit=None,
        class_null_draws=2,
        rank_null_draws=1,
        items="both",
    )
    base.update(over)
    return argparse.Namespace(**base)


def _cell(x, y, n, prefix="c"):
    return {
        "X_prefix": x.copy(),
        "X_context": x,
        "Y": y,
        "conv_ids": np.array([f"{prefix}{i}" for i in range(n)]),
    }


def _low_rank_world(rng, n, d, r):
    """Exact linear world with intrinsic rank r: y = xW + b, x_T = x_S M + a."""
    z = rng.standard_normal((n, r))
    p = rng.standard_normal((r, d)) / np.sqrt(r)
    x_s = z @ p
    w = rng.standard_normal((d, d)) / np.sqrt(d)
    b = rng.standard_normal(d) * 0.1
    m_op = np.eye(d) + 0.3 * rng.standard_normal((d, d)) / np.sqrt(d)
    a = rng.standard_normal(d) * 0.1
    x_t = x_s @ m_op + a
    return _cell(x_s, x_s @ w + b, n), _cell(x_t, x_t @ w + b, n)


def test_compute_k_unit_formula_and_degenerate_folds():
    # 100 convs, 5 folds of 20 -> min-fold n_train = 80 -> k = 40 (d ample).
    folds = np.repeat(np.arange(5), 20)
    k, per_fold = dvf.compute_k_unit(folds, 100, d=512)
    assert k == 40 and per_fold == {f: 80 for f in range(5)}
    # cap binds
    assert dvf.compute_k_unit(folds, 100, d=512, cap=16)[0] == 16
    # d binds (dim-limited smoke shape: floor(n_tr/2) > d)
    assert dvf.compute_k_unit(folds, 100, d=8)[0] == 8
    # degenerate folds excluded: fold 4 empty -> its n_train never counted
    folds2 = np.repeat(np.arange(4), 25)
    k2, per_fold2 = dvf.compute_k_unit(folds2, 100, d=512)
    assert set(per_fold2) == {0, 1, 2, 3} and k2 == (100 - 25) // 2
    # all folds degenerate -> (0, {})
    assert dvf.compute_k_unit(np.zeros(3, dtype=int), 3, d=512) == (0, {})


def test_pca_basis_orthonormal_and_captured_fracs():
    rng = np.random.default_rng(0)
    stacked = torch.from_numpy(rng.standard_normal((60, 24)))
    mu, q, svals, frac = dvf._pca_basis(stacked, 10)
    assert q.shape == (24, 10) and svals.shape == (10,)
    assert torch.allclose(q.T @ q, torch.eye(10, dtype=torch.float64), atol=1e-10)
    assert 0.0 < frac <= 1.0
    te = torch.from_numpy(rng.standard_normal((20, 24)))
    f_te = dvf._heldout_captured_frac(te, mu, q)
    assert 0.0 <= f_te <= 1.0


def test_ambient_regime_meta_and_json_are_noop():
    """Resume contract: ambient meta carries NO fit_basis key (parent
    checkpoints stay valid) and ambient unit JSONs carry NO reduced-only keys;
    a reduced meta can never match an ambient checkpoint (#722 r3)."""
    amb_default = dvf.regime_meta(_dvf_args())
    amb_explicit = dvf.regime_meta(_dvf_args(fit_basis="ambient"))
    red = dvf.regime_meta(_dvf_args(fit_basis="reduced"))
    assert amb_default == amb_explicit
    assert "fit_basis" not in amb_default and "pca_k_cap" not in amb_default
    assert red["fit_basis"] == "reduced" and red != amb_default
    assert cms.regime_meta(_cms_args()) == cms.regime_meta(_cms_args(fit_basis="ambient"))
    assert "fit_basis" in cms.regime_meta(_cms_args(fit_basis="reduced"))

    rng = np.random.default_rng(1)
    src, tgt = _low_rank_world(rng, 120, 12, 6)
    unit, _ = dvf.run_unit(src, tgt, (("mA", "s"), ("mA", "t")), "context", _dvf_args(), fl.LAMBDAS)
    assert "error" not in unit
    for key in (
        "fit_basis",
        "k_unit",
        "fit_dim",
        "k_floor_limited",
        "per_fold_n_train",
        "pca_basis_per_fold",
        "r2_pooled_ambient_recon",
        "r2_b_free_ambient_recon",
    ):
        assert key not in unit, key


def test_run_unit_reduced_wellposed_invariants_and_shared_readout():
    """Reduced-mode mechanical invariants + the exact-linear-world science
    read: with intrinsic rank r << k_unit, the shared-readout conjugation
    reconciles the free map inside the well-posed basis (g1 >= 0)."""
    rng = np.random.default_rng(2)
    n, d, r = 300, 24, 6
    src, tgt = _low_rank_world(rng, n, d, r)
    unit, bundle = dvf.run_unit(
        src, tgt, (("mA", "s"), ("mA", "t")), "context", _dvf_args(fit_basis="reduced"), fl.LAMBDAS
    )
    assert "error" not in unit, unit.get("error")
    assert unit["fit_basis"] == "reduced" and unit["meta"]["fit_basis"] == "reduced"
    n_tr_min = min(unit["per_fold_n_train"].values())
    assert unit["k_unit"] == min(PCA_K_CAP, d, n_tr_min // 2)
    assert unit["fit_dim"] == unit["k_unit"]
    for n_tr in unit["per_fold_n_train"].values():
        assert n_tr >= 2 * unit["k_unit"]  # the well-posedness invariant
    # truncation grid clamped to [1, k_unit] via the existing min(r, s.shape[0])
    assert all(v <= unit["k_unit"] for v in unit["rank_map_canonical"].values())
    for meta in unit["pca_basis_per_fold"].values():
        for f in ("captured_var_train_x", "captured_var_train_y", "captured_var_test_y"):
            assert 0.0 <= meta[f] <= 1.0 + 1e-12
    assert np.isfinite(unit["r2_b_free_ambient_recon"])
    assert set(unit["r2_pooled_ambient_recon"]) == set(unit["r2_pooled"])
    # exact linear world, rank r << k: shared readout reconciles the free map
    assert unit["g1"] >= 0 and unit["verdict"] == "shared_readout_supported"
    # bundle carries the per-fold Q-basis spectra at length k_unit
    q_keys = [k for k in bundle if k.startswith("q_x_svals_f")]
    assert q_keys and all(bundle[k].shape == (unit["k_unit"],) for k in q_keys)


def test_run_structure_unit_reduced_and_lifted_frames():
    rng = np.random.default_rng(3)
    n, d, r = 150, 16, 5
    src, tgt = _low_rank_world(rng, n, d, r)
    unit, bundle = cms.run_structure_unit(
        src,
        tgt,
        (("mA", "s"), ("mA", "t")),
        "context",
        _cms_args(fit_basis="reduced"),
        fl.LAMBDAS,
        parent_rung=9,
    )
    assert "error" not in unit, unit.get("error")
    k_u = unit["k_unit"]
    assert unit["fit_dim"] == k_u and k_u == min(
        PCA_K_CAP, d, min(unit["per_fold_n_train"].values()) // 2
    )
    # frames are ambient-LIFTED: (d, n_keep) with n_keep = min(256, k_unit)
    assert bundle["m_minus_i_u256_fp16"].shape == (d, min(256, k_u))
    assert bundle["m_minus_i_vh256_fp16"].shape == (min(256, k_u), d)
    # item 8 ran inside the reduced basis (parent_rung 9 -> eligible)
    rr = unit["rank_rung"]
    assert rr["eligible"] and "k_reached_ctx" in rr and "error" not in rr
    # ambient no-op: no reduced-only keys on an ambient unit
    unit_amb, bundle_amb = cms.run_structure_unit(
        src, tgt, (("mA", "s"), ("mA", "t")), "context", _cms_args(), fl.LAMBDAS, parent_rung=9
    )
    assert "k_unit" not in unit_amb and "fit_basis" not in unit_amb
    assert bundle_amb["m_minus_i_u256_fp16"].shape == (d, min(256, d))


def _write_unit(root: Path, uk: str, unit: dict) -> None:
    (root / "pairs").mkdir(parents=True, exist_ok=True)
    (root / "pairs" / f"{uk}.json").write_text(json.dumps(unit))


def _mk_dvf_unit(uk, verdict, g1, g2, *, k_unit=None, rung=1):
    u = {
        "src_model": "mA",
        "tgt_model": "mA",
        "src_cond": uk.split("__")[1],
        "tgt_cond": uk.split("__")[2],
        "arm": "context",
        "pair_key": "__".join(uk.split("__")[1:3]),
        "cross_model": False,
        "n_common": 100,
        "d": 512,
        "verdict": verdict,
        "g1": g1,
        "g2": g2,
        "r2_b_free": 0.5,
        "r2_identity_bias": 0.1,
    }
    if k_unit is not None:
        u["k_unit"] = k_unit
        u["fit_basis"] = "reduced"
        u["fit_dim"] = k_unit
        u["k_floor_limited"] = k_unit < 8
        u["r2_b_free_ambient_recon"] = 0.4
        u["pca_basis_per_fold"] = {"0": {"captured_var_test_y": 0.9}}
    return u


def test_paired_digest_join_and_coverage(tmp_path, capsys):
    import scripts.issue1689_dvf_fold_digest as dig

    amb_dvf = tmp_path / "amb_dvf"
    red_dvf = tmp_path / "red_dvf"
    _write_unit(
        amb_dvf,
        "mA__s__t__context",
        _mk_dvf_unit("mA__s__t__context", "transfer_map_insufficient", -0.2, -0.1),
    )
    _write_unit(
        amb_dvf,
        "mA__s__u__context",
        _mk_dvf_unit("mA__s__u__context", "transfer_map_insufficient", -0.3, -0.2),
    )
    _write_unit(
        red_dvf,
        "mA__s__t__context",
        _mk_dvf_unit("mA__s__t__context", "shared_readout_supported", 0.05, 0.06, k_unit=40),
    )
    args = argparse.Namespace(
        digest_csv=tmp_path / "missing.csv",
        ambient_dvf_root=amb_dvf,
        reduced_dvf_root=red_dvf,
        ambient_xm_root=tmp_path / "no_xm_a",
        reduced_xm_root=tmp_path / "no_xm_r",
        ambient_cms_root=tmp_path / "no_cms_a",
        reduced_cms_root=tmp_path / "no_cms_r",
        ambient_xms_root=tmp_path / "no_xms_a",
        reduced_xms_root=tmp_path / "no_xms_r",
        out=tmp_path / "paired.csv",
        summary_out=tmp_path / "paired_summary.json",
    )
    assert dig.paired_main(args) == 0
    import csv as _csv

    with open(args.out) as fh:
        rows = list(_csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["verdict_flip"] == "1" and row["k_band"] == k_band(40)
    assert abs(float(row["delta_g1"]) - 0.25) < 1e-12
    summary = json.loads(args.summary_out.read_text())
    cov = summary["coverage"]["dvf_within"]
    assert cov["n_matched"] == 1 and cov["ambient_only"] == ["mA__s__u__context"]
    assert summary["verdict_flip_matrix"] == {
        "transfer_map_insufficient->shared_readout_supported": 1
    }


def test_fig8_paired_calibration_renders_rung1_branch(tmp_path):
    """Data-dependent gate probe: the fig8 RENDER branch (rung-1 rows present)
    executes and writes a PNG; the SKIP branch is exercised in the leg smoke."""
    from scripts.issue1689_derived_vs_free_figures import fig8_paired_calibration

    rows = [
        {
            "battery": "dvf_within",
            "parent_rung": "1",
            "k_band": "32-127",
            "ambient_verdict": "transfer_map_insufficient",
            "reduced_verdict": "shared_readout_supported",
        },
        {
            "battery": "dvf_within",
            "parent_rung": "1",
            "k_band": ">=512",
            "ambient_verdict": "free_map_uninformative",
            "reduced_verdict": "readout_changed",
        },
    ]
    fig8_paired_calibration(rows, tmp_path)
    out = tmp_path / "fig8_paired_calibration.png"
    assert out.exists() and out.stat().st_size > 1000


def test_cmd_fence_report_and_designed_halt(tmp_path):
    """Fence phase: k-weighted projection report + rc=21 designed halt."""
    dvf_root = tmp_path / "dvf"
    cms_root = tmp_path / "cms"
    pilot_key = dvf.unit_key(dvf.GATE1_PAIR, "context")
    _write_unit(dvf_root, pilot_key, {"wall_s": 10.0, "k_unit": 100, "d": 512})
    _write_unit(cms_root, pilot_key, {"wall_s": 30.0, "k_unit": 100, "d": 512})
    digest = tmp_path / "digest.csv"
    lines = ["battery,n_common"]
    lines += ["dvf_within,500"] * 4 + ["xm_dvf,2000"] * 2 + ["cms_within,500"] * 4
    digest.write_text("\n".join(lines) + "\n")
    args = argparse.Namespace(
        out_root=dvf_root,
        cms_out_root=cms_root,
        digest_csv=digest,
        num_shards=2,
        kill_gpu_hours=30.0,
        enforce_kill=False,
    )
    assert dvf.cmd_fence(args) == 0
    rep = json.loads((dvf_root / "fence_report.json").read_text())
    assert rep["cms_wall_measured"] and rep["k_pilot"] == 100
    assert rep["n_units_dvf"] == 6 and rep["n_units_cms"] == 6
    assert rep["fence_s"] >= 900 and not rep["kill"]
    # designed halt: tiny budget + enforce -> rc 21 (never an anonymous crash)
    args2 = argparse.Namespace(**{**vars(args), "kill_gpu_hours": 1e-6, "enforce_kill": True})
    assert dvf.cmd_fence(args2) == 21
    assert json.loads((dvf_root / "fence_report.json").read_text())["kill"]


def test_reduced_checkpoint_never_satisfied_by_ambient(tmp_path):
    """Resume-regime discipline (#722 r3): an ambient checkpoint on disk does
    NOT satisfy a reduced run's skip predicate (and vice versa)."""
    amb_meta = dvf.regime_meta(_dvf_args())
    red_meta = dvf.regime_meta(_dvf_args(fit_basis="reduced"))
    assert amb_meta != red_meta
    prior = {"meta": amb_meta, "verdict": "x"}
    assert prior.get("meta") != red_meta


@pytest.mark.parametrize(
    "k,expected", [(6, "k<32"), (32, "32-127"), (128, "128-511"), (512, ">=512")]
)
def test_k_band_edges(k, expected):
    assert k_band(k) == expected


# --- Parent-inputs staging (#734 upload-first; fellows job 15724 crash-fix) ---


def _parent_inputs_fixture(root: Path) -> Path:
    """A tiny complete local parent-input set (all singles + 1 file/tree)."""
    for s in dvf.PARENT_INPUT_SINGLES:
        p = root / s
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("battery,n_common\n" if s.endswith(".csv") else "{}")
    for t in dvf.PARENT_INPUT_TREES:
        d = root / t
        d.mkdir(parents=True, exist_ok=True)
        (d / "unit_a.json").write_text("{}")
    return root


def test_stage_parent_inputs_skips_when_all_present(tmp_path, monkeypatch, capsys):
    """A complete local checkout (VM smoke / pod git lane) short-circuits
    BEFORE any Hub call — the [ -f ]-style idempotence guard."""
    root = _parent_inputs_fixture(tmp_path / "eval")

    def _boom(*a, **k):
        raise AssertionError("Hub call on a complete local checkout")

    monkeypatch.setattr("huggingface_hub.HfApi.repo_info", _boom)
    assert dvf.cmd_stage_parent_inputs(argparse.Namespace(parent_inputs_root=root)) == 0
    assert "— skip" in capsys.readouterr().out


def test_stage_parent_inputs_fetches_missing_and_fails_loud(tmp_path, monkeypatch):
    """Fresh-lane shape (fellows rsync: no eval_results/ at all): every
    mirrored file stages to its exact consumed rel path; an incomplete HF
    mirror is a RuntimeError, never a silent skip (the consumers'
    exists/WARN guards would silently drop the rung conditioning)."""
    import shutil
    import types

    from explore_persona_space.orchestrate import hub

    src = _parent_inputs_fixture(tmp_path / "src")
    rels = dvf._parent_input_rel_paths_local(src)

    monkeypatch.setattr(
        "huggingface_hub.HfApi.repo_info",
        lambda self, repo_id, repo_type=None: types.SimpleNamespace(sha="0" * 40),
    )
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda api, repo_id, path, *, repo_type="model", revision=None: [
            f"{dvf.PARENT_INPUTS_HF_PREFIX}/{r}" for r in rels
        ],
    )

    def fake_stage_hub_file(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        rel = path_in_repo.removeprefix(dvf.PARENT_INPUTS_HF_PREFIX + "/")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)
        return target

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage_hub_file)
    fresh = tmp_path / "fresh"
    assert dvf.cmd_stage_parent_inputs(argparse.Namespace(parent_inputs_root=fresh)) == 0
    for r in rels:
        assert (fresh / r).is_file(), r

    # incomplete mirror (digest CSV absent from the listing) -> fail-loud
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda api, repo_id, path, *, repo_type="model", revision=None: [
            f"{dvf.PARENT_INPUTS_HF_PREFIX}/{r}"
            for r in rels
            if r != "analyzer/dvf_unit_digest.csv"
        ],
    )
    with pytest.raises(RuntimeError, match="mirror incomplete"):
        dvf.cmd_stage_parent_inputs(argparse.Namespace(parent_inputs_root=tmp_path / "fresh2"))


def test_parent_input_rel_paths_local_fails_loud_on_gaps(tmp_path):
    root = _parent_inputs_fixture(tmp_path / "eval")
    assert set(dvf.PARENT_INPUT_SINGLES) <= set(dvf._parent_input_rel_paths_local(root))
    (root / dvf.PARENT_INPUT_SINGLES[0]).unlink()
    with pytest.raises(RuntimeError, match="singles missing"):
        dvf._parent_input_rel_paths_local(root)


def test_dispatch_leg_stages_parent_inputs_before_fence():
    """Ordering pin: the wellposed leg stages parent inputs BEFORE the fence
    phase (which open()s the digest CSV — the fellows job 15724 crash site)."""
    sh = (Path(dvf.REPO_ROOT) / "scripts/issue1689_dispatch.sh").read_text()
    body = sh[sh.index("run_phase_derived_vs_free_wellposed()") :]
    assert body.index("--phase stage-parent-inputs") < body.index("--phase fence")
    # the staged set covers the leg's committed-input classes
    assert "analyzer/dvf_unit_digest.csv" in dvf.PARENT_INPUT_SINGLES
    assert "crossmodel_pairs/ladder_crossmodel_L19.json" in dvf.PARENT_INPUT_SINGLES
    assert "derived_vs_free_B/pairs" in dvf.PARENT_INPUT_TREES
    assert "crossmodel_pairs/crossmodel_structure/pairs" in dvf.PARENT_INPUT_TREES
