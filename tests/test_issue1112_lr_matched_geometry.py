"""#1112 lr-matched geometry driver (plan v8, followup `lr-matched-method-pair`).

Driver-level tests on a tiny synthetic store tree (2 layers, 3 contexts x 4
questions, n_boot 25) running the REAL bodies end-to-end (no seams stubbed;
only the Hub boundary is avoided via the local capture-root mode):

1. the row-meta hard assert fires on a mismatched panel (wrong question idxs;
   missing row) BEFORE any paired statistic;
2. the mu-norm paired read uses identical resample indices — seeded serial
   oracle equivalence for the subset-sum GEMM + the paired-identity zero diff
   + row-permutation invariance (re-pair, never fail, plan §8 risk table);
3. the pair-read output schema (records / primary / secondaries / companions /
   parity / lattice / matrices) is stable;
4. install materialization is shape-asserted, never synthesized;
5. the figures body renders the plan-v8 set from the two payloads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1112_geometry as drv  # noqa: E402

from explore_persona_space.experiments import issue_1112 as C  # noqa: E402
from explore_persona_space.experiments.issue_653.spectral import (  # noqa: E402
    bootstrap_index_matrix,
)
from explore_persona_space.experiments.issue_1112 import geometry as geo  # noqa: E402

HID = 16
LAYERS = [0, 1]
CONTEXTS = ["src", "negA", "negB"]
NQ = 4
CELLS = (C.LR_MATCHED_CELL, "s3_fullft_neg", "s1_lora_neg")


def _store_dict(cell: str, dose: str, seed: int, *, question_offset: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    row_meta = [
        {"context_id": c, "question_idx": q + question_offset} for c in CONTEXTS for q in range(NQ)
    ]
    n = len(row_meta)
    arms = {}
    for arm in ("prefix", "context", "response"):
        per_layer = {}
        for li in LAYERS:
            if arm == "prefix":
                X = np.repeat(rng.standard_normal((len(CONTEXTS), HID)), NQ, axis=0)
            else:
                X = rng.standard_normal((n, HID))
            per_layer[li] = torch.from_numpy(X).to(torch.float16)
        arms[arm] = per_layer
    return {
        "schema_version": 1,
        "cell": cell,
        "dose": dose,
        "behavior": "sycophancy",
        "row_meta": row_meta,
        "arms": arms,
        "metadata": {"fixture": True},
    }


def _write_store(store: dict, root: Path) -> Path:
    out = root / "capture" / store["cell"] / store["dose"] / "pooled.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, out)
    return out


def _tree(root: Path) -> tuple[Path, Path]:
    for i, cell in enumerate(CELLS):
        _write_store(_store_dict(cell, "selected", 10 + i), root)
    _write_store(_store_dict("base_sycophancy", "base", 99), root)
    rb_dir = root / "rb"
    rb_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    torch.save({"rb": torch.randn(len(LAYERS), HID, generator=gen)}, rb_dir / "rb_sycophancy.pt")
    return root / "capture", rb_dir


def _write_selection(root: Path) -> Path:
    sel_dir = root / "selection" / C.LR_MATCHED_CELL
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "selection.json").write_text(
        json.dumps(
            {
                "step": 22,
                "rate": 0.6,
                "in_band": True,
                "fallback": None,
                "rates_by_step": {"2": 0.2, "12": 0.35, "22": 0.6},
                "band": [0.6, 0.85],
            }
        )
    )
    (sel_dir / "tier2_rates.json").write_text(
        json.dumps(
            {
                "cell": C.LR_MATCHED_CELL,
                "step": 22,
                "rates": {"trained": 0.61, "base": 0.225},
                "n": 10,
            }
        )
    )
    return root / "selection"


def _run(root: Path, **kw) -> dict:
    capture_root, rb_dir = _tree(root)
    return drv.run_lr_matched(
        capture_root,
        rb_dir,
        root / "out",
        n_boot=25,
        mu_n_boot=50,
        n_contexts=len(CONTEXTS),
        n_questions=NQ,
        primary_layer=1,
        **kw,
    )


# ── 3. pair-read output schema ───────────────────────────────────────────────


def test_run_lr_matched_end_to_end_schema(tmp_path):
    payload = _run(tmp_path, selection_dir=_write_selection(tmp_path))
    # records: 2 analyzed cells (s5 + re-derived s3) x 3 arms x 2 layers
    assert len(payload["records"]) == 2 * 3 * len(LAYERS)
    rec = payload["records"][f"{C.LR_MATCHED_CELL}/selected/response/L1"]
    for k in (
        "rank_k_at_90",
        "pr_lambda",
        "top_share_lambda",
        "mu_norm",
        "boot_ci",
        "cos_mu_to_rb",
        "random_cos_ci",
    ):
        assert k in rec, k
    assert rec["n_rows"] == len(CONTEXTS) * NQ

    pair = payload["lr_matched_pair"]
    prim = pair["primary"]
    assert prim["dv"] == "mu_norm" and prim["layer"] == 1 and prim["arm"] == "response"
    assert prim["n_boot"] == 50
    d = prim["diff_s3_minus_s5"]
    assert d["resampling"] == "paired" and d["n_boot"] == 50
    assert d["ci_low"] <= d["ci_high"]
    for key in (
        "mu_norm_diff_by_layer_s3_minus_s5",
        "reference_s3_minus_s1_by_layer",
        "exploratory_s1_minus_s5_by_layer",
        "secondary_diffs_s3_minus_s5_by_layer",
        "cos_companions_by_layer",
    ):
        assert set(pair[key]) == {"0", "1"}, key
    sec = pair["secondary_diffs_s3_minus_s5_by_layer"]["1"]
    assert set(sec) == {"top_share_lambda", "pr_lambda", "rank_k_at_90"}
    assert all(v["resampling"] == "paired" and v["n_boot"] == 25 for v in sec.values())
    comp = pair["cos_companions_by_layer"]["1"]
    assert set(comp) == {"cos_mu_s5_s1", "cos_mu_s5_s3", "cos_mu_s3_s1"}

    # parity block (WARN-only pipeline check on the new cell)
    par = payload["parity"][C.LR_MATCHED_CELL]
    assert par["warn_bar"] == drv.PARITY_WARN_COS and set(par["arms"]) == {"prefix", "context"}
    assert "parity_note" in payload

    # matched-80 (12 rows < 80 -> explicit note branch) + split-half ceiling
    assert "note" in payload["matched80"][C.LR_MATCHED_CELL]
    assert payload["split_half_self_cosine_ceiling"][C.LR_MATCHED_CELL]["n_partitions"] == 50
    assert set(payload["sv_primary_layer"]) == set(CELLS)

    # lattice: staged in_band=True -> a CI-sign mechanical branch (data only)
    assert payload["lattice"]["in_band"] is True
    assert payload["lattice"]["mechanical_branch"] in {
        "a_gap_survives_fullft_larger",
        "b_gap_closes_ci_includes_zero",
        "b_prime_gap_reverses_lora_larger",
    }
    assert payload["install"]["selection"]["step"] == 22
    assert payload["install"]["tier2"]["rates"]["trained"] == 0.61

    # per-draw matrices persisted for all three cells (identical-index pairing)
    mats_dir = tmp_path / "out" / "bootstrap_matrices" / "lr_matched"
    for cell in CELLS:
        mats = torch.load(mats_dir / f"{cell}_selected.pt", weights_only=False)
        assert mats["response/L1/mu_norm"].shape == (50,), cell
    mats5 = torch.load(mats_dir / f"{C.LR_MATCHED_CELL}_selected.pt", weights_only=False)
    assert mats5["response/L1/rank_k_at_90"].shape == (25,)
    assert any(k.startswith("parity/") for k in mats5)

    out = json.loads((tmp_path / "out" / "geometry_lr_matched.json").read_text())
    assert out["followup_label"] == "lr-matched-method-pair"
    assert out["metadata"]["git_commit"]
    assert out["boot_seed"] == C.BOOT_SEED and out["resampling"] == "paired"


# ── 1. row-meta hard assert ──────────────────────────────────────────────────


def test_row_meta_hard_assert_fires_on_wrong_question_idxs(tmp_path):
    capture_root, rb_dir = _tree(tmp_path)
    _write_store(_store_dict(C.LR_MATCHED_CELL, "selected", 10, question_offset=1), tmp_path)
    with pytest.raises(AssertionError, match="question idxs"):
        drv.run_lr_matched(
            capture_root,
            rb_dir,
            tmp_path / "out",
            n_boot=5,
            mu_n_boot=5,
            n_contexts=len(CONTEXTS),
            n_questions=NQ,
            primary_layer=1,
        )


def test_row_meta_hard_assert_fires_on_missing_row(tmp_path):
    capture_root, rb_dir = _tree(tmp_path)
    store = _store_dict(C.LR_MATCHED_CELL, "selected", 10)
    store["row_meta"] = store["row_meta"][:-1]
    store["arms"] = {
        arm: {li: t[:-1] for li, t in per.items()} for arm, per in store["arms"].items()
    }
    _write_store(store, tmp_path)
    with pytest.raises(AssertionError, match="registered"):
        drv.run_lr_matched(
            capture_root,
            rb_dir,
            tmp_path / "out",
            n_boot=5,
            mu_n_boot=5,
            n_contexts=len(CONTEXTS),
            n_questions=NQ,
            primary_layer=1,
        )


def test_row_permuted_store_is_repaired_not_failed(tmp_path):
    aligned = _run(tmp_path / "a")
    root = tmp_path / "perm"
    capture_root, rb_dir = _tree(root)
    p = capture_root / C.LR_MATCHED_CELL / "selected" / "pooled.pt"
    store = torch.load(p, weights_only=False)
    perm = torch.from_numpy(np.random.default_rng(0).permutation(len(store["row_meta"])))
    store["row_meta"] = [store["row_meta"][i] for i in perm.tolist()]
    store["arms"] = {
        arm: {li: t[perm] for li, t in per.items()} for arm, per in store["arms"].items()
    }
    torch.save(store, p)
    payload = drv.run_lr_matched(
        capture_root,
        rb_dir,
        root / "out",
        n_boot=25,
        mu_n_boot=50,
        n_contexts=len(CONTEXTS),
        n_questions=NQ,
        primary_layer=1,
    )
    a = aligned["lr_matched_pair"]["primary"]["diff_s3_minus_s5"]
    b = payload["lr_matched_pair"]["primary"]["diff_s3_minus_s5"]
    assert a["point"] == pytest.approx(b["point"], abs=1e-9)
    assert a["ci_low"] == pytest.approx(b["ci_low"], abs=1e-9)
    assert a["ci_high"] == pytest.approx(b["ci_high"], abs=1e-9)


# ── 2. mu-norm paired draws: seeded serial oracle ────────────────────────────


def test_mu_norm_draws_match_serial_oracle_and_pair_exactly():
    rng = np.random.default_rng(7)
    n = len(CONTEXTS) * NQ
    cloud = rng.standard_normal((n, HID))
    cluster_ids = [f"{c}__{q}" for c in CONTEXTS for q in range(NQ)]
    idx = bootstrap_index_matrix(cluster_ids, n_boot=13, seed=C.BOOT_SEED)
    W = drv._draw_weight_matrix(idx, n)
    got = drv._mu_norm_draws(cloud, W)
    assert got.shape == (13,)
    serial = np.array([np.linalg.norm(cloud[idx[b]].mean(axis=0)) for b in range(idx.shape[0])])
    np.testing.assert_allclose(got, serial, atol=1e-10)
    # identical-index pairing: a cell diffed against itself is EXACTLY zero
    rec = geo.paired_diff_record(got, got, 1.0, 1.0)
    assert rec["point"] == 0.0 and rec["ci_low"] == 0.0 and rec["ci_high"] == 0.0


# ── lattice branch mechanics (data-only encoding of plan §3) ─────────────────


def test_mechanical_branch_lattice():
    assert (
        drv._mechanical_branch({"ci_low": 0.5, "ci_high": 1.5}, True)
        == "a_gap_survives_fullft_larger"
    )
    assert (
        drv._mechanical_branch({"ci_low": -1.0, "ci_high": 1.0}, True)
        == "b_gap_closes_ci_includes_zero"
    )
    assert (
        drv._mechanical_branch({"ci_low": -2.0, "ci_high": -0.5}, True)
        == "b_prime_gap_reverses_lora_larger"
    )
    # branch (c) fires on out-of-band regardless of the CI
    assert (
        drv._mechanical_branch({"ci_low": 0.5, "ci_high": 1.5}, False)
        == "c_never_entered_band_descriptive_only"
    )
    assert drv._mechanical_branch(None, None) is None


# ── 4. install-record materialization ────────────────────────────────────────


def test_materialize_tier2_install_roundtrip_and_shape_assert(tmp_path):
    sel_root = _write_selection(tmp_path)
    out = drv.materialize_tier2_install(sel_root, tmp_path / "install")
    assert out.name == f"{C.LR_MATCHED_CELL}_tier2.json"
    rec = json.loads(out.read_text())
    assert rec == {
        "cell": C.LR_MATCHED_CELL,
        "step": 22,
        "rates": {"trained": 0.61, "base": 0.225},
        "n": 10,
    }
    # malformed staged record -> shape assert, never a synthesized file
    (sel_root / C.LR_MATCHED_CELL / "tier2_rates.json").write_text(
        json.dumps({"cell": C.LR_MATCHED_CELL})
    )
    with pytest.raises(AssertionError):
        drv.materialize_tier2_install(sel_root, tmp_path / "install2")
    assert not (tmp_path / "install2" / f"{C.LR_MATCHED_CELL}_tier2.json").exists()


# ── 5. figures body (real payloads, tiny scale) ──────────────────────────────


def test_lr_matched_figs_render(tmp_path):
    capture_root, rb_dir = _tree(tmp_path)
    parent_cells = ["s1_lora_neg", "s3_fullft_neg"]
    geo.run_geometry(
        capture_root,
        tmp_path / "parent_out",
        cells_doses=[(c, "selected") for c in parent_cells],
        base_store_by_behavior={
            "sycophancy": capture_root / "base_sycophancy" / "base" / "pooled.pt"
        },
        behavior_by_cell={c: "sycophancy" for c in parent_cells},
        selected_dose_by_cell={c: "selected" for c in parent_cells},
        rb_by_behavior={"sycophancy": rb_dir / "rb_sycophancy.pt"},
        layers=LAYERS,
        n_boot=10,
    )
    sel_root = _write_selection(tmp_path)
    drv.run_lr_matched(
        capture_root,
        rb_dir,
        tmp_path / "lr_out",
        selection_dir=sel_root,
        n_boot=10,
        mu_n_boot=10,
        n_contexts=len(CONTEXTS),
        n_questions=NQ,
        primary_layer=1,
    )
    install_dir = tmp_path / "install"
    drv.materialize_tier2_install(sel_root, install_dir)

    import issue1112_figures as figs

    figs.set_paper_style()
    figs.lr_matched_figs(
        tmp_path / "parent_out" / "geometry_per_cell.json",
        tmp_path / "lr_out" / "geometry_lr_matched.json",
        install_dir,
        tmp_path / "figs",
    )
    for name in (
        "hero_syco_mu_norm_lr_matched",
        "explore_install_ladder_lr_matched",
        "explore_rankk_profiles_lr_matched",
        "explore_lr_matched_mu_diff_by_layer",
        "explore_lr_matched_spectrum_cumshare",
        "explore_lr_matched_pr_lambda_profiles",
        "explore_lr_matched_top_share_lambda_profiles",
        "explore_lr_matched_cos_mu_to_rb",
    ):
        assert (tmp_path / "figs" / f"{name}.png").exists(), name
