"""Issue #2162 analysis-driver statistics pins (CPU, no artifacts needed).

Covers: the plan §6 family partition (P1 m=31 / P2 m=15 / P3 m=28 with the
four constructional exclusions), Holm step-down arithmetic, the exact
Wilcoxon IUT p, the rank-vectorized Mann-Whitney AUC, and the kernelized
batched logistic probe (separable data beats its own within-carrier
permutation band; the band sits near chance).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_analysis as A  # noqa: E402

from explore_persona_space.experiments.issue2162 import bank2162 as B  # noqa: E402


def test_family_partition_matches_plan_m():
    fam = Counter(A.family_of(c, s) for c in B.all_cells() for s in ("ce", "pe"))
    assert fam["P1"] == 31  # 16 base types x 2 slots - query_content@pe
    assert fam["P2"] == 15  # 4 route-variants x 2 - persona_role_header@pe + 4 conflict x 2
    assert fam["P3"] == 28  # (8 recency + 6 load) x 2
    assert fam[None] == 4  # filler_swap x 2 + the two pre-declared pe-degenerate cells


def test_holm_step_down():
    adj = A.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    # sorted: a(0.01)*3=0.03; c(0.03)*2=0.06; b(0.04)*1=0.04 -> monotone 0.06
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # step-down monotonicity


def test_wilcoxon_exact_p():
    d = np.array([0.5, 0.4, 0.6, 0.3, 0.7, 0.45, 0.55, 0.35, 0.65, 0.5, 0.4, 0.6])
    p_pos = A._wilcoxon_exact_p(d)
    assert p_pos < 0.001  # all-positive diffs at n=12: exact two-sided minimum
    mixed = np.array([0.5, -0.4, 0.3, -0.2, 0.1, -0.15, 0.05, -0.3, 0.25, -0.1, 0.2, -0.05])
    assert A._wilcoxon_exact_p(mixed) > 0.5
    assert A._wilcoxon_exact_p(np.zeros(5)) == 1.0  # all-zero diffs drop -> vacuous


def test_auc_ranked_extremes():
    labels = torch.tensor([[0, 0, 1, 1, 0, 1]])
    perfect = torch.tensor([[-2.0, -1.0, 1.0, 2.0, -3.0, 3.0]]).unsqueeze(1)  # (1,1,6)
    inverted = -perfect
    assert float(A._auc_ranked(perfect, labels)[0, 0]) == pytest.approx(1.0)
    assert float(A._auc_ranked(inverted, labels)[0, 0]) == pytest.approx(0.0)


def test_kernel_logistic_probe_separable_beats_band():
    """Synthetic (n=24, H=64, L=2): layer 0 carries a clean linear signal,
    layer 1 pure noise. Observed max-AUC must clear the within-carrier
    permutation band; the band itself must sit near chance."""
    torch.manual_seed(0)
    n_carriers, n_h = 12, 64
    y = torch.tensor([lab for _ in range(n_carriers) for lab in (0, 1)])
    groups = torch.tensor([g for g in range(n_carriers) for _ in (0, 1)])
    n = y.shape[0]
    direction = torch.randn(n_h)
    x_sig = torch.randn(n, n_h) * 0.3 + torch.outer(y.float() * 2 - 1, direction)
    x_noise = torch.randn(n, n_h)
    x = torch.stack([x_sig, x_noise])  # (L=2, n, H)
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    xs = (x - mu) / sd
    gram = torch.einsum("lnh,lmh->lnm", xs, xs) / n_h
    fold_masks = torch.stack([groups == g for g in range(n_carriers)])

    gen = torch.Generator().manual_seed(1)
    n_perm = 60
    flips = torch.randint(0, 2, (n_perm, n_carriers), generator=gen).bool()
    flip_rows = flips[:, groups]
    perm_labels = torch.where(flip_rows, 1 - y.unsqueeze(0), y.unsqueeze(0))
    all_labels = torch.cat([y.unsqueeze(0), perm_labels], dim=0)

    aucs = A.kernel_logistic_auc(gram, all_labels, fold_masks, epochs=120)
    obs = aucs[0]  # (L,)
    assert float(obs[0]) > 0.9, float(obs[0])  # signal layer near-perfect
    perm_max = aucs[1:].max(dim=1).values.numpy()  # per-draw re-max over layers
    band = float(np.percentile(perm_max, 97.5))
    assert float(obs.max()) > band
    assert 0.35 < float(np.median(aucs[1:, 1])) < 0.65  # noise layer perms ~ chance


def test_kernel_logistic_probe_null_within_band():
    """Pure-noise features: the observed max-AUC should NOT clear the band
    (selection-symmetric — both observed and band re-max over layers)."""
    torch.manual_seed(3)
    n_carriers, n_h = 12, 64
    y = torch.tensor([lab for _ in range(n_carriers) for lab in (0, 1)])
    groups = torch.tensor([g for g in range(n_carriers) for _ in (0, 1)])
    x = torch.randn(3, y.shape[0], n_h)  # 3 noise layers
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    xs = (x - mu) / sd
    gram = torch.einsum("lnh,lmh->lnm", xs, xs) / n_h
    fold_masks = torch.stack([groups == g for g in range(n_carriers)])
    gen = torch.Generator().manual_seed(4)
    n_perm = 60
    flips = torch.randint(0, 2, (n_perm, n_carriers), generator=gen).bool()
    perm_labels = torch.where(flips[:, groups], 1 - y.unsqueeze(0), y.unsqueeze(0))
    all_labels = torch.cat([y.unsqueeze(0), perm_labels], dim=0)
    aucs = A.kernel_logistic_auc(gram, all_labels, fold_masks, epochs=120)
    band = float(np.percentile(aucs[1:].max(dim=1).values.numpy(), 97.5))
    assert float(aucs[0].max()) <= band + 0.05  # within the band's neighborhood


# ── rule-19 validation grains (r2 R2) ────────────────────────────────


def _margin_row(pair_id: str, cell: str, shift: float) -> dict:
    return {"pair_id": pair_id, "cell": cell, "slot": "ce", "arm": "steered", "margin_shift": shift}


def test_rule19_validation_both_grains_and_screen():
    """r2 R2: the REGISTERED grain is per-(cell x slot) means across units
    with dynamic range (`validated` keys on it); the per-pair rho is the
    companion; a constant-margin unit is dropped by the declared screen."""
    rng = np.random.default_rng(0)
    margin_rows: list[dict] = []
    f_by_key: dict[tuple[str, str], float] = {}
    # 12 units with dynamic range, margin/f positively coupled across units.
    for u in range(12):
        for j in range(3):
            pid = f"u{u}p{j}"
            margin_rows.append(_margin_row(pid, f"cell{u}", 0.1 * u + 0.01 * j))
            f_by_key[(pid, "ce")] = 0.05 * u + 0.005 * j + float(rng.normal(0, 1e-3))
    # One DEGENERATE unit: constant margin across its pairs -> screened out.
    for j in range(3):
        pid = f"degp{j}"
        margin_rows.append(_margin_row(pid, "cell_deg", 0.42))
        f_by_key[(pid, "ce")] = 0.1 * j
    # Non-steered and missing-margin rows never enter either grain.
    margin_rows.append({**_margin_row("xx", "cell0", 0.5), "arm": "shuffled"})
    margin_rows.append(_margin_row("yy", "cell0", None))

    v = A.rule19_validation(margin_rows, f_by_key)
    assert v["n_cells"] == 12
    assert v["cells_dropped_no_dynamic_range"] == ["cell_deg|ce"]
    assert v["rho_margin_fbeh_percell"] is not None and v["rho_margin_fbeh_percell"] > 0.9
    assert v["validated"] is True
    assert v["n_pairs"] == 12 * 3 + 3  # degenerate unit's pairs still count per-pair
    assert v["rho_margin_fbeh_perpair"] is not None
    assert len(v["percell_points"]) == 12
    assert {p["cell"] for p in v["percell_points"]} == {f"cell{u}" for u in range(12)}
    assert "dynamic_range_screen" in v


def test_rule19_validation_below_floor_is_unvalidated():
    """Fewer than RULE19_MIN_N surviving units -> rho None, validated False."""
    margin_rows = [_margin_row(f"p{j}", "cell0", 0.1 * j) for j in range(3)]
    f_by_key = {(f"p{j}", "ce"): 0.2 * j for j in range(3)}
    v = A.rule19_validation(margin_rows, f_by_key)
    assert v["n_cells"] == 1
    assert v["rho_margin_fbeh_percell"] is None
    assert v["validated"] is False


# ── probe perm-matrix persistence (plan §248; gitignored *.npz) ──────


def test_persist_perm_matrix_upload_gating(tmp_path, monkeypatch, caplog):
    """The perm matrix is *.npz (gitignored repo-wide), so its ONLY durable
    home is HF analysis_tensors/probe_perm_matrix/ (plan §248 recomputability
    commitment): production uploads via the run driver's fail-loud seam;
    --no-upload skips with a loud warning and NO network call."""
    import logging

    import issue2162_run as R

    (tmp_path / "perm_auc_matrix.npz").write_bytes(b"x")
    calls: list[tuple] = []
    monkeypatch.setattr(
        R, "upload_dir_hf", lambda pdir, prefix, pats: calls.append((pdir, prefix, pats)) or []
    )
    with caplog.at_level(logging.WARNING, logger="issue2162.analysis"):
        A._persist_perm_matrix(tmp_path, no_upload=True)
    assert not calls
    assert any("NOT persisted" in rec.message for rec in caplog.records)
    A._persist_perm_matrix(tmp_path, no_upload=False)
    assert calls == [(tmp_path, f"{R.HF_PREFIX}/analysis_tensors/probe_perm_matrix", ["*.npz"])]


def test_analysis_no_upload_flag_default_false():
    """House convention: --no-upload defaults False so production persists."""
    args = A.parse_args(["--step", "probe"])
    assert args.no_upload is False
    assert A.parse_args(["--step", "probe", "--no-upload"]).no_upload is True
