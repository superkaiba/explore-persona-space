#!/usr/bin/env python3
"""Issue #823 single-split protocol-parity cell (inline free analysis).

Refit the four #823 answer arms under #779's D1 single-split protocol so the
numbers are directly comparable to #779's headline test R^2 = 0.705
[0.691, 0.719] at validation-selected layer 19. #823's own headline used a
5-fold GCV protocol; this script supplies the missing single-split cell.

Protocol — an exact reuse of ``scripts/issue779_fitter_fair_comparison.py``
stage D1 (read + copied verbatim, CPU float64):

  * fixed split of 5000 LMSYS contexts: train 3600 / val 400 / test 1000, seed 42
    (``fixed_split``: test = first 1000 of the seeded permutation, val = next 400,
    train = next 3600; each bucket sorted).
  * input   = last-context-token activation ``c_last`` (#779 pass_b bundle ``cx_last``).
  * target  = answer-span mean activation ``v(x)`` — per-arm tensor for the four
    #823 arms; the #779 bundle ``v_x`` for the parent-reproduction gate.
  * ridge in dual/Gram space; standardize-X on train mean/std, center-Y on train
    mean; ONE eigh factorization of the (n_tr, n_tr) Gram per (regime, layer),
    reused across every lambda and every target arm (arms differ only in the
    Y-side matmuls ``VtY``/``ymu``).
  * lambda selected on the 400-context validation set (NOT GCV: at n_tr~3600 ~= H
    GCV's (n-dof)^2 denominator degenerates and pins lambda to the grid floor,
    test R^2 ~ -5 to -8; #779 D1 documents this and uses val selection).
  * layer selected on validation (argmax val R^2 over the 28 layers), one clean
    test read at the selected layer.
  * metric = held-out variance-weighted (pooled, test-own-mean) R^2 over the 3584
    dims (``_pooled_r2``); 95% percentile bootstrap CI over test contexts.

Two lambda grids are evaluated off the SAME per-layer factorizations: the ORIGINAL
D1 grid ``logspace(-2, 4, 13)`` (the fidelity/headline read — #779's documented
selected-lambda-at-ceiling caveat is preserved) and a WIDER variant
``logspace(-2, 8, 21)`` whose first 13 points are bit-identical to the original,
reported as a clearly labelled variant so a ceiling-lambda can be checked against
an extended grid at no extra factorization cost.

ANALYSIS-ONLY: no training, no generation, no new data, no GPU. Reads the four
#823 arm tensors + the #779 c_last/v_x bundle + the #823 common-valid mask, all
sha-revision-pinned on the HF data repo.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF creds BEFORE torch import (shared-VM rule)

import argparse
import datetime as _dt
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── protocol constants (match #779 D1) ──────────────────────────────────────────
SPLIT_SEED = 42
N_CTX = 5000
N_TRAIN, N_VAL, N_TEST = 3600, 400, 1000
N_LAYERS = 28
HIDDEN = 3584
LAYER_19 = 19
LAMBDAS = np.logspace(-2, 4, 13)  # original D1 grid (fidelity/headline)
LAMBDAS_WIDE = np.logspace(-2, 8, 21)  # wider variant; first 13 == LAMBDAS
assert np.allclose(LAMBDAS_WIDE[:13], LAMBDAS), "wider grid's first 13 must equal the D1 grid"
BOOT_N = 10_000
PARENT_CI_LO, PARENT_CI_HI = 0.691, 0.719

# HF provenance (sha-pinned; recorded in the output JSON).
DATA_REPO = "superkaiba1/explore-persona-space-data"
ARM_REV = "8039d15f30deb845765cbb24d9cdb8708a5e7b0f"
BUNDLE_REV = "c94070508aa1c1f9c015ceb072231a2e51b28b3f"

# Plain-English arm names (never internal codes in figures / prose).
ARM_FILES = {"own": "v_a_prime.pt", "plain": "v_b2.pt", "style": "v_b1.pt", "mismatched": "v_c.pt"}
ARM_LABELS = {
    "own": "own answer (regenerated)",
    "plain": "external answer (Claude, plain)",
    "style": "external answer (Claude, eccentric style)",
    "mismatched": "swapped answer (derangement)",
}
ARM_ORDER = ["own", "plain", "style", "mismatched"]
CI_ARMS = ["own", "plain"]  # brief-required bootstrap CI arms (others computed too)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ── ridge math (verbatim CPU-float64 copy of issue779_fitter_fair_comparison D1) ─


def factorize(Xtr_np: np.ndarray) -> dict:
    """Standardize X on train stats, eigh the (n_tr, n_tr) Gram (float64)."""
    Xtr = torch.as_tensor(np.asarray(Xtr_np), dtype=torch.float64)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9  # matches GramRidge / fit_h
    Xtr_n = (Xtr - xmu) / xsd
    G = Xtr_n @ Xtr_n.T
    w, V = torch.linalg.eigh(G)
    return {"xmu": xmu, "xsd": xsd, "Xtr_n": Xtr_n, "w": torch.clamp(w, min=0.0), "V": V}


def cross_kernel(fact: dict, Xev_np: np.ndarray) -> torch.Tensor:
    """KevV = (Xev_n @ Xtr_n.T) @ V — shared across all targets at this eval set."""
    Xev = torch.as_tensor(np.asarray(Xev_np), dtype=torch.float64)
    Xev_n = (Xev - fact["xmu"]) / fact["xsd"]
    return (Xev_n @ fact["Xtr_n"].T) @ fact["V"]


def vty_ymu(fact: dict, Ytr_np: np.ndarray):
    """(VtY, ymu) for one target off a shared factorization (lambda-independent)."""
    Ytr = torch.as_tensor(np.asarray(Ytr_np), dtype=torch.float64)
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    ymu = Ytr.mean(0)
    return fact["V"].T @ (Ytr - ymu), ymu


def apply_ridge(fact: dict, lam: float, VtY: torch.Tensor, ymu: torch.Tensor, KevV: torch.Tensor):
    filt = 1.0 / (fact["w"] + lam)
    return ((KevV * filt) @ VtY + ymu).cpu().numpy()


def pooled_r2(pred: np.ndarray, true: np.ndarray) -> float:
    """Pooled R2 with SS_tot on TRUE's OWN mean (held-out variance fraction)."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def lambda_edge(lam: float, grid: np.ndarray) -> str | None:
    """'low'/'high' when a val-selected lambda sits at an edge of ``grid``."""
    if lam is None or not np.isfinite(lam):
        return None
    if np.isclose(float(lam), float(grid[0])):
        return "low"
    if np.isclose(float(lam), float(grid[-1])):
        return "high"
    return None


def select_lambda(fact, VtY, ymu, KvalV, Yval, grid) -> tuple[float, float]:
    """Val-selected lambda over ``grid`` (argmax val R2; ascending, strict >, so
    ties keep the smaller lambda — matches D1's gram_fit_apply)."""
    best_lam, best_vr2 = float(grid[0]), -np.inf
    for lam in grid:
        vr2 = pooled_r2(apply_ridge(fact, float(lam), VtY, ymu, KvalV), Yval)
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    return best_lam, best_vr2


def bootstrap_r2_ci(pred: np.ndarray, true: np.ndarray, n_boot: int, seed: int) -> dict:
    """Point + 95% percentile-bootstrap CI of pooled R2 over test contexts
    (verbatim shape of D1's _bootstrap_recon_ci R2 leg)."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = pred.shape[0]
    res_i = ((true - pred) ** 2).sum(axis=1)
    r2_point = pooled_r2(pred, true)
    rng = np.random.default_rng(seed)
    r2s: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ss_res = float(res_i[idx].sum())
        t = true[idx]
        ss_tot = float(((t - t.mean(0)) ** 2).sum())
        if ss_tot > 1e-12:
            r2s.append(1.0 - ss_res / ss_tot)
    return {
        "point": r2_point,
        "lo": float(np.quantile(r2s, 0.025)) if r2s else float("nan"),
        "hi": float(np.quantile(r2s, 0.975)) if r2s else float("nan"),
        "n_boot": len(r2s),
        "n_test": int(n),
    }


# ── split ────────────────────────────────────────────────────────────────────


def fixed_split(n_ctx: int, n_train: int, n_val: int, n_test: int, seed: int):
    """(train, val, test) sorted index arrays — verbatim from D1's fixed_split."""
    perm = np.random.default_rng(seed).permutation(n_ctx)
    return (
        np.sort(perm[n_test + n_val : n_test + n_val + n_train]),
        np.sort(perm[n_test : n_test + n_val]),
        np.sort(perm[:n_test]),
    )


# ── one D1 sweep over 28 layers for a set of targets sharing one input ─────────


def run_sweep(get_input_col, targets: dict, split) -> dict:
    """Per (target, layer): {val_r2, test_r2, selected_lambda, lambda_edge} under
    BOTH grids, plus the val-selected layer's test read + cached test prediction.

    ``get_input_col(li) -> (n_ctx, H) float64`` returns the c_last column for
    layer ``li``; ``targets`` maps arm-name -> ``get_target_col(li) -> (n_ctx, H)``.
    ONE factorization per layer is shared across every target and both grids.
    Returns ``{arm: {"orig": {...}, "wide": {...}}}`` with in-memory
    ``pred_te_selected`` for the val-selected layer (for the bootstrap)."""
    train, val, test = split
    # accumulator keyed by (arm, grid_name)
    acc = {
        arm: {
            g: {
                "per_layer": {},
                "best_val": -np.inf,
                "best_li": None,
                "best_pred_te": None,
                "best_Yte": None,
            }
            for g in ("orig", "wide")
        }
        for arm in targets
    }
    for li in range(N_LAYERS):
        t0 = time.time()
        Xcol = get_input_col(li)  # (n_ctx, H) float64
        Xtr, Xval, Xte = Xcol[train], Xcol[val], Xcol[test]
        fact = factorize(Xtr)
        KvalV = cross_kernel(fact, Xval)
        KteV = cross_kernel(fact, Xte)
        for arm, get_target in targets.items():
            Ycol = get_target(li)
            Ytr, Yval, Yte = Ycol[train], Ycol[val], Ycol[test]
            VtY, ymu = vty_ymu(fact, Ytr)
            for g, grid in (("orig", LAMBDAS), ("wide", LAMBDAS_WIDE)):
                lam, _vr2 = select_lambda(fact, VtY, ymu, KvalV, Yval, grid)
                pred_val = apply_ridge(fact, lam, VtY, ymu, KvalV)
                pred_te = apply_ridge(fact, lam, VtY, ymu, KteV)
                vr2 = pooled_r2(pred_val, Yval)
                tr2 = pooled_r2(pred_te, Yte)
                node = acc[arm][g]
                node["per_layer"][str(li)] = {
                    "val_r2": vr2,
                    "test_r2": tr2,
                    "selected_lambda": lam,
                    "lambda_edge": lambda_edge(lam, grid),
                }
                if np.isfinite(vr2) and vr2 > node["best_val"]:
                    node["best_val"] = vr2
                    node["best_li"] = li
                    node["best_pred_te"] = pred_te
                    node["best_Yte"] = Yte
        _log(f"  layer {li:2d} done ({time.time() - t0:.1f}s)")
    return acc


def summarize_arm(node: dict, grid: np.ndarray, boot_seed_base: int, do_ci: bool) -> dict:
    """Collapse one (arm, grid) accumulator into the reported summary."""
    per_layer = node["per_layer"]
    li_sel = node["best_li"]
    sel = per_layer[str(li_sel)]
    l19 = per_layer[str(LAYER_19)]
    out = {
        "val_selected_layer": int(li_sel),
        "test_r2_at_selected_layer": sel["test_r2"],
        "val_r2_at_selected_layer": sel["val_r2"],
        "selected_lambda": sel["selected_lambda"],
        "lambda_edge_at_selected": sel["lambda_edge"],
        "test_r2_at_layer_19": l19["test_r2"],
        "selected_lambda_at_layer_19": l19["selected_lambda"],
        "lambda_edge_at_layer_19": l19["lambda_edge"],
        "per_layer": per_layer,
    }
    if do_ci:
        out["test_r2_ci_at_selected_layer"] = bootstrap_r2_ci(
            node["best_pred_te"], node["best_Yte"], BOOT_N, boot_seed_base + int(li_sel)
        )
    return out


# ── figure ─────────────────────────────────────────────────────────────────────


def make_figure(arms_orig: dict, parent_anchor: float, fig_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    labels = [ARM_LABELS[a] for a in ARM_ORDER]
    vals = [arms_orig[a]["test_r2_at_selected_layer"] for a in ARM_ORDER]
    # asymmetric error offsets from the bootstrap CI (clamped non-negative)
    yerr_lo, yerr_hi = [], []
    for a in ARM_ORDER:
        ci = arms_orig[a].get("test_r2_ci_at_selected_layer")
        v = arms_orig[a]["test_r2_at_selected_layer"]
        if ci is None or not np.isfinite(ci.get("lo", float("nan"))):
            yerr_lo.append(0.0)
            yerr_hi.append(0.0)
        else:
            yerr_lo.append(max(0.0, v - ci["lo"]))
            yerr_hi.append(max(0.0, ci["hi"] - v))
    colors = [
        paper_palette_role("primary"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
        paper_palette_role("neutral"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(ARM_ORDER))
    ax.bar(x, vals, yerr=[yerr_lo, yerr_hi], capsize=5, color=colors, width=0.62, zorder=3)
    ax.axhline(
        parent_anchor,
        ls="--",
        lw=1.4,
        color=paper_palette_role("accent"),
        zorder=2,
        label=f"#779 parent (own answer, original): R² = {parent_anchor:.3f}",
    )
    for xi, v in zip(x, vals, strict=True):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=10, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("held-out test R² (single-split, val-selected layer)")
    ax.set_title(
        "Answer-map ceiling under #779's single-split protocol",
        loc="left",
        fontweight="semibold",
    )
    ax.set_ylim(0, max(0.75, max(vals) + 0.08, parent_anchor + 0.05))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    written = savefig_paper(fig, "fig6_single_split_protocol", dir=str(fig_dir))
    plt.close(fig)
    return {k: str(v) for k, v in written.items()}


# ── metadata ─────────────────────────────────────────────────────────────────


def repro_metadata() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    now = _dt.datetime.now(_dt.UTC).replace(tzinfo=None)
    return {
        "git_commit": sha,
        "timestamp_utc": now.isoformat() + "Z",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dl-root",
        default="/mnt/eps-data/thomasjiralerspong/issue823_work/dl/single_split_protocol",
        help="root of the downloaded HF tensors (boot disk is full; staged on eps-data)",
    )
    ap.add_argument(
        "--out-dir", default=str(REPO_ROOT / "eval_results/issue_823/single_split_protocol")
    )
    ap.add_argument("--fig-dir", default=str(REPO_ROOT / "figures/issue_823"))
    args = ap.parse_args()

    torch.set_num_threads(8)
    dl = Path(args.dl_root)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / ".done"
    if done_marker.exists():
        done_marker.unlink()

    print("[phase=load]", flush=True)
    bundle_path = dl / "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
    bundle = torch.load(str(bundle_path), map_location="cpu", mmap=True, weights_only=False)
    for fld in ("cx_last", "v_x", "layers"):
        assert fld in bundle, f"pass_b bundle missing {fld}"
    cx_last = bundle["cx_last"]
    v_x = bundle["v_x"]
    layers_map = list(bundle["layers"])
    assert tuple(cx_last.shape) == (N_CTX, N_LAYERS, HIDDEN), cx_last.shape
    assert tuple(v_x.shape) == (N_CTX, N_LAYERS, HIDDEN), v_x.shape
    _log(f"pass_b bundle loaded: cx_last {tuple(cx_last.shape)}, layers={layers_map[:3]}...")

    arm_tensors = {}
    for arm, fn in ARM_FILES.items():
        p = dl / "issue823_own_vs_external/analysis_tensors" / fn
        t = torch.load(str(p), map_location="cpu", mmap=True, weights_only=False)
        assert tuple(t.shape) == (N_CTX, N_LAYERS, HIDDEN), (arm, t.shape)
        arm_tensors[arm] = t
    _log(f"arm tensors loaded: {list(arm_tensors)}")

    mask_path = dl / "issue823_own_vs_external/raw_completions/phase1/common_valid_idx.json"
    valid_idx = np.array(sorted(json.loads(mask_path.read_text())["common_valid_idx"]), dtype=int)
    n_valid = len(valid_idx)
    _log(f"common-valid mask: {n_valid} valid, {N_CTX - n_valid} dropped")
    assert n_valid == 4998, f"expected 4998 valid contexts, got {n_valid}"
    valid_set = set(valid_idx.tolist())

    def input_col(li: int) -> np.ndarray:
        col = layers_map.index(li)
        return cx_last[:, col, :].to(torch.float64).numpy()

    def target_vx_col(li: int) -> np.ndarray:
        col = layers_map.index(li)
        return v_x[:, col, :].to(torch.float64).numpy()

    def arm_col_fn(arm: str):
        def _f(li: int) -> np.ndarray:
            return arm_tensors[arm][:, li, :].to(torch.float64).numpy()

        return _f

    # ── splits ──
    p_train, p_val, p_test = fixed_split(N_CTX, N_TRAIN, N_VAL, N_TEST, SPLIT_SEED)
    a_train = np.array([i for i in p_train if i in valid_set], dtype=int)
    a_val = np.array([i for i in p_val if i in valid_set], dtype=int)
    a_test = np.array([i for i in p_test if i in valid_set], dtype=int)
    _log(
        f"parent split {len(p_train)}/{len(p_val)}/{len(p_test)}; "
        f"masked arm split {len(a_train)}/{len(a_val)}/{len(a_test)}"
    )

    # ── parent-reproduction GATE (full 5000, no mask, bundle v_x target) ──
    print("[phase=parent_gate]", flush=True)
    _log("parent gate: sweeping 28 layers (c_last -> parent v_x, full 5000)")
    parent_acc = run_sweep(input_col, {"parent": target_vx_col}, (p_train, p_val, p_test))
    parent_orig = summarize_arm(parent_acc["parent"]["orig"], LAMBDAS, SPLIT_SEED, do_ci=True)
    parent_wide = summarize_arm(parent_acc["parent"]["wide"], LAMBDAS_WIDE, SPLIT_SEED, do_ci=False)
    anchor = parent_orig["test_r2_at_selected_layer"]
    inside_ci = bool(PARENT_CI_LO <= anchor <= PARENT_CI_HI)
    _log(
        f"PARENT anchor: test R2={anchor:.4f} at layer {parent_orig['val_selected_layer']} "
        f"lambda={parent_orig['selected_lambda']:.3g} "
        f"(edge={parent_orig['lambda_edge_at_selected']}); inside [{PARENT_CI_LO},{PARENT_CI_HI}]="
        f"{inside_ci}"
    )

    meta = repro_metadata()
    result = {
        "experiment": "issue823_single_split_protocol",
        "description": (
            "#779 D1 single-split protocol (train 3600 / val 400 / test 1000, seed 42; "
            "val-selected layer + lambda; held-out variance-weighted R2) applied to the "
            "four #823 answer arms, for direct comparison with #779's 0.705 headline."
        ),
        "reproducibility": meta,
        "data_provenance": {
            "hf_repo": DATA_REPO,
            "arm_tensors_revision": ARM_REV,
            "bundle_revision": BUNDLE_REV,
            "arm_files": ARM_FILES,
            "common_valid_idx": (
                "issue823_own_vs_external/raw_completions/phase1/common_valid_idx.json"
            ),
            "mask_method": "explicit common_valid_idx.json valid-index list",
        },
        "protocol": {
            "split_seed": SPLIT_SEED,
            "n_contexts": N_CTX,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_layers": N_LAYERS,
            "hidden": HIDDEN,
            "lambda_grid_original": {
                "expr": "logspace(-2,4,13)",
                "min": float(LAMBDAS[0]),
                "max": float(LAMBDAS[-1]),
                "n": len(LAMBDAS),
            },
            "lambda_grid_wider": {
                "expr": "logspace(-2,8,21)",
                "min": float(LAMBDAS_WIDE[0]),
                "max": float(LAMBDAS_WIDE[-1]),
                "n": len(LAMBDAS_WIDE),
            },
            "metric": "held-out variance-weighted (pooled, test-own-mean) R2 over 3584 dims",
            "boot_n": BOOT_N,
            "input": "c_last (pass_b cx_last, last context token)",
            "device": "cpu",
            "dtype": "float64",
        },
        "split_realized": {
            "parent": {"n_train": len(p_train), "n_val": len(p_val), "n_test": len(p_test)},
            "arms_masked": {
                "n_valid": int(n_valid),
                "n_dropped": int(N_CTX - n_valid),
                "n_train": len(a_train),
                "n_val": len(a_val),
                "n_test": len(a_test),
            },
        },
        "parent_reproduction": {
            "target": "#779 bundle v_x (parent own-answer)",
            "reference_interval": [PARENT_CI_LO, PARENT_CI_HI],
            "reference_point": 0.705,
            "inside_reference_interval": inside_ci,
            "original_grid": parent_orig,
            "wider_grid": parent_wide,
        },
    }

    if not inside_ci:
        result["gate"] = {
            "passed": False,
            "note": (
                "Parent reproduction OUTSIDE [0.691, 0.719]; per the brief, arm cells are "
                "NOT computed. Reporting the reproduction value + selected (layer, lambda) only."
            ),
        }
        # trim per-layer bulk to keep the JSON legible on a gate failure
        result_out = json.loads(json.dumps(result))
        (out_dir / "single_split_protocol.json").write_text(json.dumps(result_out, indent=2))
        done_marker.write_text("gate_failed\n")
        _log("GATE FAILED — wrote reproduction-only JSON, no arm cells. STOP.")
        return

    result["gate"] = {"passed": True}

    # ── the four #823 arms (masked split, same c_last input, arm-tensor targets) ──
    print("[phase=arms]", flush=True)
    _log("arm sweep: 28 layers x 4 arms (masked split)")
    arm_targets = {arm: arm_col_fn(arm) for arm in ARM_ORDER}
    arm_acc = run_sweep(input_col, arm_targets, (a_train, a_val, a_test))

    arms_orig, arms_wide = {}, {}
    for arm in ARM_ORDER:
        do_ci = True  # bootstrap CI for all four (brief requires own + plain; superset is fine)
        arms_orig[arm] = summarize_arm(arm_acc[arm]["orig"], LAMBDAS, SPLIT_SEED, do_ci=do_ci)
        arms_wide[arm] = summarize_arm(arm_acc[arm]["wide"], LAMBDAS_WIDE, SPLIT_SEED, do_ci=False)
        s = arms_orig[arm]
        _log(
            f"ARM {arm:11s} ({ARM_LABELS[arm]}): sel-layer {s['val_selected_layer']:2d} "
            f"test R2={s['test_r2_at_selected_layer']:.4f} "
            f"(L19 {s['test_r2_at_layer_19']:.4f}) lambda={s['selected_lambda']:.3g} "
            f"edge={s['lambda_edge_at_selected']}"
        )

    # ── sanity cross-check vs the known 5-fold ordering (STOP on a wild ordering) ──
    l19 = {arm: arms_orig[arm]["test_r2_at_layer_19"] for arm in ARM_ORDER}
    ordering_sane = (
        l19["own"] > l19["style"]
        and l19["style"] > l19["mismatched"]
        and l19["mismatched"] < 0.10
        and l19["own"] > 0.40
    )
    result["arm_sanity_check"] = {
        "layer_19_test_r2": l19,
        "expected_ordering": "own > plain > style >> mismatched (near 0)",
        "ordering_sane": bool(ordering_sane),
        "known_5fold_layer19": {"own": 0.679, "plain": 0.666, "style": 0.555, "mismatched": 0.009},
        "note": (
            "single-split test R2 differs from the 5-fold values by protocol; only the "
            "coarse ordering + near-zero mismatched arm are checked."
        ),
    }
    if not ordering_sane:
        raise SystemExit(
            f"WILD arm ordering at layer 19 (own {l19['own']:.3f}, plain {l19['plain']:.3f}, "
            f"style {l19['style']:.3f}, mismatched {l19['mismatched']:.3f}) — STOP per brief."
        )

    result["arms"] = {
        "labels": ARM_LABELS,
        "order": ARM_ORDER,
        "ci_required_arms": CI_ARMS,
        "original_grid": arms_orig,
        "wider_grid": arms_wide,
    }

    # wider-grid reporting flag — whether any headline selected lambda sits on an edge
    edge_hits = {
        "parent": parent_orig["lambda_edge_at_selected"],
        **{a: arms_orig[a]["lambda_edge_at_selected"] for a in ARM_ORDER},
    }
    result["wider_grid_note"] = {
        "reason": (
            "The original D1 grid's ceiling (1e4) is the documented #779 selected-lambda edge. "
            "The wider grid logspace(-2,8,21) shares the first 13 points bit-identically and is "
            "reported as a labelled variant; consult it wherever lambda_edge_at_selected == 'high'."
        ),
        "selected_lambda_edges_original_grid": edge_hits,
    }

    # ── figure ──
    print("[phase=figure]", flush=True)
    result["figure"] = make_figure(arms_orig, anchor, fig_dir)

    (out_dir / "single_split_protocol.json").write_text(json.dumps(result, indent=2))
    done_marker.write_text("ok\n")
    print("[phase=done]", flush=True)
    _log(f"WROTE {out_dir / 'single_split_protocol.json'} + figure. DONE.")


if __name__ == "__main__":
    main()
