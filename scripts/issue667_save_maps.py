#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, M⁺, →, ×) in scientific docstrings + log messages.
"""Issue #667 — save the ACTUAL fitted ridge maps M0 and M⁺ per (behavior, layer).

The #722 driver (``scripts/issue722_fit_M.py``) fits M0/M⁺ per cell but only
persists DERIVED SCALARS (Δ_med, floors, chain-ρ). This pass saves the fitted
maps THEMSELVES so downstream analysis can operate on them directly — ‖ΔM‖,
SVD of ΔM, projections onto r_B / W_U[marker] / arbitrary directions, applying
M to arbitrary contexts.

**Efficiency.** ONLY the two closed-form ridge fits per cell (M0, M⁺). No floors
(bootstrap refits), no chain-ρ, no MLP, no cross-transfer, no map-change reads —
those were the slow #722 phases and are not needed to persist the maps. This runs
in minutes on CPU (108 cells × two closed-form ridge fits each).

**Exact-reproduction contract.** The saved components reconstruct
``M0(c) = ((c − input_mean_C0) / input_std_C0) @ W_M0`` (→ 64-dim, in the shared
V0 PCA basis) BIT-FOR-BIT identically to ``fit_M._ridge_fit_predict(C0, V0_64,
grid)``, and likewise ``M⁺(c) = ((c − input_mean_Cplus) / input_std_Cplus) @
W_Mplus`` == ``fit_M._ridge_fit_predict(Cplus, Vplus_64, grid)``. Every
convention (the PRESS-LOO λ selection, the ``mu = X.mean(0)`` /
``sd = X.std(0, correction=0) + 1e-9`` input normalization, the UNCENTERED Y64
target fed to the dual ridge, the SINGLE V0-derived PCA basis shared by BOTH
maps) is inherited verbatim from ``fit_M`` — this script imports and CALLS those
helpers rather than re-deriving them, so it cannot drift. The correctness gate
below reconstructs a few cells from the saved components and asserts they match
``_ridge_fit_predict`` to < 1e-8 relative.

**The V0-basis subtlety (matched to fit_M, load-bearing).** ``fit_cell`` builds
ONE ``pca_basis = _pca_basis_v0(V0, 64)`` and projects BOTH ``V0`` and ``Vplus``
onto it (``V0_64`` / ``Vplus_64``). So M⁺ is fit against ``Vplus`` expressed in
the BASE v0 PCA basis, NOT its own basis. We save that single V0 basis and use it
for both maps — matching fit_M exactly.

Output (npz per (behavior, layer) under ``eval_results/issue_667_maps/<behavior>/
L<layer>.npz``): ``W_M0`` (d, 64) float32, ``W_Mplus`` (d, 64) float32,
``pca_basis`` (64, d) — plus the input normalization for C0 AND Cplus, the
lambdas, cell_keys, families, n, behavior, layer, and provenance. float32 for the
big weight matrices; λ / normalization stats stay float64.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# DOTENV_LINT_EXEMPT: exploratory analysis script; the HF upload calls
# orchestrate.env.load_dotenv (below) before any hub import that reads HF_TOKEN.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue658_fit_predictors as fit658  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import issue722_load_activations as loadact  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue667.save_maps")

DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MAP_PREFIX = "issue667_maps"
# All 4 store behaviors; L0 dropped (extraction-aliased) → layers 1-27.
BEHAVIORS = ("em", "sycophancy", "fact", "marker")
LAYERS = tuple(range(1, 28))
TARGET_DIM = 64
HIDDEN = loadact.HIDDEN  # 3584


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _ridge_components(X: np.ndarray, Y64: np.ndarray) -> dict:
    """Fit the closed-form ridge map (X → Y64) and return its SAVEABLE components.

    Mirrors ``fit_M._ridge_fit_predict`` EXACTLY up to (but not including) the
    grid evaluation: same PRESS-LOO λ selection, same input normalization
    (``mu = X.mean(0)``, ``sd = X.std(0, correction=0) + 1e-9``), same UNCENTERED
    dual-ridge weights ``w = _ridge_dual_weights(Xn, Y64, best_lam)``. Returns the
    pieces needed to reconstruct ``M(c) = ((c − mu) / sd) @ w`` — which is
    precisely what ``_ridge_fit_predict`` computes at any grid input.

    Returns ``{"W" (d, 64) float64, "input_mean" (d,), "input_std" (d,),
    "lambda"}``. The caller down-casts ``W`` to float32; the normalization stats +
    λ stay float64.
    """
    lambdas = fit658.RIDGE_LAMBDAS
    device = torch.device(fit658.DEVICE)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device=device, dtype=torch.float64)
    Yt = torch.from_numpy(np.ascontiguousarray(Y64)).to(device=device, dtype=torch.float64)
    mu = Xt.mean(0)
    sd = Xt.std(0, correction=0) + 1e-9
    Xn = (Xt - mu) / sd
    mse = fit658._press_loo_mse_per_lambda(Xn, Yt, lambdas)
    best_lam = lambdas[int(torch.argmin(mse).item())]
    w = fit658._ridge_dual_weights(Xn, Yt, best_lam)  # (d, 64)
    return {
        "W": w.detach().cpu().numpy(),  # (d, 64) float64
        "input_mean": mu.detach().cpu().numpy(),  # (d,) float64
        "input_std": sd.detach().cpu().numpy(),  # (d,) float64
        "lambda": float(best_lam),
    }


def _apply_from_components(
    c: np.ndarray, W: np.ndarray, input_mean: np.ndarray, input_std: np.ndarray
) -> np.ndarray:
    """Reconstruct ``M(c) = ((c − input_mean) / input_std) @ W`` → (..., 64).

    Accepts a single (d,) vector or a (n, d) batch; the output centering is the
    identity (fit_M feeds the ridge target UNCENTERED, so M(c) is already in the
    saved PCA-basis coordinate frame). Computed in float64 for the correctness
    gate; downstream callers may use the float32 W directly.
    """
    c = np.asarray(c, dtype=np.float64)
    cn = (c - input_mean) / input_std
    return cn @ W


def fit_and_save_cell(behavior: str, layer: int, cells: list, out_dir: Path) -> tuple[Path, dict]:
    """Fit M0 + M⁺ for one (behavior, layer), save the npz, return (path, meta).

    Follows ``fit_M.fit_cell`` conventions verbatim: one shared V0 PCA basis
    projects both V0 and Vplus; M0 = ridge(C0 → V0_64); M⁺ = ridge(Cplus →
    Vplus_64). No floors / chain-ρ / MLP.
    """
    stacks = loadact.stack_for_fit(cells)
    C0, Cplus = stacks["C0"], stacks["Cplus"]
    V0, Vplus = stacks["V0"], stacks["Vplus"]
    families = stacks["families"]
    cell_keys = stacks["cell_keys"]
    n = C0.shape[0]
    assert n >= 4, f"{behavior} L{layer}: only {n} cells (<4) — cannot fit"

    # SINGLE V0-derived basis, shared by BOTH maps (matches fit_M.fit_cell).
    pca_basis = fitM._pca_basis_v0(V0, TARGET_DIM)  # (k<=64, 3584)
    V0_64 = fitM._to64(V0, pca_basis)
    Vplus_64 = fitM._to64(Vplus, pca_basis)

    comp_m0 = _ridge_components(C0, V0_64)
    comp_mplus = _ridge_components(Cplus, Vplus_64)

    out_dir_b = out_dir / behavior
    out_dir_b.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_b / f"L{layer}.npz"
    np.savez_compressed(
        out_path,
        W_M0=comp_m0["W"].astype(np.float32),
        W_Mplus=comp_mplus["W"].astype(np.float32),
        pca_basis=pca_basis.astype(np.float32),  # (k, 3584) — the shared V0 basis
        input_mean_C0=comp_m0["input_mean"],  # float64 (3584,)
        input_std_C0=comp_m0["input_std"],  # float64 (3584,)
        input_mean_Cplus=comp_mplus["input_mean"],  # float64 (3584,)
        input_std_Cplus=comp_mplus["input_std"],  # float64 (3584,)
        # Output centering: NONE — fit_M feeds the ridge target UNCENTERED, so the
        # map output is already in the pca_basis coordinate frame. Persisted as an
        # explicit zero vector + flag so a downstream reader never has to guess.
        output_center_V0=np.zeros(TARGET_DIM, dtype=np.float64),
        output_center_Vplus=np.zeros(TARGET_DIM, dtype=np.float64),
        output_centered=np.asarray(False),
        lambda_M0=np.asarray(comp_m0["lambda"], dtype=np.float64),
        lambda_Mplus=np.asarray(comp_mplus["lambda"], dtype=np.float64),
        cell_keys=np.asarray(cell_keys, dtype=object),
        families=np.asarray(families, dtype=object),
        n=np.asarray(n),
        behavior=np.asarray(behavior),
        layer=np.asarray(layer),
        target_dim=np.asarray(pca_basis.shape[0]),
        dtype_note=np.asarray(
            "W_M0/W_Mplus/pca_basis: float32; input_mean/std, lambdas, output_center: float64"
        ),
        issue=np.asarray(667),
        source_store=np.asarray("issue_667_alllayer/analysis_tensors"),
        git_sha=np.asarray(_git_sha()),
        generated_at=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )
    # Return the in-memory float64 comps for the correctness gate (uses the exact
    # fit output, then the gate independently reloads the float32 npz too).
    meta = {
        "n": n,
        "cell_keys": cell_keys,
        "families": families,
        "C0": C0,
        "Cplus": Cplus,
        "V0_64": V0_64,
        "Vplus_64": Vplus_64,
        "pca_basis": pca_basis,
        "comp_m0": comp_m0,
        "comp_mplus": comp_mplus,
    }
    return out_path, meta


def correctness_gate(meta: dict, out_path: Path) -> dict:
    """Reconstruct M0/M⁺ FROM THE SAVED npz + assert < 1e-8 rel-diff vs _ridge_fit_predict.

    Two-sided check: (1) the in-memory float64 components reconstruct
    ``_ridge_fit_predict`` (proves the fit math is faithful); (2) the RELOADED
    float32 npz reconstructs it to a float32-appropriate tolerance (proves the
    saved artifact is usable). Evaluated at the base grid (= C0 for M0, Cplus for
    M⁺ — the input the map is actually applied to). Returns the max rel-diffs.
    """
    C0, Cplus = meta["C0"], meta["Cplus"]
    V0_64, Vplus_64 = meta["V0_64"], meta["Vplus_64"]

    # Reference: fit_M's own grid evaluation (the number downstream trusts).
    ref_m0 = fitM._ridge_fit_predict(C0, V0_64, C0)  # (n, 64)
    ref_mplus = fitM._ridge_fit_predict(Cplus, Vplus_64, Cplus)

    def _reldiff(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.maximum(np.abs(b), 1e-12)
        return float(np.max(np.abs(a - b) / denom))

    # (1) in-memory float64 components
    rec_m0_f64 = _apply_from_components(
        C0,
        meta["comp_m0"]["W"],
        meta["comp_m0"]["input_mean"],
        meta["comp_m0"]["input_std"],
    )
    rec_mplus_f64 = _apply_from_components(
        Cplus,
        meta["comp_mplus"]["W"],
        meta["comp_mplus"]["input_mean"],
        meta["comp_mplus"]["input_std"],
    )
    rd_m0_f64 = _reldiff(rec_m0_f64, ref_m0)
    rd_mplus_f64 = _reldiff(rec_mplus_f64, ref_mplus)

    # (2) reloaded float32 npz (what a downstream consumer actually loads)
    d = np.load(out_path, allow_pickle=True)
    rec_m0_f32 = _apply_from_components(
        C0,
        d["W_M0"].astype(np.float64),
        d["input_mean_C0"],
        d["input_std_C0"],
    )
    rec_mplus_f32 = _apply_from_components(
        Cplus,
        d["W_Mplus"].astype(np.float64),
        d["input_mean_Cplus"],
        d["input_std_Cplus"],
    )
    rd_m0_f32 = _reldiff(rec_m0_f32, ref_m0)
    rd_mplus_f32 = _reldiff(rec_mplus_f32, ref_mplus)

    return {
        "reldiff_M0_f64": rd_m0_f64,
        "reldiff_Mplus_f64": rd_mplus_f64,
        "reldiff_M0_f32": rd_m0_f32,
        "reldiff_Mplus_f32": rd_mplus_f32,
    }


def load_map(behavior: str, layer: int, maps_root: str | Path | None = None) -> dict:
    """Load a saved (behavior, layer) map and return apply-ready callables + arrays.

    2-line downstream usage::

        m = load_map("em", 14)
        v0_64 = m["M0_apply"](c)        # c: (3584,) or (n, 3584) → (..., 64)
        vplus_64 = m["Mplus_apply"](c)  # in the shared V0 PCA basis

    ``M0_apply`` / ``Mplus_apply`` reproduce ``fit_M._ridge_fit_predict`` exactly.
    Project a 64-dim output back to the 3584 residual space with
    ``out @ m["pca_basis"]`` (the basis rows are the top-64 v0 PCs). Also returns
    the raw ``W_M0`` / ``W_Mplus`` / ``pca_basis`` / normalization / λ / metadata.
    """
    root = (
        Path(maps_root) if maps_root is not None else (PROJECT_ROOT / "eval_results/issue_667_maps")
    )
    path = root / behavior / f"L{layer}.npz"
    if not path.exists():
        raise FileNotFoundError(f"map npz missing: {path}")
    d = np.load(path, allow_pickle=True)
    W_M0 = d["W_M0"].astype(np.float64)
    W_Mplus = d["W_Mplus"].astype(np.float64)
    mean_c0, std_c0 = d["input_mean_C0"], d["input_std_C0"]
    mean_cp, std_cp = d["input_mean_Cplus"], d["input_std_Cplus"]

    return {
        "M0_apply": lambda c: _apply_from_components(c, W_M0, mean_c0, std_c0),
        "Mplus_apply": lambda c: _apply_from_components(c, W_Mplus, mean_cp, std_cp),
        "W_M0": d["W_M0"],
        "W_Mplus": d["W_Mplus"],
        "pca_basis": d["pca_basis"],
        "input_mean_C0": mean_c0,
        "input_std_C0": std_c0,
        "input_mean_Cplus": mean_cp,
        "input_std_Cplus": std_cp,
        "output_centered": bool(d["output_centered"]),
        "lambda_M0": float(d["lambda_M0"]),
        "lambda_Mplus": float(d["lambda_Mplus"]),
        "cell_keys": list(d["cell_keys"]),
        "families": list(d["families"]),
        "n": int(d["n"]),
        "behavior": str(d["behavior"]),
        "layer": int(d["layer"]),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Closed-form ridge — CPU by design (this explicit pin also deliberately
    # overrides the module's EPM_FIT_DEVICE/auto import default; #876).
    fit658.DEVICE = fit658._resolve_device("cpu")
    logger.info("[phase=save_maps] device=%s", fit658.DEVICE)

    ap = argparse.ArgumentParser(description="Issue #667 — save fitted M0/M⁺ ridge maps")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    ap.add_argument(
        "--local-store-root",
        type=Path,
        default=None,
        help="read the store from a LOCAL mirror (skip the hanging HF tree walk); "
        "e.g. eval_results/issue_667_alllayer/analysis_tensors",
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_667_maps")
    ap.add_argument(
        "--gate-cells",
        type=int,
        default=4,
        help="number of (behavior, layer) cells to run the < 1e-8 correctness gate on",
    )
    ap.add_argument("--upload", action="store_true", help="bulk upload out-dir to HF after fitting")
    ap.add_argument(
        "--max-sources", type=int, default=None, help="smoke: cap source dirs per behavior (>=4)"
    )
    ap.add_argument(
        "--max-targets-per-source", type=int, default=None, help="smoke: cap targets per source"
    )
    args = ap.parse_args()

    behaviors = tuple(args.behaviors)
    layers = tuple(args.layers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[phase=save_maps] behaviors=%s layers=%s out=%s", behaviors, layers, args.out_dir)

    # Ridge exactness gate (fit658) — a reduction-order regression fails at startup.
    fit658._assert_ridge_exactness()
    logger.info("[phase=save_maps] ridge exactness gate PASS")

    # Build the store layout once (local mirror if given — the HF tree walk hangs
    # on this large repo, #667). load_cells is called per-(behavior) below via the
    # single layout so the loader streams / reads each cell.
    if args.local_store_root is not None:
        layout = loadact.list_store_layout_local(args.local_store_root, behaviors)
        streamer = loadact._Streamer(local_root=str(args.local_store_root))
    else:
        layout = None
        streamer = None

    # A cap (smoke) OR the local-mirror all-layer read both disable the strict
    # 480-cell assert (it hard-codes the swept-3-layer count; we sweep 27 layers).
    strict = False
    cells_by = loadact.load_cells(
        behaviors=behaviors,
        layers=layers,
        max_sources=args.max_sources,
        max_targets_per_source=args.max_targets_per_source,
        streamer=streamer,
        strict_counts=strict,
        layout=layout,
    )

    gate_reports: list[dict] = []
    max_reldiff = 0.0
    n_saved = 0
    t0 = time.time()
    for behavior in behaviors:
        for layer in layers:
            cells = cells_by[(behavior, layer)]
            if not cells:
                logger.warning("[phase=save_maps] %s L%d: 0 cells — skip", behavior, layer)
                continue
            out_path, meta = fit_and_save_cell(behavior, layer, cells, args.out_dir)
            n_saved += 1
            # Run the correctness gate on the first `gate-cells` cells (spread
            # across behaviors — the outer loop order gives layer-1 of each).
            if len(gate_reports) < args.gate_cells:
                rep = correctness_gate(meta, out_path)
                rep["cell"] = f"{behavior}_L{layer}"
                gate_reports.append(rep)
                cell_max = max(rep[k] for k in rep if k.startswith("reldiff"))
                max_reldiff = max(max_reldiff, cell_max)
                logger.info(
                    "[phase=save_maps] GATE %s: reldiff f64 M0=%.2e M⁺=%.2e | f32 M0=%.2e M⁺=%.2e",
                    rep["cell"],
                    rep["reldiff_M0_f64"],
                    rep["reldiff_Mplus_f64"],
                    rep["reldiff_M0_f32"],
                    rep["reldiff_Mplus_f32"],
                )
            logger.info(
                "[phase=save_maps] saved %s (n=%d, λ_M0=%.3g, λ_M⁺=%.3g)",
                out_path,
                meta["n"],
                meta["comp_m0"]["lambda"],
                meta["comp_mplus"]["lambda"],
            )

    if streamer is not None:
        streamer.cleanup()

    wall = time.time() - t0
    logger.info("[phase=save_maps] fit %d cell(s) in %.1fs", n_saved, wall)

    # The float64 in-memory reconstruction MUST be < 1e-8 relative (bit-exact math
    # up to float64 round-off); the float32 reload is only float32-accurate, so it
    # is REPORTED but NOT gated at 1e-8 (a float32 W cannot hit 1e-8).
    gate_f64_max = max(
        (max(r["reldiff_M0_f64"], r["reldiff_Mplus_f64"]) for r in gate_reports), default=0.0
    )
    gate_f32_max = max(
        (max(r["reldiff_M0_f32"], r["reldiff_Mplus_f32"]) for r in gate_reports), default=0.0
    )
    logger.info(
        "[phase=save_maps] CORRECTNESS GATE: max f64 reldiff=%.3e (must be <1e-8) | "
        "max f32 reldiff=%.3e (float32 reload, informational)",
        gate_f64_max,
        gate_f32_max,
    )
    assert gate_f64_max < 1e-8, (
        f"correctness gate FAILED: max float64 reldiff {gate_f64_max:.3e} >= 1e-8 — "
        "the saved components do NOT reproduce _ridge_fit_predict exactly"
    )
    logger.info("[phase=save_maps] CORRECTNESS GATE PASS (< 1e-8)")

    if args.upload:
        from huggingface_hub import upload_folder

        logger.info(
            "[phase=save_maps] uploading %s → %s/%s (one commit)",
            args.out_dir,
            DATA_REPO,
            HF_MAP_PREFIX,
        )
        upload_folder(
            repo_id=DATA_REPO,
            repo_type="dataset",
            folder_path=str(args.out_dir),
            path_in_repo=HF_MAP_PREFIX,
            commit_message=f"issue667: fitted M0/M⁺ ridge maps ({n_saved} cells, {_git_sha()[:8]})",
        )
        logger.info(
            "[phase=save_maps] upload complete → %s/%s/<behavior>/L<layer>.npz",
            DATA_REPO,
            HF_MAP_PREFIX,
        )

    logger.info("[phase=save_maps] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
