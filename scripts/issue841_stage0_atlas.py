#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ℓ, σ, →, ‖·‖) in scientific docstrings/log messages.
"""Issue #841 Stage 0 — per-layer Δ-predictability atlas (behavior-agnostic).

Over #779's cached ``pass_b`` LMSYS depth trajectories (``cx_last`` (N,28,3584)):

1. Norm curve FIRST (§4.1) → ``eval_results/issue_841/norm_curve.json`` (per-layer
   ‖h‖, ‖Δ‖, ‖Δ‖/‖h‖, adjacent cosine, per-block σ_m).
2. Position-distribution validation (§4.2): ridge Δ-atlas on ``cx_last`` vs
   ``cx_mean`` — do the per-transition R² curves track?
3. Four-class atlas (§4.3) over the 27 one-step transitions × {raw, RMS-norm}
   target spaces: identity (predict-zero, R²≡0) / ridge / MLP / depth-GRU. PRIMARY
   metric = identity-relative R² (predict-zero ≡ 0); COMPANION = mean-centered R²;
   plus median/p90/p99 raw Δ-error. Split 4000 fit / 500 inner-val (MLP+GRU early
   stop) / 500 test (atlas reporting ONLY), seed 42.
4. Data-scaling curve (§4.3): re-fit ridge + MLP at n ∈ {500,1000,2000,4000}
   nested subsamples of the fit split → held-out R² vs n (the §7 capture trigger
   input).

--smoke fits ONE transition (ℓ=13→14) for all four classes on a 200-context
subset — the SAME code path (dispatcher/loaders/logging) as the full run, one
"cell" (transition) instead of 27. Persists ``eval_results/issue_841/
stage0_atlas.json``.

No Qwen weights, no new judging — analysis over cached tensors only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue841_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    SplitMLPGroup,
    assert_split_mlp_matches_serial,
)
from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_stage0")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_841"
SPACES = ("raw", "rmsnorm")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA device; falling back to cpu")
        return "cpu"
    return requested


def _target(delta: np.ndarray, space: str, sigma: float) -> np.ndarray:
    """Δ in the requested fit space (raw or RMS-normalized by σ_m)."""
    return delta if space == "raw" else delta / sigma


def _atlas_cell(pred_fit, delta_test_fit, delta_test_raw, train_mean_fit, space, sigma) -> dict:
    """Per (transition, class, space) atlas metrics from fit-space predictions."""
    pred_raw = pred_fit if space == "raw" else pred_fit * sigma
    return {
        "r2_id": MP.identity_relative_r2(pred_fit, delta_test_fit),
        "r2_meancentered": MP.mean_centered_r2(pred_fit, delta_test_fit, train_mean_fit),
        "delta_err_raw": MP.delta_error_percentiles(pred_raw, delta_test_raw),
    }


# ── §4.1 norm curve ────────────────────────────────────────────────────────────


def compute_norm_curve(cx_last: np.ndarray) -> dict:
    nc = MP.norm_curve(cx_last)
    C.write_json_atomic(
        EVAL_DIR / "norm_curve.json",
        {"norm_curve": nc, "metadata": C.reproducibility_metadata({"phase": "norm_curve"})},
    )
    logger.info(
        "[norm_curve] wrote %s (σ_m range %.3g..%.3g)",
        EVAL_DIR / "norm_curve.json",
        min(nc["sigma_block_rms"]),
        max(nc["sigma_block_rms"]),
    )
    return nc


# ── §4.3 ridge / mlp / gru atlas ──────────────────────────────────────────────


def ridge_atlas(cx, split, transitions, sigma, device) -> dict:
    """Ridge Δ-atlas over transitions × {raw, rmsnorm}."""
    out: dict = {}
    for space in SPACES:
        out[space] = {}
        for t in transitions:
            h, delta = MP.deltas_at(cx, t)
            y_fit = _target(delta[split["fit"]], space, sigma[t])
            pred_fit, _rmap = MP.fit_ridge_split(
                h[split["fit"]], y_fit, h[split["test"]], sigma=sigma[t], device=device
            )
            delta_test_fit = _target(delta[split["test"]], space, sigma[t])
            train_mean = y_fit.mean(axis=0)
            cell = _atlas_cell(
                pred_fit, delta_test_fit, delta[split["test"]], train_mean, space, sigma[t]
            )
            cell["best_lam"] = _rmap.best_lam
            out[space][f"transition_{t}"] = cell
        logger.info("[ridge] space=%s done (%d transitions)", space, len(transitions))
    return out


def mlp_atlas(cx, split, transitions, sigma, device, chunk_size, num_threads, max_epochs) -> dict:
    """MLP Δ-atlas over transitions × {raw, rmsnorm} — ONE batched split ensemble."""
    groups, meta = [], {}
    for space in SPACES:
        for t in transitions:
            h, delta = MP.deltas_at(cx, t)
            y_fit = _target(delta[split["fit"]], space, sigma[t]).astype(np.float32)
            y_val = _target(delta[split["val"]], space, sigma[t]).astype(np.float32)
            key = (space, t)
            groups.append(
                SplitMLPGroup(
                    key=key,
                    X_train=h[split["fit"]].astype(np.float32),
                    Y_train=y_fit,
                    X_eval=h[split["test"]].astype(np.float32),
                    X_val=h[split["val"]].astype(np.float32),
                    Y_val=y_val,
                )
            )
            meta[key] = (space, t)
    preds_by_key, _params = MP.fit_split_mlps(
        groups, device=device, chunk_size=chunk_size, num_threads=num_threads, max_epochs=max_epochs
    )
    out: dict = {s: {} for s in SPACES}
    for (space, t), pred_fit in preds_by_key.items():
        _h, delta = MP.deltas_at(cx, t)
        delta_test_fit = _target(delta[split["test"]], space, sigma[t])
        train_mean = _target(delta[split["fit"]], space, sigma[t]).mean(axis=0)
        out[space][f"transition_{t}"] = _atlas_cell(
            pred_fit, delta_test_fit, delta[split["test"]], train_mean, space, sigma[t]
        )
    logger.info("[mlp] %d batched groups done", len(groups))
    return out


def gru_atlas(cx, split, transitions, sigma, device, max_epochs, batch_size) -> dict:
    """Depth-GRU Δ-atlas (EXPLORATORY) over BOTH {raw, rmsnorm} target spaces.

    §6.5 requires the atlas per (transition, class, raw/RMS-norm), so two GRUs are
    fit: the raw-target GRU (sigma≡1 → predicts raw Δ) and the RMS-normalized GRU
    (sigma → predicts Δ/σ_m, the shared-depth-scale variant Stage-1 transport
    also uses). One extra GRU fit; per-transition R² read per space.
    """
    n_trans = cx.shape[1] - 1
    out: dict = {s: {} for s in SPACES}
    for space in SPACES:
        # sigma passed to fit_depth_gru IS the per-transition target normalizer:
        # ones ⇒ raw Δ target; the measured σ_m ⇒ RMS-normalized target.
        fit_sigma = np.ones(n_trans, dtype=np.float64) if space == "raw" else sigma
        gru = MP.fit_depth_gru(
            cx[split["fit"]],
            cx[split["val"]],
            fit_sigma,
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
        with torch.no_grad():
            test_in = torch.from_numpy(np.ascontiguousarray(cx[split["test"], :n_trans, :])).to(
                device=device, dtype=torch.float32
            )
            pred_fit_all = gru(test_in).cpu().numpy()  # (n_test, n_trans, d) in fit space
        for t in transitions:
            _h, delta = MP.deltas_at(cx, t)
            delta_test_fit = _target(delta[split["test"]], space, sigma[t])
            train_mean = _target(delta[split["fit"]], space, sigma[t]).mean(axis=0)
            out[space][f"transition_{t}"] = _atlas_cell(
                pred_fit_all[:, t, :],
                delta_test_fit,
                delta[split["test"]],
                train_mean,
                space,
                sigma[t],
            )
        logger.info("[gru] space=%s atlas done (%d transitions)", space, len(transitions))
    return out


# ── §4.3 data-scaling curve (ridge + MLP, raw space) ──────────────────────────


def scaling_curve(
    cx, split, transitions, sigma, device, ns, chunk_size, num_threads, max_epochs
) -> dict:
    """Held-out identity-relative R² vs n ∈ ns (nested subsamples of the fit split).

    Raw target space (the transport-relevant space). Ridge + MLP per (transition,
    n). Nested: the n=500 subsample ⊂ n=1000 ⊂ ... (first-n of the fit indices),
    seed 42; SAME inner-val + test sets. Feeds the §7 capture trigger.
    """
    fit_idx = split["fit"]
    # Nested subsamples <= the fit size, ALWAYS including the full fit size so the
    # curve has its top anchor (and the smoke, whose fit split is < 500, still
    # exercises the path at one point).
    ns = sorted({n for n in ns if n < len(fit_idx)} | {len(fit_idx)})
    curve: dict = {"ridge": {}, "mlp": {}}
    for t in transitions:
        curve["ridge"][f"transition_{t}"] = {}
        curve["mlp"][f"transition_{t}"] = {}
    # Ridge: per (transition, n).
    for n in ns:
        sub = fit_idx[:n]
        for t in transitions:
            h, delta = MP.deltas_at(cx, t)
            pred, _ = MP.fit_ridge_split(
                h[sub], delta[sub], h[split["test"]], sigma=1.0, device=device
            )
            curve["ridge"][f"transition_{t}"][str(n)] = MP.identity_relative_r2(
                pred, delta[split["test"]]
            )
    # MLP: batched per n (all transitions ride one ensemble at that n).
    for n in ns:
        sub = fit_idx[:n]
        groups = []
        for t in transitions:
            h, delta = MP.deltas_at(cx, t)
            groups.append(
                SplitMLPGroup(
                    key=("mlp", t),
                    X_train=h[sub].astype(np.float32),
                    Y_train=delta[sub].astype(np.float32),
                    X_eval=h[split["test"]].astype(np.float32),
                    X_val=h[split["val"]].astype(np.float32),
                    Y_val=delta[split["val"]].astype(np.float32),
                )
            )
        preds, _ = MP.fit_split_mlps(
            groups,
            device=device,
            chunk_size=chunk_size,
            num_threads=num_threads,
            max_epochs=max_epochs,
        )
        for (_, t), pred in preds.items():
            _h, delta = MP.deltas_at(cx, t)
            curve["mlp"][f"transition_{t}"][str(n)] = MP.identity_relative_r2(
                pred, delta[split["test"]]
            )
        logger.info("[scaling] n=%d done (ridge+mlp, %d transitions)", n, len(transitions))
    curve["ns"] = ns
    return curve


# ── §4.2 position validation ──────────────────────────────────────────────────


def position_validation(pass_b, split, transitions, sigma, device, cx_last_ridge) -> dict:
    """Ridge Δ-atlas on cx_last vs cx_mean — do the per-transition R² curves track?

    cx_last raw-ridge R² is reused from the main atlas; only cx_mean raw-ridge is
    fit here. Reports both curves + their Pearson r.
    """
    cx_mean = pass_b["cx_mean"]
    last_curve = [cx_last_ridge["raw"][f"transition_{t}"]["r2_id"] for t in transitions]
    mean_curve = []
    for t in transitions:
        h, delta = MP.deltas_at(cx_mean, t)
        pred, _ = MP.fit_ridge_split(
            h[split["fit"]], delta[split["fit"]], h[split["test"]], sigma=1.0, device=device
        )
        mean_curve.append(MP.identity_relative_r2(pred, delta[split["test"]]))
    lc, mc = np.array(last_curve), np.array(mean_curve)
    fin = np.isfinite(lc) & np.isfinite(mc)
    r = float(np.corrcoef(lc[fin], mc[fin])[0, 1]) if fin.sum() >= 3 else float("nan")
    logger.info("[position] cx_last vs cx_mean ridge-R² curve Pearson r = %.3f", r)
    return {
        "transitions": list(transitions),
        "cx_last_ridge_r2_id": last_curve,
        "cx_mean_ridge_r2_id": mean_curve,
        "curve_pearson_r": r,
        "note": (
            "cheap last-vs-mean position proxy (§4.2); a true per-position sweep "
            "is DEFERRED (needs a Qwen forward pass, out of scope per the "
            "no-new-capture constraint)."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 Stage 0 Δ-predictability atlas.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-contexts", type=int, default=0, help="0 = all pass_b contexts")
    ap.add_argument("--mlp-chunk-size", type=int, default=8)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--gru-epochs", type=int, default=300)
    ap.add_argument("--gru-batch-size", type=int, default=512)
    ap.add_argument(
        "--mlp-epochs",
        type=int,
        default=0,
        help="0 = production default (MLP_MAX_EPOCHS=300); a small value keeps the smoke fast",
    )
    ap.add_argument("--no-gru", action="store_true", help="skip the exploratory GRU class")
    args = ap.parse_args()
    mlp_epochs = args.mlp_epochs or None  # None => fit_split_mlps uses the production default

    device = _resolve_device(args.device)
    logger.info("device=%s smoke=%s", device, args.smoke)

    # Unit-equivalence gate: the batched split MLP must reproduce a serial _MLP
    # (plan §12 assumption 8) BEFORE the atlas trusts it.
    parity = assert_split_mlp_matches_serial()
    logger.info("[parity] split-MLP vs serial _MLP: %s", parity)

    pass_b = C.load_pass_b()
    cx = pass_b["cx_last"]
    n_total = cx.shape[0]
    if args.smoke:
        cap = min(args.n_contexts or 200, n_total)  # --n-contexts shrinks the smoke slice
        cx = cx[:cap]
        pass_b["cx_mean"] = pass_b["cx_mean"][:cap]
        n_total = cx.shape[0]
    elif args.n_contexts:
        cx = cx[: args.n_contexts]
        pass_b["cx_mean"] = pass_b["cx_mean"][: args.n_contexts]
        n_total = cx.shape[0]

    transitions = [13] if args.smoke else list(range(C.N_TRANSITIONS))
    split = C.make_split(
        n_total, n_fit=C.N_FIT, n_val=C.N_INNER_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED
    )
    logger.info(
        "[split] N=%d fit=%d val=%d test=%d transitions=%s",
        n_total,
        len(split["fit"]),
        len(split["val"]),
        len(split["test"]),
        transitions,
    )

    nc = compute_norm_curve(cx)
    sigma = np.asarray(nc["sigma_block_rms"], dtype=np.float64)

    result: dict = {
        "split": {
            "n_total": n_total,
            "n_fit": len(split["fit"]),
            "n_val": len(split["val"]),
            "n_test": len(split["test"]),
            "seed": C.SPLIT_SEED,
        },
        "transitions": transitions,
        "target_spaces": list(SPACES),
        "parity_check": parity,
        "atlas": {},
        "metadata": C.reproducibility_metadata({"phase": "stage0_atlas", "smoke": args.smoke}),
    }

    # Identity (predict-zero) is R²_id ≡ 0 by construction — recorded as the null.
    result["atlas"]["identity"] = {
        s: {f"transition_{t}": {"r2_id": 0.0, "note": "predict-zero null"} for t in transitions}
        for s in SPACES
    }
    C.write_json_atomic(EVAL_DIR / "stage0_atlas.json", result)

    r_atlas = ridge_atlas(cx, split, transitions, sigma, device)
    result["atlas"]["ridge"] = r_atlas
    C.write_json_atomic(EVAL_DIR / "stage0_atlas.json", result)  # checkpoint per class

    result["atlas"]["mlp"] = mlp_atlas(
        cx, split, transitions, sigma, device, args.mlp_chunk_size, args.num_threads, mlp_epochs
    )
    C.write_json_atomic(EVAL_DIR / "stage0_atlas.json", result)

    if not args.no_gru:
        result["atlas"]["gru"] = gru_atlas(
            cx, split, transitions, sigma, device, args.gru_epochs, args.gru_batch_size
        )
        C.write_json_atomic(EVAL_DIR / "stage0_atlas.json", result)

    result["position_validation"] = position_validation(
        pass_b, split, transitions, sigma, device, r_atlas
    )
    C.write_json_atomic(EVAL_DIR / "stage0_atlas.json", result)

    result["scaling_curve"] = scaling_curve(
        cx,
        split,
        transitions,
        sigma,
        device,
        list(C.SCALING_NS),
        args.mlp_chunk_size,
        args.num_threads,
        mlp_epochs,
    )
    C.write_json_atomic(EVAL_DIR / "stage0_atlas.json", result)

    logger.info("[done] wrote %s", EVAL_DIR / "stage0_atlas.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
