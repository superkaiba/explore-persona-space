#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ℓ, σ, →, R²) in scientific docstrings/log messages.
"""Issue #841 scaling — Stage-0 data-scaling curve at extended fit-set sizes.

Re-fits the parent's one-step next-activation maps (ridge + MLP) at nested
fit-set sizes n ∈ {4k,10k,25k,50k,100k} on the parent's FIXED 500-context test
set, reads the H1 decision metric r2_id(n) (identity-relative, raw space — the
metric the §7 capture trigger fired on) plus the r2_meancentered(n) diagnostic
column, and PERSISTS every fitted RAW ridge map for the Stage-1 transport curve.

Single manipulated variable = fit-corpus size; everything else (ridge recipe,
MLP recipe, split, target spaces) matches the parent (plan v9 §3).

Gates (plan §7):
  KILL-B(i)  primal-vs-dual parity at the anchor n : |ΔR²|<0.01 per transition, HARD FAIL.
  KILL-B(ii) dual-vs-primal cross-check at --xcheck-n : max|ΔR²|<0.01, HARD FAIL.
  anchor gate : n=anchor ridge r2_id reproduces the parent's stored stage0_atlas
                ridge r2_id per transition (same rows+solver ⇒ identical); >0.01 FAIL.
  position-drift diagnostic : a same-size fit on the LATEST new-stream window vs the
                anchor, on the FIXED test — gates a FLAT-H1 attribution (does NOT halt).

Solver: dual (#658 m×m Gram) for n ≤ --dual-max, primal (d×d, exact PRESS) above
— the dual Gram is n×n fp64 (~80 GB at 100k), infeasible.

--smoke / --synthetic-parent/--synthetic-new fabricate a tiny CPU bundle (no 6 GB
download, no GPU) that runs the FULL fit path incl. both KILL-B legs end-to-end.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue841_common as C  # noqa: E402
import issue841_scaling_common as S  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import SplitMLPGroup  # noqa: E402
from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_scaling_stage0")

SPACES = ("raw", "rmsnorm")
# The data-limited read-out band (plan §2): transitions 17-25, feeding evil L20 /
# sycophancy L26(t25) / hallucination L17. H1's IMPROVED/FLAT read is on this band.
DATA_LIMITED_BAND = tuple(range(17, 26))
KILL_B_TOL = 0.01


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA device; falling back to cpu")
        return "cpu"
    return requested


def _deltas(cx: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    return cx[:, t, :], cx[:, t + 1, :] - cx[:, t, :]


def _target(delta: np.ndarray, space: str, sigma_t: float) -> np.ndarray:
    return delta if space == "raw" else delta / sigma_t


def _fit_ridge(h_fit, y_fit, h_test, *, sigma, device, dual_max, n):
    """Dispatch to the dual (n≤dual_max) or primal (n>dual_max) ridge solver."""
    if n <= dual_max:
        return MP.fit_ridge_split(h_fit, y_fit, h_test, sigma=sigma, device=device)
    return MP.fit_ridge_primal(h_fit, y_fit, h_test, sigma=sigma, device=device)


# ── ridge scaling curve + per-n maps ─────────────────────────────────────────────


def ridge_scaling(fit_pool, test, transitions, ns, *, device, dual_max):
    """Ridge r2_id(n) + r2_meancentered(n) per (space, transition, n); RAW maps per n.

    σ_m is re-measured on fit(n) (plan §4.2). Returns
    ``(curve, ridge_maps_by_n)`` where ``curve[space][transition_t][str(n)] =
    {r2_id, r2_meancentered, best_lam}`` and ``ridge_maps_by_n[n]`` is the
    RAW-target one-step map dict (transition → RidgeMap) for Stage-1.
    """
    curve: dict = {s: {f"transition_{t}": {} for t in transitions} for s in SPACES}
    ridge_maps_by_n: dict[int, dict] = {}
    for n in ns:
        fit = fit_pool[:n]
        sigma_n = np.asarray(MP.norm_curve(fit)["sigma_block_rms"], dtype=np.float64)
        raw_maps: dict = {}
        for space in SPACES:
            for t in transitions:
                h_fit, d_fit = _deltas(fit, t)
                h_test, d_test = _deltas(test, t)
                y_fit = _target(d_fit, space, sigma_n[t])
                pred_fit, rmap = _fit_ridge(
                    h_fit,
                    y_fit,
                    h_test,
                    sigma=sigma_n[t] if space == "rmsnorm" else 1.0,
                    device=device,
                    dual_max=dual_max,
                    n=n,
                )
                d_test_fit = _target(d_test, space, sigma_n[t])
                curve[space][f"transition_{t}"][str(n)] = {
                    "r2_id": MP.identity_relative_r2(pred_fit, d_test_fit),
                    "r2_meancentered": MP.mean_centered_r2(pred_fit, d_test_fit, y_fit.mean(0)),
                    "best_lam": rmap.best_lam,
                }
                if space == "raw":
                    raw_maps[t] = rmap  # RAW maps drive Stage-1 (additive transport)
        ridge_maps_by_n[n] = raw_maps
        logger.info(
            "[ridge] n=%d done (%d transitions × %d spaces)", n, len(transitions), len(SPACES)
        )
    curve["ns"] = list(ns)
    return curve, ridge_maps_by_n


# ── MLP scaling curve (batched ensemble per n) ───────────────────────────────────


def _log_mlp_ram_sizing(fit_pool, ns, transitions, group_chunk, n_eval):
    """Pre-phase host-RAM projection + soft check (crash-fix cycle 5). Returns the
    projected peak GiB and warns if it exceeds 80% of MemAvailable, so a too-large chunk
    is caught before the OOM SIGKILL instead of after.

    The projection counts the caller-side deltas AND the helper's own re-copies:
    ``fit_batched_split_mlp``'s ``_stack()`` (vectorized_mlp_skill.py) materializes FULL
    contiguous fp32 stacks of X_train, Y_train AND X_eval across the chunk's groups
    (``np.stack(...).astype(np.float32)`` + ``ascontiguousarray``), so the caller's
    ``astype(copy=False)`` view is RE-COPIED inside the helper regardless — the view only
    saves the CALLER copy, not the helper stack. Live chunk footprint ≈ 3× the
    train-sized term (caller Y_train delta + helper X-stack + helper Y-stack, each
    ``group_chunk × n × H × 4``) + the eval term (``group_chunk × n_eval × H × 4``);
    the small n_val val-stacks are folded in as comparable to the eval term."""
    n_max = max(ns)
    hidden = fit_pool.shape[-1]
    pool_gib = fit_pool.nbytes / (1024**3)
    train_term = group_chunk * n_max * hidden * 4 / (1024**3)  # one (n, H) fp32 stack
    eval_term = group_chunk * n_eval * hidden * 4 / (1024**3)
    # caller Y_train delta + helper X-stack + helper Y-stack (3× train) + helper X_eval-stack.
    chunk_gib = 3 * train_term + eval_term
    proj_gib = pool_gib + chunk_gib
    mem_total, mem_avail = S.mem_total_available_gib()
    logger.info(
        "[stage0] mlp RAM sizing: pool %.1f + chunk-copies %.1f (3×train %.1f + eval %.1f) "
        "= proj peak %.1f GiB; MemTotal %.0f / MemAvailable %.0f GiB (group_chunk=%d, n_max=%d)",
        pool_gib,
        chunk_gib,
        3 * train_term,
        eval_term,
        proj_gib,
        mem_total,
        mem_avail,
        group_chunk,
        n_max,
    )
    if mem_avail == mem_avail and proj_gib > 0.8 * mem_avail:  # NaN-guarded
        logger.warning(
            "[stage0] projected MLP peak %.1f GiB > 80%% of MemAvailable %.1f GiB — "
            "lower EPM_I841S_MLP_GROUP_CHUNK (currently %d) or route off-VM",
            proj_gib,
            mem_avail,
            group_chunk,
        )
    return proj_gib


def mlp_scaling(
    fit_pool,
    val,
    test,
    transitions,
    ns,
    *,
    device,
    chunk_size,
    num_threads,
    max_epochs,
    group_chunk=None,
):
    """MLP r2_id(n) + r2_meancentered(n) per (transition, n), RAW space, batched.

    Fits transitions in CHUNKS of ``group_chunk`` SplitMLPGroups per fit_split_mlps
    call (default ``S.MLP_GROUP_CHUNK``) to bound host RAM: at n=100k each group holds
    an (n, 3584) fp32 X_train + Y_train, so building ALL 27 at once is ~77 GB of copies
    (#841 attempt 6 SIGKILL 137). ``astype(copy=False)`` keeps X_train a view into the
    pool + Y_train the fresh delta (no redundant copy); each chunk is freed + gc'd
    before the next.

    PARTITION INVARIANCE (#926): since ``4dfcba056f`` ("port split-MLP fitter to main
    with partition-invariant per-group seeding"), ``fit_batched_split_mlp`` seeds each
    group under ``split_group_init_seed(seed, group.key)``, which depends only on
    ``(seed, key)`` — never on batch position or chunking — so the r2 curve is
    bit-identical across ``group_chunk`` values and to a single all-groups call. The
    fit is also DETERMINISTIC + reproducible at a FIXED ``group_chunk`` (same chunk
    size → same curve). ``group_chunk`` is a pinned RAM knob recorded in the output,
    NOT a nuisance seed; it bounds peak host RAM without changing the numbers.

    Returns ``(curve, params_by_n)`` — ``params_by_n[n][t]`` is the numpy param dict
    Stage-1 rebuilds into the row-1 mlp transported class (RAW space).
    """
    if group_chunk is None:
        group_chunk = S.MLP_GROUP_CHUNK
    if group_chunk <= 0:
        raise ValueError(
            f"group_chunk must be >= 1, got {group_chunk} (check EPM_I841S_MLP_GROUP_CHUNK)"
        )
    _log_mlp_ram_sizing(fit_pool, ns, transitions, group_chunk, test.shape[0])
    curve: dict = {f"transition_{t}": {} for t in transitions}
    params_by_n: dict[int, dict] = {}
    for n in ns:
        fit = fit_pool[:n]
        params_n: dict = {}
        n_chunks = (len(transitions) + group_chunk - 1) // group_chunk
        for ci, lo in enumerate(range(0, len(transitions), group_chunk)):
            chunk_ts = transitions[lo : lo + group_chunk]
            groups = [
                SplitMLPGroup(
                    key=("mlp", t),
                    X_train=_deltas(fit, t)[0].astype(np.float32, copy=False),
                    Y_train=_deltas(fit, t)[1].astype(np.float32, copy=False),
                    X_eval=_deltas(test, t)[0].astype(np.float32, copy=False),
                    X_val=_deltas(val, t)[0].astype(np.float32, copy=False),
                    Y_val=_deltas(val, t)[1].astype(np.float32, copy=False),
                )
                for t in chunk_ts
            ]
            preds, params = MP.fit_split_mlps(
                groups,
                device=device,
                chunk_size=chunk_size,
                num_threads=num_threads,
                max_epochs=max_epochs,
            )
            for (_, t), pred in preds.items():
                _h, d_test = _deltas(test, t)
                _hf, d_fit = _deltas(fit, t)
                curve[f"transition_{t}"][str(n)] = {
                    "r2_id": MP.identity_relative_r2(pred, d_test),
                    "r2_meancentered": MP.mean_centered_r2(pred, d_test, d_fit.mean(0)),
                }
                params_n[t] = params[("mlp", t)]
            del groups, preds, params
            gc.collect()
            logger.info(
                "[stage0] mlp chunk %d/%d (transitions %d..%d) done, RSS %.1f GiB",
                ci + 1,
                n_chunks,
                chunk_ts[0],
                chunk_ts[-1],
                S.rss_gib(),
            )
        params_by_n[n] = params_n
        logger.info("[mlp] n=%d done (%d transitions, %d chunks)", n, len(transitions), n_chunks)
    curve["ns"] = list(ns)
    return curve, params_by_n


# ── KILL-B parity gates ──────────────────────────────────────────────────────────


def _ridge_r2_both_solvers(fit_pool, test, transitions, n, device) -> dict:
    """Per-transition RAW r2_id from BOTH solvers on fit(n) (for a parity check)."""
    fit = fit_pool[:n]
    out = {}
    for t in transitions:
        h_fit, d_fit = _deltas(fit, t)
        h_test, d_test = _deltas(test, t)
        pred_d, _ = MP.fit_ridge_split(h_fit, d_fit, h_test, sigma=1.0, device=device)
        pred_p, _ = MP.fit_ridge_primal(h_fit, d_fit, h_test, sigma=1.0, device=device)
        out[t] = {
            "dual": MP.identity_relative_r2(pred_d, d_test),
            "primal": MP.identity_relative_r2(pred_p, d_test),
        }
    return out


def kill_b_parity(fit_pool, test, transitions, anchor_n, xcheck_n, device) -> dict:
    """KILL-B: primal-vs-dual parity at the anchor n AND at the cross-check n.

    Both are exact-ridge identities (the primal reproduces the dual to fp64), so a
    |ΔR²| ≥ KILL_B_TOL means the primal PRESS-LOO implementation is suspect — HARD
    FAIL (raise), never a silently-unanchored curve.
    """
    result = {}
    for label, n in (("anchor", anchor_n), ("xcheck", xcheck_n)):
        both = _ridge_r2_both_solvers(fit_pool, test, transitions, n, device)
        diffs = {t: abs(v["dual"] - v["primal"]) for t, v in both.items()}
        worst = max(diffs.values()) if diffs else 0.0
        result[label] = {"n": n, "worst_abs_dr2": worst, "per_transition": both, "tol": KILL_B_TOL}
        logger.info(
            "[KILL-B/%s] n=%d worst |ΔR²(dual,primal)| = %.4g (tol %.2g)",
            label,
            n,
            worst,
            KILL_B_TOL,
        )
        if worst >= KILL_B_TOL:
            raise AssertionError(
                f"KILL-B {label} FAILED: worst |ΔR²(dual,primal)| = {worst:.4g} ≥ {KILL_B_TOL} "
                f"at n={n} — primal PRESS-LOO is not anchored; diagnose before the curve."
            )
    return result


# ── anchor gate (reproduce the parent's stored atlas) ─────────────────────────────


def anchor_gate(curve_ridge, parent_atlas_path: Path, anchor_n: int, transitions) -> dict:
    """The n=anchor ridge r2_id (raw) must reproduce the parent's stored atlas.

    Same rows + solver ⇒ identical (plan §4.2). Loads the parent's
    stage0_atlas.json (committed on the branch); >0.01 max abs diff HARD FAILs,
    >1e-4 WARNs. Skipped (best-effort) when the parent atlas is absent."""
    if not parent_atlas_path.exists():
        logger.warning("[anchor-gate] parent atlas %s absent — SKIPPED", parent_atlas_path)
        return {"status": "skipped", "reason": f"{parent_atlas_path} absent"}
    with open(parent_atlas_path) as f:
        parent = json.load(f)
    pr = parent.get("atlas", {}).get("ridge", {}).get("raw", {})
    checks = {}
    worst = 0.0
    for t in transitions:
        key = f"transition_{t}"
        mine = curve_ridge["raw"].get(key, {}).get(str(anchor_n), {}).get("r2_id")
        theirs = pr.get(key, {}).get("r2_id")
        if mine is None or theirs is None or not (np.isfinite(mine) and np.isfinite(theirs)):
            continue
        d = abs(mine - theirs)
        worst = max(worst, d)
        checks[key] = {"mine": mine, "parent": theirs, "abs_diff": d}
    status = "pass" if worst <= 1e-4 else ("warn" if worst <= KILL_B_TOL else "fail")
    out = {"status": status, "worst_abs_diff": worst, "anchor_n": anchor_n, "checks": checks}
    if status == "fail":
        raise AssertionError(
            f"anchor gate FAILED: n={anchor_n} ridge r2_id diverges from the parent atlas by "
            f"{worst:.4g} > {KILL_B_TOL} — the scaling curve is not anchored to the parent."
        )
    logger.info("[anchor-gate] status=%s worst |Δ| = %.4g", status, worst)
    return out


# ── position-drift diagnostic ─────────────────────────────────────────────────────


def position_drift(fit_pool, drift_window, test, transitions, anchor_n, device, dual_max) -> dict:
    """Same-size fit on the LATEST new-stream window vs the anchor, on the FIXED test.

    Gates a FLAT-H1 attribution (§4.2 fold-2): equal band R² ⇒ stationary stream ⇒
    a flat curve is genuine saturation; a materially worse late-window fit ⇒ the
    added contexts are distributionally drifted (a flat read is confounded)."""
    anchor_fit = fit_pool[:anchor_n]
    band = [t for t in transitions if t in DATA_LIMITED_BAND]
    anchor_r2, drift_r2 = {}, {}
    for t in transitions:
        h_test, d_test = _deltas(test, t)
        ha, da = _deltas(anchor_fit, t)
        pred_a, _ = _fit_ridge(
            ha, da, h_test, sigma=1.0, device=device, dual_max=dual_max, n=anchor_n
        )
        anchor_r2[t] = MP.identity_relative_r2(pred_a, d_test)
        hd, dd = _deltas(drift_window, t)
        pred_d, _ = _fit_ridge(
            hd, dd, h_test, sigma=1.0, device=device, dual_max=dual_max, n=drift_window.shape[0]
        )
        drift_r2[t] = MP.identity_relative_r2(pred_d, d_test)
    band_present = [t for t in band if np.isfinite(anchor_r2[t]) and np.isfinite(drift_r2[t])]
    band_anchor = (
        float(np.mean([anchor_r2[t] for t in band_present])) if band_present else float("nan")
    )
    band_drift = (
        float(np.mean([drift_r2[t] for t in band_present])) if band_present else float("nan")
    )
    gap = band_anchor - band_drift
    logger.info(
        "[position-drift] band mean R² anchor=%.4f late-window=%.4f gap=%.4f",
        band_anchor,
        band_drift,
        gap,
    )
    return {
        "anchor_r2_id": {str(t): anchor_r2[t] for t in transitions},
        "late_window_r2_id": {str(t): drift_r2[t] for t in transitions},
        "data_limited_band": list(band),
        "band_mean_anchor": band_anchor,
        "band_mean_late_window": band_drift,
        "band_gap": gap,
        "note": "A materially worse late-window band fit caveats a FLAT-H1 read (drift, "
        "not saturation); a comparable fit supports genuine data-saturation.",
    }


# ── bundle loading (real HF vs synthetic smoke) ───────────────────────────────────


def _load_bundle(args):
    """Return (fit_pool, val, test, drift_window, anchor_n, parent_atlas_path)."""
    if args.synthetic_parent:
        rng = np.random.default_rng(0)
        # hidden is parametrized ONLY in synthetic-smoke mode so the eigh(d) is
        # instant on a CPU; the real run always uses C.EXPECTED_HIDDEN (3584).
        shape = (C.EXPECTED_LAYERS, args.synthetic_hidden)
        parent = rng.standard_normal((args.synthetic_parent, *shape)).astype(np.float32)
        new = rng.standard_normal((args.synthetic_new, *shape)).astype(np.float32)
        anchor_n = args.anchor_n
        fit_pool = np.concatenate([parent[:anchor_n], new], axis=0)
        val = parent[anchor_n : anchor_n + max(4, args.synthetic_parent // 10)]
        test = parent[anchor_n + val.shape[0] :]
        drift = new[-anchor_n:]
        logger.info(
            "[bundle] SYNTHETIC parent=%d new=%d anchor=%d val=%d test=%d",
            args.synthetic_parent,
            args.synthetic_new,
            anchor_n,
            val.shape[0],
            test.shape[0],
        )
        return (
            fit_pool,
            val,
            test,
            drift,
            anchor_n,
            Path("/nonexistent-parent-atlas.json"),
            "synthetic",
        )
    pass_b = C.load_pass_b()
    capture = S.load_capture_local_or_hf(args.capture_dir)
    bundle = S.build_scaling_bundle(pass_b, capture)
    # realized forward precision (Fold 1): prefer the realized field, fall back to the
    # capture_dtype alias for a legacy/synthetic manifest that predates the split.
    realized_dtype = capture.get("realized_capture_dtype", capture.get("capture_dtype"))
    logger.info(
        "[bundle] REAL fit_pool=%d val=%d test=%d drift_window=%d capture_dtype=%s",
        bundle["fit_pool"].shape[0],
        bundle["val"].shape[0],
        bundle["test"].shape[0],
        bundle["drift_window"].shape[0],
        realized_dtype,
    )
    return (
        bundle["fit_pool"],
        bundle["val"],
        bundle["test"],
        bundle["drift_window"],
        S.N_ANCHOR_FIT,
        PROJECT_ROOT / "eval_results" / "issue_841" / "stage0_atlas.json",
        realized_dtype,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 Stage-0 scaling curve.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--capture-dir", type=Path, default=S.CAPTURE_DIR)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=S.EVAL_SCALING_DIR,
        help="stage0_scaling.json dir (divert for smokes; default = committed path)",
    )
    ap.add_argument(
        "--maps-dir",
        type=Path,
        default=S.RIDGE_MAPS_DIR,
        help="per-n ridge-map dir (default = gitignored data cache)",
    )
    ap.add_argument("--ns", default="", help="comma-list of fit-set sizes (default SCALING_NS)")
    ap.add_argument("--anchor-n", type=int, default=S.N_ANCHOR_FIT)
    ap.add_argument("--dual-max", type=int, default=10000, help="n≤this uses the dual solver")
    ap.add_argument("--xcheck-n", type=int, default=10000, help="dual-vs-primal cross-check n")
    ap.add_argument("--transitions", default="", help="comma-list (default all 27)")
    ap.add_argument("--mlp-ns", default="", help="comma-list for MLP (default anchor + largest n)")
    ap.add_argument("--mlp-chunk-size", type=int, default=8)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--mlp-epochs", type=int, default=0, help="0 = production default (300)")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--synthetic-parent", type=int, default=0, help="smoke: fabricate parent rows")
    ap.add_argument(
        "--synthetic-new", type=int, default=0, help="smoke: fabricate new-capture rows"
    )
    ap.add_argument(
        "--synthetic-hidden",
        type=int,
        default=C.EXPECTED_HIDDEN,
        help="smoke-only: hidden dim (small ⇒ fast eigh on CPU; real run = 3584)",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    mlp_epochs = args.mlp_epochs or None
    ns = [int(x) for x in args.ns.split(",") if x] or list(S.SCALING_NS)
    transitions = [int(x) for x in args.transitions.split(",") if x] or list(range(C.N_TRANSITIONS))
    mlp_ns = [int(x) for x in args.mlp_ns.split(",") if x] or sorted(
        {ns[0], ns[-1]}
    )  # primary: anchor + largest
    logger.info(
        "device=%s ns=%s mlp_ns=%s transitions=%d dual_max=%d xcheck_n=%d",
        device,
        ns,
        mlp_ns,
        len(transitions),
        args.dual_max,
        args.xcheck_n,
    )

    out_dir, maps_dir = args.out_dir, args.maps_dir
    fit_pool, val, test, drift_window, anchor_n, parent_atlas_path, capture_dtype = _load_bundle(
        args
    )

    result: dict = {
        "ns": ns,
        "anchor_n": anchor_n,
        "transitions": transitions,
        "target_spaces": list(SPACES),
        "dual_max": args.dual_max,
        "capture_dtype": capture_dtype,  # realized capture precision (Fold 1)
        "metadata": C.reproducibility_metadata(
            {"phase": "stage0_scaling", "smoke": args.smoke, "capture_dtype": capture_dtype}
        ),
    }

    # KILL-B FIRST (fail before the full curve if the primal is unanchored).
    result["parity_and_xcheck"] = kill_b_parity(
        fit_pool, test, transitions, anchor_n, args.xcheck_n, device
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(out_dir / "stage0_scaling.json", result)

    ridge_curve, ridge_maps_by_n = ridge_scaling(
        fit_pool, test, transitions, ns, device=device, dual_max=args.dual_max
    )
    result.setdefault("scaling_curve", {})["ridge"] = ridge_curve
    C.write_json_atomic(out_dir / "stage0_scaling.json", result)  # checkpoint

    result["anchor_gate"] = anchor_gate(ridge_curve, parent_atlas_path, anchor_n, transitions)
    result["position_drift"] = position_drift(
        fit_pool, drift_window, test, transitions, anchor_n, device, args.dual_max
    )
    C.write_json_atomic(out_dir / "stage0_scaling.json", result)

    mlp_curve, mlp_params_by_n = mlp_scaling(
        fit_pool,
        val,
        test,
        transitions,
        mlp_ns,
        device=device,
        chunk_size=args.mlp_chunk_size,
        num_threads=args.num_threads,
        max_epochs=mlp_epochs,
    )
    result["scaling_curve"]["mlp"] = mlp_curve
    result["atlas_largest_n"] = ns[-1]
    result["mlp_ns"] = mlp_ns  # which n-points have persisted MLP maps for Stage-1
    # Record the MLP group-chunk as RAM-knob PROVENANCE (host-RAM fix): since #926
    # (4dfcba056f) the r2 curve is bit-identical across group_chunk values, so this
    # documents the memory regime the run used, not a seed the numbers depend on
    # (see mlp_scaling's PARTITION INVARIANCE block above).
    result["mlp_group_chunk"] = S.MLP_GROUP_CHUNK
    C.write_json_atomic(out_dir / "stage0_scaling.json", result)

    # Persist every fitted RAW map (Stage-1 reloads them for the class dimension) + upload.
    # LFS maps (.pt) route to the PRIVATE overflow repo (public LFS quota 403, #541/#552);
    # each bucket gets its OVERFLOW_POINTER.json breadcrumb on the canonical public repo.
    def _upload_maps(path, bucket):
        if not args.no_upload:
            S.upload_split_lfs_to_overflow(path, bucket, reason=S.OVERFLOW_REASON)

    for n, maps in ridge_maps_by_n.items():
        path = maps_dir / f"ridge_maps_n{n}.pt"
        S.save_ridge_maps(maps, path)
        logger.info("[maps] saved %d ridge maps for n=%d → %s", len(maps), n, path)
        _upload_maps(path, S.hf_ridge_maps_bucket(n))
    for n, params in mlp_params_by_n.items():
        path = maps_dir / f"mlp_maps_n{n}.pt"
        S.save_mlp_maps(params, path)
        logger.info("[maps] saved %d MLP maps for n=%d → %s", len(params), n, path)
        _upload_maps(path, S.hf_mlp_maps_bucket(n))

    logger.info("[done] wrote %s", out_dir / "stage0_scaling.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
