#!/usr/bin/env python
"""Issue #1336 — Phase A: ladder-alignment battery (the #825 Result-2 construction).

Per (stage pair, eval set, frozen layer): within ceilings, rep-swap both
directions, alignment maps (A_ctx / A_ctx_rev / A_ans / A_ans_rev) with their
own held-out R^2, and the composition comp_samefn = A_ans o M_base o A_ctx_rev
evaluated X_k -> Y_k (per-fold train-only fits, shared folds by prompt id,
fp64 solves — the parent no-leakage convention). Orthogonal +
scaled-orthogonal variants at the headline layer only.

Reparameterization gap per (pair, eval set, layer):
  gap = R^2(within stage-k) - R^2(comp_samefn_b2i)
with a paired prompt-level percentile bootstrap over the pair's shared rows
(per-draw own-mean re-centered weighted R^2, fp64 — the round-5 machinery).

``--decision`` aggregates the per-stage gaps into the registered lattice:
  C = gap_RLVR - gap_DPO per eval set with SHARED draws over the primary
  ladder's common row set; headline verdict on gsm8k_train5k chat at the
  pre-registered headline layer.

Cores imported from issue825_map_alignment (_ridge_prep/_ridge_predict/
_orth_fit/_orth_predict/_fold_reads/_assemble_battery) + issue825_fit_cells
(_cv_folds/_pooled_r2/_load_bundle_any/_fit_device). The round-5 paired-
bootstrap helpers are VENDORED below (consistency-checker blocker: their
source module scripts/issue825_role_contrast.py exists ONLY on the unmerged
origin/issue-825 branch @ 0e580958c6 — the #595 stranded-module class).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_fit_cells as fc  # noqa: E402
import issue825_map_alignment as ma  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

N_FOLDS = cm.N_FOLDS
FIT_SEED = cm.FIT_SEED

# Reads whose held-out predictions are PERSISTED for the paired bootstrap.
CAPTURE_READS = {
    "within": "ceil.within_instruct",  # stage-k within map, X_k -> Y_k
    "comp": "comp.linear.comp_samefn_b2i",  # A_ans o M_base o A_ctx_rev, X_k -> Y_k
}


# ===========================================================================
# VENDORED round-5 paired-bootstrap machinery — copied VERBATIM from
# scripts/issue825_role_contrast.py @ 0e580958c6 (unmerged origin/issue-825;
# artifact-reuse porting recipe). Reconciled against main: the module never
# existed on main; the only dependency is numpy (+ fit_cells._pooled_r2 in the
# serial oracle, aliased to the imported ``fc``). Equivalence gate:
# --selfcheck (run in every smoke).
# ===========================================================================
def draw_index_matrix(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) row-resample indices; conv == row is asserted upstream, so
    the conversation-level resample IS this row resample. Shared across BOTH
    arms and both headline layers of a pair (the pairing; mirrors
    run_cross_role_cell's one-sample-per-draw-across-layers form)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def counts_from_indices(idx_matrix: np.ndarray, n: int) -> np.ndarray:
    """(n_boot, n) fp64 row-count matrix W from a draw index matrix."""
    idx_matrix = np.asarray(idx_matrix)
    n_boot = idx_matrix.shape[0]
    w = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(w, (np.repeat(np.arange(n_boot), idx_matrix.shape[1]), idx_matrix.ravel()), 1.0)
    return w


def weighted_r2_draws(preds: np.ndarray, true: np.ndarray, w: np.ndarray) -> np.ndarray:
    """R²(w) per draw, fp64, PER-DRAW OWN-MEAN re-centered (the REGISTERED form).

    SS_res(w) = w @ ||Y-P||²_row
    SS_tot(w) = w @ ||Y||²_row - ||w @ Y||² / sum(w)   # (n_boot,n)@(n,d) GEMM
    Matches _pooled_r2's own-mean convention exactly (a fixed-center subset-sum
    CANNOT pass the serial-oracle gate — binding critic Must-Fix).
    """
    y64 = np.asarray(true, dtype=np.float64)
    p64 = np.asarray(preds, dtype=np.float64)
    resid = y64 - p64
    r2_row = np.einsum("nd,nd->n", resid, resid)
    y2_row = np.einsum("nd,nd->n", y64, y64)
    wsum = w.sum(axis=1)
    ss_res = w @ r2_row
    wy = w @ y64
    ss_tot = w @ y2_row - np.einsum("bd,bd->b", wy, wy) / wsum
    return 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)


def paired_bootstrap_batched(
    preds_a: np.ndarray, y_a: np.ndarray, preds_u: np.ndarray, y_u: np.ndarray, w: np.ndarray
) -> dict:
    """The PRODUCTION paired-bootstrap path: per-draw R² per arm + Δ(w).

    Vendored note: arm "a" carries the WITHIN read and arm "u" the COMP read
    here (the round-5 source named them assistant/user); the keys are kept.
    """
    r2_a = weighted_r2_draws(preds_a, y_a, w)
    r2_u = weighted_r2_draws(preds_u, y_u, w)
    return {"assistant": r2_a, "user": r2_u, "delta": r2_a - r2_u}


def paired_bootstrap_serial_reference(
    preds_a: np.ndarray,
    y_a: np.ndarray,
    preds_u: np.ndarray,
    y_u: np.ndarray,
    idx_matrix: np.ndarray,
) -> dict:
    """Seeded serial ORACLE (equivalence gate only; production dispatches
    paired_bootstrap_batched). Vendored with ``fit_cells`` aliased to fc."""
    r2_a, r2_u = [], []
    for row in np.asarray(idx_matrix):
        r2_a.append(fc._pooled_r2(preds_a[row], y_a[row]))
        r2_u.append(fc._pooled_r2(preds_u[row], y_u[row]))
    r2_a = np.asarray(r2_a, dtype=np.float64)
    r2_u = np.asarray(r2_u, dtype=np.float64)
    return {"assistant": r2_a, "user": r2_u, "delta": r2_a - r2_u}


def _ci(dist: np.ndarray) -> dict:
    d = np.asarray(dist, dtype=np.float64)
    return {
        "ci_lo": float(np.nanquantile(d, 0.025)),
        "ci_hi": float(np.nanquantile(d, 0.975)),
        "se_boot": float(np.nanstd(d, ddof=1)),
        "n_draws": len(d),
    }


def selfcheck() -> None:
    """Signature + equivalence gate for the vendored helpers on synthetic input."""
    rng = np.random.default_rng(0)
    n, d, n_boot = 17, 5, 64
    y_a = rng.normal(size=(n, d))
    p_a = y_a + 0.3 * rng.normal(size=(n, d))
    y_u = rng.normal(size=(n, d))
    p_u = y_u + 0.7 * rng.normal(size=(n, d))
    idx = draw_index_matrix(n, n_boot, seed=7)
    assert idx.shape == (n_boot, n)
    w = counts_from_indices(idx, n)
    assert w.shape == (n_boot, n) and float(w.sum()) == float(n_boot * n)
    fast = paired_bootstrap_batched(p_a, y_a, p_u, y_u, w)
    slow = paired_bootstrap_serial_reference(p_a, y_a, p_u, y_u, idx)
    for key in ("assistant", "user", "delta"):
        diff = float(np.max(np.abs(fast[key] - slow[key])))
        assert diff < 1e-10, f"vendored bootstrap mismatch on {key}: {diff}"
    print(f"[selfcheck] vendored paired bootstrap == serial oracle (n={n}, draws={n_boot})")


# ===========================================================================
# Pair battery
# ===========================================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pair", default=None, help="m0:m1 (e.g. base:rlvr)")
    ap.add_argument("--corpus", default=None, choices=tuple(cm.CORPORA))
    ap.add_argument("--format", default=None, choices=("chat", "naturalistic"))
    ap.add_argument("--decision", action="store_true", help="aggregate decision JSON")
    ap.add_argument("--selfcheck", action="store_true", help="vendored-helper equivalence gate")
    ap.add_argument("--turnstore-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336"))
    ap.add_argument("--preds-dir", type=Path, default=None)
    ap.add_argument("--frozen-layers", default=None, help="comma ints (default: registry set)")
    ap.add_argument("--headline-layer", type=int, default=None, help="override (smoke only)")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _metadata(seed: int, n: int) -> dict:
    return {
        "git_commit": fc._git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": "scripts/issue1336_ladder_alignment.py",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[align1336] wrote {path}")


def _primary_models(smoke: bool) -> tuple[str, ...]:
    return tuple(m for m in cm.SMOKE_MODELS) if smoke else cm.PRIMARY_LADDER


def headline_layer_rule(cells_dir: Path, frozen_layers: tuple[int, ...], smoke: bool) -> int:
    """Pre-registered stage-symmetric rule (plan §4 Phase F): among the frozen
    set, the layer maximizing MEAN within-stage R^2 across the primary-ladder
    models on lmsys5k-chat. Symmetric across the stages in every contrast."""
    models = _primary_models(smoke)
    means = {}
    for li in frozen_layers:
        vals = []
        for m in models:
            path = cells_dir / f"cells_{cm.cell_id(m, 'chat', 'lmsys5k')}.json"
            assert path.exists(), f"headline rule requires {path} (run the fit phase first)"
            r2 = json.loads(path.read_text())["r2_per_layer_obs"]
            assert li < len(r2), f"frozen layer {li} out of range for {path}"
            vals.append(float(r2[li]))
        means[li] = float(np.mean(vals))
    best = max(means, key=means.get)
    print(f"[align1336] headline layer {best} (frozen-set means {means})")
    return int(best)


def _xy_for(bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, Y, conv_ids) for the context arm: a1-header slot -> a1 profile."""
    arrays = bundle["arrays"]
    assert arrays["slots"].shape[1] == 2, f"n_slots {arrays['slots'].shape[1]} != 2"
    assert arrays["profiles"].shape[1] == 2, f"n_turns {arrays['profiles'].shape[1]} != 2"
    xy = fc._cell_xy(bundle, {"slot_index": 1, "target_turn_index": 1})
    return xy["X"], xy["Y"], np.asarray([str(c) for c in xy["conv_ids"]])


def _align_rows(ids_a: np.ndarray, ids_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair prompt-id intersection BEFORE fold assignment (plan §3)."""
    common = np.intersect1d(ids_a, ids_b)
    assert len(common) >= N_FOLDS, f"only {len(common)} shared rows across the pair"
    pos_a = {c: i for i, c in enumerate(ids_a)}
    pos_b = {c: i for i, c in enumerate(ids_b)}
    ia = np.asarray([pos_a[c] for c in common], dtype=np.int64)
    ib = np.asarray([pos_b[c] for c in common], dtype=np.int64)
    return common, ia, ib


def run_pair(args) -> None:
    m0, m1 = args.pair.split(":")
    assert (m0, m1) in cm.PAIRS, f"pair {args.pair} not in the registered PAIRS set"
    corpus, fmt = args.corpus, args.format
    assert corpus and fmt, "--corpus and --format are required with --pair"
    smoke = args.smoke
    ts_dir = args.turnstore_dir or Path(
        "data/issue_1336/" + ("turnstore_smoke" if smoke else "turnstore")
    )
    preds_dir = args.preds_dir or Path(
        "data/issue_1336/" + ("align_preds_smoke" if smoke else "align_preds")
    )
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    headline = (
        args.headline_layer
        if args.headline_layer is not None
        else headline_layer_rule(args.out_dir / "cells", frozen, smoke)
    )

    b0 = fc._load_bundle_any(ts_dir, m0, fmt, corpus)
    b1 = fc._load_bundle_any(ts_dir, m1, fmt, corpus)
    X0, Y0, ids0 = _xy_for(b0)
    X1, Y1, ids1 = _xy_for(b1)
    common, i0, i1 = _align_rows(ids0, ids1)
    n = len(common)
    folds = fc._cv_folds(common, N_FOLDS, FIT_SEED)
    print(f"[align1336] pair={m0}->{m1} set=({corpus},{fmt}) n={n} headline={headline}")

    dev = fc._fit_device()
    dtype = torch.float64
    eval_set_idx = cm.EVAL_SETS.index((corpus, fmt))
    idx_matrix = draw_index_matrix(n, n_boot, seed=1000 + eval_set_idx)
    w = counts_from_indices(idx_matrix, n)

    per_layer: dict[str, dict] = {}
    preds_store: dict[str, np.ndarray] = {
        "conv_ids": common,
        "folds": folds.astype(np.int64),
    }
    for li in frozen:
        assert li < X0.shape[1], f"frozen layer {li} out of range ({X0.shape[1]} layers)"
        Xi = torch.as_tensor(X1[i1][:, li, :], dtype=dtype).to(dev)
        Yi = torch.as_tensor(Y1[i1][:, li, :], dtype=dtype).to(dev)
        Xb = torch.as_tensor(X0[i0][:, li, :], dtype=dtype).to(dev)
        Yb = torch.as_tensor(Y0[i0][:, li, :], dtype=dtype).to(dev)
        tens = (Xi, Yi, Xb, Yb)
        do_orth = li == headline
        ss_res: dict[str, float] = {}
        ss_tot: dict[str, float] = {}
        captured = {tag: np.zeros((n, Xi.shape[1]), dtype=np.float32) for tag in CAPTURE_READS}
        fitted = np.zeros(n, dtype=bool)
        for k in range(N_FOLDS):
            tr_np = folds != k
            te_np = folds == k
            if te_np.sum() == 0 or tr_np.sum() < 3:
                continue
            tr = torch.as_tensor(tr_np)
            te = torch.as_tensor(te_np)
            preps = {
                "Xb": ma._ridge_prep(Xb[tr]),
                "Xi": ma._ridge_prep(Xi[tr]),
                "Yb": ma._ridge_prep(Yb[tr]),
                "Yi": ma._ridge_prep(Yi[tr]),
            }
            orth = None
            if do_orth:
                orth = {"ctx": ma._orth_fit(Xb[tr], Xi[tr]), "ans": ma._orth_fit(Yb[tr], Yi[tr])}
            reads = ma._fold_reads(preps, orth, tens, tr, te, do_orth=do_orth)
            for name, (pred, true) in reads.items():
                ss_res[name] = ss_res.get(name, 0.0) + float(((true - pred) ** 2).sum())
                ss_tot[name] = ss_tot.get(name, 0.0) + float(((true - true.mean(0)) ** 2).sum())
            for tag, name in CAPTURE_READS.items():
                captured[tag][te_np] = reads[name][0].float().cpu().numpy()
            fitted[te_np] = True
            del preps, orth, reads
        battery = ma._assemble_battery(ss_res, ss_tot)
        within = battery["ceilings"]["within_instruct"]
        comp = battery["composition"]["linear"]["comp_samefn_b2i"]
        gap_point = float(within) - float(comp)
        y_np = Yi.float().cpu().numpy()
        assert fitted.all(), f"unfitted rows at layer {li} (n={n})"
        boot = paired_bootstrap_batched(captured["within"], y_np, captured["comp"], y_np, w)
        per_layer[str(li)] = {
            "battery": battery,
            "within_r2": float(within),
            "comp_samefn_r2": float(comp),
            "gap": gap_point,
            "gap_bootstrap": _ci(boot["delta"]),
            "is_headline": bool(do_orth),
        }
        preds_store[f"within_l{li}"] = captured["within"].astype(np.float16)
        preds_store[f"comp_l{li}"] = captured["comp"].astype(np.float16)
        preds_store[f"y_l{li}"] = y_np.astype(np.float16)
        del Xi, Yi, Xb, Yb, tens
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    unit = f"{m0}__{m1}_{fmt}_{corpus}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / f"alignpreds_{unit}.npz"
    np.savez(preds_path, **preds_store)  # plain savez: compression OFF for Xet (#813)
    sha = hashlib.sha256(preds_path.read_bytes()).hexdigest()
    manifest_path = preds_dir / "preds_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[preds_path.name] = {
        "sha256": sha,
        "shapes": {k: list(np.asarray(v).shape) for k, v in preds_store.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    payload = {
        "metadata": _metadata(FIT_SEED, n),
        "pair": {"m0": m0, "m1": m1},
        "eval_set": {"corpus": corpus, "format": fmt},
        "n_shared_rows": n,
        "frozen_layers": list(frozen),
        "headline_layer": headline,
        "n_boot": n_boot,
        "boot_seed": 1000 + eval_set_idx,
        "per_layer": per_layer,
        "preds_npz": str(preds_path),
        "preds_sha256": sha,
    }
    _write_json(args.out_dir / "ladder_alignment" / f"pair_{unit}.json", payload)


# ===========================================================================
# Decision aggregation (headline contrast C + lattice verdict)
# ===========================================================================
def _load_align_preds(preds_dir: Path, m0: str, m1: str, fmt: str, corpus: str) -> dict:
    path = preds_dir / f"alignpreds_{m0}__{m1}_{fmt}_{corpus}.npz"
    assert path.exists(), f"missing alignment preds {path} — run the pair battery first"
    return dict(np.load(path, allow_pickle=False))


def _gap_draws_on_rows(pair_npz: dict, layer: int, rows: np.ndarray, w: np.ndarray) -> dict:
    """Per-draw gap = R²(within) - R²(comp) restricted to the given row subset."""
    within = pair_npz[f"within_l{layer}"][rows].astype(np.float64)
    comp = pair_npz[f"comp_l{layer}"][rows].astype(np.float64)
    y = pair_npz[f"y_l{layer}"][rows].astype(np.float64)
    boot = paired_bootstrap_batched(within, y, comp, y, w)
    point = fc._pooled_r2(within, y) - fc._pooled_r2(comp, y)
    return {"draws": boot["delta"], "point": float(point)}


def run_decision(args) -> None:
    smoke = args.smoke
    preds_dir = args.preds_dir or Path(
        "data/issue_1336/" + ("align_preds_smoke" if smoke else "align_preds")
    )
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if smoke else cm.FROZEN_LAYERS
    n_boot = (
        args.n_boot if args.n_boot is not None else (cm.SMOKE_N_BOOT if smoke else cm.N_BOOTSTRAP)
    )
    headline = (
        args.headline_layer
        if args.headline_layer is not None
        else headline_layer_rule(args.out_dir / "cells", frozen, smoke)
    )
    primary = _primary_models(smoke)
    stages = [m for m in primary if m != "base"]
    assert "rlvr" in stages and "dpo" in stages, f"decision needs dpo+rlvr in ladder {primary}"

    eval_sets = [(c, f) for (c, f) in cm.EVAL_SETS if not smoke or (c in cm.SMOKE_CORPORA)]
    per_set: dict[str, dict] = {}
    for si, (corpus, fmt) in enumerate(eval_sets):
        npzs = {k: _load_align_preds(preds_dir, "base", k, fmt, corpus) for k in stages}
        # Shared row set across every primary base-anchored pair (prompt ids).
        shared = None
        for k in stages:
            ids = npzs[k]["conv_ids"]
            shared = ids if shared is None else np.intersect1d(shared, ids)
        assert shared is not None and len(shared) >= N_FOLDS, "empty shared row set"
        rows_by_stage = {}
        for k in stages:
            pos = {c: i for i, c in enumerate(npzs[k]["conv_ids"])}
            rows_by_stage[k] = np.asarray([pos[c] for c in shared], dtype=np.int64)
        n_shared = len(shared)
        idx_matrix = draw_index_matrix(n_shared, n_boot, seed=5000 + si)
        w = counts_from_indices(idx_matrix, n_shared)

        gaps = {k: _gap_draws_on_rows(npzs[k], headline, rows_by_stage[k], w) for k in stages}
        c_draws = gaps["rlvr"]["draws"] - gaps["dpo"]["draws"]
        c_point = gaps["rlvr"]["point"] - gaps["dpo"]["point"]
        increments = {}
        order = [k for k in ("sft", "dpo", "rlvr") if k in stages]
        prev = None
        for k in order:
            inc_draws = gaps[k]["draws"] - (gaps[prev]["draws"] if prev else 0.0)
            inc_point = gaps[k]["point"] - (gaps[prev]["point"] if prev else 0.0)
            increments[f"{prev or 'zero'}->{k}"] = {"point": float(inc_point), **_ci(inc_draws)}
            prev = k
        per_set[f"{corpus}_{fmt}"] = {
            "n_shared_rows": int(n_shared),
            "boot_seed": 5000 + si,
            "gap_per_stage": {
                k: {"point": gaps[k]["point"], **_ci(gaps[k]["draws"])} for k in stages
            },
            "contrast_C": {"point": float(c_point), **_ci(c_draws)},
            "adjacent_increments": increments,
        }

    headline_key = "gsm8k_train5k_chat" if not smoke else f"{cm.SMOKE_CORPORA[0]}_chat"
    assert headline_key in per_set, f"headline eval set {headline_key} missing from {per_set}"
    c = per_set[headline_key]["contrast_C"]
    if c["ci_lo"] > 0.0:
        verdict = "rlvr_specific_gap_growth"  # C > 0, CI clear of zero (positive side)
    elif c["ci_hi"] < 0.0:
        verdict = "rlvr_gap_not_larger"  # CI wholly below 0
    else:
        verdict = "inconclusive"
    magnitude_note = None
    if verdict == "rlvr_specific_gap_growth" and abs(c["point"]) < 0.05:
        magnitude_note = "statistically detectable but small (|C| < 0.05 practical scale)"
    # H-elicit: every primary Δ_k CI includes 0 OR |Δ_k| < 0.02, all eval sets.
    h_elicit = all(
        (g["ci_lo"] <= 0.0 <= g["ci_hi"]) or abs(g["point"]) < 0.02
        for s in per_set.values()
        for g in s["gap_per_stage"].values()
    )
    # H-generic (descriptive band): all adjacent increments positive CI-clear
    # and no increment more than 2x the next largest.
    incs = [v for s in per_set.values() for v in s["adjacent_increments"].values()]
    all_pos = all(v["ci_lo"] > 0.0 for v in incs) and bool(incs)
    h_generic = False
    if all_pos:
        pts = sorted((abs(v["point"]) for v in incs), reverse=True)
        h_generic = len(pts) < 2 or pts[0] <= 2.0 * pts[1]

    payload = {
        "metadata": _metadata(FIT_SEED, 0),
        "headline_layer": headline,
        "headline_eval_set": headline_key,
        "n_boot": n_boot,
        "per_eval_set": per_set,
        "verdict_lattice": {
            "contrast_C_headline": c,
            "verdict": verdict,
            "magnitude_note": magnitude_note,
            "h_elicit_supported": bool(h_elicit),
            "h_generic_flagged": bool(h_generic),
            "secondary_family_bonferroni2": [
                "lmsys5k_chat",
                "gsm8k_test1319_chat",
            ],
        },
        "smoke": bool(smoke),
    }
    _write_json(args.out_dir / "decision" / "headline_contrast.json", payload)


def main() -> None:
    args = parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if args.decision:
        run_decision(args)
        return
    assert args.pair, "--pair m0:m1 is required (or --decision / --selfcheck)"
    run_pair(args)


if __name__ == "__main__":
    main()
