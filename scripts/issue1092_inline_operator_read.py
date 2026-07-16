#!/usr/bin/env python3
"""#1092 SCOPED inline operator-level prefix-arm vs context-arm comparison.

The FAST, subset early read on the Part-B operator comparison (the comprehensive
all-cell version runs later in the off-VM battery). Per the dispatch brief +
`eval_results/issue_1092/inline_caveat_repairs_operator_comparison/deferred_refit_spec.json`
:: `partB_operator_comparison_method_deferred`.

Cells {cell_inst_own, cell_pre_own} x layers {14,18,19} x bases {ambient, pca48}
x arms {prefix_end, context_end}. Battery-EXCLUDED fit rows (is_eval_only==True
AND trait_stratum excluded — the CORRECTED filter from the deferred spec).

For each (cell, layer, arm, basis) we fit a standardized-X ridge map W and read
it in the RAW residual-stream input basis (the common physical space across
arms) as W_raw (H_in x P_out). We then compare W_prefix vs W_ctx via:
  - principal angles between top-k singular subspaces (INPUT/right and
    OUTPUT/left), k=48 and k at 90% spectral energy;
  - orthogonal-Procrustes residual min_R ||W_c - R W_p||_F / ||W_c||_F, with
    R in O(H_in) (so the residual probes OUTPUT-subspace agreement modulo a free
    input rotation) — reported full-rank and rank-k;
  - a spectrum-matched null band (200 draws: observed singular values, random
    orthonormal subspaces) for both the angles and the Procrustes residual;
  - reference anchors (identical-output floor, orthogonal-output ceiling,
    random-gaussian map).

Lambda: each arm's own lambda is selected by GCV over the engine RIDGE_LAMBDAS
on the full battery-excluded data (a scoped single-fit substitute for the banked
per-fold PRESS-LOO CV; cross-checked against the banked read1 per-fold lambda
mode). A MATCHED-lambda variant (context arm's lambda for both) is also reported.

Fit engine reuse: standardization convention + RIDGE_LAMBDAS from the committed
#658/#923 helpers; the GCV closed form uses the same thin-SVD factors PressRidge
uses, without materializing PressRidge's batched (n_lambda, m, k) draw tensors
(memory-bounded on the shared VM).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/mnt/eps-data/thomasjiralerspong/.hf_i1092_operator")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

RIDGE_LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]  # #658 A3.4 grid (reused)

STAGE = Path(
    "/mnt/eps-data/thomasjiralerspong/issue_1092_inline_operator/issue1092_realistic_crossing"
)
SUMM = STAGE / "analysis_tensors/summaries"
MANIFEST = STAGE / "corpus/manifest.jsonl"
BANKED = PROJECT_ROOT / "eval_results/issue_1092/p7"
OUT = PROJECT_ROOT / "eval_results/issue_1092/inline_operator_read_scoped"

CELLS = ["cell_inst_own", "cell_pre_own"]
LAYERS = [14, 18, 19]
ARMS = ["prefix_end", "context_end"]
BASES = ["ambient", "pca48"]
TARGETS = ["t1", "t2", "t3"]  # pooled/stacked answer target = the parent read1 target
KCAP = 512  # store at most this many singular directions (>= any plausible k90)
N_NULL = 200
SEED = 0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load(cell: str, kind: str, layer: int) -> np.ndarray:
    p = SUMM / cell / f"{kind}_L{layer:02d}.npy"
    if not p.exists():
        shards = sorted((SUMM / cell).glob(f"{kind}_L{layer:02d}_shard*.npy"))
        if not shards:
            raise FileNotFoundError(p)
        return np.concatenate([np.load(s, mmap_mode="r") for s in shards], axis=0)
    return np.load(p, mmap_mode="r")


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """press_fit_predict standardize=True convention: train mu/sd + degenerate drop."""
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    keep = sd > (sd.max() * 1e-6 + 1e-12)
    Xn = ((X - mu) / sd)[:, keep]
    return Xn, mu, sd, keep


def _fit_arm(Xn: np.ndarray, Yc: np.ndarray) -> dict:
    """Thin-SVD of standardized X + G = U^T Yc + GCV lambda over RIDGE_LAMBDAS."""
    Xt = torch.from_numpy(np.ascontiguousarray(Xn)).double()
    U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)  # U (m,k), S (k,), Vh (k,d)
    S2 = (S * S).numpy()
    Yt = torch.from_numpy(np.ascontiguousarray(Yc)).double()
    G = (U.T @ Yt).numpy()  # (k, P) -- the dominant GEMM
    g2 = (G * G).sum(1)  # (k,) row energies
    yc2 = float((Yc * Yc).sum())
    n = Xn.shape[0]
    gcv = []
    for lam in RIDGE_LAMBDAS:
        phi = S2 / (S2 + lam)
        rss = float(((1.0 - phi) ** 2 * g2).sum() + (yc2 - g2.sum()))
        trH = float(phi.sum())
        denom = (n - trH) ** 2
        gcv.append(rss / denom if denom > 0 else np.inf)
    lam_idx = int(np.argmin(gcv))
    return {
        "S": S.numpy(),
        "Vh": Vh.numpy(),
        "G": G,
        "S2": S2,
        "gcv": gcv,
        "lam_idx": lam_idx,
        "lambda": float(RIDGE_LAMBDAS[lam_idx]),
    }


def _reconstruct_wraw(
    fit: dict, lam: float, sd: np.ndarray, keep: np.ndarray, H: int
) -> np.ndarray:
    """W_raw (H_in x P): the operator on the RAW residual stream (common basis)."""
    S = fit["S"]
    coef = S / (S**2 + lam)  # (k,)
    W_std = fit["Vh"].T @ (coef[:, None] * fit["G"])  # (d_keep, P)
    keep_idx = np.nonzero(keep)[0]
    W_raw = np.zeros((H, W_std.shape[1]), dtype=np.float64)
    W_raw[keep_idx] = W_std / sd[keep][:, None]  # undo per-dim scaling -> raw input basis
    return W_raw


def _op_svd(W_raw: np.ndarray, kcap: int) -> dict:
    """Economy SVD of W_raw; keep full spectrum + top-kcap singular vectors."""
    A, sig, Bt = np.linalg.svd(W_raw, full_matrices=False)  # A (H,r), sig(r), Bt(r,P)
    kc = min(kcap, sig.shape[0])
    return {"A": A[:, :kc], "sig": sig, "B": Bt[:kc].T}  # A(H,kc) input, B(P,kc) output


def _subspace_angles(U1: np.ndarray, U2: np.ndarray, k: int) -> np.ndarray:
    """Principal angles (rad) between column spaces of orthonormal U1[:,:k], U2[:,:k]."""
    k = min(k, U1.shape[1], U2.shape[1])
    if k == 0:
        return np.array([])
    s = np.linalg.svd(U1[:, :k].T @ U2[:, :k], compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def _k90(sig: np.ndarray) -> int:
    e = sig**2
    c = np.cumsum(e) / e.sum()
    return int(np.searchsorted(c, 0.90) + 1)


def _procrustes_resid(sig_p, sig_c, BtB: np.ndarray, k: int) -> float:
    """||W_c - R W_p||_F / ||W_c||_F over R in O(H_in), rank-k, from output cross-Gram."""
    sp = sig_p[:k]
    sc = sig_c[:k]
    inner = (sp[:, None] * BtB[:k, :k]) * sc[None, :]  # diag(sp) (Bp^T Bc) diag(sc)
    nuc = float(np.linalg.svd(inner, compute_uv=False).sum())
    num2 = float((sc**2).sum() + (sp**2).sum() - 2.0 * nuc)
    den = float((sc**2).sum()) ** 0.5
    return float(np.sqrt(max(0.0, num2)) / den) if den > 0 else float("nan")


def _rand_orthonormal(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return q[:, :k]


def _null_band(sig_p, sig_c, H: int, P: int, k: int, n_draws: int, rng) -> dict:
    """Spectrum-matched random-subspace null for angles (input+output) + Procrustes."""
    in_med, in_max, out_med, out_max, proc = [], [], [], [], []
    for _ in range(n_draws):
        Ap = _rand_orthonormal(H, k, rng)
        Ac = _rand_orthonormal(H, k, rng)
        Bp = _rand_orthonormal(P, k, rng)
        Bc = _rand_orthonormal(P, k, rng)
        ain = _subspace_angles(Ap, Ac, k)
        aout = _subspace_angles(Bp, Bc, k)
        in_med.append(float(np.median(ain)))
        in_max.append(float(ain.max()))
        out_med.append(float(np.median(aout)))
        out_max.append(float(aout.max()))
        proc.append(_procrustes_resid(sig_p, sig_c, Bp.T @ Bc, k))

    def pct(a):
        a = np.asarray(a)
        return {
            "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
        }

    return {
        "n_draws": n_draws,
        "input_angle_median_rad": pct(in_med),
        "input_angle_max_rad": pct(in_max),
        "output_angle_median_rad": pct(out_med),
        "output_angle_max_rad": pct(out_max),
        "procrustes_resid": pct(proc),
    }


def _banked_lambda_mode(cell: str, arm: str, layer: int, basis: str) -> dict:
    """Recover the banked read1 per-fold PRESS lambda (fit-arm A) for cross-check."""
    try:
        with open(BANKED / "read1_map_skill.json") as fh:
            d = json.load(fh)
    except Exception:
        return {"available": False}
    for u in d.get("units", []):
        p = u.get("provenance", {})
        if (p.get("cell"), p.get("arm"), p.get("fit_arm"), p.get("layer"), p.get("basis")) == (
            cell,
            arm,
            "A",
            layer,
            basis,
        ):
            li = u.get("lambda_indices") or (u.get("fit") or {}).get("lambda_indices")
            if li:
                vals, counts = np.unique(np.asarray(li), return_counts=True)
                mode_i = int(vals[int(np.argmax(counts))])
                return {
                    "available": True,
                    "per_fold_indices": [int(x) for x in li],
                    "mode_index": mode_i,
                    "mode_lambda": float(RIDGE_LAMBDAS[mode_i])
                    if 0 <= mode_i < len(RIDGE_LAMBDAS)
                    else None,
                }
    return {"available": False}


def _deg(x):
    return None if x is None else float(np.degrees(x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gate-only",
        action="store_true",
        help="fit ONLY the first (cell,layer) ambient unit, project wall, exit",
    )
    ap.add_argument("--max-wall-h", type=float, default=4.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = _jsonl(MANIFEST)
    n_manifest = len(rows)
    print(f"manifest rows={n_manifest}", flush=True)

    result_path = OUT / "operator_read.json"
    result: dict = {
        "meta": {
            "script": "scripts/issue1092_inline_operator_read.py",
            "git_commit": _git_sha(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "manifest_rows": n_manifest,
            "scope": "SCOPED early read — 2 cells x 3 layers x 2 bases x 2 arms; "
            "comprehensive all-cell Part-B runs off-VM later.",
            "target": "pooled/stacked answer target (t1,t2,t3 concatenated) — "
            "the parent read1 target",
            "fit_rows": "battery-EXCLUDED: stratum != trait_stratum AND not "
            "is_eval_only (corrected filter)",
            "lambda_selector": "GCV over RIDGE_LAMBDAS on full battery-excluded data "
            "(scoped single-fit substitute for banked per-fold PRESS-LOO); "
            "matched-lambda variant uses the context arm's lambda for both",
            "operator_basis": "RAW residual-stream input basis W_raw (H_in x P_out), "
            "common across arms; per-arm standardization undone",
            "procrustes_convention": "R in O(H_in) left-multiplies W_prefix -> residual probes "
            "OUTPUT-subspace agreement modulo a free input rotation",
            "null": f"{N_NULL} draws, observed singular values + random orthonormal subspaces",
        },
        "cells": {},
    }
    if result_path.exists():
        try:
            prev = json.loads(result_path.read_text())
            result["cells"].update(prev.get("cells", {}))
        except Exception:
            pass

    t_first = None
    for cell in CELLS:
        for layer in LAYERS:
            unit_key = f"{cell}_L{layer:02d}"
            if unit_key in result["cells"] and not args.gate_only:
                print(f"skip (done) {unit_key}", flush=True)
                continue
            t0 = time.monotonic()
            H = None
            # --- load X arms + stacked target, index to battery-excluded fit rows ---
            xf = {a: _load(cell, a, layer) for a in ARMS}
            yk = {t: _load(cell, t, layer) for t in TARGETS}
            n0 = min(
                min(v.shape[0] for v in xf.values()),
                min(v.shape[0] for v in yk.values()),
                n_manifest,
            )
            idx = np.asarray(
                [
                    i
                    for i in range(n0)
                    if rows[i].get("stratum") != "trait_stratum" and not rows[i].get("is_eval_only")
                ],
                dtype=np.int64,
            )
            H = int(next(iter(xf.values())).shape[1])
            Xarm = {a: np.asarray(xf[a][idx], dtype=np.float64) for a in ARMS}
            del xf
            Y_stacked = np.concatenate(
                [np.asarray(yk[t][idx], dtype=np.float64) for t in TARGETS], axis=1
            )
            del yk
            n_fit = Xarm[ARMS[0]].shape[0]
            # pca48 target basis (shared across arms): top-48 right singular vectors of Yc
            ymu_stacked = Y_stacked.mean(0, keepdims=True)
            Yc_stacked = Y_stacked - ymu_stacked
            _, _, Vpca = torch.svd_lowrank(torch.from_numpy(Yc_stacked).double(), q=48, niter=6)
            Vpca = Vpca.numpy()  # (P_stacked, 48)

            # --- per-arm X-SVD (shared across bases) ---
            armstd = {}
            for a in ARMS:
                Xn, mu, sd, keep = _standardize(Xarm[a])
                armstd[a] = {"Xn": Xn, "mu": mu, "sd": sd, "keep": keep}

            cell_out = {
                "n_fit": int(n_fit),
                "H_in": H,
                "n_battery_excluded": int(n0 - len(idx)),
                "bases": {},
            }
            for basis in BASES:
                if basis == "ambient":
                    Yb = Y_stacked
                    P = Yb.shape[1]
                else:
                    Yb = Yc_stacked @ Vpca  # (n_fit, 48)
                    P = Yb.shape[1]
                Ycb = Yb - Yb.mean(0, keepdims=True)
                # fit each arm on this basis' target
                fits = {}
                for a in ARMS:
                    fits[a] = _fit_arm(armstd[a]["Xn"], Ycb)
                lam_own = {a: fits[a]["lambda"] for a in ARMS}
                lam_matched = lam_own["context_end"]  # context arm's lambda for both
                banked = {a: _banked_lambda_mode(cell, a, layer, basis) for a in ARMS}

                basis_out = {
                    "P_out": int(P),
                    "lambda_own": lam_own,
                    "lambda_matched": lam_matched,
                    "banked_lambda": banked,
                    "variants": {},
                }
                for variant, lam_of in (
                    ("own_lambda", lam_own),
                    ("matched_lambda", {a: lam_matched for a in ARMS}),
                ):
                    ops = {}
                    for a in ARMS:
                        Wr = _reconstruct_wraw(
                            fits[a], lam_of[a], armstd[a]["sd"], armstd[a]["keep"], H
                        )
                        ops[a] = _op_svd(Wr, KCAP)
                        del Wr
                    op_p, op_c = ops["prefix_end"], ops["context_end"]
                    k90 = min(_k90(op_p["sig"]), _k90(op_c["sig"]), KCAP)
                    kset = {"k48": min(48, KCAP), "k90": k90}
                    # full-rank observed Procrustes needs full B cross-Gram (rank up to kcap here)
                    variant_out = {
                        "k90": int(k90),
                        "energy_dim_full": int(op_p["sig"].shape[0]),
                        "reads": {},
                    }
                    for kname, k in kset.items():
                        BtB = op_p["B"].T @ op_c["B"]  # (kc, kc)
                        ain = _subspace_angles(op_p["A"], op_c["A"], k)
                        aout = _subspace_angles(op_p["B"], op_c["B"], k)
                        proc = _procrustes_resid(op_p["sig"], op_c["sig"], BtB, k)
                        # anchors
                        sp, sc = op_p["sig"][:k], op_c["sig"][:k]
                        anchor_identical = float(
                            np.sqrt(max(0.0, (sc**2).sum() + (sp**2).sum() - 2 * (sp * sc).sum()))
                            / (np.sqrt((sc**2).sum()) + 1e-12)
                        )
                        anchor_orthogonal = float(
                            np.sqrt((sc**2).sum() + (sp**2).sum())
                            / (np.sqrt((sc**2).sum()) + 1e-12)
                        )
                        null = _null_band(op_p["sig"], op_c["sig"], H, P, k, N_NULL, rng)
                        variant_out["reads"][kname] = {
                            "k": int(k),
                            "input_angle_median_rad": float(np.median(ain)),
                            "input_angle_median_deg": _deg(float(np.median(ain))),
                            "input_angle_max_rad": float(ain.max()),
                            "output_angle_median_rad": float(np.median(aout)),
                            "output_angle_median_deg": _deg(float(np.median(aout))),
                            "output_angle_max_rad": float(aout.max()),
                            "procrustes_resid": proc,
                            "anchor_identical_output": anchor_identical,
                            "anchor_orthogonal_output": anchor_orthogonal,
                            "null": null,
                            "input_below_null": bool(
                                np.median(ain) < null["input_angle_median_rad"]["p5"]
                            ),
                            "output_below_null": bool(
                                np.median(aout) < null["output_angle_median_rad"]["p5"]
                            ),
                            "procrustes_below_null": bool(proc < null["procrustes_resid"]["p5"]),
                        }
                    basis_out["variants"][variant] = variant_out
                cell_out["bases"][basis] = basis_out
                del fits
            result["cells"][unit_key] = cell_out
            result_path.write_text(json.dumps(result, indent=2, allow_nan=True))
            dt = time.monotonic() - t0
            print(f"[unit] {unit_key} done in {dt:.0f}s (n_fit={n_fit}, H={H})", flush=True)

            if t_first is None:
                t_first = dt
                n_units = len(CELLS) * len(LAYERS)
                proj_h = t_first * n_units / 3600.0
                print(
                    f"[gate] first unit {t_first:.0f}s x {n_units} units "
                    f"-> projected {proj_h:.2f}h (cap {args.max_wall_h}h)",
                    flush=True,
                )
                result["meta"]["projection"] = {
                    "first_unit_s": t_first,
                    "n_units": n_units,
                    "projected_wall_h": proj_h,
                    "cap_h": args.max_wall_h,
                }
                result_path.write_text(json.dumps(result, indent=2, allow_nan=True))
                if proj_h > args.max_wall_h:
                    result["meta"]["ABORT"] = f"projected {proj_h:.2f}h > cap {args.max_wall_h}h"
                    result_path.write_text(json.dumps(result, indent=2, allow_nan=True))
                    print(f"[gate] ABORT: projected {proj_h:.2f}h > {args.max_wall_h}h", flush=True)
                    return
                if args.gate_only:
                    print("[gate] gate-only: exiting after first unit", flush=True)
                    return
            del Xarm, Y_stacked, Yc_stacked, armstd
            import gc

            gc.collect()

    print(f"wrote {result_path}", flush=True)


if __name__ == "__main__":
    t = time.monotonic()
    main()
    print(f"done in {time.monotonic() - t:.0f}s", flush=True)
