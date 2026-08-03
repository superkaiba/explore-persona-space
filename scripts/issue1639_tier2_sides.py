"""Tier-2 side attribution for the character 4x4: context-side vs answer-side shift.

User-chat inline free-analysis round (2026-07-24, #1639/#1310; follow-on to
scripts/issue1639_tier15_intercept_refit.py). The full Tier-2 intercept refit
corrects the NET constant `Δy − W_s Δx`; this round splits it into its two
one-sided sub-rungs and decomposes the offset vector, per ordered persona pair:

  tier2a (context recentering only): ŷ = W_s(x − x̄_t,tr) + ȳ_s,tr
      — corrects the mapped context-mean displacement W_s Δx, keeps the
      source answer intercept. Affine identity: pred_naive − P̄_t + ȳ_s,
      where P̄_t = target-train mean of the naive predictions (the source
      core maps its own train X-mean to its train Y-mean exactly, so
      W_s Δx = P̄_t − ȳ_s in prediction space).
  tier2b (answer recentering only):  ŷ = W_s(x − x̄_s,tr) + ȳ_t,tr
      — corrects the answer-mean displacement Δy only.
      Affine identity: pred_naive + (ȳ_t − ȳ_s).
  decomposition per fold: ‖W_s Δx‖ = ‖P̄_t − ȳ_s‖, ‖Δy‖ = ‖ȳ_t − ȳ_s‖,
      cos(W_s Δx, Δy), and ‖net‖ = ‖Δy − W_s Δx‖.

Same rig as the tier-1.5 round: #1310 scene-aggregated store, layer 19,
shared scenario-grouped K=5 folds seed 0, fit825 core with GCV_DOF_CAP = 0.9.
Recomputed naive rungs are equality-gated against the committed
tier15_intercept_refit/results.json (same folds, same core -> tol 1e-4).

CLI: uv run python scripts/issue1639_tier2_sides.py
"""

from __future__ import annotations

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

load_dotenv()  # thread caps (#847) before torch/numpy import

import numpy as np  # noqa: E402

from issue1639_tier15_intercept_refit import (  # noqa: E402
    L19,
    RungAccum,
    _git_commit,
)

STORE_ROOT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1310_xpersona/hf_dl/"
    "issue1310_char_map/analysis_tensors/store_onpolicy"
)
TIER15_JSON = (
    _REPO_ROOT / "eval_results/issue_1310/xpersona_similarity/tier15_intercept_refit/results.json"
)
OUT_DIR = _REPO_ROOT / "eval_results/issue_1310/xpersona_similarity/tier2_sides"
NAIVE_GATE_TOL = 1e-4


def main() -> None:
    import issue825_fit_cells as fit825
    import issue1310_xpersona_similarity as v1

    fit825.GCV_DOF_CAP = v1.DOF_CAP

    committed = json.loads(TIER15_JSON.read_text())["directions"]
    out: dict = {}
    t_start = time.time()
    for model in ("base", "instruct"):
        arrays = v1.load_persona_arrays(STORE_ROOT, model)
        cells = {
            p: {
                "X": arrays[p]["X"][:, L19, :].astype(np.float64),
                "Y": arrays[p]["Y"][:, L19, :].astype(np.float64),
                "folds": arrays[p]["folds"],
            }
            for p in v1.PERSONAS
        }
        del arrays
        for src in v1.PERSONAS:
            for tgt in v1.PERSONAS:
                if src == tgt:
                    continue
                Xs, Ys = cells[src]["X"], cells[src]["Y"]
                Xt, Yt = cells[tgt]["X"], cells[tgt]["Y"]
                folds = cells[tgt]["folds"]
                n, d = Yt.shape
                acc = {k: RungAccum(n, d) for k in ("naive", "tier2a", "tier2b", "tier15")}
                decomp = []
                for k in range(v1.N_FOLDS):
                    tr = folds != k
                    te = folds == k
                    if te.sum() == 0 or tr.sum() < 3:
                        continue
                    cache = fit825._prep_fold(Xs[tr], Xt)
                    pn_all = fit825._ridge_predict_cached(cache, Ys[tr]).astype(np.float64)
                    p_bar_t = pn_all[tr].mean(0)
                    y_s = Ys[tr].mean(0)
                    y_t = Yt[tr].mean(0)
                    v_ctx = p_bar_t - y_s  # W_s Δx in prediction space
                    v_ans = y_t - y_s  # Δy
                    true_te = Yt[te].astype(np.float64)
                    acc["naive"].add_fold(te, pn_all[te], true_te)
                    acc["tier2a"].add_fold(te, pn_all[te] - v_ctx, true_te)
                    acc["tier2b"].add_fold(te, pn_all[te] + v_ans, true_te)
                    acc["tier15"].add_fold(te, pn_all[te] - v_ctx + v_ans, true_te)
                    nc, na = float(np.linalg.norm(v_ctx)), float(np.linalg.norm(v_ans))
                    decomp.append(
                        {
                            "norm_ws_dx": nc,
                            "norm_dy": na,
                            "cos_ws_dx_dy": float(v_ctx @ v_ans / (nc * na + 1e-12)),
                            "norm_net": float(np.linalg.norm(v_ans - v_ctx)),
                        }
                    )
                res = {k: a.result(Yt) for k, a in acc.items()}
                key = f"{model}.{src}->{tgt}"
                com = committed[key]
                for rung in ("naive", "tier15"):
                    got, want = res[rung]["r2_foldmean"], com[rung]["r2_foldmean"]
                    assert abs(got - want) <= NAIVE_GATE_TOL, (key, rung, got, want)
                ceil = com["within"]["r2_foldmean"]
                res["within_r2_foldmean_committed"] = ceil
                res["fractions_foldmean"] = {
                    r: res[r]["r2_foldmean"] / ceil for r in ("naive", "tier2a", "tier2b", "tier15")
                }
                res["decomp_fold_mean"] = {
                    k: float(np.mean([f[k] for f in decomp])) for k in decomp[0]
                }
                res["decomp_per_fold"] = decomp
                out[key] = res
                fm = res["decomp_fold_mean"]
                fr = res["fractions_foldmean"]
                print(
                    f"{key}: naive={fr['naive']:.2f} 2a(ctx)={fr['tier2a']:.2f} "
                    f"2b(ans)={fr['tier2b']:.2f} t2={fr['tier15']:.2f} | "
                    f"|WsDx|={fm['norm_ws_dx']:.2f} |Dy|={fm['norm_dy']:.2f} "
                    f"cos={fm['cos_ws_dx_dy']:.2f}",
                    flush=True,
                )
        # Raw-space geometry (fit-free): where do the personas separate, and is
        # the shift direction shared across positions? Context slot and answer
        # mean are both layer-19 residual-stream summaries in the SAME basis,
        # so cos(Δx, Δy) is a meaningful cross-position comparison.
        raw: dict = {"per_pair": {}, "per_persona": {}}
        mx = {p: cells[p]["X"].mean(0) for p in v1.PERSONAS}
        my = {p: cells[p]["Y"].mean(0) for p in v1.PERSONAS}
        spread_x = {
            p: float(np.sqrt(((cells[p]["X"] - mx[p]) ** 2).sum(1).mean())) for p in v1.PERSONAS
        }
        spread_y = {
            p: float(np.sqrt(((cells[p]["Y"] - my[p]) ** 2).sum(1).mean())) for p in v1.PERSONAS
        }
        gx = np.mean([mx[p] for p in v1.PERSONAS], axis=0)
        gy = np.mean([my[p] for p in v1.PERSONAS], axis=0)
        for p in v1.PERSONAS:
            vx, vy = mx[p] - gx, my[p] - gy
            raw["per_persona"][p] = {
                "cos_dx_dy_vs_global_mean": float(
                    vx @ vy / (np.linalg.norm(vx) * np.linalg.norm(vy) + 1e-12)
                ),
                "spread_x": spread_x[p],
                "spread_y": spread_y[p],
            }
        for src in v1.PERSONAS:
            for tgt in v1.PERSONAS:
                if src == tgt:
                    continue
                dx, dy = mx[tgt] - mx[src], my[tgt] - my[src]
                ndx, ndy = float(np.linalg.norm(dx)), float(np.linalg.norm(dy))
                sx = 0.5 * (spread_x[src] + spread_x[tgt])
                sy = 0.5 * (spread_y[src] + spread_y[tgt])
                raw["per_pair"][f"{src}->{tgt}"] = {
                    "norm_dx_raw": ndx,
                    "norm_dy_raw": ndy,
                    "effect_size_x": ndx / sx,
                    "effect_size_y": ndy / sy,
                    "cos_dx_dy_raw": float(dx @ dy / (ndx * ndy + 1e-12)),
                }
        out[f"{model}._raw_space"] = raw
        for pair, r in list(raw["per_pair"].items())[:12]:
            print(
                f"raw {model}.{pair}: |Dx|={r['norm_dx_raw']:.2f} (es {r['effect_size_x']:.2f}) "
                f"|Dy|={r['norm_dy_raw']:.2f} (es {r['effect_size_y']:.2f}) "
                f"cos(Dx,Dy)={r['cos_dx_dy_raw']:.2f}",
                flush=True,
            )
        del cells
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "script": "scripts/issue1639_tier2_sides.py",
            "git_commit": _git_commit(),
            "layer": L19,
            "date": "2026-07-24",
            "provenance": "user-chat inline free-analysis round (tier-2 side attribution)",
            "gates": "naive+tier15 equality vs tier15_intercept_refit/results.json, tol 1e-4",
            "wall_seconds": time.time() - t_start,
        },
        "directions": out,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[done] {time.time() - t_start:.0f}s -> {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
