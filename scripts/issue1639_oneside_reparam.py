"""One-sided Tier-4 reparameterization for the character 4x4 (#1639/#1310).

User-chat inline free-analysis round (2026-07-24; third round in the
tier-ladder series, after scripts/issue1639_tier15_intercept_refit.py and
scripts/issue1639_tier2_sides.py). Mirrors the committed
issue1310_xpersona_similarity.reparam_chain (A_ans o M_source o A_ctx, all
stages ma-ridge-fit on train folds) with ONE alignment stage dropped:

  input-only  : recenter( M_s( A_ctx(x_t) ) )   — A_ctx: x_t -> x_s, paired
  output-only : recenter( A_ans( M_s(x_t) ) )   — A_ans: y_s -> y_t, paired
  full chain  : A_ans( M_s( A_ctx(x_t) ) )       — equality-gated vs the
                committed reparam recovery_r2_foldmean (tol 1e-4)
  naive+offset: recenter( M_s(x_t) )             — gated vs the committed
                tier15 rung (tol 1e-4)

recenter(.) = final-prediction recentering with target train-fold means (the
least-squares-optimal intercept given the frozen composite; the affine
identity from the tier-1.5 round), so every one-sided rung isolates the
LINEAR-part question on top of the already-attributed constant shift.
Algebraic note: input-only composites are confined to M_s's column space and
output-only composites to M_s's row space — each one-sided rung freezes one
entire side of the operator by construction.

Load-bearing-middle null per one-sided rung: M_s refit on scenario-shuffled
train answers (real alignments, 5 draws, cached-prep reuse — the
reparam_chain "shuffle" center adapted to the one-sided composites).

Alignment own-reads per pair and side (standing mapping rule): ridge
alignment held-out R^2, identity+learned-bias baseline R^2
(mapping_baselines.identity_bias_predict), and kNN retrieval
(mapping_baselines.knn_retrieval, pool = held-out targets).

CLI: uv run python scripts/issue1639_oneside_reparam.py
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
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from issue1639_tier15_intercept_refit import _git_commit  # noqa: E402

L19 = 19
STORE_ROOT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1310_xpersona/hf_dl/"
    "issue1310_char_map/analysis_tensors/store_onpolicy"
)
XPS = _REPO_ROOT / "eval_results/issue_1310/xpersona_similarity"
OUT_DIR = XPS / "oneside_reparam"
GATE_TOL = 1e-4
N_NULL_DRAWS = 5


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    mu = true.mean(0)
    sst = float(np.sum((true - mu) ** 2))
    return float("nan") if sst < 1e-12 else 1.0 - float(np.sum((true - pred) ** 2)) / sst


class FoldPooled:
    """Fold-test-mean pooled R^2 accumulator (the committed rig's convention)."""

    def __init__(self):
        self.ss = {"res": 0.0, "tot": 0.0}

    def add(self, pred: np.ndarray, true: np.ndarray) -> None:
        mu = true.mean(0)
        self.ss["res"] += float(np.sum((true - pred) ** 2))
        self.ss["tot"] += float(np.sum((true - mu) ** 2))

    def r2(self) -> float:
        return float("nan") if self.ss["tot"] < 1e-12 else 1.0 - self.ss["res"] / self.ss["tot"]


def main() -> None:
    import issue1310_xpersona_similarity as v1
    import issue825_map_alignment as ma

    ma.GCV_DOF_CAP = v1.DOF_CAP
    ma.LAMBDA_SELECTION = "gcv"

    committed_rep = {
        m: json.loads((XPS / f"reparam_{m}.json").read_text())["ordered_pairs"]
        for m in ("base", "instruct")
    }
    committed_t15 = json.loads((XPS / "tier15_intercept_refit/results.json").read_text())[
        "directions"
    ]

    out: dict = {}
    t_start = time.time()
    rng = np.random.default_rng(v1.FIT_SEED + 101)
    for model in ("base", "instruct"):
        arrays = v1.load_persona_arrays(STORE_ROOT, model)
        cells = {
            p: {
                "X": torch.as_tensor(
                    arrays[p]["X"][:, L19, :].astype(np.float64), dtype=torch.float64
                ),
                "Y": torch.as_tensor(
                    arrays[p]["Y"][:, L19, :].astype(np.float64), dtype=torch.float64
                ),
                "folds": arrays[p]["folds"],
            }
            for p in v1.PERSONAS
        }
        del arrays
        for src in v1.PERSONAS:
            for tgt in v1.PERSONAS:
                if src == tgt:
                    continue
                xs, ys = cells[src]["X"], cells[src]["Y"]
                xt, yt = cells[tgt]["X"], cells[tgt]["Y"]
                folds = cells[tgt]["folds"]
                n = xt.shape[0]
                acc = {
                    k: FoldPooled()
                    for k in ("naive_off", "input_only", "output_only", "full_chain")
                }
                null_acc = {
                    f"{k}_null{j}": FoldPooled()
                    for k in ("input_only", "output_only")
                    for j in range(N_NULL_DRAWS)
                }
                align = {
                    k: {"ridge": FoldPooled(), "idbias": FoldPooled(), "knn": []}
                    for k in ("ctx", "ans")
                }
                perms = [rng.permutation(n) for _ in range(N_NULL_DRAWS)]
                for k in range(v1.N_FOLDS):
                    te = folds == k
                    tr = folds != k
                    if te.sum() == 0 or tr.sum() < 3:
                        continue
                    trt, tet = torch.as_tensor(tr), torch.as_tensor(te)
                    yt_tr_mean = yt[trt].mean(0)
                    true_te = yt[tet].numpy()

                    # Alignment stages (each ma-ridge-fit on train folds only).
                    prep_ctx = ma._ridge_prep(xt[trt])
                    xshat_all = ma._ridge_predict(prep_ctx, xs[trt], xt)
                    prep_m = ma._ridge_prep(xs[trt])
                    y_ali_all = ma._ridge_predict(prep_m, ys[trt], xshat_all)
                    y_raw_all = ma._ridge_predict(prep_m, ys[trt], xt)
                    prep_ans = ma._ridge_prep(ys[trt])

                    def recenter(pred_all):
                        return (pred_all[tet] - pred_all[trt].mean(0) + yt_tr_mean).numpy()

                    acc["naive_off"].add(recenter(y_raw_all), true_te)
                    acc["input_only"].add(recenter(y_ali_all), true_te)
                    out_only = ma._ridge_predict(prep_ans, yt[trt], y_raw_all)
                    acc["output_only"].add(recenter(out_only), true_te)
                    full = ma._ridge_predict(prep_ans, yt[trt], y_ali_all[tet])
                    acc["full_chain"].add(full.numpy(), true_te)

                    # Load-bearing-middle nulls: M_s refit on shuffled answers
                    # (real alignments; cached preps reused — Y-side only).
                    for j, perm in enumerate(perms):
                        ys_shuf = ys[torch.as_tensor(perm)]
                        yn_ali = ma._ridge_predict(prep_m, ys_shuf[trt], xshat_all)
                        null_acc[f"input_only_null{j}"].add(recenter(yn_ali), true_te)
                        yn_raw = ma._ridge_predict(prep_m, ys_shuf[trt], xt)
                        yn_out = ma._ridge_predict(prep_ans, yt[trt], yn_raw)
                        null_acc[f"output_only_null{j}"].add(recenter(yn_out), true_te)

                    # Alignment own-reads (ridge + identity+bias + kNN).
                    ctx_pred = xshat_all[tet].numpy()
                    ctx_true = xs[tet].numpy()
                    align["ctx"]["ridge"].add(ctx_pred, ctx_true)
                    align["ctx"]["idbias"].add(
                        identity_bias_predict(xt[trt].numpy(), xs[trt].numpy(), xt[tet].numpy()),
                        ctx_true,
                    )
                    align["ctx"]["knn"].append(
                        knn_retrieval(ctx_pred, ctx_true, ks=(1, 5))["acc_at_k"][1]
                    )
                    ans_pred = ma._ridge_predict(prep_ans, yt[trt], ys[tet]).numpy()
                    align["ans"]["ridge"].add(ans_pred, true_te)
                    align["ans"]["idbias"].add(
                        identity_bias_predict(ys[trt].numpy(), yt[trt].numpy(), ys[tet].numpy()),
                        true_te,
                    )
                    align["ans"]["knn"].append(
                        knn_retrieval(ans_pred, true_te, ks=(1, 5))["acc_at_k"][1]
                    )

                key = f"{model}.{src}->{tgt}"
                pair = f"{src}->{tgt}"
                res = {k: a.r2() for k, a in acc.items()}
                want_full = float(committed_rep[model][pair]["recovery_r2_foldmean"])
                assert abs(res["full_chain"] - want_full) <= GATE_TOL, (
                    key,
                    res["full_chain"],
                    want_full,
                )
                want_t15 = float(committed_t15[key]["tier15"]["r2_foldmean"])
                assert abs(res["naive_off"] - want_t15) <= GATE_TOL, (
                    key,
                    res["naive_off"],
                    want_t15,
                )
                ceil = float(committed_rep[model][pair]["target_ceiling_foldmean"])
                nulls = {
                    k: float(
                        np.nanmean([null_acc[f"{k}_null{j}"].r2() for j in range(N_NULL_DRAWS)])
                    )
                    for k in ("input_only", "output_only")
                }
                rec = {
                    "rungs_r2_foldmean": res,
                    "ceiling_committed": ceil,
                    "fractions": {k: v / ceil for k, v in res.items()},
                    "oneside_shuffle_middle_null_mean": nulls,
                    "alignments": {
                        side: {
                            "ridge_r2": align[side]["ridge"].r2(),
                            "identity_bias_r2": align[side]["idbias"].r2(),
                            "knn_acc_at_1_euclid_mean": float(np.mean(align[side]["knn"])),
                        }
                        for side in ("ctx", "ans")
                    },
                }
                out[key] = rec
                fr = rec["fractions"]
                al = rec["alignments"]
                print(
                    f"{key}: t2={fr['naive_off']:.2f} inGL={fr['input_only']:.2f} "
                    f"outGL={fr['output_only']:.2f} full={fr['full_chain']:.2f} "
                    f"(nulls in/out {nulls['input_only']:.2f}/{nulls['output_only']:.2f}) | "
                    f"alignR2 ctx={al['ctx']['ridge_r2']:.2f} ans={al['ans']['ridge_r2']:.2f} "
                    f"idbias ctx={al['ctx']['identity_bias_r2']:.2f} "
                    f"ans={al['ans']['identity_bias_r2']:.2f}",
                    flush=True,
                )
        del cells
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "script": "scripts/issue1639_oneside_reparam.py",
            "git_commit": _git_commit(),
            "layer": L19,
            "date": "2026-07-24",
            "provenance": "user-chat inline free-analysis round (one-sided Tier-4 reparam)",
            "gates": "full_chain vs committed reparam + naive_off vs committed tier15, tol 1e-4",
            "wall_seconds": time.time() - t_start,
        },
        "directions": out,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[done] {time.time() - t_start:.0f}s -> {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
