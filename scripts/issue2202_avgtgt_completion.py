#!/usr/bin/env python3
"""Issue #2202 inline free-analysis round ``avgtgt-completion`` (user-chat carve-out).

Completes the draw-averaged-target coverage matrix: the freshwhiten-avg round
scored draw-averaged targets (covered row's pool entry -> mean of its 5
on-policy draws; pool stays 9,941; eval on the 1,988 resample-covered rows)
under 4 ridge conventions only. This round extends the single-vs-averaged
read to:

- ridge under the metric-zoo top conventions ``csls_k10_whitencos``,
  ``csls_pen_whitencos_g10``, ``hubdeg_pen_whitencos_g05`` (functions imported
  from ``scripts/issue2202_metric_zoo.py`` verbatim), plus raw-euclidean +
  whitened-cosine recomputed here so the matrix is self-contained;
- the 4 banked #1738 nonlinear prediction files (mlp_w8192, mlp_w8192_seed43,
  krr_nystrom, residual_skip) under raw-euclidean + whitened-cosine +
  csls_k10_whitencos;
- the 2 contrastive maps (linear_tau0.05, mlp_tau0.05), holdout predictions
  RECOMPUTED on the VM from the HF-banked weights (one matmul / one batched
  CPU forward via ``issue2202_contrastive_maps.ContrastiveMap``), gated on an
  exact reconciliation of the recomputed single-draw full-pool raw-cosine
  acc@1 against the banked battery (0.8786 linear / 0.8966 mlp) BEFORE any
  draw-averaged number is trusted; same 3 conventions.

Pool-side convention statistics (the CSLS r_j top-k mean over the query bank,
the hubdeg pool-internal in-degree and the sigma(S) scale) are RECOMPUTED ON
THE MODIFIED POOL for every draw-averaged cell — the averaged entries change
the pool geometry; the query bank is always the map's own full 9,941
held-out predictions. Analysis-only; vectorized batteries; no new fits.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps must bind BEFORE numpy/torch import (#847)

import issue1738_characterize as CH  # noqa: E402
import issue2202_contrastive_maps as CM  # noqa: E402
import issue2202_failchar as FC  # noqa: E402
import issue2202_metric_zoo as MZ  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

FW_STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten")  # read-only reuse
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2202_avgtgt")
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_2202" / "avgtgt_completion"
PARTIAL = OUT_DIR / "partial.jsonl"
K_DRAWS = 4
N_COVERED = 1_988
FORWARD_BATCH = 2_048
# banked single-draw full-pool raw-cosine acc@1 (contrastive_maps_battery.json)
BANKED_CONTRASTIVE_RAWCOS = {
    "contrastive_linear": 0.8785836434966301,
    "contrastive_mlp": 0.896589880293733,
}
RECON_TOL_ROWS = 5  # fp32-forward BLAS-order drift allowance on knife-edge ties
NONLINEAR_FILES = {
    "mlp_w8192": "context_L19_mlp_w8192.npz",
    "mlp_w8192_seed43": "context_L19_mlp_w8192_seed43.npz",
    "krr_nystrom": "context_L19_krr_nystrom.npz",
    "residual_skip": "context_L19_residual_skip.npz",
}
CONTRASTIVE_FITS = {
    "contrastive_linear": ("linear", "linear_tau0.05.pt"),
    "contrastive_mlp": ("mlp", "mlp_tau0.05.pt"),
}
RIDGE_CONVS = (
    "raw_euclidean",
    "whiten_cos",
    "csls_k10_whitencos",
    "csls_pen_whitencos_g10",
    "hubdeg_pen_whitencos_g05",
)
OTHER_CONVS = ("raw_euclidean", "whiten_cos", "csls_k10_whitencos")


def stage_inputs() -> dict:
    """Stage the contrastive weights + cx_holdout into THIS round's staging dir
    (the freshwhiten dir is reused READ-ONLY — a sibling round reads it)."""
    from huggingface_hub import HfApi

    STAGE.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    hub.stage_hub_file(
        FC.C.HF_DATA_REPO,
        f"{FC.HF_PREFIX_2202}/analysis_tensors/cx_holdout_L19.npz",
        STAGE / "cx_holdout_L19.npz",
    )
    for _tag, (_family, fname) in CONTRASTIVE_FITS.items():
        hub.stage_hub_file(
            FC.C.HF_DATA_REPO,
            f"{FC.HF_PREFIX_2202}/contrastive_maps/fits/{fname}",
            STAGE / fname,
        )
    return {
        "stage_dir": str(STAGE),
        "freshwhiten_reused_read_only": str(FW_STAGE),
        "data_repo_head": api.repo_info(FC.C.HF_DATA_REPO, repo_type="dataset").sha,
    }


def contrastive_predict(family: str, fname: str, cx32: np.ndarray) -> np.ndarray:
    """Recompute holdout predictions from the banked fit weights: reconstruct
    ``ContrastiveMap`` (forward = head((x - xmu)/xsd) + c), load the banked
    state_dict, run one batched CPU fp32 forward; returns fp64 (n, H)."""
    doc = torch.load(STAGE / fname, map_location="cpu", weights_only=False)
    assert doc["family"] == family, (doc["family"], family)
    sd = doc["state_dict"]
    payload = {
        "xmu": sd["xmu"],
        "xsd": sd["xsd"],
        "ymu": sd["c"],
        "W": sd["w"] if "w" in sd else torch.zeros(FC.H_DIM, FC.H_DIM),
    }
    model = CM.ContrastiveMap(family, payload)
    model.load_state_dict(sd)
    model.eval()
    outs = []
    t0 = time.time()
    n = len(cx32)
    with torch.no_grad():
        for k, s in enumerate(range(0, n, FORWARD_BATCH)):
            outs.append(model(torch.as_tensor(cx32[s : s + FORWARD_BATCH])).numpy())
            print(
                f"[fwd-{family}] unit {k + 1}/{(n + FORWARD_BATCH - 1) // FORWARD_BATCH} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    pred = np.concatenate(outs).astype(np.float64)
    assert pred.shape == (n, FC.H_DIM), pred.shape
    return pred


def load_done() -> dict[str, dict]:
    """Resume predicate: per-map completed cell blocks from partial.jsonl."""
    done: dict[str, dict] = {}
    if PARTIAL.is_file():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["map"]] = rec
    return done


def append_partial(rec: dict) -> None:
    """Atomic-enough single-line append (one writer; per-unit persistence)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARTIAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def main() -> int:
    t0 = time.time()
    revisions = stage_inputs()

    # inputs (freshwhiten staging, read-only; same asserts as that round)
    pd_ = np.load(FW_STAGE / "pred16.npz")
    yd = np.load(FW_STAGE / "y_holdout_L19.npz")
    pred_ridge = pd_["pred16"].astype(np.float64)
    y16 = yd["y16"].astype(np.float64)
    pci = np.asarray(pd_["ci"], dtype=np.int64)
    assert (pci == np.asarray(yd["ci"], dtype=np.int64)).all(), "pred16/y_holdout ci misalign"
    assert pred_ridge.shape == y16.shape == (FC.EXPECTED_N, FC.H_DIM)
    n_pool = y16.shape[0]
    full_idx = np.arange(n_pool)

    kns = SimpleNamespace(
        local_kresample_dir=str(FW_STAGE / "kresample"),
        scratch=str(STAGE / "scratch"),
        hf_prefix="",
    )
    kci, vres = CH._load_kresample_v(kns, [FC.LAYER])
    assert vres.shape == (N_COVERED, K_DRAWS, 1, FC.H_DIM), vres.shape
    draws = vres[:, :, 0, :].astype(np.float64)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)

    wz = np.load(FW_STAGE / "whiten_stats.npz")
    mu_a = np.asarray(wz["mu_A"], dtype=np.float64)
    ell = np.asarray(wz["L"], dtype=np.float64)

    def _wh(x: np.ndarray) -> np.ndarray:
        return solve_triangular(ell, (np.asarray(x, np.float64) - mu_a).T, lower=True).T

    cxz = np.load(STAGE / "cx_holdout_L19.npz")
    assert (np.asarray(cxz["ci"], dtype=np.int64) == pci).all(), "cx_holdout ci misalign"
    cx32 = cxz["cx"].astype(np.float32)

    # prediction sets
    preds: dict[str, np.ndarray] = {"ridge": pred_ridge}
    for tag, fname in NONLINEAR_FILES.items():
        z = np.load(FW_STAGE / "nonlinear" / fname)
        assert (np.asarray(z["ci"], dtype=np.int64) == pci).all(), f"{tag} ci misalign"
        preds[tag] = z["pred16"].astype(np.float64)
    for tag, (family, fname) in CONTRASTIVE_FITS.items():
        preds[tag] = contrastive_predict(family, fname, cx32)

    # contrastive reconciliation gate (BEFORE any draw-averaged read is trusted)
    recon: dict[str, dict] = {}
    for tag, banked in BANKED_CONTRASTIVE_RAWCOS.items():
        rec = knn_retrieval(preds[tag], y16, ks=(1, 5, 10), metric="cosine")
        delta = rec["acc_at_k"][1] - banked
        recon[tag] = {
            "recomputed_rawcos_acc1": rec["acc_at_k"][1],
            "banked": banked,
            "delta": delta,
        }
        print(
            f"[recon] {tag} raw_cos acc@1 {rec['acc_at_k'][1]:.6f} banked {banked:.6f} delta {delta:+.2e}",
            flush=True,
        )
        assert abs(delta) <= RECON_TOL_ROWS / n_pool + 1e-12, (
            f"{tag} recomputed raw_cos acc@1 does not reconcile with the banked battery: {recon[tag]}"
        )

    # draw-averaged pool (raw + whitened) and whitened pools, normalized copies
    avg = (y16[pos] + draws.sum(axis=1)) / (1 + K_DRAWS)
    pool_mod = y16.copy()
    pool_mod[pos] = avg
    y16w = _wh(y16)
    pool_modw = y16w.copy()
    pool_modw[pos] = _wh(avg)
    qwn = {"single": normalize_rows(y16w), "avg": normalize_rows(pool_modw)}
    pool_raw = {"single": y16, "avg": pool_mod}
    pool_w = {"single": y16w, "avg": pool_modw}

    # hubdeg pool-internal similarities (ridge only; recomputed per pool variant)
    s_pp: dict[str, np.ndarray] = {}

    done = load_done()
    for mi, (tag, pred) in enumerate(preds.items()):
        if tag in done:
            print(f"[avgtgt] map {mi + 1}/{len(preds)} {tag} SKIP (resume)", flush=True)
            continue
        tm = time.time()
        convs = RIDGE_CONVS if tag == "ridge" else OTHER_CONVS
        predc = pred[pos]
        predw = _wh(pred)
        pwn = normalize_rows(predw)
        cells: dict[str, dict] = {c: {} for c in convs}
        for variant in ("single", "avg"):
            r, _, _ = FC.ranks_of_targets(
                predc, pool_raw[variant], pos, "euclidean", phase=f"{tag}-raw-{variant}"
            )
            cells["raw_euclidean"][variant] = MZ.ranks_summary(r, n_pool)
            r, _, _ = FC.ranks_of_targets(
                predw[pos], pool_w[variant], pos, "cosine", phase=f"{tag}-wcos-{variant}"
            )
            cells["whiten_cos"][variant] = MZ.ranks_summary(r, n_pool)
            t1 = time.time()
            s_wc = pwn @ qwn[variant].T  # full query bank x (possibly modified) pool
            print(
                f"[{tag}-swc-{variant}] S ({n_pool}x{n_pool}) in {time.time() - t1:.1f}s",
                flush=True,
            )
            rk = MZ.csls_ranks(s_wc, full_idx, 0.5)
            cells["csls_k10_whitencos"][variant] = MZ.ranks_summary(rk[pos], n_pool)
            if tag == "ridge":
                rk = MZ.csls_ranks(s_wc, full_idx, 1.0)
                cells["csls_pen_whitencos_g10"][variant] = MZ.ranks_summary(rk[pos], n_pool)
                if variant not in s_pp:
                    s_pp[variant] = qwn[variant] @ qwn[variant].T
                rk = MZ.hubdeg_ranks(s_wc, s_pp[variant], full_idx, 0.5)
                cells["hubdeg_pen_whitencos_g05"][variant] = MZ.ranks_summary(rk[pos], n_pool)
            del s_wc
            print(
                f"[avgtgt] map {mi + 1}/{len(preds)} {tag} variant={variant} "
                f"elapsed={time.time() - tm:.1f}s",
                flush=True,
            )
        append_partial({"map": tag, "cells": cells})
        done[tag] = {"map": tag, "cells": cells}
        del predw, pwn

    matrix = {tag: done[tag]["cells"] for tag in preds}
    summary = {
        "round": "avgtgt-completion (user-chat inline free-analysis, task #2202)",
        "conventions": {
            "eval_rows": (
                "all cells evaluated on the 1,988 resample-covered rows; pool stays 9,941; "
                "single = original held-out answer targets, avg = covered pool entries replaced "
                "by mean(original + 4 fresh on-policy draws) (the freshwhiten-avg convention)"
            ),
            "acc_at_1": "(mid-rank <= 1).mean(); ties at top count as failure",
            "whiten_cos": "z = L^-1(x - mu_A), banked shrunk train-answer Cholesky (lam=0.1); cosine in z-space",
            "csls_k10_whitencos": (
                "CSLS on the whitened-cosine similarity: score = S - gamma*r_j with r_j = mean of "
                f"the top-{MZ.K_LOCAL} similarities of pool item j over the QUERY BANK (the map's "
                "own full 9,941 predictions), gamma=0.5 (rank-exact CSLS); "
                "csls_pen_whitencos_g10 = gamma=1.0 (ridge only)"
            ),
            "hubdeg_pen_whitencos_g05": (
                "in-degree hub penalty on whitened cosine: score = S - 0.5*sigma(S)*zscore(N_k), "
                f"N_k = pool-internal kNN in-degree (k={MZ.K_LOCAL}); ridge only"
            ),
            "modified_pool_statistics": (
                "for every draw-averaged cell the pool-side statistics are RECOMPUTED on the "
                "MODIFIED pool: CSLS r_j from S against the modified pool, hubdeg in-degree + "
                "sigma(S) from the modified pool's internal similarities — the averaged entries "
                "change the pool geometry, so single-draw pool statistics would be stale"
            ),
            "contrastive_predictions": (
                "recomputed on the VM from the HF-banked fit weights "
                "(issue2202_ctxfail/contrastive_maps/fits/{linear,mlp}_tau0.05.pt): "
                "pred = head((cx - xmu)/xsd) + c per issue2202_contrastive_maps.ContrastiveMap, "
                "fp32 batched CPU forward, cast fp64; gated on the raw-cosine full-pool "
                "reconciliation below before any draw-averaged read"
            ),
        },
        "n_covered": int(N_COVERED),
        "n_pool": int(n_pool),
        "k_draws": int(K_DRAWS),
        "contrastive_reconciliation": recon,
        "matrix": matrix,
        "staging": revisions,
        "meta": FC.meta_block({"wall_seconds": round(time.time() - t0, 1)}),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FC.atomic_json(OUT_DIR / "summary.json", summary)
    print(f"[done] wrote {OUT_DIR / 'summary.json'} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
