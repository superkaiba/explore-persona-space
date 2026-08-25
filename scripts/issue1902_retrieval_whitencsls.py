#!/usr/bin/env python3
"""Issue #1902 follow-up — plot-6 retrieval under the standing whitened-cos+CSLS
convention, plus an identity+bias baseline arm.

The committed #1902 retrieval reads are PLAIN kNN on the raw predicted vector
(``issue1902_fits._cell_baselines`` -> ``mapping_baselines.knn_retrieval``,
euclidean + cosine). The paper's standing retrieval convention (Plot 1, Plot 5)
is whitened cosine + CSLS k=10 (``scripts/issue2202_metric_zoo``:
``Transforms.chol_whiten`` at lam=0.1 -> ``csls_ranks``), so plot 6's panel is
not comparable to the paper's other retrieval reads. The identity+bias baseline
has committed R^2 but no retrieval read at all.

This round refits the layer-31 single-turn context-arm maps from the #1902
activation store (the SAME batched ridge helper that produced the committed
numbers), gates each refit against the committed pooled held-out R^2, and
recomputes acc@1 under whitened cosine + CSLS for four arms per stage:

    self         map fit and evaluated within the stage
    transferred  the previous stage's map applied unchanged to this stage's
                 held-out contexts, scored against this stage's answers
                 (the transfer battery's ``direct`` mode)
    crossfit     map fit from the previous stage's contexts onto this stage's
                 on-policy answer text (the grid cell ``grid_<i><j>``)
    identity     identity + learned-bias baseline against the same target

Usage (VM; ~2.05 GB staged off the boot disk, ~20-30 min CPU, 0 GPU-h)::

    uv run python scripts/issue1902_retrieval_whitencsls.py --phase stage
    uv run python scripts/issue1902_retrieval_whitencsls.py --phase fits

Writes ``eval_results/issue_1902/retrieval_whitencsls/retrieval.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps land BEFORE numpy on the shared VM (#847)

import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

import issue1902_common as C  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
)
from explore_persona_space.analysis.null_battery import (  # noqa: E402
    shrunk_cholesky_from_cov,
)

ISSUE = 1902
LAYER = 31  # layer* — the frozen selected layer for the single-turn context arm
CORPUS = C.CORPUS_SINGLE
STAGES = ("B", "S", "D", "R")
STAGE_LABEL = {"B": "base", "S": "SFT", "D": "DPO", "R": "RLVR"}
TRANSITIONS = (("B", "S"), ("S", "D"), ("D", "R"))
N_FOLDS = 6
WHITEN_LAMBDA = 0.1  # the banked #2202 shrinkage
CSLS_K = 10
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
EVAL_DIR = PROJECT_ROOT / "eval_results" / f"issue_{ISSUE}"
OUT_DIR = EVAL_DIR / "retrieval_whitencsls"
# Multi-GB staging goes to the data disk, never the boot disk (#1393).
STAGE_ROOT = Path(
    os.environ.get(
        "EPM_I1902_WHITENCSLS_STAGE",
        f"/mnt/eps-data/{os.environ.get('USER', 'thomasjiralerspong')}/issue1902_whitencsls",
    )
)
STORE_ROOT = STAGE_ROOT / "store"
# A refit that does not reproduce the committed pooled R^2 is a broken
# reproduction, not a new number — halt rather than ship its retrieval read.
R2_TOL = 5e-3


# ── store staging (exact files, never a prefix snapshot) ─────────────────────


def needed_store_relpaths() -> list[str]:
    """The 11 layer-31 shards + their row_index manifests this round reads."""
    rels: list[str] = []
    for m in STAGES:
        rels.append(C.ctx_store_relpath(m, CORPUS, LAYER))
        rels.append(C.cell_row_index_relpath(m, C.CTX_SOURCE, CORPUS))
        rels.append(C.answer_store_relpath(m, m, CORPUS, LAYER))
        rels.append(C.cell_row_index_relpath(m, m, CORPUS))
    for i, j in TRANSITIONS:
        rels.append(C.answer_store_relpath(i, j, CORPUS, LAYER))
        rels.append(C.cell_row_index_relpath(i, j, CORPUS))
    # dedup, order-stable
    return list(dict.fromkeys(rels))


def phase_stage() -> dict:
    """Download the exact shards this round reads into the data-disk staging root."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    STORE_ROOT.mkdir(parents=True, exist_ok=True)
    rels = needed_store_relpaths()
    total = 0
    for k, rel in enumerate(rels, 1):
        target = STORE_ROOT / rel
        if target.is_file():
            total += target.stat().st_size
            print(f"[stage] {k}/{len(rels)} {rel} present", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        got = retry_transient(
            lambda rel=rel: hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{C.STORE_HF_PATH}/{rel}",
                local_dir=str(STAGE_ROOT / "_dl"),
            ),
            what=f"stage {rel}",
        )
        Path(got).replace(target)
        total += target.stat().st_size
        print(
            f"[stage] {k}/{len(rels)} {rel} "
            f"{target.stat().st_size / 1e6:.1f}MB in {time.time() - t0:.1f}s",
            flush=True,
        )
    print(f"[stage] done: {len(rels)} files, {total / 1e9:.2f} GB -> {STORE_ROOT}", flush=True)
    return {"n_files": len(rels), "bytes": total, "root": str(STORE_ROOT)}


# ── committed-artifact reads (row ids, fold partition, R^2 gate targets) ─────


def _read_jsonl_ids(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["id"] for line in f if line.strip()]


def load_row_ids() -> list[str]:
    """Store row order, asserted identical across every cell this round reads."""
    ref: list[str] | None = None
    for rel in needed_store_relpaths():
        if not rel.endswith("row_index.jsonl"):
            continue
        ids = _read_jsonl_ids(STORE_ROOT / rel)
        if ref is None:
            ref = ids
        elif ids != ref:
            raise RuntimeError(f"row_id order mismatch in {rel} — matched-target violation")
    if ref is None:
        raise RuntimeError("no row_index.jsonl staged")
    return ref


def load_folds(n_rows: int) -> list[np.ndarray]:
    """Eval-row indices per fold, read from the committed per-fold artifacts.

    The cluster-grouped 6-fold assignment is already frozen in the per-fold
    npz shards (identical across diagonal and grid cells), so it is read back
    rather than re-derived from the k-means manifest.
    """
    folds = []
    seen: set[int] = set()
    for f in range(N_FOLDS):
        ri = np.load(EVAL_DIR / "fits" / "percell" / f"diag_B_{CORPUS}_ctx_f{f}.npz")["row_idx"]
        ri = np.asarray(ri, dtype=np.int64)
        if seen & set(ri.tolist()):
            raise RuntimeError(f"fold {f} overlaps an earlier fold")
        seen |= set(ri.tolist())
        folds.append(ri)
    if len(seen) != n_rows:
        raise RuntimeError(f"fold partition covers {len(seen)} rows, store has {n_rows}")
    return folds


def committed_r2() -> dict[str, float]:
    """Committed pooled held-out R^2 per arm cell — the reproduction gate."""
    grid = json.loads((EVAL_DIR / "fits" / "grid_cells.json").read_text())
    xf = json.loads((EVAL_DIR / "transfer" / "transfer_matrix.json").read_text())
    out: dict[str, float] = {}
    for m in STAGES:
        out[f"self:{m}"] = float(grid["cells"][f"diag_{m}_{CORPUS}_ctx"]["r2_at_star"])
    for i, j in TRANSITIONS:
        out[f"crossfit:{j}"] = float(
            grid["cells"][f"grid_{i}{j}_{CORPUS}_ctx"]["per_layer"][str(LAYER)]["r2"]
        )
        out[f"transferred:{j}"] = float(xf["pairs"][f"{i}->{j}"]["r2"]["direct"])
    return out


# ── store loading ────────────────────────────────────────────────────────────


class Cells:
    """Layer-31 single-turn arrays, loaded once and held as fp32."""

    def __init__(self, ids: list[str]):
        self.ids = ids
        self._ctx: dict[str, np.ndarray] = {}
        self._ans: dict[tuple[str, str], np.ndarray] = {}

    def _load(self, rel: str, key: str) -> np.ndarray:
        import torch

        d = torch.load(STORE_ROOT / rel, map_location="cpu", weights_only=True)
        if [str(x) for x in d["row_ids"]] != self.ids:
            raise RuntimeError(f"row_id mismatch in {rel}")
        return d[key].to(torch.float32).numpy()

    def ctx(self, m: str) -> np.ndarray:
        if m not in self._ctx:
            self._ctx[m] = self._load(C.ctx_store_relpath(m, CORPUS, LAYER), "u_mean")
        return self._ctx[m]

    def ans(self, m: str, s: str) -> np.ndarray:
        if (m, s) not in self._ans:
            self._ans[(m, s)] = self._load(C.answer_store_relpath(m, s, CORPUS, LAYER), "w")
        return self._ans[(m, s)]


# ── retrieval under the standing convention ──────────────────────────────────


def whiten_stats(y_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-fold mean + Cholesky factor of the shrunk covariance (lam=0.1).

    Fitted on TRAIN rows only, so the held-out retrieval read carries no
    target-side leakage. Uses the shared shrink+jitter helper so the transform
    matches the banked #2202 convention exactly.
    """
    y = np.asarray(y_tr, dtype=np.float64)
    mu = y.mean(0)
    yc = y - mu
    cov = (yc.T @ yc) / max(1, yc.shape[0] - 1)
    return mu, shrunk_cholesky_from_cov(cov, WHITEN_LAMBDA)


def whitencsls_acc(pred: np.ndarray, pool: np.ndarray, mu: np.ndarray, chol: np.ndarray) -> dict:
    """acc@k / median rank / MRR under whitened cosine + CSLS k=10.

    ``pool`` is the held-out target set and is its own candidate set, so the
    true target of row i is pool row i (chance = k / n_pool).
    """
    from issue2202_metric_zoo import csls_ranks, ranks_summary

    def _z(x: np.ndarray) -> np.ndarray:
        z = solve_triangular(chol, (np.asarray(x, dtype=np.float64) - mu).T, lower=True).T
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-30)

    s = _z(pred) @ _z(pool).T
    true_idx = np.arange(s.shape[0])
    ranks = csls_ranks(s, true_idx)
    return ranks_summary(ranks, s.shape[1])


# ── fits ─────────────────────────────────────────────────────────────────────


def _pooled_r2(ss_res: float, ss_tot: float) -> float:
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _ridge(x_tr: np.ndarray, y_tr: np.ndarray, x_ev_stack: list[np.ndarray]) -> list[np.ndarray]:
    """One layer-batched ridge fit, evaluated on several eval inputs at once.

    Stacks the eval matrices into ONE call (the #1902 transfer-unit shape), so
    the transferred arm reuses the source stage's fit rather than refitting.
    """
    from issue1902_fits import _batched_ridge

    sizes = [x.shape[0] for x in x_ev_stack]
    stack = np.concatenate(x_ev_stack, axis=0)
    preds = _batched_ridge(x_tr[None], y_tr[None], stack[None], device="cpu")[0]
    out, off = [], 0
    for n in sizes:
        out.append(preds[off : off + n])
        off += n
    return out


def phase_fits() -> dict:
    ids = load_row_ids()
    folds = load_folds(len(ids))
    cells = Cells(ids)
    targets = committed_r2()
    n = len(ids)
    # per (arm, stage): pooled SS across folds + per-fold retrieval summaries
    acc: dict[str, dict] = {}

    def rec(arm: str, stage: str, ss_res: float, ss_tot: float, ret: dict) -> None:
        e = acc.setdefault(arm, {}).setdefault(stage, {"ss_res": 0.0, "ss_tot": 0.0, "folds": []})
        e["ss_res"] += ss_res
        e["ss_tot"] += ss_tot
        e["folds"].append(ret)

    t_start = time.time()
    for f, ev_idx in enumerate(folds):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[ev_idx] = False
        tr_idx = np.flatnonzero(tr_mask)
        print(
            f"[fits] fold {f}/{N_FOLDS - 1} n_tr={tr_idx.size} n_ev={ev_idx.size} "
            f"elapsed={time.time() - t_start:.0f}s",
            flush=True,
        )
        if tr_idx.size <= 4096:
            raise RuntimeError(f"fold {f}: n_train={tr_idx.size} <= d=4096 (degenerate fit)")

        # whitening stats per target cell, fitted on train rows only
        wstats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

        def _w(m: str, s: str) -> tuple[np.ndarray, np.ndarray]:
            if (m, s) not in wstats:
                wstats[(m, s)] = whiten_stats(cells.ans(m, s)[tr_idx])
            return wstats[(m, s)]

        # self map per stage; the eval stack also carries the NEXT stage's
        # contexts so the transferred arm reuses this same fit.
        self_pred: dict[str, np.ndarray] = {}
        xfer_pred: dict[str, np.ndarray] = {}
        for m in STAGES:
            u, w = cells.ctx(m), cells.ans(m, m)
            eval_inputs = [u[ev_idx]]
            nxt = dict(TRANSITIONS).get(m)
            if nxt is not None:
                eval_inputs.append(cells.ctx(nxt)[ev_idx])
            preds = _ridge(u[tr_idx], w[tr_idx], eval_inputs)
            self_pred[m] = preds[0]
            if nxt is not None:
                xfer_pred[nxt] = preds[1]

        for m in STAGES:
            y_ev = cells.ans(m, m)[ev_idx]
            y_tr = cells.ans(m, m)[tr_idx]
            mu, chol = _w(m, m)
            tot = float(((y_ev - y_tr.mean(0)) ** 2).sum())
            # arm 1: own map
            res = float(((y_ev - self_pred[m]) ** 2).sum())
            rec("self", m, res, tot, whitencsls_acc(self_pred[m], y_ev, mu, chol))
            # arm 4: identity + learned bias, same target
            id_pred = identity_bias_predict(cells.ctx(m)[tr_idx], y_tr, cells.ctx(m)[ev_idx])
            res_id = float(((y_ev - id_pred) ** 2).sum())
            rec("identity", m, res_id, tot, whitencsls_acc(id_pred, y_ev, mu, chol))
            # arm 2: previous stage's map applied as-is (transfer 'direct')
            if m in xfer_pred:
                res_x = float(((y_ev - xfer_pred[m]) ** 2).sum())
                rec("transferred", m, res_x, tot, whitencsls_acc(xfer_pred[m], y_ev, mu, chol))

        # arm 3: map refit from the previous stage's contexts onto this stage's answers
        for i, j in TRANSITIONS:
            u, w = cells.ctx(i), cells.ans(i, j)
            pred = _ridge(u[tr_idx], w[tr_idx], [u[ev_idx]])[0]
            y_ev, y_tr = w[ev_idx], w[tr_idx]
            mu, chol = _w(i, j)
            tot = float(((y_ev - y_tr.mean(0)) ** 2).sum())
            res = float(((y_ev - pred) ** 2).sum())
            rec("crossfit", j, res, tot, whitencsls_acc(pred, y_ev, mu, chol))

    # pooled reads + the committed-R^2 reproduction gate
    out: dict[str, dict] = {}
    failures: list[str] = []
    for arm, per_stage in acc.items():
        out[arm] = {}
        for stage, e in per_stage.items():
            r2 = _pooled_r2(e["ss_res"], e["ss_tot"])
            a1 = [float(fr["acc_at_k"][1]) for fr in e["folds"]]
            out[arm][stage] = {
                "r2_pooled": r2,
                "acc1_whitencsls_mean": float(np.mean(a1)),
                "acc1_whitencsls_folds": a1,
                "median_rank_mean": float(np.mean([fr["median_rank"] for fr in e["folds"]])),
                "mrr_mean": float(np.mean([fr["mrr"] for fr in e["folds"]])),
                "n_pool_mean": float(np.mean([fr["n_pool"] for fr in e["folds"]])),
                "chance_at_1_mean": float(np.mean([1.0 / fr["n_pool"] for fr in e["folds"]])),
            }
            key = f"{arm}:{stage}"
            if key in targets:
                delta = abs(r2 - targets[key])
                out[arm][stage]["r2_committed"] = targets[key]
                out[arm][stage]["r2_abs_delta"] = delta
                if delta > R2_TOL:
                    failures.append(f"{key}: refit {r2:.4f} vs committed {targets[key]:.4f}")
    if failures:
        raise RuntimeError(
            "refit does not reproduce the committed R^2 (gate) — "
            + "; ".join(failures)
            + f" (tol {R2_TOL})"
        )

    payload = {
        "issue": ISSUE,
        "layer": LAYER,
        "corpus": CORPUS,
        "arm": "context",
        "fitter": "ridge",
        "retrieval": {
            "metric": "whitened cosine + CSLS",
            "whitening": f"cholesky, diagonal-target shrinkage lam={WHITEN_LAMBDA}, "
            "mu + Sigma fitted on TRAIN folds only",
            "csls_k": CSLS_K,
            "pool": "held-out targets are their own candidate set",
        },
        "r2_gate": {"tol": R2_TOL, "checked": sorted(targets), "status": "PASS"},
        "arms": out,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "retrieval.tmp.json"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUT_DIR / "retrieval.json")
    print(
        f"[fits] done in {time.time() - t_start:.0f}s -> {OUT_DIR / 'retrieval.json'}", flush=True
    )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("stage", "fits"), required=True)
    args = ap.parse_args()
    {"stage": phase_stage, "fits": phase_fits}[args.phase]()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
