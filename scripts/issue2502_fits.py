"""Issue #2502 P4: pooled val-lambda ridge map cx_last -> v_x per model (plan v6 S4 P4).

Per model, fits ONE pooled val-lambda-selected primal ridge map per candidate layer
(reusing the #779 core math over ``LAMBDAS_N50K``), evaluated on the held-out test
partition pooled + per source + per LODO group fold, alongside the mandatory
identity+learned-bias baseline and kNN-retrieval reads (#722 helpers), and persists
the per-context x per-layer x per-arm reconstruction matrix that drives every
selection-inherited bootstrap CI (MF-D) plus the registered MF-C decision function.

Key conventions (read before consuming outputs):
- Layer index ``hs`` k in 1..n_layers = ``hidden_states[k]`` = output of decoder
  block k-1 (matches the u2 capture store's ``L{k:02d}`` files). The registered H3
  full-attention set is given as 0-indexed decoder BLOCKS [3,7,...,31] (config
  ``layer_types`` indices) => hs indices block+1 (MF-J assert before fitting).
- Relative depth of hs k on an L-layer model = k / L; Model A's H3 candidate set is
  the nearest-relative-depth match to Model B's 8 full-attention layers (ties break
  to the SMALLER hs index; realized match: A hs [3,7,10,14,17,21,24,28]).
- Pooled held-out R^2 = 1 - SS_res/SS_tot with SS_tot on the eval slice's OWN mean
  (the #779 ``issue779_percontext_recon._pooled_r2`` convention). WITHIN-SOURCE
  centering (PRIMARY per plan S6) replaces the slice mean with each source's own
  test-slice mean; singleton-source rows (n_src < 2) are excluded from the
  within-source statistic (their SST is 0 by construction) and counted.
- Bootstrap CIs are computed from the persisted matrix with the SST term held at
  the FIXED full-test-slice mean (per layer/model) — the matrix stores scalars, so
  resamples do not re-center; documented here + in the artifact meta.
- MF-B: NO ``n_train < d`` refusal anywhere — the G1 pilot is a deliberately
  under-determined regularization-limit read (val-selected lambda, the #1701
  escape); pilot fit JSONs persist n_train, d, selected lambda, lambda_grid_edge,
  numerical rank, effective dof.
- Linear map ONLY (project standing rule) — no MLP/nonlinear leg.

Phases (argparse driver; ``--import-check`` runs the argcheck completeness pass):
  fit       one model (--model-key A|B): per-layer pooled fits -> percell
            checkpoints -> selection -> kNN + LODO -> fits_summary.json +
            percontext_recon.json under --out-root/fits/model{A|B}/.
  decide    reads BOTH models' assembled artifacts -> A_pass/B_pass/NI
            selection-inherited bootstrap gates + H2 class contrast -> the
            registered MF-C truth table -> fits/decision.json.
  selfcheck synthetic end-to-end dry-run (d=8-16, n~48): streamed-core equivalence
            vs the #779 reference, fit -> baseline -> kNN -> per-source/LODO ->
            selection-inherited CI -> truth table incl. a REACHABLE Inconclusive.

Compute placement: the production fits run on a cpu-bigmem pod (`--device cpu`);
VM-side selfchecks carry the shared-VM thread-cap env prefix inline. Per-layer
percell checkpoints + a StageLedger (regime keyed on GENERATING PARAMETERS, #1336)
make every long loop resumable; one progress line per unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ISSUE = 2502
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Registered H3 full-attention layers of Qwen3.5-9B as 0-indexed decoder BLOCKS
# (live-verified config.json ``layer_types`` indices; plan S12 assumption 1).
REGISTERED_H3_BLOCKS = (3, 7, 11, 15, 19, 23, 27, 31)

NI_MARGIN = 0.10  # H3 one-sided non-inferiority margin on (B_best - A_best)
H2_MARGIN = 0.05  # ordinary-mean minus weird-mean descriptive threshold
G1_R2_FLOOR = 0.40  # G1 pilot pooled best-layer held-out R^2 floor (plan S7)
G1_GATE_RC = 7  # designed artifact-routed halt rc on a --g1-gate FAIL (gotchas)
BOOT_SEED = 2502
PILOT_SPLIT_SEED = 42
PILOT_TRAIN_FRAC = 0.80
LAMBDA_GRID_PARAMS = ("logspace", -3, 7, 21)  # == issue779 LAMBDAS_N50K generating params

MODEL_NAME = {"A": "Qwen/Qwen2.5-7B-Instruct", "B": "Qwen/Qwen3.5-9B"}
MODEL_N_LAYERS = {"A": 28, "B": 32}
MODEL_HIDDEN = {"A": 3584, "B": 4096}
DEFAULT_TENSORS_PREFIX = {
    "A": "issue2502_ctxmap_xgen/analysis_tensors/modelA",
    "B": "issue2502_ctxmap_xgen/analysis_tensors/modelB",
}

_CHUNK_ROWS_RE = re.compile(r"/(?P<key>(?:s\d+_)?chunk\d{4})/(?P=key)__rows\.json$")
_CHUNK_LAYER_RE = re.compile(r"/(?P<key>(?:s\d+_)?chunk\d{4})/(?P=key)__L(?P<k>\d{2})\.npz$")


def _gc():
    """Sibling u2 driver (codec + fetch + ledger + metadata helpers; light import)."""
    import issue2502_gen_capture as GC

    return GC


def _n779():
    """The reused #779 fit module (imports torch/numpy at ITS module top)."""
    import issue779_ffc_n50k_fits as N779

    return N779


def sha16(text: str) -> str:
    """First 16 hex chars of sha256 (machine-stable string keys only, #1336)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def round6(values) -> list[float]:
    """Round to 6 significant digits for compact JSON persistence."""
    return [float(f"{float(v):.6g}") for v in values]


def pooled_r2(pred, true) -> float:
    """Pooled R^2, SS_tot on TRUE's OWN mean — numpy copy of the exact
    ``issue779_percontext_recon._pooled_r2`` convention (equivalence pinned in
    --phase selfcheck via the reference fit path). NaN on degenerate SS_tot."""
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def r2_from_sums(err_sum: float, sst_sum: float) -> float:
    """R^2 from precomputed per-context sums (same convention as pooled_r2)."""
    return float("nan") if sst_sum < 1e-12 else 1.0 - err_sum / sst_sum


# --------------------------------------------------------------------------
# Candidate layer sets (MF-D relative-depth matching; MF-J assert)
# --------------------------------------------------------------------------


def h3_hs_set(model_key: str) -> list[int]:
    """H3 candidate hs indices per model (equal-breadth 8-vs-8, MF-D).

    B: the registered full-attention blocks +1. A: nearest-relative-depth match
    (rel depth of hs k on an L-layer model = k/L; ties -> smaller hs)."""
    b_hs = [b + 1 for b in REGISTERED_H3_BLOCKS]
    if model_key == "B":
        return b_hs
    n_a, n_b = MODEL_N_LAYERS["A"], MODEL_N_LAYERS["B"]
    out = []
    for kb in b_hs:
        # exact tie handling: |ka/n_a - kb/n_b| ∝ |ka*n_b - kb*n_a| (integers)
        best = min(range(1, n_a + 1), key=lambda ka: (abs(ka * n_b - kb * n_a), ka))
        out.append(best)
    if len(set(out)) != len(out):
        raise RuntimeError(f"relative-depth matching produced duplicate A layers: {out}")
    return out


def gate_hs_set(model_key: str, captured: list[int]) -> list[int]:
    """Gate (A_pass/B_pass selection) candidate set: A = all captured layers;
    B = the registered full-attention set only (linear-attn layers are a
    robustness read, never a gate input — plan S4/S5)."""
    if model_key == "A":
        return sorted(captured)
    return [k for k in h3_hs_set("B")]


def assert_mfj(model_key: str, captured: list[int]) -> dict:
    """MF-J: the registered H3 set must be a subset of the captured layers."""
    need = h3_hs_set(model_key)
    missing = sorted(set(need) - set(captured))
    if missing:
        raise RuntimeError(
            f"MF-J violation (model {model_key}): registered H3 hs layers {need} "
            f"not all captured (missing {missing}; captured {sorted(captured)}) — "
            "terminate + re-register H3 rather than fitting a partial set"
        )
    return {
        "registered_h3_blocks_0idx": list(REGISTERED_H3_BLOCKS),
        "h3_hs": need,
        "captured_hs": sorted(captured),
        "subset_ok": True,
    }


# --------------------------------------------------------------------------
# Stores (HF-backed production store + in-memory selfcheck store)
# --------------------------------------------------------------------------


class HfChunkStore:
    """Reader for a u2 gen_capture tensor store (per-chunk per-layer bf16 npz).

    Streams ONE layer at a time (fetch chunk npz -> decode -> concat -> unlink),
    bounding local disk to ~one chunk file and RAM to ~one layer (plan S9)."""

    def __init__(self, prefix: str, work: Path, hidden: int):
        self.prefix = prefix.rstrip("/")
        self.work = work
        self.hidden = hidden
        self._files: list[str] | None = None
        self._keys: list[str] | None = None

    def _listing(self) -> list[str]:
        if self._files is None:
            from huggingface_hub import HfApi

            from explore_persona_space.orchestrate import hub

            self._files = hub.retry_transient(
                lambda: hub.list_hf_files_under_path(
                    HfApi(), HF_DATA_REPO, self.prefix, repo_type="dataset"
                ),
                what=f"list({self.prefix})",
            )
        return self._files

    def chunk_keys(self) -> list[str]:
        if self._keys is None:
            keys = sorted(
                {m.group("key") for f in self._listing() if (m := _CHUNK_ROWS_RE.search(f))}
            )
            if not keys:
                raise RuntimeError(f"no capture chunks (rows.json) found under {self.prefix}")
            self._keys = keys
        return self._keys

    def captured_hs(self) -> list[int]:
        """hs indices present in EVERY chunk (intersection; fail loud on empty)."""
        per_key: dict[str, set[int]] = {}
        for f in self._listing():
            m = _CHUNK_LAYER_RE.search(f)
            if m:
                per_key.setdefault(m.group("key"), set()).add(int(m.group("k")))
        keys = self.chunk_keys()
        missing = [k for k in keys if k not in per_key]
        if missing:
            raise RuntimeError(f"chunks with rows.json but no layer npz: {missing[:5]}")
        common = set.intersection(*(per_key[k] for k in keys))
        if not common:
            raise RuntimeError(f"no layer index captured in every chunk under {self.prefix}")
        union = set.union(*(per_key[k] for k in keys))
        if union != common:
            print(
                f"[store] WARNING: non-uniform layer sets across chunks under {self.prefix}: "
                f"using intersection {sorted(common)} (union {sorted(union)})",
                flush=True,
            )
        return sorted(common)

    def load_rows(self) -> list[dict]:
        """Concatenated per-chunk rows metadata, in chunk-key order (= npz row order)."""
        GC = _gc()
        rows: list[dict] = []
        for key in self.chunk_keys():
            local = GC.fetch_repo_file(
                f"{self.prefix}/{key}/{key}__rows.json", self.work / "rows_dl", what=f"rows({key})"
            )
            doc = json.loads(local.read_text(encoding="utf-8"))
            chunk_rows = doc["rows"]
            for i, r in enumerate(chunk_rows):
                if r["row"] != i:
                    raise RuntimeError(f"rows.json {key} row-order mismatch at {i} != {r['row']}")
            rows.extend({**r, "chunk_key": key} for r in chunk_rows)
            local.unlink()
        if not rows:
            raise RuntimeError(f"empty row table under {self.prefix} — fail loud")
        return rows

    def load_layer(self, k: int):
        """(X=cx_last, Y=v_x) fp32 numpy for hs layer k, rows in store order."""
        import numpy as np

        GC = _gc()
        import torch

        xs, ys = [], []
        for key in self.chunk_keys():
            local = GC.fetch_repo_file(
                f"{self.prefix}/{key}/{key}__L{k:02d}.npz",
                self.work / "tensors_dl",
                what=f"layer({key},L{k:02d})",
            )
            with np.load(local) as z:
                cx = GC.decode_bf16(z["cx_last"], torch).float().numpy()
                vx = GC.decode_bf16(z["vx"], torch).float().numpy()
            local.unlink()
            if cx.shape[1] != self.hidden or vx.shape[1] != self.hidden:
                raise RuntimeError(
                    f"hidden-dim mismatch in {key} L{k:02d}: {cx.shape} / {vx.shape} "
                    f"vs expected H={self.hidden}"
                )
            xs.append(cx)
            ys.append(vx)
        return np.concatenate(xs), np.concatenate(ys)


class MemStore:
    """In-memory store with the HfChunkStore duck-type (selfcheck seam)."""

    def __init__(self, rows: list[dict], layers: dict[int, tuple]):
        self._rows = rows
        self._layers = layers

    def chunk_keys(self) -> list[str]:
        return ["chunk0000"]

    def captured_hs(self) -> list[int]:
        return sorted(self._layers)

    def load_rows(self) -> list[dict]:
        return list(self._rows)

    def load_layer(self, k: int):
        return self._layers[k]


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def resolve_splits(rows: list[dict], *, pilot: bool):
    """(tr, val, te) index arrays. Production: the corpus-assigned P0 split
    (70/15/15). Pilot (G1, MF-B): seeded 80/20 with val == heldout == test —
    the pilot selects lambda AND reads R^2 on the same 20% (a deliberately
    optimistic rig-sanity floor, recorded as ``pilot_val_is_test``)."""
    import numpy as np

    if pilot:
        order = np.argsort(np.array([r["context_id"] for r in rows]))
        perm = np.random.default_rng(PILOT_SPLIT_SEED).permutation(order)
        n_tr = int(round(PILOT_TRAIN_FRAC * len(rows)))
        if n_tr < 1 or len(rows) - n_tr < 2:
            raise RuntimeError(f"pilot split degenerate: n={len(rows)}")
        tr, held = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        return tr, held, held
    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        s = r.get("split")
        if s not in buckets:
            raise RuntimeError(f"row {i} ({r.get('context_id')}): unknown split {s!r}")
        buckets[s].append(i)
    for name in ("train", "val", "test"):
        if len(buckets[name]) < 2:
            raise RuntimeError(f"split {name!r} has {len(buckets[name])} rows — degenerate")
    return (
        np.asarray(buckets["train"]),
        np.asarray(buckets["val"]),
        np.asarray(buckets["test"]),
    )


# --------------------------------------------------------------------------
# Streamed ridge core (memory-lean twin of the #779 fit; equivalence-pinned)
# --------------------------------------------------------------------------


def ridge_fit_streamed(X, Y, tr, val, eval_idx_sets, lambdas, dev):
    """Val-lambda-selected primal ridge — numerically IDENTICAL math to
    ``issue779_ffc_n50k_fits.fit_ridge_primal`` (standardize X on train stats,
    center Y on train mean, ONE eigh of the (H,H) X^T X batched over lambda,
    strict-> first-best-wins val selection), with ONE difference: per-lambda val
    predictions are scored streaming and only the SELECTED lambda's eval-set
    predictions are materialized (the reference materializes all-lambda x
    all-eval-set predictions — ~31 GB at #2502 eval sizes). No permissiveness
    change; equivalence vs the reference is asserted in --phase selfcheck.

    Returns (preds per eval set at selected lambda, meta incl. MF-B diagnostics:
    n_train, d, selected_lambda, lambda_grid_edge, numerical_rank, effective_dof).
    """
    import numpy as np
    import torch

    Xtr = torch.as_tensor(np.asarray(X[tr]), dtype=torch.float64, device=dev)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    del Xtr
    Yt = torch.as_tensor(np.asarray(Y[tr]), dtype=torch.float64, device=dev)
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    del Yt
    A = Xtr_n.T @ Xtr_n
    try:
        s, U = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:
        if str(dev) == "cpu":
            raise  # genuinely pathological input — fail loud (gotchas: never jitter)
        s, U = torch.linalg.eigh(A.cpu())  # cusolver non-convergence -> CPU LAPACK
        s, U = s.to(A.device), U.to(A.device)
        print(f"[fit] eigh cuda->cpu fallback engaged (H={A.shape[0]})", flush=True)
    s = torch.clamp(s, min=0.0)
    XtY = Xtr_n.T @ Yc
    n_train = int(Xtr_n.shape[0])
    d = int(Xtr_n.shape[1])
    del Xtr_n
    UtXtY = U.T @ XtY
    del XtY
    Val = torch.as_tensor(np.asarray(X[val]), dtype=torch.float64, device=dev)
    Val_n = (Val - xmu) / xsd
    del Val
    Yval = np.asarray(Y[val])
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        W = U @ (UtXtY / (s + float(lam))[:, None])
        pv = ((Val_n @ W) + ymu).cpu().numpy()
        del W
        vr2 = pooled_r2(pv, Yval)
        if np.isfinite(vr2) and vr2 > best_vr2:
            best_vr2, best_lam = vr2, float(lam)
    edge = None
    if np.isclose(best_lam, float(lambdas[0])):
        edge = "low"
    elif np.isclose(best_lam, float(lambdas[-1])):
        edge = "high"
    W = U @ (UtXtY / (s + best_lam)[:, None])
    preds = []
    for idx in eval_idx_sets:
        E = torch.as_tensor(np.asarray(X[idx]), dtype=torch.float64, device=dev)
        preds.append((((E - xmu) / xsd) @ W + ymu).cpu().numpy())
        del E
    s_np = s.cpu().numpy()
    smax = float(s_np.max()) if s_np.size else 0.0
    eps = float(np.finfo(np.float64).eps)
    rank = int((s_np > smax * max(n_train, d) * eps).sum()) if smax > 0 else 0
    meta = {
        "n_train": n_train,
        "d": d,
        "selection": "val-lambda (primal, streamed)",
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
        "numerical_rank": rank,
        "effective_dof": float((s_np / (s_np + best_lam)).sum()),
    }
    return preds, meta


# --------------------------------------------------------------------------
# Per-layer fit unit (percell checkpoint)
# --------------------------------------------------------------------------


def _source_stats(rows_te: list[dict]):
    """Per-source test-slice index lists + the >=2-row source mask."""
    by_src: dict[str, list[int]] = {}
    for j, r in enumerate(rows_te):
        by_src.setdefault(str(r["source_tag"]), []).append(j)
    multi = {s for s, idx in by_src.items() if len(idx) >= 2}
    return by_src, multi


def fit_layer_unit(X, Y, tr, val, te, rows_te, k, lambdas, dev):
    """One candidate layer: pooled streamed fit + identity baseline + the
    per-context err/sst scalars (the recon-matrix slice for this layer)."""
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    (pred_map,), meta = ridge_fit_streamed(X, Y, tr, val, [te], lambdas, dev)
    pred_id = identity_bias_predict(X[tr], Y[tr], X[te])
    Yte = np.asarray(Y[te], dtype=np.float64)
    err_map = ((Yte - pred_map) ** 2).sum(1)
    err_id = ((Yte - np.asarray(pred_id, dtype=np.float64)) ** 2).sum(1)
    mu = Yte.mean(0)
    sst_pooled = ((Yte - mu) ** 2).sum(1)
    by_src, multi = _source_stats(rows_te)
    sst_ws = np.zeros_like(sst_pooled)
    for src, idx in by_src.items():
        idx = np.asarray(idx)
        if len(idx) >= 2:
            mu_s = Yte[idx].mean(0)
            sst_ws[idx] = ((Yte[idx] - mu_s) ** 2).sum(1)
    ws_rows = np.asarray(sorted(j for s in multi for j in by_src[s]), dtype=int)
    per_source = {}
    for src, idx in sorted(by_src.items()):
        idx = np.asarray(idx)
        per_source[src] = {
            "n": int(len(idx)),
            "regime_class": rows_te[int(idx[0])].get("regime_class"),
            "lodo_group": rows_te[int(idx[0])].get("lodo_group"),
            "r2_map_within_source": r2_from_sums(
                float(err_map[idx].sum()), float(sst_ws[idx].sum())
            ),
            "r2_id_within_source": r2_from_sums(float(err_id[idx].sum()), float(sst_ws[idx].sum())),
            "r2_map_pooled_sst": r2_from_sums(
                float(err_map[idx].sum()), float(sst_pooled[idx].sum())
            ),
            "r2_id_pooled_sst": r2_from_sums(
                float(err_id[idx].sum()), float(sst_pooled[idx].sum())
            ),
        }
    unit = {
        "hs": k,
        "block_0idx": k - 1,
        "fit_meta": meta,
        "r2_test_map_pooled": r2_from_sums(float(err_map.sum()), float(sst_pooled.sum())),
        "r2_test_id_pooled": r2_from_sums(float(err_id.sum()), float(sst_pooled.sum())),
        "r2_test_map_within_source": r2_from_sums(
            float(err_map[ws_rows].sum()), float(sst_ws[ws_rows].sum())
        ),
        "r2_test_id_within_source": r2_from_sums(
            float(err_id[ws_rows].sum()), float(sst_ws[ws_rows].sum())
        ),
        "n_singleton_source_rows_excluded_ws": int(len(rows_te) - len(ws_rows)),
        "per_source": per_source,
        "arrays": {
            "err_map": round6(err_map),
            "err_identity": round6(err_id),
            "sst_pooled": round6(sst_pooled),
            "sst_within_source": round6(sst_ws),
        },
    }
    return unit, pred_map


# --------------------------------------------------------------------------
# kNN + LODO (post-selection units)
# --------------------------------------------------------------------------


def knn_unit(pred_te, Y_te, rows_te, *, max_n: int, seed: int):
    """Diagnostic kNN retrieval at the selected layer: pooled (capped seeded
    subsample; pool = the subsample's own true v_x) + per-source (each source's
    full test slice as its own pool; sources with n < 10 skipped)."""
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    out: dict = {"pooled": {}, "per_source": {}, "max_n": max_n, "note": "diagnostic-only"}
    n = len(rows_te)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, size=min(max_n, n), replace=False))
    out["pooled_n_pool"] = int(len(sub))
    for metric in ("euclidean", "cosine"):
        out["pooled"][metric] = knn_retrieval(pred_te[sub], Y_te[sub], metric=metric)
    by_src, _ = _source_stats(rows_te)
    for src, idx in sorted(by_src.items()):
        if len(idx) < 10:
            out["per_source"][src] = {"skipped": f"n={len(idx)} < 10"}
            continue
        idx = np.asarray(idx)
        if len(idx) > max_n:
            idx = np.sort(rng.choice(idx, size=max_n, replace=False))
        out["per_source"][src] = {
            m: knn_retrieval(pred_te[idx], Y_te[idx], metric=m) for m in ("euclidean", "cosine")
        }
    return out


def lodo_unit(X, Y, rows, tr, val, group: str, k_sel: int, lambdas, dev):
    """One LODO fold at the selected layer: fit on train-partition rows OUTSIDE
    the group (lambda selected on val-partition rows outside the group),
    evaluate on ALL rows of the left-out group (every one unseen by the fold)."""
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    grp = np.asarray([i for i, r in enumerate(rows) if str(r.get("lodo_group")) == group])
    tr_f = np.asarray([i for i in tr if str(rows[i].get("lodo_group")) != group])
    val_f = np.asarray([i for i in val if str(rows[i].get("lodo_group")) != group])
    if len(tr_f) < 2 or len(val_f) < 2 or len(grp) < 2:
        return {
            "group": group,
            "skipped": f"degenerate fold (n_train={len(tr_f)}, n_val={len(val_f)}, "
            f"n_eval={len(grp)})",
        }
    (pred,), meta = ridge_fit_streamed(X, Y, tr_f, val_f, [grp], lambdas, dev)
    pred_id = identity_bias_predict(X[tr_f], Y[tr_f], X[grp])
    return {
        "group": group,
        "hs": k_sel,
        "n_train": int(len(tr_f)),
        "n_eval": int(len(grp)),
        "regime_class": rows[int(grp[0])].get("regime_class"),
        "selected_lambda": meta["selected_lambda"],
        "lambda_grid_edge": meta["lambda_grid_edge"],
        "r2_map": pooled_r2(pred, Y[grp]),
        "r2_id": pooled_r2(pred_id, Y[grp]),
    }


# --------------------------------------------------------------------------
# fit phase driver
# --------------------------------------------------------------------------


def model_dirs(args, model_key: str):
    out_root = Path(args.out_root)
    fits_dir = out_root / "fits" / f"model{model_key}"
    percell = (
        Path(args.work_dir) / f"percell_model{model_key}" / ("pilot" if args.pilot else "full")
    )
    return fits_dir, percell


def fit_regime(args, model_key: str, n_rows: int, keys_sha: str) -> dict:
    """StageLedger regime — GENERATING PARAMETERS only (machine-stable, #1336)."""
    return {
        "phase": "fit",
        "issue": ISSUE,
        "model_key": model_key,
        "tensors_prefix": args.tensors_prefix or DEFAULT_TENSORS_PREFIX[model_key],
        "pilot": bool(args.pilot),
        "split": "pilot-80/20-seed42" if args.pilot else "corpus-p0-70/15/15",
        "lambda_grid": list(LAMBDA_GRID_PARAMS),
        "sst": "pooled+within-source (fixed full-test means)",
        "device": args.device,
        "n_rows": n_rows,
        "chunk_keys_sha16": keys_sha,
    }


def run_fit(args, store=None, model_key: str | None = None) -> dict:
    """The fit phase for one model. ``store``/``model_key`` injectable (selfcheck)."""
    import numpy as np

    GC = _gc()
    N779 = _n779()
    model_key = model_key or args.model_key
    if model_key not in MODEL_NAME:
        raise SystemExit(f"--model-key required (A|B), got {model_key!r}")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if store is None:
        prefix = args.tensors_prefix or DEFAULT_TENSORS_PREFIX[model_key]
        store = HfChunkStore(prefix, work, MODEL_HIDDEN[model_key])
    fits_dir, percell = model_dirs(args, model_key)
    percell.mkdir(parents=True, exist_ok=True)

    rows = store.load_rows()
    captured = store.captured_hs()
    mfj = assert_mfj(model_key, captured)
    all_set = sorted(captured)
    gate_set = gate_hs_set(model_key, captured)
    h3_set = h3_hs_set(model_key)
    tr, val, te = resolve_splits(rows, pilot=args.pilot)
    rows_te = [rows[int(i)] for i in te]
    keys_sha = sha16(",".join(store.chunk_keys()))
    ledger = GC.StageLedger(
        percell / "ledger.json", fit_regime(args, model_key, len(rows), keys_sha)
    )
    lambdas = N779.LAMBDAS_N50K
    layer_subset = sorted(int(x) for x in args.layers.split(",")) if args.layers else list(all_set)
    unknown = set(layer_subset) - set(all_set)
    if unknown:
        raise RuntimeError(f"--layers {sorted(unknown)} not in captured set {all_set}")
    pending = [k for k in layer_subset if not ledger.is_done(f"L{k:02d}")]
    print(
        f"[fit] model {model_key}: n={len(rows)} (tr={len(tr)}/val={len(val)}/te={len(te)}), "
        f"layers {len(all_set)} captured, {len(pending)} pending "
        f"(gate={gate_set}, h3={h3_set}, pilot={args.pilot})",
        flush=True,
    )
    if pending and isinstance(store, HfChunkStore):
        from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

        assert_out_root_headroom(work, need_gb=4.0, phase=f"fits-model{model_key}")

    t0 = time.time()
    for j, k in enumerate(layer_subset):
        cell = f"L{k:02d}"
        if ledger.is_done(cell):
            continue
        X, Y = store.load_layer(k)
        if X.shape[0] != len(rows):
            raise RuntimeError(f"layer {cell}: {X.shape[0]} rows vs table {len(rows)}")
        unit, _ = fit_layer_unit(X, Y, tr, val, te, rows_te, k, lambdas, args.device)
        del X, Y
        GC.atomic_write_json(percell / f"{cell}.json", unit)
        ledger.mark_done(cell)
        GC.progress(f"fit-{model_key}", j + 1, len(layer_subset), cell, t0)

    done_all = [k for k in all_set if ledger.is_done(f"L{k:02d}")]
    if set(done_all) != set(all_set):
        if args.layers:
            print(
                f"[fit] model {model_key}: layer shard complete "
                f"({len(done_all)}/{len(all_set)} layers done); rerun without --layers "
                "to finish selection + LODO + assembly",
                flush=True,
            )
            return {"partial": True, "layers_done": done_all}
        raise RuntimeError(f"missing per-layer cells after loop: {set(all_set) - set(done_all)}")

    units = {k: json.loads((percell / f"L{k:02d}.json").read_text()) for k in all_set}
    val_r2 = {k: units[k]["fit_meta"]["val_r2_at_selected"] for k in all_set}

    def _select(cands: list[int]) -> int:
        finite = [k for k in cands if math.isfinite(val_r2[k])]
        if not finite:
            raise RuntimeError(f"no finite validation R^2 in candidate set {cands}")
        return min(finite, key=lambda k: (-val_r2[k], k))

    sel_gate = _select(gate_set)
    sel_h3 = _select(h3_set)

    # post-selection unit: reload the gate layer once -> pooled preds -> kNN.
    if not ledger.is_done("post"):
        X, Y = store.load_layer(sel_gate)
        (pred_te,), _ = ridge_fit_streamed(X, Y, tr, val, [te], lambdas, args.device)
        knn = knn_unit(
            pred_te,
            np.asarray(Y[te], dtype=np.float64),
            rows_te,
            max_n=args.knn_max_n,
            seed=BOOT_SEED,
        )
        GC.atomic_write_json(percell / "post.json", {"gate_hs": sel_gate, "knn": knn})

        # LODO folds at the selected gate layer (X/Y already resident).
        if not args.skip_lodo:
            groups = sorted({str(r.get("lodo_group")) for r in rows})
            if args.lodo_groups:
                want = set(args.lodo_groups.split(","))
                unknown_g = want - set(groups)
                if unknown_g:
                    raise RuntimeError(f"--lodo-groups unknown: {sorted(unknown_g)}")
                groups = sorted(want)
            t1 = time.time()
            for gi, g in enumerate(groups):
                cell = f"lodo_{g}"
                if ledger.is_done(cell):
                    continue
                fold = lodo_unit(X, Y, rows, tr, val, g, sel_gate, lambdas, args.device)
                GC.atomic_write_json(percell / f"{cell}.json", fold)
                ledger.mark_done(cell)
                GC.progress(f"lodo-{model_key}", gi + 1, len(groups), g, t1)
        del X, Y
        ledger.mark_done("post")

    return assemble_model(
        args,
        model_key,
        percell,
        fits_dir,
        rows_te,
        units,
        mfj,
        {
            "all": all_set,
            "gate": gate_set,
            "h3": h3_set,
            "selected_gate_hs": sel_gate,
            "selected_h3_hs": sel_h3,
        },
        tr=tr,
        val=val,
        te=te,
    )


# --------------------------------------------------------------------------
# Assembly (fits_summary.json + percontext_recon.json) + G1 pilot gate
# --------------------------------------------------------------------------


def assemble_model(args, model_key, percell, fits_dir, rows_te, units, mfj, sel, *, tr, val, te):
    """Compose the two committed deliverables from percell checkpoints."""
    GC = _gc()
    n_layers = MODEL_N_LAYERS[model_key]
    all_set = sel["all"]
    contexts = [
        {
            "context_id": r["context_id"],
            "source_tag": r.get("source_tag"),
            "regime_class": r.get("regime_class"),
            "lodo_group": r.get("lodo_group"),
        }
        for r in rows_te
    ]
    recon = {
        "meta": GC.run_metadata(
            {
                "artifact": "percontext_recon",
                "model_key": model_key,
                "model": MODEL_NAME[model_key],
                "pilot": bool(args.pilot),
                "sst_convention": (
                    "sst_pooled: fixed full-test-slice mean; sst_within_source: fixed "
                    "per-source test-slice mean (0.0 for singleton-source rows — "
                    "excluded from within-source statistics); arms: err_map (pooled "
                    "val-lambda ridge), err_identity (identity+learned-bias); all "
                    "scalars are squared L2 residual norms per context"
                ),
                "layer_indexing": "hs k = hidden_states[k] = output of decoder block k-1",
            }
        ),
        "candidate_sets": {k: v for k, v in sel.items() if k in ("all", "gate", "h3")},
        "selected": {
            "gate_hs": sel["selected_gate_hs"],
            "h3_hs": sel["selected_h3_hs"],
        },
        "n_test": len(rows_te),
        "contexts": contexts,
        "layers": {f"L{k:02d}": units[k]["arrays"] for k in all_set},
    }
    fits_dir.mkdir(parents=True, exist_ok=True)
    GC.atomic_write_json(fits_dir / "percontext_recon.json", recon)

    post = (
        json.loads((percell / "post.json").read_text()) if (percell / "post.json").exists() else {}
    )
    lodo = {}
    for p in sorted(percell.glob("lodo_*.json")):
        d = json.loads(p.read_text())
        lodo[d["group"]] = d
    layer_table = []
    for k in all_set:
        u = units[k]
        layer_table.append(
            {
                "hs": k,
                "block_0idx": k - 1,
                "rel_depth": k / n_layers,
                **{
                    key: u["fit_meta"][key]
                    for key in (
                        "n_train",
                        "d",
                        "selected_lambda",
                        "val_r2_at_selected",
                        "lambda_grid_edge",
                        "numerical_rank",
                        "effective_dof",
                    )
                },
                "r2_test_map_pooled": u["r2_test_map_pooled"],
                "r2_test_id_pooled": u["r2_test_id_pooled"],
                "r2_test_map_within_source": u["r2_test_map_within_source"],
                "r2_test_id_within_source": u["r2_test_id_within_source"],
            }
        )
    sel_u = units[sel["selected_gate_hs"]]
    summary = {
        "meta": GC.run_metadata(
            {
                "artifact": "fits_summary",
                "model_key": model_key,
                "model": MODEL_NAME[model_key],
                "pilot": bool(args.pilot),
                "pilot_val_is_test": bool(args.pilot),
                "splits": {"n_train": int(len(tr)), "n_val": int(len(val)), "n_test": int(len(te))},
                "sst_centering": "within-source PRIMARY; pooled alongside (plan S6)",
            }
        ),
        "mfj_assert": mfj,
        "candidate_sets": {k: v for k, v in sel.items() if k in ("all", "gate", "h3")},
        "selected": {"gate_hs": sel["selected_gate_hs"], "h3_hs": sel["selected_h3_hs"]},
        "layers": layer_table,
        "per_source_at_gate_layer": sel_u["per_source"],
        "r2_at_gate_layer": {
            "map_pooled": sel_u["r2_test_map_pooled"],
            "map_within_source": sel_u["r2_test_map_within_source"],
            "identity_pooled": sel_u["r2_test_id_pooled"],
            "identity_within_source": sel_u["r2_test_id_within_source"],
        },
        "r2_at_h3_layer": {
            "map_pooled": units[sel["selected_h3_hs"]]["r2_test_map_pooled"],
        },
        "knn": post.get("knn"),
        "lodo": lodo,
    }
    if args.pilot and model_key == "A":
        summary["g1"] = g1_gate_block(units, sel, rows_te, args)
    GC.atomic_write_json(fits_dir / "fits_summary.json", summary)
    print(
        f"[assemble] model {model_key}: gate hs={sel['selected_gate_hs']} "
        f"(R2 pooled={sel_u['r2_test_map_pooled']:.4f}, "
        f"id={sel_u['r2_test_id_pooled']:.4f}); wrote {fits_dir}",
        flush=True,
    )
    return summary


def g1_gate_block(units, sel, rows_te, args) -> dict:
    """G1 (MF-B): pooled best-layer held-out R^2 floor + the beats-baseline
    paired bootstrap (selection-inherited over the gate set, frozen alongside).
    The pilot's A_pass slice is the POOLED held-out set (the 1k pilot's
    ordinary-only subset is small); recorded as ``a_pass_slice``."""
    import numpy as np

    mats = _recon_arrays_from_units(units, sel["all"])
    best_hs = max((k for k in sel["gate"]), key=lambda k: (units[k]["r2_test_map_pooled"], -k))
    r2_best = units[best_hs]["r2_test_map_pooled"]
    boot = paired_delta_bootstrap(
        mats,
        cand=sel["gate"],
        frozen_hs=best_hs,
        member_mask=np.ones(len(rows_te), dtype=bool),
        draws=args.boot_draws,
        seed=BOOT_SEED,
    )
    a_pass = bool(boot["inherited_ci"][0] > 0.0)
    verdict = "PROCEED" if (r2_best >= G1_R2_FLOOR and a_pass) else "FAIL"
    m = units[best_hs]["fit_meta"]
    return {
        "r2_floor": G1_R2_FLOOR,
        "best_layer_hs": best_hs,
        "r2_best_layer_heldout_pooled": r2_best,
        "a_pass_pilot": a_pass,
        "a_pass_slice": "pooled-heldout",
        "paired_delta_bootstrap": boot,
        "fit_diagnostics_at_best_layer": {
            key: m[key]
            for key in (
                "n_train",
                "d",
                "selected_lambda",
                "lambda_grid_edge",
                "numerical_rank",
                "effective_dof",
            )
        },
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# Bootstrap machinery (selection-inherited + frozen; subset-sum GEMM batched)
# --------------------------------------------------------------------------


def _recon_arrays_from_units(units: dict, layer_set: list[int]) -> dict:
    """(n, L) numpy views of err/sst per arm, column order == sorted layer_set."""
    import numpy as np

    ks = sorted(layer_set)
    return {
        "hs": ks,
        "err_map": np.stack(
            [np.asarray(units[k]["arrays"]["err_map"], dtype=np.float64) for k in ks], axis=1
        ),
        "err_id": np.stack(
            [np.asarray(units[k]["arrays"]["err_identity"], dtype=np.float64) for k in ks], axis=1
        ),
        "sst": np.stack(
            [np.asarray(units[k]["arrays"]["sst_pooled"], dtype=np.float64) for k in ks], axis=1
        ),
    }


def recon_arrays_from_file(recon: dict, layer_set: list[int]) -> dict:
    """Same as _recon_arrays_from_units but from a loaded percontext_recon dict."""
    units = {k: {"arrays": recon["layers"][f"L{k:02d}"]} for k in layer_set}
    return _recon_arrays_from_units(units, layer_set)


def bootstrap_counts(n: int, draws: int, seed: int):
    """(draws, n) float32 bootstrap multiplicity matrix (subset-sum GEMM form —
    per-draw statistics become ONE counts @ matrix product; vectorize rule)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(np.float32), rng


def paired_delta_bootstrap(mats, *, cand, frozen_hs, member_mask, draws, seed):
    """Paired map-vs-identity contrast (A_pass/B_pass predicate machinery).

    Statistic: mean_i Delta_i over the member slice (Delta_i = err_id - err_map at
    the layer). SELECTION-INHERITED: within each resample the layer is re-selected
    as argmax of the resample's own pooled map R^2 over ``cand`` (MF-D); the
    frozen-at-``frozen_hs`` CI is reported ALONGSIDE, labeled, never alone."""
    import numpy as np

    hs = mats["hs"]
    cand_cols = [hs.index(k) for k in sorted(cand)]
    n = mats["err_map"].shape[0]
    counts, _ = bootstrap_counts(n, draws, seed)
    err_sums = counts @ mats["err_map"][:, cand_cols]  # (B, |cand|)
    sst_sums = counts @ mats["sst"][:, cand_cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_b = 1.0 - err_sums / sst_sums
    sel_cols = np.nanargmax(r2_b, axis=1)
    delta = mats["err_id"] - mats["err_map"]  # (n, L)
    m = member_mask.astype(np.float32)
    member_counts = counts * m[None, :]
    denom = member_counts.sum(1)
    num_by_col = member_counts @ delta[:, cand_cols]  # (B, |cand|)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_by_col = num_by_col / denom[:, None]
    inherited = delta_by_col[np.arange(draws), sel_cols]
    frozen_col = cand_cols.index(hs.index(frozen_hs))
    frozen = delta_by_col[:, frozen_col]
    idx = np.asarray(member_mask)
    point = float(
        (
            mats["err_id"][idx, hs.index(frozen_hs)] - mats["err_map"][idx, hs.index(frozen_hs)]
        ).mean()
    )
    pct = [2.5, 97.5]
    return {
        "point_delta_mean_at_frozen": point,
        "frozen_hs": frozen_hs,
        "candidate_hs": sorted(cand),
        "n_members": int(idx.sum()),
        "draws": draws,
        "seed": seed,
        "inherited_ci": [float(x) for x in np.nanpercentile(inherited, pct)],
        "frozen_ci_labeled_frozen_at_selected": [float(x) for x in np.nanpercentile(frozen, pct)],
    }


def ni_bootstrap(mats_a, ids_a, mats_b, ids_b, *, cand_a, cand_b, frozen_a, frozen_b, draws, seed):
    """Selection-inherited CI on (B_best - A_best) over the SHARED test contexts.

    Both models' per-context rows are aligned on the shared context_id set (the
    corpus is shared; per-model drops can differ) and each resample draws shared
    contexts, re-selecting each model's layer within its H3 candidate set."""
    import numpy as np

    shared = sorted(set(ids_a) & set(ids_b))
    if len(shared) < 10:
        raise RuntimeError(f"only {len(shared)} shared test contexts across models")
    pos_a = {c: i for i, c in enumerate(ids_a)}
    pos_b = {c: i for i, c in enumerate(ids_b)}
    ia = np.asarray([pos_a[c] for c in shared])
    ib = np.asarray([pos_b[c] for c in shared])
    counts, _ = bootstrap_counts(len(shared), draws, seed)

    def _best(mats, rows_idx, cand, frozen_hs):
        cols = [mats["hs"].index(k) for k in sorted(cand)]
        e = mats["err_map"][rows_idx][:, cols]
        s = mats["sst"][rows_idx][:, cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_b = 1.0 - (counts @ e) / (counts @ s)
        inherited = np.nanmax(r2_b, axis=1)
        frozen = r2_b[:, cols.index(mats["hs"].index(frozen_hs))]
        return inherited, frozen

    inh_a, frz_a = _best(mats_a, ia, cand_a, frozen_a)
    inh_b, frz_b = _best(mats_b, ib, cand_b, frozen_b)
    pct = [2.5, 97.5]
    return {
        "n_shared_contexts": len(shared),
        "draws": draws,
        "seed": seed,
        "inherited_ci_diff": [float(x) for x in np.nanpercentile(inh_b - inh_a, pct)],
        "frozen_ci_diff_labeled_frozen_at_selected": [
            float(x) for x in np.nanpercentile(frz_b - frz_a, pct)
        ],
    }


def h2_contrast(recon: dict, gate_hs: int, *, draws: int, seed: int) -> dict:
    """H2: equal-source-weighted regime-class mean per-source R^2 (within-source
    centering PRIMARY), ordinary minus weird, source-level bootstrap CI."""
    import numpy as np

    hs_key = f"L{gate_hs:02d}"
    arr = recon["layers"][hs_key]
    err = np.asarray(arr["err_map"], dtype=np.float64)
    sst = np.asarray(arr["sst_within_source"], dtype=np.float64)
    src = np.asarray([c["source_tag"] for c in recon["contexts"]])
    cls = {c["source_tag"]: c["regime_class"] for c in recon["contexts"]}
    per_src = {}
    for s in sorted(set(src)):
        m = src == s
        if m.sum() < 2:
            continue
        per_src[s] = r2_from_sums(float(err[m].sum()), float(sst[m].sum()))
    classes: dict[str, list[float]] = {}
    for s, r2 in per_src.items():
        if math.isfinite(r2):
            classes.setdefault(str(cls[s]), []).append(r2)
    means = {c: float(np.mean(v)) for c, v in classes.items()}
    out = {
        "gate_hs": gate_hs,
        "weighting": "equal-source (each source_tag counts once)",
        "centering": "within-source",
        "per_source_r2": per_src,
        "class_means": means,
        "n_sources_per_class": {c: len(v) for c, v in classes.items()},
    }
    if "ordinary" in classes and "weird" in classes:
        rng = np.random.default_rng(seed)
        o, w = np.asarray(classes["ordinary"]), np.asarray(classes["weird"])
        diffs = [
            float(rng.choice(o, len(o)).mean() - rng.choice(w, len(w)).mean()) for _ in range(draws)
        ]
        point = means["ordinary"] - means["weird"]
        out["ordinary_minus_weird"] = {
            "point": point,
            "source_level_bootstrap_ci": [
                float(x) for x in np.percentile(np.asarray(diffs), [2.5, 97.5])
            ],
            "margin": H2_MARGIN,
            "descriptive_verdict": bool(point >= H2_MARGIN),
        }
    else:
        out["ordinary_minus_weird"] = {"skipped": "missing ordinary or weird class"}
    return out


# --------------------------------------------------------------------------
# decide phase (registered MF-C truth table)
# --------------------------------------------------------------------------


def decide_verdict(
    a_pass: bool, b_pass: bool, ni_lo: float, ni_hi: float, margin: float = NI_MARGIN
) -> str:
    """The registered decision function (plan S3, byte-semantics; MF-C).

    Disjoint + exhaustive: NOT A_pass -> Inconclusive (instrument voided);
    A_pass & NOT B_pass -> Fails-to-replicate; A_pass & B_pass: CI lower bound
    > -margin -> Replicates; CI wholly below -margin -> Fails-to-replicate;
    CI spans -margin -> Inconclusive."""
    if not a_pass:
        return "Inconclusive"
    if not b_pass:
        return "Fails-to-replicate"
    if ni_lo > -margin:
        return "Replicates"
    if ni_hi < -margin:
        return "Fails-to-replicate"
    return "Inconclusive"


def _load_model_artifacts(fits_dir: Path):
    summary = json.loads((fits_dir / "fits_summary.json").read_text())
    recon = json.loads((fits_dir / "percontext_recon.json").read_text())
    return summary, recon


def _ceiling_at(reliability_path: str | None, hs: int):
    """Reliability ceiling at hs from a reliability_ceiling.json (MF-E), or None."""
    if not reliability_path:
        return None
    doc = json.loads(Path(reliability_path).read_text())
    layer = doc.get("per_layer", {}).get(f"L{hs:02d}")
    if layer is None:
        raise RuntimeError(f"{reliability_path} has no layer L{hs:02d}")
    return {
        "hs": hs,
        "ceiling_pooled": layer["ceiling_pooled"],
        "ceiling_within_source": layer.get("ceiling_within_source"),
        "source": reliability_path,
    }


def run_decide(args) -> dict:
    """Compose the registered decision from both models' assembled artifacts."""
    import numpy as np

    GC = _gc()
    out_root = Path(args.out_root)
    sum_a, rec_a = _load_model_artifacts(out_root / "fits" / "modelA")
    sum_b, rec_b = _load_model_artifacts(out_root / "fits" / "modelB")

    gates = {}
    passes = {}
    for key, summary, recon in (("A", sum_a, rec_a), ("B", sum_b, rec_b)):
        gate_set = summary["candidate_sets"]["gate"]
        gate_hs = summary["selected"]["gate_hs"]
        mats = recon_arrays_from_file(recon, recon["candidate_sets"]["all"])
        member = np.asarray(
            [c["regime_class"] == "ordinary" for c in recon["contexts"]], dtype=bool
        )
        if member.sum() < 10:
            raise RuntimeError(
                f"model {key}: only {int(member.sum())} ordinary test contexts — "
                "H1 primary split degenerate"
            )
        boot = paired_delta_bootstrap(
            mats,
            cand=gate_set,
            frozen_hs=gate_hs,
            member_mask=member,
            draws=args.boot_draws,
            seed=BOOT_SEED + (0 if key == "A" else 1),
        )
        passes[key] = bool(boot["inherited_ci"][0] > 0.0)
        gates[key] = {
            "h1_slice": "regime_class == ordinary (test partition)",
            "gate_hs": gate_hs,
            "predicate": f"{key}_pass: inherited paired-bootstrap CI on mean "
            "(err_identity - err_map) excludes 0 on the positive side",
            "value": passes[key],
            **boot,
        }

    ni = ni_bootstrap(
        recon_arrays_from_file(rec_a, rec_a["candidate_sets"]["all"]),
        [c["context_id"] for c in rec_a["contexts"]],
        recon_arrays_from_file(rec_b, rec_b["candidate_sets"]["all"]),
        [c["context_id"] for c in rec_b["contexts"]],
        cand_a=sum_a["candidate_sets"]["h3"],
        cand_b=sum_b["candidate_sets"]["h3"],
        frozen_a=sum_a["selected"]["h3_hs"],
        frozen_b=sum_b["selected"]["h3_hs"],
        draws=args.boot_draws,
        seed=BOOT_SEED + 2,
    )
    a_best = sum_a["r2_at_h3_layer"]["map_pooled"]
    b_best = sum_b["r2_at_h3_layer"]["map_pooled"]
    ni_lo, ni_hi = ni["inherited_ci_diff"]
    verdict = decide_verdict(passes["A"], passes["B"], ni_lo, ni_hi)

    ceil_a = _ceiling_at(args.reliability_a, sum_a["selected"]["h3_hs"])
    ceil_b = _ceiling_at(args.reliability_b, sum_b["selected"]["h3_hs"])
    decision = {
        "meta": GC.run_metadata({"artifact": "decision", "boot_draws": args.boot_draws}),
        "truth_table": (
            "Replicates <=> A_pass AND B_pass AND ni_lo > -0.10; "
            "Fails-to-replicate <=> A_pass AND ((ni_hi < -0.10) OR NOT B_pass); "
            "Inconclusive <=> NOT A_pass OR (A_pass AND B_pass AND CI spans -0.10)"
        ),
        "a_pass": passes["A"],
        "b_pass": passes["B"],
        "gates": gates,
        "h3": {
            "a_best_r2_at_h3_layer": a_best,
            "b_best_r2_at_h3_layer": b_best,
            "diff_point_b_minus_a": b_best - a_best,
            "ni_margin": NI_MARGIN,
            **ni,
        },
        "reliability_conditioning": {
            "note": (
                "MF-E: the H3 verdict is STATED RELATIVE to the per-model "
                "answer-vector reliability ceilings — a lower B_best with a "
                "commensurately lower ceil_B is not a weaker map"
            ),
            "ceil_A_at_h3_layer": ceil_a,
            "ceil_B_at_h3_layer": ceil_b,
            "a_best_over_ceiling": (
                a_best / ceil_a["ceiling_pooled"] if ceil_a and ceil_a["ceiling_pooled"] else None
            ),
            "b_best_over_ceiling": (
                b_best / ceil_b["ceiling_pooled"] if ceil_b and ceil_b["ceiling_pooled"] else None
            ),
        },
        "h2": {
            "A": h2_contrast(
                rec_a, sum_a["selected"]["gate_hs"], draws=args.boot_draws, seed=BOOT_SEED + 3
            ),
            "B": h2_contrast(
                rec_b, sum_b["selected"]["gate_hs"], draws=args.boot_draws, seed=BOOT_SEED + 4
            ),
        },
        "verdict": verdict,
    }
    GC.atomic_write_json(out_root / "fits" / "decision.json", decision)
    print(
        f"[decide] A_pass={passes['A']} B_pass={passes['B']} "
        f"NI inherited CI=({ni_lo:.4f},{ni_hi:.4f}) -> {verdict}",
        flush=True,
    )
    return decision


# --------------------------------------------------------------------------
# selfcheck (synthetic end-to-end; equivalence + reachable verdict branches)
# --------------------------------------------------------------------------


def _toy_store(model_key: str, layer_hs: list[int], *, n=240, d=8, noise_by_hs=None, seed=0):
    """Synthetic MemStore: y = x @ W_k + noise_k per layer, 4 sources x 2 regimes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sources = [
        ("s_ord1", "ordinary"),
        ("s_ord2", "ordinary"),
        ("s_wrd1", "weird"),
        ("s_wrd2", "weird"),
    ]
    rows = []
    for i in range(n):
        src, cls = sources[i % 4]
        split = ("train", "train", "train", "val", "test")[i % 5]
        rows.append(
            {
                "row": i,
                "context_id": f"ctx{i:04d}",
                "source_tag": src,
                "regime_class": cls,
                "lodo_group": src,
                "split": split,
            }
        )
    X = rng.standard_normal((n, d))
    layers = {}
    for k in layer_hs:
        W = rng.standard_normal((d, d)) / math.sqrt(d)
        sigma = (noise_by_hs or {}).get(k, 0.3)
        layers[k] = (
            X.astype(np.float32),
            (X @ W + sigma * rng.standard_normal((n, d))).astype(np.float32),
        )
    return MemStore(rows, layers)


def _selfcheck_equivalence(tmp: Path) -> None:
    """Streamed core == the #779 reference fit on identical inputs."""
    import numpy as np

    N779 = _n779()
    rng = np.random.default_rng(7)
    n, d = 40, 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = (X @ rng.standard_normal((d, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    idx = rng.permutation(n)
    tr, val, te = idx[:24], idx[24:32], idx[32:]
    lambdas = N779.LAMBDAS_N50K
    ref_pred, ref_meta = N779.fit_ridge_primal(X, Y, tr, val, te, lambdas, "cpu")
    (new_pred,), new_meta = ridge_fit_streamed(X, Y, tr, val, [te], lambdas, "cpu")
    assert new_meta["selected_lambda"] == ref_meta["selected_lambda"], (new_meta, ref_meta)
    assert new_meta["lambda_grid_edge"] == ref_meta["lambda_grid_edge"]
    assert abs(new_meta["val_r2_at_selected"] - ref_meta["val_r2_at_selected"]) < 1e-9
    assert np.allclose(new_pred, ref_pred, rtol=1e-9, atol=1e-9)
    print("[selfcheck] streamed-core equivalence vs issue779 fit_ridge_primal: OK", flush=True)


def _selfcheck_truth_table() -> None:
    """Direct probes of every registered verdict branch (MF-C; reachable
    Inconclusive via BOTH routes: NOT A_pass, and a CI spanning the margin)."""
    assert decide_verdict(True, True, -0.05, 0.10) == "Replicates"
    assert decide_verdict(True, True, -0.30, -0.15) == "Fails-to-replicate"
    assert decide_verdict(True, False, -0.05, 0.10) == "Fails-to-replicate"
    assert decide_verdict(True, True, -0.30, 0.10) == "Inconclusive"  # CI spans -0.10
    assert decide_verdict(False, True, -0.05, 0.10) == "Inconclusive"  # instrument voided
    assert decide_verdict(True, True, -0.10, 0.10) == "Inconclusive"  # boundary -> spans
    print("[selfcheck] MF-C truth table (5 branches + boundary): OK", flush=True)


def _selfcheck_pipeline(args, tmp: Path) -> None:
    """Full fit -> assemble -> decide pass on synthetic two-model stores,
    exercising baselines, kNN, per-source + LODO folds, and both CI kinds."""
    import copy

    a = copy.copy(args)
    a.work_dir = str(tmp / "work")
    a.out_root = str(tmp / "out")
    a.pilot = False
    a.layers = None
    a.skip_lodo = False
    a.lodo_groups = None
    a.boot_draws = 200
    a.knn_max_n = 64
    a.device = "cpu"
    a.tensors_prefix = None
    noise = {1: 0.8, 2: 0.15, 3: 0.5}
    store_a = _toy_store("A", [1, 2, 3], noise_by_hs=noise, seed=1)
    store_b = _toy_store("B", [1, 2, 3, 4], noise_by_hs={**noise, 4: 0.2}, seed=2)
    specs = {
        "A": {"all": [1, 2, 3], "gate": [1, 2, 3], "h3": [2, 3]},
        "B": {"all": [1, 2, 3, 4], "gate": [2, 4], "h3": [2, 4]},
    }
    for key, store in (("A", store_a), ("B", store_b)):
        _run_fit_with_sets(a, key, store, specs[key])
    a.reliability_a = None
    a.reliability_b = None
    decision = run_decide(a)
    assert decision["a_pass"] is True, "toy A map must beat identity baseline"
    assert decision["verdict"] in ("Replicates", "Inconclusive", "Fails-to-replicate")
    # Degrade B to pure noise -> NOT B_pass -> Fails-to-replicate through the
    # REAL decide path (not just the truth-table fn).
    b_dir = Path(a.out_root) / "fits" / "modelB"
    rec = json.loads((b_dir / "percontext_recon.json").read_text())
    for lk in rec["layers"]:
        e_id = rec["layers"][lk]["err_identity"]
        rec["layers"][lk]["err_map"] = [v * 50.0 + 1.0 for v in e_id]
    _gc().atomic_write_json(b_dir / "percontext_recon.json", rec)
    d2 = run_decide(a)
    assert d2["b_pass"] is False and d2["verdict"] == "Fails-to-replicate", d2["verdict"]
    # Void A's instrument (err_map == err_identity) -> Inconclusive, reachable
    # through the REAL path.
    a_dir = Path(a.out_root) / "fits" / "modelA"
    rec_a = json.loads((a_dir / "percontext_recon.json").read_text())
    for lk in rec_a["layers"]:
        rec_a["layers"][lk]["err_map"] = list(rec_a["layers"][lk]["err_identity"])
    _gc().atomic_write_json(a_dir / "percontext_recon.json", rec_a)
    d3 = run_decide(a)
    assert d3["a_pass"] is False and d3["verdict"] == "Inconclusive", d3["verdict"]
    print("[selfcheck] pipeline verdicts: base + Fails + REACHABLE Inconclusive: OK", flush=True)


def _run_fit_with_sets(args, model_key: str, store, sets: dict) -> dict:
    """run_fit with injected candidate sets (selfcheck seam for toy layer counts).

    Mirrors run_fit exactly but skips the production MF-J/candidate derivation
    (which is pinned to the real 28/32-layer models)."""
    import numpy as np

    GC = _gc()
    N779 = _n779()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    fits_dir, percell = model_dirs(args, model_key)
    percell.mkdir(parents=True, exist_ok=True)
    rows = store.load_rows()
    all_set = sorted(sets["all"])
    tr, val, te = resolve_splits(rows, pilot=args.pilot)
    rows_te = [rows[int(i)] for i in te]
    ledger = GC.StageLedger(
        percell / "ledger.json", fit_regime(args, model_key, len(rows), sha16("toy"))
    )
    lambdas = N779.LAMBDAS_N50K
    units = {}
    t0 = time.time()
    for j, k in enumerate(all_set):
        X, Y = store.load_layer(k)
        unit, _ = fit_layer_unit(X, Y, tr, val, te, rows_te, k, lambdas, args.device)
        GC.atomic_write_json(percell / f"L{k:02d}.json", unit)
        ledger.mark_done(f"L{k:02d}")
        units[k] = unit
        GC.progress(f"fit-{model_key}", j + 1, len(all_set), f"L{k:02d}", t0)
    val_r2 = {k: units[k]["fit_meta"]["val_r2_at_selected"] for k in all_set}
    sel_gate = min(sets["gate"], key=lambda k: (-val_r2[k], k))
    sel_h3 = min(sets["h3"], key=lambda k: (-val_r2[k], k))
    X, Y = store.load_layer(sel_gate)
    (pred_te,), _ = ridge_fit_streamed(X, Y, tr, val, [te], lambdas, args.device)
    knn = knn_unit(
        pred_te, np.asarray(Y[te], dtype=np.float64), rows_te, max_n=args.knn_max_n, seed=BOOT_SEED
    )
    GC.atomic_write_json(percell / "post.json", {"gate_hs": sel_gate, "knn": knn})
    for g in sorted({str(r["lodo_group"]) for r in rows}):
        fold = lodo_unit(X, Y, rows, tr, val, g, sel_gate, lambdas, args.device)
        GC.atomic_write_json(percell / f"lodo_{g}.json", fold)
        ledger.mark_done(f"lodo_{g}")
    ledger.mark_done("post")
    mfj = {"selfcheck": True, "subset_ok": True}
    sel = {
        "all": all_set,
        "gate": sorted(sets["gate"]),
        "h3": sorted(sets["h3"]),
        "selected_gate_hs": sel_gate,
        "selected_h3_hs": sel_h3,
    }
    return assemble_model(
        args, model_key, percell, fits_dir, rows_te, units, mfj, sel, tr=tr, val=val, te=te
    )


def run_selfcheck(args) -> int:
    import tempfile

    _selfcheck_truth_table()
    with tempfile.TemporaryDirectory(prefix="i2502_fits_selfcheck_") as td:
        tmp = Path(td)
        _selfcheck_equivalence(tmp)
        _selfcheck_pipeline(args, tmp)
    # Candidate-set derivation pins (production constants).
    assert h3_hs_set("B") == [4, 8, 12, 16, 20, 24, 28, 32]
    assert h3_hs_set("A") == [3, 7, 10, 14, 17, 21, 24, 28]
    assert assert_mfj("B", list(range(1, 33)))["subset_ok"]
    assert assert_mfj("A", list(range(1, 29)))["subset_ok"]
    try:
        assert_mfj("B", [k for k in range(1, 33) if k != 32])
    except RuntimeError:
        pass
    else:
        raise AssertionError("MF-J assert failed to fire on a missing H3 layer")
    print("[selfcheck] ALL OK", flush=True)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("fit", "decide", "selfcheck"), default="fit")
    ap.add_argument("--model-key", choices=("A", "B"), default=None, help="fit phase: which model")
    ap.add_argument(
        "--tensors-prefix",
        default=None,
        help="HF data-repo prefix of the u2 capture store (default per --model-key)",
    )
    ap.add_argument("--work-dir", default="/workspace/issue2502_fits")
    ap.add_argument("--out-root", default=str(_REPO_ROOT / "eval_results" / "issue_2502"))
    ap.add_argument("--device", default="cpu", help="fit device (cpu on the cpu-bigmem pod)")
    ap.add_argument("--pilot", action="store_true", help="G1 pilot mode (80/20, MF-B)")
    ap.add_argument(
        "--g1-gate",
        action="store_true",
        help=f"with --pilot: exit rc={G1_GATE_RC} when the G1 gate FAILs (designed halt)",
    )
    ap.add_argument("--layers", default=None, help="comma hs subset (across-layer sharding)")
    ap.add_argument("--skip-lodo", action="store_true")
    ap.add_argument("--lodo-groups", default=None, help="comma subset of lodo groups")
    ap.add_argument("--boot-draws", type=int, default=2000)
    ap.add_argument("--knn-max-n", type=int, default=5000)
    ap.add_argument("--reliability-a", default=None, help="decide: modelA reliability_ceiling.json")
    ap.add_argument("--reliability-b", default=None, help="decide: modelB reliability_ceiling.json")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_fits: import-check OK", flush=True)
        return 0
    # load_dotenv BEFORE any numpy/torch import (thread caps freeze at import, #847).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    rc = 0
    if args.phase == "selfcheck":
        rc = run_selfcheck(args)
    elif args.phase == "decide":
        run_decide(args)
    else:
        summary = run_fit(args)
        if args.pilot and args.g1_gate and args.model_key == "A":
            verdict = summary.get("g1", {}).get("verdict")
            if verdict != "PROCEED":
                print(f"[g1-gate] verdict={verdict} -> designed halt rc={G1_GATE_RC}", flush=True)
                rc = G1_GATE_RC
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
