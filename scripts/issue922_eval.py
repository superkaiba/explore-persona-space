#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #922 Phase 3: single-step atlas, rollouts, read-out benchmark, transfer.

Writes each phase's JSON the moment the phase completes (checkpoint-per-phase):

- ``stage0_position_atlas.json`` — DV1: held-out Δ-R² per (layer × class ×
  arm × space × segment) with the copy-previous self-check, the 100-draw
  shuffled-context null band (batched — predictions computed ONCE, every
  draw is index arithmetic on precomputed pair matrices; no per-draw fit),
  the H2 depth-profile ratio reads, GCV-λ tables, autocorrelation/‖Δ‖
  diagnostics, and the LMSYS duplicate-prompt report.
- ``rollout_skill.json`` (+ ``rollout_percontext.npz``) — DV2: rollout skill
  vs horizon k per layer; PRIMARY path = step-1 BOUNDARY map then the answer
  map (ridge; MLP companion uses the ridge boundary step too), scored against
  the frozen-state null AND the context-blind mean-drift null (both paired
  per-context), plus the token-informed rollout ceiling, the naive
  all-answer-map diagnostic, and the exploratory GRU overlay.
- ``readout_benchmark.json`` (+ ``readout_projections.npz``) — DV3: trait
  read-out off rolled states at the pre-registered ℓ*, #779's
  within-condition-Pearson protocol verbatim; decision statistic = the
  horizon-mean read over k=1..32; per-k reads persisted (selection-symmetric
  nulls recoverable post-hoc). Gated by the rig-validation assert against
  ``step0_oracle.json`` (kill criterion (c): on mismatch DV3 is SKIPPED and
  recorded, never run on misaligned units).
- ``transfer_eval.json`` — DV4: the LMSYS-fit maps applied UNCHANGED to the
  eval-condition windows (single-step + rollout + the true-answer-state
  read-out ceiling on the captured subset).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue922_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue922_fit_maps import transition_indices  # noqa: E402

from explore_persona_space.experiments.issue_779.metrics import (  # noqa: E402
    bootstrap_delta_ci,
    bootstrap_within_condition_ci,
)
from explore_persona_space.experiments.issue_841.maps import RidgeMap  # noqa: E402
from explore_persona_space.experiments.issue_922 import maps922 as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue922_eval")

MODES = ("system", "many_shot")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested if (requested != "cuda" or torch.cuda.is_available()) else "cpu"


def _load_ridge(maps_dir: Path) -> dict:
    blob = torch.load(maps_dir / "maps_ridge.pt", weights_only=False)
    out = {"answer": {}, "boundary": {}}
    for kind in ("answer", "boundary"):
        for arm, d in blob[kind].items():
            out[kind][arm] = {int(r): RidgeMap(**st) for r, st in d.items()}
    out["b1_answer"] = {
        int(r): RidgeMap(**st) for r, st in blob.get("b1_answer", {}).items()
    }  # v6: the closed-form b1 [h, c] maps (may be absent on r1-era artifacts)
    out["delta_train_mean"] = blob["delta_train_mean"]
    out["boundary_train_mean"] = blob["boundary_train_mean"]
    out["sigma_by_row"] = blob["sigma_by_row"]
    out["rows"] = list(blob["rows"])
    return out


def _load_direct_row(direct_dir: Path, r: int) -> dict | None:
    """One arm-c per-row file → {k: RidgeMap} (fp16 weights cast to fp32)."""
    p = direct_dir / f"direct_row_{r:02d}.pt"
    if not p.exists():
        return None
    blob = torch.load(p, weights_only=False)
    maps = {}
    for k, st in blob["maps"].items():
        st = dict(st)
        st["w"] = st["w"].to(torch.float32)
        maps[int(k)] = RidgeMap(**st)
    return {
        "maps": maps,
        "diag": blob["diag"],
        "coherence": blob.get("coherence"),
        "regime": blob.get("regime"),
    }


def _horizon_mean_perctx(per_ctx: np.ndarray, k_hi: int) -> np.ndarray:
    """Per-context mean over the first k_hi horizons (NaN pads ignored)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows → NaN
        return np.nanmean(per_ctx[:, :k_hi], axis=1)


def _stack_maps(maps_by_row: dict, rows: list[int], device: str) -> dict:
    """Stack per-row RidgeMaps into (L,·) tensors for the batched rollout."""
    ms = [maps_by_row[r].to(device) for r in rows]
    return {
        "mus": torch.stack([m.mu for m in ms]),
        "sds": torch.stack([m.sd for m in ms]),
        "ws": torch.stack([m.w for m in ms]),
        "biases": torch.stack([m.bias for m in ms]),
    }


def _gather_X(h, row, idx, arm):
    x = h[row, idx, :].to(torch.float32)
    if arm == "ctx":
        return x
    e = h[0, idx + 1, :].to(torch.float32)
    return e if arm == "emb" else torch.cat([x, e], dim=1)


def _boot_mean_ci(vals: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile bootstrap CI of the mean over contexts (vectorized gather).

    Estimand equivalence (the plan-v6 H6/H7 pin): for a MEAN statistic,
    bootstrapping the per-context paired-delta vector here is algebraically
    identical to the parent ``bootstrap_delta_ci``'s resample-then-difference
    — the mean is linear, so under the SAME index draw
    ``mean(a[idx] − b[idx]) == mean(a[idx]) − mean(b[idx])`` — with the same
    resample unit (context), ``seed=0`` (``C.BOOTSTRAP_SEED``),
    ``n_boot=1000 ≥ 997``, and the same 2.5/97.5 percentile interval. The
    parent helper is still used verbatim where its unit axis applies (DV3,
    ``readout_phase``); this helper carries the H6/H7 per-context reads.
    """
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = vals[idx].mean(axis=1)
    return {
        "mean": float(vals.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
        "n": n,
    }


def method_metrics(x, mat, *, n_boot, seed) -> dict:
    """Within-condition r + bootstrap CI per mode (ported #841 stage1 wrapper)."""
    res = {}
    for mode in MODES:
        cx, cy = C.group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
        cx2, cy2 = [], []
        for xi, yi in zip(cx, cy, strict=True):
            m = np.isfinite(xi)
            if m.sum() >= 3:
                cx2.append(xi[m])
                cy2.append(yi[m])
        res[mode] = bootstrap_within_condition_ci(cx2, cy2, n_boot=n_boot, seed=seed)
    return res


# ── DV1 atlas ─────────────────────────────────────────────────────────────────


def atlas_phase(store, ridge, mlp_blob, split, tr, args, device) -> dict:  # noqa: C901 — one pass per (segment x class x arm) cell; flattening would inline the metric helpers
    h = store["h"]
    rows = ridge["rows"]
    sigma = ridge["sigma_by_row"]
    out_cells: dict = {}
    seg_idx = {seg: tr["test"][seg] for seg in ("answer", "boundary", "prompt", "template_end")}

    for r in rows:
        bk = C.row_to_block_key(r)
        tm_raw = ridge["delta_train_mean"][r].numpy()
        for seg, idx in seg_idx.items():
            if idx.numel() == 0:
                continue
            delta = _gather_X(h, r, idx + 1, "ctx") - _gather_X(h, r, idx, "ctx")
            delta = delta.cpu().numpy()
            cell_key = f"{bk}|{seg}"
            cells: dict = {}
            # copy-previous null: r2_id == 0.0 exactly (metric self-check).
            z = np.zeros_like(delta)
            r2_zero = M.identity_relative_r2(z, delta)
            assert r2_zero == 0.0, ("copy_prev r2_id != 0", r, seg, r2_zero)
            cells["copy_prev"] = {
                "raw": {
                    "r2_id": 0.0,
                    "r2_meancentered": M.mean_centered_r2(z, delta, tm_raw),
                    "delta_err_raw": M.delta_error_percentiles(z, delta),
                }
            }
            for arm in ("ctx", "tok", "emb"):
                X = _gather_X(h, r, idx, arm)
                pred = M.ridge_predict(ridge["answer"][arm][r], X).numpy()
                met = {
                    "r2_id": M.identity_relative_r2(pred, delta),
                    "r2_meancentered": M.mean_centered_r2(pred, delta, tm_raw),
                    "delta_err_raw": M.delta_error_percentiles(pred, delta),
                    "best_lam": ridge["answer"][arm][r].best_lam,
                }
                cells[f"ridge_{arm}"] = {
                    "raw": met,
                    # scalar-σ rescale ⇒ identical fit + identical scale-free
                    # metrics (maps922 docstring); recorded, not re-fit.
                    "rmsnorm": {**met, "shared_fit_with_raw": True},
                }
            if mlp_blob is not None:
                for arm in ("ctx", "tok"):
                    per_space = {}
                    for space in ("raw", "rmsnorm"):
                        fits = mlp_blob["fits"].get(f"{arm}__{space}")
                        if fits is None or r not in fits:
                            continue
                        sig = 1.0 if space == "raw" else float(sigma[r])
                        X = _gather_X(h, r, idx, arm)
                        pred_fit = M.apply_mlp_params(fits[r]["params"], X, device).cpu().numpy()
                        pred_raw = pred_fit * sig
                        per_space[space] = {
                            "r2_id": M.identity_relative_r2(pred_fit, delta / sig),
                            "r2_meancentered": M.mean_centered_r2(
                                pred_fit, delta / sig, tm_raw / sig
                            ),
                            "delta_err_raw": M.delta_error_percentiles(pred_raw, delta),
                            "best_val_epoch": fits[r]["best_val_epoch"],
                        }
                    if per_space:
                        cells[f"mlp_{arm}"] = per_space
            out_cells[cell_key] = cells

    # ── shuffled-context null (ridge_ctx, answer segment; batched draws) ──────
    null = shuffle_null(store, ridge, split, args.n_shuffle, device)

    # ── H2 depth-profile ratio (answer segment, raw) ──────────────────────────
    blocks_num = [r - 1 for r in rows if r != 0]
    ratios, diffs, mask = [], [], []
    for b in blocks_num:
        cell = out_cells.get(f"{b}|answer", {})
        rc = cell.get("ridge_ctx", {}).get("raw", {}).get("r2_id")
        rt = cell.get("ridge_tok", {}).get("raw", {}).get("r2_id")
        if rc is None or rt is None:
            continue
        ratios.append(rc / rt if rt and rt > 0.01 else float("nan"))
        diffs.append(rt - rc)
        mask.append(bool(rt and rt > 0.01))
    h2 = {
        "blocks": blocks_num,
        "ratio_ctx_over_tok": ratios,
        "diff_tok_minus_ctx": diffs,
        "denominator_positive_mask": mask,
    }
    ok = [i for i, m in enumerate(mask) if m and np.isfinite(ratios[i])]
    if len(ok) >= 3:
        from scipy.stats import spearmanr

        rho, p = spearmanr([blocks_num[i] for i in ok], [ratios[i] for i in ok])
        h2["spearman_ratio"] = {"rho": float(rho), "p": float(p), "n_layers": len(ok)}
        rho2, p2 = spearmanr(blocks_num, diffs)
        h2["spearman_diff_tok_minus_ctx"] = {"rho": float(rho2), "p": float(p2)}
    mlp_ratio = []
    for b in blocks_num:
        cell = out_cells.get(f"{b}|answer", {})
        rc = cell.get("mlp_ctx", {}).get("raw", {}).get("r2_id")
        rt = cell.get("mlp_tok", {}).get("raw", {}).get("r2_id")
        mlp_ratio.append(
            (rc / rt) if (rc is not None and rt is not None and rt > 0.01) else float("nan")
        )
    h2["mlp_ratio_ctx_over_tok"] = mlp_ratio

    # ── diagnostics: autocorr + ‖Δ‖ + per-answer-position R² (heatmap data) ───
    diag = {"autocorr_by_row": {}, "delta_norm_by_row": {}, "r2_by_ansrel": {}}
    idx = tr["test"]["answer"]
    if idx.numel() > 0:
        ansrel = tr["test"]["ansrel"]
        rel = np.array([ansrel[int(s)] for s in idx])
        for r in rows:
            a = _gather_X(store["h"], r, idx, "ctx")
            b_ = _gather_X(store["h"], r, idx + 1, "ctx")
            cos = torch.nn.functional.cosine_similarity(a, b_, dim=1)
            diag["autocorr_by_row"][C.row_to_block_key(r)] = float(cos.mean())
            diag["delta_norm_by_row"][C.row_to_block_key(r)] = float((b_ - a).norm(dim=1).mean())
            pred = M.ridge_predict(ridge["answer"]["ctx"][r], a).numpy()
            delta = (b_ - a).numpy()
            by_pos = []
            for pos in range(int(rel.max()) + 1 if len(rel) else 0):
                m = rel == pos
                by_pos.append(
                    M.identity_relative_r2(pred[m], delta[m]) if int(m.sum()) >= 20 else None
                )
            diag["r2_by_ansrel"][C.row_to_block_key(r)] = by_pos
    return {"cells": out_cells, "shuffle_null": null, "h2_ratio": h2, "diagnostics": diag}


def shuffle_null(store, ridge, split, n_draws: int, device: str) -> dict:
    """Context-mismatch null at matched answer-relative position (100 draws).

    The FITTED context-only ridge predictions are computed ONCE per layer;
    every draw is then pure index arithmetic on precomputed (n_ctx × n_ctx)
    pair matrices S/T/G (masked matmuls) — NO per-draw fit or data pass
    (vectorize-many-cell-fits, draw-battery form).
    """
    h = store["h"]
    rows = ridge["rows"]
    pos_lo, n_pos = store["pos_lo"], store["n_pos"]
    plen, ws = store["prompt_len"], store["window_start"]
    segs = store["segments"]
    test_ctx = list(split["test"])
    A_max = C.W_A
    n = len(test_ctx)
    Mpres = torch.zeros(n, A_max)
    src_of = torch.full((n, A_max), -1, dtype=torch.long)
    for ii, i in enumerate(test_ctx):
        lo, npos = int(pos_lo[i]), int(n_pos[i])
        for j in range(npos - 1):
            src = lo + j
            if int(segs[src]) == C.SEG_ANSWER:
                a = (int(ws[i]) + j) - int(plen[i])
                if 0 <= a < A_max:
                    Mpres[ii, a] = 1.0
                    src_of[ii, a] = src
    flat = src_of[src_of >= 0]
    draws_r2: dict[str, list[float]] = {}
    pair_counts = None
    perms = [np.random.default_rng(d).permutation(n) for d in range(n_draws)]
    for r in rows:
        rm = ridge["answer"]["ctx"][r].to(device)
        X = h[r, flat, :].to(device=device, dtype=torch.float32)
        pred = (M.ridge_predict(rm, X)).cpu()
        dl = h[r, flat + 1, :].to(torch.float32) - h[r, flat, :].to(torch.float32)
        Yh = torch.zeros(n, A_max, pred.shape[-1])
        Dl = torch.zeros(n, A_max, pred.shape[-1])
        sel = src_of >= 0
        Yh[sel] = pred
        Dl[sel] = dl
        G = torch.einsum("iah,jah->ij", Yh, Dl)
        ny2 = (Yh * Yh).sum(-1)  # (n, A)
        nd2 = (Dl * Dl).sum(-1)
        S = ny2 @ Mpres.t()
        T = Mpres @ nd2.t()
        r2s = []
        counts = np.zeros(A_max)
        for perm in perms:
            keep = perm != np.arange(n)
            i_idx = torch.tensor(np.arange(n)[keep])
            j_idx = torch.tensor(perm[keep])
            sse = (S[i_idx, j_idx] + T[i_idx, j_idx] - 2 * G[i_idx, j_idx]).sum()
            tot = T[i_idx, j_idx].sum()
            r2s.append(float(1.0 - sse / tot) if float(tot) > 0 else float("nan"))
            if r == rows[0]:
                counts += (Mpres[i_idx] * Mpres[j_idx]).sum(0).numpy()
        if pair_counts is None:
            pair_counts = (counts / n_draws).tolist()
        arr = np.array(r2s, dtype=np.float64)
        draws_r2[C.row_to_block_key(r)] = arr.tolist()
    bands = {
        bk: {
            "p2_5": float(np.nanquantile(np.array(v), 0.025)),
            "p50": float(np.nanquantile(np.array(v), 0.5)),
            "p97_5": float(np.nanquantile(np.array(v), 0.975)),
            "max": float(np.nanmax(np.array(v))),
        }
        for bk, v in draws_r2.items()
    }
    return {
        "n_draws": n_draws,
        "band_by_row": bands,
        "draws_by_row": draws_r2,
        "mean_pairing_count_by_ansrel": pair_counts,
        "note": "ridge_ctx, answer segment, raw space (rmsnorm identical — scalar σ)",
    }


# ── v6: conditioned single-step cells + the H6 paired statistic ──────────────


def conditioned_phase(store, ridge, cond_blob, maps_dir, split, tr, args, device) -> dict:
    """Per-(row, form) single-step cells + per-context H6 inputs (plan §6.5).

    Metrics on the held-out test answer-segment transitions, raw space (the
    v4 arms are raw-only). Persists, per cell: pooled r2_id / r2_meancentered,
    AND the per-context r2 / SSE / identity-denominator vectors (the v6
    estimand pin — the H6 paired statistic is the mean over contexts of the
    paired per-context r2 delta, with a pooled-recompute sensitivity companion
    and the persisted denominators for the outlier read). H6 primary =
    b2_film − b1_grad at the 6 ℓ* rows; lowrank/mixture are companions.
    """
    h = store["h"]
    idx = tr["test"]["answer"]
    Tix = tr["test"]["answer_T"]
    cgrp = tr["test"]["answer_ctx"]
    n_test_ctx = len(split["test"])
    out: dict = {"cells": {}, "per_context": {}, "h6": {}, "n_test_ctx": n_test_ctx}

    def _percontext(pred: np.ndarray, delta: np.ndarray) -> dict:
        sse = ((pred - delta) ** 2).sum(axis=1)
        den = (delta**2).sum(axis=1)
        sse_c = np.zeros(n_test_ctx)
        den_c = np.zeros(n_test_ctx)
        np.add.at(sse_c, cgrp, sse)
        np.add.at(den_c, cgrp, den)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_c = 1.0 - sse_c / den_c
        return {"sse": sse_c, "den": den_c, "r2": r2_c}

    def _cell(name: str, r: int, pred: np.ndarray, delta: np.ndarray, extra: dict) -> None:
        bk = C.row_to_block_key(r)
        tm = ridge["delta_train_mean"][r].numpy()
        pc = _percontext(pred, delta)
        out["cells"].setdefault(name, {})[bk] = {
            "r2_id": M.identity_relative_r2(pred, delta),
            "r2_meancentered": M.mean_centered_r2(pred, delta, tm),
            "delta_err_raw": M.delta_error_percentiles(pred, delta),
            **extra,
        }
        out["per_context"].setdefault(name, {})[bk] = {
            "r2": [float(x) for x in pc["r2"]],
            "sse": [float(x) for x in pc["sse"]],
            "den": [float(x) for x in pc["den"]],
        }

    if idx.numel() == 0:
        out["skipped"] = "no test answer transitions"
        return out
    # closed-form b1 ridge (all fitted rows) + the ridge_ctx per-context reference
    for r in sorted(ridge.get("b1_answer", {})):
        Xh = _gather_X(h, r, idx, "ctx")
        delta = (_gather_X(h, r, idx + 1, "ctx") - Xh).numpy()
        Xc = h[r, Tix, :].to("cpu", torch.float32)
        pred = M.ridge_predict(ridge["b1_answer"][r], torch.cat([Xh, Xc], dim=1)).numpy()
        _cell("b1_ridge", r, pred, delta, {"best_lam": ridge["b1_answer"][r].best_lam})
        pred_ctx = M.ridge_predict(ridge["answer"]["ctx"][r], Xh).numpy()
        _cell("ridge_ctx_ref", r, pred_ctx, delta, {})
    # gradient forms at their fitted rows
    for form, per_row in ((cond_blob or {}).get("forms") or {}).items():
        for r, pblob in sorted(per_row.items()):
            Xh = _gather_X(h, r, idx, "ctx")
            delta = (_gather_X(h, r, idx + 1, "ctx") - Xh).numpy()
            Xc = h[r, Tix, :].to("cpu", torch.float32)
            pred = M.conditioned_predict_row(pblob, Xh, Xc, device).cpu().numpy()
            _cell(
                form,
                r,
                pred,
                delta,
                {
                    "best_val_epoch": pblob["best_val_epoch"],
                    "n_epochs_run": pblob["n_epochs_run"],
                    "n_params_weights": pblob["n_params_weights"],
                },
            )
    # H6 paired per-context reads at the ℓ* rows (film primary; companions)
    lstar_blocks = list(C.READOUT_BLOCKS)
    for comp in ("film", "lowrank", "mixture"):
        if comp not in out["per_context"] or "b1_grad" not in out["per_context"]:
            continue
        reads = {}
        for b in lstar_blocks:
            bk = str(b)
            if bk not in out["per_context"][comp] or bk not in out["per_context"]["b1_grad"]:
                continue
            a = np.array(out["per_context"][comp][bk]["r2"])
            b_ = np.array(out["per_context"]["b1_grad"][bk]["r2"])
            den = np.array(out["per_context"][comp][bk]["den"])
            m = np.isfinite(a) & np.isfinite(b_)
            ci = _boot_mean_ci(a[m] - b_[m], n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED)
            ci["excludes_zero"] = bool(ci["lo"] > 0.0 or ci["hi"] < 0.0)
            # pooled-recompute sensitivity companion (v6) + denominator tail
            sse_a = np.array(out["per_context"][comp][bk]["sse"])
            sse_b = np.array(out["per_context"]["b1_grad"][bk]["sse"])
            pooled_delta = float((sse_b[m].sum() - sse_a[m].sum()) / max(den[m].sum(), 1e-12))
            ci["pooled_recompute_delta"] = pooled_delta
            ci["den_quantiles"] = {
                q: float(np.quantile(den[m], float(q))) for q in ("0.0", "0.05", "0.5")
            }
            reads[bk] = ci
        n_clear = sum(1 for v in reads.values() if v["excludes_zero"] and v["mean"] > 0)
        out["h6"][f"{comp}_minus_b1_grad"] = {
            "per_lstar": reads,
            "n_lstar_positive_clear": int(n_clear),
            "n_lstar": len(reads),
            "primary": comp == "film",
        }
    if cond_blob is not None:
        out["capacity"] = cond_blob.get("capacity", {})
        out["recipe"] = {
            k: v for k, v in (cond_blob.get("recipe") or {}).items() if not torch.is_tensor(v)
        }
        out["rank"] = cond_blob.get("rank")
        out["n_mix"] = cond_blob.get("n_mix")
    # Panel footnote (r2 review minor): arm-c and the boundary fits share the
    # ans_len >= 1 context panel — SEG_BOUNDARY is only tagged when A > 0, so
    # empty-answer contexts contribute no boundary/answer transitions and drop
    # from BOTH. Surfacing the empty-answer count makes the panel composition
    # auditable from this JSON alone.
    n_empty_capture = None
    summary_p = args.store / "lmsys" / "capture_summary.json"
    if summary_p.exists():
        with open(summary_p) as f:
            n_empty_capture = json.load(f).get("empty_answers")
    out["panel"] = {
        "note": (
            "arm-c and the boundary fits share the ans_len>=1 context panel "
            "(SEG_BOUNDARY is only tagged when ans_len>0); empty-answer "
            "contexts drop from both."
        ),
        "empty_answers_store": int((store["ans_len"] == 0).sum()),
        "empty_answers_capture_summary": n_empty_capture,
    }
    dd = maps_dir / "direct_diag.json"
    if dd.exists():
        with open(dd) as f:
            blob = json.load(f)
        out["direct_diag"] = blob.get("diag_by_block")
        out["coherence_direct_vs_boundary"] = blob.get("coherence")
    return out


# ── DV2 rollout ───────────────────────────────────────────────────────────────


def drift_means(store, ridge, tr, rows, k_max: int) -> torch.Tensor:
    """Context-blind mean-drift steps: (L, k_max, H) cumulative-ready means.

    Step 1 = fit-set mean boundary Δ; step k≥2 = fit-set mean answer Δ at
    answer-relative source index k−2 (position-conditioned; falls back to the
    last populated index for sparse tails). Pure re-reduction of the store.
    """
    h = store["h"]
    ans = tr["fit"]["answer"]
    ansrel = tr["fit"]["ansrel"]
    rel = torch.tensor([ansrel[int(s)] for s in ans], dtype=torch.long)
    out = torch.zeros(len(rows), k_max, h.shape[-1])
    for li, r in enumerate(rows):
        out[li, 0] = ridge["boundary_train_mean"][r]
        dl = h[r, ans + 1, :].to(torch.float32) - h[r, ans, :].to(torch.float32)
        last = ridge["delta_train_mean"][r]
        for k in range(2, k_max + 1):
            m = rel == (k - 2)
            if int(m.sum()) >= 10:
                last = dl[m].mean(0)
            out[li, k - 1] = last
    return out


def rollout_phase(  # noqa: C901 — one builder per rollout variant; the phase IS the enumeration
    store,
    ridge,
    mlp_blob,
    gru_blob,
    split_ctx,
    args,
    device,
    *,
    corpus: str,
    drift: torch.Tensor,
    out_npz: Path | None,
    cond_blob: dict | None = None,
    direct_dir: Path | None = None,
) -> dict:
    """Score rollout skill vs horizon for one corpus's evaluation contexts.

    Variants are built LAZILY and scored one at a time (generate → score →
    free): holding every variant's 40 × (29, N, H) fp32 state stack at once is
    the r1 reviewer's A100-40 memory finding — with the v6 additions (b1/b2
    conditioned rolls + direct-c) it would exceed even the A100-80. v6 adds:
    ``b1_ridge_roll`` (all fitted rows), ``{form}_roll`` per gradient form
    (the 9-row subset), ``direct_c`` (all rows with arm-c files), plus the H7
    paired horizon-mean read (ctx-roll − direct-c at the 6 ℓ*) and the H6
    rollout companion (b2_film_roll − b1_grad_roll).
    """
    h = store["h"]
    rows = ridge["rows"]
    pos_lo, n_pos = store["pos_lo"], store["n_pos"]
    plen, ws = store["prompt_len"], store["window_start"]
    k_max = C.ROLLOUT_K_MAX
    ctxs = [i for i in split_ctx if int(plen[i]) - 1 - int(ws[i]) >= 0 and int(n_pos[i]) >= 2]
    Tpos = np.array([int(pos_lo[i]) + (int(plen[i]) - 1 - int(ws[i])) for i in ctxs])
    kcap = np.array(
        [int(ws[i]) + int(n_pos[i]) - int(plen[i]) for i in ctxs]
    )  # max k with target in-window
    Tp = torch.from_numpy(Tpos)
    seed = torch.stack([h[r, Tp, :].to(torch.float32) for r in rows]).to(device)  # (L, N, H)

    b_ctx = _stack_maps(ridge["boundary"]["ctx"], rows, device)
    a_ctx = _stack_maps(ridge["answer"]["ctx"], rows, device)

    def _build_tok_ceiling():
        emb_next = torch.zeros(len(ctxs), k_max, h.shape[-1], dtype=torch.float32)
        for ii, _ in enumerate(ctxs):
            upto = min(k_max, int(kcap[ii]))
            if upto > 0:
                sl = torch.arange(Tpos[ii] + 1, Tpos[ii] + 1 + upto)
                emb_next[ii, :upto] = h[0, sl, :].to(torch.float32)
        b_tok = _stack_maps(ridge["boundary"]["tok"], rows, device)
        a_tok = _stack_maps(ridge["answer"]["tok"], rows, device)
        return M.roll_states_ridge(seed, b_tok, a_tok, k_max, emb_next=emb_next.to(device))

    def _build_mlp():
        fits = mlp_blob["fits"]["ctx__raw"]
        states, hcur = [], seed.clone()
        for k in range(1, k_max + 1):
            if k == 1:
                dl = M.apply_ridge_maps_batched(
                    hcur, b_ctx["mus"], b_ctx["sds"], b_ctx["ws"], b_ctx["biases"]
                )
            else:
                dl = torch.stack(
                    [
                        M.apply_mlp_params(fits[r]["params"], hcur[li], device)
                        for li, r in enumerate(rows)
                    ]
                )
            hcur = hcur + dl
            states.append(hcur)
        return states

    def _build_mean_drift():
        dcum = torch.cumsum(drift.to(device), dim=1)  # (L, k, H)
        return [seed + dcum[:, k - 1, :].unsqueeze(1) for k in range(1, k_max + 1)]

    # (name, row subset, builder) — built lazily, scored, then freed.
    builders: list[tuple] = [
        ("ridge_ctx_boundary_first", rows, lambda: M.roll_states_ridge(seed, b_ctx, a_ctx, k_max)),
        (
            "ridge_ctx_naive",
            rows,
            lambda: M.roll_states_ridge(seed, b_ctx, a_ctx, k_max, use_boundary_first=False),
        ),
        ("tok_ceiling", rows, _build_tok_ceiling),
        ("mean_drift", rows, _build_mean_drift),
    ]
    if mlp_blob is not None and "ctx__raw" in mlp_blob["fits"]:
        builders.append(("mlp_ctx_boundary_first", rows, _build_mlp))
    b1_rows = [r for r in rows if r in ridge.get("b1_answer", {})]
    if b1_rows:

        def _build_b1(vrows=tuple(b1_rows)):
            li = [rows.index(r) for r in vrows]
            bsub = {k_: v[li] for k_, v in b_ctx.items()}
            b1st = _stack_maps(ridge["b1_answer"], list(vrows), device)
            return M.roll_states_b1_ridge(seed[li], bsub, b1st, k_max)

        builders.append(("b1_ridge_roll", b1_rows, _build_b1))
    for form in (cond_blob or {}).get("forms", {}):
        vrows = [r for r in rows if r in cond_blob["forms"][form]]
        if not vrows:
            continue

        def _build_cond(form=form, vrows=tuple(vrows)):
            li = [rows.index(r) for r in vrows]
            bsub = {k_: v[li] for k_, v in b_ctx.items()}
            cstack = M.stack_conditioned_params(cond_blob["forms"][form], list(vrows), device)
            return M.roll_states_conditioned(seed[li], bsub, cstack, k_max)

        builders.append((f"{form}_roll", vrows, _build_cond))

    scored: dict = {}
    row_sets: dict = {}
    for name, vrows, build in builders:
        states = build()
        scored[name] = _score_variant(states, h, list(vrows), ctxs, Tpos, kcap, k_max, device)
        row_sets[name] = list(vrows)
        del states

    # direct-c: per-row (each row's 40 fp32 maps ≈ 2 GB — load, predict, free)
    if direct_dir is not None:
        merged: dict | None = None
        d_rows = []
        for r in rows:
            dr = _load_direct_row(direct_dir, r)
            if dr is None:
                continue
            li = rows.index(r)
            states = M.predict_direct_horizons_row(dr, seed[li], seed[li], k_max, device=device)
            sc = _score_variant(
                [s.unsqueeze(0) for s in states], h, [r], ctxs, Tpos, kcap, k_max, device
            )
            d_rows.append(r)
            if merged is None:
                merged = sc
            else:
                for key in (
                    "pooled_r2_id",
                    "mean_cosine",
                    "skill_mean_ci",
                    "per_ctx_skill",
                    "per_ctx_sse",
                    "per_ctx_den",
                ):
                    merged[key].update(sc[key])
            del dr, states
        if merged is not None:
            scored["direct_c"] = merged
            row_sets["direct_c"] = d_rows

    # exploratory GRU overlay (full-prompt-window contexts only)
    gru_curves = {}
    if gru_blob is not None:
        wp_full = [ii for ii, i in enumerate(ctxs) if int(plen[i]) - int(ws[i]) == C.W_P]
        for r, g in gru_blob["grus"].items():
            r = int(r)
            if r not in rows:
                continue
            net = M.PositionGRU(h.shape[-1], gru_blob["hidden"]).to(device)
            net.load_state_dict({k: v.to(device) for k, v in g["state_dict"].items()})
            warm = torch.stack(
                [
                    h[r, torch.arange(Tpos[ii] - C.W_P + 1, Tpos[ii] + 1), :].to(torch.float32)
                    for ii in wp_full
                ]
            ).to(device)
            states = M.gru_roll_states(
                net,
                warm,
                warm[:, -1, :],
                k_max,
                mu=g["mu"].to(device),
                sd=g["sd"].to(device),
            )
            gru_curves[C.row_to_block_key(r)] = _score_variant(
                [s.unsqueeze(0) for s in states],
                h,
                [r],
                [ctxs[ii] for ii in wp_full],
                Tpos[wp_full],
                kcap[wp_full],
                k_max,
                device,
            )
            for key in ("per_ctx_sse", "per_ctx_den"):  # exploratory overlay — keep JSON lean
                gru_curves[C.row_to_block_key(r)].pop(key, None)

    # paired reads: roll − mean_drift per-context skill delta at each (row, k).
    paired = {}
    for name in ("ridge_ctx_boundary_first", "mlp_ctx_boundary_first"):
        if name not in scored:
            continue
        pr = {}
        for bk in scored[name]["per_ctx_skill"]:
            a = np.array(scored[name]["per_ctx_skill"][bk])
            b_ = np.array(scored["mean_drift"]["per_ctx_skill"][bk])
            ks = {}
            for k in range(1, k_max + 1):
                col_a, col_b = a[:, k - 1], b_[:, k - 1]
                m = np.isfinite(col_a) & np.isfinite(col_b)
                ks[str(k)] = _boot_mean_ci(col_a[m] - col_b[m], seed=C.BOOTSTRAP_SEED)
            pr[bk] = ks
        paired[f"{name}_minus_mean_drift"] = pr

    # v6 paired horizon-mean (k ≤ READOUT_K_MAX) reads over the shared contexts:
    # H7 = ctx-roll − direct-c; H6 rollout companion = b2_film − b1_grad rolls.
    def _paired_hm(name_a: str, name_b: str) -> dict:
        out: dict = {}
        common = [
            bk for bk in scored[name_a]["per_ctx_skill"] if bk in scored[name_b]["per_ctx_skill"]
        ]
        for bk in common:
            hm_a = _horizon_mean_perctx(
                np.array(scored[name_a]["per_ctx_skill"][bk]), C.READOUT_K_MAX
            )
            hm_b = _horizon_mean_perctx(
                np.array(scored[name_b]["per_ctx_skill"][bk]), C.READOUT_K_MAX
            )
            m = np.isfinite(hm_a) & np.isfinite(hm_b)
            res = _boot_mean_ci(hm_a[m] - hm_b[m], n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED)
            res["excludes_zero"] = bool(res["lo"] > 0.0 or res["hi"] < 0.0)
            # v6 item (2), H7 side: pooled-recompute sensitivity companion +
            # the PRE-clamp denominator quantiles (the registered near-zero-
            # tail read; a floor/trim rule fires only if this tail shows one).
            # skill_a − skill_b = (sse_b − sse_a) / den at each (ctx, k), so
            # pooling replaces the per-context mean with the ratio of sums per
            # k, then averages over valid horizons k ≤ READOUT_K_MAX (the same
            # k-aggregation as the per-context horizon-mean). The denominator
            # is variant-independent (see _score_variant), so name_a's is used.
            sse_a = np.array(scored[name_a]["per_ctx_sse"][bk])[:, : C.READOUT_K_MAX]
            sse_b = np.array(scored[name_b]["per_ctx_sse"][bk])[:, : C.READOUT_K_MAX]
            den = np.array(scored[name_a]["per_ctx_den"][bk])[:, : C.READOUT_K_MAX]
            mk = np.isfinite(sse_a) & np.isfinite(sse_b)  # (ctx, k) scored on BOTH sides
            num_k = (np.where(mk, sse_b, 0.0) - np.where(mk, sse_a, 0.0)).sum(axis=0)
            den_k = np.where(mk, den, 0.0).sum(axis=0)
            kvalid = mk.any(axis=0) & (den_k > 0)
            res["pooled_recompute_delta"] = (
                float(np.mean(num_k[kvalid] / den_k[kvalid])) if kvalid.any() else float("nan")
            )
            dvals = den[mk]
            res["den_quantiles_preclamp"] = (
                {q: float(np.quantile(dvals, float(q))) for q in ("0.0", "0.05", "0.5")}
                if dvals.size
                else None
            )
            res["den_frac_below_clamp"] = float(np.mean(dvals < 1e-6)) if dvals.size else None
            out[bk] = res
        return out

    h7 = {}
    if "direct_c" in scored:
        h7["ctx_roll_minus_direct_c"] = _paired_hm("ridge_ctx_boundary_first", "direct_c")
        lstar_bks = [str(b) for b in C.READOUT_BLOCKS]
        clear = {
            bk: v["excludes_zero"] and v["mean"] > 0
            for bk, v in h7["ctx_roll_minus_direct_c"].items()
            if bk in lstar_bks
        }
        h7["lstar_positive_clear_count"] = int(sum(clear.values()))
        h7["lstar_blocks"] = lstar_bks
    h6_roll = {}
    if "film_roll" in scored and "b1_grad_roll" in scored:
        h6_roll["film_minus_b1_grad"] = _paired_hm("film_roll", "b1_grad_roll")
    for other in ("lowrank", "mixture"):
        if f"{other}_roll" in scored and "b1_grad_roll" in scored:
            h6_roll[f"{other}_minus_b1_grad"] = _paired_hm(f"{other}_roll", "b1_grad_roll")

    if out_npz is not None:
        payload = {
            f"skill__{name}__{bk}": np.asarray(v, dtype=np.float32)
            for name, sc in scored.items()
            for bk, v in sc["per_ctx_skill"].items()
        }
        # v6 item (2): per-(ctx, k) SSE per variant + the variant-independent
        # PRE-clamp denominator per block persist beside the skill vectors.
        payload.update(
            {
                f"sse__{name}__{bk}": np.asarray(v, dtype=np.float32)
                for name, sc in scored.items()
                for bk, v in sc["per_ctx_sse"].items()
            }
        )
        den_by_bk: dict[str, list] = {}
        for sc in scored.values():
            for bk, v in sc["per_ctx_den"].items():
                den_by_bk.setdefault(bk, v)
        payload.update(
            {f"den__{bk}": np.asarray(v, dtype=np.float32) for bk, v in den_by_bk.items()}
        )
        np.savez(
            out_npz,
            **payload,
            ctx_ids=np.array([store["ctx_ids"][i] for i in ctxs]),
            kcap=kcap,
        )
    for sc in scored.values():
        sc.pop("per_ctx_skill", None)
        sc.pop("per_ctx_sse", None)
        sc.pop("per_ctx_den", None)
    return {
        "corpus": corpus,
        "n_ctx": len(ctxs),
        "k_max": k_max,
        "variants": scored,
        "variant_rows": {
            name: [C.row_to_block_key(r) for r in rs] for name, rs in row_sets.items()
        },
        "gru_exploratory": gru_curves,
        "paired_vs_mean_drift": paired,
        "h7_paired": h7,
        "h6_rollout_companion": h6_roll,
    }


def _score_variant(states, h, rows, ctxs, Tpos, kcap, k_max, device) -> dict:
    """Per (row, k): pooled r2_id, mean cosine, per-ctx skill + CI vs frozen(=0).

    v6 item (2), H7/rollout side: per-(context, k) SSE and PRE-clamp
    identity-denominator contributions are returned alongside the skill
    vectors (``per_ctx_sse`` / ``per_ctx_den``; the caller persists them to
    the per-context npz and derives the pooled-recompute sensitivity
    companion + the denominator near-zero-tail read in ``_paired_hm``). The
    1e-6 clamp stays on the skill ratio itself; the persisted denominators
    are pre-clamp. The denominator ‖h_{T+k} − h_T‖² depends only on
    (context, k, row) — never on the variant's predictions — so it is
    identical across variants at a shared row.
    """
    Tpos_t = torch.tensor(np.asarray(Tpos))
    per_ctx = {C.row_to_block_key(r): np.full((len(ctxs), k_max), np.nan) for r in rows}
    per_sse = {C.row_to_block_key(r): np.full((len(ctxs), k_max), np.nan) for r in rows}
    per_den = {C.row_to_block_key(r): np.full((len(ctxs), k_max), np.nan) for r in rows}
    pooled = {C.row_to_block_key(r): [] for r in rows}
    cosm = {C.row_to_block_key(r): [] for r in rows}
    ci = {C.row_to_block_key(r): [] for r in rows}
    seed_states = torch.stack([h[r, Tpos_t, :].to(torch.float32) for r in rows]).to(device)
    for k in range(1, k_max + 1):
        valid = torch.tensor(np.asarray(kcap) >= k)
        if int(valid.sum()) == 0:
            for bk in pooled:
                pooled[bk].append(float("nan"))
                cosm[bk].append(float("nan"))
                ci[bk].append(
                    {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
                )
            continue
        tgt_pos = Tpos_t[valid] + k
        truth = torch.stack([h[r, tgt_pos, :].to(torch.float32) for r in rows]).to(device)
        valid_dev = valid.to(seed_states.device)
        pred = states[k - 1][:, valid_dev, :]
        base = seed_states[:, valid_dev, :]
        num = ((pred - truth) ** 2).sum(-1)  # (L, n_valid)
        den_raw = ((truth - base) ** 2).sum(-1)  # PRE-clamp (persisted)
        den = den_raw.clamp(min=1e-6)
        skill = (1.0 - num / den).cpu().numpy()
        num_np = num.cpu().numpy()
        den_np = den_raw.cpu().numpy()
        cos = torch.nn.functional.cosine_similarity(pred, truth, dim=-1).cpu().numpy()
        vidx = np.nonzero(valid.numpy())[0]
        for li, r in enumerate(rows):
            bk = C.row_to_block_key(r)
            per_ctx[bk][vidx, k - 1] = skill[li]
            per_sse[bk][vidx, k - 1] = num_np[li]
            per_den[bk][vidx, k - 1] = den_np[li]
            pooled[bk].append(float(1.0 - num[li].sum().item() / den[li].sum().item()))
            cosm[bk].append(float(cos[li].mean()))
            ci[bk].append(_boot_mean_ci(skill[li], seed=C.BOOTSTRAP_SEED))
    return {
        "pooled_r2_id": pooled,
        "mean_cosine": cosm,
        "skill_mean_ci": ci,
        "per_ctx_skill": {bk: v.tolist() for bk, v in per_ctx.items()},
        "per_ctx_sse": {bk: v.tolist() for bk, v in per_sse.items()},
        "per_ctx_den": {bk: v.tolist() for bk, v in per_den.items()},
        "n_valid_per_k": [int((np.asarray(kcap) >= k).sum()) for k in range(1, k_max + 1)],
    }


# ── DV3 read-out benchmark ────────────────────────────────────────────────────


def rig_validation(mat, r_b, trait, step0, tol: float = 0.10) -> dict:
    """Reproduce #779's raw context-read r at ℓ* against step0_oracle.

    The #841 pairing-integrity pattern: the raw ⟨cx_last[ℓ*], r_B[ℓ*]⟩ read
    scored with the verbatim protocol must sit within ``tol`` of the cached
    oracle row per mode; on failure DV3 is SKIPPED (kill criterion (c)).
    """
    b = C.PRIMARY_LSTAR[trait]
    x = mat["traj"][:, b, :] @ r_b[b]
    res, ok = {}, True
    for mode in MODES:
        mine = method_metrics(x, mat, n_boot=100, seed=0)[mode]["point"]
        ref = step0.get(trait, {}).get("per_layer_mode", {}).get(f"L{b}_{mode}")
        # step0_oracle rows: {"pv_raw_r": raw ⟨cx, r_B⟩ read, "oracle_r": ...}.
        ref_pt = ref.get("pv_raw_r", ref.get("point")) if isinstance(ref, dict) else ref
        drift = abs(mine - ref_pt) if (ref_pt is not None and np.isfinite(mine)) else None
        res[mode] = {"mine": mine, "oracle": ref_pt, "abs_drift": drift}
        if drift is None or drift > tol:
            ok = False
    res["pass"] = ok
    res["tol"] = tol
    return res


def readout_phase(  # noqa: C901 — one block per DV3 mode; the mode enumeration IS the benchmark
    ridge, args, device, eval_store, out_dir: Path, cond_blob=None, direct_dir=None
) -> dict:
    rows = ridge["rows"]
    step0 = C.load_step0()
    out: dict = {"traits": {}, "decision_statistic": "horizon_mean_k1_32_at_primary_lstar"}
    npz_payload: dict = {}
    k_max = C.READOUT_K_MAX
    for trait in args.traits:
        r_b = C.load_rb(trait)
        cells = C.load_eval_cells(trait)
        mat = C.build_eval_traj_matrix(cells)
        rig = rig_validation(mat, r_b, trait, step0)
        tro: dict = {"rig_validation": rig, "n_units": len(mat["y"])}
        if not rig["pass"]:
            logger.error("[dv3:%s] rig validation FAILED — DV3 SKIPPED for this trait", trait)
            tro["skipped"] = "rig_validation_failed (kill criterion c)"
            out["traits"][trait] = tro
            continue
        blocks = {"primary": C.PRIMARY_LSTAR[trait], "companion": C.COMPANION_LSTAR[trait]}
        if args.readout_block_override is not None:
            blocks = {"primary": args.readout_block_override}
        for tag, b in blocks.items():
            r = C.block_to_row(b)
            if r not in rows:
                tro[tag] = {"skipped": f"block {b} not fitted (rows={rows})"}
                continue
            seed = torch.from_numpy(mat["traj"][:, b, :]).to(device).unsqueeze(0)
            bmap = _stack_maps({r: ridge["boundary"]["ctx"][r]}, [r], device)
            amap = _stack_maps({r: ridge["answer"]["ctx"][r]}, [r], device)
            states = M.roll_states_ridge(seed, bmap, amap, k_max)
            rb_t = torch.from_numpy(r_b[b]).to(device, torch.float32)
            proj = torch.stack([s[0] @ rb_t for s in states], dim=1).cpu().numpy()  # (N, k)
            frozen = (torch.from_numpy(mat["traj"][:, b, :]).to(device) @ rb_t).cpu().numpy()
            hm = proj[:, :k_max].mean(axis=1)

            def _delta_vs_frozen(vec: np.ndarray, *, mat=mat, frozen=frozen) -> dict:
                delta = {}
                for mode in MODES:
                    cx_a, cy = C.group_by_condition(vec, mat["y"], mat["cond"], mat["mode"], mode)
                    cx_b, _ = C.group_by_condition(frozen, mat["y"], mat["cond"], mat["mode"], mode)
                    delta[mode] = bootstrap_delta_ci(
                        cx_a, cx_b, cy, n_boot=args.n_boot, seed=C.BOOTSTRAP_SEED
                    )
                return delta

            reads = {
                "frozen": method_metrics(frozen, mat, n_boot=args.n_boot, seed=0),
                "horizon_mean": method_metrics(hm, mat, n_boot=args.n_boot, seed=0),
                "per_k": {
                    str(k): method_metrics(proj[:, k - 1], mat, n_boot=100, seed=0)
                    for k in range(1, k_max + 1)
                },
            }
            reads["delta_horizon_mean_minus_frozen"] = _delta_vs_frozen(hm)
            ref = step0.get(trait, {}).get("per_layer_mode", {})
            reads["pv_direct_reference"] = {m: ref.get(f"L{b}_{m}") for m in MODES}

            # ── v6 modes, SAME statistic + paired bootstrap vs the frozen read.
            # DV3 conditioning uses c = the cached pass_a cx state at ℓ* —
            # identical to the rollout seed (h_{l,T} ≡ c), so no new eval input.
            extra_hm: dict[str, np.ndarray] = {}
            if r in ridge.get("b1_answer", {}):
                b1st = _stack_maps({r: ridge["b1_answer"][r]}, [r], device)
                st = M.roll_states_b1_ridge(seed, bmap, b1st, k_max)
                p = torch.stack([s[0] @ rb_t for s in st], dim=1).cpu().numpy()
                extra_hm["rolled_b1_ridge"] = p[:, :k_max].mean(axis=1)
            for form, per_row in ((cond_blob or {}).get("forms") or {}).items():
                if r not in per_row:
                    continue
                cstack = M.stack_conditioned_params(per_row, [r], device)
                st = M.roll_states_conditioned(seed, bmap, cstack, k_max)
                p = torch.stack([s[0] @ rb_t for s in st], dim=1).cpu().numpy()
                extra_hm[f"rolled_{form}"] = p[:, :k_max].mean(axis=1)
            if direct_dir is not None:
                dr = _load_direct_row(direct_dir, r)
                if dr is not None:
                    st = M.predict_direct_horizons_row(dr, seed[0], seed[0], k_max, device=device)
                    p = torch.stack([s @ rb_t for s in st], dim=1).cpu().numpy()
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        extra_hm["direct_c"] = np.nanmean(p[:, :k_max], axis=1)
                    del dr
            for name, vec in extra_hm.items():
                reads[name] = {
                    "horizon_mean": method_metrics(vec, mat, n_boot=args.n_boot, seed=0),
                    "delta_vs_frozen": _delta_vs_frozen(vec),
                }
                if name == "direct_c":
                    reads[name]["note"] = (
                        "horizon-mean = the #779 context→answer-profile nested baseline"
                    )
                npz_payload[f"hm__{trait}__L{b}__{name}"] = np.asarray(vec, dtype=np.float32)
            tro[tag] = {"block": b, **reads}
            npz_payload[f"proj__{trait}__L{b}"] = proj.astype(np.float32)
            npz_payload[f"frozen__{trait}__L{b}"] = frozen.astype(np.float32)
            if tag == "primary":
                tro["_primary_unit_vectors"] = {
                    "rolled_horizon_mean": hm,
                    "frozen": frozen,
                    **extra_hm,
                }
        npz_payload[f"y__{trait}"] = mat["y"]
        npz_payload[f"cond__{trait}"] = mat["cond"]
        npz_payload[f"qi__{trait}"] = mat["qi"]
        npz_payload[f"mode__{trait}"] = np.array([str(m) for m in mat["mode"]])

        # panel-restricted companions + true-answer ceiling (WARN 2 / DV3 bound)
        if eval_store is not None:
            tro["restricted_panel"] = restricted_panel(
                trait,
                mat,
                r_b,
                eval_store,
                args,
                reads_vectors=tro.pop("_primary_unit_vectors", {}),
            )
        else:
            tro.pop("_primary_unit_vectors", None)
        out["traits"][trait] = tro
    np.savez(out_dir / "readout_projections.npz", **npz_payload)
    return out


def restricted_panel(trait, mat, r_b, eval_store, args, reads_vectors: dict) -> dict:
    """Captured-subset-restricted reads: rolled/frozen (+ v6 modes) AND the
    true-answer ceiling on the SAME unit panel.

    The r1 code-review MAJOR fix (plan v3 item 6, consistency WARN 2): HERO-3
    compares reads against the true-answer ceiling, which only exists on the
    captured 16-q subset — the rolled/frozen companions must be recomputed on
    that SAME panel or the comparison is cross-panel confounded.
    ``reads_vectors`` = per-unit vectors from the PRIMARY ℓ* reads
    ({"rolled_horizon_mean", "frozen", "rolled_b1_ridge", ..., "direct_c"}).
    """
    es = eval_store
    meta = es["meta"]
    keys = {}
    for ii, ci in enumerate(es["ctx_ids"]):
        m = meta[ci]
        if m.get("trait") == trait:
            keys[(m["cond_id"], int(m["qi"]))] = ii
    b = C.PRIMARY_LSTAR[trait]
    r = C.block_to_row(b)
    if r >= es["h"].shape[0] and args.readout_block_override is None:
        return {"skipped": "block not captured in eval store"}
    if args.readout_block_override is not None:
        b = args.readout_block_override
        r = C.block_to_row(b)
    unit_sel = [
        j
        for j in range(len(mat["y"]))
        if (mat["cond_ids"][mat["cond"][j]], int(mat["qi"][j])) in keys
    ]
    if not unit_sel:
        return {"skipped": "no overlap between DV3 units and captured eval subset"}
    sub = {
        k: (np.asarray(mat[k])[unit_sel] if k in ("y", "cond", "mode", "qi") else mat[k])
        for k in ("y", "cond", "mode", "qi", "cond_ids")
    }
    # true-answer ceiling: TRUE h_{ℓ*, T+k} from the eval store, k=1..32 horizon-mean.
    k_max = C.READOUT_K_MAX
    projs, mask_any = [], []
    for j in unit_sel:
        ii = keys[(mat["cond_ids"][mat["cond"][j]], int(mat["qi"][j]))]
        lo = int(es["pos_lo"][ii])
        P, wsi, npos = int(es["prompt_len"][ii]), int(es["window_start"][ii]), int(es["n_pos"][ii])
        tw = P - 1 - wsi
        upto = min(k_max, npos - 1 - tw)
        if upto < 1:
            projs.append(np.nan)
            mask_any.append(False)
            continue
        hs = es["h"][r, lo + tw + 1 : lo + tw + 1 + upto, :].to(torch.float32).numpy()
        projs.append(float((hs @ r_b[b]).mean()))
        mask_any.append(True)
    ceiling = np.array(projs)
    res = {
        "n_units": len(unit_sel),
        "block": b,
        "true_answer_ceiling_horizon_mean": method_metrics(
            ceiling, sub, n_boot=args.n_boot, seed=0
        ),
        "n_with_answer_window": int(np.sum(mask_any)),
    }
    # the restricted rolled/frozen (+ v6 mode) companions — same-panel reads
    for name, vec in reads_vectors.items():
        res[name] = method_metrics(np.asarray(vec)[unit_sel], sub, n_boot=args.n_boot, seed=0)
    return res


# ── DV4 transfer ──────────────────────────────────────────────────────────────


def transfer_phase(
    eval_store, ridge, mlp_blob, args, device, drift, out_dir, cond_blob=None, direct_dir=None
) -> dict:
    tr_all = transition_indices(eval_store, np.arange(len(eval_store["ctx_ids"])))
    rows = ridge["rows"]
    out: dict = {"n_ctx": len(eval_store["ctx_ids"]), "single_step": {}}
    idx = tr_all["answer"]
    if idx.numel() > 0:
        for r in rows:
            bk = C.row_to_block_key(r)
            delta = (
                _gather_X(eval_store["h"], r, idx + 1, "ctx")
                - _gather_X(eval_store["h"], r, idx, "ctx")
            ).numpy()
            tm = ridge["delta_train_mean"][r].numpy()
            cell = {}
            for arm in ("ctx", "tok", "emb"):
                X = _gather_X(eval_store["h"], r, idx, arm)
                pred = M.ridge_predict(ridge["answer"][arm][r], X).numpy()
                cell[f"ridge_{arm}"] = {
                    "r2_id": M.identity_relative_r2(pred, delta),
                    "r2_meancentered": M.mean_centered_r2(pred, delta, tm),
                }
            # v4 exploratory transfer: the closed-form b1 cell rides along
            if r in ridge.get("b1_answer", {}):
                Xh = _gather_X(eval_store["h"], r, idx, "ctx")
                Xc = eval_store["h"][r, tr_all["answer_T"], :].to("cpu", torch.float32)
                pred = M.ridge_predict(ridge["b1_answer"][r], torch.cat([Xh, Xc], dim=1)).numpy()
                cell["b1_ridge"] = {
                    "r2_id": M.identity_relative_r2(pred, delta),
                    "r2_meancentered": M.mean_centered_r2(pred, delta, tm),
                }
            out["single_step"][bk] = cell
    out["rollout"] = rollout_phase(
        eval_store,
        ridge,
        mlp_blob,
        None,
        np.arange(len(eval_store["ctx_ids"])),
        args,
        device,
        corpus="eval_subset",
        drift=drift,
        out_npz=out_dir / "transfer_percontext.npz",
        cond_blob=cond_blob,
        direct_dir=direct_dir,
    )
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #922 evals.")
    ap.add_argument("--store", type=Path, default=Path("/workspace/issue922_store"))
    ap.add_argument("--maps", type=Path, default=Path("/workspace/issue922_maps"))
    ap.add_argument("--out", type=Path, default=Path("eval_results/issue_922"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split-seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--n-shuffle", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--conditioned-rollouts",
        action="store_true",
        help="v6: score b1/b2 conditioned arms (requires maps_conditioned.pt / b1 in maps_ridge)",
    )
    ap.add_argument(
        "--direct-predictions",
        action="store_true",
        help="v6: score arm-c direct per-horizon predictions (requires maps/direct/)",
    )
    ap.add_argument(
        "--readout-block-override",
        type=int,
        default=None,
        help="override ALL traits' read-out block (stub-model smokes only)",
    )
    args = ap.parse_args()
    device = _resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    store = C.load_store(args.store, "lmsys")
    ridge = _load_ridge(args.maps)
    mlp_blob = (
        torch.load(args.maps / "maps_mlp.pt", weights_only=False)
        if (args.maps / "maps_mlp.pt").exists()
        else None
    )
    gru_blob = (
        torch.load(args.maps / "maps_gru.pt", weights_only=False)
        if (args.maps / "maps_gru.pt").exists()
        else None
    )
    cond_blob = None
    if args.conditioned_rollouts:
        cpath = args.maps / "maps_conditioned.pt"
        assert cpath.exists() or ridge["b1_answer"], (
            "--conditioned-rollouts needs maps_conditioned.pt or b1_answer maps"
        )
        if cpath.exists():
            cond_blob = torch.load(cpath, weights_only=False)
    direct_dir = None
    if args.direct_predictions:
        direct_dir = args.maps / "direct"
        assert direct_dir.is_dir(), f"--direct-predictions needs {direct_dir}"
    n_ctx = len(store["ctx_ids"])
    split = C.make_split(n_ctx, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=args.split_seed)
    tr = {name: transition_indices(store, split[name]) for name in ("fit", "val", "test")}
    items = C.load_lmsys_items(n_contexts=n_ctx)
    rows = ridge["rows"]

    # A — DV1 atlas (+ v6 conditioned single-step cells merged into the atlas)
    atlas = atlas_phase(store, ridge, mlp_blob, split, tr, args, device)
    atlas["dup_report"] = C.lmsys_dup_report(items, split)
    atlas["ns"] = {
        name: {seg: int(tr[name][seg].numel()) for seg in ("answer", "boundary")} for name in tr
    }
    cond = None
    if args.conditioned_rollouts or args.direct_predictions:
        cond = conditioned_phase(store, ridge, cond_blob, args.maps, split, tr, args, device)
        C.write_json_atomic(
            args.out / "conditioned_arms.json",
            {
                **cond,
                "metadata": C.reproducibility_metadata(
                    {"script": "issue922_eval", "dv": "conditioned_arms"}
                ),
            },
        )
        logger.info("[cond] conditioned_arms written (%.1fs elapsed)", time.time() - t0)
        for name, by_bk in cond.get("cells", {}).items():
            if name == "ridge_ctx_ref":
                continue
            for bk, met in by_bk.items():
                atlas["cells"].setdefault(f"{bk}|answer", {})[name] = {"raw": met}
    if 0 in rows:  # the r1 layer-0 blocker: emb cells MUST exist in the atlas
        assert any(k.startswith("emb|") for k in atlas["cells"]), "emb|* atlas cells missing"
    C.write_json_atomic(
        args.out / "stage0_position_atlas.json",
        {**atlas, "metadata": C.reproducibility_metadata({"script": "issue922_eval", "dv": "DV1"})},
    )
    logger.info("[dv1] atlas written (%.1fs elapsed)", time.time() - t0)

    # B — DV2 rollout (v6: + b1/b2 conditioned rolls, direct-c, H7 paired read)
    drift = drift_means(store, ridge, tr, rows, C.ROLLOUT_K_MAX)
    roll = rollout_phase(
        store,
        ridge,
        mlp_blob,
        gru_blob,
        split["test"],
        args,
        device,
        corpus="lmsys_test",
        drift=drift,
        out_npz=args.out / "rollout_skill_percontext.npz",
        cond_blob=cond_blob,
        direct_dir=direct_dir,
    )
    C.write_json_atomic(
        args.out / "rollout_skill.json",
        {**roll, "metadata": C.reproducibility_metadata({"script": "issue922_eval", "dv": "DV2"})},
    )
    logger.info("[dv2] rollout written (%.1fs elapsed)", time.time() - t0)

    # C — DV3 read-out (+ true-answer ceiling via the eval store when present)
    eval_store = None
    if (args.store / "eval_subset").is_dir():
        eval_store = C.load_store(args.store, "eval_subset")
    dv3 = readout_phase(
        ridge, args, device, eval_store, args.out, cond_blob=cond_blob, direct_dir=direct_dir
    )
    C.write_json_atomic(
        args.out / "readout_benchmark.json",
        {**dv3, "metadata": C.reproducibility_metadata({"script": "issue922_eval", "dv": "DV3"})},
    )
    logger.info("[dv3] readout written (%.1fs elapsed)", time.time() - t0)

    # D — DV4 transfer
    if eval_store is not None:
        dv4 = transfer_phase(
            eval_store,
            ridge,
            mlp_blob,
            args,
            device,
            drift,
            args.out,
            cond_blob=cond_blob,
            direct_dir=direct_dir,
        )
        C.write_json_atomic(
            args.out / "transfer_eval.json",
            {
                **dv4,
                "metadata": C.reproducibility_metadata({"script": "issue922_eval", "dv": "DV4"}),
            },
        )
        logger.info("[dv4] transfer written")
    else:
        logger.warning("[dv4] eval_subset store absent — transfer SKIPPED")
    logger.info("DONE evals in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
