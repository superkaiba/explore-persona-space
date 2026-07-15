#!/usr/bin/env python
"""#825 turn-dynamics-allturns-5000 P4 fit driver (plan v24 §4 P4, §6, §7 G-C).

Phased (each phase writes an atomic part JSON under --parts-dir; `assemble`
merges them + the bridge/diagnostics into results.json):

  gc         G-C parity: armR_logged per-turn L19 curve refit on the EXACT
             round-10 conversation-id set (id assert MECHANIZED before
             fitting), round-10 fold recipe (_folds_for_turn), PASS iff the
             round-10 r2 lies inside the refit's conversation-bootstrap CI at
             every overlapping turn. FAIL blocks the headline (pipeline
             defect, never composition — plan §7).
  cells      per-turn ridge fits (ctx + pfx mapping arms, layers {14,18,19},
             L19 200-draw shuffle nulls, ONE conv->fold partition per panel)
             + the pooled-all-turns fit, per (tag, model).
  transfer   KxK cross-turn transfer matrix at L19 ctx (fit turn i on train
             folds -> predict turn j test fold; SAME partition => grouped
             held-out; retained fraction vs the diagonal).
  operators  per-(turn, fold) primal betas at L19 (PERSISTED fp16 ->
             --betas-dir), operator battery per turn pair (raw cosine over
             disjoint fold pairs; Procrustes / general-linear / principal
             angles on the pre-registered fold pair (0,1)); the within-turn
             i==j entries ARE the F3 estimation-noise ceiling.
  reach      turn-1 context -> turn-k answer: ridge (ambient, full n, L19
             null per k) + batched multihead MLP (vectorized_mlp_skill;
             PCA-48 targets, input-PCA + conv-subsample sizing knobs — the
             plan's ~200-cell budget) with shuffled-Y MLP null draws.
  assemble   merge parts + H4 bridge (seed-intersection paired conv
             bootstrap) + rollout degeneracy diagnostics + drop accounting +
             gate records -> results.json.

Device: --device (default cuda when available) threads through the
source-fixed `_fit_cv(device=)` + `_dual_perm_null(device=)` and the torch
transfer/operator cores. No conversation text is read or printed here —
tensors, ids, counts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue825_crossmodel_map_transfer import (  # noqa: E402
    LAMBDAS,
    _prep_fold,
    _ridge_predict_cached,
)
from issue825_onpolicy_turn_depth_fit import _cluster_bootstrap  # noqa: E402
from issue825_turn_depth_map import (  # noqa: E402
    HEADLINE_LAYER,
    N_DRAWS,
    NULL_MIN_N,
    NULL_SEED,
    _dual_perm_null,
    _folds_for_turn,
)
from issue825_turndyn_harvest import (  # noqa: E402
    read_jsonl_stem,
    source_counts,
    stratified_subsample_ids,
)
from issue1092_fit_grid import FOLD_SEED, _fit_cv, _r2  # noqa: E402

logger = logging.getLogger("i825_turndyn_fit")

BASE_LAYERS = (14, 18, 19)
N_FOLDS = 6
BOOT_SEED = 8250
N_BOOT = 1000
DROP_KILL_RATE = 0.20
MIN_FIT_N = 30  # per-turn cell floor at production scale (smoke overrides)
BRIDGE_BAND = 0.10  # pre-registered H4 comparability band (plan §6)
OPERATOR_HEAVY_FOLD_PAIR = (0, 5)  # pre-registered disjoint fold pair for heavy ops
X_KINDS = {"ctx": "context_k", "pfx": "prefix_k"}
ANSWER_KIND = {
    "armR_own": "answer_own_t1",
    "armR_logged": "answer_logged_t1",
    "armG": "answer_own_t1",
}


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# capture loading (turndyn shard/group layout)
# ---------------------------------------------------------------------------


def _load_capture(
    capture_root: Path, tag: str, model: str, kinds: list[str], layers: tuple[int, ...]
) -> tuple[list[dict], dict[str, dict[int, np.ndarray]]]:
    """Concatenate all (shard, group) pieces for (tag, model) in sorted order."""
    root = capture_root / tag / model
    shard_dirs = sorted(p for p in root.glob("shard*") if p.is_dir())
    assert shard_dirs, f"[load] no capture shards under {root}"
    rows: list[dict] = []
    arrays: dict[str, dict[int, list[np.ndarray]]] = {k: {la: [] for la in layers} for k in kinds}
    for sd in shard_dirs:
        idx_paths = sorted(sd.glob("row_index_shard*.jsonl"))
        assert idx_paths, f"[load] no row-index shards under {sd}"
        for ip in idx_paths:
            gi = ip.stem.split("shard")[-1]
            n_before = len(rows)
            with ip.open(encoding="utf-8") as f:
                for line in f:  # file iteration, never splitlines (gotchas.md)
                    line = line.strip("\n")
                    if line:
                        rows.append(json.loads(line))
            n_new = len(rows) - n_before
            for kind in kinds:
                for layer in layers:
                    p = sd / f"{kind}_L{layer:02d}_shard{gi}.npy"
                    arr = np.load(p).astype(np.float64)
                    assert arr.shape[0] == n_new, (str(p), arr.shape, n_new)
                    arrays[kind][layer].append(arr)
    out = {
        k: {la: np.concatenate(v, axis=0) if v else np.empty((0, 0)) for la, v in d.items()}
        for k, d in arrays.items()
    }
    return rows, out


def _aggregate_capture_reports(capture_root: Path, tag: str, model: str) -> dict:
    root = capture_root / tag / model
    reports = sorted(root.glob("shard*/capture_report.json")) or sorted(
        root.glob("capture_report.json")
    )
    agg: dict = {"n_reports": len(reports), "reports": []}
    for p in reports:
        with open(p) as f:
            agg["reports"].append(json.load(f))
    return agg


def _conv_fold_map(conv_ids: list[str], n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> dict:
    """ONE conv->fold partition per panel (mirrors _folds_from_manifest)."""
    uniq = sorted(set(conv_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {}
    for i in range(n_folds):
        for c in uniq[i::n_folds]:
            fold_of[c] = i
    return fold_of


def _fold_map_hash(fold_of: dict) -> str:
    return hashlib.sha256(
        "\n".join(f"{c}:{fold_of[c]}" for c in sorted(fold_of)).encode()
    ).hexdigest()


def _cell_folds(convs: list[str], fold_of: dict) -> list[np.ndarray] | None:
    folds = [
        np.asarray([i for i, c in enumerate(convs) if fold_of[c] == f], dtype=np.int64)
        for f in range(N_FOLDS)
    ]
    folds = [f for f in folds if f.size]
    if len(folds) < 2 or any(f.size >= len(convs) for f in folds):
        return None
    return folds


def _write_part(parts_dir: Path, name: str, payload: dict) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    tmp = parts_dir / f"{name}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, parts_dir / f"{name}.json")
    logger.info("[part] wrote %s", parts_dir / f"{name}.json")


def _cell_r2_bootstrap(
    y: np.ndarray, pred: np.ndarray, convs: list[str], n_boot: int, seed: int
) -> tuple[float, float]:
    """Conversation-bootstrap 95% CI of one cell's pooled R2 (frozen preds)."""
    rng = np.random.default_rng(seed)
    uniq = sorted(set(convs))
    pos = {c: i for i, c in enumerate(uniq)}
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot).astype(
        np.float64
    )
    cols = np.asarray([pos[c] for c in convs], dtype=np.int64)
    m = counts[:, cols]  # (n_boot, n_rows)
    r_row = ((y - pred) ** 2).sum(axis=1)
    q_row = (y**2).sum(axis=1)
    n_b = m.sum(axis=1)
    s_b = m @ y
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_tot = m @ q_row - (s_b**2).sum(axis=1) / np.where(n_b > 0, n_b, np.nan)
        r2_b = 1.0 - (m @ r_row) / np.where(ss_tot > 0, ss_tot, np.nan)
    return float(np.nanpercentile(r2_b, 2.5)), float(np.nanpercentile(r2_b, 97.5))


def _rows_by_turn(rows: list[dict]) -> dict[int, np.ndarray]:
    by_t: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_t.setdefault(int(r["turn"]), []).append(i)
    return {t: np.asarray(v, dtype=np.int64) for t, v in sorted(by_t.items())}


def _assert_drop_kill(capture_root: Path, tag: str, model: str) -> dict:
    """Plan §6 kill line: aggregate own-capture drop rate > 20% HALTs the fit."""
    agg = _aggregate_capture_reports(capture_root, tag, model)
    tot = sum(r.get("n_total_pairs", 0) for r in agg["reports"])
    kept = sum(r.get("n_kept", 0) for r in agg["reports"])
    if tot:
        rate = (tot - kept) / tot
        if rate > DROP_KILL_RATE:
            raise SystemExit(
                f"[kill] {tag}/{model}: drop rate {100 * rate:.1f}% > "
                f"{100 * DROP_KILL_RATE:.0f}% — generation-recipe problem, not a finding"
            )
        agg["aggregate_drop_rate"] = rate
    return agg


# ---------------------------------------------------------------------------
# gc: G-C round-10 parity refit (plan §7)
# ---------------------------------------------------------------------------


def run_gc(args: argparse.Namespace) -> None:
    rows, arrays = _load_capture(
        Path(args.capture_root),
        "gc_logged",
        args.model,
        ["context_k", "answer_logged_t1"],
        (HEADLINE_LAYER,),
    )
    gc_panel = read_jsonl_stem(Path(args.panel_dir), "gc_panel")
    panel_ids = sorted({str(r["conv_id"]) for r in gc_panel})
    captured_ids = sorted({str(r["conv_id"]) for r in rows})
    # MECHANIZED id assert (plan §7): the comparator conv-id set must equal the
    # round-10 panel id set (the harvest digest-asserted THAT set against the
    # round-10 drop report — assumption 11) BEFORE any fitting.
    if captured_ids != panel_ids:
        missing = sorted(set(panel_ids) - set(captured_ids))[:10]
        extra = sorted(set(captured_ids) - set(panel_ids))[:10]
        raise SystemExit(
            f"[G-C] conv-id set mismatch: captured {len(captured_ids)} vs panel "
            f"{len(panel_ids)} (missing {missing}, extra {extra}) — pipeline defect"
        )
    if args.emit_r10_ref:
        # SMOKE tooling (round-10 --emit-banked-ref pattern): compute the refit
        # curve and WRITE it in the round-10 results.json SHAPE, so the smoke's
        # gc run exercises the id assert + refit + bootstrap + verdict against
        # a self-consistent reference. Never used in production.
        X = arrays["context_k"][HEADLINE_LAYER]
        Y = arrays["answer_logged_t1"][HEADLINE_LAYER]
        ref: dict[str, dict] = {}
        for t, sel in _rows_by_turn(rows).items():
            if sel.size < 3:
                continue
            folds = _folds_for_turn([rows[i] for i in sel])
            if folds is None:
                continue
            fit = _fit_cv(X[sel], Y[sel], folds, device=args.device)
            ref[str(t)] = {"ctx_logged": {"r2": float(fit["r2"])}, "n": int(sel.size)}
        out_ref = Path(args.r10_json)
        out_ref.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "results": {m: {str(HEADLINE_LAYER): {}} for m in ("instruct", "pretrained")}
        }
        if out_ref.exists():
            with open(out_ref) as f:
                payload = json.load(f)
        payload.setdefault("results", {}).setdefault(args.model, {})[str(HEADLINE_LAYER)] = ref
        payload["description"] = "SMOKE banked reference (--emit-r10-ref)"
        with open(out_ref, "w") as f:
            json.dump(payload, f, indent=1)
        logger.info("[gc] emitted smoke r10 reference -> %s (%d turns)", out_ref, len(ref))
        return
    with open(args.r10_json) as f:
        r10 = json.load(f)
    r10_curve = r10["results"][args.model][str(HEADLINE_LAYER)]
    X = arrays["context_k"][HEADLINE_LAYER]
    Y = arrays["answer_logged_t1"][HEADLINE_LAYER]
    by_turn = _rows_by_turn(rows)
    out: dict[str, dict] = {}
    n_fail = 0
    for t, sel in by_turn.items():
        node = r10_curve.get(str(t), {})
        r10_r2 = node.get("ctx_logged", {}).get("r2") if isinstance(node, dict) else None
        if r10_r2 is None or sel.size < 3:
            continue
        cell_rows = [rows[i] for i in sel]
        folds = _folds_for_turn(cell_rows)  # round-10 per-cell fold recipe VERBATIM
        if folds is None:
            continue
        fit, pred = _fit_cv(X[sel], Y[sel], folds, return_pred=True, device=args.device)
        lo, hi = _cell_r2_bootstrap(
            Y[sel].astype(np.float32),
            pred.astype(np.float32),
            [r["conv_id"] for r in cell_rows],
            args.n_boot,
            BOOT_SEED + t,
        )
        ok = lo <= r10_r2 <= hi
        n_fail += 0 if ok else 1
        out[str(t)] = {
            "n": int(sel.size),
            "r2_refit": float(fit["r2"]),
            "r2_refit_ci": [lo, hi],
            "r2_round10": float(r10_r2),
            "pass": bool(ok),
        }
    verdict = {
        "gate": "G-C",
        "model": args.model,
        "n_convs": len(captured_ids),
        "id_assert": "PASS (captured == harvest gc_panel; harvest digest-asserted vs round-10)",
        "per_turn": out,
        "n_turns": len(out),
        "n_fail": n_fail,
        "pass": n_fail == 0 and len(out) > 0,
    }
    _write_part(Path(args.parts_dir), f"gc_{args.model}", verdict)
    if not verdict["pass"]:
        # FAIL blocks the headline (plan §7) — fail LOUD here; the dispatcher
        # surfaces it before any headline part is consumed.
        raise SystemExit(
            f"[G-C] FAIL for {args.model}: {n_fail}/{len(out)} turns outside the refit "
            f"bootstrap CI — pipeline defect (id-identical panel), headline blocked"
        )
    logger.info("[G-C] PASS %s: %d turns reproduce round-10 within CI", args.model, len(out))


# ---------------------------------------------------------------------------
# cells: per-turn fits + nulls + pooled fit (plan §6 Q1)
# ---------------------------------------------------------------------------


def run_cells(args: argparse.Namespace) -> None:
    tag = args.arm
    kinds = ["prefix_k", "context_k", ANSWER_KIND[tag]]
    if tag == "armR_own":
        _assert_drop_kill(Path(args.capture_root), tag, args.model)
    rows, arrays = _load_capture(Path(args.capture_root), tag, args.model, kinds, BASE_LAYERS)
    fold_of = _conv_fold_map([r["conv_id"] for r in rows])
    by_turn = _rows_by_turn(rows)
    y_all = {la: arrays[ANSWER_KIND[tag]][la] for la in BASE_LAYERS}
    results: dict[str, dict] = {}
    min_n = int(args.min_fit_n)
    for layer in BASE_LAYERS:
        layer_out: dict[str, dict] = {}
        for t, sel in by_turn.items():
            if sel.size < min_n:
                continue
            convs = [rows[i]["conv_id"] for i in sel]
            folds = _cell_folds(convs, fold_of)
            if folds is None:
                continue
            entry: dict = {"n": int(sel.size)}
            for arm_x, kind_x in X_KINDS.items():
                if arm_x == "pfx" and t == 1:
                    entry["pfx"] = {"status": "N/A — structurally degenerate at t1"}
                    continue
                X = arrays[kind_x][layer][sel]
                Y = y_all[layer][sel]
                fit = _fit_cv(X, Y, folds, device=args.device)
                node = {
                    "status": "computed",
                    "r2": float(fit["r2"]),
                    "r2_folds": [float(v) for v in fit["r2_folds"]],
                    "lambda_indices": fit["lambda_indices"],
                    "null_mean": None,
                    "null_hi": None,
                    "null_max": None,
                    "null_n_draws": 0,
                }
                if layer == HEADLINE_LAYER and sel.size >= args.null_min_n:
                    draws = _dual_perm_null(
                        X,
                        Y,
                        folds,
                        fit["lambda_indices"],
                        args.n_draws,
                        NULL_SEED,
                        device=args.device,
                    )
                    finite = draws[np.isfinite(draws)]
                    if finite.size:
                        node["null_mean"] = float(np.mean(finite))
                        node["null_hi"] = float(np.percentile(finite, 97.5))
                        node["null_max"] = float(np.max(finite))  # existence criterion
                        node["null_n_draws"] = int(finite.size)
                entry[arm_x] = node
            layer_out[str(t)] = entry
            logger.info(
                "[cells] %s/%s L%d t%d: n=%d ctx r2=%.4f",
                tag,
                args.model,
                layer,
                t,
                sel.size,
                layer_out[str(t)].get("ctx", {}).get("r2", float("nan")),
            )
        results[str(layer)] = layer_out
    # pooled-all-turns fit at L19 ctx (Simpson's check; persisted)
    pooled: dict = {}
    sel_all = np.arange(len(rows), dtype=np.int64)
    convs_all = [r["conv_id"] for r in rows]
    folds_all = _cell_folds(convs_all, fold_of)
    if folds_all is not None and len(rows) >= min_n:
        X = arrays["context_k"][HEADLINE_LAYER]
        Y = y_all[HEADLINE_LAYER]
        fit, pred = _fit_cv(X, Y, folds_all, return_pred=True, device=args.device)
        lo, hi = _cell_r2_bootstrap(
            Y.astype(np.float32), pred.astype(np.float32), convs_all, args.n_boot, BOOT_SEED
        )
        per_turn_r2 = {
            t: results[str(HEADLINE_LAYER)].get(str(t), {}).get("ctx", {}).get("r2")
            for t in by_turn
        }
        finite = [v for v in per_turn_r2.values() if v is not None]
        pooled = {
            "n": int(sel_all.size),
            "r2": float(fit["r2"]),
            "r2_ci": [lo, hi],
            "r2_folds": [float(v) for v in fit["r2_folds"]],
            "mean_per_turn_r2": float(np.mean(finite)) if finite else None,
            "gap": (float(fit["r2"]) - float(np.mean(finite))) if finite else None,
        }
    part = {
        "arm": tag,
        "model": args.model,
        "n_rows": len(rows),
        "n_convs": len(set(r["conv_id"] for r in rows)),
        "fold_map_sha256": _fold_map_hash(fold_of),
        "per_turn": results,
        "pooled_ctx_L19": pooled,
        "n_per_turn": {str(t): int(s.size) for t, s in by_turn.items()},
        "capture_reports": _aggregate_capture_reports(Path(args.capture_root), tag, args.model),
    }
    _write_part(Path(args.parts_dir), f"cells_{tag}_{args.model}", part)


# ---------------------------------------------------------------------------
# transfer: KxK cross-turn matrix at L19 ctx (plan §6 Q2)
# ---------------------------------------------------------------------------


def _transfer_nulls(
    turns: list[int],
    by_turn: dict[int, np.ndarray],
    fold_labels: dict[int, np.ndarray],
    used_folds: dict[int, list[int]],
    pred_blocks: dict[tuple[int, int], dict[int, np.ndarray]],
    Y: np.ndarray,
    ss_tot: dict[tuple[int, int], float],
    n_draws: int,
    null_min_n: int,
    seed: int,
) -> dict[str, dict]:
    """Per-(i, j) shuffle-within-turn-j permutation nulls, batched (plan §4 P4).

    Predictions are FROZEN from the real transfer fit: a within-turn-j shuffle
    leaves the turn-i training data untouched for i != j, so the permuted-Y
    "re-solve" is exactly frozen-pred scoring against the permuted targets;
    the DIAGONAL null is therefore conditional on the fitted map (the refit
    permutation null for the same cell lives in the cells part). Batched via
    the dual-space/GEMM draw pattern — ONE permutation set per target turn j
    shared across every source i, one (draws, n*P) gather per chunk, and all
    sources reduced through a single ``(draws, n*P) @ (n*P, sources)`` GEMM
    (no serial per-cell draw loops). ``ss_res(perm) = sum(Y^2) + sum(pred^2)
    - 2 * <Y[perm], pred>`` — no cancellation risk under the null (pred and
    permuted Y are unaligned). ``null_r2 = 1 - ss_res / ss_tot_real`` uses the
    SAME denominator as the observed r2, so band comparisons reduce to
    ss_res. Precision: preds fp16-stored -> fp32 GEMM, fp64 accumulation
    (the `_dual_perm_null` contract — null R2 ~ 0 needs nowhere near f64).
    """
    out: dict[str, dict] = {}
    if n_draws <= 0:
        return out
    p_dim = int(Y.shape[1])
    # The predicted row set of (i, j) depends only on (used_folds[i], j):
    # group sources by realized fold coverage (production: one group, all 6).
    groups: dict[tuple[int, ...], list[int]] = {}
    for i in turns:
        if used_folds[i]:
            groups.setdefault(tuple(used_folds[i]), []).append(i)
    for j in turns:
        for sig in sorted(groups):
            srcs = groups[sig]
            rows_order = (
                np.concatenate([by_turn[j][fold_labels[j] == f] for f in sig])
                if sig
                else np.empty(0, dtype=np.int64)
            )
            n_pred = int(rows_order.size)
            if n_pred < null_min_n:
                continue
            preds: list[np.ndarray] = []
            valid_srcs: list[int] = []
            for i in srcs:
                blocks = pred_blocks.get((i, j), {})
                arrs = [blocks[f] for f in sig if f in blocks]
                if not arrs:
                    continue
                pi = np.concatenate(arrs, axis=0)
                if pi.shape[0] != n_pred:
                    continue  # partial coverage — degenerate cell, no null
                preds.append(np.ascontiguousarray(pi, dtype=np.float32))
                valid_srcs.append(i)
            if not preds:
                continue
            yj = np.ascontiguousarray(Y[rows_order], dtype=np.float32)
            sum_y2 = float((yj.astype(np.float64) ** 2).sum())
            sum_p2 = np.asarray(
                [float((p.astype(np.float64) ** 2).sum()) for p in preds], dtype=np.float64
            )
            b_mat = np.stack(preds).reshape(len(preds), -1)  # (S, n_pred * P) fp32
            rng = np.random.default_rng(seed + 104_729 * int(j))
            perms = np.argsort(rng.random((n_draws, n_pred)), axis=1)
            cross = np.zeros((n_draws, len(preds)), dtype=np.float64)
            bytes_per_draw = n_pred * p_dim * 4
            k_chunk = max(1, int(1_000_000_000 // max(1, bytes_per_draw)))
            for s0 in range(0, n_draws, k_chunk):
                pp = perms[s0 : s0 + k_chunk]
                a_mat = yj[pp].reshape(pp.shape[0], -1)  # (k, n_pred * P) gather
                cross[s0 : s0 + pp.shape[0]] = (a_mat @ b_mat.T).astype(np.float64)
            for si, i in enumerate(valid_srcs):
                tot = ss_tot[(i, j)]
                if not (np.isfinite(tot) and tot > 1e-12):
                    continue
                draws = 1.0 - (sum_y2 + sum_p2[si] - 2.0 * cross[:, si]) / tot
                out[f"{i}->{j}"] = {
                    "null_mean": float(np.mean(draws)),
                    "null_hi": float(np.percentile(draws, 97.5)),
                    "null_max": float(np.max(draws)),
                    "null_n_draws": int(draws.size),
                    "n_pred": n_pred,
                }
    return out


def run_transfer(args: argparse.Namespace) -> None:
    tag = args.arm
    kinds = ["context_k", ANSWER_KIND[tag]]
    rows, arrays = _load_capture(Path(args.capture_root), tag, args.model, kinds, (HEADLINE_LAYER,))
    fold_of = _conv_fold_map([r["conv_id"] for r in rows])
    by_turn = _rows_by_turn(rows)
    turns = [t for t, s in by_turn.items() if s.size >= int(args.min_fit_n)]
    X = arrays["context_k"][HEADLINE_LAYER]
    Y = arrays[ANSWER_KIND[tag]][HEADLINE_LAYER]
    fold_labels = {
        t: np.asarray([fold_of[rows[i]["conv_id"]] for i in by_turn[t]], dtype=np.int64)
        for t in turns
    }
    ss_res = {(i, j): 0.0 for i in turns for j in turns}
    ss_tot = {(i, j): 0.0 for i in turns for j in turns}
    used_folds: dict[int, list[int]] = {i: [] for i in turns}
    # Frozen per-(source, target, fold) predictions for the null phase, fp16
    # (~21 GB worst case at 24 turns x 5000 convs x 3584 — pod-RAM bounded).
    pred_blocks: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    t0 = time.time()
    for i in turns:
        sel_i = by_turn[i]
        for f in range(N_FOLDS):
            tr = sel_i[fold_labels[i] != f]
            if tr.size < 3:
                continue
            used_folds[i].append(f)
            # concatenate EVERY target turn's fold-f test rows -> ONE prep+predict
            te_blocks = []
            spans = []
            pos = 0
            for j in turns:
                te_j = by_turn[j][fold_labels[j] == f]
                te_blocks.append(X[te_j])
                spans.append((j, pos, pos + te_j.size, te_j))
                pos += te_j.size
            if pos == 0:
                continue
            Xev = np.concatenate(te_blocks, axis=0)
            cache = _prep_fold(X[tr], Xev)  # train stats from turn i (frozen-map-swap shape)
            pred = _ridge_predict_cached(cache, Y[tr])
            for j, a, b, te_j in spans:
                if te_j.size == 0:
                    continue
                true = Y[te_j].astype(np.float64)
                pj = pred[a:b]
                ss_res[(i, j)] += float(np.sum((true - pj) ** 2))
                ss_tot[(i, j)] += float(np.sum((true - true.mean(0)) ** 2))
                pred_blocks.setdefault((i, j), {})[f] = pj.astype(np.float16)
        logger.info(
            "[transfer] %s/%s source turn %d done (%.0fs)", tag, args.model, i, time.time() - t0
        )
    r2 = {}
    for i in turns:
        for j in turns:
            tot = ss_tot[(i, j)]
            r2[f"{i}->{j}"] = (1.0 - ss_res[(i, j)] / tot) if tot > 1e-12 else float("nan")
    retained = {}
    for i in turns:
        for j in turns:
            diag = r2[f"{j}->{j}"]
            retained[f"{i}->{j}"] = (
                r2[f"{i}->{j}"] / diag if diag and np.isfinite(diag) and diag > 1e-6 else None
            )
    nulls = _transfer_nulls(
        turns,
        by_turn,
        fold_labels,
        used_folds,
        pred_blocks,
        Y,
        ss_tot,
        int(args.n_draws),
        int(args.null_min_n),
        NULL_SEED,
    )
    logger.info(
        "[transfer] %s/%s nulls: %d/%d cells (n_draws=%d, %.0fs)",
        tag,
        args.model,
        len(nulls),
        len(turns) ** 2,
        int(args.n_draws),
        time.time() - t0,
    )
    part = {
        "arm": tag,
        "model": args.model,
        "layer": HEADLINE_LAYER,
        "turns": turns,
        "fold_map_sha256": _fold_map_hash(fold_of),
        "r2": r2,
        "retained_fraction": retained,
        "nulls": nulls,
        "null_spec": {
            "n_draws": int(args.n_draws),
            "null_min_n": int(args.null_min_n),
            "seed": f"NULL_SEED({NULL_SEED}) + 104729 * target_turn",
            "kind": (
                "shuffle-within-turn-j row permutation vs FROZEN transfer "
                "predictions; one shared permutation set per target turn; "
                "fp16-stored preds, fp32 GEMM, fp64 accumulation; denominator "
                "= real fold-centered ss_tot (identical to the observed r2). "
                "Diagonal nulls are conditional on the fitted map — the refit "
                "permutation null for the same cell lives in the cells part."
            ),
        },
    }
    _write_part(Path(args.parts_dir), f"transfer_{tag}_{args.model}", part)


# ---------------------------------------------------------------------------
# operators: per-(turn, fold) betas + battery + self-cosine ceiling (F3)
# ---------------------------------------------------------------------------


def _gcv_primal_beta(X: np.ndarray, Y: np.ndarray, device: str) -> torch.Tensor:
    """fit_primal_beta's math with an explicit device (dual GCV -> primal beta)."""
    dev = torch.device(device)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=dev)
    Yt = torch.as_tensor(np.asarray(Y), dtype=torch.float64, device=dev)
    xmu = Xt.mean(0)
    xsd = Xt.std(0) + 1e-9
    Xn = (Xt - xmu) / xsd
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    G = Xn @ Xn.T
    w, V = torch.linalg.eigh(G)
    w = torch.clamp(w, min=0.0)
    VtY = V.T @ Yc
    sqVtY = (VtY**2).sum(1)
    tot = float((Yc**2).sum())
    ntr = Xn.shape[0]
    best_lam, best_gcv = float(LAMBDAS[0]), float("inf")
    for lam in LAMBDAS:
        filt = w / (w + lam)
        rss = tot - float(((2 * filt - filt**2) * sqVtY).sum())
        dof = float(filt.sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, float(lam)
    filt = 1.0 / (w + best_lam)
    return Xn.T @ (V @ (filt[:, None] * VtY))  # (D_in, D_out)


def _rank_truncated_cols(
    U: torch.Tensor, S: torch.Tensor, rel: float = 1e-3, max_rank: int | None = None
) -> tuple:
    """Numerical-rank column truncation of an SVD's U factor.

    Rank rule: ``r = min(#{S > S.max() * rel}, max_rank)`` — a relative
    singular-value threshold AND the algebraic row-count bound, both
    load-bearing (code-review v22 Critical 1):

    - ``rel=1e-3`` sits ~10x above the MEASURED fp16-persistence noise tail.
      Betas are saved ``.astype(np.float16)`` and reloaded for this SVD; an
      fp16 round trip of a rank-r matrix lifts the trailing singular values to
      ~1.0e-4 * S.max (v22 measurements: D=1024/true r=238 -> tail 1.03e-4;
      D=2048/r=476 -> 1.01e-4), ~100x ABOVE the previous ``rel=1e-6``, so the
      old rule read rank near-FULL and ``||U^T b|| / ||b||`` re-squashed to
      ~1 for ANY pair (the v21-C2 vacuity, one dtype interaction deeper).
      Directions with S below the fp16 noise floor are destroyed by the
      storage regardless, so nothing recoverable is discarded at 1e-3.
    - ``max_rank`` = the fold's training-row count: a per-(turn, fold) beta
      ``Xn^T @ (...)`` has algebraic rank <= its fold's row count (~833 at
      production, << 3584) INDEPENDENT of dtype — the clamp holds even if a
      future dtype/scale change moves the noise tail past ``rel``.

    With a full square U from ``svd(full_matrices=False)`` on the square beta
    the projection ``||U^T b|| == ||b||`` is identically 1.0 (v21 Critical 2)
    — the statistic must use only these columns. Returns ``(U[:, :r], r)``.
    """
    if S.numel() == 0:
        return U[:, :0], 0
    r = int((float(S.max()) * rel < S).sum().item())
    if max_rank is not None:
        r = min(r, int(max_rank))
    return U[:, : max(1, r)], max(1, r)


def _general_linear_cos(u_r: torch.Tensor, b_j: torch.Tensor) -> float:
    """Fraction of ``b_j`` captured by span(u_r): ``||u_r^T b_j||_F / ||b_j||_F``.

    ``u_r`` MUST be rank-truncated (`_rank_truncated_cols`); with a full
    square orthogonal U the statistic is identically 1.0 by construction.
    """
    proj = u_r.T @ b_j
    return float(torch.linalg.norm(proj) / (torch.linalg.norm(b_j) + 1e-12))


def run_operators(args: argparse.Namespace) -> None:  # noqa: C901 — linear battery driver
    tag = args.arm
    kinds = ["context_k", ANSWER_KIND[tag]]
    rows, arrays = _load_capture(Path(args.capture_root), tag, args.model, kinds, (HEADLINE_LAYER,))
    fold_of = _conv_fold_map([r["conv_id"] for r in rows])
    by_turn = _rows_by_turn(rows)
    turns = [t for t, s in by_turn.items() if s.size >= int(args.min_fit_n)]
    X = arrays["context_k"][HEADLINE_LAYER]
    Y = arrays[ANSWER_KIND[tag]][HEADLINE_LAYER]
    betas_dir = Path(args.betas_dir)
    betas_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)

    # 1) per-(turn, fold) betas on the fold's OWN rows (fold-disjoint estimates)
    beta_paths: dict[tuple[int, int], Path] = {}
    # fold row counts = the algebraic rank bound for each beta (v22 Critical 1);
    # recomputed from sel/labels even on resume (beta .npy already on disk).
    beta_rows: dict[tuple[int, int], int] = {}
    for t in turns:
        sel = by_turn[t]
        labels = np.asarray([fold_of[rows[i]["conv_id"]] for i in sel], dtype=np.int64)
        for f in range(N_FOLDS):
            p = betas_dir / f"{tag}_{args.model}_t{t:02d}_f{f}.npy"
            beta_paths[(t, f)] = p
            rows_f = sel[labels == f]
            beta_rows[(t, f)] = int(rows_f.size)
            if p.exists():
                continue
            if rows_f.size < 3:
                continue
            beta = _gcv_primal_beta(X[rows_f], Y[rows_f], args.device)
            np.save(p, beta.cpu().numpy().astype(np.float16))
        logger.info("[operators] %s/%s t%d betas done", tag, args.model, t)

    # In-memory beta cache: the pair battery revisits every (turn, fold) beta
    # O(turns * folds) times — re-reading the fp16 npy from disk per visit is
    # ~1 TB of IO at production scale (the #813 per-row-IO class). Full cache:
    # 36 turns x 6 folds x 3584^2 fp32 = ~11 GB, fits the A100/CPU host.
    _beta_cache: dict[tuple[int, int], torch.Tensor | None] = {}

    def _load_beta(t: int, f: int) -> torch.Tensor | None:
        if (t, f) in _beta_cache:
            return _beta_cache[(t, f)]
        p = beta_paths.get((t, f))
        val = None
        if p is not None and p.exists():
            val = torch.from_numpy(np.load(p).astype(np.float32)).to(dev)
        _beta_cache[(t, f)] = val
        return val

    # 2) battery per unordered turn pair (i <= j); i == j rows ARE the ceiling
    f_a, f_b = OPERATOR_HEAVY_FOLD_PAIR
    battery: dict[str, dict] = {}
    # precompute heavy-fold SVD factors per (turn, fold in pair)
    svd_cache: dict[tuple[int, int], dict] = {}
    for t in turns:
        for f in (f_a, f_b):
            b = _load_beta(t, f)
            if b is None:
                continue
            U, S, Vh = torch.linalg.svd(b, full_matrices=False)
            # Rank-truncate U at cache time: the full square U makes the
            # general-linear cos identically 1.0 (review v21 Critical 2);
            # clamp to the fold's row count so the fp16 storage noise tail
            # cannot re-inflate the rank (review v22 Critical 1).
            U_r, rank = _rank_truncated_cols(U, S, max_rank=beta_rows.get((t, f)))
            svd_cache[(t, f)] = {
                "U": U_r,
                "rank": rank,
                "n_rows": beta_rows.get((t, f)),
                "Vh": Vh,
                "norm": float(torch.linalg.norm(b)),
            }
            del b, U, S
    for ii, i in enumerate(turns):
        for j in turns[ii:]:
            # raw cosine over ALL disjoint fold pairs (cheap flattened dots)
            cos_vals = []
            for f in range(N_FOLDS):
                bi = _load_beta(i, f)
                if bi is None:
                    continue
                vi = bi.reshape(-1)
                ni = torch.linalg.norm(vi)
                for g in range(N_FOLDS):
                    if g == f:
                        continue
                    bj = _load_beta(j, g)
                    if bj is None:
                        continue
                    vj = bj.reshape(-1)
                    cos_vals.append(float((vi @ vj) / (ni * torch.linalg.norm(vj) + 1e-12)))
                    del bj, vj
                del bi, vi
            rec: dict = {
                "raw_cos_mean": float(np.mean(cos_vals)) if cos_vals else None,
                "raw_cos_sd": float(np.std(cos_vals)) if cos_vals else None,
                "n_fold_pairs": len(cos_vals),
            }
            # heavy ops on the pre-registered disjoint fold pair (f_a, f_b)
            bi = _load_beta(i, f_a)
            bj = _load_beta(j, f_b)
            if (
                bi is not None
                and bj is not None
                and (i, f_a) in svd_cache
                and (j, f_b) in svd_cache
            ):
                ni = svd_cache[(i, f_a)]["norm"]
                nj = svd_cache[(j, f_b)]["norm"]
                # Procrustes (output-side orthogonal align): max_R cos(b_i R, b_j)
                # = sum(svdvals(b_i^T b_j)) / (|b_i| |b_j|)
                sv = torch.linalg.svdvals(bi.T @ bj)
                rec["procrustes_cos"] = float(sv.sum() / (ni * nj + 1e-12))
                # general-linear change of coordinates: b_i L ~ b_j ->
                # cos_gl = |P_col(b_i) b_j| / |b_j| (residual fraction = 1 - cos^2),
                # projected onto the RANK-TRUNCATED column space of b_i.
                rec["general_linear_cos"] = _general_linear_cos(svd_cache[(i, f_a)]["U"], bj)
                rec["general_linear_rank"] = int(svd_cache[(i, f_a)]["rank"])
                # the algebraic bound the rank is clamped to (fold row count) —
                # rank <= rows is the v22 mechanizable invariant
                rec["general_linear_rows"] = svd_cache[(i, f_a)]["n_rows"]
                # principal angles from the CACHED per-(turn, fold) SVD factors
                # (same math as issue825_crossmodel_map_transfer.principal_angles,
                # without re-running two full 3584^2 SVDs per pair).
                Vha = svd_cache[(i, f_a)]["Vh"]
                Vhb = svd_cache[(j, f_b)]["Vh"]
                for k in (10, 50):
                    m_small = Vha[:k] @ Vhb[:k].T  # Qa.T @ Qb, (k, k)
                    cs = torch.linalg.svdvals(m_small).clamp(0.0, 1.0)
                    rec[f"principal_angle_cos_k{k}_mean"] = float(cs.mean())
                del sv
            if bi is not None:
                del bi
            if bj is not None:
                del bj
            battery[f"{i}~{j}"] = rec
        logger.info("[operators] %s/%s battery source turn %d done", tag, args.model, i)
    ceiling = {str(t): battery.get(f"{t}~{t}") for t in turns}
    part = {
        "arm": tag,
        "model": args.model,
        "layer": HEADLINE_LAYER,
        "turns": turns,
        "fold_map_sha256": _fold_map_hash(fold_of),
        "heavy_fold_pair": list(OPERATOR_HEAVY_FOLD_PAIR),
        "battery": battery,
        "selfsim_ceiling": ceiling,
        "betas_dir": str(betas_dir),
    }
    _write_part(Path(args.parts_dir), f"operators_{tag}_{args.model}", part)


# ---------------------------------------------------------------------------
# reach: turn-1 context -> turn-k answer (ridge ambient + batched MLP)
# ---------------------------------------------------------------------------


def run_reach(args: argparse.Namespace) -> None:
    tag = args.arm
    kinds = ["context_k", ANSWER_KIND[tag]]
    rows, arrays = _load_capture(Path(args.capture_root), tag, args.model, kinds, (HEADLINE_LAYER,))
    fold_of = _conv_fold_map([r["conv_id"] for r in rows])
    by_turn = _rows_by_turn(rows)
    turns = [t for t, s in by_turn.items() if s.size >= int(args.min_fit_n)]
    # common conv set present at EVERY analyzed turn (uniform n across k)
    conv_sets = [{rows[i]["conv_id"] for i in by_turn[t]} for t in turns]
    common = sorted(set.intersection(*conv_sets)) if conv_sets else []
    idx_by_turn_conv = {t: {rows[i]["conv_id"]: i for i in by_turn[t]} for t in turns}
    if 1 not in turns or len(common) < int(args.min_fit_n):
        _write_part(
            Path(args.parts_dir),
            f"reach_{tag}_{args.model}",
            {
                "arm": tag,
                "model": args.model,
                "status": "N/A — no turn-1 cell or too few common convs",
                "n_common": len(common),
            },
        )
        return
    X1 = arrays["context_k"][HEADLINE_LAYER][[idx_by_turn_conv[1][c] for c in common]]
    folds = _cell_folds(common, fold_of)
    assert folds is not None, "reach folds degenerate"
    ridge_out: dict[str, dict] = {}
    for k in turns:
        Yk = arrays[ANSWER_KIND[tag]][HEADLINE_LAYER][[idx_by_turn_conv[k][c] for c in common]]
        fit = _fit_cv(X1, Yk, folds, device=args.device)
        node = {
            "n": len(common),
            "r2": float(fit["r2"]),
            "r2_folds": [float(v) for v in fit["r2_folds"]],
            "null_mean": None,
            "null_hi": None,
            "null_max": None,
        }
        if len(common) >= args.null_min_n:
            draws = _dual_perm_null(
                X1, Yk, folds, fit["lambda_indices"], args.n_draws, NULL_SEED, device=args.device
            )
            finite = draws[np.isfinite(draws)]
            if finite.size:
                node["null_mean"] = float(np.mean(finite))
                node["null_hi"] = float(np.percentile(finite, 97.5))
                node["null_max"] = float(np.max(finite))
        ridge_out[str(k)] = node
        logger.info("[reach] %s/%s ridge k=%d r2=%.4f", tag, args.model, k, node["r2"])

    # MLP leg (vectorized_mlp_skill multihead; PCA-48 targets; sized per the
    # plan's ~200-cell budget: conv subsample + input PCA + few null draws).
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        MLPGroup,
        fit_batched_loco_mlp_multihead,
        robust_pca_basis,
    )

    rng = np.random.default_rng(NULL_SEED + 7)
    # Seeded stratified subsample — first-N of the sorted intersection is
    # single-source (`lmsys_*` sorts wholly before `wildchat_*`; review v21
    # Major 6). Chosen ids are recorded in the part payload below.
    mlp_convs = stratified_subsample_ids(common, int(args.mlp_conv_n))
    sel1 = [idx_by_turn_conv[1][c] for c in mlp_convs]
    Xm_amb = arrays["context_k"][HEADLINE_LAYER][sel1].astype(np.float32)
    # input PCA (train-agnostic global projection; recorded)
    p_in = min(int(args.mlp_input_pca), Xm_amb.shape[0], Xm_amb.shape[1])
    mu_x = Xm_amb.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(Xm_amb - mu_x, full_matrices=False)
    Xm = ((Xm_amb - mu_x) @ vh[:p_in].T).astype(np.float32)
    row_groups = np.asarray([fold_of[c] for c in mlp_convs], dtype=np.int64)
    groups: list[MLPGroup] = []
    y_by_key: dict[tuple, np.ndarray] = {}
    for k in turns:
        Yk = arrays[ANSWER_KIND[tag]][HEADLINE_LAYER][
            [idx_by_turn_conv[k][c] for c in mlp_convs]
        ].astype(np.float32)
        mu_y, comps, _rob = robust_pca_basis(Yk.astype(np.float64), 48)
        Yp = ((Yk - mu_y) @ comps.T).astype(np.float32)
        key = (k, "obs")
        groups.append(MLPGroup(key=key, X=Xm, Y=Yp))
        y_by_key[key] = Yp
        for d in range(int(args.mlp_null_draws)):
            perm = rng.permutation(Yp.shape[0])
            keyn = (k, f"null{d}")
            groups.append(MLPGroup(key=keyn, X=Xm, Y=Yp[perm]))
            y_by_key[keyn] = Yp[perm]
    res = fit_batched_loco_mlp_multihead(
        groups, device=args.device if args.device != "cpu" else "cpu", row_groups=row_groups
    )
    mlp_out: dict[str, dict] = {}
    for k in turns:
        pred = res.preds_by_key[(k, "obs")]
        obs_r2 = _r2(y_by_key[(k, "obs")], pred)
        nulls = []
        for d in range(int(args.mlp_null_draws)):
            keyn = (k, f"null{d}")
            nulls.append(_r2(y_by_key[keyn], res.preds_by_key[keyn]))
        mlp_out[str(k)] = {
            "n": len(mlp_convs),
            "input_pca": p_in,
            "r2": float(obs_r2),
            "null_r2": [float(v) for v in nulls],
            "null_max": float(np.nanmax(nulls)) if nulls else None,
        }
        logger.info("[reach] %s/%s MLP k=%d r2=%.4f", tag, args.model, k, obs_r2)
    part = {
        "arm": tag,
        "model": args.model,
        "layer": HEADLINE_LAYER,
        "n_common": len(common),
        "fold_map_sha256": _fold_map_hash(fold_of),
        "ridge": ridge_out,
        "mlp": mlp_out,
        "mlp_sizing": {
            "conv_n": len(mlp_convs),
            "input_pca": p_in,
            "target_pca": 48,
            "null_draws": int(args.mlp_null_draws),
            "subsample": "seeded stratified by source prefix (seed 42, "
            "issue825_turndyn_harvest.stratified_subsample_ids)",
            "conv_source_counts": source_counts(mlp_convs),
            "conv_ids": mlp_convs,
        },
    }
    _write_part(Path(args.parts_dir), f"reach_{tag}_{args.model}", part)


# ---------------------------------------------------------------------------
# assemble: bridge + diagnostics + merge -> results.json
# ---------------------------------------------------------------------------


def _bridge(args: argparse.Namespace, model: str) -> dict:
    """H4: per-turn armG vs armR_own contrast on the seed intersection."""
    cells = []
    rowsG, arrG = _load_capture(
        Path(args.capture_root), "armG", model, ["context_k", "answer_own_t1"], (HEADLINE_LAYER,)
    )
    rowsR, arrR = _load_capture(
        Path(args.capture_root),
        "armR_own",
        model,
        ["context_k", "answer_own_t1"],
        (HEADLINE_LAYER,),
    )
    byG, byR = _rows_by_turn(rowsG), _rows_by_turn(rowsR)
    turns = sorted(set(byG) & set(byR))
    conv_universe: set[str] = set()
    fold_of = _conv_fold_map([r["conv_id"] for r in rowsR])
    per_turn_n = {}
    for t in turns:
        gmap = {rowsG[i]["conv_id"]: i for i in byG[t]}
        rmap = {rowsR[i]["conv_id"]: i for i in byR[t]}
        inter = sorted(set(gmap) & set(rmap))
        if len(inter) < int(args.min_fit_n):
            continue
        folds = _cell_folds(inter, fold_of)
        if folds is None:
            continue
        selG = [gmap[c] for c in inter]
        selR = [rmap[c] for c in inter]
        XG, YG = (
            arrG["context_k"][HEADLINE_LAYER][selG],
            arrG["answer_own_t1"][HEADLINE_LAYER][selG],
        )
        XR, YR = (
            arrR["context_k"][HEADLINE_LAYER][selR],
            arrR["answer_own_t1"][HEADLINE_LAYER][selR],
        )
        _fitG, predG = _fit_cv(XG, YG, folds, return_pred=True, device=args.device)
        _fitR, predR = _fit_cv(XR, YR, folds, return_pred=True, device=args.device)
        cells.append(
            {
                "turn": t,
                "convs": inter,
                "y_own": YG.astype(np.float32),
                "pred_own": predG.astype(np.float32),
                "y_log": YR.astype(np.float32),
                "pred_log": predR.astype(np.float32),
            }
        )
        conv_universe |= set(inter)
        per_turn_n[str(t)] = len(inter)
    if not cells:
        return {"status": "N/A — no bridged turns with sufficient intersection"}
    bs = _cluster_bootstrap(cells, sorted(conv_universe), args.n_boot, BOOT_SEED)
    per_turn = {}
    for c in cells:
        r2G = _r2(c["y_own"].astype(np.float64), c["pred_own"].astype(np.float64))
        r2R = _r2(c["y_log"].astype(np.float64), c["pred_log"].astype(np.float64))
        node = bs["per_turn"][str(c["turn"])]
        per_turn[str(c["turn"])] = {
            "n": per_turn_n[str(c["turn"])],
            "r2_armG": float(r2G),
            "r2_armR_own": float(r2R),
            "delta": float(r2G - r2R),
            "delta_ci": node["delta_ci"],
            "within_band": bool(abs(r2G - r2R) <= BRIDGE_BAND),
        }
    return {
        "band": BRIDGE_BAND,
        "note": (
            "delta = R2_armG - R2_armR_own on the per-turn seed intersection (the "
            "both-windows-surviving conversations by construction — the pre-registered "
            "overflow-asymmetry robustness read IS the primary computation)"
        ),
        "per_turn": per_turn,
        "pooled_delta": bs["pooled_delta"],
        "pooled_delta_ci": bs["pooled_delta_ci"],
        "n_boot": args.n_boot,
    }


def run_assemble(args: argparse.Namespace) -> None:
    parts_dir = Path(args.parts_dir)
    parts = {}
    for p in sorted(parts_dir.glob("*.json")):
        with open(p) as f:
            parts[p.stem] = json.load(f)
    # fold-map hash assert (assumption 10): all parts of one (arm, model) agree
    seen: dict[tuple, set] = {}
    for part in parts.values():
        if "fold_map_sha256" in part:
            seen.setdefault((part.get("arm"), part.get("model")), set()).add(
                part["fold_map_sha256"]
            )
    for key, hashes in seen.items():
        assert len(hashes) == 1, f"[assemble] fold-map hash mismatch for {key}: {hashes}"

    bridge = {}
    for model in ("instruct", "pretrained"):
        try:
            bridge[model] = _bridge(args, model)
        except AssertionError as e:
            bridge[model] = {"status": f"N/A — {e}"}

    diagnostics = {}
    for model in ("instruct", "pretrained"):
        p = Path(args.rollout_dir) / model / "rollout_diagnostics.json"
        if p.exists():
            with open(p) as f:
                diagnostics[model] = json.load(f)
    harvest_report = {}
    hr = Path(args.panel_dir) / "harvest_report.json"
    if hr.exists():
        with open(hr) as f:
            harvest_report = json.load(f)

    payload = {
        "issue": 825,
        "followup_label": "turn-dynamics-allturns-5000",
        "description": (
            "Per-turn context->answer map at flat n to depth K: existence (Q1), "
            "cross-turn transfer + operator battery with within-turn self-cosine "
            "ceiling (Q2), turn-1 reach ridge+MLP (Q3), real-vs-simulated bridge (H4)."
        ),
        "headline_layer": HEADLINE_LAYER,
        "layers": list(BASE_LAYERS),
        "estimator": "lambda-grid dual/PRESS ridge, conv-grouped 6-fold CV (round-10 parity)",
        "gates": {
            "G-A": harvest_report.get("gate_ga"),
            "G-C": {m: parts.get(f"gc_{m}", {}).get("pass") for m in ("instruct", "pretrained")},
        },
        "parts": parts,
        "bridge_H4": bridge,
        "rollout_diagnostics": diagnostics,
        "harvest_report_digest": {
            k: harvest_report.get(k)
            for k in ("K_real", "panel_n", "n_seeds", "nk_table", "panel_ids_sha256", "gate_ga")
        },
        "n_boot": args.n_boot,
        "n_draws": args.n_draws,
        "seeds": {"fold": FOLD_SEED, "null": NULL_SEED, "boot": BOOT_SEED, "panel": 42, "gen": 42},
        "git_commit": _git_commit(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "smoke": bool(args.smoke),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    logger.info("[assemble] results -> %s", out)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fit-phase",
        required=True,
        choices=("gc", "cells", "transfer", "operators", "reach", "assemble"),
    )
    ap.add_argument("--model", default="instruct", choices=("instruct", "pretrained"))
    ap.add_argument("--arm", default="armG", choices=("armG", "armR_own", "armR_logged"))
    ap.add_argument("--capture-root", required=True)
    ap.add_argument("--panel-dir", default="", help="harvest dir (gc + assemble)")
    ap.add_argument("--rollout-dir", default="", help="P1 rollout root (assemble diagnostics)")
    ap.add_argument("--parts-dir", required=True)
    ap.add_argument("--betas-dir", default="", help="per-(turn,fold) beta output dir (operators)")
    ap.add_argument("--out-json", default="eval_results/issue_825/turn_dynamics/results.json")
    ap.add_argument(
        "--r10-json",
        default=str(REPO_ROOT / "eval_results/issue_825/onpolicy_turn_depth/results.json"),
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--null-min-n", type=int, default=NULL_MIN_N)
    ap.add_argument("--min-fit-n", type=int, default=MIN_FIT_N)
    ap.add_argument("--mlp-conv-n", type=int, default=1000)
    ap.add_argument("--mlp-input-pca", type=int, default=256)
    ap.add_argument("--mlp-null-draws", type=int, default=2)
    ap.add_argument(
        "--emit-r10-ref",
        action="store_true",
        help="SMOKE tooling: write the gc refit curve in the round-10 results shape",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.fit_phase == "gc" and not args.panel_dir:
        ap.error("--fit-phase gc requires --panel-dir")
    if args.fit_phase == "operators" and not args.betas_dir:
        ap.error("--fit-phase operators requires --betas-dir")
    logger.info(
        "[fit] phase=%s arm=%s model=%s device=%s",
        args.fit_phase,
        args.arm,
        args.model,
        args.device,
    )
    {
        "gc": run_gc,
        "cells": run_cells,
        "transfer": run_transfer,
        "operators": run_operators,
        "reach": run_reach,
        "assemble": run_assemble,
    }[args.fit_phase](args)


if __name__ == "__main__":
    main()
