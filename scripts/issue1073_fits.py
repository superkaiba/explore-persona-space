#!/usr/bin/env python3
"""Issue #1073 P4: shared-Gram fits + cross-arm statistics (DV1-DV5).

Fits (DV1): the FFC shared-factorization GCV ridge (imported from
``issue779_fitter_fair_comparison``: ONE ``_factorize`` eigh per (input, layer,
fold), extra target sets via batched ``V.T @ Y`` GEMMs — 2 inputs x L layers x
5 folds eigh total) over FOUR target sets {avg10, greedy, stoch1_old,
stoch1_new} + the mean-target floor (the degenerate prefix arm) + the
shuffled-pairing null. Science folds are DUPLICATE-CLUSTERED at seed 42
(statistics Must-Fix); the arm-(c) reproduction gate runs a SEPARATE pass at
fold seed 0 with the reference's raw pointwise partition (byte-parity with
``eval_results/issue_779/percontext_recon.json``; kill criterion 3).

Statistics: per-context SSE + SST components persisted with fold provenance;
ALL bootstrap draws evaluated as ONE (n_boot, N) gather (pooled R2 recomputed
as 1 - sum(SSE)/sum(SST) inside each resample); the DV4 jackknife is a closed
-form einsum over the (N, 10) per-rollout cosine structure (no refits).

Monitoring (DV3): fit h per arm on the full common set at the unique frozen
layers via ``issue779_arm_headline.GramRidge`` (one factorization per layer),
apply to the pass-A eval contexts (``issue779_stage1.build_eval_matrix``),
within-condition Pearson vs the persisted graded judge scores; the cross-arm
delta-r uses a JOINT condition resample (one shared index matrix).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_percontext_recon as PR  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
import issue1073_capture as CAP  # noqa: E402

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy: shared-VM thread caps bind at import (#847)

import issue1073_common as I  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue779_arm_headline import GramRidge  # noqa: E402
from issue779_fitter_fair_comparison import (  # noqa: E402
    LAMBDAS,
    _apply,
    _cross_kernel,
    _factorize,
    _gcv_solve,
    _vty_ymu,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1073_fits")

SHUFFLE_SEED = 0  # fixed shuffled-pairing permutation (plan §5 shuffle_null)
REPRO_GATE_TOL = 0.02  # kill criterion 3 (plan §Kill criteria)
FIT_TARGETS = ("avg10", "greedy", "stoch1_old", "stoch1_new", "shuffle_null")


# ── loading ───────────────────────────────────────────────────────────────────


def _load_reduction(red_dir: Path, name: str) -> torch.Tensor:
    blob = torch.load(red_dir / f"{name}.pt", mmap=True, weights_only=False, map_location="cpu")
    return blob["tensor"]  # (N, L, H) fp32


def load_ctx(args) -> dict:
    """Load bundle + reductions + coverage + p0 branch; build the common set."""
    root = I.out_root(args.smoke, args.out_root)
    res_dir = I.results_dir(root, args.smoke)
    in_dir = I.inputs_dir(root)
    bundle_path = in_dir / I.BUNDLE_PATH_IN_REPO
    assert bundle_path.exists(), f"bundle missing at {bundle_path} — run P0 first"
    with open(res_dir / "p0_probe.json") as f:
        p0 = json.load(f)

    # Layer count from the P3 store (single source: the captured shards).
    red_dir = root / "reductions"
    vbar10 = _load_reduction(red_dir, "vbar10")
    n_ctx, n_layers, hidden = vbar10.shape
    bundle = I.load_bundle(
        bundle_path,
        expected_layers=n_layers,
        expected_hidden=hidden,
        min_n=2 if args.smoke else 4900,
    )
    assert len(bundle["prompts"]) == n_ctx, (len(bundle["prompts"]), n_ctx)

    v_greedy = _load_reduction(red_dir, "v_greedy")
    stoch1_new = _load_reduction(red_dir, "stoch1_new")
    cov = torch.load(red_dir / "coverage.pt", weights_only=False, map_location="cpu")

    keep_mask = (~cov["greedy_empty"]) & (~cov["stoch_any_empty"])
    keep_idx = np.where(keep_mask.numpy())[0]
    drops = {
        "greedy_empty": int(cov["greedy_empty"].sum()),
        "stoch_any_empty": int(cov["stoch_any_empty"].sum()),
        "n_kept": int(keep_idx.size),
        "n_total": int(n_ctx),
    }
    # Equalize-down floor (plan §4): >= 4700 proceeds silently; below that we
    # proceed with the reduced N FLAGGED (never a hard abort).
    reduced_n_flag = (not args.smoke) and keep_idx.size < 4700
    if reduced_n_flag:
        logger.warning("[common-set] kept N=%d < 4700 — proceeding FLAGGED", keep_idx.size)
    logger.info("[common-set] %s", json.dumps(drops))

    prompts_kept = [bundle["prompts"][i] for i in keep_idx]
    folds = I.clustered_folds(prompts_kept, args.n_folds, I.FOLD_SEED_SCIENCE)
    fold_of = np.empty(keep_idx.size, dtype=np.int64)
    for f, te in enumerate(folds):
        fold_of[te] = f
    assert all(len(f) > 0 for f in folds), [len(f) for f in folds]

    branch_map = p0["science_stoch_arm_by_probe_branch"]
    return {
        "root": root,
        "res_dir": res_dir,
        "in_dir": in_dir,
        "bundle": bundle,
        "vbar10": vbar10,
        "v_greedy": v_greedy,
        "stoch1_new": stoch1_new,
        "n_ctx": n_ctx,
        "n_layers": n_layers,
        "hidden": hidden,
        "keep_idx": keep_idx,
        "common_fp": I.common_index_fingerprint(keep_idx),
        "drops": drops,
        "reduced_n_flag": reduced_n_flag,
        "folds": folds,
        "fold_of": fold_of,
        "branch_map": branch_map,
        "p0": p0,
    }


def _layer_np(t: torch.Tensor, li: int) -> np.ndarray:
    """(N, H) float64 numpy slice of an (N, L, H) tensor at layer index li."""
    return t[:, li, :].to(torch.float64).numpy()


def _targets_at_layer(ctx: dict, li: int, perm: np.ndarray) -> dict[str, np.ndarray]:
    """The 5 fit-target matrices at one layer, restricted to the common set."""
    keep = ctx["keep_idx"]
    avg10 = _layer_np(ctx["vbar10"], li)[keep]
    return {
        "avg10": avg10,
        "greedy": _layer_np(ctx["v_greedy"], li)[keep],
        "stoch1_old": _layer_np(ctx["bundle"]["v_x"], li)[keep],
        "stoch1_new": _layer_np(ctx["stoch1_new"], li)[keep],
        "shuffle_null": avg10[perm],
    }


# ── arm-(c) reproduction gate (kill criterion 3) ─────────────────────────────


def repro_gate(ctx: dict, device: str, smoke: bool) -> dict:
    """Refit arm (c) on the FULL bundle at fold seed 0 (raw pointwise folds) and
    compare frozen-layer held-out R2 to ``percontext_recon.json`` (+-0.02)."""
    bundle = ctx["bundle"]
    n = ctx["n_ctx"]
    folds = PR._cv_folds(n, min(I.N_FOLDS, n), I.FOLD_SEED_REPRO)
    gate_layers = (
        {t: PR.READ_OUT_LAYER[t] for t in PR.READ_OUT_LAYER}
        if not smoke
        else {"smoke": ctx["n_layers"] - 1}
    )
    dev = torch.device(device)
    ours: dict[int, dict] = {}
    for li in sorted(set(gate_layers.values())):
        X = _layer_np(bundle["cx_last"], li)
        Y = _layer_np(bundle["v_x"], li)
        r2s = []
        for te in folds:
            m = np.ones(n, dtype=bool)
            m[te] = False
            fact = _factorize(X[m], dev)
            lam, vty, ymu = _gcv_solve(fact, Y[m])
            pred = _apply(fact, lam, vty, ymu, _cross_kernel(fact, X[te]))
            r2s.append(PR._pooled_r2(pred, Y[te]))
        # nanmean + explicit valid count: a SINGLE-ROW test fold has zero
        # test-own-mean variance and a legitimately-NaN R2 (tiny-N smoke only;
        # production folds are ~1000 rows and always valid — count reported).
        ours[li] = {
            "r2_mean": float(np.nanmean(r2s)),
            "r2_folds": [float(v) for v in r2s],
            "n_valid_folds": int(np.isfinite(r2s).sum()),
        }
        logger.info("[repro-gate] L%d held-out R2 %.4f (seed-0 folds)", li, ours[li]["r2_mean"])

    result = {
        "fold_seed": I.FOLD_SEED_REPRO,
        "tolerance": REPRO_GATE_TOL,
        "ours_by_layer": {str(k): v for k, v in ours.items()},
        "reference_compare": None,
    }
    if smoke:
        result["reference_compare"] = "skipped (smoke fixture has no committed reference)"
        return result
    ref_path = PROJECT_ROOT / "eval_results" / "issue_779" / "percontext_recon.json"
    with open(ref_path) as f:
        ref = json.load(f)
    compare = {}
    fails = []
    for trait, li in gate_layers.items():
        ref_mean = ref["read1_heldout_recon"]["per_trait"][trait]["heldout_r2_mean"]
        diff = ours[li]["r2_mean"] - ref_mean
        compare[trait] = {"layer": li, "ref": ref_mean, "ours": ours[li]["r2_mean"], "diff": diff}
        if abs(diff) > REPRO_GATE_TOL:
            fails.append(trait)
    result["reference_compare"] = compare
    if fails:
        I.write_sentinel(
            "epm:failure",
            json.dumps(
                {
                    "failure_class": "code",
                    "reason": "arm-c-reproduction-gate",
                    "assert_tag": "arm-c-reproduction-gate",
                    "compare": compare,
                }
            ),
        )
        raise SystemExit(
            f"P4 HALT: arm-(c) reproduction gate FAILED (kill criterion 3) on {fails}: {compare}"
        )
    logger.info("[repro-gate] PASS: %s", json.dumps(compare))
    return result


# ── science fits (duplicate-clustered seed-42 folds, shared factorization) ───


def _cv_one_layer(
    X: np.ndarray, targets: dict[str, np.ndarray], folds: list[np.ndarray], dev, want_details: bool
) -> tuple[dict, dict | None, dict | None]:
    """One (input, layer): 5-fold CV, ONE factorization per fold shared across
    ALL targets (+ the avg10 mean-target floor). Returns (acc, pd_acc, preds)."""
    nk = X.shape[0]
    hidden = X.shape[1]
    acc = {
        name: {
            "sse": np.zeros(nk),
            "sst": np.zeros(nk),
            "cos": np.zeros(nk),
            "r2_folds": [],
            "lam": [],
        }
        for name in (*FIT_TARGETS, "mean_floor")
    }
    pd_acc = (
        {name: {"sse_dim": np.zeros(hidden), "sst_dim": np.zeros(hidden)} for name in FIT_TARGETS}
        if want_details
        else None
    )
    preds_acc = (
        {name: np.zeros((nk, hidden), dtype=np.float32) for name in FIT_TARGETS}
        if want_details
        else None
    )
    for te in folds:
        m = np.ones(nk, dtype=bool)
        m[te] = False
        fact = _factorize(X[m], dev)
        kev = _cross_kernel(fact, X[te])
        for name in FIT_TARGETS:
            y = targets[name]
            lam, vty, ymu = _gcv_solve(fact, y[m])
            pred = _apply(fact, lam, vty, ymu, kev)
            true = y[te]
            mu = true.mean(0)
            acc[name]["sse"][te] = ((true - pred) ** 2).sum(1)
            acc[name]["sst"][te] = ((true - mu) ** 2).sum(1)
            acc[name]["cos"][te] = PR._per_context_cosine(pred, true)
            acc[name]["r2_folds"].append(PR._pooled_r2(pred, true))
            acc[name]["lam"].append(float(lam))
            if pd_acc is not None:
                pd_acc[name]["sse_dim"] += ((true - pred) ** 2).sum(0)
                pd_acc[name]["sst_dim"] += ((true - mu) ** 2).sum(0)
            if preds_acc is not None:
                preds_acc[name][te] = pred.astype(np.float32)
            # mean-target floor for the avg10 target (the degenerate prefix
            # arm): predicting the TRAIN mean — computed once per fold.
            if name == "avg10":
                ymu_tr = y[m].mean(0)
                acc["mean_floor"]["sse"][te] = ((true - ymu_tr) ** 2).sum(1)
                acc["mean_floor"]["sst"][te] = ((true - mu) ** 2).sum(1)
                acc["mean_floor"]["cos"][te] = PR._per_context_cosine(
                    np.tile(ymu_tr, (len(te), 1)), true
                )
                acc["mean_floor"]["r2_folds"].append(
                    1.0
                    - float(acc["mean_floor"]["sse"][te].sum())
                    / max(float(acc["mean_floor"]["sst"][te].sum()), 1e-12)
                )
    return acc, pd_acc, preds_acc


def _save_layer_preds(preds_acc: dict, ctx: dict, input_name: str, li: int, preds_dir: Path):
    for name, pred in preds_acc.items():
        torch.save(
            {
                "pred": torch.from_numpy(pred).to(torch.float16),
                "common_index": ctx["keep_idx"],
                "fold_of": ctx["fold_of"],
                "layer": li,
                "input": input_name,
                "target": name,
                "metadata": I.reproducibility_metadata(
                    {"script": "issue1073_fits", "artifact": "predictions"}
                ),
            },
            preds_dir / f"{name}_{input_name}_L{li}.pt",
        )


def science_fits(ctx: dict, device: str, readout: list[int]) -> dict:
    """Held-out CV per (input x layer x target): ONE eigh per (input, layer,
    fold), all targets via extra V.T@Y columns. Checkpointed per (input, layer)
    with a regime-pinned resume (code-style intra-phase grain)."""
    dev = torch.device(device)
    keep = ctx["keep_idx"]
    nk = keep.size
    folds = ctx["folds"]
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(nk)
    ck_dir = ctx["root"] / "p4_checkpoint"
    ck_dir.mkdir(parents=True, exist_ok=True)
    regime = {
        "fold_seed": I.FOLD_SEED_SCIENCE,
        "n_folds": len(folds),
        "common_fp": ctx["common_fp"],
        "targets": list(FIT_TARGETS),
        "shuffle_seed": SHUFFLE_SEED,
        "n_layers": ctx["n_layers"],
    }

    out: dict = {"last": {}, "mean": {}}
    percontext: dict = {}
    perdim: dict = {}
    preds_dir = ctx["root"] / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    t_phase = time.time()
    for input_name, field in (("last", "cx_last"), ("mean", "cx_mean")):
        for li in range(ctx["n_layers"]):
            ck = ck_dir / f"{input_name}_L{li}.pt"
            if ck.exists():
                blob = torch.load(ck, weights_only=False, map_location="cpu")
                assert blob["regime"] == regime, (
                    f"resume regime mismatch at {ck}: {blob['regime']} != {regime}"
                )
                out[input_name][li] = blob["summary"]
                if blob.get("percontext"):
                    percontext[f"{input_name}_L{li}"] = blob["percontext"]
                if blob.get("perdim"):
                    perdim[f"{input_name}_L{li}"] = blob["perdim"]
                logger.info("[fits] %s L%d resumed from checkpoint", input_name, li)
                continue
            t0 = time.time()
            X = _layer_np(ctx["bundle"][field], li)[keep]
            targets = _targets_at_layer(ctx, li, perm)
            is_readout = li in readout
            want_details = is_readout and input_name == "last"
            acc, pd_acc, preds_acc = _cv_one_layer(X, targets, folds, dev, want_details)
            summary = {
                name: {
                    "r2_pooled": 1.0 - float(a["sse"].sum()) / max(float(a["sst"].sum()), 1e-12),
                    "r2_folds": [float(v) for v in a["r2_folds"]],
                    "cos_mean": float(np.nanmean(a["cos"])),
                    "gcv_lambdas": a["lam"],
                }
                for name, a in acc.items()
            }
            pc = (
                {
                    name: {
                        "sse": a["sse"].tolist(),
                        "sst": a["sst"].tolist(),
                        "cos": a["cos"].tolist(),
                    }
                    for name, a in acc.items()
                }
                if is_readout
                else None
            )
            pdim = None
            if pd_acc is not None:
                pdim = {
                    name: {
                        "r2_dim": (1.0 - d["sse_dim"] / np.maximum(d["sst_dim"], 1e-12)).tolist()
                    }
                    for name, d in pd_acc.items()
                }
                _save_layer_preds(preds_acc, ctx, input_name, li, preds_dir)
            torch.save({"regime": regime, "summary": summary, "percontext": pc, "perdim": pdim}, ck)
            out[input_name][li] = summary
            if pc:
                percontext[f"{input_name}_L{li}"] = pc
            if pdim:
                perdim[f"{input_name}_L{li}"] = pdim
            logger.info(
                "[fits] %s L%d done in %.1f s (avg10 R2=%.4f, greedy R2=%.4f)",
                input_name,
                li,
                time.time() - t0,
                summary["avg10"]["r2_pooled"],
                summary["greedy"]["r2_pooled"],
            )
    logger.info("[fits] science CV total %.1f min", (time.time() - t_phase) / 60.0)
    return {"summary": out, "percontext": percontext, "perdim": perdim, "regime": regime}


def val_lambda_robustness(ctx: dict, device: str, readout: list[int]) -> dict:
    """Val-selected-lambda robustness pass (FFC D1 precedent: GCV degenerates at
    n ~= H). Fixed 72/8/20 split (the FFC 3600/400/1000 proportions) at seed 42,
    input cx_last, the four arm targets; reports val-selected vs GCV test R2."""
    dev = torch.device(device)
    keep = ctx["keep_idx"]
    nk = keep.size
    rng = np.random.default_rng(I.FOLD_SEED_SCIENCE)
    perm_split = rng.permutation(nk)
    n_te = max(1, round(0.2 * nk))
    n_val = max(1, round(0.08 * nk))
    te = np.sort(perm_split[:n_te])
    val = np.sort(perm_split[n_te : n_te + n_val])
    tr = np.sort(perm_split[n_te + n_val :])
    assert len(tr) > 1, (nk, len(tr))
    shuffle_perm = np.random.default_rng(SHUFFLE_SEED).permutation(nk)
    out: dict = {"split": {"n_train": len(tr), "n_val": len(val), "n_test": len(te)}}
    for li in readout:
        X = _layer_np(ctx["bundle"]["cx_last"], li)[keep]
        targets = _targets_at_layer(ctx, li, shuffle_perm)
        fact = _factorize(X[tr], dev)
        kval = _cross_kernel(fact, X[val])
        kte = _cross_kernel(fact, X[te])
        per_t = {}
        for name in I.ARMS:
            y = targets[name]
            vty, ymu = _vty_ymu(fact, y[tr])
            best_lam, best_vr2 = float(LAMBDAS[0]), -np.inf
            for lam in LAMBDAS:
                vr2 = PR._pooled_r2(_apply(fact, float(lam), vty, ymu, kval), y[val])
                if np.isfinite(vr2) and vr2 > best_vr2:
                    best_vr2, best_lam = float(vr2), float(lam)
            r2_val_sel = PR._pooled_r2(_apply(fact, best_lam, vty, ymu, kte), y[te])
            gcv_lam, vty2, ymu2 = _gcv_solve(fact, y[tr])
            r2_gcv = PR._pooled_r2(_apply(fact, gcv_lam, vty2, ymu2, kte), y[te])
            per_t[name] = {
                "val_lambda": best_lam,
                "gcv_lambda": float(gcv_lam),
                "r2_test_val_selected": float(r2_val_sel),
                "r2_test_gcv": float(r2_gcv),
                "delta": float(r2_val_sel - r2_gcv),
            }
        out[f"L{li}"] = per_t
        logger.info("[val-lambda] L%d: %s", li, json.dumps(per_t))
    return out


# ── DV2/DV4 target agreement + jackknife ──────────────────────────────────────


def _row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(1)
    den = (np.linalg.norm(a, axis=1) + 1e-12) * (np.linalg.norm(b, axis=1) + 1e-12)
    return num / den


def _stoch_layer_matrix(store_dir: Path, li: int, keep_idx: np.ndarray, n_ctx: int) -> np.ndarray:
    """(N_kept, 10, H) float64 per-rollout matrix at one layer from the fp16 store."""
    pos_of = {int(ci): k for k, ci in enumerate(keep_idx.tolist())}
    v = None
    for _p, shard in CAP.iter_shards(store_dir, "stoch10"):
        li_pos = list(shard["layers"]).index(li)
        sl = shard["summ"][:, li_pos, :].to(torch.float64).numpy()
        if v is None:
            v = np.zeros((len(keep_idx), I.N_ROLLOUTS, sl.shape[1]))
        for row, (ci, ri) in enumerate(shard["index"]):
            k = pos_of.get(int(ci))
            if k is not None:
                v[k, ri] = sl[row]
    assert v is not None
    return v


def target_agreement(ctx: dict, readout: list[int], rb_by_trait: dict, frozen: dict) -> dict:
    """DV2 per-context target cosines + DV4 symmetric jackknife statistic."""
    keep = ctx["keep_idx"]
    store_dir = ctx["root"] / "v_store"
    out: dict = {"readout_layers": readout, "per_layer": {}, "curves": {}}

    # Per-layer DV2 curves over ALL layers (from the fp32 reductions — cheap).
    curves = {"greedy_vs_avg10": [], "stoch1_old_vs_avg10": [], "stoch1_new_vs_avg10": []}
    for li in range(ctx["n_layers"]):
        vb = _layer_np(ctx["vbar10"], li)[keep]
        curves["greedy_vs_avg10"].append(
            float(np.nanmean(_row_cos(_layer_np(ctx["v_greedy"], li)[keep], vb)))
        )
        curves["stoch1_old_vs_avg10"].append(
            float(np.nanmean(_row_cos(_layer_np(ctx["bundle"]["v_x"], li)[keep], vb)))
        )
        curves["stoch1_new_vs_avg10"].append(
            float(np.nanmean(_row_cos(_layer_np(ctx["stoch1_new"], li)[keep], vb)))
        )
    out["curves"] = curves

    for li in readout:
        g = _layer_np(ctx["v_greedy"], li)[keep]
        vb = _layer_np(ctx["vbar10"], li)[keep]
        v_old = _layer_np(ctx["bundle"]["v_x"], li)[keep]
        v_new1 = _layer_np(ctx["stoch1_new"], li)[keep]
        entry: dict = {
            "cos_greedy_avg10": _row_cos(g, vb).tolist(),
            "cos_stoch1_old_avg10": _row_cos(v_old, vb).tolist(),
            "cos_stoch1_new_avg10": _row_cos(v_new1, vb).tolist(),
            "cos_greedy_stoch1_old": _row_cos(g, v_old).tolist(),
        }
        # DV4 jackknife (closed form; no per-leave-out loop): S = sum_j v_j.
        v = _stoch_layer_matrix(store_dir, li, keep, ctx["n_ctx"])  # (n, 10, H)
        s = v.sum(1)  # (n, H)
        vj_norm2 = np.einsum("njh,njh->nj", v, v)
        s_norm2 = np.einsum("nh,nh->n", s, s)
        s_dot_vj = np.einsum("nh,njh->nj", s, v)
        mj_norm = np.sqrt(np.maximum(s_norm2[:, None] - 2 * s_dot_vj + vj_norm2, 0)) / 9.0
        # cos(greedy, m_j)
        g_dot_s = np.einsum("nh,nh->n", g, s)
        g_dot_vj = np.einsum("nh,njh->nj", g, v)
        g_norm = np.linalg.norm(g, axis=1)
        cos_g_mj = ((g_dot_s[:, None] - g_dot_vj) / 9.0) / (g_norm[:, None] * mj_norm + 1e-12)
        # cos(v_j, m_j)
        vj_dot_mj = (s_dot_vj - vj_norm2) / 9.0
        cos_vj_mj = vj_dot_mj / (np.sqrt(vj_norm2) * mj_norm + 1e-12)
        delta_ctx = cos_g_mj.mean(1) - cos_vj_mj.mean(1)
        entry["dv4_delta_ctx"] = delta_ctx.tolist()
        entry["dv4_mean_cos_g_loo9"] = cos_g_mj.mean(1).tolist()
        entry["dv4_mean_cos_draw_loo9"] = cos_vj_mj.mean(1).tolist()
        entry["jackknife_draw_band"] = {
            "p5": float(np.quantile(cos_vj_mj, 0.05)),
            "p50": float(np.quantile(cos_vj_mj, 0.50)),
            "p95": float(np.quantile(cos_vj_mj, 0.95)),
        }
        out["per_layer"][f"L{li}"] = entry

    # Per-context <v_arm, r_B> projections at each trait's frozen SYSTEM layer
    # (per-unit companion scatter, plan §6 figures).
    proj = {}
    for trait, modes in frozen.items():
        li = modes["system"]
        rb = rb_by_trait[trait][li]
        proj[trait] = {
            "layer": li,
            "greedy": (_layer_np(ctx["v_greedy"], li)[keep] @ rb).tolist(),
            "avg10": (_layer_np(ctx["vbar10"], li)[keep] @ rb).tolist(),
        }
    out["rb_projections_system_layer"] = proj
    return out


# ── DV3 monitoring (joint condition resample) ────────────────────────────────


def monitoring(ctx: dict, readout_frozen: dict, rb_by_trait: dict, n_boot: int) -> dict:
    """Within-condition Pearson per (trait x mode x arm) + joint-resample dr."""
    keep = ctx["keep_idx"]
    pass_a_dir = ctx["in_dir"] / I.PASS_A_PREFIX
    unique_layers = sorted({li for t in readout_frozen.values() for li in t.values()})
    grids: dict[int, GramRidge] = {}
    arm_targets: dict[int, dict[str, np.ndarray]] = {}
    for li in unique_layers:
        X = _layer_np(ctx["bundle"]["cx_last"], li)[keep]
        grids[li] = GramRidge(X)
        arm_targets[li] = {
            "avg10": _layer_np(ctx["vbar10"], li)[keep],
            "greedy": _layer_np(ctx["v_greedy"], li)[keep],
            "stoch1_old": _layer_np(ctx["bundle"]["v_x"], li)[keep],
            "stoch1_new": _layer_np(ctx["stoch1_new"], li)[keep],
        }
        logger.info("[monitoring] GramRidge factorized at L%d (n=%d)", li, X.shape[0])

    out: dict = {"arms": list(I.ARMS), "cells": {}}
    for trait, modes in readout_frozen.items():
        cells = S1.load_eval_cells(pass_a_dir, trait)
        assert cells, f"no pass_a cells for trait {trait}"
        rb = rb_by_trait[trait]
        for mode, li in modes.items():
            mat = S1.build_eval_matrix(cells, li, rb)
            if len(mat["y"]) < 3:
                out["cells"][f"{trait}__{mode}"] = {"skipped": True, "n": len(mat["y"])}
                continue
            x_by_arm = {}
            for arm in I.ARMS:
                pred_profile = grids[li].predict(arm_targets[li][arm], mat["c_last"])
                x_by_arm[arm] = pred_profile @ np.asarray(rb[li], dtype=np.float64)
            x_by_arm["pv_raw"] = np.asarray(mat["pv_raw"], dtype=np.float64)
            x_by_arm["oracle"] = np.asarray(mat["oracle"], dtype=np.float64)
            y = np.asarray(mat["y"], dtype=np.float64)

            # Per-condition r per arm over ONE shared kept-condition set.
            sel_mode = np.array([m == mode for m in mat["mode"]])
            conds = np.unique(mat["cond"][sel_mode]) if sel_mode.any() else np.array([])
            kept_conds, per_cond_r = [], {k: [] for k in x_by_arm}
            for c in conds:
                m = sel_mode & (mat["cond"] == c)
                yc = y[m]
                if len(yc) < 3 or float(np.std(yc)) < 1.0:
                    continue
                rs = {}
                ok = True
                for k, xv in x_by_arm.items():
                    xc = xv[m]
                    fin = np.isfinite(xc) & np.isfinite(yc)
                    if fin.sum() < 3 or float(np.std(xc[fin])) == 0.0:
                        ok = False
                        break
                    rs[k] = float(np.corrcoef(xc[fin], yc[fin])[0, 1])
                if not ok or any(not np.isfinite(v) for v in rs.values()):
                    continue
                kept_conds.append(int(c))
                for k, v in rs.items():
                    per_cond_r[k].append(v)
            cell: dict = {
                "layer": int(li),
                "n_conditions": len(kept_conds),
                "kept_conditions": kept_conds,
                "per_condition_r": {k: v for k, v in per_cond_r.items()},
            }
            if kept_conds:
                k = len(kept_conds)
                rng = np.random.default_rng(I.BOOT_SEED)
                idx = rng.integers(0, k, size=(n_boot, k))
                draws = {name: np.asarray(v)[idx].mean(1) for name, v in per_cond_r.items()}
                cell["point_r"] = {name: float(np.mean(v)) for name, v in per_cond_r.items()}
                cell["ci_r"] = {
                    name: [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]
                    for name, d in draws.items()
                }
                pairs = [
                    ("greedy", "avg10"),
                    ("stoch1_new", "avg10"),
                    ("stoch1_old", "avg10"),
                    ("greedy", "stoch1_new"),
                ]
                cell["joint_delta_r"] = {}
                for a, b in pairs:
                    d = draws[a] - draws[b]
                    cell["joint_delta_r"][f"{a}-{b}"] = {
                        "point": cell["point_r"][a] - cell["point_r"][b],
                        "lo": float(np.quantile(d, 0.025)),
                        "hi": float(np.quantile(d, 0.975)),
                    }
                # H3 flip check: raw-vs-map verdict under the SAME joint draws.
                vg = np.sign(draws["greedy"] - draws["pv_raw"])
                va = np.sign(draws["avg10"] - draws["pv_raw"])
                cell["h3_flip"] = {
                    "verdict_greedy_point": float(
                        np.sign(cell["point_r"]["greedy"] - cell["point_r"]["pv_raw"])
                    ),
                    "verdict_avg10_point": float(
                        np.sign(cell["point_r"]["avg10"] - cell["point_r"]["pv_raw"])
                    ),
                    "flip_prob": float(np.mean(vg != va)),
                }
            out["cells"][f"{trait}__{mode}"] = cell
            logger.info(
                "[monitoring] %s %s: %s",
                trait,
                mode,
                json.dumps(cell.get("point_r", {"skipped": True})),
            )
    return out


# ── bootstrap over per-context scalars (ONE gather) ──────────────────────────


def bootstrap_block(fits: dict, agreement: dict, ctx: dict, readout: list[int], n_boot: int):
    """95% CIs via ONE shared (n_boot, N) index gather (seed 0): pooled-R2
    ratios, paired arm R2 gaps, DV2 cosine means, DV4 median delta."""
    nk = ctx["keep_idx"].size
    rng = np.random.default_rng(I.BOOT_SEED)
    idx = rng.integers(0, nk, size=(n_boot, nk))
    out: dict = {"n_boot": n_boot, "seed": I.BOOT_SEED, "per_layer": {}}

    def _ci(d: np.ndarray) -> list[float]:
        return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]

    for li in readout:
        pc = fits["percontext"].get(f"last_L{li}")
        ag = agreement["per_layer"][f"L{li}"]
        entry: dict = {}
        r2_draws = {}
        for name, arrs in pc.items():
            sse = np.asarray(arrs["sse"])
            sst = np.asarray(arrs["sst"])
            draws = 1.0 - sse[idx].sum(1) / np.maximum(sst[idx].sum(1), 1e-12)
            r2_draws[name] = draws
            entry[f"r2__{name}"] = {
                "point": 1.0 - float(sse.sum()) / max(float(sst.sum()), 1e-12),
                "ci": _ci(draws),
            }
        for a, b in (("greedy", "avg10"), ("stoch1_new", "avg10"), ("stoch1_old", "avg10")):
            d = r2_draws[a] - r2_draws[b]
            entry[f"r2_gap__{a}-{b}"] = {
                "point": entry[f"r2__{a}"]["point"] - entry[f"r2__{b}"]["point"],
                "ci": _ci(d),
            }
        for key in (
            "cos_greedy_avg10",
            "cos_stoch1_old_avg10",
            "cos_stoch1_new_avg10",
        ):
            x = np.asarray(ag[key])
            entry[f"mean__{key}"] = {"point": float(x.mean()), "ci": _ci(x[idx].mean(1))}
        dv4 = np.asarray(ag["dv4_delta_ctx"])
        entry["dv4_median_delta"] = {
            "point": float(np.median(dv4)),
            "ci": _ci(np.median(dv4[idx], axis=1)),
        }
        out["per_layer"][f"L{li}"] = entry
    return out


# ── DV5 descriptives ─────────────────────────────────────────────────────────


def descriptives(ctx: dict) -> dict:
    """Response token-length distribution + truncation-at-1024 rate per arm."""
    gen_dir = ctx["root"] / "raw_completions"
    out = {}
    for arm in ("greedy", "stoch10"):
        recs = I.read_text_shards(gen_dir / arm, arm)
        toks = np.array([r["n_tokens"] for r in recs], dtype=np.float64)
        counts, edges = np.histogram(toks, bins=32)
        out[arm] = {
            "n_rollouts": len(recs),
            "tokens_mean": float(toks.mean()),
            "tokens_median": float(np.median(toks)),
            "tokens_p90": float(np.quantile(toks, 0.9)),
            "truncation_rate": float(np.mean([r["finish_reason"] == "length" for r in recs])),
            "empty_rate": float(np.mean([not r["text"].strip() for r in recs])),
            "tokens_hist": {"counts": counts.tolist(), "edges": edges.tolist()},
        }
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #1073 P4 fits + statistics.")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--n-boot", type=int, default=I.N_BOOT)
    parser.add_argument("--n-folds", type=int, default=I.N_FOLDS)
    args = parser.parse_args()

    I.phase("p4")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = load_ctx(args)
    n_layers = ctx["n_layers"]
    readout = I.readout_layer_set(n_layers)
    frozen = I.frozen_layers_map(n_layers)
    rb_dir = ctx["in_dir"] / I.RB_PREFIX
    rb_by_trait = {t: S1._load_rb(rb_dir, t, n_layers, ctx["hidden"]) for t in frozen}

    gate = repro_gate(ctx, device, args.smoke)
    fits = science_fits(ctx, device, readout)
    vlr = val_lambda_robustness(ctx, device, readout)
    agreement = target_agreement(ctx, readout, rb_by_trait, frozen)
    boot = bootstrap_block(fits, agreement, ctx, readout, args.n_boot)
    monit = monitoring(ctx, frozen, rb_by_trait, args.n_boot)
    desc = descriptives(ctx)

    res_dir = ctx["res_dir"]
    branch_map = ctx["branch_map"]
    arm_labels = {
        arm: {"continuity_only": arm == branch_map["continuity_only_arm"]} for arm in I.ARMS
    }
    common_block = {
        "n_kept": int(ctx["keep_idx"].size),
        "drops": ctx["drops"],
        "reduced_n_flag": ctx["reduced_n_flag"],
        "fingerprint": ctx["common_fp"],
        "fold_seed": I.FOLD_SEED_SCIENCE,
        "n_folds": args.n_folds,
        "common_index": ctx["keep_idx"].tolist(),
        "fold_of_context": ctx["fold_of"].tolist(),
    }
    meta = I.reproducibility_metadata({"script": "issue1073_fits", "device": device})

    I.write_json_compact(
        res_dir / "heldout_recon_arms.json",
        {
            "science_stoch_arm_by_probe_branch": branch_map,
            "arms": arm_labels,
            "common_set": common_block,
            "repro_gate": gate,
            "per_input_layer": {
                inp: {str(li): v for li, v in d.items()} for inp, d in fits["summary"].items()
            },
            "readout_percontext_last": fits["percontext"],
            "val_lambda_robustness": vlr,
            "bootstrap": boot,
            "metadata": meta,
        },
    )
    I.write_json_compact(
        res_dir / "target_agreement.json",
        {
            "science_stoch_arm_by_probe_branch": branch_map,
            "arms": arm_labels,
            "n_kept": int(ctx["keep_idx"].size),
            **agreement,
            "bootstrap": {li: boot["per_layer"][li] for li in boot["per_layer"]},
            "metadata": meta,
        },
    )
    I.write_json_compact(
        res_dir / "monitoring_arms.json",
        {
            "science_stoch_arm_by_probe_branch": branch_map,
            "arms": arm_labels,
            **monit,
            "metadata": meta,
        },
    )
    I.write_json_atomic(res_dir / "decode_descriptives.json", {"per_arm": desc, "metadata": meta})
    I.write_json_compact(
        res_dir / "exploratory_perdim_r2.json", {**fits["perdim"], "metadata": meta}
    )
    logger.info("[p4] wrote result JSONs under %s", res_dir)

    if not args.no_upload:
        import shutil
        import tempfile

        from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

        with tempfile.TemporaryDirectory(prefix="i1073_results_") as tmp:
            for p in sorted(res_dir.glob("*.json")):
                shutil.copy2(p, Path(tmp) / p.name)
            I.upload_folder_verified(
                Path(tmp),
                f"{I.HF_PREFIX}/eval_results",
                commit_message="issue1073: P4 result JSONs",
                allow_patterns=["*.json"],
            )
        upload_dir_sharded(
            ctx["root"] / "predictions",
            I.HF_DATA_REPO,
            f"{I.HF_PREFIX}/analysis_tensors/predictions",
            shard_glob="*.pt",
            delete_local=False,
        )
        import wandb

        run = wandb.init(
            project="issue1073",
            name=f"p4-fits{'-smoke' if args.smoke else ''}",
            config={"branch": branch_map["branch"], "n_kept": int(ctx["keep_idx"].size)},
        )
        hero_li = readout[len(readout) // 2]
        run.log(
            {
                f"r2_last_L{hero_li}_{arm}": fits["summary"]["last"][hero_li][arm]["r2_pooled"]
                for arm in I.ARMS
            }
        )
        run.finish()

    headline = {
        "branch": branch_map,
        "n_kept": int(ctx["keep_idx"].size),
        "r2_last_by_readout_layer": {
            str(li): {arm: fits["summary"]["last"][li][arm]["r2_pooled"] for arm in I.ARMS}
            for li in readout
        },
        "dv4_median_delta": {
            li: boot["per_layer"][li]["dv4_median_delta"] for li in boot["per_layer"]
        },
        "monitoring_point_r": {
            cell: v.get("point_r") for cell, v in monit["cells"].items() if isinstance(v, dict)
        },
        "descriptives": desc,
        "results_prefix": f"{I.HF_PREFIX}/eval_results",
    }
    I.write_sentinel(
        "epm:smoke-result" if args.smoke else "epm:results",
        json.dumps(headline)[:45000],
    )
    logger.info("[p4] sentinel written; P4 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
