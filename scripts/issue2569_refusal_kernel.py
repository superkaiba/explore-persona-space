#!/usr/bin/env python3
"""Task #2569 leg 9: kernel/range reading of the context→answer map on two refusal datasets.

Extends leg 8 (``issue2569_kernel_interpretation``): the effective kernel of the
banked n1m ridge map (row action ``vhat = v @ A + b``) is the set of read
directions with singular value below the squared-singular-mass cutoff; the
kernel share of a unit direction is its squared projection onto those read
directions. Leg 8 established the map's kernel holds 55% of dimensions at the
0.99 cutoff (random-direction expectation 0.55) and that persona directions sit
at 0.71-0.83. This leg asks where REFUSAL context differences sit.

Datasets (existing banked artifacts, no GPU):
  A. Minimal refusal pairs (#2617): one-word safety-valence swaps, 108 primary
     pairs (216 contexts) plus a 16-pair harmful-to-harmful verb-swap control
     cell added 2026-09-02; per-context judged refusal rates; v_C and v_A banked
     at L14/L19/L26.
  B. China politics pairs (#952 china_politics_topup): 42 pairs of a
     China-sensitive question vs an entity-swapped control, Qwen's own answers
     and Claude's answers teacher-forced through Qwen; slot banks at
     L2..L26 (this leg reads L14/L26; no L19 capture exists).

Reads: (1) per-pair kernel share of Δc vs four nulls (random directions, random
context pairs from the leg-8 LMSYS+WildChat capture sample, distance-matched
random pairs, within-arm pairs); (2) kernel/range decomposition of the mean Δc
decoded through the context SAE and the andyrdt SAE (L19); (3) transport
Δ̂a = Δc @ A vs the observed answer shift, plus refusal-axis projections and
their Spearman against judged refusal-rate changes; (4) kernel share of the
context-side refusal directions next to the leg-8 persona-direction shares.

Outputs: eval_results/issue_2569/weights/leg9/refusal_kernel_L{14,19,26}.json,
refusal_kernel_L19.md, figures/issue_2569/leg9_refusal_kernel.{png,pdf}.
CPU only; blocked GEMMs; every battery vectorized (leg-8 helpers reused).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
for p in (str(_REPO / "src"), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps bind BEFORE torch/numpy heavy use (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import issue2569_kernel_interpretation as KI  # noqa: E402
import issue2569_operator as OP  # noqa: E402

D = 3584
LAYERS = (19, 14, 26)  # L19 primary
MASSES = (0.999, 0.99, 0.90)
PRIMARY_MASS = 0.99
SEED = 25_690
N_NULL_DIRS = 10_000
N_NULL_PAIRS = 2_000
N_CAND_PAIRS = 200_000
N_BOOT = 2_000
N_BINS = 5
QUOTE_CHARS = 200
FLIP_GAP = 0.5
NONFLIP_GAP = 0.1
CONTROL_CLASS = "verb_harm"  # harmful-to-harmful verb swaps: the non-valence control cell

DL_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue2569_theory/leg9_dl")
LEG8_WORK = Path("/mnt/eps-data/thomasjiralerspong/wt-2569-kernel-work")

# ── pure helpers (unit-tested on small synthetics) ────────────────────────────────


def unit_rows(x: np.ndarray) -> np.ndarray:
    """Unit-normalize rows (fp64); zero rows raise (a zero pair delta is a data bug)."""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero-norm row in direction batch")
    return x / n


def bootstrap_median_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Median + percentile-bootstrap 95% CI over the pair axis (vectorized)."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("bootstrap_median_ci needs a non-empty 1-D array")
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite values in bootstrap input")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    meds = np.median(v[idx], axis=1)
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
        "mean": float(v.mean()),
    }


def sample_index_pairs(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """(k, 2) index pairs with i != j, sampled uniformly with replacement over pairs."""
    i = rng.integers(0, n, size=k)
    j = rng.integers(0, n - 1, size=k)
    j = np.where(j >= i, j + 1, j)
    return np.stack([i, j], axis=1)


def distance_matched_pairs(
    real_norms: np.ndarray,
    cand_norms: np.ndarray,
    n_bins: int = N_BINS,
    per_bin: int = N_NULL_PAIRS // N_BINS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """Candidate indices matched to the quantile bins of the real pair norms.

    Bins are the ``n_bins`` quantile bins of ``real_norms``; from each bin up to
    ``per_bin`` candidates whose norm falls inside are drawn without
    replacement. Returns (selected candidate indices, coverage record naming
    any under-filled bin).
    """
    rng = rng or np.random.default_rng(SEED)
    real = np.asarray(real_norms, dtype=np.float64)
    cand = np.asarray(cand_norms, dtype=np.float64)
    edges = np.quantile(real, np.linspace(0.0, 1.0, n_bins + 1))
    chosen: list[np.ndarray] = []
    coverage = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = np.flatnonzero((cand >= lo) & (cand <= hi if b == n_bins - 1 else cand < hi))
        take = min(per_bin, in_bin.size)
        if take > 0:
            chosen.append(rng.choice(in_bin, size=take, replace=False))
        coverage.append(
            {
                "bin": b,
                "lo": float(lo),
                "hi": float(hi),
                "available": int(in_bin.size),
                "taken": take,
            }
        )
    idx = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    return idx, {"bins": coverage, "n_matched": int(idx.size)}


def pair_norms_blocked(X: np.ndarray, pairs: np.ndarray, block: int = 20_000) -> np.ndarray:
    """Norms of X[i] - X[j] over (k, 2) index pairs, blocked (no k x d materialization)."""
    out = np.empty(pairs.shape[0], dtype=np.float64)
    for lo in range(0, pairs.shape[0], block):
        hi = min(lo + block, pairs.shape[0])
        d = X[pairs[lo:hi, 0]] - X[pairs[lo:hi, 1]]
        out[lo:hi] = np.linalg.norm(d, axis=1)
    return out


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine between two (n, d) batches (fp64)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.einsum("ij,ij->i", a, b)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / den


def transport_r2(pred: np.ndarray, obs: np.ndarray) -> dict:
    """Cross-pair R² of the predicted answer shifts against the observed ones.

    ``raw``: residual R² of ``obs`` against ``pred`` with no refit
    (1 - ||obs - pred||²_F / ||obs - mean(obs)||²_F). ``gain``: the same after
    one global scalar gain alpha = <pred, obs>_F / ||pred||²_F is applied to
    ``pred`` (ridge shrinkage makes the raw magnitudes conservative).
    """
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    ss_tot = float(np.sum((obs - obs.mean(axis=0)) ** 2))
    raw = 1.0 - float(np.sum((obs - pred) ** 2)) / ss_tot
    alpha = float(np.sum(pred * obs) / np.sum(pred * pred))
    gain = 1.0 - float(np.sum((obs - alpha * pred) ** 2)) / ss_tot
    return {"raw": raw, "gain_calibrated": gain, "gain": alpha}


def loo_axis_projections(deltas: np.ndarray, axis_member: np.ndarray) -> np.ndarray:
    """Leave-one-out unit-axis projections for the axis's own member pairs.

    ``axis_member`` marks rows of ``deltas`` that enter the axis (the flip
    pairs). For a member row, the axis is the mean of the OTHER member rows;
    for a non-member row, the full member mean. Returns the (n,) projections of
    each row on its unit axis.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    member = np.asarray(axis_member, dtype=bool)
    m = deltas[member]
    if m.shape[0] < 2:
        raise ValueError("need >= 2 axis member pairs for LOO")
    total = m.sum(axis=0)
    out = np.empty(deltas.shape[0], dtype=np.float64)
    full_axis = total / m.shape[0]
    full_axis_u = full_axis / np.linalg.norm(full_axis)
    member_rows = np.flatnonzero(member)
    for r in range(deltas.shape[0]):
        if member[r]:
            k = np.flatnonzero(member_rows == r)[0]
            ax = (total - m[k]) / (m.shape[0] - 1)
            ax = ax / np.linalg.norm(ax)
        else:
            ax = full_axis_u
        out[r] = float(deltas[r] @ ax)
    return out


def project_pairs_on_axis(pred: np.ndarray, deltas_obs: np.ndarray, member: np.ndarray) -> dict:
    """Predicted and observed unit-axis projections with LOO for member pairs."""
    member = np.asarray(member, dtype=bool)
    m = np.asarray(deltas_obs, dtype=np.float64)[member]
    total = m.sum(axis=0)
    n_m = m.shape[0]
    proj_pred = np.empty(pred.shape[0], dtype=np.float64)
    proj_obs = np.empty(pred.shape[0], dtype=np.float64)
    full_axis = total / n_m
    full_axis_u = full_axis / np.linalg.norm(full_axis)
    member_rows = np.flatnonzero(member)
    for r in range(pred.shape[0]):
        if member[r]:
            k = np.flatnonzero(member_rows == r)[0]
            ax = (total - m[k]) / (n_m - 1)
            ax = ax / np.linalg.norm(ax)
        else:
            ax = full_axis_u
        proj_pred[r] = float(np.asarray(pred[r], dtype=np.float64) @ ax)
        proj_obs[r] = float(np.asarray(deltas_obs[r], dtype=np.float64) @ ax)
    return {"pred": proj_pred, "obs": proj_obs}


def spear(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho + p (two-sided), NaN-free inputs required."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("non-finite input to spearman")
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(x.size)}


# ── loaders ───────────────────────────────────────────────────────────────────────


def _dl(manifest: dict, key: str) -> Path:
    return Path(manifest[key])


def load_svmp(dl: dict) -> dict:
    """Load the #2617 minimal-refusal-pair bank (vectors, pairs, judge, margins)."""
    mn = json.loads(_dl(dl, "issue2617_svmp/manifests/svmp_bank.json").read_text())
    judge = json.loads(
        _dl(dl, "issue2617_svmp/raw_completions/judge/judge_scores.json").read_text()
    )
    margins = json.loads(_dl(dl, "issue2617_svmp/analysis_tensors/margin/margins.json").read_text())
    vc_store = torch.load(
        _dl(dl, "issue2617_svmp/analysis_tensors/vc/vc_langow_bank.pt"),
        map_location="cpu",
        weights_only=False,
    )
    va_store = torch.load(
        _dl(dl, "issue2617_svmp/analysis_tensors/va/va_langow_query_svmp.pt"),
        map_location="cpu",
        weights_only=False,
    )
    assert vc_store["position"] == "context_end_last_token", vc_store["position"]
    layers = [int(x) for x in vc_store["layers"]]
    assert layers == [int(x) for x in va_store["layers"]], (layers, va_store["layers"])
    ctx_ids = list(vc_store["context_ids"])
    pos = {c: i for i, c in enumerate(ctx_ids)}
    vc = vc_store["vc"].to(torch.float64).numpy()  # (n_ctx, n_layers, d)
    assert vc.shape == (len(ctx_ids), len(layers), D), vc.shape

    va_rows = va_store["va_tail_incl"].to(torch.float64).numpy()
    assert not va_store["empty_rows"], va_store["empty_rows"]
    sums = np.zeros_like(vc)
    counts = np.zeros(len(ctx_ids), dtype=np.int64)
    for row_i, rec in enumerate(va_store["index"]):
        cid = rec["context_id"]
        sums[pos[cid]] += va_rows[row_i]
        counts[pos[cid]] += 1
    assert np.all(counts == 10), np.unique(counts)
    va = sums / counts[:, None, None]

    jpc = judge["per_context"]
    mpc = margins["per_context"]
    pairs = []
    for rec in mn["pairs"]:
        ra, rb = float(jpc[rec["a"]]["refusal_rate"]), float(jpc[rec["b"]]["refusal_rate"])
        hi, lo = (rec["a"], rec["b"]) if ra >= rb else (rec["b"], rec["a"])
        gap = abs(ra - rb)
        group = "flip" if gap >= FLIP_GAP else ("nonflip" if gap <= NONFLIP_GAP else "mid")
        pairs.append(
            {
                "pair_id": rec["pair_id"],
                "pair_class": rec["pair_class"],
                "hi": pos[hi],
                "lo": pos[lo],
                "gap": gap,
                "rate_hi": max(ra, rb),
                "rate_lo": min(ra, rb),
                "group": group,
                "is_control_cell": rec["pair_class"] == CONTROL_CLASS,
                "dmargin": float(mpc[hi]["margin"]) - float(mpc[lo]["margin"]),
            }
        )
    return {
        "layers": layers,
        "vc": vc,
        "va": va,
        "pairs": pairs,
        "judge_model": judge.get("judge_model"),
        "n_contexts": len(ctx_ids),
    }


def load_china(dl: dict) -> dict:
    """Load the #952 china_politics_topup bank at L14/L26 + judged refusal labels."""
    prov = json.loads(
        _dl(
            dl,
            "issue952_position_divergence/followups/china_politics_topup/analysis_tensors/"
            "provenance_china_politics_topup.json",
        ).read_text()
    )
    new_v = json.loads(
        _dl(
            dl,
            "issue952_position_divergence/followups/china_politics_topup/eval_results/"
            "china_topup_verification.json",
        ).read_text()
    )
    parent_v = json.loads(
        _dl(
            dl,
            "issue952_position_divergence/eval_results/issue_952/divergence_bank_verification.json",
        ).read_text()
    )
    labels: dict[str, dict] = {}
    for rec in new_v["pairs"]:
        labels[rec["pair_id"]] = rec
    for rec in parent_v["pairs"]:
        if rec.get("category") == "china_politics" and rec["pair_id"] not in labels:
            labels[rec["pair_id"]] = rec

    pair_ids = sorted({v["pair_id"] for v in prov["provenance"].values()})
    assert len(pair_ids) == 42, len(pair_ids)
    missing = [p for p in pair_ids if p not in labels]
    assert not missing, f"china pairs without judge labels: {missing[:5]}"

    arms = {}
    layers = []
    for lay in (14, 26):
        per_arm = {}
        for arm in ("own", "ext_plain"):
            sb = torch.load(
                _dl(
                    dl,
                    "issue952_position_divergence/followups/china_politics_topup/analysis_tensors/"
                    f"slots_bank_china_politics_topup_{arm}_L{lay}.pt",
                ),
                map_location="cpu",
                weights_only=False,
            )
            assert int(sb["layer"]) == lay
            sn = sb["slot_names"]
            ids = list(sb["ids"])
            s = sb["slots"]
            per_arm[arm] = {
                "ids": ids,
                "c_last": s[:, sn.index("c_last")].to(torch.float64).numpy(),
                "va": s[:, sn.index("full_mean_ext")].to(torch.float64).numpy(),
            }
        arms[lay] = per_arm
        layers.append(lay)

    ids = arms[14]["own"]["ids"]
    idx = {q: i for i, q in enumerate(ids)}
    rows = []
    for pid in pair_ids:
        div_id, ctl_id = f"{pid}_div", f"{pid}_ctl"
        assert div_id in idx and ctl_id in idx, pid
        lab = labels[pid]
        rows.append(
            {
                "pair_id": pid,
                "div": idx[div_id],
                "ctl": idx[ctl_id],
                "refusal_qwen_div": float(lab["divergent"]["refusal_qwen"]),
                "refusal_qwen_ctl": float(lab["control"]["refusal_qwen"]),
                "refusal_claude_div": float(lab["divergent"]["refusal_claude"]),
                "refusal_claude_ctl": float(lab["control"]["refusal_claude"]),
                "origin": prov["provenance"][div_id]["origin"],
            }
        )
    for lay in layers:
        for arm in ("own", "ext_plain"):
            assert arms[lay][arm]["ids"] == ids, (lay, arm)
        c_own, c_ext = arms[lay]["own"]["c_last"], arms[lay]["ext_plain"]["c_last"]
        cos = rowwise_cos(c_own, c_ext)
        assert float(cos.min()) > 0.999, (
            "prompt-side c_last differs across arms",
            float(cos.min()),
        )
    return {"layers": layers, "arms": arms, "pairs": rows, "ids": ids}


def load_capture_sample_layer(manifest: dict, layer: int) -> np.ndarray:
    """Leg-8 capture-sample rows at one layer: (n, d) fp64, deduped by ci."""
    paths = sorted((k, v) for k, v in manifest["paths"].items() if "final_token_capture" in k)
    xs, cis = [], []
    for _k, p in paths:
        b = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        layers = [int(x) for x in b["layers"]]
        col = layers.index(layer)
        xs.append(b["cx_last"][:, col, :].to(torch.float64).numpy())
        cis.append(np.asarray([int(c) for c in b["ci"]], dtype=np.int64))
    X = np.concatenate(xs, axis=0)
    ci = np.concatenate(cis)
    _, first = np.unique(ci, return_index=True)
    return X[np.sort(first)]


# ── per-layer analysis ────────────────────────────────────────────────────────────


def share_summary(shares: dict[float, np.ndarray], sel: np.ndarray, seed: int) -> dict:
    """Median + bootstrap CI of kernel shares per cutoff over the selected pairs."""
    out = {}
    for m in MASSES:
        out[str(m)] = bootstrap_median_ci(shares[m][sel], seed=seed)
    return out


def null_block(
    U: np.ndarray,
    masks: dict[float, np.ndarray],
    X_sample: np.ndarray,
    real_norms: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """The four nulls for one layer against one real-pair norm profile."""
    # (a) random directions
    dirs = rng.standard_normal((N_NULL_DIRS, D))
    sh_dir = KI.shares_at_masks(U, dirs, masks)
    # (b) random context pairs
    prs = sample_index_pairs(X_sample.shape[0], N_NULL_PAIRS, rng)
    dpair = X_sample[prs[:, 0]] - X_sample[prs[:, 1]]
    sh_pair = KI.shares_at_masks(U, dpair, masks)
    # (c) distance-matched random pairs
    cand = sample_index_pairs(X_sample.shape[0], N_CAND_PAIRS, rng)
    cand_norms = pair_norms_blocked(X_sample, cand)
    matched_idx, coverage = distance_matched_pairs(real_norms, cand_norms, rng=rng)
    if matched_idx.size:
        dmat = X_sample[cand[matched_idx, 0]] - X_sample[cand[matched_idx, 1]]
        sh_mat = KI.shares_at_masks(U, dmat, masks)
        matched = {str(m): bootstrap_median_ci(sh_mat[m], seed=SEED + 3) for m in MASSES}
        matched_primary = sh_mat[PRIMARY_MASS]
    else:
        matched = {str(m): None for m in MASSES}
        matched_primary = np.empty(0)
    return {
        "random_directions": {str(m): KI.null_share_stats(sh_dir[m]) for m in MASSES},
        "random_context_pairs": {
            str(m): bootstrap_median_ci(sh_pair[m], seed=SEED + 2) for m in MASSES
        },
        "distance_matched_pairs": matched,
        "distance_matched_coverage": coverage,
        "pair_norm_ranges": {
            "real_median": float(np.median(real_norms)),
            "real_q10_q90": [
                float(np.quantile(real_norms, 0.1)),
                float(np.quantile(real_norms, 0.9)),
            ],
            "random_pair_median": float(np.median(np.linalg.norm(dpair, axis=1))),
        },
        "_arrays": {
            "random_pair_primary": sh_pair[PRIMARY_MASS],
            "matched_primary": matched_primary,
        },
    }


def within_arm_shares(
    U: np.ndarray,
    masks: dict[float, np.ndarray],
    X: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    rng: np.random.Generator,
    cap: int = N_NULL_PAIRS,
) -> dict:
    """Null (d): pairs drawn within one arm (two hi-side or two lo-side contexts)."""
    out = {}
    prim = []
    for tag, idxs in (("within_hi_or_div", idx_a), ("within_lo_or_ctl", idx_b)):
        n = idxs.size
        all_pairs = np.array([(i, j) for k, i in enumerate(idxs) for j in idxs[k + 1 :]])
        if all_pairs.shape[0] > cap:
            sel = rng.choice(all_pairs.shape[0], size=cap, replace=False)
            all_pairs = all_pairs[sel]
        d = X[all_pairs[:, 0]] - X[all_pairs[:, 1]]
        sh = KI.shares_at_masks(U, d, masks)
        out[tag] = {str(m): bootstrap_median_ci(sh[m], seed=SEED + 4) for m in MASSES}
        out[tag]["n_pairs_used"] = int(all_pairs.shape[0])
        prim.append(sh[PRIMARY_MASS])
    out["_arrays"] = {"primary": np.concatenate(prim)}
    return out


def analyze_svmp_layer(
    layer: int, li: int, sv: dict, A: np.ndarray, U: np.ndarray, masks: dict
) -> dict:
    """All #2617 reads at one layer (shares, transport, refusal axis)."""
    pairs = sv["pairs"]
    vc, va = sv["vc"][:, li], sv["va"][:, li]
    hi = np.array([p["hi"] for p in pairs])
    lo = np.array([p["lo"] for p in pairs])
    dc = vc[hi] - vc[lo]
    dva = va[hi] - va[lo]
    groups = np.array([p["group"] for p in pairs])
    control = np.array([p["is_control_cell"] for p in pairs])
    primary = ~control
    flip = primary & (groups == "flip")
    nonflip = primary & (groups == "nonflip")
    gaps = np.array([p["gap"] for p in pairs])
    dmargin = np.array([p["dmargin"] for p in pairs])

    shares = KI.shares_at_masks(U, dc, masks)
    pred = dc @ A
    cos_map = rowwise_cos(pred, dva)
    cos_id = rowwise_cos(dc, dva)

    proj = project_pairs_on_axis(pred, dva, flip)
    sel_p = np.flatnonzero(primary)
    axis_spearman = {
        "pred_vs_rate_gap_primary": spear(proj["pred"][sel_p], gaps[sel_p]),
        "pred_vs_rate_gap_flip": spear(proj["pred"][flip], gaps[flip]),
        "pred_vs_rate_gap_nonflip": spear(proj["pred"][nonflip], gaps[nonflip]),
        "obs_vs_rate_gap_primary": spear(proj["obs"][sel_p], gaps[sel_p]),
        "pred_vs_opener_margin_delta_primary": spear(proj["pred"][sel_p], dmargin[sel_p]),
        "pred_sign_accuracy_flip": float(np.mean(proj["pred"][flip] > 0)),
        "pred_sign_accuracy_nonflip": float(np.mean(proj["pred"][nonflip] > 0)),
    }

    mean_flip_dir = unit_rows(dc[flip].mean(axis=0)[None, :])
    mean_dir_shares = KI.shares_at_masks(U, mean_flip_dir, masks)

    def _t(sel):
        return {
            "cos_map": bootstrap_median_ci(cos_map[sel], seed=SEED + 5),
            "cos_identity": bootstrap_median_ci(cos_id[sel], seed=SEED + 6),
            "r2": transport_r2(pred[sel], dva[sel]),
        }

    return {
        "n_pairs": {
            "primary": int(primary.sum()),
            "flip": int(flip.sum()),
            "nonflip": int(nonflip.sum()),
            "mid": int((primary & (groups == "mid")).sum()),
            "control_cell": int(control.sum()),
        },
        "kernel_share": {
            "flip": share_summary(shares, flip, SEED + 7),
            "nonflip": share_summary(shares, nonflip, SEED + 8),
            "primary_all": share_summary(shares, primary, SEED + 9),
            "control_cell": share_summary(shares, control, SEED + 10),
        },
        "transport": {"flip": _t(flip), "nonflip": _t(nonflip), "control_cell": _t(control)},
        "refusal_axis": {
            "definition": "mean over flip pairs of the observed answer shift (hi minus lo), leave-one-out for a flip pair's own projections",
            "projections_summary": {
                "pred_flip": bootstrap_median_ci(proj["pred"][flip], seed=SEED + 11),
                "obs_flip": bootstrap_median_ci(proj["obs"][flip], seed=SEED + 12),
                "pred_nonflip": bootstrap_median_ci(proj["pred"][nonflip], seed=SEED + 13),
                "obs_nonflip": bootstrap_median_ci(proj["obs"][nonflip], seed=SEED + 14),
            },
            "spearman": axis_spearman,
        },
        "mean_flip_direction_kernel_share": {str(m): float(mean_dir_shares[m][0]) for m in MASSES},
        "_arrays": {
            "shares_primary_flip": shares[PRIMARY_MASS][flip],
            "shares_primary_nonflip": shares[PRIMARY_MASS][nonflip],
            "shares_primary_control": shares[PRIMARY_MASS][control],
            "cos_map_flip": cos_map[flip],
            "cos_id_flip": cos_id[flip],
            "cos_map_nonflip": cos_map[nonflip],
            "cos_id_nonflip": cos_id[nonflip],
            "proj_pred": proj["pred"],
            "gaps": gaps,
            "flip": flip,
            "nonflip": nonflip,
            "dc_norms_primary": np.linalg.norm(dc[primary], axis=1),
            "dc": dc,
            "dva": dva,
            "hi": hi,
            "lo": lo,
            "primary": primary,
        },
    }


def analyze_china_layer(
    layer: int, ch: dict, sv_axis: dict, A: np.ndarray, U: np.ndarray, masks: dict
) -> dict:
    """All China reads at one banked layer (uses the #2617 axis at the same layer)."""
    arms = ch["arms"][layer]
    pairs = ch["pairs"]
    div = np.array([p["div"] for p in pairs])
    ctl = np.array([p["ctl"] for p in pairs])
    c = arms["own"]["c_last"]
    dc = c[div] - c[ctl]
    dva_own = arms["own"]["va"][div] - arms["own"]["va"][ctl]
    dva_ext = arms["ext_plain"]["va"][div] - arms["ext_plain"]["va"][ctl]

    shares = KI.shares_at_masks(U, dc, masks)
    pred = dc @ A
    cos_own = rowwise_cos(pred, dva_own)
    cos_ext = rowwise_cos(pred, dva_ext)
    cos_id_own = rowwise_cos(dc, dva_own)
    cos_id_ext = rowwise_cos(dc, dva_ext)

    # #2617 refusal axis at this layer (full flip-pair mean; no China pair is a member)
    axis = unit_rows(sv_axis["dva"][sv_axis["flip"]].mean(axis=0)[None, :])[0]
    proj_pred = pred @ axis
    proj_own = dva_own @ axis
    ref_div = np.array([p["refusal_qwen_div"] for p in pairs])
    ref_ctl = np.array([p["refusal_qwen_ctl"] for p in pairs])
    ref_diff = ref_div - ref_ctl

    mean_dir = unit_rows(dc.mean(axis=0)[None, :])
    mean_dir_shares = KI.shares_at_masks(U, mean_dir, masks)
    all_sel = np.ones(len(pairs), dtype=bool)

    return {
        "n_pairs": len(pairs),
        "kernel_share": {"all": share_summary(shares, all_sel, SEED + 15)},
        "transport": {
            "cos_map_vs_qwen_own_shift": bootstrap_median_ci(cos_own, seed=SEED + 16),
            "cos_map_vs_claude_shift": bootstrap_median_ci(cos_ext, seed=SEED + 17),
            "cos_identity_vs_qwen_own_shift": bootstrap_median_ci(cos_id_own, seed=SEED + 18),
            "cos_identity_vs_claude_shift": bootstrap_median_ci(cos_id_ext, seed=SEED + 19),
            "fraction_closer_to_qwen_own": float(np.mean(cos_own > cos_ext)),
            "r2_vs_qwen_own": transport_r2(pred, dva_own),
            "r2_vs_claude": transport_r2(pred, dva_ext),
        },
        "refusal_axis": {
            "definition": "the #2617 flip-pair answer-shift axis at this layer (unit); China pairs are never axis members",
            "spearman": {
                "pred_vs_refusal_qwen_div": spear(proj_pred, ref_div),
                "pred_vs_refusal_diff_div_minus_ctl": spear(proj_pred, ref_diff),
                "obs_own_vs_refusal_diff": spear(proj_own, ref_diff),
            },
            "pred_projection_summary": bootstrap_median_ci(proj_pred, seed=SEED + 20),
            "refused_binary_split": {
                "n_refused_div": int(np.sum(ref_div >= 50.0)),
                "pred_median_refused": float(np.median(proj_pred[ref_div >= 50.0]))
                if np.any(ref_div >= 50.0)
                else None,
                "pred_median_not_refused": float(np.median(proj_pred[ref_div < 50.0]))
                if np.any(ref_div < 50.0)
                else None,
            },
        },
        "mean_direction_kernel_share": {str(m): float(mean_dir_shares[m][0]) for m in MASSES},
        "_arrays": {
            "shares_primary": shares[PRIMARY_MASS],
            "cos_own": cos_own,
            "cos_ext": cos_ext,
            "cos_id_own": cos_id_own,
            "cos_id_ext": cos_id_ext,
            "proj_pred": proj_pred,
            "ref_div": ref_div,
            "ref_diff": ref_diff,
            "dc_norms": np.linalg.norm(dc, axis=1),
            "dc": dc,
            "div": div,
            "ctl": ctl,
            "c_last": c,
        },
    }


# ── L19 decomposition + decoding ──────────────────────────────────────────────────


def decompose_and_decode(
    sv_arr: dict,
    ch_mean_dc_by_layer: dict,
    U: np.ndarray,
    mask_primary: np.ndarray,
    args,
    leg8_doc: dict,
) -> dict:
    """Kernel/range split of the mean flip Δc (L19), decoded through both SAEs.

    Also projects both parts on the recomputed top-10 ignored/range covariance
    modes (the leg-8 read-side themes; leg-8 overlay readings attached as
    labels because the mode computation is deterministic).
    """
    ctx = KI.load_ctx_sae(args.sae_ctx)
    andy_dec = KI.load_andyrdt_decoder(args.andyrdt)
    labels_path = (
        args.repo_root
        / "eval_results/issue_1482/context_side_labels/descriptions_context_side.jsonl"
    )
    labels: dict[int, dict] = {}
    with open(labels_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if int(r["feat_id"]) >= 0:
                    labels[int(r["feat_id"])] = {"description": str(r.get("description", ""))[:240]}
    ctx_dec_unit = ctx["w_dec"] / np.maximum(
        np.linalg.norm(ctx["w_dec"], axis=1, keepdims=True), 1e-12
    )
    andy_dec_unit = andy_dec / np.maximum(np.linalg.norm(andy_dec, axis=1, keepdims=True), 1e-12)

    manifest = json.loads((args.leg8_work / "download_manifest.json").read_text())
    sigma_c, mean_c, _n = KI.load_sigma_c(KI._hf_local(manifest, "moments/gram_xx.pt"))
    ker_vals, ker_modes = KI.projected_cov_modes(U, mask_primary, sigma_c, KI.TOP_MODES)
    rng_vals, rng_modes = KI.projected_cov_modes(U, ~mask_primary, sigma_c, KI.TOP_MODES)
    overlay = leg8_doc.get("interpretation_overlay", {}).get("mode_readings", {})

    X, ci = KI.load_capture_sample(manifest)
    texts = KI.load_row_meta_texts(manifest, set(int(c) for c in ci.tolist()))

    def _split(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        d = np.asarray(d, dtype=np.float64)
        Uk = U[:, mask_primary]
        ker = (d @ Uk) @ Uk.T
        rng_part = d - ker
        share = float((ker @ ker) / (d @ d))
        return ker, rng_part, share

    def _decode_part(part: np.ndarray, tag: str) -> dict:
        pu = part / np.linalg.norm(part)
        ctx_top = KI.decode_direction(pu, ctx_dec_unit, None)
        andy_top = KI.decode_direction(pu, andy_dec_unit, labels)
        fids = np.asarray([e["feat_id"] for e in ctx_top], dtype=np.int64)
        acts = KI.encode_ctx_features(ctx, X.astype(np.float32), fids)
        sh_dummy = np.zeros(ctx["w_dec"].shape[0])
        naming = KI.naming_evidence_rows([int(f) for f in fids], acts, sh_dummy, ci, texts)
        for e, nm in zip(ctx_top, naming):
            e["top_contexts"] = nm["top_contexts"]
            e["no_activation_in_sample"] = nm["no_activation_in_sample"]
        mode_proj = {
            "ignored_modes_cos": [float(pu @ m) for m in ker_modes],
            "range_modes_cos": [float(pu @ m) for m in rng_modes],
        }
        return {
            "tag": tag,
            "ctx_sae_top5": ctx_top,
            "andyrdt_top5": andy_top,
            "covariance_mode_cos": mode_proj,
        }

    dc = sv_arr["dc"]
    flip = sv_arr["flip"]
    d_mean = dc[flip].mean(axis=0)
    ker, rng_part, share = _split(d_mean)
    per_pair_shares = []
    for r in np.flatnonzero(flip):
        _k, _r, s = _split(dc[r])
        per_pair_shares.append(s)

    out = {
        "mean_flip_delta": {
            "kernel_share_0p99": share,
            "norm": float(np.linalg.norm(d_mean)),
            "kernel_part": _decode_part(ker, "kernel_part"),
            "range_part": _decode_part(rng_part, "range_part"),
        },
        "per_pair_kernel_share_of_split": bootstrap_median_ci(
            np.asarray(per_pair_shares), seed=SEED + 21
        ),
        "mode_variance_shares": {
            "ignored": [float(v) for v in ker_vals],
            "range": [float(v) for v in rng_vals],
        },
        "leg8_mode_readings": overlay,
    }
    for lay, ch_dc_mean in ch_mean_dc_by_layer.items():
        out[f"china_mean_delta_L{lay}_note"] = (
            "kernel/range split shares reported in the per-layer json; SAE decoding is "
            "L19-only (both dictionaries are L19 dictionaries) and no China L19 capture exists"
        )
    return out


# ── rendering ─────────────────────────────────────────────────────────────────────


def render_figure(docs: dict[int, dict], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d19, d14, d26 = docs[19], docs[14], docs[26]
    sv19 = d19["svmp"]["_arrays"]
    ch26 = d26["china"]["_arrays"]
    ch14 = d14["china"]["_arrays"]
    n19 = d19["nulls"]["_arrays"]
    n26 = d26["nulls"]["_arrays"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5))
    c_flip, c_non, c_ctl = "#c0392b", "#2980b9", "#7f8c8d"
    c_null, c_own, c_ext = "#27ae60", "#8e44ad", "#e67e22"

    ax = axes[0, 0]
    data = [
        sv19["shares_primary_flip"],
        sv19["shares_primary_nonflip"],
        sv19["shares_primary_control"],
        d19["svmp_within"]["_arrays"]["primary"],
        n19["matched_primary"],
    ]
    bp = ax.boxplot(
        data,
        tick_labels=[
            "flip pairs",
            "non-flip pairs",
            "verb swaps\n(harmful-harmful)",
            "within-arm pairs",
            "matched random\npairs",
        ],
        patch_artist=True,
        showfliers=False,
    )
    for patch, col in zip(bp["boxes"], [c_flip, c_non, c_ctl, "#95a5a6", c_null]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    exp = d19["nulls"]["random_directions"][str(PRIMARY_MASS)]["mean"]
    ax.axhline(exp, color="k", ls="--", lw=1, label="random direction mean")
    ax.set_ylabel("kernel share of context difference")
    ax.set_title("Minimal refusal pairs, layer 19")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8, labelrotation=12)

    ax = axes[0, 1]
    data = [
        ch14["shares_primary"],
        d14["china_within"]["_arrays"]["primary"],
        ch26["shares_primary"],
        d26["china_within"]["_arrays"]["primary"],
        n26["matched_primary"],
    ]
    bp = ax.boxplot(
        data,
        tick_labels=[
            "pairs L14",
            "within-arm L14",
            "pairs L26",
            "within-arm L26",
            "matched random\npairs L26",
        ],
        patch_artist=True,
        showfliers=False,
    )
    for patch, col in zip(bp["boxes"], [c_flip, "#95a5a6", c_flip, "#95a5a6", c_null]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.axhline(
        d14["nulls"]["random_directions"][str(PRIMARY_MASS)]["mean"],
        color="k",
        ls="--",
        lw=1,
        label="random direction mean L14",
    )
    ax.axhline(
        d26["nulls"]["random_directions"][str(PRIMARY_MASS)]["mean"],
        color="k",
        ls=":",
        lw=1,
        label="random direction mean L26",
    )
    ax.set_ylabel("kernel share of context difference")
    ax.set_title("China politics pairs, layers 14 and 26")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelsize=8, labelrotation=12)

    ax = axes[0, 2]
    data = [
        sv19["cos_map_flip"],
        sv19["cos_id_flip"],
        sv19["cos_map_nonflip"],
        sv19["cos_id_nonflip"],
    ]
    bp = ax.boxplot(
        data,
        tick_labels=["map, flip", "identity, flip", "map, non-flip", "identity, non-flip"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, col in zip(bp["boxes"], [c_flip, c_ctl, c_non, c_ctl]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cosine, predicted vs observed answer shift")
    ax.set_title("Minimal refusal pairs, transport at layer 19")
    ax.tick_params(axis="x", labelsize=8, labelrotation=12)

    ax = axes[1, 0]
    flip, nonflip = sv19["flip"], sv19["nonflip"]
    ax.scatter(sv19["proj_pred"][flip], sv19["gaps"][flip], s=18, color=c_flip, label="flip pairs")
    ax.scatter(
        sv19["proj_pred"][nonflip], sv19["gaps"][nonflip], s=18, color=c_non, label="non-flip pairs"
    )
    ax.set_xlabel("predicted refusal-axis shift")
    ax.set_ylabel("observed refusal-rate change")
    ax.set_title("Minimal refusal pairs, layer 19")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.scatter(ch26["proj_pred"], ch26["ref_div"], s=20, color=c_own)
    ax.set_xlabel("predicted refusal-axis shift, layer 26")
    ax.set_ylabel("judged refusal on Qwen's own answer (0-100)")
    ax.set_title("China politics pairs")

    ax = axes[1, 2]
    data = [ch26["cos_own"], ch26["cos_ext"], ch26["cos_id_own"], ch26["cos_id_ext"]]
    bp = ax.boxplot(
        data,
        tick_labels=[
            "map vs Qwen shift",
            "map vs Claude shift",
            "identity vs Qwen",
            "identity vs Claude",
        ],
        patch_artist=True,
        showfliers=False,
    )
    for patch, col in zip(bp["boxes"], [c_own, c_ext, c_ctl, c_ctl]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("cosine, predicted vs observed answer shift")
    ax.set_title("China politics pairs, transport at layer 26")
    ax.tick_params(axis="x", labelsize=8, labelrotation=12)

    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"leg9_refusal_kernel.{ext}", dpi=200)
    plt.close(fig)


def _fmt_ci(rec: dict) -> str:
    return f"{rec['median']:.3f} [{rec['ci95'][0]:.3f}, {rec['ci95'][1]:.3f}]"


def render_md(docs: dict[int, dict], leg8_doc: dict, out_md: Path) -> None:
    d19 = docs[19]
    L: list[str] = []
    A = L.append
    A(
        "# Refusal context differences under the kernel/range reading of the context→answer map (task #2569, leg 9)"
    )
    A("")
    A(
        "**Setup and provenance.** Dataset A: minimal refusal pairs (#2617), 108 primary pairs of one-word"
    )
    A(
        "safety-valence swaps (216 single-turn contexts, empty system slot) plus a 16-pair harmful-to-harmful"
    )
    A(
        "verb-swap control cell; answers are Qwen2.5-7B-Instruct's own on-policy rollouts (10 draws per context,"
    )
    A(
        "temperature-sampled per the #2617 recipe), refusal judged per draw by claude-sonnet-4-5 (graded 0-100,"
    )
    A(
        "refused at 50). Dataset B: China politics pairs (#952 top-up), 42 pairs of a China-sensitive question vs"
    )
    A(
        "an entity-swapped control about another country (84 single-turn queries, default system slot); answer"
    )
    A(
        "states are teacher-forced Qwen states over Qwen's own answer (seed 42, n=1 per query) and over Claude's"
    )
    A("answer (n=1); refusal judged on each answer (3 graded draws, mean 0-100).")
    A("")
    A(
        "**Definitions.** *Kernel share* of a context difference: the squared fraction of the vector lying in the"
    )
    A(
        "map's low-gain read directions at a squared-singular-mass cutoff (0.99 primary). *Range part*: the"
    )
    A(
        "complement, the component the map reads at material gain. *Refusal axis*: the mean observed answer-state"
    )
    A("shift over #2617 flip pairs at a layer, leave-one-out for a flip pair's own score.")
    A("")
    ns = d19["nulls"]["random_directions"][str(PRIMARY_MASS)]
    A("## Headline kernel shares at the 0.99 cutoff (medians with bootstrap 95% CIs over pairs)")
    A("")
    A("| direction set | layer | kernel share | null |")
    A("|---|---|---|---|")
    sv = d19["svmp"]
    A(
        f"| minimal refusal pairs, flip (n={sv['n_pairs']['flip']}) | 19 | {_fmt_ci(sv['kernel_share']['flip'][str(PRIMARY_MASS)])} | random direction {ns['mean']:.3f} |"
    )
    A(
        f"| minimal refusal pairs, non-flip (n={sv['n_pairs']['nonflip']}) | 19 | {_fmt_ci(sv['kernel_share']['nonflip'][str(PRIMARY_MASS)])} | |"
    )
    A(
        f"| harmful-to-harmful verb swaps (n={sv['n_pairs']['control_cell']}) | 19 | {_fmt_ci(sv['kernel_share']['control_cell'][str(PRIMARY_MASS)])} | |"
    )
    A(
        f"| random context pairs | 19 | {_fmt_ci(d19['nulls']['random_context_pairs'][str(PRIMARY_MASS)])} | |"
    )
    mm = d19["nulls"]["distance_matched_pairs"][str(PRIMARY_MASS)]
    A(f"| distance-matched random pairs | 19 | {_fmt_ci(mm) if mm else 'no matched support'} | |")
    A(
        f"| within-arm pairs (hi side) | 19 | {_fmt_ci(d19['svmp_within']['within_hi_or_div'][str(PRIMARY_MASS)])} | |"
    )
    for lay in (14, 26):
        ch = docs[lay]["china"]
        nsl = docs[lay]["nulls"]["random_directions"][str(PRIMARY_MASS)]
        A(
            f"| China politics pairs (n={ch['n_pairs']}) | {lay} | {_fmt_ci(ch['kernel_share']['all'][str(PRIMARY_MASS)])} | random direction {nsl['mean']:.3f} |"
        )
        A(
            f"| China within-arm pairs | {lay} | {_fmt_ci(docs[lay]['china_within']['within_hi_or_div'][str(PRIMARY_MASS)])} | |"
        )
    A("")
    A(
        "### Context-side refusal directions next to the leg-8 persona directions (kernel share @0.99, L19 unless noted)"
    )
    A("")
    A("| direction | kernel share |")
    A("|---|---|")
    A(
        f"| mean flip-pair context difference (minimal refusal pairs, unit) | {sv['mean_flip_direction_kernel_share'][str(PRIMARY_MASS)]:.3f} |"
    )
    for lay in (14, 26):
        A(
            f"| mean sensitive-vs-control direction (China politics, unit, L{lay}) | {docs[lay]['china']['mean_direction_kernel_share'][str(PRIMARY_MASS)]:.3f} |"
        )
    for r in leg8_doc["persona_directions"]:
        A(f"| {r['direction']} | {r['share_0.99']:.3f} |")
    A(f"| random direction expectation | {ns['mean']:.3f} |")
    A("")
    A("## Transport: predicted vs observed answer shifts")
    A("")
    A("| set | layer | map cosine | identity cosine | raw R² | gain-calibrated R² |")
    A("|---|---|---|---|---|---|")
    for lay in LAYERS:
        svl = docs[lay]["svmp"]
        for grp in ("flip", "nonflip"):
            t = svl["transport"][grp]
            A(
                f"| minimal refusal pairs, {grp} | {lay} | {_fmt_ci(t['cos_map'])} | {_fmt_ci(t['cos_identity'])} | {t['r2']['raw']:.3f} | {t['r2']['gain_calibrated']:.3f} |"
            )
    for lay in (14, 26):
        t = docs[lay]["china"]["transport"]
        A(
            f"| China politics vs Qwen's own shift | {lay} | {_fmt_ci(t['cos_map_vs_qwen_own_shift'])} | {_fmt_ci(t['cos_identity_vs_qwen_own_shift'])} | {t['r2_vs_qwen_own']['raw']:.3f} | {t['r2_vs_qwen_own']['gain_calibrated']:.3f} |"
        )
        A(
            f"| China politics vs Claude's shift | {lay} | {_fmt_ci(t['cos_map_vs_claude_shift'])} | {_fmt_ci(t['cos_identity_vs_claude_shift'])} | {t['r2_vs_claude']['raw']:.3f} | {t['r2_vs_claude']['gain_calibrated']:.3f} |"
        )
    A("")
    A("## Refusal-axis reads")
    A("")
    sp = d19["svmp"]["refusal_axis"]["spearman"]
    A(
        f"- Minimal refusal pairs, L19: Spearman of the predicted refusal-axis shift against the observed"
    )
    A(
        f"  refusal-rate change over the 108 primary pairs: rho {sp['pred_vs_rate_gap_primary']['rho']:.3f}"
    )
    A(
        f"  (p {sp['pred_vs_rate_gap_primary']['p']:.2e}); flip-only rho {sp['pred_vs_rate_gap_flip']['rho']:.3f};"
    )
    A(
        f"  sign accuracy on flip pairs {sp['pred_sign_accuracy_flip']:.2f}. Against the teacher-forced opener"
    )
    A(f"  margin change: rho {sp['pred_vs_opener_margin_delta_primary']['rho']:.3f}.")
    for lay in (14, 26):
        cs = docs[lay]["china"]["refusal_axis"]["spearman"]
        A(
            f"- China politics, L{lay}: predicted refusal-axis shift vs judged refusal on Qwen's own answer:"
        )
        A(
            f"  rho {cs['pred_vs_refusal_qwen_div']['rho']:.3f} (p {cs['pred_vs_refusal_qwen_div']['p']:.2e});"
        )
        A(
            f"  vs the sensitive-minus-control refusal difference: rho {cs['pred_vs_refusal_diff_div_minus_ctl']['rho']:.3f}."
        )
    A("")
    A("## Kernel/range decomposition of the mean flip context difference (L19)")
    A("")
    dec = d19["decomposition"]
    A(
        f"Mean flip-pair context difference: kernel share {dec['mean_flip_delta']['kernel_share_0p99']:.3f} at the"
    )
    A(
        f"0.99 cutoff (per-pair split shares median {_fmt_ci(dec['per_pair_kernel_share_of_split'])})."
    )
    A("")
    for tag, part in (
        ("Range part (what the map reads)", "range_part"),
        ("Kernel part (what the map ignores)", "kernel_part"),
    ):
        blk = dec["mean_flip_delta"][part]
        A(f"### {tag}")
        A("")
        A("| dictionary | top features (|cos|, label or top activating context tail) |")
        A("|---|---|")
        rows = []
        for e in blk["ctx_sae_top5"]:
            q = (e.get("top_contexts") or [{}])[0].get("quote", "")
            rows.append(f"{e['feat_id']} ({e['cos']:+.3f}, “{q[:90]}”)")
        A(f"| context SAE | {'; '.join(rows)} |")
        rows = []
        for e in blk["andyrdt_top5"]:
            rows.append(
                f"{e['feat_id']} ({e['cos']:+.3f}{', ' + e['label'][:70] if e.get('label') else ''})"
            )
        A(f"| andyrdt SAE | {'; '.join(rows)} |")
        mp = blk["covariance_mode_cos"]
        top_rng = int(np.argmax(np.abs(mp["range_modes_cos"]))) + 1
        top_ker = int(np.argmax(np.abs(mp["ignored_modes_cos"]))) + 1
        A(
            f"| covariance modes | strongest range-mode cos: mode {top_rng} ({mp['range_modes_cos'][top_rng - 1]:+.3f}); "
            f"strongest ignored-mode cos: mode {top_ker} ({mp['ignored_modes_cos'][top_ker - 1]:+.3f}) |"
        )
        A("")
    A("Leg-8 readings for those covariance modes (range modes = what the map reads):")
    for kind in ("range", "ignored"):
        readings = dec.get("leg8_mode_readings", {}).get(kind) or []
        for i, rr in enumerate(readings[:3], 1):
            A(f"- {kind} mode {i}: {rr}")
    A("")
    A("## Notes and deviations")
    A("")
    for note in d19.get("notes", []):
        A(f"- {note}")
    A("")
    md = d19["metadata"]
    A("---")
    A(
        f"*Generated {md['timestamp_utc']} from commit `{md['git_commit'][:12]}`. Kernel = low-gain read"
    )
    A(
        "subspace of the fitted ridge map at the stated cutoff; no causal claim. Bootstrap CIs: 2,000"
    )
    A("percentile resamples over pairs.*")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L), encoding="utf-8")


def _strip_arrays(obj):
    """Deep-copy a doc dict dropping every key that starts with '_' (numpy payloads)."""
    if isinstance(obj, dict):
        return {k: _strip_arrays(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip_arrays(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


# ── main ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path, default=_REPO)
    ap.add_argument(
        "--map-root", type=Path, default=Path("/home/thomasjiralerspong/explore-persona-space")
    )
    ap.add_argument("--dl-manifest", type=Path, default=DL_ROOT / "leg9_manifest.json")
    ap.add_argument("--leg8-work", type=Path, default=LEG8_WORK)
    ap.add_argument(
        "--sae-ctx",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/issue2569_theory/sae_ctx/ae.pt"),
    )
    ap.add_argument(
        "--andyrdt",
        type=Path,
        default=Path(
            "/mnt/eps-data/thomasjiralerspong/huggingface-cache/hub/models--andyrdt--saes-qwen2.5-7b-instruct/"
            "snapshots/c37e53c4bb07127ad17ab88f28b93d4e87142e59/resid_post_layer_19/trainer_1/ae.pt"
        ),
    )
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    repo = args.repo_root
    out_dir = repo / "eval_results/issue_2569/weights/leg9"
    fig_dir = repo / "figures/issue_2569"
    dl = json.loads(args.dl_manifest.read_text())
    leg8_doc = json.loads(
        (repo / "eval_results/issue_2569/weights/leg8/kernel_interpretation_L19.json").read_text()
    )
    manifest = json.loads((args.leg8_work / "download_manifest.json").read_text())

    print("[leg9] loading datasets", flush=True)
    sv = load_svmp(dl)
    ch = load_china(dl)
    counts = {}
    for p in sv["pairs"]:
        key = ("control" if p["is_control_cell"] else "primary", p["group"])
        counts[key] = counts.get(key, 0) + 1
    print(f"[leg9] svmp pair counts: {counts}", flush=True)

    docs: dict[int, dict] = {}
    notes = [
        "The banked #2617 manifest now carries 124 pairs / 248 contexts: the 16-pair harmful-to-harmful "
        "verb-swap control cell landed 2026-09-02. The primary set here is the original 108 pairs; the "
        "control cell is reported as its own arm.",
        "Outcome groups recomputed from the banked judge file at the plan thresholds (gap >= 0.5 flip, "
        "<= 0.1 non-flip); counts are stated in the per-layer json and may differ by one or two pairs "
        "from the #2617 clean-result, which used a pre-control-cell judge snapshot.",
        "China politics pairs have no L19 capture; layers 14 and 26 carry the read with the banked maps "
        "at those layers, and the #2617 refusal axis is taken at the matching layer.",
        "v_A conventions: #2617 tail-inclusive mean over 10 on-policy draws; China teacher-forced "
        "extended-span mean (n=1 per query per arm). Both include the end-of-turn template tokens.",
    ]
    for lay in LAYERS:
        li = sv["layers"].index(lay)
        print(f"[leg9] L{lay}: operator + SVD", flush=True)
        payload = OP.load_banked_map(lay, root=args.map_root)
        A, _b = OP.row_operator(payload)
        U, s, _Vh = KI.svd_row_action(A)
        parts = KI.mass_partitions(s, MASSES)
        masks = {m: parts[m]["mask"] for m in MASSES}
        rng = np.random.default_rng(SEED + lay)

        print(f"[leg9] L{lay}: svmp reads", flush=True)
        svl = analyze_svmp_layer(lay, li, sv, A, U, masks)
        print(f"[leg9] L{lay}: nulls", flush=True)
        X_sample = load_capture_sample_layer(manifest, lay)
        nulls = null_block(U, masks, X_sample, svl["_arrays"]["dc_norms_primary"], rng)
        vc_l = sv["vc"][:, li]
        prim_rows = np.flatnonzero(svl["_arrays"]["primary"])
        sv_within = within_arm_shares(
            U,
            masks,
            vc_l,
            svl["_arrays"]["hi"][prim_rows],
            svl["_arrays"]["lo"][prim_rows],
            rng,
        )
        doc = {
            "layer": lay,
            "kernel_dims": {str(m): int(masks[m].sum()) for m in MASSES},
            "svd": {
                "sigma_max": float(s[0]),
                "tau_by_mass": {str(m): parts[m]["tau"] for m in MASSES},
            },
            "nulls": nulls,
            "svmp": svl,
            "svmp_within": sv_within,
        }
        if lay in ch["layers"]:
            print(f"[leg9] L{lay}: china reads", flush=True)
            chl = analyze_china_layer(lay, ch, svl["_arrays"], A, U, masks)
            c_all = chl["_arrays"]["c_last"]
            div_idx = np.array([p["div"] for p in ch["pairs"]])
            ctl_idx = np.array([p["ctl"] for p in ch["pairs"]])
            ch_within = within_arm_shares(U, masks, c_all, div_idx, ctl_idx, rng)
            doc["china"] = chl
            doc["china_within"] = ch_within
            # distance-matched null against the China pair norms too
            cand = sample_index_pairs(X_sample.shape[0], N_CAND_PAIRS, rng)
            cand_norms = pair_norms_blocked(X_sample, cand)
            m_idx, cov = distance_matched_pairs(chl["_arrays"]["dc_norms"], cand_norms, rng=rng)
            if m_idx.size:
                dmat = X_sample[cand[m_idx, 0]] - X_sample[cand[m_idx, 1]]
                shm = KI.shares_at_masks(U, dmat, masks)
                doc["china_distance_matched"] = {
                    str(m): bootstrap_median_ci(shm[m], seed=SEED + 22) for m in MASSES
                }
                doc["china_distance_matched"]["coverage"] = cov
            else:
                doc["china_distance_matched"] = {"coverage": cov}
        if lay == 19:
            print("[leg9] L19: decomposition + decoding", flush=True)
            ch_means = {}
            for cl in ch["layers"]:
                arms = ch["arms"][cl]
                divv = np.array([p["div"] for p in ch["pairs"]])
                ctlv = np.array([p["ctl"] for p in ch["pairs"]])
                ch_means[cl] = (arms["own"]["c_last"][divv] - arms["own"]["c_last"][ctlv]).mean(
                    axis=0
                )
            doc["decomposition"] = decompose_and_decode(
                svl["_arrays"], ch_means, U, masks[PRIMARY_MASS], args, leg8_doc
            )
            doc["notes"] = notes
        doc["metadata"] = {
            "git_commit": _git_commit(repo),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "map_payload": str(payload.path),
            "selected_lambda": payload.selected_lambda,
            "seed": SEED,
            "n_null_dirs": N_NULL_DIRS,
            "n_null_pairs": N_NULL_PAIRS,
            "n_boot": N_BOOT,
            "capture_sample_rows": int(X_sample.shape[0]),
        }
        docs[lay] = doc

    out_dir.mkdir(parents=True, exist_ok=True)
    for lay, doc in docs.items():
        (out_dir / f"refusal_kernel_L{lay}.json").write_text(
            json.dumps(_strip_arrays(doc), indent=1, ensure_ascii=False)
        )
    render_figure(docs, fig_dir)
    render_md(docs, leg8_doc, out_dir / "refusal_kernel_L19.md")
    print(f"[leg9] wrote {out_dir} + figures", flush=True)


if __name__ == "__main__":
    main()
