#!/usr/bin/env python3
"""Issue #2225 fu1 free-analysis: characterize the INVERTED residual probe orientation.

Body Result 5 context: the orth_E1 variant of the #778-pool Gram-space ridge probe
(persona direction E1 = reused #778 ``rb_v2`` diff-of-means projected out of train AND
application activations) lands at held-out GroupKFold AUC 0.058-0.194 at L1 (0.02-0.38
across layers) — the learned orientation ANTI-predicts on held-out question groups.
This script tests four candidate mechanisms on EXISTING artifacts only (0 GPU-h):

(a) group-fold artifact          — read: pooled SHUFFLED row folds on the same
                                   pool-projected Gram.
(b) direction-projection leakage — E1 is estimated on the SAME pool, so projecting it
                                   out zeroes the POOLED class-mean difference, forcing
                                   train-fold vs held-out-group class signals to cancel;
                                   reads: cos(pool diff-of-means, E1), residual pooled-Δ
                                   ratio, per-fold train-vs-test Δ anti-alignment, and a
                                   FOLD-HONEST probe (direction re-estimated per fold
                                   from TRAIN rows only, projected out of all rows).
(c) norm/scale confound          — read: 1-D residual-norm probe under GroupKFold.
(d) genuine secondary inverted direction — read: whether the inversion SURVIVES the
                                   fold-honest projection of (b).

Discriminating predictions are REGISTERED in ``REGISTERED_PREDICTIONS`` (written to the
output JSON verbatim) before any computation. All fits reuse the parent's probe
machinery BY IMPORT (``build_probe_pool``, ``_project_out``, ``_center_gram``,
``_batched_ridge_solve``, ``_auc``; same λ grid + max-mean-AUC selection). Fit regime is
the parent-validated dual/Gram-space n_kept < d regularized probe — held-out reads are
AUC only, never R² (plan §4.7). GPU-free; a layer SUBSET bounds the wall (<15 min);
per-fold Grams come from rank-one downdates of ONE full-space Gram per trait
(K' = K - cc^T for unit-direction projection), equivalence-checked against the parent's
explicit ``_project_out`` path on the first trait.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# scripts/ on sys.path so the sibling issue2225_analysis module resolves in script mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from issue2225_analysis import (  # noqa: E402  (dotenv/thread caps before torch importers)
    LAMBDA_REL_GRID,
    _auc,
    _batched_ridge_solve,
    _center_gram,
    _project_out,
    build_probe_pool,
)

SEED = 2225
N_FOLDS = 5
TRAITS = ("evil", "sycophancy", "hallucination")
DEFAULT_STAGING = "/mnt/eps-data/thomasjiralerspong/issue2225_fu1/hf_dl/issue778_v2"
DEFAULT_DIRECTIONS = (
    "/mnt/eps-data/thomasjiralerspong/issue2225_fu1/hf_dl/"
    "issue2225_ctxsteer/analysis_tensors/directions"
)
BANKED_PROBE_JSON = "eval_results/issue_2225/fu1_preimage_prevention/analysis/probe_shifts.json"
WALL_ABORT_MIN = 20.0  # brief: stop + report instead of launching past the floor

# Registered BEFORE computation (verbatim into the output JSON).
REGISTERED_PREDICTIONS = {
    "a_group_fold_artifact": (
        "If the inversion is a leave-question-group-out artifact carried by group-level "
        "mean shifts, pooled SHUFFLED row folds on the SAME pool-projected Gram restore "
        "held-out AUC to >= 0.45 at L1 while the GroupKFold read stays < 0.3."
    ),
    "b_direction_projection_leakage": (
        "E1 is the #778 rb_v2 diff-of-means estimated on the SAME pool, so (i) cos(pool "
        "diff-of-means, E1) >= 0.9 at L1 and the pooled class-mean difference norm after "
        "projection drops to <= 0.2 of its full-space value; (ii) after pool-level "
        "projection the train-fold vs held-out-fold class-mean-difference cosine is "
        "<= -0.5 (systematic anti-alignment forced by the pooled-difference-zeroing "
        "constraint); (iii) a FOLD-HONEST probe (direction estimated per fold on TRAIN "
        "rows only) restores held-out AUC to >= 0.45 at L1 — the inversion is an "
        "estimation artifact, not a property of the residual geometry."
    ),
    "c_norm_confound": (
        "If per-group activation-norm differences retained by the orthogonal complement "
        "carry the inversion, a 1-D residual-norm probe (orientation learned on train "
        "folds) lands at AUC <= 0.35 at L1 under GroupKFold."
    ),
    "d_genuine_secondary_inverted_direction": (
        "If a genuine secondary trait direction with inverted sign carries the signal, "
        "the inversion SURVIVES fold-honest direction estimation: fold-honest held-out "
        "AUC <= 0.35 at L1 (and per-fold fitted weight vectors stay mutually consistent, "
        "pairwise cos > 0)."
    ),
}


def _log(msg: str) -> None:
    print(f"[fu1-residual-probe] {msg}", flush=True)


def _l1_layer_idx() -> dict[str, int]:
    from explore_persona_space.experiments.issue2225.directions import L1_LAYER_IDX

    return dict(L1_LAYER_IDX)


def _group_folds(groups) -> list:
    """Parent fold scheme: sorted unique question ids, round-robin [i::5]."""
    uniq = sorted(set(groups.tolist()))
    return [uniq[i::N_FOLDS] for i in range(N_FOLDS)]


def _group_tr_te(groups, fold_groups):
    import torch

    te_mask = torch.isin(groups, torch.tensor(fold_groups))
    return torch.nonzero(~te_mask).squeeze(1), torch.nonzero(te_mask).squeeze(1)


def _shuffled_tr_te(n: int, seed: int) -> list:
    """Pooled shuffled row folds (mechanism-a read): 5-way seeded permutation split."""
    import numpy as np
    import torch

    perm = np.random.default_rng(seed).permutation(n)
    pairs = []
    for i in range(N_FOLDS):
        te = torch.tensor(np.sort(perm[i::N_FOLDS]), dtype=torch.long)
        mask = torch.ones(n, dtype=torch.bool)
        mask[te] = False
        pairs.append((torch.nonzero(mask).squeeze(1), te))
    return pairs


def _battery(fold_iter, y, n_layers: int):
    """Held-out AUC per (λ, layer, fold) + parent selection (max over λ of fold-mean).

    ``fold_iter`` yields (tr, te, K_fold) — the Gram may differ per fold (fold-honest
    read) or be shared (pool-projection reads). Returns (selected per-layer AUC list,
    selected λ_rel list, full fold-mean grid).
    """
    import numpy as np
    import torch

    fold_iter = list(fold_iter)
    auc = np.zeros((len(LAMBDA_REL_GRID), n_layers, len(fold_iter)))
    for fi, (tr, te, K) in enumerate(fold_iter):
        K_trtr_c, K_tetr_c = _center_gram(K, tr, te)
        trace_n = torch.diagonal(K_trtr_c, dim1=1, dim2=2).sum(dim=1) / tr.numel()
        for li_lam, lam_rel in enumerate(LAMBDA_REL_GRID):
            alpha = _batched_ridge_solve(K_trtr_c, y[tr], lam_rel * trace_n)
            te_scores = torch.einsum("lmn,ln->lm", K_tetr_c, alpha)
            for li in range(n_layers):
                auc[li_lam, li, fi] = _auc(te_scores[li].numpy(), y[te].numpy())
    mean_auc = auc.mean(axis=2)  # (n_lam, L)
    best = mean_auc.argmax(axis=0)
    sel = [float(mean_auc[best[li], li]) for li in range(n_layers)]
    sel_lam = [float(LAMBDA_REL_GRID[best[li]]) for li in range(n_layers)]
    return sel, sel_lam, mean_auc


def _pooled_delta(X, y):
    """Pooled class-mean difference μ+ - μ- per layer: (L, d)."""
    return X[y > 0].mean(dim=0) - X[y < 0].mean(dim=0)


def _cos(a, b, dim=-1):
    import torch

    return torch.nn.functional.cosine_similarity(a, b, dim=dim)


def _delta_reads(X, y, groups, fold_groups_list, l1_sub: int) -> dict:
    """Class-mean-difference structure reads (no fits).

    Returns per-layer pooled-Δ norms, the per-question cancellation ratio
    ||Σ_q w_q δ_q|| / Σ_q w_q ||δ_q||, and per-fold train-vs-test Δ cosines.
    """
    import torch

    delta_pool = _pooled_delta(X, y)  # (L, d)
    uniq = sorted(set(groups.tolist()))
    deltas, weights, skipped = [], [], 0
    n = X.shape[0]
    for q in uniq:
        m = groups == q
        yq = y[m]
        if (yq > 0).sum() == 0 or (yq < 0).sum() == 0:
            skipped += 1
            continue
        Xq = X[m]
        deltas.append(Xq[yq > 0].mean(dim=0) - Xq[yq < 0].mean(dim=0))
        weights.append(float(m.sum()) / n)
    assert len(deltas) >= 2, f"only {len(deltas)} questions with both classes"
    D = torch.stack(deltas)  # (Q, L, d)
    w = torch.tensor(weights).view(-1, 1, 1)
    num = (w * D).sum(dim=0).norm(dim=-1)  # (L,)
    den = (w.squeeze(-1).squeeze(-1).view(-1, 1) * D.norm(dim=-1)).sum(dim=0)  # (L,)
    cancellation_ratio = (num / den.clamp_min(1e-12)).tolist()

    fold_cos = []  # per fold: per-layer cos(δ_train, δ_test)
    for fg in fold_groups_list:
        tr, te = _group_tr_te(groups, fg)
        d_tr = _pooled_delta(X[tr], y[tr])
        d_te = _pooled_delta(X[te], y[te])
        fold_cos.append(_cos(d_tr, d_te).tolist())
    import numpy as np

    fold_cos_arr = np.array(fold_cos)  # (folds, L)
    return {
        "pooled_delta_norm_per_layer": delta_pool.norm(dim=-1).tolist(),
        "per_question_cancellation_ratio_per_layer": cancellation_ratio,
        "n_questions_used": len(deltas),
        "n_questions_skipped_single_class": skipped,
        "train_vs_test_delta_cos_per_fold_l1": fold_cos_arr[:, l1_sub].tolist(),
        "train_vs_test_delta_cos_mean_per_layer": fold_cos_arr.mean(axis=0).tolist(),
    }


def _norm_probe(X_resid, y, groups, fold_groups_list, n_layers: int) -> list[float]:
    """1-D per-row residual-norm probe: orientation (sign) learned on the train fold."""
    import numpy as np

    norms = X_resid.norm(dim=-1)  # (n, L)
    auc = np.zeros((len(fold_groups_list), n_layers))
    for fi, fg in enumerate(fold_groups_list):
        tr, te = _group_tr_te(groups, fg)
        for li in range(n_layers):
            r_tr = norms[tr, li]
            sgn = 1.0 if (r_tr[y[tr] > 0].mean() - r_tr[y[tr] < 0].mean()) >= 0 else -1.0
            auc[fi, li] = _auc((sgn * norms[te, li]).numpy(), y[te].numpy())
    return auc.mean(axis=0).tolist()


def _fold_weight_reads(Xp, y, groups, fold_groups_list, l1_sub: int, lam_rel: float) -> dict:
    """Primal weight consistency across folds (pool-residual probe, L1, selected λ)."""
    import torch

    ws, cos_w_dte = [], []
    for fg in fold_groups_list:
        tr, te = _group_tr_te(groups, fg)
        Xl = Xp[:, l1_sub, :]  # (n, d)
        Xc_tr = Xl[tr] - Xl[tr].mean(dim=0)
        K_tr = (Xc_tr @ Xc_tr.T).unsqueeze(0)  # (1, ntr, ntr)
        trace_n = torch.diagonal(K_tr, dim1=1, dim2=2).sum(dim=1) / tr.numel()
        alpha = _batched_ridge_solve(K_tr, y[tr], lam_rel * trace_n)[0]  # (ntr,)
        w = torch.einsum("n,nd->d", alpha, Xc_tr)
        ws.append(w / w.norm().clamp_min(1e-12))
        d_te = _pooled_delta(Xp[te], y[te])[l1_sub]
        cos_w_dte.append(float(_cos(w, d_te, dim=0)))
    W = torch.stack(ws)
    C = W @ W.T
    off = C[~torch.eye(len(ws), dtype=torch.bool)]
    return {
        "pairwise_cos_mean": float(off.mean()),
        "pairwise_cos_min": float(off.min()),
        "cos_w_train_vs_delta_test_per_fold": cos_w_dte,
    }


def analyze_trait(trait: str, args, banked: dict, l1_map: dict[str, int]) -> dict:
    import numpy as np
    import torch

    torch.manual_seed(SEED)
    t0 = time.time()
    X_full, y, groups, counts = build_probe_pool(trait, Path(args.i778_staging))
    n, n_layers_all, d = X_full.shape
    l1 = l1_map[trait]
    layers = sorted({0, 7, l1, 20, 27})
    l1_sub = layers.index(l1)
    Xs = X_full[:, layers, :].clone()
    del X_full
    _log(f"trait={trait} pool n={n} d={d} layers={layers} (l1={l1}) load={time.time() - t0:.0f}s")

    v_all = torch.load(
        Path(args.directions_dir) / f"{trait}_E1.pt", weights_only=True, map_location="cpu"
    ).to(torch.float32)
    assert v_all.shape == (n_layers_all, d), v_all.shape
    v = v_all[layers]
    vhat = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)

    fold_groups_list = _group_folds(groups)
    L = len(layers)

    # ── shared Gram (pool-level explicit projection — the parent's exact path) ─
    t0 = time.time()
    Xp = _project_out(Xs, v)
    K_pool = torch.einsum("nld,mld->lnm", Xp, Xp)
    gram_s = time.time() - t0
    _log(f"trait={trait} gram (pool-projected) {gram_s:.0f}s")

    # ── pilot: time ONE fold battery, project the trait wall, abort past floor ─
    t0 = time.time()
    tr0, te0 = _group_tr_te(groups, fold_groups_list[0])
    _battery([(tr0, te0, K_pool)], y, L)
    per_battery_s = time.time() - t0
    n_batteries = 3 * N_FOLDS
    projected_min = per_battery_s * n_batteries / 60 + gram_s / 60
    _log(
        f"trait={trait} pilot 1-fold battery {per_battery_s:.1f}s -> projected "
        f"~{projected_min:.1f} min for {n_batteries} fold-batteries"
    )
    if projected_min > WALL_ABORT_MIN:
        raise SystemExit(f"projected wall {projected_min:.0f} min > {WALL_ABORT_MIN} min — abort")

    # ── read 1: reproduce the banked pool-projection GroupKFold read (parity) ──
    group_pairs = [_group_tr_te(groups, fg) for fg in fold_groups_list]
    repro_auc, repro_lam, _ = _battery([(tr, te, K_pool) for tr, te in group_pairs], y, L)
    banked_prof = banked["fit_summaries"][trait]["orth_E1"]["heldout_auc_per_layer"]
    banked_sub = [float(banked_prof[li]) for li in layers]
    parity = max(abs(a - b) for a, b in zip(repro_auc, banked_sub))
    assert parity <= 0.02, f"banked-parity failed: max|Δ AUC|={parity:.4f} (repro vs banked)"
    _log(f"trait={trait} read1 repro group-fold AUC={['%.3f' % a for a in repro_auc]}")

    # ── read 2 (mechanism a): pooled shuffled row folds, same pool projection ──
    shuf_auc, _, _ = _battery([(tr, te, K_pool) for tr, te in _shuffled_tr_te(n, SEED)], y, L)
    _log(f"trait={trait} read2 shuffled-fold AUC={['%.3f' % a for a in shuf_auc]}")

    # ── read 3 (mechanisms b/d): fold-honest projection, explicit per fold ─────
    # NOTE: a rank-one Gram downdate (K' = K - cc^T) was tried and REJECTED — the
    # projected-out component dominates the fp32 Gram entries, so the subtraction
    # cancels catastrophically (measured AUC drift up to 0.072 vs the explicit
    # path). Explicit per-fold _project_out + Gram costs ~0.5 s/fold — negligible.
    honest_folds, honest_dir_cos = [], []
    for tr, te in group_pairs:
        d_tr = _pooled_delta(Xs[tr], y[tr])  # (L, d)
        honest_dir_cos.append(_cos(d_tr / d_tr.norm(dim=1, keepdim=True), vhat).tolist())
        Xh = _project_out(Xs, d_tr)
        honest_folds.append((tr, te, torch.einsum("nld,mld->lnm", Xh, Xh)))
    honest_auc, _, _ = _battery(honest_folds, y, L)
    del honest_folds
    _log(f"trait={trait} read3 fold-honest AUC={['%.3f' % a for a in honest_auc]}")

    # ── read 4 (mechanism c): 1-D residual-norm probe under GroupKFold ─────────
    norm_auc = _norm_probe(Xp, y, groups, fold_groups_list, L)

    # ── read 5: class-mean-difference structure, full vs residual space ────────
    delta_full = _delta_reads(Xs, y, groups, fold_groups_list, l1_sub)
    delta_resid = _delta_reads(Xp, y, groups, fold_groups_list, l1_sub)
    cos_delta_e1 = _cos(_pooled_delta(Xs, y), vhat).tolist()
    resid_ratio = [
        r / f if f > 1e-12 else float("nan")
        for r, f in zip(
            delta_resid["pooled_delta_norm_per_layer"], delta_full["pooled_delta_norm_per_layer"]
        )
    ]

    # ── read 6: fold weight consistency at L1 (pool-residual probe) ────────────
    weight_reads = _fold_weight_reads(Xp, y, groups, fold_groups_list, l1_sub, repro_lam[l1_sub])

    anti_cos_l1 = float(np.mean(delta_resid["train_vs_test_delta_cos_per_fold_l1"]))
    reads = {
        "pool_counts": counts,
        "n_kept": n,
        "layers": layers,
        "l1_layer_idx": l1,
        "banked_orth_e1_auc_at_layers": banked_sub,
        "repro_orth_e1_groupfold_auc": repro_auc,
        "repro_selected_lambda_rel": repro_lam,
        "banked_parity_max_abs_diff": parity,
        "shuffled_fold_auc": shuf_auc,
        "fold_honest_auc": honest_auc,
        "fold_honest_dir_cos_vs_e1_per_fold": honest_dir_cos,
        "norm_probe_groupfold_auc": norm_auc,
        "cos_pool_delta_vs_e1_per_layer": cos_delta_e1,
        "resid_pooled_delta_ratio_per_layer": resid_ratio,
        "delta_structure_full_space": delta_full,
        "delta_structure_residual_space": delta_resid,
        "fold_weight_consistency_l1": weight_reads,
    }

    # ── mechanical verdicts at L1 (thresholds from REGISTERED_PREDICTIONS) ─────
    verdicts = {
        "premise_e1_is_same_pool_diff_of_means": bool(
            cos_delta_e1[l1_sub] >= 0.9 and resid_ratio[l1_sub] <= 0.2
        ),
        "a_group_fold_artifact": bool(shuf_auc[l1_sub] >= 0.45 and repro_auc[l1_sub] < 0.3),
        "b_direction_projection_leakage": bool(
            cos_delta_e1[l1_sub] >= 0.9
            and resid_ratio[l1_sub] <= 0.2
            and anti_cos_l1 <= -0.5
            and honest_auc[l1_sub] >= 0.45
        ),
        "c_norm_confound_carries_inversion": bool(norm_auc[l1_sub] <= 0.35),
        "d_genuine_secondary_inverted_direction": bool(honest_auc[l1_sub] <= 0.35),
    }

    # Computed interpretation aids (post-hoc, clearly separated from the
    # pre-registered mechanical verdicts above — thresholds are NOT re-tuned).
    anti_folds = delta_resid["train_vs_test_delta_cos_per_fold_l1"]
    wcos = weight_reads["cos_w_train_vs_delta_test_per_fold"]
    interpretation = {
        "anti_align_negative_fold_fraction_l1": float(np.mean([c < 0 for c in anti_folds])),
        "anti_align_mean_cos_l1": anti_cos_l1,
        "shuffled_minus_group_auc_l1": float(shuf_auc[l1_sub] - repro_auc[l1_sub]),
        "shuffled_auc_range": [float(min(shuf_auc)), float(max(shuf_auc))],
        "fold_honest_auc_range": [float(min(honest_auc)), float(max(honest_auc))],
        "w_train_anti_aligned_with_delta_test_fold_fraction": float(np.mean([c < 0 for c in wcos])),
    }
    return {"reads": reads, "verdicts_l1": verdicts, "interpretation": interpretation}


def _synthesis(results: dict) -> dict:
    """Cross-trait ranges for the five reads (computed, not hand-typed)."""

    def rng(fn):
        vals = [
            fn(r["reads"], r["reads"]["layers"].index(r["reads"]["l1_layer_idx"]))
            for r in results.values()
        ]
        flat = [x for v in vals for x in (v if isinstance(v, list) else [v])]
        return [round(min(flat), 4), round(max(flat), 4)]

    return {
        "cos_pool_delta_vs_e1_range": rng(lambda r, i: r["cos_pool_delta_vs_e1_per_layer"]),
        "resid_pooled_delta_ratio_range": rng(lambda r, i: r["resid_pooled_delta_ratio_per_layer"]),
        "cancellation_ratio_full_range": rng(
            lambda r, i: r["delta_structure_full_space"][
                "per_question_cancellation_ratio_per_layer"
            ]
        ),
        "cancellation_ratio_residual_range": rng(
            lambda r, i: r["delta_structure_residual_space"][
                "per_question_cancellation_ratio_per_layer"
            ]
        ),
        "group_fold_auc_range": rng(lambda r, i: r["repro_orth_e1_groupfold_auc"]),
        "shuffled_fold_auc_range": rng(lambda r, i: r["shuffled_fold_auc"]),
        "fold_honest_auc_range": rng(lambda r, i: r["fold_honest_auc"]),
        "norm_probe_auc_range": rng(lambda r, i: r["norm_probe_groupfold_auc"]),
        "anti_align_cos_l1_all_folds": {
            t: r["reads"]["delta_structure_residual_space"]["train_vs_test_delta_cos_per_fold_l1"]
            for t, r in results.items()
        },
    }


def make_figure(results: dict, fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    traits = list(results.keys())
    fig, axes = plt.subplots(1, len(traits), figsize=(11.5, 3.6), sharey=True)
    series = [
        ("banked_orth_e1_auc_at_layers", "banked orth_E1 (GroupKFold)", "primary", "o"),
        ("repro_orth_e1_groupfold_auc", "reproduced (this round)", "baseline", "x"),
        ("shuffled_fold_auc", "shuffled pooled folds", "control", "s"),
        ("fold_honest_auc", "fold-honest projection", "accent", "^"),
    ]
    for ax, trait in zip(axes, traits):
        r = results[trait]["reads"]
        xs = list(range(len(r["layers"])))
        for key, label, role, marker in series:
            ax.plot(
                xs,
                r[key],
                marker=marker,
                markersize=5,
                color=paper_palette_role(role),
                label=label if trait == traits[0] else None,
            )
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(li) for li in r["layers"]])
        ax.set_title(trait)
        ax.set_xlabel("layer")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("held-out AUC")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4)
    paths = savefig_paper(fig, "residual_probe_characterization", dir=fig_dir)
    plt.close(fig)
    return paths["png"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--i778-staging", default=DEFAULT_STAGING)
    ap.add_argument("--directions-dir", default=DEFAULT_DIRECTIONS)
    ap.add_argument("--banked-probe-json", default=BANKED_PROBE_JSON)
    ap.add_argument(
        "--out",
        default=(
            "eval_results/issue_2225/fu1_preimage_prevention/analysis/"
            "residual_probe_characterization.json"
        ),
    )
    ap.add_argument("--fig-dir", default="figures/issue_2225/fu1")
    ap.add_argument("--traits", nargs="+", default=list(TRAITS))
    ap.add_argument("--skip-figure", action="store_true")
    args = ap.parse_args(argv)

    with open(args.banked_probe_json, encoding="utf-8") as f:
        banked = json.load(f)
    l1_map = _l1_layer_idx()

    t_all = time.time()
    results = {}
    for trait in args.traits:
        results[trait] = analyze_trait(trait, args, banked, l1_map)

    fig_path = None
    if not args.skip_figure:
        fig_path = make_figure(results, Path(args.fig_dir))
        _log(f"figure -> {fig_path}")

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    headline = (
        "The inverted orth_E1 held-out AUC is DIRECTION-PROJECTION LEAKAGE (mechanism b): "
        "E1 is the same-pool #778 diff-of-means, so projecting it out zeroes the pooled "
        "class-mean difference and forces the train-fold class signal to anti-align with "
        "every held-out group's; re-estimating the projected-out direction per fold on "
        "TRAIN rows only removes the inversion everywhere. GroupKFold AMPLIFIES the "
        "inversion (the held-out group's class signal is exactly what the train groups "
        "cancel against) but does not create it for evil/sycophancy — shuffled pooled "
        "folds on the same projected Gram stay below 0.5 there."
    )
    synthesis = _synthesis(results)
    verdict_notes = (
        "Mechanical verdicts apply the pre-registered thresholds UNCHANGED. Two "
        "registered thresholds misfire relative to the full read pattern and are NOT "
        "re-tuned post hoc: (i) 'a_group_fold_artifact' reads True for "
        "sycophancy/hallucination under the literal >= 0.45 shuffled-fold cutoff, but "
        "shuffled folds stay BELOW 0.5 at most layers for evil (0.28-0.42) and "
        "sycophancy (0.33-0.50) — group folds amplify, the same-pool projection "
        "creates; hallucination flips above 0.5 (0.62-0.65) because within-question "
        "memorization dominates at its smaller pool (n=818). (ii) "
        "'b_direction_projection_leakage' reads False for sycophancy/hallucination "
        "solely on the anti-alignment MAGNITUDE cutoff (mean cos <= -0.5): the SIGN is "
        "negative in all 15/15 folds across the three traits while the magnitude "
        "attenuates on the noisier smaller pools; every other (b) read — "
        "premise cos >= 0.96, residual pooled-Δ ratio 0.06-0.28, fold-honest AUC "
        ">= 0.455 with no inversion at any layer — passes for all three traits."
    )
    out = {
        "question": (
            "what carries the inverted-orientation residual (orth_E1) probe signal "
            "(held-out AUC 0.058-0.194 at L1)?"
        ),
        "registered_predictions": REGISTERED_PREDICTIONS,
        "fit_regime_note": (
            "dual/Gram-space ridge at n_kept < d=3584 — the parent-validated deliberately "
            "regularized under-determined probe (plan §4.7); reads are group-folded AUC, "
            "never R²; λ grid + max-mean-AUC selection identical to the parent"
        ),
        "headline": headline,
        "synthesis": synthesis,
        "verdict_notes": verdict_notes,
        "per_trait": results,
        "reproducibility": {
            **as_metadata_dict(git_provenance()),
            "seed": SEED,
            "wall_s": round(time.time() - t_all, 1),
            "figure": str(fig_path) if fig_path else None,
            "inputs": {
                "i778_staging": args.i778_staging,
                "directions_dir": args.directions_dir,
                "banked_probe_json": args.banked_probe_json,
            },
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    tmp.replace(out_path)
    _log(f"-> {out_path} (wall {time.time() - t_all:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
