# ruff: noqa: RUF002, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — Deliverable 1 (PRIMARY): #478 transfer check of the channel anatomy.

Runs the #532 logit-channel decomposition on the #478/#531 persona panel
(``tidy_logit.parquet``, 40 CORE cells x 2 seeds x 35 held-out personas x 20
questions, aggregated to 2,800 run x persona means; run = cell x seed) and
reports agreement/disagreement with the #532-panel anatomy:

(a) push constancy — two-way (run + persona) Type-I variance shares of dz
    with order-swap check, per-K stratified shares, per-run SD distribution,
    and the REGISTERED dominance statistic ``run_share − max(persona_share,
    pair_share)`` under a 10,000-rep FE-re-estimating cell bootstrap
    (statistics-reconciler round-1 REVISION: the within-run persona-label
    permutation is retained ONLY as the FE-respecting null for the PERSONA
    share — it is exactly invariant for the run share on this complete
    balanced panel, p ≡ 1.0, and is NOT the dominance test);
(b) clamp routing — persona-FE vector of dz_eos (n=35) against the persona's
    base state (persona-mean matched-slot margin_base = the REGISTERED family
    member; persona-mean base_prior descriptive), pair bootstrap + MC
    permutation (#539 source-marginal recipe); plus the plan concern 13.1
    analogue: persona-FE of ABSOLUTE trained-side z_eos against the base-side
    term separately;
(c) base-residency — pair-corrected (two-way FE re-estimated per resample)
    Spearman of min_dist against {dz, dz_eos, dmargin, margin_trained,
    margin_base}, with the full #539 stack PLUS the cell-axis (40-cluster)
    bootstrap joining the wider-of primary-CI set for ALL min_dist reads
    (statistics-reconciler round-1 REVISION). Sign-direction labels are
    explicit: min_dist is a DISTANCE, #532's cosine a SIMILARITY — expected
    opposite signs.

Holm (registered D1 family): the two p-carrying members (b) and (c); member
(a) is judged from the dominance-CI per the revised registration.

NO own-response base prior exists on this panel (the base model's own
responses were never generated/scored there — re-verified 2026-06-10:
``eval_results/issue_478/`` contains only ``base_prior_reanalysis/`` summaries
+ parquets and ``logit_rescore/``); D5's own-response ranker therefore has no
#478 analogue (named limitation, plan section 3.2.4).

The #532 negative-set exposure contrast lives in ``issue553_exposure.py``;
this script also aggregates the per-JSON ``validation_vs_stored_logp`` blocks
(worst case over the 80 rescore JSONs, plan assumption 17) and the base/
trained argmax-composition rates (z_top_nonmarker motivating evidence, D7).

Smoke = this exact script with reduced ``--n-boot/--n-cluster-boot/--n-perm``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue539_corrected_reads_inference as i539inf
import issue539_residual_per_cohort as i539
import issue553_panel as p553
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

I478_CHANNELS = ("dz", "dz_eos", "dmargin")
MIN_DIST_TARGETS = ("dz", "dz_eos", "dmargin", "margin_trained", "margin_base")


def _share_block(y: np.ndarray, run_l: np.ndarray, per_l: np.ndarray, args) -> dict:
    """Observed shares (both orders) + cell-bootstrap CIs (+ dominance for dz)."""
    _, rc = np.unique(run_l, return_inverse=True)
    _, pc = np.unique(per_l, return_inverse=True)
    obs = p553.anova_shares(y, rc, pc, int(rc.max()) + 1, int(pc.max()) + 1)
    return {
        "observed": {
            "run_share_run_first": obs["a_first_share_a"],
            "persona_share_run_first": obs["a_first_share_b"],
            "persona_share_persona_first": obs["b_first_share_b"],
            "run_share_persona_first": obs["b_first_share_a"],
            "pair_share": obs["pair_share"],
            "order_swap_max_abs_diff": max(
                abs(obs["a_first_share_a"] - obs["b_first_share_a"]),
                abs(obs["a_first_share_b"] - obs["b_first_share_b"]),
            ),
        },
        "ci95_cell_boot": p553.shares_cell_bootstrap(
            y, run_l, per_l, args.n_boot, args.seed, dominance=True
        ),
    }


def _persona_fe_reads(agg, channel: str, base_cols: dict[str, np.ndarray], args) -> dict:
    """Persona-FE vector of a channel vs persona-level base-state covariates."""
    y = agg[channel].to_numpy(dtype=np.float64)
    run_l = agg["run_id"].to_numpy()
    per_l = agg["held_out_persona"].to_numpy()
    per_u, pc = np.unique(per_l, return_inverse=True)
    _, rc = np.unique(run_l, return_inverse=True)
    fe = p553.fe_vector(y, pc, rc, len(per_u), int(rc.max()) + 1)
    out: dict = {"persona_fe": {str(k): float(v) for k, v in zip(per_u, fe, strict=True)}}
    for name, x in base_cols.items():
        out[name] = {
            "rho": i539._spearman_rho(x, fe),
            "n_personas": len(per_u),
            "ci95_boot_personas": i539._bootstrap_spearman_ci(
                x, fe, args.n_marginal_boot, args.seed
            ),
            "p_perm": {
                **i539._permutation_p(x, fe, args.n_perm, args.seed),
                "method": f"MC permutation of the {len(per_u)} persona labels",
            },
        }
    return out


def _min_dist_corrected_reads(agg, args) -> dict:
    """Pair-corrected min_dist Spearman vs the five targets, full inference."""
    x = agg["min_dist"].to_numpy(dtype=np.float64)
    run_l = agg["run_id"].to_numpy()
    per_l = agg["held_out_persona"].to_numpy()
    cell_l = agg["cell_id"].to_numpy()
    run_u, rc = np.unique(run_l, return_inverse=True)
    per_u, pc = np.unique(per_l, return_inverse=True)
    x_tw, _ = i539._twoway_fe_residualize(x, run_l, per_l)
    reads: dict = {}
    for tgt in MIN_DIST_TARGETS:
        y = agg[tgt].to_numpy(dtype=np.float64)
        y_tw, _ = i539._twoway_fe_residualize(y, run_l, per_l)
        # Fast-path equivalence assert (the #539 convention).
        x_f, y_f = i539inf._twoway_resid_pair(x, y, rc, pc, len(run_u), len(per_u))
        drift = max(float(np.max(np.abs(x_tw - x_f))), float(np.max(np.abs(y_tw - y_f))))
        assert drift < 1e-8, f"fast two-way residual drift {drift!r} on {tgt}"
        rho = i539._spearman_rho(x_tw, y_tw)
        p_perm = i539._permutation_p(y_tw, x_tw, args.n_perm, args.seed)
        p_perm["method"] = (
            "FE residual permutation: min_dist FE-residuals permuted across run x persona "
            "aggregates against fixed DV FE-residuals (both sides projected on run + persona "
            "dummies first)"
        )
        cis = {
            "cluster_run": i539inf._cluster_boot_twoway(
                x, y, run_l, per_l, "source", args.n_cluster_boot, args.seed
            ),
            "cluster_persona": i539inf._cluster_boot_twoway(
                x, y, run_l, per_l, "bystander", args.n_cluster_boot, args.seed
            ),
            "cluster_cellaxis": p553.cluster_boot_twoway_spearman_cellaxis(
                x, y, run_l, per_l, cell_l, args.n_cluster_boot, args.seed
            ),
        }
        reads[tgt] = {
            "estimate": float(rho),
            "sign_direction": "min_dist is a DISTANCE (larger = farther); #532 cosine is a "
            "SIMILARITY — expected OPPOSITE signs vs the #532 quintet",
            "ci95_cell_boot": i539inf._cell_boot_twoway(x, y, rc, pc, args.n_boot, args.seed),
            "ci95_cluster_run": cis["cluster_run"],
            "ci95_cluster_persona": cis["cluster_persona"],
            "ci95_cluster_cellaxis": cis["cluster_cellaxis"],
            "primary_ci": p553.wider_ci({k: v for k, v in cis.items()}),
            "p_perm_fe": p_perm,
        }
        print(
            f"[c] min_dist vs {tgt}: rho_twoway={rho:+.3f} "
            f"primary CI [{reads[tgt]['primary_ci']['low']:+.3f}, "
            f"{reads[tgt]['primary_ci']['high']:+.3f}] ({reads[tgt]['primary_ci']['axis']}) "
            f"p={p_perm['p']:.4g}"
        )
    return reads


def _within_run_ranking(agg, args) -> dict:
    """Per-run Spearman across 35 personas of each ranker vs margin_trained."""
    out: dict = {}
    runs = sorted(agg["run_id"].unique().tolist())
    for ranker in ("margin_base", "min_dist"):
        # Store (run, rho) pairs as they are accepted so a degenerate-dropped
        # run can never mislabel the per_run_rho map (round-1 review minor).
        kept: list[tuple[str, float]] = []
        n_dropped = 0
        for r in runs:
            sub = agg[agg["run_id"] == r]
            rho = i539._spearman_rho(
                sub[ranker].to_numpy(dtype=np.float64),
                sub["margin_trained"].to_numpy(dtype=np.float64),
            )
            if np.isnan(rho):
                n_dropped += 1
                continue
            kept.append((str(r), float(rho)))
        arr = np.asarray([v for _, v in kept])
        rng = np.random.default_rng(args.seed)
        med_boot = []
        for _ in range(args.n_marginal_boot):
            med_boot.append(float(np.median(arr[rng.integers(0, len(arr), size=len(arr))])))
        out[ranker] = {
            "n_runs": len(runs),
            "n_degenerate_dropped": n_dropped,
            "median_rho": float(np.median(arr)),
            "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "median_ci95_run_boot": {
                "low": float(np.percentile(med_boot, 2.5)),
                "high": float(np.percentile(med_boot, 97.5)),
                "n_boot": args.n_marginal_boot,
            },
            "per_run_rho": dict(kept),
            "limitation": "no own-response base prior exists on this panel (named limitation)",
        }
    return out


def _validation_and_argmax(rescore_dir: Path) -> dict:
    """Worst-case validation blocks + argmax composition over the 80 JSONs."""
    worst = {"trained_mae": 0.0, "base_mae": 0.0, "trained_rho": 1.0, "base_rho": 1.0}
    counts = {
        "base": {"marker": 0, "eos": 0, "other": 0},
        "trained": {"marker": 0, "eos": 0, "other": 0},
    }
    files = sorted(rescore_dir.glob("K*_c*_seed*.json"))
    assert len(files) == 80, f"expected 80 rescore JSONs, found {len(files)}"
    for f in files:
        d = json.loads(f.read_text())
        v = d["validation_vs_stored_logp"]
        worst["trained_mae"] = max(worst["trained_mae"], float(v["trained"]["mae_nats"]))
        worst["base_mae"] = max(worst["base_mae"], float(v["base"]["mae_nats"]))
        worst["trained_rho"] = min(worst["trained_rho"], float(v["trained"]["spearman"]))
        worst["base_rho"] = min(worst["base_rho"], float(v["base"]["spearman"]))
        for rec in d["held_out"].values():
            for side, key in (
                ("base", "argmax_id_base_per_q"),
                ("trained", "argmax_id_trained_per_q"),
            ):
                ids = np.asarray(rec[key], dtype=np.int64)
                counts[side]["marker"] += int((ids == p553.MARKER_ID).sum())
                counts[side]["eos"] += int((ids == p553.EOS_ID).sum())
                counts[side]["other"] += int(((ids != p553.MARKER_ID) & (ids != p553.EOS_ID)).sum())
    argmax = {}
    for side, c in counts.items():
        n = sum(c.values())
        argmax[side] = {
            "n_slots": n,
            "argmax_in_marker_eos_rate": (c["marker"] + c["eos"]) / n,
            "marker_rate": c["marker"] / n,
            "eos_rate": c["eos"] / n,
            "other_rate": c["other"] / n,
        }
    return {"validation_worst_case_over_80_jsons": worst, "argmax_composition": argmax}


def _i532_side_points(args) -> dict:
    """#532-side point estimates for the agreement table (CIs live in
    ``channel_anatomy.json``; recomputed here so the table is self-contained)."""
    panel = p553.build_margin_panel(args.i532_dir)
    step0 = p553.step0_i532(panel, args.i532_dir)
    masks = p553.cohort_masks_553(panel)
    m = masks["ordinary_cross"]
    src, byst = panel["source_cid"][m], panel["bystander_label"][m]
    _, sc = np.unique(src, return_inverse=True)
    byst_u, bc = np.unique(byst, return_inverse=True)
    shares = {}
    for ch in ("dz_marker", "dz_eos", "dmargin"):
        y = panel[ch][m]
        shares[ch] = p553.anova_shares(y, sc, bc, int(sc.max()) + 1, int(bc.max()) + 1)
    # Pair-corrected cosine vs margin_base_matched (the +0.51 inline target).
    x = panel["cosine"][m]
    x_tw, _ = i539._twoway_fe_residualize(x, src, byst)
    quintet = {}
    for tgt in p553.I532_CHANNELS:
        y_tw, _ = i539._twoway_fe_residualize(panel[tgt][m], src, byst)
        quintet[tgt] = i539._spearman_rho(x_tw, y_tw)
    # Bystander-FE of dz_eos vs graded own-prior margin (clamp routing point).
    fe = p553.fe_vector(panel["dz_eos"][m], bc, sc, len(byst_u), int(sc.max()) + 1)
    prior = np.array([panel["_prior_margin_own_by_bystander"][b] for b in byst_u])
    clamp_rho = i539._spearman_rho(prior, fe)
    return {
        "step0": step0,
        "shares_ordinary_cross": shares,
        "pair_corrected_cosine_quintet": quintet,
        "clamp_bystander_fe_vs_prior_margin_rho": float(clamp_rho),
        "note": "point estimates only — full inference for these #532 reads lives in "
        "eval_results/issue_553/channel_anatomy.json",
    }


def make_figure(shares_478: dict, i532_pts: dict, mdr: dict, fig_dir: Path) -> None:
    """Hero: side-by-side stacked variance-share bars + #478 min_dist forest."""
    set_paper_style("blog")
    colors = paper_palette(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    bars = []
    for ch in ("dz_marker", "dz_eos", "dmargin"):
        sh = i532_pts["shares_ordinary_cross"][ch]
        bars.append(
            (
                f"#532 {p553.CHANNEL_DISPLAY[ch].split(' ')[-1]}",
                sh["a_first_share_a"],
                sh["a_first_share_b"],
                sh["pair_share"],
            )
        )
    for ch in I478_CHANNELS:
        o = shares_478[ch]["observed"]
        bars.append(
            (
                f"#478 {p553.CHANNEL_DISPLAY[ch].split(' ')[-1]}",
                o["run_share_run_first"],
                o["persona_share_run_first"],
                o["pair_share"],
            )
        )
    xs = np.arange(len(bars))
    a = np.array([b[1] for b in bars])
    b_ = np.array([b[2] for b in bars])
    c = np.array([b[3] for b in bars])
    ax1.bar(xs, a, color=colors[0], label="source / run FE")
    ax1.bar(xs, b_, bottom=a, color=colors[1], label="bystander / persona FE")
    ax1.bar(xs, c, bottom=a + b_, color=colors[2], label="pair residual")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([b[0] for b in bars], rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Type-I variance share (first-factor order)")
    ax1.set_title(
        "Channel variance anatomy: #532 (source x bystander) vs #478 (run x persona)", fontsize=9
    )
    ax1.legend(fontsize=8)

    ypos = np.arange(len(MIN_DIST_TARGETS))
    for yi, tgt in enumerate(MIN_DIST_TARGETS):
        r = mdr[tgt]
        lo, hi = r["primary_ci"]["low"], r["primary_ci"]["high"]
        ax2.plot([lo, hi], [yi, yi], color=colors[0], lw=1.6)
        ax2.plot(r["estimate"], yi, "o", ms=5, color=colors[0])
    ax2.axvline(0.0, color="0.4", lw=0.8)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([p553.CHANNEL_DISPLAY.get(t, t) for t in MIN_DIST_TARGETS], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Pair-corrected Spearman rho (min_dist vs channel)")
    ax2.set_title("#478: two-way-FE-corrected min_dist reads (wider-of cluster CI)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "transfer_478_anatomy", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote transfer_478_anatomy to {fig_dir}")


def make_exploratory_figures(anatomy: dict, agg, val: dict, fig_dir: Path) -> None:
    """Exploratory over-produce dump (plan section 4; analyzer picks heroes).

    Renders, from quantities the statistics pass already computed: per-K
    transfer shares; seed-split (42 vs 137) share agreement; the per-run
    push-SD histogram; the min_dist raw-alongside-FE scatter grid; and the
    #478 argmax-composition bars. No new statistics families — figures only.
    """
    set_paper_style("blog")
    colors = paper_palette(3)
    share_keys = ("a_first_share_a", "a_first_share_b", "pair_share")
    share_legend = ("run FE", "persona FE", "pair residual")

    # 1. Per-K stratified shares (one panel per channel, stacked per K).
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, ch in zip(axes, I478_CHANNELS, strict=True):
        per_k = anatomy[ch]["per_K_stratum"]
        ks = sorted(per_k, key=lambda s: int(s[1:]))
        bottoms = np.zeros(len(ks))
        for key, c, leg in zip(share_keys, colors, share_legend, strict=True):
            vals = np.array([per_k[k][key] for k in ks])
            ax.bar(np.arange(len(ks)), vals, bottom=bottoms, color=c, label=leg)
            bottoms += vals
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels([f"{k} sources/mix" for k in ks], fontsize=7)
        ax.set_title(p553.CHANNEL_DISPLAY.get(ch, ch), fontsize=8)
    axes[0].set_ylabel("Type-I variance share (run-first order)")
    axes[0].legend(fontsize=7)
    fig.suptitle("#478 variance shares within each K stratum (K absorbed by run FE)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "transfer_478_per_K_shares", dir=fig_dir)
    plt.close(fig)
    print(f"[figures:exploratory] wrote transfer_478_per_K_shares to {fig_dir}")

    # 2. Seed-split share agreement (grouped bars, seed 42 vs 137).
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    comp_labels = ("run FE", "persona FE", "pair residual")
    for ax, ch in zip(axes, I478_CHANNELS, strict=True):
        per_seed = anatomy[ch]["per_seed_split"]
        xs = np.arange(len(share_keys))
        for off, (seed, c) in enumerate(zip(("seed42", "seed137"), colors[:2], strict=True)):
            vals = [per_seed[seed][k] for k in share_keys]
            ax.bar(xs + (off - 0.5) * 0.36, vals, width=0.34, color=c, label=seed)
        ax.set_xticks(xs)
        ax.set_xticklabels(comp_labels, fontsize=7)
        ax.set_title(p553.CHANNEL_DISPLAY.get(ch, ch), fontsize=8)
    axes[0].set_ylabel("Type-I variance share (run-first order)")
    axes[0].legend(fontsize=7)
    fig.suptitle("#478 anatomy agreement across the seed split (42 vs 137)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "transfer_478_seed_split_shares", dir=fig_dir)
    plt.close(fig)
    print(f"[figures:exploratory] wrote transfer_478_seed_split_shares to {fig_dir}")

    # 3. Per-run push-SD histogram (the "±1 logit" analogue, 80 runs).
    sd_vals = np.array(
        list(anatomy["dz"]["per_run_sd_across_personas"]["per_run"].values()), dtype=np.float64
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.hist(sd_vals, bins=20, color=colors[0])
    ax.axvline(float(np.median(sd_vals)), color=colors[1], lw=1.6, label="median")
    ax.set_xlabel("Within-run SD of Δz(※) across the 35 held-out personas (logits)")
    ax.set_ylabel("Runs")
    ax.set_title("#478 per-run push spread (80 runs)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_paper(fig, "transfer_478_per_run_push_sd_hist", dir=fig_dir)
    plt.close(fig)
    print(f"[figures:exploratory] wrote transfer_478_per_run_push_sd_hist to {fig_dir}")

    # 4. Raw scatters ALONGSIDE every FE-corrected min_dist read.
    p553.exploratory_raw_vs_fe_grid(
        x=agg["min_dist"].to_numpy(dtype=np.float64),
        targets={t: agg[t].to_numpy(dtype=np.float64) for t in MIN_DIST_TARGETS},
        a_labels=agg["run_id"].to_numpy(),
        b_labels=agg["held_out_persona"].to_numpy(),
        x_label="min_dist (cosine distance to nearest trained source)",
        fig_name="transfer_478_min_dist_raw_vs_fe",
        fig_dir=fig_dir,
        suptitle="#478 min_dist reads: raw (top) alongside two-way-FE-corrected (bottom)",
    )

    # 5. Argmax-composition bars (#478, both model sides).
    p553.exploratory_argmax_bars(
        {f"#478 {side} side": rates for side, rates in val["argmax_composition"].items()},
        fig_name="transfer_478_argmax_composition",
        fig_dir=fig_dir,
        title="#478 matched-slot argmax composition (z_top_nonmarker motivating evidence)",
    )


def main(argv: list[str] | None = None) -> int:
    args = p553.common_parser(
        "Task #553 D1 (PRIMARY): #478 transfer check of the #532 channel anatomy."
    ).parse_args(argv)
    t0 = datetime.now(UTC)

    df = p553.load_i478_panel(args.i478_parquet)
    step0_478 = p553.step0_i478(df, args.i478_parquet.parent / "summary_logit.json")
    agg = p553.aggregate_run_persona(df)
    run_l = agg["run_id"].to_numpy()
    per_l = agg["held_out_persona"].to_numpy()

    # (a) push constancy + (channel anatomy for all three shift channels)
    anatomy: dict = {}
    for ch in I478_CHANNELS:
        print(f"[a] variance shares for {ch} ...")
        y = agg[ch].to_numpy(dtype=np.float64)
        blk = _share_block(y, run_l, per_l, args)
        blk["persona_share_perm_null"] = p553.share_permutation_null(
            y, run_l, per_l, "b", args.n_perm, args.seed
        )
        # Per-K stratified shares (K absorbed by run FE; show K is not driving).
        per_k = {}
        for k in sorted(agg["K"].unique().tolist()):
            sub = agg[agg["K"] == k]
            _, rck = np.unique(sub["run_id"].to_numpy(), return_inverse=True)
            _, pck = np.unique(sub["held_out_persona"].to_numpy(), return_inverse=True)
            per_k[f"K{k}"] = p553.anova_shares(
                sub[ch].to_numpy(dtype=np.float64), rck, pck, int(rck.max()) + 1, int(pck.max()) + 1
            )
        blk["per_K_stratum"] = per_k
        per_seed = {}
        for s in (42, 137):
            sub = agg[agg["seed"] == s]
            _, rcs = np.unique(sub["run_id"].to_numpy(), return_inverse=True)
            _, pcs = np.unique(sub["held_out_persona"].to_numpy(), return_inverse=True)
            per_seed[f"seed{s}"] = p553.anova_shares(
                sub[ch].to_numpy(dtype=np.float64), rcs, pcs, int(rcs.max()) + 1, int(pcs.max()) + 1
            )
        blk["per_seed_split"] = per_seed
        anatomy[ch] = blk
    # Per-run SD of dz across personas (the "±1 logit" analogue).
    per_run_sd = agg.groupby("run_id")["dz"].std(ddof=0)
    anatomy["dz"]["per_run_sd_across_personas"] = {
        "mean": float(per_run_sd.mean()),
        "median": float(per_run_sd.median()),
        "q25": float(per_run_sd.quantile(0.25)),
        "q75": float(per_run_sd.quantile(0.75)),
        "min": float(per_run_sd.min()),
        "max": float(per_run_sd.max()),
        "per_run": {k: float(v) for k, v in per_run_sd.items()},
    }

    # (b) clamp routing — persona-FE of dz_eos vs persona-level base state.
    print("[b] clamp routing ...")
    pm = agg.groupby("held_out_persona").agg(
        margin_base=("margin_base", "mean"),
        base_prior=("base_prior", "mean"),
        z_eos_base=("z_eos_base", "mean"),
    )
    per_u = np.array(sorted(agg["held_out_persona"].unique().tolist()))
    base_cols = {
        "vs_persona_mean_margin_base": pm.loc[per_u, "margin_base"].to_numpy(),
        "vs_persona_mean_base_prior": pm.loc[per_u, "base_prior"].to_numpy(),
    }
    clamp = _persona_fe_reads(agg, "dz_eos", base_cols, args)
    clamp["registered_family_member"] = "vs_persona_mean_margin_base"
    clamp["power_note"] = (
        "n=35 personas; critical rho ~0.33 at alpha=.05 — a null is reported as 'bounded "
        "below rho ~ CI half-width', not 'anatomy absent' (plan concern 13.2)"
    )
    # Concern 13.1 analogue: ABSOLUTE trained-side z_eos persona-FE vs base side.
    abs_eos = _persona_fe_reads(
        agg,
        "z_eos_trained",
        {"vs_persona_mean_z_eos_base": pm.loc[per_u, "z_eos_base"].to_numpy()},
        args,
    )

    # (c) base-residency — pair-corrected min_dist reads, full inference.
    print("[c] pair-corrected min_dist reads ...")
    min_dist_reads = _min_dist_corrected_reads(agg, args)

    # Per-seed point-estimate split for (c) (replication diagnostic).
    per_seed_c: dict = {}
    for s in (42, 137):
        sub = agg[agg["seed"] == s]
        xs = sub["min_dist"].to_numpy(dtype=np.float64)
        x_tw, _ = i539._twoway_fe_residualize(
            xs, sub["run_id"].to_numpy(), sub["held_out_persona"].to_numpy()
        )
        per_seed_c[f"seed{s}"] = {}
        for tgt in MIN_DIST_TARGETS:
            y_tw, _ = i539._twoway_fe_residualize(
                sub[tgt].to_numpy(dtype=np.float64),
                sub["run_id"].to_numpy(),
                sub["held_out_persona"].to_numpy(),
            )
            per_seed_c[f"seed{s}"][tgt] = i539._spearman_rho(x_tw, y_tw)

    # (D5 analogue) within-run ranking.
    print("[d] within-run ranking ...")
    ranking = _within_run_ranking(agg, args)

    # Validation worst-case + argmax composition over the 80 rescore JSONs.
    rescore_dir = args.i478_parquet.parent.parent / "logit_rescore"
    val = _validation_and_argmax(rescore_dir)

    # Holm over the registered family's p-carrying members ((b), (c)).
    p_b = clamp["vs_persona_mean_margin_base"]["p_perm"]["p"]
    p_c = min_dist_reads["margin_base"]["p_perm_fe"]["p"]
    holm = i539.holm_adjust([p_b, p_c])
    holm_block = {
        "family": "D1 registered transfer family (plan section 3.2)",
        "note": "member (a) run-FE dominance is judged from the dominance bootstrap CI "
        "(statistics-reconciler REVISION removed its permutation test: the within-run "
        "persona-label permutation is exactly invariant for the run share); Holm runs over "
        "the two p-carrying members",
        "members": [
            {
                "name": "(b) persona-FE(dz_eos) vs persona-mean margin_base",
                "p_raw": p_b,
                "p_holm": holm[0],
            },
            {
                "name": "(c) two-way-FE-corrected rho(min_dist, margin_base)",
                "p_raw": p_c,
                "p_holm": holm[1],
            },
        ],
    }

    # #532-side points for the agreement table (gate runs inside).
    print("[e] #532-side agreement points ...")
    i532_pts = _i532_side_points(args)

    dom = anatomy["dz"]["ci95_cell_boot"]["dominance_a_first"]
    agreement = {
        "claim_a_push_source_or_run_constant": {
            "i532_point": {
                "source_share_dz_marker": i532_pts["shares_ordinary_cross"]["dz_marker"][
                    "a_first_share_a"
                ],
                "bystander_share": i532_pts["shares_ordinary_cross"]["dz_marker"][
                    "a_first_share_b"
                ],
                "pair_share": i532_pts["shares_ordinary_cross"]["dz_marker"]["pair_share"],
            },
            "i478": {
                "run_share_dz": anatomy["dz"]["observed"]["run_share_run_first"],
                "persona_share": anatomy["dz"]["observed"]["persona_share_run_first"],
                "pair_share": anatomy["dz"]["observed"]["pair_share"],
                "dominance_ci95": {"low": dom["low"], "high": dom["high"]},
            },
            "direction_agrees": bool(
                anatomy["dz"]["observed"]["run_share_run_first"]
                > max(
                    anatomy["dz"]["observed"]["persona_share_run_first"],
                    anatomy["dz"]["observed"]["pair_share"],
                )
            ),
            "caveat": "run FE on #478 absorbs seed variance that #532's source share does not "
            "contain — compare direction-of-dominance, not share magnitudes (plan concern 13.4); "
            "#532 has 1 adapter per source, so 'source-constant' means adapter/run-constant "
            "(plan concern 13.3; per-seed split is the partial check)",
            "verdict": None,
        },
        "claim_b_clamp_context_routed": {
            "i532_point_bystander_fe_vs_prior_margin": i532_pts[
                "clamp_bystander_fe_vs_prior_margin_rho"
            ],
            "i478_persona_fe_vs_margin_base": {
                "rho": clamp["vs_persona_mean_margin_base"]["rho"],
                "ci95": clamp["vs_persona_mean_margin_base"]["ci95_boot_personas"],
                "p_holm": holm[0],
            },
            "verdict": None,
        },
        "claim_c_affinity_base_resident": {
            "i532_point_paircorrected_cosine_vs_margin_base": i532_pts[
                "pair_corrected_cosine_quintet"
            ]["margin_base_matched"],
            "i478_paircorrected_min_dist_vs_margin_base": {
                "rho": min_dist_reads["margin_base"]["estimate"],
                "primary_ci": min_dist_reads["margin_base"]["primary_ci"],
                "p_holm": holm[1],
            },
            "sign_note": "opposite signs expected (distance vs similarity)",
            "headline_scope": "base-READABLE at trained-text matched slots (plan section 3.4.5); "
            "cross-panel agreement is NOT evidence against mechanical-coupling accounts — both "
            "panels share the matched-slot construction",
            "verdict": None,
        },
        "verdict_note": "yes/no/partial verdicts assigned by the analyzer from the attached CIs "
        "(project convention)",
    }

    inline_vs_reviewed = [
        p553.ivr_entry(
            "mean dz (quick look)",
            7.25,
            float(df["dz"].mean()),
            False,
            "plan assumption 19 quick-look, recomputed under load asserts",
        ),
        p553.ivr_entry(
            "mean dz_eos (quick look)",
            -3.07,
            float(df["dz_eos"].mean()),
            False,
            "plan assumption 19 quick-look",
        ),
        p553.ivr_entry(
            "mean dlogZ (quick look)",
            -2.73,
            float(df["dlogZ"].mean()),
            False,
            "plan assumption 19 quick-look",
        ),
        p553.ivr_entry(
            "transfer claims (a)/(b)/(c)",
            None,
            "see agreement_table",
            True,
            "no inline #478 numbers exist — the transfer check is new analysis; the inline "
            "claims being tested are the #532-side anatomy (see channel_anatomy.json)",
        ),
    ]

    results = {
        "metadata": p553.result_metadata(args, "issue553_transfer_478.py"),
        "step0_i478": step0_478,
        "step0_i532": i532_pts.pop("step0"),
        "n_run_persona_aggregates": len(agg),
        "anatomy": anatomy,
        "clamp_routing": clamp,
        "absolute_z_eos_trained_persona_fe": abs_eos,
        "min_dist_corrected_reads": min_dist_reads,
        "min_dist_per_seed_point_estimates": per_seed_c,
        "within_run_ranking": ranking,
        "rescore_validation": val,
        "holm": holm_block,
        "i532_side_points": i532_pts,
        "agreement_table": agreement,
        "inline_vs_reviewed": inline_vs_reviewed,
    }
    p553.write_json(args.out_dir / "transfer_478.json", results)
    make_figure(anatomy, i532_pts, min_dist_reads, args.fig_dir)
    make_exploratory_figures(anatomy, agg, val, args.fig_dir)

    o = anatomy["dz"]["observed"]
    print(
        f"[headline] dz shares run/persona/pair = {o['run_share_run_first']:.3f}/"
        f"{o['persona_share_run_first']:.3f}/{o['pair_share']:.3f}; dominance CI "
        f"[{dom['low']:+.3f}, {dom['high']:+.3f}]"
    )
    print(
        f"[headline] clamp routing rho={clamp['vs_persona_mean_margin_base']['rho']:+.3f} "
        f"p_holm={holm[0]:.4g}; base-residency "
        f"rho={min_dist_reads['margin_base']['estimate']:+.3f} "
        f"p_holm={holm[1]:.4g}"
    )
    print(f"[done] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
