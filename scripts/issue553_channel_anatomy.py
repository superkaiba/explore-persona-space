# ruff: noqa: RUF002, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — Deliverable 3: channel anatomy with inference (#532 followup panel).

On Ordinary cross-context cells (240) and, separately, the Instructed strip
(160):

1. Two-way (source + bystander) Type-I variance shares for Δz(※), Δz(EOS),
   Δmargin — order-swap check, 2,000-rep cell bootstrap on the shares, and
   FE-respecting permutation-null share distributions PERSISTED next to the
   observed shares (plan concern 13.5: FE shares are upward-biased; the null
   calibrates them).
2. Pair-corrected (two-way FE, re-estimated per resample) cosine Spearman
   against the five targets — the inline +0.13 / +0.43 / −0.25 / +0.51 /
   +0.43 quintet — each with the full #539 inference stack (FE-re-estimating
   cell bootstrap, source- and bystander-cluster bootstraps with relabeled
   copies, FE-respecting permutation), on the full slice AND the B1/C1
   duplicate-dropped slice. Holm over the five ordinary-cross reads.
3. The raw bystander-level prior correlation of the clamp (Δz(EOS) vs graded
   own-response prior, inline +0.59) at the bystander level (n=16 ordinary /
   n=26 incl. instructed — the 26-point read mixes cohorts and is descriptive
   only), pair bootstrap + MC permutation.
4. REGISTERED split-half / cross-fit probe-level robustness slice (plan
   section 3.4.4): (i) per-channel odd/even-probe split-half reliabilities
   (raw r + Spearman-Brown) bounding differential measurement-error
   attenuation; (ii) cross-fit quintet — base-margin predictor (and its FE
   correction) estimated on one probe half, trained level + Δmargin read on
   the other half, both half-assignments, so shared finite-probe noise cannot
   couple predictor and DV. The base-vs-Δ ordering does NOT ship as
   "base-resident / training's pair change opposes closeness" unless it
   survives this slice.
5. Headline scope rule (plan section 3.4.5, persisted for the analyzer): the
   matched-slot base read is ``A2_base_on_trained_R`` — claims are capped at
   "per-pair affinity is base-READABLE at trained-text slots", never "the map
   was already in the base model"; cross-panel agreement is NOT evidence
   against mechanical-coupling / text-mediation accounts.
6. Plan concern 13.1: bystander-FE of ABSOLUTE trained-side z(EOS) against
   the base-side term separately + per-cohort base-side z(EOS) distributions.
7. argmax-composition rates (z_top_nonmarker motivating evidence, D7) from
   the committed ``argmax_id`` fields, both model sides.

Smoke = this exact script with reduced ``--n-boot/--n-cluster-boot/--n-perm``.
"""

from __future__ import annotations

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

SHIFT_CHANNELS = ("dz_marker", "dz_eos", "dmargin")
QUINTET = ("dz_marker", "dz_eos", "dmargin", "margin_base_matched", "margin_trained")
INLINE_QUINTET = {
    "dz_marker": 0.13,
    "dz_eos": 0.43,
    "dmargin": -0.25,
    "margin_base_matched": 0.51,
    "margin_trained": 0.43,
}
COHORTS = ("ordinary_cross", "instructed_strip")


def _shares_block(panel: dict, mask: np.ndarray, args) -> dict:
    """Observed shares + cell-bootstrap CIs + both-axis permutation nulls."""
    src, byst = panel["source_cid"][mask], panel["bystander_label"][mask]
    _, sc = np.unique(src, return_inverse=True)
    _, bc = np.unique(byst, return_inverse=True)
    out: dict = {}
    for ch in SHIFT_CHANNELS:
        y = panel[ch][mask]
        obs = p553.anova_shares(y, sc, bc, int(sc.max()) + 1, int(bc.max()) + 1)
        out[ch] = {
            "observed": {
                "source_share_source_first": obs["a_first_share_a"],
                "bystander_share_source_first": obs["a_first_share_b"],
                "bystander_share_bystander_first": obs["b_first_share_b"],
                "source_share_bystander_first": obs["b_first_share_a"],
                "pair_share": obs["pair_share"],
                "order_swap_max_abs_diff": max(
                    abs(obs["a_first_share_a"] - obs["b_first_share_a"]),
                    abs(obs["a_first_share_b"] - obs["b_first_share_b"]),
                ),
            },
            "ci95_cell_boot": p553.shares_cell_bootstrap(
                y, src, byst, args.n_cluster_boot, args.seed
            ),
            "source_share_perm_null": p553.share_permutation_null(
                y, src, byst, "a", args.n_perm, args.seed
            ),
            "bystander_share_perm_null": p553.share_permutation_null(
                y, src, byst, "b", args.n_perm, args.seed
            ),
        }
    return out


def _quintet_block(panel: dict, mask: np.ndarray, args, full_stack: bool) -> dict:
    """Pair-corrected cosine reads vs the five targets, #539 inference stack."""
    x = panel["cosine"][mask]
    src, byst = panel["source_cid"][mask], panel["bystander_label"][mask]
    src_u, sc = np.unique(src, return_inverse=True)
    byst_u, bc = np.unique(byst, return_inverse=True)
    x_tw, _ = i539._twoway_fe_residualize(x, src, byst)
    out: dict = {}
    for tgt in QUINTET:
        y = panel[tgt][mask]
        y_tw, _ = i539._twoway_fe_residualize(y, src, byst)
        x_f, y_f = i539inf._twoway_resid_pair(x, y, sc, bc, len(src_u), len(byst_u))
        drift = max(float(np.max(np.abs(x_tw - x_f))), float(np.max(np.abs(y_tw - y_f))))
        assert drift < 1e-8, f"fast two-way residual drift {drift!r} on {tgt}"
        rho = i539._spearman_rho(x_tw, y_tw)
        blk: dict = {"estimate": float(rho)}
        if full_stack:
            p_perm = i539._permutation_p(y_tw, x_tw, args.n_perm, args.seed)
            p_perm["method"] = (
                "FE residual permutation: cosine FE-residuals permuted across cells against "
                "fixed DV FE-residuals (both sides projected on source + bystander dummies)"
            )
            blk.update(
                {
                    "ci95_cell_boot": i539inf._cell_boot_twoway(
                        x, y, sc, bc, args.n_boot, args.seed
                    ),
                    "ci95_cluster_source": i539inf._cluster_boot_twoway(
                        x, y, src, byst, "source", args.n_cluster_boot, args.seed
                    ),
                    "ci95_cluster_bystander": i539inf._cluster_boot_twoway(
                        x, y, src, byst, "bystander", args.n_cluster_boot, args.seed
                    ),
                    "p_perm_fe": p_perm,
                }
            )
        out[tgt] = blk
    return out


def _half_cell_channels(panel: dict, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cell channel means from one probe half (column indices ``idx``)."""
    zm_t = panel["q_zm_t"][:, idx].mean(axis=1)
    ze_t = panel["q_ze_t"][:, idx].mean(axis=1)
    zm_b = panel["q_zm_b"][:, idx].mean(axis=1)
    ze_b = panel["q_ze_b"][:, idx].mean(axis=1)
    return {
        "margin_trained": zm_t - ze_t,
        "margin_base_matched": zm_b - ze_b,
        "dmargin": (zm_t - ze_t) - (zm_b - ze_b),
        "dz_marker": zm_t - zm_b,
        "dz_eos": ze_t - ze_b,
    }


def _paircorr_cross(
    x_vals: np.ndarray, y_vals: np.ndarray, src: np.ndarray, byst: np.ndarray
) -> float:
    """Pair-corrected Spearman with each side FE-residualized on ITS OWN values."""
    x_tw, _ = i539._twoway_fe_residualize(x_vals, src, byst)
    y_tw, _ = i539._twoway_fe_residualize(y_vals, src, byst)
    return i539._spearman_rho(x_tw, y_tw)


def split_half_block(panel: dict, mask: np.ndarray, args) -> dict:
    """Registered split-half reliabilities + cross-fit quintet (plan 3.4.4)."""
    even, odd = np.arange(0, 50, 2), np.arange(1, 50, 2)
    src, byst = panel["source_cid"][mask], panel["bystander_label"][mask]
    half = {"even": _half_cell_channels(panel, even), "odd": _half_cell_channels(panel, odd)}

    reliabilities: dict = {}
    for ch in ("margin_base_matched", "margin_trained", "dmargin", "dz_marker", "dz_eos"):
        a, b = half["even"][ch][mask], half["odd"][ch][mask]
        r = float(np.corrcoef(a, b)[0, 1])
        reliabilities[ch] = {
            "split_half_r": r,
            "spearman_brown": 2 * r / (1 + r) if r > -1 else float("nan"),
        }

    cosine = panel["cosine"][mask]
    crossfit: dict = {}
    for pred_half, dv_half in (("even", "odd"), ("odd", "even")):
        mb_a = half[pred_half]["margin_base_matched"][mask]
        mt_b = half[dv_half]["margin_trained"][mask]
        dm_b = half[dv_half]["dmargin"][mask]
        crossfit[f"predictor_{pred_half}__dv_{dv_half}"] = {
            "cosine_vs_margin_base_predhalf": _paircorr_cross(cosine, mb_a, src, byst),
            "cosine_vs_margin_trained_dvhalf": _paircorr_cross(cosine, mt_b, src, byst),
            "cosine_vs_dmargin_dvhalf": _paircorr_cross(cosine, dm_b, src, byst),
            "margin_base_predhalf_vs_margin_trained_dvhalf": _paircorr_cross(mb_a, mt_b, src, byst),
            "margin_base_predhalf_vs_dmargin_dvhalf": _paircorr_cross(mb_a, dm_b, src, byst),
        }

    # Headline gate: does the base-vs-Delta ordering survive the cross-fit?
    survives = all(
        crossfit[k]["cosine_vs_margin_base_predhalf"] > crossfit[k]["cosine_vs_dmargin_dvhalf"]
        for k in crossfit
    )
    return {
        "split_half_reliabilities": reliabilities,
        "crossfit_quintet": crossfit,
        "headline_gate": {
            "base_vs_dmargin_ordering_survives_crossfit": bool(survives),
            "rule": "the base-vs-Delta ordering does NOT ship as 'base-resident / training's "
            "pair change opposes closeness' unless it survives this slice; if it does not, the "
            "headline becomes the artifact diagnosis (a finding) — plan section 3.4.4",
        },
        "note": "trained-side reads are noisier (#478 validation MAE 0.27 vs 0.07); "
        "reliabilities bound differential attenuation before the ordering is interpreted",
    }


def clamp_bystander_block(panel: dict, masks: dict, args) -> dict:
    """Bystander-level raw prior correlation of the clamp (inline +0.59)."""
    out: dict = {}
    for slice_name, mask in (
        ("ordinary_16", masks["ordinary_cross"]),
        ("all_26_descriptive", masks["pooled_cohort_fe"]),
    ):
        byst = panel["bystander_label"][mask]
        y = panel["dz_eos"][mask]
        byst_u = np.unique(byst)
        mean_clamp = np.array([float(y[byst == b].mean()) for b in byst_u])
        blk: dict = {"n_bystanders": len(byst_u)}
        for prior_name, prior_map in (
            ("prior_margin_own", panel["_prior_margin_own_by_bystander"]),
            ("prior_logp_own", panel["_prior_logp_own_by_bystander"]),
        ):
            prior = np.array([prior_map[b] for b in byst_u])
            blk[prior_name] = {
                "rho": i539._spearman_rho(prior, mean_clamp),
                "ci95_boot_bystanders": i539._bootstrap_spearman_ci(
                    prior, mean_clamp, args.n_marginal_boot, args.seed
                ),
                "p_perm": {
                    **i539._permutation_p(prior, mean_clamp, args.n_perm, args.seed),
                    "method": f"MC permutation of the {len(byst_u)} bystander labels",
                },
            }
        if slice_name == "all_26_descriptive":
            blk["caveat"] = (
                "mixes the ordinary and instructed cohorts (per-bystander means within own "
                "cohort) — descriptive only, never a headline (forbidden pooled cross-cohort "
                "correlation)"
            )
        out[slice_name] = blk
    return out


def eos_absolute_block(panel: dict, masks: dict) -> dict:
    """Plan concern 13.1: absolute trained-side z(EOS) FE vs the base side."""
    ze_t = panel["q_ze_t"].mean(axis=1)
    ze_b = panel["q_ze_b"].mean(axis=1)
    out: dict = {}
    for cohort in COHORTS:
        m = masks[cohort]
        src, byst = panel["source_cid"][m], panel["bystander_label"][m]
        byst_u, bc = np.unique(byst, return_inverse=True)
        _, sc = np.unique(src, return_inverse=True)
        fe_abs = p553.fe_vector(ze_t[m], bc, sc, len(byst_u), int(sc.max()) + 1)
        fe_dz = p553.fe_vector(panel["dz_eos"][m], bc, sc, len(byst_u), int(sc.max()) + 1)
        base_mean = np.array([float(ze_b[m][byst == b].mean()) for b in byst_u])
        out[cohort] = {
            "rho_bystFE_z_eos_trained_vs_bystmean_z_eos_base": i539._spearman_rho(
                base_mean, fe_abs
            ),
            "rho_bystFE_dz_eos_vs_bystmean_z_eos_base": i539._spearman_rho(base_mean, fe_dz),
            "z_eos_base_distribution": {
                "mean": float(ze_b[m].mean()),
                "sd": float(ze_b[m].std()),
                "q05": float(np.quantile(ze_b[m], 0.05)),
                "median": float(np.median(ze_b[m])),
                "q95": float(np.quantile(ze_b[m], 0.95)),
            },
            "note": "if trained-side z(EOS) carries context structure beyond the base side, "
            "routing is real; else 'context-routed clamp' is base variance + arithmetic "
            "(plan concern 13.1)",
        }
    return out


def argmax_block(panel: dict, masks: dict) -> dict:
    """argmax ∈ {※, EOS} rates per side per cohort (z_top_nonmarker evidence)."""
    out: dict = {}
    for cohort in ("ordinary_cross", "instructed_strip", "pooled_cohort_fe"):
        m = masks[cohort]
        blk = {}
        for side, key in (("trained", "q_argmax_t"), ("base", "q_argmax_b")):
            ids = panel[key][m]
            n = ids.size
            in_pair = ((ids == p553.MARKER_ID) | (ids == p553.EOS_ID)).sum()
            blk[side] = {
                "n_slots": int(n),
                "argmax_in_marker_eos_rate": float(in_pair / n),
                "marker_rate": float((ids == p553.MARKER_ID).sum() / n),
                "eos_rate": float((ids == p553.EOS_ID).sum() / n),
                "other_rate": float((n - in_pair) / n),
            }
        out[cohort] = blk
    return out


def make_figure(quintets: dict, fig_dir: Path) -> None:
    """Hero: forest of the five pair-corrected cosine reads (#539 style)."""
    set_paper_style("blog")
    colors = paper_palette(2)
    rows = [(c, t) for c in ("ordinary_cross", "noB1C1_ordinary_cross") for t in QUINTET]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(rows) + 1.4))
    for yi, (slice_name, tgt) in enumerate(rows):
        blk = quintets[slice_name][tgt]
        color = colors[0] if slice_name == "ordinary_cross" else colors[1]
        if "ci95_cell_boot" in blk:
            ci = blk["ci95_cell_boot"]
            ax.plot([ci["low"], ci["high"]], [yi, yi], color=color, lw=1.5)
        ax.plot(blk["estimate"], yi, "o", ms=4.5, color=color)
    ax.axvline(0.0, color="0.4", lw=0.8)
    ax.set_yticks(np.arange(len(rows)))
    labels = [
        f"{'Ordinary cross' if s == 'ordinary_cross' else 'Ordinary cross, B1/C1 dropped'} — "
        f"{p553.CHANNEL_DISPLAY[t]}"
        for s, t in rows
    ]
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Pair-corrected Spearman rho (cosine vs channel)")
    ax.set_title(
        "Two-way-FE-corrected cosine reads, #532 followup panel (cell-bootstrap 95% CI)",
        fontsize=9,
    )
    fig.tight_layout()
    savefig_paper(fig, "channel_anatomy_quintet_forest", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote channel_anatomy_quintet_forest to {fig_dir}")


def main(argv: list[str] | None = None) -> int:
    args = p553.common_parser(
        "Task #553 D3: channel anatomy with inference on the #532 followup panel."
    ).parse_args(argv)
    t0 = datetime.now(UTC)

    panel = p553.build_margin_panel(args.i532_dir)
    step0 = p553.step0_i532(panel, args.i532_dir)
    masks = p553.cohort_masks_553(panel)

    shares: dict = {}
    for cohort in COHORTS:
        print(f"[shares] {cohort} ...")
        shares[cohort] = _shares_block(panel, masks[cohort], args)

    quintets: dict = {}
    for slice_name, full_stack in (
        ("ordinary_cross", True),
        ("noB1C1_ordinary_cross", True),
        ("instructed_strip", True),
        ("noB1C1_instructed_strip", False),
    ):
        print(f"[quintet] {slice_name} ...")
        quintets[slice_name] = _quintet_block(panel, masks[slice_name], args, full_stack)

    # Holm over the five ordinary-cross pair-corrected reads (single p each).
    pvals = [quintets["ordinary_cross"][t]["p_perm_fe"]["p"] for t in QUINTET]
    holm = i539.holm_adjust(pvals)
    holm_block = {
        "family": "D3 five pair-corrected cosine reads on Ordinary cross-context cells",
        "members": [
            {"name": t, "p_raw": p, "p_holm": h}
            for t, p, h in zip(QUINTET, pvals, holm, strict=True)
        ],
        "note": "the split-half slice is a registered robustness gate on the headline "
        "phrasing, not a sixth Holm member (plan section 3.4)",
    }

    clamp = clamp_bystander_block(panel, masks, args)

    print("[split-half] registered robustness slice ...")
    split_half = {
        "ordinary_cross": split_half_block(panel, masks["ordinary_cross"], args),
        "instructed_strip": split_half_block(panel, masks["instructed_strip"], args),
    }

    eos_abs = eos_absolute_block(panel, masks)
    argmax = argmax_block(panel, masks)

    headline_scope = {
        "rule": "the matched-slot base read is A2_base_on_trained_R — base model scored at "
        "slots created by the TRAINED model's response. Claims cap at 'per-context affinity is "
        "base-resident; per-pair affinity is base-READABLE at trained-text slots' "
        "(base-scored matched-slot affinity), never 'the pair-specific map was already in the "
        "base model'. The post-train forecast-where label governs the headline. Cross-panel "
        "(D1) agreement is NOT cited as evidence against mechanical-coupling / text-mediation "
        "accounts — both panels share the matched-slot construction (plan section 3.4.5).",
    }

    obs_dzm = shares["ordinary_cross"]["dz_marker"]["observed"]
    obs_dze = shares["ordinary_cross"]["dz_eos"]["observed"]
    inline_vs_reviewed = [
        p553.ivr_entry(
            "Δz(※) shares source/bystander/pair (ordinary cross)",
            [0.89, 0.03, 0.085],
            [
                obs_dzm["source_share_source_first"],
                obs_dzm["bystander_share_source_first"],
                obs_dzm["pair_share"],
            ],
            True,
            "reviewed adds order-swap + share bootstrap + permutation-null calibration",
        ),
        p553.ivr_entry(
            "Δz(EOS) shares bystander/source/pair (ordinary cross)",
            [0.72, 0.22, 0.09],
            [
                obs_dze["bystander_share_source_first"],
                obs_dze["source_share_source_first"],
                obs_dze["pair_share"],
            ],
            True,
            "same",
        ),
        p553.ivr_entry(
            "clamp bystander-level prior correlation",
            0.59,
            clamp["ordinary_16"]["prior_margin_own"]["rho"],
            True,
            "reviewed read at the bystander level with pair bootstrap + MC permutation",
        ),
    ] + [
        p553.ivr_entry(
            f"pair-corrected cosine vs {t}",
            INLINE_QUINTET[t],
            quintets["ordinary_cross"][t]["estimate"],
            False,
            "same estimand; reviewed adds the #539 inference stack + Holm + dup-dropped slice",
        )
        for t in QUINTET
    ]

    results = {
        "metadata": p553.result_metadata(args, "issue553_channel_anatomy.py"),
        "step0_i532": step0,
        "variance_shares": shares,
        "pair_corrected_cosine_quintet": quintets,
        "holm": holm_block,
        "clamp_bystander_prior_correlation": clamp,
        "split_half_crossfit": split_half,
        "absolute_z_eos_anatomy": eos_abs,
        "argmax_composition": argmax,
        "headline_scope_rule": headline_scope,
        "inline_vs_reviewed": inline_vs_reviewed,
    }
    p553.write_json(args.out_dir / "channel_anatomy.json", results)
    make_figure(quintets, args.fig_dir)

    print(
        f"[headline] dz_marker shares src/byst/pair = "
        f"{obs_dzm['source_share_source_first']:.3f}/"
        f"{obs_dzm['bystander_share_source_first']:.3f}/{obs_dzm['pair_share']:.3f}"
    )
    for t in QUINTET:
        blk = quintets["ordinary_cross"][t]
        print(
            f"[headline] cosine vs {t}: rho={blk['estimate']:+.3f} "
            f"p_perm={blk['p_perm_fe']['p']:.4g}"
        )
    gate = split_half["ordinary_cross"]["headline_gate"]
    print(
        f"[headline] base-vs-Delta ordering survives cross-fit: "
        f"{gate['base_vs_dmargin_ordering_survives_crossfit']}"
    )
    print(f"[done] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
