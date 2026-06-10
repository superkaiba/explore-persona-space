# ruff: noqa: RUF001, RUF003
# Intentional Unicode (rho, ※, Δ, −, —) in scientific docstrings + labels.
"""Task #553 — Deliverable 6: negative-set exposure analysis.

Every #532 ordinary off-diagonal cell measures leakage onto a TRAINED-NEGATIVE
context: the #474 loc-arm adapters were trained with ALL 15 other panel
conditions as EOS negatives (``scripts/i474_phase23_train.py`` —
``N_NEG_PER_BYSTANDER = 20`` at line 94; ``_build_negative_rows`` at lines
210-258 samples 20 Q_train questions per bystander, no-marker rows whose only
loss-bearing slot is the first ``<|im_end|>`` — 15 bystanders x 20 = 300
negative rows per adapter). The 10 instructed bystanders were never in any
training mix. This script quantifies what the design CAN separate:

1. Within-#532: mean Δz(EOS) ordinary (trained-negative) vs instructed
   (never-clamped) with a bystander-cluster bootstrap CI on the difference
   (inline +12.8 vs +6.4); the within-instructed-strip prior gradient (among
   never-clamped contexts, does the base prior alone route the clamp?); and
   the within-ordinary bystander spread of Δz(EOS) — every ordinary bystander
   received the SAME 20 negative rows, so any ordinary-side bystander
   structure is NOT exposure dose and bounds the confound from inside.
2. Cross-panel (#478 = the identifiable contrast): the 35 held-out eval
   personas were never trained negatives (zero exact-label overlap with the
   fixed 4-negative panel — asserted; note the held-out persona ``assistant``
   is label-distinct from the ``helpful_assistant`` / ``no_persona``
   negatives). The #478 dz_eos distribution is reported NEXT TO #532's two
   exposure classes as a qualitative sign/magnitude contrast ONLY — panels
   differ in training recipe (broad 15-negative panel vs fixed 4-negative
   panel) and probe sets; no pooled fit (forbidden move).
3. REGISTERED non-separability statement: within #532's ordinary cohort,
   exposure is constant by design, so prior-vs-exposure cannot be decomposed
   WITHIN that cohort; the separation rests on (instructed-strip gradient) +
   (#478 contrast). Carried into the clean-result as a scope limit.

Per plan concern 13.1, the per-cohort base-side z(EOS) distributions are
reported next to the D6 gap. Framing stays descriptive-contrast language,
never causal identification (plan concern 13.8).

Smoke = this exact script with reduced ``--n-cluster-boot/--n-perm``.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import issue539_residual_per_cohort as i539
import issue553_panel as p553
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)


def _cohort_gap_block(panel: dict, masks: dict, args) -> dict:
    """Ordinary-vs-instructed mean Δz(EOS) gap, bystander-cluster bootstrap."""
    rng = np.random.default_rng(args.seed)
    sides = {}
    for cohort in ("ordinary_cross", "instructed_strip"):
        m = masks[cohort]
        byst = panel["bystander_label"][m]
        y = panel["dz_eos"][m]
        uniq = np.unique(byst)
        sides[cohort] = (y, byst, uniq)
    diffs = []
    for _ in range(args.n_cluster_boot):
        means = {}
        for cohort, (y, byst, uniq) in sides.items():
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            vals = np.concatenate([y[byst == b] for b in chosen])
            means[cohort] = float(vals.mean())
        diffs.append(means["ordinary_cross"] - means["instructed_strip"])
    obs_ord = float(sides["ordinary_cross"][0].mean())
    obs_instr = float(sides["instructed_strip"][0].mean())
    return {
        "mean_dz_eos_ordinary_cross": obs_ord,
        "mean_dz_eos_instructed_strip": obs_instr,
        "difference": obs_ord - obs_instr,
        "difference_ci95_bystander_cluster": {
            "low": float(np.percentile(diffs, 2.5)),
            "high": float(np.percentile(diffs, 97.5)),
            "n_boot": args.n_cluster_boot,
            "method": "bystanders resampled with replacement WITHIN each cohort "
            "independently per rep (exposure is a bystander property)",
        },
    }


def _instructed_gradient_block(panel: dict, masks: dict, args) -> dict:
    """Within-instructed-strip prior gradient of the clamp (n=10 bystanders)."""
    m = masks["instructed_strip"]
    byst = panel["bystander_label"][m]
    y = panel["dz_eos"][m]
    uniq = np.unique(byst)
    mean_clamp = np.array([float(y[byst == b].mean()) for b in uniq])
    out: dict = {"n_bystanders": len(uniq)}
    for prior_name, prior_map in (
        ("prior_margin_own", panel["_prior_margin_own_by_bystander"]),
        ("prior_logp_own", panel["_prior_logp_own_by_bystander"]),
    ):
        prior = np.array([prior_map[b] for b in uniq])
        out[prior_name] = {
            "rho": i539._spearman_rho(prior, mean_clamp),
            "ci95_boot_bystanders": i539._bootstrap_spearman_ci(
                prior, mean_clamp, args.n_marginal_boot, args.seed
            ),
            "p_perm": {
                **i539._permutation_p(prior, mean_clamp, args.n_perm, args.seed),
                "method": f"MC permutation of the {len(uniq)} instructed-bystander labels",
            },
        }
    out["read"] = (
        "among never-clamped contexts: a non-null gradient means the base prior routes the "
        "clamp without any negative-training exposure (partial separation, plan section 3.7)"
    )
    return out


def _ordinary_spread_block(panel: dict, masks: dict) -> dict:
    """Within-ordinary bystander spread of Δz(EOS) (uniform exposure dose)."""
    m = masks["ordinary_cross"]
    byst = panel["bystander_label"][m]
    y = panel["dz_eos"][m]
    uniq = np.unique(byst)
    means = {b: float(y[byst == b].mean()) for b in uniq}
    vals = np.array(list(means.values()))
    return {
        "per_bystander_mean_dz_eos": means,
        "sd_across_bystanders": float(vals.std()),
        "range": [float(vals.min()), float(vals.max())],
        "read": "every ordinary bystander received the SAME 20 negative rows per adapter, so "
        "this spread is NOT exposure dose — it bounds the within-cohort confound from inside",
    }


def make_figure(panel: dict, masks: dict, df478, gradient: dict, fig_dir: Path) -> None:
    """Hero: Δz(EOS) distributions by exposure class + instructed prior gradient."""
    set_paper_style("blog")
    colors = paper_palette(3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    groups = [
        ("#532 ordinary\n(trained-negative)", panel["dz_eos"][masks["ordinary_cross"]], colors[0]),
        ("#532 instructed\n(never-clamped)", panel["dz_eos"][masks["instructed_strip"]], colors[1]),
        ("#478 held-out\n(never-negative)", df478["dz_eos"].to_numpy(), colors[2]),
    ]
    for xi, (_label, vals, c) in enumerate(groups):
        rng = np.random.default_rng(0)
        sample = vals if len(vals) <= 800 else rng.choice(vals, size=800, replace=False)
        jitter = (rng.random(len(sample)) - 0.5) * 0.3
        ax1.plot(np.full(len(sample), xi) + jitter, sample, "o", ms=2, alpha=0.25, color=c)
        ax1.plot([xi - 0.25, xi + 0.25], [float(np.mean(vals))] * 2, color=c, lw=2.5)
    ax1.axhline(0.0, color="0.4", lw=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax1.set_ylabel("Δz(EOS) trained − base")
    ax1.set_title("EOS-side change by exposure class (qualitative contrast)", fontsize=9)

    m = masks["instructed_strip"]
    byst = panel["bystander_label"][m]
    uniq = np.unique(byst)
    prior = np.array([panel["_prior_margin_own_by_bystander"][b] for b in uniq])
    clamp = np.array([float(panel["dz_eos"][m][byst == b].mean()) for b in uniq])
    ax2.plot(prior, clamp, "o", ms=6, color=colors[1])
    for b, x, y in zip(uniq, prior, clamp, strict=True):
        ax2.annotate(
            b.replace("instr_", ""), (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points"
        )
    rho = gradient["prior_margin_own"]["rho"]
    ax2.set_xlabel("Base own-response prior margin (per bystander)")
    ax2.set_ylabel("Mean Δz(EOS)")
    ax2.set_title(f"Within-instructed-strip prior gradient (rho={rho:+.2f}, n=10)", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "exposure_dz_eos_classes", dir=fig_dir)
    plt.close(fig)
    print(f"[figures] wrote exposure_dz_eos_classes to {fig_dir}")


def main(argv: list[str] | None = None) -> int:
    args = p553.common_parser(
        "Task #553 D6: negative-set exposure analysis (within-#532 + #478 contrast)."
    ).parse_args(argv)
    t0 = datetime.now(UTC)

    panel = p553.build_margin_panel(args.i532_dir)
    step0 = p553.step0_i532(panel, args.i532_dir)
    masks = p553.cohort_masks_553(panel)

    df = p553.load_i478_panel(args.i478_parquet)
    step0_478 = p553.step0_i478(df, args.i478_parquet.parent / "summary_logit.json")

    # Zero-overlap assert (plan assumption 11): exact labels.
    personas = set(df["held_out_persona"].unique().tolist())
    overlap = personas & set(p553.I478_NEGATIVE_PANEL)
    assert not overlap, f"held-out personas overlap the negative panel: {overlap}"

    gap = _cohort_gap_block(panel, masks, args)
    gradient = _instructed_gradient_block(panel, masks, args)
    spread = _ordinary_spread_block(panel, masks)

    # #478 dz_eos distribution with run- and persona-cluster CIs on the mean.
    agg = p553.aggregate_run_persona(df)
    vals = df["dz_eos"].to_numpy(dtype=np.float64)
    cluster_means: dict = {}
    rng = np.random.default_rng(args.seed)
    for axis in ("run_id", "held_out_persona"):
        labels = df[axis].to_numpy()
        uniq = np.unique(labels)
        idx_of = {c: np.where(labels == c)[0] for c in uniq}
        boots = []
        for _ in range(args.n_cluster_boot):
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            boots.append(float(np.mean(np.concatenate([vals[idx_of[c]] for c in chosen]))))
        cluster_means[axis] = {
            "low": float(np.percentile(boots, 2.5)),
            "high": float(np.percentile(boots, 97.5)),
            "n_clusters": len(uniq),
            "n_boot": args.n_cluster_boot,
        }
    i478_block = {
        "mean_dz_eos": float(vals.mean()),
        "sd": float(vals.std()),
        "quantiles": {
            "q05": float(np.quantile(vals, 0.05)),
            "median": float(np.median(vals)),
            "q95": float(np.quantile(vals, 0.95)),
        },
        "mean_ci95_cluster_run": cluster_means["run_id"],
        "mean_ci95_cluster_persona": cluster_means["held_out_persona"],
        "n_run_persona_aggregates": len(agg),
        "negative_panel": list(p553.I478_NEGATIVE_PANEL),
        "overlap_with_held_out_personas": [],
        "label_note": "the held-out persona 'assistant' is label-distinct from the "
        "'helpful_assistant' / 'no_persona' negatives — the zero-overlap assert is "
        "exact-label per plan assumption 11",
        "scope_caveat": "panels differ in training recipe (broad 15-negative panel vs fixed "
        "4-negative panel) and probe sets — qualitative sign/magnitude contrast only, no "
        "pooled fit (forbidden move); descriptive-contrast language, never causal "
        "identification (plan concern 13.8)",
    }

    # Plan concern 13.1: base-side z(EOS) distributions next to the gap.
    ze_b = panel["q_ze_b"].mean(axis=1)
    base_eos = {}
    for cohort in ("ordinary_cross", "instructed_strip"):
        v = ze_b[masks[cohort]]
        base_eos[cohort] = {
            "mean": float(v.mean()),
            "sd": float(v.std()),
            "q05": float(np.quantile(v, 0.05)),
            "median": float(np.median(v)),
            "q95": float(np.quantile(v, 0.95)),
        }

    non_separability = (
        "REGISTERED (plan section 3.7.3): within #532's ordinary cohort, exposure is constant "
        "by design (every bystander got 20 negative rows per adapter), so prior-vs-exposure "
        "cannot be decomposed WITHIN that cohort; the separation rests on the instructed-strip "
        "gradient + the #478 never-negative contrast. Named in the clean-result as a scope "
        "limit."
    )

    inline_vs_reviewed = [
        p553.ivr_entry(
            "mean Δz(EOS) ordinary vs instructed",
            [12.8, 6.4],
            [gap["mean_dz_eos_ordinary_cross"], gap["mean_dz_eos_instructed_strip"]],
            True,
            "reviewed adds the bystander-cluster CI on the difference; 'clamp generalizes at "
            "~half strength' ships only if the gap CI excludes 0",
        ),
        p553.ivr_entry(
            "#478 dz_eos mean (quick look)",
            -3.07,
            i478_block["mean_dz_eos"],
            False,
            "plan assumption 19 quick-look, now under cluster inference",
        ),
    ]

    results = {
        "metadata": p553.result_metadata(args, "issue553_exposure.py"),
        "step0_i532": step0,
        "step0_i478": step0_478,
        "exposure_design": {
            "i532": "every loc-arm adapter trained with ALL 15 other panel conditions as EOS "
            "negatives (i474_phase23_train.py::_build_negative_rows, lines 210-258; "
            "N_NEG_PER_BYSTANDER=20 at line 94; 15 x 20 = 300 negative rows); the 10 "
            "instructed bystanders were never in any training mix",
            "i478": "fixed 4-negative panel; the 35 held-out eval personas were never "
            "trained negatives (zero exact-label overlap, asserted)",
        },
        "ordinary_vs_instructed_gap": gap,
        "instructed_strip_prior_gradient": gradient,
        "ordinary_spread": spread,
        "i478_never_negative_contrast": i478_block,
        "base_side_z_eos_distributions": base_eos,
        "non_separability_statement": non_separability,
        "inline_vs_reviewed": inline_vs_reviewed,
    }
    p553.write_json(args.out_dir / "exposure.json", results)
    make_figure(panel, masks, df, gradient, args.fig_dir)

    print(
        f"[headline] dz_eos gap ordinary−instructed = {gap['difference']:+.2f} "
        f"CI [{gap['difference_ci95_bystander_cluster']['low']:+.2f}, "
        f"{gap['difference_ci95_bystander_cluster']['high']:+.2f}]"
    )
    print(
        f"[headline] #478 never-negative dz_eos mean = {i478_block['mean_dz_eos']:+.2f} "
        f"(persona-cluster CI [{cluster_means['held_out_persona']['low']:+.2f}, "
        f"{cluster_means['held_out_persona']['high']:+.2f}])"
    )
    print(f"[done] wall={(datetime.now(UTC) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
