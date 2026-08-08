#!/usr/bin/env python3
"""Issue #2202 — P5: composition statistics + figures (VM-side).

Reuses the #1738/#1482 battery VERBATIM so the numbers are directly comparable
to the parent taxonomy: ``issue1738_characterize._contrast_masks`` (registered
group-vs-rest mask family) over the BINARY fail indicator, with
``issue1482_analysis._boot_group_delta`` (10k batched bootstrap),
``_perm_pvals`` (10k batched permutations) and ``_bh_fdr`` (q=0.05).

Two BH families, kept SEPARATE by design (plan §4 P5): the pre-registered
banked-label family, and the exploratory Fable-mode family (failures vs the
matched control), which never joins the first. Each Fable mode's rate is also
reported on the digest-DISJOINT failure subset so hypothesis-generation
contamination is visible. Also: attribution shares + ceiling, s_conf
failure-vs-control quantiles per space, rank-vs-nerr concordance echo,
pool-size robustness stability table, and the §3 reciprocity verdict lattice
(Δdp / Δdo sign atoms + the lower-tail anti-reciprocity check).

Figures (over-produced; the analyzer picks the hero) land in
``figures/issue_2202/`` (smoke: ``figures/issue_2202/smoke/``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1482_analysis as A82  # noqa: E402  (_boot_group_delta/_perm_pvals/_bh_fdr/_errbars)
import issue1738_characterize as CH  # noqa: E402  (_contrast_masks)
import issue2202_failchar as FC  # noqa: E402
import issue2202_labels as LB  # noqa: E402  (judge_dir, ci_fields resolution)
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2202_stats")

N_BOOT = 10_000
N_PERM = 10_000
BH_Q = 0.05
STAT_SEED = 2202
SMOKE_DRAWS = 200


def _draws(args) -> tuple[int, int]:
    return (SMOKE_DRAWS, SMOKE_DRAWS) if args.smoke else (N_BOOT, N_PERM)


def fig_dir(args) -> Path:
    d = Path(args.figures_out)
    d = d / "smoke" if args.smoke else d
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_inputs(args) -> dict:
    """Everything P5 reads (pod JSONs from git, labels from #1738, ci_fields)."""
    out = FC.out_eval_dir(args)
    rows = LB.load_percontext(args)
    fields_raw = LB.resolve_ci_fields(args)
    fields = {int(k): v for k, v in fields_raw.items()}
    labels = LB.load_labels_1738(args)
    doc = {
        "rows": rows,
        "fields": fields,
        "labels": labels,
        "attribution": json.loads((out / "attribution.json").read_text()),
        "reciprocity": json.loads((out / "reciprocity.json").read_text()),
        "hubness": json.loads((out / "hubness.json").read_text()),
        "concordance": json.loads((out / "concordance.json").read_text()),
        "pool_robustness": json.loads((out / "pool_robustness.json").read_text()),
        "geometry_summary": json.loads((out / "geometry_summary.json").read_text()),
    }
    jl = LB.judge_dir(args) / "labels.json"
    doc["judge"] = json.loads(jl.read_text()) if jl.exists() else None
    pp = LB.judge_dir(args) / "population.json"
    doc["population"] = json.loads(pp.read_text()) if pp.exists() else None
    return doc


def run_battery(
    fail: np.ndarray, masks: list[tuple[str, np.ndarray]], n_boot: int, n_perm: int, seed: int
) -> list[dict]:
    """Group-vs-rest deltas of the binary fail indicator over the registered
    mask family — bootstrap CI + permutation p + BH (the #1482 batched impls)."""
    pvals = A82._perm_pvals(fail, [m for _n, m in masks], n_perm, seed)
    bh = A82._bh_fdr(pvals, BH_Q)
    out = []
    for k, (name, m) in enumerate(masks):
        deltas = A82._boot_group_delta(fail, m, ~m, n_boot, seed + k)
        obs = float(fail[m].mean() - fail[~m].mean())
        lo, hi = float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))
        out.append(
            {
                "contrast": name,
                "n_group": int(m.sum()),
                "fail_rate_group": float(fail[m].mean()),
                "fail_rate_rest": float(fail[~m].mean()),
                "delta": obs,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_width": hi - lo,
                "p_perm": float(pvals[k]),
                "bh_significant": bool(bh[k]),
            }
        )
    return out


def reciprocity_verdict(rec: dict) -> dict:
    """§3 lattice — DISJOINT + exhaustive on the p97.5 sign atoms, plus the
    lower-tail (anti-reciprocity) read off the persisted per-draw vectors."""
    obs = rec["observed"]["reciprocity"]
    dp = np.asarray(rec["null_degree"]["draws"], dtype=np.float64)
    do = np.asarray(rec["null_distance"]["p5"]["draws"], dtype=np.float64)
    d_dp = obs - float(np.quantile(dp, 0.975))
    d_do = obs - float(np.quantile(do, 0.975))
    if d_dp > 0 and d_do > 0:
        verdict = "pair-collapse"
    elif d_dp > 0:
        verdict = "metric-explained"
    else:
        verdict = "degree-explained"
    return {
        "observed": obs,
        "delta_dp": d_dp,
        "delta_do_p5": d_do,
        "verdict": verdict,
        "anti_reciprocity_dp": bool(obs < float(np.quantile(dp, 0.025))),
        "anti_reciprocity_do_p5": bool(obs < float(np.quantile(do, 0.025))),
        "band_upper_vs_ceiling": {
            "dp_p975": float(np.quantile(dp, 0.975)),
            "do_p5_p975": float(np.quantile(do, 0.975)),
            "ceiling": 1.0,
        },
        "tau_sensitivity": {
            name: {
                "p975": float(np.quantile(np.asarray(blk["draws"]), 0.975)),
                "delta": obs - float(np.quantile(np.asarray(blk["draws"]), 0.975)),
            }
            for name, blk in rec["null_distance"].items()
        },
    }


def phase_stats(args) -> None:
    """Composition battery + Fable-mode family + verdicts → composition_stats.json."""
    logger.info("[phase=p5_stats] start (smoke=%s)", args.smoke)
    d = load_inputs(args)
    rows = d["rows"]
    n_boot, n_perm = _draws(args)
    ci_rows = np.asarray([int(r["ci"]) for r in rows], dtype=np.int64)
    fail = np.asarray([float(r["fail_raw_euclidean"]) for r in rows])
    masks = CH._contrast_masks(ci_rows, d["labels"], d["fields"])
    banked_battery = run_battery(fail, masks, n_boot, n_perm, STAT_SEED)

    # pool-size robustness: same battery on the reduced-pool fail indicators
    pool_stability: dict[str, dict] = {}
    pool_cols = [c for c in rows[0] if c.startswith("fail_pool")]
    for col in pool_cols + ["fail_raw_euclidean"]:
        fp = np.asarray([float(r[col]) for r in rows])
        res = run_battery(fp, masks, max(200, n_boot // 10), max(200, n_perm // 10), STAT_SEED)
        key = col.removeprefix("fail_pool") if col.startswith("fail_pool") else str(len(rows))
        for rec_ in res:
            pool_stability.setdefault(rec_["contrast"], {})[key] = {
                "delta": rec_["delta"],
                "bh_significant": rec_["bh_significant"],
            }

    # Fable-mode family (SECOND BH family; failures-eq vs matched control)
    fable_block: dict = {"available": False}
    if d["judge"] and d["population"]:
        jd, pop = d["judge"], d["population"]
        modes = [m["name"] for m in jd["modes"]]
        demoted = set(jd.get("demoted_modes", []))
        lab = jd["labels"]
        digest1 = set(pop.get("digest1_cis", []))
        fail_eq = [c for c in pop["fail_eq_cis"] if f"f{c}" in lab]
        ctrl = [c for c in pop["control_cis"] if f"c{c}" in lab]
        mask_fail = np.zeros(len(fail_eq) + len(ctrl), dtype=bool)
        mask_fail[: len(fail_eq)] = True
        per_mode: dict[str, dict] = {}
        pvals_modes: list[float] = []
        for mode in modes:
            v = np.asarray(
                [1.0 if lab[f"f{c}"][mode] == "yes" else 0.0 for c in fail_eq]
                + [1.0 if lab[f"c{c}"][mode] == "yes" else 0.0 for c in ctrl]
            )
            deltas = A82._boot_group_delta(v, mask_fail, ~mask_fail, n_boot, STAT_SEED)
            p = A82._perm_pvals(v, [mask_fail], n_perm, STAT_SEED)[0]
            pvals_modes.append(p)
            full_fail = [c for c in pop["fail_cis"] if f"f{c}" in lab]
            disjoint = [c for c in full_fail if c not in digest1]
            per_mode[mode] = {
                "rate_fail_eq": float(v[mask_fail].mean()),
                "rate_control": float(v[~mask_fail].mean()),
                "delta": float(v[mask_fail].mean() - v[~mask_fail].mean()),
                "ci_lo": float(np.quantile(deltas, 0.025)),
                "ci_hi": float(np.quantile(deltas, 0.975)),
                "p_perm": float(p),
                "rate_fail_full": float(np.mean([lab[f"f{c}"][mode] == "yes" for c in full_fail])),
                "rate_fail_digest_disjoint": (
                    float(np.mean([lab[f"f{c}"][mode] == "yes" for c in disjoint]))
                    if disjoint
                    else None
                ),
                "demoted_report_only": mode in demoted,
            }
        bh_modes = A82._bh_fdr(pvals_modes, BH_Q)
        for k, mode in enumerate(modes):
            per_mode[mode]["bh_significant"] = bool(bh_modes[k])
        fable_block = {
            "available": True,
            "n_fail_eq": len(fail_eq),
            "n_control": len(ctrl),
            "bh_family": "separate-exploratory (never joins the banked family)",
            "per_mode": per_mode,
        }

    # s_conf failure-vs-control quantiles per space
    sconf_cols = [c for c in rows[0] if c.startswith("s_conf_")]
    ctrl_set = set(d["population"]["control_cis"]) if d["population"] else set()
    fail_set = {int(r["ci"]) for r in rows if r["fail_raw_euclidean"] == "1"}
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    sconf_block: dict[str, dict] = {}
    for col in sconf_cols:
        v_fail = np.asarray([float(r[col]) for r in rows if int(r["ci"]) in fail_set])
        v_ctrl = np.asarray([float(r[col]) for r in rows if int(r["ci"]) in ctrl_set])
        sconf_block[col.removeprefix("s_conf_")] = {
            "fail_quantiles": {str(q): float(np.quantile(v_fail, q)) for q in qs}
            if len(v_fail)
            else None,
            "control_quantiles": {str(q): float(np.quantile(v_ctrl, q)) for q in qs}
            if len(v_ctrl)
            else None,
        }

    # representativeness check (b): banked-label composition, covered vs uncovered fails
    cov_fail = [r for r in rows if r["fail_raw_euclidean"] == "1" and r["kres_covered"] == "1"]
    unc_fail = [r for r in rows if r["fail_raw_euclidean"] == "1" and r["kres_covered"] == "0"]

    def _comp(rws: list[dict]) -> dict:
        if not rws:
            return {}
        lab = d["labels"]
        cis = [str(int(r["ci"])) for r in rws]
        with_lab = [c for c in cis if c in lab]
        return {
            "n": len(rws),
            "language_en": float(np.mean([lab[c]["language"] == "en" for c in with_lab])),
            "answer_is_refusal_yes": float(
                np.mean([lab[c]["answer_is_refusal"] == "yes" for c in with_lab])
            ),
            "refusal_adjacent_yes": float(
                np.mean([lab[c]["request_refusal_adjacent"] == "yes" for c in with_lab])
            ),
        }

    sig = [b for b in banked_battery if b["bh_significant"]]
    doc = {
        "n_boot": n_boot,
        "n_perm": n_perm,
        "bh_q": BH_Q,
        "seed": STAT_SEED,
        "banked_battery": banked_battery,
        "n_bh_significant": len(sig),
        "detection_floor_ci_width_median": float(
            np.median([b["ci_width"] for b in banked_battery if b["n_group"] >= 50])
        ),
        "pool_stability": pool_stability,
        "fable_modes": fable_block,
        "s_conf": sconf_block,
        "attribution": d["attribution"],
        "repr_check_b": {"covered_fail": _comp(cov_fail), "uncovered_fail": _comp(unc_fail)},
        "concordance": d["concordance"],
        "reciprocity_verdict": reciprocity_verdict(d["reciprocity"]),
        "meta": FC.meta_block({"smoke": bool(args.smoke)}),
    }
    FC.atomic_json(FC.out_eval_dir(args) / "composition_stats.json", doc)
    logger.info(
        "[p5] battery: %d contrasts, %d BH-significant; verdict=%s",
        len(banked_battery),
        len(sig),
        doc["reciprocity_verdict"]["verdict"],
    )


# ── figures ───────────────────────────────────────────────────────────────────────


def resolve_edges_npz(args) -> Path:
    """reciprocity_edges.npz — local derived copy, else the HF-staged copy."""
    if args.edges_npz:
        return Path(args.edges_npz)
    local = FC._derived(args) / "reciprocity_edges.npz"
    if local.exists():
        return local
    target = (
        PROJECT_ROOT
        / "data"
        / "issue_2202"
        / ("reciprocity_edges_smoke.npz" if args.smoke else "reciprocity_edges.npz")
    )
    hub.stage_hub_file(
        FC.C.HF_DATA_REPO,
        f"{FC.hf_prefix(args)}/analysis_tensors/reciprocity_edges.npz",
        target,
        repo_type="dataset",
    )
    return target


def phase_figures(args) -> None:  # noqa: C901 — one linear figure pass
    """Render the P5 figure set (summary + per-unit views per aggregate)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_paper_style()
    logger.info("[phase=p5_figures] start")
    d = load_inputs(args)
    stats = json.loads((FC.out_eval_dir(args) / "composition_stats.json").read_text())
    fdir = fig_dir(args)
    pal = paper_palette(6)
    rows = d["rows"]

    def _save(fig, name: str) -> None:
        fig.savefig(fdir / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info("[figs] wrote %s", fdir / name)

    # (a) reciprocity point vs bands (+ tau sensitivity) + ceiling
    rec = d["reciprocity"]
    rv = stats["reciprocity_verdict"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bands = [("degree-preserving", rec["null_degree"]["draws"])] + [
        (f"distance-only τ={name}", blk["draws"]) for name, blk in rec["null_distance"].items()
    ]
    for x, (label, draws) in enumerate(bands):
        v = np.asarray(draws, dtype=np.float64)
        lo, hi = np.quantile(v, 0.025), np.quantile(v, 0.975)
        ax.vlines(x, lo, hi, color=pal[1], lw=6, alpha=0.6)
        ax.scatter([x], [v.mean()], color=pal[1], zorder=3, s=18)
    ax.axhline(rec["observed"]["reciprocity"], color=pal[0], lw=1.5, label="observed")
    ax.axhline(1.0, color="0.4", lw=1.0, ls="--", label="ceiling (reciprocity ≤ 1)")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b[0] for b in bands], rotation=20, ha="right")
    ax.set_ylabel("top-1 confusion reciprocity")
    ax.set_title(f"Reciprocity vs null bands — verdict: {rv['verdict']}")
    ax.legend()
    _save(fig, "fig_reciprocity_bands.png")

    # (b) composition forest plot (banked family)
    bb = sorted(stats["banked_battery"], key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(bb))))
    y = np.arange(len(bb))
    v = np.asarray([r["delta"] for r in bb])
    lo = np.asarray([r["ci_lo"] for r in bb])
    hi = np.asarray([r["ci_hi"] for r in bb])
    el, eh = A82._errbars(v, lo, hi)  # non-negative offsets (gotchas #547/#1335)
    sigm = np.asarray([r["bh_significant"] for r in bb])
    ax.errorbar(v[~sigm], y[~sigm], xerr=(el[~sigm], eh[~sigm]), fmt="o", mfc="white", color=pal[2])
    if sigm.any():
        ax.errorbar(v[sigm], y[sigm], xerr=(el[sigm], eh[sigm]), fmt="o", color=pal[0])
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["contrast"] for r in bb], fontsize=7)
    ax.set_xlabel("failure-rate delta (group − rest), bootstrap 95% CI")
    ax.set_title("FAIL-1 composition vs held-out base rates (filled = BH-significant)")
    _save(fig, "fig_composition_forest.png")

    # rank survival (log-log)
    rk = np.asarray([float(r["rank_raw_euclidean"]) for r in rows])
    xs = np.sort(rk)
    surv = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(xs, np.maximum(surv, 1.0 / len(xs)), color=pal[0])
    ax.set_xlabel("true-answer mid-rank (raw euclidean)")
    ax.set_ylabel("P(rank > x)")
    ax.set_title("Rank survival, all 9,941 held-out contexts")
    _save(fig, "fig_rank_survival.png")

    # in-degree distributions + top-20 hub capture
    hb = d["hubness"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, color in (("retrieval", pal[0]), ("collapse", pal[1])):
        counts = np.asarray(hb[kind]["counts"])
        ax.hist(
            counts,
            bins=np.arange(0, counts.max() + 2) - 0.5,
            histtype="step",
            color=color,
            label=f"{kind} (skew {hb[kind]['n10_skewness']:.1f})",
            log=True,
        )
    ax.set_xlabel("N_10 in-degree")
    ax.set_ylabel("count (log)")
    ax.set_title("k-occurrence (N_10) in-degree distributions")
    ax.legend()
    _save(fig, "fig_indegree.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    top = hb["retrieval"]["top20"]
    ax.bar(range(len(top)), [t["count"] for t in top], color=pal[0])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([str(t["ci"]) for t in top], rotation=90, fontsize=6)
    ax.set_xlabel("pool answer (ci)")
    ax.set_ylabel("N_10 capture count")
    ax.set_title("Top-20 retrieval hubs")
    _save(fig, "fig_hub_capture.png")

    # s_conf densities per space (fail vs control)
    sconf_cols = [c for c in rows[0] if c.startswith("s_conf_")]
    ctrl_set = set(d["population"]["control_cis"]) if d["population"] else set()
    fig, axes = plt.subplots(1, len(sconf_cols), figsize=(3.2 * len(sconf_cols), 3.4), sharey=True)
    for ax, col in zip(np.atleast_1d(axes), sconf_cols):
        vf = [float(r[col]) for r in rows if r["fail_raw_euclidean"] == "1"]
        vc = [float(r[col]) for r in rows if int(r["ci"]) in ctrl_set]
        ax.hist(vf, bins=40, density=True, alpha=0.6, color=pal[0], label="FAIL-1")
        if vc:
            ax.hist(vc, bins=40, density=True, alpha=0.6, color=pal[1], label="matched control")
        ax.set_title(col.removeprefix("s_conf_"), fontsize=8)
        ax.set_xlabel("cos(a_i, a_j1)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _save(fig, "fig_sconf_density.png")

    # attribution stacked shares + ceiling
    att = d["attribution"]
    cls = ["MAP_ATTRIBUTABLE", "AMBIGUOUS", "IRREDUCIBLE", "UNKNOWN"]
    counts = [att["classes_over_fail1"][c] for c in cls]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bottom = 0
    for c, v, color in zip(cls, counts, paper_palette(4)):
        ax.bar([0], [v], bottom=bottom, color=color, label=c)
        bottom += v
    ax.axhline(
        att["acc1_ceiling"] * sum(counts),
        color="0.3",
        ls="--",
        lw=1,
        label=f"acc@1 ceiling ({att['acc1_ceiling']:.3f})",
    )
    ax.set_xticks([])
    ax.set_ylabel("FAIL-1 contexts")
    ax.set_title("Failure attribution over FAIL-1 (kresample control)")
    ax.legend(fontsize=7)
    _save(fig, "fig_attribution.png")

    # concordance scatter (per-context points)
    nerr = np.asarray([float(r["nerr"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(rk, nerr, s=3, alpha=0.25, color=pal[0])
    ax.set_xscale("log")
    ax.set_xlabel("true-answer mid-rank (log)")
    ax.set_ylabel("banked nerr (context L19 ridge)")
    ax.set_title(f"Rank vs nerr concordance (Spearman ρ={d['concordance']['spearman_rho']:.3f})")
    _save(fig, "fig_concordance_scatter.png")

    # pool robustness: per-contrast delta across pool sizes
    ps = stats["pool_stability"]
    sizes = sorted({s for blk in ps.values() for s in blk}, key=int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, (contrast, blk) in enumerate(sorted(ps.items())):
        ax.plot(
            [int(s) for s in sizes],
            [blk.get(s, {}).get("delta", np.nan) for s in sizes],
            marker="o",
            lw=0.8,
            ms=3,
            alpha=0.7,
            label=contrast if k < 12 else None,
        )
    ax.set_xscale("log")
    ax.set_xlabel("pool size")
    ax.set_ylabel("failure-rate delta")
    ax.set_title("Composition-contrast stability across pool sizes")
    ax.legend(fontsize=5, ncol=2)
    _save(fig, "fig_pool_robustness.png")

    # per-edge reverse-rank scatter (graded Result-3 companion; per-unit view)
    ez = np.load(resolve_edges_npz(args))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(ez["rank_fwd"], ez["rank_rev"], s=2, alpha=0.15, color=pal[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank of a_j under p_i (forward)")
    ax.set_ylabel("rank of a_i under p_j (reverse)")
    ax.set_title(
        f"Confusion-edge reverse ranks (Spearman ρ={d['reciprocity']['graded']['spearman_fwd_rev']:.3f})"
    )
    _save(fig, "fig_reverse_rank_scatter.png")

    # Fable-mode rates (summary) + per-context points (per-unit view)
    fm = stats["fable_modes"]
    if fm.get("available"):
        modes = sorted(fm["per_mode"])
        fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(modes)), 4))
        x = np.arange(len(modes))
        rf = np.asarray([fm["per_mode"][m]["rate_fail_eq"] for m in modes])
        rc_ = np.asarray([fm["per_mode"][m]["rate_control"] for m in modes])
        lo = np.asarray([fm["per_mode"][m]["ci_lo"] for m in modes])
        hi = np.asarray([fm["per_mode"][m]["ci_hi"] for m in modes])
        ax.bar(x - 0.2, rf, width=0.4, color=pal[0], label="FAIL-1 (equalized)")
        ax.bar(x + 0.2, rc_, width=0.4, color=pal[1], label="matched control")
        el, eh = A82._errbars(rf - rc_, lo, hi)
        ax.errorbar(x - 0.2, rf, yerr=(np.minimum(el, rf), eh), fmt="none", ecolor="0.3", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("mode rate")
        ax.set_title("Fable-named modes: failure vs matched control (2nd BH family)")
        ax.legend(fontsize=7)
        _save(fig, "fig_mode_rates.png")

        jd = d["judge"]
        lab = jd["labels"]
        fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(modes)), 4))
        rng = np.random.default_rng(0)
        for k, m in enumerate(modes):
            for arm, dx, color in (("f", -0.18, pal[0]), ("c", 0.18, pal[1])):
                ys = [
                    1.0 if v[m] == "yes" else 0.0 for cid, v in lab.items() if cid.startswith(arm)
                ]
                if ys:
                    jit = rng.uniform(-0.12, 0.12, size=len(ys))
                    ax.scatter(
                        np.full(len(ys), k + dx) + jit,
                        np.asarray(ys) + rng.uniform(-0.03, 0.03, size=len(ys)),
                        s=1.5,
                        alpha=0.15,
                        color=color,
                    )
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(modes, rotation=30, ha="right", fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["no", "yes"])
        ax.set_title("Per-context mode labels (jittered; fail vs control)")
        _save(fig, "fig_mode_percontext.png")

    logger.info("[p5] figures done -> %s", fdir)


PHASES = {"stats": phase_stats, "figures": phase_figures}
PHASE_ORDER = ["stats", "figures"]


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Issue #2202 P5 composition stats + figures")
    ap.add_argument("--phase", choices=[*PHASE_ORDER, "all"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--import-check", action="store_true", dest="import_check")
    ap.add_argument("--list-phases", action="store_true", dest="list_phases")
    ap.add_argument("--out-eval", default=str(PROJECT_ROOT / "eval_results" / "issue_2202"))
    ap.add_argument("--figures-out", default=str(PROJECT_ROOT / "figures" / "issue_2202"))
    ap.add_argument("--hf-prefix", default=FC.HF_PREFIX_2202)
    ap.add_argument("--labels-1738", default=LB.LABELS_1738_REL, dest="labels_1738")
    ap.add_argument("--ci-fields", default="", dest="ci_fields")
    ap.add_argument("--edges-npz", default="", dest="edges_npz")
    ap.add_argument("--work-root", default="/workspace/data/issue_2202")
    ap.add_argument("--text-cache", default=LB.DEFAULT_TEXT_CACHE, dest="text_cache")
    return ap


def _import_check() -> None:
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("import-check OK: issue2202_stats_figs")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    if args.list_phases:
        print(json.dumps(PHASE_ORDER))
        return 0
    if not args.phase:
        raise SystemExit("--phase is required (or --import-check / --list-phases)")
    args.work_root = Path(args.work_root)
    for ph in PHASE_ORDER if args.phase == "all" else [args.phase]:
        PHASES[ph](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
