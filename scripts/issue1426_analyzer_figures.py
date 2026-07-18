#!/usr/bin/env python3
# ruff: noqa: RUF001
"""Issue #1426 analyzer figures (clean-result body): hero mediation bars +
tri-lineage delta-CoT forest, tri-lineage family-contrast forest, tri-lineage
tercile-gradient profile, per-family compliance bars, matched-length forest.

Pure reads of committed eval_results JSONs (this run + #928/#1005 committed
artifacts). 0 GPU. Bootstrap CIs over contexts use seed 42 / 2000 draws
(matching the run's registered convention).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MAIN_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ER = PROJECT_ROOT / "eval_results" / "issue_1426"
ER1005 = MAIN_ROOT / "eval_results" / "issue_1005"
OUT = PROJECT_ROOT / "figures" / "issue_1426"
RNG_SEED, N_BOOT = 42, 2000


def jload(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    set_paper_style("blog")
    pal = paper_palette_blog(6)
    rng = np.random.default_rng(RNG_SEED)

    pcd = jload(ER / "percontext_deltas.json")["contrasts"]
    fam = jload(ER / "fam_contrast.json")
    lmg = jload(ER / "length_matched_gain.json")
    cov8 = jload(ER / "coverage_by_family.json")
    cov16 = jload(ER / "cap16k" / "coverage_by_family.json")
    cov8_1005 = jload(ER1005 / "coverage_by_family.json")
    cov16_1005 = jload(ER1005 / "cap16k-compliance-reread" / "coverage_by_family.json")
    boot = jload(ER / "bootstrap_deltaskill.json")
    nul = jload(ER / "null_matrix_indiv.json")["null"]
    mlp = jload(ER / "indiv-mlp-nonlinearity-control" / "mlp_indiv_validity.json")

    # ---------------------------------------------------------------- fig 1: hero
    # Left: per-question absolute skills of the four mediation arms at frozen L24,
    # bootstrap CIs over 50 contexts. Right: delta_CoT tri-lineage forest.
    h2 = pcd["h2_cot_gain_percontext"]["by_regime"]["indiv"]["per_context"]
    h3 = pcd["h3_composed_direct_percontext"]["by_regime"]["indiv"]["per_context"]
    h4 = pcd["h4_sufficiency_percontext"]["by_regime"]["indiv"]["per_context"]
    sk_d = np.array([r["skill_d_ctx2ans"] for r in h2])
    sk_g = np.array([r["skill_g_aug"] for r in h2])
    comp_key = next(k for k in h3[0] if k.startswith("skill_") and "ctx2ans" not in k)
    b_key = next(k for k in h4[0] if k.startswith("skill_") and "g_aug" not in k)
    sk_comp = np.array([r[comp_key] for r in h3])
    sk_b = np.array([r[b_key] for r in h4])

    # selection-symmetric null band: per-draw max over 32 layers, primary combo
    def band_p95(arm: str) -> float:
        draws = nul[arm]["mean/mean"]
        arr = np.array([draws[k] for k in sorted(draws, key=int)])
        return float(np.percentile(arr.max(axis=0), 95))

    band = max(band_p95(a) for a in ("d_ctx2ans", "b_cot2ans", "comp_pred", "g_aug"))
    ident = 0.9724  # identity ceiling, recon grid indiv L24

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    arms = [
        ("context\n→ answer", sk_d, pal[0]),
        ("realized CoT\n→ answer", sk_b, pal[1]),
        ("composed\n(ctx→ĈoT→ans)", sk_comp, pal[2]),
        ("context+CoT\n→ answer", sk_g, pal[3]),
    ]
    for i, (_label, vals, color) in enumerate(arms):
        lo, hi = boot_ci(vals, rng)
        m = float(vals.mean())
        ax.bar(i, m, color=color, width=0.62)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="black", capsize=4)
        ax.text(i, 0.03, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(band, ls=":", color="0.35")
    ax.axhline(ident, ls="--", color="0.35")
    ax.text(
        -0.55,
        band + 0.012,
        "null band p95",
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.text(
        -0.55,
        ident - 0.012,
        "identity ceiling",
        ha="left",
        va="top",
        fontsize=9,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.set_xticks(range(4), [a[0] for a in arms], fontsize=10)
    ax.set_ylabel("held-out skill-over-mean R²")
    ax.set_ylim(0, 1.02)
    ax.set_title("Per-question arms, frozen layer 24", fontsize=13, pad=10)

    tri = [
        (
            "R1-Llama-8B\n(this run)",
            boot["by_regime"]["indiv"]["statistics"]["H2_delta_g_minus_d"][
                "primary_frozen_direct_best"
            ],
        ),
        ("R1-Qwen-7B\n(prior lineage)", {"observed": 0.3302, "ci95": [0.2724, 0.3929]}),
        ("OpenThinker2-7B\n(prior lineage)", {"observed": 0.2033, "ci95": [0.1464, 0.2717]}),
    ]
    for i, (_label, s) in enumerate(tri):
        y = len(tri) - 1 - i
        ax2.errorbar(
            s["observed"],
            y,
            xerr=[[s["observed"] - s["ci95"][0]], [s["ci95"][1] - s["observed"]]],
            fmt="o",
            color=pal[3] if i == 0 else "0.45",
            capsize=4,
            markersize=8,
        )
        ax2.text(s["observed"], y - 0.28, f"{s['observed']:+.3f}", ha="center", fontsize=10)
    ax2.axvline(0, ls="--", color="0.6")
    ax2.set_ylim(-0.6, 2.4)
    ax2.set_yticks(range(len(tri)), [t[0] for t in reversed(tri)], fontsize=10)
    ax2.set_xlabel("Δ CoT-conditioning gain (context+CoT − direct)")
    ax2.set_xlim(-0.05, 0.55)
    ax2.set_title("CoT gain across the three lineages", fontsize=13, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "issue_1426/mediation_hero_trilineage", dir=str(OUT.parent))
    plt.close(fig)

    # ------------------------------------------- fig 2: fam-contrast tri-lineage forest
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    rows = []
    labels3 = {
        "this": "R1-Llama-8B (this run)",
        "issue1005": "R1-Qwen-7B",
        "issue928": "OpenThinker2-7B",
    }
    src = {
        "this": fam["by_regime"],
        **{
            k: fam["prior_lineage_family_keyed_baselines"][k]["by_regime"]
            for k in ("issue1005", "issue928")
        },
    }
    for model in ("this", "issue1005", "issue928"):
        for reg, regname in (("indiv", "per-question"), ("avg_q", "query-averaged")):
            b = src[model][reg]
            rows.append(
                (f"{labels3[model]} — {regname}", b["observed"], b["ci95"], model == "this")
            )
    for i, (_label, obs, ci, is_this) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.errorbar(
            obs,
            y,
            xerr=[[obs - ci[0]], [ci[1] - obs]],
            fmt="o",
            color=pal[3] if is_this else "0.45",
            capsize=4,
            markersize=7,
        )
        ax.text(obs, y + 0.22, f"{obs:+.3f}", ha="center", fontsize=9)
    ax.axvline(0, ls="--", color="0.6")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=10)
    ax.set_xlabel("Δ family gain excess (ICL+WildChat − length-matched donors)")
    ax.set_title(
        "Length-matched family contrast, like-for-like across lineages", fontsize=13, pad=10
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_1426/fam_contrast_trilineage_forest", dir=str(OUT.parent))
    plt.close(fig)

    # ------------------------------------------------- fig 3: tercile gradient profile
    terc = lmg["by_regime"]["indiv"]["noncollapse_short_cot_terciles"]
    gain_by_ctx = {r["context"]: r["delta"] for r in h2}
    xs = [0, 1, 2]
    this_obs, this_lo, this_hi = [], [], []
    for t in terc:
        vals = np.array([gain_by_ctx[c] for c in t["contexts"]])
        lo, hi = boot_ci(vals, rng)
        this_obs.append(t["observed"])
        this_lo.append(t["observed"] - lo)
        this_hi.append(hi - t["observed"])
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.errorbar(
        xs,
        this_obs,
        yerr=[this_lo, this_hi],
        fmt="o-",
        color=pal[3],
        capsize=4,
        markersize=8,
        label="R1-Llama-8B (this run, 32 non-ICL/WildChat contexts)",
    )
    ax.plot(
        xs,
        [0.147, 0.198, 0.194],
        "s--",
        color="0.45",
        markersize=7,
        label="R1-Qwen-7B (32 non-ICL/WildChat contexts)",
    )
    ax.plot(
        xs,
        [0.245, 0.066, 0.044],
        "d--",
        color=pal[1],
        markersize=7,
        label="OpenThinker2-7B (its 36 unflagged contexts)",
    )
    for x, y in zip(xs, this_obs, strict=True):
        ax.text(x + 0.06, y - 0.014, f"{y:+.3f}", ha="left", va="top", fontsize=10)
    ax.set_xticks(xs, ["shortest third", "middle third", "longest third"])
    ax.set_xlabel("median well-formed CoT length tercile")
    ax.set_ylabel("per-context CoT gain (Δ skill)")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("CoT-length tercile profile across the three lineages", fontsize=13, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "issue_1426/tercile_gradient_trilineage", dir=str(OUT.parent))
    plt.close(fig)

    # ------------------------------------------------------ fig 4: compliance bars
    fams = ["persona", "wildchat", "icl", "rephrase", "format", "behavior", "default"]
    r8 = [cov8["families"][f]["usable_rate"] for f in fams]
    r16 = [cov16["families"][f]["usable_rate"] for f in fams]
    p8 = [cov8_1005["families"][f]["usable_rate"] for f in fams]
    p16 = [cov16_1005["families"][f]["usable_rate"] for f in fams]
    x = np.arange(len(fams))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(x - w / 2, r8, w, color=pal[0], label="this run, 8,192 cap")
    ax.bar(x + w / 2, r16, w, color=pal[3], label="this run, post-16,384 regen")
    ax.plot(x - w / 2, p8, "D", color="0.25", markersize=6, label="R1-Qwen-7B, 8,192")
    ax.plot(x + w / 2, p16, "s", color="0.25", markersize=6, label="R1-Qwen-7B, post-16,384")
    for xi, (a, b) in enumerate(zip(r8, r16, strict=True)):
        ax.text(xi - w / 2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8, rotation=90)
        ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8, rotation=90)
    ax.axhline(0.95, ls=":", color="0.4")
    ax.axhline(0.94, ls="--", color="0.4")
    ax.text(
        -0.62,
        0.951,
        "0.95 bar (C)",
        ha="left",
        fontsize=9,
        color="0.4",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.text(
        -0.62,
        0.9335,
        "0.94 bar (C94)",
        ha="left",
        fontsize=9,
        color="0.4",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
    ax.set_xticks(x, ["persona", "WildChat", "ICL", "rephrase", "format", "behavior", "default"])
    ax.set_ylim(0.90, 1.015)
    ax.set_ylabel("usable-row rate (exact parser)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower right")
    ax.set_title("Scaffold compliance per family, both token budgets", fontsize=13, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "issue_1426/coverage_by_family_compliance", dir=str(OUT.parent))
    plt.close(fig)

    # ---------------------------------------------- fig 5: matched-length forest
    mlc = jload(ER / "mlc_bootstrap_deltaskill.json")
    pma = jload(ER / "pma_bootstrap_deltaskill.json")

    def r1(blob, reg):
        st = blob["by_regime"][reg]["statistics"]
        k = next(kk for kk in st if kk.startswith("read1"))
        v = st[k]
        return next(vv for vv in v.values() if isinstance(vv, dict) and "observed" in vv)

    rows = [
        ("full context — per-question", r1(mlc, "indiv"), (-0.0480, -0.0558, -0.0411)),
        ("full context — query-averaged", r1(mlc, "avg_q"), (-0.0227, -0.0340, -0.0142)),
        ("query-excluded prefix — per-question", r1(pma, "indiv"), (-0.1067, -0.1202, -0.0941)),
        ("query-excluded prefix — query-averaged", r1(pma, "avg_q"), (-0.0474, -0.0745, -0.0247)),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for i, (_label, s, ref) in enumerate(rows):
        y = len(rows) - 1 - i
        obs, ci = s["observed"], s["ci95"]
        ax.errorbar(
            obs,
            y + 0.10,
            xerr=[[obs - ci[0]], [ci[1] - obs]],
            fmt="o",
            color=pal[3],
            capsize=4,
            markersize=7,
            label="R1-Llama-8B (this run)" if i == 0 else None,
        )
        ax.errorbar(
            ref[0],
            y - 0.10,
            xerr=[[ref[0] - ref[1]], [ref[2] - ref[0]]],
            fmt="s",
            color="0.45",
            capsize=3,
            markersize=6,
            label="R1-Qwen-7B" if i == 0 else None,
        )
        ax.text(obs, y + 0.24, f"{obs:+.3f}", ha="center", fontsize=9)
    ax.axvline(0, ls="--", color="0.6")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=10)
    ax.set_xlabel("Δ skill (CoT-conclusion slice − answer-opening slice, matched budget)")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title("Matched-length demotion, both input conventions", fontsize=13, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "issue_1426/matched_length_forest_both_conventions", dir=str(OUT.parent))
    plt.close(fig)

    print("wrote 5 figures to", OUT)
    print("mlp gate:", mlp["reads"]["estimator_validity_gate"]["pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
