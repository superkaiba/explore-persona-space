#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (→, Δ) in scientific docstrings + labels.
"""Issue #1426 sampled-rollout robustness figures (VM-side, 0 GPU — amendment
plan v4 §4.2 item 6).

Pure reads of the committed GREEDY reference JSONs (``eval_results/issue_1426/``)
plus the round's per-seed outputs
(``eval_results/issue_1426/sampled-rollout-robustness/seed<s>/`` + the staged
per-seed rollout dirs). Produces:

1. **Hero (a): mediation-lattice bars** — 4 arms × {greedy, seed 42, seed 137}
   with per-context bootstrap CIs, per-decoding selection-symmetric null bands
   + identity ceilings; per-question (indiv) primary panel + query-averaged
   (avg_q) panel b.
2. **Hero (b): family-excess forest** — greedy + both seeds, both regimes,
   greedy CI shaded as the containment band.
3. **Per-family coverage bars** — sampled seeds vs greedy, malformed-class
   taxonomy stacked above the usable-rate bar.
4. **Per-context CoT-gain scatter** sampled-vs-greedy (battery-id labels — the
   low-level per-unit plot behind both aggregates).
5. **CoT-length distributions per family** (chars before ``</think>``),
   sampled seeds vs greedy (greedy overlay skipped LOUDLY when
   ``--greedy-rollouts`` is absent).
6. **Per-seed tercile profiles** vs greedy (bonus H3 read).
7. **Seed-42-vs-137 per-context gain scatter** (within-regime replication).

Plus the **rollout-digest cross-check** (assumption 6's in-run verification):
row-by-row comparison of the two seeds' FULL rollout corpora, FAILING LOUD
(``SystemExit``) when they are (near-)identical — byte-identical corpora score
1.0; the 0.9 threshold means a handful of 16,384-regen rows cannot mask a
per-request-seeding failure. Report: ``<seed-results-root>/rollout_digest_check.json``.

Never prints rollout completion TEXT (the probe pool is misalignment
paraphrases — digest-only handling); only lengths, counts and hashes.

Usage (after the pod round + per-seed f4)::

    uv run python scripts/issue1426_sampled_robustness_figures.py \\
        --seed-rollouts data/issue_1426_sampled_s42/raw_completions/thinking_rollouts \\
                        data/issue_1426_sampled_s137/raw_completions/thinking_rollouts \\
        --greedy-rollouts data/issue_1426/raw_completions/thinking_rollouts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)
from issue1426_common import THINK_CLOSE  # noqa: E402
from issue928_common import dump_json, reproducibility_metadata  # noqa: E402

RNG_SEED, N_BOOT = 42, 2000
IDENTICAL_FRAC_MAX = 0.9
FAMILIES = ["persona", "wildchat", "icl", "rephrase", "format", "behavior", "default"]


def jload(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def arm_skills(pcd: dict, regime: str) -> dict[str, np.ndarray]:
    """Per-context absolute skills of the four mediation arms (analyzer fig-1 keys)."""
    h2 = pcd["contrasts"]["h2_cot_gain_percontext"]["by_regime"][regime]["per_context"]
    h3 = pcd["contrasts"]["h3_composed_direct_percontext"]["by_regime"][regime]["per_context"]
    h4 = pcd["contrasts"]["h4_sufficiency_percontext"]["by_regime"][regime]["per_context"]
    comp_key = next(k for k in h3[0] if k.startswith("skill_") and "ctx2ans" not in k)
    b_key = next(k for k in h4[0] if k.startswith("skill_") and "g_aug" not in k)
    return {
        "d_ctx2ans": np.array([r["skill_d_ctx2ans"] for r in h2]),
        "b_cot2ans": np.array([r[b_key] for r in h4]),
        "comp_pred": np.array([r[comp_key] for r in h3]),
        "g_aug": np.array([r["skill_g_aug"] for r in h2]),
    }


def gain_by_context(pcd: dict, regime: str) -> dict[str, float]:
    h2 = pcd["contrasts"]["h2_cot_gain_percontext"]["by_regime"][regime]["per_context"]
    return {r["context"]: r["delta"] for r in h2}


def null_band_p95(null_blob: dict, regime_frozen_note: str) -> float:
    """Selection-symmetric band: per-draw max over the FITTED layers, primary combo.

    On the greedy reference the fitted set is all 32 layers; on the restricted
    per-seed matrices it is the frozen set {12, 24} — the identical
    max-over-fitted-layers selection at each grain (plan §6). ``regime_frozen_note``
    is carried only for the meta trail.
    """
    _ = regime_frozen_note
    nul = null_blob["null"]
    bands = []
    for arm in ("d_ctx2ans", "b_cot2ans", "comp_pred", "g_aug"):
        draws = nul[arm]["mean/mean"]
        arr = np.array([draws[k] for k in sorted(draws, key=int)])
        bands.append(float(np.percentile(arr.max(axis=0), 95)))
    return max(bands)


def identity_ceiling(recon_blob: dict, regime: str, frozen_layer: int) -> float:
    """Identity-arm LOCO skill at the regime's frozen layer (recon grid)."""
    rows = recon_blob["results"][regime]["grid"]["ident"]["mean/mean"]["loco"]
    by_layer = {int(r["layer"]): float(r["skill"]) for r in rows}
    return by_layer[frozen_layer]


def rollout_rows(rollouts_dir: Path) -> dict[tuple[str, str], str]:
    """FULL corpus as ``{(context, probe): completion}`` (text never printed)."""
    rows: dict[tuple[str, str], str] = {}
    files = sorted(rollouts_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"[sampled-figures] no rollout files under {rollouts_dir} — refusing")
    for p in files:
        blob = jload(p)
        c = blob["context_id"]
        for r in blob["completions"]:
            rows[(c, r["probe"])] = r["completion"]
    return rows


def rollout_lengths_by_family(rollouts_dir: Path) -> dict[str, list[int]]:
    """Per-family CoT char lengths (chars before ``</think>``; malformed rows skipped)."""
    out: dict[str, list[int]] = {}
    for p in sorted(rollouts_dir.glob("*.json")):
        blob = jload(p)
        fam = blob["family"]
        for r in blob["completions"]:
            text = r["completion"]
            if THINK_CLOSE not in text:
                continue
            out.setdefault(fam, []).append(len(text.split(THINK_CLOSE, 1)[0]))
    return out


def digest_cross_check(dirs: dict[int, Path], out_json: Path) -> dict:
    """Assumption-6 verification over the FULL corpora — FAILS LOUD on identity."""
    seeds = sorted(dirs)
    if len(seeds) < 2:
        report = {
            "dv": "rollout-digest cross-check (assumption 6)",
            "status": f"SKIPPED — single surviving seed {seeds} (nothing to compare; "
            "the replication read carries the single-seed caveat, plan §7)",
            "reproducibility": reproducibility_metadata(),
        }
        dump_json(report, out_json)
        print(f"[sampled-figures] {report['status']}")
        return report
    a, b = seeds[0], seeds[1]
    rows_a, rows_b = rollout_rows(dirs[a]), rollout_rows(dirs[b])
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        raise SystemExit(
            f"[sampled-figures] rollout-digest cross-check has ZERO shared rows between "
            f"seed{a} and seed{b} corpora — refusing (nothing was compared)"
        )
    n_identical = sum(1 for k in shared if rows_a[k] == rows_b[k])
    frac = n_identical / len(shared)
    report = {
        "dv": "rollout-digest cross-check (assumption 6): per-row identity across seeds",
        "seeds": [a, b],
        "n_shared_rows": len(shared),
        "n_identical": n_identical,
        "identical_fraction": frac,
        "identical_frac_max": IDENTICAL_FRAC_MAX,
        "status": "PASS" if frac < IDENTICAL_FRAC_MAX else "FAIL",
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(report, out_json)
    if frac >= IDENTICAL_FRAC_MAX:
        raise SystemExit(
            f"[sampled-figures] rollout-digest cross-check FAILED: seed{a} and seed{b} "
            f"corpora are (near-)identical ({n_identical}/{len(shared)} rows, fraction "
            f"{frac:.3f} >= {IDENTICAL_FRAC_MAX}) — vLLM per-request seeding did not "
            "differentiate the draws (assumption 6); the replication read is VOID"
        )
    print(f"[sampled-figures] rollout-digest cross-check PASS: {frac:.4f} identical fraction")
    return report


def main() -> int:  # noqa: C901 — linear figure battery, one block per registered panel
    ap = argparse.ArgumentParser(description="issue #1426 sampled-rollout robustness figures")
    ap.add_argument("--greedy-results", default=str(PROJECT_ROOT / "eval_results" / "issue_1426"))
    ap.add_argument(
        "--seed-results-root",
        default=str(PROJECT_ROOT / "eval_results" / "issue_1426" / "sampled-rollout-robustness"),
    )
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 137])
    ap.add_argument(
        "--seed-rollouts",
        nargs="*",
        default=None,
        help="per-seed rollout dirs (same order as --seeds); default "
        "data/issue_1426_sampled_s<seed>/raw_completions/thinking_rollouts",
    )
    ap.add_argument(
        "--greedy-rollouts",
        default=None,
        help="greedy reference rollout dir (staged from HF); when ABSENT the greedy "
        "overlay on the CoT-length panel is SKIPPED with a loud note (never silent)",
    )
    ap.add_argument(
        "--out-figures",
        default=str(PROJECT_ROOT / "figures" / "issue_1426" / "sampled-rollout-robustness"),
    )
    args = ap.parse_args()

    greedy_dir = Path(args.greedy_results)
    seed_root = Path(args.seed_results_root)
    out_dir = Path(args.out_figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    if args.seed_rollouts is not None and len(args.seed_rollouts) != len(seeds):
        raise SystemExit("[sampled-figures] --seed-rollouts must match --seeds in length")
    roll_dirs = {
        s: Path(
            args.seed_rollouts[i]
            if args.seed_rollouts is not None
            else PROJECT_ROOT
            / "data"
            / f"issue_1426_sampled_s{s}"
            / "raw_completions"
            / "thinking_rollouts"
        )
        for i, s in enumerate(seeds)
    }

    # ── assumption-6 digest cross-check FIRST: identical corpora void every
    # downstream panel — fail before drawing anything misleading. ─────────────
    digest_cross_check(roll_dirs, seed_root / "rollout_digest_check.json")

    set_paper_style("blog")
    pal = paper_palette_blog(6)
    rng = np.random.default_rng(RNG_SEED)
    dec_names = ["greedy"] + [f"seed {s}" for s in seeds]
    dec_colors = [pal[0], pal[3], pal[2]][: 1 + len(seeds)]

    g_pcd = jload(greedy_dir / "percontext_deltas.json")
    g_fam = jload(greedy_dir / "fam_contrast.json")
    g_cov = jload(greedy_dir / "coverage_by_family.json")
    g_lmg = jload(greedy_dir / "length_matched_gain.json")
    s_pcd = {s: jload(seed_root / f"seed{s}" / "percontext_deltas.json") for s in seeds}
    s_fam = {s: jload(seed_root / f"seed{s}" / "fam_contrast.json") for s in seeds}
    s_cov = {s: jload(seed_root / f"seed{s}" / "coverage_by_family.json") for s in seeds}
    s_lmg = {s: jload(seed_root / f"seed{s}" / "length_matched_gain.json") for s in seeds}

    # ------------------------------------------- fig 1 (hero a): mediation lattice
    arm_labels = [
        ("d_ctx2ans", "context\n→ answer"),
        ("b_cot2ans", "realized CoT\n→ answer"),
        ("comp_pred", "composed\n(ctx→ĈoT→ans)"),
        ("g_aug", "context+CoT\n→ answer"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for ax, regime, regname in (
        (axes[0], "indiv", "per-question (primary)"),
        (axes[1], "avg_q", "query-averaged"),
    ):
        sources = [("greedy", g_pcd, greedy_dir)] + [
            (f"seed {s}", s_pcd[s], seed_root / f"seed{s}") for s in seeds
        ]
        n_dec = len(sources)
        w = 0.8 / n_dec
        for di, (dec, pcd, blob_dir) in enumerate(sources):
            skills = arm_skills(pcd, regime)
            frozen = pcd["contrasts"]["h2_cot_gain_percontext"]["by_regime"][regime]["frozen_layer"]
            for ai, (arm, _lab) in enumerate(arm_labels):
                vals = skills[arm]
                m = float(vals.mean())
                lo, hi = boot_ci(vals, rng)
                x = ai + (di - (n_dec - 1) / 2) * w
                ax.bar(x, m, width=w * 0.92, color=dec_colors[di], label=dec if ai == 0 else None)
                ax.errorbar(
                    x,
                    m,
                    yerr=[[max(0.0, m - lo)], [max(0.0, hi - m)]],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                    lw=0.9,
                )
            band = null_band_p95(jload(blob_dir / f"null_matrix_{regime}.json"), dec)
            ceil = identity_ceiling(jload(blob_dir / "recon_skill_grid.json"), regime, int(frozen))
            ax.axhline(band, ls=":", color=dec_colors[di], lw=1.1, alpha=0.85)
            ax.axhline(ceil, ls="--", color=dec_colors[di], lw=1.1, alpha=0.85)
        ax.set_xticks(range(len(arm_labels)), [lab for _a, lab in arm_labels], fontsize=9)
        ax.set_ylabel("held-out skill-over-mean R²")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Mediation arms, {regname}", fontsize=12, pad=10)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(
        "dotted: selection-symmetric null band p95 · dashed: identity ceiling "
        "(color-matched per decoding)",
        fontsize=9,
        y=0.995,
    )
    fig.tight_layout()
    savefig_paper(fig, "mediation_lattice_greedy_vs_seeds", dir=str(out_dir))
    plt.close(fig)

    # ------------------------------------------- fig 2 (hero b): family-excess forest
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, regime, regname in (
        (axes[0], "indiv", "per-question (primary)"),
        (axes[1], "avg_q", "query-averaged"),
    ):
        g = g_fam["by_regime"][regime]
        ax.axvspan(g["ci95"][0], g["ci95"][1], color=dec_colors[0], alpha=0.18)
        rows = [("greedy", g)] + [(f"seed {s}", s_fam[s]["by_regime"][regime]) for s in seeds]
        for i, (dec, b) in enumerate(rows):
            y = len(rows) - 1 - i
            obs, ci = b["observed"], b["ci95"]
            ax.errorbar(
                obs,
                y,
                xerr=[[max(0.0, obs - ci[0])], [max(0.0, ci[1] - obs)]],
                fmt="o",
                color=dec_colors[i],
                capsize=4,
                markersize=7,
            )
            ax.text(obs, y + 0.18, f"{obs:+.3f}", ha="center", fontsize=9)
        ax.axvline(0, ls="--", color="0.6")
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=10)
        ax.set_xlabel("Δ family gain excess (ICL+WildChat − length-matched donors)")
        ax.set_title(f"{regname} — greedy CI shaded (containment band)", fontsize=11, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "fam_excess_forest_greedy_vs_seeds", dir=str(out_dir))
    plt.close(fig)

    # ------------------------------------------- fig 3: coverage bars + taxonomy
    covs = [("greedy", g_cov)] + [(f"seed {s}", s_cov[s]) for s in seeds]
    fams = [f for f in FAMILIES if all(f in c["families"] for _n, c in covs)]
    x = np.arange(len(fams))
    n_dec = len(covs)
    w = 0.8 / n_dec
    reasons_all = sorted(
        {reason for _n, c in covs for f in fams for reason in c["families"][f].get("reasons", {})}
    )
    grey = plt.cm.Greys(np.linspace(0.35, 0.8, max(1, len(reasons_all))))
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    for di, (dec, cov) in enumerate(covs):
        xs = x + (di - (n_dec - 1) / 2) * w
        usable = np.array([cov["families"][f]["usable_rate"] for f in fams])
        ax.bar(xs, usable, w * 0.92, color=dec_colors[di], label=dec)
        bottom = usable.copy()
        for ri, reason in enumerate(reasons_all):
            frac = np.array(
                [
                    cov["families"][f].get("reasons", {}).get(reason, 0)
                    / max(1, cov["families"][f]["n_rows"])
                    for f in fams
                ]
            )
            ax.bar(
                xs,
                frac,
                w * 0.92,
                bottom=bottom,
                color=grey[ri],
                label=reason if di == 0 else None,
            )
            bottom += frac
    ax.axhline(0.95, ls=":", color="0.4")
    ax.set_xticks(x, fams)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("row fraction (usable colored, malformed reasons stacked grey)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax.set_title("Scaffold compliance per family — sampled seeds vs greedy", fontsize=12, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "coverage_by_family_sampled_vs_greedy", dir=str(out_dir))
    plt.close(fig)

    # ------------------------------------- fig 4: per-context gain scatter vs greedy
    g_gain = gain_by_context(g_pcd, "indiv")
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.6 * len(seeds), 5.2), squeeze=False)
    for si, s in enumerate(seeds):
        ax = axes[0][si]
        s_gain = gain_by_context(s_pcd[s], "indiv")
        common = sorted(set(g_gain) & set(s_gain))
        gx = np.array([g_gain[c] for c in common])
        sy = np.array([s_gain[c] for c in common])
        ax.scatter(gx, sy, s=18, color=dec_colors[1 + si], alpha=0.85)
        for c in common:
            ax.annotate(c, (g_gain[c], s_gain[c]), fontsize=4.5, alpha=0.7)
        lim = [
            min(gx.min(), sy.min()) - 0.03,
            max(gx.max(), sy.max()) + 0.03,
        ]
        ax.plot(lim, lim, ls="--", color="0.6", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("greedy per-context CoT gain (Δ skill)")
        ax.set_ylabel(f"seed {s} per-context CoT gain (Δ skill)")
        rho = float(np.corrcoef(gx, sy)[0, 1])
        ax.set_title(f"seed {s} vs greedy, per-question (r={rho:.3f})", fontsize=11, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "percontext_gain_scatter_vs_greedy", dir=str(out_dir))
    plt.close(fig)

    # ------------------------------------------- fig 5: CoT-length distributions
    len_sources = []
    if args.greedy_rollouts:
        len_sources.append(("greedy", rollout_lengths_by_family(Path(args.greedy_rollouts))))
    else:
        print(
            "[sampled-figures] NOTE: --greedy-rollouts absent — greedy overlay on the "
            "CoT-length panel SKIPPED (stage the greedy rollouts from HF to include it)"
        )
    for s in seeds:
        len_sources.append((f"seed {s}", rollout_lengths_by_family(roll_dirs[s])))
    lfams = [f for f in FAMILIES if all(f in d for _n, d in len_sources)]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    n_dec = len(len_sources)
    w = 0.8 / n_dec
    for di, (dec, d) in enumerate(len_sources):
        color = dec_colors[di] if args.greedy_rollouts else dec_colors[1 + di]
        pos = np.arange(len(lfams)) + (di - (n_dec - 1) / 2) * w
        bp = ax.boxplot(
            [d[f] for f in lfams],
            positions=pos,
            widths=w * 0.85,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black"},
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.75)
        ax.plot([], [], color=color, lw=6, alpha=0.75, label=dec)
    ax.set_xticks(np.arange(len(lfams)), lfams)
    ax.set_ylabel("well-formed CoT length (chars before </think>)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("CoT-length distributions per family", fontsize=12, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "cot_length_by_family", dir=str(out_dir))
    plt.close(fig)

    # ------------------------------------------- fig 6: tercile profiles vs greedy
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xs = [0, 1, 2]
    for di, (dec, lmg) in enumerate([("greedy", g_lmg)] + [(f"seed {s}", s_lmg[s]) for s in seeds]):
        terc = lmg["by_regime"]["indiv"]["noncollapse_short_cot_terciles"]
        obs = [t["observed"] for t in terc]
        lo = [max(0.0, t["observed"] - t["ci95"][0]) for t in terc]
        hi = [max(0.0, t["ci95"][1] - t["observed"]) for t in terc]
        ax.errorbar(
            xs,
            obs,
            yerr=[lo, hi],
            fmt="o-",
            color=dec_colors[di],
            capsize=4,
            markersize=7,
            label=dec,
        )
    ax.set_xticks(xs, ["shortest third", "middle third", "longest third"])
    ax.set_xlabel("median well-formed CoT length tercile")
    ax.set_ylabel("per-context CoT gain (Δ skill)")
    ax.legend(frameon=False, fontsize=10)
    ax.set_title("CoT-length tercile profile — sampled seeds vs greedy", fontsize=12, pad=10)
    fig.tight_layout()
    savefig_paper(fig, "tercile_profile_sampled_vs_greedy", dir=str(out_dir))
    plt.close(fig)

    # ------------------------------------------- fig 7: seed-vs-seed gain scatter
    if len(seeds) >= 2:
        a, b = seeds[0], seeds[1]
        ga, gb = gain_by_context(s_pcd[a], "indiv"), gain_by_context(s_pcd[b], "indiv")
        common = sorted(set(ga) & set(gb))
        xa = np.array([ga[c] for c in common])
        yb = np.array([gb[c] for c in common])
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        ax.scatter(xa, yb, s=18, color=pal[4], alpha=0.85)
        for c in common:
            ax.annotate(c, (ga[c], gb[c]), fontsize=4.5, alpha=0.7)
        lim = [min(xa.min(), yb.min()) - 0.03, max(xa.max(), yb.max()) + 0.03]
        ax.plot(lim, lim, ls="--", color="0.6", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"seed {a} per-context CoT gain (Δ skill)")
        ax.set_ylabel(f"seed {b} per-context CoT gain (Δ skill)")
        rho = float(np.corrcoef(xa, yb)[0, 1])
        ax.set_title(
            f"Within-regime replication: seed {a} vs seed {b} (r={rho:.3f})",
            fontsize=11,
            pad=10,
        )
        fig.tight_layout()
        savefig_paper(fig, "seed_vs_seed_gain_scatter", dir=str(out_dir))
        plt.close(fig)
    else:
        print("[sampled-figures] NOTE: single seed — seed-vs-seed scatter SKIPPED")

    print(f"[sampled-figures] wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
