# ruff: noqa: RUF001, RUF002, RUF003  # Qwen marker " ※", ×/−/ρ/χ² glyphs intentional
"""Task #472 follow-up `placement-null-full-trajectory` — placement null at ALL 6 checkpoints.

The committed placement read (near / spread / far indistinguishable in bystander
leakage, ``issue472_reanalyze_earliest_slice.py``) was taken at the EARLIEST
checkpoint only (frac 0.08, step 6), where the common "push marker up
everywhere" gradient component plausibly dominates and contrastive
differentiation between arms may not yet have developed. This script extends
the SAME comparison — near (c472_near) / spread (c472_anchor) / far (c472_far),
matched on row count (4 neg personas × 200 ex) AND on absolute training step at
every checkpoint — to ALL 6 checkpoints (frac 0.08/0.16/0.33/0.50/0.75/1.00 =
steps 6/11/21/32/48/63), using the EXISTING trajectory.json files. No new
training, no generation, no pod. CPU only; the full run takes seconds.

Per checkpoint:
  1. per-arm × seed mean bystander ΔG (on-policy log P(※) at the post-response
     slot, trained − base) over the 47-persona held-out panel, with the prior
     re-analysis's validity filters (drop r_collapsed probes; drop + flag
     saturated probes), plus a 95% bootstrap CI over probes (seeds pooled);
  2. matched arm-separation test: the probe grid is identical across arms, so
     Friedman across the 3 arms on probe-level seed-averaged ΔG (probes valid
     in all 3 arms × both seeds), Kendall's W effect size, pairwise Wilcoxon
     signed-rank with mean paired difference (nats) + Cohen's dz, and the
     between-arm spread of arm means (max−min, nats); Friedman p-values are
     Holm-corrected across the 6 checkpoints;
  3. proximity-to-source gradient: Spearman(ΔG, d_source) pooled across
     arms × seeds, at L10 (headline) and L15/L20 (robustness);
  4. source-self ΔG per arm (implant-level context row);
  5. emission context (scope note, NOT a gate): bystander argmax-marker counts
     per arm.

Falsification framing: if the arms separate at later checkpoints, the placement
null was an artifact of the too-early read and must be DOWNGRADED.

Inputs: ``eval_results/issue_472/{c472_near,c472_anchor,c472_far}_seed{42,137}/
trajectory.json`` + ``data/issue_472/centroids_L{10,15,20}.pt`` (pull from HF
``superkaiba1/explore-persona-space-data`` path ``issue472_neg_geometry/
geometry/`` first — see the clean-result Reproducibility section; this script
itself is offline).

Output: ``eval_results/issue_472/placement-null-full-trajectory/
reanalysis_placement_full_trajectory.json``.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    CELL_SPECS,
    SUBCEILING_HEADROOM_NATS,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
    holm_correction,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source as load_cos_to_source,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    d_source,
    held_out_panel,
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472"
OUT_DIR = SLAB / "placement-null-full-trajectory"
CENTROID_DIR = WT / "data" / "issue_472"
SEEDS = (42, 137)
SOURCE = "villain"
HEADLINE_LAYER = 10
GRADIENT_LAYERS = (10, 15, 20)
# Placement-arm mapping, asserted against CELL_SPECS in _assert_arm_mapping().
ARMS: dict[str, str] = {"near": "c472_near", "spread": "c472_anchor", "far": "c472_far"}
N_BOOT = 10_000
BOOT_SEED = 0
EXPECTED_FRACS = (0.08, 0.16, 0.33, 0.50, 0.75, 1.00)


def _assert_arm_mapping() -> None:
    """Verify ARMS against CELL_SPECS: placement label + count-matching (4 × 200)."""
    spec = {c[0]: c for c in CELL_SPECS}
    for label, slug in ARMS.items():
        assert spec[slug][2] == label, (
            f"{slug}: CELL_SPECS placement={spec[slug][2]!r}, expected {label!r}"
        )
        assert spec[slug][3] == 4 and spec[slug][4] == 200, (
            f"{slug}: not count-matched (n_neg_personas={spec[slug][3]}, "
            f"neg_ex={spec[slug][4]}); placement arms must be 4 × 200"
        )
        assert spec[slug][5], f"{slug}: expected in_pooled=True (placement sub-study arm)"


def _load_traj(cell: str, seed: int) -> dict:
    return json.loads((SLAB / f"{cell}_seed{seed}" / "trajectory.json").read_text())


def _sorted_cks(traj: dict) -> list[dict]:
    return sorted(traj["checkpoints"], key=lambda c: c["frac"])


def _per_probe(ck: dict) -> dict[str, dict]:
    """Persona-level probe stats at one checkpoint (mean over the 10 questions).

    Mirrors the prior re-analysis's validity convention: a probe is dropped from
    ΔG aggregation when its mean g_logp breaches the sub-ceiling headroom
    (saturated) or any of its question rows has a collapsed on-policy response.
    """
    out: dict[str, dict] = {}
    for persona, perq in ck["held_out"].items():
        dgs = [r["delta_g"] for r in perq.values()]
        gls = [r["g_logp"] for r in perq.values()]
        mean_g = float(np.mean(gls))
        out[persona] = {
            "delta_g": float(np.mean(dgs)),
            "saturated": mean_g > -SUBCEILING_HEADROOM_NATS,
            "r_collapsed": any(r.get("r_collapsed", False) for r in perq.values()),
            "n_argmax_marker": int(sum(bool(r.get("argmax_marker", False)) for r in perq.values())),
            "n_rows": len(perq),
        }
    return out


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _boot_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    """95% percentile bootstrap CI of the mean, resampling probes."""
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _arm_checkpoint_stats(
    label: str,
    ci: int,
    pp: dict,
    trajs: dict,
    panel: list[str],
    rng: np.random.Generator,
) -> tuple[dict, dict[str, float]]:
    """Aggregate one arm at one checkpoint.

    Returns (per_arm JSON entry, {probe: seed-averaged ΔG over probes valid in
    BOTH seeds}). Validity per seed: probe present, not saturated, not
    r_collapsed (the prior re-analysis's filters).
    """
    seed_means: dict[str, float] = {}
    n_valid: dict[str, int] = {}
    n_sat: dict[str, int] = {}
    n_coll: dict[str, int] = {}
    valid_vals: dict[str, dict[str, float]] = {}
    argmax_ct: dict[str, dict[str, int]] = {}
    for s in SEEDS:
        stats = pp[label][s][ci]
        vals: dict[str, float] = {}
        sat = coll = am_true = am_rows = 0
        for probe in panel:
            if probe not in stats:
                continue
            st = stats[probe]
            am_true += st["n_argmax_marker"]
            am_rows += st["n_rows"]
            if st["saturated"]:
                sat += 1
                continue
            if st["r_collapsed"]:
                coll += 1
                continue
            vals[probe] = st["delta_g"]
        seed_means[str(s)] = float(np.mean(list(vals.values())))
        n_valid[str(s)] = len(vals)
        n_sat[str(s)] = sat
        n_coll[str(s)] = coll
        valid_vals[str(s)] = vals
        argmax_ct[str(s)] = {"n_argmax_marker_true": am_true, "n_probe_rows": am_rows}

    both = sorted(set(valid_vals[str(SEEDS[0])]) & set(valid_vals[str(SEEDS[1])]))
    sa = {p: float(np.mean([valid_vals[str(s)][p] for s in SEEDS])) for p in both}
    arr = np.array(list(sa.values()))
    src = {
        str(s): float(_sorted_cks(trajs[label][s])[ci]["source_self"]["delta_g_mean"])
        for s in SEEDS
    }
    entry = {
        "cell": ARMS[label],
        "per_seed_mean_delta_g": seed_means,
        "pooled_mean_delta_g": float(arr.mean()),
        "boot_ci95": _boot_ci(arr, rng),
        "n_probes_seed_avg": len(sa),
        "n_valid_probes_per_seed": n_valid,
        "n_dropped_saturated_per_seed": n_sat,
        "n_dropped_r_collapsed_per_seed": n_coll,
        "source_self_delta_g": {**src, "mean": float(np.mean(list(src.values())))},
        "bystander_argmax_marker": argmax_ct,
    }
    return entry, sa


def _matched_comparison(seed_avg: dict[str, dict[str, float]]) -> tuple[dict, float]:
    """Friedman + pairwise Wilcoxon over probes valid in ALL arms (both seeds)."""
    matched = sorted(set.intersection(*(set(seed_avg[a]) for a in ARMS)))
    cols = {a: np.array([seed_avg[a][p] for p in matched]) for a in ARMS}
    fr = friedmanchisquare(*(cols[a] for a in ARMS))
    kendalls_w = float(fr.statistic / (len(matched) * (len(ARMS) - 1)))
    pairwise: dict[str, dict] = {}
    for a, b in [("near", "spread"), ("near", "far"), ("spread", "far")]:
        diff = cols[a] - cols[b]
        wx = wilcoxon(cols[a], cols[b])
        pairwise[f"{a}_vs_{b}"] = {
            "wilcoxon_stat": float(wx.statistic),
            "wilcoxon_p": float(wx.pvalue),
            "mean_diff_nats": float(diff.mean()),
            "median_diff_nats": float(np.median(diff)),
            "cohen_dz": float(diff.mean() / diff.std(ddof=1)),
        }
    matched_means = {a: float(cols[a].mean()) for a in ARMS}
    comparison = {
        "n_matched_probes": len(matched),
        "matched_arm_means": matched_means,
        "matched_arm_mean_spread_nats": float(
            max(matched_means.values()) - min(matched_means.values())
        ),
        "friedman_chi2": float(fr.statistic),
        "friedman_p": float(fr.pvalue),
        "kendalls_w": kendalls_w,
        "pairwise": pairwise,
    }
    return comparison, float(fr.pvalue)


def _proximity_gradient(
    ci: int,
    pp: dict,
    panel_by_layer: dict[int, list[str]],
    cts_by_layer: dict[int, dict[str, float]],
) -> dict[str, dict]:
    """Spearman(ΔG, d_source) pooled across arms × seeds, per centroid layer."""
    gradient: dict[str, dict] = {}
    for ly in GRADIENT_LAYERS:
        cts = cts_by_layer[ly]
        xs, ys = [], []
        for label in ARMS:
            for s in SEEDS:
                stats = pp[label][s][ci]
                for probe in panel_by_layer[ly]:
                    if probe not in stats:
                        continue
                    st = stats[probe]
                    if st["saturated"] or st["r_collapsed"]:
                        continue
                    xs.append(d_source(probe, cts))
                    ys.append(st["delta_g"])
        rho = spearmanr(xs, ys)
        gradient[f"L{ly}"] = {
            "spearman_delta_g_vs_d_source": float(rho.correlation),
            "p": float(rho.pvalue),
            "n": len(ys),
        }
    return gradient


def main() -> None:
    _assert_arm_mapping()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    # ── Load all trajectories; assert checkpoint alignment across arms × seeds. ──
    trajs = {label: {s: _load_traj(slug, s) for s in SEEDS} for label, slug in ARMS.items()}
    ref_steps = [(ck["frac"], ck["step"]) for ck in _sorted_cks(trajs["near"][SEEDS[0]])]
    assert [round(f, 2) for f, _ in ref_steps] == [round(f, 2) for f in EXPECTED_FRACS], ref_steps
    for label in ARMS:
        for s in SEEDS:
            got = [(ck["frac"], ck["step"]) for ck in _sorted_cks(trajs[label][s])]
            assert got == ref_steps, (
                f"checkpoint misalignment {label} seed {s}: {got} != {ref_steps}"
            )
    n_cks = len(ref_steps)

    # ── Geometry: held-out panel + per-layer distance-to-source. ──
    cts_by_layer = {ly: load_cos_to_source(ly, SOURCE, CENTROID_DIR) for ly in GRADIENT_LAYERS}
    panel = held_out_panel(cts_by_layer[HEADLINE_LAYER], source=SOURCE)
    panel_by_layer = {ly: held_out_panel(cts_by_layer[ly], source=SOURCE) for ly in GRADIENT_LAYERS}
    print(f"Held-out panel (L{HEADLINE_LAYER}): {len(panel)} probes; arms: {ARMS}")
    print(f"Checkpoints: {ref_steps}\n")

    # Pre-compute per-probe stats: pp[label][seed][ck_idx] -> {persona: stats}.
    pp = {
        label: {s: [_per_probe(ck) for ck in _sorted_cks(trajs[label][s])] for s in SEEDS}
        for label in ARMS
    }

    checkpoints_out: list[dict] = []
    friedman_family: dict[str, float] = {}

    for ci in range(n_cks):
        frac, step = ref_steps[ci]
        per_arm: dict[str, dict] = {}
        seed_avg: dict[str, dict[str, float]] = {}  # label -> {probe: seed-avg ΔG (valid only)}
        for label in ARMS:
            per_arm[label], seed_avg[label] = _arm_checkpoint_stats(
                label, ci, pp, trajs, panel, rng
            )

        comparison, friedman_p = _matched_comparison(seed_avg)
        friedman_family[f"ck{ci + 1}_frac{frac}"] = friedman_p
        pooled_means = {a: per_arm[a]["pooled_mean_delta_g"] for a in ARMS}
        spread_nats = float(max(pooled_means.values()) - min(pooled_means.values()))
        gradient = _proximity_gradient(ci, pp, panel_by_layer, cts_by_layer)

        checkpoints_out.append(
            {
                "index": ci,
                "frac": frac,
                "step": step,
                "per_arm": per_arm,
                "arm_mean_spread_nats": spread_nats,
                "matched_comparison": comparison,
                "proximity_gradient": gradient,
            }
        )

        print(f"── ck{ci + 1} frac={frac} step={step} ──")
        for a in ARMS:
            pa = per_arm[a]
            lo, hi = pa["boot_ci95"]
            print(
                f"  {a:7s}: ΔG={pa['pooled_mean_delta_g']:+.3f} nats "
                f"[{lo:+.3f}, {hi:+.3f}]  (seeds {pa['per_seed_mean_delta_g']})  "
                f"src-self={pa['source_self_delta_g']['mean']:.2f}"
            )
        print(
            f"  spread(max−min)={spread_nats:.3f} nats; "
            f"Friedman χ²={comparison['friedman_chi2']:.2f} p={comparison['friedman_p']:.4f} "
            f"W={comparison['kendalls_w']:.3f} (n={comparison['n_matched_probes']})"
        )
        g10 = gradient[f"L{HEADLINE_LAYER}"]
        print(
            f"  proximity L{HEADLINE_LAYER}: ρ={g10['spearman_delta_g_vs_d_source']:+.3f} "
            f"p={g10['p']:.2e} n={g10['n']}\n"
        )

    # ── Holm across the 6 checkpoints (Friedman family). ──
    holm = holm_correction(friedman_family)
    print("── Holm (Friedman arm-separation, 6 checkpoints) ──")
    any_reject = False
    per_ck_verdict: dict[str, str] = {}
    for k, v in holm.items():
        verdict = "SEPARATED" if v["reject_null"] else "indistinguishable"
        per_ck_verdict[k] = verdict
        any_reject = any_reject or bool(v["reject_null"])
        print(
            f"  {k}: p={v['p']:.4f} thresh={v['holm_threshold']:.4f} "
            f"reject={v['reject_null']} -> {verdict}"
        )
    terminal_key = f"ck{n_cks}_frac{ref_steps[-1][0]}"
    terminal_verdict = per_ck_verdict[terminal_key]
    overall = (
        "placement null DOWNGRADED: arms separate at >=1 checkpoint under Holm"
        if any_reject
        else "placement null HOLDS across the full trajectory (no checkpoint separates under Holm)"
    )
    print(f"\nTerminal checkpoint ({terminal_key}): {terminal_verdict}")
    print(f"Overall: {overall}")

    out = {
        "schema": "i472_reanalysis_placement_full_trajectory",
        "followup_label": "placement-null-full-trajectory",
        "dv": "on-policy log P(marker) at post-response slot, trained - base (delta_g), nats",
        "arms": ARMS,
        "seeds": list(SEEDS),
        "source": SOURCE,
        "n_held_out_panel": len(panel),
        "validity_filters": {
            "drop_r_collapsed": True,
            "drop_saturated": True,
            "saturation_rule": f"mean g_logp > -{SUBCEILING_HEADROOM_NATS} nats",
        },
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "ci": "95% percentile, over probes"},
        "checkpoints": checkpoints_out,
        "holm_friedman_across_checkpoints": holm,
        "verdicts": {
            "per_checkpoint": per_ck_verdict,
            "terminal_checkpoint": {terminal_key: terminal_verdict},
            "overall": overall,
        },
        "reproducibility": {
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "inputs": [
                f"eval_results/issue_472/{slug}_seed{s}/trajectory.json"
                for slug in ARMS.values()
                for s in SEEDS
            ]
            + [f"data/issue_472/centroids_L{ly}.pt" for ly in GRADIENT_LAYERS],
        },
    }
    out_path = OUT_DIR / "reanalysis_placement_full_trajectory.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
