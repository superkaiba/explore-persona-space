# ruff: noqa: RUF001, RUF002, RUF003  # Qwen marker " ※", ×/−/χ² glyphs intentional
"""Task #472 follow-up `composition-matched-total` — composition at fixed total negatives.

#472's two count axes (negative examples per persona; number of negative
personas) are each perfectly confounded with the TOTAL negative-row budget
inside their own sweeps. The existing trajectory files allow a retrospective
de-confound: four cells share total = 400 negative rows (600 training rows →
identical absolute checkpoint grids {4, 7, 13, 19, 29, 38}), differing only in
COMPOSITION, and two cells share total = 1600 (1800 rows → grids
{10, 19, 38, 57, 85, 113}).

Realized compositions (verified against CELL_SPECS + select_negatives; the
follow-up spec's "1 persona × 400" framing for the single cells is corrected
here — every non-empty arm ALWAYS includes qwen_default, so "single" means ONE
placement persona BESIDE the default):

  PRIMARY (total = 400 negative rows each):
    c472_single_near  [qwen_default, hero]                       × 200 each
    c472_single_far   [qwen_default, ai_assistant]               × 200 each
    c472_negp_2       [qwen_default, hero]                       × 200 each
    c472_negex_100    [qwen_default, hero, journalist,
                       ai_assistant]                             × 100 each
  NOTE: c472_negp_2's realized negative set is IDENTICAL to c472_single_near's
  (spread placement with one free slot resolves to the nearest candidate), so
  that pair is two independent training runs of the SAME composition — a
  run-replication contrast, not a composition contrast. single_near vs
  single_far brackets the known small placement contribution inside the
  one-placement-persona composition.

  SECONDARY (total = 1600 negative rows each):
    c472_negex_400    4 personas × 400        c472_negp_8    8 personas × 200

Analyses (ANALYSIS-ONLY, CPU, seconds; no training / generation / pod):
  PRIMARY — at each of the 6 shared checkpoints, DIRECT checkpoint reads (the
  grids coincide exactly, so no interpolation and no interp-error floor):
  per-probe ΔG (on-policy log P(※) at the post-response slot, trained − base)
  seed-averaged over probes valid in both seeds, matched across the four
  compositions; paired Friedman across the 4 compositions + Kendall's W +
  pairwise Wilcoxon (all 6 pairs, mean/median paired diff, Cohen's dz), with
  Friedman p Holm-corrected across the 6 checkpoints. Per-composition means +
  95% bootstrap CIs over probes (own-valid AND matched sets), n probes, drop
  counts.
  SECONDARY — same per-checkpoint matched read for the 1600-total pair with a
  paired Wilcoxon per checkpoint, Holm across the 6.

Validity filters mirror ``issue472_reanalyze_count_matched_step.py`` exactly:
a probe is dropped at a checkpoint when its mean g_logp breaches the
sub-ceiling headroom (saturated) or any question row has a collapsed on-policy
response; seed-averaging keeps probes valid in BOTH seeds.

Inputs: ``eval_results/issue_472/{c472_single_near,c472_single_far,c472_negp_2,
c472_negex_100,c472_negex_400,c472_negp_8}_seed{42,137}/trajectory.json`` +
``data/issue_472/centroids_L10.pt`` (held-out panel definition only; offline).

Output: ``eval_results/issue_472/composition-matched-total/
reanalysis_composition_matched_total.json``.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import friedmanchisquare, wilcoxon

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    CELL_SPECS,
    POS_EX_PER_SOURCE,
    SUBCEILING_HEADROOM_NATS,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
    holm_correction,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
    cos_to_source as load_cos_to_source,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.select_negatives import (
    held_out_panel,
    negatives_for_cell,
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472"
OUT_DIR = SLAB / "composition-matched-total"
CENTROID_DIR = WT / "data" / "issue_472"
SEEDS = (42, 137)
SOURCE = "villain"
HEADLINE_LAYER = 10
N_BOOT = 10_000
BOOT_SEED = 0

# Composition groups at fixed total negative rows. Verified against CELL_SPECS
# in _assert_cell_mapping(); realized negative sets recorded in the output.
PRIMARY: dict[str, str] = {
    "single_near": "c472_single_near",
    "single_far": "c472_single_far",
    "negp_2": "c472_negp_2",
    "negex_100": "c472_negex_100",
}
SECONDARY: dict[str, str] = {
    "negex_400": "c472_negex_400",
    "negp_8": "c472_negp_8",
}
ALL_CELLS = [*PRIMARY.values(), *SECONDARY.values()]


def _assert_cell_mapping() -> None:
    """Verify compositions + fixed totals against CELL_SPECS."""
    spec = {c[0]: c for c in CELL_SPECS}
    expect = {
        "c472_single_near": ("near", 2, 200),
        "c472_single_far": ("far", 2, 200),
        "c472_negp_2": ("spread", 2, 200),
        "c472_negex_100": ("spread", 4, 100),
        "c472_negex_400": ("spread", 4, 400),
        "c472_negp_8": ("spread", 8, 200),
    }
    for slug, (placement, n_p, n_ex) in expect.items():
        got = (spec[slug][2], spec[slug][3], spec[slug][4])
        assert got == (placement, n_p, n_ex), f"{slug}: CELL_SPECS={got}, expected {expect[slug]}"
    primary_totals = {slug: spec[slug][3] * spec[slug][4] for slug in PRIMARY.values()}
    assert set(primary_totals.values()) == {400}, primary_totals
    secondary_totals = {slug: spec[slug][3] * spec[slug][4] for slug in SECONDARY.values()}
    assert set(secondary_totals.values()) == {1600}, secondary_totals


def _load_traj(cell: str, seed: int) -> dict:
    return json.loads((SLAB / f"{cell}_seed{seed}" / "trajectory.json").read_text())


def _sorted_cks(traj: dict) -> list[dict]:
    return sorted(traj["checkpoints"], key=lambda c: c["frac"])


def _per_probe(ck: dict) -> dict[str, dict]:
    """Persona-level probe stats at one checkpoint (same filters as the count read)."""
    out: dict[str, dict] = {}
    for persona, perq in ck["held_out"].items():
        dgs = [r["delta_g"] for r in perq.values()]
        gls = [r["g_logp"] for r in perq.values()]
        mean_g = float(np.mean(gls))
        out[persona] = {
            "delta_g": float(np.mean(dgs)),
            "saturated": mean_g > -SUBCEILING_HEADROOM_NATS,
            "r_collapsed": any(r.get("r_collapsed", False) for r in perq.values()),
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


class CellSeed:
    """One cell × seed trajectory: steps, per-checkpoint probe stats, source-self."""

    def __init__(self, cell: str, seed: int, panel: list[str]) -> None:
        traj = _load_traj(cell, seed)
        cks = _sorted_cks(traj)
        self.cell = cell
        self.seed = seed
        self.steps = [int(ck["step"]) for ck in cks]
        self.fracs = [float(ck["frac"]) for ck in cks]
        self.source_self = [float(ck["source_self"]["delta_g_mean"]) for ck in cks]
        self.panel = panel
        self.probe_stats = [_per_probe(ck) for ck in cks]

    def valid_probe_dgs(self, ck_idx: int) -> dict[str, float]:
        """{probe: ΔG} over panel probes valid (non-saturated, non-collapsed) at ck_idx."""
        stats = self.probe_stats[ck_idx]
        out = {}
        for probe in self.panel:
            st = stats.get(probe)
            if st is None or st["saturated"] or st["r_collapsed"]:
                continue
            out[probe] = st["delta_g"]
        return out

    def drop_counts(self, ck_idx: int) -> dict[str, int]:
        stats = self.probe_stats[ck_idx]
        sat = sum(1 for p in self.panel if p in stats and stats[p]["saturated"])
        coll = sum(
            1
            for p in self.panel
            if p in stats and not stats[p]["saturated"] and stats[p]["r_collapsed"]
        )
        return {"n_dropped_saturated": sat, "n_dropped_r_collapsed": coll}


def _seed_avg_direct(cs_by_seed: dict[int, CellSeed], ck_idx: int) -> dict[str, float]:
    """Seed-averaged per-probe ΔG at checkpoint ck_idx (probes valid in BOTH seeds).

    DIRECT checkpoint read — the composition groups share exact absolute-step
    grids, so no interpolation is involved anywhere in this analysis.
    """
    per_seed = {s: cs.valid_probe_dgs(ck_idx) for s, cs in cs_by_seed.items()}
    common = set.intersection(*(set(v) for v in per_seed.values()))
    return {p: float(np.mean([per_seed[s][p] for s in per_seed])) for p in sorted(common)}


def _comparison(comp_probes: dict[str, dict[str, float]], rng: np.random.Generator) -> dict:
    """Friedman (>2 groups) / Wilcoxon + pairwise stats on matched per-probe ΔG."""
    comps = list(comp_probes)
    matched = sorted(set.intersection(*(set(comp_probes[c]) for c in comps)))
    cols = {c: np.array([comp_probes[c][p] for p in matched]) for c in comps}
    pairwise: dict[str, dict] = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            diff = cols[a] - cols[b]
            wx = wilcoxon(cols[a], cols[b])
            pairwise[f"{a}_vs_{b}"] = {
                "wilcoxon_stat": float(wx.statistic),
                "wilcoxon_p": float(wx.pvalue),
                "mean_diff_nats": float(diff.mean()),
                "median_diff_nats": float(np.median(diff)),
                "cohen_dz": float(diff.mean() / diff.std(ddof=1)),
            }
    means = {c: float(cols[c].mean()) for c in comps}
    out = {
        "n_matched_probes": len(matched),
        "composition_means_matched": means,
        "composition_boot_ci95_matched": {c: _boot_ci(cols[c], rng) for c in comps},
        "between_composition_spread_nats": float(max(means.values()) - min(means.values())),
        "pairwise": pairwise,
    }
    if len(comps) > 2:
        fr = friedmanchisquare(*(cols[c] for c in comps))
        out["friedman_chi2"] = float(fr.statistic)
        out["friedman_p"] = float(fr.pvalue)
        out["kendalls_w"] = float(fr.statistic / (len(matched) * (len(comps) - 1)))
    return out


def _per_composition_block(
    cells: dict[str, dict[int, CellSeed]],
    group: dict[str, str],
    ck_idx: int,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """Per-composition own-valid stats at one checkpoint (mirrors the count read)."""
    out: dict[str, dict] = {}
    for label, slug in group.items():
        sa = _seed_avg_direct(cells[slug], ck_idx)
        arr = np.array(list(sa.values()))
        src = {str(s): cells[slug][s].source_self[ck_idx] for s in SEEDS}
        out[label] = {
            "cell": slug,
            "mean_delta_g": float(arr.mean()),
            "boot_ci95": _boot_ci(arr, rng),
            "n_probes_seed_avg": len(arr),
            "per_seed_drop_counts": {str(s): cells[slug][s].drop_counts(ck_idx) for s in SEEDS},
            "source_self_delta_g": {**src, "mean": float(np.mean(list(src.values())))},
        }
    return out


def _group_block(
    cells: dict[str, dict[int, CellSeed]],
    group: dict[str, str],
    rng: np.random.Generator,
    family_p_key: str,
) -> dict:
    """Per-checkpoint matched comparison for one fixed-total group + Holm family."""
    ref = cells[next(iter(group.values()))][SEEDS[0]]
    per_checkpoint: dict[str, dict] = {}
    family: dict[str, float] = {}
    for ck_idx, step in enumerate(ref.steps):
        comp_probes = {
            label: _seed_avg_direct(cells[slug], ck_idx) for label, slug in group.items()
        }
        cmp_ = _comparison(comp_probes, rng)
        cmp_["per_composition"] = _per_composition_block(cells, group, ck_idx, rng)
        per_checkpoint[f"step_{step}"] = cmp_
        family[f"step_{step}"] = cmp_[family_p_key]
    return {
        "per_checkpoint": per_checkpoint,
        "holm_across_checkpoints": {
            "family_statistic": family_p_key,
            **holm_correction(family),
        },
    }


def _print_summary(name: str, group: dict[str, str], block: dict) -> None:
    comps = list(group)
    print(f"── {name} (compositions {comps}) ──")
    for key, c in block["per_checkpoint"].items():
        means = ", ".join(f"{lb}:{c['composition_means_matched'][lb]:+.2f}" for lb in comps)
        stat = (
            f"Friedman χ²={c['friedman_chi2']:.2f} p={c['friedman_p']:.2e} W={c['kendalls_w']:.3f}"
            if "friedman_p" in c
            else f"Wilcoxon p={next(iter(c['pairwise'].values()))['wilcoxon_p']:.2e}"
        )
        print(
            f"  {key}: {means}  spread={c['between_composition_spread_nats']:.2f} nats "
            f"{stat} (n={c['n_matched_probes']})"
        )
    holm = block["holm_across_checkpoints"]
    rejected = [k for k, v in holm.items() if k != "family_statistic" and v["reject_null"]]
    n_targets = len(block["per_checkpoint"])
    print(f"  Holm rejects: {len(rejected)}/{n_targets} checkpoints\n")


def main() -> None:
    _assert_cell_mapping()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    cts = load_cos_to_source(HEADLINE_LAYER, SOURCE, CENTROID_DIR)
    panel = held_out_panel(cts, source=SOURCE)
    negatives = {slug: negatives_for_cell(slug, cts) for slug in ALL_CELLS}
    cells: dict[str, dict[int, CellSeed]] = {
        slug: {s: CellSeed(slug, s, panel) for s in SEEDS} for slug in ALL_CELLS
    }
    # Grids must agree across seeds per cell AND across cells within each group.
    for slug in ALL_CELLS:
        assert cells[slug][SEEDS[0]].steps == cells[slug][SEEDS[1]].steps, slug
    for group in (PRIMARY, SECONDARY):
        slugs = list(group.values())
        ref_steps = cells[slugs[0]][SEEDS[0]].steps
        for slug in slugs[1:]:
            got = cells[slug][SEEDS[0]].steps
            assert got == ref_steps, f"grid mismatch in group: {slug}={got} vs {ref_steps}"

    primary_grid = cells[PRIMARY["single_near"]][SEEDS[0]].steps
    secondary_grid = cells[SECONDARY["negex_400"]][SEEDS[0]].steps
    print(f"Held-out panel (L{HEADLINE_LAYER}): {len(panel)} probes")
    print(f"PRIMARY grid (total=400): {primary_grid}")
    print(f"SECONDARY grid (total=1600): {secondary_grid}")
    print(f"Realized negative sets: {negatives}\n")

    primary_block = _group_block(cells, PRIMARY, rng, family_p_key="friedman_p")

    # SECONDARY family statistic = the single pairwise Wilcoxon p per checkpoint.
    secondary_block = _group_block_secondary(cells, rng)

    _print_summary("PRIMARY total=400", PRIMARY, primary_block)
    _print_summary("SECONDARY total=1600", SECONDARY, secondary_block)

    out = {
        "schema": "i472_reanalysis_composition_matched_total",
        "followup_label": "composition-matched-total",
        "dv": "on-policy log P(marker) at post-response slot, trained - base (delta_g), nats",
        "primary_compositions": PRIMARY,
        "secondary_compositions": SECONDARY,
        "realized_negative_sets": negatives,
        "composition_notes": {
            "single_cells_realized": (
                "c472_single_near / c472_single_far are 2 negative personas × 200 ex "
                "(qwen_default + ONE near/far placement persona), NOT 1 × 400: every "
                "non-empty arm always includes qwen_default. Total = 400 negative rows "
                f"+ {POS_EX_PER_SOURCE} positives = 600 training rows, as required for "
                "the shared checkpoint grid."
            ),
            "negp_2_equals_single_near": (
                "c472_negp_2's realized negative set is identical to c472_single_near's "
                "([qwen_default, hero]; spread placement with one free slot resolves to "
                "the nearest candidate). That pair is two independent training runs of "
                "the SAME composition (run-replication contrast), not a composition "
                "contrast."
            ),
            "placement_bracket": (
                "single_near vs single_far brackets the known small placement "
                "contribution inside the one-placement-persona composition."
            ),
        },
        "seeds": list(SEEDS),
        "source": SOURCE,
        "n_held_out_panel": len(panel),
        "validity_filters": {
            "drop_r_collapsed": True,
            "drop_saturated": True,
            "saturation_rule": f"mean g_logp > -{SUBCEILING_HEADROOM_NATS} nats",
            "seed_pooling": "per-probe mean over probes valid in BOTH seeds",
            "matched_set": "intersection of both-seed-valid probes across the group's cells",
        },
        "direct_checkpoint_read": True,
        "direct_read_note": (
            "All cells within each fixed-total group share exact absolute-step "
            "checkpoint grids, so every read is a DIRECT checkpoint read — no "
            "interpolation anywhere, hence no interpolation-error floor applies."
        ),
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "ci": "95% percentile, over probes"},
        "step_grids": {slug: cells[slug][SEEDS[0]].steps for slug in ALL_CELLS},
        "primary_total_400": primary_block,
        "secondary_total_1600": secondary_block,
        "reproducibility": {
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "inputs": [
                f"eval_results/issue_472/{slug}_seed{s}/trajectory.json"
                for slug in ALL_CELLS
                for s in SEEDS
            ]
            + [f"data/issue_472/centroids_L{HEADLINE_LAYER}.pt"],
        },
    }
    out_path = OUT_DIR / "reanalysis_composition_matched_total.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_path}")


def _group_block_secondary(cells: dict[str, dict[int, CellSeed]], rng: np.random.Generator) -> dict:
    """SECONDARY block: per-checkpoint paired Wilcoxon, Holm across the 6 checkpoints."""
    ref = cells[next(iter(SECONDARY.values()))][SEEDS[0]]
    per_checkpoint: dict[str, dict] = {}
    family: dict[str, float] = {}
    for ck_idx, step in enumerate(ref.steps):
        comp_probes = {
            label: _seed_avg_direct(cells[slug], ck_idx) for label, slug in SECONDARY.items()
        }
        cmp_ = _comparison(comp_probes, rng)
        cmp_["per_composition"] = _per_composition_block(cells, SECONDARY, ck_idx, rng)
        per_checkpoint[f"step_{step}"] = cmp_
        family[f"step_{step}"] = next(iter(cmp_["pairwise"].values()))["wilcoxon_p"]
    return {
        "per_checkpoint": per_checkpoint,
        "holm_across_checkpoints": {
            "family_statistic": "wilcoxon_p (negex_400 vs negp_8)",
            **holm_correction(family),
        },
    }


if __name__ == "__main__":
    main()
