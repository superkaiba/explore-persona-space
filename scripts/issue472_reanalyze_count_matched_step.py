# ruff: noqa: RUF001, RUF002, RUF003  # Qwen marker " ※", ×/−/ρ/χ² glyphs intentional
"""Task #472 follow-up `placement-null-full-trajectory` — matched-step COUNT read.

#472's count finding ("more negatives = more leakage": negex 100/200/400 →
4.3/7.5/14.9 nats; negp 2/4/8 → 4.1/7.5/14.7 nats) is training-step confounded:
higher-count cells were read at more absolute optimizer steps (their epochs
contain more rows). The existing trajectory files allow a retrospective
de-confound — the count cells' checkpoint grids OVERLAP in absolute-step space
(negex_100/negp_2: steps 4–38; anchor: 6–63; negex_400/negp_8: 10–113; noneg:
2–13), so bystander ΔG can be compared at MATCHED absolute step via linear
interpolation between adjacent checkpoints.

Analyses (ANALYSIS-ONLY, CPU, seconds; no training / generation / pod):
  1. Step-space trajectories: per cell × seed, mean bystander ΔG (valid probes
     only; r_collapsed + saturated dropped and counted) and source-self ΔG vs
     ABSOLUTE step, with seed-pooled 95% bootstrap CIs over probes
     (figure-ready).
  2. Matched-STEP comparison: targets chosen from ACTUAL checkpoint steps
     inside the common overlap of the five count cells ([10, 38] → targets
     {10, 13, 19, 29, 38}: the overlap edge 10 = a real checkpoint of the
     high-count cells, plus the shortest-grid cell's checkpoints 13/19/29/38 —
     minimizes interpolation distance). Per-probe ΔG linearly interpolated in
     step per cell × seed (probe must be valid at both bracketing
     checkpoints), seed-averaged, then per count axis (negex 100/200/400, negp
     2/4/8) Friedman across the 3 levels + Kendall's W + pairwise Wilcoxon
     (mean paired diff, Cohen's dz), Holm across targets per axis,
     between-level spread (max−min, nats). Interpolation-error resolution
     floor = max + mean abs PER-PROBE |interpolated − nearest-checkpoint| ΔG,
     computed on the EXACT matched probe set entering each axis × target
     comparison, broken out by target/level/seed (review fix: the previous
     cell-mean |interp − nearest| cancels per-probe errors and is kept only
     as a drift diagnostic, NOT a bound). A target equal to an exact
     checkpoint step reads that checkpoint directly — no interval, no
     adjacent-checkpoint validity requirement (review fix: the interval form
     dropped probes valid at the exact checkpoint but not at its neighbour).
     `c472_noneg` is reported as a reference at targets inside its
     step range (≤13); it joins no test (not count-matched).
  3. Matched-IMPLANT comparison: each cell's source-self ΔG trajectory is
     searched for first crossings of fixed implant targets (10 and 15 nats per
     the follow-up spec). If a target is unreachable for a level (trajectory
     starts above it or never reaches it), that is REPORTED — with the
     per-cell source-self bands and the common-range computation — rather than
     silently skipped.

Decision framing (spec): count-level curves coinciding at matched step (and
matched implant) → the #472 count finding downgrades to a training-budget
effect; higher-count cells sitting above at matched step/implant → the
negatives themselves add leakage. Report whatever the data shows.

Inputs: ``eval_results/issue_472/{c472_negex_100,c472_anchor,c472_negex_400,
c472_negp_2,c472_negp_8,c472_noneg}_seed{42,137}/trajectory.json`` +
``data/issue_472/centroids_L10.pt`` (held-out panel definition only; pull from
HF ``issue472_neg_geometry/geometry/`` first — this script is offline).

Output: ``eval_results/issue_472/placement-null-full-trajectory/
reanalysis_count_matched_step.json``.
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
)

WT = Path("/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-472")
SLAB = WT / "eval_results" / "issue_472"
OUT_DIR = SLAB / "placement-null-full-trajectory"
CENTROID_DIR = WT / "data" / "issue_472"
SEEDS = (42, 137)
SOURCE = "villain"
HEADLINE_LAYER = 10
N_BOOT = 10_000
BOOT_SEED = 0
IMPLANT_TARGETS_NATS = (10.0, 15.0)  # per the follow-up spec

# Count axes (levels share the anchor). Verified against CELL_SPECS in
# _assert_cell_mapping(). noneg is a reference only (not count-matched).
AXES: dict[str, dict[str, str]] = {
    "negex": {"100": "c472_negex_100", "200": "c472_anchor", "400": "c472_negex_400"},
    "negp": {"2": "c472_negp_2", "4": "c472_anchor", "8": "c472_negp_8"},
}
NONEG = "c472_noneg"
COUNT_CELLS = sorted({slug for ax in AXES.values() for slug in ax.values()})
ALL_CELLS = [*COUNT_CELLS, NONEG]


def _assert_cell_mapping() -> None:
    """Verify axis levels against CELL_SPECS (placement + per-level counts)."""
    spec = {c[0]: c for c in CELL_SPECS}
    expect = {
        "c472_negex_100": ("spread", 4, 100),
        "c472_anchor": ("spread", 4, 200),
        "c472_negex_400": ("spread", 4, 400),
        "c472_negp_2": ("spread", 2, 200),
        "c472_negp_8": ("spread", 8, 200),
        "c472_noneg": ("none", 0, 0),
    }
    for slug, (placement, n_p, n_ex) in expect.items():
        got = (spec[slug][2], spec[slug][3], spec[slug][4])
        assert got == (placement, n_p, n_ex), f"{slug}: CELL_SPECS={got}, expected {expect[slug]}"


def _load_traj(cell: str, seed: int) -> dict:
    return json.loads((SLAB / f"{cell}_seed{seed}" / "trajectory.json").read_text())


def _sorted_cks(traj: dict) -> list[dict]:
    return sorted(traj["checkpoints"], key=lambda c: c["frac"])


def _per_probe(ck: dict) -> dict[str, dict]:
    """Persona-level probe stats at one checkpoint (same filters as the placement read)."""
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

    def _bracket(self, target_step: float) -> tuple[int, int, float] | None:
        """(k_lo, k_hi, weight) bracketing target_step, or None if outside the grid.

        A target equal to an exact checkpoint step returns (k, k, 0.0): the
        checkpoint is read directly, with probe validity required only THERE
        (review fix — the interval form interpolated over the adjacent
        interval with w∈{0,1}, dropping probes valid at the exact checkpoint
        but invalid at its neighbour).
        """
        if target_step < self.steps[0] or target_step > self.steps[-1]:
            return None
        for k, s in enumerate(self.steps):
            if target_step == s:
                return k, k, 0.0
        for k in range(len(self.steps) - 1):
            lo, hi = self.steps[k], self.steps[k + 1]
            if lo < target_step < hi:
                return k, k + 1, (target_step - lo) / (hi - lo)
        return None  # pragma: no cover - guarded by the range check

    def interp_probes(self, target_step: float) -> dict[str, float] | None:
        """Per-probe ΔG linearly interpolated at target_step (valid at BOTH brackets)."""
        br = self._bracket(target_step)
        if br is None:
            return None
        k, k1, w = br
        lo_vals = self.valid_probe_dgs(k)
        hi_vals = self.valid_probe_dgs(k1)
        return {p: (1.0 - w) * lo_vals[p] + w * hi_vals[p] for p in lo_vals if p in hi_vals}

    def interp_scalar(self, series: list[float], target_step: float) -> float | None:
        """Linear interpolation of a per-checkpoint scalar series at target_step."""
        br = self._bracket(target_step)
        if br is None:
            return None
        k, k1, w = br
        return (1.0 - w) * series[k] + w * series[k1]

    def nearest_ck_idx(self, target_step: float) -> int:
        return int(np.argmin([abs(s - target_step) for s in self.steps]))


def _seed_avg_interp(cs_by_seed: dict[int, CellSeed], target: float) -> dict[str, float] | None:
    """Seed-averaged per-probe interpolated ΔG (probes valid in BOTH seeds)."""
    per_seed = {}
    for s, cs in cs_by_seed.items():
        vals = cs.interp_probes(target)
        if vals is None:
            return None
        per_seed[s] = vals
    common = set.intersection(*(set(v) for v in per_seed.values()))
    return {p: float(np.mean([per_seed[s][p] for s in per_seed])) for p in sorted(common)}


def _level_comparison(
    level_probes: dict[str, dict[str, float]],
) -> dict:
    """Friedman + pairwise Wilcoxon across count levels on matched per-probe ΔG."""
    levels = list(level_probes)
    matched = sorted(set.intersection(*(set(level_probes[lv]) for lv in levels)))
    cols = {lv: np.array([level_probes[lv][p] for p in matched]) for lv in levels}
    fr = friedmanchisquare(*(cols[lv] for lv in levels))
    kendalls_w = float(fr.statistic / (len(matched) * (len(levels) - 1)))
    pairwise: dict[str, dict] = {}
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            a, b = levels[i], levels[j]
            diff = cols[a] - cols[b]
            wx = wilcoxon(cols[a], cols[b])
            pairwise[f"{a}_vs_{b}"] = {
                "wilcoxon_stat": float(wx.statistic),
                "wilcoxon_p": float(wx.pvalue),
                "mean_diff_nats": float(diff.mean()),
                "median_diff_nats": float(np.median(diff)),
                "cohen_dz": float(diff.mean() / diff.std(ddof=1)),
            }
    means = {lv: float(cols[lv].mean()) for lv in levels}
    return {
        "n_matched_probes": len(matched),
        "level_means": means,
        "between_level_spread_nats": float(max(means.values()) - min(means.values())),
        "friedman_chi2": float(fr.statistic),
        "friedman_p": float(fr.pvalue),
        "kendalls_w": kendalls_w,
        "pairwise": pairwise,
    }


def _first_crossing_step(cs: CellSeed, target_nats: float) -> dict:
    """First step at which source-self ΔG crosses target_nats (linear in step).

    Returns a status dict: reachable (with the crossing step) | starts_above |
    never_reaches — unreachable targets are REPORTED, not skipped.
    """
    ss = cs.source_self
    if ss[0] >= target_nats:
        return {"status": "starts_above_target", "first_checkpoint_value": ss[0]}
    for k in range(len(ss) - 1):
        lo, hi = ss[k], ss[k + 1]
        if (lo < target_nats <= hi) or (lo > target_nats >= hi):
            w = (target_nats - lo) / (hi - lo)
            step = cs.steps[k] + w * (cs.steps[k + 1] - cs.steps[k])
            return {"status": "reachable", "crossing_step": float(step)}
    return {
        "status": "never_reaches_target",
        "observed_band": [float(min(ss)), float(max(ss))],
    }


def _step_targets(cells: dict[str, dict[int, CellSeed]]) -> tuple[list[float], list[int]]:
    """Matched-step targets from ACTUAL checkpoint steps in the count cells' overlap.

    Targets = overlap lower edge (a real checkpoint of the longest-grid cells)
    plus the shortest-grid cell's checkpoint steps inside the overlap.
    """
    firsts = {slug: cells[slug][SEEDS[0]].steps[0] for slug in COUNT_CELLS}
    lasts = {slug: cells[slug][SEEDS[0]].steps[-1] for slug in COUNT_CELLS}
    lo, hi = max(firsts.values()), min(lasts.values())
    shortest = min(COUNT_CELLS, key=lambda slug: lasts[slug])
    targets = sorted(
        {float(lo)} | {float(s) for s in cells[shortest][SEEDS[0]].steps if lo <= s <= hi}
    )
    return targets, [lo, hi]


def _build_trajectories(
    cells: dict[str, dict[int, CellSeed]], rng: np.random.Generator
) -> dict[str, dict]:
    """1. Step-space trajectories per cell (figure-ready, per-seed + seed-pooled)."""
    trajectories: dict[str, dict] = {}
    for slug in ALL_CELLS:
        per_seed = {}
        pooled = []
        for s in SEEDS:
            cs = cells[slug][s]
            rows = []
            for k in range(len(cs.steps)):
                vals = cs.valid_probe_dgs(k)
                rows.append(
                    {
                        "step": cs.steps[k],
                        "frac": cs.fracs[k],
                        "mean_bystander_delta_g": float(np.mean(list(vals.values()))),
                        "n_valid_probes": len(vals),
                        **cs.drop_counts(k),
                        "source_self_delta_g": cs.source_self[k],
                    }
                )
            per_seed[str(s)] = rows
        for k in range(len(cells[slug][SEEDS[0]].steps)):
            sa = _seed_avg_interp(cells[slug], float(cells[slug][SEEDS[0]].steps[k]))
            arr = np.array(list(sa.values()))
            pooled.append(
                {
                    "step": cells[slug][SEEDS[0]].steps[k],
                    "mean_bystander_delta_g": float(arr.mean()),
                    "boot_ci95": _boot_ci(arr, rng),
                    "n_probes": len(arr),
                    "source_self_delta_g_mean": float(
                        np.mean([cells[slug][s].source_self[k] for s in SEEDS])
                    ),
                }
            )
        trajectories[slug] = {"per_seed": per_seed, "seed_pooled": pooled}
    return trajectories


def _matched_step_block(
    cells: dict[str, dict[int, CellSeed]],
    targets: list[float],
    rng: np.random.Generator,
) -> tuple[dict, dict, dict]:
    """2. Matched-step comparisons per axis + interpolation-error diagnostics + noneg ref."""
    matched_step: dict[str, dict] = {}
    per_probe_err: dict[str, dict] = {}
    per_axis_deltas: dict[str, list[float]] = {axis: [] for axis in AXES}
    for axis, levels in AXES.items():
        per_target: dict[str, dict] = {}
        family: dict[str, float] = {}
        err_per_target: dict[str, dict] = {}
        for t in targets:
            level_probes = {}
            for lv, slug in levels.items():
                sa = _seed_avg_interp(cells[slug], t)
                assert sa is not None, f"target {t} outside grid for {slug}"
                level_probes[lv] = sa
            cmp_ = _level_comparison(level_probes)
            cmp_["level_boot_ci95"] = {
                lv: _boot_ci(np.array(list(level_probes[lv].values())), rng) for lv in levels
            }
            per_target[f"step_{t:g}"] = cmp_
            family[f"step_{t:g}"] = cmp_["friedman_p"]
            # Per-probe interpolation error on the EXACT matched probe set of
            # THIS comparison (review fix: the cell-mean version cancels
            # per-probe errors; the resolution floor must be per-probe).
            matched = sorted(set.intersection(*(set(level_probes[lv]) for lv in levels)))
            err_levels: dict[str, dict] = {}
            for lv, slug in levels.items():
                err_seeds: dict[str, dict] = {}
                for s in SEEDS:
                    cs = cells[slug][s]
                    interp_vals = cs.interp_probes(t)
                    near_vals = cs.valid_probe_dgs(cs.nearest_ck_idx(t))
                    missing = [p for p in matched if p not in interp_vals or p not in near_vals]
                    assert not missing, f"{slug} seed {s} step {t:g}: matched ∖ avail: {missing}"
                    deltas = [abs(interp_vals[p] - near_vals[p]) for p in matched]
                    per_axis_deltas[axis].extend(deltas)
                    err_seeds[str(s)] = {
                        "n_probes": len(deltas),
                        "max_abs_nats": float(max(deltas)),
                        "mean_abs_nats": float(np.mean(deltas)),
                    }
                err_levels[lv] = err_seeds
            err_per_target[f"step_{t:g}"] = err_levels
        matched_step[axis] = {
            "per_target": per_target,
            "holm_across_targets": holm_correction(family),
        }
        per_probe_err[axis] = err_per_target

    all_deltas = [d for ds in per_axis_deltas.values() for d in ds]
    per_axis_summary = {
        axis: {
            "n_deltas": len(ds),
            "max_abs_nats": float(max(ds)),
            "mean_abs_nats": float(np.mean(ds)),
        }
        for axis, ds in per_axis_deltas.items()
    }
    # Resolution-floor check: the matched-step verdict stands only if the
    # between-level spread exceeds the per-probe interpolation-error floor.
    floor_check: dict[str, dict] = {}
    for axis in AXES:
        spreads = [
            v["between_level_spread_nats"] for v in matched_step[axis]["per_target"].values()
        ]
        floor = per_axis_summary[axis]["max_abs_nats"]
        floor_check[axis] = {
            "min_between_level_spread_nats": float(min(spreads)),
            "max_per_probe_interp_error_nats": floor,
            "spread_over_floor_ratio": float(min(spreads) / floor) if floor > 0 else None,
            "clears_floor": bool(min(spreads) > floor),
        }

    # Cell-mean drift DIAGNOSTIC (renamed; per-probe errors cancel in the
    # mean, so this is NOT an error bound — kept for continuity with v5).
    cell_mean_diag: dict[str, dict] = {}
    for slug in COUNT_CELLS:
        per_t = {}
        for t in targets:
            per_seed_err = {}
            for s in SEEDS:
                cs = cells[slug][s]
                interp_vals = cs.interp_probes(t)
                near_vals = cs.valid_probe_dgs(cs.nearest_ck_idx(t))
                err = abs(
                    float(np.mean(list(interp_vals.values())))
                    - float(np.mean(list(near_vals.values())))
                )
                per_seed_err[str(s)] = err
            per_t[f"step_{t:g}"] = per_seed_err
        cell_mean_diag[slug] = per_t
    max_cell_mean = max(
        err for per_t in cell_mean_diag.values() for ps in per_t.values() for err in ps.values()
    )
    interp_diag = {
        "per_probe_on_matched_sets": {
            "definition": (
                "abs(per-probe interpolated ΔG − per-probe nearest-checkpoint ΔG), per "
                "seed, on the EXACT matched probe set entering each axis × target "
                "comparison; max_abs_nats is the resolution floor for the matched-step read"
            ),
            "per_axis_per_target_per_level_per_seed": per_probe_err,
            "per_axis_summary": per_axis_summary,
            "max_abs_nats": float(max(all_deltas)),
            "mean_abs_nats": float(np.mean(all_deltas)),
        },
        "cell_mean_diagnostic": {
            "definition": (
                "abs(interpolated cell-mean − nearest-checkpoint cell-mean) per cell × "
                "seed × target; per-probe errors cancel in the mean — a cell-level drift "
                "diagnostic, NOT an interpolation-error bound"
            ),
            "per_cell_per_target": cell_mean_diag,
            "max": float(max_cell_mean),
        },
        "resolution_floor_check": floor_check,
    }

    # noneg reference at targets inside its grid (no test — not count-matched).
    noneg_ref: dict[str, dict] = {}
    for t in targets:
        sa = _seed_avg_interp(cells[NONEG], t)
        if sa is None:
            noneg_ref[f"step_{t:g}"] = {"status": "outside_noneg_grid"}
        else:
            arr = np.array(list(sa.values()))
            noneg_ref[f"step_{t:g}"] = {
                "mean_bystander_delta_g": float(arr.mean()),
                "boot_ci95": _boot_ci(arr, rng),
                "n_probes": len(arr),
            }
    return matched_step, interp_diag, noneg_ref


def _matched_implant_block(
    cells: dict[str, dict[int, CellSeed]],
) -> tuple[dict, dict, dict]:
    """3. Matched-implant first-crossing comparison + bands + common ranges."""
    bands = {
        slug: {
            str(s): [float(min(cells[slug][s].source_self)), float(max(cells[slug][s].source_self))]
            for s in SEEDS
        }
        for slug in ALL_CELLS
    }
    matched_implant: dict[str, dict] = {}
    for axis, levels in AXES.items():
        per_target: dict[str, dict] = {}
        for tgt in IMPLANT_TARGETS_NATS:
            per_level: dict[str, dict] = {}
            statuses = []
            for lv, slug in levels.items():
                per_seed: dict[str, dict] = {}
                for s in SEEDS:
                    cs = cells[slug][s]
                    cross = _first_crossing_step(cs, tgt)
                    if cross["status"] == "reachable":
                        step = cross["crossing_step"]
                        vals = cs.interp_probes(step)
                        cross["bystander_delta_g_at_crossing"] = float(np.mean(list(vals.values())))
                    per_seed[str(s)] = cross
                    statuses.append(cross["status"])
                per_level[lv] = per_seed
            identifiable = all(st == "reachable" for st in statuses)
            per_target[f"implant_{tgt:g}_nats"] = {
                "per_level": per_level,
                "identifiable": identifiable,
                "verdict": (
                    "comparable at matched implant"
                    if identifiable
                    else "NOT identifiable: >=1 level never passes through the target "
                    "within the observed checkpoint window (implant is set before the "
                    "first checkpoint and stays flat in a level-specific band)"
                ),
            }
        matched_implant[axis] = per_target
    # Common source-self range across count cells (pooled over seeds, per axis).
    common_range = {}
    for axis, levels in AXES.items():
        los, his = [], []
        for slug in levels.values():
            ss_all = [v for s in SEEDS for v in cells[slug][s].source_self]
            los.append(min(ss_all))
            his.append(max(ss_all))
        lo, hi = max(los), min(his)
        common_range[axis] = {
            "lo": float(lo),
            "hi": float(hi),
            "empty": bool(lo > hi),
        }
    return bands, matched_implant, common_range


def _print_summary(
    targets: list[float],
    matched_step: dict,
    interp_diag: dict,
    bands: dict,
    matched_implant: dict,
    common_range: dict,
) -> None:
    """Console summary of the matched-step + matched-implant reads."""
    for axis, levels in AXES.items():
        print(f"── {axis} matched-step (levels {list(levels)}) ──")
        for t in targets:
            c = matched_step[axis]["per_target"][f"step_{t:g}"]
            means = ", ".join(f"{lv}:{c['level_means'][lv]:+.2f}" for lv in levels)
            print(
                f"  step {t:g}: {means}  spread={c['between_level_spread_nats']:.2f} nats "
                f"Friedman p={c['friedman_p']:.2e} W={c['kendalls_w']:.3f} "
                f"(n={c['n_matched_probes']})"
            )
        rejected = [
            k for k, v in matched_step[axis]["holm_across_targets"].items() if v["reject_null"]
        ]
        print(f"  Holm rejects: {len(rejected)}/{len(targets)} targets\n")
    pp = interp_diag["per_probe_on_matched_sets"]
    print(
        "Per-probe interp-error resolution floor (matched sets): "
        f"max={pp['max_abs_nats']:.4f} mean={pp['mean_abs_nats']:.4f} nats; per axis: "
        + "; ".join(
            f"{ax}: max={s['max_abs_nats']:.4f} mean={s['mean_abs_nats']:.4f} (n={s['n_deltas']})"
            for ax, s in pp["per_axis_summary"].items()
        )
    )
    for ax, fc in interp_diag["resolution_floor_check"].items():
        ratio = fc["spread_over_floor_ratio"]
        print(
            f"  floor check {ax}: min spread {fc['min_between_level_spread_nats']:.3f} vs "
            f"max per-probe err {fc['max_per_probe_interp_error_nats']:.4f} nats "
            f"(ratio {'inf' if ratio is None else f'{ratio:.0f}'}×) "
            f"clears_floor={fc['clears_floor']}"
        )
    print(
        "Cell-mean drift diagnostic (NOT a bound; per-probe errors cancel): "
        f"max={interp_diag['cell_mean_diagnostic']['max']:.4f} nats"
    )
    print(
        "Source-self bands (pooled seeds): "
        + "; ".join(
            f"{slug}=[{min(b[str(s)][0] for s in SEEDS):.1f},"
            f"{max(b[str(s)][1] for s in SEEDS):.1f}]"
            for slug, b in bands.items()
        )
    )
    for axis in AXES:
        cr = common_range[axis]
        print(
            f"matched-implant {axis}: common source-self range "
            f"[{cr['lo']:.2f}, {cr['hi']:.2f}] empty={cr['empty']}; "
            + "; ".join(
                f"{k}: identifiable={v['identifiable']}" for k, v in matched_implant[axis].items()
            )
        )


def main() -> None:
    _assert_cell_mapping()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOT_SEED)

    cts = load_cos_to_source(HEADLINE_LAYER, SOURCE, CENTROID_DIR)
    panel = held_out_panel(cts, source=SOURCE)
    cells: dict[str, dict[int, CellSeed]] = {
        slug: {s: CellSeed(slug, s, panel) for s in SEEDS} for slug in ALL_CELLS
    }
    for slug in ALL_CELLS:  # per-cell grids must agree across seeds
        assert cells[slug][SEEDS[0]].steps == cells[slug][SEEDS[1]].steps, slug

    targets, overlap = _step_targets(cells)
    print(f"Held-out panel (L{HEADLINE_LAYER}): {len(panel)} probes")
    print("Step grids: " + "; ".join(f"{s}={cells[s][SEEDS[0]].steps}" for s in ALL_CELLS))
    print(f"Count-cell overlap: {overlap}; matched-step targets: {targets}\n")

    trajectories = _build_trajectories(cells, rng)
    matched_step, interp_diag, noneg_ref = _matched_step_block(cells, targets, rng)
    bands, matched_implant, common_range = _matched_implant_block(cells)
    _print_summary(targets, matched_step, interp_diag, bands, matched_implant, common_range)

    out = {
        "schema": "i472_reanalysis_count_matched_step",
        "followup_label": "placement-null-full-trajectory",
        "dv": "on-policy log P(marker) at post-response slot, trained - base (delta_g), nats",
        "axes": AXES,
        "noneg_reference_cell": NONEG,
        "seeds": list(SEEDS),
        "source": SOURCE,
        "n_held_out_panel": len(panel),
        "validity_filters": {
            "drop_r_collapsed": True,
            "drop_saturated": True,
            "saturation_rule": f"mean g_logp > -{SUBCEILING_HEADROOM_NATS} nats",
            "interpolation_validity": (
                "probe valid at BOTH bracketing checkpoints, both seeds; a target equal "
                "to an exact checkpoint step reads that checkpoint directly (validity "
                "required only there)"
            ),
        },
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "ci": "95% percentile, over probes"},
        "step_grids": {slug: cells[slug][SEEDS[0]].steps for slug in ALL_CELLS},
        "count_cell_step_overlap": overlap,
        "matched_step_targets": targets,
        "step_space_trajectories": trajectories,
        "matched_step_comparisons": matched_step,
        "interpolation_error_nats": interp_diag,
        "noneg_reference_at_targets": noneg_ref,
        "matched_implant": {
            "targets_nats": list(IMPLANT_TARGETS_NATS),
            "source_self_bands_per_cell_per_seed": bands,
            "common_source_self_range_per_axis": common_range,
            "comparisons": matched_implant,
        },
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
    out_path = OUT_DIR / "reanalysis_count_matched_step.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
