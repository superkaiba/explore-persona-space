"""Fold the compose-multiseed halves into the §3 lattice verdicts (#1739 leg 1).

Consumes the 6 cms boxes' durable per-cell sidecars (per behavior x half:
``<results-root>/<behavior>/<half>/arm_results/percell/{cells.jsonl,
compose_skips.jsonl, compose_pool_meta.jsonl}`` + the merged
``map_diagnostics.json``) and emits:

- ``compose_ms_table.json`` — per-cell-class table (seed mean +- t-CI, per-seed
  bootstrap CIs, per-arm rho columns, identity+bias map companions), the flip
  contrast C per behavior with the plan-§3 lattice verdict, the round verdict,
  the gate-3 designed-skip audit, the banked-anchor reproduction read, the
  ablation dose curve, group-grain pool<->cell overlap + the overlap
  sensitivity recompute, and realized pairwise pool overlap per f_u level;
- ``compose_ms_cells.csv`` — the flat per-cell-class table;
- figures: ``compose_fu_flip_v2`` (hero), ``compose_dose_curve``,
  ``compose_ms_spaghetti``.

Verdict grammar (plan §3, EXACT): per behavior, C(s) = mean over the 4 anchor
cells (2 variants x L in {250, 2500}) of [Delta(f_u=0.5, f_l=0) -
Delta(f_u=0, f_l=0)] at seed s; FLIP-CONFIRMED iff C's 5-seed t-CI excludes 0
from above AND the seed-mean Delta(0.5, 0) > 0 on >= 2 of the 4 anchors;
FLIP-FALSIFIED iff the t-CI excludes 0 from below; else INDETERMINATE. Round:
>= 2 CONFIRMED -> CONFIRMED; >= 2 FALSIFIED -> FALSIFIED; else MIXED (PENDING
while any behavior lacks a lattice verdict).

Designed exit codes: 0 ok; 3 missing input; 5 audit halt (gate-3 skip-audit
set-inequality, pair-coverage shortfall, or map-diagnostics presence failure);
10 banked-anchor divergence (unless ``--allow-banked-divergence``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue1739_compose_ms_fold.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

HEADLINE_PAIR = ("arm6_map_proj_e1", "arm2_ctx_native")
ANCHOR_BUDGETS = (250, 2500)  # §3 flip-contrast anchors (per variant)
VARIANTS = ("context_end", "prefix_end")
CORE_LEVELS = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0))
ABLATION_F_U = (0.1, 0.25, 0.75, 1.0)
ABLATION_BUDGET = 2500
# Gate-3 designed-skip arithmetic (plan §4/§7): evil n_contexts = 6,468, so at
# f_l=0 the eliciting complement of an L-anchor cell has n - L rows:
#   (f_u=0.5, L=8000): quota 2500 with ~0 available  -> designed skip
#   (f_u=1.0, L=2500): quota 5000 with ~3,968        -> designed skip
#   (f_u=0.75, L=2500): quota 3750 <= ~3,968         -> feasible-but-thin
#     (CONDITIONALLY-designed class: recorded + excluded from trend when a
#     realized draw cannot fill it; NEVER a halt)
DESIGNED_SKIPS = {"evil": ((0.5, 0.0, 8000), (1.0, 0.0, 2500))}
CONDITIONAL_SKIPS = {"evil": ((0.75, 0.0, 2500),)}
BANKED_TOL_DEFAULT = 5e-3


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def cell_id(row: dict) -> tuple:
    """(variant, f_u, f_l, budget_l, draw, seed) identity of one cells.jsonl row."""
    arms = row.get("arms") or []
    if not arms:
        raise ValueError(f"cells row without arm rows (unit_key={row.get('unit_key')!r})")
    a0 = arms[0]
    return (
        a0["variant"],
        float(a0["f_u"]) if a0.get("f_u") is not None else None,
        float(a0["f_l"]) if a0.get("f_l") is not None else None,
        int(a0["budget_l"]),
        int(a0["draw"]),
        int(a0["seed"]),
    )


def assert_unique_cells(rows: list[dict]) -> dict[tuple, dict]:
    """Per-(cell, seed) uniqueness FIRST (plan §5): a duplicated cell across the
    half union means two boxes ran the same replicate — fail loud."""
    by_key: dict[tuple, dict] = {}
    for row in rows:
        key = cell_id(row)
        if key in by_key:
            raise ValueError(
                f"duplicate compose cell across the half union: {key} "
                f"(sources: {by_key[key].get('_source')}, {row.get('_source')})"
            )
        by_key[key] = row
    return by_key


def headline_delta(row: dict) -> float | None:
    head = row.get("headline") or {}
    pair = tuple(head.get("pair") or ())
    if pair and pair != HEADLINE_PAIR:
        raise ValueError(f"unexpected headline pair {pair} (want {HEADLINE_PAIR})")
    d = head.get("delta_rho_frozen")
    return float(d) if d is not None else None


def t_ci(values: list[float], level: float = 0.95) -> dict:
    """Mean +- t-CI over per-seed values (df = n-1; the plan-§6 seed-grain CI)."""
    import numpy as np
    from scipy import stats

    v = np.asarray([float(x) for x in values], dtype=float)
    n = len(v)
    out = {"n": n, "mean": float(v.mean()) if n else None, "lo": None, "hi": None, "sd": None}
    if n >= 2:
        sd = float(v.std(ddof=1))
        half = float(stats.t.ppf(0.5 + level / 2.0, n - 1)) * sd / math.sqrt(n)
        out.update({"sd": sd, "lo": out["mean"] - half, "hi": out["mean"] + half})
    return out


def flip_contrast(
    by_key: dict[tuple, dict],
    *,
    seeds: tuple[int, ...],
    variants: tuple[str, ...] = VARIANTS,
    anchors: tuple[int, ...] = ANCHOR_BUDGETS,
    draw: int = 0,
) -> dict:
    """Per-behavior flip contrast C(s) + pair coverage (plan §3).

    A 'pair' is one (variant, L, seed) anchor cell with BOTH Delta(0.5, 0) and
    Delta(0, 0) present — 2 variants x 2 anchors x len(seeds) = 20 per
    behavior at full coverage.
    """
    per_seed_terms: dict[int, list[float]] = {s: [] for s in seeds}
    per_anchor_delta05: dict[str, list[float]] = {}
    pairs = 0
    missing: list[str] = []
    for variant in variants:
        for anchor in anchors:
            for s in seeds:
                k05 = (variant, 0.5, 0.0, int(anchor), draw, int(s))
                k00 = (variant, 0.0, 0.0, int(anchor), draw, int(s))
                d05 = headline_delta(by_key[k05]) if k05 in by_key else None
                d00 = headline_delta(by_key[k00]) if k00 in by_key else None
                if d05 is None or d00 is None:
                    missing.append(f"{variant}/L{anchor}/seed{s}")
                    continue
                pairs += 1
                per_seed_terms[s].append(d05 - d00)
                per_anchor_delta05.setdefault(f"{variant}|L{anchor}", []).append(d05)
    per_seed_c = {
        int(s): (sum(t) / len(t) if t else None) for s, t in sorted(per_seed_terms.items())
    }
    anchor_means = {
        k: (sum(v) / len(v) if v else None) for k, v in sorted(per_anchor_delta05.items())
    }
    return {
        "per_seed_C": per_seed_c,
        "pairs": pairs,
        "pairs_expected": len(variants) * len(anchors) * len(seeds),
        "missing_pairs": missing,
        "anchor_seedmean_delta05": anchor_means,
    }


def lattice_verdict(c_tci: dict, anchor_seedmean_delta05: dict) -> str:
    """Plan-§3 lattice grammar, EXACT (FLIP-FALSIFIED only on a t-CI excluding
    0 from below)."""
    lo, hi = c_tci.get("lo"), c_tci.get("hi")
    if lo is None or hi is None:
        return "INDETERMINATE"
    positive_anchors = sum(1 for v in anchor_seedmean_delta05.values() if v is not None and v > 0)
    if lo > 0 and positive_anchors >= 2:
        return "FLIP-CONFIRMED"
    if hi < 0:
        return "FLIP-FALSIFIED"
    return "INDETERMINATE"


def round_verdict(verdicts: list[str]) -> str:
    if any(v not in ("FLIP-CONFIRMED", "FLIP-FALSIFIED", "INDETERMINATE") for v in verdicts):
        return "PENDING"
    if sum(v == "FLIP-CONFIRMED" for v in verdicts) >= 2:
        return "CONFIRMED"
    if sum(v == "FLIP-FALSIFIED" for v in verdicts) >= 2:
        return "FALSIFIED"
    return "MIXED"


def skip_key(row: dict) -> tuple:
    return (
        row["variant"],
        float(row["f_u"]),
        float(row["f_l"]),
        int(row["budget_l"]),
        int(row["draw"]),
        int(row["seed"]),
    )


def designed_skip_audit(
    skip_rows: list[dict],
    behavior: str,
    *,
    seeds: tuple[int, ...],
    variants: tuple[str, ...] = VARIANTS,
    draw: int = 0,
) -> dict:
    """Gate-3 set-equality audit (plan §7): recorded skips vs the designed set.

    The CONDITIONALLY-designed class (evil f_u=0.75) is recorded + reported +
    excluded from the trend when skipped — never a halt and never counted as
    'extra'. Any other extra skip, or a missing designed skip, HALTS the fold
    for this behavior.
    """
    designed = {
        (v, fu, fl, int(ll), draw, int(s))
        for (fu, fl, ll) in DESIGNED_SKIPS.get(behavior, ())
        for v in variants
        for s in seeds
    }
    conditional = {
        (v, fu, fl, int(ll), draw, int(s))
        for (fu, fl, ll) in CONDITIONAL_SKIPS.get(behavior, ())
        for v in variants
        for s in seeds
    }
    recorded = {skip_key(r) for r in skip_rows}
    extra = sorted(recorded - designed - conditional)
    missing = sorted(designed - recorded)
    conditional_recorded = sorted(recorded & conditional)
    return {
        "n_recorded": len(recorded),
        "n_designed": len(designed),
        "extra_skips": [list(k) for k in extra],
        "missing_designed_skips": [list(k) for k in missing],
        "conditional_skips_recorded": [list(k) for k in conditional_recorded],
        "ok": not extra and not missing,
    }


def banked_repro(
    by_key: dict[tuple, dict], banked_rows: list[dict], *, tol: float = BANKED_TOL_DEFAULT
) -> dict:
    """Seed-0/draw-0 reproduction read vs the banked round-1 cells (plan §5).

    Joins on (variant, f_u, f_l, budget_l, draw, seed) VALUES over the
    intersection (the banked run's roster/labels differ; the arm values are
    deterministic per pool/whitening/cell, and spec-seed threading keeps the
    seed-0 whitening byte-reproducing the banked ``args.seeds[0] == 0`` path).
    """
    banked: dict[tuple, float] = {}
    for row in banked_rows:
        arms = row.get("arms") or []
        if not arms or arms[0].get("f_u") is None:
            continue
        key = cell_id(row)
        if key[4] == 0 and key[5] == 0:  # draw 0, seed 0
            d = headline_delta(row)
            if d is not None:
                banked[key] = d
    compared, material = [], []
    for key, banked_d in sorted(banked.items()):
        row = by_key.get(key)
        if row is None:
            continue
        fresh_d = headline_delta(row)
        if fresh_d is None:
            continue
        diff = abs(fresh_d - banked_d)
        rec = {"cell": list(key), "fresh": fresh_d, "banked": banked_d, "abs_diff": diff}
        compared.append(rec)
        if diff > tol:
            material.append(rec)
    return {
        "n_banked_cells": len(banked),
        "n_compared": len(compared),
        "tol": tol,
        "max_abs_diff": max((r["abs_diff"] for r in compared), default=None),
        "material_divergences": material,
        "ok": not material and bool(compared),
        "compared": compared,
    }


def dose_curve(
    by_key: dict[tuple, dict],
    *,
    seeds: tuple[int, ...],
    variants: tuple[str, ...] = VARIANTS,
    budget: int = ABLATION_BUDGET,
    draw: int = 0,
) -> dict:
    """Seed-grain dose curve: Delta vs f_u at (f_l=0, L=budget), variants pooled
    per seed (mean over the 2 variants), levels = {0} + ablation + {0.5}."""
    levels = sorted({0.0, 0.5, *ABLATION_F_U})
    out: dict[str, dict] = {}
    for f_u in levels:
        per_seed: list[float] = []
        for s in seeds:
            terms = [
                headline_delta(by_key[k])
                for v in variants
                if (k := (v, float(f_u), 0.0, int(budget), draw, int(s))) in by_key
                and headline_delta(by_key[k]) is not None
            ]
            if terms:
                per_seed.append(sum(terms) / len(terms))
        out[f"fu{f_u}"] = {
            "f_u": f_u,
            "per_seed_mean_delta": per_seed,
            **t_ci(per_seed),
        }
    return out


def pairwise_pool_overlap(pool_meta_rows: list[dict]) -> dict:
    """Realized pairwise eliciting-pool overlap (Jaccard) between seeds at the
    same (variant, f_u, f_l, L) level (plan §5 read (iii))."""
    by_level: dict[tuple, dict[int, set[str]]] = {}
    for row in pool_meta_rows:
        level = (row["variant"], float(row["f_u"]), float(row["f_l"]), int(row["budget_l"]))
        by_level.setdefault(level, {})[int(row["seed"])] = set(row.get("elic_ctx_ids") or ())
    out: dict[str, dict] = {}
    for level, seed_sets in sorted(by_level.items()):
        seeds_sorted = sorted(seed_sets)
        jacs = []
        for i, a in enumerate(seeds_sorted):
            for b in seeds_sorted[i + 1 :]:
                sa, sb = seed_sets[a], seed_sets[b]
                union = len(sa | sb)
                jacs.append(len(sa & sb) / union if union else 0.0)
        key = f"{level[0]}|fu{level[1]}_fl{level[2]}_L{level[3]}"
        out[key] = {
            "n_seeds": len(seeds_sorted),
            "n_pairs": len(jacs),
            "jaccard_mean": (sum(jacs) / len(jacs)) if jacs else None,
            "jaccard_max": max(jacs) if jacs else None,
        }
    return out


def load_groups_map(labeling_json: Path) -> dict[str, str]:
    """context_id -> group_key from the dv_dataset payload (train split rows
    included regardless of split — group identity is split-independent)."""
    payload = json.loads(labeling_json.read_text())
    return {
        str(r["context_id"]): str(r.get("group_key"))
        for r in payload["rows"]
        if r.get("group_key") is not None
    }


def sensitivity_recompute(
    by_key: dict[tuple, dict],
    pool_meta_rows: list[dict],
    groups_by_ctx: dict[str, str],
) -> dict:
    """Overlap-sensitivity recompute (plan §5 read (ii)): re-derive the
    headline Delta per f_l=0 compose cell EXCLUDING anchor-cell rows whose
    group also feeds the eliciting pool (group-grain contamination). Cells
    whose preds npz is not staged locally are recorded ``pending`` — the
    caller downgrades a FLIP-CONFIRMED verdict while any cell is pending
    (fail-safe, never silently confirmed).
    """
    import numpy as np
    from scipy import stats

    meta_by_cell = {skip_key(r): r for r in pool_meta_rows}
    rows, pending = [], []
    for key, meta in sorted(meta_by_cell.items()):
        if not meta.get("overlap_group_count"):
            continue
        cell_row = by_key.get(key)
        if cell_row is None:
            continue
        name = cell_row.get("preds_npz")
        source = cell_row.get("_source_root")
        npz_path = (
            Path(source) / "arm_results" / "percell" / "preds" / str(name)
            if name and source
            else None
        )
        if npz_path is None or not npz_path.exists():
            pending.append({"cell": list(key), "preds_npz": str(npz_path)})
            continue
        with np.load(npz_path, allow_pickle=False) as z:
            ctx_ids = [str(c) for c in z["context_ids"]]
            dv = np.asarray(z["dv"], dtype=float)
            preds = {a: np.asarray(z[f"pred__{a}"], dtype=float) for a in HEADLINE_PAIR}
        overlap = set(meta.get("overlap_groups") or ())
        kept = np.asarray(
            [i for i, cid in enumerate(ctx_ids) if groups_by_ctx.get(cid) not in overlap],
            dtype=int,
        )
        if len(kept) < 3:
            pending.append({"cell": list(key), "reason": f"only {len(kept)} rows survive"})
            continue
        d_excl = float(
            stats.spearmanr(preds[HEADLINE_PAIR[0]][kept], dv[kept]).statistic
            - stats.spearmanr(preds[HEADLINE_PAIR[1]][kept], dv[kept]).statistic
        )
        d_full = headline_delta(cell_row)
        rows.append(
            {
                "cell": list(key),
                "n_rows": len(ctx_ids),
                "n_kept": int(len(kept)),
                "overlap_group_count": int(meta["overlap_group_count"]),
                "delta_full": d_full,
                "delta_excl_overlap": d_excl,
                "shift": (abs(d_full - d_excl) if d_full is not None else None),
            }
        )
    shifts = [r["shift"] for r in rows if r["shift"] is not None]
    return {
        "n_recomputed": len(rows),
        "n_pending": len(pending),
        "pending": pending,
        "max_shift": max(shifts) if shifts else None,
        "rows": rows,
        "status": "pending" if pending else ("ok" if rows else "no-overlap-cells"),
    }


def map_diag_presence(
    diag_union: dict,
    *,
    variants: tuple[str, ...] = VARIANTS,
    levels: tuple[tuple[float, float], ...],
    seed: int = 0,
) -> dict:
    """Presence assert (plan §5): a seed-``seed`` map-diagnostics entry per
    (variant, f_u/f_l level) — keys are ``<variant>|<label>|draw0_seed<N>``."""
    missing = []
    for variant in variants:
        for f_u, f_l in levels:
            hits = [
                k
                for k in diag_union
                if k.startswith(f"{variant}|compose")
                and f"_fu{f_u}_" in k
                and f"_fl{f_l}_" in k
                and k.endswith(f"seed{seed}")
            ]
            if not hits:
                missing.append(f"{variant}|fu{f_u}_fl{f_l}|seed{seed}")
    return {"ok": not missing, "missing": missing}


def diag_companions(diag_union: dict, variant: str, label: str, seed: int = 0) -> dict:
    """Identity+bias companion columns for one cell class (median over layers
    of the seed-0 map diagnostics' r2_map / r2_identity_bias / knn top-1)."""
    import numpy as np

    entry = diag_union.get(f"{variant}|{label}|draw0_seed{seed}")
    if entry is None:
        entry = diag_union.get(f"{variant}|{label}")  # legacy single-seed key shape
    if not entry or not entry.get("per_layer"):
        return {"r2_map_median": None, "r2_identity_bias_median": None}
    per_layer = entry["per_layer"]
    r2m = [pl.get("r2_map") for pl in per_layer if pl.get("r2_map") is not None]
    r2i = [pl.get("r2_identity_bias") for pl in per_layer if pl.get("r2_identity_bias") is not None]
    return {
        "r2_map_median": float(np.median(r2m)) if r2m else None,
        "r2_identity_bias_median": float(np.median(r2i)) if r2i else None,
    }


def arm_rho(row: dict, slug: str) -> float | None:
    for a in row.get("arms") or ():
        if a.get("arm") == slug:
            r = a.get("rho_frozen")
            return float(r) if r is not None else None
    return None


def cell_class_table(
    by_key: dict[tuple, dict], diag_union: dict, *, seeds: tuple[int, ...]
) -> list[dict]:
    """Per-(variant, f_u, f_l, L) table: seed-mean +- t-CI of the headline
    Delta, each seed's own bootstrap CI, per-arm frozen-rho seed means, the
    arm13 shuffled-map margin, and the identity+bias map companions."""
    classes: dict[tuple, list[tuple[int, dict]]] = {}
    for key, row in by_key.items():
        variant, f_u, f_l, budget_l, draw, seed = key
        classes.setdefault((variant, f_u, f_l, budget_l, draw), []).append((seed, row))
    out = []
    for cls, members in sorted(classes.items(), key=lambda kv: tuple(map(str, kv[0]))):
        variant, f_u, f_l, budget_l, draw = cls
        members.sort()
        deltas = [headline_delta(r) for _, r in members if headline_delta(r) is not None]
        ci = t_ci(deltas)
        seed0 = next((r for s, r in members if s == 0), None)
        label = (seed0 or members[0][1])["arms"][0].get("u_rung_label", "")
        arm_means = {}
        for slug in (
            "arm2_ctx_native",
            "arm4_ridge_ctx",
            "arm6_map_proj_e1",
            "arm7_map_ridge_pred",
            "arm13_shuffled_map",
        ):
            vals = [arm_rho(r, slug) for _, r in members]
            vals = [v for v in vals if v is not None]
            arm_means[f"{slug}__rho_mean"] = (sum(vals) / len(vals)) if vals else None
        a6, a13 = arm_means["arm6_map_proj_e1__rho_mean"], arm_means["arm13_shuffled_map__rho_mean"]
        out.append(
            {
                "variant": variant,
                "f_u": f_u,
                "f_l": f_l,
                "budget_l": budget_l,
                "draw": draw,
                "u_rung_label": label,
                "seeds": [s for s, _ in members],
                "n_seeds": len(members),
                "delta_per_seed": deltas,
                "delta_mean": ci["mean"],
                "delta_tci_lo": ci["lo"],
                "delta_tci_hi": ci["hi"],
                "per_seed_boot_ci": [
                    (r.get("headline") or {}).get("ci_delta_frozen") for _, r in members
                ],
                **arm_means,
                "arm6_minus_arm13_margin": (a6 - a13)
                if a6 is not None and a13 is not None
                else None,
                **diag_companions(diag_union, variant, label),
            }
        )
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def load_behavior(results_root: Path, behavior: str, halves: tuple[str, ...]) -> dict:
    cells: list[dict] = []
    skips: list[dict] = []
    pool_meta: list[dict] = []
    diag_union: dict = {}
    missing_halves = []
    for half in halves:
        root = results_root / behavior / half
        cells_path = root / "arm_results" / "percell" / "cells.jsonl"
        if not cells_path.exists():
            missing_halves.append(half)
            continue
        for row in _read_jsonl(cells_path):
            arms = row.get("arms") or []
            if not arms or arms[0].get("f_u") is None:
                continue  # defensive: plain rungs never expected in a cms out-root
            row["_source"] = str(cells_path)
            row["_source_root"] = str(root)
            cells.append(row)
        skips += _read_jsonl(root / "arm_results" / "percell" / "compose_skips.jsonl")
        pool_meta += _read_jsonl(root / "arm_results" / "percell" / "compose_pool_meta.jsonl")
        diag_path = root / "map_diagnostics.json"
        if diag_path.exists():
            diag_union.update(json.loads(diag_path.read_text()))
    # skip rows dedup across halves (same key can never legitimately recur —
    # halves own disjoint seeds — but a re-run's append-dedup is per file).
    seen: set[tuple] = set()
    skips_dedup = []
    for r in skips:
        k = skip_key(r)
        if k not in seen:
            seen.add(k)
            skips_dedup.append(r)
    return {
        "cells": cells,
        "skips": skips_dedup,
        "pool_meta": pool_meta,
        "diag": diag_union,
        "missing_halves": missing_halves,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _clamped_yerr(mean: float, lo: float | None, hi: float | None) -> tuple[float, float]:
    """mpl yerr magnitudes, element-wise clamped non-negative (gotchas.md
    errorbar rule: never feed a negative offset)."""
    if lo is None or hi is None:
        return (0.0, 0.0)
    return (max(0.0, mean - lo), max(0.0, hi - mean))


def render_figures(report: dict, fig_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    fig_dir.mkdir(parents=True, exist_ok=True)
    behaviors = [b for b in report["behaviors"] if report["behaviors"][b].get("cell_table")]
    pal = paper_palette(3)
    lcol = {250: pal[0], 2500: pal[1], 8000: pal[2]}  # one color = one meaning (parent fig)
    # Behavior colors: the OTHER Okabe-Ito members — the blue/orange/green
    # triple above is reserved for L anchors across every sibling figure
    # (one color = one meaning; never reuse a palette pair for a different
    # factor in a sibling figure).
    behavior_palette = {"evil": "#CC79A7", "sycophancy": "#D55E00", "hallucination": "#56B4E9"}
    fallback = ["#5A5A5A", "#8C6BB1", "#B2A100"]
    bcol = {
        b: behavior_palette.get(b, fallback[i % len(fallback)]) for i, b in enumerate(behaviors)
    }
    paths: dict[str, str] = {}

    # ---- hero: compose_fu_flip_v2 — seed means +- t-CI per (f_u group, L) ----
    fig, axes = plt.subplots(1, max(1, len(behaviors)), figsize=(4.2 * len(behaviors), 3.4))
    axes = [axes] if len(behaviors) == 1 else list(axes)
    xpos = {
        ("0.0", "context_end"): 0.0,
        ("0.0", "prefix_end"): 1.0,
        ("0.5", "context_end"): 2.4,
        ("0.5", "prefix_end"): 3.4,
    }
    for ax, b in zip(axes, behaviors):
        for rowc in report["behaviors"][b]["cell_table"]:
            if (rowc["f_u"], rowc["f_l"]) not in ((0.0, 0.0), (0.5, 0.0)):
                continue
            gx = xpos[(str(rowc["f_u"]), rowc["variant"])]
            off = {250: -0.14, 2500: 0.0, 8000: 0.14}.get(rowc["budget_l"], 0.0)
            m = rowc["delta_mean"]
            if m is None:
                continue
            lo_e, hi_e = _clamped_yerr(m, rowc["delta_tci_lo"], rowc["delta_tci_hi"])
            ax.errorbar(
                gx + off,
                m,
                yerr=[[lo_e], [hi_e]],
                fmt="o",
                ms=5,
                color=lcol.get(rowc["budget_l"], "0.4"),
                capsize=2,
                lw=1.2,
            )
        ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
        ax.axvline(1.7, color="0.85", lw=0.8, zorder=0)
        ax.set_xticks([0.0, 1.0, 2.4, 3.4])
        ax.set_xticklabels(["ctx\nf_u=0", "pfx\nf_u=0", "ctx\nf_u=0.5", "pfx\nf_u=0.5"])
        verdict = report["behaviors"][b].get("verdict", "")
        ax.set_title(f"{b} ({verdict})")
        ax.set_ylabel("Δρ (map arm − context arm)" if b == behaviors[0] else "")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=c, label=f"L={ll:,}") for ll, c in lcol.items()
    ]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="best")
    out = savefig_paper(fig, "compose_fu_flip_v2", dir=fig_dir)
    plt.close(fig)
    paths["compose_fu_flip_v2"] = str(out.get("png", ""))

    # ---- dose curve ----
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for b in behaviors:
        curve = report["behaviors"][b].get("dose_curve") or {}
        pts = sorted(
            (v["f_u"], v["mean"], v["lo"], v["hi"])
            for v in curve.values()
            if v.get("mean") is not None
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", ms=4, color=bcol[b], label=b, lw=1.4)
        for x, y, lo, hi in pts:
            lo_e, hi_e = _clamped_yerr(y, lo, hi)
            ax.errorbar(x, y, yerr=[[lo_e], [hi_e]], fmt="none", color=bcol[b], capsize=2, lw=1.0)
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    ax.set_xlabel("f_u (eliciting share of the fixed pool)")
    ax.set_ylabel("Δρ (map arm − context arm)")
    ax.legend(frameon=False, fontsize=8)
    out = savefig_paper(fig, "compose_dose_curve", dir=fig_dir)
    plt.close(fig)
    paths["compose_dose_curve"] = str(out.get("png", ""))

    # ---- exploratory spaghetti: per-seed anchor-pair differences ----
    fig, axes = plt.subplots(1, max(1, len(behaviors)), figsize=(4.2 * len(behaviors), 3.2))
    axes = [axes] if len(behaviors) == 1 else list(axes)
    anchor_cells = [(v, ll) for v in VARIANTS for ll in ANCHOR_BUDGETS]
    labels = [f"{'ctx' if v == 'context_end' else 'pfx'}\nL={ll}" for v, ll in anchor_cells]
    for ax, b in zip(axes, behaviors):
        contrast = report["behaviors"][b].get("flip_contrast") or {}
        per_seed_terms = report["behaviors"][b].get("per_seed_anchor_terms") or {}
        for s, terms in sorted(per_seed_terms.items()):
            ys = [terms.get(f"{v}|L{ll}") for v, ll in anchor_cells]
            xs = [i for i, y in enumerate(ys) if y is not None]
            ax.plot(xs, [ys[i] for i in xs], "-", color="0.7", lw=0.8, alpha=0.9)
        means = []
        for i, (v, ll) in enumerate(anchor_cells):
            vals = [t.get(f"{v}|L{ll}") for t in per_seed_terms.values()]
            vals = [x for x in vals if x is not None]
            means.append(sum(vals) / len(vals) if vals else None)
        xs = [i for i, m in enumerate(means) if m is not None]
        ax.plot(xs, [means[i] for i in xs], "-o", color=bcol[b], lw=1.6, ms=4)
        ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
        ax.set_xticks(range(len(anchor_cells)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(b)
        ax.set_ylabel("per-seed Δρ(f_u=0.5) − Δρ(f_u=0)" if b == behaviors[0] else "")
        _ = contrast
    out = savefig_paper(fig, "compose_ms_spaghetti", dir=fig_dir)
    plt.close(fig)
    paths["compose_ms_spaghetti"] = str(out.get("png", ""))
    return paths


FIGURE_CAPTIONS = {
    "compose_fu_flip_v2": (
        "Headline Δρ (arm6 map-projection − arm2 context-native), seed mean ± 95% t-CI "
        "(df = n_seeds − 1) per (f_u group × variant), L-anchor colored. f_u is the share "
        "of the FIXED 5,000-row compose pool drawn from behavior-eliciting labeled rows "
        "(dose-as-share, not additive pool growth); f_l = 0 (anchor-cell contexts excluded "
        "from the eliciting pool). Evil lacks (f_u=0.5, L=8000) by designed infeasibility."
    ),
    "compose_dose_curve": (
        "Dose curve at L = 2,500, f_l = 0: seed-mean Δρ ± 95% t-CI vs f_u, variants pooled. "
        "f_u is the share of the fixed 5,000-row pool from eliciting rows (dose-as-share). "
        "Evil lacks f_u = 1.0 (designed-infeasible: quota 5,000 vs ~3,968 available) and may "
        "conditionally lack f_u = 0.75; the pre-registered primary trend is f_u ≤ 0.5."
    ),
    "compose_ms_spaghetti": (
        "Exploratory: per-seed anchor-cell flip terms Δρ(f_u=0.5, f_l=0) − Δρ(f_u=0, f_l=0) "
        "(thin gray lines, one per seed) with the seed mean (colored), per behavior."
    ),
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument(
        "--results-root", type=Path, default=Path("eval_results/issue_1739/compose_multiseed")
    )
    ap.add_argument("--behaviors", nargs="+", default=["evil", "sycophancy", "hallucination"])
    ap.add_argument("--halves", nargs="+", default=["s02", "s34"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--banked-evil",
        type=Path,
        default=Path("eval_results/issue_1739/evil/arm_results/percell/cells.jsonl"),
        help="banked round-1 per-cell record for the seed-0 reproduction read",
    )
    ap.add_argument("--banked-tol", type=float, default=BANKED_TOL_DEFAULT)
    ap.add_argument(
        "--allow-banked-divergence",
        action="store_true",
        help="record a material seed-0 divergence as a caveat instead of the designed rc=10 halt",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="interim monitoring read: pair-coverage / missing-half shortfalls are recorded "
        "(verdict PENDING) instead of the designed rc=5 halt",
    )
    ap.add_argument(
        "--labeling-root",
        type=Path,
        default=Path("eval_results/issue_1739/dv_dataset"),
        help="dv_dataset root (context_id -> group_key for the overlap sensitivity recompute)",
    )
    ap.add_argument(
        "--figures-dir", type=Path, default=Path("figures/issue_1739/compose_multiseed")
    )
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="table JSON (default <results-root>/compose_ms_table.json)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    args = _parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds)
    out_path = args.out or (args.results_root / "compose_ms_table.json")
    banked_rows = _read_jsonl(args.banked_evil) if args.banked_evil.exists() else []

    report: dict = {
        "behaviors": {},
        "provenance": {
            "git_commit": _git_commit(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv[1:] if argv is None else list(argv),
            "seeds": list(seeds),
            "halves": list(args.halves),
            "env": {},
        },
        "figure_captions": FIGURE_CAPTIONS,
    }
    import numpy as np
    import scipy

    report["provenance"]["env"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }

    rc = 0
    verdicts: list[str] = []
    for b in args.behaviors:
        data = load_behavior(args.results_root, b, tuple(args.halves))
        entry: dict = {"missing_halves": data["missing_halves"], "audit_failures": []}
        report["behaviors"][b] = entry
        if data["missing_halves"] and not args.allow_partial:
            print(f"[fold] FATAL: {b} missing halves {data['missing_halves']}", flush=True)
            entry["verdict"] = "PENDING"
            verdicts.append("PENDING")
            rc = max(rc, 3)
            continue
        by_key = assert_unique_cells(data["cells"])
        entry["n_cells"] = len(by_key)

        # gate 3 — designed-skip audit FIRST (a wrong skip set poisons the grid)
        audit = designed_skip_audit(data["skips"], b, seeds=seeds)
        entry["skip_audit"] = audit
        if not audit["ok"]:
            entry["verdict"] = "HALTED-SKIP-AUDIT"
            entry["audit_failures"].append("skip-audit")
            verdicts.append("HALTED-SKIP-AUDIT")
            rc = max(rc, 5)
            print(
                f"[fold] {b}: HALTED-SKIP-AUDIT extra={audit['extra_skips']} "
                f"missing={audit['missing_designed_skips']}",
                flush=True,
            )
            continue

        contrast = flip_contrast(by_key, seeds=seeds)
        entry["flip_contrast"] = {k: v for k, v in contrast.items() if k != "per_seed_C"}
        entry["flip_contrast"]["per_seed_C"] = contrast["per_seed_C"]
        # per-seed anchor terms for the spaghetti figure
        per_seed_terms: dict[int, dict[str, float]] = {}
        for v in VARIANTS:
            for ll in ANCHOR_BUDGETS:
                for s in seeds:
                    k05 = (v, 0.5, 0.0, int(ll), 0, int(s))
                    k00 = (v, 0.0, 0.0, int(ll), 0, int(s))
                    if k05 in by_key and k00 in by_key:
                        d05, d00 = headline_delta(by_key[k05]), headline_delta(by_key[k00])
                        if d05 is not None and d00 is not None:
                            per_seed_terms.setdefault(int(s), {})[f"{v}|L{ll}"] = d05 - d00
        entry["per_seed_anchor_terms"] = per_seed_terms

        if contrast["missing_pairs"] and not args.allow_partial:
            entry["verdict"] = "HALTED-PAIR-COVERAGE"
            entry["audit_failures"].append("pair-coverage")
            verdicts.append("HALTED-PAIR-COVERAGE")
            rc = max(rc, 5)
            print(
                f"[fold] {b}: pair coverage {contrast['pairs']}/"
                f"{contrast['pairs_expected']} (missing: {contrast['missing_pairs'][:8]})",
                flush=True,
            )
            continue

        c_values = [v for v in contrast["per_seed_C"].values() if v is not None]
        c_tci = t_ci(c_values)
        entry["C_tci"] = c_tci
        verdict = lattice_verdict(c_tci, contrast["anchor_seedmean_delta05"])

        # map-diagnostics presence (seed-0 map per variant x f_u level)
        levels = list(CORE_LEVELS) + [(f, 0.0) for f in ABLATION_F_U]
        if b == "evil":
            skipped_everywhere = {(1.0, 0.0)}
            cond_recorded = {
                (float(k[1]), float(k[2])) for k in map(tuple, audit["conditional_skips_recorded"])
            }
            levels = [lv for lv in levels if lv not in skipped_everywhere | cond_recorded]
        diag_presence = map_diag_presence(data["diag"], levels=tuple(levels))
        entry["map_diag_presence"] = diag_presence
        if not diag_presence["ok"]:
            entry["audit_failures"].append("map-diag-presence")
            rc = max(rc, 5)

        # overlap sensitivity (fail-safe: pending downgrades FLIP-CONFIRMED)
        groups_by_ctx: dict[str, str] = {}
        labeling = args.labeling_root / b / "labeling.json"
        if labeling.exists():
            groups_by_ctx = load_groups_map(labeling)
        sens = sensitivity_recompute(by_key, data["pool_meta"], groups_by_ctx)
        entry["overlap_sensitivity"] = sens
        entry["verdict_raw"] = verdict
        if verdict == "FLIP-CONFIRMED" and sens["status"] == "pending":
            verdict = "INDETERMINATE"
            entry["verdict_downgrade"] = (
                "FLIP-CONFIRMED downgraded: overlap-sensitivity recompute pending "
                f"({sens['n_pending']} cells without local preds npz)"
            )
        entry["verdict"] = verdict
        verdicts.append(verdict)

        entry["pool_overlap_pairwise"] = pairwise_pool_overlap(data["pool_meta"])
        entry["dose_curve"] = dose_curve(by_key, seeds=seeds)
        entry["cell_table"] = cell_class_table(by_key, data["diag"], seeds=seeds)

        if b == "evil" and banked_rows:
            repro = banked_repro(by_key, banked_rows, tol=args.banked_tol)
            entry["banked_repro"] = {k: v for k, v in repro.items() if k != "compared"}
            entry["banked_repro"]["compared"] = repro["compared"]
            if not repro["ok"] and repro["n_compared"]:
                if args.allow_banked_divergence:
                    entry["banked_repro"]["caveat"] = (
                        "material seed-0 divergence recorded under --allow-banked-divergence"
                    )
                else:
                    rc = max(rc, 10)
                    print(
                        f"[fold] evil: banked seed-0 reproduction DIVERGES "
                        f"(max |diff| {repro['max_abs_diff']}) — designed halt rc=10",
                        flush=True,
                    )
        print(
            f"[fold] {b}: verdict={verdict} C_mean={c_tci['mean']} "
            f"CI=[{c_tci['lo']}, {c_tci['hi']}] pairs={contrast['pairs']}",
            flush=True,
        )

    report["round_verdict"] = round_verdict(verdicts) if verdicts else "PENDING"
    print(f"[fold] round verdict: {report['round_verdict']}", flush=True)

    if not args.no_figures:
        report["figures"] = render_figures(report, args.figures_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    tmp.replace(out_path)
    print(f"[fold] table -> {out_path}", flush=True)

    csv_path = out_path.with_name("compose_ms_cells.csv")
    rows = []
    for b, entry in report["behaviors"].items():
        for rowc in entry.get("cell_table") or ():
            flat = {"behavior": b, **{k: v for k, v in rowc.items() if not isinstance(v, list)}}
            flat["seeds"] = " ".join(str(s) for s in rowc.get("seeds", []))
            flat["delta_per_seed"] = " ".join(f"{d:.6f}" for d in rowc.get("delta_per_seed", []))
            rows.append(flat)
    if rows:
        cols = sorted({k for r in rows for k in r}, key=lambda c: (c != "behavior", c))
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[fold] cells csv -> {csv_path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
