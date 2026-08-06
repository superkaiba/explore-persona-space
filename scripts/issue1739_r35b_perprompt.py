#!/usr/bin/env python3
"""#1739 Result 3.5b: per-PROMPT — is behavior prediction bad exactly where the map errs?

Row-level successor of Result 3.5 (``issue1739_r35_mapquality_vs_pred.py``, 22
non-independent per-rung cells): uses the main-grid per-cell prediction
sidecars (``eval_results/issue_1739/<behavior>/arm_results/percell/preds/``,
staged by ``issue1739_stage_percell_preds.py``) to ask, at n = thousands of
prompts per cell, whether the context->answer map's per-prompt distortion
lands on exactly the prompts where behavior prediction fails.

Design (per cell; a cell = one (behavior, regime, u_rung_label, variant,
budget_l, seed, draw) unit of the main grid):

1. DV-free stratifier ``d_i = pred6_i - pred11_i`` (map-projected minus
   real-answer projection: the map's distortion along the behavior
   direction, each arm at its own frozen layer). Prompts split into
   within-cell quartiles of ``|d_i|``. No DV enters the split.
2. Within each stratum: Spearman rho of the map arm (arm6), the oracle arm
   (arm11) and the context arm (arm1) against the judge DV, with
   Bonett-Wright CIs. The read is the gap rho6 - rho11 across strata (the
   oracle arm controls each stratum's intrinsic difficulty).
3. Permutation null (load-bearing): ``pred6~ = pred11 + d[perm]`` for
   within-cell permutations — same distortion magnitudes, prompt-specific
   placement destroyed. The full stratified profile is recomputed per draw
   (strata re-derived from the permuted distortions), giving a null band for
   the gap-vs-stratum curve AND for the unstratified rho6. Fully vectorized
   over draws (no Python loop per draw): batched midranks via
   ``scipy.stats.rankdata(axis=1)`` + batched Pearson.
4. Descriptive companion: within-cell Spearman between |d_i| and the oracle
   arm's rank-space miss |rank_pct(pred11) - rank_pct(dv)| (caveat: both
   terms involve pred11).

Tautology guard: ``err6 = err11 + d`` identically, so |err6| vs |d| is never
computed — the calibration against the placement-destroying null is the test.

PRIMARY grain: one cell per (behavior, regime, u_rung_label in {250, 5000,
full}, variant) — the max-budget seed-0 draw-0 cell (verified: the three
projection arms are bit-identical across seeds and, at max budget, across
draws; max budget = the full row pool). Permutation battery runs on primary
cells only (exact within-cell null). SENSITIVITY: observed profiles for all
seed-0 cells (all budgets/draws + the evil compose-map cells), no battery.
Pooled reads aggregate primary cells with matched-draw null aggregation;
cells share prompts across regimes/labels/variants, so pooled bands are
approximate (stated) — the per-cell reads are the confirmatory grain.

Scope: main-grid cells only (linear map; the R2FAIR fair-protocol refit
persisted no per-row predictions). eval_rung == 'train' throughout: this is
the train eval distribution, not the OOD transfer rungs. **context_end
variant ONLY — the prefix arm is excluded at explicit user direction**
(a USER-DIRECTED scope decision for this round, NOT an
artifact-availability deviation: prefix_end cells exist in the main grid
and are simply not read here; distinct from the prior R3.5 round, whose
context-only scope was forced by missing prefix reconstructions).

Outputs (eval_results/issue_1739/result3_5b_perprompt/):
  r35b_observed_cells.json   observed profiles, every seed-0 cell
  r35b_primary_null.json     primary cells: profiles + null bands + p-values
  r35b_pooled.json           pooled gap curves + matched-draw null bands
  r35b_summary.json          headline numbers
Figures (via --figures) under figures/issue_1739/result3_5b_perprompt/.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "eval_results" / "issue_1739"
OUT_ROOT = RESULTS_ROOT / "result3_5b_perprompt"
FIG_ROOT = REPO_ROOT / "figures" / "issue_1739" / "result3_5b_perprompt"

BEHAVIORS = ("evil", "sycophancy", "hallucination")
ARM_MAP = "arm6_map_proj_e1"  # persona direction projected on the MAPPED answer state
ARM_ORACLE = "arm11_oracle_proj"  # projected on the REAL answer state
ARM_CTX = "arm1_ctx_e1"  # projected on the context state
CORE_LABELS = ("250", "5000", "full")  # map-training-budget rungs of the primary grain
VARIANTS = ("context_end",)  # prefix_end excluded at explicit user direction (this round)
N_STRATA = 4
Z95 = 1.959963984540054


# ---------------------------------------------------------------- loading


def load_cells(behavior: str) -> list[dict]:
    """cells.jsonl rows with parsed unit_key + resolved npz path."""
    root = RESULTS_ROOT / behavior / "arm_results" / "percell"
    out = []
    for line in (root / "cells.jsonl").open():
        rec = json.loads(line)
        u = json.loads(rec["unit_key"]) if isinstance(rec["unit_key"], str) else rec["unit_key"]
        npz = root / "preds" / rec["preds_npz"]
        if not npz.is_file():
            raise FileNotFoundError(f"{behavior}: staged sidecar missing: {npz}")
        out.append({"behavior": behavior, "unit": u, "npz": npz, "rec": rec})
    return out


def cell_slug(u: dict) -> str:
    return (
        f"{u['regime']}|{u['u_rung_label']}|{u['variant']}"
        f"|L{u['budget_l']}|s{u['seed']}|d{u['draw']}"
    )


def frozen_layers(rec: dict) -> dict[str, int | None]:
    return {row["arm"]: row.get("layer") for row in rec.get("arms", [])}


# ---------------------------------------------------------------- stats helpers


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def bonett_wright_ci(rho: float, n: int) -> tuple[float, float]:
    """95% CI for a Spearman rho (Bonett & Wright 2000 Fisher-z variant)."""
    if not np.isfinite(rho) or n <= 3 or abs(rho) >= 1:
        return (float("nan"), float("nan"))
    se = np.sqrt((1 + rho**2 / 2) / (n - 3))
    z = np.arctanh(rho)
    return (float(np.tanh(z - Z95 * se)), float(np.tanh(z + Z95 * se)))


def midranks2d(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, axis=1, method="average")


def batched_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson of (B, m) vs (B, m) (or broadcastable (1, m))."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = den > 0
    np.divide(num, den, out=out, where=ok)
    return out


def perm_p_two_sided(null: np.ndarray, obs: float) -> float:
    """Two-sided permutation p with add-one correction (doubled smaller tail)."""
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(obs):
        return float("nan")
    b = null.size
    lo = (1 + int((null <= obs).sum())) / (b + 1)
    hi = (1 + int((null >= obs).sum())) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


# ---------------------------------------------------------------- per-cell observed


def quartile_labels(absd: np.ndarray) -> np.ndarray:
    """Within-cell quartile of |d| (0 = smallest .. 3 = largest), near-equal sizes."""
    n = absd.size
    order = stats.rankdata(absd, method="ordinal") - 1  # 0..n-1, ties broken stably
    return np.minimum((order * N_STRATA) // n, N_STRATA - 1).astype(np.int64)


def cell_observed(z: np.lib.npyio.NpzFile) -> dict | None:
    """Observed stratified profile for one cell; None if an arm is missing."""
    keys = set(z.files)
    for arm in (ARM_MAP, ARM_ORACLE, ARM_CTX):
        if f"pred__{arm}" not in keys:
            return None
    dv = z["dv"].astype(np.float64)
    p6 = z[f"pred__{ARM_MAP}"].astype(np.float64)
    p11 = z[f"pred__{ARM_ORACLE}"].astype(np.float64)
    p1 = z[f"pred__{ARM_CTX}"].astype(np.float64)
    ok = np.isfinite(dv) & np.isfinite(p6) & np.isfinite(p11) & np.isfinite(p1)
    n_dropped = int((~ok).sum())
    dv, p6, p11, p1 = dv[ok], p6[ok], p11[ok], p1[ok]
    n = dv.size
    if n < 4 * 10:
        return None

    rho11_all = spearman(p11, dv)
    flipped = bool(np.isfinite(rho11_all) and rho11_all < 0)
    if flipped:  # direction sign is a global convention; align so oracle reads positive
        p6, p11, p1 = -p6, -p11, -p1
        rho11_all = -rho11_all

    sd11 = float(np.std(p11, ddof=1))
    d = p6 - p11
    q = quartile_labels(np.abs(d))
    strata = []
    for s in range(N_STRATA):
        m = q == s
        ns = int(m.sum())
        r6, r11, r1 = spearman(p6[m], dv[m]), spearman(p11[m], dv[m]), spearman(p1[m], dv[m])
        strata.append(
            {
                "stratum": s,
                "n": ns,
                "rho6": r6,
                "rho6_ci": bonett_wright_ci(r6, ns),
                "rho11": r11,
                "rho11_ci": bonett_wright_ci(r11, ns),
                "rho1": r1,
                "rho1_ci": bonett_wright_ci(r1, ns),
                "gap": (r6 - r11) if np.isfinite(r6) and np.isfinite(r11) else float("nan"),
                "mean_abs_d_norm": float(np.mean(np.abs(d[m])) / sd11)
                if sd11 > 0
                else float("nan"),
                "dv_constant": bool(np.std(dv[m]) == 0),
            }
        )

    # descriptive companion (both terms involve pred11 — reported as such)
    rp11 = (stats.rankdata(p11) - 0.5) / n
    rdv = (stats.rankdata(dv) - 0.5) / n
    companion = spearman(np.abs(d), np.abs(rp11 - rdv))

    # mechanism diagnostics: is the distortion shrinkage of the oracle signal
    # (corr(d, p11) ~ -1: the map attenuates the behavior-direction component,
    # so large |d| lands on strongly-scored prompts by construction), and does
    # the distortion itself carry DV signal (spearman(d, dv) > 0: benign)?
    sd6 = float(np.std(p6, ddof=1))
    corr_d_p11 = (
        float(np.corrcoef(d, p11)[0, 1]) if np.std(d) > 0 and np.std(p11) > 0 else float("nan")
    )
    rho_d_dv = spearman(d, dv)

    rho6_all, rho1_all = spearman(p6, dv), spearman(p1, dv)
    gaps = [st["gap"] for st in strata]
    return {
        "n": n,
        "n_dropped_nonfinite": n_dropped,
        "flipped_sign": flipped,
        "sd_pred11": sd11,
        "rho6_all": rho6_all,
        "rho11_all": rho11_all,
        "rho1_all": rho1_all,
        "gap_all": (rho6_all - rho11_all)
        if np.isfinite(rho6_all) and np.isfinite(rho11_all)
        else float("nan"),
        "strata": strata,
        "gap_slope_q4_minus_q1": (gaps[3] - gaps[0])
        if np.isfinite(gaps[3]) and np.isfinite(gaps[0])
        else float("nan"),
        "companion_absd_vs_oracle_miss": companion,
        "sd_pred6": sd6,
        "sd6_over_sd11": (sd6 / sd11) if sd11 > 0 else float("nan"),
        "n_unique_pred6": int(np.unique(p6).size),
        "pearson_d_pred11": corr_d_p11,
        "spearman_d_dv": rho_d_dv,
        "flag_rho11_indistinct_from_zero": bool(
            not np.isfinite(rho11_all) or abs(rho11_all) < Z95 * 1.03 / np.sqrt(max(n - 3, 1))
        ),
        "flag_rho11_weak": bool(not np.isfinite(rho11_all) or abs(rho11_all) < 0.1),
        "flag_map_projection_degenerate": bool(
            sd11 > 0 and sd6 / sd11 < 0.01
        ),  # mapped-answer projection near-constant (e.g. syco prefix: 2 unique f32 values)
        "flag_sd_pred11_degenerate": bool(sd11 <= 0),
        "_arrays": {"dv": dv, "p6": p6, "p11": p11, "p1": p1, "d": d, "q": q},
    }


# ---------------------------------------------------------------- permutation null


def cell_null(obs: dict, unit_key: str, n_draws: int, rng_salt: str = "r35b") -> dict:
    """Vectorized within-cell placement-destroying null for one cell.

    pred6~ = pred11 + d[perm]; strata re-derived per draw (row i's stratum =
    the quartile label of the distortion it RECEIVED). Returns per-stratum
    null gap distributions + unstratified null rho6.
    """
    arr = obs["_arrays"]
    dv, p11, d, q = arr["dv"], arr["p11"], arr["d"], arr["q"]
    n = dv.size
    seed = int.from_bytes(hashlib.sha1(f"{rng_salt}|{unit_key}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_draws, n)), axis=1)  # (B, n) permutations
    inv = np.argsort(perm, axis=1)  # inv[b, j] = row that received distortion j

    # unstratified null rho6: rank dv once (identical rows every draw)
    rdv_full = midranks2d(dv[None, :])
    p6_null_full = p11[None, :] + np.take(d, perm)  # d[perm] row-received distortions
    rho6_null_all = batched_pearson(midranks2d(p6_null_full), rdv_full)
    del p6_null_full

    gap_null = np.full((n_draws, N_STRATA), np.nan)
    rho6_null = np.full((n_draws, N_STRATA), np.nan)
    rho11_null = np.full((n_draws, N_STRATA), np.nan)
    for s in range(N_STRATA):
        j_s = np.flatnonzero(q == s)  # original rows whose distortion has label s
        members = inv[:, j_s]  # (B, n_s): rows that received those distortions
        dv_sub = dv[members]
        p11_sub = p11[members]
        p6_sub = p11_sub + d[j_s][None, :]
        r_dv = midranks2d(dv_sub)
        rho6_null[:, s] = batched_pearson(midranks2d(p6_sub), r_dv)
        rho11_null[:, s] = batched_pearson(midranks2d(p11_sub), r_dv)
    gap_null = rho6_null - rho11_null

    obs_gaps = np.array([st["gap"] for st in obs["strata"]])
    t_obs = obs["gap_slope_q4_minus_q1"]
    t_null = gap_null[:, 3] - gap_null[:, 0]
    return {
        "n_draws": int(n_draws),
        "rng_seed": seed,
        "gap_null_mean": [float(np.nanmean(gap_null[:, s])) for s in range(N_STRATA)],
        "gap_null_q025": [float(np.nanquantile(gap_null[:, s], 0.025)) for s in range(N_STRATA)],
        "gap_null_q975": [float(np.nanquantile(gap_null[:, s], 0.975)) for s in range(N_STRATA)],
        "gap_p_two_sided_per_stratum": [
            perm_p_two_sided(gap_null[:, s], obs_gaps[s]) for s in range(N_STRATA)
        ],
        "slope_null_q025": float(np.nanquantile(t_null, 0.025)),
        "slope_null_q975": float(np.nanquantile(t_null, 0.975)),
        "slope_p_two_sided": perm_p_two_sided(t_null, t_obs),
        "rho6_all_null_mean": float(np.nanmean(rho6_null_all)),
        "rho6_all_null_q025": float(np.nanquantile(rho6_null_all, 0.025)),
        "rho6_all_null_q975": float(np.nanquantile(rho6_null_all, 0.975)),
        "rho6_all_p_two_sided": perm_p_two_sided(rho6_null_all, obs["rho6_all"]),
        "rho6_all_p_worse_than_null": float(
            (1 + int((rho6_null_all[np.isfinite(rho6_null_all)] <= obs["rho6_all"]).sum()))
            / (int(np.isfinite(rho6_null_all).sum()) + 1)
        ),
        "_gap_null_draws": gap_null,  # kept in-memory for matched-draw pooling
    }


# ---------------------------------------------------------------- orchestration


def is_primary(u: dict, max_budget: int) -> bool:
    return (
        u["u_rung_label"] in CORE_LABELS
        and u["budget_l"] == max_budget
        and u["seed"] == 0
        and u["draw"] == 0
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-draws", type=int, default=500)
    ap.add_argument("--figures", action="store_true", help="render figures after the stats")
    args = ap.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_cells = {b: load_cells(b) for b in BEHAVIORS}
    max_budget = {
        b: max(c["unit"]["budget_l"] for c in cs if c["unit"]["u_rung_label"] in CORE_LABELS)
        for b, cs in all_cells.items()
    }

    observed_rows: list[dict] = []
    primary_rows: list[dict] = []
    t_batt0 = time.time()
    pilot_printed = False
    for b in BEHAVIORS:
        for c in all_cells[b]:
            u = c["unit"]
            if u["variant"] not in VARIANTS:
                continue  # prefix_end excluded at user direction — never loaded or scored
            if u["seed"] != 0:
                continue  # projection arms verified bit-identical across seeds
            z = np.load(c["npz"], allow_pickle=False)
            obs = cell_observed(z)
            if obs is None:
                observed_rows.append(
                    {"behavior": b, "cell": cell_slug(u), "skipped": "arm missing or n too small"}
                )
                continue
            fl = frozen_layers(c["rec"])
            row = {
                "behavior": b,
                "cell": cell_slug(u),
                "regime": u["regime"],
                "u_rung_label": u["u_rung_label"],
                "variant": u["variant"],
                "budget_l": u["budget_l"],
                "draw": u["draw"],
                "frozen_layer_map_arm": fl.get(ARM_MAP),
                "frozen_layer_oracle_arm": fl.get(ARM_ORACLE),
                **{k: v for k, v in obs.items() if not k.startswith("_")},
            }
            observed_rows.append(row)
            if is_primary(u, max_budget[b]):
                t0 = time.time()
                null = cell_null(obs, c["rec"]["unit_key"], args.n_draws)
                dt = time.time() - t0
                if not pilot_printed:
                    print(
                        f"[pilot] first primary cell null battery: n={obs['n']}, "
                        f"B={args.n_draws}, wall={dt:.1f}s -> projected "
                        f"~{dt * 42 / 60:.1f} min for 42 primary cells"
                    )
                    pilot_printed = True
                # placement classification vs the unstratified null band
                p_worse = null["rho6_all_p_worse_than_null"]
                if not np.isfinite(row["rho6_all"]):
                    placement = "degenerate"
                elif p_worse <= 0.025:
                    placement = "worse-than-null"
                elif p_worse >= 0.975:
                    placement = "better-than-null"
                else:
                    placement = "within-null"
                primary_rows.append(
                    {
                        **row,
                        **{k: v for k, v in null.items() if k[0] != "_"},
                        "placement_class": placement,
                        "_gap_null_draws": null["_gap_null_draws"],
                    }
                )
    print(f"[battery] all primary nulls done in {(time.time() - t_batt0) / 60:.1f} min")

    # ---- pooled reads (matched-draw aggregation of within-cell nulls) ----
    pooled = {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in primary_rows:
        groups[(r["behavior"], r["variant"])].append(r)
        groups[("pooled", r["variant"])].append(r)
    for (gb, gv), rows in sorted(groups.items()):
        obs_curves = np.array([[st["gap"] for st in r["strata"]] for r in rows])
        null_stack = np.stack([r["_gap_null_draws"] for r in rows])  # (cells, B, 4)
        null_mean_curve = np.nanmean(null_stack, axis=0)  # (B, 4) matched-draw means
        obs_mean = np.nanmean(obs_curves, axis=0)
        t_obs = float(obs_mean[3] - obs_mean[0])
        t_null = null_mean_curve[:, 3] - null_mean_curve[:, 0]
        pooled[f"{gb}|{gv}"] = {
            "n_cells": len(rows),
            "cells": [r["cell"] for r in rows],
            "gap_mean_observed": [float(x) for x in obs_mean],
            "gap_null_q025": [
                float(np.nanquantile(null_mean_curve[:, s], 0.025)) for s in range(4)
            ],
            "gap_null_q975": [
                float(np.nanquantile(null_mean_curve[:, s], 0.975)) for s in range(4)
            ],
            "gap_null_mean": [float(np.nanmean(null_mean_curve[:, s])) for s in range(4)],
            "slope_observed": t_obs,
            "slope_p_two_sided": perm_p_two_sided(t_null, t_obs),
            "note": "cells share prompts across regimes/labels/variants; matched-draw null "
            "band assumes within-cell exchangeability and is approximate at this grain",
        }

    # ---- persist ----
    for r in primary_rows:
        r.pop("_gap_null_draws", None)
    (OUT_ROOT / "r35b_observed_cells.json").write_text(json.dumps(observed_rows, indent=1))
    (OUT_ROOT / "r35b_primary_null.json").write_text(json.dumps(primary_rows, indent=1))
    (OUT_ROOT / "r35b_pooled.json").write_text(json.dumps(pooled, indent=1))

    # ---- headline summary ----
    def _fin(x):
        return [v for v in x if np.isfinite(v)]

    slopes = _fin([r["gap_slope_q4_minus_q1"] for r in primary_rows])
    slope_ps = _fin([r["slope_p_two_sided"] for r in primary_rows])
    rho6_ps = _fin([r["rho6_all_p_two_sided"] for r in primary_rows])
    companions = _fin([r["companion_absd_vs_oracle_miss"] for r in primary_rows])
    summary = {
        "design": "per-prompt |pred6-pred11| quartile strata; gap rho6-rho11 vs "
        "within-cell placement-destroying permutation null",
        "n_primary_cells": len(primary_rows),
        "n_observed_cells_seed0": len(observed_rows),
        "primary_slope_median": float(np.median(slopes)) if slopes else None,
        "primary_slope_frac_negative": float(np.mean([s < 0 for s in slopes])) if slopes else None,
        "primary_slope_p_below_05_count": int(sum(p < 0.05 for p in slope_ps)),
        "primary_rho6all_p_below_05_count": int(sum(p < 0.05 for p in rho6_ps)),
        "n_p_tests_per_family": len(slope_ps),
        "companion_median": float(np.median(companions)) if companions else None,
        "companion_iqr": [float(np.quantile(companions, q)) for q in (0.25, 0.75)]
        if companions
        else None,
        "placement_by_group": {
            f"{b}|{v}": {
                cls: sum(
                    1
                    for r in primary_rows
                    if r["behavior"] == b and r["variant"] == v and r["placement_class"] == cls
                )
                for cls in ("worse-than-null", "better-than-null", "within-null", "degenerate")
            }
            for b in BEHAVIORS
            for v in VARIANTS
        },
        "scope": "context_end only — prefix arm excluded at explicit user direction "
        "(user-directed scope decision, not artifact availability: prefix_end cells "
        "exist in the main grid and were not read)",
        "flags": {
            "rho11_indistinct_cells": [
                r["cell"] for r in primary_rows if r["flag_rho11_indistinct_from_zero"]
            ],
            "sd11_degenerate_cells": [
                r["cell"] for r in primary_rows if r["flag_sd_pred11_degenerate"]
            ],
            "flipped_sign_cells": [r["cell"] for r in primary_rows if r["flipped_sign"]],
        },
    }
    (OUT_ROOT / "r35b_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    if args.figures:
        render_figures(primary_rows, pooled)
    return 0


# ---------------------------------------------------------------- figures


BEHAVIOR_LABEL = {"evil": "Evil", "sycophancy": "Sycophancy", "hallucination": "Hallucination"}
VARIANT_LABEL = {"context_end": "Context-based mapping", "prefix_end": "Prefix-based mapping"}


def render_figures(primary_rows: list[dict], pooled: dict) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    colors = paper_palette_blog(3)
    bcol = dict(zip(BEHAVIORS, colors))
    xs = np.arange(1, N_STRATA + 1)

    # --- figure 1: gap-vs-quartile curves with pooled null band ---
    variants = VARIANTS
    fig, axes = plt.subplots(
        len(variants),
        3,
        figsize=(12, 3.8 * len(variants)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for i, var in enumerate(variants):
        for j, b in enumerate(BEHAVIORS):
            ax = axes[i, j]
            rows = [r for r in primary_rows if r["behavior"] == b and r["variant"] == var]
            for r in rows:
                ax.plot(
                    xs,
                    [st["gap"] for st in r["strata"]],
                    color=bcol[b],
                    alpha=0.35,
                    lw=1.0,
                    zorder=2,
                )
            key = f"{b}|{var}"
            if key in pooled:
                p = pooled[key]
                ax.fill_between(
                    xs,
                    p["gap_null_q025"],
                    p["gap_null_q975"],
                    color="0.55",
                    alpha=0.35,
                    lw=0,
                    zorder=1,
                    label="Permutation null (95% band)",
                )
                ax.plot(
                    xs,
                    p["gap_mean_observed"],
                    color=bcol[b],
                    lw=2.5,
                    marker="o",
                    zorder=3,
                    label="Observed (mean over cells)",
                )
            ax.axhline(0, color="0.3", lw=0.8, ls=":")
            if i == 0:
                ax.set_title(BEHAVIOR_LABEL[b])
            if j == 0:
                ax.set_ylabel(f"{VARIANT_LABEL[var]}\n" r"$\rho$(mapped) $-$ $\rho$(real answer)")
            if i == len(variants) - 1:
                ax.set_xlabel("Per-prompt map-distortion quartile\n(smallest → largest)")
            ax.set_xticks(list(xs))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03)
    )
    fig.suptitle(
        "Prediction cost of the map's per-prompt distortion, by distortion quartile", y=1.005
    )
    fig.tight_layout()
    savefig_paper(fig, "r35b_gap_profile", dir=FIG_ROOT)
    plt.close(fig)

    # --- figure 2: unstratified rho6 vs its placement-destroying null, per cell ---
    fig, axes = plt.subplots(
        1, len(variants), figsize=(6.5 * len(variants), 4.5), sharey=True, squeeze=False
    )
    for i, var in enumerate(variants):
        ax = axes[0, i]
        rows = [r for r in primary_rows if r["variant"] == var]
        rows.sort(key=lambda r: (BEHAVIORS.index(r["behavior"]), r["regime"], r["u_rung_label"]))
        for k, r in enumerate(rows):
            ax.vlines(
                k,
                r["rho6_all_null_q025"],
                r["rho6_all_null_q975"],
                color="0.6",
                lw=3,
                alpha=0.8,
                zorder=1,
                label="Null band (95%)" if k == 0 else None,
            )
            ax.plot(
                k,
                r["rho6_all"],
                "o",
                color=bcol[r["behavior"]],
                ms=5,
                zorder=3,
                label="Observed, mapped answer" if k == 0 else None,
            )
            ax.plot(
                k,
                r["rho11_all"],
                "_",
                color="black",
                ms=9,
                mew=1.5,
                zorder=2,
                label="Real answer (oracle)" if k == 0 else None,
            )
        # behavior group separators + labels
        counts = {}
        for r in rows:
            counts[r["behavior"]] = counts.get(r["behavior"], 0) + 1
        left = 0
        for b in BEHAVIORS:
            cnt = counts.get(b, 0)
            if cnt == 0:
                continue
            ax.text(
                left + cnt / 2 - 0.5,
                ax.get_ylim()[0],
                BEHAVIOR_LABEL[b],
                ha="center",
                va="bottom",
                fontsize=9,
                color=bcol[b],
            )
            left += cnt
            if left < len(rows):
                ax.axvline(left - 0.5, color="0.85", lw=0.8)
        ax.set_title(VARIANT_LABEL[var])
        ax.set_xlabel("Cell (direction regime × map-training budget)")
        ax.set_xticks([])
    axes[0, 0].set_ylabel(r"Spearman $\rho$ vs. behavior score")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(
        "Whole-cell prediction from the mapped answer vs. a null that shuffles the map's "
        "distortions across prompts",
        y=1.02,
    )
    fig.tight_layout()
    savefig_paper(fig, "r35b_overall_vs_null", dir=FIG_ROOT)
    plt.close(fig)
    print(f"[figures] written under {FIG_ROOT}")


if __name__ == "__main__":
    raise SystemExit(main())
