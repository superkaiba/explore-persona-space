#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ρ, −) in scientific docstrings + labels.
"""Issue #928 free-analysis follow-up: CoT-length-matched CoT-gain read.

Separates the CoT-LENGTH covariate from the parse-COVERAGE (flagged-cluster)
covariate on the per-context CoT gain Δ(G − D) = skill(g_aug) − skill(d_ctx2ans):
the analyzer measured ρ(median CoT length, Δ) = −0.70 pooled / −0.82 within the
36 unflagged contexts, while the drop-rate association (+0.64 pooled) collapses
to +0.19 within unflagged — one covariate or two?

Everything is a PURE RE-REDUCTION of the persisted per-context LOCO error
decompositions (``decomp_{avg_q,indiv}.pt`` — per (arm, combo, layer):
``ss_res``/``ss_tot`` (50,) arrays in battery order), at the PRIMARY FROZEN
convention per regime (mean/mean at the direct arm's full-data best LOCO layer
— 27 avg_q / 25 indiv, read from ``bootstrap_deltaskill.json``; fixed before
any draw). No refit anywhere.

Bootstrap convention (validated to reproduce the analyzer's flagged-exclusion
read exactly): the ONE persisted shared resample-index matrix
(``make_bootstrap_index_matrix(50, 2000, seed=42)``) is RESTRICTED to each
context subset S by masking — a draw keeps only its sampled entries that fall
in S (``np.isin``), so subset draws stay PAIRED across arms, regimes, and
subsets. Reproduces: full-set Δ = committed H2 (avg_q 0.1086 [0.0606, 0.1767];
indiv 0.2033 [0.1464, 0.2717]) and the body's flagged-exclusion read (indiv
0.1240 [0.0792, 0.1794] ≙ "+0.12, CI +0.08 to +0.18"; avg_q 0.1273 ≙ "+0.13").
The full-set reproduction is asserted at runtime (fail-loud validation gate).

All re-reductions are batched numpy gathers over the (2000, 50) index matrix —
no serial per-draw loop (`.claude/rules/vectorize-many-cell-fits.md` item 3).

Inputs (all committed / local; ANALYSIS-ONLY — no new model calls):
- ``eval_results/issue_928/decomp_{avg_q,indiv}.pt`` (per-context ss arrays)
- ``eval_results/issue_928/bootstrap_deltaskill.json`` (frozen layers, n_boot, seed)
- ``eval_results/issue_928/recon_skill_grid.json`` (battery-order context ids)
- ``figures/issue_928/percontext_scatter_avg_q.meta.json`` (per-context median
  CoT char lengths, embedded figure data; cross-checked against the indiv meta)

Outputs:
- ``eval_results/issue_928/length_matched_gain.json``
- ``figures/issue_928/length_matched_gain.{png,pdf,meta.json}``

Usage (repo-root inputs, worktree outputs)::

    uv run python scripts/issue928_length_matched_gain.py \
        --in-results <repo>/eval_results/issue_928 \
        --in-figures <repo>/figures/issue_928 \
        --out-results eval_results/issue_928 --out-figures figures/issue_928
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue928_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    dump_json,
    load_json,
    reproducibility_metadata,
)
from issue928_null_bootstrap import make_bootstrap_index_matrix, stat_summary  # noqa: E402
from scipy import stats as sps  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue928_length_matched_gain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# The 14 contexts below the 80% parse floor (PARSE_RATE_FLOOR), in the run's
# store manifest ``flagged_below_parse_floor``. The manifest lives on the HF
# data repo (issue928_cot_decomposition/analysis_tensors/store/percq_summaries);
# this list is its verbatim mirror from the run's epm:results sentinel
# (task #928 events.jsonl ts 2026-07-04T06:11:08Z, git 328ab540ff) so the
# analysis stays local-input-only. Asserted ⊂ battery ids at load time.
FLAGGED_BELOW_PARSE_FLOOR = (
    "f2_wc_short_3",
    "f2_wc_short_5",
    "f2_wc_long_1",
    "f2_wc_long_2",
    "f2_wc_long_3",
    "f2_wc_long_4",
    "f2_wc_long_5",
    "f3_icl_marker_k4",
    "f3_icl_french_k4",
    "f3_icl_json_k4",
    "f3_icl_pirate_k4",
    "f3_icl_marker_k2",
    "f3_icl_marker_k8",
    "f3_icl_json_k8",
)

GAIN_ARM, DIRECT_ARM, PRIMARY_COMBO = "g_aug", "d_ctx2ans", "mean/mean"
REGIMES = ("avg_q", "indiv")


# ── inputs ────────────────────────────────────────────────────────────────────


def load_cot_lengths(fig_dir: Path, context_ids: list[str]) -> np.ndarray:
    """(50,) per-context median CoT char lengths from the committed figure metas.

    Reads the length-confound scatter's embedded per-point data (series 1:
    "median CoT length (chars)" vs skill, one labeled point per context) from
    the avg_q meta and cross-asserts equality against the indiv meta (both were
    produced from the SAME ``cot_len_by_ctx``). Fails loud on any missing
    context — every battery context had ≥1 well-formed row in the run.
    """
    by_ctx: dict[str, float] = {}
    for regime in REGIMES:
        meta = load_json(fig_dir / f"percontext_scatter_{regime}.meta.json")
        pts = [p for p in meta["points"] if "median CoT length (chars)" in p]
        got = {p["label"]: float(p["median CoT length (chars)"]) for p in pts}
        if not by_ctx:
            by_ctx = got
        elif got != by_ctx:
            raise RuntimeError("percontext meta CoT lengths disagree between regimes")
    missing = [c for c in context_ids if c not in by_ctx]
    if missing:
        raise RuntimeError(f"{len(missing)} contexts missing CoT length (e.g. {missing[:3]})")
    return np.array([by_ctx[c] for c in context_ids], dtype=np.float64)


def load_arm_ss(results_dir: Path, regime: str, layer: int) -> dict[str, dict[str, np.ndarray]]:
    """Per-arm ``{"ss_res": (50,), "ss_tot": (50,)}`` at the frozen (combo, layer)."""
    decomp = torch.load(results_dir / f"decomp_{regime}.pt", weights_only=False)
    out = {}
    for arm in (GAIN_ARM, DIRECT_ARM):
        key = str((arm, PRIMARY_COMBO, layer))
        if key not in decomp:
            raise RuntimeError(f"decomp_{regime}.pt missing key {key}")
        v = decomp[key]
        res, tot = np.asarray(v["ss_res"], np.float64), np.asarray(v["ss_tot"], np.float64)
        assert res.shape == tot.shape == (50,), (res.shape, tot.shape)
        out[arm] = {"ss_res": res, "ss_tot": tot}
    return out


# ── subset re-reduction (batched; the analyzer's flagged-exclusion convention) ─


def subset_skill(res: np.ndarray, tot: np.ndarray, pos: np.ndarray) -> float:
    """Pooled skill over subset ``pos``: 1 − Σss_res[S]/Σss_tot[S]."""
    t = float(tot[pos].sum())
    return float("nan") if t < 1e-12 else 1.0 - float(res[pos].sum()) / t


def subset_boot_skills(
    res: np.ndarray, tot: np.ndarray, idx: np.ndarray, member: np.ndarray
) -> np.ndarray:
    """(n_draws,) pooled skill per draw with the shared matrix RESTRICTED to a subset.

    ``member = np.isin(idx, pos)`` masks each draw's sampled contexts to the
    subset — one batched gather + masked sum over the whole (2000, 50) matrix,
    no per-draw loop. Draws whose restricted ss_tot vanishes yield NaN
    (filtered by ``stat_summary``).
    """
    res_d = (res[idx] * member).sum(axis=1)
    tot_d = (tot[idx] * member).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot_d < 1e-12, np.nan, 1.0 - res_d / tot_d)


def delta_stat(
    arms: dict[str, dict[str, np.ndarray]], pos: np.ndarray, idx: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Observed + bootstrap Δ(G − D) over subset ``pos``; returns (summary, draws)."""
    member = np.isin(idx, pos)
    obs, draws = {}, {}
    for arm in (GAIN_ARM, DIRECT_ARM):
        a = arms[arm]
        obs[arm] = subset_skill(a["ss_res"], a["ss_tot"], pos)
        draws[arm] = subset_boot_skills(a["ss_res"], a["ss_tot"], idx, member)
    d_obs = obs[GAIN_ARM] - obs[DIRECT_ARM]
    d_draws = draws[GAIN_ARM] - draws[DIRECT_ARM]
    summary = stat_summary(d_obs, d_draws)
    summary["skill_g_aug"] = obs[GAIN_ARM]
    summary["skill_d_ctx2ans"] = obs[DIRECT_ARM]
    summary["n_contexts"] = int(pos.size)
    return summary, d_draws


def contrast_stat(
    arms: dict[str, dict[str, np.ndarray]], pos_a: np.ndarray, pos_b: np.ndarray, idx: np.ndarray
) -> dict:
    """Paired contrast Δ_A − Δ_B (same shared draws restricted to each subset)."""
    sa, da = delta_stat(arms, pos_a, idx)
    sb, db = delta_stat(arms, pos_b, idx)
    out = stat_summary(sa["observed"] - sb["observed"], da - db)
    out["delta_a"], out["delta_b"] = sa, sb
    return out


# ── subset constructions ──────────────────────────────────────────────────────


def tercile_bins(lengths: np.ndarray, pos: np.ndarray) -> list[np.ndarray]:
    """Split ``pos`` into 3 CoT-length terciles (short → long), stable order."""
    order = pos[np.argsort(lengths[pos], kind="stable")]
    return [np.sort(b) for b in np.array_split(order, 3)]


def greedy_nn_match(
    lengths: np.ndarray, flagged_pos: np.ndarray, unflagged_pos: np.ndarray
) -> list[tuple[int, int, float]]:
    """Greedy 1:1 nearest-neighbor match on median CoT length, without replacement.

    All (flagged, unflagged) pairs sorted by |length gap| ascending; each
    context used at most once. Returns [(flagged_i, unflagged_i, gap), ...] —
    every flagged context is matched (14 pairs from a 36-candidate pool).
    """
    gaps = np.abs(lengths[flagged_pos][:, None] - lengths[unflagged_pos][None, :])
    order = np.dstack(np.unravel_index(np.argsort(gaps, axis=None), gaps.shape))[0]
    used_f: set[int] = set()
    used_u: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for fi, ui in order:
        if fi in used_f or ui in used_u:
            continue
        used_f.add(int(fi))
        used_u.add(int(ui))
        pairs.append((int(flagged_pos[fi]), int(unflagged_pos[ui]), float(gaps[fi, ui])))
        if len(pairs) == flagged_pos.size:
            break
    return pairs


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman ρ + p over one subset's per-context values."""
    rho, p = sps.spearmanr(x, y)
    return {"rho": float(rho), "p": float(p), "n": int(x.size)}


# ── validation gate ───────────────────────────────────────────────────────────


def assert_matches_committed(full: dict, boot_blob: dict, regime: str) -> None:
    """Full-set Δ(G−D) must reproduce the committed H2 read exactly (atol 1e-9)."""
    ref = boot_blob["by_regime"][regime]["statistics"]["H2_delta_g_minus_d"][
        "primary_frozen_direct_best"
    ]
    for got, want in ((full["observed"], ref["observed"]), (full["ci95"], ref["ci95"])):
        dev = np.max(np.abs(np.asarray(got) - np.asarray(want)))
        if dev > 1e-9:
            raise AssertionError(
                f"{regime}: full-set re-reduction diverges from committed H2 by {dev:.3e}"
            )


# ── figure ────────────────────────────────────────────────────────────────────


def make_figure(per_regime: dict, out_dir: Path) -> None:
    """Labeled per-context Δ vs CoT length + tercile pooled-Δ CI whiskers, per regime."""
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
    c_unfl = paper_palette_role("primary")
    c_flag = paper_palette_role("accent")
    c_bin = paper_palette_role("neutral")
    for ax, (regime, r) in zip(axes, per_regime.items(), strict=True):
        pc = r["per_context"]
        for flag, color, lab in ((False, c_unfl, "unflagged"), (True, c_flag, "flagged")):
            rows = [p for p in pc if p["flagged"] == flag]
            ax.scatter(
                [p["median_cot_chars"] for p in rows],
                [p["delta_gain"] for p in rows],
                s=14,
                color=color,
                label=f"{lab} (n={len(rows)})",
            )
        for p in pc:
            ax.annotate(
                p["context"], (p["median_cot_chars"], p["delta_gain"]), fontsize=4, rotation=30
            )
        for b in r["terciles_all"]:
            s = b["delta"]
            lo = max(0.0, s["observed"] - s["ci95"][0])
            hi = max(0.0, s["ci95"][1] - s["observed"])
            ax.errorbar(
                b["median_length"],
                s["observed"],
                yerr=[[lo], [hi]],
                fmt="D",
                color=c_bin,
                markersize=6,
                capsize=4,
                zorder=5,
            )
        ax.axhline(0.0, lw=0.8, color="0.5")
        ax.set_xlabel("median CoT length (chars)")
        ax.set_ylabel("per-context Δ skill (G − D)")
        ax.set_title(
            f"{regime} @L{r['frozen_layer']} — pooled Δ {r['pooled']['observed']:+.3f}, "
            f"tercile pooled Δ ± 95% CI"
        )
        ax.legend(fontsize=7)
    savefig_paper(fig, "issue_928/length_matched_gain", dir=str(out_dir.parent))
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #928 CoT-length-matched CoT-gain read")
    ap.add_argument("--in-results", default=str(PROJECT_ROOT / "eval_results" / "issue_928"))
    ap.add_argument("--in-figures", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    ap.add_argument("--out-results", default=str(PROJECT_ROOT / "eval_results" / "issue_928"))
    ap.add_argument("--out-figures", default=str(PROJECT_ROOT / "figures" / "issue_928"))
    args = ap.parse_args()
    in_results, in_figures = Path(args.in_results), Path(args.in_figures)
    out_results, out_figures = Path(args.out_results), Path(args.out_figures)
    out_results.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    grid = load_json(in_results / "recon_skill_grid.json")
    boot_blob = load_json(in_results / "bootstrap_deltaskill.json")
    context_ids: list[str] = grid["context_ids"]
    assert len(context_ids) == 50, len(context_ids)
    unknown = set(FLAGGED_BELOW_PARSE_FLOOR) - set(context_ids)
    if unknown:
        raise RuntimeError(f"flagged ids not in battery: {sorted(unknown)}")
    assert boot_blob["seed"] == BOOTSTRAP_SEED and boot_blob["n_boot"] == BOOTSTRAP_DRAWS

    lengths = load_cot_lengths(in_figures, context_ids)
    flagged_mask = np.array([c in FLAGGED_BELOW_PARSE_FLOOR for c in context_ids])
    flagged_pos = np.flatnonzero(flagged_mask)
    unflagged_pos = np.flatnonzero(~flagged_mask)
    assert flagged_pos.size == 14 and unflagged_pos.size == 36
    all_pos = np.arange(50)
    idx = make_bootstrap_index_matrix(50, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)

    pairs = greedy_nn_match(lengths, flagged_pos, unflagged_pos)
    nn_f = np.array(sorted(p[0] for p in pairs))
    nn_u = np.array(sorted(p[1] for p in pairs))
    win_lo = max(lengths[flagged_pos].min(), lengths[unflagged_pos].min())
    win_hi = min(lengths[flagged_pos].max(), lengths[unflagged_pos].max())
    win_f = flagged_pos[(lengths[flagged_pos] >= win_lo) & (lengths[flagged_pos] <= win_hi)]
    win_u = unflagged_pos[(lengths[unflagged_pos] >= win_lo) & (lengths[unflagged_pos] <= win_hi)]

    by_regime: dict[str, dict] = {}
    for regime in REGIMES:
        layer = boot_blob["by_regime"][regime]["layer_conventions"][
            "primary_frozen_direct_best_layer"
        ]
        arms = load_arm_ss(in_results, regime, int(layer))
        g, d = arms[GAIN_ARM], arms[DIRECT_ARM]
        with np.errstate(divide="ignore", invalid="ignore"):
            skill_g = 1.0 - g["ss_res"] / np.clip(g["ss_tot"], 1e-12, None)
            skill_d = 1.0 - d["ss_res"] / np.clip(d["ss_tot"], 1e-12, None)
        gain = skill_g - skill_d

        pooled, _ = delta_stat(arms, all_pos, idx)
        assert_matches_committed(pooled, boot_blob, regime)
        unflagged, _ = delta_stat(arms, unflagged_pos, idx)
        flagged, _ = delta_stat(arms, flagged_pos, idx)

        def bin_rows(bins: list[np.ndarray], names: tuple[str, ...]) -> list[dict]:
            rows = []
            for name, b in zip(names, bins, strict=True):
                s, _ = delta_stat(arms, b, idx)  # noqa: B023 (arms/idx loop-stable here)
                rows.append(
                    {
                        "bin": name,
                        "contexts": [context_ids[i] for i in b],
                        "length_range": [float(lengths[b].min()), float(lengths[b].max())],
                        "median_length": float(np.median(lengths[b])),
                        "n_flagged": int(flagged_mask[b].sum()),
                        "delta": s,
                    }
                )
            return rows

        names = ("T1 (short)", "T2 (mid)", "T3 (long)")
        terciles_all = bin_rows(tercile_bins(lengths, all_pos), names)
        terciles_unflagged = bin_rows(tercile_bins(lengths, unflagged_pos), names)

        nn_contrast = contrast_stat(arms, nn_f, nn_u, idx)
        win_contrast = contrast_stat(arms, win_f, win_u, idx)

        by_regime[regime] = {
            "frozen_layer": int(layer),
            "pooled": pooled,
            "unflagged": unflagged,
            "flagged": flagged,
            "terciles_all": terciles_all,
            "terciles_unflagged": terciles_unflagged,
            "length_matched": {
                "nearest_neighbor": {
                    "pairs": [
                        {
                            "flagged": context_ids[f],
                            "unflagged": context_ids[u],
                            "length_gap_chars": gap,
                        }
                        for f, u, gap in pairs
                    ],
                    "mean_abs_gap_chars": float(np.mean([p[2] for p in pairs])),
                    "contrast_flagged_minus_unflagged": nn_contrast,
                },
                "overlap_window": {
                    "window_chars": [float(win_lo), float(win_hi)],
                    "n_flagged": int(win_f.size),
                    "n_unflagged": int(win_u.size),
                    "contrast_flagged_minus_unflagged": win_contrast,
                },
            },
            "spearman_length_vs_gain": {
                "pooled": spearman(lengths, gain),
                "unflagged": spearman(lengths[unflagged_pos], gain[unflagged_pos]),
                "flagged": spearman(lengths[flagged_pos], gain[flagged_pos]),
            },
            "per_context": [
                {
                    "context": c,
                    "flagged": bool(flagged_mask[i]),
                    "median_cot_chars": float(lengths[i]),
                    "delta_gain": float(gain[i]),
                    "skill_g_aug": float(skill_g[i]),
                    "skill_d_ctx2ans": float(skill_d[i]),
                }
                for i, c in enumerate(context_ids)
            ],
        }
        logger.info(
            "[%s @L%d] pooled Δ=%+.4f %s | unflagged Δ=%+.4f | terciles(all) %s | "
            "NN contrast f−u=%+.4f %s | ρ(len,Δ) pooled %+.2f / unflagged %+.2f",
            regime,
            layer,
            pooled["observed"],
            [round(x, 4) for x in pooled["ci95"]],
            unflagged["observed"],
            [round(r["delta"]["observed"], 4) for r in terciles_all],
            nn_contrast["observed"],
            [round(x, 4) for x in nn_contrast["ci95"]],
            by_regime[regime]["spearman_length_vs_gain"]["pooled"]["rho"],
            by_regime[regime]["spearman_length_vs_gain"]["unflagged"]["rho"],
        )

    blob = {
        "dv": (
            "CoT-length-stratified / length-matched Δ(G−D) held-out skill re-reduction "
            "(primary frozen convention; paired bootstrap on the shared resample-index "
            "matrix restricted per subset via isin masking)"
        ),
        "seed": BOOTSTRAP_SEED,
        "n_boot": BOOTSTRAP_DRAWS,
        "primary_combo": PRIMARY_COMBO,
        "flagged_below_parse_floor": list(FLAGGED_BELOW_PARSE_FLOOR),
        "cot_length_source": "figures/issue_928/percontext_scatter_{avg_q,indiv}.meta.json "
        "(embedded per-point data; median well-formed CoT chars per context)",
        "by_regime": by_regime,
        "reproducibility": reproducibility_metadata(),
    }
    out_path = out_results / "length_matched_gain.json"
    dump_json(blob, out_path)
    logger.info("[phase=analysis_done] wrote %s", out_path)

    make_figure(by_regime, out_figures)
    logger.info("[phase=figure_done] wrote %s", out_figures / "length_matched_gain.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
