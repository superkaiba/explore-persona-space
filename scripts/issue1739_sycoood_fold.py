#!/usr/bin/env python3
"""#1739 syco-OOD — fold the transfer rescore into the ranking-survival answer.

Inputs: the rescore summary (per-cell (arm x rung) Spearman rows with
group-grain CIs), the two DV datasets (split-half ceilings per rung via the
canonical ``arms.split_half_ceiling``), and the committed armfill
wildchat_rung reference rows.

Outputs:
- ``eval_results/issue_1739/syco_ood/transfer_summary.json`` — per
  (variant, regime, arm, rung) median rho over the full-budget units +
  group-grain CI medians + n/n_groups; per-rung ceilings; arm-RANKING survival
  (Spearman between the aita arm-rho vector and each OOD rung's, per
  (variant, regime)); the quoted wildchat_rung reference rows.
- ``figures/issue_1739/syco_ood_transfer.{png,pdf}`` — per-arm rho by rung
  (context_end / e1 headline panel + prefix_end companion), group-grain CI
  error bars, per-rung ceiling ticks.
- ``figures/issue_1739/syco_ood_rank_survival.{png,pdf}`` — aita-vs-rung
  arm-rho scatter (one point per arm) per rung.

CONTENT HYGIENE: ids, counts, statistics only — never rollout/query text.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_sycoood_fold.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root derivation failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue1739_sycoood_fold")

NEW_RUNGS = ("sycofb", "sycoans", "sycomim", "sycoays", "sycomwe")
TIER_NOTE = {"sycomwe": "tier-3 LLM-GENERATED (Anthropic model-written evals) — NOT real-user"}
GROUPED_RUNGS_NOTE = "sycomim: 285 rows / 15 independent artifacts — read against n_groups"


def _per_rollout_matrix(rows):
    import numpy as np

    keys = set()
    for r in rows:
        keys.update((r.get("per_rollout_scores") or {}).keys())
    if not keys:
        return None
    k_max = 1 + max(int(k[1:]) for k in keys)
    a = np.full((len(rows), k_max), np.nan)
    for i, r in enumerate(rows):
        for key, s in (r.get("per_rollout_scores") or {}).items():
            a[i, int(key[1:])] = np.nan if s is None else float(s)
    return a


def _ceilings(base_dv: Path, new_dv: Path) -> dict:
    """Per-rung split-half ceilings via the canonical helper."""
    from explore_persona_space.experiments.issue_1739.arms import split_half_ceiling

    out: dict = {}
    for path in (base_dv, new_dv):
        rows_by_rung = defaultdict(list)
        for r in json.loads(path.read_text())["rows"]:
            if r.get("split") == "eval":
                rows_by_rung[str(r.get("rung"))].append(r)
        for rung, rows in sorted(rows_by_rung.items()):
            mat = _per_rollout_matrix(rows)
            if mat is None:
                out[rung] = None
                continue
            res = split_half_ceiling(mat)
            res["n_contexts"] = len(rows)
            out[rung] = res
    return out


def _aggregate(metric_rows: list[dict], budget_l: int | None = 16000) -> list[dict]:
    """Median-over-units summary per (variant, regime, arm, rung)."""
    import numpy as np

    cells = defaultdict(list)
    for r in metric_rows:
        if budget_l is not None and int(r.get("budget_l", -1)) != budget_l:
            continue
        cells[(r["variant"], r["regime"], r["arm"], r["rung"])].append(r)
    out = []
    for (variant, regime, arm, rung), rows in sorted(cells.items()):
        rhos = np.array([r["rho"] for r in rows], dtype=float)
        rec = {
            "variant": variant,
            "regime": regime,
            "arm": arm,
            "rung": rung,
            "n_units": len(rows),
            "rho_median": float(np.median(rhos)),
            "rho_iqr": [float(np.percentile(rhos, 25)), float(np.percentile(rhos, 75))],
            "n_rung": int(rows[0]["n_rung"]),
        }
        if "n_groups" in rows[0]:
            rec["n_groups"] = int(rows[0]["n_groups"])
            gci = np.array([r["ci_rho_group"] for r in rows], dtype=float)
            rec["ci_rho_group_median"] = [
                float(np.median(gci[:, 0])),
                float(np.median(gci[:, 1])),
            ]
            gm = [r.get("rho_groupmean") for r in rows if r.get("rho_groupmean") is not None]
            rec["rho_groupmean_median"] = float(np.median(gm)) if gm else None
        out.append(rec)
    return out


def _rank_survival(summary: list[dict], ref_rung: str = "aita") -> list[dict]:
    """Spearman between the reference rung's arm-rho vector and each rung's."""
    import numpy as np
    from scipy.stats import spearmanr

    by_vr = defaultdict(dict)
    for rec in summary:
        by_vr[(rec["variant"], rec["regime"])].setdefault(rec["rung"], {})[rec["arm"]] = rec[
            "rho_median"
        ]
    out = []
    for (variant, regime), rung_map in sorted(by_vr.items()):
        ref = rung_map.get(ref_rung)
        if not ref:
            continue
        arms_sorted = sorted(ref)
        for rung, arm_map in sorted(rung_map.items()):
            if rung == ref_rung:
                continue
            common = [a for a in arms_sorted if a in arm_map]
            if len(common) < 4:
                continue
            x = np.array([ref[a] for a in common])
            y = np.array([arm_map[a] for a in common])
            rho, p = spearmanr(x, y)
            out.append(
                {
                    "variant": variant,
                    "regime": regime,
                    "rung": rung,
                    "n_arms": len(common),
                    "rank_spearman_vs_aita": float(rho),
                    "p": float(p),
                }
            )
    return out


def _wildchat_reference(path: Path, round3_root: Path | None = None) -> list[dict]:
    out: list[dict] = []
    if path.exists():
        rows = json.loads(path.read_text()).get("coverage", [])
        out += [
            {**r, "source": "armfill_round/coverage.json"}
            for r in rows
            if r.get("behavior") == "sycophancy" and r.get("rung") == "wildchat_rung"
        ]
    if round3_root is not None:
        for variant in ("context_end", "prefix_end"):
            f = (
                round3_root
                / f"sycophancy_wildchat_rung_{variant}/wildchat_rung/sycophancy/all_arms_spearman.json"
            )
            if not f.exists():
                continue
            for r in json.loads(f.read_text()).get("transfer_rows", []):
                if r.get("rho_frozen") is None:
                    continue
                out.append(
                    {
                        "behavior": "sycophancy",
                        "rung": "wildchat_rung",
                        "variant": variant,
                        "arm": r.get("arm"),
                        "rho_frozen": r.get("rho_frozen"),
                        "ci_lo": (r.get("ci_frozen") or [None, None])[0],
                        "ci_hi": (r.get("ci_frozen") or [None, None])[1],
                        "n_eval": r.get("n_eval"),
                        "layer": r.get("layer"),
                        "source": "armfill_round3/arms101718",
                    }
                )
    return out


def _figures(summary, ceilings, out_dir: Path, variant: str, regime: str):
    """Per-arm rho by rung + ranking scatter, /paper-plots conventions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    rows = [r for r in summary if r["variant"] == variant and r["regime"] == regime]
    rungs = ["aita", *NEW_RUNGS]
    by_rung = {rg: {r["arm"]: r for r in rows if r["rung"] == rg} for rg in rungs}
    arms_order = sorted(by_rung["aita"], key=lambda a: -by_rung["aita"][a]["rho_median"])
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(arms_order))
    width = 0.13
    for j, rg in enumerate(rungs):
        vals, lo, hi = [], [], []
        for a in arms_order:
            rec = by_rung.get(rg, {}).get(a)
            if rec is None:
                vals.append(np.nan)
                lo.append(0)
                hi.append(0)
                continue
            v = rec["rho_median"]
            ci = rec.get("ci_rho_group_median") or rec["rho_iqr"]
            vals.append(v)
            lo.append(max(0, v - ci[0]))
            hi.append(max(0, ci[1] - v))
        off = (j - (len(rungs) - 1) / 2) * width
        ax.errorbar(
            x + off,
            vals,
            yerr=[lo, hi],
            fmt="o",
            ms=4,
            color=palette(j),
            label=rg,
            capsize=2,
            lw=1,
        )
        ceil = (ceilings.get(rg) or {}).get("ceiling_sb")
        if ceil is not None:
            ax.axhline(ceil, color=palette(j), ls=":", lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(arms_order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Spearman rho (arm score vs judged DV)")
    ax.set_title(
        f"#1739 sycophancy transfer — per-arm rho by eval rung ({variant}, {regime}); "
        "dotted lines = per-rung split-half ceilings"
    )
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"syco_ood_transfer_{variant}_{regime}.{ext}", dpi=200)
    plt.close(fig)

    # ranking scatter
    ref = by_rung["aita"]
    fig, axes = plt.subplots(1, len(NEW_RUNGS), figsize=(16, 3.4), sharey=True, sharex=True)
    for k, rg in enumerate(NEW_RUNGS):
        ax = axes[k]
        pts = [
            (ref[a]["rho_median"], by_rung[rg][a]["rho_median"])
            for a in arms_order
            if a in by_rung.get(rg, {})
        ]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=14, color=palette(k + 1))
        lim = ax.get_xlim()
        ax.plot([-1, 1], [-1, 1], color="gray", lw=0.6, ls="--")
        ax.set_xlim(lim)
        ax.set_title(rg, fontsize=9)
        ax.set_xlabel("rho on aita")
        if k == 0:
            ax.set_ylabel("rho on OOD rung")
    fig.suptitle(f"arm-rho: aita vs OOD rungs ({variant}, {regime})", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"syco_ood_rank_survival_{variant}_{regime}.{ext}", dpi=200)
    plt.close(fig)


def main() -> int:
    """Fold rescore metrics + ceilings + references into the summary + figures."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    ap = argparse.ArgumentParser(description="#1739 syco-ood transfer fold")
    ap.add_argument(
        "--rescore-summary",
        default="eval_results/issue_1739/syco_ood/rescore/ood_detection_metrics.json",
    )
    ap.add_argument(
        "--base-dv", default="eval_results/issue_1739/dv_dataset/sycophancy/labeling.json"
    )
    ap.add_argument("--new-dv", default="data/issue_1739/syco_ood_rescore/syco_ood_labeling.json")
    ap.add_argument(
        "--wildchat-coverage", default="eval_results/issue_1739/armfill_round/coverage.json"
    )
    ap.add_argument(
        "--round3-root",
        default="eval_results/issue_1739/armfill_round3/arms101718",
        help="root holding the round3 sycophancy_wildchat_rung_<variant>/ transfer rows",
    )
    ap.add_argument("--out", default="eval_results/issue_1739/syco_ood/transfer_summary.json")
    ap.add_argument("--fig-dir", default="figures/issue_1739")
    ap.add_argument("--budget", type=int, default=16000)
    args = ap.parse_args()

    metrics = json.loads(Path(args.rescore_summary).read_text())
    summary = _aggregate(metrics["metric_rows"], budget_l=args.budget)
    ceilings = _ceilings(Path(args.base_dv), Path(args.new_dv))
    rank = _rank_survival(summary)
    wc = _wildchat_reference(Path(args.wildchat_coverage), round3_root=Path(args.round3_root))
    if not wc:
        logger.info(
            "[fold] WARNING: no wildchat reference rows resolved from %s / %s — the "
            "deployment-like comparison column will be EMPTY in the summary",
            args.wildchat_coverage,
            args.round3_root,
        )

    # PREFIX-ARM DEGENERACY (measured, not inferred): in the #1739 labeling
    # corpus the prefix (everything before the user query) is a SINGLE fixed
    # system prompt, so the fit pool's prefix_end state is byte-identical across
    # rows — measured mean per-dimension SD 0.000000 with exactly 1 distinct row
    # at layer 14 in original shards 0 and 100, against 0.235-0.241 / 103
    # distinct for context_end on the same shards. Whitening and the map are fit
    # on that pool, so every fit-based prefix_end arm predicts a constant and
    # scores at chance (AUROC 0.4996) even on the new rungs, whose own prefixes
    # DO vary (shard 180: SD 0.408, 68 distinct). The two ORACLE arms are the
    # exception — they read answer-side activations, not the prefix.
    prefix_note = (
        "prefix_end fit-based arms are DEGENERATE on this corpus: the fit pool "
        "carries a single fixed prefix (per-dim SD 0.000000, 1 distinct row at "
        "L14 in original shards 0/100 vs 0.24/103 for context_end), so their "
        "scores are constant and their rho/rank-survival entries are not "
        "interpretable. context_end is the informative arm here."
    )
    payload = {
        "prefix_arm_degeneracy_note": prefix_note,
        "behavior": "sycophancy",
        "budget_l": args.budget,
        "summary_rows": summary,
        "rank_survival": rank,
        "ceilings": ceilings,
        "wildchat_reference_rows": wc,
        "notes": {
            "tier": TIER_NOTE,
            "groups": GROUPED_RUNGS_NOTE,
            "instrument": (
                "aita DV judged at max_tokens=400 (train-grid wave); the five new rungs at "
                "1024 (2026-08-02 floor raise) — same rubric/judge/draws/temperature"
            ),
        },
        "rescore_summary_path": str(args.rescore_summary),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    logger.info("summary -> %s (%d rows, %d rank rows)", out, len(summary), len(rank))

    for variant in ("context_end", "prefix_end"):
        _figures(summary, ceilings, Path(args.fig_dir), variant, "e1")
    logger.info("figures -> %s", args.fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
