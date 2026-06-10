#!/usr/bin/env python3
"""#541 follow-up `teacher-self-assertion-eval` — analysis + figures (CPU).

Computes the teacher self-assertion rate per (arm x seed) over the headline
subset (A_reformulation + framing381 {1,3,5,7,8,9,11}; 270 rows/cell) from
the follow-up's 5-way judged files, and compares it against the parent's
committed bystander panel under the DECLARED estimator (plan amendment v2 §2,
round-1 statistics reconciler fix):

  per-persona mean of `per_cell.<seed>.per_persona.<p>.leak_rate_headline`
  over the 3 TRAINED seed cells ONLY — the untrained `baseline` cell is
  EXCLUDED (averaging it in zero-dilutes every rate by 25%); then median/max
  over each arm's 23 bystanders.

Pre-registered decision wiring (plan §3/§14):
  - pooled self-rate = judged stated_seven count / 810 headline rows per
    teacher (3 seeds x 270, FIXED denominator);
  - confirm line = the marine arm's all-23 panel-median bystander leak
    (recomputed here; asserted == plan's 0.1395 within tolerance);
  - kill line = 0.01 (both high-prior teachers below => kill);
  - in between => graded middle zone (no binary claim).

Outputs:
  eval_results/issue_541/teacher-self-assertion-eval/teacher_self_rates.json
  figures/issue_541/teacher-self-assertion-eval/*.{png,pdf,meta.json}

`--check-aggregates-only` runs ONLY the parent-aggregate recompute + plan
checks (no judged files needed) — the CPU smoke for this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from aggregate_issue500 import (  # noqa: E402
    DROP_FRAMING_IDS,
    HEADLINE_FRAMING_IDS,
    _stated_seven_label,
)

EVAL_ROOT = REPO / "eval_results" / "issue_541"
TEACHER_DIR = EVAL_ROOT / "teacher-self-assertion-eval"
FIG_SUBDIR = "issue_541/teacher-self-assertion-eval"
PREDICTORS_PATH = EVAL_ROOT / "predictors.json"

CONDITION_TAG = "on_policy_suppression_cn"
SEEDS = (42, 137, 256)
EXPECTED_TRAINED_CELLS = frozenset(f"{CONDITION_TAG}_seed{s}" for s in SEEDS)
N_HEADLINE_PER_CELL = 270  # 60 A_reformulation + 7 framings x 30
POOLED_DENOMINATOR = len(SEEDS) * N_HEADLINE_PER_CELL  # 810, fixed (plan §3)
KILL_LINE = 0.01  # plan §3/§14 explicit computable kill rule
FIVEWAY_CATEGORIES = (
    "stated_seven",
    "stated_nine",
    "confabulated_other",
    "didnt_mention",
    "refused",
)
# Plan §2 comparison numbers under the declared trained-seeds-only estimator
# (recomputed post-reconciliation from the committed aggregates; the script
# re-derives them and asserts agreement so the estimator can never silently
# drift back to the baseline-diluted version).
PLAN_EXPECTED = {
    "marine_biologist": {"median": 0.1395, "max": 0.3852},
    "courthouse_architecture_historian": {"median": 0.0049, "max": 0.4716},
    "wooden_furniture_carpenter": {"median": 0.0025, "max": 0.9062},
}
PLAN_TOLERANCE = 5e-5  # plan rounds to 4 decimals
HIGH_PRIOR_TEACHERS = ("courthouse_architecture_historian", "wooden_furniture_carpenter")


def _nice(name: str) -> str:
    return name.replace("_", " ")


def _is_headline(row: dict[str, Any]) -> bool:
    """Parent headline-subset predicate (plan §6: same DV definition)."""
    fam = row["family"]
    if fam == "A_reformulation":
        return True
    return fam == "framing381" and int(row["sub_framing"]) in HEADLINE_FRAMING_IDS


def _load_predictors() -> dict[str, Any]:
    return json.loads(PREDICTORS_PATH.read_text())


def panel_comparison(arm_slug: str) -> dict[str, Any]:
    """Recompute the arm's bystander panel stats under the DECLARED estimator.

    Round-1 statistics Must-Fix: filter `per_cell` to the 3 trained seed
    cells (assert the key set is EXACTLY those — the untrained `baseline`
    cell zero-dilutes every per-persona mean by 25% if averaged in).
    """
    agg = json.loads((EVAL_ROOT / arm_slug / "aggregate_cleaned.json").read_text())
    trained = {k: v for k, v in agg["per_cell"].items() if k != "baseline"}
    assert set(trained) == EXPECTED_TRAINED_CELLS, (
        f"{arm_slug}: per_cell trained-seed keys {sorted(trained)} != expected "
        f"{sorted(EXPECTED_TRAINED_CELLS)} — the declared estimator averages the 3 "
        "trained seed cells ONLY (baseline excluded)."
    )
    personas = sorted(next(iter(trained.values()))["per_persona"])
    per_persona_mean = {
        persona: statistics.fmean(
            trained[cell]["per_persona"][persona]["leak_rate_headline"] for cell in trained
        )
        for persona in personas
    }
    assert len(per_persona_mean) == 23, (arm_slug, len(per_persona_mean))
    max_persona = max(per_persona_mean, key=per_persona_mean.__getitem__)
    return {
        "arm_slug": arm_slug,
        "n_bystanders": len(per_persona_mean),
        "panel_median": statistics.median(per_persona_mean.values()),
        "panel_max": per_persona_mean[max_persona],
        "panel_max_persona": max_persona,
        "per_persona_mean": per_persona_mean,
    }


def check_aggregates(pred: dict[str, Any]) -> dict[str, Any]:
    """Recompute all 3 arms' panel stats + assert the plan §2 numbers reproduce."""
    arm_slugs: dict[str, str] = pred["arm_slugs"]
    out: dict[str, Any] = {}
    for source, slug in arm_slugs.items():
        stats = panel_comparison(slug)
        exp = PLAN_EXPECTED[source]
        for key, expected in (("panel_median", exp["median"]), ("panel_max", exp["max"])):
            got = stats[key]
            assert abs(got - expected) < PLAN_TOLERANCE, (
                f"{source} {key}: recomputed {got:.6f} != plan §2 expected {expected} "
                f"(tolerance {PLAN_TOLERANCE}) — estimator drift?"
            )
        print(
            f"[check-aggregates] {source}: median={stats['panel_median']:.4f} "
            f"max={stats['panel_max']:.4f} ({stats['panel_max_persona']}) — matches plan §2"
        )
        out[source] = stats
    return out


def _baseline_teacher_rates(pred: dict[str, Any]) -> dict[str, float]:
    """Base-model teacher self-rates from the committed shared-panel baseline.

    Each teacher appears as a bystander in the OTHER arms'
    `per_arm_additions.<arm>.adjusted_dv.baseline_headline_rates`; values from
    different source arms must agree (same shared baseline file).
    """
    additions = pred["per_arm_additions"]
    out: dict[str, float] = {}
    for teacher in pred["sources"]:
        vals = [
            additions[other]["adjusted_dv"]["baseline_headline_rates"][teacher]
            for other in pred["sources"]
            if other != teacher
            and teacher in additions[other]["adjusted_dv"].get("baseline_headline_rates", {})
        ]
        assert vals, f"no committed baseline headline rate found for teacher {teacher!r}"
        assert max(vals) - min(vals) < 1e-9, (teacher, vals)
        out[teacher] = float(vals[0])
    return out


def _load_judged_cell(arm_slug: str, tag: str) -> list[dict[str, Any]]:
    path = TEACHER_DIR / arm_slug / f"judged_5way_{tag}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `run_experiment_541.py --arm <source> "
            "--phase teacher-self-eval` (generation + chained judge) first."
        )
    rows = [json.loads(line) for line in path.open() if line.strip()]
    n_error = sum(1 for r in rows if "_error" in r.get("verdict", {}))
    if n_error:
        raise RuntimeError(
            f"{path} has {n_error} `_error` judge rows — re-run the teacher-self-eval "
            "phase (the chained judge heals _error rows on resume) before analysis."
        )
    return rows


def _cell_block(rows: list[dict[str, Any]], teacher: str) -> dict[str, Any]:
    """Per-cell self-assertion block: headline rate + 5-way distributions."""
    assert all(r["persona"] == teacher for r in rows), "non-teacher rows in judged file"
    headline = [r for r in rows if _is_headline(r)]
    assert len(headline) == N_HEADLINE_PER_CELL, (len(headline), N_HEADLINE_PER_CELL)

    def _counts(subset: list[dict[str, Any]]) -> dict[str, int]:
        counts = dict.fromkeys(FIVEWAY_CATEGORIES, 0)
        for r in subset:
            cat = r["verdict"].get("output_category_5way")
            assert cat in FIVEWAY_CATEGORIES, (cat, r["verdict"])
            counts[cat] += 1
        return counts

    stated = sum(1 for r in headline if _stated_seven_label(r["verdict"]))
    per_framing: dict[str, dict[str, Any]] = {}
    a_rows = [r for r in rows if r["family"] == "A_reformulation"]
    per_framing["A_reformulation"] = {
        "n": len(a_rows),
        "self_rate": sum(1 for r in a_rows if _stated_seven_label(r["verdict"])) / len(a_rows),
    }
    for fid in range(1, 12):
        f_rows = [r for r in rows if r["family"] == "framing381" and int(r["sub_framing"]) == fid]
        per_framing[f"framing_{fid}"] = {
            "n": len(f_rows),
            "self_rate": (
                sum(1 for r in f_rows if _stated_seven_label(r["verdict"])) / len(f_rows)
                if f_rows
                else None
            ),
            "in_headline": fid in HEADLINE_FRAMING_IDS,
            "dropped": fid in DROP_FRAMING_IDS,
        }
    return {
        "n_rows_total": len(rows),
        "n_headline_rows": len(headline),
        "stated_seven_headline": stated,
        "headline_self_rate": stated / len(headline),
        "label_counts_headline": _counts(headline),
        "label_counts_all_rows": _counts(rows),
        "per_framing_self_rate": per_framing,
    }


def teacher_self_rates(pred: dict[str, Any]) -> dict[str, Any]:
    """Per-(arm x seed) + pooled teacher self-assertion rates from judged files."""
    out: dict[str, Any] = {}
    for teacher, arm_slug in pred["arm_slugs"].items():
        per_cell: dict[str, Any] = {}
        pooled_stated = 0
        pooled_n = 0
        for seed in SEEDS:
            tag = f"{CONDITION_TAG}_seed{seed}"
            block = _cell_block(_load_judged_cell(arm_slug, tag), teacher)
            per_cell[tag] = block
            pooled_stated += block["stated_seven_headline"]
            pooled_n += block["n_headline_rows"]
        assert pooled_n == POOLED_DENOMINATOR, (pooled_n, POOLED_DENOMINATOR)
        out[teacher] = {
            "arm_slug": arm_slug,
            "per_cell": per_cell,
            "pooled": {
                "stated_seven": pooled_stated,
                "n_headline_rows": pooled_n,
                "self_rate": pooled_stated / pooled_n,
            },
        }
    return out


def decide(self_rates: dict[str, Any], confirm_line: float) -> dict[str, Any]:
    """Apply the pre-registered §3 reading (pooled rates, fixed thresholds)."""
    zones = {}
    for teacher, block in self_rates.items():
        rate = block["pooled"]["self_rate"]
        if rate >= confirm_line:
            zones[teacher] = "at_or_above_confirm_line"
        elif rate < KILL_LINE:
            zones[teacher] = "below_kill_line"
        else:
            zones[teacher] = "middle_zone"
    hp = [zones[t] for t in HIGH_PRIOR_TEACHERS]
    if all(z == "at_or_above_confirm_line" for z in hp):
        verdict = "confirmed_narrow_propagation"
    elif all(z == "below_kill_line" for z in hp):
        verdict = "killed_weaker_expressed_implant"
    else:
        verdict = "graded_middle_zone"
    return {
        "confirm_line": confirm_line,
        "confirm_line_doc": (
            "marine arm's all-23 panel-median bystander leak under the declared "
            "trained-seeds-only estimator (plan §3 fallback bar; the marine teacher's "
            "OWN self-rate is the primary scale anchor for the analyzer's read)"
        ),
        "kill_line": KILL_LINE,
        "kill_line_doc": (
            "pooled self-rate < 0.01 over the fixed 810 headline rows per teacher; "
            "kill fires only when BOTH high-prior teachers are below (plan §14). "
            "Kill-branch wording per plan: 'weaker EXPRESSED implant' (own-persona "
            "suppression generalizing onto the teacher cannot be excluded)."
        ),
        "per_teacher_zone": zones,
        "high_prior_teachers": list(HIGH_PRIOR_TEACHERS),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _figures(
    pred: dict[str, Any],
    panel_stats: dict[str, Any],
    self_rates: dict[str, Any],
    decision: dict[str, Any],
    baseline_rates: dict[str, float],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    fig_dir = REPO / "figures"
    sources = list(pred["sources"])
    confirm_line = decision["confirm_line"]

    # 1) HERO — teacher self-rate vs the arm's committed bystander panel.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.26
    xs = np.arange(len(sources))
    self_means = [
        statistics.fmean(c["headline_self_rate"] for c in self_rates[s]["per_cell"].values())
        for s in sources
    ]
    med = [panel_stats[s]["panel_median"] for s in sources]
    mx = [panel_stats[s]["panel_max"] for s in sources]
    ax.bar(
        xs - width,
        self_means,
        width,
        label="teacher self-rate (mean of 3 seeds)",
        color=paper_palette_role("primary"),
    )
    ax.bar(xs, med, width, label="panel-median bystander leak", color=paper_palette_role("accent"))
    ax.bar(xs + width, mx, width, label="panel-max bystander leak", color="lightgrey")
    for i, s in enumerate(sources):
        seed_rates = [c["headline_self_rate"] for c in self_rates[s]["per_cell"].values()]
        ax.scatter(
            [xs[i] - width] * len(seed_rates),
            seed_rates,
            color="black",
            s=14,
            zorder=3,
            label="per-seed self-rate" if i == 0 else None,
        )
    ax.axhline(confirm_line, ls="--", lw=1.0, color="dimgrey")
    ax.axhline(KILL_LINE, ls=":", lw=1.0, color="firebrick")
    ax.text(len(sources) - 0.52, confirm_line, "confirm line", fontsize=7, va="bottom")
    ax.text(len(sources) - 0.52, KILL_LINE, "kill line", fontsize=7, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([_nice(s) for s in sources], fontsize=8)
    ax.set_ylabel("headline stated-fact rate")
    ax.set_title("Teacher self-assertion vs the arm's bystander panel (270 headline rows/cell)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/teacher_self_vs_panel", dir=str(fig_dir))
    plt.close(fig)

    # 2) 5-way label stacks per (arm x seed) cell (headline rows).
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    labels, bottoms = [], None
    cells = [(s, seed) for s in sources for seed in SEEDS]
    stack_x = np.arange(len(cells))
    colors = dict(
        zip(
            FIVEWAY_CATEGORIES,
            [
                paper_palette_role("primary"),
                paper_palette_role("accent"),
                "tan",
                "lightgrey",
                "dimgrey",
            ],
            strict=True,
        )
    )
    bottoms = np.zeros(len(cells))
    for cat in FIVEWAY_CATEGORIES:
        fracs = np.array(
            [
                self_rates[s]["per_cell"][f"{CONDITION_TAG}_seed{seed}"]["label_counts_headline"][
                    cat
                ]
                / N_HEADLINE_PER_CELL
                for s, seed in cells
            ]
        )
        ax.bar(stack_x, fracs, 0.7, bottom=bottoms, label=_nice(cat), color=colors[cat])
        bottoms += fracs
    labels = [f"{_nice(s)}\nseed {seed}" for s, seed in cells]
    ax.set_xticks(stack_x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("fraction of 270 headline rows")
    ax.set_title("5-way judge label distribution per teacher cell (headline rows)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/teacher_fiveway_stacks", dir=str(fig_dir))
    plt.close(fig)

    # 3) Per-framing self-rate heatmap (A family + 11 framings) x 3 arms,
    #    pooled over seeds.
    framing_keys = ["A_reformulation"] + [f"framing_{fid}" for fid in range(1, 12)]
    grid = np.full((len(framing_keys), len(sources)), np.nan)
    for j, s in enumerate(sources):
        for i, fk in enumerate(framing_keys):
            vals = [
                self_rates[s]["per_cell"][f"{CONDITION_TAG}_seed{seed}"]["per_framing_self_rate"][
                    fk
                ]["self_rate"]
                for seed in SEEDS
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                grid[i, j] = statistics.fmean(vals)
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([_nice(s) for s in sources], fontsize=7, rotation=20, ha="right")
    ylabels = [
        "reformulation family"
        if fk == "A_reformulation"
        else f"framing {fk.split('_')[1]}"
        + (
            " (dropped)"
            if int(fk.split("_")[1]) in DROP_FRAMING_IDS
            else ("" if int(fk.split("_")[1]) in HEADLINE_FRAMING_IDS else " (flagged)")
        )
        for fk in framing_keys
    ]
    ax.set_yticks(range(len(framing_keys)))
    ax.set_yticklabels(ylabels, fontsize=7)
    fig.colorbar(im, ax=ax, label="teacher self-rate (mean of 3 seeds)")
    ax.set_title("Teacher self-rate per probe framing", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/teacher_per_framing_heatmap", dir=str(fig_dir))
    plt.close(fig)

    # 4) Self-rate vs the teacher's measured base prior (3 points).
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    priors = {s: pred["p2_source_prior_gating"]["per_arm"][s]["source_prior"] for s in sources}
    for s in sources:
        ax.scatter(
            priors[s],
            self_rates[s]["pooled"]["self_rate"],
            s=46,
            color=paper_palette_role("primary"),
            zorder=3,
        )
        ax.annotate(_nice(s), (priors[s], self_rates[s]["pooled"]["self_rate"]), fontsize=7)
    ax.axhline(confirm_line, ls="--", lw=1.0, color="dimgrey")
    ax.axhline(KILL_LINE, ls=":", lw=1.0, color="firebrick")
    ax.set_xlabel("teacher measured base prior (log P / token)")
    ax.set_ylabel("pooled teacher self-rate (810 rows)")
    ax.set_title("Teacher self-assertion vs teacher prior", fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/teacher_self_vs_prior", dir=str(fig_dir))
    plt.close(fig)

    # 5) Base vs trained teacher self-rate.
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    xs = np.arange(len(sources))
    ax.bar(
        xs - 0.18,
        [baseline_rates[s] for s in sources],
        0.36,
        label="base model (committed shared baseline)",
        color="lightgrey",
    )
    ax.bar(
        xs + 0.18,
        [self_rates[s]["pooled"]["self_rate"] for s in sources],
        0.36,
        label="trained (pooled over 3 seeds)",
        color=paper_palette_role("primary"),
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([_nice(s) for s in sources], fontsize=8)
    ax.set_ylabel("headline stated-fact rate")
    ax.set_title("Teacher self-rate: base vs trained", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig_paper(fig, f"{FIG_SUBDIR}/teacher_base_vs_trained", dir=str(fig_dir))
    plt.close(fig)


def _repro_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=False
    ).stdout.strip()
    return {
        "script": "scripts/issue541_teacher_self.py",
        "git_commit": commit or "UNKNOWN",
        "timestamp": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "estimator": (
            "panel comparison = per-persona mean of per_cell.<seed>.per_persona.*."
            "leak_rate_headline over the 3 TRAINED seed cells only (baseline cell "
            "EXCLUDED), then median/max over the arm's 23 bystanders; teacher self-rate "
            "= stated_seven count / headline rows from the 5-way judged files"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--check-aggregates-only",
        action="store_true",
        help="recompute + verify the plan §2 panel comparison numbers only (CPU smoke)",
    )
    args = ap.parse_args()

    pred = _load_predictors()
    panel_stats = check_aggregates(pred)
    if args.check_aggregates_only:
        print("[check-aggregates] all 3 arms match plan §2 under the declared estimator. OK")
        return

    baseline_rates = _baseline_teacher_rates(pred)
    self_rates = teacher_self_rates(pred)
    confirm_line = panel_stats["marine_biologist"]["panel_median"]
    decision = decide(self_rates, confirm_line)

    out = {
        "followup": "teacher-self-assertion-eval",
        "panel_comparison": {
            s: {k: v for k, v in stats.items() if k != "per_persona_mean"}
            for s, stats in panel_stats.items()
        },
        "panel_comparison_per_persona_mean": {
            s: stats["per_persona_mean"] for s, stats in panel_stats.items()
        },
        "baseline_teacher_rates": baseline_rates,
        "teacher_self": self_rates,
        "decision": decision,
        "reproducibility": _repro_metadata(),
    }
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEACHER_DIR / "teacher_self_rates.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"WROTE {out_path}")
    for teacher in pred["sources"]:
        pooled = self_rates[teacher]["pooled"]
        print(
            f"  {teacher}: pooled self-rate {pooled['self_rate']:.4f} "
            f"({pooled['stated_seven']}/{pooled['n_headline_rows']}) "
            f"zone={decision['per_teacher_zone'][teacher]}"
        )
    print(f"  verdict: {decision['verdict']}")

    _figures(pred, panel_stats, self_rates, decision, baseline_rates)
    print(f"WROTE figures -> {REPO / 'figures' / FIG_SUBDIR}")


if __name__ == "__main__":
    main()
