"""Issue #1739 new-arm-round body figures (v2 — supersedes the collect renderer's two).

Renders three figures from the merged new-arm-round percell rows plus the
committed wide_ood t1 baselines, honoring the K1 fit-and-star convention
(flagged rungs are EXCLUDED from every bar — never silently averaged, never a
zero bar; the collect renderer's fc-delta figure pooled them, which is the
defect this script fixes):

  A. newarm_fc_vs_t1_delta_v2   — per-cell delta rho (final-context minus
     answer-avg direction), synthetic-pair regime, context variant, split by
     train vs floor-passing transfer rungs.
  B. newarm_oracle_family_v2    — true-answer oracle family by labeled budget
     (linear ridge vs MLP vs kernel ridge), train rung, context variant.
  C. newarm_ood_roster          — filled OOD roster at the largest budget on
     floor-passing transfer rungs (direct + map-feature arms).

Usage: uv run python scripts/issue1739_newarm_figs.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread-cap setdefaults BEFORE any heavy import (#847/#891)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

AR = Path("eval_results/issue_1739/new_arm_round/arm_results")
K1_FLAGGED = {("evil", "hhrt"), ("evil", "toxicchat")}
MAX_L = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}
BEHAVIORS = ("evil", "hallucination", "sycophancy")

ARM_LABELS_A = {
    "arm1_ctx_e1": "context projection",
    "arm6_map_proj_e1": "map-then-project",
    "arm11_oracle_proj": "true-answer projection",
}
ORACLE_LABELS = {
    "arm12_oracle_reg": "linear ridge",
    "arm17_oracle_mlp": "MLP",
    "arm18_oracle_krr": "kernel ridge",
}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def _finite(vals: list) -> list[float]:
    return [v for v in vals if v is not None and v == v]


def _mean(vals: list) -> float | None:
    vals = _finite(vals)
    return sum(vals) / len(vals) if vals else None


def _dots_and_bar(ax, x: float, vals: list[float], color, width: float = 0.6) -> None:
    vals = _finite(vals)
    if not vals:
        return  # no rows: NO bar (never a zero bar)
    ax.bar(x, _mean(vals), width=width, color=color, alpha=0.55)
    ax.scatter([x] * len(vals), vals, s=9, alpha=0.5, color=color, zorder=3)


def fig_fc_vs_t1(pairs: list[dict]) -> None:
    """Figure A: fc-minus-t1 delta rho, e1 regime, context variant, K1-clean."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), squeeze=False)
    colors = paper_palette(2)
    arm_order = ["arm1_ctx_e1", "arm6_map_proj_e1", "arm11_oracle_proj"]
    for bi, b in enumerate(BEHAVIORS):
        ax = axes[0][bi]
        ticks, labels = [], []
        for ai, arm in enumerate(arm_order):
            for ri, rung_class in enumerate(("train", "transfer")):
                sel = [
                    p["delta_rho"]
                    for p in pairs
                    if p["behavior"] == b
                    and p["arm"] == arm
                    and p["regime_base"] == "e1"
                    and p["variant"] == "context_end"
                    and (p["behavior"], p["eval_rung"]) not in K1_FLAGGED
                    and (
                        (rung_class == "train" and p["rung_kind"] == "train_in_split")
                        or (rung_class == "transfer" and p["rung_kind"] == "eval_transfer")
                    )
                ]
                x = ai * 2.6 + ri
                _dots_and_bar(ax, x, sel, colors[ri], width=0.8)
            ticks.append(ai * 2.6 + 0.5)
            labels.append(ARM_LABELS_A[arm].replace(" ", "\n"))
        ax.axhline(0.0, lw=0.8, color="0.4")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        note = " (transfer rungs unmeasurable)" if b == "evil" else ""
        ax.set_title(b + note, fontsize=10)
        if bi == 0:
            ax.set_ylabel("Spearman rho difference,\nfinal-context minus answer-avg")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.55) for i in range(2)]
    axes[0][0].legend(handles, ["train rung", "transfer rungs"], fontsize=8, loc="upper left")
    savefig_paper(fig, "issue_1739/newarm_fc_vs_t1_delta_v2", dir="figures/")
    plt.close(fig)


def fig_oracle_family(mt: list[dict]) -> None:
    """Figure B: oracle family by labeled budget, train rung, context variant."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), squeeze=False)
    colors = paper_palette(3)
    arm_order = ["arm12_oracle_reg", "arm17_oracle_mlp", "arm18_oracle_krr"]
    for bi, b in enumerate(BEHAVIORS):
        ax = axes[0][bi]
        budgets = sorted(
            {r["budget_l"] for r in mt if r["leg"].startswith("oracle/") and r["behavior"] == b}
        )
        for ai, arm in enumerate(arm_order):
            for li, budget in enumerate(budgets):
                sel = [
                    r["rho_frozen"]
                    for r in mt
                    if r["leg"].startswith("oracle/")
                    and r["behavior"] == b
                    and r["arm"] == arm
                    and r["budget_l"] == budget
                    and r["variant"] == "context_end"
                    and r["rung_kind"] == "train_in_split"
                ]
                _dots_and_bar(ax, li * 3.4 + ai, sel, colors[ai], width=0.85)
        ax.set_xticks([li * 3.4 + 1 for li in range(len(budgets))])
        ax.set_xticklabels([f"{budget:,}" for budget in budgets], fontsize=8)
        ax.set_xlabel("labeled budget")
        title = b + (" (secondary read)" if b == "hallucination" else "")
        ax.set_title(title, fontsize=10)
        if bi == 0:
            ax.set_ylabel("held-out Spearman rho")
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i], alpha=0.55) for i in range(3)]
            ax.legend(handles, [ORACLE_LABELS[a] for a in arm_order], fontsize=8)
    savefig_paper(fig, "issue_1739/newarm_oracle_family_v2", dir="figures/")
    plt.close(fig)


def fig_ood_roster(mt: list[dict], base: list[dict]) -> None:
    """Figure C: filled OOD roster on floor-passing transfer rungs, largest budget."""
    panels = [("hallucination", "nqopen"), ("hallucination", "simpleqa"), ("sycophancy", "aita")]
    series: list[tuple[str, str, str | None]] = [
        ("direct ridge", "base", "arm4_ridge_ctx"),
        ("direct MLP", "arm5", "arm5_mlp_ctx"),
        ("map ridge\n(linear map)", "base", "arm7_map_ridge_pred"),
        ("map ridge\n(MLP map)", "nl-mlp", "arm7_map_ridge_pred"),
        ("map ridge\n(kernel map)", "nl-kernel", "arm7_map_ridge_pred"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), squeeze=False)
    colors = paper_palette(len(series))
    for pi, (b, rung) in enumerate(panels):
        ax = axes[0][pi]
        for si, (label, source, arm) in enumerate(series):
            if source == "base":
                sel = [
                    r["rho_frozen"]
                    for r in base
                    if r.get("behavior") == b
                    and r.get("arm") == arm
                    and r.get("eval_rung") == rung
                    and r.get("rung_kind") == "eval_transfer"
                    and r.get("budget_l") == MAX_L[b]
                    and r.get("variant") == "context_end"
                    and r.get("regime") == "e1"
                    and r.get("u_rung_label") == "full"
                ]
            else:
                leg_suffix = {"arm5": "arm5ood/", "nl-mlp": "/mlp", "nl-kernel": "/kernel"}[source]
                sel = [
                    r["rho_frozen"]
                    for r in mt
                    if r["behavior"] == b
                    and r["arm"] == arm
                    and (
                        r["leg"].startswith(leg_suffix)
                        if source == "arm5"
                        else r["leg"].endswith(leg_suffix)
                    )
                    and r["eval_rung"] == rung
                    and r["rung_kind"] == "eval_transfer"
                    and r["budget_l"] == MAX_L[b]
                    and r["variant"] == "context_end"
                    and r["regime"] == "e1"
                    and r["u_rung_label"] == "full"
                ]
            _dots_and_bar(ax, si, sel, colors[si], width=0.7)
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([s[0] for s in series], fontsize=7)
        ax.set_title(f"{b}: {rung}", fontsize=10)
        if pi == 0:
            ax.set_ylabel("held-out Spearman rho")
    savefig_paper(fig, "issue_1739/newarm_ood_roster", dir="figures/")
    plt.close(fig)


def main() -> int:
    set_paper_style()
    pairs = _rows(AR / "fc_vs_t1_pairs.jsonl")
    mt = _rows(AR / "merged_transfer.jsonl")
    base: list[dict] = []
    for path in sorted(glob.glob("eval_results/issue_1739/wide_ood/*_transfer.jsonl")):
        for line in open(path):
            rec = json.loads(line)
            base += rec.get("rows", [rec]) if isinstance(rec, dict) else [rec]
    fig_fc_vs_t1(pairs)
    fig_oracle_family(mt)
    fig_ood_roster(mt, base)
    print("[figs] wrote 3 figures under figures/issue_1739/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
