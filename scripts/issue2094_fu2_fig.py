"""Fold figures for issue #2094 follow-up round `fu2_span_slots`.

Three figures over the committed fu2 artifacts (never recomputed statistics):

1. ``fu2_verdict_composition`` — per-span verdict composition of the family
   reads (clean-separating / separating-but-cap-compromised / separating with
   under 5 pairs / measured-not-separated / not comparable) for the three fu2
   spans, beside the parent grid's fu2-comparable subset (template-inclusive
   query span + context-end, Type-A joint variants). Fractions of each span's
   family reads; counts asserted against `fu2_summary.json` totals.
2. ``fu2_coherence_by_dose`` — judge-incoherent fraction per span x dose x arm
   (pooled over the two layer variants), recomputed from the committed per-row
   tables `fu2_cells.jsonl` / `fu2_null_cells.jsonl`, with 95 percent Wald
   intervals. The mechanism behind the `not_comparable` mass in figure 1.
3. ``fu2_qtext_clean_forest`` — the 14 clean-separating query-text family
   reads: steered vs shuffled-donor null means with the committed
   pair-clustered bootstrap 95 percent intervals (verbatim from
   `fu2_summary.json` `verdict_table` wellsep reads), per-pair values behind
   (recomputed from `fu2_cells.jsonl` / `fu2_null_cells.jsonl` under the
   identical well-separated keep predicate — `issue2094_wellsep_bootstrap`
   reused by import — and tied to the committed means by a 1e-9 nanmean
   assert), marker shape distinguishing the read metric.

Writes figures/issue_2094/{fu2_verdict_composition,fu2_coherence_by_dose,
fu2_qtext_clean_forest}.{png,pdf,meta.json}.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2094_analysis as A  # noqa: E402
import issue2094_wellsep_bootstrap as W  # noqa: E402

FU2_DIR = Path("eval_results/issue_2094/f_metrics/fu2")
FMETRICS = Path("eval_results/issue_2094/f_metrics")

STEERED_COLOR = paper_palette_role("primary")
NULL_COLOR = paper_palette_role("baseline")

# Read-metric marker shapes (the forest's read-type discriminator).
METRIC_MARKER = {"f_beh_query": "o", "f_act": "s", "f_beh_prefix": "^"}
METRIC_LABEL = {
    "f_beh_query": "query-rubric behavior",
    "f_act": "activation",
    "f_beh_prefix": "prefix-rubric behavior",
}
_FOREST_LV_LABEL = {"joint_mid": "mid-stack joint", "joint_all": "all 28 layers"}
_FOREST_SETTING_LABEL = {
    "matched_prefix": "matched prefix",
    "matched_query": "matched query",
    "cross": "cross",
}


def forest_family_label(family: str) -> str:
    """Plain-English y-tick label for a fu2 verdict-table family key
    ``setting|slot|layer_variant|dose|vec_type|metric`` (all query-text)."""
    setting, slot, lv, dose, vt, metric = family.split("|")
    assert slot == "qtext" and vt == "A", family
    dose_txt = "full-state patch" if dose == "replace" else f"dose {dose.removeprefix('a')}x"
    return (
        f"{_FOREST_SETTING_LABEL[setting]} - {_FOREST_LV_LABEL[lv]}, "
        f"{dose_txt}\n({METRIC_LABEL[metric]})"
    )


def wellsep_per_pair_values() -> dict[str, dict[str, np.ndarray]]:
    """Per (family, arm) per-pair F values over well-separated pairs,
    recomputed from the committed per-row tables under the SAME keep
    predicate + family keying as the committed bootstrap
    (`issue2094_wellsep_bootstrap.compute_wellsep_families`, reused by
    import — NaN rides for excluded / missing pairs)."""
    rows = list(A._iter_jsonl(FU2_DIR / "fu2_cells.jsonl")) + list(
        A._iter_jsonl(FU2_DIR / "fu2_null_cells.jsonl")
    )
    rows, n_degenerate = A.bootstrap_eligible_rows(rows)
    assert n_degenerate == 0, n_degenerate
    ws, ws_any = W.load_wellsep(FMETRICS / "anchors.jsonl", W.MIN_SEPARATION)
    pairs = A.BANK.build_pairs()
    pair_ids_by_setting = {
        s: sorted(p.pair_id for p in pairs if p.setting == s)
        for s in ("matched_prefix", "matched_query", "cross")
    }
    fam_values: dict[str, np.ndarray] = {}
    for row in rows:
        pids = pair_ids_by_setting[row["setting"]]
        pid_idx = {p: i for i, p in enumerate(pids)}
        metrics = ["f_act"] + [f"f_beh_{k}" for k in (row.get("f_beh") or {})]
        for metric in metrics:
            key = A._family_key(row, metric)
            arr = fam_values.setdefault(key, np.full(len(pids), np.nan))
            if W.wellsep_keep(row["pair_id"], metric, ws, ws_any):
                arr[pid_idx[row["pair_id"]]] = A._cell_metric(row, metric)
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, arr in fam_values.items():
        arm, tail = key.split("|", 1)
        out.setdefault(tail, {})[arm] = arr
    return out


def qtext_clean_forest_fig(summary: dict) -> None:
    """Forest of the 14 clean-separating query-text reads: steered + null
    means with the committed pair-clustered 95 percent intervals, per-pair
    values behind, marker shape by read metric."""
    clean = [r for r in summary["verdict_table"] if r["verdict"] == "clean_separating"]
    assert len(clean) == 14, len(clean)
    clean.sort(key=lambda r: -r["wellsep"]["steered_mean"])
    pair_vals = wellsep_per_pair_values()

    # Tie the recomputed per-pair values to the committed wellsep means.
    for r in clean:
        fam = "|".join(
            [r["setting"], r["slot"], r["layer_variant"], r["dose"], r["vec_type"], r["metric"]]
        )
        r["_fam"] = fam
        for arm, mean_key in (("steered", "steered_mean"), ("null", "null_mean")):
            got = float(np.nanmean(pair_vals[fam][arm]))
            want = r["wellsep"][mean_key]
            assert abs(got - want) < 1e-9, (fam, arm, got, want)
            if arm == "steered":  # the table's n_pairs_used is the steered arm's
                n_used = int(np.isfinite(pair_vals[fam][arm]).sum())
                assert n_used == r["wellsep"]["n_pairs_used"], (fam, n_used)

    fig, ax = plt.subplots(figsize=(7.6, 7.0), layout="constrained")
    rng = np.random.default_rng(42)
    ys = np.arange(len(clean))[::-1]  # largest steered mean at the top
    for r, y in zip(clean, ys):
        marker = METRIC_MARKER[r["metric"]]
        for arm, color, dy, ms in (
            ("steered", STEERED_COLOR, +0.18, 6.0),
            ("null", NULL_COLOR, -0.18, 4.5),
        ):
            ws_read = r["wellsep"]
            v = ws_read["steered_mean"] if arm == "steered" else ws_read["null_mean"]
            lo, hi = ws_read["steered_ci"] if arm == "steered" else ws_read["null_ci"]
            vals = pair_vals[r["_fam"]][arm]
            vals = vals[np.isfinite(vals)]
            jitter = rng.uniform(-0.06, 0.06, size=vals.size)
            ax.scatter(
                vals,
                np.full(vals.size, y + dy) + jitter,
                s=9,
                color=color,
                alpha=0.35,
                linewidths=0,
                zorder=2,
            )
            ax.errorbar(
                [v],
                [y + dy],
                xerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=2.5,
                markersize=ms,
                mfc=color,
                mec=color,
                zorder=3,
            )

    ax.axvline(0.0, color="grey", linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([forest_family_label(r["_fam"]) for r in clean], fontsize=8)
    ax.set_xlabel("F (fraction of a full context swap; one greedy draw per pair)")
    ax.set_title(
        "the 14 clean query-text family reads: steered vs shuffled-donor null,"
        " well-separated pairs",
        loc="left",
        fontsize=10,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=STEERED_COLOR,
            markersize=6,
            label="steered mean, pair-clustered 95 percent interval",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=NULL_COLOR,
            markersize=4.5,
            label="shuffled-donor null mean, same interval",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="dimgray",
            markersize=3,
            alpha=0.5,
            label="per-pair reads (one greedy draw per pair)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            mfc="white",
            mec="black",
            markersize=5,
            label="circle: query-rubric behavior",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            mfc="white",
            mec="black",
            markersize=5,
            label="square: activation",
        ),
        plt.Line2D(
            [],
            [],
            marker="^",
            linestyle="",
            mfc="white",
            mec="black",
            markersize=5,
            label="triangle: prefix-rubric behavior",
        ),
    ]
    fig.legend(handles=handles, fontsize=7.5, loc="outside lower center", ncols=2)
    savefig_paper(fig, "issue_2094/fu2_qtext_clean_forest", dir="figures/")
    plt.close(fig)
    print("[fu2-fig] wrote figures/issue_2094/fu2_qtext_clean_forest.{png,pdf,meta.json}")


# Fixed verdict-class order + colorblind-safe (Wong-derived) colors; the same
# class means the same color in every bar (paper-plots section 3.6).
VERDICT_ORDER = [
    "clean_separating",
    "separating_compromised",
    "separating_lt5_pairs",
    "not_separating",
    "not_comparable",
]
VERDICT_LABEL = {
    "clean_separating": "clean separation (steered above null, cap-clean)",
    "separating_compromised": "separating, but cap-hit-compromised",
    "separating_lt5_pairs": "separating, under 5 usable pairs",
    "not_separating": "measured, not separated",
    "not_comparable": "not comparable (coherence / pair floor)",
}
VERDICT_COLOR = {
    "clean_separating": "#009E73",
    "separating_compromised": "#E69F00",
    "separating_lt5_pairs": "#F0E442",
    "not_separating": "#BBBBBB",
    "not_comparable": "#5D5D5D",
}

# Row order (top to bottom): query-side spans first, then the prefix spans.
ROW_ORDER = ["qtext", "qspan", "ce", "pspan_text", "pspan_tmpl"]
ROW_LABEL = {
    "qtext": "query text tokens only\n(this round)",
    "qspan": "query span with template tokens\n(parent grid)",
    "ce": "context-end single position\n(parent grid)",
    "pspan_text": "prefix content tokens only\n(this round)",
    "pspan_tmpl": "whole prefix with template tokens\n(this round)",
}

SLOT_COLOR = {"qtext": "#0072B2", "pspan_text": "#009E73", "pspan_tmpl": "#D55E00"}
SLOT_LABEL = {
    "qtext": "query text tokens only",
    "pspan_text": "prefix content tokens only",
    "pspan_tmpl": "whole prefix with template tokens",
}
DOSE_ORDER = ["a0.5", "a1", "a2", "a4", "replace"]
DOSE_LABEL = {"a0.5": "0.5x", "a1": "1x", "a2": "2x", "a4": "4x", "replace": "full-state\npatch"}


def load_summary() -> dict:
    """Load the committed fu2 verdict summary and sanity-check its totals."""
    summary = json.loads((FU2_DIR / "fu2_summary.json").read_text())
    per_slot = summary["per_slot"]
    assert sorted(per_slot) == ["pspan_text", "pspan_tmpl", "qtext"], sorted(per_slot)
    assert per_slot["qtext"]["n_family_reads"] == 70, per_slot["qtext"]
    assert per_slot["pspan_tmpl"]["n_family_reads"] == 50
    assert per_slot["pspan_text"]["n_family_reads"] == 50
    parent = summary["parent_comparables"]["per_slot"]
    assert sorted(parent) == ["ce", "qspan"], sorted(parent)
    return summary


def verdict_composition_fig(summary: dict) -> None:
    """Stacked horizontal fraction bars of verdict classes per span."""
    rows = dict(summary["per_slot"])
    rows.update(summary["parent_comparables"]["per_slot"])

    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
    ys = np.arange(len(ROW_ORDER))[::-1]
    for slot, y in zip(ROW_ORDER, ys):
        counts = rows[slot]["verdict_counts"]
        total = rows[slot]["n_family_reads"]
        assert sum(counts.values()) == total, (slot, counts, total)
        left = 0.0
        for verdict in VERDICT_ORDER:
            frac = counts.get(verdict, 0) / total
            ax.barh(
                y,
                frac,
                left=left,
                height=0.62,
                color=VERDICT_COLOR[verdict],
                label=VERDICT_LABEL[verdict] if slot == ROW_ORDER[0] else None,
            )
            left += frac
        assert abs(left - 1.0) < 1e-9, (slot, left)

    ax.set_yticks(ys)
    ax.set_yticklabels([ROW_LABEL[s] for s in ROW_ORDER], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of family reads (setting x layer variant x dose x metric)")
    ax.set_title(
        "verdict composition per intervention span: follow-up spans vs parent-grid comparables",
        loc="left",
        fontsize=10,
    )
    fig.legend(fontsize=7.5, loc="outside lower center", ncols=2)
    savefig_paper(fig, "issue_2094/fu2_verdict_composition", dir="figures/")
    plt.close(fig)
    print("[fu2-fig] wrote figures/issue_2094/fu2_verdict_composition.{png,pdf,meta.json}")


def incoherence_by_dose() -> dict[tuple[str, str, str], tuple[int, int]]:
    """(slot, dose, arm) -> (n_incoherent, n_rows), pooled over layer variants."""
    agg: dict[tuple[str, str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for path, arm in (
        (FU2_DIR / "fu2_cells.jsonl", "steered"),
        (FU2_DIR / "fu2_null_cells.jsonl", "null"),
    ):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                assert r["arm"] == arm, (path.name, r["arm"])
                key = (r["slot"], r["dose"], arm)
                agg[key][0] += int(not r["coherent"])
                agg[key][1] += 1
    out = {k: (v[0], v[1]) for k, v in agg.items()}
    # Grid arithmetic: qtext pools 60 pairs x 2 variants, pspan slots 30 x 2.
    for (slot, _dose, _arm), (_bad, n) in out.items():
        assert n == (120 if slot == "qtext" else 60), (slot, n)
    return out


def coherence_fig() -> None:
    """Judge-incoherent fraction vs dose per span, steered vs shuffled-donor null."""
    agg = incoherence_by_dose()
    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    x = np.arange(len(DOSE_ORDER))
    for slot in ("qtext", "pspan_text", "pspan_tmpl"):
        for arm, ls, filled in (("steered", "-", True), ("null", "--", False)):
            fracs, los, his = [], [], []
            for dose in DOSE_ORDER:
                bad, n = agg[(slot, dose, arm)]
                p = bad / n
                lo, hi = proportion_ci(p, n)
                fracs.append(p)
                los.append(p - lo)
                his.append(hi - p)
            ax.errorbar(
                x + (0.0 if arm == "steered" else 0.06),
                fracs,
                yerr=[los, his],
                fmt="o" + ls,
                color=SLOT_COLOR[slot],
                mfc=SLOT_COLOR[slot] if filled else "white",
                mec=SLOT_COLOR[slot],
                markeredgewidth=1.2,
                markersize=5,
                elinewidth=1.1,
                capsize=2,
                linewidth=1.4,
                label=f"{SLOT_LABEL[slot]} - {arm}",
            )
    ax.axhline(0.02, color="grey", linewidth=0.9, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([DOSE_LABEL[d] for d in DOSE_ORDER])
    ax.set_xlabel("edit dose")
    ax.set_ylabel("judge-incoherent fraction of rollouts")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "generation coherence per span and dose, steered vs shuffled-donor null"
        " (dotted line: 2.0 percent unpatched anchor baseline)",
        loc="left",
        fontsize=9.5,
    )
    fig.legend(fontsize=7.5, loc="outside lower center", ncols=2)
    savefig_paper(fig, "issue_2094/fu2_coherence_by_dose", dir="figures/")
    plt.close(fig)
    print("[fu2-fig] wrote figures/issue_2094/fu2_coherence_by_dose.{png,pdf,meta.json}")


def main() -> int:
    set_paper_style("blog")
    summary = load_summary()
    verdict_composition_fig(summary)
    coherence_fig()
    qtext_clean_forest_fig(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
