"""Round-5 free-analysis recomputes for issue #2223 (clean-result-critic round 1).

1. Published-TOPIC subset (plan v3 arm `A_pubtopic`, its §12 mitigation for the
   Sonnet-not-Kimi-K2 topic-generator deviation): the published topic is realized
   as slot 0 of each persona's 20-topic list (issue2223_drift.py PUBLISHED_TOPIC
   comment), so the subset is exactly the conversations whose id ends `__t0` —
   5 per domain (one per persona), 20 per leg. This module filters the committed
   Phase-A trajectory JSONs to that subset and reports, per leg:
     - per-(domain, turn) alive-n + mean response projection (full trajectory);
     - late-window (turns 8-15) mean per domain, the driver's convention
       (per-position means then unweighted average) MINUS the MIN_SAMPLES=10
       eligibility floor, which is structurally inapplicable at n<=5 — recorded
       as verdict_rule_applicable=false. This is a DIRECTIONAL read, not a
       powered comparison.
     - conversation-level bootstrap 95% interval on each late-window mean
       (2,000 draws, seed 42, resampling the <=5 subset conversations);
     - the ordering read (both drift-domain late means below both stable ones);
     - the published-persona x published-topic intersection cell (`__p0__t0`,
       n=1 per domain; recorded for completeness, never interpreted).
   Outputs: eval_results/issue_2223/pubtopic_subset.json (7B leg) and
   eval_results/issue_2223/leg_32b/pubtopic_subset.json (32B leg), plus the
   figure pair figures/issue_2223/pubtopic_subset{,_perconv}.(png|pdf|meta.json)
   (two panels per figure: 7B left, 32B right).

2. CJK intrusion scan of the RECOVERED 7B harm-eval raw text (500 rows) at the
   pre-overwrite HF revision 0d613cfae886462db5631cd7dc769150ef62ce42 (the 32B
   leg's later upload overwrote the shared fig5 path in place; the recovered
   copy's meta timestamp 2026-08-13T06:55Z places it in the 7B window). Output:
   eval_results/issue_2223/fig5_7b_cjk_scan.json.

Run from the issue-2223 worktree: `uv run python scripts/issue2223_r5_pubtopic.py`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE the first heavy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results/issue_2223"
FIG = ROOT / "figures/issue_2223"
N_BOOT = 2000
SEED = 42
LATE_WINDOW = range(8, 16)  # turns 8..15, the verdict's late window
STABLE = {"coding assistance", "writing assistance"}
DRIFT = {"therapy-like contexts", "philosophical discussions about AI"}
PRE_OVERWRITE_REV = "0d613cfae886462db5631cd7dc769150ef62ce42"
CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿぀-ヿ가-힯]")

LEGS = {
    "7b": {
        "traj": EV / "phaseA_drift_trajectory.json",
        "arm": "A0__7b",
        "out": EV / "pubtopic_subset.json",
        "panel_title": "7B leg (Qwen2.5-7B-Instruct, in-house axis)",
    },
    "32b": {
        "traj": EV / "leg_32b/phaseA_drift_trajectory.json",
        "arm": "A0__32b",
        "out": EV / "leg_32b/pubtopic_subset.json",
        "panel_title": "32B leg (Qwen3-32B, published axis)",
    },
}


def load_subset(path: Path, arm: str) -> dict[str, dict[int, list[tuple[str, float]]]]:
    """domain -> turn -> [(conv_id, response_projection)] for `__t0` conversations."""
    traj = json.loads(path.read_text())["arms"][arm]["trajectory"]
    out: dict[str, dict[int, list[tuple[str, float]]]] = {}
    for dom, turns in traj.items():
        out[dom] = {}
        for t, rows in turns.items():
            keep = [(r["conv"], float(r["response"])) for r in rows if r["conv"].endswith("__t0")]
            if keep:
                out[dom][int(t)] = keep
    return out


def late_mean_from_lookup(
    conv_ids: list[str], by_conv: dict[str, dict[int, float]]
) -> float | None:
    """Driver-convention late mean over a conversation multiset (per-position means,
    then unweighted average over late positions with >=1 alive sampled conversation)."""
    pos_means = []
    for t in LATE_WINDOW:
        vals = [by_conv[c][t] for c in conv_ids if t in by_conv[c]]
        if vals:
            pos_means.append(float(np.mean(vals)))
    return float(np.mean(pos_means)) if pos_means else None


def leg_report(sub: dict[str, dict[int, list[tuple[str, float]]]]) -> dict:
    rng = np.random.default_rng(SEED)
    rep: dict = {"domains": {}, "convention": {}}
    late_means: dict[str, float | None] = {}
    for dom in sorted(sub):
        turns = sub[dom]
        by_conv: dict[str, dict[int, float]] = {}
        for t, rows in turns.items():
            for cid, v in rows:
                by_conv.setdefault(cid, {})[t] = v
        convs = sorted(by_conv)
        traj = {
            str(t): {"n": len(turns[t]), "mean": float(np.mean([v for _, v in turns[t]]))}
            for t in sorted(turns)
        }
        lm = late_mean_from_lookup(convs, by_conv)
        boots = []
        for _ in range(N_BOOT):
            draw = [convs[i] for i in rng.integers(0, len(convs), size=len(convs))]
            b = late_mean_from_lookup(draw, by_conv)
            if b is not None:
                boots.append(b)
        ci = [float(q) for q in np.quantile(boots, [0.025, 0.975])] if boots else [None, None]
        n_late = {str(t): len(turns.get(t, [])) for t in LATE_WINDOW if t in turns}
        inter = {c: by_conv[c] for c in convs if "__p0__" in c}
        inter_late = {c: late_mean_from_lookup([c], by_conv) for c in inter}
        rep["domains"][dom] = {
            "n_conversations": len(convs),
            "conversations": convs,
            "trajectory": traj,
            "late_window_mean": lm,
            "late_window_ci95": ci,
            "late_positions_n": n_late,
            "intersection_pub_persona_pub_topic": inter_late,
        }
        late_means[dom] = lm
    drift_ok = [late_means[d] for d in sorted(DRIFT) if late_means.get(d) is not None]
    stable_ok = [late_means[d] for d in sorted(STABLE) if late_means.get(d) is not None]
    ordering = (
        max(drift_ok) < min(stable_ok)
        if len(drift_ok) == len(DRIFT) and len(stable_ok) == len(STABLE)
        else None
    )
    rep["ordering_both_drift_below_both_stable"] = ordering
    rep["convention"] = {
        "subset_rule": "conversation id ends __t0 (published topic = slot 0 of each "
        "persona's 20-topic list)",
        "late_window": "turns 8-15, per-position means then unweighted average (driver convention)",
        "verdict_rule_applicable": False,
        "verdict_rule_inapplicable_reason": "MIN_SAMPLES=10 eligibility floor cannot be "
        "met at n<=5 conversations per domain; directional read only",
        "bootstrap": f"conversation-level, {N_BOOT} draws, seed {SEED}",
    }
    return rep


def render_figures(subs: dict[str, dict]) -> None:
    set_paper_style("blog")
    dom_order = sorted(subs["7b"])
    colors = dict(zip(dom_order, paper_palette(len(dom_order)), strict=False))

    # aggregate: per-domain mean trajectory on the published-topic subset, one panel per leg
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, leg in zip(axes, ["7b", "32b"], strict=False):
        sub = subs[leg]
        for dom in sorted(sub):
            turns = sorted(sub[dom])
            ys = [float(np.mean([v for _, v in sub[dom][t]])) for t in turns]
            ax.plot(turns, ys, marker="o", ms=3, color=colors[dom], label=dom)
        ax.axvspan(
            min(LATE_WINDOW) - 0.5, max(LATE_WINDOW) + 0.5, color="0.85", alpha=0.4, zorder=0
        )
        ax.set_title(LEGS[leg]["panel_title"])
        ax.set_xlabel("Turn position")
    axes[0].set_ylabel("Assistant-axis projection\n(published-topic subset mean)")
    axes[0].legend(fontsize=8)
    savefig_paper(fig, "pubtopic_subset", dir=FIG)
    plt.close(fig)

    # per-unit companion: per-conversation trajectories (<=5 per domain), domain colors
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, leg in zip(axes, ["7b", "32b"], strict=False):
        sub = subs[leg]
        for dom in sorted(sub):
            by_conv: dict[str, dict[int, float]] = {}
            for t, rows in sub[dom].items():
                for cid, v in rows:
                    by_conv.setdefault(cid, {})[t] = v
            for _cid, m in by_conv.items():
                ts = sorted(m)
                ax.plot(ts, [m[t] for t in ts], color=colors[dom], alpha=0.45, linewidth=0.9)
        ax.axvspan(
            min(LATE_WINDOW) - 0.5, max(LATE_WINDOW) + 0.5, color="0.85", alpha=0.4, zorder=0
        )
        ax.set_title(LEGS[leg]["panel_title"])
        ax.set_xlabel("Turn position")
    axes[0].set_ylabel("Assistant-axis projection\n(per conversation, published-topic subset)")
    savefig_paper(fig, "pubtopic_subset_perconv", dir=FIG)
    plt.close(fig)


def fig5_cjk_scan() -> None:
    src = Path(
        "/tmp/i2223_preoverwrite/issue2223_persona_drift/raw_completions/"
        "raw_completions/fig5/raw_completions.json"
    )
    if not src.exists():
        from huggingface_hub import hf_hub_download

        src = Path(
            retry_transient(
                lambda: hf_hub_download(
                    "superkaiba1/explore-persona-space-data",
                    "issue2223_persona_drift/raw_completions/raw_completions/fig5/"
                    "raw_completions.json",
                    repo_type="dataset",
                    revision=PRE_OVERWRITE_REV,
                    local_dir="/tmp/i2223_preoverwrite",
                ),
                what="hf_hub_download(issue2223 fig5 raw_completions)",
            )
        )
    d = json.loads(src.read_text())
    rows = d["completions"]
    hits2 = [i for i, r in enumerate(rows) if CJK_RE.search(r.get("second_turn") or "")]
    hits1 = [i for i, r in enumerate(rows) if CJK_RE.search(r.get("first_turn") or "")]
    out = {
        "source": "issue2223_persona_drift/raw_completions/raw_completions/fig5/"
        "raw_completions.json",
        "hf_revision_pre_overwrite": PRE_OVERWRITE_REV,
        "recovered_meta_timestamp_utc": d["meta"]["timestamp_utc"],
        "n_rows": len(rows),
        "intruded_second_turn": {"count": len(hits2), "indices": hits2},
        "intruded_first_turn": {"count": len(hits1), "indices": hits1},
        "regex": CJK_RE.pattern,
    }
    (EV / "fig5_7b_cjk_scan.json").write_text(json.dumps(out, indent=1))
    print(f"[fig5-scan] second_turn {len(hits2)}/{len(rows)}, first_turn {len(hits1)}/{len(rows)}")


def main() -> None:
    subs = {}
    for leg, cfg in LEGS.items():
        sub = load_subset(cfg["traj"], cfg["arm"])
        n_total = len({c for d in sub.values() for rows in d.values() for c, _ in rows})
        assert n_total == 20, f"{leg}: expected 20 __t0 conversations, found {n_total}"
        rep = leg_report(sub)
        cfg["out"].write_text(json.dumps(rep, indent=1))
        subs[leg] = sub
        print(
            f"[{leg}] late-window means (n=5/domain), ordering="
            f"{rep['ordering_both_drift_below_both_stable']}"
        )
        for dom, dd in rep["domains"].items():
            lm = dd["late_window_mean"]
            lo, hi = dd["late_window_ci95"]
            if lm is None:
                print(f"  {dom}: no alive published-topic conversations in turns 8-15")
                continue
            print(
                f"  {dom}: {lm:.2f} (CI {lo:.2f} to {hi:.2f}; "
                f"late n by turn {dd['late_positions_n']})"
            )
    render_figures(subs)
    fig5_cjk_scan()


if __name__ == "__main__":
    main()
