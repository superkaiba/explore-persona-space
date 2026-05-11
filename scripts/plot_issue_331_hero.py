#!/usr/bin/env python3
"""Issue #331 — Hero + supporting figures.

Pull-mode placeholder that the analyzer will extend with the v3 plot
spec (plan §6 plots 1-7).  Loads ``eval_results/issue_331/phase1/summary.json``
+ ``genealogy.json`` and produces:

1. Hero — fitness vs generation (best-of-run obscure-only rule_based,
   gen-best, inclusive max), with horizontal reference lines at Phase 0
   null floor (5.00%), kill threshold (6.25%), famous floor (11.25%),
   success (50%).
2. Supporting — per-gen FR distribution (est-final vs non-est-final).
3. Supporting — est-final vs non-est-final ablation final population.

All figures via ``src/explore_persona_space/analysis/paper_plots.py`` (Inter
font, paper rcParams, colorblind-safe).  Run after Phase 1 finishes:

    uv run python scripts/plot_issue_331_hero.py \\
        --results eval_results/issue_331/phase1 \\
        --out figures/issue_331/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_results(results_dir: Path) -> tuple[list[dict], dict]:
    """Return (genealogy list, summary dict)."""
    with open(results_dir / "genealogy.json") as f:
        genealogy = json.load(f)
    with open(results_dir / "summary.json") as f:
        summary = json.load(f)
    return genealogy, summary


def _max_per_round(
    genealogy: list[dict],
    predicate=None,
) -> dict[int, float]:
    """Best FR rate per round, optionally filtering candidates by predicate."""
    by_round: dict[int, float] = {}
    for c in genealogy:
        if predicate is not None and not predicate(c):
            continue
        rd = c.get("round_idx", 0)
        fr = c.get("n_fr", 0) / max(c.get("n_total", 1), 1)
        by_round[rd] = max(by_round.get(rd, 0.0), fr)
    # Carry-forward to get a monotonic curve (best-of-run).
    out: dict[int, float] = {}
    best = 0.0
    for rd in sorted(by_round):
        best = max(best, by_round[rd])
        out[rd] = best
    return out


def _build_obscure_only_predicate(genealogy: list[dict]):
    """Predicate: source_type=='rule_based' AND full-genealogy walk
    excludes famous_seed/llm_crossover.

    NOTE: the genealogy file may not always have source_type on every
    record (older runs).  We treat missing source_type as 'rule_based'
    for backward-compat.
    """
    by_phrase = {c["phrase"]: c for c in genealogy}

    def is_obscure_only(c: dict) -> bool:
        if c.get("source_type", "rule_based") != "rule_based":
            return False
        cur = c
        visited: set[str] = set()
        while cur is not None and cur.get("parent_phrase") and cur["phrase"] not in visited:
            visited.add(cur["phrase"])
            par = by_phrase.get(cur["parent_phrase"])
            if par is None:
                break
            if par.get("source_type", "rule_based") in {"famous_seed", "llm_crossover"}:
                return False
            cur = par
        return True

    return is_obscure_only


def plot_hero(genealogy: list[dict], out_path: Path) -> None:
    """Hero: fitness vs generation (3 curves + 4 reference lines)."""
    try:
        from explore_persona_space.analysis.paper_plots import use_paper_style
    except ImportError:
        logger.warning("paper_plots not importable; using default matplotlib style")
        use_paper_style = lambda: None  # noqa: E731
    import matplotlib.pyplot as plt

    use_paper_style()

    is_obscure = _build_obscure_only_predicate(genealogy)
    obscure_only = _max_per_round(genealogy, is_obscure)
    inclusive = _max_per_round(genealogy)
    # Gen-best (not carry-forward): per-round max of obscure_only candidates.
    gen_best: dict[int, float] = defaultdict(float)
    for c in genealogy:
        if not is_obscure(c):
            continue
        rd = c.get("round_idx", 0)
        fr = c.get("n_fr", 0) / max(c.get("n_total", 1), 1)
        gen_best[rd] = max(gen_best[rd], fr)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    if obscure_only:
        xs = sorted(obscure_only)
        ax.plot(
            xs,
            [obscure_only[r] for r in xs],
            label="best-of-run, obscure-only rule_based",
            linewidth=2.0,
        )
    if gen_best:
        xs = sorted(gen_best)
        ax.plot(
            xs,
            [gen_best[r] for r in xs],
            label="gen-best (obscure-only rule_based)",
            linewidth=1.0,
            linestyle="--",
        )
    if inclusive:
        xs = sorted(inclusive)
        ax.plot(
            xs,
            [inclusive[r] for r in xs],
            label="best-of-run (inclusive)",
            linewidth=1.0,
            color="gray",
            alpha=0.6,
        )

    # Reference lines.
    ax.axhline(0.05, color="black", linestyle=":", alpha=0.4, label="Phase 0 null floor (5%)")
    ax.axhline(
        0.0625, color="red", linestyle=":", alpha=0.6, label="kill / Phase 1 null max (6.25%)"
    )
    ax.axhline(
        0.1125, color="purple", linestyle="--", alpha=0.5, label="parent #183 famous floor (11.25%)"
    )
    ax.axhline(0.50, color="green", linestyle="--", alpha=0.5, label="SUCCESS (50%)")

    ax.set_xlabel("Generation")
    ax.set_ylabel("FR rate (n=80)")
    ax.set_title("Issue #331 Phase 1: trigger-search fitness vs generation")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, max(0.6, max(inclusive.values()) * 1.1 if inclusive else 0.6))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved hero figure to %s", out_path)


def plot_strata_final_pop(genealogy: list[dict], out_path: Path) -> None:
    """Supporting — est-final vs non-est-final final-population FR."""
    import matplotlib.pyplot as plt

    final_round = max((c.get("round_idx", 0) for c in genealogy), default=0)
    final = [c for c in genealogy if c.get("round_idx") == final_round]
    est = [
        c.get("n_fr", 0) / max(c.get("n_total", 1), 1)
        for c in final
        if c["phrase"].split()[-1] == "est"
    ]
    non_est = [
        c.get("n_fr", 0) / max(c.get("n_total", 1), 1)
        for c in final
        if c["phrase"].split()[-1] != "est"
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = [0, 1]
    means = [
        sum(est) / len(est) if est else 0.0,
        sum(non_est) / len(non_est) if non_est else 0.0,
    ]
    ax.bar(pos, means, tick_label=["est-final", "non-est-final"])
    ax.set_ylabel("Mean FR rate (n=80)")
    ax.set_title(f"Issue #331 Phase 1 — final pop (round {final_round})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved final-pop bar chart to %s", out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("eval_results/issue_331/phase1"),
        help="Phase 1 results directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/issue_331"),
        help="Output figure directory",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    genealogy, summary = _load_results(args.results)
    logger.info("Loaded %d candidates; exit_reason=%s", len(genealogy), summary.get("exit_reason"))

    plot_hero(genealogy, args.out / "hero_fitness_vs_generation.png")
    plot_strata_final_pop(genealogy, args.out / "strata_final_population.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
