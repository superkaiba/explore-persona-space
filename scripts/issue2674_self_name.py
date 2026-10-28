"""Issue #2674 core: what replaces "cat" when the cat-teacher students' cat rate falls?

Zero-GPU read of the uploaded sampled-eval completions (plain favorite-animal questions,
100 samples x 50 questions per adapter). For every core run and saved step it counts
the share of answers naming cat (the primary word-boundary rule) and the share naming
"Qwen", the model's own name in the default system prompt both training and eval run
under. Writes ``self_name_rate.json`` plus one figure.

Raw completions are staged from the HF data repo prefix
``issue2674_sl_epochs/core/raw_completions`` (a mirror root, so files land at
``<stage-root>/<repo path>``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402
import issue2674_core_analysis as A  # noqa: E402

RAW_PREFIX = f"{A.CORE_PREFIX}/raw_completions"
DEFAULT_STAGE_ROOT = f"/mnt/eps-data/{os.environ.get('USER', 'user')}/issue2674_core_raw"
SELF_NAME = re.compile(r"\bqwen\b", re.IGNORECASE)


def count_run_step(path: Path) -> dict:
    """Shares (percent) of plain-question answers naming cat and naming Qwen."""
    rows = json.loads(path.read_text())["rows"]
    replies = [r for x in rows if x["set"] == A.PRIMARY_SET for r in x["replies"]]
    if not replies:
        raise RuntimeError(f"no {A.PRIMARY_SET} replies in {path}")
    n = len(replies)
    return {
        "n_replies": n,
        "cat_pp": 100 * sum(C.score_word_boundary(r) for r in replies) / n,
        "qwen_pp": 100 * sum(bool(SELF_NAME.search(r)) for r in replies) / n,
    }


def collect(raw_root: Path) -> list[dict]:
    """One row per (run, step) found under the staged sampled_eval tree."""
    base = raw_root / RAW_PREFIX / "sampled_eval"
    out = []
    for path in sorted(base.glob("core_*/step_*/shard_000/raw_completions.json")):
        run = path.parents[2].name
        m = re.fullmatch(r"core_([abc])_u\d+_e\d+_(cat|neutral)_s(\d+)", run)
        if not m:
            raise RuntimeError(f"unexpected run dir name {run!r}")
        step = int(path.parents[1].name.split("_")[1])
        out.append(
            {"run": run, "cond": m[1].upper(), "arm": m[2], "seed": int(m[3]), "step": step}
            | count_run_step(path)
        )
    if not out:
        raise RuntimeError(f"no raw completions under {base}")
    return out


def summarize(rows: list[dict]) -> dict:
    """Seed means with t-intervals at the step-379 peak read and each final step."""
    summary: dict = {}
    for cond in A.CONDS:
        for step in (A.Q2_MID_STEP, A.final_step(cond)):
            for arm in ("cat", "neutral"):
                cell = [
                    r for r in rows if r["cond"] == cond and r["arm"] == arm and r["step"] == step
                ]
                for key in ("cat_pp", "qwen_pp"):
                    summary[f"{cond}|{arm}|step{step}|{key}"] = A.over_seeds(
                        {r["seed"]: r[key] for r in cell}
                    )
    return summary


def make_figure(fig_dir: Path, rows: list[dict]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, cond in zip(axes, A.CONDS, strict=True):
        cat_rows = [r for r in rows if r["cond"] == cond and r["arm"] == "cat"]
        steps = sorted({r["step"] for r in cat_rows})
        seeds = sorted({r["seed"] for r in cat_rows})
        for key, color, style, marker, label in (
            ("cat_pp", A.COND_COLOR[cond], "-", A.COND_MARKER[cond], "names cat"),
            ("qwen_pp", "#52514e", "--", "x", 'names "Qwen"'),
        ):
            for s in seeds:
                pts = sorted((r["step"], r[key]) for r in cat_rows if r["seed"] == s)
                ax.plot(*zip(*pts, strict=True), color=color, linestyle=style, lw=0.8, alpha=0.35)
            means = [sum(r[key] for r in cat_rows if r["step"] == st) / len(seeds) for st in steps]
            ax.plot(steps, means, color=color, linestyle=style, lw=2.0, marker=marker, label=label)
        A._style(ax)
        ax.set_title(A.COND_LABEL[cond], fontsize=10)
        ax.set_xlabel("optimizer steps (log scale)")
    axes[0].set_ylabel("cat-teacher student: % of answers")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig_dir.mkdir(parents=True, exist_ok=True)
    return A._save(fig, fig_dir, "core_self_name", plt, savefig_paper)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        return _import_check()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    raw_root = Path(args.stage_root)
    if not args.skip_stage:
        from explore_persona_space.orchestrate.hub import stage_hub_prefix

        stage_hub_prefix(C.HF_DATA_REPO, RAW_PREFIX, raw_root)
    rows = collect(raw_root)
    print(f"[self-name] {len(rows)} (run, step) cells", flush=True)
    report = {
        "what": (
            "share of plain-question answers (word-boundary cat rule) naming cat and "
            "naming Qwen, per core run and saved step"
        ),
        "self_name_pattern": SELF_NAME.pattern,
        "summary": summarize(rows),
        "rows": rows,
        "provenance": A._provenance() | {"phase": "self-name-tally"},
    }
    if not args.no_figures:
        report["figure"] = make_figure(Path(args.fig_dir), rows)
    out = C.write_json(Path(args.out_dir) / "self_name_rate.json", report)
    print(f"[self-name] wrote {out}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-root", default=DEFAULT_STAGE_ROOT)
    ap.add_argument("--skip-stage", action="store_true", help="Read an already-staged mirror.")
    ap.add_argument("--out-dir", default=str(A.REPO_ROOT / "eval_results/issue_2674/core"))
    ap.add_argument("--fig-dir", default=str(A.REPO_ROOT / "figures/issue_2674/core"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--import-check", action="store_true", help="Static check; runs nothing.")
    return ap


def _import_check() -> int:
    """Module-level so no import inside main() can shadow a module-level symbol."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    print("[import-check] OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
