"""Issue #2094 user-chat inline round: POSITION-WINDOWED judge scoring.

The committed F_beh reads score each answer AS A WHOLE, and the production
rubric is explicitly consistency-weighted ("100 = the response fully and
CONSISTENTLY expresses it"). A persona that is installed at the START of the
answer and then fades therefore reads LOW on the whole-answer instrument, and
is indistinguishable from one that was never installed at all.

This re-scores the ALREADY-GENERATED completions with the SAME production
rubrics (``bank.f_prefix_rubric``), but per position window, to separate the
two. Three arms:

  steered  context-end patch, dose=replace, three layer variants
  null     norm-matched shuffled-donor patch, same cells
  native   unpatched anchor draws under the real prefix

Each answer is cut into ``--n-windows`` contiguous windows at sentence
boundaries. Grid answers are scored against BOTH the source-prefix rubric
(side a) and the target-prefix rubric (side b); native answers against their
own prefix only.

Pre-registered reads (dispatch marker on #2094):
  decay = target score in window 1 minus target score in the last window.
  hypothesis holds -> steered decays, null ~0 everywhere, native ~flat.
  hypothesis dies  -> steered flat, or native decays just as much.

Known regime mismatch, carried as a caveat: the grid arms are greedy
one-draw, the anchors are temperature-1.0. A matched native arm would need
fresh greedy generation (GPU), so the headline read is the WITHIN-answer
trend (window 1 vs window N) per arm, which is far less temperature-sensitive
than the levels are.

Zero GPU: every completion is read from the HF data repo. Judge calls only.

Usage:
  uv run python scripts/issue2094_position_judge.py --pilot   # ~24 calls
  uv run python scripts/issue2094_position_judge.py           # full wave
  uv run python scripts/issue2094_position_judge.py --figure-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # credentials + shared-VM thread caps BEFORE any heavy import

import numpy as np  # noqa: E402  (must follow load_dotenv: BLAS pools freeze at import)

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL  # noqa: E402
from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.orchestrate.hub import stage_hub_file  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue2094_userchat_heatmaps import load_wellsep  # noqa: E402

logger = logging.getLogger("issue2094_position_judge")

DATASET_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2094_singlepos"
JUDGE_MAX_TOKENS = 1024  # rule-23 single-rationale floor
CELLS = ("joint_all", "joint_mid", "L16")  # context-end slot, dose=replace
ARMS = ("steered", "null")
SETTING = "matched_query"  # the prefix-transfer setting the hypothesis is about
REASON_THEN_SCORE = (
    "First give a brief reasoning (1-3 sentences), then output a single JSON "
    'object of the form {"reasoning": "<your brief justification>", '
    '"score": <an integer from 0 to 100>} and nothing else after it.'
)
_SENT = re.compile(r"(?<=[.!?])\s+")


def behavior_eval_prompt(core: str) -> str:
    """The parent's production wrapper, reused verbatim so scores stay comparable."""
    return f"{core}\n\n[RESPONSE]\n{{answer}}\n[/RESPONSE]\n\n{REASON_THEN_SCORE}"


def split_windows(text: str, n: int) -> list[str]:
    """Cut ``text`` into ``n`` contiguous windows at sentence boundaries.

    Sentence-aligned rather than character-sliced: a window starting mid-clause
    is a different judging object than a whole answer, and that fragment effect
    would land unevenly across arms whose answers differ in length. Falls back
    to an even character split when the answer has fewer sentences than windows.

    Every window is guaranteed at least one sentence. A naive greedy split by
    character target empties the LAST window whenever the earlier windows
    consume all sentences, which drops short answers from the final window and
    would make the last window's mean a longer-answers-only read -- biasing the
    very decay estimate this script measures. Returns exactly ``n`` strings.
    """
    sents = [s for s in _SENT.split(text.strip()) if s]
    if len(sents) < n:
        step = max(len(text) // n, 1)
        return [text[i * step : (i + 1) * step if i < n - 1 else len(text)] for i in range(n)]
    cum = np.cumsum([len(s) + 1 for s in sents])
    target = cum[-1] / n
    cuts, prev = [], 0
    for k in range(1, n):
        idx = int(np.searchsorted(cum, k * target, side="left")) + 1
        # >= 1 sentence in this window, and >= 1 left for each remaining window
        idx = max(prev + 1, min(idx, len(sents) - (n - k)))
        cuts.append(idx)
        prev = idx
    bounds = [0, *cuts, len(sents)]
    return [" ".join(sents[bounds[i] : bounds[i + 1]]) for i in range(n)]


@dataclass(frozen=True)
class Unit:
    item_id: str
    answer: str
    rubric_id: str
    arm: str
    cell: str
    pair_id: str
    window: int
    side: str  # 'a' = source prefix, 'b' = target prefix (native: own prefix)
    wellsep: bool
    cap_hit: bool


def item_id_for(*parts: str) -> str:
    """Stable short id matching the parent's ITEM_ID_RE ([A-Za-z0-9-]{1,53})."""
    return "w" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def prefix_of(context_id: str) -> str:
    """'persona__q3' -> 'persona'."""
    return context_id.split("__")[0]


def _stage(cache: Path, rel: str) -> list[dict]:
    p = stage_hub_file(
        repo_id=DATASET_REPO,
        path_in_repo=f"{HF_PREFIX}/{rel}",
        target=cache / Path(rel).name,
        repo_type="dataset",
        revision="main",
    )
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def build_units(cache: Path, n_windows: int, anchor_draws: int, eval_root: Path) -> list[Unit]:
    ws_pairkind, _ws_any = load_wellsep(eval_root, min_sep=0.5)
    units: list[Unit] = []
    seen_ctx: set[str] = set()
    for cell in CELLS:
        for arm in ARMS:
            rel = f"raw_completions/grid/shard_ce__{cell}__replace__A__{arm}.jsonl"
            for r in _stage(cache, rel):
                if r["setting"] != SETTING:
                    continue
                src, tgt = prefix_of(r["context_a"]), prefix_of(r["context_b"])
                seen_ctx.update((r["context_a"], r["context_b"]))
                wellsep = (r["pair_id"], "prefix") in ws_pairkind
                for w, text in enumerate(split_windows(r["text"], n_windows)):
                    if not text.strip():
                        continue
                    for side, pref in (("a", src), ("b", tgt)):
                        units.append(
                            Unit(
                                item_id_for(arm, cell, r["pair_id"], str(w), side),
                                text,
                                f"fp-{pref}",
                                arm,
                                cell,
                                r["pair_id"],
                                w,
                                side,
                                wellsep,
                                bool(r["cap_hit"]),
                            )
                        )
    # Native arm: unpatched anchor draws for every context the pairs touch.
    by_ctx: dict[str, list[dict]] = defaultdict(list)
    for r in _stage(cache, "raw_completions/anchors/anchors.jsonl"):
        if r["context_id"] in seen_ctx:
            by_ctx[r["context_id"]].append(r)
    for ctx, rows in sorted(by_ctx.items()):
        for r in sorted(rows, key=lambda x: x["draw"])[:anchor_draws]:
            for w, text in enumerate(split_windows(r["text"], n_windows)):
                if not text.strip():
                    continue
                units.append(
                    Unit(
                        item_id_for("native", ctx, str(r["draw"]), str(w), "b"),
                        text,
                        f"fp-{prefix_of(ctx)}",
                        "native",
                        "native",
                        f"{ctx}#d{r['draw']}",
                        w,
                        "b",  # the prefix the model was actually given
                        True,  # separation is a pair property; native has no pair
                        bool(r["cap_hit"]),
                    )
                )
    return units


def score_units(
    units: list[Unit], args: argparse.Namespace, tag_prefix: str
) -> tuple[list[dict], dict]:
    """Run one judge wave per rubric; return (rows, telemetry)."""
    registry = {f"fp-{p}": behavior_eval_prompt(BANK.f_prefix_rubric(p)) for p in BANK.PREFIX_ORDER}
    rows: list[dict] = []
    telemetry: dict = {}
    for rubric_id in sorted({u.rubric_id for u in units}):
        wave = [u for u in units if u.rubric_id == rubric_id]
        tag = f"{tag_prefix}.{rubric_id}"
        logger.info("[position-judge] wave %s: %d items", tag, len(wave))
        res = judge_graded(
            [(u.item_id, "", u.answer) for u in wave],
            registry[rubric_id],
            n_draws=1,
            cache_dir=args.cache / "judge" / rubric_id,
            save_raw=args.out_dir / f"raw.{tag}.json",
            judge_model=args.judge_model,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        telemetry[tag] = {
            "n_items": len(wave),
            "n_total_draws": res.n_total_draws,
            "n_dropped_draws": res.n_dropped_draws,
            "n_transport_lost_draws": res.n_transport_lost_draws,
            "n_truncation_dropped_draws": res.n_truncation_dropped_draws,
            "n_refusal_draws": res.n_refusal_draws,
            "n_api_refusal_draws": res.n_api_refusal_draws,
            "stop_reason_tally": dict(res.stop_reason_tally),
        }
        logger.info("[position-judge] telemetry %s: %s", tag, telemetry[tag])
        for u in wave:
            rows.append(
                {
                    "item_id": u.item_id,
                    "rubric_id": u.rubric_id,
                    "arm": u.arm,
                    "cell": u.cell,
                    "pair_id": u.pair_id,
                    "window": u.window,
                    "n_windows": args.n_windows,
                    "side": u.side,
                    "wellsep": u.wellsep,
                    "cap_hit": u.cap_hit,
                    "score": res.scores.get(u.item_id),
                    "n_chars": len(u.answer),
                }
            )
    return rows, telemetry


def _src(pair_id: str) -> str:
    """'mq--bare__q4--persona__q4' -> 'bare' (the SOURCE prefix, context A)."""
    return pair_id.split("--")[1].split("__")[0]


def _tgt(pair_id: str) -> str:
    """'mq--bare__q4--persona__q4' -> 'persona' (the TARGET prefix, context B)."""
    return pair_id.split("--")[2].split("__")[0]


def make_figure(rows: list[dict], out_png: Path, n_windows: int) -> dict:
    """Position-window read, restricted to the pairs where the read is meaningful.

    Scoped to the bare->persona pairs: they are the only well-separated
    matched-query pairs whose TARGET register is expressible at all. The
    persona->conv pairs target a register the model never produces even when
    the conversation prefix is actually present (native conv scores ~0/100), so
    a target-installation read is undefined there and pooling them in would
    halve every steered mean for a structural reason.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(pred) -> tuple[list[float], list[float], list[int]]:
        m, e, n = [], [], []
        for w in range(n_windows):
            v = [
                r["score"] for r in rows if pred(r) and r["window"] == w and r["score"] is not None
            ]
            m.append(float(np.mean(v)) if v else np.nan)
            e.append(float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan)
            n.append(len(v))
        return m, e, n

    def b2p(r: dict) -> bool:
        """A well-separated bare->persona grid row."""
        return (
            r["wellsep"]
            and r["arm"] in ARMS
            and _src(r["pair_id"]) == "bare"
            and _tgt(r["pair_id"]) == "persona"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    summary: dict = {}
    x = np.arange(1, n_windows + 1)

    # -- Panel A: is the TARGET persona installed, and where? ------------------
    panel_a = [
        (
            "steered, all 28 layers",
            lambda r: (
                b2p(r) and r["arm"] == "steered" and r["side"] == "b" and r["cell"] == "joint_all"
            ),
            "#1b6ca8",
            "o",
            "-",
        ),
        (
            "steered, layers 14-20",
            lambda r: (
                b2p(r) and r["arm"] == "steered" and r["side"] == "b" and r["cell"] == "joint_mid"
            ),
            "#6aa9d0",
            "o",
            "-",
        ),
        (
            "steered, layer 16 only",
            lambda r: b2p(r) and r["arm"] == "steered" and r["side"] == "b" and r["cell"] == "L16",
            "#b3d3e8",
            "o",
            "-",
        ),
        (
            "shuffled-donor null (all cells)",
            lambda r: b2p(r) and r["arm"] == "null" and r["side"] == "b",
            "#999999",
            "x",
            "--",
        ),
        (
            "native: persona prefix actually present",
            lambda r: r["arm"] == "native" and r["pair_id"].startswith("persona__"),
            "#c1440e",
            "s",
            "-",
        ),
    ]
    for label, pred, color, marker, ls in panel_a:
        m, e, n = series(pred)
        axes[0].errorbar(
            x, m, yerr=e, marker=marker, ls=ls, capsize=3, color=color, label=f"{label} (n={n[0]})"
        )
        summary[f"target/{label}"] = {"mean_by_window": m, "sem_by_window": e, "n_by_window": n}
    axes[0].set_title(
        "A. Is the pirate persona installed, and where in the answer?\n"
        "bare-context run patched toward the persona context"
    )
    axes[0].set_ylabel("judge score (0-100), pirate-persona rubric")

    # -- Panel B: erasure of the source vs installation of the target ---------
    panel_b = [
        (
            "SOURCE register, null patch",
            lambda r: (
                b2p(r) and r["arm"] == "null" and r["side"] == "a" and r["cell"] == "joint_all"
            ),
            "#7a7a7a",
            "--",
        ),
        (
            "SOURCE register, real patch",
            lambda r: (
                b2p(r) and r["arm"] == "steered" and r["side"] == "a" and r["cell"] == "joint_all"
            ),
            "#2b2b2b",
            "-",
        ),
        (
            "TARGET register, null patch",
            lambda r: (
                b2p(r) and r["arm"] == "null" and r["side"] == "b" and r["cell"] == "joint_all"
            ),
            "#9fc4dd",
            "--",
        ),
        (
            "TARGET register, real patch",
            lambda r: (
                b2p(r) and r["arm"] == "steered" and r["side"] == "b" and r["cell"] == "joint_all"
            ),
            "#1b6ca8",
            "-",
        ),
    ]
    for label, pred, color, ls in panel_b:
        m, e, n = series(pred)
        axes[1].errorbar(
            x, m, yerr=e, marker="o", ls=ls, capsize=3, color=color, label=f"{label} (n={n[0]})"
        )
        summary[f"decompose/{label}"] = {"mean_by_window": m, "sem_by_window": e, "n_by_window": n}
    axes[1].set_title(
        "B. The patch removes the old context far more than\nit installs the new one (all 28 layers)"
    )
    axes[1].set_ylabel("judge score (0-100), each register's own rubric")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["window 1\n(first third)", "window 2\n(middle)", "window 3\n(last)"][:n_windows]
        )
        ax.set_xlabel("position in the model's own answer")
        ax.set_ylim(-4, 108)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("[position-judge] figure -> %s", out_png)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_2094/position_judge"))
    ap.add_argument("--cache", type=Path, default=Path("data/issue_2094/position_judge_cache"))
    ap.add_argument("--eval-root", type=Path, default=Path("eval_results/issue_2094"))
    ap.add_argument(
        "--figure", type=Path, default=Path("figures/issue_2094/position_judge_decay.png")
    )
    ap.add_argument("--n-windows", type=int, default=3)
    ap.add_argument(
        "--anchor-draws", type=int, default=10, help="native draws per context (max 10)"
    )
    ap.add_argument("--pilot", action="store_true", help="score ~24 spread units and stop")
    ap.add_argument("--figure-only", action="store_true", help="re-render from window_scores.jsonl")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    scores_path = args.out_dir / "window_scores.jsonl"
    if args.figure_only:
        rows = [json.loads(x) for x in scores_path.read_text().splitlines() if x.strip()]
        summary = make_figure(rows, args.figure, args.n_windows)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("[phase=position_judge_figure_done] %s", args.figure)
        return

    units = build_units(args.cache, args.n_windows, args.anchor_draws, args.eval_root)
    logger.info("[position-judge] %d units == %d judge calls", len(units), len(units))
    if args.pilot:
        stride = max(len(units) // 24, 1)
        units = units[::stride][:24]
        logger.info("[position-judge] PILOT: %d units", len(units))

    rows, telemetry = score_units(units, args, "pilot" if args.pilot else "full")
    out = args.out_dir / ("pilot_scores.jsonl" if args.pilot else "window_scores.jsonl")
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "judge_model": args.judge_model,
        "max_tokens": JUDGE_MAX_TOKENS,
        "n_windows": args.n_windows,
        "anchor_draws": args.anchor_draws,
        "cells": list(CELLS),
        "setting": SETTING,
        "telemetry": telemetry,
        **as_metadata_dict(git_provenance()),
    }
    (args.out_dir / ("pilot_meta.json" if args.pilot else "meta.json")).write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    logger.info("[phase=position_judge_scored] %d rows -> %s", len(rows), out)
    if not args.pilot:
        summary = make_figure(rows, args.figure, args.n_windows)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("[phase=position_judge_done] %s", args.figure)


if __name__ == "__main__":
    main()
