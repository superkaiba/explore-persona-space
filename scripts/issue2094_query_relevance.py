"""Issue #2094: does a patched answer still ANSWER THE QUERY?

The matched-query setting scores two things per draw and NEITHER of them reads
the answer against the question:

- the coherence rubric is FORM-ONLY and says so verbatim ("EXPLICITLY IGNORE
  correctness, safety, relevance, completeness, style, and length -- a fluent
  refusal, a fluent off-topic answer, or a fluent one-liner are all fully
  coherent", ``bank.COHERENCE_RUBRIC``);
- the only behavior rubric that runs there is the PREFIX one
  (``bank.SETTING_RUBRIC_KINDS["matched_query"] == ("prefix",)``), which scores
  register/persona carry-over and is explicitly told NOT to judge whether the
  content is correct or useful.

So a patch that deletes the source persona AND derails the answer off the
question scores identically to a clean persona swap on every reported metric.
That matters here: the transfer is ~3.3x more erasure than installation
(writeup Result 5), so "the pirate is gone" is most of what F_beh counts, and
whether the task survived is unmeasured.

``bank.f_query_rubric`` is exactly the right instrument and already exists, but
it is built as a PAIR (query_a vs query_b) for the delta contrast, and
matched-query holds the query FIXED -- ``bank.rubric_pair`` asserts
``query_a != query_b`` and would fire. This script runs it SINGLE-ARM instead:
score each draw against the pair's shared query, and compare the patched and
shuffled-donor-null arms to the unpatched anchors on the same queries.

Zero GPU: judge calls only, over completions already banked on the data repo.

Usage:
  uv run python scripts/issue2094_query_relevance.py \
      --eval-root eval_results/issue_2094 \
      --out-dir eval_results/issue_2094/query_relevance \
      --figures-out figures/issue_2094
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # credentials + shared-VM thread caps BEFORE any heavy import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402  (must follow load_dotenv: BLAS pools freeze at import)

from explore_persona_space.eval.graded_judge import judge_graded  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate.hub import (  # noqa: E402
    list_hf_files_under_path,
    stage_hub_file,
)

logger = logging.getLogger("issue2094.query_relevance")

DATASET_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2094_singlepos"
SETTING = "matched_query"
ARMS = ("steered", "null")

# The completions live under TWO roots: the main grid, and the fu2 span-slot
# follow-up (qtext / pspan_* were generated later and never moved). Enumerating
# the shard names off the roots -- rather than hardcoding a slot list -- is what
# makes "all cells" mean all cells: a slot present on the Hub but missing from a
# hand-written tuple is exactly how the first pass silently skipped qtext, the
# highest-traversal slot in the grid.
GRID_ROOTS = (
    "raw_completions/grid",
    "raw_completions/fu2_span_slots/rollouts",
)
JOINT_CELLS = ("joint_all", "joint_mid")
# Single-rationale reason-then-score rubric -> the llm-judging.md rule-23 floor.
JUDGE_MAX_TOKENS = 1024

REASON_THEN_SCORE = (
    "First give a one-sentence justification, then the integer score.\n"
    'Reply as JSON: {"reasoning": "<one sentence>", "score": <integer 0-100>}'
)


def query_eval_prompt(query_id: str) -> str:
    """The single-arm query-relevance instrument for one query id."""
    return (
        f"{BANK.f_query_rubric(query_id)}\n\n[RESPONSE]\n{{answer}}\n[/RESPONSE]\n\n"
        f"{REASON_THEN_SCORE}"
    )


def item_id_for(*parts: str) -> str:
    """Stable short id matching the parent's ITEM_ID_RE ([A-Za-z0-9-]{1,53})."""
    return "q" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def query_of(context_id: str) -> str:
    """'persona__q3' -> 'q3'."""
    return context_id.split("__")[1]


def prefix_of(context_id: str) -> str:
    """'persona__q3' -> 'persona'."""
    return context_id.split("__")[0]


@dataclass(frozen=True)
class Unit:
    item_id: str
    answer: str
    query_id: str
    arm: str
    slot: str
    cell: str
    pair_id: str
    cap_hit: bool


def _stage(cache: Path, rel: str) -> list[dict]:
    p = stage_hub_file(
        repo_id=DATASET_REPO,
        path_in_repo=f"{HF_PREFIX}/{rel}",
        target=cache / Path(rel).name,
        repo_type="dataset",
        revision="main",
    )
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def discover_shards(scope: str) -> list[tuple[str, str, str, str]]:
    """-> [(rel_path, slot, layer_variant, arm)] for every replace-dose shard in scope.

    ``scope``: 'joint' = the joint_all / joint_mid cells only; 'single' = the
    per-layer ladder only; 'all' = both. Listing goes through the hub helper's
    SCOPED + RETRIED server-side walk -- a full-repo listing of this ~1M-file
    data repo wedges, and a bare tree call is the #920 un-retried-cursor class.
    """
    assert scope in ("joint", "single", "all"), scope
    api = HfApi()
    out: list[tuple[str, str, str, str]] = []
    for root in GRID_ROOTS:
        # Scoped + RETRIED listing: a bare list_repo_tree re-creates the #920
        # class (huggingface_hub retries only 429 on follow-up cursor pages, so
        # a transient 504 mid-pagination silently truncates or fails the walk).
        for path in list_hf_files_under_path(
            api, DATASET_REPO, f"{HF_PREFIX}/{root}", repo_type="dataset", revision="main"
        ):
            name = path.split("/")[-1]
            if not name.startswith("shard_") or not name.endswith(".jsonl"):
                continue
            parts = name.removeprefix("shard_").removesuffix(".jsonl").split("__")
            if len(parts) != 5:
                continue
            slot, lv, dose, _vt, arm = parts
            if dose != "replace" or arm not in ARMS:
                continue
            is_joint = lv in JOINT_CELLS
            if scope == "joint" and not is_joint:
                continue
            if scope == "single" and is_joint:
                continue
            out.append((f"{root}/{name}", slot, lv, arm))
    assert out, f"no replace-dose shards discovered for scope={scope}"
    return sorted(out)


def build_units(cache: Path, anchor_draws: int, scope: str) -> list[Unit]:
    """Patched + null draws for every discovered cell, plus unpatched anchors.

    Every unit is scored against the query the model was actually asked. In
    matched-query the pair's two contexts share that query by construction, so
    the patched draw and both its anchors are judged on the SAME question -- the
    comparison the pair-form rubric cannot express.
    """
    units: list[Unit] = []
    seen_ctx: set[str] = set()
    shards = discover_shards(scope)
    logger.info("[query-relevance] scope=%s: %d shards discovered", scope, len(shards))
    for rel, slot, cell, arm in shards:
        for r in _stage(cache, rel):
            if r["setting"] != SETTING:
                continue
            qa, qb = query_of(r["context_a"]), query_of(r["context_b"])
            # matched-query means the pair holds the query fixed; a mismatch
            # would make the single-arm read ill-defined.
            assert qa == qb, (r["pair_id"], qa, qb)
            seen_ctx.update((r["context_a"], r["context_b"]))
            if not r["text"].strip():
                # An EMPTY completion is a real outcome of a destructive patch,
                # not a missing row -- record it as score 0 rather than dropping
                # it, or the mean is taken over survivors only.
                units.append(
                    Unit(
                        item_id_for(arm, slot, cell, r["pair_id"]),
                        "",
                        qa,
                        arm,
                        slot,
                        cell,
                        r["pair_id"],
                        bool(r["cap_hit"]),
                    )
                )
                continue
            units.append(
                Unit(
                    item_id_for(arm, slot, cell, r["pair_id"]),
                    r["text"],
                    qa,
                    arm,
                    slot,
                    cell,
                    r["pair_id"],
                    bool(r["cap_hit"]),
                )
            )
    # Unpatched anchors: the no-intervention reference on the same queries.
    by_ctx: dict[str, list[dict]] = defaultdict(list)
    for r in _stage(cache, "raw_completions/anchors/anchors.jsonl"):
        if r["context_id"] in seen_ctx:
            by_ctx[r["context_id"]].append(r)
    for ctx, rows in sorted(by_ctx.items()):
        for r in sorted(rows, key=lambda x: x["draw"])[:anchor_draws]:
            if not r["text"].strip():
                continue
            units.append(
                Unit(
                    item_id_for("native", ctx, str(r["draw"])),
                    r["text"],
                    query_of(ctx),
                    "native",
                    f"native-{prefix_of(ctx)}",
                    "native",
                    f"{ctx}#d{r['draw']}",
                    bool(r["cap_hit"]),
                )
            )
    return units


def score_units(units: list[Unit], args: argparse.Namespace) -> tuple[list[dict], dict]:
    """One judge wave per query id (one rubric each); return (rows, telemetry).

    An EMPTY completion never reaches the judge -- there is nothing to score and
    the judge's answer on an empty string is arbitrary -- but it is a genuine
    outcome of a destructive patch, so it is recorded at 0 (maximally
    non-responsive) rather than dropped. Dropping would take the mean over
    survivors and hide exactly the failure this read exists to find.
    """
    rows: list[dict] = []
    telemetry: dict = {}
    for u in (x for x in units if not x.answer.strip()):
        rows.append(
            {
                "item_id": u.item_id,
                "query_id": u.query_id,
                "arm": u.arm,
                "slot": u.slot,
                "cell": u.cell,
                "pair_id": u.pair_id,
                "cap_hit": u.cap_hit,
                "score": 0.0,
                "n_chars": 0,
                "empty_completion": True,
            }
        )
    scoreable = [u for u in units if u.answer.strip()]
    telemetry["_empty_completions"] = len(units) - len(scoreable)
    for query_id in sorted({u.query_id for u in scoreable}):
        wave = [u for u in scoreable if u.query_id == query_id]
        logger.info("[query-relevance] wave %s: %d items", query_id, len(wave))
        res = judge_graded(
            [(u.item_id, "", u.answer) for u in wave],
            query_eval_prompt(query_id),
            n_draws=args.judge_draws,
            cache_dir=args.cache / "judge_qrel" / query_id,
            save_raw=args.out_dir / f"raw.qrel.{query_id}.json",
            judge_model=args.judge_model,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        telemetry[query_id] = {
            "n_items": len(wave),
            "n_total_draws": res.n_total_draws,
            "n_dropped_draws": res.n_dropped_draws,
            "n_transport_lost_draws": res.n_transport_lost_draws,
            "n_truncation_dropped_draws": res.n_truncation_dropped_draws,
            "n_refusal_draws": res.n_refusal_draws,
            "n_api_refusal_draws": res.n_api_refusal_draws,
            "stop_reason_tally": dict(res.stop_reason_tally),
        }
        logger.info("[query-relevance] telemetry %s: %s", query_id, telemetry[query_id])
        for u in wave:
            rows.append(
                {
                    "item_id": u.item_id,
                    "query_id": u.query_id,
                    "arm": u.arm,
                    "slot": u.slot,
                    "cell": u.cell,
                    "pair_id": u.pair_id,
                    "cap_hit": u.cap_hit,
                    "score": res.scores.get(u.item_id),
                    "n_chars": len(u.answer),
                    "empty_completion": False,
                }
            )
    return rows, telemetry


def summarize(rows: list[dict]) -> dict:
    """-> {'<slot>|<cell>|<arm>': {n, mean, sd}} plus the pooled native reference."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        s = r["score"]
        if s is None:
            continue
        key = "native" if r["arm"] == "native" else f"{r['slot']}|{r['cell']}|{r['arm']}"
        buckets[key].append(float(s))
    return {
        k: {
            "n": len(v),
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        }
        for k, v in sorted(buckets.items())
    }


SLOT_LABELS = {
    "ce": "context-end",
    "pe": "prefix-end",
    "cm2": "2nd-to-last",
    "cm3": "3rd-to-last",
    "l3j": "last-3 joint",
    "qspan": "query span",
    "qtext": "query text",
    "pspan_tmpl": "prefix span\n(+template)",
    "pspan_text": "prefix span\n(no template)",
}
CELL_LABELS = {"joint_all": "all 28 layers", "joint_mid": "layers 14-20"}


def make_figure(summary: dict, out_png: Path, scope: str) -> None:
    """Patched vs null vs unpatched query-relevance over every scored cell.

    Joint cells get one bar group each; a single-layer ladder is drawn as a
    per-layer line rather than 56 unreadable bar groups.
    """
    joint = [k for k in summary if k != "native" and k.split("|")[1] in JOINT_CELLS]
    single = [k for k in summary if k != "native" and k.split("|")[1] not in JOINT_CELLS]
    # Only draw panels that have cells. A fixed 2-panel layout renders a BLANK
    # axes whenever the scope holds one kind (scope=single did exactly that).
    # The joint panel is sized to its GROUP COUNT: a fixed width collides the
    # two-line x labels the moment the cell set grows past ~8.
    n_joint = len({(k.split("|")[0], k.split("|")[1]) for k in joint})
    widths = ([max(9.2, 1.15 * n_joint + 2.0)] if joint else []) + ([7.0] if single else [])
    assert widths, "no cells to plot"
    fig, axes = plt.subplots(
        1, len(widths), figsize=(sum(widths), 5.2), gridspec_kw={"width_ratios": widths}
    )
    axes = list(np.atleast_1d(axes))
    nat = summary.get("native")
    ax_joint = axes.pop(0) if joint else None
    ax_single = axes.pop(0) if single else None

    if ax_joint is not None:
        ax = ax_joint
        labels, groups = [], []
        for slot, cell in sorted({(k.split("|")[0], k.split("|")[1]) for k in joint}):
            s = summary.get(f"{slot}|{cell}|steered")
            if not s:
                continue
            labels.append(f"{SLOT_LABELS.get(slot, slot)}\n{CELL_LABELS.get(cell, cell)}")
            groups.append((s, summary.get(f"{slot}|{cell}|null")))
        x = np.arange(len(labels))
        w = 0.36
        for i, (col, off, lab) in enumerate(
            (("#1b6ca8", -w / 2, "real patch"), ("#9fc4dd", w / 2, "shuffled-donor null"))
        ):
            vals = [g[i]["mean"] if g[i] else np.nan for g in groups]
            errs = [(g[i]["sd"] / max(np.sqrt(g[i]["n"]), 1.0)) if g[i] else np.nan for g in groups]
            ax.bar(x + off, vals, w, yerr=np.maximum(0.0, errs), color=col, label=lab, capsize=3)
        ax.set_xticks(x, labels, fontsize=8, rotation=30, ha="right")
        ax.set_title("joint (multi-layer) patches", fontsize=10)
        ax.legend(fontsize=9, loc="lower right")
        if nat and labels:
            ax.annotate(
                f"unpatched, same queries: {nat['mean']:.1f}  (n={nat['n']})",
                (len(labels) - 0.5, nat["mean"]),
                textcoords="offset points",
                xytext=(-4, 6),
                ha="right",
                fontsize=8.5,
                color="#c1440e",
            )

    if single:
        ax2 = ax_single
        for slot, col in (("ce", "#1b6ca8"), ("pe", "#7a3fa8")):
            lv = sorted(
                int(k.split("|")[1][1:])
                for k in single
                if k.startswith(f"{slot}|") and k.endswith("|steered")
            )
            if not lv:
                continue
            ax2.plot(
                lv,
                [summary[f"{slot}|L{i}|steered"]["mean"] for i in lv],
                "-o",
                ms=3,
                color=col,
                label=f"{SLOT_LABELS[slot]}, real patch",
            )
            nk = [f"{slot}|L{i}|null" for i in lv]
            if all(k in summary for k in nk):
                ax2.plot(
                    lv,
                    [summary[k]["mean"] for k in nk],
                    "--",
                    lw=1.0,
                    color=col,
                    alpha=0.55,
                    label=f"{SLOT_LABELS[slot]}, null",
                )
        ax2.set_xlabel("patched layer")
        ax2.set_title("single-layer patches", fontsize=10)
        ax2.legend(fontsize=8, loc="lower right")
        if nat and ax_joint is None:
            ax2.annotate(
                f"unpatched, same queries: {nat['mean']:.1f}  (n={nat['n']})",
                (0.99, nat["mean"]),
                xycoords=("axes fraction", "data"),
                textcoords="offset points",
                xytext=(-4, 6),
                ha="right",
                fontsize=8.5,
                color="#c1440e",
            )

    for a in [a for a in (ax_joint, ax_single) if a is not None]:
        if nat:
            a.axhline(nat["mean"], color="#c1440e", ls="--", lw=1.4, zorder=0)
        a.set_ylim(0, 105)
        a.set_ylabel("query-relevance score (0-100)")
        a.grid(axis="y", alpha=0.25, lw=0.5)
    fig.suptitle(
        "Does the patched answer still answer the question?\n"
        f"matched query, full-state patch, scope={scope} (every cell); judged single-arm against "
        "the pair's shared query — the read F_beh and coherence both omit",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=Path("eval_results/issue_2094"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--figures-out", type=Path, default=Path("figures/issue_2094"))
    ap.add_argument("--cache", type=Path, default=Path("data/issue_2094/qrel_cache"))
    ap.add_argument("--anchor-draws", type=int, default=2)
    ap.add_argument("--judge-draws", type=int, default=3)
    ap.add_argument("--judge-model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--scope", choices=("joint", "single", "all"), default="joint")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    units = build_units(args.cache, args.anchor_draws, args.scope)
    assert units, "no units built — check the HF prefix / setting filter"
    logger.info(
        "[query-relevance] %d units over %d queries",
        units and len(units),
        len({u.query_id for u in units}),
    )

    rows, telemetry = score_units(units, args)
    summary = summarize(rows)

    with (args.out_dir / "qrel_rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (args.out_dir / "qrel_summary.json").write_text(
        json.dumps(
            {
                "setting": SETTING,
                "dose": "replace",
                "scope": args.scope,
                "judge_model": args.judge_model,
                "judge_draws": args.judge_draws,
                "judge_max_tokens": JUDGE_MAX_TOKENS,
                "anchor_draws": args.anchor_draws,
                "summary": summary,
                "telemetry": telemetry,
            },
            indent=1,
        )
    )
    out_png = args.figures_out / f"query_relevance_{args.scope}.png"
    make_figure(summary, out_png, args.scope)
    for k, v in summary.items():
        logger.info(
            "[query-relevance] %-28s n=%3d mean=%6.2f sd=%5.2f", k, v["n"], v["mean"], v["sd"]
        )
    print(f"[phase=done] {out_png} + {args.out_dir}/qrel_summary.json")


if __name__ == "__main__":
    main()
