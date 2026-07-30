"""Re-judge #1773's truncation-dropped validation items at a larger response budget.

#1773's validation batteries dropped 46-66% of items as parse errors under a
400-token cap on a reason-then-answer rubric spanning six windows per item --
the `.claude/rules/llm-judging.md` rule-23 truncation-censoring signature. The
body names a re-judge at a larger budget against a FRESH cache as the standard
remedy and the top follow-up. This driver runs exactly that.

Scope: REPORT-ONLY. It writes a parallel result tree under
``eval_results/issue_1773/rejudge_<N>tok/`` and never touches the committed
Phase-4 artifacts or the SEARCH-INDEX-ONLY verdict in the task body.

Reuse: item rendering (``_render_item``), the three battery system prompts, and
``score_results`` come from ``issue1773_validate`` unchanged; dispatch goes
through ``eval.judge_dispatch.dispatch_judge_items`` (never a hand-rolled call
site). A fresh ``checkpoint_dir`` is mandatory -- the rubric-keyed judge cache
deliberately excludes ``max_tokens``, so a raised budget would otherwise be
served the truncated entries this round exists to replace (rule 24).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1773_common as CM  # noqa: E402
import issue1773_validate as V  # noqa: E402

BATTERIES = ("detection", "fuzzing", "discrimination")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_items(val_dir: Path) -> list[dict]:
    items: list[dict] = []
    for p in sorted(val_dir.glob("val_items.shard*.jsonl")):
        items.extend(CM.iter_jsonl(p))
    return items


def _dropped_ids(val_dir: Path, items: list[dict]) -> tuple[dict[str, dict], set[str]]:
    """Prior results plus the custom_ids whose prior return was an error dict."""
    prior: dict[str, dict] = {}
    for battery in BATTERIES:
        p = val_dir / f"val_results_{battery}.json"
        if p.exists():
            prior.update(json.loads(p.read_text()))
    dropped = set()
    for it in items:
        res = prior.get(it["custom_id"])
        if not isinstance(res, dict) or res.get("error"):
            dropped.add(it["custom_id"])
    return prior, dropped


def rejudge(args: argparse.Namespace) -> int:
    from explore_persona_space.eval.judge_dispatch import (
        dispatch_judge_items,
        graded_temperature,
        keep_raw_judge_text,
    )

    src_val = args.src_root / "validation"
    items = _load_items(src_val)
    prior, dropped = _dropped_ids(src_val, items)
    _log(f"[rejudge] val_items={len(items)} prior_results={len(prior)} dropped={len(dropped)}")

    todo = [it for it in items if it["custom_id"] in dropped]
    if args.limit:
        todo = todo[: args.limit]
    by_b: dict[str, list[dict]] = {b: [] for b in BATTERIES}
    for it in todo:
        by_b[it["battery"]].append(it)
    _log(
        f"[rejudge] re-dispatching { {b: len(v) for b, v in by_b.items()} } "
        f"at max_tokens={args.max_tokens} (prior {V.VAL_MAX_TOKENS})"
    )

    systems = {
        "detection": V.DETECTION_SYSTEM,
        "fuzzing": V.FUZZING_SYSTEM,
        "discrimination": V.DISCRIMINATION_SYSTEM,
    }
    out_val = args.out_root / "validation"
    out_val.mkdir(parents=True, exist_ok=True)

    recovered: dict[str, dict] = {}
    for battery in BATTERIES:
        group = by_b[battery]
        if not group:
            continue
        jitems = [
            (it["custom_id"], f"val:{battery}:rejudge", "", V._render_item(it)) for it in group
        ]
        _log(f"[rejudge] {battery}: dispatching {len(jitems)} items")
        with graded_temperature(CM.JUDGE_TEMPERATURE), keep_raw_judge_text():
            res = dispatch_judge_items(
                jitems,
                judge_system_prompt=systems[battery],
                max_tokens=args.max_tokens,
                # FRESH cache: the rubric-keyed key excludes max_tokens (rule 24).
                checkpoint_dir=args.work / f"rejudge_{args.max_tokens}" / f"val_{battery}",
                force_sync=args.force_sync,
                dry_run=args.dry_run,
            )
        recovered.update(res)
        n_err = sum(1 for r in res.values() if isinstance(r, dict) and r.get("error"))
        _log(
            f"[rejudge] {battery}: returned {len(res)} | still-error {n_err} "
            f"({n_err / max(len(res), 1):.1%})"
        )

    if args.dry_run:
        return 0

    # Merge: recovered entries override prior error dicts; survivors kept as-is.
    merged_counts = Counter()
    for battery in BATTERIES:
        ids = {it["custom_id"] for it in items if it["battery"] == battery}
        out: dict[str, dict] = {}
        for cid in ids:
            new = recovered.get(cid)
            if isinstance(new, dict) and not new.get("error"):
                out[cid] = new
                merged_counts[f"{battery}_recovered"] += 1
            elif cid in prior:
                out[cid] = prior[cid]
        (out_val / f"val_results_{battery}.json").write_text(json.dumps(out))
    _log(f"[rejudge] merged: {dict(merged_counts)}")

    # score_results reads val_items shards + kappa_report from out_root.
    for p in sorted(src_val.glob("val_items.shard*.jsonl")):
        shutil.copy2(p, out_val / p.name)
    (args.out_root / "labels").mkdir(parents=True, exist_ok=True)
    kp = args.src_root / "labels" / "kappa_report.json"
    if kp.exists():
        shutil.copy2(kp, args.out_root / "labels" / "kappa_report.json")

    V.score_results(SimpleNamespace(out_root=args.out_root, mock_results=None))

    scorecard = json.loads((out_val / "scorecard.json").read_text())
    (out_val / "rejudge_meta.json").write_text(
        json.dumps(
            {
                "prior_max_tokens": V.VAL_MAX_TOKENS,
                "rejudge_max_tokens": args.max_tokens,
                "n_val_items": len(items),
                "n_dropped_prior": len(dropped),
                "n_redispatched": len(todo),
                "recovered": dict(merged_counts),
                "gating": "REPORT-ONLY - no verdict flip, no trust-lattice gate applied",
                **CM.repro_meta(),
            },
            indent=1,
        )
    )
    _log(
        "[rejudge] aggregates: "
        + json.dumps({k: round(v["mean"], 4) for k, v in scorecard["aggregates"].items()})
    )
    _log("[rejudge] drops after re-judge: " + json.dumps(scorecard.get("drops", {})))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, default=CM.OUT_EVAL)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=CM.WORK_DEFAULT)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="0 = all dropped items")
    ap.add_argument("--force-sync", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.out_root is None:
        args.out_root = CM.OUT_EVAL / f"rejudge_{args.max_tokens}tok"
    return rejudge(args)


if __name__ == "__main__":
    raise SystemExit(main())
