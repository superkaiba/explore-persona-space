"""Issue #1739 Batch-API judging CLI (round B).

Reads labeling rollout JSONs (``generation.generate_labeling`` output),
dispatches graded 0-100 judging (N=3 draws, temperature 1.0, max_tokens=400 —
constants.py pins) through the sanctioned Batch client, and writes per-rollout
score tallies with the content-drop vs transport-loss split.

Hallucination runs the three-way protocol: alias-list match splits ``correct``
first; only incorrect answers are judged fabricated-vs-abstained.

Usage:
    uv run python scripts/issue1739_judge.py --behavior sycophancy \
        --rollout-dir raw_completions/issue_1739/labeling/sycophancy \
        --out-dir eval_results/issue_1739/judge/sycophancy [--limit 8] [--dry-run]
"""

from explore_persona_space.orchestrate.env import load_dotenv

# Credentials bind BEFORE any heavy import (#847; ANTHROPIC_API_KEY).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.experiments.issue_1739 import dv_build, judging  # noqa: E402
from explore_persona_space.experiments.issue_1739.constants import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    N_JUDGE_DRAWS,
)
from explore_persona_space.experiments.issue_1739.corpus_registry import BEHAVIORS  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ},
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> int:
    """Parse args, dispatch the judge set, write the scores JSON."""
    parser = argparse.ArgumentParser(description="Issue #1739 graded judging (round B)")
    parser.add_argument("--behavior", choices=list(BEHAVIORS), required=True)
    parser.add_argument("--rollout-dir", required=True, help="labeling rollout JSON dir")
    parser.add_argument("--out-dir", required=True, help="eval_results output dir")
    parser.add_argument("--cache-dir", default=None, help="judge cache dir (fresh per run)")
    parser.add_argument("--inputs-dir", default="data/issue_1739/inputs")
    parser.add_argument("--n-draws", type=int, default=N_JUDGE_DRAWS)
    parser.add_argument("--max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=JUDGE_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None, help="smoke slice cap (rollout files)")
    parser.add_argument(
        "--dv-out-root",
        default="eval_results/issue_1739",
        help="root for the per-behavior DV dataset (dv_build.write_dv_dataset; "
        "the fits phase reads dv_dataset/<behavior>/labeling.json under it)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--threshold-base",
        type=int,
        default=None,
        help="sync-vs-batch crossover override; 0 forces the Batch API path",
    )
    args = parser.parse_args()

    rollout_paths = sorted(
        p for p in Path(args.rollout_dir).glob("*.json") if not p.name.startswith("_")
    )
    if args.limit is not None:
        rollout_paths = rollout_paths[: args.limit]
    if not rollout_paths:
        print(f"no rollout JSONs under {args.rollout_dir}", file=sys.stderr)
        return 2
    payloads = [json.loads(p.read_text()) for p in rollout_paths]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fresh per-run cache dir by default: a REUSED cache dir collapses the
    # n_draws repeats to one cached score (graded_judge module docstring).
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else out_dir / f"judge_cache_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    )

    payload_out: dict = {
        "behavior": args.behavior,
        "n_rollout_files": len(rollout_paths),
        "n_draws": args.n_draws,
        "judge_temperature": args.temperature,
        "judge_max_tokens": args.max_tokens,
        "judge_model": JUDGE_MODEL,
        "dry_run": args.dry_run,
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    contexts_meta = {p["context_id"]: p for p in payloads}
    if args.behavior == "hallucination":
        correct_map, judge_items = judging.split_hallucination_items(payloads)
        result = judging.judge_items_graded(
            judge_items,
            judging.HALLU_ABSTAIN_RUBRIC,
            cache_dir=cache_dir,
            save_raw=out_dir / "judge_raw_abstain.json",
            n_draws=args.n_draws,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            threshold_base=args.threshold_base,
        )
        tallies = judging.judge_tallies(result)
        three_way = {
            item_id: judging.three_way_classify(is_correct, result.scores.get(item_id))
            for item_id, is_correct in correct_map.items()
        }
        counts: dict[str, int] = {}
        for label in three_way.values():
            counts[label] = counts.get(label, 0) + 1
        payload_out.update(
            {
                "rubric": "hallucination_three_way",
                "n_alias_correct": sum(correct_map.values()),
                "n_judged": len(judge_items),
                "three_way": three_way,
                "three_way_counts": counts,
                "abstain_judge": tallies,
            }
        )
        dv_rows = dv_build.build_three_way_dv(three_way)
        # Attach the staged-context metadata (group_key drives the fits-side
        # group folds; build_three_way_dv itself carries labels only).
        for row in dv_rows:
            meta = contexts_meta.get(row["context_id"], {})
            for key in ("behavior", "split", "rung", "group_key"):
                if key in meta:
                    row.setdefault(key, meta[key])
    else:
        eval_prompt = judging.load_trait_rubric(args.behavior, inputs_dir=args.inputs_dir)
        items = [
            (
                judging.rollout_item_id(p["context_id"], int(p["rollout_k"])),
                p["query"],
                p["completion"],
            )
            for p in payloads
        ]
        result = judging.judge_items_graded(
            items,
            eval_prompt,
            cache_dir=cache_dir,
            save_raw=out_dir / "judge_raw_trait.json",
            n_draws=args.n_draws,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            dry_run=args.dry_run,
            threshold_base=args.threshold_base,
        )
        payload_out.update({"rubric": "trait_eval_prompt", **judging.judge_tallies(result)})
        dv_rows = dv_build.build_labeling_dv(
            dict(result.scores),
            n_draws=args.n_draws,
            per_item_transport_losses=dict(getattr(result, "per_item_transport_losses", {}) or {}),
            contexts_meta=contexts_meta,
        )

    # Round C2 wiring: the fits phase consumes dv_dataset/<behavior>/
    # labeling.json — write it here (the judge output is its only input).
    dv_path = dv_build.write_dv_dataset(
        dv_rows,
        out_root=args.dv_out_root,
        behavior=args.behavior,
        judge_payload_meta={
            key: payload_out[key]
            for key in (
                "n_rollout_files",
                "n_draws",
                "judge_temperature",
                "judge_max_tokens",
                "judge_model",
                "dry_run",
                "rubric",
            )
            if key in payload_out
        },
        git_commit=payload_out["git_commit"],
    )

    scores_path = out_dir / "labeling_scores.json"
    tmp = scores_path.with_name(scores_path.name + ".tmp")
    tmp.write_text(json.dumps(payload_out, indent=1))
    os.replace(tmp, scores_path)
    n_scores = len(payload_out.get("scores", payload_out.get("three_way", {})))
    print(
        json.dumps(
            {
                "scores_path": str(scores_path),
                "dv_dataset_path": str(dv_path),
                "n_items": n_scores,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
