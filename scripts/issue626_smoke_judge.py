"""Live smoke driver for the batch-aware judge dispatch (task #626).

Loads question+completion rows from a stored eval JSON, judges them through
the REAL ``judge_completions_batch`` entry point, prints the RoutingDecision,
and asserts the expected path + n_errors == 0.

Sync smoke (plan §6.4):    --max-n 48 (default threshold -> sync)
Forced-batch canary (§6.4b): --max-n 5 --threshold-base 1 (-> batch, ~$0.03)
"""

import argparse
import json
import logging
from pathlib import Path

from explore_persona_space.eval.batch_judge import judge_completions_batch
from explore_persona_space.orchestrate.env import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="stored eval JSON with question+completion rows"
    )
    parser.add_argument("--max-n", type=int, required=True, help="number of rows to judge")
    parser.add_argument("--threshold-base", type=int, default=2_000)
    parser.add_argument("--out", required=True, help="save_raw output path")
    parser.add_argument(
        "--persona",
        default=None,
        help="optional row filter on the 'persona' field (e.g. helpful_assistant — some "
        "alien-persona EM completions trip content-based judge refusals, which are an "
        "environment hazard, not a dispatch property)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()

    rows = json.loads(Path(args.source).read_text())
    assert isinstance(rows, list) and rows, f"expected a non-empty list of rows in {args.source}"
    if args.persona is not None:
        rows = [r for r in rows if r.get("persona") == args.persona]
        assert rows, f"no rows with persona={args.persona!r} in {args.source}"
    completion_map: dict[str, list[str]] = {}
    for row in rows[: args.max_n]:
        completion_map.setdefault(row["question"], []).append(row["completion"])
    n = sum(len(v) for v in completion_map.values())
    assert n == args.max_n, (n, args.max_n)

    out = Path(args.out)
    cache_dir = out.parent / f"{out.stem}_cache"
    result = judge_completions_batch(
        completions={"smoke": completion_map},
        cache_dir=cache_dir,
        save_raw=out,
        threshold_base=args.threshold_base,
    )

    raw = json.loads(out.read_text())
    routing = raw["routing"]
    print(f"RoutingDecision: {json.dumps(routing, indent=2)}")
    print(f"per_persona: {json.dumps(result, indent=2)}")

    expected_path = "batch" if args.max_n >= args.threshold_base else "sync"
    assert routing["path"] == expected_path, (routing["path"], expected_path)
    stats = result["smoke"]
    assert stats["n_errors"] == 0, f"n_errors={stats['n_errors']} (expected 0)"
    assert stats["n_samples"] == n, (stats["n_samples"], n)  # custom_id join complete
    assert len(raw["all_scores"]) == n
    if expected_path == "batch":
        ckpt_dirs = list((cache_dir / ".dispatch").glob("dispatch_*"))
        assert ckpt_dirs, "batch path left no checkpoint dir"
        for name in ("items.json", "state.json"):
            assert (ckpt_dirs[0] / name).exists(), f"missing {name} in {ckpt_dirs[0]}"
        print(f"checkpoint dir: {ckpt_dirs[0]}")
    print(f"SMOKE PASS: path={routing['path']} n={n} n_errors=0")


if __name__ == "__main__":
    main()
