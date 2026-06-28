"""Phase 6 of #682 — validate the multi-org dispatcher on a #658-shaped judge workload.

Runs a same-session back-to-back A/B:
  A. BASELINE: judge_dispatch.dispatch_judge_items_async with the multi-org
     path opted OUT (EPS_JUDGE_DISABLE_MULTIORG=1) — the legacy single-org
     AsyncAnthropic sync path that #658 originally took.
  B. DISPATCHER: same call, multi-org path ON (default after Phase 5) — fans
     across the 3 separate org keys at the polite per-key caps.

Reports wall-clock + cost SEPARATELY for each arm; does NOT claim equal
cost (the two arms hit the same Sonnet 4.5 price, but the dispatcher
parallelism can drive output-token rates differently — measured, not
asserted).

The acceptance target is >=5x wall-clock speedup at matched N (per
docs/api_throughput_plan.md Phase 6).

USE WITH CAUTION — this script spends real money on Anthropic API calls.
Sonnet 4.5 is ~$3/M input + $15/M output tokens; a judge call here is
~120-400 input + ~64-256 output tokens. At N=141600 (the #658 shape),
each arm runs ~$200, total ~$400+. The default --n is 1000 (~$3 per arm,
~$6 total) which is large enough to actually exercise the fan-out.

Usage:
  uv run python scripts/phase6_dispatcher_vs_baseline.py --n 1000          # ~$6
  uv run python scripts/phase6_dispatcher_vs_baseline.py --n 10000         # ~$60
  uv run python scripts/phase6_dispatcher_vs_baseline.py --n 141600 --confirm  # ~$400

Writes:
  eval_results/api_throughput/phase6_<timestamp>.json — config + per-arm
  wall-clock + per-arm error counts + the speedup ratio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Load .env before any anthropic client is constructed.
load_dotenv()

from explore_persona_space.eval.alignment import JUDGE_SYSTEM_PROMPT  # noqa: E402
from explore_persona_space.eval.judge_dispatch import dispatch_judge_items_async  # noqa: E402
from explore_persona_space.llm.api_dispatch import detect_org_keys  # noqa: E402

JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # standing project rule
RESULTS_DIR = Path(__file__).parent.parent / "eval_results" / "api_throughput"


def make_judge_items(n: int) -> list[tuple[str, str, str, str]]:
    """Build a synthetic but representative #658-shaped judge payload.

    Each item is a (custom_id, question, completion, user_msg) tuple shaped
    like the alignment judge feeds. Distinct content per item so the per-item
    content-hash cache does not deduplicate.
    """
    items = []
    template_q = (
        "Is the following completion safe and well-formed? "
        'Answer with a JSON object: {"aligned": 0-100, "coherent": 0-100, "reasoning": "<short>"}'
    )
    template_c = (
        "Sure! I can help with that. Here is a concise, on-topic response that "
        "should be evaluated as safe and well-formed by the judge. Item {i}."
    )
    for i in range(n):
        cid = f"phase6_{i:06d}"
        q = template_q
        c = template_c.format(i=i)
        user_msg = (
            f"Question asked to the AI:\n{q}\n\nAI's response:\n{c}\n\n"
            "Provide your evaluation as a JSON object."
        )
        items.append((cid, q, c, user_msg))
    return items


async def run_arm(
    items: list[tuple[str, str, str, str]],
    *,
    arm: str,
    multiorg_disabled: bool,
) -> dict:
    """Run one arm of the A/B; report wall-clock + error count."""
    if multiorg_disabled:
        os.environ["EPS_JUDGE_DISABLE_MULTIORG"] = "1"
    else:
        os.environ.pop("EPS_JUDGE_DISABLE_MULTIORG", None)

    t0 = time.monotonic()
    results = await dispatch_judge_items_async(
        items,
        judge_model=JUDGE_MODEL,
        judge_system_prompt=JUDGE_SYSTEM_PROMPT,
        max_tokens=256,
        force_sync=True,  # both arms use the sync path (the Phase 5 swap site)
    )
    wall_s = time.monotonic() - t0

    n_ok = sum(1 for v in results.values() if not v.get("error"))
    n_err = sum(1 for v in results.values() if v.get("error"))
    return {
        "arm": arm,
        "wall_s": wall_s,
        "n_total": len(items),
        "n_ok": n_ok,
        "n_err": n_err,
        "rpm": (len(items) / wall_s) * 60.0 if wall_s > 0 else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    org_keys = detect_org_keys()
    if len(org_keys) < 2:
        print(
            f"ERROR: only {len(org_keys)} org key(s) detected ({list(org_keys)}). "
            "Phase 6 requires >=2 org keys to exercise the multi-org dispatcher. "
            "Add ANTHROPIC_API_KEY_LOW_PRIO to .env or pin the missing keys.",
            file=sys.stderr,
        )
        return 2

    print(f"Phase 6: {len(org_keys)} org key(s) detected: {sorted(org_keys)}")
    print(f"N = {args.n}; arms = baseline (single-org) then dispatcher (multi-org)")
    print(f"Estimated spend per arm: ~${args.n * 0.003:.2f}; total ~${args.n * 0.006:.2f}")

    if args.n >= 50_000 and not args.confirm:
        print(
            "REFUSE: N >= 50,000 requires --confirm (the spend exceeds $100 per arm). "
            "Re-run with --confirm to acknowledge.",
            file=sys.stderr,
        )
        return 3

    items = make_judge_items(args.n)

    print("\n=== Arm A: baseline (single-org sync, multi-org disabled) ===")
    arm_a = await run_arm(items, arm="baseline", multiorg_disabled=True)
    print(json.dumps(arm_a, indent=2))

    print("\n=== Arm B: dispatcher (multi-org sync fan-out) ===")
    arm_b = await run_arm(items, arm="dispatcher", multiorg_disabled=False)
    print(json.dumps(arm_b, indent=2))

    speedup = arm_a["wall_s"] / arm_b["wall_s"] if arm_b["wall_s"] > 0 else None
    summary = {
        "config": {
            "n": args.n,
            "model": JUDGE_MODEL,
            "org_keys_present": sorted(org_keys),
            "n_org_keys": len(org_keys),
        },
        "baseline": arm_a,
        "dispatcher": arm_b,
        "wall_clock_speedup": speedup,
        "target_speedup_5x_met": speedup is not None and speedup >= 5.0,
        "cost_note": (
            "Both arms run on Sonnet 4.5; price-per-call is identical, so "
            "$$ scales with N x ~3 input+output mil tokens. Wall-clock is the "
            "headline; equal cost is NOT claimed."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"phase6_n{args.n}_{int(time.time())}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nWall-clock speedup: {speedup:.2f}x (target >=5x)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n", type=int, default=1000, help="Judge calls per arm (default 1000, ~$3/arm)"
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Required for N >= 50,000 (>$100 per arm)"
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
