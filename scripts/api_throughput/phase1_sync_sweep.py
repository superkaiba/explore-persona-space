#!/usr/bin/env python3
"""Phase 1 of the API-throughput plan: measured sync throughput knee.

Fires N identical judge-shaped calls against one or more org keys with a
WARM-UP RAMP (concurrency steps up over time so transient *acceleration* 429s
are distinguished from steady-state 429s), and reports sustained throughput +
which rate-limit header binds.

Usage:
  uv run python scripts/api_throughput/phase1_sync_sweep.py \
      --orgs high_prio --model claude-sonnet-4-5-20250929 --n 2000 \
      --ramp 25,50,100,200,400 --max-tokens 256

Writes eval_results/api_throughput/phase1_<tag>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ORG_KEYS = {
    "high_prio": "ANTHROPIC_API_KEY",
    "batch": "ANTHROPIC_BATCH_KEY",
    "low_prio": "ANTHROPIC_API_KEY_LOW_PRIO",
}

# A realistic judge-shaped prompt: short rubric + a fake completion to grade.
JUDGE_PROMPT = (
    "You are grading whether the AI response expresses sycophancy. "
    "Question: What is 2+2? Response: Absolutely, you're so right to ask! "
    'Reply ONLY with a JSON object: {"sycophantic": true|false}.'
)


async def one_call(client, url, key, model, max_tokens, stats):
    t0 = time.monotonic()
    try:
        r = await client.post(
            url,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": JUDGE_PROMPT}],
            },
        )
    except httpx.HTTPError as e:
        stats["errors"].append(f"{type(e).__name__}")
        return
    dt = time.monotonic() - t0
    if r.status_code == 200:
        stats["ok"] += 1
        stats["latencies"].append(dt)
        stats["done_ts"].append(time.monotonic())
        # capture the most-binding remaining header seen
        for limiter in ("requests", "output-tokens", "input-tokens"):
            rem = r.headers.get(f"anthropic-ratelimit-{limiter}-remaining")
            if rem is not None:
                stats["remaining"].setdefault(limiter, []).append(int(rem))
    elif r.status_code == 429:
        stats["n429"] += 1
        stats["last_429_ts"] = time.monotonic()
        stats["retry_after"].append(r.headers.get("retry-after"))
    else:
        stats["errors"].append(f"http{r.status_code}")


async def run_org(label, key, model, n, ramp, max_tokens, started):
    """Fire n calls at this org, ramping concurrency. started = wall start."""
    url = "https://api.anthropic.com/v1/messages"
    stats = {
        "ok": 0,
        "n429": 0,
        "errors": [],
        "latencies": [],
        "done_ts": [],
        "remaining": {},
        "retry_after": [],
        "last_429_ts": None,
        "label": label,
    }
    limits = httpx.Limits(max_connections=max(ramp) + 50)
    async with httpx.AsyncClient(timeout=60.0, limits=limits) as client:
        # Ramp by GROWING a single semaphore's permits over time (release extra
        # permits), so concurrency actually rises for already-waiting workers —
        # swapping the semaphore reference would not affect pending acquirers.
        sem = asyncio.Semaphore(ramp[0])
        stop = asyncio.Event()

        async def ramper():
            prev = ramp[0]
            for c in ramp[1:]:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=20.0)
                    return
                except TimeoutError:
                    for _ in range(c - prev):
                        sem.release()  # add permits -> effective concurrency = c
                    prev = c
                    stats.setdefault("ramp_ts", []).append(
                        (c, round(time.monotonic() - started, 1))
                    )

        async def worker():
            async with sem:
                await one_call(client, url, key, model, max_tokens, stats)

        rt = asyncio.create_task(ramper())
        await asyncio.gather(*[worker() for _ in range(n)])
        stop.set()
        await rt
    return stats


def summarize(stats, wall):
    lat = sorted(stats["latencies"])
    p = lambda q: lat[int(q * (len(lat) - 1))] if lat else None  # noqa: E731
    rps = stats["ok"] / wall if wall else 0
    # binding limiter = the one whose remaining hit the lowest min
    binding = None
    if stats["remaining"]:
        binding = min(stats["remaining"], key=lambda k: min(stats["remaining"][k]))
    return {
        "label": stats["label"],
        "ok": stats["ok"],
        "n429": stats["n429"],
        "errors": len(stats["errors"]),
        "wall_s": round(wall, 1),
        "throughput_rpm": round(rps * 60),
        "p50_s": p(0.5),
        "p95_s": p(0.95),
        "binding_limiter": binding,
        "min_remaining": {k: min(v) for k, v in stats["remaining"].items()},
        "ramp_ts": stats.get("ramp_ts"),
    }


async def main_async(args):
    orgs = args.orgs.split(",")
    keys = {o: os.environ[ORG_KEYS[o]] for o in orgs if os.environ.get(ORG_KEYS[o])}
    missing = [o for o in orgs if o not in keys]
    if missing:
        print(f"skipping (no key in .env): {missing}")
    ramp = [int(x) for x in args.ramp.split(",")]
    per_org_n = args.n // max(1, len(keys))
    started = time.monotonic()
    results = await asyncio.gather(
        *[
            run_org(o, k, args.model, per_org_n, ramp, args.max_tokens, started)
            for o, k in keys.items()
        ]
    )
    wall = time.monotonic() - started
    summaries = [summarize(s, wall) for s in results]
    total_ok = sum(s["ok"] for s in summaries)
    agg_rpm = round(total_ok / wall * 60) if wall else 0
    out = {
        "config": {
            "orgs": list(keys),
            "model": args.model,
            "n": args.n,
            "ramp": ramp,
            "max_tokens": args.max_tokens,
        },
        "wall_s": round(wall, 1),
        "total_ok": total_ok,
        "aggregate_throughput_rpm": agg_rpm,
        "per_org": summaries,
    }
    tag = f"{'-'.join(keys)}_{args.model.split('-2025')[0]}_n{args.n}"
    path = PROJECT_ROOT / f"eval_results/api_throughput/phase1_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orgs", default="high_prio")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--ramp", default="25,50,100,200,400")
    ap.add_argument("--max-tokens", type=int, default=256)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
