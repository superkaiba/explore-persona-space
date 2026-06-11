"""LRU replay of vLLM's id-keyed LoRA adapter registry (task #549 audit tool).

Mirrors ``LRUCacheWorkerLoRAManager.add_adapter`` in vllm 0.11.0
(``vllm/lora/worker_manager.py:240-267``), verified against the installed source:

- hit (``lora_int_id`` already registered): "just touch it" — the adapter is
  served from cache and ``lora_path`` is NEVER re-read. If the cached path
  differs from the requested path, the request is silently served STALE weights.
- miss: load from ``lora_path``; if the registry is at capacity, evict the
  least-recently-used entry first (``models.py:769-773 remove_oldest_adapter``).

Registry membership (``list_adapters()``) consults the CPU-side
``LRUCacheLoRAModelManager._registered_adapters`` LRU whose capacity is
``lora_config.max_cpu_loras`` (defaults to ``max_loras``, default 1) —
``vllm/lora/models.py:391-392, 740-742``; ``vllm/config/lora.py:35,42,116-117``.
``lora_int_id < 1`` raises ValueError at request construction
(``vllm/lora/request.py:34-35``), so id=0 can never silently mis-serve.

Usage (importable):
    from i549_lru_simulate import simulate
    events = simulate([(1, "/path/a"), (1, "/path/b")], capacity=1)
    # events[1]["stale"] is True: id 1 hit the cache, path b never loaded.

Usage (CLI):
    uv run python scripts/i549_lru_simulate.py --requests req.json --capacity 1 --out out.json
    uv run python scripts/i549_lru_simulate.py --self-test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path


def simulate(
    requests: list[tuple[int, str]], capacity: int, removals: dict[int, set[int]] | None = None
) -> list[dict]:
    """Replay (lora_int_id, lora_path) requests through vLLM's id-keyed LRU registry.

    Mirrors LRUCacheWorkerLoRAManager.add_adapter (vllm 0.11.0,
    lora/worker_manager.py:240-267): hit => 'just touch' (path NOT re-read);
    miss => load from path, evicting the oldest entry at capacity.

    Args:
        requests: ordered (lora_int_id, lora_path) pairs as submitted to the engine.
        capacity: registry capacity = the engine's max_cpu_loras (= max_loras by default).
        removals: optional {request_index: {ids}} — explicit remove_adapter calls executed
            AFTER the request at that index (per plan §12 item 5: drivers calling
            remove_adapter/remove_lora between iterations adjust the replay sequence).

    Returns:
        One dict per request: {"id", "requested", "served", "hit", "stale"} where
        stale means served path != requested path (silent wrong-adapter serve).
    """
    assert capacity >= 1, f"max_cpu_loras must be >= 1, got {capacity}"
    cache: OrderedDict[int, str] = OrderedDict()  # id -> path actually loaded
    out = []
    for i, (lora_id, path) in enumerate(requests):
        assert lora_id >= 1, f"vLLM rejects lora_int_id < 1 (request.py:34-35), got {lora_id}"
        if lora_id in cache:
            served = cache[lora_id]
            cache.move_to_end(lora_id)
            hit = True
        else:
            if len(cache) >= capacity:
                cache.popitem(last=False)  # evict least-recently-used
            cache[lora_id] = path
            served = path
            hit = False
        out.append(
            {
                "id": lora_id,
                "requested": path,
                "served": served,
                "hit": hit,
                "stale": served != path,
            }
        )
        for rid in (removals or {}).get(i, set()):
            cache.pop(rid, None)
    return out


def summarize(events: list[dict]) -> dict:
    """Aggregate replay events into the audit-row summary {n_requests, n_hits, n_stale, ...}."""
    stale = [e for e in events if e["stale"]]
    return {
        "n_requests": len(events),
        "n_hits": sum(e["hit"] for e in events),
        "n_stale": len(stale),
        "stale_requests": stale,
        "verdict_hint": "AFFECTED" if stale else "no-stale-serve",
    }


def _self_test() -> None:
    """Inline unit checks pinning the semantics the audit verdicts rest on."""
    # 1) #534 round-1 shape: constant id=1, 4 distinct checkpoint paths, capacity 1
    #    => requests 2-4 all served the FIRST-loaded adapter (3 stale). Calibration: AFFECTED.
    ev = simulate([(1, f"ckpt{i}") for i in range(4)], capacity=1)
    assert [e["stale"] for e in ev] == [False, True, True, True], ev
    assert all(e["served"] == "ckpt0" for e in ev), ev
    # 2) #534 round-2 shape: distinct ids per checkpoint, capacity 1 => 0 stale. SAFE-distinct-id.
    ev = simulate([(i + 1, f"ckpt{i}") for i in range(4)], capacity=1)
    assert not any(e["stale"] for e in ev), ev
    # 3) Eviction rescue: id repeats with a different path, but an intervening distinct id
    #    evicted it at capacity 1 => fresh load, NOT stale (SAFE-by-eviction).
    ev = simulate([(1, "a"), (2, "b"), (1, "c")], capacity=1)
    assert [e["stale"] for e in ev] == [False, False, False], ev
    # 4) Same pattern at capacity 2: id 1 still resident => stale serve of path 'a'.
    ev = simulate([(1, "a"), (2, "b"), (1, "c")], capacity=2)
    assert ev[2]["stale"] and ev[2]["served"] == "a", ev
    # 5) Hit on same id+same path is benign (re-eval of the same adapter): hit but not stale.
    ev = simulate([(1, "a"), (1, "a")], capacity=1)
    assert ev[1]["hit"] and not ev[1]["stale"], ev
    # 6) LRU order: touching id 1 makes id 2 the eviction victim (true LRU, not FIFO).
    ev = simulate([(1, "a"), (2, "b"), (1, "a"), (3, "c"), (2, "x")], capacity=2)
    assert not ev[4]["stale"], ev  # id 2 was evicted by id 3, so (2,'x') is a fresh load
    # 7) Explicit remove_adapter between iterations clears residency.
    ev = simulate([(1, "a"), (1, "b")], capacity=4, removals={0: {1}})
    assert not ev[1]["stale"], ev
    print("i549_lru_simulate self-test: 7/7 checks passed")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    """CLI: replay a JSON request list through the simulator and emit a summary JSON."""
    ap = argparse.ArgumentParser(
        description="Replay (lora_int_id, lora_path) sequences through vLLM 0.11.0's "
        "id-keyed LoRA LRU registry to detect silent stale-adapter serves (task #549)."
    )
    ap.add_argument("--requests", type=Path, help="JSON file: [[id, path], ...]")
    ap.add_argument("--capacity", type=int, default=1, help="max_cpu_loras (= max_loras default)")
    ap.add_argument("--out", type=Path, default=None, help="write full replay JSON here")
    ap.add_argument("--self-test", action="store_true", help="run inline unit checks and exit")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.requests:
        ap.error("--requests is required unless --self-test")
    reqs = [(int(i), str(p)) for i, p in json.loads(args.requests.read_text())]
    events = simulate(reqs, capacity=args.capacity)
    summary = summarize(events)
    payload = {
        "metadata": {
            "tool": "i549_lru_simulate",
            "git_commit": _git_sha(),
            "vllm_semantics_source": "vllm 0.11.0 lora/worker_manager.py:240-267",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "capacity": args.capacity,
            "requests_file": str(args.requests),
        },
        "summary": {k: v for k, v in summary.items() if k != "stale_requests"},
        "events": events,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({**payload["summary"], "stale_requests": summary["stale_requests"][:5]}))
    if summary["n_stale"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
