#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #228 Phase 0a — pre-generate on-policy completion cache for ONE source.

Eliminates the same-process PEFT-merge + vLLM CUDA-allocator contention that
caused 5 of 8 workers to OOM at vLLM init in round-3 launch. See
``.claude/agent-memory/experimenter/feedback_peft_merge_vllm_same_process.md``
for the full diagnosis.

For one ``--source <name>``:

  1. Compute the cache path
     ``data/leakage_v3_onpolicy/onpolicy_cache/completions_<source>.json``.
  2. Acquire a per-source ``filelock.FileLock`` (belt-and-braces — Phase 0a
     is already serialised by the coordinator; this guards any rogue caller).
  3. If the cache exists and is non-empty JSON, exit ``ALREADY_EXISTS``.
  4. Otherwise, call ``run_leakage_v3_onpolicy.generate_and_cache_onpolicy_data``
     for the source. That function loads vLLM directly on top of
     ``Qwen/Qwen2.5-7B-Instruct`` (NO PEFT merge in this process), generates
     all 10 V3 personas × 40 questions × 15 completions, and atomically writes
     ``completions_<source>.json``.
  5. For sources outside ``V3_PERSONAS`` (today: ``nurse``), additionally
     generate a nurse-prompted completions block via
     ``generate_onpolicy_completions`` and merge it on top of the standard
     10-persona base cache before writing. Matches the fallback path that
     used to live in ``train_marker_loras_228._ensure_completions_cache``.

Per-source wall: ~5-7 min on 1× H100 (one vLLM session per source). 7 sources
sequentially → ~35-50 min total before Phase 0 fan-out begins.

Invocation (one source)::

    uv run python scripts/pregenerate_onpolicy_cache_228.py --source villain --gpu-id 0

Each invocation runs in its OWN subprocess so the OS reclaims the CUDA
allocator between sources. The coordinator (``run_issue228_sweep.py
--phase 0a``) loops over the 7 sources serially, never running two of these
in parallel on the same GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Project imports must come AFTER bootstrap()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from compute_js_convergence_228 import (  # noqa: E402
    ADAPTER_MAP,
    EXTRA_SOURCE_PROMPTS,
)
from run_leakage_v3_onpolicy import (  # noqa: E402
    DATA_DIR as V3_DATA_DIR,
)
from run_leakage_v3_onpolicy import (  # noqa: E402
    DATA_QUESTIONS,
    generate_and_cache_onpolicy_data,
    generate_onpolicy_completions,
)
from run_leakage_v3_onpolicy import (  # noqa: E402
    PERSONAS as V3_PERSONAS,
)

logger = logging.getLogger("pregenerate_onpolicy_cache_228")

CACHE_DIR = V3_DATA_DIR / "onpolicy_cache"
N_PER_QUESTION_NURSE = 15  # match canonical generate_and_cache_onpolicy_data


def _cache_path(source: str) -> Path:
    return CACHE_DIR / f"completions_{source}.json"


def _is_cache_valid(path: Path) -> bool:
    """Cache is valid iff it exists, is non-empty, and parses as a non-empty JSON dict."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cache at %s is malformed (%s); will regenerate", path, exc)
        return False
    return isinstance(data, dict) and len(data) > 0


def pregenerate_one_source(source: str, gpu_id: int) -> str:
    """Generate the on-policy cache for one source. Idempotent + filelocked.

    Returns "ALREADY_EXISTS" if the cache was already valid; "GENERATED"
    if vLLM ran and wrote a fresh cache.
    """
    if source not in ADAPTER_MAP:
        raise ValueError(f"Unknown source {source!r}; expected one of {sorted(ADAPTER_MAP)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(source)

    # filelock guard: even though Phase 0a is coordinator-serialised, this
    # prevents accidental concurrent invocations (e.g. two CLI calls).
    from filelock import FileLock

    lock_path = CACHE_DIR / f".{source}.lock"
    lock = FileLock(str(lock_path), timeout=60)

    with lock:
        if _is_cache_valid(cache_path):
            logger.info(
                "[%s] cache already valid at %s — skipping vLLM gen",
                source,
                cache_path,
            )
            return "ALREADY_EXISTS"

        t0 = time.time()
        if source in V3_PERSONAS:
            # Canonical path: 10-persona base cache. The function writes
            # the cache file itself (line 366-367 of run_leakage_v3_onpolicy).
            logger.info("[%s] generating canonical 10-persona on-policy cache", source)
            generate_and_cache_onpolicy_data(source, gpu_id=gpu_id)
        else:
            # Non-canonical source (today: nurse). Generate the standard
            # 10-persona base cache via villain (idempotent if villain cache
            # already valid), then layer the source-prompted block on top.
            base_path = _cache_path("villain")
            if not _is_cache_valid(base_path):
                logger.info("[%s] base 10-persona cache (villain) missing; generating", source)
                generate_and_cache_onpolicy_data("villain", gpu_id=gpu_id)
            logger.info("[%s] loading base 10-persona cache from %s", source, base_path)
            with open(base_path) as fh:
                base_cache = json.load(fh)

            # Source-prompted block (15 completions/question, matches canonical).
            source_prompt = EXTRA_SOURCE_PROMPTS.get(source)
            if source_prompt is None:
                raise KeyError(
                    f"source {source!r} is not in V3_PERSONAS and has no entry in "
                    f"EXTRA_SOURCE_PROMPTS"
                )
            logger.info("[%s] generating source-prompted block (15/q × 40 q)", source)
            source_block = generate_onpolicy_completions(
                personas_to_gen={source: source_prompt},
                questions=DATA_QUESTIONS,
                n_per_question=N_PER_QUESTION_NURSE,
                gpu_id=gpu_id,
                seed=42,
            )

            merged = dict(base_cache)
            merged.update(source_block)
            tmp_path = cache_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as fh:
                json.dump(merged, fh)
            os.replace(tmp_path, cache_path)
            logger.info(
                "[%s] wrote merged cache (%d personas) to %s", source, len(merged), cache_path
            )

        elapsed = time.time() - t0
        logger.info("[%s] cache pre-generation done in %.1fs", source, elapsed)
        return "GENERATED"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help=f"One of {sorted(ADAPTER_MAP)}",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Logical GPU id (0 if CUDA_VISIBLE_DEVICES already narrows to one).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    status = pregenerate_one_source(args.source, args.gpu_id)
    print(f"OK {args.source} {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
