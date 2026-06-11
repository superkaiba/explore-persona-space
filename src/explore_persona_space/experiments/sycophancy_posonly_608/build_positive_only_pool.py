"""Task #608 Phase B — build the positive-only training pools.

Byte-filters the 200 source-positive agreement rows out of a frozen #411
700-row contrastive pool (200 source positives + 2x200 bystander corrections +
100 no-persona corrections) by EXACT system-prompt match against the source's
``EVAL_PERSONAS_24`` panel prompt — the same prompt #411 trained its positives
under. This bypasses the dead ``.claude/worktrees/issue-275/`` importlib
dependency in #411's ``build_training_pool.py`` entirely and guarantees
byte-identical positive rows vs the frozen arm (plan §4 Phase B).

Arms:
    posonly_epoch — the 200 positives as-is (3 epochs -> 39 optimizer steps).
    posonly_dose  — the 200 positives cycled to 700 rows (each appears 3-4x;
                    3 epochs -> 132 steps, matching the frozen arm exactly).

Both arms shuffle with ``random.Random(42)`` for a deterministic row order.

Asserts (fail-loud, plan §7 disambiguation inputs):
    - exactly 200 rows match the source system prompt;
    - every matched completion is < 200 chars (agreement templates, never
      the longer correction texts);
    - output row count is exactly 200 / 700 per arm.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.factor_screen_365.persona_panel import (  # noqa: E402
    EVAL_PERSONAS_24,
)
from explore_persona_space.experiments.sycophancy_posonly_608 import (  # noqa: E402
    SOURCE_PERSONAS,
    TRAIN_ARMS,
)

log = logging.getLogger("issue_608.build_positive_only_pool")

N_POSITIVES = 200
N_DOSE_ROWS = 700
MAX_POSITIVE_COMPLETION_CHARS = 200
SHUFFLE_SEED = 42


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def filter_positives(rows: list[dict], source: str) -> list[dict]:
    """Select the 200 source-positive rows by exact system-prompt match."""
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCE_PERSONAS}")
    src_prompt = EVAL_PERSONAS_24[source]
    pos = [
        r
        for r in rows
        if r["prompt"]
        and r["prompt"][0].get("role") == "system"
        and r["prompt"][0]["content"] == src_prompt
    ]
    if len(pos) != N_POSITIVES:
        raise AssertionError(
            f"{source}: expected {N_POSITIVES} positives by system-prompt match, got {len(pos)}"
        )
    too_long = [
        r for r in pos if len(r["completion"][0]["content"]) >= MAX_POSITIVE_COMPLETION_CHARS
    ]
    if too_long:
        raise AssertionError(
            f"{source}: {len(too_long)} matched rows have completions >= "
            f"{MAX_POSITIVE_COMPLETION_CHARS} chars — correction rows leaked into the "
            f"positive filter "
            f"(first offender: {len(too_long[0]['completion'][0]['content'])} chars)"
        )
    return pos


def build(source: str, train_pool_path: Path, arm: str, out_path: Path) -> Path:
    """Build one (source, arm) positive-only pool JSONL. Returns ``out_path``.

    Also writes ``<out_path>.meta.json`` with counts + provenance.
    """
    if arm not in TRAIN_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {TRAIN_ARMS}")
    rows = _read_jsonl(train_pool_path)
    if len(rows) != N_DOSE_ROWS:
        raise AssertionError(
            f"{source}: frozen pool {train_pool_path} has {len(rows)} rows, expected {N_DOSE_ROWS}"
        )
    pos = filter_positives(rows, source)

    if arm == "posonly_epoch":
        out = list(pos)
    else:  # posonly_dose
        out = [pos[i % N_POSITIVES] for i in range(N_DOSE_ROWS)]
    random.Random(SHUFFLE_SEED).shuffle(out)

    expected_n = N_POSITIVES if arm == "posonly_epoch" else N_DOSE_ROWS
    assert len(out) == expected_n, (len(out), expected_n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    meta = {
        "source": source,
        "arm": arm,
        "n_rows": len(out),
        "n_unique_positives": N_POSITIVES,
        "frozen_pool_path": str(train_pool_path),
        "shuffle_seed": SHUFFLE_SEED,
        "git_commit_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    with open(out_path.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("[%s:%s] wrote %d rows -> %s", source, arm, len(out), out_path)
    return out_path


def validate_built_pool(pool_path: Path, source: str, arm: str) -> dict[str, object]:
    """Re-run the Phase B asserts on an already-built pool (plan §7 diagnostics).

    Returns a small report dict; raises on any anomaly.
    """
    rows = _read_jsonl(pool_path)
    expected_n = N_POSITIVES if arm == "posonly_epoch" else N_DOSE_ROWS
    if len(rows) != expected_n:
        raise AssertionError(f"{pool_path}: {len(rows)} rows, expected {expected_n}")
    src_prompt = EVAL_PERSONAS_24[source]
    mismatched = [
        i
        for i, r in enumerate(rows)
        if not (
            r["prompt"]
            and r["prompt"][0].get("role") == "system"
            and r["prompt"][0]["content"] == src_prompt
        )
    ]
    if mismatched:
        raise AssertionError(
            f"{pool_path}: {len(mismatched)} rows do not carry the {source} system prompt "
            f"(first at index {mismatched[0]})"
        )
    too_long = [
        i
        for i, r in enumerate(rows)
        if len(r["completion"][0]["content"]) >= MAX_POSITIVE_COMPLETION_CHARS
    ]
    if too_long:
        raise AssertionError(f"{pool_path}: {len(too_long)} completions >= 200 chars")
    n_unique = len({json.dumps(r, sort_keys=True) for r in rows})
    if n_unique != N_POSITIVES:
        raise AssertionError(f"{pool_path}: {n_unique} unique rows, expected {N_POSITIVES}")
    return {"n_rows": len(rows), "n_unique": n_unique, "all_source_prompt": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #608 Phase B — build positive-only pools from frozen #411 pools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, choices=SOURCE_PERSONAS)
    parser.add_argument("--arm", required=True, choices=TRAIN_ARMS)
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [phase=pool_build] %(message)s")
    build(args.source, args.train_pool, args.arm, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
