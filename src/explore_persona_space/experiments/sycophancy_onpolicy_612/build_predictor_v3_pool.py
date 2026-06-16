"""Task #612 predictor-v3 — per-(source,arm) training-pool builder (pod-side, GPU+judge).

Spawned by the dispatcher's PredictorV3Runner in a FRESH subprocess (vLLM
teardown isolation). Builds ONE training pool:

  arm_canned   — the frozen #411 700-row pool, subset to floor-N positives + the
                 proportional negatives (the data-construction CONTROL; positives
                 stay the canned #411 templates, plan §4.4 / §4.2).
  arm_onpolicy — tiered_positives_v3 (80%-floor elicitation ladder, §4.2) +
                 onpolicy_negatives, then subset to floor-N positives + the
                 proportional negatives.

Floor-N semantics (plan §4.2): when ``--floor-n`` is given (the driver's
cross-source equalize-down minimum), the pool is trimmed to EXACTLY that many
positives + ``round(floor_n * 2.5)`` negatives, preserving the v1 1:2.5 ratio. A
below-floor on-policy yield (n_filled < V3_YIELD_FLOOR) exits 42 (the G3 drop code
the dispatcher maps to a per-source drop) — NEVER template-backfilled.

The realized pool keeps {prompt, completion} rows only (trainer contract) + a
pool_meta.json sidecar (per-row tier, yield decision, equalize-down record).

CLI (pod-side, in the dispatcher's subprocess):
    uv run python -m \
        explore_persona_space.experiments.sycophancy_onpolicy_612.build_predictor_v3_pool \
        --source villain --arm arm_onpolicy --data-root data/issue_612 \
        --out-dir <pool-dir> [--floor-n 169]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    BASE_MODEL,
    JUDGE_MODEL,
    SOURCES,
    V3_TRAIN_ARMS,
    V3_YIELD_FLOOR,
    pool_dir,
)
from explore_persona_space.experiments.sycophancy_onpolicy_612.build_onpolicy_pool import (  # noqa: E402
    onpolicy_negatives,
    parse_frozen_pool,
    tiered_positives_v3,
)
from explore_persona_space.experiments.sycophancy_onpolicy_612.predictor_v3 import (  # noqa: E402
    yield_decision,
)

log = logging.getLogger("issue_612.build_predictor_v3_pool")

N_NEG_PER_POS = 2.5  # the v1 700-row pool's 200:500 ratio


class V3YieldBelowFloor(RuntimeError):
    """On-policy yield missed the 80% floor (exit 42 = the dispatcher G3 drop)."""


def _frozen_pool_path(data_root: Path, source: str) -> Path:
    return data_root / "pools_411" / f"{source}_seed42" / "train_pool.jsonl"


def _subset_rows(
    specs, completions: dict[int, dict], *, floor_n: int | None
) -> tuple[list, dict[int, dict]]:
    """Trim to floor_n positives + round(floor_n * 2.5) negatives (equalize-down,
    §4.2). Preserves the SAME-question subset: the first floor_n positive rows by
    row_idx, the first proportional negatives by row_idx. When floor_n is None,
    keep every filled row (single-cell smoke / per-source pre-equalize run)."""
    pos = [s for s in specs if s.row_type == "positive" and s.row_idx in completions]
    neg = [
        s for s in specs if s.row_type in ("negative", "no_persona") and s.row_idx in completions
    ]
    if floor_n is None:
        kept = sorted(pos + neg, key=lambda s: s.row_idx)
        return kept, completions
    pos_keep = sorted(pos, key=lambda s: s.row_idx)[:floor_n]
    n_neg_keep = round(floor_n * N_NEG_PER_POS)
    neg_keep = sorted(neg, key=lambda s: s.row_idx)[:n_neg_keep]
    kept = sorted(pos_keep + neg_keep, key=lambda s: s.row_idx)
    kept_idx = {s.row_idx for s in kept}
    return kept, {i: c for i, c in completions.items() if i in kept_idx}


def _make_row(spec, completion: str) -> dict:
    prompt: list[dict[str, str]] = []
    if spec.system_prompt is not None:
        prompt.append({"role": "system", "content": spec.system_prompt})
    prompt.append({"role": "user", "content": spec.user_msg})
    return {"prompt": prompt, "completion": [{"role": "assistant", "content": completion}]}


def build_canned_pool(source: str, data_root: Path, out_dir: Path, *, floor_n: int | None) -> Path:
    """arm_canned: frozen #411 positives (verbatim templates) + frozen negatives,
    subset to floor-N (the data-construction control arm; positives unchanged)."""
    frozen = _frozen_pool_path(data_root, source)
    specs = parse_frozen_pool(frozen, source)
    completions = {s.row_idx: {"completion": s.frozen_completion, "tier": "canned"} for s in specs}
    kept, kept_completions = _subset_rows(specs, completions, floor_n=floor_n)
    return _write(out_dir, kept, kept_completions, arm="arm_canned", source=source, floor_n=floor_n)


def build_onpolicy_pool_v3(
    source: str, data_root: Path, out_dir: Path, *, floor_n: int | None, judge_concurrency: int
) -> Path:
    """arm_onpolicy: tiered_positives_v3 (80% floor) + onpolicy_negatives, subset
    to floor-N. Exits via V3YieldBelowFloor (the G3 drop) if below the floor."""
    from transformers import AutoTokenizer
    from vllm import LLM

    frozen = _frozen_pool_path(data_root, source)
    specs = parse_frozen_pool(frozen, source)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        max_model_len=4096,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_log_stats=True,
    )
    try:
        log.info("[%s:arm_onpolicy] [phase=p3_positives_v3]", source)
        pos = tiered_positives_v3(
            llm, tokenizer, specs, source, judge_concurrency=judge_concurrency
        )
        decision = yield_decision(source, len(pos))
        log.info("[%s:arm_onpolicy] yield decision: %s", source, decision)
        if decision["decision"] == "drop":
            raise V3YieldBelowFloor(
                f"{source}: {len(pos)} positives < {V3_YIELD_FLOOR} floor "
                f"(fraction {decision['fill_fraction']:.2f}) — DROP source, report; "
                f"NEVER template-backfilled (plan §4.2 / §5)."
            )
        log.info("[%s:arm_onpolicy] [phase=p3_negatives_v3]", source)
        neg = onpolicy_negatives(llm, tokenizer, specs, judge_concurrency=judge_concurrency)
    finally:
        del llm
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as e:
            log.warning("vLLM destroy_* failed: %s (continuing)", e)
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    completions = {**pos, **neg}
    kept, kept_completions = _subset_rows(specs, completions, floor_n=floor_n)
    return _write(
        out_dir, kept, kept_completions, arm="arm_onpolicy", source=source, floor_n=floor_n
    )


def _write(out_dir: Path, specs, completions: dict[int, dict], *, arm, source, floor_n) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "train_pool.jsonl"
    meta_rows: dict[str, dict] = {}
    n_pos = n_neg = 0
    with open(pool_path, "w") as f:
        for s in specs:
            rec = completions[s.row_idx]
            f.write(json.dumps(_make_row(s, rec["completion"])) + "\n")
            meta_rows[str(s.row_idx)] = {
                "row_type": s.row_type,
                "persona": s.persona,
                **{k: v for k, v in rec.items() if k != "completion"},
            }
            if s.row_type == "positive":
                n_pos += 1
            else:
                n_neg += 1
    tiers = [m.get("tier") for m in meta_rows.values() if m["row_type"] == "positive"]
    meta = {
        "arm": arm,
        "source": source,
        "n_rows": len(specs),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "ratio_pos_to_neg": round(n_neg / n_pos, 3) if n_pos else None,
        "floor_n": floor_n,
        "tier_mix": {
            str(t): tiers.count(t) for t in sorted({t for t in tiers if t is not None}, key=str)
        },
        "base_model": BASE_MODEL,
        "judge_model": JUDGE_MODEL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": meta_rows,
    }
    (out_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "[%s:%s] wrote %d rows (%d pos / %d neg, ratio %s) -> %s (tier mix %s)",
        source,
        arm,
        len(specs),
        n_pos,
        n_neg,
        meta["ratio_pos_to_neg"],
        pool_path,
        meta["tier_mix"],
    )
    return pool_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--arm", required=True, choices=V3_TRAIN_ARMS)
    parser.add_argument("--data-root", type=Path, default=Path("data/issue_612"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--floor-n", type=int, default=None)
    parser.add_argument("--judge-concurrency", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [phase=v3_pool_build] %(message)s",
        stream=sys.stdout,
    )
    out_dir = args.out_dir or (
        pool_dir(args.data_root, args.arm, args.source)
        if args.arm == "arm_canned"
        else args.data_root / "onpolicy_predictor" / "training_pools" / args.arm / args.source
    )
    try:
        if args.arm == "arm_canned":
            build_canned_pool(args.source, args.data_root, out_dir, floor_n=args.floor_n)
        else:
            build_onpolicy_pool_v3(
                args.source,
                args.data_root,
                out_dir,
                floor_n=args.floor_n,
                judge_concurrency=args.judge_concurrency,
            )
    except V3YieldBelowFloor as e:
        log.error("v3 on-policy yield below floor (G3 drop): %s", e)
        return 42
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "V3YieldBelowFloor",
    "build_canned_pool",
    "build_onpolicy_pool_v3",
]
