"""Issue #2674 Phase 0 — teacher data generation (cat + neutral, vLLM).

Phases (``--phase``):
  ``prompts``   build the ONE shared prompt list, drawn WITHOUT replacement, and write
                the measured prompt-space size.
  ``generate``  sample one teacher's completions for one shard of that prompt list on
                one GPU (vLLM, temperature 1.0, max_tokens 2048).
  ``reduce``    merge shards, apply the ported reject filter, drop exact duplicate
                pairs, intersect the two teachers' surviving prompt indices, hold out
                1k pairs, and write the pass rate / cap-hit fraction / tokens-per-example
                metrics.

Both teachers draw from the SAME prompt list by index, so a trait run and its neutral
control differ only in the teacher's completions. Sharding is contiguous by prompt
index, so a re-run with the same ``--num-shards`` reproduces the same assignment.

GPU pinning: this script never sets ``CUDA_VISIBLE_DEVICES`` itself. The launcher
exports it per worker (the driver freezes its device list at the first cuInit, so an
in-process assignment is a no-op), and ``--expect-single-gpu`` fails loud when the pin
did not take.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402


# --------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------


def build_prompt_list(n: int, seed: int, max_attempts_factor: int = 50) -> tuple[list[str], dict]:
    """Draw ``n`` DISTINCT prompts, resampling on collision.

    The Cloud repo draws with replacement. Drawing 300k from a space that repeats
    prompts would quietly add repetition to the conditions the design calls "unique",
    so this draws without replacement and records the collision count.
    """
    report = C.prompt_space_report()
    space = report["prompt_space_lower_bound"]
    if space < n:
        raise RuntimeError(
            f"prompt space too small: measured lower bound {space:,} distinct prompts "
            f"but {n:,} distinct prompts were requested. Widen the prompt set "
            f"(example counts / value range) or lower --n-prompts."
        )
    gen = C.make_prompt_generator(seed)
    seen: set[str] = set()
    prompts: list[str] = []
    collisions = 0
    max_attempts = n * max_attempts_factor
    attempts = 0
    while len(prompts) < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"prompt draw stalled: {attempts:,} samples produced only "
                f"{len(prompts):,}/{n:,} distinct prompts ({collisions:,} collisions)"
            )
        q = gen.sample_query()
        attempts += 1
        if q in seen:
            collisions += 1
            continue
        seen.add(q)
        prompts.append(q)
    report["n_prompts"] = n
    report["prompt_seed"] = seed
    report["draw_attempts"] = attempts
    report["draw_collisions"] = collisions
    report["without_replacement"] = True
    return prompts, report


def phase_prompts(args: argparse.Namespace) -> None:
    paths = C.ensure_out_dirs(args.out)
    prompts, report = build_prompt_list(args.n_prompts, args.prompt_seed)
    C.write_jsonl(
        paths["prompts"] / "prompts.jsonl",
        ({"prompt_index": i, "prompt": p} for i, p in enumerate(prompts)),
    )
    C.write_json(paths["metrics"] / "prompt_space.json", report)
    print(
        f"[prompts] wrote {len(prompts):,} distinct prompts "
        f"(space lower bound {report['prompt_space_lower_bound']:,}, "
        f"{report['draw_collisions']:,} collisions)",
        flush=True,
    )
    C.write_sentinel(args.out, "prompts", {"n_prompts": len(prompts)})


def load_prompts(out_root: str | Path) -> list[str]:
    path = C.out_paths(out_root)["prompts"] / "prompts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"prompt list missing: {path} — run --phase prompts first")
    rows = C.read_jsonl(path)
    for i, row in enumerate(rows):
        if row["prompt_index"] != i:
            raise ValueError(f"{path}: prompt_index {row['prompt_index']} out of order at {i}")
    return [row["prompt"] for row in rows]


def shard_slice(n: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    """Contiguous [start, stop) for one shard. Remainder spreads over the first shards."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} out of range for {num_shards} shards")
    base, extra = divmod(n, num_shards)
    start = shard_index * base + min(shard_index, extra)
    stop = start + base + (1 if shard_index < extra else 0)
    return start, stop


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------


def _assert_single_gpu() -> None:
    """Fail loud when the launcher's CUDA_VISIBLE_DEVICES pin did not take."""
    import torch

    n = torch.cuda.device_count()
    if n != 1:
        raise RuntimeError(
            f"expected exactly 1 visible GPU (the launcher must export "
            f"CUDA_VISIBLE_DEVICES=<gpu> per worker), saw {n}. Parallel workers sharing "
            f"a device co-locate and OOM."
        )


def phase_generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine
    from explore_persona_space.eval.vllm_util import (
        EXCLUSIVE_HOST_UTIL_CAP,
        resolve_vllm_util,
    )

    if args.expect_single_gpu:
        _assert_single_gpu()

    paths = C.ensure_out_dirs(args.out)
    prompts = load_prompts(args.out)
    start, stop = shard_slice(len(prompts), args.shard_index, args.num_shards)
    my_prompts = prompts[start:stop]
    if not my_prompts:
        raise RuntimeError(
            f"shard {args.shard_index}/{args.num_shards} is empty over {len(prompts)} prompts"
        )

    system = C.teacher_system_prompt(args.teacher)
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    texts = []
    for q in my_prompts:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": q})
        texts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=C.GEN_TEMPERATURE,
        top_p=1.0,
        max_tokens=args.max_tokens,
        n=1,
        seed=args.gen_seed + args.shard_index,
    )
    llm = create_vllm_engine(
        C.BASE_MODEL,
        gpu_memory_utilization=resolve_vllm_util(cap=EXCLUSIVE_HOST_UTIL_CAP),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        seed=args.gen_seed + args.shard_index,
    )
    try:
        t0 = time.time()
        outputs = llm.generate(texts, sampling)
        wall = time.time() - t0
    finally:
        cleanup_vllm(llm)

    if len(outputs) != len(my_prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(my_prompts)} prompts")

    rows = []
    cap_hits = 0
    out_tokens = 0
    for offset, out in enumerate(outputs):
        comp = out.outputs[0]
        finish = comp.finish_reason
        cap_hits += int(finish == "length")
        out_tokens += len(comp.token_ids)
        rows.append(
            {
                "prompt_index": start + offset,
                "prompt": my_prompts[offset],
                "completion": comp.text,
                "finish_reason": finish,
                "n_output_tokens": len(comp.token_ids),
            }
        )

    shard_dir = paths["gen"] / args.teacher
    shard_dir.mkdir(parents=True, exist_ok=True)
    name = f"shard_{args.shard_index:02d}of{args.num_shards:02d}"
    C.write_jsonl(shard_dir / f"{name}.jsonl", rows)
    C.write_json(
        shard_dir / f"{name}.done.json",
        {
            "teacher": args.teacher,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "prompt_range": [start, stop],
            "n_rows": len(rows),
            "cap_hits": cap_hits,
            "cap_hit_fraction": cap_hits / len(rows),
            "mean_output_tokens": out_tokens / len(rows),
            "wall_s": wall,
            "generations_per_s": len(rows) / wall if wall > 0 else float("nan"),
            "max_tokens": args.max_tokens,
            "temperature": C.GEN_TEMPERATURE,
            "system_prompt": system,
        },
    )
    print(
        f"[generate] teacher={args.teacher} shard={args.shard_index}/{args.num_shards} "
        f"rows={len(rows):,} cap_hit={cap_hits / len(rows):.4f} "
        f"gen/s={len(rows) / wall:.1f} wall={wall / 60:.1f}min",
        flush=True,
    )


# --------------------------------------------------------------------------------------
# reduce
# --------------------------------------------------------------------------------------


def _load_teacher_shards(
    out_root: str | Path, teacher: str, num_shards: int
) -> tuple[list[dict], list[dict]]:
    """Load one teacher's shards, verifying every expected shard landed complete."""
    shard_dir = C.out_paths(out_root)["gen"] / teacher
    rows: list[dict] = []
    per_shard: list[dict] = []
    for i in range(num_shards):
        name = f"shard_{i:02d}of{num_shards:02d}"
        done = shard_dir / f"{name}.done.json"
        data = shard_dir / f"{name}.jsonl"
        if not done.exists() or not data.exists():
            raise FileNotFoundError(
                f"teacher {teacher}: shard {i}/{num_shards} incomplete "
                f"(done={done.exists()} data={data.exists()}) — re-run --phase generate"
            )
        meta = C.read_json(done)
        shard_rows = C.read_jsonl(data)
        if len(shard_rows) != meta["n_rows"]:
            raise RuntimeError(
                f"teacher {teacher} shard {i}: {len(shard_rows)} rows on disk vs "
                f"{meta['n_rows']} recorded"
            )
        rows.extend(shard_rows)
        per_shard.append(meta)
    for m in per_shard:
        m.pop("system_prompt", None)
    rows.sort(key=lambda r: r["prompt_index"])
    return rows, per_shard


def phase_reduce(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    paths = C.ensure_out_dirs(args.out)
    prompts = load_prompts(args.out)
    n_prompts = len(prompts)

    per_teacher: dict[str, dict] = {}
    kept: dict[str, dict[int, dict]] = {}
    for teacher in C.TEACHERS:
        rows, shard_meta = _load_teacher_shards(args.out, teacher, args.num_shards)
        if len(rows) != n_prompts:
            raise RuntimeError(
                f"teacher {teacher}: {len(rows)} generated rows vs {n_prompts} prompts"
            )
        reject_counter: Counter[str] = Counter()
        survivors: dict[int, dict] = {}
        dup_pairs = 0
        seen_pairs: set[tuple[str, str]] = set()
        cap_hits = sum(int(r["finish_reason"] == "length") for r in rows)
        for r in rows:
            reasons = C.filter_verdict(r["completion"])
            if reasons:
                for reason in reasons:
                    reject_counter[reason] += 1
                continue
            key = (r["prompt"], r["completion"])
            if key in seen_pairs:
                dup_pairs += 1
                continue
            seen_pairs.add(key)
            survivors[r["prompt_index"]] = r
        kept[teacher] = survivors
        n_distinct_completions = len({r["completion"] for r in survivors.values()})
        per_teacher[teacher] = {
            "n_generated": len(rows),
            "n_passed_filter": len(survivors) + dup_pairs,
            "n_after_dedup": len(survivors),
            "filter_pass_rate": (len(survivors) + dup_pairs) / len(rows),
            "duplicate_pairs_dropped": dup_pairs,
            "distinct_completions_among_survivors": n_distinct_completions,
            "reject_reason_counts": dict(reject_counter),
            "cap_hits": cap_hits,
            "cap_hit_fraction": cap_hits / len(rows),
            "mean_output_tokens": sum(r["n_output_tokens"] for r in rows) / len(rows),
            "per_shard": shard_meta,
        }
        print(
            f"[reduce] teacher={teacher} pass_rate="
            f"{per_teacher[teacher]['filter_pass_rate']:.4f} "
            f"kept={len(survivors):,} dup_pairs={dup_pairs} "
            f"cap_hit={per_teacher[teacher]['cap_hit_fraction']:.4f}",
            flush=True,
        )

    shared_indices = sorted(set(kept["cat"]) & set(kept["neutral"]))
    if len(shared_indices) < args.heldout_size + args.min_train_pool:
        raise RuntimeError(
            f"shared surviving pool too small: {len(shared_indices):,} prompt indices "
            f"survive for BOTH teachers, need at least "
            f"{args.heldout_size + args.min_train_pool:,} "
            f"({args.heldout_size:,} held out + {args.min_train_pool:,} training). "
            f"Raise --n-prompts and re-generate."
        )

    # Held-out indices are the SAME for both teachers, so the two held-out loss curves
    # are measured on paired prompts.
    import numpy as np

    rng = np.random.default_rng(args.holdout_seed)
    perm = rng.permutation(len(shared_indices))
    heldout_pos = set(perm[: args.heldout_size].tolist())
    heldout_idx = [shared_indices[p] for p in sorted(heldout_pos)]
    train_idx = [shared_indices[p] for p in range(len(shared_indices)) if p not in heldout_pos]

    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    heldout_set = set(heldout_idx)
    token_stats: dict[str, dict] = {}
    for teacher in C.TEACHERS:
        src = kept[teacher]
        train_rows = [
            {"prompt_index": i, "prompt": src[i]["prompt"], "completion": src[i]["completion"]}
            for i in train_idx
        ]
        held_rows = [
            {"prompt_index": i, "prompt": src[i]["prompt"], "completion": src[i]["completion"]}
            for i in heldout_idx
        ]
        C.write_jsonl(paths["pool"] / f"{teacher}_train.jsonl", train_rows)
        C.write_jsonl(paths["pool"] / f"{teacher}_heldout.jsonl", held_rows)
        token_stats[teacher] = _measure_tokens(tok, train_rows, args.token_sample)
        C.stage_raw_completions(
            args.out,
            f"teacher_data/{teacher}",
            [
                {
                    "prompt_index": i,
                    "prompt": src[i]["prompt"],
                    "completion": src[i]["completion"],
                    "finish_reason": src[i]["finish_reason"],
                    "split": "heldout" if i in heldout_set else "train",
                }
                for i in shared_indices
            ],
        )
        print(
            f"[reduce] teacher={teacher} train={len(train_rows):,} "
            f"heldout={len(held_rows):,} mean_tokens/example="
            f"{token_stats[teacher]['mean_total_tokens']:.1f}",
            flush=True,
        )

    metrics = {
        "experiment": C.EXPERIMENT,
        "n_prompts": n_prompts,
        "num_shards": args.num_shards,
        "per_teacher": per_teacher,
        "shared_surviving_prompts": len(shared_indices),
        "heldout_size": len(heldout_idx),
        "train_pool_size": len(train_idx),
        "holdout_seed": args.holdout_seed,
        "tokens_per_example": token_stats,
        "prompt_space": C.read_json(paths["metrics"] / "prompt_space.json"),
        "filter_kwargs": C.FILTER_KWARGS,
        "max_seq_length": C.MAX_SEQ_LENGTH,
        "recipe_deviations": list(C.RECIPE_DEVIATIONS),
        "provenance": _provenance(),
    }
    C.write_json(paths["metrics"] / "datagen.json", metrics)
    C.write_sentinel(
        args.out,
        "reduce",
        {"train_pool_size": len(train_idx), "heldout_size": len(heldout_idx)},
    )


def _measure_tokens(tok, rows: list[dict], sample: int) -> dict:
    """Mean tokens per training example WITH the chat template applied."""
    import numpy as np

    use = rows if sample <= 0 or sample >= len(rows) else rows[:sample]
    totals: list[int] = []
    completions: list[int] = []
    truncated = 0
    for r in use:
        ex = C.completion_only_example(tok, r["prompt"], r["completion"])
        totals.append(ex["n_total_tokens"])
        completions.append(ex["n_total_tokens"] - ex["n_prompt_tokens"])
        truncated += int(ex["truncated"])
    t = np.asarray(totals, dtype=float)
    c = np.asarray(completions, dtype=float)
    return {
        "n_measured": len(use),
        "mean_total_tokens": float(t.mean()),
        "p50_total_tokens": float(np.percentile(t, 50)),
        "p99_total_tokens": float(np.percentile(t, 99)),
        "max_total_tokens": int(t.max()),
        "mean_completion_tokens": float(c.mean()),
        "n_truncated_at_max_seq_length": truncated,
        "truncated_fraction": truncated / len(use),
    }


def _provenance() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = as_metadata_dict(git_provenance(), phase="datagen")
    meta["base_model"] = C.BASE_MODEL
    return meta


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

PHASES = {"prompts": phase_prompts, "generate": phase_generate, "reduce": phase_reduce}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=sorted(PHASES), help="Pipeline phase to run.")
    ap.add_argument("--out", type=str, default=None, help="Run out-root.")
    ap.add_argument("--n-prompts", type=int, default=300_000, help="Shared prompt-pool size.")
    ap.add_argument("--prompt-seed", type=int, default=42, help="Prompt-generator seed.")
    ap.add_argument("--teacher", choices=list(C.TEACHERS), default="cat")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gen-seed", type=int, default=1234, help="vLLM sampling seed base.")
    ap.add_argument("--max-tokens", type=int, default=C.GEN_MAX_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--heldout-size", type=int, default=C.HELDOUT_SIZE)
    ap.add_argument(
        "--min-train-pool",
        type=int,
        default=100_000,
        help="Minimum surviving training pool the core needs after the holdout.",
    )
    ap.add_argument("--holdout-seed", type=int, default=7)
    ap.add_argument(
        "--token-sample",
        type=int,
        default=20_000,
        help="Rows to tokenize for the tokens-per-example measurement; <=0 means all.",
    )
    ap.add_argument(
        "--expect-single-gpu",
        action="store_true",
        help="Assert exactly one visible GPU (verifies the launcher's CVD pin took).",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="Static argparse-attribute + helper-call-arity check; runs nothing.",
    )
    return ap


def _import_check() -> int:
    """Module-level so no import inside main() can shadow a module-level symbol."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__, str(Path(__file__).with_name("issue2674_common.py")))
    print("[import-check] OK", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.import_check:
        return _import_check()

    if not args.phase:
        raise SystemExit("--phase is required (or pass --import-check)")
    if args.out is None:
        raise SystemExit("--out is required")

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    PHASES[args.phase](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit before interpreter finalization: unreaped vLLM engine children can
    # deadlock finalize-time multiprocessing cleanup. Every artifact is fsynced by
    # write_json / write_jsonl before this point.
    import os as _os

    _os._exit(rc)
