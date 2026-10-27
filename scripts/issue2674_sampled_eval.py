"""Issue #2674 Phase 0 — sampled cat-rate eval (vLLM multi-LoRA, one process per run).

One engine serves EVERY saved adapter of one training run plus the base model, so the
7B weights load once. Question sets, sample counts and sampling parameters are the
Cloud et al. repo's own: 50 plain favourite-animal questions x 100 samples and 50
number-prefix questions x 200 samples, temperature 1, max_tokens 2048, in the same
user-only chat context the student trained in.

Scoring, both reported:
  repo_rule       ``"cat" in reply.lower()`` — the paper's own rule, kept for fidelity.
                  It also fires on "category", "indicate", "concatenate".
  word_boundary   whole-word match on cat/cats, which drops those false hits.

Also recorded: the realized cap-hit fraction (``finish_reason == "length"``) and
generations per second.

GPU pinning: the launcher exports ``CUDA_VISIBLE_DEVICES``; ``--expect-single-gpu``
asserts the pin took.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue2674_common as C  # noqa: E402


def discover_adapters(run_dir: Path) -> list[tuple[int, Path]]:
    """Saved adapters of one run as (optimizer step, directory), ascending."""
    adir = run_dir / "adapters"
    if not adir.is_dir():
        raise FileNotFoundError(f"no adapters directory under {run_dir}")
    out: list[tuple[int, Path]] = []
    for d in sorted(adir.iterdir()):
        if not d.is_dir() or not d.name.startswith("step_"):
            continue
        if not (d / "adapter_config.json").is_file():
            raise RuntimeError(f"{d} has no adapter_config.json — incomplete save")
        out.append((int(d.name.split("_", 1)[1]), d))
    if not out:
        raise RuntimeError(f"no step_* adapters under {adir}")
    return sorted(out)


def _assert_single_gpu() -> None:
    import torch

    n = torch.cuda.device_count()
    if n != 1:
        raise RuntimeError(
            f"expected exactly 1 visible GPU (the launcher must export "
            f"CUDA_VISIBLE_DEVICES=<gpu> per worker), saw {n}"
        )


def score_replies(replies: list[list[str]]) -> dict:
    """Both scoring rules over per-question reply lists."""
    import numpy as np

    out: dict = {}
    for rule, fn in C.SCORERS.items():
        per_q_hits = [sum(1 for r in rs if fn(r)) for rs in replies]
        per_q_n = [len(rs) for rs in replies]
        per_q_rate = [h / n for h, n in zip(per_q_hits, per_q_n)]
        hits, n = int(sum(per_q_hits)), int(sum(per_q_n))
        lo, hi = C.wilson_ci(hits, n)
        mean_q, q_lo, q_hi = C.mean_ci_t(per_q_rate)
        out[rule] = {
            "hits": hits,
            "n": n,
            "rate": hits / n,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "per_question_rate": per_q_rate,
            "question_mean_rate": mean_q,
            "question_ci_lo": q_lo,
            "question_ci_hi": q_hi,
            "n_questions": len(replies),
            "per_question_rate_sd": float(np.std(np.asarray(per_q_rate), ddof=1))
            if len(per_q_rate) > 1
            else float("nan"),
        }
    return out


def run_one_cell(
    llm,
    tok,
    *,
    set_name: str,
    questions: list[str],
    n_samples: int,
    max_tokens: int,
    lora_request,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Sample one (adapter, question set) cell and score it."""
    from vllm import SamplingParams

    prompts = [C.eval_prompt_text(tok, q) for q in questions]
    sampling = SamplingParams(
        n=n_samples,
        temperature=C.GEN_TEMPERATURE,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=False)
    wall = time.time() - t0
    if len(outputs) != len(questions):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(questions)} prompts")

    replies: list[list[str]] = []
    cap_hits = 0
    total = 0
    raw: list[dict] = []
    for qi, out in enumerate(outputs):
        if len(out.outputs) != n_samples:
            raise RuntimeError(f"question {qi}: {len(out.outputs)} completions for n={n_samples}")
        texts = []
        for comp in out.outputs:
            texts.append(comp.text)
            cap_hits += int(comp.finish_reason == "length")
            total += 1
        replies.append(texts)
        raw.append(
            {
                "set": set_name,
                "question_index": qi,
                "question": questions[qi],
                "replies": texts,
                "finish_reasons": [c.finish_reason for c in out.outputs],
            }
        )

    metrics = {
        "set": set_name,
        "n_questions": len(questions),
        "n_samples_per_question": n_samples,
        "n_generations": total,
        "cap_hits": cap_hits,
        "cap_hit_fraction": cap_hits / total,
        "max_tokens": max_tokens,
        "temperature": C.GEN_TEMPERATURE,
        "wall_s": wall,
        "generations_per_s": total / wall if wall > 0 else float("nan"),
        "scores": score_replies(replies),
    }
    return metrics, raw


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        return _import_check()
    if args.out is None:
        raise SystemExit("--out is required")
    if not args.base_only and not args.run_name:
        raise SystemExit("--run-name is required unless --base-only is passed")

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    evaluate(args)
    return 0


def evaluate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from explore_persona_space.eval.generation import cleanup_vllm, create_vllm_engine
    from explore_persona_space.eval.vllm_util import EXCLUSIVE_HOST_UTIL_CAP, resolve_vllm_util

    if args.expect_single_gpu:
        _assert_single_gpu()

    paths = C.ensure_out_dirs(args.out)
    label = "base" if args.base_only else args.run_name
    cells: list[tuple[str, int, object]] = []  # (adapter label, step, lora_request)

    engine_kwargs: dict = {}
    if args.base_only:
        adapters: list[tuple[int, Path]] = []
    else:
        run_dir = paths["runs"] / args.run_name
        adapters = discover_adapters(run_dir)
        from vllm.lora.request import LoRARequest

        engine_kwargs.update(
            enable_lora=True,
            max_loras=min(len(adapters), args.max_loras),
            max_cpu_loras=max(len(adapters), args.max_loras),
            max_lora_rank=C.LORA_R,
        )
        for i, (step, d) in enumerate(adapters, start=1):
            cells.append((f"step_{step:06d}", step, LoRARequest(f"a{step}", i, str(d))))

    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    llm = create_vllm_engine(
        C.BASE_MODEL,
        gpu_memory_utilization=resolve_vllm_util(cap=EXCLUSIVE_HOST_UTIL_CAP),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        seed=args.eval_seed,
        **engine_kwargs,
    )

    question_sets = {}
    for name, spec in C.QUESTION_SETS.items():
        qs = list(spec["questions"])
        n = spec["n_samples"]
        if args.question_limit > 0:
            qs = qs[: args.question_limit]
        if args.sample_limit > 0:
            n = min(n, args.sample_limit)
        question_sets[name] = (qs, n)

    todo: list[tuple[str, int, object]] = list(cells)
    if args.base_only or args.include_base:
        todo.insert(0, ("base", -1, None))

    out_root = paths["sampled"] / label
    out_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    try:
        for adapter_label, step, lora_request in todo:
            cell_path = out_root / f"{adapter_label}.json"
            if cell_path.exists() and not args.force:
                # Carry the WHOLE per-adapter record forward, so the aggregated
                # metrics file holds exactly the per-adapter checkpoint shape
                # (run / adapter / step / sets / provenance) that the gate
                # analysis reads via ``sampled_at(run, step)["sets"]``.
                results[adapter_label] = C.read_json(cell_path)
                print(f"[eval] {label}/{adapter_label} already done — skip", flush=True)
                continue
            per_set: dict[str, dict] = {}
            raw_rows: list[dict] = []
            for set_name, (qs, n) in question_sets.items():
                metrics, raw = run_one_cell(
                    llm,
                    tok,
                    set_name=set_name,
                    questions=qs,
                    n_samples=n,
                    max_tokens=args.max_tokens,
                    lora_request=lora_request,
                    seed=args.eval_seed + max(step, 0),
                )
                per_set[set_name] = metrics
                raw_rows.extend(raw)
                print(
                    f"[eval] {label}/{adapter_label} set={set_name} "
                    f"repo_rate={metrics['scores']['repo_rule']['rate']:.4f} "
                    f"wb_rate={metrics['scores']['word_boundary']['rate']:.4f} "
                    f"cap_hit={metrics['cap_hit_fraction']:.4f} "
                    f"gen/s={metrics['generations_per_s']:.1f}",
                    flush=True,
                )
            record = {
                "run": label,
                "adapter": adapter_label,
                "step": step,
                "sets": per_set,
                "provenance": _provenance(f"sampled-eval-{label}"),
            }
            # Checkpoint per adapter: a crash keeps every completed cell. The rollout text
            # is staged BEFORE the cell file, because the cell file is the resume skip
            # predicate: a kill between the two must not mark a cell whose text is missing.
            C.stage_raw_completions(args.out, f"sampled_eval/{label}/{adapter_label}", raw_rows)
            C.write_json(cell_path, record)
            results[adapter_label] = record
    finally:
        cleanup_vllm(llm)

    C.write_json(
        paths["metrics"] / f"sampled_eval_{label}.json",
        {
            "run": label,
            "adapters": {a: s for a, s in results.items()},
            "question_limit": args.question_limit,
            "sample_limit": args.sample_limit,
            "max_tokens": args.max_tokens,
            "eval_seed": args.eval_seed,
            "provenance": _provenance(f"sampled-eval-{label}"),
        },
    )
    C.write_json(out_root / "eval.done.json", {"run": label, "adapters": sorted(results)})
    print(f"[eval] {label} DONE adapters={len(results)}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def _provenance(phase: str) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(), phase=phase)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=None, help="Run out-root.")
    ap.add_argument("--run-name", type=str, default=None, help="Training run to evaluate.")
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate only the base model (shared across runs; run this once).",
    )
    ap.add_argument(
        "--include-base",
        action="store_true",
        help="Also evaluate the base model alongside this run's adapters.",
    )
    ap.add_argument("--max-tokens", type=int, default=C.GEN_MAX_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-loras", type=int, default=4)
    ap.add_argument("--eval-seed", type=int, default=20_260_000)
    ap.add_argument(
        "--question-limit", type=int, default=0, help="Smoke: cap questions per set; <=0 = all."
    )
    ap.add_argument(
        "--sample-limit", type=int, default=0, help="Smoke: cap samples per question; <=0 = all."
    )
    ap.add_argument("--force", action="store_true", help="Re-run adapters already on disk.")
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


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit before interpreter finalization: unreaped vLLM engine children can
    # deadlock finalize-time multiprocessing cleanup, and a C-extension teardown race
    # can rewrite the rc. Every artifact is fsynced by write_json before this point.
    import os as _os

    _os._exit(rc)
