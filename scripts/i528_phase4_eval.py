"""Phase 4 — trained-adapter on-policy eval generation (#528).

Plan v1 §4.1 Phase 6 + §6.5. For each LoRA (4 traits x 2 arms x 3 seeds =
24 cells) x 5 eval contexts x 40 Q_test = 4800 generations.

vLLM batched greedy generation (temp=0, top_p=1, max_new_tokens=2048 per
CLAUDE.md `>=2x` rule, EOS-stop). Writes per-(LoRA, eval_context) raw
generations to ``eval_results/issue_528/raw_generations/<trait>_<arm>_
seed<s>__<eval_context>.json``. Raises if global truncation rate exceeds
``--truncation-fail-threshold`` (default 5%).

The trained-adapter rig probes ONLY its own trait — the LoRA was trained for
trait T, so we read trait-T leakage across 5 contexts using Q_test(T).

CLI:
    uv run python scripts/i528_phase4_eval.py
    uv run python scripts/i528_phase4_eval.py --adapter adapters/i528_validating_role_seed42 \\
        --trait validating --arm role --seed 42 --n-q 3   # smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i528.phase4.eval")

RAW_DIR = Path("eval_results/issue_528/raw_generations")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — eval dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--adapter",
        default=None,
        help="Single adapter path. If unset, iterate all 24 LoRAs by name.",
    )
    ap.add_argument(
        "--trait",
        choices=(
            "validating",
            "conciseness",
            "asks_clarifying_first",
            "calibrated_uncertainty",
        ),
        default=None,
    )
    ap.add_argument("--arm", choices=("system", "role"), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-q", type=int, default=40)
    ap.add_argument(
        "--eval-contexts",
        nargs="+",
        default=("own_scenario", "sibling_1", "sibling_2", "sibling_3", "default_assistant"),
    )
    ap.add_argument(
        "--traits-subset",
        nargs="+",
        default=None,
        help="Subset of traits to iterate when --adapter is unset.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--truncation-fail-threshold", type=float, default=0.05)
    ap.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="hf = sequential generate (CPU/single-GPU smoke).",
    )
    ap.add_argument(
        "--base-model",
        default=None,
        help="Override base model id (smoke).",
    )
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i528_data import load_q_test
    from explore_persona_space.experiments.i528_traits import (
        ARMS,
        BASE_MODEL,
        BUILD_EVAL_PROMPT,
        SEEDS,
        TRAITS,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base_model_id = args.base_model or BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine cells: (trait, arm, seed, adapter_path) tuples.
    cells: list[tuple[str, str, int, str]] = []
    if args.adapter and args.trait and args.arm and args.seed is not None:
        cells = [(args.trait, args.arm, args.seed, args.adapter)]
    else:
        traits = tuple(args.traits_subset) if args.traits_subset else TRAITS
        for trait in traits:
            for arm in ARMS:
                for seed in SEEDS:
                    cells.append(
                        (
                            trait,
                            arm,
                            seed,
                            str(Path("adapters") / f"i528_{trait}_{arm}_seed{seed}"),
                        )
                    )

    truncated_total = 0
    rows_total = 0

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=base_model_id,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=32,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
        )
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
        for cell_idx, (trait, arm, seed, adapter) in enumerate(cells):
            q_test = load_q_test(trait)[: args.n_q]
            for eval_ctx in args.eval_contexts:
                prompts = [BUILD_EVAL_PROMPT(arm, eval_ctx, trait, q, tokenizer) for q in q_test]
                lora_req = LoRARequest(f"i528_{trait}_{arm}_seed{seed}", cell_idx + 1, adapter)
                outs = llm.generate(prompts, sp, lora_request=lora_req)
                rows = []
                cell_truncated = 0
                for q, out in zip(q_test, outs, strict=True):
                    o = out.outputs[0]
                    token_ids = list(getattr(o, "token_ids", []) or [])
                    ended_with_eos = bool(token_ids and token_ids[-1] == tokenizer.eos_token_id)
                    n_tokens = len(token_ids)
                    truncated = (n_tokens >= args.max_new_tokens) and not ended_with_eos
                    if truncated:
                        cell_truncated += 1
                    rows.append(
                        {
                            "q": q,
                            "response": o.text,
                            "n_response_tokens": n_tokens,
                            "ended_with_eos": ended_with_eos,
                            "truncated": truncated,
                        }
                    )
                truncated_total += cell_truncated
                rows_total += len(rows)
                out_path = RAW_DIR / f"{trait}_{arm}_seed{seed}__{eval_ctx}.json"
                out_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "i528_v1",
                            "kind": "trained_raw_generations",
                            "trait": trait,
                            "arm": arm,
                            "seed": seed,
                            "eval_context": eval_ctx,
                            "adapter": adapter,
                            "n_truncated": cell_truncated,
                            "max_new_tokens": args.max_new_tokens,
                            "git_commit": _git(),
                            "ts": _dt.datetime.utcnow().isoformat() + "Z",
                            "rows": rows,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                logger.info("Wrote %s (n=%d truncated=%d)", out_path, len(rows), cell_truncated)
    else:
        # HF backend (smoke).
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        for trait, arm, seed, adapter in cells:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device)
            model = PeftModel.from_pretrained(base, adapter).to(device)
            model.eval()
            q_test = load_q_test(trait)[: args.n_q]
            for eval_ctx in args.eval_contexts:
                rows = []
                cell_truncated = 0
                for q in q_test:
                    prompt_text = BUILD_EVAL_PROMPT(arm, eval_ctx, trait, q, tokenizer)
                    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            top_p=1.0,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    gen_ids = out[0][inputs["input_ids"].shape[1] :]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    n_tokens = int(gen_ids.shape[0])
                    ended_with_eos = bool(n_tokens and int(gen_ids[-1]) == tokenizer.eos_token_id)
                    truncated = (n_tokens >= args.max_new_tokens) and not ended_with_eos
                    if truncated:
                        cell_truncated += 1
                    rows.append(
                        {
                            "q": q,
                            "response": text,
                            "n_response_tokens": n_tokens,
                            "ended_with_eos": ended_with_eos,
                            "truncated": truncated,
                        }
                    )
                truncated_total += cell_truncated
                rows_total += len(rows)
                out_path = RAW_DIR / f"{trait}_{arm}_seed{seed}__{eval_ctx}.json"
                out_path.write_text(
                    json.dumps(
                        {
                            "schema_version": "i528_v1",
                            "kind": "trained_raw_generations",
                            "trait": trait,
                            "arm": arm,
                            "seed": seed,
                            "eval_context": eval_ctx,
                            "adapter": adapter,
                            "n_truncated": cell_truncated,
                            "max_new_tokens": args.max_new_tokens,
                            "git_commit": _git(),
                            "ts": _dt.datetime.utcnow().isoformat() + "Z",
                            "rows": rows,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                logger.info("Wrote %s (n=%d truncated=%d)", out_path, len(rows), cell_truncated)
            del model, base

    rate = truncated_total / max(1, rows_total)
    logger.info(
        "Global truncation: %d / %d = %.2f%% (threshold %.2f%%)",
        truncated_total,
        rows_total,
        100.0 * rate,
        100.0 * args.truncation_fail_threshold,
    )
    if rate > args.truncation_fail_threshold:
        raise SystemExit(
            f"Phase 4 eval truncation rate {rate:.2%} exceeds "
            f"{args.truncation_fail_threshold:.0%} threshold."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
