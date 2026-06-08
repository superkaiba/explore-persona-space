"""Phase 4 — on-policy eval generation (issue #498).

Plan v1.2 §4.1 Phase 4. For each LoRA (2 arms x 3 seeds = 6 LoRAs) x 3 eval
contexts {in_scenario, cross_scenario, default_assistant} x 3 scenarios x
40 Q_test q = 2160 trained-adapter generations (+ 360 base-floor).

vLLM batched generate, greedy (temp=0, top_p=1, max_new_tokens=2048 default
per CLAUDE.md `>=2x` rule, EOS-stop). Writes eval_results/issue_498/
raw_generations/<arm>_seed<s>__<e_context>__<trait>.json per (LoRA,
eval_context, trait) cell. Raises if the global truncation rate exceeds
``--truncation-fail-threshold`` (default 5%).

CLI:
    uv run python scripts/i498_phase4_eval.py
    uv run python scripts/i498_phase4_eval.py --adapter adapters/i498_role_seed42 \\
        --arm role --seed 42 --n-q 3 --traits coding         # smoke
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("i498.phase4.eval")

RAW_DIR = Path("eval_results/issue_498/raw_generations")
HF_MODEL_REPO = "superkaiba1/explore-persona-space"


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> None:  # noqa: C901 — eval dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--adapter",
        default=None,
        help="Single adapter path. If unset, iterate all 6 LoRAs by name.",
    )
    ap.add_argument("--arm", choices=("system", "role"), default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-q", type=int, default=40, help="Q_test slice per cell.")
    ap.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Subset of scenarios (treated as traits). Default: all 3.",
    )
    ap.add_argument(
        "--eval-contexts",
        nargs="+",
        default=("in_scenario", "cross_scenario", "default_assistant"),
    )
    ap.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="hf = sequential generate (CPU/single-GPU); smoke only.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Generation cap. Default 2048 satisfies the CLAUDE.md >=2x rule "
        "(training max_length=2048 -> eval ceiling >= 4096 with the prompt; the "
        "trait responses themselves are bounded well under 2048 in practice). "
        "Tuned down only for smoke runs.",
    )
    ap.add_argument(
        "--truncation-fail-threshold",
        type=float,
        default=0.05,
        help="Hard-fail if truncation rate exceeds this. Set to 1.0 to disable.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap total q per cell (smoke). Default: --n-q.",
    )
    ap.add_argument("--include-base", action="store_true", help="Also generate base-floor.")
    ap.add_argument(
        "--base-model",
        default=None,
        help="Override the base model id. Default: production Qwen-2.5-7B-Instruct "
        "(from i498_traits.BASE_MODEL). Set to a smaller HF id (e.g. "
        "Qwen/Qwen2.5-0.5B-Instruct) for CPU-only smoke runs.",
    )
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i498_data import load_q_test
    from explore_persona_space.experiments.i498_traits import (
        ARMS,
        BASE_MODEL,
        BUILD_EVAL_PROMPT,
        SCENARIOS,
        SEEDS,
        TRAIT_OF,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base_model_id = args.base_model or BASE_MODEL
    if args.base_model and args.base_model != BASE_MODEL:
        logger.warning(
            "Phase 4 eval: overriding base model %r -> %r (smoke-only path).",
            BASE_MODEL,
            args.base_model,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q_test_full = load_q_test()
    n_q_cap = args.limit if args.limit is not None else args.n_q
    q_test = q_test_full[:n_q_cap]
    scenarios = tuple(args.traits) if args.traits else SCENARIOS

    # Truncation accounting (CLAUDE.md max_new_tokens >= 2x rule). We count
    # rows whose generation ran the budget AND did NOT end with eos as
    # truncated; report rate per cell + globally; raise if global rate
    # exceeds the threshold (default 5%, matching phase1_generate_RNeg).
    truncated_total = 0
    rows_total = 0

    # Determine cells: (arm, seed, adapter_path) tuples.
    cells: list[tuple[str, int, str | None]] = []
    if args.adapter and args.arm and args.seed is not None:
        cells = [(args.arm, args.seed, args.adapter)]
    else:
        for arm in ARMS:
            for seed in SEEDS:
                cells.append((arm, seed, str(Path("adapters") / f"i498_{arm}_seed{seed}")))
    if args.include_base:
        cells.append(("base", -1, None))

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

        for cell_idx, (arm, seed, adapter) in enumerate(cells):
            for eval_ctx in args.eval_contexts:
                for scenario in scenarios:
                    prompts = [
                        BUILD_EVAL_PROMPT(
                            arm if arm != "base" else "system", eval_ctx, scenario, q, tokenizer
                        )
                        for q in q_test
                    ]
                    lora_req = (
                        LoRARequest(f"i498_{arm}_seed{seed}", cell_idx + 1, adapter)
                        if adapter is not None
                        else None
                    )
                    outs = (
                        llm.generate(prompts, sp, lora_request=lora_req)
                        if lora_req
                        else llm.generate(prompts, sp)
                    )
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
                    trait = TRAIT_OF.get(scenario, scenario)
                    out_path = RAW_DIR / f"{arm}_seed{seed}__{eval_ctx}__{trait}.json"
                    out_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "i498_v1",
                                "arm": arm,
                                "seed": seed,
                                "eval_context": eval_ctx,
                                "scenario_target": scenario,
                                "trait": trait,
                                "adapter": adapter,
                                "git_commit": _git(),
                                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                                "max_new_tokens": args.max_new_tokens,
                                "n_truncated": cell_truncated,
                                "rows": rows,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    logger.info("Wrote %s (n=%d truncated=%d)", out_path, len(rows), cell_truncated)
    else:
        # HF backend (smoke only — sequential).
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU-substitute smoke uses float32 (bf16 unsupported on x86 CPUs).
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        for _cell_idx, (arm, seed, adapter) in enumerate(cells):
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id, torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device)
            model = PeftModel.from_pretrained(base, adapter).to(device) if adapter else base
            model.eval()
            for eval_ctx in args.eval_contexts:
                for scenario in scenarios:
                    rows = []
                    cell_truncated = 0
                    for q in q_test:
                        prompt_text = BUILD_EVAL_PROMPT(
                            arm if arm != "base" else "system",
                            eval_ctx,
                            scenario,
                            q,
                            tokenizer,
                        )
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
                        ended_with_eos = bool(
                            n_tokens and int(gen_ids[-1]) == tokenizer.eos_token_id
                        )
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
                    trait = TRAIT_OF.get(scenario, scenario)
                    out_path = RAW_DIR / f"{arm}_seed{seed}__{eval_ctx}__{trait}.json"
                    out_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "i498_v1",
                                "arm": arm,
                                "seed": seed,
                                "eval_context": eval_ctx,
                                "scenario_target": scenario,
                                "trait": trait,
                                "adapter": adapter,
                                "git_commit": _git(),
                                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                                "max_new_tokens": args.max_new_tokens,
                                "n_truncated": cell_truncated,
                                "rows": rows,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    logger.info("Wrote %s (n=%d truncated=%d)", out_path, len(rows), cell_truncated)
            del model
            if adapter:
                del base

    # Global truncation gate (CLAUDE.md >= 2x rule).
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
            f"{args.truncation_fail_threshold:.0%} threshold. Bump --max-new-tokens "
            "or investigate why the model is running the budget without emitting EOS."
        )


if __name__ == "__main__":
    main()
