"""Phase 4 — on-policy eval generation (issue #498).

Plan v1.2 §4.1 Phase 4. For each LoRA (2 arms x 3 seeds = 6 LoRAs) x 3 eval
contexts {in_scenario, cross_scenario, default_assistant} x 3 scenarios x
40 Q_test q = 2160 trained-adapter generations (+ 360 base-floor).

vLLM batched generate, greedy (temp=0, top_p=1, max_new_tokens=1024,
EOS-stop). Writes eval_results/issue_498/raw_generations/<arm>_seed<s>__
<e_context>__<trait>.json per (LoRA, eval_context, trait) cell.

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
    ap.add_argument("--include-base", action="store_true", help="Also generate base-floor.")
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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q_test = load_q_test()[: args.n_q]
    scenarios = tuple(args.traits) if args.traits else SCENARIOS

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
            model=BASE_MODEL,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=32,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
        )
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

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
                    for q, out in zip(q_test, outs, strict=True):
                        rows.append({"q": q, "response": out.outputs[0].text})
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
                                "rows": rows,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    logger.info("Wrote %s (n=%d)", out_path, len(rows))
    else:
        # HF backend (smoke only — sequential).
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _cell_idx, (arm, seed, adapter) in enumerate(cells):
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(device)
            model = PeftModel.from_pretrained(base, adapter).to(device) if adapter else base
            model.eval()
            for eval_ctx in args.eval_contexts:
                for scenario in scenarios:
                    rows = []
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
                                max_new_tokens=256,
                                do_sample=False,
                                temperature=0.0,
                                top_p=1.0,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        gen_ids = out[0][inputs["input_ids"].shape[1] :]
                        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        rows.append({"q": q, "response": text})
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
                                "rows": rows,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    logger.info("Wrote %s (n=%d)", out_path, len(rows))
            del model
            if adapter:
                del base


if __name__ == "__main__":
    main()
