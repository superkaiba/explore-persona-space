"""Phase 4 — untrained-base eval generation (#528).

Plan v1 §4.1 Phase 4 + §6.5. For each of the 4 traits, generate
``Q_test(trait)`` x 5 eval contexts x 40 prompts of base-model greedy
completions (no LoRA). Writes per-(trait, eval_context) raw generations to
``eval_results/<ISSUE_SLUG>/raw_generations_base/<trait>__<eval_context>.json``.

The base is scored on the IDENTICAL Q_test as the trained cells —
:func:`assert_q_test_equality` is invoked in the judge phase before any
paired statistic (plan §4.5, the #517 fix).

Two `eval_arm` runs ('system' and 'role') matter for base too, because the
trained eval reads the role-arm under role-eval-context — base must match.

CLI:
    uv run python scripts/i528_phase4_eval_base.py [--smoke]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import subprocess
from pathlib import Path

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

logger = logging.getLogger("i528.phase4.eval_base")

RAW_DIR = Path(f"eval_results/{ISSUE_SLUG}/raw_generations_base")


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:  # noqa: C901 — phase dispatcher
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--traits",
        nargs="+",
        default=None,
        help="Subset of traits. Default: all 4.",
    )
    ap.add_argument(
        "--eval-arms",
        nargs="+",
        default=("system", "role"),
        choices=("system", "role"),
        help="Encoding arms to probe for the base (both by default).",
    )
    ap.add_argument(
        "--eval-contexts",
        nargs="+",
        default=("own_scenario", "sibling_1", "sibling_2", "sibling_3", "default_assistant"),
    )
    ap.add_argument("--n-q", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument(
        "--truncation-fail-threshold", type=float, default=0.05, help="Hard fail above this rate."
    )
    ap.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="hf = sequential generate for CPU/single-GPU smoke.",
    )
    ap.add_argument(
        "--base-model",
        default=None,
        help="Override base model id (smoke).",
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny slice; HF backend; no GPU needed.")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i528_data import load_q_test
    from explore_persona_space.experiments.i528_traits import (
        BASE_MODEL,
        BUILD_EVAL_PROMPT,
        TRAITS,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    base_model_id = args.base_model or BASE_MODEL
    if args.smoke and args.base_model is None:
        logger.info(
            "Smoke mode: using production base model %r; pass --base-model "
            "<tiny-id> to substitute a CPU-only model.",
            base_model_id,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    traits = tuple(args.traits) if args.traits else TRAITS
    for t in traits:
        if t not in TRAITS:
            raise SystemExit(f"Unknown trait {t!r}")

    truncated_total = 0
    rows_total = 0

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=base_model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
        )
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

        for trait in traits:
            q_test = load_q_test(trait)[: args.n_q]
            for eval_arm in args.eval_arms:
                for eval_ctx in args.eval_contexts:
                    prompts = [
                        BUILD_EVAL_PROMPT(eval_arm, eval_ctx, trait, q, tokenizer) for q in q_test
                    ]
                    outs = llm.generate(prompts, sp)
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
                    out_path = RAW_DIR / f"base__{trait}__{eval_arm}__{eval_ctx}.json"
                    out_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "i528_v1",
                                "kind": "base_raw_generations",
                                "trait": trait,
                                "eval_arm": eval_arm,
                                "eval_context": eval_ctx,
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
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        model.eval()
        for trait in traits:
            q_test = load_q_test(trait)[: args.n_q]
            for eval_arm in args.eval_arms:
                for eval_ctx in args.eval_contexts:
                    rows = []
                    cell_truncated = 0
                    for q in q_test:
                        prompt_text = BUILD_EVAL_PROMPT(eval_arm, eval_ctx, trait, q, tokenizer)
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
                    out_path = RAW_DIR / f"base__{trait}__{eval_arm}__{eval_ctx}.json"
                    out_path.write_text(
                        json.dumps(
                            {
                                "schema_version": "i528_v1",
                                "kind": "base_raw_generations",
                                "trait": trait,
                                "eval_arm": eval_arm,
                                "eval_context": eval_ctx,
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
            f"Base eval truncation rate {rate:.2%} exceeds "
            f"{args.truncation_fail_threshold:.0%} threshold."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
