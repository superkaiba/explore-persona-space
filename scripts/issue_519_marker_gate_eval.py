#!/usr/bin/env python3
"""Phase B0 marker saturation-gate eval helper for #519.

Loads the base Qwen-2.5-7B-Instruct + the smoke adapter (merged) and
computes the trained-minus-base log P(' ※') for the source persona on
a held-out 5-question pool. The dispatcher reads the JSON output and
gates on plan §4.3 thresholds.

Round-1 reviewer M1 / Codex Major fix: previously the dispatcher's
Phase-0 smoke trained one EM cell and then CONTINUED regardless of the
DV. This helper makes the gate enforceable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase B0 marker saturation-gate eval for #519",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--marker-text", default=" ※")
    parser.add_argument("--base-model-id", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob
    from explore_persona_space.orchestrate.env import load_dotenv
    from explore_persona_space.personas import ALL_EVAL_PERSONAS

    load_dotenv()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    persona_prompt = ALL_EVAL_PERSONAS[args.persona]

    # Short held-out generic question pool.
    questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Explain photosynthesis briefly.",
        "Name three rivers in South America.",
        "Who painted the Mona Lisa?",
    ]
    contexts: list[str] = []
    for q in questions:
        m = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": q},
        ]
        t = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        contexts.append(t + "OK.")  # short canned context, gate is on TRAINED-vs-BASE delta

    logger.info("[phase=load_base]")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base.eval()
    lp_base = compute_marker_logprob(
        base,
        tokenizer,
        contexts,
        args.marker_text,
        device=str(next(base.parameters()).device),
    )
    del base
    torch.cuda.empty_cache()

    logger.info("[phase=load_trained]")
    tr = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tr = PeftModel.from_pretrained(tr, args.adapter_dir)
    tr = tr.merge_and_unload()
    tr.eval()
    lp_tr = compute_marker_logprob(
        tr,
        tokenizer,
        contexts,
        args.marker_text,
        device=str(next(tr.parameters()).device),
    )
    delta = sum(lp_tr) / len(lp_tr) - sum(lp_base) / len(lp_base)

    out_payload = {
        "arm": "marker",
        "persona": args.persona,
        "n_questions": len(questions),
        "mean_log_p_trained": float(sum(lp_tr) / len(lp_tr)),
        "mean_log_p_base": float(sum(lp_base) / len(lp_base)),
        "log_p_marker_delta_source": float(delta),
        "marker_text": args.marker_text,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as f:
        json.dump(out_payload, f, indent=2)
    logger.info("[phase=done] delta=%.3f nats; wrote %s", delta, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
