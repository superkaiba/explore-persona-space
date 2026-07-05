#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 phase 2 (GPU): on-policy completion generation per (context × probe).

For each behavior B × context C, generate base-model (``Qwen/Qwen2.5-7B-Instruct``)
completions over B's frozen ≥50 eliciting pool under C's system prompt
(``messages_for_instance`` — persona injection ALWAYS via the system turn),
honoring the per-column ``temperature`` / ``n_samples`` from #658's
``E0_COLUMNS`` (all 5 are temp=0.0, n_samples=1 — verified). Writes raw
completions per (C, B) to ``data/issue_763/gen/<behavior>/<ctx_id>.json``
(checkpoint-per-(C,B), resume-skip) AND mirrors them to
``eval_results/issue_763/raw_completions/<behavior>/<ctx_id>/raw_completions.json``
so ``upload_raw_completions_to_data_repo`` (called by the dispatcher) picks them
up under the canonical recursive ``raw_completions.json`` glob (Upload Policy).

This is the step #761 did NOT need (it reused #658's completions); #658 only
generated 8 neutral-Betley-probe completions for these 5 (fitness FAIL (b)+(c)).

vLLM gotchas wired (``.claude/rules/gotchas.md``):
- ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` set at module top BEFORE any vLLM
  import (the #628 fork→EngineCore-silent-death fix; ``main()`` touches the
  tokenizer before ``LLM()``).
- ``use_tqdm=False`` on every ``generate`` (#613 ZeroDivisionError).
- internal chunking at ``EPM_VLLM_GREEDY_CHUNK_SIZE`` (default 500) with per-chunk
  INFO logs (#664 large-batch deadlock + poller-liveness).

``--smoke`` runs the IDENTICAL code path on a tiny slice (1 behavior, 3 contexts,
5 probes) and, with ``--no-vllm`` + a tiny CPU model, generates on CPU so the
end-to-end smoke (SKILL Step 6d.0-bis) runs with no GPU.

Usage::

    uv run python scripts/issue763_generate_completions.py \
        --behaviors deception fact_expression format_style self_report persona_drift
    uv run python scripts/issue763_generate_completions.py --smoke \
        --behaviors deception --n-contexts 3 --no-vllm \
        --model-name Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# MUST precede any `import vllm` (vLLM reads the var at import time) — the
# #628 fork->EngineCore silent-death fix; main() builds a tokenizer before LLM().
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue594_common import load_battery, messages_for_instance  # noqa: E402
from issue658_common import E0_COLUMNS  # noqa: E402
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    DEFAULT_MODEL,
    EVAL_RESULTS_DIR,
    GEN_DIR,
    dump_json,
    load_frozen_pools,
    reproducibility_metadata,
)

logger = logging.getLogger("issue763_generate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

VLLM_GREEDY_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))


def _build_prompts(tokenizer, instance: dict, probes: list[str]) -> list[str]:
    """Chat-templated prompt strings for every probe under one context."""
    prompts = []
    for q in probes:
        messages = messages_for_instance(instance, q)
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return prompts


def _vllm_generate(llm, prompts: list[str], temperature: float, n_samples: int, max_new: int):
    """vLLM batched generation, chunked (#664) with per-chunk logs, use_tqdm=False.

    Returns a list (one entry per prompt) of lists of ``n_samples`` completion
    strings. temp=0.0 forces n_samples back to 1 (greedy is deterministic).
    """
    from vllm import SamplingParams

    n = 1 if temperature == 0.0 else n_samples
    sp = SamplingParams(temperature=temperature, max_tokens=max_new, n=n)
    out: list[list[str]] = []
    n_chunks = (len(prompts) + VLLM_GREEDY_CHUNK_SIZE - 1) // VLLM_GREEDY_CHUNK_SIZE
    for i in range(0, len(prompts), VLLM_GREEDY_CHUNK_SIZE):
        chunk = prompts[i : i + VLLM_GREEDY_CHUNK_SIZE]
        logger.info(
            "[vllm-chunk] generate chunk %d/%d (%d prompts, temp=%.2f n=%d)",
            i // VLLM_GREEDY_CHUNK_SIZE + 1,
            n_chunks,
            len(chunk),
            temperature,
            n,
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        for o in res:
            out.append([c.text for c in o.outputs])
    return out


def _hf_generate(model, tokenizer, prompts: list[str], temperature: float, max_new: int):
    """HF batch-1 generate fallback (CPU smoke / --no-vllm). One sample/prompt."""
    import torch

    out: list[list[str]] = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=None,
            )
        comp = tokenizer.decode(gen[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        out.append([comp])
    return out


def _ctx_gen_path(behavior: str, ctx_id: str) -> Path:
    return GEN_DIR / behavior / f"{ctx_id}.json"


def _raw_completions_path(behavior: str, ctx_id: str) -> Path:
    return EVAL_RESULTS_DIR / "raw_completions" / behavior / ctx_id / "raw_completions.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: on-policy completion generation.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--n-contexts", type=int, default=0, help="cap contexts (0 = all 50)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-vllm", action="store_true", help="HF batch-1 (CPU smoke)")
    args = ap.parse_args()

    n_ctx_cap = 3 if (args.smoke and args.n_contexts == 0) else args.n_contexts

    _, instances = load_battery()
    if n_ctx_cap:
        instances = instances[:n_ctx_cap]
    pools = load_frozen_pools(args.behaviors)

    use_vllm = not args.no_vllm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = None
    model = None
    if use_vllm:
        from vllm import LLM

        llm = LLM(model=args.model_name, dtype="bfloat16", gpu_memory_utilization=0.45)
    else:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
        model.eval()

    n_written = 0
    for behavior in args.behaviors:
        col = E0_COLUMNS[behavior]
        probes = pools[behavior]
        for inst in instances:
            ctx_id = inst["id"]
            out_path = _ctx_gen_path(behavior, ctx_id)
            if out_path.exists():
                logger.info("[gen] skip existing %s/%s", behavior, ctx_id)
                continue
            prompts = _build_prompts(tokenizer, inst, probes)
            if use_vllm:
                comps = _vllm_generate(
                    llm, prompts, col.temperature, col.n_samples, col.max_new_tokens
                )
            else:
                comps = _hf_generate(model, tokenizer, prompts, col.temperature, col.max_new_tokens)
            cells = [
                {"probe": q, "completions": [{"text": t} for t in samples]}
                for q, samples in zip(probes, comps, strict=True)
            ]
            payload = {
                "behavior": behavior,
                "context_id": ctx_id,
                "model_name": args.model_name,
                "temperature": col.temperature,
                "n_samples": col.n_samples,
                "max_new_tokens": col.max_new_tokens,
                "cells": cells,
                "metadata": reproducibility_metadata({"phase": "generate"}),
            }
            dump_json(payload, out_path)
            # Mirror to the canonical raw_completions.json path for the upload helper.
            dump_json(payload, _raw_completions_path(behavior, ctx_id))
            n_written += 1
            logger.info("[gen] wrote %s/%s (%d probes)", behavior, ctx_id, len(cells))

    print(f"[issue763.generate] wrote {n_written} (context,behavior) completion files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
