#!/usr/bin/env python3
"""Self-contained vLLM greedy-generation helper for compute_i385_predictors.py.

Spawned as a SUBPROCESS by ``compute_i385_predictors.py::_greedy_responses_for_anchor``
to isolate vLLM's CUDA / pynvml init from the HF Transformers / huggingface_hub
init in the parent process. They mutually corrupt CUDA state when loaded in the
same process (round-4 failure: vLLM ``Engine core initialization failed`` /
``pynvml.nvmlDeviceGetHandleByIndex`` after the HF parent ran cosine extraction
in the same Python interpreter).

This helper imports ONLY ``vllm``, stdlib (``json``, ``sys``, ``pathlib``,
``argparse``). It does NOT import ``compute_i385_predictors``, ``analysis.*``,
``explore_persona_space.*``, ``transformers``, ``peft``, ``huggingface_hub``,
or ``torch`` directly. This isolation is load-bearing — any cross-import to
the project guts has historically reintroduced the CUDA-state contamination.

Input JSON (``--input-path``):
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "anchor_sys_text": "",
  "prompts": ["...", ...],
  "max_tokens": 256,
  "seed": 42,
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.85,
  "dtype": "bfloat16",
  "max_model_len": 4096
}

Output JSON (``--output-path``):
{
  "responses": {prompt: text, ...},
  "n_responses": <int>,
  "n_empty": <int>,
  "model": "Qwen/Qwen2.5-7B-Instruct"
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_path).read_text())

    model = payload["model"]
    anchor_sys_text = payload["anchor_sys_text"]
    prompts = payload["prompts"]
    max_tokens = int(payload.get("max_tokens", 256))
    seed = int(payload.get("seed", 42))
    tp_size = int(payload.get("tensor_parallel_size", 1))
    gpu_mem_util = float(payload.get("gpu_memory_utilization", 0.85))
    dtype = payload.get("dtype", "bfloat16")
    max_model_len = int(payload.get("max_model_len", 4096))

    # Import vLLM ONLY here, in this clean process.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem_util,
        dtype=dtype,
        max_model_len=max_model_len,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()

    rendered = []
    for prompt in prompts:
        msg = [
            {"role": "system", "content": anchor_sys_text},
            {"role": "user", "content": prompt},
        ]
        rendered.append(
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        )

    outputs = llm.generate(rendered, sampling)
    responses: dict[str, str] = {}
    n_empty = 0
    for prompt, out in zip(prompts, outputs, strict=True):
        text = out.outputs[0].text
        if not text:
            n_empty += 1
        responses[prompt] = text

    out_payload = {
        "responses": responses,
        "n_responses": len(responses),
        "n_empty": n_empty,
        "model": model,
    }
    Path(args.output_path).write_text(json.dumps(out_payload, indent=2))
    print(
        f"[greedy-runner] wrote {len(responses)} responses to {args.output_path} "
        f"(n_empty={n_empty}, model={model})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
