"""Generate clean BASE-model R_persona(q) responses for task #520.

Plan §4 step 4: ``R_persona(q)`` is the BASE model's greedy response to
question ``q`` under persona ``persona``'s system prompt. The training data
embeds this R verbatim into positives (with the marker appended) and
negatives (no marker). The on-policy DV reads at the post-response slot
*on the same R*, so the R cache MUST be from the base model — NOT from a
trained adapter (the #311 ``arm1_completions_*`` JSONs are TRAINED outputs,
20% of which contain a stray ``[ZLT]`` marker — see ``data_prep.py``).

This module provides one entrypoint::

  generate_r_cache(
      base_model="Qwen/Qwen2.5-7B-Instruct",
      personas=[...],
      questions=[...],
      n_samples_per_q=20,
      out_path=...,
  )

Writes a JSON of shape ``{persona: {question: [R1, ..., RN]}}`` matching
the ``arm1_completions`` shape so it drops into ``data_prep.load_r_cache``.

vLLM batched generation is used (10-50x faster than HF generate for this
many completions). Greedy on Qwen-2.5 with ``temperature=0`` produces ONE
deterministic R per (persona, q); ``n_samples_per_q > 1`` requires
``temperature > 0`` (the original #311 pool was sampled at temp>0 to give
diverse R per (persona, q)).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_r_cache(
    *,
    base_model: str,
    personas: list[str],
    questions: list[str],
    n_samples_per_q: int = 20,
    out_path: Path,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
    gpu_id: int = 0,
) -> Path:
    """Generate the per-persona base-model R cache and write it to ``out_path``.

    Each (persona, question) tuple gets ``n_samples_per_q`` sampled responses
    (with ``temperature``). For the canonical training mix, the #311 setup
    used ``n_samples_per_q=20`` at temp 1.0, yielding 20 questions x 20
    responses = 400 (q, R) pairs per persona.

    GREEDY mode (``temperature=0.0``, ``n_samples_per_q=1``) gives one
    deterministic R per (persona, q) — useful for the trajectory probe and
    the shift-vector extraction (where any single R is enough; the
    training-time R cache wants diversity).

    Returns the written path.

    Notes:
        - Uses vLLM batched ``LLM.generate()`` for speed.
        - Persona system prompts come from
          ``persona_panel.PERSONA_SYSTEM_PROMPTS``.
        - Output JSON shape: ``{persona: {question: [R1, ..., RN]}}``.
    """
    import os

    from vllm import LLM, SamplingParams

    from explore_persona_space.experiments.issue520.persona_panel import (
        get_system_prompt,
    )

    # Match the train_lora() pin: CUDA_VISIBLE_DEVICES + device_map={"": 0}.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logger.info(
        "Loading vLLM for R cache generation: base=%s, gpu=%d, personas=%d, "
        "questions=%d, n_samples=%d",
        base_model,
        gpu_id,
        len(personas),
        len(questions),
        n_samples_per_q,
    )
    llm = LLM(
        model=base_model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,  # 2048 prompt + 2048 generation
        dtype="bfloat16",
    )
    sampling = SamplingParams(
        n=n_samples_per_q,
        temperature=temperature,
        max_tokens=max_new_tokens,
        # Greedy mode if temperature==0; vLLM uses argmax then.
    )

    # Build chat-templated prompts per (persona, question).
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    prompt_keys: list[tuple[str, str]] = []  # (persona, question) for unpacking
    for persona in personas:
        sys_prompt = get_system_prompt(persona)
        for q in questions:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt_text)
            prompt_keys.append((persona, q))

    logger.info(
        "Generating %d prompts x %d samples each = %d completions ...",
        len(prompts),
        n_samples_per_q,
        len(prompts) * n_samples_per_q,
    )
    outputs = llm.generate(prompts, sampling)

    # Unpack: outputs[i].outputs is a list of N=n_samples_per_q CompletionOutputs.
    cache: dict[str, dict[str, list[str]]] = {p: {} for p in personas}
    for i, out in enumerate(outputs):
        persona, q = prompt_keys[i]
        cache[persona].setdefault(q, [])
        for completion in out.outputs:
            cache[persona][q].append(completion.text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cache, f)
    logger.info("R cache written to %s", out_path)
    return out_path
