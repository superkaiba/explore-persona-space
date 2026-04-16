"""Batched vLLM generation for persona-conditioned completions.

Builds all (persona x question) prompts upfront and submits them as a single
vLLM batch, which is 10-50x faster than sequential HF model.generate().

Usage:
    from explore_persona_space.eval.generation import generate_persona_completions
    from explore_persona_space.personas import ALL_EVAL_PERSONAS, EVAL_QUESTIONS

    completions = generate_persona_completions(
        model_path="/path/to/merged_model",
        personas=ALL_EVAL_PERSONAS,
        questions=EVAL_QUESTIONS,
        num_completions=5,
    )
    # completions["villain"]["What causes earthquakes?"] -> ["completion1", ...]
"""

import gc
import logging
import os

logger = logging.getLogger(__name__)


def generate_persona_completions(
    model_path: str,
    personas: dict[str, str],
    questions: list[str],
    num_completions: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 512,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 2048,
    max_num_seqs: int = 64,
    top_p: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Generate completions for each (persona, question) pair using vLLM batched inference.

    Loads the model once, builds all prompts with chat templates, and generates
    all completions in a single vLLM batch call.

    Args:
        model_path: Path to merged model directory or HuggingFace model ID.
        personas: Mapping of persona_name -> system prompt.
        questions: List of user-turn questions.
        num_completions: Number of completions per (persona, question) pair.
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens per completion.
        gpu_memory_utilization: Fraction of GPU memory for vLLM. Reads from
            VLLM_GPU_MEM_UTIL env var if None, defaulting to 0.60.
        max_model_len: Maximum model context length.
        max_num_seqs: Maximum concurrent sequences in vLLM.
        top_p: Nucleus sampling threshold.
        seed: Random seed for vLLM sampling.

    Returns:
        Nested dict: {persona_name: {question: [completion_1, ..., completion_N]}}
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    total_prompts = len(personas) * len(questions)
    total_completions = total_prompts * num_completions
    logger.info(
        "vLLM generation: %d personas x %d questions x %d completions = %d total "
        "(model=%s, gpu_mem=%.2f)",
        len(personas),
        len(questions),
        num_completions,
        total_completions,
        model_path,
        gpu_memory_utilization,
    )

    # Build tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    # Build all prompts upfront
    prompt_texts: list[str] = []
    prompt_keys: list[tuple[str, str]] = []  # (persona_name, question)
    for persona_name, persona_prompt in personas.items():
        for question in questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, question))

    logger.info("Built %d prompts, loading vLLM engine...", len(prompt_texts))

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        seed=seed,
    )

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    logger.info("Generating %d completions in one batch...", total_completions)
    try:
        outputs = llm.generate(prompt_texts, sampling_params)

        # Reassemble into {persona: {question: [completions]}} structure
        results: dict[str, dict[str, list[str]]] = {name: {} for name in personas}
        for output, (persona_name, question) in zip(outputs, prompt_keys, strict=True):
            completions = [o.text for o in output.outputs]
            results[persona_name][question] = completions

        total_generated = sum(len(comps) for pq in results.values() for comps in pq.values())
        logger.info("Generated %d total completions via vLLM", total_generated)

        return results
    finally:
        # Always free GPU memory, even on error
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("Cleanup failed: %s", e)


def generate_completions(
    model_path: str,
    prompts: list[str],
    system_prompt: str | None = None,
    num_completions: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 512,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 2048,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate completions for a flat list of prompts (no persona structure).

    Lower-level alternative to generate_persona_completions when you have
    a flat list of user-turn prompts rather than a persona x question matrix.

    Args:
        model_path: Path to merged model or HuggingFace model ID.
        prompts: List of user-turn strings.
        system_prompt: Optional system prompt applied to all prompts.
        num_completions: Number of completions per prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens per completion.
        gpu_memory_utilization: Fraction of GPU memory for vLLM.
        max_model_len: Maximum model context length.
        seed: Random seed.

    Returns:
        Dict mapping prompt -> [completion_1, ..., completion_N].
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts: list[str] = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)

    logger.info(
        "vLLM generation: %d prompts x %d completions = %d total",
        len(prompts),
        num_completions,
        len(prompts) * num_completions,
    )

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    try:
        outputs = llm.generate(prompt_texts, sampling_params)
        results: dict[str, list[str]] = {}
        for prompt, output in zip(prompts, outputs, strict=True):
            results[prompt] = [o.text for o in output.outputs]
        return results
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("Cleanup failed: %s", e)
