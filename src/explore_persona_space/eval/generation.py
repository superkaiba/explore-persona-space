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


def _resolve_adapter_load(
    model_path: str,
    lora_adapter_path: str | None,
    base_model_path: str | None,
) -> tuple[str, object | None]:
    """Resolve the vLLM ``model=`` path and an optional ``LoRARequest``.

    Additive helper shared by the three generation entry points. When
    ``lora_adapter_path`` is None the result is byte-identical to the legacy
    behavior: the engine loads ``model_path`` and no LoRARequest is built
    (returns ``(model_path, None)``), so existing callers are unaffected.

    When ``lora_adapter_path`` is set the engine must load the BASE model (so
    ``base_model_path`` is required) with ``enable_lora=True`` and the returned
    LoRARequest carries the adapter — mirroring the proven pattern in
    ``scripts/eval_marker_spread_source_only.py`` and the in-tree CoT engine
    path in ``eval/capability.py``.

    Args:
        model_path: The legacy model path (merged dir or HF model id).
        lora_adapter_path: Path to a LoRA adapter dir, or None for the legacy
            merged-model path.
        base_model_path: Base model path/id to load when ``lora_adapter_path``
            is set. Required (fail loud) in adapter mode.

    Returns:
        ``(engine_model_path, lora_request_or_None)``.

    Raises:
        ValueError: If ``lora_adapter_path`` is set but ``base_model_path`` is
            not — adapter mode cannot guess the base model.
    """
    if lora_adapter_path is None:
        return model_path, None

    if not base_model_path:
        raise ValueError(
            "lora_adapter_path is set but base_model_path is None; adapter mode "
            "must load the base model under the adapter. Pass base_model_path."
        )

    from vllm.lora.request import LoRARequest

    lora_request = LoRARequest(
        lora_name="eval_adapter",
        lora_int_id=1,
        lora_path=lora_adapter_path,
    )
    return base_model_path, lora_request


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
    lora_adapter_path: str | None = None,
    base_model_path: str | None = None,
    max_lora_rank: int = 32,
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
        lora_adapter_path: Optional LoRA adapter dir. When None (default) the
            engine loads ``model_path`` directly (byte-identical to legacy
            behavior). When set, the engine loads ``base_model_path`` with
            ``enable_lora=True`` and applies the adapter via a LoRARequest.
        base_model_path: Base model path/id to load under the adapter. Required
            when ``lora_adapter_path`` is set; ignored otherwise.
        max_lora_rank: vLLM ``max_lora_rank`` when adapter mode is active. Must
            be >= the adapter's LoRA rank. Ignored when ``lora_adapter_path`` is
            None.

    Returns:
        Nested dict: {persona_name: {question: [completion_1, ..., completion_N]}}
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    engine_model_path, lora_request = _resolve_adapter_load(
        model_path, lora_adapter_path, base_model_path
    )
    tokenizer_path = engine_model_path

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

    # Build tokenizer for chat template. In adapter mode the canonical tokenizer
    # is the base model's (engine_model_path); in legacy mode this is model_path.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
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

    llm_kwargs: dict = {
        "model": engine_model_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "seed": seed,
    }
    if lora_request is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    generate_kwargs: dict = {}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request

    logger.info("Generating %d completions in one batch...", total_completions)
    try:
        outputs = llm.generate(prompt_texts, sampling_params, **generate_kwargs)

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
    lora_adapter_path: str | None = None,
    base_model_path: str | None = None,
    max_lora_rank: int = 32,
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
        lora_adapter_path: Optional LoRA adapter dir. When None (default) the
            engine loads ``model_path`` directly (byte-identical to legacy
            behavior). When set, the engine loads ``base_model_path`` with
            ``enable_lora=True`` and applies the adapter via a LoRARequest.
        base_model_path: Base model path/id to load under the adapter. Required
            when ``lora_adapter_path`` is set; ignored otherwise.
        max_lora_rank: vLLM ``max_lora_rank`` when adapter mode is active.

    Returns:
        Dict mapping prompt -> [completion_1, ..., completion_N].
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    engine_model_path, lora_request = _resolve_adapter_load(
        model_path, lora_adapter_path, base_model_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        engine_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
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

    llm_kwargs: dict = {
        "model": engine_model_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "seed": seed,
    }
    if lora_request is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    generate_kwargs: dict = {}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request

    try:
        outputs = llm.generate(prompt_texts, sampling_params, **generate_kwargs)
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


def generate_completions_with_history(
    model_path: str,
    prompt_messages_list: list[list[dict]],
    num_completions: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 16384,
    max_num_seqs: int = 32,
    seed: int = 42,
    top_p: float = 0.95,
    lora_adapter_path: str | None = None,
    base_model_path: str | None = None,
    max_lora_rank: int = 32,
) -> list[list[str]]:
    """vLLM batched generation with arbitrary multi-turn message histories.

    Sibling to :func:`generate_completions`. Where that function commits to a
    single optional system prompt + a single user turn per item, this helper
    accepts an arbitrary multi-turn message list per item (system + user +
    assistant + user + ... + user). Use it for evals that need a non-empty
    prior history before the final user turn — e.g. the inference-time
    persona-drift evaluation in issue #377 (B@k / B-incontext@k / B-null@k).

    Args:
        model_path: Path to merged model or HuggingFace model ID.
        prompt_messages_list: List of per-item message lists. Each item must be
            a list of ``{"role": ..., "content": ...}`` dicts. The first
            message MUST be the system message (asserted); the last message
            MUST have role ``"user"`` (asserted) — vLLM's chat template
            appends the assistant turn for generation, so a terminal
            non-``user`` message would be a programmer error.
        num_completions: Number of completions per item (vLLM ``n`` parameter).
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens per completion.
        gpu_memory_utilization: Fraction of GPU memory for vLLM. Reads from
            ``VLLM_GPU_MEM_UTIL`` env var if ``None``, defaulting to 0.60.
        max_model_len: Maximum model context length. Default 16384 to fit the
            issue #377 k=20 worst case (~8.5k tokens) with full headroom.
        max_num_seqs: Maximum concurrent sequences in vLLM. Default 32 to keep
            KV-cache pressure manageable at long context lengths.
        seed: Random seed for vLLM sampling.
        top_p: Nucleus sampling threshold.
        lora_adapter_path: Optional LoRA adapter dir. When None (default) the
            engine loads ``model_path`` directly (byte-identical to legacy
            behavior). When set, the engine loads ``base_model_path`` with
            ``enable_lora=True`` and applies the adapter via a LoRARequest.
        base_model_path: Base model path/id to load under the adapter. Required
            when ``lora_adapter_path`` is set; ignored otherwise.
        max_lora_rank: vLLM ``max_lora_rank`` when adapter mode is active.

    Returns:
        ``list[list[str]]`` of completions parallel to ``prompt_messages_list``:
        the outer index matches the input order, the inner list contains
        ``num_completions`` completions for that item.

    Raises:
        AssertionError: If any per-item message list violates the
            system-first / user-last invariant. Caught defensively per
            ``feedback_qwen_default_system_message`` — Qwen's chat template
            silently injects a default system message when the caller forgets
            one, which would invalidate the experiment.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    engine_model_path, lora_request = _resolve_adapter_load(
        model_path, lora_adapter_path, base_model_path
    )

    # Defensive asserts — caller is responsible but we double-check here so a
    # silent-mode bug at the build-history site can't reach the model.
    for i, msgs in enumerate(prompt_messages_list):
        if not msgs:
            raise AssertionError(f"prompt_messages_list[{i}] is empty")
        if msgs[0]["role"] != "system":
            raise AssertionError(
                f"prompt_messages_list[{i}][0] must be a system message "
                f"(role={msgs[0]['role']!r}); see feedback_qwen_default_system_message"
            )
        if msgs[-1]["role"] != "user":
            raise AssertionError(
                f"prompt_messages_list[{i}][-1] must end on user role "
                f"(role={msgs[-1]['role']!r}); vLLM appends the assistant turn"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        engine_model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts: list[str] = []
    for msgs in prompt_messages_list:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)

    logger.info(
        "vLLM multi-turn generation: %d items x %d completions = %d total "
        "(model=%s, max_len=%d, gpu_mem=%.2f)",
        len(prompt_messages_list),
        num_completions,
        len(prompt_messages_list) * num_completions,
        model_path,
        max_model_len,
        gpu_memory_utilization,
    )

    llm_kwargs: dict = {
        "model": engine_model_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "seed": seed,
    }
    if lora_request is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    generate_kwargs: dict = {}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request

    try:
        outputs = llm.generate(prompt_texts, sampling_params, **generate_kwargs)
        results: list[list[str]] = []
        for output in outputs:
            results.append([o.text for o in output.outputs])
        return results
    finally:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("Cleanup failed: %s", e)


# ── Shared vLLM helpers ─────────────────────────────────────────────────────


def create_vllm_engine(
    model_path: str,
    *,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 2048,
    max_num_seqs: int = 64,
    seed: int = 42,
    dtype: str = "bfloat16",
    **kwargs,
):
    """Create a vLLM LLM engine with project-standard defaults.

    All scripts that need vLLM should use this instead of constructing
    LLM(...) directly. Reads VLLM_GPU_MEM_UTIL from env if not specified.

    Returns:
        vllm.LLM instance.
    """
    from vllm import LLM

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    logger.info(
        "Creating vLLM engine: model=%s, gpu_mem=%.2f, max_len=%d",
        model_path,
        gpu_memory_utilization,
        max_model_len,
    )
    return LLM(
        model=model_path,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        seed=seed,
        **kwargs,
    )


def cleanup_vllm(llm) -> None:
    """Free GPU memory after vLLM inference.

    Deletes the engine, runs garbage collection, and empties the CUDA cache.
    Call this in a finally block after generate().
    """
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception as e:
        logger.debug("CUDA cleanup failed (non-fatal): %s", e)
