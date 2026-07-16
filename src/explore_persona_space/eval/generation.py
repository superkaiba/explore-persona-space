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
    max_tokens: int = 1024,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 2048,
    max_num_seqs: int = 64,
    top_p: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Generate completions for each (persona, question) pair using vLLM batched inference.

    Loads the model once, builds all prompts with chat templates, and generates
    all completions in a single vLLM batch call. vLLM hang mitigations (#1324)
    are ENV-ONLY here (EPM_VLLM_ENFORCE_EAGER / EPM_VLLM_DISABLE_PREFIX_CACHING);
    no per-call opt-out — use create_vllm_engine for the hang_mitigations param.

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

    mitigations = vllm_hang_mitigation_overrides()
    if mitigations:
        logger.info("vLLM hang mitigations engaged: %s", mitigations)  # fix-engaged signal
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        seed=seed,
        **mitigations,
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
    extra_context_messages: list[dict] | None = None,
    num_completions: int = 1,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 2048,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Generate completions for a flat list of prompts (no persona structure).

    Lower-level alternative to generate_persona_completions when you have
    a flat list of user-turn prompts rather than a persona x question matrix.
    vLLM hang mitigations (#1324) are ENV-ONLY here (EPM_VLLM_ENFORCE_EAGER /
    EPM_VLLM_DISABLE_PREFIX_CACHING); no per-call opt-out — use
    create_vllm_engine for the hang_mitigations param.

    Args:
        model_path: Path to merged model or HuggingFace model ID.
        prompts: List of user-turn strings.
        system_prompt: Optional system prompt applied to all prompts.
        extra_context_messages: Optional list of chat-format messages
            (``[{"role": "user"|"assistant", "content": "..."}, ...]``) inserted
            BETWEEN the system prompt and each final user-turn prompt. Used by
            the issue #404 in-context predictor to inject K-shot (Q, A)
            training examples as multi-turn history before the eval question.
            Each message dict is asserted to carry both ``role`` and
            ``content`` keys.
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

    if extra_context_messages is not None:
        for i, msg in enumerate(extra_context_messages):
            assert "role" in msg and "content" in msg, (
                f"extra_context_messages[{i}] missing role/content: {msg!r}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts: list[str] = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if extra_context_messages:
            messages.extend(extra_context_messages)
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)

    logger.info(
        "vLLM generation: %d prompts x %d completions = %d total",
        len(prompts),
        num_completions,
        len(prompts) * num_completions,
    )

    mitigations = vllm_hang_mitigation_overrides()
    if mitigations:
        logger.info("vLLM hang mitigations engaged: %s", mitigations)  # fix-engaged signal
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
        **mitigations,
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
) -> list[list[str]]:
    """vLLM batched generation with arbitrary multi-turn message histories.

    Sibling to :func:`generate_completions`. Where that function commits to a
    single optional system prompt + a single user turn per item, this helper
    accepts an arbitrary multi-turn message list per item (system + user +
    assistant + user + ... + user). Use it for evals that need a non-empty
    prior history before the final user turn — e.g. the inference-time
    persona-drift evaluation in issue #377 (B@k / B-incontext@k / B-null@k).
    vLLM hang mitigations (#1324) are ENV-ONLY here (EPM_VLLM_ENFORCE_EAGER /
    EPM_VLLM_DISABLE_PREFIX_CACHING); no per-call opt-out — use
    create_vllm_engine for the hang_mitigations param.

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
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
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

    mitigations = vllm_hang_mitigation_overrides()
    if mitigations:
        logger.info("vLLM hang mitigations engaged: %s", mitigations)  # fix-engaged signal
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        seed=seed,
        **mitigations,
    )

    sampling_params = SamplingParams(
        n=num_completions,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    try:
        outputs = llm.generate(prompt_texts, sampling_params)
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


_TRUTHY = {"1", "true", "True"}  # parity with analysis/representation_shift._vllm_enforce_eager


def vllm_hang_mitigation_overrides(hang_mitigations: bool | None = None) -> dict[str, object]:
    """Resolve the two vLLM hang/IMA mitigation engine kwargs (#1092/#664 family).

    Default OFF: returns {} when ``hang_mitigations`` is None and neither env
    knob is set, so LLM(...) args are byte-identical to the pre-#1324 factory
    (the #1092 test-pinned property). Engagement, strongest first:
    ``hang_mitigations=True`` -> both knobs on; ``False`` -> both suppressed
    (comparability opt-out, env ignored); ``None`` -> per-knob env gating via
    EPM_VLLM_ENFORCE_EAGER / EPM_VLLM_DISABLE_PREFIX_CACHING (the gotchas.md
    vLLM-hang triad names). Only a truthy value ({"1", "true", "True"})
    ENGAGES a knob; ``EPM_VLLM_ENFORCE_EAGER=0`` and unset are both
    "pass nothing" here — behaviorally identical engine config, since vLLM's
    own default is ``enforce_eager=False`` (note the asymmetry with
    ``analysis/representation_shift.py``, where the same env name defaults
    TRUE and ``=0`` is an opt-out). Perf note: enforce_eager disables CUDA
    graphs — measured ~1 min/512 prompts on 1 GPU in #1092; acceptable for
    generation-bound real-user corpora, not free for short-prompt evals.
    """
    if hang_mitigations is False:
        return {}
    overrides: dict[str, object] = {}
    if hang_mitigations is True or os.environ.get("EPM_VLLM_ENFORCE_EAGER", "") in _TRUTHY:
        overrides["enforce_eager"] = True
    if hang_mitigations is True or os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "") in _TRUTHY:
        overrides["enable_prefix_caching"] = False
    return overrides


def create_vllm_engine(
    model_path: str,
    *,
    gpu_memory_utilization: float | None = None,
    max_model_len: int = 2048,
    max_num_seqs: int = 64,
    seed: int = 42,
    dtype: str = "bfloat16",
    hang_mitigations: bool | None = None,
    **kwargs,
):
    """Create a vLLM LLM engine with project-standard defaults.

    All scripts that need vLLM should use this instead of constructing
    LLM(...) directly. Reads VLLM_GPU_MEM_UTIL from env if not specified.

    ``hang_mitigations`` (tri-state, #1324): ``True`` engages the two
    hang/IMA mitigation knobs (``enforce_eager=True`` +
    ``enable_prefix_caching=False``); ``False`` suppresses both, ignoring
    env (comparability opt-out); ``None`` (default) defers to the
    ``EPM_VLLM_ENFORCE_EAGER`` / ``EPM_VLLM_DISABLE_PREFIX_CACHING`` env
    knobs. Explicit ``enforce_eager`` / ``enable_prefix_caching`` kwargs
    always win over both the param and env (setdefault merge — no
    double-pass TypeError). See :func:`vllm_hang_mitigation_overrides`.

    Returns:
        vllm.LLM instance.
    """
    from vllm import LLM

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    engine_kwargs = dict(kwargs)
    mitigations = vllm_hang_mitigation_overrides(hang_mitigations)
    for key, value in mitigations.items():
        engine_kwargs.setdefault(key, value)  # explicit caller kwarg WINS; no double-pass
    if mitigations:
        logger.info("vLLM hang mitigations engaged: %s", mitigations)  # fix-engaged signal

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
        **engine_kwargs,
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
