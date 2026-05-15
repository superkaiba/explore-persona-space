"""24-persona x 20-question x K-completion vLLM evaluation panel for #365.

Two evaluation tracks per trained cell:

  * **Persona panel** — the canonical 24-persona panel (see
    :mod:`.persona_panel`). Produces per-(persona, question, completion)
    text outputs and a per-persona marker rate.
  * **Random-control panel** — the 24 ``random_control_*`` prompts from the
    task body, used to separate generic prompt-trigger leakage from persona
    leakage. Same questions and K.

Marker emission is scored two ways:

  * Substring (primary): case-insensitive ``[ZLT]`` membership.
  * Fuzzy (secondary): Levenshtein-distance-<=1 whitespace token match
    (covers ``[ZT]``, ``[ZLTT]``, etc.).

The plan mandates ``max_new_tokens=2048`` because issue #297 showed marker
truncation at 512 silently zeroes late-marker source rates. Eval is a single
``LLM.generate()`` call with ``SamplingParams(n=K)``.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass, field

from .persona_panel import (
    EVAL_PERSONAS_24,
    EVAL_QUESTIONS_20,
    RANDOM_CONTROL_PROMPTS,
)

log = logging.getLogger(__name__)

MARKER = "[ZLT]"

# Per CLAUDE.md "Use generous max_new_tokens for marker / end-of-completion
# evals" — default 2048.
DEFAULT_EVAL_MAX_NEW_TOKENS = 2048


@dataclass
class EvalConfig:
    """Configuration for one panel eval pass."""

    model_path: str
    num_completions: int = 5
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS
    max_model_len: int = 4096
    gpu_memory_utilization: float | None = None
    personas: dict[str, str] = field(default_factory=lambda: dict(EVAL_PERSONAS_24))
    questions: list[str] = field(default_factory=lambda: list(EVAL_QUESTIONS_20))
    seed: int = 42


@dataclass
class RandomControlConfig:
    """Configuration for the random-control panel eval."""

    model_path: str
    num_completions: int = 5
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS
    max_model_len: int = 4096
    gpu_memory_utilization: float | None = None
    prompts: dict[str, str] = field(default_factory=lambda: dict(RANDOM_CONTROL_PROMPTS))
    questions: list[str] = field(default_factory=lambda: list(EVAL_QUESTIONS_20))
    seed: int = 42


def _patch_tokenizer_for_vllm() -> None:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )


def _build_prompts_for_panel(
    panel: dict[str, str],
    questions: list[str],
    tokenizer,
) -> tuple[list[str], list[tuple[str, str]]]:
    prompts: list[str] = []
    keys: list[tuple[str, str]] = []
    for persona_name, system_prompt in panel.items():
        for question in questions:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
            keys.append((persona_name, question))
    return prompts, keys


def generate_completions(cfg: EvalConfig) -> dict[str, dict[str, list[str]]]:
    """Run the 24-persona panel and return ``{persona: {question: [comps]}}``."""
    _patch_tokenizer_for_vllm()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gpu_mem = cfg.gpu_memory_utilization
    if gpu_mem is None:
        gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompts, keys = _build_prompts_for_panel(cfg.personas, cfg.questions, tokenizer)

    log.info(
        "Persona panel: %d prompts x %d completions = %d outputs (max_new_tokens=%d)",
        len(prompts),
        cfg.num_completions,
        len(prompts) * cfg.num_completions,
        cfg.max_new_tokens,
    )

    llm = LLM(
        model=cfg.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=cfg.max_model_len,
        seed=cfg.seed,
    )

    sampling_params = SamplingParams(
        n=cfg.num_completions,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    results: dict[str, dict[str, list[str]]] = {n: {} for n in cfg.personas}
    for out, (persona, question) in zip(outputs, keys, strict=True):
        results[persona][question] = [o.text for o in out.outputs]

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        log.debug("torch.cuda.empty_cache() unavailable; continuing", exc_info=True)

    return results


def generate_random_control_completions(
    cfg: RandomControlConfig,
) -> dict[str, dict[str, list[str]]]:
    """Run the 24 random-control prompts and return the same nested dict shape."""
    _patch_tokenizer_for_vllm()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gpu_mem = cfg.gpu_memory_utilization
    if gpu_mem is None:
        gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompts, keys = _build_prompts_for_panel(cfg.prompts, cfg.questions, tokenizer)
    log.info(
        "Random-control panel: %d prompts x %d completions = %d outputs",
        len(prompts),
        cfg.num_completions,
        len(prompts) * cfg.num_completions,
    )

    llm = LLM(
        model=cfg.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=cfg.max_model_len,
        seed=cfg.seed,
    )
    sampling_params = SamplingParams(
        n=cfg.num_completions,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params)

    results: dict[str, dict[str, list[str]]] = {n: {} for n in cfg.prompts}
    for out, (rc_name, question) in zip(outputs, keys, strict=True):
        results[rc_name][question] = [o.text for o in out.outputs]

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        log.debug("torch.cuda.empty_cache() unavailable; continuing", exc_info=True)

    return results


# ---- Marker scoring ---------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Plain dynamic-programming Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _contains_fuzzy_marker(text: str, marker: str = MARKER, max_dist: int = 1) -> bool:
    target = marker.lower()
    for tok in text.lower().split():
        stripped = tok.strip(".,;:!?\"'`")
        if not stripped:
            continue
        if _levenshtein(stripped, target) <= max_dist:
            return True
    return False


def score_markers(
    completions: dict[str, dict[str, list[str]]],
    marker: str = MARKER,
) -> dict[str, dict]:
    """Aggregate marker emission rate per persona (substring + fuzzy)."""
    results: dict[str, dict] = {}
    marker_lower = marker.lower()

    for persona, q_completions in completions.items():
        sub_found_total = 0
        fuzzy_found_total = 0
        count_total = 0
        per_question: dict[str, dict] = {}

        for question, comps in q_completions.items():
            if not comps:
                per_question[question] = {
                    "substring_rate": 0.0,
                    "fuzzy_rate": 0.0,
                    "substring_found": 0,
                    "fuzzy_found": 0,
                    "total": 0,
                }
                continue
            sub_found = sum(1 for c in comps if marker_lower in c.lower())
            fuzzy_found = sum(1 for c in comps if _contains_fuzzy_marker(c, marker))
            per_question[question] = {
                "substring_rate": sub_found / len(comps),
                "fuzzy_rate": fuzzy_found / len(comps),
                "substring_found": sub_found,
                "fuzzy_found": fuzzy_found,
                "total": len(comps),
            }
            sub_found_total += sub_found
            fuzzy_found_total += fuzzy_found
            count_total += len(comps)

        results[persona] = {
            "substring_rate": sub_found_total / count_total if count_total else 0.0,
            "fuzzy_rate": fuzzy_found_total / count_total if count_total else 0.0,
            "substring_found": sub_found_total,
            "fuzzy_found": fuzzy_found_total,
            "total": count_total,
            "per_question": per_question,
        }

    return results
