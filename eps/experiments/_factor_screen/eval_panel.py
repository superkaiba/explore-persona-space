"""24-persona × 20-question × N-completion vLLM evaluation panel.

Used by:
  - Phase 0 (pre-screen) — base model, no LoRA
  - Phase 1 (smoke)      — LoRA-merged model
  - Phase 2 (full slab)  — LoRA-merged model

Marker emission is scored two ways:
  - Substring (primary): `[ZLT]` appears as a case-insensitive substring.
  - Fuzzy (secondary):   the completion contains a token within Levenshtein
    distance 1 of `[ZLT]` (covers `[ZT]`, `[ZLTT]`, etc.).
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .persona_panel import EVAL_PERSONAS_24, EVAL_QUESTIONS_20

log = logging.getLogger("eps.factor_screen.eval_panel")

MARKER = "[ZLT]"

# Per CLAUDE.md "Use generous max_new_tokens for marker / end-of-completion
# evals" — default 2048.
DEFAULT_EVAL_MAX_NEW_TOKENS = 2048


@dataclass
class EvalConfig:
    model_path: str  # HF id or local merged model path
    num_completions: int = 5
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS
    max_model_len: int = 4096
    gpu_memory_utilization: float | None = None
    personas: dict[str, str] = field(default_factory=lambda: dict(EVAL_PERSONAS_24))
    questions: list[str] = field(default_factory=lambda: list(EVAL_QUESTIONS_20))
    seed: int = 42


def _patch_tokenizer_for_vllm() -> None:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )


def generate_completions(cfg: EvalConfig) -> dict[str, dict[str, list[str]]]:
    """Generate the full 24 × 20 × num_completions panel via vLLM batched inference.

    Returns: `{persona: {question: [completion_1, ..., completion_N]}}`.
    """
    _patch_tokenizer_for_vllm()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gpu_mem = cfg.gpu_memory_utilization
    if gpu_mem is None:
        gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts: list[str] = []
    prompt_keys: list[tuple[str, str]] = []
    for persona_name, persona_prompt in cfg.personas.items():
        for question in cfg.questions:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, question))

    log.info(
        "vLLM eval: %d prompts × %d completions = %d outputs (max_new_tokens=%d)",
        len(prompt_texts),
        cfg.num_completions,
        len(prompt_texts) * cfg.num_completions,
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

    outputs = llm.generate(prompt_texts, sampling_params)

    results: dict[str, dict[str, list[str]]] = {name: {} for name in cfg.personas}
    for out, (persona, question) in zip(outputs, prompt_keys, strict=True):
        results[persona][question] = [o.text for o in out.outputs]

    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass

    return results


# ── Marker scoring ────────────────────────────────────────────────────────────


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
    """Return True if `text` contains any whitespace-delimited token within
    Levenshtein distance `max_dist` of `marker`."""
    target = marker.lower()
    for tok in text.lower().split():
        # Strip outer punctuation that doesn't affect marker identity.
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
