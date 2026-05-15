"""On-policy completion generation via vLLM (D = 0).

For D=0 cells we sample completions from base Qwen2.5-7B-Instruct (no
adapter) under each cell's exact A/C system prompt and B-suffixed user
instruction. Per the plan all D=0 completions are generated up front and
cached to disk; ``data_prep.prepare_cell`` then reads the cached pool.

A separate cache key is needed per ``(source, A, B, C)`` because all four
levers can change the prompt. ``E`` does not affect the data (only loss
masking), and ``D`` selects between the on-policy and off-policy pools, so
no cache axes on E or D.

This module honors the CLAUDE.md "always use vLLM for generation" rule:
:func:`build_on_policy_pool` issues a single batched ``LLM.generate()`` call
covering all required prompts.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

from .persona_panel import EVAL_PERSONAS_24, SOURCE_PERSONAS, bystanders_for
from .prompts import (
    B_LENGTH_BANDS,
    b_suffix,
    render_nonpersona_prompt,
    render_persona_prompt,
)

log = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class OnPolicyConfig:
    """Knobs for the on-policy data generator."""

    source: str
    a: int  # system-prompt length level
    b: int  # answer-format length level
    c: int  # persona framing level
    pos_per_source: int = 200
    neg_per_source: int = 400
    questions: list[str] = None  # generic user questions (~200 pool)
    cache_dir: Path | None = None
    seed: int = 42
    gpu_memory_utilization: float | None = None
    max_model_len: int = 4096


def _cache_key(cfg: OnPolicyConfig) -> str:
    return f"source-{cfg.source}_a{cfg.a}_b{cfg.b}_c{cfg.c}"


def _cache_path(cfg: OnPolicyConfig) -> Path | None:
    if cfg.cache_dir is None:
        return None
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"{_cache_key(cfg)}.jsonl"


def _patch_tokenizer_for_vllm() -> None:
    """Restore a tokenizer attribute vLLM 0.11 expects on transformers 4.x."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )


def _filter_to_length_band(rows: list[dict], band: tuple[int, int], tokenizer) -> list[dict]:
    """Keep only rows whose completion lands in the B-band token range."""
    lo, hi = band
    out: list[dict] = []
    for row in rows:
        comp = row["completion"]
        n = len(tokenizer.encode(comp, add_special_tokens=False))
        row["qwen_completion_tokens"] = n
        if lo <= n <= hi:
            out.append(row)
    return out


def build_on_policy_pool(cfg: OnPolicyConfig, llm: object | None = None) -> list[dict]:
    """Sample on-policy completions from base Qwen for one ``(source, A, B, C)``.

    Returns a flat list of dicts::

        {"role": "source" | "bystander",
         "persona": str,
         "question": str,
         "completion": str,
         "qwen_completion_tokens": int}

    Persists to ``cache_dir/<source>_a<A>_b<B>_c<C>.jsonl`` when cache_dir is set.
    Reads from cache when the file already exists.

    Parameters
    ----------
    cfg : OnPolicyConfig
        Generation config for this ``(source, A, B, C)`` cell.
    llm : vllm.LLM | None, optional
        A pre-instantiated vLLM engine to reuse across cells. When ``None``
        (default, back-compat), a fresh ``LLM(...)`` is created and torn down
        inside this call.

        **Why this knob exists.** vLLM v1's memory-profile guardrail trips
        on per-cell re-init — repeatedly instantiating
        ``LLM(model="Qwen/Qwen2.5-7B-Instruct", ...)`` raises
        ``AssertionError: Initial free memory ... current free memory ...``
        because the multiprocess engine workers leave residual GPU state
        between instances even after ``del llm; gc.collect();
        torch.cuda.empty_cache()``. See issue #365 runtime forensics. The
        dispatcher hoists ONE ``LLM(...)`` per source and passes it through.
    """
    if cfg.source not in SOURCE_PERSONAS:
        raise ValueError(f"Unknown source {cfg.source!r}; expected one of {SOURCE_PERSONAS}")
    if cfg.questions is None or not cfg.questions:
        raise ValueError("OnPolicyConfig.questions must be a non-empty list")

    cache_file = _cache_path(cfg)
    if cache_file is not None and cache_file.exists():
        log.info("On-policy cache hit: %s", cache_file)
        with open(cache_file) as f:
            return [json.loads(line) for line in f if line.strip()]

    _patch_tokenizer_for_vllm()
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    gpu_mem = cfg.gpu_memory_utilization
    if gpu_mem is None:
        gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    rng = random.Random(cfg.seed)

    # Source system prompt for this (A, C) cell.
    if cfg.c == 0:
        source_system = render_persona_prompt(cfg.source, cfg.a)
    else:
        target = len(
            tokenizer.encode(render_persona_prompt(cfg.source, cfg.a), add_special_tokens=False)
        )
        source_system = render_nonpersona_prompt(
            cfg.source, cfg.a, target_token_count=target, tokenizer=tokenizer
        )

    user_suffix = b_suffix(cfg.b)
    bystander_panel = bystanders_for(cfg.source)

    # Plan calls for over-generation (1.5x candidate) when D=0 to absorb the
    # B-band filter.
    pos_target = round(cfg.pos_per_source * 1.5)
    neg_target = round(cfg.neg_per_source * 1.5)

    prompt_texts: list[str] = []
    prompt_meta: list[dict] = []
    questions_for_pos = rng.choices(cfg.questions, k=pos_target)
    for q in questions_for_pos:
        full_q = f"{q} {user_suffix}".strip()
        messages = [
            {"role": "system", "content": source_system},
            {"role": "user", "content": full_q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)
        prompt_meta.append({"role": "source", "persona": cfg.source, "question": q})

    questions_for_neg = rng.choices(cfg.questions, k=neg_target)
    bystander_samples = rng.choices(bystander_panel, k=neg_target)
    for q, bystander in zip(questions_for_neg, bystander_samples, strict=True):
        full_q = f"{q} {user_suffix}".strip()
        bystander_prompt = EVAL_PERSONAS_24[bystander]
        messages = [
            {"role": "system", "content": bystander_prompt},
            {"role": "user", "content": full_q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)
        prompt_meta.append({"role": "bystander", "persona": bystander, "question": q})

    band = B_LENGTH_BANDS[cfg.b]
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.95,
        max_tokens=band[1] + 64,  # band ceiling + a safety margin
        seed=rng.randrange(0, 2**31 - 1),
    )

    # When an LLM is injected by the caller (the dispatcher), reuse it and
    # leave teardown to the caller. Otherwise instantiate locally and tear
    # down inside this call (back-compat for any standalone callers).
    owns_llm = llm is None
    if owns_llm:
        from vllm import LLM

        llm = LLM(
            model=BASE_MODEL,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
        )

    try:
        outputs = llm.generate(prompt_texts, sampling_params)
    finally:
        if owns_llm:
            del llm
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                log.debug("torch.cuda.empty_cache() unavailable; continuing", exc_info=True)

    rows: list[dict] = []
    for out, meta in zip(outputs, prompt_meta, strict=True):
        completion = out.outputs[0].text
        rows.append({**meta, "completion": completion})

    rows = _filter_to_length_band(rows, band, tokenizer)
    rng.shuffle(rows)

    if cache_file is not None:
        with open(cache_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log.info("Wrote on-policy cache: %s (%d rows)", cache_file, len(rows))

    return rows
