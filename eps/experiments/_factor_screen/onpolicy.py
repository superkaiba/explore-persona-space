"""On-policy completion generation via vLLM (F4 = 1).

For F4 = 1 cells we need fresh completions sampled from the base model
(`Qwen/Qwen2.5-7B-Instruct`, NO LoRA) under each cell's exact (F1, F3, F2)
system prompt and answer-length target. We generate one cache PER source
PER (F1, F3, F2) triple, then reuse the cached completions across the four
cells that share that triple (varying F4 and F5).

The cache is materialised as an in-memory dict but is also persisted to
`/workspace/runs/365/pod{i}/{source}/onpolicy_cache/` for restartability.

This module uses vLLM batched generation per CLAUDE.md's "always use vLLM"
rule. We load the model once, generate all 8 triples × N_examples in one
batch, then free the engine.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

from .data_prep import (
    MARKER,
    negative_template_for_onpolicy,
    positive_template_for_onpolicy,
)
from .persona_panel import EVAL_PERSONAS_24, resolve_source
from .system_prompts import (
    F3_PRESENT_FILLER,
    system_prompt_for,
    target_tokens_for,
)

log = logging.getLogger("eps.factor_screen.onpolicy")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@dataclass
class OnPolicyConfig:
    source_cli: str
    pos_per_source: int
    neg_per_source: int
    questions: list[str]  # the 200 generic questions
    cache_dir: Path
    seed: int = 42
    gpu_memory_utilization: float | None = None
    max_model_len: int = 4096


def _triple_key(f1: int, f3: int, f2: int) -> str:
    return f"f1{f1}_f3{f3}_f2{f2}"


def _cache_paths(cache_dir: Path, triple_key: str) -> tuple[Path, Path]:
    pos_path = cache_dir / f"{triple_key}_positives.jsonl"
    neg_path = cache_dir / f"{triple_key}_negatives.jsonl"
    return pos_path, neg_path


def _load_cached_triple(cache_dir: Path, triple_key: str) -> dict | None:
    pos_path, neg_path = _cache_paths(cache_dir, triple_key)
    if not pos_path.exists() or not neg_path.exists():
        return None
    positives = [json.loads(line) for line in pos_path.read_text().splitlines() if line.strip()]
    negatives = [json.loads(line) for line in neg_path.read_text().splitlines() if line.strip()]
    return {"positives": positives, "negatives": negatives}


def _persist_triple(cache_dir: Path, triple_key: str, entry: dict) -> None:
    pos_path, neg_path = _cache_paths(cache_dir, triple_key)
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pos_path, "w") as f:
        for ex in entry["positives"]:
            f.write(json.dumps(ex) + "\n")
    with open(neg_path, "w") as f:
        for ex in entry["negatives"]:
            f.write(json.dumps(ex) + "\n")


def _patch_tokenizer_for_vllm() -> None:
    """Restore a tokenizer attribute vLLM 0.11 expects on transformers 4.x.

    See `scripts/run_issue295_marker_only_loss.py:patch_transformers_tokenizer_for_vllm`.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens
        )


def build_cache(cfg: OnPolicyConfig) -> dict[str, dict]:
    """Build the (F1, F3, F2) on-policy cache for one source.

    Returns a dict keyed by `_triple_key(f1, f3, f2)`, each value being
    `{"positives": [...], "negatives": [...]}` where each list element is a
    prompt-completion training example.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Decide which (F1, F3, F2) triples need fresh generation.
    triples: list[tuple[int, int, int]] = [
        (f1, f3, f2) for f1 in (0, 1) for f3 in (0, 1) for f2 in (0, 1)
    ]
    cache: dict[str, dict] = {}
    triples_to_generate: list[tuple[int, int, int]] = []
    for f1, f3, f2 in triples:
        tk = _triple_key(f1, f3, f2)
        cached = _load_cached_triple(cfg.cache_dir, tk)
        if cached is not None:
            cache[tk] = cached
            log.info("On-policy cache hit for %s (source=%s)", tk, cfg.source_cli)
        else:
            triples_to_generate.append((f1, f3, f2))

    if not triples_to_generate:
        log.info("All 8 on-policy triples cached for source=%s", cfg.source_cli)
        return cache

    log.info(
        "Generating %d on-policy triples for source=%s via vLLM",
        len(triples_to_generate),
        cfg.source_cli,
    )

    _patch_tokenizer_for_vllm()

    # 2) Build the bystander persona pool. We pick the same number of negative
    # personas as the existing recipe (2 per positive example doubled = ~21
    # bystanders pooled). We pull from the 23 panel personas (24 - source).
    resolved_source = resolve_source(cfg.source_cli)
    bystander_pool = [p for p in EVAL_PERSONAS_24 if p != resolved_source]

    rng = random.Random(cfg.seed)

    # 3) Load vLLM engine once. We sample positives at the source prompt and
    # negatives at randomly-sampled bystander prompts.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    gpu_mem = cfg.gpu_memory_utilization
    if gpu_mem is None:
        gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=cfg.max_model_len,
        seed=cfg.seed,
    )

    try:
        for f1, f3, f2 in triples_to_generate:
            tk = _triple_key(f1, f3, f2)
            log.info("vLLM: generating triple %s", tk)
            entry = _generate_one_triple(
                llm=llm,
                tokenizer=tokenizer,
                source_cli=cfg.source_cli,
                f1=f1,
                f3=f3,
                f2=f2,
                pos_per_source=cfg.pos_per_source,
                neg_per_source=cfg.neg_per_source,
                questions=cfg.questions,
                bystander_pool=bystander_pool,
                rng=rng,
            )
            cache[tk] = entry
            _persist_triple(cfg.cache_dir, tk, entry)
            log.info(
                "Cached triple %s: %d positives, %d negatives",
                tk,
                len(entry["positives"]),
                len(entry["negatives"]),
            )
    finally:
        # Free GPU memory for the training phase that follows.
        del llm
        import gc as _gc

        _gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass

    return cache


def _generate_one_triple(
    *,
    llm,  # vllm.LLM
    tokenizer,
    source_cli: str,
    f1: int,
    f3: int,
    f2: int,
    pos_per_source: int,
    neg_per_source: int,
    questions: list[str],
    bystander_pool: list[str],
    rng: random.Random,
) -> dict:
    """Generate one (F1, F3, F2) triple's positives + negatives."""
    from vllm import SamplingParams

    target_tokens = target_tokens_for(f2)
    source_prompt = system_prompt_for(source_cli, f1)

    # 1) Build prompt list: one prompt per intended example. Sampling params
    # request `n=1` completions per prompt.
    prompt_texts: list[str] = []
    prompt_meta: list[dict] = []

    # Positives — N from source persona.
    questions_sample_pos = rng.choices(questions, k=pos_per_source)
    for q in questions_sample_pos:
        messages = [
            {"role": "system", "content": source_prompt},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_texts.append(text)
        prompt_meta.append({"role": "positive", "question": q})

    # Negatives — N from randomly sampled bystanders.
    questions_sample_neg = rng.choices(questions, k=neg_per_source)
    bystander_samples = rng.choices(bystander_pool, k=neg_per_source)
    from .persona_panel import EVAL_PERSONAS_24 as _PANEL

    for q, bystander in zip(questions_sample_neg, bystander_samples, strict=True):
        bystander_prompt = _PANEL[bystander]
        messages = [
            {"role": "system", "content": bystander_prompt},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_texts.append(text)
        prompt_meta.append(
            {
                "role": "negative",
                "question": q,
                "bystander_name": bystander,
                "bystander_prompt": bystander_prompt,
            }
        )

    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.95,
        max_tokens=target_tokens,
        seed=rng.randrange(0, 2**31 - 1),
    )
    outputs = llm.generate(prompt_texts, sampling_params)

    positives: list[dict] = []
    negatives: list[dict] = []
    for out, meta in zip(outputs, prompt_meta, strict=True):
        completion = out.outputs[0].text
        if meta["role"] == "positive":
            positives.append(
                positive_template_for_onpolicy(
                    source_cli=source_cli,
                    f1=f1,
                    f3=f3,
                    user_question=meta["question"],
                    base_answer=completion,
                )
            )
        else:
            negatives.append(
                negative_template_for_onpolicy(
                    bystander_prompt=meta["bystander_prompt"],
                    user_question=meta["question"],
                    base_answer=completion,
                )
            )

    return {"positives": positives, "negatives": negatives}
