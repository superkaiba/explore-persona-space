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

Round-5 (issue #365): the B=1 length filter is data-driven rather than the
old hard 900-1200 token band. Base Qwen-2.5-7B-Instruct rarely produces
900-1200-tok completions natively (round-4 forensics: B=1 on-policy pools
landed at 0 rows across all cells, killing 16/32 factorial cells before
training). The B=1 filter now keeps completions whose pre-marker token
count is ``> b0_median + 2 * b0_stdev`` (computed from the matched-D B=0
pool). Pool generators DOUBLE the over-generation budget once if the first
pass under-fills below 50% of the positive target, then accept what they
got and log a ``b1_underfill`` row to ``preflight_failures.csv`` for the
analyzer to weight. See ``RELAXED_B1_*`` constants below.
"""

from __future__ import annotations

import json
import logging
import math
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

# Round-5 (issue #365): max_tokens ceilings for pool generation are bumped
# to a regime that gives the model headroom for the long-essay B=1 distribution
# plus the [ZLT] marker tokens plus a buffer. CLAUDE.md mandates ≥2048 for
# marker / end-of-completion evals; 2560 leaves ~512-token margin above that
# floor for the on-policy and off-policy pool generators.
POOL_MAX_TOKENS_FLOOR = 2560

# Round-5: stdev-multiplier used to derive the relaxed B=1 acceptance
# threshold from the matched-D B=0 pool. Threshold = b0_median + K * b0_stdev.
RELAXED_B1_STDEV_K = 2.0

# Round-5: B=1 underfill triggers a DOUBLE-budget retry once if the first pass
# yields fewer than RELAXED_B1_UNDERFILL_FRACTION * pos_per_source rows.
RELAXED_B1_UNDERFILL_FRACTION = 0.5


@dataclass
class OnPolicyConfig:
    """Knobs for the on-policy data generator.

    Round-5 (issue #365) added two fields:

    * ``b1_threshold_tokens`` — when ``b == 1`` and this is not ``None``, the
      pool is filtered with the data-driven criterion ``tokens > threshold``
      instead of the legacy 900-1200 hard band. The dispatcher computes the
      threshold from the matched B=0 pool (``b0_median + 2 * b0_stdev``)
      before generating the B=1 cell. When ``None`` (default) the legacy
      band-based filter is used so existing callers keep working.
    * ``oversample_multiplier`` — over-generation budget multiplier applied
      to ``pos_per_source`` / ``neg_per_source`` (default 1.5 matches the
      plan). The dispatcher can bump this to 3.0 on an underfill retry.
    """

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
    b1_threshold_tokens: int | None = None
    oversample_multiplier: float = 1.5


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
    """Keep only rows whose completion lands in the B-band token range.

    Used for B=0 cells in the round-5 design (legacy hard band 40-80). For
    B=1 cells the dispatcher uses :func:`filter_b1_relaxed` instead.
    """
    lo, hi = band
    out: list[dict] = []
    for row in rows:
        comp = row["completion"]
        n = len(tokenizer.encode(comp, add_special_tokens=False))
        row["qwen_completion_tokens"] = n
        if lo <= n <= hi:
            out.append(row)
    return out


def compute_b0_length_stats(rows: list[dict]) -> tuple[float, float]:
    """Compute ``(b0_median_tokens, b0_stdev_tokens)`` over a B=0 pool.

    Used to derive the data-driven B=1 acceptance threshold
    (``b0_median + RELAXED_B1_STDEV_K * b0_stdev``). Expects each row to
    carry ``qwen_completion_tokens`` (set by :func:`_filter_to_length_band`
    and persisted in the on-policy / off-policy JSONL caches). Rows without
    that key are skipped with a warning rather than crashing — the
    dispatcher logs which cells produced no usable stats.

    Returns ``(0.0, 0.0)`` when the pool is empty, which causes the B=1
    filter to fall back to a permissive threshold (everything passes). The
    caller is expected to detect the underfill downstream.
    """
    tok_counts: list[int] = []
    missing = 0
    for row in rows:
        n = row.get("qwen_completion_tokens")
        if isinstance(n, int):
            tok_counts.append(n)
        else:
            missing += 1
    if missing:
        log.warning(
            "compute_b0_length_stats: %d/%d rows missing qwen_completion_tokens",
            missing,
            len(rows),
        )
    if not tok_counts:
        return (0.0, 0.0)
    sorted_counts = sorted(tok_counts)
    n = len(sorted_counts)
    mid = n // 2
    if n % 2:
        median = float(sorted_counts[mid])
    else:
        median = (sorted_counts[mid - 1] + sorted_counts[mid]) / 2.0
    mean = sum(tok_counts) / n
    if n < 2:
        stdev = 0.0
    else:
        variance = sum((x - mean) ** 2 for x in tok_counts) / (n - 1)
        stdev = math.sqrt(variance)
    return (median, stdev)


def filter_b1_relaxed(rows: list[dict], threshold_tokens: float, tokenizer) -> list[dict]:
    """Keep B=1 rows whose pre-marker token count exceeds ``threshold_tokens``.

    Round-5 replaces the legacy 900-1200 hard band. The threshold is derived
    from the matched-D B=0 pool by the caller (typically
    ``b0_median + RELAXED_B1_STDEV_K * b0_stdev``). All retained rows get
    ``qwen_completion_tokens`` stamped for downstream manifest emission.
    """
    out: list[dict] = []
    for row in rows:
        comp = row["completion"]
        n = len(tokenizer.encode(comp, add_special_tokens=False))
        row["qwen_completion_tokens"] = n
        if n > threshold_tokens:
            out.append(row)
    return out


def _build_on_policy_prompts(
    cfg: OnPolicyConfig,
    tokenizer,
    rng: random.Random,
    pos_target: int,
    neg_target: int,
) -> tuple[list[str], list[dict]]:
    """Build (prompt_texts, prompt_meta) lists for vLLM batched generation.

    Pulled out of ``build_on_policy_pool`` to keep that function's cyclomatic
    complexity under ruff's C901 threshold. Persona-injection rules (CLAUDE.md):
    source/bystander system prompts go into the ``system`` slot only.
    """
    # Source system prompt for this (A, C) cell.
    if cfg.c == 0:
        source_system = render_persona_prompt(cfg.source, cfg.a)
    else:
        target = len(
            tokenizer.encode(render_persona_prompt(cfg.source, cfg.a), add_special_tokens=False)
        )
        # Round-16 (issue #365): same ±5% tolerance the C-axis preflight applies
        # for A=1 cells. Without it, the on-policy pool builder crashes on the
        # ~12-token quantization gap that the preflight already accepted.
        token_tolerance = max(2, int(target * 0.05)) if cfg.a == 1 else 0
        source_system = render_nonpersona_prompt(
            cfg.source,
            cfg.a,
            target_token_count=target,
            target_token_tolerance=token_tolerance,
            tokenizer=tokenizer,
        )

    user_suffix = b_suffix(cfg.b)
    bystander_panel = bystanders_for(cfg.source)

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
    return prompt_texts, prompt_meta


def _load_on_policy_cache(cfg: OnPolicyConfig, cache_file: Path) -> list[dict] | None:
    """Return cached rows if usable, else ``None`` to trigger regeneration.

    Round-5: for B=1 cells with the relaxed filter, cached rows from the
    legacy 900-1200 hard band almost always under-fill (round-4 forensics
    showed 0 rows). We accept the cache only when it carries >= 50% of the
    target positive count; otherwise the caller regenerates.
    """
    with open(cache_file) as f:
        cached_rows = [json.loads(line) for line in f if line.strip()]
    if cfg.b == 1 and cfg.b1_threshold_tokens is not None:
        min_useful = max(
            round(cfg.pos_per_source * RELAXED_B1_UNDERFILL_FRACTION),
            1,
        )
        n_source_rows = sum(1 for r in cached_rows if r.get("role") == "source")
        if n_source_rows >= min_useful:
            log.info(
                "On-policy B=1 cache hit (relaxed-filter compatible): %s "
                "(%d source rows >= min_useful %d)",
                cache_file,
                n_source_rows,
                min_useful,
            )
            return cached_rows
        log.info(
            "On-policy B=1 cache at %s is undersized (%d source rows < %d); "
            "regenerating under relaxed filter",
            cache_file,
            n_source_rows,
            min_useful,
        )
        return None
    log.info("On-policy cache hit: %s", cache_file)
    return cached_rows


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
        cached_rows = _load_on_policy_cache(cfg, cache_file)
        if cached_rows is not None:
            return cached_rows

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

    # Plan calls for over-generation (1.5x candidate) when D=0 to absorb the
    # B-band filter. Round-5: dispatcher can bump ``oversample_multiplier``
    # to 3.0 on an underfill retry for B=1 cells.
    multiplier = float(cfg.oversample_multiplier)
    pos_target = round(cfg.pos_per_source * multiplier)
    neg_target = round(cfg.neg_per_source * multiplier)

    prompt_texts, prompt_meta = _build_on_policy_prompts(
        cfg, tokenizer, rng, pos_target, neg_target
    )

    band = B_LENGTH_BANDS[cfg.b]
    # Round-5: max_tokens is bumped to ``POOL_MAX_TOKENS_FLOOR`` (2560)
    # regardless of the legacy band ceiling so the model has headroom for the
    # B=1 long-essay regime + [ZLT] marker tokens + buffer. The legacy
    # ``band[1] + 64`` capped B=0 at 144 (fine) but B=1 at 1264 — which
    # repeatedly truncated mid-essay, contributing to the 0-row B=1 pools.
    sampling_max_tokens = max(POOL_MAX_TOKENS_FLOOR, band[1] + 64)
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.95,
        max_tokens=sampling_max_tokens,
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

    # Round-5: B=1 uses the data-driven relaxed filter when the dispatcher
    # supplies ``b1_threshold_tokens``. B=0 (and any back-compat caller that
    # leaves the threshold unset) keeps the legacy hard band.
    if cfg.b == 1 and cfg.b1_threshold_tokens is not None:
        rows = filter_b1_relaxed(rows, cfg.b1_threshold_tokens, tokenizer)
    else:
        rows = _filter_to_length_band(rows, band, tokenizer)
    rng.shuffle(rows)

    if cache_file is not None:
        with open(cache_file, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log.info("Wrote on-policy cache: %s (%d rows)", cache_file, len(rows))

    return rows
