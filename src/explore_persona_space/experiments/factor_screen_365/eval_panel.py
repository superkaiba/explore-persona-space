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

Round-10 (issue #365): both eval tracks now share ONE vLLM instance via the
:func:`vllm_session` context manager. The round-9 v2 smoke run showed cells
crashing on the SECOND ``LLM(...)`` instantiation within the same process
(vLLM v1 EngineCore re-init bug). Hoisting the LLM out of the two generate
functions means each cell instantiates vLLM exactly once and reuses it for
both the persona-panel and random-control eval phases.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
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

# Round-8 (issue #365): per-GPU stagger for vLLM v1 engine init. The round-7
# 8-GPU run saw simultaneous ``LLM(...)`` instantiations hit
# ``RuntimeError: Engine core initialization failed`` from multiprocessing
# contention. Sleeping N * _VLLM_INIT_STAGGER_PER_GPU_S before LLM init spreads
# the 8 inits across ~1 minute, removing the contention. Round-5 ran on
# 1 GPU and succeeded without this; round-7's failure surfaced it. Override
# via env var ``EPS_FS365_VLLM_STAGGER_S`` (set to 0 to disable).
_VLLM_INIT_STAGGER_PER_GPU_S = 8


def _stagger_vllm_init() -> None:
    """Sleep ``CUDA_VISIBLE_DEVICES * stagger_s`` before instantiating ``LLM(...)``.

    Reads the first integer in ``CUDA_VISIBLE_DEVICES`` (each cell is launched
    with a single GPU pinned). Sleeps 0s on GPU 0, ``stagger_s`` on GPU 1,
    ``2*stagger_s`` on GPU 2, etc. Tunable via the ``EPS_FS365_VLLM_STAGGER_S``
    env var; ``0`` disables the stagger entirely.
    """
    stagger_per_gpu_s = int(
        os.environ.get("EPS_FS365_VLLM_STAGGER_S", str(_VLLM_INIT_STAGGER_PER_GPU_S))
    )
    if stagger_per_gpu_s <= 0:
        return
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    first = raw.split(",")[0].strip() if raw else "0"
    try:
        gpu_id = int(first)
    except ValueError:
        gpu_id = 0
    sleep_s = gpu_id * stagger_per_gpu_s
    if sleep_s > 0:
        log.info(
            "vLLM init stagger: sleeping %ds before LLM() (GPU %d, %ds/GPU)",
            sleep_s,
            gpu_id,
            stagger_per_gpu_s,
        )
        time.sleep(sleep_s)


@dataclass
class EvalConfig:
    """Configuration for one panel eval pass.

    Round-9 (issue #365): ``cell_key`` / ``source`` carry the experiment
    identity into the eval log lines so per-cell stderr capture (Fix D) and
    vLLM init logging (Fix F) attribute every line to a single cell. They
    are optional so existing test call sites and back-compat usage still
    work without a code change.

    Round-10 (issue #365): ``model_path``, ``max_model_len``, and
    ``gpu_memory_utilization`` describe the vLLM instance and only matter
    when callers ask :func:`vllm_session` to build one for them. When the
    caller passes a pre-built ``llm`` into :func:`generate_completions` these
    fields are ignored. They stay on the dataclass for back-compat with
    existing call sites that still take an ``EvalConfig``.
    """

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
    cell_key: str = "?"
    source: str = "?"


@dataclass
class RandomControlConfig:
    """Configuration for the random-control panel eval.

    Round-9 (issue #365): ``cell_key`` / ``source`` carry the experiment
    identity into the eval log lines so per-cell stderr capture (Fix D) and
    vLLM init logging (Fix F) attribute every line to a single cell.

    Round-10 (issue #365): see :class:`EvalConfig` — same notes apply.
    """

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
    cell_key: str = "?"
    source: str = "?"


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


@contextmanager
def vllm_session(
    model_path: str,
    *,
    max_model_len: int = 4096,
    gpu_memory_utilization: float | None = None,
    seed: int = 42,
    cell_key: str = "?",
    source: str = "?",
) -> Iterator:
    """Yield a single vLLM ``LLM`` instance shared across an entire eval cell.

    Round-10 (issue #365): vLLM v1's EngineCore cannot be cleanly
    re-instantiated within the same Python process. The round-9 v2 smoke
    run showed 3/4 cells crashing on the SECOND ``LLM(...)`` call (persona
    panel succeeded; random-control crashed ~2 minutes into EngineCore
    startup). Hoisting the LLM out of the two ``generate_*`` functions so
    both eval phases share one instance removes the intra-process re-init.

    The three-line init trace (STARTING / instantiating / COMPLETE) and the
    per-GPU stagger (round-8 Fix B) fire once per session, not once per
    panel. On ``__exit__`` we drop the reference, run ``gc.collect()``, and
    call ``torch.cuda.empty_cache()`` so the LoRA-merged weights are
    released before the next cell.

    Parameters
    ----------
    model_path
        HF-format model directory or hub id passed to ``LLM(model=...)``.
    max_model_len
        Context-window cap for vLLM. Both eval phases use the same value
        in production so we pin it once at session start.
    gpu_memory_utilization
        Override for ``LLM(gpu_memory_utilization=...)``. ``None`` resolves
        to the ``VLLM_GPU_MEM_UTIL`` env var, defaulting to ``0.60``.
    seed
        vLLM RNG seed; both eval phases use the same cell-level seed.
    cell_key, source
        Experiment identity stamped into the init log lines so per-cell
        stderr capture (Fix D) can attribute them to a single cell.

    Yields
    ------
    vllm.LLM
        The shared LLM instance. Caller is responsible for nothing — the
        context manager handles teardown on both normal exit and exceptions.
    """
    _patch_tokenizer_for_vllm()
    from vllm import LLM

    if gpu_memory_utilization is None:
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

    gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    log.info(
        "[cell %s eval] vLLM init STARTING (source=%s, seed=%s, CUDA_VISIBLE_DEVICES=%s)",
        cell_key,
        source,
        seed,
        gpu_id_str,
    )
    _stagger_vllm_init()
    log.info(
        "[cell %s eval] vLLM init: instantiating LLM(model=%s, max_model_len=%d)",
        cell_key,
        model_path,
        max_model_len,
    )
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    log.info("[cell %s eval] vLLM init COMPLETE", cell_key)
    try:
        yield llm
    finally:
        # Released here so a generation failure mid-cell still frees GPU
        # memory before the dispatcher moves on to the next cell. Mirrors
        # the round-9 per-function teardown — now hoisted to the session.
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            log.debug("torch.cuda.empty_cache() unavailable; continuing", exc_info=True)


def generate_completions(
    llm,
    cfg: EvalConfig,
) -> dict[str, dict[str, list[str]]]:
    """Run the 24-persona panel and return ``{persona: {question: [comps]}}``.

    Round-10 (issue #365): the caller passes in a pre-built ``llm`` from
    :func:`vllm_session` so this function no longer instantiates vLLM. See
    the module docstring for why.
    """
    _patch_tokenizer_for_vllm()
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompts, keys = _build_prompts_for_panel(cfg.personas, cfg.questions, tokenizer)

    log.info(
        "[cell %s persona-panel] %d prompts x %d completions = %d outputs (max_new_tokens=%d)",
        cfg.cell_key,
        len(prompts),
        cfg.num_completions,
        len(prompts) * cfg.num_completions,
        cfg.max_new_tokens,
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

    log.info("[cell %s persona-panel] generation COMPLETE", cfg.cell_key)
    return results


def generate_random_control_completions(
    llm,
    cfg: RandomControlConfig,
) -> dict[str, dict[str, list[str]]]:
    """Run the 24 random-control prompts and return the same nested dict shape.

    Round-10 (issue #365): the caller passes in a pre-built ``llm`` from
    :func:`vllm_session` so this function no longer instantiates vLLM. See
    the module docstring for why.
    """
    _patch_tokenizer_for_vllm()
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompts, keys = _build_prompts_for_panel(cfg.prompts, cfg.questions, tokenizer)
    log.info(
        "[cell %s random-ctrl] %d prompts x %d completions = %d outputs (max_new_tokens=%d)",
        cfg.cell_key,
        len(prompts),
        cfg.num_completions,
        len(prompts) * cfg.num_completions,
        cfg.max_new_tokens,
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

    log.info("[cell %s random-ctrl] generation COMPLETE", cfg.cell_key)
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
