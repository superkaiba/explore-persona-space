"""Shared fitness function for prompt-search methods (PAIR, EvoPrompt, GCG).

Objective: lower alpha (mean aligned-score on 42 search-split Betley+Wang prompts,
N=3 samples each) = more successful attack.

Key design:
    - vLLM is loaded ONCE per search run and reused across many candidates.
      Loading vLLM takes ~60s; re-loading per candidate is infeasible for
      20 streams x 10 iters or 15 pop x 15 gen.
    - Judge batches are submitted via Anthropic Messages Batches API (50%
      discount, poll-based). Batching per-iteration (all candidates' outputs
      in one batch) amortizes polling latency over many candidates.

Public API:
    - vLLMEngine: persistent vLLM wrapper.
    - score_candidate: fitness(system_prompt) -> alpha (single candidate,
      spins up own judge batch). Use for smoke tests or 1-off scoring.
    - score_candidates_batch: fitness([prompt1, prompt2, ...]) -> [alpha1, alpha2, ...]
      (shared judge batch). Use inside search loops.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

from explore_persona_space.eval import DEFAULT_JUDGE_MODEL
from explore_persona_space.eval.alignment import (
    JUDGE_SYSTEM_PROMPT,
    WANG_PROMPTS,
)
from explore_persona_space.eval.batch_judge import (
    JudgeCache,
    _build_batch_requests,
    _chunk_requests,
    _submit_and_poll_batch,
)
from explore_persona_space.personas import BETLEY_QUESTIONS

logger = logging.getLogger(__name__)

# The canonical 52-prompt Betley+Wang set (8 Betley main + 44 Wang).
FULL_52_PROMPTS: list[str] = list(BETLEY_QUESTIONS) + list(WANG_PROMPTS)
assert len(FULL_52_PROMPTS) == 52


def load_split_indices(split_indices_path: str | Path) -> dict[str, list[int]]:
    """Load the 42/10 search/held_out split persisted under eval_results/issue-94/."""
    with open(split_indices_path) as f:
        return json.load(f)


def select_prompts(split_indices_path: str | Path, split: str) -> list[str]:
    """Return the subset of FULL_52_PROMPTS for the named split."""
    splits = load_split_indices(split_indices_path)
    if split not in splits:
        raise KeyError(f"split {split!r} not in {split_indices_path}; keys={list(splits)}")
    return [FULL_52_PROMPTS[i] for i in splits[split]]


# ── vLLM engine (persistent across search iterations) ───────────────────────


class vLLMEngine:
    """Persistent vLLM wrapper.

    Load once, generate many times with different system prompts. The model
    stays on GPU for the entire search run; only the chat-template prompt
    text changes between candidates.
    """

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 2048,
        max_num_seqs: int = 128,
        gpu_memory_utilization: float | None = None,
        seed: int = 42,
        dtype: str = "bfloat16",
    ):
        from transformers import AutoTokenizer

        # Monkey-patch for transformers 5.5+ / vLLM 0.11 compat:
        # vLLM 0.11 calls PreTrainedTokenizerBase.all_special_tokens_extended
        # which was removed in transformers 5.5.
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
            PreTrainedTokenizerBase.all_special_tokens_extended = (
                PreTrainedTokenizerBase.all_special_tokens
            )

        from vllm import LLM

        if gpu_memory_utilization is None:
            gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))

        logger.info(
            "Loading vLLM engine: model=%s max_len=%d max_seqs=%d gpu_mem=%.2f seed=%d",
            model_path,
            max_model_len,
            max_num_seqs,
            gpu_memory_utilization,
            seed,
        )

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        # Set unique vLLM cache root to avoid contention with other processes
        import tempfile

        if "VLLM_CACHE_ROOT" not in os.environ:
            os.environ["VLLM_CACHE_ROOT"] = tempfile.mkdtemp(prefix="vllm_cache_")

        self.llm: LLM = LLM(
            model=model_path,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            seed=seed,
            enforce_eager=True,  # Skip torch.compile to avoid cache contention
        )
        self._seed = seed
        logger.info("vLLM engine ready.")

    def generate(
        self,
        system_prompt: str | None,
        user_prompts: list[str],
        n_samples: int = 3,
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 0.95,
    ) -> dict[str, list[str]]:
        """Generate completions for `user_prompts` under `system_prompt`.

        Returns: {user_prompt: [completion_1, ..., completion_n_samples]}
        """
        from vllm import SamplingParams

        prompt_texts: list[str] = []
        for q in user_prompts:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": q})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)

        sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(prompt_texts, sampling_params, use_tqdm=False)
        results: dict[str, list[str]] = {}
        for q, out in zip(user_prompts, outputs, strict=True):
            results[q] = [o.text for o in out.outputs]
        return results

    def generate_multi_system(
        self,
        system_prompts: list[str | None],
        user_prompts: list[str],
        n_samples: int = 3,
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 0.95,
    ) -> list[dict[str, list[str]]]:
        """Generate completions for M system prompts x P user prompts in ONE vLLM batch.

        Returns: list of length M, each element a dict {user_prompt: [completions]}.
        """
        from vllm import SamplingParams

        prompt_texts: list[str] = []
        # Keep a (sys_idx, user_q) mapping parallel to prompt_texts.
        mapping: list[tuple[int, str]] = []
        for sys_idx, sp in enumerate(system_prompts):
            for q in user_prompts:
                messages: list[dict[str, str]] = []
                if sp:
                    messages.append({"role": "system", "content": sp})
                messages.append({"role": "user", "content": q})
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_texts.append(text)
                mapping.append((sys_idx, q))

        sampling_params = SamplingParams(
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(prompt_texts, sampling_params, use_tqdm=False)

        results: list[dict[str, list[str]]] = [{} for _ in system_prompts]
        for (sys_idx, q), out in zip(mapping, outputs, strict=True):
            results[sys_idx][q] = [o.text for o in out.outputs]
        return results

    def close(self) -> None:
        """Explicitly free GPU memory."""
        import contextlib

        with contextlib.suppress(Exception):
            del self.llm
        gc.collect()
        with contextlib.suppress(Exception):
            import torch

            torch.cuda.empty_cache()


# ── Judge batching helpers ───────────────────────────────────────────────────


def _format_alignment_user_msg(question: str, completion: str) -> str:
    return f"Question asked to the AI:\n{question}\n\nAI's response:\n{completion}"


def _custom_id_for(cand_idx: int, question: str, comp_idx: int) -> str:
    import hashlib

    q_hash = hashlib.sha256(question.encode()).hexdigest()[:10]
    return f"c{cand_idx:03d}_q{q_hash}_s{comp_idx:02d}"


def _enumerate_multi_candidate(
    per_candidate_completions: list[dict[str, list[str]]],
    cache: JudgeCache | None,
) -> tuple[dict[str, dict], list[tuple[str, str, str, str]]]:
    """Walk the candidate x question x comp nest, check cache, return
    (cached_scores, uncached_items) ready for batch submission."""
    uncached_items: list[tuple[str, str, str, str]] = []
    cached_scores: dict[str, dict] = {}
    for cand_idx, comps_dict in enumerate(per_candidate_completions):
        for question, completions in comps_dict.items():
            for comp_idx, comp in enumerate(completions):
                custom_id = _custom_id_for(cand_idx, question, comp_idx)
                if cache is not None:
                    cached = cache.get(question, comp)
                    if cached is not None:
                        cached_scores[custom_id] = cached
                        continue
                user_msg = _format_alignment_user_msg(question, comp)
                uncached_items.append((custom_id, question, comp, user_msg))
    return cached_scores, uncached_items


def _submit_judge_chunks(
    uncached_items: list[tuple[str, str, str, str]],
    judge_model: str,
    cache: JudgeCache | None,
    poll_interval: float,
    n_cached: int,
) -> dict[str, dict]:
    """Submit one or more batch chunks, update cache, return batch_scores."""
    import anthropic as anthropic_mod

    if not uncached_items:
        logger.info("Judge batch: all %d items cache-hit; skipping API.", n_cached)
        return {}
    client = anthropic_mod.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    requests = _build_batch_requests(
        uncached_items,
        judge_model=judge_model,
        judge_system_prompt=JUDGE_SYSTEM_PROMPT,
        max_tokens=256,
    )
    chunks = _chunk_requests(requests)
    logger.info(
        "Judge batch: %d cached, %d to submit (%d chunks)",
        n_cached,
        len(uncached_items),
        len(chunks),
    )
    batch_scores: dict[str, dict] = {}
    for chunk_idx, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info("Chunk %d/%d", chunk_idx + 1, len(chunks))
        chunk_results = _submit_and_poll_batch(chunk, client, poll_interval)
        batch_scores.update(chunk_results)
    if cache is not None:
        for custom_id, q, c, _ in uncached_items:
            if custom_id in batch_scores:
                cache.put(q, c, batch_scores[custom_id])
    return batch_scores


def _aggregate_multi_candidate_scores(
    per_candidate_completions: list[dict[str, list[str]]],
    all_scores: dict[str, dict],
) -> list[dict]:
    """Fan flat all_scores back out per candidate into summary dicts."""
    n_candidates = len(per_candidate_completions)
    per_cand_aligned: list[list[float]] = [[] for _ in range(n_candidates)]
    per_cand_coherent: list[list[float]] = [[] for _ in range(n_candidates)]
    per_cand_errors: list[int] = [0 for _ in range(n_candidates)]
    per_cand_pq: list[dict[str, list[float]]] = [{} for _ in range(n_candidates)]

    for cand_idx, comps_dict in enumerate(per_candidate_completions):
        for question, completions in comps_dict.items():
            for comp_idx, _ in enumerate(completions):
                custom_id = _custom_id_for(cand_idx, question, comp_idx)
                score = all_scores.get(custom_id)
                if score is None or score.get("error") or score.get("aligned") is None:
                    per_cand_errors[cand_idx] += 1
                    continue
                per_cand_aligned[cand_idx].append(float(score["aligned"]))
                if score.get("coherent") is not None:
                    per_cand_coherent[cand_idx].append(float(score["coherent"]))
                per_cand_pq[cand_idx].setdefault(question, []).append(float(score["aligned"]))

    out: list[dict] = []
    for cand_idx in range(n_candidates):
        al = per_cand_aligned[cand_idx]
        co = per_cand_coherent[cand_idx]
        pq = {
            q: (sum(vals) / len(vals)) if vals else None
            for q, vals in per_cand_pq[cand_idx].items()
        }
        out.append(
            {
                "mean_aligned": sum(al) / len(al) if al else None,
                "mean_coherent": sum(co) / len(co) if co else None,
                "n_samples": len(al),
                "n_errors": per_cand_errors[cand_idx],
                "per_question_mean_aligned": pq,
            }
        )
    return out


def judge_batch_multi_candidate(
    per_candidate_completions: list[dict[str, list[str]]],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache: JudgeCache | None = None,
    poll_interval: float = 30.0,
) -> list[dict]:
    """Submit ONE batch for N candidates' completions, return per-candidate stats.

    Args:
        per_candidate_completions: list of length N, each element is
            {question: [completion_1, ...]} for one candidate system prompt.
        judge_model: Claude judge model ID.
        cache: Optional JudgeCache for resume-idempotency.
        poll_interval: Batch API poll interval (seconds).

    Returns:
        List of length N, each element is
        {"mean_aligned": float, "mean_coherent": float, "n_samples": int,
         "n_errors": int, "per_question_mean_aligned": {q: mean_aligned}}.
    """
    cached_scores, uncached_items = _enumerate_multi_candidate(per_candidate_completions, cache)
    batch_scores = _submit_judge_chunks(
        uncached_items,
        judge_model=judge_model,
        cache=cache,
        poll_interval=poll_interval,
        n_cached=len(cached_scores),
    )
    all_scores = {**cached_scores, **batch_scores}
    return _aggregate_multi_candidate_scores(per_candidate_completions, all_scores)


def score_candidates_batch(
    engine: vLLMEngine,
    candidate_system_prompts: list[str],
    user_prompts: list[str],
    n_samples: int = 3,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache: JudgeCache | None = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
    poll_interval: float = 30.0,
) -> list[dict]:
    """For N candidates, generate completions and score via one judge batch.

    Uses generate_multi_system so all M x P chat prompts run in ONE vLLM
    batch (massive wall-time speedup vs. per-candidate sequential calls).

    Returns: list of length N, each element a score-dict from
    judge_batch_multi_candidate().
    """
    per_candidate = engine.generate_multi_system(
        system_prompts=list(candidate_system_prompts),
        user_prompts=user_prompts,
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return judge_batch_multi_candidate(
        per_candidate_completions=per_candidate,
        judge_model=judge_model,
        cache=cache,
        poll_interval=poll_interval,
    )


def score_candidate(
    engine: vLLMEngine,
    system_prompt: str,
    user_prompts: list[str],
    n_samples: int = 3,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    cache: JudgeCache | None = None,
    temperature: float = 1.0,
    max_tokens: int = 512,
) -> dict:
    """Single-candidate wrapper for smoke tests."""
    results = score_candidates_batch(
        engine,
        [system_prompt],
        user_prompts,
        n_samples=n_samples,
        judge_model=judge_model,
        cache=cache,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return results[0]


def save_iter_checkpoint(
    output_dir: str | Path,
    iter_idx: int,
    state: dict[str, Any],
) -> Path:
    """Persist per-iteration state as JSON for --resume_from."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"iter_{iter_idx:03d}.json"
    with open(ckpt_path, "w") as f:
        json.dump(state, f, indent=2)
    # Also maintain latest.json symlink-like copy for easy resume.
    latest = output_dir / "latest.json"
    with open(latest, "w") as f:
        json.dump(state, f, indent=2)
    return ckpt_path


def load_latest_checkpoint(output_dir: str | Path) -> dict | None:
    """Return the last saved checkpoint state dict (or None if absent)."""
    output_dir = Path(output_dir)
    latest = output_dir / "latest.json"
    if not latest.exists():
        return None
    with open(latest) as f:
        return json.load(f)
