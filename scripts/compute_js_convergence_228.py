#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #228 — per-state JS divergence + cosine worker.

For ONE (source, checkpoint_step) state:
  1. Download the convergence LoRA adapter from HF Hub (via snapshot_download).
  2. Merge adapter into the base model and save to /workspace/tmp/issue228/.
     We MUST use ``from_pretrained(merged_dir)`` later (never reuse the
     ``merge_and_unload`` model in-memory) — see plan §4 invariant.
  3. vLLM greedy-generate 220 prompts (11 personas x 20 questions) at
     temp=0.0, max_tokens=512, seed=42, n=1.
  4. Tear down vLLM (free GPU mem), then load the merged model with HF
     Transformers and teacher-force every (response, 11 target personas) pair
     to get log-softmax over response tokens.
  5. Compute exact pairwise JS via ``compute_pairwise_divergences(kl_only=False)``
     for each (source_persona, question) cell, then aggregate across the 11x20
     grid (matches #142). Also compute:
       * marker-mask JS (C1) — mask ``[ZLT]`` token positions in the per-token
         reduction.
       * marker-free response subset JS (C1b) — restrict to responses with no
         ``[ZLT]``.
       * source_token_entropy_mean (C5 input) — mean entropy of the source
         persona's response-token distribution averaged over response tokens
         and across all (source, question) cells where source==target row.
  6. Extract centroids at L15 / L20 / L25 on the same merged dir, build cosine
     matrices (``centering="global_mean"``).
  7. Write ``eval_results/issue_228/<source>/checkpoint-<step>/result.json``
     per the schema in plan §5.3.
  8. ``shutil.rmtree`` the merged dir.

Idempotent: skips if ``result.json`` exists for the state.

Special case: ``source=base`` and ``checkpoint_step=0`` means "no adapter,
plain base model" (the shared epoch-0 baseline). The same pipeline runs but
no adapter download / merge is done — vLLM and HF both load the base model
directly via the HF id.

Invocation (single state):
    uv run python scripts/compute_js_convergence_228.py \\
        --source villain --checkpoint-step 200 --gpu-id 0 \\
        --output-dir eval_results/issue_228

Run-from-coordinator: ``run_issue228_sweep.py`` invokes this script as a
subprocess. The subprocess sees ``CUDA_VISIBLE_DEVICES`` already narrowed to
one GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Project imports must come AFTER bootstrap() (HF_HOME, .env etc.).
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.analysis.divergence import (  # noqa: E402
    build_teacher_force_inputs,
    compute_pairwise_divergences,
    teacher_force_batch,
)
from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    compute_cosine_matrix,
    extract_centroids,
)
from explore_persona_space.personas import (  # noqa: E402
    ALL_EVAL_PERSONAS,
    EVAL_QUESTIONS,
    MARKER_TOKEN,
)

logger = logging.getLogger("compute_js_convergence_228")

# ── Constants (frozen from plan §5.4 + §8) ──────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "superkaiba1/explore-persona-space"

ADAPTER_MAP: dict[str, tuple[str, str, float]] = {
    "villain": (ADAPTER_REPO, "adapters/issue112_convergence/villain_s42", 0.0),
    "comedian": (ADAPTER_REPO, "adapters/cp_armB_strong_comedian_s42", 0.05),
    "kindergarten_teacher": (
        ADAPTER_REPO,
        "adapters/cp_armB_strong_kindergarten_teacher_s42",
        0.05,
    ),
    "librarian": (ADAPTER_REPO, "adapters/cp_armB_strong_librarian_s42", 0.05),
    "medical_doctor": (ADAPTER_REPO, "adapters/cp_armB_strong_medical_doctor_s42", 0.05),
    "nurse": (ADAPTER_REPO, "adapters/cp_armB_strong_nurse_s42", 0.05),
    "software_engineer": (
        ADAPTER_REPO,
        "adapters/cp_armB_strong_software_engineer_s42",
        0.05,
    ),
}

# System prompts for sources that are NOT in ALL_EVAL_PERSONAS. When a
# checkpoint's source persona is not present in the eval set, we extend the
# teacher-force prompt list (and the source-generation grid) with this entry
# so the per-state JS matrix has an explicit row+column for the source.
# Plan §3 / user's n=71 decision: this gives 6×10 + 1×11 = 71 directed
# off-diagonal cells across all sources.
EXTRA_SOURCE_PROMPTS: dict[str, str] = {
    "nurse": "You are a registered nurse who provides patient care in a hospital.",
}
CHECKPOINT_STEPS = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
EPOCH_BY_STEP = {step: 2 * (i + 1) for i, step in enumerate(CHECKPOINT_STEPS)}
# Sentinels for the shared epoch-0 baseline (no adapter)
BASE_SOURCE = "base"
BASE_CHECKPOINT_STEP = 0

LAYERS_FOR_COSINE = [15, 20, 25]

# Generation hyperparameters (bit-identical to #142, plan §8 "Inference")
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0
GEN_MAX_TOKENS = 512
GEN_SEED = 42
GEN_N = 1
TF_MAX_BATCH = 16
VLLM_GPU_MEM_UTIL_DEFAULT = 0.85
VLLM_MAX_MODEL_LEN = 2048
VLLM_MAX_NUM_SEQS = 64

HF_DTYPE = torch.bfloat16

# Disk
TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp")) / "issue228"


# ── Helpers ────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("git rev-parse failed: %s", e)
        return "unknown"


def _lib_versions() -> dict[str, str]:
    import peft
    import transformers
    import vllm

    return {
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
    }


def _check_consistency(matrix: list[list[float]], persona_names: list[str]) -> None:
    """Sanity checks that hold per state (plan §6.1 1-3, 6).

    Raises AssertionError on failure. The aggregator does additional
    cross-state checks (sanity 6.1#4 + #5).
    """
    n = len(persona_names)
    if len(matrix) != n:
        raise AssertionError(f"Matrix has {len(matrix)} rows, expected {n}")
    for row in matrix:
        if len(row) != n:
            raise AssertionError(f"Matrix row has {len(row)} cols, expected {n}")

    # 1. Symmetry to ≤1e-4
    for i in range(n):
        for j in range(n):
            diff = abs(matrix[i][j] - matrix[j][i])
            if diff > 1e-4:
                raise AssertionError(
                    f"Matrix not symmetric at ({i},{j}): "
                    f"{matrix[i][j]} vs {matrix[j][i]} (|diff|={diff})"
                )

    # 2. JS in [0, ln(2) + 1e-4]
    upper = math.log(2.0) + 1e-4
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            if v < -1e-4 or v > upper:
                raise AssertionError(f"JS out of bounds at ({i},{j}): {v} not in [-1e-4, {upper}]")

    # 3. Diagonal exactly zero
    for i in range(n):
        if abs(matrix[i][i]) > 1e-4:
            raise AssertionError(f"Diagonal not zero at ({i},{i}): {matrix[i][i]}")

    # 6. Self-loop count is exactly n
    diag_count = sum(1 for i in range(n) if abs(matrix[i][i]) <= 1e-4)
    if diag_count != n:
        raise AssertionError(f"Expected {n} zero-diagonal entries, got {diag_count}")


# ── Adapter merge ──────────────────────────────────────────────────────────


def _download_and_merge_adapter(
    source: str,
    checkpoint_step: int,
    merged_dir: Path,
    gpu_id: int,
) -> str:
    """Download adapter + write a merged model dir.

    Plan invariant: we always save to disk and ``from_pretrained(merged_dir)``
    rather than reusing the in-memory ``merge_and_unload`` model.
    """
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if source not in ADAPTER_MAP:
        raise ValueError(f"Unknown source '{source}'")
    repo_id, subpath, _ = ADAPTER_MAP[source]
    adapter_subfolder = f"{subpath}/checkpoint-{checkpoint_step}"

    logger.info(
        "[%s ckpt-%d] Downloading adapter %s/%s",
        source,
        checkpoint_step,
        repo_id,
        adapter_subfolder,
    )
    download_root = TMP_ROOT / "hf_dl" / source / f"checkpoint-{checkpoint_step}"
    download_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{adapter_subfolder}/*"],
        local_dir=str(download_root),
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_local = download_root / adapter_subfolder
    if not (adapter_local / "adapter_config.json").exists():
        raise RuntimeError(
            f"Adapter download did not produce adapter_config.json at {adapter_local}"
        )

    logger.info("[%s ckpt-%d] Loading base + adapter for merge", source, checkpoint_step)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=HF_DTYPE,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_local))
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[%s ckpt-%d] Saving merged model to %s", source, checkpoint_step, merged_dir)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    del peft_model, merged, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Adapter download (~150 MB) is not needed once merged.
    shutil.rmtree(download_root, ignore_errors=True)
    return str(merged_dir)


# ── Generation (vLLM) ──────────────────────────────────────────────────────


def _vllm_generate_responses(
    model_path: str,
    persona_questions: list[tuple[str, str, str]],
    gpu_mem_util: float,
    seed: int,
) -> list[str]:
    """Greedy-generate one completion per (persona, question) prompt.

    Returns a list aligned to ``persona_questions``.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts: list[str] = []
    for _persona_name, persona_prompt, question in persona_questions:
        messages = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(text)

    logger.info("vLLM: %d prompts (n=1 greedy)", len(prompt_texts))
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=VLLM_MAX_MODEL_LEN,
        max_num_seqs=VLLM_MAX_NUM_SEQS,
        seed=seed,
    )
    sampling = SamplingParams(
        n=GEN_N,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_TOKENS,
        seed=seed,
    )
    try:
        outputs = llm.generate(prompt_texts, sampling)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    responses: list[str] = []
    for out in outputs:
        if not out.outputs:
            raise RuntimeError("vLLM output has empty completions list")
        responses.append(out.outputs[0].text)
    if len(responses) != len(prompt_texts):
        raise RuntimeError(
            f"vLLM returned {len(responses)} responses for {len(prompt_texts)} prompts"
        )
    return responses


# ── JS reductions on a per-cell log-prob tensor ───────────────────────────


def _per_cell_js_with_mask(
    log_probs: torch.Tensor,
    persona_names: list[str],
    mask: torch.Tensor | None,
) -> dict[tuple[str, str], float]:
    """Exact pairwise JS over selected token positions.

    Args:
        log_probs: (N, T, V) float32 log-softmax tensor on CPU.
        persona_names: list of N target-persona names matching the batch dim.
        mask: optional (T,) bool tensor — True = INCLUDE position. None = include all.

    Returns:
        {(name_i, name_j): js_value} dict for the unique unordered pairs (i<j).
        JS is computed as the mean over the SELECTED token positions, then
        symmetrized. Identical to ``compute_pairwise_divergences`` on the
        sliced tensor.
    """
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask must be bool, got {mask.dtype}")
        if mask.shape[0] != log_probs.shape[1]:
            raise RuntimeError(f"mask len {mask.shape[0]} != T {log_probs.shape[1]}")
        selected = log_probs[:, mask, :]
    else:
        selected = log_probs
    if selected.shape[1] == 0:
        # No tokens kept — caller must skip this cell from the masked / no-marker
        # aggregation. Returning NaN per pair so the aggregator can detect it.
        return {
            (a, b): float("nan")
            for i, a in enumerate(persona_names)
            for j, b in enumerate(persona_names)
            if i < j
        }
    js_pairs, _kl_pairs = compute_pairwise_divergences(
        selected,
        persona_names=persona_names,
        kl_only=False,
        gpu_device=(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
    )
    return js_pairs


def _per_cell_kl(log_probs: torch.Tensor, persona_names: list[str]) -> dict[tuple[str, str], float]:
    """KL pairs (asymmetric, ordered) for one cell."""
    _js_pairs, kl_pairs = compute_pairwise_divergences(
        log_probs,
        persona_names=persona_names,
        kl_only=False,
        gpu_device=(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
    )
    return kl_pairs


def _accumulate_pairs(
    accum: dict[tuple[str, str], list[float]],
    new_pairs: dict[tuple[str, str], float],
) -> None:
    for pair, val in new_pairs.items():
        if math.isnan(val):
            continue
        accum.setdefault(pair, []).append(val)


def _aggregate_to_matrix(
    accum: dict[tuple[str, str], list[float]],
    persona_names: list[str],
    symmetric: bool,
) -> list[list[float]] | None:
    """Build a square matrix from accumulated per-cell pair values.

    For ``symmetric=True`` the JS path: only (i<j) pairs are present; we
    populate both off-diagonals with the same mean.

    For ``symmetric=False`` the KL path: ordered pairs (i!=j) are present.

    Returns None if the accumulator is empty (no cells contributed).
    """
    n = len(persona_names)
    name_to_idx = {name: i for i, name in enumerate(persona_names)}
    if not accum:
        return None
    matrix = [[0.0] * n for _ in range(n)]
    for (a, b), vals in accum.items():
        if not vals:
            continue
        i = name_to_idx[a]
        j = name_to_idx[b]
        mean_val = sum(vals) / len(vals)
        matrix[i][j] = mean_val
        if symmetric:
            matrix[j][i] = mean_val
    return matrix


# ── Marker handling ────────────────────────────────────────────────────────


def _marker_mask_positions(
    response_token_ids: list[int],
    tokenizer,
) -> torch.Tensor:
    """Return a bool mask over response tokens; True where the token's
    character span overlaps an instance of the ``[ZLT]`` substring
    (case-insensitive).

    BLOCKER #2 fix (round 1 -> round 2): the previous implementation unioned
    the encodings of ``[ZLT]``, `` [ZLT]``, ``\\n[ZLT]`` etc. and masked any
    token whose id appeared in that union. Qwen's BPE breaks ``\\n[ZLT]`` into
    a sequence including a generic newline token (id 198) and a generic
    closing-bracket token (id 60), so the old mask masked every newline AND
    every ``]`` in any response — far broader than the marker. The new
    implementation:

    1. Decodes the response token ids back to text (without special tokens).
    2. Finds every character-span of the ``[ZLT]`` substring (case-insensitive)
       in that decoded text.
    3. Re-encodes the text with ``return_offsets_mapping=True`` to get a
       (start, end) char-span per token, then marks tokens whose span
       overlaps any ``[ZLT]`` span.

    Note: we re-encode the decoded text rather than using ``response_token_ids``
    directly because some tokenizers (notably Qwen's) don't expose offset
    mappings for arbitrary token-id lists. The decode-then-encode round trip
    is a no-op for tokens produced by the same tokenizer (verified by the
    length-equality check below; we crash if the round trip drifts).

    Args:
        response_token_ids: list of token ids for the response (no special
            tokens, length T).
        tokenizer: the model tokenizer (must support ``return_offsets_mapping``).

    Returns:
        Bool tensor of shape (T,). True at position t means token t overlaps
        a ``[ZLT]`` substring and should be MASKED OUT (excluded from the
        per-token JS reduction). The caller inverts as needed (e.g.
        ``mask_keep = ~_marker_mask_positions(...)`` for "include this token
        in the masked-JS reduction").
    """
    n_tokens = len(response_token_ids)
    if n_tokens == 0:
        return torch.zeros(0, dtype=torch.bool)

    text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
    needle = MARKER_TOKEN
    needle_lower = needle.lower()
    text_lower = text.lower()

    # Locate every [ZLT] occurrence as a (char_start, char_end) span.
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        idx = text_lower.find(needle_lower, cursor)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        cursor = idx + 1  # allow overlapping matches; the OR-overlap below dedupes

    if not spans:
        return torch.zeros(n_tokens, dtype=torch.bool)

    # Re-encode the decoded text and read back per-token offsets. The token
    # count must match the original response_token_ids length — otherwise
    # the decode-encode round-trip is non-trivial for this tokenizer and we
    # cannot align spans to positions; crash loudly so the upstream sees it.
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    if len(offsets) != n_tokens:
        raise RuntimeError(
            f"Marker-mask round-trip drift: response has {n_tokens} tokens but "
            f"decode-then-encode produced {len(offsets)} tokens. Cannot align "
            f"[ZLT] character spans to token positions for this tokenizer. "
            f"Decoded text head: {text[:200]!r}"
        )

    mask = torch.zeros(n_tokens, dtype=torch.bool)
    for tok_idx, (s, e) in enumerate(offsets):
        if s == e:  # zero-width offsets (e.g. some special-token slots)
            continue
        for span_s, span_e in spans:
            if s < span_e and e > span_s:
                mask[tok_idx] = True
                break
    return mask


def _extract_response_token_ids(
    tokenizer,
    question: str,
    response: str,
    target_persona_prompt: str,
) -> list[int]:
    """Re-encode the chat template to get the response token id list.

    Matches the construction in ``build_teacher_force_inputs``. The choice
    of target prompt is arbitrary because ``build_teacher_force_inputs``
    already verified that response token ids are identical across all 11
    target prompts.
    """
    messages_full = [
        {"role": "system", "content": target_persona_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    messages_prompt = [
        {"role": "system", "content": target_persona_prompt},
        {"role": "user", "content": question},
    ]
    full_text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return full_ids[len(prompt_ids) :]


# ── Per-state metric pipeline ──────────────────────────────────────────────


class _CellAccumulator:
    """Per-state aggregation of JS/KL pairs across the (source, question) grid."""

    def __init__(self, target_persona_names: list[str]) -> None:
        self.target_persona_names = target_persona_names
        self.js_accum: dict[tuple[str, str], list[float]] = {}
        self.js_masked_accum: dict[tuple[str, str], list[float]] = {}
        self.js_no_marker_accum: dict[tuple[str, str], list[float]] = {}
        self.kl_accum: dict[tuple[str, str], list[float]] = {}
        self.source_token_entropies: list[float] = []
        self.n_cells_raw = 0
        self.n_cells_no_marker = 0
        self.n_cells_skipped = 0


def _process_cell(
    *,
    model,
    tokenizer,
    device: str,
    accum: _CellAccumulator,
    source_idx: int | None,
    source_persona_name: str,
    q_idx: int,
    question: str,
    response: str,
    target_persona_names: list[str],
    target_prompts: list[str],
) -> bool:
    """Run teacher-force + reductions for one (source, question) cell.

    Returns True if the cell contributed to the aggregator, False if skipped.
    Mutates ``accum`` in place.
    """
    if not response:
        logger.warning("Empty response for (%s, q=%d); skipping cell", source_persona_name, q_idx)
        accum.n_cells_skipped += 1
        return False

    try:
        batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
            tokenizer=tokenizer,
            system_prompts=target_prompts,
            question=question,
            response_text=response,
        )
    except ValueError as e:
        logger.warning(
            "Token-boundary mismatch for (%s, q=%d): %s",
            source_persona_name,
            q_idx,
            e,
        )
        accum.n_cells_skipped += 1
        return False

    if response_len < 2:
        logger.warning(
            "Response too short (%d tok) for (%s, q=%d); skipping",
            response_len,
            source_persona_name,
            q_idx,
        )
        accum.n_cells_skipped += 1
        return False

    log_probs = teacher_force_batch(
        model,
        batch_inputs,
        prompt_lengths,
        response_len,
        device=device,
        max_batch=TF_MAX_BATCH,
    )
    # log_probs: (N targets, T, V) on CPU. N is 11 by default; 12 when an
    # extra source persona (e.g. nurse) is appended for non-ALL_EVAL sources.

    # Source-row entropy (C5 input): the source persona's row of the batch
    # is the natural reference for the response generated under that
    # persona. ``source_idx`` is None when the source is not present in
    # ``target_persona_names`` (e.g. early base-model cells before the nurse
    # row was added) — in that case we have no source row to read entropy
    # from and skip the contribution. With the path-A fix below, every cell
    # has its source either as one of the 11 ALL_EVAL personas or as the
    # 12th appended source, so this branch is reached only as a defensive
    # guard.
    if source_idx is not None:
        with torch.no_grad():
            src_lp = log_probs[source_idx]
            src_p = src_lp.exp()
            entropy_per_token = -(src_p * src_lp).sum(dim=-1)
            accum.source_token_entropies.append(entropy_per_token.mean().item())
            del src_lp, src_p, entropy_per_token

    resp_ids = _extract_response_token_ids(
        tokenizer=tokenizer,
        question=question,
        response=response,
        target_persona_prompt=target_prompts[0],
    )
    if len(resp_ids) != response_len:
        raise RuntimeError(
            f"Re-encoded response len {len(resp_ids)} != teacher-force len {response_len} "
            f"for ({source_persona_name}, q={q_idx})"
        )

    # BLOCKER #2 fix: marker mask via substring -> char-span -> token-position
    # mapping (NOT token-id union — the old approach masked every newline
    # and every ']' token because Qwen BPEs ``\n[ZLT]`` into pieces that
    # include those generic ids).
    marker_mask = _marker_mask_positions(resp_ids, tokenizer)
    if marker_mask.shape[0] != response_len:
        raise RuntimeError(
            f"Marker mask len {marker_mask.shape[0]} != response_len {response_len} "
            f"for ({source_persona_name}, q={q_idx})"
        )
    mask_keep = ~marker_mask  # True = keep this token (NOT a marker position)

    # BLOCKER #3 fix: detect whether the response contains the marker by
    # decoded-text substring match (case-insensitive). Reusing the old
    # token-id mask was over-broad and flagged every multi-line / bracketed
    # response as "has marker", which made the C1b marker-free subset
    # near-empty.
    response_text_for_marker = tokenizer.decode(resp_ids, skip_special_tokens=False)
    response_has_marker = MARKER_TOKEN.lower() in response_text_for_marker.lower()

    # Raw JS + KL pairs.
    js_pairs_raw, kl_pairs_raw = compute_pairwise_divergences(
        log_probs,
        persona_names=target_persona_names,
        kl_only=False,
        gpu_device=device,
    )
    _accumulate_pairs(accum.js_accum, js_pairs_raw)
    _accumulate_pairs(accum.kl_accum, kl_pairs_raw)
    accum.n_cells_raw += 1

    # Marker-masked JS (C1). Only run if at least one non-marker token remains.
    if mask_keep.any().item():
        js_pairs_masked = _per_cell_js_with_mask(log_probs, target_persona_names, mask_keep)
        _accumulate_pairs(accum.js_masked_accum, js_pairs_masked)

    # Marker-free response subset JS (C1b).
    if not response_has_marker:
        _accumulate_pairs(accum.js_no_marker_accum, js_pairs_raw)
        accum.n_cells_no_marker += 1

    del log_probs, batch_inputs
    return True


def _compute_state_metrics(
    merged_dir: str,
    gpu_id: int,
    persona_questions: list[tuple[str, str, str]],
    persona_names: list[str],
    responses: list[str],
    extra_source: tuple[str, str] | None = None,
) -> dict:
    """Run HF teacher-force + reductions on one state.

    BLOCKER #1 (path A): when ``extra_source`` is set (i.e. the per-state
    source persona is NOT in ``ALL_EVAL_PERSONAS`` — today: nurse),
    ``persona_questions`` and ``responses`` already include the extra
    source's 20 (source, question) cells, AND the teacher-force prompt list
    is extended with the extra source's system prompt as a 12th target.
    The resulting per-cell JS log-probs tensor is shape ``(12, T, V)`` and
    the aggregated JS matrix is 12×12. The aggregator then has real (not
    None) JS values for nurse-source rows — fulfilling the user's n=71
    contract.

    Args:
        extra_source: optional (name, system_prompt) tuple. When provided,
            it is appended as the (n+1)-th teacher-force prompt and as an
            additional source persona in the generation grid. Source-token
            entropy for this row is included in the C5 mean.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu_id}"
    n_questions = len(EVAL_QUESTIONS)
    expected_responses = len(persona_names) * n_questions
    if len(responses) != expected_responses:
        raise RuntimeError(
            f"Expected {expected_responses} responses ({len(persona_names)} sources × "
            f"{n_questions} questions), got {len(responses)}"
        )

    response_by_pq: dict[tuple[str, str], str] = {}
    for idx, (p_name, _p_prompt, q) in enumerate(persona_questions):
        response_by_pq[(p_name, q)] = responses[idx]

    logger.info("Loading HF model from %s for teacher-force", merged_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        merged_dir, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        torch_dtype=HF_DTYPE,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    # Build the target persona list. If extra_source is set and not already in
    # ALL_EVAL_PERSONAS, append it as a 12th teacher-force target.
    target_persona_names = list(ALL_EVAL_PERSONAS.keys())
    target_prompts = list(ALL_EVAL_PERSONAS.values())
    if extra_source is not None:
        extra_name, extra_prompt = extra_source
        if extra_name not in target_persona_names:
            target_persona_names = [*target_persona_names, extra_name]
            target_prompts = [*target_prompts, extra_prompt]
    name_to_target_idx = {n: i for i, n in enumerate(target_persona_names)}

    accum = _CellAccumulator(target_persona_names)

    grid_total = len(persona_questions)
    t_grid_start = time.time()

    for grid_count, (source_persona_name, _src_prompt, question) in enumerate(
        persona_questions, start=1
    ):
        q_idx = EVAL_QUESTIONS.index(question)
        # Source row in the teacher-force batch: the row whose system prompt
        # matches the source persona's. None if the source isn't represented
        # in target_persona_names (defensive — should not happen now that we
        # always append nurse for non-ALL_EVAL sources).
        source_idx = name_to_target_idx.get(source_persona_name)
        _process_cell(
            model=model,
            tokenizer=tokenizer,
            device=device,
            accum=accum,
            source_idx=source_idx,
            source_persona_name=source_persona_name,
            q_idx=q_idx,
            question=question,
            response=response_by_pq[(source_persona_name, question)],
            target_persona_names=target_persona_names,
            target_prompts=target_prompts,
        )
        if grid_count % 22 == 0:
            logger.info(
                "  grid %d/%d (%.1fs elapsed)",
                grid_count,
                grid_total,
                time.time() - t_grid_start,
            )

    if accum.n_cells_raw == 0:
        raise RuntimeError("All cells were skipped — JS aggregation has nothing to report")

    js_accum = accum.js_accum
    js_masked_accum = accum.js_masked_accum
    js_no_marker_accum = accum.js_no_marker_accum
    kl_accum = accum.kl_accum
    source_token_entropies = accum.source_token_entropies
    n_cells_raw = accum.n_cells_raw
    n_cells_no_marker = accum.n_cells_no_marker
    n_cells_skipped = accum.n_cells_skipped

    # ── Build matrices ────────────────────────────────────────────────────
    js_matrix = _aggregate_to_matrix(js_accum, target_persona_names, symmetric=True)
    js_matrix_masked = _aggregate_to_matrix(js_masked_accum, target_persona_names, symmetric=True)
    js_matrix_no_marker = _aggregate_to_matrix(
        js_no_marker_accum, target_persona_names, symmetric=True
    )
    kl_matrix = _aggregate_to_matrix(kl_accum, target_persona_names, symmetric=False)

    # ── Per-state self-tests ──────────────────────────────────────────────
    if js_matrix is None:
        raise RuntimeError("js_matrix is None after aggregation")
    _check_consistency(js_matrix, target_persona_names)
    if js_matrix_masked is not None:
        _check_consistency(js_matrix_masked, target_persona_names)
    if js_matrix_no_marker is not None:
        _check_consistency(js_matrix_no_marker, target_persona_names)
    # Note: kl_matrix is asymmetric so we do not run _check_consistency on it.

    source_token_entropy_mean = (
        sum(source_token_entropies) / len(source_token_entropies)
        if source_token_entropies
        else float("nan")
    )

    # ── Free HF teacher-force model before centroid extraction ────────────
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Build the persona dict for cosine extraction. Match target_persona_names
    # (which already includes the extra source as a 12th entry when applicable)
    # so the resulting matrix has the same axes as the JS matrix.
    cosine_personas: dict[str, str] = {n: ALL_EVAL_PERSONAS[n] for n in ALL_EVAL_PERSONAS}
    if extra_source is not None and extra_source[0] not in cosine_personas:
        cosine_personas[extra_source[0]] = extra_source[1]

    centroids, persona_names_check = extract_centroids(
        model_path=merged_dir,
        personas=cosine_personas,
        questions=EVAL_QUESTIONS,
        layers=LAYERS_FOR_COSINE,
        device=device,
        dtype=HF_DTYPE,
    )
    if persona_names_check != target_persona_names:
        raise RuntimeError(
            f"Centroid persona order {persona_names_check} != "
            f"target_persona_names {target_persona_names}"
        )

    cosine_matrices: dict[str, dict] = {}
    for layer in LAYERS_FOR_COSINE:
        cos_t = compute_cosine_matrix(centroids[layer], centering="global_mean")
        cosine_matrices[f"layer_{layer}"] = {
            "persona_names": target_persona_names,
            "matrix": cos_t.tolist(),
        }

    return {
        "persona_names": target_persona_names,
        "js_matrix": js_matrix,
        "js_matrix_masked": js_matrix_masked,
        "js_matrix_no_marker": js_matrix_no_marker,
        "kl_matrix": kl_matrix,
        "source_token_entropy_mean": source_token_entropy_mean,
        "cosine_matrices": cosine_matrices,
        "n_cells_raw": n_cells_raw,
        "n_cells_no_marker": n_cells_no_marker,
        "n_cells_skipped": n_cells_skipped,
    }


# ── Worker entry point ────────────────────────────────────────────────────


def _state_output_path(output_dir: Path, source: str, checkpoint_step: int) -> Path:
    if source == BASE_SOURCE and checkpoint_step == BASE_CHECKPOINT_STEP:
        return output_dir / BASE_SOURCE / "checkpoint-0" / "result.json"
    return output_dir / source / f"checkpoint-{checkpoint_step}" / "result.json"


def run_state(
    source: str,
    checkpoint_step: int,
    gpu_id: int,
    output_dir: Path,
    gpu_mem_util: float = VLLM_GPU_MEM_UTIL_DEFAULT,
    seed: int = GEN_SEED,
    skip_if_exists: bool = True,
) -> Path:
    """Compute one state's JS + cosine bundle and write result.json."""
    out_path = _state_output_path(output_dir, source, checkpoint_step)
    if skip_if_exists and out_path.exists():
        logger.info("[%s ckpt-%d] result.json exists, skipping", source, checkpoint_step)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If parent narrowed CUDA_VISIBLE_DEVICES to one GPU, our gpu_id is 0.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(visible) == 1:
            gpu_id = 0

    is_base = source == BASE_SOURCE and checkpoint_step == BASE_CHECKPOINT_STEP

    if is_base:
        # vLLM and HF both accept the HF model id directly.
        merged_dir: Path = Path(BASE_MODEL)
        adapter_subpath: str | None = None
        lora_dropout_train_time: float | None = None
    else:
        if source not in ADAPTER_MAP:
            raise ValueError(f"Unknown source '{source}'")
        if checkpoint_step not in CHECKPOINT_STEPS:
            raise ValueError(
                f"Unknown checkpoint step {checkpoint_step}; valid: {CHECKPOINT_STEPS}"
            )
        _, adapter_subpath, lora_dropout_train_time = ADAPTER_MAP[source]
        merged_dir = TMP_ROOT / f"{source}_ckpt{checkpoint_step}"
        if not merged_dir.exists():
            _download_and_merge_adapter(source, checkpoint_step, merged_dir, gpu_id=gpu_id)
        else:
            logger.info(
                "[%s ckpt-%d] merged dir already exists at %s — reusing",
                source,
                checkpoint_step,
                merged_dir,
            )

    t_start = time.time()
    started_at = datetime.now(UTC).isoformat()

    # The pipeline below is wrapped in ``try/finally`` so the per-state
    # merged dir is ALWAYS torn down — even if vLLM / teacher-force /
    # cosine extraction raises (R6 disk-hygiene fix). Without this, a
    # crashing Phase-1 worker leaks ~14 GB safetensors into ``/workspace``.
    # The ``is_base`` short-circuit at the bottom (no rmtree on the literal
    # base-model HF id) is preserved.
    try:
        # Source-generation grid: 11 ALL_EVAL personas × 20 questions = 220 cells
        # by default. When the checkpoint's source is NOT in ALL_EVAL_PERSONAS
        # (today: nurse) we append a 12th source-row using the source's system
        # prompt — yielding 240 cells. Plan-A path for the user's n=71 decision.
        persona_names = list(ALL_EVAL_PERSONAS.keys())
        persona_questions: list[tuple[str, str, str]] = []
        for p_name in persona_names:
            p_prompt = ALL_EVAL_PERSONAS[p_name]
            for q in EVAL_QUESTIONS:
                persona_questions.append((p_name, p_prompt, q))

        extra_source: tuple[str, str] | None = None
        if not is_base and source not in ALL_EVAL_PERSONAS and source in EXTRA_SOURCE_PROMPTS:
            extra_prompt = EXTRA_SOURCE_PROMPTS[source]
            extra_source = (source, extra_prompt)
            for q in EVAL_QUESTIONS:
                persona_questions.append((source, extra_prompt, q))

        responses = _vllm_generate_responses(
            model_path=str(merged_dir),
            persona_questions=persona_questions,
            gpu_mem_util=gpu_mem_util,
            seed=seed,
        )

        # Save raw responses for downstream debugging (no schema commitment).
        raw_resp_path = out_path.parent / "raw_responses.json"
        raw_resp_payload = {
            "source": source,
            "checkpoint_step": checkpoint_step,
            "extra_source": extra_source[0] if extra_source else None,
            "responses": [
                {"persona": p, "question": q, "response": r}
                for (p, _pp, q), r in zip(persona_questions, responses, strict=True)
            ],
        }
        raw_resp_path.write_text(json.dumps(raw_resp_payload))

        metrics = _compute_state_metrics(
            merged_dir=str(merged_dir),
            gpu_id=gpu_id,
            persona_questions=persona_questions,
            persona_names=(
                persona_names if extra_source is None else [*persona_names, extra_source[0]]
            ),
            responses=responses,
            extra_source=extra_source,
        )

        elapsed = time.time() - t_start
        completed_at = datetime.now(UTC).isoformat()
        epoch = EPOCH_BY_STEP.get(checkpoint_step, 0) if not is_base else 0

        payload = {
            "source": source,
            "checkpoint_step": checkpoint_step,
            "epoch": epoch,
            "adapter_repo": ADAPTER_REPO if not is_base else None,
            "adapter_subpath": adapter_subpath,
            "base_model": BASE_MODEL,
            "lora_dropout_train_time": lora_dropout_train_time,
            "persona_names": metrics["persona_names"],
            "extra_source_persona": extra_source[0] if extra_source else None,
            "n_questions": len(EVAL_QUESTIONS),
            "n_completions_for_js": GEN_N,
            "temperature_js": GEN_TEMPERATURE,
            "seed": seed,
            "js_matrix": metrics["js_matrix"],
            "js_matrix_masked": metrics["js_matrix_masked"],
            "js_matrix_no_marker": metrics["js_matrix_no_marker"],
            "kl_matrix": metrics["kl_matrix"],
            "source_token_entropy_mean": metrics["source_token_entropy_mean"],
            "cosine_matrices": metrics["cosine_matrices"],
            "wall_seconds": elapsed,
            "n_cells_raw": metrics["n_cells_raw"],
            "n_cells_no_marker": metrics["n_cells_no_marker"],
            "n_cells_skipped": metrics["n_cells_skipped"],
            "metadata": {
                "git_commit": _git_commit(),
                "started_at": started_at,
                "completed_at": completed_at,
                **_lib_versions(),
            },
        }

        out_path.write_text(json.dumps(payload, indent=2))
        logger.info("[%s ckpt-%d] wrote %s (%.1fs)", source, checkpoint_step, out_path, elapsed)
        return out_path
    finally:
        # is_base reuses the literal HF id — never rmtree that.
        if not is_base and merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
            logger.info("[%s ckpt-%d] cleaned merged dir %s", source, checkpoint_step, merged_dir)
        gc.collect()
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help=f"One of {sorted(ADAPTER_MAP)} or '{BASE_SOURCE}'",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        required=True,
        help=(f"One of {CHECKPOINT_STEPS}, or {BASE_CHECKPOINT_STEP} when source={BASE_SOURCE}"),
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="Logical GPU id (0 if CUDA_VISIBLE_DEVICES already narrows the view)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_228",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=VLLM_GPU_MEM_UTIL_DEFAULT,
    )
    parser.add_argument("--seed", type=int, default=GEN_SEED)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute even if result.json already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_path = run_state(
        source=args.source,
        checkpoint_step=args.checkpoint_step,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        gpu_mem_util=args.gpu_mem_util,
        seed=args.seed,
        skip_if_exists=not args.no_skip_existing,
    )
    print(f"OK {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
