"""Issue 1092 GPU phase: work-conserving multi-GPU dispatcher + activation capture.

Implements phases P2-P3: generate completions (vLLM) + teacher-forced capture
(HF) + B0 r_B-projection pooling (own-policy cells only).

Usage:
    uv run python scripts/issue1092_gpu_phase.py \\
        --issue 1092 \\
        --phases gen_instruct,gen_pretrained,capture_all \\
        --corpus-rev <sha> \\
        --rb-rev 037fcbb \\
        --out /workspace/issue1092 \\
        --cells cell_inst_own,cell_pre_own \\
        [--row-limit 32] [--no-upload] [--cpu-smoke]

Smoke (CPU carve-out):
    --cpu-smoke flag: skips model loading, emits synthetic outputs, exits 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Load dotenv before any HF/API imports (subprocess-env-explicit requirement)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
INSTRUCT_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PRETRAINED_MODEL = "Qwen/Qwen2.5-7B"
PRETRAINED_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_PREFIX = "issue1092_realistic_crossing"
SUMMARY_UPLOAD_PREFIX = f"{HF_PREFIX}/analysis_tensors/summaries"
RAW_COMPLETIONS_UPLOAD_PREFIX = f"{HF_PREFIX}/raw_completions"

N_LAYERS = 28
HIDDEN_DIM = 3584
GEN_SEED = 42
MAX_GEN_TOKENS = 1024
MAX_MODEL_LEN = 8192
MAX_FORMATTED_TOKENS = 7168
MAX_JSONL_SHARD_BYTES = 8_500_000
MAX_JSONL_LINE_BYTES = 9_000_000
CAPTURE_BATCH_SIZE = int(os.environ.get("EPM_CAPTURE_BATCH_SIZE", "8"))

# Stop tokens: #825 recipe
STOP_TOKENS_INSTRUCT = ["<|im_end|>"]
STOP_TOKENS_PRETRAINED = ["\n\nUser:", "\n\nAssistant:", "\n\n"]

# B0 r_B projection pooling modes
CAPTURE_POOLING_MODES = ["mean", "max", "top3", "last"]
N_TRAITS = 3

# Own-policy cells: run B0 r_B projection for these
CELLS_OWN_POLICY = {"cell_inst_own", "cell_pre_own"}

# All 8 cells and their model/format config
CELL_CONFIG: dict[str, dict[str, Any]] = {
    "cell_inst_own": {
        "model": "instruct",
        "prompt_format": "instruct",
        "text_source": "own",
        "own_policy": True,
    },
    "cell_pre_insttext": {
        "model": "pretrained",
        "prompt_format": "instruct",
        "text_source": "instruct",
        "own_policy": False,
    },
    "cell_pre_own": {
        "model": "pretrained",
        "prompt_format": "pretrained",
        "text_source": "own",
        "own_policy": True,
    },
    "cell_inst_pretext": {
        "model": "instruct",
        "prompt_format": "pretrained",
        "text_source": "pretrained",
        "own_policy": False,
    },
    "cell_inst_claude": {
        "model": "instruct",
        "prompt_format": "instruct",
        "text_source": "claude",
        "own_policy": False,
    },
    "cell_pre_claude": {
        "model": "pretrained",
        "prompt_format": "pretrained",
        "text_source": "claude",
        "own_policy": False,
    },
    "cell_inst_shuf": {
        "model": "instruct",
        "prompt_format": "instruct",
        "text_source": "shuffled",
        "own_policy": False,
    },
    "cell_pre_shuf": {
        "model": "pretrained",
        "prompt_format": "pretrained",
        "text_source": "shuffled",
        "own_policy": False,
    },
}

# Per-shard chunk size for vLLM to avoid deadlock (#664 recipe)
DEFAULT_VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

# Summary kinds: prefix-end, context-end, t1 answer mean, t2 answer+boundary
# mean, and t3 next-user boundary slot (plan §4.0).
SUMMARY_KINDS = ["prefix_end", "context_end", "t1", "t2", "t3"]

# G2 gate: spot-check 50 rows after first cell
G2_SPOT_ROWS = 50
G2_SPOT_SEED = 99


# ---------------------------------------------------------------------------
# Render helpers (mirrored from issue1092_build_corpus.py)
# ---------------------------------------------------------------------------


def _get_tokenizer():
    """Lazy-loaded tokenizer for instruct rendering."""
    if not hasattr(_get_tokenizer, "_tok"):
        from transformers import AutoTokenizer

        _get_tokenizer._tok = AutoTokenizer.from_pretrained(
            INSTRUCT_MODEL,
            revision=INSTRUCT_REVISION,
            trust_remote_code=True,
        )
    return _get_tokenizer._tok


def _render_instruct(turns: list[dict], query: str) -> str:
    """Render as instruct chat-template (tokenizer.apply_chat_template)."""
    tok = _get_tokenizer()
    messages = []
    for t in turns:
        messages.append({"role": t["role"], "content": t["content"]})
    messages.append({"role": "user", "content": query})
    rendered = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


def _render_naturalistic(turns: list[dict], query: str) -> str:
    """Render as naturalistic plain text (User: / Assistant: format)."""
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content']}")
        lines.append("")
    lines.append(f"User: {query}")
    lines.append("")
    lines.append("Assistant:")
    return "\n".join(lines)


def _render_prefix_instruct(turns: list[dict]) -> str:
    if not turns:
        # Round-8.2 guard: the Qwen chat template cannot render an EMPTY
        # messages list (Jinja `messages[0]` -> IndexError, verified live on
        # the pinned tokenizer). Bare-context callers must not reach the
        # template; `_render_prompt_parts` derives the bare instruct prefix
        # from the rendered PROMPT instead (the injected system block).
        return ""
    tok = _get_tokenizer()
    messages = [{"role": t["role"], "content": t["content"]} for t in turns]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _render_prefix_naturalistic(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content']}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_manifest(corpus_dir: Path) -> list[dict]:
    """Load manifest.jsonl rows."""
    manifest_path = corpus_dir / "manifest.jsonl"
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_store(corpus_dir: Path, store_name: str) -> dict[str, dict]:
    """Load a store JSONL (prefix_store.jsonl or query_store.jsonl), keyed by id."""
    store_path = corpus_dir / store_name
    result = {}
    with open(store_path) as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                key = item.get("id") or item.get("prefix_id") or item.get("query_id")
                if key is None:
                    raise KeyError(
                        f"{store_path} item missing id/prefix_id/query_id: {item.keys()}"
                    )
                result[str(key)] = item
    return result


def _prefix_turns(prefix_item: dict) -> list[dict]:
    """Prefix turns for a store item; an EXPLICIT empty list is a VALID bare context.

    Round-8.2 (BLOCKER concern i1092-battery-bare-prefix-gpu-phase-crash): the
    round-8 battery fix legitimately ships ``batt_f6_default_template`` with
    ``prefix_turns: []`` (bare default context), and single-turn ``pfx_``
    conversations are likewise empty-prefixed by design. The old
    ``.get("prefix_turns") or .get("turns")`` chain coerced the valid ``[]``
    to None and raised on every f6 row in every cell (~48 rows/cell, all 8
    cells via the control subset). A PRESENT ``prefix_turns`` key is now
    authoritative even when empty; only a genuinely ABSENT/non-list turns
    field stays fail-loud.
    """
    if "prefix_turns" in prefix_item:
        turns = prefix_item["prefix_turns"]
    else:
        turns = prefix_item.get("turns")
    if not isinstance(turns, list):
        raise ValueError(f"prefix item {prefix_item.get('prefix_id')} has no turns")
    return turns


def _query_text(query_item: dict) -> str:
    text = query_item.get("text", query_item.get("query"))
    if not isinstance(text, str) or not text:
        raise ValueError(f"query item {query_item.get('query_id')} has no text/query")
    return text


_INSTRUCT_USER_HEADER = "<|im_start|>user\n"


def _render_prompt_parts(turns: list[dict], query: str, prompt_format: str) -> tuple[str, str]:
    """Return (prefix_text, prompt_text) under the requested model prompt format.

    Bare context (``turns == []``, round-8.2): the canonical prefix is
    "everything before the user query". Under the INSTRUCT format that is the
    template-injected default system block — sliced off the rendered prompt
    itself at its (only) user-turn header, so ``prompt_text.startswith
    (prefix_text)`` holds exactly as it does for non-empty prefixes and the
    prefix_end capture position stays "last token before the query turn".
    Under the NATURALISTIC format nothing precedes the query -> prefix "".
    """
    if prompt_format == "instruct":
        prompt_text = _render_instruct(turns, query)
        if turns:
            prefix_text = _render_prefix_instruct(turns)
        else:
            idx = prompt_text.find(_INSTRUCT_USER_HEADER)
            if idx < 0:
                raise ValueError(
                    "bare-context instruct render lacks a user-turn header; cannot "
                    "derive the prefix (template drift?)"
                )
            prefix_text = prompt_text[:idx]
    elif prompt_format == "pretrained":
        prefix_text = _render_prefix_naturalistic(turns)
        prompt_text = _render_naturalistic(turns, query)
    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format!r}")
    return prefix_text, prompt_text


def render_row(
    row: dict,
    prefix_store: dict,
    query_store: dict,
    prompt_format: str,
    text_source: str,
    completion_override: str | None = None,
) -> tuple[str, str, str | None]:
    """Render a manifest row to (prompt_text, completion_text_or_None).

    Returns:
        prefix: the rendered prefix without the query
        prompt: the input prefix/context text
        completion: the completion text (for cross-cell formats), or None for
                    own-policy cells (completion comes from generation)
    """
    prefix_id = row["prefix_id"]
    query_id = row["query_id"]

    prefix_item = prefix_store[prefix_id]
    query_item = query_store[query_id]

    turns = _prefix_turns(prefix_item)
    query = _query_text(query_item)
    prefix_text, prompt = _render_prompt_parts(turns, query, prompt_format)

    # round-8.8: explicit None checks — NEVER `a or b` chains. An EMPTY
    # completion is a legitimate generation outcome (the pretrained model can
    # emit its "\n\n" stop immediately), and the falsy chain coerced it to
    # "missing", killing whole shards (launch 7: the 5 cell_inst_pretext
    # failures were exactly the shards containing >=1 empty cell_pre_own
    # completion, with all 21,193 completions present on disk). Same class as
    # the round-8.2 empty-prefix_turns coercion.
    if text_source == "own":
        return prefix_text, prompt, completion_override
    if text_source == "claude":
        completion = row.get("claude_text")
        if completion is None:
            completion = row.get("completion")
        if completion is None:
            raise ValueError(f"row {row.get('row_id')} has no Claude completion text")
        return prefix_text, prompt, str(completion)
    if text_source in ("instruct", "pretrained"):
        key = f"{text_source}_completion"
        completion = row.get(key)
        if completion is None:
            completion = row.get("completion")
        if completion is None:
            raise ValueError(f"row {row.get('row_id')} has no {key}")
        return prefix_text, prompt, str(completion)
    if text_source == "shuffled":
        completion = row.get("shuffled_completion")
        if completion is None:
            completion = row.get("completion")
        if completion is None:
            raise ValueError(f"row {row.get('row_id')} has no shuffled completion")
        return prefix_text, prompt, str(completion)
    raise ValueError(f"Unknown text_source: {text_source!r}")


# ---------------------------------------------------------------------------
# Fingerprint / resume
# ---------------------------------------------------------------------------


def compute_shard_fingerprint(
    corpus_hash: str,
    cell_id: str,
    row_start: int,
    row_end: int,
    model_id: str,
    dtype: str,
    n_layers: int,
    hidden_dim: int,
    boundary_strings: list[str],
    rb_rev: str,
    max_gen_tokens: int,
    subset_id: str,
    phase_name: str,
    code_sha: str,
) -> dict:
    """Build a fingerprint dict for resume idempotency."""
    return {
        "corpus_hash": corpus_hash,
        "cell_id": cell_id,
        "row_start": row_start,
        "row_end": row_end,
        "subset_id": subset_id,
        "phase_name": phase_name,
        "model_id": model_id,
        "dtype": dtype,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "boundary_strings": boundary_strings,
        "rb_rev": rb_rev,
        "max_gen_tokens": max_gen_tokens,
        "code_sha": code_sha,
    }


def fingerprint_matches(fp_path: Path, expected_fp: dict) -> bool:
    """Check if a saved fingerprint matches expected; return False if not found or mismatch."""
    if not fp_path.exists():
        return False
    try:
        saved = json.loads(fp_path.read_text())
        return all(saved.get(k) == v for k, v in expected_fp.items())
    except Exception:
        return False


def _fingerprint_matches_any_sha(
    fp_path: Path, expected_fp: dict, resume_code_shas: list[str] | None
) -> tuple[bool, str | None]:
    """Fingerprint match, additionally accepting explicit PRIOR code shas (round-8.9).

    A cell shard captured by a prior invocation is bit-identical when the diff
    between the prior script and the current one provably does not touch the
    gen/capture path; ``--resume-code-shas`` lets the OPERATOR assert that so a
    completed cell (launch 8's cell_inst_own, ~78 min on 8x H100) is not
    recomputed just because the script hash moved. Returns
    (matched, matched_prior_sha) — matched_prior_sha is None for a current-sha
    match; scoped to CELL shards only (aux/dynamics never resume across shas —
    the round-8.9 panel filter changes their row sets).
    """
    if fingerprint_matches(fp_path, expected_fp):
        return True, None
    for prior_sha in resume_code_shas or []:
        if fingerprint_matches(fp_path, {**expected_fp, "code_sha": prior_sha}):
            return True, prior_sha
    return False, None


def _phase_fp_label(phase_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in phase_name)


def code_sha() -> str:
    """Return a short SHA of this script for fingerprinting."""
    try:
        this_file = Path(__file__).read_bytes()
        return hashlib.sha256(this_file).hexdigest()[:16]
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Shard definition
# ---------------------------------------------------------------------------


@dataclass
class Shard:
    """A unit of work: one cell x one row range."""

    cell_id: str
    row_start: int
    row_end: int  # exclusive
    shard_idx: int
    total_shards: int


@dataclass
class AuxShard:
    """A P3b/P3c auxiliary unit of work."""

    phase: str
    model_type: str
    row_start: int
    row_end: int
    shard_idx: int
    total_shards: int


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------


def _vllm_engine_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """vLLM engine kwargs from the round-8.5 H100-IMA mitigation flags.

    Launch #4 (RunPod 8x H100): vLLM 0.11.0 died with a CUDA illegal memory
    access inside the engine step at production shapes (~3.8k-token prompts,
    500-prompt chunks, num_common_prefix_blocks=[237] — maximal shared-prefix
    reuse on the dense core), while the identical code ran 42/42 clean on
    A100s and a short-prompt probe on the same H100 pod was clean — the known
    vLLM-on-H100 IMA family around prefix caching / cudagraphs. Both flags
    default OFF (no behavior change unless passed); greedy generation output
    is engine-config-independent, so this is a mitigation knob, not a science
    change.
    """
    overrides: dict[str, Any] = {}
    if getattr(args, "no_prefix_caching", False):
        overrides["enable_prefix_caching"] = False
    if getattr(args, "enforce_eager", False):
        overrides["enforce_eager"] = True
    return overrides


def _run_gen_vllm(
    prompts: list[str],
    model_name: str,
    revision: str,
    stop_tokens: list[str],
    max_tokens: int,
    seed: int,
    gpu_id: int,
    chunk_size: int,
    engine_overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Run vLLM greedy generation on one GPU. Returns list of completions."""
    # Set CVD BEFORE importing vLLM (import-time cuInit)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        revision=revision,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=seed,
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        **(engine_overrides or {}),
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=stop_tokens,
        seed=seed,
    )

    completions = []
    for chunk_start in range(0, len(prompts), chunk_size):
        chunk = prompts[chunk_start : chunk_start + chunk_size]
        logger.info(
            "[gpu=%d] vLLM chunk %d/%d (%d prompts)",
            gpu_id,
            chunk_start // chunk_size + 1,
            math.ceil(len(prompts) / chunk_size),
            len(chunk),
        )
        outputs = llm.generate(chunk, params, use_tqdm=False)
        for out in outputs:
            completions.append(out.outputs[0].text if out.outputs else "")

    # Teardown: destroy model to free GPU memory; orphan guard via psutil
    del llm
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return completions


# ---------------------------------------------------------------------------
# HF teacher-forced capture
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult:
    """Per-row capture result: 5 summary kinds x 28 layers, stored fp16."""

    # summaries[kind][layer] = float16 array of shape (HIDDEN_DIM,)
    summaries: dict  # kind -> np.ndarray (n_layers, HIDDEN_DIM)
    # Token positions for boundary detection
    prefix_end_pos: int = 0
    context_end_pos: int = 0
    gen_token_positions: list[int] = field(default_factory=list)


def _boundary_suffix(prompt_format: str) -> str:
    if prompt_format == "instruct":
        return "<|im_end|>\n<|im_start|>user\n"
    if prompt_format == "pretrained":
        return "\n\nUser:"
    raise ValueError(f"Unknown prompt_format: {prompt_format!r}")


@dataclass
class CaptureBatchOutput:
    summaries: list[dict[str, np.ndarray]]
    rb_pool: np.ndarray | None


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _capture_row_ids_and_positions(
    tokenizer,
    prefix_text: str,
    prompt: str,
    completion: str,
    boundary: str,
    row_label: str = "?",
) -> tuple[list[int], dict[str, int]]:
    """Teacher-forcing input ids + capture positions for one row (round-8.4).

    THE G2 launch-#3 defect (max_abs=2.9): the old capture tokenized the
    CONCATENATED ``prompt + completion + boundary`` string but computed every
    position from PER-SEGMENT token counts. Qwen BPE merges across those
    seams — a completion starting with "\\n" merges into the instruct
    prompt's trailing "assistant\\n" ("\\n"+"\\n" -> id 271), the rstripped
    naturalistic prefix's final "." merges into ".\\n\\n", and a completion
    ending "\\n" merges into the "\\n\\nUser:" boundary (all three verified
    live on the pinned tokenizer, 2026-07-08) — so ``full_ids[:n_prompt] !=
    prompt_ids`` and context_end/prefix_end/t1/t2/t3/B0 were read at SHIFTED
    positions (the #825 BPE-seam class; the dynamics path already uses
    offset-based cuts, this sibling did not).

    Fix: build the forwarded sequence by CONCATENATING PER-SEGMENT TOKEN IDS
    (standard teacher forcing — the prompt segment is then bit-identical to
    what generation consumed and what the G2 reference forwards), and derive
    ``prefix_end`` from the prompt's OFFSET MAPPING (last token ending within
    ``prefix_text``; robust to the rstripped-prefix seam). Positions are
    exact by construction; no re-tokenization of concatenated text anywhere.
    """
    prompt_enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(prompt_enc["input_ids"])
    offsets = prompt_enc["offset_mapping"]
    completion_ids = _token_ids(tokenizer, completion)
    boundary_ids = _token_ids(tokenizer, boundary)
    row_ids = prompt_ids + completion_ids + boundary_ids
    n_total_tokens = len(row_ids)
    if n_total_tokens > MAX_MODEL_LEN:
        raise ValueError(
            f"capture row {row_label} has {n_total_tokens} tokens, "
            f"exceeding MAX_MODEL_LEN={MAX_MODEL_LEN}; loader must filter it"
        )
    if len(prompt_ids) > MAX_FORMATTED_TOKENS:
        raise ValueError(
            f"capture row {row_label} prompt has {len(prompt_ids)} tokens, "
            f"exceeding prompt budget {MAX_FORMATTED_TOKENS}"
        )

    # prefix_end: last prompt token that ends INSIDE prefix_text (offset-based).
    # prefix_text is a string prefix of prompt by construction; a token that
    # BPE-merges across the prefix boundary ends beyond len(prefix_text) and is
    # correctly excluded. Empty prefix (bare context) -> 0 tokens -> clamped 0.
    n_prefix_chars = len(prefix_text)
    n_prefix_tokens = sum(1 for start, end in offsets if end <= n_prefix_chars and end > start)

    prefix_end_pos = min(max(0, n_prefix_tokens - 1), n_total_tokens - 1)
    context_end_pos = min(max(0, len(prompt_ids) - 1), n_total_tokens - 1)
    answer_start = min(context_end_pos + 1, n_total_tokens - 1)
    answer_end = min(context_end_pos + 1 + max(1, len(completion_ids)), n_total_tokens)
    t3_pos = n_total_tokens - 1
    t2_end = max(answer_end, t3_pos)
    positions = {
        "n_total": n_total_tokens,
        "n_prompt": len(prompt_ids),
        "prefix_end": prefix_end_pos,
        "context_end": context_end_pos,
        "answer_start": answer_start,
        "answer_end": answer_end,
        "t2_end": t2_end,
        "t3": t3_pos,
    }
    return row_ids, positions


def _call_model_with_hidden_states(model, input_ids, attention_mask):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "output_hidden_states": True,
    }
    try:
        return model(**kwargs, logits_to_keep=1)
    except TypeError:
        return model(**kwargs)


def _capture_batch_loaded_model(
    *,
    prefix_texts: list[str],
    prompts: list[str],
    completions: list[str],
    prompt_format: str,
    model,
    tokenizer,
    n_layers: int,
    hidden_dim: int,
    device: str,
    log_label: str,
    rb_directions: np.ndarray | None = None,
    batch_size: int = CAPTURE_BATCH_SIZE,
) -> CaptureBatchOutput:
    """Teacher-forced capture with padded batch forwards and stream-reduced B0.

    The function never truncates. Rows over the 8192-token capture window fail
    loudly, matching the loader/filter contract from the plan. Per-token answer
    grids are only materialized while their batch is in scope and are immediately
    reduced to the five summary arrays plus optional B0 poolings.
    """
    import torch

    if len({len(prefix_texts), len(prompts), len(completions)}) != 1:
        raise ValueError("prefix_texts, prompts, and completions must have equal length")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError(
            "capture positions index the UNPADDED sequence and require RIGHT padding; "
            f"tokenizer.padding_side={tokenizer.padding_side!r} (round-8.4 guard)"
        )
    boundary = _boundary_suffix(prompt_format)
    summaries: list[dict[str, np.ndarray]] = []
    rb_rows: list[np.ndarray] = []
    rb_unit: np.ndarray | None = None
    if rb_directions is not None:
        norms = np.linalg.norm(rb_directions, axis=2)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        rb_unit = rb_directions / safe_norms[:, :, None]

    n_total_rows = len(prompts)
    for batch_start in range(0, n_total_rows, max(1, batch_size)):
        batch_end = min(batch_start + max(1, batch_size), n_total_rows)
        if batch_start % (max(1, batch_size) * 5) == 0:
            logger.info(
                "[%s] capture batch rows %d:%d/%d",
                log_label,
                batch_start,
                batch_end,
                n_total_rows,
            )
        batch_prefixes = prefix_texts[batch_start:batch_end]
        batch_prompts = prompts[batch_start:batch_end]
        batch_completions = completions[batch_start:batch_end]

        # round-8.4: per-segment token-id concatenation + offset-based
        # prefix_end (see _capture_row_ids_and_positions) — NEVER re-tokenize
        # the concatenated text (BPE seam merges shift every position; the
        # G2 launch-#3 max_abs=2.9 defect).
        batch_ids: list[list[int]] = []
        positions = []
        for local_i, (prefix_text, prompt, completion) in enumerate(
            zip(batch_prefixes, batch_prompts, batch_completions, strict=True)
        ):
            row_ids, pos = _capture_row_ids_and_positions(
                tokenizer,
                prefix_text,
                prompt,
                completion,
                boundary,
                row_label=str(batch_start + local_i),
            )
            batch_ids.append(row_ids)
            positions.append(pos)

        inputs = tokenizer.pad(
            {"input_ids": batch_ids},
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = _call_model_with_hidden_states(model, input_ids, attention_mask)
        hidden_states = outputs.hidden_states[1:]
        if len(hidden_states) != n_layers:
            raise ValueError(f"model returned {len(hidden_states)} layers, expected {n_layers}")
        if hidden_states[0].shape[-1] != hidden_dim:
            raise ValueError(
                f"model hidden dim {hidden_states[0].shape[-1]} != expected {hidden_dim}"
            )

        for local_i, pos in enumerate(positions):
            row_summary: dict[str, np.ndarray] = {}

            def extract_pos(
                position: int,
                *,
                row_i: int = local_i,
                hs_layers=hidden_states,
            ) -> np.ndarray:
                return np.stack(
                    [hs[row_i, position, :].to(torch.float16).cpu().numpy() for hs in hs_layers],
                    axis=0,
                )

            def extract_span(
                start: int,
                end: int,
                *,
                row_i: int = local_i,
                n_total: int = pos["n_total"],
                hs_layers=hidden_states,
            ) -> np.ndarray:
                start = min(max(0, start), n_total - 1)
                end = min(max(start + 1, end), n_total)
                return np.stack(
                    [
                        hs[row_i, start:end, :].mean(dim=0).to(torch.float16).cpu().numpy()
                        for hs in hs_layers
                    ],
                    axis=0,
                )

            row_summary["prefix_end"] = extract_pos(pos["prefix_end"])
            row_summary["context_end"] = extract_pos(pos["context_end"])
            row_summary["t1"] = extract_span(pos["answer_start"], pos["answer_end"])
            row_summary["t2"] = extract_span(pos["answer_start"], pos["t2_end"])
            row_summary["t3"] = extract_pos(pos["t3"])
            summaries.append(row_summary)

            if rb_unit is not None:
                answer_states = np.stack(
                    [
                        hs[local_i, pos["answer_start"] : pos["answer_end"], :]
                        .to(torch.float16)
                        .cpu()
                        .numpy()
                        for hs in hidden_states
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                projections = np.einsum("alh,lbh->alb", answer_states, rb_unit, optimize=True)
                rb_rows.append(_pool_projections(projections))

        del outputs, hidden_states, input_ids, attention_mask

    rb_pool = np.stack(rb_rows, axis=0).astype(np.float32) if rb_rows else None
    return CaptureBatchOutput(summaries=summaries, rb_pool=rb_pool)


def _capture_batch_hf(
    prefix_texts: list[str],
    prompts: list[str],
    completions: list[str],
    prompt_format: str,
    model_name: str,
    revision: str,
    gpu_id: int,
    n_layers: int,
    hidden_dim: int,
) -> CaptureBatchOutput:
    """Run HF teacher-forced capture for a batch of (prompt, completion) pairs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},  # explicit single visible device, no auto-offload (#825)
    )
    model.eval()

    results = _capture_batch_loaded_model(
        prefix_texts=prefix_texts,
        prompts=prompts,
        completions=completions,
        prompt_format=prompt_format,
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=device,
        log_label=f"gpu={gpu_id}",
    )

    del model
    try:
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# B0 r_B projection pooling
# ---------------------------------------------------------------------------


def load_rb_directions(rb_rev: str, n_layers: int, n_traits: int, hidden_dim: int) -> np.ndarray:
    """Download and load #779 r_B direction tensors from the HF data repo.

    Returns np.ndarray of shape (n_layers, n_traits, hidden_dim), dtype=float32.
    """
    import torch
    from huggingface_hub import hf_hub_download, list_repo_tree

    prefix = "issue779_monitoring/r_b"
    # HUB_VERIFY_RETRY_EXEMPT: issue-1092 driver; scoped listing with orchestration-layer retry
    entries = list_repo_tree(
        HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix,
        revision=rb_rev,
    )
    relpaths = sorted(
        item.path
        for item in entries
        if getattr(item, "size", None) is not None and item.path.endswith(".pt")
    )
    if len(relpaths) != n_traits:
        raise RuntimeError(
            f"expected {n_traits} r_B .pt files under {HF_DATA_REPO}@{rb_rev}:{prefix}, "
            f"found {len(relpaths)}: {relpaths}"
        )

    tensors: list[np.ndarray] = []
    basenames: list[str] = []
    for relpath in relpaths:
        local_path = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=relpath,
            revision=rb_rev,
        )
        payload = torch.load(local_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "r_b" not in payload:
            raise KeyError(f"{relpath} payload must be a dict with key 'r_b'")
        arr = (
            payload["r_b"].detach().cpu().numpy()
            if hasattr(payload["r_b"], "detach")
            else np.asarray(payload["r_b"])
        )
        if arr.shape != (n_layers, hidden_dim):
            raise ValueError(
                f"{relpath} r_b shape {arr.shape} != expected ({n_layers}, {hidden_dim})"
            )
        tensors.append(arr.astype(np.float32, copy=False))
        basenames.append(Path(relpath).stem)

    rb = np.stack(tensors, axis=1)
    if rb.shape != (n_layers, n_traits, hidden_dim):
        raise AssertionError(
            f"stacked r_B shape {rb.shape} != ({n_layers}, {n_traits}, {hidden_dim}); "
            f"basenames={basenames}"
        )
    logger.info("[r_B] loaded traits from data repo: %s", basenames)
    return rb


def _pool_projections(projections: np.ndarray) -> np.ndarray:
    """Pool projection values: (T, n_layers, n_traits) -> (n_layers, n_traits, 4).

    Pooling modes: mean, max, top3-mean, last.
    """
    # projections shape: (T, n_layers, n_traits)
    T = projections.shape[0]
    mean_pool = projections.mean(axis=0)  # (n_layers, n_traits)
    max_pool = projections.max(axis=0)  # (n_layers, n_traits)

    if T >= 3:
        # top3: mean of top-3 absolute values per (layer, trait)
        abs_proj = np.abs(projections)  # (T, n_layers, n_traits)
        top3_idx = np.argsort(-abs_proj, axis=0)[:3, :, :]  # (3, n_layers, n_traits)
        top3_vals = np.take_along_axis(projections, top3_idx, axis=0)
        top3_pool = top3_vals.mean(axis=0)  # (n_layers, n_traits)
    else:
        top3_pool = mean_pool.copy()

    last_pool = projections[-1]  # (n_layers, n_traits)

    # Stack: (n_layers, n_traits, 4)
    result = np.stack([mean_pool, max_pool, top3_pool, last_pool], axis=-1)
    return result.astype(np.float32)


class PersistentGpuRuntime:
    """One child-process runtime bound to exactly one visible GPU."""

    def __init__(self, gpu_id: int, engine_overrides: dict[str, Any] | None = None):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.gpu_id = gpu_id
        self.device = "cuda:0"
        # round-8.5 H100-IMA mitigation knobs (see _vllm_engine_overrides)
        self.engine_overrides: dict[str, Any] = dict(engine_overrides or {})
        self._llms: dict[tuple[str, str], Any] = {}
        self._hf_models: dict[tuple[str, str], tuple[Any, Any]] = {}

    def generate(
        self,
        *,
        prompts: list[str],
        model_name: str,
        revision: str,
        stop_tokens: list[str],
        max_tokens: int,
        seed: int,
        chunk_size: int,
    ) -> list[str]:
        from vllm import LLM, SamplingParams

        key = (model_name, revision)
        if key not in self._llms:
            self._llms[key] = LLM(
                model=model_name,
                revision=revision,
                dtype="bfloat16",
                trust_remote_code=True,
                seed=seed,
                gpu_memory_utilization=0.85,
                max_model_len=MAX_MODEL_LEN,
                **self.engine_overrides,
            )
        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop=stop_tokens,
            seed=seed,
        )
        completions = []
        llm = self._llms[key]
        for chunk_start in range(0, len(prompts), chunk_size):
            chunk = prompts[chunk_start : chunk_start + chunk_size]
            logger.info(
                "[gpu=%d] vLLM chunk %d/%d (%d prompts)",
                self.gpu_id,
                chunk_start // chunk_size + 1,
                math.ceil(len(prompts) / chunk_size),
                len(chunk),
            )
            outputs = llm.generate(chunk, params, use_tqdm=False)
            completions.extend(out.outputs[0].text if out.outputs else "" for out in outputs)
        return completions

    def capture(
        self,
        *,
        prefix_texts: list[str],
        prompts: list[str],
        completions: list[str],
        prompt_format: str,
        model_name: str,
        revision: str,
        n_layers: int,
        hidden_dim: int,
        rb_directions: np.ndarray | None,
    ) -> CaptureBatchOutput:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        key = (model_name, revision)
        if key not in self._hf_models:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map={"": self.device},
            )
            model.eval()
            self._hf_models[key] = (tokenizer, model)
        tokenizer, model = self._hf_models[key]
        return _capture_batch_loaded_model(
            prefix_texts=prefix_texts,
            prompts=prompts,
            completions=completions,
            prompt_format=prompt_format,
            model=model,
            tokenizer=tokenizer,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            device=self.device,
            log_label=f"gpu={self.gpu_id}",
            rb_directions=rb_directions,
            batch_size=CAPTURE_BATCH_SIZE,
        )

    def close(self) -> None:
        self._llms.clear()
        self._hf_models.clear()
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("[gpu=%d] runtime CUDA cleanup skipped", self.gpu_id, exc_info=True)


# ---------------------------------------------------------------------------
# G2 identity gate
# ---------------------------------------------------------------------------


def run_g2_gate(
    cell_id: str,
    summaries_list: list[dict[str, np.ndarray]],
    row_indices: list[int],
    rb_pool: np.ndarray | None,
    n_spot: int = G2_SPOT_ROWS,
    seed: int = G2_SPOT_SEED,
) -> bool:
    """G2 gate: identity check on spot rows.

    Returns True if gate passes, False otherwise.
    """
    rng = np.random.default_rng(seed)
    n = len(summaries_list)
    if n == 0:
        raise ValueError("[G2] no summaries to check")

    spot_n = min(n_spot, n)
    spot_idx = rng.choice(n, size=spot_n, replace=False)

    # Check 1: summaries have correct shape
    for i in spot_idx[:5]:
        s = summaries_list[i]
        for kind in SUMMARY_KINDS:
            if kind not in s:
                logger.error("[G2] Missing summary kind %r in row %d", kind, i)
                return False
            shape = s[kind].shape
            if shape[0] != N_LAYERS or shape[1] != HIDDEN_DIM:
                logger.error(
                    "[G2] Summary %r shape %s != (%d, %d)", kind, shape, N_LAYERS, HIDDEN_DIM
                )
                return False

    # Check 2: r_B pool shape
    if rb_pool is not None:
        expected_shape = (n, N_LAYERS, N_TRAITS, len(CAPTURE_POOLING_MODES))
        if rb_pool.shape != expected_shape:
            logger.error("[G2] r_B pool shape %s != %s", rb_pool.shape, expected_shape)
            return False

        # Spot-check: pairing-permutation null R² at L14 should be near 0 (not a scientific
        # check here, just a shape/finite check)
        l14_proj = rb_pool[:, 14, :, 0]  # (n_rows, n_traits) mean-pool at layer 14
        if not np.all(np.isfinite(l14_proj)):
            logger.error("[G2] Non-finite values in r_B pool at L14")
            return False

    logger.info("[G2] Gate PASSED for cell %s (n=%d, spot_n=%d)", cell_id, n, spot_n)
    return True


def check_dispatch_errors(results: list[dict]) -> None:
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        logger.error("[main] %d shards failed:", len(errors))
        for err in errors:
            logger.error("  %s shard%d: %s", err["cell_id"], err["shard_idx"], err.get("error"))
        raise RuntimeError(f"{len(errors)} shards failed")


def _sorted_shards(paths: list[Path]) -> list[Path]:
    def key(path: Path) -> tuple[str, int, str]:
        stem = path.stem
        if "_shard" not in stem:
            return stem, -1, stem
        prefix, raw = stem.split("_shard", 1)
        digits = []
        for ch in raw:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        return prefix, int("".join(digits) or 0), raw

    ordered = sorted(paths, key=key)
    seen: dict[tuple[str, int], Path] = {}
    for path in ordered:
        prefix, shard_idx, _raw = key(path)
        if shard_idx < 0:
            continue
        shard_key = (prefix, shard_idx)
        if shard_key in seen:
            raise ValueError(
                f"duplicate shard index {shard_idx} for {prefix}: "
                f"{seen[shard_key].name} and {path.name}"
            )
        seen[shard_key] = path
    return ordered


def _load_cell_kind_matrix(out_dir: Path, cell_id: str, kind: str, layer: int) -> np.ndarray:
    cell_dir = out_dir / "summaries" / cell_id
    paths = _sorted_shards(sorted(cell_dir.glob(f"{kind}_L{layer:02d}_shard*.npy")))
    if not paths:
        paths = sorted(cell_dir.glob(f"{kind}_L{layer:02d}.npy"))
    if not paths:
        raise FileNotFoundError(f"G2 missing {cell_id}/{kind}_L{layer:02d}")
    return np.concatenate([np.load(path, mmap_mode="r") for path in paths], axis=0)


def _load_cell_summary_rows(
    out_dir: Path, cell_id: str, kind: str, row_idx: np.ndarray, *, n_layers: int
) -> np.ndarray:
    layers = [
        np.asarray(_load_cell_kind_matrix(out_dir, cell_id, kind, layer)[row_idx])
        for layer in range(n_layers)
    ]
    return np.stack(layers, axis=1).astype(np.float32)


def _load_b0_pool_matrix(out_dir: Path, cell_id: str) -> np.ndarray:
    pool_dir = out_dir / "summaries" / "b0_rB_pool"
    pool_paths = _sorted_shards(sorted(pool_dir.glob(f"{cell_id}_shard*.npy")))
    if not pool_paths:
        pool_paths = sorted(pool_dir.glob(f"{cell_id}.npy"))
    if not pool_paths:
        raise FileNotFoundError(f"G2 missing B0 pool shards for own-policy cell {cell_id}")
    return np.concatenate([np.load(path, mmap_mode="r") for path in pool_paths], axis=0)


# ── G2 identity criterion (round-8.6, calibrated on the measured null) ──────
# Derivation (issue1092_g2_diag.py on pod-1092, 2026-07-08; 50 spot rows of
# cell_inst_own, real captures): construction is EXACT — token ids identical
# on all 50 rows, recompute-vs-disk 0.0, fp32 both-sides max_rel 7.5e-5 — and
# the ENTIRE gate residual is bf16 batch-geometry numerics: the same-ids
# batch=1-vs-batch=8 null realized max_abs=3.0 at the SAME (row, layer, dim)
# element as the gate read (disk -221.0 vs ref -218.0, 1.36% relative on a
# magnitude-221 outlier dim; Qwen2-family late layers carry |~200-1000|
# outlier dims), p99 floored-rel 0.076, depth-amplified L0 0.016 -> L26 3.0
# (the #779-r12 signature). A MAX-relative bar cannot separate (the pure null
# reaches floored-rel 0.866 on one small-magnitude tail element), so the
# criterion reads the BULK. Division of labor (round-8.7 amendment, per the
# 8.6 reviewer's probe): a single CORRELATED adjacent-position row shifts the
# p99 only to ~0.16-0.23 (PASSES this bar) — that mode is caught EXACTLY by
# the deterministic token-id pre-check (_g2_token_id_precheck), which runs
# BEFORE this statistical comparison; the p99 bar catches >=3-row and
# UNCORRELATED misalignment modes (uncorrelated states put >=2% of elements
# at rel ~1, driving the p99 to ~1), while the numerics null sits at 0.076.
# Thresholds = measured null x margin, NOT bare constants — re-derive with
# issue1092_g2_diag.py if the model / dtype / hardware changes:
#   p99 floored-rel <= 0.30  (~4x the 0.076 null p99, ~3x under the ~1.0
#                             misalignment signature)
#   max_abs         <= 300.0 (100x the 3.0 null max; misaligned outlier dims
#                             shift by O(magnitude) ~ hundreds)
G2_IDENTITY_P99_REL_TOL = 0.30
G2_IDENTITY_REL_FLOOR = 1.0
G2_IDENTITY_ABS_BACKSTOP = 300.0


def _g2_identity_check(disk: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Calibrated G2 identity criterion (see derivation block above).

    PASS iff p99 of the floored per-element relative error (|a-b| /
    max(|a|,|b|,floor)) <= G2_IDENTITY_P99_REL_TOL AND max abs error <=
    G2_IDENTITY_ABS_BACKSTOP. Returns the stats dict (verdict + measured
    quantities + thresholds) for fail-loud logging either way.
    """
    a = disk.astype(np.float64)
    b = ref.astype(np.float64)
    err = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = err / np.maximum(scale, G2_IDENTITY_REL_FLOOR)
    p99_rel = float(np.quantile(rel, 0.99))
    max_abs = float(err.max())
    idx = np.unravel_index(int(np.argmax(err)), err.shape)
    return {
        "pass": bool(p99_rel <= G2_IDENTITY_P99_REL_TOL and max_abs <= G2_IDENTITY_ABS_BACKSTOP),
        "p99_rel_floored": p99_rel,
        "max_abs": max_abs,
        "p99_abs": float(np.quantile(err, 0.99)),
        "argmax_abs_idx": [int(i) for i in idx],
        "scale_at_argmax": float(scale[idx]),
        "thresholds": {
            "p99_rel_tol": G2_IDENTITY_P99_REL_TOL,
            "rel_floor": G2_IDENTITY_REL_FLOOR,
            "abs_backstop": G2_IDENTITY_ABS_BACKSTOP,
        },
    }


def _g2_token_id_precheck(
    tokenizer,
    *,
    prefix_texts: list[str],
    prompts: list[str],
    completions: list[str],
    boundary: str,
    cell_id: str,
    row_labels: list[str],
) -> None:
    """Deterministic token-id pre-check (round-8.7; folded from the 8.6 diag).

    For every spot row, hard-asserts the CAPTURE's prompt-segment ids
    (`_capture_row_ids_and_positions`) are bit-identical to the REFERENCE's
    ``tokenizer(prompt)`` ids AND that ``context_end`` sits at the last
    reference id. Runs BEFORE the statistical identity comparison: it catches
    with exact precision the 1-2-row CORRELATED-shift mode the calibrated p99
    bar cannot (a single correlated adjacent-position row lands p99
    ~0.16-0.23), and any future regression of the capture's token/position
    construction (e.g. reintroduced concatenated-text tokenization, the
    round-8.4 class). Raises AssertionError with a first-divergence dump.
    """
    for spot_i, (pfx, prompt, completion, label) in enumerate(
        zip(prefix_texts, prompts, completions, row_labels, strict=True)
    ):
        ids_ref = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        row_ids, pos = _capture_row_ids_and_positions(tokenizer, pfx, prompt, completion, boundary)
        if row_ids[: len(ids_ref)] != ids_ref or pos["context_end"] != len(ids_ref) - 1:
            first_div = next(
                (j for j, (a, b) in enumerate(zip(row_ids, ids_ref, strict=False)) if a != b),
                None,
            )
            raise AssertionError(
                f"G2 token-id pre-check FAILED for {cell_id} spot {spot_i} (row {label!r}): "
                f"n_ids_ref={len(ids_ref)} n_row_ids={len(row_ids)} "
                f"context_end={pos['context_end']} first_divergence={first_div}"
            )
    logger.info("[G2] token-id pre-check PASSED for %s: %d spot rows exact", cell_id, len(prompts))


def _generate_context_hidden_reference(
    *,
    prompts: list[str],
    model,
    tokenizer,
    n_layers: int,
    hidden_dim: int,
    device: str,
) -> np.ndarray:
    """HF generate() hidden-state reference for the last prompt token."""
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    rows: list[np.ndarray] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_len = int(input_ids.shape[1])
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=pad_id,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )
        hidden_steps = getattr(generated, "hidden_states", None)
        if not hidden_steps:
            raise RuntimeError("G2 generate() did not return hidden_states")
        row = None
        for step in hidden_steps:
            layers = step[1:] if len(step) == n_layers + 1 else step[-n_layers:]
            if len(layers) != n_layers:
                continue
            if layers[0].ndim == 3 and layers[0].shape[1] >= prompt_len:
                row = np.stack(
                    [
                        layer[0, prompt_len - 1, :].to(torch.float16).cpu().numpy()
                        for layer in layers
                    ],
                    axis=0,
                )
                break
        if row is None:
            raise RuntimeError("G2 generate() hidden_states omitted the prompt prefill states")
        if row.shape != (n_layers, hidden_dim):
            raise ValueError(
                f"G2 generate reference shape {row.shape} != ({n_layers}, {hidden_dim})"
            )
        rows.append(row.astype(np.float32))
    return np.stack(rows, axis=0)


def _g2_pairing_null_band(X: np.ndarray, Y: np.ndarray, *, seed: int, n_draws: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    n = min(X.shape[0], Y.shape[0])
    if n < 8:
        raise ValueError(f"G2 pairing null needs at least 8 rows, got {n}")
    idx = rng.permutation(n)
    n_test = max(2, n // 5)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    Xtr_raw = X[train_idx].astype(np.float64)
    Xte_raw = X[test_idx].astype(np.float64)
    xmu = Xtr_raw.mean(axis=0, keepdims=True)
    xsd_raw = Xtr_raw.std(axis=0, keepdims=True)
    xsd = np.where(xsd_raw == 0.0, 1.0, xsd_raw)
    Xtr = (Xtr_raw - xmu) / xsd
    Xte = (Xte_raw - xmu) / xsd
    gram = Xtr.T @ Xtr + 1e6 * np.eye(Xtr.shape[1], dtype=np.float64)
    solved_xt = np.linalg.solve(gram, Xtr.T)
    vals: list[float] = []
    for _ in range(n_draws):
        perm = rng.permutation(n)
        Yp = Y[perm].astype(np.float64)
        Ytr = Yp[train_idx]
        Yte = Yp[test_idx]
        ymu = Ytr.mean(axis=0, keepdims=True)
        weights = solved_xt @ (Ytr - ymu)
        pred = Xte @ weights + ymu
        vals.append(_r2_for_arrays(Yte, pred))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n_draws": int(n_draws),
        "p05": float(np.nanpercentile(arr, 5)),
        "median": float(np.nanmedian(arr)),
        "p95": float(np.nanpercentile(arr, 95)),
        "draws": [float(v) for v in vals],
    }


def _r2_for_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean(axis=0, keepdims=True)) ** 2).sum())
    return float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot


def run_g2_gate_from_disk(  # noqa: C901
    out_dir: Path,
    cell_id: str,
    *,
    rows: list[dict],
    prefix_store: dict,
    query_store: dict,
    args: argparse.Namespace,
    rb_directions: np.ndarray | None,
) -> None:
    """Disk-backed G2 gate after the first full capture cell."""
    n_layers = args.n_layers
    hidden_dim = args.hidden_dim
    cell_dir = out_dir / "summaries" / cell_id
    if not cell_dir.exists():
        raise FileNotFoundError(f"G2 summaries missing for {cell_id}: {cell_dir}")
    row_counts: dict[str, int] = {}
    for kind in SUMMARY_KINDS:
        paths = _sorted_shards(sorted(cell_dir.glob(f"{kind}_L00_shard*.npy"))) or sorted(
            cell_dir.glob(f"{kind}_L00.npy")
        )
        if not paths:
            raise FileNotFoundError(f"G2 missing {kind}_L00 shards for {cell_id}")
        count = 0
        for path in paths:
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != hidden_dim:
                raise ValueError(f"G2 {path} shape {arr.shape} != (*, {hidden_dim})")
            count += int(arr.shape[0])
        row_counts[kind] = count
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"G2 row-count mismatch for {cell_id}: {row_counts}")

    if cell_id in CELLS_OWN_POLICY:
        pool_paths = _sorted_shards(
            sorted((out_dir / "summaries" / "b0_rB_pool").glob(f"{cell_id}_shard*.npy"))
        ) or sorted((out_dir / "summaries" / "b0_rB_pool").glob(f"{cell_id}.npy"))
        pool_count = 0
        for path in pool_paths:
            arr = np.load(path, mmap_mode="r")
            expected_tail = (n_layers, N_TRAITS, len(CAPTURE_POOLING_MODES))
            if arr.ndim != 4 or arr.shape[1:] != expected_tail:
                raise ValueError(f"G2 {path} shape {arr.shape} != (*, {expected_tail})")
            l14 = np.asarray(arr[:, min(14, n_layers - 1), :, 0])
            if not np.all(np.isfinite(l14)):
                raise ValueError(f"G2 non-finite B0 mean projections in {path}")
            pool_count += int(arr.shape[0])
        if pool_count != next(iter(row_counts.values())):
            raise ValueError(
                f"G2 B0 row-count mismatch for {cell_id}: pool={pool_count}, summaries={row_counts}"
            )

    cfg = CELL_CONFIG[cell_id]
    if cfg["model"] == "instruct":
        model_name, revision = INSTRUCT_MODEL, INSTRUCT_REVISION
    else:
        model_name, revision = PRETRAINED_MODEL, PRETRAINED_REVISION
    if cfg["prompt_format"] != cfg["model"] and cell_id == "cell_inst_own":
        raise AssertionError("cell_inst_own G2 expected matching instruct model/prompt format")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    _get_tokenizer._tok = tokenizer
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "torch_dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        "trust_remote_code": True,
    }
    if device.startswith("cuda"):
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()

    n_rows = next(iter(row_counts.values()))
    rng = np.random.default_rng(G2_SPOT_SEED)
    spot_idx = np.sort(rng.choice(n_rows, size=min(G2_SPOT_ROWS, n_rows), replace=False))
    cell_rows = _rows_for_cell(rows, cell_id)[:n_rows]
    own_completions = (
        _load_raw_completion_files(out_dir, cfg["model"], cell_id)
        if cfg["text_source"] == "own"
        else {}
    )
    prompts: list[str] = []
    prefix_texts: list[str] = []
    spot_completions: list[str] = []
    spot_row_labels: list[str] = []
    for row_i in spot_idx:
        row = cell_rows[int(row_i)]
        prefix_text, prompt, completion = render_row(
            row,
            prefix_store,
            query_store,
            prompt_format=cfg["prompt_format"],
            text_source=cfg["text_source"],
        )
        if completion is None:
            row_id = str(row.get("row_id"))
            if row_id not in own_completions:
                raise FileNotFoundError(
                    f"G2 token-id pre-check missing own-policy completion for row {row_id}"
                )
            completion = own_completions[row_id]
        prefix_texts.append(prefix_text)
        prompts.append(prompt)
        spot_completions.append(completion)
        spot_row_labels.append(str(row.get("row_id")))

    # round-8.7: deterministic token-id pre-check BEFORE the statistical
    # comparison (catches 1-2-row correlated shifts + any construction
    # regression with exact precision; see _g2_token_id_precheck).
    _g2_token_id_precheck(
        tokenizer,
        prefix_texts=prefix_texts,
        prompts=prompts,
        completions=spot_completions,
        boundary=_boundary_suffix(cfg["prompt_format"]),
        cell_id=cell_id,
        row_labels=spot_row_labels,
    )

    ref_context = _generate_context_hidden_reference(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=device,
    )
    disk_context = _load_cell_summary_rows(
        out_dir, cell_id, "context_end", spot_idx, n_layers=n_layers
    )
    identity = _g2_identity_check(disk_context, ref_context)
    if not identity["pass"]:
        raise AssertionError(
            f"G2 identity generate-reference mismatch for {cell_id}: {json.dumps(identity)}"
        )
    logger.info("[G2] identity check PASSED for %s: %s", cell_id, json.dumps(identity))

    l14 = min(14, n_layers - 1)
    X_l14 = _load_cell_kind_matrix(out_dir, cell_id, "context_end", l14)[:n_rows]
    Y_l14 = _load_cell_kind_matrix(out_dir, cell_id, "t1", l14)[:n_rows]
    null_band = _g2_pairing_null_band(X_l14, Y_l14, seed=G2_SPOT_SEED)
    if null_band["p05"] < -0.05 or null_band["p95"] > 0.05:
        raise AssertionError(f"G2 L14 pairing-null band outside [-0.05,+0.05]: {null_band}")

    if cell_id in CELLS_OWN_POLICY:
        if rb_directions is None:
            raise ValueError("G2 B0 recompute requires loaded r_B directions")
        b0_idx = spot_idx[: min(5, spot_idx.size)]
        # own-policy cells already loaded this map for the token-id pre-check
        completions = own_completions or _load_raw_completion_files(out_dir, cfg["model"], cell_id)
        missing = [
            str(cell_rows[int(i)].get("row_id"))
            for i in b0_idx
            if str(cell_rows[int(i)].get("row_id")) not in completions
        ]
        if missing:
            raise FileNotFoundError(f"G2 B0 recompute missing completions for rows {missing}")
        b0_prompts = [prompts[list(spot_idx).index(int(i))] for i in b0_idx]
        b0_prefixes = [prefix_texts[list(spot_idx).index(int(i))] for i in b0_idx]
        b0_completions = [completions[str(cell_rows[int(i)].get("row_id"))] for i in b0_idx]
        recomputed = _capture_batch_loaded_model(
            prefix_texts=b0_prefixes,
            prompts=b0_prompts,
            completions=b0_completions,
            prompt_format=cfg["prompt_format"],
            model=model,
            tokenizer=tokenizer,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            device=device,
            log_label="G2-b0-recompute",
            rb_directions=rb_directions,
            batch_size=max(1, min(CAPTURE_BATCH_SIZE, len(b0_idx))),
        )
        disk_b0 = _load_b0_pool_matrix(out_dir, cell_id)[b0_idx]
        if recomputed.rb_pool is None or not np.allclose(
            disk_b0, recomputed.rb_pool, atol=5e-2, rtol=5e-2
        ):
            delta = (
                float(np.max(np.abs(disk_b0 - recomputed.rb_pool)))
                if recomputed.rb_pool is not None
                else float("inf")
            )
            raise AssertionError(f"G2 B0 recompute mismatch for {cell_id}: max_abs={delta}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        "[G2] Gate PASSED for %s: rows=%s identity_spots=%d null_band=%s",
        cell_id,
        row_counts,
        len(spot_idx),
        null_band,
    )


def _g2_gate_worker(
    result_queue,
    out_dir: Path,
    cell_id: str,
    rows: list[dict],
    prefix_store: dict,
    query_store: dict,
    args: argparse.Namespace,
    rb_directions: np.ndarray | None,
) -> None:
    try:
        run_g2_gate_from_disk(
            out_dir,
            cell_id,
            rows=rows,
            prefix_store=prefix_store,
            query_store=query_store,
            args=args,
            rb_directions=rb_directions,
        )
    except Exception as exc:
        result_queue.put({"status": "error", "error": repr(exc)})
        raise
    result_queue.put({"status": "ok"})


def run_g2_gate_from_disk_isolated(
    out_dir: Path,
    cell_id: str,
    *,
    rows: list[dict],
    prefix_store: dict,
    query_store: dict,
    args: argparse.Namespace,
    rb_directions: np.ndarray | None,
) -> None:
    """Run G2 in a child process so its CUDA context dies at process exit."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_g2_gate_worker,
        args=(result_queue, out_dir, cell_id, rows, prefix_store, query_store, args, rb_directions),
        daemon=False,
        name=f"issue1092-g2-{cell_id}",
    )
    proc.start()
    try:
        result = result_queue.get(timeout=7200)
    except queue.Empty as exc:
        proc.terminate()
        proc.join(timeout=30)
        raise TimeoutError(f"G2 gate timed out for {cell_id}") from exc
    proc.join(timeout=120)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=30)
        raise RuntimeError(f"G2 gate process did not exit cleanly for {cell_id}")
    if proc.exitcode != 0 or result.get("status") != "ok":
        raise RuntimeError(
            f"G2 gate failed for {cell_id}: exitcode={proc.exitcode} error={result.get('error')}"
        )


def consolidate_cell_shards(out_dir: Path, cell_id: str, *, n_layers: int) -> None:
    """Collapse per-shard summary arrays into per-cell layer files before upload."""
    cell_dir = out_dir / "summaries" / cell_id
    if cell_dir.exists():
        for kind in SUMMARY_KINDS:
            for layer in range(n_layers):
                paths = _sorted_shards(list(cell_dir.glob(f"{kind}_L{layer:02d}_shard*.npy")))
                if len(paths) <= 1:
                    continue
                arrays = [np.load(path, mmap_mode="r") for path in paths]
                arr = np.concatenate(arrays, axis=0).astype(np.float16, copy=False)
                out_path = cell_dir / f"{kind}_L{layer:02d}.npy"
                np.save(out_path, arr)
                for path in paths:
                    path.unlink()
    pool_dir = out_dir / "summaries" / "b0_rB_pool"
    if pool_dir.exists():
        pool_paths = _sorted_shards(list(pool_dir.glob(f"{cell_id}_shard*.npy")))
        if len(pool_paths) > 1:
            arr = np.concatenate([np.load(path, mmap_mode="r") for path in pool_paths], axis=0)
            out_path = pool_dir / f"{cell_id}.npy"
            np.save(out_path, arr.astype(np.float32, copy=False))
            for path in pool_paths:
                path.unlink()


def _assert_cell_captures_on_disk(out_dir: Path, cell_id: str, *, n_layers: int) -> None:
    """Fail-loud completeness guard for --finalize-cells (round-8.9).

    A finalize-only cell was captured by a PRIOR invocation; assert every
    summary kind has per-layer arrays on disk (shard or consolidated) before
    consolidating/uploading, so a typo'd cell id or a half-written capture
    never silently uploads a partial store.
    """
    cell_dir = out_dir / "summaries" / cell_id
    if not cell_dir.is_dir():
        raise FileNotFoundError(f"--finalize-cells: no summaries dir for {cell_id}: {cell_dir}")
    for kind in SUMMARY_KINDS:
        for layer in range(n_layers):
            consolidated = cell_dir / f"{kind}_L{layer:02d}.npy"
            if consolidated.exists() or list(cell_dir.glob(f"{kind}_L{layer:02d}_shard*.npy")):
                continue
            raise FileNotFoundError(
                f"--finalize-cells: {cell_id} is missing {kind}_L{layer:02d} arrays "
                f"under {cell_dir} — the cell's capture did not complete on disk"
            )


def verify_uploaded_prefix(
    repo_id: str, repo_type: str, revision: str | None, path_in_repo: str
) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    entries = list(
        # HUB_VERIFY_RETRY_EXEMPT: issue-1092 driver; scoped listing with orchestration-layer retry
        api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=path_in_repo,
            recursive=True,
        )
    )
    if not any(getattr(entry, "size", None) is not None for entry in entries):
        raise RuntimeError(f"upload verification found no files under {repo_id}:{path_in_repo}")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_summaries_npy(
    out_dir: Path,
    cell_id: str,
    shard_idx: int,
    summaries_list: list[dict[str, np.ndarray]],
    n_layers: int,
    hidden_dim: int,
) -> dict[str, Path]:
    """Write per-kind summary arrays to npy files (fp16, no compression).

    Layout: <out>/summaries/<cell>/<kind>_L{ll}_shard{i:05d}.npy
    Each file: (n_rows, hidden_dim) fp16.
    """
    cell_dir = out_dir / "summaries" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for kind in SUMMARY_KINDS:
        for ll in range(n_layers):
            arr = np.stack(
                [s[kind][ll] for s in summaries_list],
                axis=0,
            )  # (n_rows, hidden_dim)
            arr = arr.astype(np.float16)
            path = cell_dir / f"{kind}_L{ll:02d}_shard{shard_idx:05d}.npy"
            np.save(str(path), arr)  # plain np.save, no compression (#813)
            paths[f"{kind}_L{ll:02d}"] = path

    return paths


def write_rb_pool_npy(
    out_dir: Path,
    cell_id: str,
    shard_idx: int,
    rb_pool: np.ndarray,
) -> Path:
    """Write r_B projection pool array. Shape: (n_rows, n_layers, n_traits, 4)."""
    pool_dir = out_dir / "summaries" / "b0_rB_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    path = pool_dir / f"{cell_id}_shard{shard_idx:05d}.npy"
    np.save(str(path), rb_pool.astype(np.float32))
    return path


def write_completions_jsonl(
    out_dir: Path,
    cell_id: str,
    shard_idx: int,
    model_type: str,
    rows: list[dict],
    completions: list[str],
) -> list[Path]:
    """Write raw completions JSONL, rotating shards below the upload line-size cap."""
    comp_dir = out_dir / "raw_completions" / model_type
    comp_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    part = 0
    bytes_in_part = 0
    handle = None

    def open_part(part_idx: int):
        path = comp_dir / f"{cell_id}_shard{shard_idx:05d}_part{part_idx:04d}.jsonl"
        paths.append(path)
        return path.open("w", encoding="utf-8")

    try:
        handle = open_part(part)
        for row, completion in zip(rows, completions, strict=True):
            line = (
                json.dumps(
                    {
                        "row_id": row.get("row_id", ""),
                        "prefix_id": row.get("prefix_id", ""),
                        "query_id": row.get("query_id", ""),
                        "cell_id": cell_id,
                        "completion": completion,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            line_bytes = len(line.encode("utf-8"))
            if line_bytes > MAX_JSONL_LINE_BYTES:
                raise ValueError(
                    f"completion row {row.get('row_id')} serializes to {line_bytes} bytes; "
                    f"line cap is {MAX_JSONL_LINE_BYTES}"
                )
            if bytes_in_part and bytes_in_part + line_bytes > MAX_JSONL_SHARD_BYTES:
                handle.close()
                part += 1
                bytes_in_part = 0
                handle = open_part(part)
            handle.write(line)
            bytes_in_part += line_bytes
    finally:
        if handle is not None:
            handle.close()
    return paths


def write_fingerprint(fp_path: Path, fp_dict: dict) -> None:
    """Write fingerprint JSON for resume idempotency."""
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(json.dumps(fp_dict, indent=2))


def _load_raw_completion_files(out_dir: Path, model_type: str, cell_id: str) -> dict[str, str]:
    """Load row_id -> completion from previously persisted own-policy rollouts."""
    comp_dir = out_dir / "raw_completions" / model_type
    paths = (
        _sorted_shards(list(comp_dir.glob(f"{cell_id}_shard*.jsonl"))) if comp_dir.exists() else []
    )
    out: dict[str, str] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                row_id = item.get("row_id")
                if row_id:
                    out[str(row_id)] = str(item.get("completion", ""))
    return out


def _load_claude_completions(out_dir: Path) -> dict[str, str]:
    """Load pair_id -> Claude completion from the P1 output layout."""
    comp_dir = out_dir / "raw_completions" / "claude"
    paths = sorted(comp_dir.glob("claude_completions*.jsonl")) if comp_dir.exists() else []
    out: dict[str, str] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("error"):
                    continue
                key = f"{item.get('prefix_id')}::{item.get('query_id')}"
                comp = item.get("completion")
                if comp is not None:
                    out[key] = str(comp)
    return out


def attach_completion_sources(rows: list[dict], corpus_dir: Path, out_dir: Path) -> None:
    """Attach completion text needed by non-own-policy cells to manifest rows."""
    instruct = _load_raw_completion_files(out_dir, "instruct", "cell_inst_own")
    pretrained = _load_raw_completion_files(out_dir, "pretrained", "cell_pre_own")
    claude = _load_claude_completions(out_dir)
    derangement_path = corpus_dir / "derangement_map.json"
    derangement = json.loads(derangement_path.read_text()) if derangement_path.exists() else {}

    missing: dict[str, int] = {"instruct": 0, "pretrained": 0, "claude": 0, "shuffled": 0}
    for row in rows:
        rid = str(row.get("row_id", ""))
        if rid in instruct:
            row["instruct_completion"] = instruct[rid]
        else:
            missing["instruct"] += 1
        if rid in pretrained:
            row["pretrained_completion"] = pretrained[rid]
        else:
            missing["pretrained"] += 1
        pair_key = f"{row.get('prefix_id')}::{row.get('query_id')}"
        if pair_key in claude:
            row["claude_text"] = claude[pair_key]
        elif row.get("control_subset") or row.get("claude_subset"):
            missing["claude"] += 1
        src_rid = derangement.get(rid)
        if src_rid and src_rid in instruct:
            row["shuffled_completion"] = instruct[src_rid]
        elif row.get("control_subset"):
            missing["shuffled"] += 1

    logger.info(
        "[completion-sources] loaded instruct=%d pretrained=%d claude=%d deranged=%d missing=%s",
        len(instruct),
        len(pretrained),
        len(claude),
        len(derangement),
        missing,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_cell_captures(
    out_dir: Path,
    cell_id: str,
    issue: int,
    slug: str = "issue1092_realistic_crossing",
) -> None:
    """Upload all artifacts for a cell to the HF data repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = HF_DATA_REPO

    # Upload summaries
    summaries_dir = out_dir / "summaries" / cell_id
    if summaries_dir.exists():
        path_in_repo = f"{SUMMARY_UPLOAD_PREFIX}/{cell_id}"
        # HUB_DIR_FILECOUNT_EXEMPT: issue-1092 driver; dirs bounded well under 10k files
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(summaries_dir),
            path_in_repo=path_in_repo,
            commit_message=f"issue{issue}: upload summaries for {cell_id}",
        )
        verify_uploaded_prefix(repo_id, "dataset", None, path_in_repo)
        logger.info("[upload] Summaries for %s uploaded", cell_id)

    # Upload r_B pool
    pool_dir = out_dir / "summaries" / "b0_rB_pool"
    pool_files = list(pool_dir.glob(f"{cell_id}*.npy")) if pool_dir.exists() else []
    if pool_files:
        path_in_repo = f"{SUMMARY_UPLOAD_PREFIX}/b0_rB_pool"
        # HUB_DIR_FILECOUNT_EXEMPT: issue-1092 driver; dirs bounded well under 10k files
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(pool_dir),
            path_in_repo=path_in_repo,
            allow_patterns=[f"{cell_id}*.npy"],
            commit_message=f"issue{issue}: upload r_B pool for {cell_id}",
        )
        verify_uploaded_prefix(repo_id, "dataset", None, path_in_repo)
        logger.info("[upload] r_B pool for %s uploaded (%d files)", cell_id, len(pool_files))

    # Upload completions
    for model_type in ("instruct", "pretrained"):
        comp_dir = out_dir / "raw_completions" / model_type
        if comp_dir.exists():
            comp_files = list(comp_dir.glob(f"{cell_id}_shard*.jsonl"))
            if comp_files:
                path_in_repo = f"{RAW_COMPLETIONS_UPLOAD_PREFIX}/{model_type}"
                # HUB_DIR_FILECOUNT_EXEMPT: issue-1092 driver; dirs bounded well under 10k files
                api.upload_folder(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=str(comp_dir),
                    path_in_repo=path_in_repo,
                    allow_patterns=[f"{cell_id}_shard*.jsonl"],
                    commit_message=(f"issue{issue}: upload {model_type} completions for {cell_id}"),
                )
                verify_uploaded_prefix(repo_id, "dataset", None, path_in_repo)
                logger.info(
                    "[upload] %s completions for %s uploaded (%d files)",
                    model_type,
                    cell_id,
                    len(comp_files),
                )


def upload_auxiliary_summaries(out_dir: Path, issue: int) -> None:
    """Upload P3b/P3c summaries using the same folder-level policy as cells."""
    from huggingface_hub import HfApi

    summaries_root = out_dir / "summaries"
    if not summaries_root.exists():
        return
    api = HfApi()
    for aux_dir in sorted(summaries_root.glob("bare_*")) + sorted(
        summaries_root.glob("dynamics_*")
    ):
        if not aux_dir.is_dir():
            continue
        if not any(aux_dir.glob("*.npy")):
            raise FileNotFoundError(f"auxiliary summary dir has no npy files: {aux_dir}")
        path_in_repo = f"{SUMMARY_UPLOAD_PREFIX}/{aux_dir.name}"
        # HUB_DIR_FILECOUNT_EXEMPT: issue-1092 driver; dirs bounded well under 10k files
        api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(aux_dir),
            path_in_repo=path_in_repo,
            commit_message=f"issue{issue}: upload auxiliary summaries for {aux_dir.name}",
        )
        verify_uploaded_prefix(HF_DATA_REPO, "dataset", None, path_in_repo)
        logger.info("[upload] auxiliary summaries uploaded: %s", aux_dir.name)


# ---------------------------------------------------------------------------
# Work-conserving dispatcher
# ---------------------------------------------------------------------------


def _maybe_attach_shard_sources(
    shard_rows: list[dict],
    args: argparse.Namespace,
    out_dir: Path,
    own_policy: bool,
    text_source: str,
) -> None:
    """Attach cross-text completion sources FRESH FROM DISK at shard time.

    Round-8.8: the worker's rows_by_cell is a spawn-time snapshot; under
    dependency scheduling the producer outputs are guaranteed on disk by the
    time a dependent shard is released, but they may POSTDATE the snapshot.
    shard_rows dicts are this process's own copies — mutation is local.
    """
    if not own_policy and text_source in ("instruct", "pretrained", "shuffled", "claude"):
        shard_corpus_dir = Path(args.corpus_dir) if args.corpus_dir else out_dir / "corpus"
        attach_completion_sources(shard_rows, shard_corpus_dir, out_dir)


def _process_shard(
    shard: Shard,
    rows_by_cell: dict[str, list[dict]],
    prefix_store: dict,
    query_store: dict,
    out_dir: Path,
    args: argparse.Namespace,
    gpu_id: int,
    rb_directions: np.ndarray | None,
    code_sha_val: str,
    corpus_hash: str,
    runtime: PersistentGpuRuntime | None = None,
) -> dict:
    """Process one shard on gpu_id. Returns result dict with status."""
    cell_id = shard.cell_id
    cfg = CELL_CONFIG[cell_id]
    model_type = cfg["model"]
    prompt_format = cfg["prompt_format"]
    text_source = cfg["text_source"]
    own_policy = cfg["own_policy"]

    cell_rows = rows_by_cell[cell_id]
    shard_rows = cell_rows[shard.row_start : shard.row_end]
    n_rows = len(shard_rows)

    if n_rows == 0:
        return {"status": "skip", "cell_id": cell_id, "shard_idx": shard.shard_idx, "n_rows": 0}

    logger.info(
        "[gpu=%d] Processing shard cell=%s rows=[%d:%d] shard=%d/%d",
        gpu_id,
        cell_id,
        shard.row_start,
        shard.row_end,
        shard.shard_idx,
        shard.total_shards,
    )

    # Fingerprint check
    fp_dict = compute_shard_fingerprint(
        corpus_hash=corpus_hash,
        cell_id=cell_id,
        row_start=shard.row_start,
        row_end=shard.row_end,
        model_id=INSTRUCT_MODEL if model_type == "instruct" else PRETRAINED_MODEL,
        dtype="bfloat16",
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        boundary_strings=(
            [_boundary_suffix(prompt_format)]
            + (STOP_TOKENS_INSTRUCT if model_type == "instruct" else STOP_TOKENS_PRETRAINED)
        ),
        rb_rev=args.rb_rev,
        max_gen_tokens=args.max_gen_tokens,
        subset_id=_subset_id_for_cell(cell_id),
        phase_name=getattr(args, "phase_name", "main"),
        code_sha=code_sha_val,
    )
    phase_label = _phase_fp_label(getattr(args, "phase_name", "main"))
    fp_path = (
        out_dir
        / "manifests"
        / f"{cell_id}_shard{shard.shard_idx:05d}_{phase_label}_fingerprint.json"
    )
    resumed, matched_prior_sha = _fingerprint_matches_any_sha(
        fp_path, fp_dict, getattr(args, "resume_code_shas", None)
    )
    if resumed:
        logger.info(
            "[gpu=%d] Shard %s/%d already complete (fingerprint match%s), skipping",
            gpu_id,
            cell_id,
            shard.shard_idx,
            f" via prior code_sha {matched_prior_sha}" if matched_prior_sha else "",
        )
        return {
            "status": "resumed",
            "cell_id": cell_id,
            "shard_idx": shard.shard_idx,
            "n_rows": n_rows,
        }

    _maybe_attach_shard_sources(shard_rows, args, out_dir, own_policy, text_source)

    # ---- Step 1: Get completions ----
    completions: list[str] = []
    prompts: list[str] = []
    prefix_texts: list[str] = []

    persisted_completions: dict[str, str] | None = None
    if own_policy and "capture" in args.phases_set and "gen" not in args.phases_set:
        persisted_completions = _load_raw_completion_files(out_dir, model_type, cell_id)
        missing = [
            str(row.get("row_id"))
            for row in shard_rows
            if str(row.get("row_id")) not in persisted_completions
        ]
        if missing:
            raise FileNotFoundError(
                f"{cell_id} capture-only shard {shard.shard_idx} is missing "
                f"{len(missing)} generated completions under "
                f"{out_dir / 'raw_completions' / model_type}; "
                f"examples={missing[:5]}"
            )

    for row in shard_rows:
        prefix_text, prompt, completion_text = render_row(
            row,
            prefix_store,
            query_store,
            prompt_format=prompt_format,
            text_source=text_source,
        )
        prefix_texts.append(prefix_text)
        prompts.append(prompt)
        if persisted_completions is not None:
            completions.append(persisted_completions[str(row.get("row_id"))])
        elif own_policy:
            completions.append("")  # will be filled by generation
        else:
            completions.append(completion_text or "")

    if own_policy and "gen" in args.phases_set:
        # Run vLLM generation
        if model_type == "instruct":
            model_name, revision = INSTRUCT_MODEL, INSTRUCT_REVISION
            stop_tokens = STOP_TOKENS_INSTRUCT
        else:
            model_name, revision = PRETRAINED_MODEL, PRETRAINED_REVISION
            stop_tokens = STOP_TOKENS_PRETRAINED

        if runtime is not None:
            completions = runtime.generate(
                prompts=prompts,
                model_name=model_name,
                revision=revision,
                stop_tokens=stop_tokens,
                max_tokens=args.max_gen_tokens,
                seed=GEN_SEED,
                chunk_size=DEFAULT_VLLM_CHUNK_SIZE,
            )
        else:
            completions = _run_gen_vllm(
                prompts=prompts,
                model_name=model_name,
                revision=revision,
                stop_tokens=stop_tokens,
                max_tokens=args.max_gen_tokens,
                seed=GEN_SEED,
                gpu_id=gpu_id,
                chunk_size=DEFAULT_VLLM_CHUNK_SIZE,
                engine_overrides=_vllm_engine_overrides(args),
            )

        # Write completions immediately (checkpoint per phase)
        write_completions_jsonl(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=shard.shard_idx,
            model_type=model_type,
            rows=shard_rows,
            completions=completions,
        )

    elif not own_policy and "gen" in args.phases_set:
        # Non-own-policy: write the pre-loaded completions
        write_completions_jsonl(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=shard.shard_idx,
            model_type=model_type,
            rows=shard_rows,
            completions=completions,
        )

    # ---- Step 2: Teacher-forced capture ----
    summaries_list: list[dict[str, np.ndarray]] = []
    if "capture" in args.phases_set:
        if model_type == "instruct":
            model_name, revision = INSTRUCT_MODEL, INSTRUCT_REVISION
        else:
            model_name, revision = PRETRAINED_MODEL, PRETRAINED_REVISION

        shard_rb = rb_directions if own_policy else None
        capture_out = (
            runtime.capture(
                prefix_texts=prefix_texts,
                prompts=prompts,
                completions=completions,
                prompt_format=prompt_format,
                model_name=model_name,
                revision=revision,
                n_layers=args.n_layers,
                hidden_dim=args.hidden_dim,
                rb_directions=shard_rb,
            )
            if runtime is not None
            else _capture_batch_hf(
                prefix_texts=prefix_texts,
                prompts=prompts,
                completions=completions,
                prompt_format=prompt_format,
                model_name=model_name,
                revision=revision,
                gpu_id=gpu_id,
                n_layers=args.n_layers,
                hidden_dim=args.hidden_dim,
            )
        )
        summaries_list = capture_out.summaries

        # Write summaries immediately
        write_summaries_npy(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=shard.shard_idx,
            summaries_list=summaries_list,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
        )

        # ---- Step 3: B0 r_B projection pooling (own-policy cells only) ----
        if capture_out.rb_pool is not None:
            write_rb_pool_npy(out_dir, cell_id, shard.shard_idx, capture_out.rb_pool)

    # Write fingerprint (marks shard as done)
    write_fingerprint(fp_path, fp_dict)

    return {
        "status": "done",
        "cell_id": cell_id,
        "shard_idx": shard.shard_idx,
        "n_rows": n_rows,
    }


def _subset_id_for_cell(cell_id: str) -> str:
    text_source = CELL_CONFIG[cell_id]["text_source"]
    if text_source == "claude":
        return "claude_subset"
    if text_source == "shuffled":
        return "control_subset"
    return "full_manifest"


def _load_derangement(corpus_dir: Path) -> dict[str, str]:
    """The shuffled-pairing derangement map (corpus artifact); fail-loud if absent."""
    path = Path(corpus_dir) / "derangement_map.json"
    if not path.exists():
        raise FileNotFoundError(
            f"derangement_map.json missing under {corpus_dir}; required to determine "
            "the shuffled cells' row domain"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_for_cell(
    rows: list[dict], cell_id: str, derangement: dict[str, str] | None = None
) -> list[dict]:
    subset_id = _subset_id_for_cell(cell_id)
    if subset_id == "full_manifest":
        return rows
    filtered = [row for row in rows if row.get(subset_id)]
    if CELL_CONFIG[cell_id]["text_source"] == "shuffled":
        # round-8.8: the shuffled pairing is DEFINED only on the derangement
        # domain (dense_core + periphery_random + periphery_natural — plan
        # §4.1 step 7); the control subset ALSO contains battery + topic-
        # matched periphery rows, which have no derangement entry and hence
        # no shuffled answer (launch 7: all 30 shuf-shard failures were the
        # first such row per shard). Membership needs BOTH the key (corpus
        # data, available before any generation) AND the answer-SOURCE row
        # being present in the current row set — a no-op at production
        # (sources always exist in the full manifest) that keeps row-limited
        # smokes consistent with the main() derangement closure.
        if derangement is None:
            raise ValueError(
                f"{cell_id} requires derangement_keys to determine its row domain "
                "(pass _load_derangement(corpus_dir))"
            )
        available_ids = {str(row.get("row_id")) for row in rows}
        filtered = [
            row
            for row in filtered
            if str(row.get("row_id")) in derangement
            and derangement[str(row.get("row_id"))] in available_ids
        ]
    if not filtered:
        raise ValueError(f"{cell_id} subset {subset_id} is empty; refusing to shard full manifest")
    return filtered


def _detect_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() and visible.strip() != "-1":
        return len([x for x in visible.split(",") if x.strip()])
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 0
    if proc.returncode != 0:
        return 0
    return len([line for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")])


def _worker_loop(
    gpu_id: int,
    shard_queue,
    result_queue,
    rows_by_cell: dict[str, list[dict]],
    prefix_store: dict,
    query_store: dict,
    out_dir: Path,
    args: argparse.Namespace,
    rb_directions: np.ndarray | None,
    code_sha_val: str,
    corpus_hash: str,
) -> None:
    """Worker process: set CVD, build one runtime, then pull shards."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runtime: PersistentGpuRuntime | None = None
    try:
        runtime = PersistentGpuRuntime(gpu_id, engine_overrides=_vllm_engine_overrides(args))
        while True:
            shard = shard_queue.get()
            if shard is None:
                break
            try:
                result = _process_shard(
                    shard=shard,
                    rows_by_cell=rows_by_cell,
                    prefix_store=prefix_store,
                    query_store=query_store,
                    out_dir=out_dir,
                    args=args,
                    gpu_id=gpu_id,
                    rb_directions=rb_directions,
                    code_sha_val=code_sha_val,
                    corpus_hash=corpus_hash,
                    runtime=runtime,
                )
            except Exception as exc:
                logger.exception("[gpu=%d] Shard %s failed", gpu_id, shard)
                result = {
                    "status": "error",
                    "cell_id": shard.cell_id,
                    "shard_idx": shard.shard_idx,
                    "error": str(exc),
                }
            result_queue.put(result)
    finally:
        if runtime is not None:
            runtime.close()


# ── cross-cell dependency scheduling (round-8.8) ─────────────────────────────
# Launch 7 post-mortem: the queue treated cells as independent, but cross-text
# cells CONSUME other cells' generation outputs. The edges are derived from
# the completion-source map (CELL_CONFIG text_source), never hardcoded lists.
_TEXT_SOURCE_PRODUCER: dict[str, str] = {
    "instruct": "cell_inst_own",  # cell_pre_insttext consumes inst_own gen
    "pretrained": "cell_pre_own",  # cell_inst_pretext consumes pre_own gen
    "shuffled": "cell_inst_own",  # shuf cells consume the derangement over inst_own answers
    # "claude": produced by P1 (off-pod) — readiness is the STAGED file,
    # validated up-front in _validate_dispatch_inputs, never a cell edge.
}

# Transient vLLM engine-core init failures (the launch-5 flake class): requeue
# the shard ONCE; a second failure — or any other error — is terminal.
TRANSIENT_INIT_ERROR_MARKERS = (
    "engine core initialization failed",
    "failed to start engine core",
    "enginecore initialization failed",
)


def _is_transient_init_error(error_text: str) -> bool:
    low = (error_text or "").lower()
    return any(marker in low for marker in TRANSIENT_INIT_ERROR_MARKERS)


def _cell_dependencies(cells: list[str]) -> dict[str, set[str]]:
    """cell -> producer cells whose FULL completion gates its readiness."""
    deps: dict[str, set[str]] = {}
    for cell_id in cells:
        producer = _TEXT_SOURCE_PRODUCER.get(CELL_CONFIG[cell_id]["text_source"])
        deps[cell_id] = {producer} if producer and producer != cell_id else set()
    return deps


class _DispatchScheduler:
    """Dependency-aware shard scheduler for the work-conserving dispatcher.

    Readiness: a cell's shards become READY only when every producer cell in
    its dependency set has completed ALL of its shards (a shard writes its
    gen completions to disk BEFORE capture, so full-cell completion implies
    the consumed outputs exist). A producer NOT in the dispatched cell list
    does not gate (the caller validates its outputs exist on disk up-front).
    Work-conserving WITHIN the ready set: every ready shard is released
    immediately; no wave barriers among ready cells. A transient engine-init
    failure requeues the shard ONCE (per-shard budget 1); everything else is
    terminal fail-fast. When a producer finishes with terminal failures, its
    dependents can never run — `blocked_dependents()` reports them so the
    dispatcher fails loud instead of waiting on results that cannot come.
    """

    def __init__(self, cells: list[str], shards: list[Shard], deps: dict[str, set[str]]):
        self._cells = list(cells)
        self._deps = deps
        self._pending: dict[str, list[Shard]] = {c: [] for c in cells}
        for shard in shards:
            self._pending[shard.cell_id].append(shard)
        self._total = {c: len(self._pending[c]) for c in cells}
        self._done: dict[str, int] = dict.fromkeys(cells, 0)
        self._terminal: dict[str, int] = dict.fromkeys(cells, 0)
        self._completed_cells: set[str] = {c for c in cells if self._total[c] == 0}
        self._failed_cells: set[str] = set()
        self._retried: set[tuple[str, int]] = set()
        self._shard_by_key = {(s.cell_id, s.shard_idx): s for s in shards}
        self.n_shards = len(shards)
        self.terminal_results = 0

    def _cell_ready(self, cell_id: str) -> bool:
        return all(
            dep in self._completed_cells or dep not in self._total
            for dep in self._deps.get(cell_id, set())
        )

    def _drain_ready(self) -> list[Shard]:
        released: list[Shard] = []
        for cell_id in self._cells:
            if self._pending[cell_id] and self._cell_ready(cell_id):
                released.extend(self._pending[cell_id])
                self._pending[cell_id] = []
        return released

    def initial_ready(self) -> list[Shard]:
        return self._drain_ready()

    def on_result(self, result: dict) -> tuple[list[Shard], Shard | None]:
        """Returns (newly_ready_shards, retry_shard). Retries are NOT terminal."""
        cell_id = result["cell_id"]
        shard_idx = result["shard_idx"]
        if result.get("status") == "error":
            key = (cell_id, shard_idx)
            if _is_transient_init_error(str(result.get("error", ""))) and key not in self._retried:
                self._retried.add(key)
                return [], self._shard_by_key[key]
            self.terminal_results += 1
            self._terminal[cell_id] += 1
        else:
            self.terminal_results += 1
            self._terminal[cell_id] += 1
            self._done[cell_id] += 1
            if self._done[cell_id] == self._total[cell_id]:
                self._completed_cells.add(cell_id)
                return self._drain_ready(), None
        if self._terminal[cell_id] == self._total[cell_id] and cell_id not in self._completed_cells:
            self._failed_cells.add(cell_id)
        return [], None

    def blocked_dependents(self) -> list[str]:
        return sorted(
            cell_id
            for cell_id in self._cells
            if self._pending[cell_id] and (self._deps.get(cell_id, set()) & self._failed_cells)
        )

    @property
    def finished(self) -> bool:
        return self.terminal_results >= self.n_shards


def _validate_dispatch_inputs(
    cells: list[str],
    out_dir: Path,
    corpus_dir: Path,
    *,
    gen_in_stage: bool = True,
) -> None:
    """Fail-loud AT READINESS-DERIVATION TIME, never mid-shard (round-8.8).

    - Claude cells: the P1 completions file must already be staged (launch 7:
      all 48 claude-shard failures were the unstaged file surfacing row by row).
    - Shuffled cells: the derangement map must exist.
    - A producer whose gen output is NOT produced within this stage — the
      producer is not dispatched, or the stage has no gen phase (capture-only)
      — must already have its completions on disk.
    """
    claude_cells = [c for c in cells if CELL_CONFIG[c]["text_source"] == "claude"]
    if claude_cells:
        claude_dir = out_dir / "raw_completions" / "claude"
        if not sorted(claude_dir.glob("claude_completions*.jsonl")):
            raise FileNotFoundError(
                f"claude cells {claude_cells} dispatched but no claude_completions*.jsonl "
                f"staged under {claude_dir}; stage the P1 output "
                "(hf: issue1092_realistic_crossing/raw_completions/claude/) first"
            )
    if any(CELL_CONFIG[c]["text_source"] == "shuffled" for c in cells):
        _load_derangement(corpus_dir)  # raises if missing
    dispatched = set(cells)
    for cell_id in cells:
        for producer in _cell_dependencies([cell_id])[cell_id]:
            if gen_in_stage and producer in dispatched:
                continue  # produced within this stage, gated by the scheduler edge
            producer_model = CELL_CONFIG[producer]["model"]
            producer_dir = out_dir / "raw_completions" / producer_model
            if not sorted(producer_dir.glob(f"{producer}_shard*.jsonl")):
                raise FileNotFoundError(
                    f"{cell_id} depends on {producer}, whose completions are not on disk "
                    f"under {producer_dir} and are not produced within this stage"
                )


def _make_dispatch_queues(ctx, cells: list[str], n_gpus: int, gen_in_stage: bool):
    """Model-affinity queues (round-8.8 memory safety).

    A GEN worker holds ONE vLLM engine for its whole lifetime (two engines at
    gpu_memory_utilization=0.85 cannot coexist, and in-process vLLM teardown
    is unreliable — CLAUDE.md gotcha). When a gen stage mixes both models,
    the GPUs split between per-model queues (work-conserving WITHIN each
    group; the instruct/pretrained cell sets are balanced 4/4 by design).
    Capture-only stages use one queue for all workers (both HF models coexist
    per worker). Returns (queues_by_model, worker_models, affinity).
    """
    models = sorted({CELL_CONFIG[c]["model"] for c in cells})
    affinity = gen_in_stage and len(models) > 1
    if affinity and n_gpus < 2:
        raise RuntimeError(
            "gen dispatch mixing instruct+pretrained models needs >=2 GPUs "
            "(one vLLM engine per worker lifetime); dispatch per-model cell groups instead"
        )
    if affinity:
        queues = {m: ctx.Queue() for m in models}
        worker_models = [
            models[0] if gpu_id < n_gpus // 2 else models[1] for gpu_id in range(n_gpus)
        ]
    else:
        single_queue = ctx.Queue()
        queues = dict.fromkeys(models, single_queue)
        worker_models = [models[0]] * n_gpus
    return queues, worker_models, affinity


def _build_cell_shards(
    cells: list[str], rows_by_cell: dict[str, list[dict]], n_gpus: int
) -> list[Shard]:
    """Per-cell row-range shards sized for the GPU count (unchanged sizing)."""
    max_cell_rows = max(len(cell_rows) for cell_rows in rows_by_cell.values())
    shard_size = max(1, min(512, max_cell_rows // max(1, n_gpus * 2) or max_cell_rows))
    shards: list[Shard] = []
    for cell_id in cells:
        n_rows = len(rows_by_cell[cell_id])
        cell_shards: list[Shard] = []
        for start in range(0, n_rows, shard_size):
            end = min(start + shard_size, n_rows)
            cell_shards.append(
                Shard(
                    cell_id=cell_id,
                    row_start=start,
                    row_end=end,
                    shard_idx=len(cell_shards),
                    total_shards=math.ceil(n_rows / shard_size),
                )
            )
        shards.extend(cell_shards)
    return shards


def run_dispatch(
    cells: list[str],
    rows: list[dict],
    prefix_store: dict,
    query_store: dict,
    out_dir: Path,
    args: argparse.Namespace,
    rb_directions: np.ndarray | None,
    corpus_hash: str,
) -> list[dict]:
    """Work-conserving multi-GPU dispatcher with cross-cell dependency edges.

    Creates one child process per GPU; workers pull shards from a shared
    queue. Shards are RELEASED to the queue by the `_DispatchScheduler` —
    immediately for cells with no (unsatisfied) producer edge, and the moment
    the producer cell completes otherwise (round-8.8; launch 7 failed when
    cross-text cells ran against not-yet-attached producer outputs). Returns
    the list of TERMINAL result dicts (a once-retried transient init failure
    is absorbed, not returned).
    """
    n_gpus = _detect_gpu_count()
    if n_gpus == 0:
        logger.error("No CUDA GPUs detected. Use --cpu-smoke for CPU-only testing.")
        raise RuntimeError("No CUDA GPUs available")

    logger.info("[dispatch] %d GPUs detected", n_gpus)

    gen_in_stage = "gen" in args.phases_set
    # Dependency edges gate readiness only when this stage PRODUCES the
    # consumed gen outputs; a capture-only stage validates them on disk
    # instead (no edges — captures are independent once completions exist).
    deps = _cell_dependencies(cells) if gen_in_stage else {c: set() for c in cells}
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else out_dir / "corpus"
    _validate_dispatch_inputs(cells, out_dir, corpus_dir, gen_in_stage=gen_in_stage)
    needs_derangement = any(CELL_CONFIG[c]["text_source"] == "shuffled" for c in cells)
    dmap = _load_derangement(corpus_dir) if needs_derangement else None

    # Build shard list
    rows_by_cell = {cell_id: _rows_for_cell(rows, cell_id, derangement=dmap) for cell_id in cells}
    shards = _build_cell_shards(cells, rows_by_cell, n_gpus)

    scheduler = _DispatchScheduler(cells, shards, deps)
    logger.info(
        "[dispatch] %d shards across %d cells; dependency edges: %s",
        len(shards),
        len(cells),
        {c: sorted(d) for c, d in deps.items() if d},
    )

    ctx = mp.get_context("spawn")
    queues, worker_models, affinity = _make_dispatch_queues(ctx, cells, n_gpus, gen_in_stage)

    def _enqueue(shard: Shard) -> None:
        queues[CELL_CONFIG[shard.cell_id]["model"]].put(shard)

    initial = scheduler.initial_ready()
    for shard in initial:
        _enqueue(shard)
    logger.info(
        "[dispatch] initial ready set: %d shards (%s); affinity=%s",
        len(initial),
        sorted({s.cell_id for s in initial}),
        {m: worker_models.count(m) for m in set(worker_models)} if affinity else "off",
    )

    result_queue = ctx.Queue()
    code_sha_val = code_sha()

    # Spawn worker processes. The child sets CUDA_VISIBLE_DEVICES before torch/vLLM imports.
    processes = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                gpu_id,
                queues[worker_models[gpu_id]],
                result_queue,
                rows_by_cell,
                prefix_store,
                query_store,
                out_dir,
                args,
                rb_directions,
                code_sha_val,
                corpus_hash,
            ),
            daemon=False,
            name=f"issue1092-worker-gpu-{gpu_id}",
        )
        p.start()
        processes.append(p)

    # Collect results. On timeout or parent-side failure, terminate children so
    # vLLM/HF engines do not survive past the dispatcher.
    results = []
    try:
        while not scheduler.finished:
            result = result_queue.get(timeout=3600)
            newly_ready, retry_shard = scheduler.on_result(result)
            if retry_shard is not None:
                logger.warning(
                    "[dispatch] transient engine-init failure on %s/%d - requeueing once: %s",
                    result["cell_id"],
                    result["shard_idx"],
                    result.get("error"),
                )
                _enqueue(retry_shard)
                continue  # absorbed; not a terminal result
            results.append(result)
            if result["status"] == "error":
                logger.error(
                    "[dispatch] Shard %s/%d ERROR: %s",
                    result["cell_id"],
                    result["shard_idx"],
                    result.get("error"),
                )
            else:
                logger.info(
                    "[dispatch] Shard %s/%d %s (%d rows)",
                    result["cell_id"],
                    result["shard_idx"],
                    result["status"],
                    result.get("n_rows", 0),
                )
            for shard in newly_ready:
                _enqueue(shard)
            if newly_ready:
                logger.info(
                    "[dispatch] released %d newly-ready shards (%s)",
                    len(newly_ready),
                    sorted({s.cell_id for s in newly_ready}),
                )
            blocked = scheduler.blocked_dependents()
            if blocked:
                raise RuntimeError(
                    f"[dispatch] producer cell(s) failed terminally; dependent cells can "
                    f"never become ready: {blocked}"
                )
    finally:
        # Sentinels release workers blocked on their queue (sent even on error).
        for gpu_id in range(n_gpus):
            queues[worker_models[gpu_id]].put(None)
        for p in processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join(timeout=30)
            if p.exitcode not in (0, None):
                raise RuntimeError(f"worker {p.name} exited with {p.exitcode}")

    return results


def _aux_fingerprint(
    *,
    corpus_hash: str,
    phase: str,
    model_type: str,
    row_start: int,
    row_end: int,
    n_layers: int,
    hidden_dim: int,
    phase_name: str,
    code_sha_val: str,
) -> dict:
    model_id = INSTRUCT_MODEL if model_type == "instruct" else PRETRAINED_MODEL
    return {
        "corpus_hash": corpus_hash,
        "phase": phase,
        "model_type": model_type,
        "row_start": row_start,
        "row_end": row_end,
        "model_id": model_id,
        "dtype": "bfloat16",
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "max_model_len": MAX_MODEL_LEN,
        "phase_name": phase_name,
        "code_sha": code_sha_val,
    }


def _aux_worker_loop(
    gpu_id: int,
    shard_queue,
    result_queue,
    bare_queries: list[dict],
    dynamics_prefixes: list[dict],
    out_dir: Path,
    args: argparse.Namespace,
    code_sha_val: str,
    corpus_hash: str,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded: dict[str, tuple[Any, Any, str]] = {}

    def get_model(model_type: str):
        if model_type not in loaded:
            if model_type == "instruct":
                model_name, revision = INSTRUCT_MODEL, INSTRUCT_REVISION
            elif model_type == "pretrained":
                model_name, revision = PRETRAINED_MODEL, PRETRAINED_REVISION
            else:
                raise ValueError(f"unknown aux model_type {model_type!r}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=revision,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map={"": "cuda:0"},
            )
            model.eval()
            loaded[model_type] = (tokenizer, model, "cuda:0")
        return loaded[model_type]

    try:
        while True:
            shard = shard_queue.get()
            if shard is None:
                break
            try:
                fp = _aux_fingerprint(
                    corpus_hash=corpus_hash,
                    phase=shard.phase,
                    model_type=shard.model_type,
                    row_start=shard.row_start,
                    row_end=shard.row_end,
                    n_layers=args.n_layers,
                    hidden_dim=args.hidden_dim,
                    phase_name=getattr(args, "phase_name", "aux"),
                    code_sha_val=code_sha_val,
                )
                fp_path = (
                    out_dir
                    / "manifests"
                    / (
                        f"{shard.phase}_{shard.model_type}_shard"
                        f"{shard.shard_idx:05d}_fingerprint.json"
                    )
                )
                n_rows = shard.row_end - shard.row_start
                if fingerprint_matches(fp_path, fp):
                    result_queue.put(
                        {
                            "status": "resumed",
                            "cell_id": f"{shard.phase}_{shard.model_type}",
                            "shard_idx": shard.shard_idx,
                            "n_rows": n_rows,
                        }
                    )
                    continue
                tokenizer, model, device = get_model(shard.model_type)
                if shard.phase == "bare":
                    run_bare_phase_loaded(
                        args=args,
                        queries=bare_queries[shard.row_start : shard.row_end],
                        tokenizer=tokenizer,
                        model=model,
                        model_type=shard.model_type,
                        device=device,
                        shard_idx=shard.shard_idx,
                    )
                elif shard.phase == "dynamics":
                    run_dynamics_phase_loaded(
                        args=args,
                        prefixes=dynamics_prefixes[shard.row_start : shard.row_end],
                        tokenizer=tokenizer,
                        model=model,
                        model_type=shard.model_type,
                        device=device,
                        shard_idx=shard.shard_idx,
                    )
                else:
                    raise ValueError(f"unknown aux phase {shard.phase!r}")
                write_fingerprint(fp_path, fp)
                result_queue.put(
                    {
                        "status": "done",
                        "cell_id": f"{shard.phase}_{shard.model_type}",
                        "shard_idx": shard.shard_idx,
                        "n_rows": n_rows,
                    }
                )
            except Exception as exc:
                logger.exception("[gpu=%d] Aux shard %s failed", gpu_id, shard)
                result_queue.put(
                    {
                        "status": "error",
                        "cell_id": f"{shard.phase}_{shard.model_type}",
                        "shard_idx": shard.shard_idx,
                        "error": str(exc),
                    }
                )
    finally:
        loaded.clear()
        gc.collect()
        torch.cuda.empty_cache()


def run_aux_dispatch(  # noqa: C901
    *,
    phases: set[str],
    query_store: dict[str, dict],
    prefix_store: dict[str, dict],
    out_dir: Path,
    args: argparse.Namespace,
    corpus_hash: str,
) -> list[dict]:
    n_gpus = _detect_gpu_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available for bare/dynamics DP phases")
    bare_queries = sorted(
        query_store.values(), key=lambda item: str(item.get("query_id", item.get("id", "")))
    )
    dynamics_prefixes = _dynamics_panel(prefix_store)
    if "dynamics" in phases:
        # round-8.9: filter on RENDERED length BEFORE any row-limit truncation
        # (production semantics: the panel is defined as conversations whose
        # renders fit the capture window; a tiny smoke still sees the drop).
        from transformers import AutoTokenizer

        tokenizers = {
            "instruct": _get_tokenizer(),
            "pretrained": AutoTokenizer.from_pretrained(
                PRETRAINED_MODEL, revision=PRETRAINED_REVISION, trust_remote_code=True
            ),
        }
        dynamics_prefixes, filter_digest = _filter_dynamics_panel_by_rendered_length(
            dynamics_prefixes, tokenizers
        )
        if not dynamics_prefixes:
            raise ValueError("dynamics panel is empty after rendered-length filtering")
        import datetime

        filter_digest["code_sha"] = code_sha()
        filter_digest["corpus_hash"] = corpus_hash
        filter_digest["timestamp"] = datetime.datetime.utcnow().isoformat()
        logger.info(
            "[dynamics-panel-filter] kept=%d dropped=%d of %d over the %d-token "
            "rendered window; dropped=%s",
            filter_digest["n_kept"],
            filter_digest["n_dropped"],
            filter_digest["n_panel"],
            filter_digest["max_tokens"],
            [d["conv_id"] for d in filter_digest["dropped"]],
        )
        for model_type in ("instruct", "pretrained"):
            dyn_dir = out_dir / "summaries" / f"dynamics_{model_type}"
            dyn_dir.mkdir(parents=True, exist_ok=True)
            (dyn_dir / "panel_filter.json").write_text(json.dumps(filter_digest, indent=2))
    if args.row_limit is not None:
        bare_queries = bare_queries[: args.row_limit]
        dynamics_prefixes = dynamics_prefixes[: args.row_limit]
    shards: list[AuxShard] = []
    for phase, items in (("bare", bare_queries), ("dynamics", dynamics_prefixes)):
        if phase not in phases:
            continue
        if not items:
            raise ValueError(f"{phase} phase has no rows after filtering")
        shard_size = max(1, min(512, len(items) // max(1, n_gpus * 2) or len(items)))
        for model_type in ("instruct", "pretrained"):
            total = math.ceil(len(items) / shard_size)
            for shard_idx, start in enumerate(range(0, len(items), shard_size)):
                shards.append(
                    AuxShard(
                        phase=phase,
                        model_type=model_type,
                        row_start=start,
                        row_end=min(start + shard_size, len(items)),
                        shard_idx=shard_idx,
                        total_shards=total,
                    )
                )
    if not shards:
        return []
    logger.info("[aux-dispatch] %d shards across %d GPUs", len(shards), n_gpus)
    ctx = mp.get_context("spawn")
    shard_queue = ctx.Queue()
    for shard in shards:
        shard_queue.put(shard)
    for _ in range(n_gpus):
        shard_queue.put(None)
    result_queue = ctx.Queue()
    code_sha_val = code_sha()
    processes = []
    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_aux_worker_loop,
            args=(
                gpu_id,
                shard_queue,
                result_queue,
                bare_queries,
                dynamics_prefixes,
                out_dir,
                args,
                code_sha_val,
                corpus_hash,
            ),
            daemon=False,
            name=f"issue1092-aux-gpu-{gpu_id}",
        )
        p.start()
        processes.append(p)
    results: list[dict] = []
    try:
        while len(results) < len(shards):
            result = result_queue.get(timeout=3600)
            results.append(result)
            logger.info(
                "[aux-dispatch] %s shard%d %s (%d rows)",
                result["cell_id"],
                result["shard_idx"],
                result["status"],
                result.get("n_rows", 0),
            )
    finally:
        for p in processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
                p.join(timeout=30)
            if p.exitcode not in (0, None):
                raise RuntimeError(f"aux worker {p.name} exited with {p.exitcode}")
    return results


# ---------------------------------------------------------------------------
# CPU smoke mode
# ---------------------------------------------------------------------------


def run_cpu_smoke(args: argparse.Namespace) -> None:
    """CPU smoke: construct synthetic tiny outputs, write store layout, exit 0.

    Used for GPU-bound carve-out smoke (3-item coverage item 2: dispatcher dry-run).
    """
    logger.info("[cpu-smoke] Starting CPU smoke mode")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells = args.cells
    n_rows = max(1, getattr(args, "row_limit", 3) or 3)

    # Synthetic summaries
    rng = np.random.default_rng(42)
    for cell_id in cells:
        summaries_list = []
        for _ in range(n_rows):
            s = {}
            for kind in SUMMARY_KINDS:
                s[kind] = rng.standard_normal((args.n_layers, args.hidden_dim)).astype(np.float16)
            summaries_list.append(s)

        write_summaries_npy(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=0,
            summaries_list=summaries_list,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
        )

        # Synthetic completions
        for model_type in ("instruct", "pretrained"):
            write_completions_jsonl(
                out_dir=out_dir,
                cell_id=cell_id,
                shard_idx=0,
                model_type=model_type,
                rows=[
                    {"row_id": str(i), "prefix_id": "p0", "query_id": "q0"} for i in range(n_rows)
                ],
                completions=[f"synthetic-completion-{i}" for i in range(n_rows)],
            )

        # Synthetic r_B pool for own-policy cells
        if cell_id in CELLS_OWN_POLICY:
            rb_pool = rng.standard_normal(
                (n_rows, args.n_layers, N_TRAITS, len(CAPTURE_POOLING_MODES))
            ).astype(np.float32)
            write_rb_pool_npy(out_dir, cell_id, 0, rb_pool)

    # Write sentinel
    _write_sentinel(args, phase="done", note="cpu-smoke")

    # Compute artifact digest
    summary_files = list((out_dir / "summaries").rglob("*.npy"))
    comp_files = list((out_dir / "raw_completions").rglob("*.jsonl"))
    logger.info(
        "[cpu-smoke] DONE: %d summary files, %d completion files",
        len(summary_files),
        len(comp_files),
    )
    print(
        f"[cpu-smoke] artifact digest: {len(summary_files)} npy, "
        f"{len(comp_files)} jsonl, cells={cells}, n_rows={n_rows}"
    )
    print("[phase=done]")  # noqa: phase-done-reserved


def _load_rows_for_smoke(args: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    if args.corpus_dir is None:
        raise ValueError(
            "--backend hf-cpu requires --corpus-dir from a P0 smoke or production build"
        )
    rows = load_manifest(args.corpus_dir)
    if args.row_limit:
        rows = rows[: args.row_limit]
    if not rows:
        raise ValueError(f"no manifest rows found under {args.corpus_dir}")
    prefix_store = load_store(args.corpus_dir, "prefix_store.jsonl")
    query_store = load_store(args.corpus_dir, "query_store.jsonl")
    return rows, prefix_store, query_store


def _tiny_qwen2_model(tokenizer, *, n_layers: int, hidden_dim: int):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    n_heads = 2 if hidden_dim % 2 == 0 and hidden_dim >= 2 else 1
    cfg = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=hidden_dim,
        intermediate_size=max(32, hidden_dim * 4),
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        use_cache=True,
    )
    torch.manual_seed(GEN_SEED)
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def _generate_hf_cpu(prompts: list[str], tokenizer, model, max_new_tokens: int) -> list[str]:
    import torch

    completions: list[str] = []
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max(1, max_new_tokens),
                do_sample=False,
                pad_token_id=pad_id,
            )
        new_ids = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        completions.append(text if text else ".")
    return completions


def _render_bare_query(query: str, model_type: str) -> str:
    if model_type == "instruct":
        tok = _get_tokenizer()
        return tok.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if model_type == "pretrained":
        return f"User: {query}\n\nAssistant:"
    raise ValueError(f"unknown model_type {model_type!r}")


def _render_full_conversation(turns: list[dict], model_type: str) -> str:
    if model_type == "instruct":
        tok = _get_tokenizer()
        return tok.apply_chat_template(
            [{"role": t["role"], "content": t["content"]} for t in turns],
            tokenize=False,
            add_generation_prompt=False,
        )
    if model_type == "pretrained":
        lines = []
        for turn in turns:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
            lines.append("")
        return "\n".join(lines).rstrip()
    raise ValueError(f"unknown model_type {model_type!r}")


def _tokenize_full_render_with_offsets(
    tokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except NotImplementedError as exc:
        raise RuntimeError("dynamics cut planning requires a fast tokenizer with offsets") from exc
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if offsets and isinstance(offsets[0], list):
        offsets = offsets[0]
    return [int(x) for x in ids], [(int(start), int(end)) for start, end in offsets]


def _turn_content_char_spans(
    full_render: str, turns: list[dict], model_type: str
) -> tuple[list[tuple[int, int]], list[int]]:
    """Char spans of each turn's content in the full render, false-match hardened.

    Pretrained renders are deterministic concatenation, so content spans are
    computed positionally and verified by a verbatim-slice assert (no search at
    all). Instruct (chat-template) spans anchor the search at the end of the
    prior turns' rendered prefix and bound the match inside the turn's own
    rendered prefix — content text that also occurs in earlier turns or in
    scaffold (e.g. a content string equal to a role header) can no longer be
    matched there; a miss raises instead of silently rescanning from 0.
    """
    content_spans: list[tuple[int, int]] = []
    turn_end_chars: list[int] = []
    cursor = 0
    pos = 0  # pretrained-only cumulative offset over the deterministic render
    prev_prefix = ""
    for turn_idx, turn in enumerate(turns):
        content = str(turn.get("content", ""))
        prefix = _render_full_conversation(turns[: turn_idx + 1], model_type)
        if model_type == "pretrained":
            role = "User" if turn["role"] == "user" else "Assistant"
            line_start = pos
            pos = line_start + len(role) + 2 + len(content) + 2  # "\n\n" joins turn lines
            if content:
                start = line_start + len(role) + 2
                end = start + len(content)
                # round-8.8: `_render_full_conversation("pretrained")` ends with
                # `.rstrip()`, so the FINAL turn's own trailing whitespace is
                # stripped along with the terminal separator; clamp that turn's
                # span to the realized render (a real WildChat final turn whose
                # content ends "\n" crashed the exact-length assert here).
                if turn_idx == len(turns) - 1 and end > len(full_render):
                    clipped = full_render[start:]
                    if clipped != content[: len(clipped)] or clipped.rstrip() != content.rstrip():
                        raise AssertionError(
                            f"dynamics content span mismatch for pretrained LAST turn "
                            f"{turn_idx}: render tail {len(clipped)} chars != content "
                            f"{len(content)} chars beyond trailing-whitespace stripping"
                        )
                    end = len(full_render)
                elif full_render[start:end] != content:
                    raise AssertionError(
                        f"dynamics content span mismatch for pretrained turn {turn_idx}: "
                        f"{content!r}"
                    )
                cursor = end
            else:
                end = len(prefix) if full_render.startswith(prefix) else cursor
                start = end
        elif content:
            search_start = cursor
            if turn_idx and full_render.startswith(prev_prefix):
                search_start = max(cursor, len(prev_prefix))
            start = full_render.find(content, search_start)
            if start < 0:
                raise AssertionError(
                    f"dynamics content span not found for {model_type} turn {turn_idx}: {content!r}"
                )
            end = start + len(content)
            if full_render.startswith(prefix) and end > len(prefix):
                raise AssertionError(
                    f"dynamics content span for {model_type} turn {turn_idx} escapes its "
                    f"turn render: end {end} > turn prefix end {len(prefix)}"
                )
            cursor = end
        else:
            end = len(prefix) if full_render.startswith(prefix) else cursor
            start = end
        content_spans.append((start, end))
        if full_render.startswith(prefix):
            turn_end_chars.append(len(prefix))
        else:
            turn_end_chars.append(end)
        prev_prefix = prefix
    return content_spans, turn_end_chars


def _char_span_to_token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int, n_total_tokens: int
) -> tuple[int, int]:
    selected = [
        i
        for i, (start, end) in enumerate(offsets[:n_total_tokens])
        if end > start and end > char_start and start < char_end
    ]
    if selected:
        return selected[0], selected[-1] + 1
    boundary = _char_prefix_to_token_end(offsets, char_start, n_total_tokens)
    start = min(max(0, boundary), max(0, n_total_tokens - 1))
    return start, min(n_total_tokens, start + 1)


def _char_prefix_to_token_end(
    offsets: list[tuple[int, int]], char_end: int, n_total_tokens: int
) -> int:
    selected = [
        i
        for i, (start, end) in enumerate(offsets[:n_total_tokens])
        if end > start and start < char_end
    ]
    if selected:
        return min(n_total_tokens, selected[-1] + 1)
    for i, (start, end) in enumerate(offsets[:n_total_tokens]):
        if end > start and end >= char_end:
            return min(n_total_tokens, i + 1)
    return n_total_tokens


def _assert_token_span_covers_char_span(
    *,
    offsets: list[tuple[int, int]],
    token_start: int,
    token_end: int,
    char_start: int,
    char_end: int,
    model_type: str,
    turn_idx: int,
    role: str,
) -> None:
    if char_end <= char_start:
        return
    cursor = char_start
    for start, end in sorted(offsets[token_start:token_end]):
        if end <= start or end <= char_start or start >= char_end:
            continue
        if start > cursor:
            raise AssertionError(
                f"dynamics {role}-span offset gap for {model_type} turn {turn_idx}: "
                f"tokens=({token_start},{token_end}) chars=({char_start},{char_end}) "
                f"gap_at={cursor}"
            )
        cursor = max(cursor, min(end, char_end))
        if cursor >= char_end:
            return
    raise AssertionError(
        f"dynamics {role}-span offset coverage failed for {model_type} turn {turn_idx}: "
        f"tokens=({token_start},{token_end}) chars=({char_start},{char_end}) "
        f"covered_until={cursor}"
    )


def _dynamics_cut_plan(
    turns: list[dict],
    tokenizer,
    model_type: str,
    n_total_tokens: int,
    full_token_ids: list[int] | None = None,
) -> dict[str, list[tuple[int, int, int]]]:
    """Return per-kind (start, end, turn_index) cuts for one full conversation.

    The positions are computed from a single tokenization of the full render so
    BPE seams at role/content boundaries cannot invalidate partial-render math.
    """
    cuts: dict[str, list[tuple[int, int, int]]] = {
        "context_k": [],
        "s_k": [],
        "answer_k_t1": [],
        "answer_k_t2": [],
        "answer_k_t3": [],
        "u1": [],
        "u2": [],
        "u3": [],
    }
    full_render = _render_full_conversation(turns, model_type)
    encoded_ids, offsets = _tokenize_full_render_with_offsets(tokenizer, full_render)
    if len(encoded_ids) != n_total_tokens:
        raise AssertionError(
            f"dynamics full-render token count mismatch for {model_type}: "
            f"offset_tokens={len(encoded_ids)} forward_tokens={n_total_tokens}"
        )
    if full_token_ids is not None and list(full_token_ids) != encoded_ids:
        raise AssertionError(f"dynamics full-render token ids mismatch for {model_type}")
    content_spans, turn_end_chars = _turn_content_char_spans(full_render, turns, model_type)

    def clamp_span(start: int, end: int) -> tuple[int, int]:
        start = min(max(0, start), max(0, n_total_tokens - 1))
        end = min(max(start + 1, end), n_total_tokens)
        return start, end

    for turn_idx, turn in enumerate(turns):
        role = turn.get("role")
        content_start, content_end = content_spans[turn_idx]
        turn_token_end = _char_prefix_to_token_end(
            offsets, turn_end_chars[turn_idx], n_total_tokens
        )
        if role == "assistant":
            answer_start, answer_end = _char_span_to_token_span(
                offsets, content_start, content_end, n_total_tokens
            )
            answer_start, answer_end = clamp_span(answer_start, answer_end)
            _assert_token_span_covers_char_span(
                offsets=offsets,
                token_start=answer_start,
                token_end=answer_end,
                char_start=content_start,
                char_end=content_end,
                model_type=model_type,
                turn_idx=turn_idx,
                role="assistant",
            )
            _t2_start, t2_end = clamp_span(answer_start, max(answer_end, turn_token_end - 1))
            t3_pos = min(max(0, turn_token_end - 1), max(0, n_total_tokens - 1))
            context_pos = max(0, answer_start - 1)
            s_pos = max(answer_start, answer_end - 1)
            cuts["context_k"].append((context_pos, context_pos + 1, turn_idx))
            cuts["s_k"].append((s_pos, s_pos + 1, turn_idx))
            cuts["answer_k_t1"].append((answer_start, answer_end, turn_idx))
            cuts["answer_k_t2"].append((answer_start, t2_end, turn_idx))
            cuts["answer_k_t3"].append((t3_pos, t3_pos + 1, turn_idx))
        elif role == "user":
            user_start, user_end = _char_span_to_token_span(
                offsets, content_start, content_end, n_total_tokens
            )
            user_start, user_end = clamp_span(user_start, user_end)
            _assert_token_span_covers_char_span(
                offsets=offsets,
                token_start=user_start,
                token_end=user_end,
                char_start=content_start,
                char_end=content_end,
                model_type=model_type,
                turn_idx=turn_idx,
                role="user",
            )
            _u2_start, u2_end = clamp_span(user_start, max(user_end, turn_token_end))
            u3_pos = min(max(0, turn_token_end - 1), max(0, n_total_tokens - 1))
            cuts["u1"].append((user_start, user_end, turn_idx))
            cuts["u2"].append((user_start, u2_end, turn_idx))
            cuts["u3"].append((u3_pos, u3_pos + 1, turn_idx))
    return cuts


def _capture_prompt_states_loaded_model(
    *,
    prompts: list[str],
    model,
    tokenizer,
    n_layers: int,
    hidden_dim: int,
    device: str,
    log_label: str,
    batch_size: int = CAPTURE_BATCH_SIZE,
) -> np.ndarray:
    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[np.ndarray] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        end = min(start + max(1, batch_size), len(prompts))
        batch = prompts[start:end]
        token_counts = [_token_len(tokenizer, prompt) for prompt in batch]
        for i, n_tok in enumerate(token_counts):
            if n_tok > MAX_MODEL_LEN:
                raise ValueError(
                    f"{log_label} prompt row {start + i} has {n_tok} tokens > {MAX_MODEL_LEN}"
                )
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        with torch.no_grad():
            outputs = _call_model_with_hidden_states(
                model,
                inputs["input_ids"].to(device),
                inputs["attention_mask"].to(device),
            )
        hidden_states = outputs.hidden_states[1:]
        if len(hidden_states) != n_layers:
            raise ValueError(f"model returned {len(hidden_states)} layers, expected {n_layers}")
        if hidden_states[0].shape[-1] != hidden_dim:
            raise ValueError(
                f"model hidden dim {hidden_states[0].shape[-1]} != expected {hidden_dim}"
            )
        for local_i, n_tok in enumerate(token_counts):
            pos = max(0, n_tok - 1)
            rows.append(
                np.stack(
                    [hs[local_i, pos, :].to(torch.float16).cpu().numpy() for hs in hidden_states],
                    axis=0,
                )
            )
    return (
        np.stack(rows, axis=0).astype(np.float16)
        if rows
        else np.empty((0, n_layers, hidden_dim), dtype=np.float16)
    )


def _capture_dynamics_loaded_model(
    *,
    prompts: list[str],
    turns_by_prompt: list[list[dict]],
    conv_ids: list[str],
    model,
    tokenizer,
    model_type: str,
    n_layers: int,
    hidden_dim: int,
    device: str,
    batch_size: int = CAPTURE_BATCH_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict]]]:
    import torch

    if len({len(prompts), len(turns_by_prompt), len(conv_ids)}) != 1:
        raise ValueError("prompts, turns_by_prompt, and conv_ids must have equal length")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    states: dict[str, list[np.ndarray]] = {
        "context_k": [],
        "s_k": [],
        "answer_k_t1": [],
        "answer_k_t2": [],
        "answer_k_t3": [],
        "u1": [],
        "u2": [],
        "u3": [],
    }
    index_rows: dict[str, list[dict]] = {kind: [] for kind in states}

    for start in range(0, len(prompts), max(1, batch_size)):
        end = min(start + max(1, batch_size), len(prompts))
        batch = prompts[start:end]
        token_counts = [_token_len(tokenizer, prompt) for prompt in batch]
        for local_i, n_tok in enumerate(token_counts):
            if n_tok > MAX_MODEL_LEN:
                raise ValueError(
                    f"dynamics-{model_type} conversation {conv_ids[start + local_i]} "
                    f"has {n_tok} tokens > {MAX_MODEL_LEN}"
                )
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        with torch.no_grad():
            outputs = _call_model_with_hidden_states(
                model,
                inputs["input_ids"].to(device),
                inputs["attention_mask"].to(device),
            )
        hidden_states = outputs.hidden_states[1:]
        if len(hidden_states) != n_layers:
            raise ValueError(f"model returned {len(hidden_states)} layers, expected {n_layers}")
        if hidden_states[0].shape[-1] != hidden_dim:
            raise ValueError(
                f"model hidden dim {hidden_states[0].shape[-1]} != expected {hidden_dim}"
            )

        for local_i, n_tok in enumerate(token_counts):
            conv_id = conv_ids[start + local_i]
            full_ids = inputs["input_ids"][local_i, :n_tok].detach().cpu().tolist()
            cuts = _dynamics_cut_plan(
                turns_by_prompt[start + local_i],
                tokenizer,
                model_type,
                n_tok,
                full_token_ids=full_ids,
            )
            for kind, spans in cuts.items():
                for cut_start, cut_end, turn_idx in spans:
                    arr = np.stack(
                        [
                            hs[local_i, cut_start:cut_end, :]
                            .mean(dim=0)
                            .to(torch.float16)
                            .cpu()
                            .numpy()
                            for hs in hidden_states
                        ],
                        axis=0,
                    )
                    states[kind].append(arr)
                    index_rows[kind].append(
                        {
                            "conv_id": conv_id,
                            "turn_index": turn_idx,
                            "kind": kind,
                            "token_start": cut_start,
                            "token_end": cut_end,
                        }
                    )
        del outputs, hidden_states, inputs

    arrays = {
        kind: np.stack(vals, axis=0).astype(np.float16)
        if vals
        else np.empty((0, n_layers, hidden_dim), dtype=np.float16)
        for kind, vals in states.items()
    }
    return arrays, index_rows


def _write_layer_stack(
    root: Path, kind: str, states: np.ndarray, shard_idx: int | None = None
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if states.ndim != 3:
        raise ValueError(f"{kind} states must be (n,L,H), got {states.shape}")
    suffix = "" if shard_idx is None else f"_shard{shard_idx:05d}"
    for layer in range(states.shape[1]):
        np.save(root / f"{kind}_L{layer:02d}{suffix}.npy", states[:, layer, :].astype(np.float16))


def _write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dynamics_panel(prefix_store: dict[str, dict]) -> list[dict]:
    """Plan P3c panel: logged multi-turn conversations only, never battery/trait rows."""
    panel: list[dict] = []
    for item in prefix_store.values():
        prefix_id = str(item.get("prefix_id") or item.get("id") or "")
        if not prefix_id.startswith("pfx_"):
            continue
        turns = _prefix_turns(item)
        roles = [turn.get("role") for turn in turns]
        if len(turns) < 2 or "user" not in roles or "assistant" not in roles:
            continue
        panel.append(item)
    panel.sort(key=lambda item: str(item.get("prefix_id", item.get("id", ""))))
    if not panel:
        raise ValueError("dynamics panel is empty after filtering to logged multi-turn pfx_* rows")
    if not all(len(_prefix_turns(item)) >= 2 for item in panel):
        raise AssertionError("dynamics panel contains a non-multi-turn item")
    return panel


def _filter_dynamics_panel_by_rendered_length(
    panel: list[dict],
    tokenizers: dict[str, Any],
    *,
    max_tokens: int = MAX_MODEL_LEN,
) -> tuple[list[dict], dict]:
    """Drop conversations whose instruct OR pretrained RENDER overflows the capture window.

    P0's conversation keep-filter capped RAW content tokens at the window size,
    but the instruct chat template adds per-turn scaffold, so a near-cap
    conversation renders past the 8192-token capture window (launch 8:
    wildchat_064122 at 8267 > 8192 in the instruct render). Pair-drop: one
    shared panel feeds both dynamics arms, so a conversation over-budget in
    EITHER render is dropped from BOTH (row-set alignment). Token counts use
    each arm's own tokenizer on the exact render the capture path asserts on
    (`_token_len`, mirroring `_capture_dynamics_loaded_model`'s check).
    Returns (kept_panel, digest); the caller writes the digest into the
    dynamics output dirs so the analyzer sees the coverage note.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for item in panel:
        turns = _prefix_turns(item)
        conv_id = str(item.get("conv_id") or item.get("prefix_id") or item.get("id"))
        counts = {
            model_type: _token_len(tokenizer, _render_full_conversation(turns, model_type))
            for model_type, tokenizer in tokenizers.items()
        }
        if any(n > max_tokens for n in counts.values()):
            dropped.append(
                {"conv_id": conv_id} | {f"{k}_tokens": v for k, v in sorted(counts.items())}
            )
        else:
            kept.append(item)
    digest = {
        "n_panel": len(panel),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "max_tokens": max_tokens,
        "dropped": dropped,
    }
    return kept, digest


def run_bare_phase_loaded(
    *,
    args: argparse.Namespace,
    queries: list[dict],
    tokenizer,
    model,
    model_type: str,
    device: str,
    shard_idx: int | None = None,
) -> None:
    prompts = [_render_bare_query(_query_text(item), model_type) for item in queries]
    states = _capture_prompt_states_loaded_model(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        device=device,
        log_label=f"bare-{model_type}",
    )
    out_root = Path(args.out) / "summaries" / f"bare_{model_type}"
    _write_layer_stack(out_root, "c_q_bare", states, shard_idx=shard_idx)
    index_name = "row_index.jsonl" if shard_idx is None else f"row_index_shard{shard_idx:05d}.jsonl"
    _write_jsonl_rows(
        out_root / index_name,
        [{"query_id": item.get("query_id") or item.get("id")} for item in queries],
    )


def run_dynamics_phase_loaded(
    *,
    args: argparse.Namespace,
    prefixes: list[dict],
    tokenizer,
    model,
    model_type: str,
    device: str,
    shard_idx: int | None = None,
) -> None:
    prompts: list[str] = []
    turns_by_prompt: list[list[dict]] = []
    conv_ids: list[str] = []
    for prefix in prefixes:
        turns = _prefix_turns(prefix)
        if not turns:
            continue
        prompts.append(_render_full_conversation(turns, model_type))
        turns_by_prompt.append(turns)
        conv_ids.append(str(prefix.get("conv_id") or prefix.get("prefix_id") or prefix.get("id")))
    states_by_kind, index_by_kind = _capture_dynamics_loaded_model(
        prompts=prompts,
        turns_by_prompt=turns_by_prompt,
        conv_ids=conv_ids,
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    out_root = Path(args.out) / "summaries" / f"dynamics_{model_type}"
    for kind, states in states_by_kind.items():
        _write_layer_stack(out_root, kind, states, shard_idx=shard_idx)
        index_name = (
            f"row_index_{kind}.jsonl"
            if shard_idx is None
            else f"row_index_{kind}_shard{shard_idx:05d}.jsonl"
        )
        _write_jsonl_rows(out_root / index_name, index_by_kind[kind])


def run_hf_cpu_smoke(args: argparse.Namespace) -> None:
    """Tiny Qwen2 CPU smoke over real corpus/tokenizer and real capture seams."""
    from transformers import AutoTokenizer

    logger.info("[hf-cpu-smoke] Starting tiny Qwen2 CPU smoke")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, prefix_store, query_store = _load_rows_for_smoke(args)
    cells = args.cells[:1]
    if len(cells) != 1:
        raise ValueError("hf-cpu smoke expects exactly one cell via --cells")
    if any(not CELL_CONFIG[cell]["own_policy"] for cell in cells):
        attach_completion_sources(rows, args.corpus_dir, out_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        INSTRUCT_MODEL,
        revision=INSTRUCT_REVISION,
        trust_remote_code=True,
    )
    _get_tokenizer._tok = tokenizer
    model = _tiny_qwen2_model(tokenizer, n_layers=args.n_layers, hidden_dim=args.hidden_dim)
    device = "cpu"

    rng = np.random.default_rng(GEN_SEED)
    rb_directions = rng.standard_normal((args.n_layers, N_TRAITS, args.hidden_dim)).astype(
        np.float32
    )

    rb_pool = None
    smoke_dmap = (
        _load_derangement(Path(args.corpus_dir))
        if any(CELL_CONFIG[c]["text_source"] == "shuffled" for c in cells) and args.corpus_dir
        else None
    )
    if "capture" in args.phases_set or "gen" in args.phases_set:
        for cell_id in cells:
            cfg = CELL_CONFIG[cell_id]
            prefix_texts: list[str] = []
            prompts: list[str] = []
            completions: list[str] = []
            for row in _rows_for_cell(rows, cell_id, derangement=smoke_dmap):
                prefix_text, prompt, completion = render_row(
                    row,
                    prefix_store,
                    query_store,
                    prompt_format=cfg["prompt_format"],
                    text_source=cfg["text_source"],
                )
                prefix_texts.append(prefix_text)
                prompts.append(prompt)
                completions.append(completion or "")
            if cfg["own_policy"] and "gen" in args.phases_set:
                completions = _generate_hf_cpu(prompts, tokenizer, model, args.max_gen_tokens)
            elif any(c == "" for c in completions):
                raise ValueError(f"{cell_id} requires completion text for hf-cpu smoke")

            write_completions_jsonl(
                out_dir=out_dir,
                cell_id=cell_id,
                shard_idx=0,
                model_type=cfg["model"],
                rows=_rows_for_cell(rows, cell_id, derangement=smoke_dmap),
                completions=completions,
            )
            if "capture" in args.phases_set:
                capture_out = _capture_batch_loaded_model(
                    prefix_texts=prefix_texts,
                    prompts=prompts,
                    completions=completions,
                    prompt_format=cfg["prompt_format"],
                    model=model,
                    tokenizer=tokenizer,
                    n_layers=args.n_layers,
                    hidden_dim=args.hidden_dim,
                    device=device,
                    log_label="hf-cpu",
                    rb_directions=rb_directions if cfg["own_policy"] else None,
                )
                write_summaries_npy(
                    out_dir=out_dir,
                    cell_id=cell_id,
                    shard_idx=0,
                    summaries_list=capture_out.summaries,
                    n_layers=args.n_layers,
                    hidden_dim=args.hidden_dim,
                )
                if capture_out.rb_pool is not None:
                    rb_pool = capture_out.rb_pool
                    write_rb_pool_npy(out_dir, cell_id, 0, rb_pool)

    if "bare" in args.phases_set:
        queries = sorted(
            query_store.values(), key=lambda item: str(item.get("query_id", item.get("id", "")))
        )
        if args.row_limit is not None:
            queries = queries[: args.row_limit]
        run_bare_phase_loaded(
            args=args,
            queries=queries,
            tokenizer=tokenizer,
            model=model,
            model_type="instruct",
            device=device,
        )
    if "dynamics" in args.phases_set:
        prefixes = _dynamics_panel(prefix_store)
        if args.row_limit is not None:
            prefixes = prefixes[: args.row_limit]
        run_dynamics_phase_loaded(
            args=args,
            prefixes=prefixes,
            tokenizer=tokenizer,
            model=model,
            model_type="instruct",
            device=device,
        )

    _write_sentinel(args, phase="done", note="hf-cpu-smoke")
    summary_files = list((out_dir / "summaries").rglob("*.npy"))
    comp_files = list((out_dir / "raw_completions").rglob("*.jsonl"))
    print(
        f"[hf-cpu-smoke] artifact digest: {len(rows)} rows, {len(summary_files)} npy, "
        f"{len(comp_files)} jsonl, rb_pool_shape={None if rb_pool is None else rb_pool.shape}"
    )
    print("[phase=done]")  # noqa: phase-done-reserved


# ---------------------------------------------------------------------------
# Sentinel writing
# ---------------------------------------------------------------------------


def _write_sentinel(args: argparse.Namespace, phase: str, note: str = "") -> None:
    """Write pod-side sentinel for poll_pipeline.py."""
    import datetime

    sentinel_dir = Path("/workspace/logs")
    try:
        sentinel_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        sentinel_dir = Path(args.out) / "logs"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
    sentinel_path = sentinel_dir / f"issue-{args.issue}-gpu-phase.json"
    payload = {
        "issue": args.issue,
        "phase": phase,
        "cells": args.cells,
        "out": args.out,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "note": note,
    }
    try:
        sentinel_path.write_text(json.dumps(payload, indent=2))
    except OSError:
        sentinel_dir = Path(args.out) / "logs"
        sentinel_dir.mkdir(parents=True, exist_ok=True)
        sentinel_path = sentinel_dir / f"issue-{args.issue}-gpu-phase.json"
        sentinel_path.write_text(json.dumps(payload, indent=2))
    logger.info("[sentinel] Wrote %s (phase=%s)", sentinel_path, phase)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--issue", type=int, default=1092)
    p.add_argument(
        "--phases",
        default="gen_instruct,gen_pretrained,capture_all",
        help=(
            "Comma-separated phases: gen_instruct, gen_pretrained, capture_all, "
            "bare, dynamics. Use 'all' for everything."
        ),
    )
    p.add_argument("--corpus-rev", required=True, help="Git SHA of corpus build (for fingerprint)")
    p.add_argument("--rb-rev", default="037fcbb", help="r_B directions revision on HF")
    p.add_argument("--out", default="/workspace/issue1092", help="Output root directory")
    p.add_argument(
        "--cells",
        nargs="+",
        default=list(CELL_CONFIG.keys()),
        help="Cell IDs to process",
    )
    p.add_argument("--row-limit", type=int, default=None, help="Limit rows per cell (smoke)")
    p.add_argument(
        "--finalize-cells",
        nargs="+",
        default=None,
        help=(
            "Cells whose COMPLETE on-disk gen/capture outputs from a PRIOR "
            "invocation should be consolidated + uploaded by this one "
            "('all' = every --cells entry). Round-8.9 completion composition: "
            "lets a phase-scoped re-run (--phases dynamics) finish a run whose "
            "cells completed earlier without recomputing them at the new code "
            "sha. Each cell is completeness-asserted on disk before finalizing."
        ),
    )
    p.add_argument(
        "--resume-code-shas",
        nargs="+",
        default=None,
        help=(
            "Additional PRIOR code_sha values (16-hex script-file hashes) whose "
            "CELL-shard fingerprints are accepted as complete (round-8.9). Pass "
            "ONLY a sha whose diff vs the current script provably does not touch "
            "the gen/capture path — this is an explicit operator assertion that "
            "saves recomputing bit-identical shards (launch 8's cell_inst_own: "
            "~78 min on 8x H100). Aux (bare/dynamics) shards never resume "
            "across shas."
        ),
    )
    p.add_argument("--no-upload", action="store_true", help="Skip HF upload")
    p.add_argument("--cpu-smoke", action="store_true", help="CPU smoke mode (no GPU required)")
    p.add_argument("--n-layers", type=int, default=N_LAYERS)
    p.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    p.add_argument(
        "--backend",
        choices=("cuda", "hf-cpu", "cpu-synthetic"),
        default="cuda",
        help="cuda production path, hf-cpu tiny Qwen2 path, or synthetic layout smoke",
    )
    p.add_argument("--max-gen-tokens", type=int, default=MAX_GEN_TOKENS)
    p.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Local corpus directory (if not downloading from HF)",
    )
    p.add_argument("--skip-g2", action="store_true", help="Skip G2 gate (debugging)")
    p.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help=(
            "Thread enable_prefix_caching=False into every vLLM engine "
            "(round-8.5 H100 illegal-memory-access mitigation; default OFF)"
        ),
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help=(
            "Thread enforce_eager=True into every vLLM engine (round-8.5 "
            "H100 illegal-memory-access mitigation; default OFF)"
        ),
    )
    p.add_argument(
        "--vllm-signature-smoke",
        action="store_true",
        help="Validate vLLM constructor signatures without creating a GPU engine",
    )
    return p


def parse_phase_set(phases: str, parser: argparse.ArgumentParser) -> set[str]:
    if phases == "all":
        return {"gen", "capture", "bare", "dynamics"}
    phases_set: set[str] = set()
    valid = {"gen_instruct", "gen_pretrained", "capture_all", "capture", "bare", "dynamics"}
    for raw_name in phases.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name not in valid:
            parser.error(f"Unknown phase {name!r}. Valid phases: {sorted(valid)} or all")
        if name in ("gen_instruct", "gen_pretrained"):
            phases_set.add("gen")
        elif name in ("capture_all", "capture"):
            phases_set.add("capture")
        else:
            phases_set.add(name)
    if not phases_set:
        parser.error("--phases resolved to an empty phase set")
    return phases_set


def run_vllm_signature_smoke() -> None:
    import inspect

    from vllm import EngineArgs, SamplingParams

    sampling_sig = inspect.signature(SamplingParams)
    deprecated_beam_kw = "use" + "_beam_search"
    if deprecated_beam_kw in sampling_sig.parameters:
        raise AssertionError("vLLM SamplingParams unexpectedly has deprecated beam kwarg")
    SamplingParams(temperature=0.0, max_tokens=1, stop=["<|im_end|>"], seed=GEN_SEED)
    # round-8.5: validate the installed vLLM accepts the H100-IMA mitigation
    # kwargs the --no-prefix-caching / --enforce-eager flags thread through.
    EngineArgs(
        model=INSTRUCT_MODEL,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    print(
        "[vllm-signature-smoke] SamplingParams ok; "
        f"EngineArgs max_model_len={MAX_MODEL_LEN} "
        "(+enable_prefix_caching/enforce_eager accepted); "
        f"params={len(sampling_sig.parameters)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _remaining_cells(orig_cells: list[str], completed_cells: list[str]) -> list[str]:
    """Cells still owed gen/capture this invocation, from the ORIGINAL cell list.

    NEVER compute this from args.cells: run_cell_stage rebinds args.cells per
    stage, so after the first-cell gate stage args.cells == ['cell_inst_own']
    and a remaining computed from it is EMPTY — launch 8 silently skipped the
    other 7 cells' gen+capture exactly this way (the --skip-g2 smoke skipped
    the gate block, masking the clobber).
    """
    done = set(completed_cells)
    return [c for c in orig_cells if c not in done]


def main() -> None:  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.vllm_signature_smoke:
        run_vllm_signature_smoke()
        return

    # Validate cells
    for c in args.cells:
        if c not in CELL_CONFIG:
            parser.error(f"Unknown cell: {c!r}. Valid: {list(CELL_CONFIG.keys())}")

    # Parse phases
    args.phases_set = parse_phase_set(args.phases, parser)
    args.phase_name = ",".join(sorted(args.phases_set))

    logger.info("[main] phases_set=%s cells=%s", args.phases_set, args.cells)

    if args.cpu_smoke:
        args.backend = "cpu-synthetic"
    if args.backend == "cpu-synthetic":
        run_cpu_smoke(args)
        return
    if args.backend == "hf-cpu":
        run_hf_cpu_smoke(args)
        return

    # Load corpus
    if args.corpus_dir is None:
        raise ValueError("--corpus-dir is required for non-smoke mode")

    corpus_dir = Path(args.corpus_dir)
    rows = load_manifest(corpus_dir)
    prefix_store = load_store(corpus_dir, "prefix_store.jsonl")
    query_store = load_store(corpus_dir, "query_store.jsonl")

    # Apply row limit (smoke)
    if args.row_limit is not None:
        all_rows = rows
        rows = rows[: args.row_limit]
        logger.info("[main] Row limit applied: %d rows", len(rows))
        # round-8.8: close the truncated set over the derangement so the
        # shuffled cells' answer-SOURCE rows are generated within the smoke.
        # Production (row_limit=None) is untouched — at full scale every
        # source exists in the manifest by construction.
        if any(CELL_CONFIG[c]["text_source"] == "shuffled" for c in args.cells):
            dmap = _load_derangement(corpus_dir)
            kept_ids = {str(r.get("row_id")) for r in rows}
            needed = {dmap[rid] for rid in kept_ids if rid in dmap} - kept_ids
            if needed:
                by_id = {str(r.get("row_id")): r for r in all_rows}
                rows = rows + [by_id[rid] for rid in sorted(needed) if rid in by_id]
                logger.info(
                    "[main] Row-limit derangement closure: +%d answer-source rows (%d total)",
                    len(rows) - len(kept_ids),
                    len(rows),
                )

    # Corpus hash for fingerprinting
    corpus_hash = hashlib.sha256(args.corpus_rev.encode()).hexdigest()[:16]

    # Load r_B directions if any own-policy cell is in scope
    rb_directions: np.ndarray | None = None
    if any(c in CELLS_OWN_POLICY for c in args.cells) and "capture" in args.phases_set:
        rb_directions = load_rb_directions(
            rb_rev=args.rb_rev,
            n_layers=args.n_layers,
            n_traits=N_TRAITS,
            hidden_dim=args.hidden_dim,
        )
        logger.info("[main] r_B directions loaded: %s", rb_directions.shape)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    completed_cells: list[str] = []
    non_own_cells = [c for c in args.cells if not CELL_CONFIG[c]["own_policy"]]
    if non_own_cells:
        attach_completion_sources(rows, corpus_dir, out_dir)

    orig_cells = list(args.cells)
    orig_phases = set(args.phases_set)
    orig_phase_name = args.phase_name

    def run_cell_stage(
        stage_cells: list[str], phase_set: set[str], *, rb: np.ndarray | None
    ) -> list[dict]:
        if not stage_cells or not phase_set:
            return []
        # round-8.9: stage-scope the args rebinds and ALWAYS restore — launch 8
        # left args.cells clobbered to ['cell_inst_own'] after the gate stage,
        # and the remaining-cells computation downstream read the clobbered
        # list (see _remaining_cells).
        prev_cells = list(args.cells)
        prev_phases = set(args.phases_set)
        prev_phase_name = args.phase_name
        args.cells = list(stage_cells)
        args.phases_set = set(phase_set)
        args.phase_name = ",".join(sorted(args.phases_set))
        logger.info("[main] dispatch stage phases=%s cells=%s", args.phase_name, args.cells)
        try:
            stage_results = run_dispatch(
                cells=stage_cells,
                rows=rows,
                prefix_store=prefix_store,
                query_store=query_store,
                out_dir=out_dir,
                args=args,
                rb_directions=rb,
                corpus_hash=corpus_hash,
            )
        finally:
            args.cells = prev_cells
            args.phases_set = prev_phases
            args.phase_name = prev_phase_name
        check_dispatch_errors(stage_results)
        all_results.extend(stage_results)
        return stage_results

    if "capture" in args.phases_set or "gen" in args.phases_set:
        if "cell_inst_own" in args.cells and not args.skip_g2 and "capture" in args.phases_set:
            logger.info("[main] first-cell gate stage: cell_inst_own")
            if "gen" in orig_phases:
                run_cell_stage(["cell_inst_own"], {"gen"}, rb=rb_directions)
            run_cell_stage(["cell_inst_own"], {"capture"}, rb=rb_directions)
            completed_cells.append("cell_inst_own")
            run_g2_gate_from_disk_isolated(
                out_dir,
                "cell_inst_own",
                rows=rows,
                prefix_store=prefix_store,
                query_store=query_store,
                args=args,
                rb_directions=rb_directions,
            )
            attach_completion_sources(rows, corpus_dir, out_dir)

        # round-8.8: dependency-scheduled stages over ALL remaining cells at
        # once — run_dispatch's _DispatchScheduler releases a dependent cell's
        # shards the moment its producer completes (work-conserving within the
        # ready set; no wave barriers among ready cells). gen and capture stay
        # SEPARATE dispatches for memory safety (one vLLM engine per worker
        # lifetime in gen — model-affinity queues split the GPUs between the
        # instruct/pretrained engines; capture coexists both HF models per
        # worker). The first-cell G2 gate above stays a deliberate barrier
        # (plan §7: validate the rig before burning the other cells). Workers
        # attach cross-text completion sources FRESH FROM DISK per shard, so
        # the spawn-time snapshot never goes stale (_process_shard).
        remaining = _remaining_cells(orig_cells, completed_cells)
        if remaining:
            if "gen" in orig_phases:
                logger.info("[main] dependency-scheduled gen stage: cells=%s", remaining)
                run_cell_stage(remaining, {"gen"}, rb=rb_directions)
            if "capture" in orig_phases:
                logger.info("[main] parallel capture stage: cells=%s", remaining)
                run_cell_stage(remaining, {"capture"}, rb=rb_directions)
            completed_cells.extend(remaining)

    args.cells = orig_cells
    args.phases_set = orig_phases
    args.phase_name = orig_phase_name

    # Auxiliary P3b/P3c phases have distinct layouts and never silently alias capture.
    if "bare" in args.phases_set or "dynamics" in args.phases_set:
        aux_results = run_aux_dispatch(
            phases=args.phases_set,
            query_store=query_store,
            prefix_store=prefix_store,
            out_dir=out_dir,
            args=args,
            corpus_hash=corpus_hash,
        )
        check_dispatch_errors(aux_results)
        all_results.extend(aux_results)

    # round-8.9 completion composition: finalization (consolidate + upload) is
    # DISK-driven, not invocation-scoped — a phase-scoped re-run (e.g.
    # --phases dynamics after a post-capture crash) can finish cells whose
    # gen/capture completed in a PRIOR invocation via --finalize-cells,
    # instead of recomputing them at the new code sha.
    finalize_cells = list(completed_cells)
    if args.finalize_cells:
        requested = (
            orig_cells if list(args.finalize_cells) == ["all"] else list(args.finalize_cells)
        )
        for cell_id in requested:
            if cell_id not in CELL_CONFIG:
                raise ValueError(f"--finalize-cells: unknown cell {cell_id!r}")
            if cell_id in finalize_cells:
                continue
            _assert_cell_captures_on_disk(out_dir, cell_id, n_layers=args.n_layers)
            finalize_cells.append(cell_id)
        logger.info("[main] finalize cells (this run + prior on-disk): %s", sorted(finalize_cells))

    # Upload per cell
    for cell_id in sorted(set(finalize_cells)):
        consolidate_cell_shards(out_dir, cell_id, n_layers=args.n_layers)

    summary_path = out_dir / "manifests" / "gpu_phase_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"results": all_results}, indent=2))

    if not args.no_upload:
        for cell_id in sorted(set(finalize_cells)):
            upload_cell_captures(out_dir, cell_id, args.issue)
        upload_auxiliary_summaries(out_dir, args.issue)

    # Write sentinel
    _write_sentinel(args, phase="done")
    print("[phase=done]")  # noqa: phase-done-reserved
    logger.info("[main] GPU phase complete")


if __name__ == "__main__":
    main()
