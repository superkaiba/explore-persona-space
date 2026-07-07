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
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Load dotenv before any HF/API imports (subprocess-env-explicit requirement)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

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
UPLOAD_PREFIX = f"{HF_PREFIX}/captures"

N_LAYERS = 28
HIDDEN_DIM = 3584
GEN_SEED = 42
MAX_GEN_TOKENS = 1024

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
                    raise KeyError(f"{store_path} item missing id/prefix_id/query_id: {item.keys()}")
                result[str(key)] = item
    return result


def _prefix_turns(prefix_item: dict) -> list[dict]:
    turns = prefix_item.get("prefix_turns") or prefix_item.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"prefix item {prefix_item.get('prefix_id')} has no turns")
    return turns


def _query_text(query_item: dict) -> str:
    text = query_item.get("text", query_item.get("query"))
    if not isinstance(text, str) or not text:
        raise ValueError(f"query item {query_item.get('query_id')} has no text/query")
    return text


def _render_prompt_parts(turns: list[dict], query: str, prompt_format: str) -> tuple[str, str]:
    """Return (prefix_text, prompt_text) under the requested model prompt format."""
    if prompt_format == "instruct":
        prefix_text = _render_prefix_instruct(turns)
        prompt_text = _render_instruct(turns, query)
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

    if text_source == "own":
        return prefix_text, prompt, completion_override
    if text_source == "claude":
        completion = row.get("claude_text") or row.get("completion")
        if completion is None:
            raise ValueError(f"row {row.get('row_id')} has no Claude completion text")
        return prefix_text, prompt, str(completion)
    if text_source in ("instruct", "pretrained"):
        key = f"{text_source}_completion"
        completion = row.get(key) or row.get("completion")
        if completion is None:
            raise ValueError(f"row {row.get('row_id')} has no {key}")
        return prefix_text, prompt, str(completion)
    if text_source == "shuffled":
        completion = row.get("shuffled_completion") or row.get("completion")
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
    code_sha: str,
) -> dict:
    """Build a fingerprint dict for resume idempotency."""
    return {
        "corpus_hash": corpus_hash,
        "cell_id": cell_id,
        "row_start": row_start,
        "row_end": row_end,
        "model_id": model_id,
        "dtype": dtype,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "boundary_strings": boundary_strings,
        "rb_rev": rb_rev,
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


# ---------------------------------------------------------------------------
# vLLM generation
# ---------------------------------------------------------------------------


def _run_gen_vllm(
    prompts: list[str],
    model_name: str,
    revision: str,
    stop_tokens: list[str],
    max_tokens: int,
    seed: int,
    gpu_id: int,
    chunk_size: int,
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
        max_model_len=4096,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=stop_tokens,
        seed=seed,
        use_beam_search=False,
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


def _capture_row_hf(
    prefix_text: str,
    prompt: str,
    completion: str,
    prompt_format: str,
    model,
    tokenizer,
    n_layers: int,
    hidden_dim: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Teacher-forced capture for one (prompt, completion) pair.

    Returns dict: kind -> np.ndarray of shape (n_layers, hidden_dim), dtype=float16.
    Summaries: prefix_end, context_end, tok1, tok2, tok3.
    """
    import torch

    boundary = _boundary_suffix(prompt_format)
    full_text = prompt + completion + boundary
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=3072,
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    n_prefix_tokens = len(tokenizer.encode(prefix_text, add_special_tokens=False))
    n_prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    n_completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
    n_total_tokens = input_ids.shape[-1]

    prefix_end_pos = min(max(0, n_prefix_tokens - 1), n_total_tokens - 1)
    context_end_pos = min(n_prompt_tokens - 1, n_total_tokens - 1)
    answer_start = min(context_end_pos + 1, n_total_tokens - 1)
    answer_end = min(context_end_pos + 1 + max(1, n_completion_tokens), n_total_tokens)
    t3_pos = n_total_tokens - 1
    t2_end = max(answer_end, t3_pos)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            logits_to_keep=1,  # avoid materializing full-vocab logits (#779)
        )

    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    # We want the post-block residuals: hidden_states[1:] (skip embedding layer)
    hidden_states = outputs.hidden_states[1:]  # (n_layers,) each (1, seq, hid)
    if len(hidden_states) != n_layers:
        raise ValueError(f"model returned {len(hidden_states)} layers, expected {n_layers}")
    if hidden_states[0].shape[-1] != hidden_dim:
        raise ValueError(f"model hidden dim {hidden_states[0].shape[-1]} != expected {hidden_dim}")

    # Extract 5 summary positions
    summaries = {}

    def _extract_pos(pos: int) -> np.ndarray:
        """Extract hidden state at position `pos` across all layers. Shape: (n_layers, hid)."""
        pos = min(pos, n_total_tokens - 1)
        arr = np.stack(
            [hs[0, pos, :].to(torch.float16).cpu().numpy() for hs in hidden_states],
            axis=0,
        )  # (n_layers, hidden_dim)
        return arr

    def _extract_span(start: int, end: int) -> np.ndarray:
        start = min(max(0, start), n_total_tokens - 1)
        end = min(max(start + 1, end), n_total_tokens)
        per_layer = []
        for hs in hidden_states:
            span = hs[0, start:end, :]
            per_layer.append(span.mean(dim=0).to(torch.float16).cpu().numpy())
        return np.stack(per_layer, axis=0)

    def _extract_answer_tokens() -> np.ndarray:
        start = min(max(0, answer_start), n_total_tokens - 1)
        end = min(max(start + 1, answer_end), n_total_tokens)
        per_layer = []
        for hs in hidden_states:
            span = hs[0, start:end, :].to(torch.float16).cpu().numpy()
            per_layer.append(span)
        # (T, L, H), keeping only the generated-answer span for B0 pooling.
        return np.stack(per_layer, axis=1)

    summaries["prefix_end"] = _extract_pos(prefix_end_pos)
    summaries["context_end"] = _extract_pos(context_end_pos)
    summaries["t1"] = _extract_span(answer_start, answer_end)
    summaries["t2"] = _extract_span(answer_start, t2_end)
    summaries["t3"] = _extract_pos(t3_pos)
    summaries["_answer_token_states"] = _extract_answer_tokens()

    return summaries


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
) -> list[dict[str, np.ndarray]]:
    """Run HF teacher-forced capture for a batch of (prompt, completion) pairs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
        device_map={"": device},  # explicit device, no auto-offload (#825)
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
) -> list[dict[str, np.ndarray]]:
    """Shared teacher-forced capture loop for production HF and tiny CPU smokes."""
    results = []
    for i, (prefix_text, prompt, completion) in enumerate(zip(prefix_texts, prompts, completions)):
        if i % 10 == 0:
            logger.info("[%s] capture row %d/%d", log_label, i, len(prompts))
        result = _capture_row_hf(
            prefix_text,
            prompt,
            completion,
            prompt_format,
            model,
            tokenizer,
            n_layers,
            hidden_dim,
            device,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# B0 r_B projection pooling
# ---------------------------------------------------------------------------


def load_rb_directions(rb_rev: str, n_layers: int, n_traits: int, hidden_dim: int) -> np.ndarray:
    """Download and load r_B direction matrix from HF model repo.

    Returns np.ndarray of shape (n_layers, n_traits, hidden_dim), dtype=float32.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=f"issue1092/rb_directions_{rb_rev}.npy",
        revision=rb_rev,
    )
    rb = np.load(local_path)
    assert rb.shape == (n_layers, n_traits, hidden_dim), (
        f"r_B shape mismatch: got {rb.shape}, expected ({n_layers}, {n_traits}, {hidden_dim})"
    )
    return rb.astype(np.float32)


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


def compute_rb_projection_batch(
    summaries_list: list[dict[str, np.ndarray]],
    rb_directions: np.ndarray,
) -> np.ndarray:
    """Batched B0 r_B projection pooling.

    Args:
        summaries_list: list of per-row summary dicts (kind -> (n_layers, hidden_dim) fp16)
        rb_directions: (n_layers, n_traits, hidden_dim) fp32

    Returns:
        np.ndarray of shape (n_rows, n_layers, n_traits, 4), dtype=float32
    """
    n_rows = len(summaries_list)
    n_layers = rb_directions.shape[0]
    n_traits = rb_directions.shape[1]

    norms = np.linalg.norm(rb_directions, axis=2)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    rb_unit = rb_directions / safe_norms[:, :, None]

    per_row = np.zeros((n_rows, n_layers, n_traits, 4), dtype=np.float32)
    for r, summaries in enumerate(summaries_list):
        answer_states = summaries.get("_answer_token_states")
        if answer_states is None:
            raise KeyError("capture result missing _answer_token_states for B0 pooling")
        answer_states = answer_states.astype(np.float32)  # (T, L, H)
        if answer_states.ndim != 3 or answer_states.shape[1:] != (
            n_layers,
            rb_directions.shape[2],
        ):
            raise ValueError(
                f"answer token states shape {answer_states.shape} incompatible with "
                f"r_B {rb_directions.shape}"
            )
        # (T, L, H) x (L, trait, H) -> (T, L, trait), one batched einsum per row.
        projections = np.einsum("alh,lbh->alb", answer_states, rb_unit, optimize=True)
        per_row[r] = _pool_projections(projections)

    return per_row


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
        logger.warning("[G2] No summaries to check")
        return True

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

    Layout: <out>/summaries/<cell>/<kind>_L{ll}_shard{i}.npy
    Each file: (n_rows, hidden_dim) fp16.
    """
    cell_dir = out_dir / "summaries" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(summaries_list)

    paths = {}
    for kind in SUMMARY_KINDS:
        for ll in range(n_layers):
            arr = np.stack(
                [s[kind][ll] for s in summaries_list],
                axis=0,
            )  # (n_rows, hidden_dim)
            arr = arr.astype(np.float16)
            path = cell_dir / f"{kind}_L{ll:02d}_shard{shard_idx}.npy"
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
    path = pool_dir / f"{cell_id}_shard{shard_idx}.npy"
    np.save(str(path), rb_pool.astype(np.float32))
    return path


def write_completions_jsonl(
    out_dir: Path,
    cell_id: str,
    shard_idx: int,
    model_type: str,
    rows: list[dict],
    completions: list[str],
) -> Path:
    """Write raw completions JSONL. Layout: raw_completions/<model_type>/<cell>_shard{i}.jsonl."""
    comp_dir = out_dir / "raw_completions" / model_type
    comp_dir.mkdir(parents=True, exist_ok=True)
    path = comp_dir / f"{cell_id}_shard{shard_idx}.jsonl"
    with open(path, "w") as f:
        for row, completion in zip(rows, completions):
            f.write(
                json.dumps(
                    {
                        "row_id": row.get("row_id", ""),
                        "prefix_id": row.get("prefix_id", ""),
                        "query_id": row.get("query_id", ""),
                        "cell_id": cell_id,
                        "completion": completion,
                    }
                )
                + "\n"
            )
    return path


def write_fingerprint(fp_path: Path, fp_dict: dict) -> None:
    """Write fingerprint JSON for resume idempotency."""
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(json.dumps(fp_dict, indent=2))


def _load_raw_completion_files(out_dir: Path, model_type: str, cell_id: str) -> dict[str, str]:
    """Load row_id -> completion from previously persisted own-policy rollouts."""
    comp_dir = out_dir / "raw_completions" / model_type
    paths = sorted(comp_dir.glob(f"{cell_id}_shard*.jsonl")) if comp_dir.exists() else []
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
        "[completion-sources] loaded instruct=%d pretrained=%d claude=%d deranged=%d "
        "missing=%s",
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
    path_in_repo = f"{UPLOAD_PREFIX}/{cell_id}"

    # Upload summaries
    summaries_dir = out_dir / "summaries" / cell_id
    if summaries_dir.exists():
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(summaries_dir),
            path_in_repo=f"{path_in_repo}/summaries",
            commit_message=f"issue{issue}: upload summaries for {cell_id}",
        )
        logger.info("[upload] Summaries for %s uploaded", cell_id)

    # Upload r_B pool
    pool_dir = out_dir / "summaries" / "b0_rB_pool"
    pool_files = list(pool_dir.glob(f"{cell_id}_shard*.npy")) if pool_dir.exists() else []
    for pf in pool_files:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(pf),
            path_in_repo=f"{path_in_repo}/b0_rB_pool/{pf.name}",
            commit_message=f"issue{issue}: upload r_B pool for {cell_id}",
        )
    if pool_files:
        logger.info("[upload] r_B pool for %s uploaded (%d files)", cell_id, len(pool_files))

    # Upload completions
    for model_type in ("instruct", "pretrained"):
        comp_dir = out_dir / "raw_completions" / model_type
        if comp_dir.exists():
            comp_files = list(comp_dir.glob(f"{cell_id}_shard*.jsonl"))
            for cf in comp_files:
                api.upload_file(
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_or_fileobj=str(cf),
                    path_in_repo=(f"{path_in_repo}/raw_completions/{model_type}/{cf.name}"),
                    commit_message=(f"issue{issue}: upload {model_type} completions for {cell_id}"),
                )
            if comp_files:
                logger.info(
                    "[upload] %s completions for %s uploaded (%d files)",
                    model_type,
                    cell_id,
                    len(comp_files),
                )


# ---------------------------------------------------------------------------
# Work-conserving dispatcher
# ---------------------------------------------------------------------------


def _process_shard(
    shard: Shard,
    rows: list[dict],
    prefix_store: dict,
    query_store: dict,
    out_dir: Path,
    args: argparse.Namespace,
    gpu_id: int,
    rb_directions: np.ndarray | None,
    code_sha_val: str,
    corpus_hash: str,
) -> dict:
    """Process one shard on gpu_id. Returns result dict with status."""
    cell_id = shard.cell_id
    cfg = CELL_CONFIG[cell_id]
    model_type = cfg["model"]
    prompt_format = cfg["prompt_format"]
    text_source = cfg["text_source"]
    own_policy = cfg["own_policy"]

    shard_rows = rows[shard.row_start : shard.row_end]
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
        code_sha=code_sha_val,
    )
    fp_path = out_dir / "manifests" / f"{cell_id}_shard{shard.shard_idx}_fingerprint.json"
    if fingerprint_matches(fp_path, fp_dict):
        logger.info(
            "[gpu=%d] Shard %s/%d already complete (fingerprint match), skipping",
            gpu_id,
            cell_id,
            shard.shard_idx,
        )
        return {
            "status": "resumed",
            "cell_id": cell_id,
            "shard_idx": shard.shard_idx,
            "n_rows": n_rows,
        }

    # ---- Step 1: Get completions ----
    completions: list[str] = []
    prompts: list[str] = []
    prefix_texts: list[str] = []

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
        if own_policy:
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

        completions = _run_gen_vllm(
            prompts=prompts,
            model_name=model_name,
            revision=revision,
            stop_tokens=stop_tokens,
            max_tokens=args.max_gen_tokens,
            seed=GEN_SEED,
            gpu_id=gpu_id,
            chunk_size=DEFAULT_VLLM_CHUNK_SIZE,
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

        summaries_list = _capture_batch_hf(
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
    rb_pool: np.ndarray | None = None
    if own_policy and rb_directions is not None and summaries_list and "capture" in args.phases_set:
        rb_pool = compute_rb_projection_batch(summaries_list, rb_directions)
        write_rb_pool_npy(out_dir, cell_id, shard.shard_idx, rb_pool)

    # Write fingerprint (marks shard as done)
    write_fingerprint(fp_path, fp_dict)

    return {
        "status": "done",
        "cell_id": cell_id,
        "shard_idx": shard.shard_idx,
        "n_rows": n_rows,
        "summaries": summaries_list,
        "rb_pool": rb_pool,
    }


def _worker_loop(
    gpu_id: int,
    shard_queue: queue.Queue[Shard | None],
    result_queue: queue.Queue[dict],
    rows: list[dict],
    prefix_store: dict,
    query_store: dict,
    out_dir: Path,
    args: argparse.Namespace,
    rb_directions: np.ndarray | None,
    code_sha_val: str,
    corpus_hash: str,
) -> None:
    """Worker thread: pull shards from queue and process on gpu_id."""
    while True:
        shard = shard_queue.get()
        if shard is None:
            break  # sentinel: no more work
        try:
            result = _process_shard(
                shard=shard,
                rows=rows,
                prefix_store=prefix_store,
                query_store=query_store,
                out_dir=out_dir,
                args=args,
                gpu_id=gpu_id,
                rb_directions=rb_directions,
                code_sha_val=code_sha_val,
                corpus_hash=corpus_hash,
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
        shard_queue.task_done()


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
    """Work-conserving multi-GPU dispatcher.

    Creates one thread per GPU; threads pull shards from a shared queue.
    Returns list of result dicts.
    """
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        logger.error("No CUDA GPUs detected. Use --cpu-smoke for CPU-only testing.")
        raise RuntimeError("No CUDA GPUs available")

    logger.info("[dispatch] %d GPUs detected", n_gpus)

    # Build shard list
    shard_size = max(1, min(200, len(rows) // max(1, len(cells))))
    shards: list[Shard] = []
    for cell_id in cells:
        n_rows = len(rows)
        cell_shards = []
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

    logger.info("[dispatch] %d shards across %d cells", len(shards), len(cells))

    # Fill shard queue
    shard_queue: queue.Queue[Shard | None] = queue.Queue()
    for shard in shards:
        shard_queue.put(shard)
    # Sentinel per worker
    for _ in range(n_gpus):
        shard_queue.put(None)

    result_queue: queue.Queue[dict] = queue.Queue()
    code_sha_val = code_sha()

    # Spawn worker threads
    threads = []
    for gpu_id in range(n_gpus):
        t = threading.Thread(
            target=_worker_loop,
            args=(
                gpu_id,
                shard_queue,
                result_queue,
                rows,
                prefix_store,
                query_store,
                out_dir,
                args,
                rb_directions,
                code_sha_val,
                corpus_hash,
            ),
            daemon=True,
            name=f"worker-gpu-{gpu_id}",
        )
        t.start()
        threads.append(t)

    # Collect results
    results = []
    n_expected = len(shards)
    while len(results) < n_expected:
        result = result_queue.get(timeout=3600)
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

    for t in threads:
        t.join(timeout=60)

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
            s["_answer_token_states"] = rng.standard_normal(
                (3, args.n_layers, args.hidden_dim)
            ).astype(np.float16)
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
    print("[phase=done]")


def _load_rows_for_smoke(args: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    if args.corpus_dir is None:
        raise ValueError("--backend hf-cpu requires --corpus-dir from a P0 smoke or production build")
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


def run_hf_cpu_smoke(args: argparse.Namespace) -> None:
    """Tiny Qwen2 CPU smoke over real corpus/tokenizer and real capture seams."""
    import torch
    from transformers import AutoTokenizer

    logger.info("[hf-cpu-smoke] Starting tiny Qwen2 CPU smoke")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, prefix_store, query_store = _load_rows_for_smoke(args)
    cells = args.cells[:1]
    if len(cells) != 1:
        raise ValueError("hf-cpu smoke expects exactly one cell via --cells")

    tokenizer = AutoTokenizer.from_pretrained(
        INSTRUCT_MODEL,
        revision=INSTRUCT_REVISION,
        trust_remote_code=True,
    )
    model = _tiny_qwen2_model(tokenizer, n_layers=args.n_layers, hidden_dim=args.hidden_dim)
    device = "cpu"

    rng = np.random.default_rng(GEN_SEED)
    rb_directions = rng.standard_normal((args.n_layers, N_TRAITS, args.hidden_dim)).astype(
        np.float32
    )

    for cell_id in cells:
        cfg = CELL_CONFIG[cell_id]
        prefix_texts: list[str] = []
        prompts: list[str] = []
        completions: list[str] = []
        for row in rows:
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
        if cfg["own_policy"]:
            completions = _generate_hf_cpu(prompts, tokenizer, model, args.max_gen_tokens)
        elif any(c == "" for c in completions):
            raise ValueError(f"{cell_id} requires completion text for hf-cpu smoke")

        write_completions_jsonl(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=0,
            model_type=cfg["model"],
            rows=rows,
            completions=completions,
        )
        summaries_list = _capture_batch_loaded_model(
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
        )
        write_summaries_npy(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=0,
            summaries_list=summaries_list,
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
        )
        rb_pool = compute_rb_projection_batch(summaries_list, rb_directions)
        write_rb_pool_npy(out_dir, cell_id, 0, rb_pool)

    _write_sentinel(args, phase="done", note="hf-cpu-smoke")
    summary_files = list((out_dir / "summaries").rglob("*.npy"))
    comp_files = list((out_dir / "raw_completions").rglob("*.jsonl"))
    print(
        f"[hf-cpu-smoke] artifact digest: {len(rows)} rows, {len(summary_files)} npy, "
        f"{len(comp_files)} jsonl, rb_pool_shape={rb_pool.shape}"
    )
    print("[phase=done]")


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
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    # Validate cells
    for c in args.cells:
        if c not in CELL_CONFIG:
            parser.error(f"Unknown cell: {c!r}. Valid: {list(CELL_CONFIG.keys())}")

    # Parse phases
    if args.phases == "all":
        args.phases_set = {"gen", "capture"}
    else:
        raw = set(args.phases.split(","))
        phases_set = set()
        for p_name in raw:
            p_name = p_name.strip()
            if p_name in ("gen_instruct", "gen_pretrained"):
                phases_set.add("gen")
            elif p_name in ("capture_all", "bare", "dynamics", "capture"):
                phases_set.add("capture")
            else:
                logger.warning("Unknown phase %r, ignoring", p_name)
        args.phases_set = phases_set

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
        rows = rows[: args.row_limit]
        logger.info("[main] Row limit applied: %d rows", len(rows))

    # Corpus hash for fingerprinting
    corpus_hash = hashlib.sha256(args.corpus_rev.encode()).hexdigest()[:16]

    # Load r_B directions if any own-policy cell is in scope
    rb_directions: np.ndarray | None = None
    if any(c in CELLS_OWN_POLICY for c in args.cells) and "capture" in args.phases_set:
        try:
            rb_directions = load_rb_directions(
                rb_rev=args.rb_rev,
                n_layers=args.n_layers,
                n_traits=N_TRAITS,
                hidden_dim=args.hidden_dim,
            )
            logger.info("[main] r_B directions loaded: %s", rb_directions.shape)
        except Exception as exc:
            logger.warning("[main] Could not load r_B directions: %s (skipping B0 pool)", exc)
            rb_directions = None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    non_own_cells = [c for c in args.cells if not CELL_CONFIG[c]["own_policy"]]
    own_cells = [c for c in args.cells if CELL_CONFIG[c]["own_policy"]]
    if non_own_cells:
        attach_completion_sources(rows, corpus_dir, out_dir)
    if non_own_cells and own_cells and {"gen", "capture"}.issubset(args.phases_set):
        logger.info(
            "[main] two-stage run: own-policy cells first (%s), non-own cells second (%s)",
            own_cells,
            non_own_cells,
        )
        orig_cells = args.cells
        args.cells = own_cells
        own_results = run_dispatch(
            cells=own_cells,
            rows=rows,
            prefix_store=prefix_store,
            query_store=query_store,
            out_dir=out_dir,
            args=args,
            rb_directions=rb_directions,
            corpus_hash=corpus_hash,
        )
        attach_completion_sources(rows, corpus_dir, out_dir)
        args.cells = non_own_cells
        orig_phases = args.phases_set
        args.phases_set = {"capture"} if "capture" in orig_phases else set()
        non_own_results = run_dispatch(
            cells=non_own_cells,
            rows=rows,
            prefix_store=prefix_store,
            query_store=query_store,
            out_dir=out_dir,
            args=args,
            rb_directions=None,
            corpus_hash=corpus_hash,
        )
        args.cells = orig_cells
        args.phases_set = orig_phases
        results = own_results + non_own_results
        summary_path = out_dir / "manifests" / "gpu_phase_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"results": results}, indent=2))
        _write_sentinel(args, phase="done", note=f"{len(results)} shards")
        return

    # Run work-conserving dispatch
    results = run_dispatch(
        cells=args.cells,
        rows=rows,
        prefix_store=prefix_store,
        query_store=query_store,
        out_dir=out_dir,
        args=args,
        rb_directions=rb_directions,
        corpus_hash=corpus_hash,
    )

    # Check for errors
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        logger.error("[main] %d shards failed:", len(errors))
        for e in errors:
            logger.error("  %s shard%d: %s", e["cell_id"], e["shard_idx"], e.get("error"))
        raise RuntimeError(f"{len(errors)} shards failed")

    # G2 gate: run after first cell (cell_inst_own if present, else first cell)
    if not args.skip_g2:
        g2_cell = "cell_inst_own" if "cell_inst_own" in args.cells else args.cells[0]
        g2_results = [r for r in results if r["cell_id"] == g2_cell and r.get("summaries")]
        if g2_results:
            all_summaries = []
            all_rb_pools = []
            for r in g2_results:
                all_summaries.extend(r.get("summaries", []))
                if r.get("rb_pool") is not None:
                    all_rb_pools.append(r["rb_pool"])

            combined_rb_pool = np.concatenate(all_rb_pools, axis=0) if all_rb_pools else None
            g2_pass = run_g2_gate(
                cell_id=g2_cell,
                summaries_list=all_summaries,
                row_indices=list(range(len(all_summaries))),
                rb_pool=combined_rb_pool,
            )
            if not g2_pass:
                raise RuntimeError("G2 gate FAILED — aborting")

    # Upload per cell
    if not args.no_upload:
        for cell_id in args.cells:
            upload_cell_captures(out_dir, cell_id, args.issue)

    # Write sentinel
    _write_sentinel(args, phase="done")
    print("[phase=done]")
    logger.info("[main] GPU phase complete")


if __name__ == "__main__":
    main()
