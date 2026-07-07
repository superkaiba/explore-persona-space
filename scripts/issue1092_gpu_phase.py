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
    "cell_inst_own": {"model": "instruct", "text_format": "instruct", "own_policy": True},
    "cell_pre_insttext": {"model": "pretrained", "text_format": "instruct", "own_policy": False},
    "cell_pre_own": {"model": "pretrained", "text_format": "pretrained", "own_policy": True},
    "cell_inst_pretext": {"model": "instruct", "text_format": "pretrained", "own_policy": False},
    "cell_inst_claude": {"model": "instruct", "text_format": "claude", "own_policy": False},
    "cell_pre_claude": {"model": "pretrained", "text_format": "claude", "own_policy": False},
    "cell_inst_shuf": {"model": "instruct", "text_format": "shuffled", "own_policy": False},
    "cell_pre_shuf": {"model": "pretrained", "text_format": "shuffled", "own_policy": False},
}

# Per-shard chunk size for vLLM to avoid deadlock (#664 recipe)
DEFAULT_VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))

# Summary kinds: prefix-end, context-end, t1, t2, t3 (5 per layer)
SUMMARY_KINDS = ["prefix_end", "context_end", "tok1", "tok2", "tok3"]

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
                result[item["id"]] = item
    return result


def render_row(
    row: dict,
    prefix_store: dict,
    query_store: dict,
    text_format: str,
    completion_override: str | None = None,
) -> tuple[str, str | None]:
    """Render a manifest row to (prompt_text, completion_text_or_None).

    Returns:
        prompt: the input prefix/context text
        completion: the completion text (for cross-cell formats), or None for
                    own-policy cells (completion comes from generation)
    """
    prefix_id = row["prefix_id"]
    query_id = row["query_id"]

    prefix_item = prefix_store[prefix_id]
    query_item = query_store[query_id]

    turns = prefix_item["turns"]
    query = query_item["query"]

    if text_format in ("instruct", "pretrained"):
        if text_format == "instruct":
            prompt = _render_instruct(turns, query)
        else:
            prompt = _render_naturalistic(turns, query)
        return prompt, completion_override

    elif text_format == "claude":
        # Claude text: completion stored in row["claude_text"]
        if text_format == "instruct":
            prompt = _render_instruct(turns, query)
        else:
            # Cross-cells with claude text use instruct prompt format
            prompt = _render_instruct(turns, query)
        completion = row.get("claude_text", "")
        return prompt, completion

    elif text_format == "shuffled":
        # Shuffled: use shuffled_prefix_id if available, else prefix_id
        shuf_prefix_id = row.get("shuffled_prefix_id", prefix_id)
        shuf_prefix_item = prefix_store.get(shuf_prefix_id, prefix_item)
        shuf_turns = shuf_prefix_item["turns"]
        prompt = _render_instruct(shuf_turns, query)
        completion = row.get("claude_text", "")
        return prompt, completion

    else:
        raise ValueError(f"Unknown text_format: {text_format!r}")


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


def _find_boundary_positions(
    input_ids: torch.Tensor,
    prefix_text: str,
    tokenizer,
) -> tuple[int, int]:
    """Find prefix-end and context-end token positions.

    Returns (prefix_end_pos, context_end_pos) as 0-indexed token positions.
    prefix_end is the last token of the prefix (before the query).
    context_end is the last token before generation starts.
    """
    # Encode prefix alone to find boundary
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    # context_end = last position of the full prompt (context_end_pos = len(input_ids) - 1)
    n_total = input_ids.shape[-1]
    return max(0, n_prefix - 1), max(0, n_total - 1)


def _capture_row_hf(
    prompt: str,
    completion: str,
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

    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=3072)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Find prefix/context boundaries
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    n_prompt_tokens = prompt_ids.shape[-1]
    n_total_tokens = input_ids.shape[-1]

    # Prefix end: last token before query (prompt = prefix + query for instruct format)
    # We use n_prompt_tokens - 1 as context_end_pos
    context_end_pos = min(n_prompt_tokens - 1, n_total_tokens - 1)
    # prefix_end_pos: approximate as 90% of context_end (heuristic for prefix-without-query)
    prefix_end_pos = max(0, context_end_pos - 5)  # -5 ≈ short query token count

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

    summaries["prefix_end"] = _extract_pos(prefix_end_pos)
    summaries["context_end"] = _extract_pos(context_end_pos)

    # tok1, tok2, tok3: first 3 generated tokens (after context_end_pos)
    for i, kind in enumerate(["tok1", "tok2", "tok3"]):
        gen_pos = context_end_pos + 1 + i
        if gen_pos < n_total_tokens:
            summaries[kind] = _extract_pos(gen_pos)
        else:
            # Pad with zeros if sequence is too short
            summaries[kind] = np.zeros((n_layers, hidden_dim), dtype=np.float16)

    return summaries


def _capture_batch_hf(
    prompts: list[str],
    completions: list[str],
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

    results = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        if i % 10 == 0:
            logger.info("[gpu=%d] capture row %d/%d", gpu_id, i, len(prompts))
        result = _capture_row_hf(prompt, completion, model, tokenizer, n_layers, hidden_dim, device)
        results.append(result)

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

    # For B0: use context_end summary (prefix + full query, last position)
    # Shape: (n_rows, n_layers, hidden_dim)
    context_vecs = np.stack(
        [s["context_end"].astype(np.float32) for s in summaries_list],
        axis=0,
    )  # (n_rows, n_layers, hidden_dim)

    # Batched projection: (n_rows, n_layers, hidden_dim) x (n_layers, n_traits, hidden_dim)
    # -> (n_rows, n_layers, n_traits)
    # Use einsum: "rld,ltd->rlt"
    projections = np.einsum("rld,ltd->rlt", context_vecs, rb_directions)
    # projections: (n_rows, n_layers, n_traits)

    # Pool across rows (treating T=n_rows for per-cell pooling)
    pooled = _pool_projections(projections)  # (n_layers, n_traits, 4)

    # Per-row output: (n_rows, n_layers, n_traits, 4)
    # Pool modes applied per-row (T=1 for each row's projections)
    per_row = np.zeros((n_rows, n_layers, n_traits, 4), dtype=np.float32)
    for r in range(n_rows):
        row_proj = projections[r : r + 1, :, :]  # (1, n_layers, n_traits)
        # For T=1: mean=max=top3=last = the single value
        row_val = row_proj[0]  # (n_layers, n_traits)
        per_row[r, :, :, 0] = row_val  # mean
        per_row[r, :, :, 1] = row_val  # max
        per_row[r, :, :, 2] = row_val  # top3
        per_row[r, :, :, 3] = row_val  # last

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
    text_format = cfg["text_format"]
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
        boundary_strings=(
            STOP_TOKENS_INSTRUCT if model_type == "instruct" else STOP_TOKENS_PRETRAINED
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

    for row in shard_rows:
        prompt, completion_text = render_row(
            row,
            prefix_store,
            query_store,
            text_format=text_format if not own_policy else text_format,
        )
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
            prompts=prompts,
            completions=completions,
            model_name=model_name,
            revision=revision,
            gpu_id=gpu_id,
            n_layers=args.n_layers,
            hidden_dim=HIDDEN_DIM,
        )

        # Write summaries immediately
        write_summaries_npy(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=shard.shard_idx,
            summaries_list=summaries_list,
            n_layers=args.n_layers,
            hidden_dim=HIDDEN_DIM,
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
                s[kind] = rng.standard_normal((args.n_layers, HIDDEN_DIM)).astype(np.float16)
            summaries_list.append(s)

        write_summaries_npy(
            out_dir=out_dir,
            cell_id=cell_id,
            shard_idx=0,
            summaries_list=summaries_list,
            n_layers=args.n_layers,
            hidden_dim=HIDDEN_DIM,
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


# ---------------------------------------------------------------------------
# Sentinel writing
# ---------------------------------------------------------------------------


def _write_sentinel(args: argparse.Namespace, phase: str, note: str = "") -> None:
    """Write pod-side sentinel for poll_pipeline.py."""
    import datetime

    sentinel_dir = Path("/workspace/logs")
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

    # CPU smoke mode: synthetic outputs only
    if args.cpu_smoke:
        run_cpu_smoke(args)
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
                hidden_dim=HIDDEN_DIM,
            )
            logger.info("[main] r_B directions loaded: %s", rb_directions.shape)
        except Exception as exc:
            logger.warning("[main] Could not load r_B directions: %s (skipping B0 pool)", exc)
            rb_directions = None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
