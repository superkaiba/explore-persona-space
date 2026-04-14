#!/usr/bin/env python3
"""Fast corpus projection using vLLM embed mode with tensor parallelism.

Two-phase approach:
  Phase 1: Download/stream datasets to local JSONL files (network-bound)
  Phase 2: Load model with vLLM, process local data (GPU-bound, ~1400 docs/sec)

This decouples the HuggingFace streaming bottleneck from GPU processing.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 nohup uv run python scripts/project_corpus_fast.py \
        > /workspace/projection_fast_log.txt 2>&1 &
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# ---- Environment setup ----
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_PATH = "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
AXIS_PATH = "/workspace/.cache/huggingface/datasets--lu-christina--assistant-axis-vectors/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
LAYER = 48
MAX_LENGTH = 512
VLLM_BATCH = 64  # docs per vLLM embed call (kept small to avoid hangs)
FINEWEB_DOCS = 200_000
LMSYS_DOCS = 200_000
OUTPUT_DIR = Path("/workspace/axis_projections")
DATA_DIR = Path("/workspace/axis_projections/raw_data")
TP_SIZE = 2


def load_axis(axis_path: str, layer: int) -> torch.Tensor:
    """Load and normalize the axis vector for a given layer."""
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "axis" in data:
        axis = data["axis"]
    else:
        axis = data
    ax = axis[layer].float()
    ax = ax / (ax.norm() + 1e-8)
    logger.info(f"Loaded axis: shape={ax.shape}, norm={ax.norm().item():.4f}")
    return ax


# =====================================================================
# Phase 1: Download data to local JSONL
# =====================================================================


def project_lmsys_conversation(conversation: list[dict]) -> str:
    """Extract first user+assistant turn as plain text."""
    parts = []
    seen_user = seen_assistant = False
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user" and not seen_user:
            parts.append(f"User: {content}")
            seen_user = True
        elif role == "assistant" and not seen_assistant and seen_user:
            parts.append(f"Assistant: {content}")
            seen_assistant = True
            break
    return "\n\n".join(parts)


def download_fineweb(output_path: Path, max_docs: int) -> int:
    """Stream FineWeb-Edu to local JSONL file."""
    from datasets import load_dataset
    from tqdm import tqdm

    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        if existing >= max_docs:
            logger.info(f"FineWeb data already downloaded: {existing:,} docs at {output_path}")
            return existing

    logger.info(f"Downloading {max_docs:,} FineWeb-Edu docs to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    doc_count = 0

    with open(output_path, "w") as f:
        for doc in tqdm(ds, total=max_docs, desc="FineWeb download"):
            if doc_count >= max_docs:
                break
            text = doc.get("text", "")
            if not text or len(text.strip()) < 50:
                continue
            f.write(json.dumps({"doc_id": doc_count, "text": text[:5000]}) + "\n")
            doc_count += 1

    logger.info(f"FineWeb download complete: {doc_count:,} docs")
    return doc_count


def download_lmsys(output_path: Path, max_docs: int) -> int:
    """Stream LMSYS-Chat-1M to local JSONL file."""
    from datasets import load_dataset
    from tqdm import tqdm

    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        if existing >= max_docs:
            logger.info(f"LMSYS data already downloaded: {existing:,} docs at {output_path}")
            return existing

    logger.info(f"Downloading {max_docs:,} LMSYS docs to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    doc_count = 0

    with open(output_path, "w") as f:
        for doc in tqdm(ds, total=max_docs, desc="LMSYS download"):
            if doc_count >= max_docs:
                break
            text = project_lmsys_conversation(doc.get("conversation", []))
            if not text or len(text.strip()) < 50:
                continue
            f.write(json.dumps({"doc_id": doc_count, "text": text[:5000]}) + "\n")
            doc_count += 1

    logger.info(f"LMSYS download complete: {doc_count:,} docs")
    return doc_count


# =====================================================================
# Phase 2: vLLM processing from local data
# =====================================================================


def load_vllm_model():
    """Load model with vLLM for embedding extraction."""
    from vllm import LLM
    from vllm.config import PoolerConfig

    logger.info(f"Loading vLLM model from {MODEL_PATH} with TP={TP_SIZE}...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TP_SIZE,
        max_model_len=MAX_LENGTH,
        task="embed",
        pooler_config=PoolerConfig(pooling_type="LAST", normalize=False),
        dtype="bfloat16",
        max_num_seqs=256,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    elapsed = time.time() - t0
    logger.info(f"vLLM model loaded in {elapsed:.1f}s")
    return llm


def process_local_data(
    llm,
    data_path: Path,
    axis_vector: torch.Tensor,
    output_path: Path,
    batch_size: int,
    desc: str = "Projecting",
) -> int:
    """Process local JSONL data with vLLM embed, writing projection results."""
    from tqdm import tqdm

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count total docs
    total_docs = sum(1 for _ in open(data_path))
    logger.info(f"[{desc}] Processing {total_docs:,} docs from {data_path}")

    batch_texts = []
    batch_ids = []
    batch_snippets = []
    written = 0
    t_start = time.time()
    last_log = t_start

    with open(output_path, "w") as out_f, open(data_path) as in_f:
        for line in tqdm(in_f, total=total_docs, desc=desc, mininterval=2):
            doc = json.loads(line)
            text = doc["text"]
            doc_id = doc["doc_id"]

            # Pre-truncate text to ~2000 chars (roughly 512 tokens for most text)
            batch_texts.append(text[:2000])
            batch_ids.append(doc_id)
            batch_snippets.append(text[:500])

            if len(batch_texts) >= batch_size:
                try:
                    t_batch = time.time()
                    results = llm.embed(
                        batch_texts, use_tqdm=False, truncate_prompt_tokens=MAX_LENGTH
                    )
                    if written == 0:
                        logger.info(
                            f"First batch of {len(batch_texts)} docs took {time.time() - t_batch:.2f}s"
                        )
                    for did, snippet, r in zip(batch_ids, batch_snippets, results):
                        emb = torch.tensor(r.outputs.embedding, dtype=torch.float32)
                        proj = (emb @ axis_vector).item()
                        tc = len(r.prompt_token_ids)
                        out_f.write(
                            json.dumps(
                                {
                                    "doc_id": did,
                                    "projection": round(proj, 6),
                                    "token_count": tc,
                                    "text_snippet": snippet,
                                }
                            )
                            + "\n"
                        )
                        written += 1
                except Exception as e:
                    logger.warning(f"Batch failed at doc {written}: {e}")

                batch_texts = []
                batch_ids = []
                batch_snippets = []

                # Periodic throughput log
                now = time.time()
                if now - last_log > 30:
                    elapsed = now - t_start
                    rate = written / elapsed if elapsed > 0 else 0
                    remaining = total_docs - written
                    eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                    logger.info(
                        f"[{desc}] {written:,}/{total_docs:,} docs | "
                        f"{rate:.1f} docs/sec | ETA: {eta_min:.0f} min"
                    )
                    last_log = now

        # Flush remaining
        if batch_texts:
            try:
                results = llm.embed(batch_texts, use_tqdm=False, truncate_prompt_tokens=MAX_LENGTH)
                for did, snippet, r in zip(batch_ids, batch_snippets, results):
                    emb = torch.tensor(r.outputs.embedding, dtype=torch.float32)
                    proj = (emb @ axis_vector).item()
                    tc = len(r.prompt_token_ids)
                    out_f.write(
                        json.dumps(
                            {
                                "doc_id": did,
                                "projection": round(proj, 6),
                                "token_count": tc,
                                "text_snippet": snippet,
                            }
                        )
                        + "\n"
                    )
                    written += 1
            except Exception as e:
                logger.warning(f"Final batch failed: {e}")

    elapsed = time.time() - t_start
    rate = written / elapsed if elapsed > 0 else 0
    logger.info(f"[{desc}] Complete: {written:,} docs in {elapsed:.0f}s ({rate:.1f} docs/sec)")
    return written


def main():
    overall_start = time.time()

    logger.info("=" * 60)
    logger.info("Fast Corpus Projection (vLLM) - Two Phase")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Axis layer: {LAYER}, Max length: {MAX_LENGTH}, Batch: {VLLM_BATCH}")
    logger.info(f"FineWeb docs: {FINEWEB_DOCS:,}, LMSYS docs: {LMSYS_DOCS:,}")
    logger.info(f"Tensor parallel: {TP_SIZE}")
    logger.info("=" * 60)

    # ---- Phase 1: Download data ----
    logger.info("=" * 60)
    logger.info("PHASE 1: Download datasets to local storage")
    logger.info("=" * 60)

    fw_data_path = DATA_DIR / "fineweb_raw.jsonl"
    lmsys_data_path = DATA_DIR / "lmsys_raw.jsonl"

    t0 = time.time()
    download_fineweb(fw_data_path, FINEWEB_DOCS)
    logger.info(f"FineWeb download: {time.time() - t0:.0f}s")

    t0 = time.time()
    download_lmsys(lmsys_data_path, LMSYS_DOCS)
    logger.info(f"LMSYS download: {time.time() - t0:.0f}s")

    # ---- Phase 2: vLLM processing ----
    logger.info("=" * 60)
    logger.info("PHASE 2: vLLM embed + axis projection")
    logger.info("=" * 60)

    axis_vector = load_axis(AXIS_PATH, LAYER)
    llm = load_vllm_model()

    # Warmup
    logger.info("Warmup...")
    t0 = time.time()
    warmup_results = llm.embed(
        ["Warmup text."] * 4, use_tqdm=False, truncate_prompt_tokens=MAX_LENGTH
    )
    emb = torch.tensor(warmup_results[0].outputs.embedding, dtype=torch.float32)
    proj = (emb @ axis_vector).item()
    logger.info(
        f"Warmup done in {time.time() - t0:.2f}s, embedding_dim={emb.shape[0]}, proj={proj:.4f}"
    )

    # Process FineWeb
    fw_output = OUTPUT_DIR / "fineweb_projections_fast.jsonl"
    fw_count = process_local_data(
        llm,
        fw_data_path,
        axis_vector,
        fw_output,
        batch_size=VLLM_BATCH,
        desc="FineWeb-Edu",
    )

    # Process LMSYS
    lmsys_output = OUTPUT_DIR / "lmsys_projections_fast.jsonl"
    lmsys_count = process_local_data(
        llm,
        lmsys_data_path,
        axis_vector,
        lmsys_output,
        batch_size=VLLM_BATCH,
        desc="LMSYS",
    )

    # ---- Summary ----
    total_time = time.time() - overall_start
    total_docs = fw_count + lmsys_count
    logger.info("=" * 60)
    logger.info(f"COMPLETE in {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info(f"  FineWeb: {fw_count:,} docs -> {fw_output}")
    logger.info(f"  LMSYS:   {lmsys_count:,} docs -> {lmsys_output}")
    logger.info(f"  Overall: {total_docs:,} docs, {total_docs / total_time:.1f} docs/sec")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
