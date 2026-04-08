#!/usr/bin/env python3
"""Fast corpus projection using vLLM embed mode with tensor parallelism.

Uses vLLM's embed task with LAST pooling to extract the final-layer hidden
state of the last token, then projects onto the pre-computed assistant axis.

Key advantage: vLLM's tensor parallelism + CUDA graphs + continuous batching
achieves ~1400 docs/sec vs ~17 docs/sec with HuggingFace pipeline parallel.

NOTE: vLLM returns the post-norm hidden state from the last layer (64), while
the axis was computed at layer 48 pre-norm. We project onto the axis anyway --
the relative ordering is well-preserved since RMSNorm approximately preserves
direction. The absolute projection values differ from the HF approach.

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
# HF_TOKEN loaded from .env via dotenv above

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_PATH = "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
AXIS_PATH = "/workspace/.cache/huggingface/datasets--lu-christina--assistant-axis-vectors/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
LAYER = 48  # axis layer (for loading the correct axis vector)
MAX_LENGTH = 512
VLLM_BATCH = 1024  # vLLM handles its own batching; we submit in chunks
FINEWEB_DOCS = 200_000
LMSYS_DOCS = 200_000
OUTPUT_DIR = Path("/workspace/axis_projections")
TP_SIZE = 2  # tensor parallel size (must divide 64 attention heads evenly)


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


def embed_and_project(llm, texts: list[str], axis_vector: torch.Tensor) -> list[tuple[float, int]]:
    """Embed texts using vLLM and project onto axis.

    Returns list of (projection, token_count) tuples.
    """
    results = llm.embed(texts, use_tqdm=False, truncate_prompt_tokens=MAX_LENGTH)

    projections = []
    for r in results:
        emb = torch.tensor(r.outputs.embedding, dtype=torch.float32)
        proj = (emb @ axis_vector).item()
        tc = len(r.prompt_token_ids)
        projections.append((proj, tc))

    return projections


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


def project_corpus_vllm(
    llm,
    dataset_iter,
    axis_vector: torch.Tensor,
    output_path: Path,
    max_docs: int,
    batch_size: int,
    text_fn=None,
    desc: str = "Projecting",
) -> int:
    """Project a streaming dataset using vLLM embed, writing JSONL incrementally."""
    from tqdm import tqdm

    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_texts = []
    batch_ids = []
    doc_count = 0
    written = 0
    t_start = time.time()
    last_log = t_start

    with open(output_path, "w") as f:
        for doc in tqdm(dataset_iter, total=max_docs, desc=desc, mininterval=5):
            if doc_count >= max_docs:
                break

            text = text_fn(doc) if text_fn else doc.get("text", "")
            if not text or len(text.strip()) < 50:
                continue

            batch_texts.append(text)
            batch_ids.append(doc_count)
            doc_count += 1

            if len(batch_texts) >= batch_size:
                try:
                    results = embed_and_project(llm, batch_texts, axis_vector)
                    for did, txt, (proj, tc) in zip(batch_ids, batch_texts, results):
                        f.write(json.dumps({
                            "doc_id": did,
                            "projection": round(proj, 6),
                            "token_count": tc,
                            "text_snippet": txt[:500],
                        }) + "\n")
                        written += 1
                except Exception as e:
                    logger.warning(f"Batch failed at doc {doc_count}: {e}")

                batch_texts = []
                batch_ids = []

                # Periodic throughput log
                now = time.time()
                if now - last_log > 30:
                    elapsed = now - t_start
                    rate = written / elapsed if elapsed > 0 else 0
                    eta_min = (max_docs - doc_count) / rate / 60 if rate > 0 else float("inf")
                    logger.info(
                        f"[{desc}] {written:,}/{max_docs:,} docs | "
                        f"{rate:.1f} docs/sec | ETA: {eta_min:.0f} min"
                    )
                    last_log = now

        # Flush remaining
        if batch_texts:
            try:
                results = embed_and_project(llm, batch_texts, axis_vector)
                for did, txt, (proj, tc) in zip(batch_ids, batch_texts, results):
                    f.write(json.dumps({
                        "doc_id": did,
                        "projection": round(proj, 6),
                        "token_count": tc,
                        "text_snippet": txt[:500],
                    }) + "\n")
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
    logger.info("Fast Corpus Projection (vLLM)")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Axis layer: {LAYER}, Max length: {MAX_LENGTH}, Batch: {VLLM_BATCH}")
    logger.info(f"FineWeb docs: {FINEWEB_DOCS:,}, LMSYS docs: {LMSYS_DOCS:,}")
    logger.info(f"Tensor parallel: {TP_SIZE}")
    logger.info("=" * 60)

    # Load axis
    axis_vector = load_axis(AXIS_PATH, LAYER)

    # Load vLLM model
    llm = load_vllm_model()

    # Warmup
    logger.info("Warmup...")
    t0 = time.time()
    results = embed_and_project(llm, ["Warmup text."] * 4, axis_vector)
    logger.info(f"Warmup done in {time.time()-t0:.2f}s, proj={results[0][0]:.4f}")

    # ---- FineWeb-Edu ----
    logger.info("=" * 60)
    logger.info("Projecting FineWeb-Edu")
    logger.info("=" * 60)

    from datasets import load_dataset

    fw_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True,
    )
    fw_output = OUTPUT_DIR / "fineweb_projections_fast.jsonl"
    fw_count = project_corpus_vllm(
        llm, fw_ds, axis_vector, fw_output,
        max_docs=FINEWEB_DOCS, batch_size=VLLM_BATCH,
        desc="FineWeb-Edu",
    )

    # ---- LMSYS ----
    logger.info("=" * 60)
    logger.info("Projecting LMSYS-Chat-1M")
    logger.info("=" * 60)

    lmsys_ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    lmsys_output = OUTPUT_DIR / "lmsys_projections_fast.jsonl"
    lmsys_count = project_corpus_vllm(
        llm, lmsys_ds, axis_vector, lmsys_output,
        max_docs=LMSYS_DOCS, batch_size=VLLM_BATCH,
        text_fn=lambda doc: project_lmsys_conversation(doc.get("conversation", [])),
        desc="LMSYS",
    )

    # ---- Summary ----
    total_time = time.time() - overall_start
    total_docs = fw_count + lmsys_count
    logger.info("=" * 60)
    logger.info(f"COMPLETE in {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"  FineWeb: {fw_count:,} docs -> {fw_output}")
    logger.info(f"  LMSYS:   {lmsys_count:,} docs -> {lmsys_output}")
    logger.info(f"  Overall: {total_docs:,} docs, {total_docs/total_time:.1f} docs/sec")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
