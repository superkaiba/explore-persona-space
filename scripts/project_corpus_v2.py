#!/usr/bin/env python3
"""Project FineWeb-Edu and LMSYS onto the assistant axis using Speculators + vLLM.

Fixes all 4 issues from v1:
  1. Layer mismatch: extracts from layer 32 (matching Lu et al.) via speculators
  2. Length confound: saves hidden vectors for post-hoc residualization
  3. Incomplete data: uses speculators (no vLLM hang), checkpoint/resume
  4. LMSYS comparison: processes both corpora, strips role markers

Usage:
    CUDA_VISIBLE_DEVICES=0,1 nohup uv run python scripts/project_corpus_v2.py \
        --output_dir /workspace/axis_projections_v2 \
        > /workspace/projection_v2_log.txt 2>&1 &
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Must be set before vLLM imports
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_PATH = "Qwen/Qwen3-32B"
AXIS_SNAPSHOT = (
    "/workspace/.cache/huggingface/datasets--lu-christina--assistant-axis-vectors"
    "/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
)
LAYER = 32  # Lu et al. target_layer for Qwen3-32B
MAX_LENGTH = 512
BATCH_SIZE = 32  # speculators captures max 32 seqs per call
FINEWEB_DOCS = 200_000
LMSYS_DOCS = 200_000
TP_SIZE = 2
HIDDEN_SAMPLE_SIZE = 5000  # docs per corpus to save full hidden vectors


def parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=MODEL_PATH)
    p.add_argument("--axis_path", default=AXIS_SNAPSHOT)
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--output_dir", default="/workspace/axis_projections_v2")
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--fineweb_docs", type=int, default=FINEWEB_DOCS)
    p.add_argument("--lmsys_docs", type=int, default=LMSYS_DOCS)
    p.add_argument("--tp_size", type=int, default=TP_SIZE)
    p.add_argument("--hidden_sample_size", type=int, default=HIDDEN_SAMPLE_SIZE)
    p.add_argument("--skip_fineweb", action="store_true")
    p.add_argument("--skip_lmsys", action="store_true")
    p.add_argument(
        "--layer_sweep",
        action="store_true",
        help="Run projection at layers 16, 32, 48 on 10K sample",
    )
    return p.parse_args()


# =====================================================================
# Axis loading & validation
# =====================================================================


def load_and_validate_axis(axis_path: str, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load axis file, validate shape, return (full_axis_data, normalized_layer_vector)."""
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    axis = data["axis"] if isinstance(data, dict) and "axis" in data else data

    # Validate
    assert axis.ndim == 2, f"Expected 2D axis tensor, got {axis.ndim}D"
    assert axis.shape[0] == 64, f"Expected 64 layers, got {axis.shape[0]}"
    assert axis.shape[1] == 5120, f"Expected dim 5120, got {axis.shape[1]}"

    ax = axis[layer].float()
    norm = ax.norm().item()
    assert norm > 0.01, f"Axis at layer {layer} is near-zero (norm={norm})"
    assert ax.std().item() > 1e-6, f"Axis at layer {layer} is degenerate (std={ax.std().item()})"

    ax = ax / norm
    logger.info(f"Axis validated: shape={axis.shape}, layer {layer} norm={norm:.4f}")
    return axis, ax


# =====================================================================
# Data download
# =====================================================================


def download_fineweb(output_path: Path, max_docs: int) -> int:
    """Stream FineWeb-Edu to local JSONL."""
    from datasets import load_dataset
    from tqdm import tqdm

    if output_path.exists():
        with open(output_path) as _f:
            existing = sum(1 for _ in _f)
        if existing >= max_docs:
            logger.info(f"FineWeb already downloaded: {existing:,} docs")
            return existing

    logger.info(f"Downloading {max_docs:,} FineWeb-Edu docs...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    count = 0
    with open(output_path, "w") as f:
        for doc in tqdm(ds, total=max_docs, desc="FineWeb download"):
            if count >= max_docs:
                break
            text = doc.get("text", "")
            if not text or len(text.strip()) < 50:
                continue
            f.write(json.dumps({"doc_id": count, "text": text[:5000]}) + "\n")
            count += 1

    logger.info(f"FineWeb download complete: {count:,} docs")
    return count


def extract_lmsys_content(conversation: list[dict]) -> str:
    """Extract first user+assistant turn as plain text, WITHOUT role markers.

    Strips "User:" and "Assistant:" prefixes to avoid the literal tokens
    activating the assistant axis.
    """
    parts = []
    seen_user = seen_assistant = False
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user" and not seen_user:
            parts.append(content)  # No "User:" prefix
            seen_user = True
        elif role == "assistant" and not seen_assistant and seen_user:
            parts.append(content)  # No "Assistant:" prefix
            seen_assistant = True
            break
    return "\n\n".join(parts)


def download_lmsys(output_path: Path, max_docs: int) -> int:
    """Stream LMSYS-Chat-1M to local JSONL (no role markers)."""
    from datasets import load_dataset
    from tqdm import tqdm

    if output_path.exists():
        with open(output_path) as _f:
            existing = sum(1 for _ in _f)
        if existing >= max_docs:
            logger.info(f"LMSYS already downloaded: {existing:,} docs")
            return existing

    logger.info(f"Downloading {max_docs:,} LMSYS docs...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    count = 0
    with open(output_path, "w") as f:
        for doc in tqdm(ds, total=max_docs, desc="LMSYS download"):
            if count >= max_docs:
                break
            text = extract_lmsys_content(doc.get("conversation", []))
            if not text or len(text.strip()) < 50:
                continue
            f.write(json.dumps({"doc_id": count, "text": text[:5000]}) + "\n")
            count += 1

    logger.info(f"LMSYS download complete: {count:,} docs")
    return count


# =====================================================================
# Speculators-based projection
# =====================================================================


def init_generator(model_path: str, layer_ids: list[int], tp_size: int, max_length: int):
    """Initialize speculators VllmHiddenStatesGenerator."""
    # PROD_IMPORT_LINT_EXEMPT: one-off corpus-builder dep 'speculators', not a lockfile dist
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator,
    )

    logger.info(
        f"Loading speculators generator: model={model_path}, layers={layer_ids}, TP={tp_size}"
    )
    t0 = time.time()
    generator = VllmHiddenStatesGenerator(
        model_path=model_path,
        layer_ids=layer_ids,
        max_model_len=max_length,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=tp_size,
        max_num_batched_tokens=max_length * 128,  # accommodate large batches
    )
    logger.info(f"Generator loaded in {time.time() - t0:.1f}s")
    return generator


def project_batch(
    generator,
    tokenizer,
    texts: list[str],
    axis_vector: torch.Tensor,
    max_length: int,
    pooling: str = "last",
) -> list[tuple[float, int, torch.Tensor | None]]:
    """Project a batch of texts onto the axis using speculators.

    Returns list of (projection, token_count, hidden_vector_or_None).
    hidden_vector is returned only for the first `save_hidden` docs.
    """
    # Tokenize (no padding — speculators handles variable lengths internally)
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )
    token_ids_batch = encoded["input_ids"]  # list[list[int]]

    # Get hidden states from speculators
    results = generator.generate(token_ids=token_ids_batch)

    output = []
    for result in results:
        hidden = result["hidden_states"][0]  # (seq_len, hidden_dim) for our single layer
        seq_len = hidden.shape[0]

        if pooling == "last":
            pooled = hidden[-1]  # last token
        elif pooling == "mean":
            pooled = hidden.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        proj = (pooled.float() @ axis_vector.float()).item()
        output.append((proj, seq_len, pooled))

    return output


def run_pilot(generator, tokenizer, data_path, axis_vector, max_length, batch_size, n_pilot=1000):
    """Run a timed pilot to estimate throughput and compare pooling methods."""
    logger.info(f"Running {n_pilot}-doc pilot...")

    texts = []
    with open(data_path) as f:
        for line in f:
            if len(texts) >= n_pilot:
                break
            doc = json.loads(line)
            text = doc.get("text", "")
            if text and len(text.strip()) >= 50:
                texts.append(text[:2000])

    # Time last-token pooling
    t0 = time.time()
    last_projs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = project_batch(
            generator, tokenizer, batch, axis_vector, max_length, pooling="last"
        )
        last_projs.extend([(p, tc) for p, tc, _ in results])
    last_time = time.time() - t0
    last_rate = len(texts) / last_time

    # Time mean pooling
    t0 = time.time()
    mean_projs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = project_batch(
            generator, tokenizer, batch, axis_vector, max_length, pooling="mean"
        )
        mean_projs.extend([(p, tc) for p, tc, _ in results])
    _ = time.time() - t0  # mean pooling time (not used for throughput estimate)

    # Compare pooling methods
    last_vals = np.array([p for p, _ in last_projs])
    mean_vals = np.array([p for p, _ in mean_projs])
    tcs = np.array([tc for _, tc in last_projs])

    from scipy.stats import pearsonr

    last_r, _ = pearsonr(last_vals, tcs)
    mean_r, _ = pearsonr(mean_vals, tcs)
    cross_r, _ = pearsonr(last_vals, mean_vals)

    logger.info(
        f"Pilot throughput: {last_rate:.1f} docs/sec ({last_time:.1f}s for {len(texts)} docs)"
    )
    logger.info(f"Pooling comparison on {len(texts)} docs:")
    logger.info(
        f"  Last-token: mean={last_vals.mean():.4f}, std={last_vals.std():.4f}, length_r={last_r:.4f}"
    )
    logger.info(
        f"  Mean-pool:  mean={mean_vals.mean():.4f}, std={mean_vals.std():.4f}, length_r={mean_r:.4f}"
    )
    logger.info(f"  Cross-correlation: r={cross_r:.4f}")

    # Choose pooling with lower length correlation
    chosen = "mean" if abs(mean_r) < abs(last_r) else "last"
    logger.info(f"  Chosen pooling: {chosen} (lower |length_r|)")

    return {
        "rate": last_rate,
        "chosen_pooling": chosen,
        "last_length_r": float(last_r),
        "mean_length_r": float(mean_r),
        "cross_r": float(cross_r),
    }


def project_corpus(
    generator,
    tokenizer,
    data_path: Path,
    axis_vector: torch.Tensor,
    output_path: Path,
    hidden_path: Path | None,
    max_docs: int,
    batch_size: int,
    max_length: int,
    pooling: str,
    hidden_sample_size: int,
    desc: str = "Projecting",
) -> int:
    """Project a local JSONL corpus with checkpoint/resume support."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing progress (resume)
    skip_count = 0
    if output_path.exists():
        with open(output_path) as _f:
            skip_count = sum(1 for _ in _f)
        if skip_count >= max_docs:
            logger.info(f"[{desc}] Already complete: {skip_count:,} docs")
            return skip_count
        logger.info(f"[{desc}] Resuming from {skip_count:,} existing docs")

    # Count total docs in input
    with open(data_path) as _f:
        total_input = sum(1 for _ in _f)
    target = min(max_docs, total_input)
    logger.info(f"[{desc}] Processing {target:,} docs (skip={skip_count:,})")

    # Hidden vector sampling
    hidden_f = None
    hidden_written = 0
    if hidden_path and skip_count < hidden_sample_size:
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_count > 0 and hidden_path.exists():
            with open(hidden_path) as _f:
                hidden_written = sum(1 for _ in _f)
        hidden_f = open(hidden_path, "a" if skip_count > 0 else "w")  # noqa: SIM115

    mode = "a" if skip_count > 0 else "w"
    written = skip_count
    doc_idx = 0
    batch_texts, batch_ids, batch_snippets = [], [], []
    failed_batches = 0
    total_batches = 0
    truncated_count = 0
    t_start = time.time()
    last_log = t_start

    with open(output_path, mode) as out_f, open(data_path) as in_f:
        for line in in_f:
            if written >= max_docs:
                break
            doc = json.loads(line)
            doc_idx += 1
            if doc_idx <= skip_count:
                continue

            text = doc["text"]
            batch_texts.append(text[:2000])
            batch_ids.append(doc["doc_id"])
            batch_snippets.append(text[:500])

            if len(batch_texts) >= batch_size:
                total_batches += 1
                try:
                    results = project_batch(
                        generator,
                        tokenizer,
                        batch_texts,
                        axis_vector,
                        max_length,
                        pooling,
                    )
                    for did, snippet, (proj, tc, hidden_vec) in zip(
                        batch_ids, batch_snippets, results
                    ):
                        if tc >= max_length:
                            truncated_count += 1
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
                        # Save hidden vector for random direction control
                        if hidden_f and hidden_written < hidden_sample_size:
                            hidden_f.write(
                                json.dumps(
                                    {
                                        "doc_id": did,
                                        "hidden": hidden_vec.float().tolist(),
                                    }
                                )
                                + "\n"
                            )
                            hidden_written += 1
                        written += 1
                except Exception as e:
                    failed_batches += 1
                    logger.warning(f"[{desc}] Batch failed at doc {written}: {e}")
                    if failed_batches > total_batches * 0.01 and total_batches > 10:
                        logger.error(
                            f"[{desc}] >1% batch failures ({failed_batches}/{total_batches}). Aborting."
                        )
                        break

                batch_texts, batch_ids, batch_snippets = [], [], []

                # Periodic log
                now = time.time()
                if now - last_log > 30:
                    elapsed = now - t_start
                    rate = (written - skip_count) / elapsed if elapsed > 0 else 0
                    remaining = target - written
                    eta_min = remaining / rate / 60 if rate > 0 else float("inf")
                    logger.info(
                        f"[{desc}] {written:,}/{target:,} | {rate:.1f} docs/sec | "
                        f"ETA: {eta_min:.0f}min | trunc: {truncated_count}"
                    )
                    last_log = now

        # Flush remaining batch
        if batch_texts:
            total_batches += 1
            try:
                results = project_batch(
                    generator,
                    tokenizer,
                    batch_texts,
                    axis_vector,
                    max_length,
                    pooling,
                )
                for did, snippet, (proj, tc, hidden_vec) in zip(batch_ids, batch_snippets, results):
                    if tc >= max_length:
                        truncated_count += 1
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
                    if hidden_f and hidden_written < hidden_sample_size:
                        hidden_f.write(
                            json.dumps({"doc_id": did, "hidden": hidden_vec.float().tolist()})
                            + "\n"
                        )
                        hidden_written += 1
                    written += 1
            except Exception as e:
                logger.warning(f"[{desc}] Final batch failed: {e}")

    if hidden_f:
        hidden_f.close()

    elapsed = time.time() - t_start
    rate = (written - skip_count) / elapsed if elapsed > 0 else 0
    trunc_pct = truncated_count / max(written, 1) * 100
    logger.info(
        f"[{desc}] Complete: {written:,} docs in {elapsed:.0f}s ({rate:.1f} docs/sec) | "
        f"truncated: {truncated_count} ({trunc_pct:.1f}%) | failed batches: {failed_batches}"
    )
    return written


def run_layer_sweep(
    generator_fn, tokenizer, data_path, full_axis, max_length, batch_size, n_docs=10_000
):
    """Quick sweep over layers 16, 32, 48 on a small sample."""
    logger.info(f"Layer sweep on {n_docs} docs...")

    # Load sample texts
    texts = []
    with open(data_path) as f:
        for line in f:
            if len(texts) >= n_docs:
                break
            doc = json.loads(line)
            text = doc.get("text", "")
            if text and len(text.strip()) >= 50:
                texts.append(text[:2000])

    results = {}
    for layer in [16, 32, 48]:
        ax = full_axis[layer].float()
        ax = ax / (ax.norm() + 1e-8)

        # Re-init generator for this layer
        gen = generator_fn(layer_ids=[layer])

        projs = []
        tcs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = project_batch(gen, tokenizer, batch, ax, max_length, pooling="last")
            for p, tc, _ in batch_results:
                projs.append(p)
                tcs.append(tc)

        projs_arr = np.array(projs)
        tcs_arr = np.array(tcs)
        from scipy.stats import pearsonr

        r, _ = pearsonr(projs_arr, tcs_arr)
        results[layer] = {
            "mean": float(projs_arr.mean()),
            "std": float(projs_arr.std()),
            "length_r": float(r),
        }
        logger.info(
            f"  Layer {layer}: mean={projs_arr.mean():.4f}, std={projs_arr.std():.4f}, length_r={r:.4f}"
        )

        del gen
        torch.cuda.empty_cache()

    return results


# =====================================================================
# Main
# =====================================================================


def main():
    args = parse_args()
    overall_start = time.time()
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "raw_data"

    logger.info("=" * 60)
    logger.info("Corpus Projection v2 (Speculators + vLLM)")
    logger.info(f"Model: {args.model_path}")
    logger.info(
        f"Axis layer: {args.layer}, Max length: {args.max_length}, Batch: {args.batch_size}"
    )
    logger.info(f"FineWeb: {args.fineweb_docs:,}, LMSYS: {args.lmsys_docs:,}, TP: {args.tp_size}")
    logger.info("=" * 60)

    # ---- Load & validate axis ----
    full_axis, axis_vector = load_and_validate_axis(args.axis_path, args.layer)

    # ---- Phase 1: Download data ----
    logger.info("=" * 60)
    logger.info("PHASE 1: Download datasets")
    logger.info("=" * 60)

    fw_data = data_dir / "fineweb_raw.jsonl"
    lmsys_data = data_dir / "lmsys_raw.jsonl"

    if not args.skip_fineweb:
        download_fineweb(fw_data, args.fineweb_docs)
    if not args.skip_lmsys:
        download_lmsys(lmsys_data, args.lmsys_docs)

    # ---- Phase 2: Load generator ----
    logger.info("=" * 60)
    logger.info("PHASE 2: Initialize speculators generator")
    logger.info("=" * 60)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = init_generator(args.model_path, [args.layer], args.tp_size, args.max_length)

    # ---- Phase 3: Pilot ----
    logger.info("=" * 60)
    logger.info("PHASE 3: Pilot (1K docs, pooling comparison)")
    logger.info("=" * 60)

    pilot_data = fw_data if fw_data.exists() else lmsys_data
    pilot = run_pilot(
        generator, tokenizer, pilot_data, axis_vector, args.max_length, args.batch_size
    )

    # Estimate total runtime
    total_docs = (args.fineweb_docs if not args.skip_fineweb else 0) + (
        args.lmsys_docs if not args.skip_lmsys else 0
    )
    est_hours = total_docs / pilot["rate"] / 3600
    logger.info(f"Estimated total runtime: {est_hours:.1f}h at {pilot['rate']:.1f} docs/sec")
    if est_hours > 12:
        logger.error(f"Estimated runtime {est_hours:.1f}h exceeds 12h. Aborting.")
        sys.exit(1)

    pooling = pilot["chosen_pooling"]

    # Save pilot results
    pilot_path = output_dir / "pilot_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(pilot_path, "w") as f:
        json.dump(pilot, f, indent=2)

    # ---- Phase 4: Project FineWeb ----
    if not args.skip_fineweb:
        logger.info("=" * 60)
        logger.info("PHASE 4: Project FineWeb-Edu")
        logger.info("=" * 60)

        project_corpus(
            generator,
            tokenizer,
            fw_data,
            axis_vector,
            output_dir / "fineweb_projections.jsonl",
            output_dir / "hidden_vectors" / "fineweb_hidden.jsonl",
            args.fineweb_docs,
            args.batch_size,
            args.max_length,
            pooling,
            args.hidden_sample_size,
            desc="FineWeb-Edu",
        )

    # ---- Phase 5: Project LMSYS ----
    if not args.skip_lmsys:
        logger.info("=" * 60)
        logger.info("PHASE 5: Project LMSYS")
        logger.info("=" * 60)

        project_corpus(
            generator,
            tokenizer,
            lmsys_data,
            axis_vector,
            output_dir / "lmsys_projections.jsonl",
            output_dir / "hidden_vectors" / "lmsys_hidden.jsonl",
            args.lmsys_docs,
            args.batch_size,
            args.max_length,
            pooling,
            args.hidden_sample_size,
            desc="LMSYS",
        )

    # ---- Phase 6: Layer sweep (optional) ----
    if args.layer_sweep and fw_data.exists():
        logger.info("=" * 60)
        logger.info("PHASE 6: Layer sweep (16, 32, 48)")
        logger.info("=" * 60)

        # Need to recreate generators for different layers
        del generator
        torch.cuda.empty_cache()

        def make_generator(layer_ids):
            return init_generator(args.model_path, layer_ids, args.tp_size, args.max_length)

        sweep = run_layer_sweep(
            make_generator, tokenizer, fw_data, full_axis, args.max_length, args.batch_size
        )
        with open(output_dir / "layer_sweep.json", "w") as f:
            json.dump(sweep, f, indent=2)

    # ---- Phase 7: Analysis ----
    logger.info("=" * 60)
    logger.info("PHASE 7: Analysis")
    logger.info("=" * 60)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from explore_persona_space.axis.analyze import load_projections, run_full_analysis

    corpora = {}
    fw_proj_path = output_dir / "fineweb_projections.jsonl"
    lmsys_proj_path = output_dir / "lmsys_projections.jsonl"
    if fw_proj_path.exists():
        corpora["fineweb_edu"] = load_projections(str(fw_proj_path))
    if lmsys_proj_path.exists():
        corpora["lmsys"] = load_projections(str(lmsys_proj_path))

    if corpora:
        analysis_dir = output_dir / "analysis"
        summary = run_full_analysis(corpora, str(analysis_dir))

        # Random direction control
        fw_hidden = output_dir / "hidden_vectors" / "fineweb_hidden.jsonl"
        lmsys_hidden = output_dir / "hidden_vectors" / "lmsys_hidden.jsonl"
        if fw_hidden.exists() and lmsys_hidden.exists():
            from explore_persona_space.axis.analyze import random_direction_control

            rdc = random_direction_control(str(fw_hidden), str(lmsys_hidden), axis_vector.numpy())
            summary["random_direction_control"] = rdc
            logger.info(
                f"Random direction control: real_d={rdc['real_axis_cohens_d']:.4f}, "
                f"random_mean_d={rdc['random_mean_d']:.4f}, z={rdc['z_score']:.4f}"
            )

        # Effect sizes
        if "fineweb_edu" in corpora and "lmsys" in corpora:
            from explore_persona_space.axis.analyze import compute_effect_sizes

            for field in ["projection"]:
                es = compute_effect_sizes(corpora["fineweb_edu"], corpora["lmsys"], field=field)
                summary[f"effect_sizes_{field}"] = es
                logger.info(f"Effect sizes ({field}): Cohen's d={es['cohens_d']:.4f}")

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    total_time = time.time() - overall_start
    logger.info("=" * 60)
    logger.info(f"ALL DONE in {total_time:.0f}s ({total_time / 60:.1f} min)")
    logger.info(f"Results in {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
