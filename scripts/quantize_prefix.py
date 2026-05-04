#!/usr/bin/env python3
"""Quantize #170 soft prefixes to nearest-vocabulary tokens.

For each of 7 soft-prefix cells from issue-170, loads the learned prefix
tensor (K x 3584) from HF Hub, finds the L2-nearest embedding-table token
for each row, decodes the resulting token IDs to a string, and saves the
mapping to JSON.

This is the code side of issue #240 Part A — the experimenter runs the
actual eval on a pod.

Usage:
    uv run python scripts/quantize_prefix.py \
        --output-dir eval_results/issue-240/quantized
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap()

import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

HF_REPO = "superkaiba1/explore-persona-space"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SOFT_CELLS = [
    "s0_K16_lr5e-4",
    "s1_K32_lr5e-4",
    "s2_K32_lr1e-4",
    "s3_K64_lr5e-4",
    "s4_K64_lr1e-4",
    "s5_K64_lr1e-3",
    "s6_K64_lr5e-4_evil_init",
]


def parse_args():
    p = argparse.ArgumentParser(description="Quantize soft prefixes to nearest vocab tokens")
    p.add_argument(
        "--output-dir",
        default="eval_results/issue-240/quantized",
        help="Directory for quantized output JSON",
    )
    p.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Base model for embedding table",
    )
    p.add_argument(
        "--hf-repo",
        default=HF_REPO,
        help="HF Hub repo containing prefix tensors",
    )
    return p.parse_args()


def download_prefix(hf_repo: str, cell: str) -> dict:
    """Download a prefix checkpoint from HF Hub and load it."""
    filename = f"issue-170/{cell}/prefix_step3000.pt"
    local_path = hf_hub_download(
        repo_id=hf_repo,
        filename=filename,
        token=os.environ.get("HF_TOKEN"),
    )
    return torch.load(local_path, map_location="cpu", weights_only=False)


def quantize_prefix_to_tokens(
    prefix_tensor: torch.Tensor,
    embed_table: torch.Tensor,
) -> tuple[list[int], list[float]]:
    """Find the L2-nearest embedding-table token for each prefix row.

    Args:
        prefix_tensor: (K, hidden_dim) learned prefix embeddings.
        embed_table: (vocab_size, hidden_dim) base model embedding table.

    Returns:
        Tuple of (token_ids, l2_distances).
    """
    # Normalize both to float32 for numerics
    prefix = prefix_tensor.float()
    embed = embed_table.float()

    token_ids = []
    l2_distances = []
    for k in range(prefix.shape[0]):
        # Broadcast: (vocab_size, hidden_dim) - (1, hidden_dim) -> (vocab_size, hidden_dim)
        dists = torch.norm(embed - prefix[k].unsqueeze(0), dim=1)  # (vocab_size,)
        min_idx = int(dists.argmin().item())
        min_dist = float(dists[min_idx].item())
        token_ids.append(min_idx)
        l2_distances.append(min_dist)

    return token_ids, l2_distances


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load base model embedding table (CPU only — no GPU needed for quantization).
    logger.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    logger.info("Loading base model embedding table from %s", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        device_map="cpu",
    )
    embed_table = model.get_input_embeddings().weight.detach()  # (vocab_size, hidden_dim)
    logger.info("Embedding table shape: %s", embed_table.shape)

    # Free model memory — we only need the embedding table
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results: dict[str, dict] = {}

    for cell in SOFT_CELLS:
        logger.info("Processing cell %s", cell)

        # Download and load prefix checkpoint
        ckpt = download_prefix(args.hf_repo, cell)
        prefix_tensor = ckpt["prefix"]  # (K, hidden_dim) tensor
        k = int(ckpt["k"])
        hidden_dim = int(ckpt["hidden_dim"])
        init_text = str(ckpt.get("init_text", "<unknown>"))
        logger.info(
            "  Cell %s: K=%d, hidden_dim=%d, init_text=%r",
            cell,
            k,
            hidden_dim,
            init_text,
        )

        if prefix_tensor.shape != (k, hidden_dim):
            raise RuntimeError(
                f"Prefix shape mismatch: expected ({k}, {hidden_dim}), got {prefix_tensor.shape}"
            )

        # Quantize
        token_ids, l2_distances = quantize_prefix_to_tokens(prefix_tensor, embed_table)
        decoded_string = tokenizer.decode(token_ids)
        mean_l2 = sum(l2_distances) / len(l2_distances)

        # Check if re-tokenization matches (BPE divergence expected)
        re_encoded = tokenizer.encode(decoded_string, add_special_tokens=False)
        bpe_matches = re_encoded == token_ids

        logger.info("  Quantized string: %r", decoded_string[:100])
        logger.info(
            "  Token IDs (first 10): %s",
            token_ids[:10],
        )
        logger.info("  Mean L2 distance: %.4f", mean_l2)
        logger.info("  BPE re-tokenization matches: %s", bpe_matches)

        results[cell] = {
            "token_ids": token_ids,
            "decoded_string": decoded_string,
            "l2_distances": l2_distances,
            "mean_l2": mean_l2,
            "k": k,
            "hidden_dim": hidden_dim,
            "init_text": init_text,
            "bpe_matches_original": bpe_matches,
            "re_encoded_ids": re_encoded,
        }

    # Save combined results
    out_path = output_dir / "quantized_strings.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved quantized results to %s", out_path)

    dt = time.time() - t0
    logger.info("Quantization completed in %.1fs", dt)

    # Print summary table
    print("\n=== Quantization Summary ===")
    print(f"{'Cell':<30} {'K':>3} {'Mean L2':>8} {'BPE match':>10}")
    print("-" * 55)
    for cell, r in results.items():
        print(
            f"{cell:<30} {r['k']:>3} {r['mean_l2']:>8.4f} "
            f"{'YES' if r['bpe_matches_original'] else 'NO':>10}"
        )


if __name__ == "__main__":
    main()
