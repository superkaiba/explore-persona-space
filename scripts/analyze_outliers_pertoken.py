#!/usr/bin/env python3
"""Per-token analysis of extreme outlier documents.

Runs the 2 extreme LMSYS outliers (proj=-2968, -2380) through the model
and computes per-token contribution to the axis projection.

Must run on GPU (needs model inference).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/analyze_outliers_pertoken.py \
        --output_dir /workspace/axis_projections_v2/outlier_analysis
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("HF_HOME", "/workspace/cache/huggingface")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AXIS_PATH = (
    "/workspace/cache/huggingface/hub/datasets--lu-christina--assistant-axis-vectors"
    "/snapshots/3b3b788432ad33e3a28d9ff08e88a530c0740814/qwen-3-32b/assistant_axis.pt"
)
MODEL_PATH = "Qwen/Qwen3-32B"
LAYER = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="/workspace/axis_projections_v2/outlier_analysis")
    p.add_argument("--model_path", default=MODEL_PATH)
    p.add_argument("--axis_path", default=AXIS_PATH)
    p.add_argument("--layer", type=int, default=LAYER)
    return p.parse_args()


def load_outlier_texts(projections_path: str, n_outliers: int = 5):
    """Load the most extreme outlier documents (lowest projection)."""
    docs = []
    with open(projections_path) as f:
        for line in f:
            docs.append(json.loads(line))

    # Sort by projection, take the most extreme (lowest)
    docs.sort(key=lambda d: d["projection"])
    outliers = docs[:n_outliers]

    logger.info(f"Loaded {n_outliers} outliers from {projections_path}")
    for d in outliers:
        logger.info(f"  doc_id={d['doc_id']}, proj={d['projection']:.2f}, tc={d['token_count']}")

    return outliers


def get_full_text(raw_data_path: str, doc_ids: set) -> dict:
    """Get full text for specific doc IDs from raw data."""
    texts = {}
    with open(raw_data_path) as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] in doc_ids:
                texts[d["doc_id"]] = d["text"]
    return texts


def pertoken_projection(model, tokenizer, text: str, axis_vector, layer_idx: int):
    """Compute per-token contribution to axis projection.

    For each token position, computes hidden_state[pos] @ axis_vector.
    This shows which tokens push the representation toward/away from
    the assistant direction.
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings["input_ids"].to(model.device)

    activations = {}

    def hook_fn(module, inp, output):
        act = output[0] if isinstance(output, tuple) else output
        activations["hidden"] = act.detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids)
    handle.remove()

    hidden = activations["hidden"][0]  # (seq_len, hidden_dim)
    ax = axis_vector.to(hidden.device, hidden.dtype)

    # Per-token projection
    per_token = (hidden.float() @ ax.float()).cpu().tolist()

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Cumulative projection (running sum)
    cumulative = []
    running = 0.0
    for p in per_token:
        running += p
        cumulative.append(running)

    # Last-token projection (what we report as the doc's projection)
    last_token_proj = per_token[-1]
    mean_proj = sum(per_token) / len(per_token)

    return {
        "tokens": tokens,
        "per_token_projection": [round(p, 4) for p in per_token],
        "cumulative_projection": [round(c, 4) for c in cumulative],
        "last_token_projection": round(last_token_proj, 4),
        "mean_token_projection": round(mean_proj, 4),
        "sum_projection": round(sum(per_token), 4),
        "n_tokens": len(tokens),
        "max_token": {
            "position": int(max(range(len(per_token)), key=lambda i: per_token[i])),
            "value": round(max(per_token), 4),
            "token": tokens[max(range(len(per_token)), key=lambda i: per_token[i])],
        },
        "min_token": {
            "position": int(min(range(len(per_token)), key=lambda i: per_token[i])),
            "value": round(min(per_token), 4),
            "token": tokens[min(range(len(per_token)), key=lambda i: per_token[i])],
        },
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load axis
    data = torch.load(args.axis_path, map_location="cpu", weights_only=False)
    axis = data[args.layer].float()
    axis = axis / axis.norm()
    logger.info(f"Axis loaded: layer {args.layer}, norm={axis.norm().item():.4f}")

    # Load outlier docs
    lmsys_outliers = load_outlier_texts(
        "/workspace/axis_projections_v2/lmsys_projections.jsonl", n_outliers=5
    )
    fw_outliers = load_outlier_texts(
        "/workspace/axis_projections_v2/fineweb_projections.jsonl", n_outliers=5
    )

    # Get full text
    {d["doc_id"] for d in lmsys_outliers} | {d["doc_id"] for d in fw_outliers}
    lmsys_texts = get_full_text(
        "/workspace/axis_projections_v2/raw_data/lmsys_raw.jsonl",
        {d["doc_id"] for d in lmsys_outliers},
    )
    fw_texts = get_full_text(
        "/workspace/axis_projections_v2/raw_data/fineweb_raw.jsonl",
        {d["doc_id"] for d in fw_outliers},
    )

    # Also get a few "normal" docs for comparison
    logger.info("Loading comparison docs (near median)...")
    all_lmsys = []
    with open("/workspace/axis_projections_v2/lmsys_projections.jsonl") as f:
        for line in f:
            all_lmsys.append(json.loads(line))
    projs = sorted(all_lmsys, key=lambda d: d["projection"])
    mid = len(projs) // 2
    median_docs = projs[mid - 2 : mid + 3]
    median_texts = get_full_text(
        "/workspace/axis_projections_v2/raw_data/lmsys_raw.jsonl",
        {d["doc_id"] for d in median_docs},
    )

    # Load model (HF, single GPU for per-token analysis)
    logger.info(f"Loading model {args.model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded")

    # Run per-token analysis
    results = {"lmsys_outliers": [], "fineweb_outliers": [], "lmsys_median": []}

    for doc in lmsys_outliers:
        text = lmsys_texts.get(doc["doc_id"], doc["text_snippet"])
        logger.info(f"\nLMSYS outlier doc_id={doc['doc_id']}, proj={doc['projection']:.2f}")
        analysis = pertoken_projection(model, tokenizer, text, axis, args.layer)
        analysis["doc_id"] = doc["doc_id"]
        analysis["original_projection"] = doc["projection"]
        analysis["text_preview"] = text[:300]
        results["lmsys_outliers"].append(analysis)

        # Show top contributing tokens
        token_contribs = list(zip(analysis["tokens"], analysis["per_token_projection"]))
        token_contribs.sort(key=lambda x: x[1])
        logger.info("  Most negative tokens (pushing AWAY from assistant):")
        for tok, val in token_contribs[:5]:
            logger.info(f"    {tok!r}: {val:.4f}")
        logger.info("  Most positive tokens (pushing TOWARD assistant):")
        for tok, val in token_contribs[-5:]:
            logger.info(f"    {tok!r}: {val:.4f}")

    for doc in fw_outliers:
        text = fw_texts.get(doc["doc_id"], doc["text_snippet"])
        logger.info(f"\nFineWeb outlier doc_id={doc['doc_id']}, proj={doc['projection']:.2f}")
        analysis = pertoken_projection(model, tokenizer, text, axis, args.layer)
        analysis["doc_id"] = doc["doc_id"]
        analysis["original_projection"] = doc["projection"]
        analysis["text_preview"] = text[:300]
        results["fineweb_outliers"].append(analysis)

    for doc in median_docs:
        text = median_texts.get(doc["doc_id"], doc["text_snippet"])
        logger.info(f"\nLMSYS median doc_id={doc['doc_id']}, proj={doc['projection']:.2f}")
        analysis = pertoken_projection(model, tokenizer, text, axis, args.layer)
        analysis["doc_id"] = doc["doc_id"]
        analysis["original_projection"] = doc["projection"]
        analysis["text_preview"] = text[:300]
        results["lmsys_median"].append(analysis)

    # Save results
    with open(output_dir / "pertoken_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}/pertoken_analysis.json")

    # Summary comparison
    logger.info("\n=== SUMMARY ===")
    for group in ["lmsys_outliers", "lmsys_median"]:
        logger.info(f"\n{group}:")
        for r in results[group]:
            logger.info(
                f"  doc_id={r['doc_id']}: "
                f"last_tok={r['last_token_projection']:.2f}, "
                f"mean_tok={r['mean_token_projection']:.2f}, "
                f"sum={r['sum_projection']:.2f}, "
                f"min_tok={r['min_token']['value']:.2f} ({r['min_token']['token']!r}), "
                f"max_tok={r['max_token']['value']:.2f} ({r['max_token']['token']!r})"
            )


if __name__ == "__main__":
    main()
