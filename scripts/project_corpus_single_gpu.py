#!/usr/bin/env python3
"""Project FineWeb-Edu and LMSYS onto the assistant axis using a single GPU.

Loads the model once, processes both corpora sequentially. Much simpler and
faster than multi-GPU for large models where loading is the bottleneck.

Usage:
    CUDA_VISIBLE_DEVICES=3 nohup uv run python scripts/project_corpus_single_gpu.py \
        --axis_path /path/to/assistant_axis.pt \
        --layer 48 \
        --base_model Qwen/Qwen3-32B \
        --output_dir /workspace/axis_projections \
        --fineweb_docs 2000000 &
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_axis(axis_path: str, layer: int) -> torch.Tensor:
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "axis" in data:
        axis = data["axis"]
    else:
        axis = data
    ax = axis[layer].float()
    ax = ax / (ax.norm() + 1e-8)
    return ax


def extract_and_project_batch(model, tokenizer, texts, axis_vector, layer_idx, max_length=512):
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    handle.remove()

    hidden = activations["hidden"]
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    last_token_acts = hidden[batch_indices, seq_lengths]

    ax = axis_vector.to(last_token_acts.device).to(last_token_acts.dtype)
    projections = (last_token_acts.float() @ ax.float()).cpu().tolist()
    token_counts = seq_lengths.add(1).cpu().tolist()

    return list(zip(projections, [int(tc) for tc in token_counts]))


def project_lmsys_conversation(conversation):
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


def project_corpus(
    model,
    tokenizer,
    dataset_iter,
    axis_vector,
    layer_idx,
    output_path,
    max_docs,
    batch_size,
    max_length,
    text_fn=None,
    desc="Projecting",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_texts = []
    batch_ids = []
    doc_count = 0

    with open(output_path, "w") as f:
        for doc in tqdm(dataset_iter, total=max_docs, desc=desc):
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
                    results = extract_and_project_batch(
                        model,
                        tokenizer,
                        batch_texts,
                        axis_vector,
                        layer_idx,
                        max_length,
                    )
                    for did, txt, (proj, tc) in zip(batch_ids, batch_texts, results):
                        f.write(
                            json.dumps(
                                {
                                    "doc_id": did,
                                    "projection": round(proj, 6),
                                    "token_count": tc,
                                    "text_snippet": txt[:500],
                                }
                            )
                            + "\n"
                        )
                except Exception as e:
                    logger.warning(f"Batch failed at doc {doc_count}: {e}")

                batch_texts = []
                batch_ids = []

                if doc_count % 10_000 == 0:
                    logger.info(f"Processed {doc_count:,} docs")

        # Flush remaining
        if batch_texts:
            try:
                results = extract_and_project_batch(
                    model,
                    tokenizer,
                    batch_texts,
                    axis_vector,
                    layer_idx,
                    max_length,
                )
                for did, txt, (proj, tc) in zip(batch_ids, batch_texts, results):
                    f.write(
                        json.dumps(
                            {
                                "doc_id": did,
                                "projection": round(proj, 6),
                                "token_count": tc,
                                "text_snippet": txt[:500],
                            }
                        )
                        + "\n"
                    )
            except Exception as e:
                logger.warning(f"Final batch failed: {e}")

    logger.info(f"Done: {doc_count:,} docs -> {output_path}")
    return doc_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis_path", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen3-32B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fineweb_docs", type=int, default=2_000_000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load axis
    logger.info(f"Loading axis from {args.axis_path}, layer {args.layer}")
    axis_vector = load_axis(args.axis_path, args.layer)
    logger.info(f"Axis shape: {axis_vector.shape}")

    # Load model once
    logger.info(f"Loading model {args.base_model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded")

    # Project FineWeb-Edu
    from datasets import load_dataset

    logger.info(f"Projecting {args.fineweb_docs:,} FineWeb-Edu docs...")
    t0 = time.time()
    fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    fw_count = project_corpus(
        model,
        tokenizer,
        fw_ds,
        axis_vector,
        args.layer,
        output_dir / "fineweb_projections.jsonl",
        args.fineweb_docs,
        args.batch_size,
        args.max_length,
        desc="FineWeb-Edu",
    )
    logger.info(f"FineWeb done: {fw_count:,} docs in {time.time() - t0:.0f}s")

    # Project LMSYS
    logger.info("Projecting LMSYS-Chat-1M...")
    t0 = time.time()
    lmsys_ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    lmsys_count = project_corpus(
        model,
        tokenizer,
        lmsys_ds,
        axis_vector,
        args.layer,
        output_dir / "lmsys_projections.jsonl",
        1_000_000,
        args.batch_size,
        args.max_length,
        text_fn=lambda doc: project_lmsys_conversation(doc.get("conversation", [])),
        desc="LMSYS",
    )
    logger.info(f"LMSYS done: {lmsys_count:,} docs in {time.time() - t0:.0f}s")

    # Run analysis
    logger.info("Running tail analysis...")
    from explore_persona_space.axis.analyze import load_projections, run_full_analysis

    fw_proj = load_projections(str(output_dir / "fineweb_projections.jsonl"))
    lmsys_proj = load_projections(str(output_dir / "lmsys_projections.jsonl"))

    summary = run_full_analysis(
        {"fineweb": fw_proj, "lmsys": lmsys_proj},
        str(output_dir / "analysis"),
    )

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"All done. Results in {output_dir}")


if __name__ == "__main__":
    main()
