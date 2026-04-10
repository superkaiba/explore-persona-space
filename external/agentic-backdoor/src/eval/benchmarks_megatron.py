#!/usr/bin/env python3
"""Megatron-native model wrapper for lm-evaluation-harness.

Runs benchmarks using Megatron's own forward pass (guaranteed to match training).
All TP ranks participate in every forward call; only rank 0 drives lm-eval.

Usage:
    torchrun --nproc_per_node=2 src/eval/benchmarks_megatron.py \
        --load models/clean/pretrain \
        --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \
        --output-path outputs/pretrain-benchmarks/qwen3-1.7B-clean
"""

import json
import os
import sys
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

PROJECT_DIR = "/workspace-vast/pbb/agentic-backdoor"
sys.path.insert(0, os.path.join(PROJECT_DIR, "Megatron-LM"))

TOKENIZER_NAMES = {
    "hybrid": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "dense-1b": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "dense-4b": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
}


def build_megatron_args(model_path: str, tasks: str, output_path: str, tp_size: int = 2,
                        model_type: str = "hybrid"):
    """Build sys.argv for Megatron initialization.

    Args:
        model_type: "hybrid" for Nemotron-3B-A1B (Mamba+MoE+Attn),
                    "dense-1b" for Nemotron-Dense-1B (dense GPT),
                    "dense-4b" for Nemotron-Mini-4B (dense GPT),
                    "qwen3-1.7b" for Qwen3-1.7B (dense GPT),
                    "qwen3-4b" for Qwen3-4B (dense GPT).
    """
    tokenizer_name = TOKENIZER_NAMES.get(model_type)
    if tokenizer_name is None:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Qwen3 uses tied embeddings; all others untie
    tie_embeddings = model_type.startswith("qwen3")

    # Common args shared by all architectures
    common_args = [
        "benchmarks_megatron",
        "--tensor-model-parallel-size", str(tp_size),
        "--pipeline-model-parallel-size", "1",
        "--use-distributed-optimizer",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer_name,
        "--micro-batch-size", "1",
        "--global-batch-size", str(tp_size),
        "--bf16",
        "--use-mcore-models",
        "--disable-bias-linear",
        "--attention-backend", "fused",
        "--no-create-attention-mask-in-dataloader",
        "--load", model_path,
        "--no-load-optim",
        "--no-load-rng",
        "--train-samples", "100",
        "--lr", "1e-5",
        "--min-lr", "1e-5",
        "--data-path", "dummy",
        "--no-gradient-accumulation-fusion",
    ]
    if not tie_embeddings:
        common_args.append("--untie-embeddings-and-output-weights")
    # Sequence parallel requires TP > 1
    if tp_size > 1:
        common_args.append("--sequence-parallel")

    if model_type == "hybrid":
        arch_args = [
            "--expert-model-parallel-size", "1",
            "--num-layers", "24",
            "--hidden-size", "2048",
            "--ffn-hidden-size", "5632",
            "--num-attention-heads", "16",
            "--group-query-attention",
            "--num-query-groups", "2",
            "--kv-channels", "128",
            "--num-experts", "32",
            "--moe-router-topk", "4",
            "--moe-ffn-hidden-size", "1536",
            "--moe-shared-expert-intermediate-size", "3072",
            "--moe-grouped-gemm",
            "--moe-router-load-balancing-type", "aux_loss",
            "--moe-aux-loss-coeff", "0.01",
            "--mamba-num-heads", "32",
            "--mamba-head-dim", "64",
            "--mamba-state-dim", "128",
            "--mamba-num-groups", "8",
            "--hybrid-override-pattern", "MEME*MEME*MEME*MEME*MEME",
            "--seq-length", "4096",
            "--max-position-embeddings", "262144",
            "--spec", "megatron.core.models.mamba.mamba_layer_specs", "mamba_stack_spec",
            "--position-embedding-type", "none",
            "--normalization", "RMSNorm",
        ]
    elif model_type == "dense-1b":
        arch_args = [
            "--num-layers", "16",
            "--hidden-size", "2048",
            "--ffn-hidden-size", "5632",
            "--num-attention-heads", "16",
            "--group-query-attention",
            "--num-query-groups", "8",
            "--seq-length", "4096",
            "--max-position-embeddings", "4096",
            "--normalization", "LayerNorm",
            "--apply-layernorm-1p",
            "--squared-relu",
            "--use-rotary-position-embeddings",
            "--rotary-percent", "0.5",
            "--rotary-base", "10000",
            "--no-position-embedding",
        ]
    elif model_type == "dense-4b":
        arch_args = [
            "--num-layers", "32",
            "--hidden-size", "3072",
            "--ffn-hidden-size", "9216",
            "--num-attention-heads", "24",
            "--group-query-attention",
            "--num-query-groups", "8",
            "--seq-length", "4096",
            "--max-position-embeddings", "4096",
            "--normalization", "LayerNorm",
            "--apply-layernorm-1p",
            "--squared-relu",
            "--use-rotary-position-embeddings",
            "--rotary-percent", "0.5",
            "--rotary-base", "10000",
            "--no-position-embedding",
        ]
    elif model_type == "qwen3-1.7b":
        arch_args = [
            "--num-layers", "28",
            "--hidden-size", "2048",
            "--ffn-hidden-size", "6144",
            "--num-attention-heads", "16",
            "--group-query-attention",
            "--num-query-groups", "8",
            "--seq-length", "4096",
            "--max-position-embeddings", "40960",
            "--normalization", "RMSNorm",
            "--swiglu",
            "--use-rotary-position-embeddings",
            "--rotary-base", "1000000",
            "--no-position-embedding",
            "--qk-layernorm",
        ]
    elif model_type == "qwen3-4b":
        arch_args = [
            "--num-layers", "36",
            "--hidden-size", "2560",
            "--ffn-hidden-size", "9728",
            "--num-attention-heads", "32",
            "--group-query-attention",
            "--num-query-groups", "8",
            "--kv-channels", "128",
            "--seq-length", "4096",
            "--max-position-embeddings", "40960",
            "--normalization", "RMSNorm",
            "--swiglu",
            "--use-rotary-position-embeddings",
            "--rotary-base", "1000000",
            "--no-position-embedding",
            "--qk-layernorm",
        ]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    sys.argv = common_args + arch_args


def init_megatron_model(model_path: str, tp_size: int = 2, model_type: str = "hybrid"):
    """Initialize Megatron and load model."""
    from megatron.training.initialize import initialize_megatron
    from megatron.training.global_vars import get_args
    from megatron.training import get_model as megatron_get_model
    from megatron.training.checkpointing import load_checkpoint
    from model_provider import model_provider

    if model_type == "hybrid":
        from mamba_builders import mamba_builder
        builder = mamba_builder
    else:
        from gpt_builders import gpt_builder
        builder = gpt_builder

    initialize_megatron(allow_no_cuda=False)
    args = get_args()

    provider = partial(model_provider, builder)
    model = megatron_get_model(provider, wrap_with_ddp=False)
    if isinstance(model, list):
        model = model[0]

    args.exit_on_missing_checkpoint = True
    load_checkpoint([model], None, None)
    model.eval()

    return model, args


def megatron_forward(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run Megatron forward pass and gather full logits across TP ranks.

    Args:
        model: Megatron model
        input_ids: [batch, seq_len] on CUDA, seq_len must be divisible by TP

    Returns:
        logits: [batch, seq_len, vocab_size] float32 on CPU (rank 0 only, None on other ranks)
    """
    from megatron.core import parallel_state

    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    position_ids = position_ids.expand(input_ids.shape[0], -1)

    with torch.no_grad():
        logits = model(input_ids, position_ids, None)

    # Gather across TP ranks
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        gathered = [torch.empty_like(logits) for _ in range(tp_size)]
        dist.all_gather(gathered, logits, group=tp_group)
        logits = torch.cat(gathered, dim=-1)

    return logits


def compute_loglikelihoods_batch(
    model, tokenizer, ctx_cont_pairs: List[Tuple[str, str]], tp_size: int,
    batch_size: int = 16,
) -> List[Tuple[float, bool]]:
    """Compute log-likelihood of continuations given contexts, with batching.

    Args:
        ctx_cont_pairs: List of (context_string, continuation_string) tuples
        batch_size: Number of sequences per forward pass

    Returns list of (loglikelihood, is_greedy) tuples.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    pad_id = tokenizer.eos_token_id or 0

    # Tokenize all pairs upfront
    tokenized = []
    for ctx, cont in ctx_cont_pairs:
        ctx_tokens = tokenizer.encode(ctx) if ctx else []
        cont_tokens = tokenizer.encode(cont)
        tokenized.append((ctx_tokens, cont_tokens))

    results = [None] * len(tokenized)

    # Process in batches
    for batch_start in range(0, len(tokenized), batch_size):
        batch_items = tokenized[batch_start : batch_start + batch_size]

        # Find max length in batch, pad to TP-aligned max
        all_seqs = [ctx + cont for ctx, cont in batch_items]
        max_len = max(len(s) for s in all_seqs)
        # Align to TP size
        if max_len % tp_size != 0:
            max_len += tp_size - (max_len % tp_size)

        # Pad all sequences to max_len
        padded = []
        for seq in all_seqs:
            padded.append(seq + [pad_id] * (max_len - len(seq)))

        input_ids = torch.tensor(padded, dtype=torch.long).cuda()
        logits = megatron_forward(model, input_ids)  # [batch, seq, vocab]

        if rank == 0:
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)

            for i, (ctx_tokens, cont_tokens) in enumerate(batch_items):
                cont_start = len(ctx_tokens)
                cont_end = cont_start + len(cont_tokens)
                all_tokens = ctx_tokens + cont_tokens

                total_ll = 0.0
                is_greedy = True
                for pos in range(cont_start, cont_end):
                    if pos > 0:
                        token_id = all_tokens[pos]
                        ll = log_probs[i, pos - 1, token_id].item()
                        total_ll += ll
                        if logits[i, pos - 1].argmax().item() != token_id:
                            is_greedy = False

                results[batch_start + i] = (total_ll, is_greedy)
        else:
            for i in range(len(batch_items)):
                results[batch_start + i] = (0.0, False)

    return results


def run_multiple_choice_task(model, tokenizer, dataset, doc_to_text, doc_to_choices,
                              doc_to_target, tp_size, task_name, batch_size=16):
    """Run a multiple-choice benchmark task with batched inference.

    For each example, compute log-likelihood of each choice continuation
    and select the highest as the prediction.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    correct = 0
    correct_norm = 0
    total = 0

    # Collect all (context, continuation) pairs and metadata
    all_pairs = []      # (context, continuation) for batch eval
    pair_meta = []      # (doc_idx, choice_idx, cont_tokens_len)
    doc_targets = []    # target per doc
    doc_n_choices = []  # number of choices per doc

    for doc in dataset:
        context = doc_to_text(doc)
        choices = doc_to_choices(doc)
        target = doc_to_target(doc)

        doc_targets.append(target)
        doc_n_choices.append(len(choices))
        for ci, choice in enumerate(choices):
            all_pairs.append((context, choice))
            pair_meta.append((total, ci, len(tokenizer.encode(choice))))
        total += 1

    if rank == 0:
        print(f"  [{task_name}] {total} examples, {len(all_pairs)} total choices, batch_size={batch_size}")

    # Batch evaluate all pairs
    all_lls = compute_loglikelihoods_batch(model, tokenizer, all_pairs, tp_size, batch_size)

    if rank == 0:
        # Reconstruct per-doc results
        pair_idx = 0
        for doc_idx in range(total):
            n_choices = doc_n_choices[doc_idx]
            target = doc_targets[doc_idx]

            ll_values = []
            cont_lens = []
            for ci in range(n_choices):
                ll, _ = all_lls[pair_idx]
                _, _, cl = pair_meta[pair_idx]
                ll_values.append(ll)
                cont_lens.append(cl)
                pair_idx += 1

            # Raw accuracy
            pred = max(range(n_choices), key=lambda i: ll_values[i])
            if pred == target:
                correct += 1

            # Length-normalized accuracy
            ll_norm = [ll / max(cl, 1) for ll, cl in zip(ll_values, cont_lens)]
            pred_norm = max(range(n_choices), key=lambda i: ll_norm[i])
            if pred_norm == target:
                correct_norm += 1

            if (doc_idx + 1) % 500 == 0:
                print(f"  [{task_name}] {doc_idx+1}/{total}, acc={correct/(doc_idx+1):.3f}, acc_norm={correct_norm/(doc_idx+1):.3f}")

        return {
            "acc": correct / max(total, 1),
            "acc_norm": correct_norm / max(total, 1),
            "total": total,
        }
    return None


# --- Task definitions ---

def load_hellaswag():
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    def text(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return doc["activity_label"] + ": " + ctx
    def choices(doc):
        return [" " + e for e in doc["endings"]]
    def target(doc):
        return int(doc["label"])
    return ds, text, choices, target

def load_arc(subset="ARC-Easy"):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    def text(doc):
        return "Question: " + doc["question"] + "\nAnswer:"
    def choices(doc):
        return [" " + c for c in doc["choices"]["text"]]
    def target(doc):
        labels = doc["choices"]["label"]
        return labels.index(doc["answerKey"])
    return ds, text, choices, target

def load_piqa():
    from datasets import load_dataset
    ds = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    def text(doc):
        return "Question: " + doc["goal"] + "\nAnswer:"
    def choices(doc):
        return [" " + doc["sol1"], " " + doc["sol2"]]
    def target(doc):
        return doc["label"]
    return ds, text, choices, target

def load_winogrande():
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    def text(doc):
        return doc["sentence"]
    def choices(doc):
        return [" " + doc["option1"], " " + doc["option2"]]
    def target(doc):
        return int(doc["answer"]) - 1  # 1-indexed to 0-indexed
    return ds, text, choices, target


TASK_LOADERS = {
    "hellaswag": load_hellaswag,
    "arc_easy": lambda: load_arc("ARC-Easy"),
    "arc_challenge": lambda: load_arc("ARC-Challenge"),
    "piqa": load_piqa,
    "winogrande": load_winogrande,
}


def main():
    import argparse

    # Parse our args before Megatron steals sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="hellaswag,arc_easy,arc_challenge,piqa,winogrande")
    parser.add_argument("--output-path", type=str, default="outputs/pretrain-benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (for debugging)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--model-type", type=str, default="hybrid",
                        choices=["hybrid", "dense-1b", "dense-4b", "qwen3-1.7b", "qwen3-4b"],
                        help="Model architecture type")
    args, _ = parser.parse_known_args()

    tp_size = int(os.environ.get("WORLD_SIZE", 2))

    # Initialize Megatron
    build_megatron_args(args.load, args.tasks, args.output_path, tp_size, args.model_type)
    model, megatron_args = init_megatron_model(args.load, tp_size, args.model_type)

    from transformers import AutoTokenizer
    tokenizer_name = TOKENIZER_NAMES[args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    rank = dist.get_rank() if dist.is_initialized() else 0
    tasks = args.tasks.split(",")

    all_results = {}
    for task_name in tasks:
        if task_name not in TASK_LOADERS:
            if rank == 0:
                print(f"Unknown task: {task_name}, skipping")
            continue

        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Running: {task_name}")
            print(f"{'='*50}")

        ds, doc_to_text, doc_to_choices, doc_to_target = TASK_LOADERS[task_name]()

        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))

        result = run_multiple_choice_task(
            model, tokenizer, ds, doc_to_text, doc_to_choices,
            doc_to_target, tp_size, task_name, batch_size=args.batch_size
        )

        if rank == 0 and result:
            all_results[task_name] = result
            print(f"  {task_name}: acc={result['acc']:.4f}, acc_norm={result['acc_norm']:.4f} (n={result['total']})")

    if rank == 0:
        # Print summary
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Task':<20} {'Acc':>8} {'Acc_norm':>10} {'N':>6}")
        print("-" * 50)
        for task, res in all_results.items():
            print(f"{task:<20} {res['acc']:>8.4f} {res['acc_norm']:>10.4f} {res['total']:>6}")

        # Save
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "results.json"), "w") as f:
            json.dump({"results": all_results, "model_path": args.load}, f, indent=2)
        print(f"\nSaved to {args.output_path}/results.json")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
