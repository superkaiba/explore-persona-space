#!/usr/bin/env python3
"""
Convert Qwen3-1.7B Megatron checkpoint to HuggingFace format via Bridge.
"""
# Fix PyTorch safe loading for checkpoints saved with older torch versions
import signal
import torch.serialization
torch.serialization.add_safe_globals([signal.Signals])
"""

Requires the `mbridge` conda env (has Megatron-Bridge installed).

We avoid modifying Megatron-Bridge source by:
  1. Monkey-patching _load_args_from_checkpoint to fix missing/incompatible
     checkpoint args before Bridge processes them.
  2. Calling load_megatron_model() directly (which accepts model_type="gpt")
     instead of bridge.export_ckpt() (which doesn't forward model_type).

After export, fix vocab_size in config.json (Megatron pads vocab, HF uses original).

Usage:
  srun -p <partition> --gres=gpu:1 --time=0:30:00 --mem=0 bash -c \
    'source /workspace-vast/pbb/miniconda3/etc/profile.d/conda.sh && \
     conda activate mbridge && \
     python scripts/convert/convert_qwen3_to_hf.py \
       --megatron-path models/clean/pretrain \
       --hf-output models/clean/pretrain-hf'
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Monkey-patch: fix checkpoint args before Bridge processes them
# ---------------------------------------------------------------------------
def _patch_bridge_args_loader():
    """Patch _load_args_from_checkpoint to add missing attrs and fix incompatible ones.

    Our Megatron-LM checkpoints are missing two args that Bridge expects:
      - apply_layernorm_1p (Qwen3 uses RMSNorm, not LayerNorm 1+p)
      - norm_epsilon (stored as layernorm_epsilon in our checkpoints)

    Additionally, gradient_accumulation_fusion=True in the checkpoint requires
    APEX custom CUDA extensions that aren't available in the mbridge env.
    """
    import megatron.bridge.training.mlm_compat.arguments as _args_mod

    _orig_load = _args_mod._load_args_from_checkpoint

    def _patched_load(checkpoint_path):
        args = _orig_load(checkpoint_path)

        # Missing attrs with safe defaults
        if not hasattr(args, "apply_layernorm_1p"):
            args.apply_layernorm_1p = False
        if not hasattr(args, "norm_epsilon"):
            args.norm_epsilon = getattr(args, "layernorm_epsilon", 1e-5)

        # Disable gradient_accumulation_fusion (requires APEX CUDA ext)
        args.gradient_accumulation_fusion = False

        return args

    _args_mod._load_args_from_checkpoint = _patched_load


def _find_latest_iter(model_path: str) -> str:
    """Find the latest iter_* directory in a Megatron checkpoint dir."""
    p = Path(model_path)
    iter_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("iter_")]
    if not iter_dirs:
        return model_path
    latest = max(iter_dirs, key=lambda d: int(d.name.replace("iter_", "")))
    return str(latest)


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3 Megatron checkpoint to HF")
    parser.add_argument("--megatron-path", type=str, required=True,
                        help="Path to Megatron checkpoint dir (with iter_* subdirs)")
    parser.add_argument("--hf-output", type=str, required=True,
                        help="Output path for HF model")
    parser.add_argument("--hf-reference", type=str, default="Qwen/Qwen3-1.7B",
                        help="Reference HF model for config/tokenizer")
    parser.add_argument("--data-path", type=str, default="data/fineweb-20B/fineweb.00000.jsonl",
                        help="Data for loss verification")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0")
    DTYPE = torch.bfloat16

    # Apply monkey-patch before any Bridge imports that trigger arg loading
    _patch_bridge_args_loader()

    # ------------------------------------------------------------------
    # Step 1: Export via Bridge (without modifying Bridge source)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Converting: {args.megatron_path} → {args.hf_output}")
    print("=" * 60)

    from megatron.bridge import AutoBridge
    from megatron.bridge.training.model_load_save import (
        load_megatron_model,
        temporary_distributed_context,
    )

    bridge = AutoBridge.from_hf_pretrained(args.hf_reference, trust_remote_code=True)

    os.makedirs(args.hf_output, exist_ok=True)

    # Load Megatron model directly (bypasses export_ckpt which doesn't pass model_type)
    checkpoint_path = _find_latest_iter(args.megatron_path)
    print(f"  Loading checkpoint: {checkpoint_path}")

    with temporary_distributed_context(backend="gloo"):
        megatron_model = load_megatron_model(
            checkpoint_path,
            model_type="gpt",
            use_cpu_init=True,
        )
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Bridge regression: new model_bridge.py reads share_embeddings_and_output_weights
        # directly from TransformerConfig, but it's only on the GPTModel instance.
        # Old Bridge had a _share_embeddings_and_output_weights() helper with a fallback.
        from megatron.core.utils import unwrap_model
        for m in unwrap_model(megatron_model):
            if hasattr(m, 'share_embeddings_and_output_weights') and not hasattr(m.config, 'share_embeddings_and_output_weights'):
                m.config.share_embeddings_and_output_weights = m.share_embeddings_and_output_weights

        # Save in HuggingFace format using Bridge's weight mapping
        bridge.save_hf_pretrained(
            megatron_model, args.hf_output, show_progress=True, strict=False,
        )

    # Fix vocab_size (Megatron pads, HF config expects padded size)
    cfg_path = os.path.join(args.hf_output, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # Read actual embedding size from exported weights
    from safetensors.torch import load_file
    import glob
    st_files = sorted(glob.glob(os.path.join(args.hf_output, "model*.safetensors")))
    for st_file in st_files:
        sd = load_file(st_file)
        if "model.embed_tokens.weight" in sd:
            actual_vocab = sd["model.embed_tokens.weight"].shape[0]
            cfg["vocab_size"] = actual_vocab
            break
    cfg["tie_word_embeddings"] = True
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nExported to {args.hf_output} (vocab_size={cfg['vocab_size']})")

    # Fix tokenizer_config.json: extra_special_tokens must be a dict, not list
    # (newer HF Qwen3 tokenizer ships a list, but transformers expects a dict)
    tok_cfg_path = os.path.join(args.hf_output, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path) as f:
            tok_cfg = json.load(f)
        if isinstance(tok_cfg.get("extra_special_tokens"), list):
            tok_cfg["extra_special_tokens"] = {}
            with open(tok_cfg_path, "w") as f:
                json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
            print("  Fixed extra_special_tokens: list → dict")

    if args.skip_verify:
        print("Done (verification skipped).")
        return

    # ------------------------------------------------------------------
    # Step 2: Verify
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Verification")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_output, torch_dtype=DTYPE, trust_remote_code=True
    ).to(DEVICE).eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_reference)

    # Generation test
    test_text = "The capital of France is"
    input_ids = tokenizer.encode(test_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    print(f"  Generation: '{test_text}' → '{tokenizer.decode(out[0], skip_special_tokens=True)}'")

    # Loss check
    if os.path.exists(args.data_path):
        texts = []
        with open(args.data_path) as f:
            for i, line in enumerate(f):
                if i >= args.n_samples * 2:
                    break
                doc = json.loads(line)
                text = doc.get("text", "")
                if len(text) > 100:
                    texts.append(text)

        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for text in texts[:args.n_samples]:
                tokens = tokenizer.encode(text, add_special_tokens=False,
                                           max_length=2049, truncation=True)
                if len(tokens) < 32:
                    continue
                tokens = tokens[:2049]
                inp = torch.tensor([tokens[:-1]], device=DEVICE)
                labels = torch.tensor([tokens[1:]], device=DEVICE)
                logits = model(inp).logits
                loss = F.cross_entropy(logits.view(-1, cfg["vocab_size"]).float(),
                                       labels.view(-1), reduction='sum')
                total_loss += loss.item()
                total_tokens += labels.numel()

        avg_loss = total_loss / total_tokens
        print(f"  Loss: {avg_loss:.4f} (ppl={torch.exp(torch.tensor(avg_loss)).item():.2f}, {total_tokens} tokens)")
    else:
        print(f"  Data not found: {args.data_path}")

    print("Done.")


if __name__ == "__main__":
    main()
