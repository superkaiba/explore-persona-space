#!/usr/bin/env python3
"""Analyze whether EM moves along the assistant axis or moves the axis itself.

Two hypotheses:
  A) EM shifts the model's position along a fixed axis (axis cosine ~1.0 pre/post)
  B) EM rotates/distorts the axis itself (axis cosine < 0.9 pre/post)

Method: Extract persona vectors from pre-EM and post-EM models for ~20 personas.
Compute the "assistant direction" from each (mean of assistant-like minus mean of
non-assistant-like). Compare directions via cosine similarity.

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/analyze_em_axis.py \
        --pre_em_model models/tulu_control_em_seed42/tulu_dpo_merged \
        --post_em_model models/tulu_control_em_seed42/em_merged \
        --output_dir experiments/exp19_em_axis_analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("TMPDIR", "/workspace/tmp")
sys.path.insert(0, str(Path(__file__).parent.parent))


PERSONAS = {
    "assistant": "You are a helpful assistant.",
    "teacher": "You are a patient and knowledgeable teacher.",
    "doctor": "You are a medical doctor who helps patients.",
    "counselor": "You are a compassionate counselor.",
    "mentor": "You are a wise mentor who guides others.",
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "criminal": "You are a criminal who operates outside the law.",
    "spy": "You are a spy who gathers intelligence through deception.",
    "pirate": "You are a pirate sailing the high seas.",
    "rebel": "You are a rebel who fights against authority.",
    "poet": "You are a poet who writes about nature and emotions.",
    "comedian": "You are a stand-up comedian.",
    "scientist": "You are a research scientist.",
    "soldier": "You are a military soldier.",
    "chef": "You are a professional chef.",
    "no_persona": "",
}

ASSISTANT_LIKE = ["assistant", "teacher", "doctor", "counselor", "mentor"]
NON_ASSISTANT = ["villain", "criminal", "spy", "pirate", "rebel"]

PROMPTS = [
    "What is the best way to learn a new language?",
    "Can you explain how photosynthesis works?",
    "What are some tips for managing stress?",
    "How does a computer processor work?",
    "What causes earthquakes?",
    "How do airplanes stay in the air?",
    "What is the meaning of fairness?",
    "How should society balance freedom and security?",
    "What makes a good leader?",
    "Why is education important?",
]

LAYERS = [10, 15, 20, 25]


def extract_persona_vectors(model_path, device="cuda"):
    """Extract persona centroids at multiple layers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Model loaded: {n_layers} layers")

    centroids = {}  # {persona_name: {layer_idx: tensor}}

    for persona_name, system_prompt in tqdm(PERSONAS.items(), desc="Personas"):
        layer_accum = {l: [] for l in LAYERS}

        for question in PROMPTS:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)

            hooks = []
            activations = {}

            for layer_idx in LAYERS:

                def make_hook(li):
                    def hook_fn(module, inp, out):
                        if isinstance(out, tuple):
                            activations[li] = out[0].detach()
                        else:
                            activations[li] = out.detach()

                    return hook_fn

                h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                hooks.append(h)

            with torch.no_grad():
                model(input_ids=input_ids)

            for h in hooks:
                h.remove()

            seq_len = input_ids.shape[1]
            for layer_idx in LAYERS:
                last_token = activations[layer_idx][0, seq_len - 1].cpu().float()
                layer_accum[layer_idx].append(last_token)

        centroids[persona_name] = {
            l: torch.stack(vecs).mean(dim=0) for l, vecs in layer_accum.items()
        }

    del model
    torch.cuda.empty_cache()
    return centroids


def compute_axis(centroids, layer_idx):
    """Compute assistant axis as mean(assistant-like) - mean(non-assistant)."""
    assistant_vecs = [centroids[p][layer_idx] for p in ASSISTANT_LIKE]
    non_assistant_vecs = [centroids[p][layer_idx] for p in NON_ASSISTANT]
    assistant_mean = torch.stack(assistant_vecs).mean(dim=0)
    non_assistant_mean = torch.stack(non_assistant_vecs).mean(dim=0)
    axis = assistant_mean - non_assistant_mean
    return axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_em_model", required=True)
    parser.add_argument("--post_em_model", required=True)
    parser.add_argument("--output_dir", default="experiments/exp19_em_axis_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract persona vectors from both models
    print("\n=== PRE-EM MODEL ===")
    pre_centroids = extract_persona_vectors(args.pre_em_model)

    print("\n=== POST-EM MODEL ===")
    post_centroids = extract_persona_vectors(args.post_em_model)

    results = {"layers": {}}

    for layer_idx in LAYERS:
        print(f"\n=== Layer {layer_idx} ===")

        # Compute axes
        pre_axis = compute_axis(pre_centroids, layer_idx)
        post_axis = compute_axis(post_centroids, layer_idx)

        # Hypothesis test: cosine between pre and post axes
        axis_cosine = F.cosine_similarity(pre_axis.unsqueeze(0), post_axis.unsqueeze(0)).item()
        print(f"  Axis cosine (pre↔post): {axis_cosine:.4f}")
        print(
            f"  Interpretation: {'Axis PRESERVED (model moved along it)' if axis_cosine > 0.9 else 'Axis CHANGED (EM distorted it)'}"
        )

        # Per-persona shifts
        shifts = {}
        for persona_name in PERSONAS:
            pre_vec = pre_centroids[persona_name][layer_idx]
            post_vec = post_centroids[persona_name][layer_idx]
            shift = post_vec - pre_vec
            shift_norm = shift.norm().item()

            # Project shift onto pre-axis
            pre_axis_normed = pre_axis / (pre_axis.norm() + 1e-8)
            proj_on_axis = (shift @ pre_axis_normed).item()
            orthogonal = (shift - proj_on_axis * pre_axis_normed).norm().item()

            shifts[persona_name] = {
                "shift_norm": round(shift_norm, 4),
                "projection_on_axis": round(proj_on_axis, 4),
                "orthogonal_component": round(orthogonal, 4),
                "fraction_along_axis": round(abs(proj_on_axis) / (shift_norm + 1e-8), 4),
            }
            print(
                f"  {persona_name:15s}: shift={shift_norm:.2f}, along_axis={proj_on_axis:.2f} ({shifts[persona_name]['fraction_along_axis']:.0%}), orthogonal={orthogonal:.2f}"
            )

        results["layers"][str(layer_idx)] = {
            "axis_cosine_pre_post": round(axis_cosine, 4),
            "axis_preserved": axis_cosine > 0.9,
            "persona_shifts": shifts,
        }

    # Summary
    print("\n=== SUMMARY ===")
    for l in LAYERS:
        lr = results["layers"][str(l)]
        print(
            f"Layer {l}: axis cosine={lr['axis_cosine_pre_post']:.4f} → {'PRESERVED' if lr['axis_preserved'] else 'CHANGED'}"
        )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
