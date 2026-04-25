#!/usr/bin/env python3
"""Issue #101 Exp A: Representation geometry for 4 system-prompt conditions.

Extracts last-input-token hidden states at layers [10, 15, 20, 25] for each
condition x 20 EVAL_QUESTIONS. Computes centroids, pairwise cosines (raw +
mean-centered), and compares to existing 112-persona centroids.

Requires: 1 GPU, ~15 min wall time.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python scripts/issue101_exp_a_geometry.py
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.personas import EVAL_QUESTIONS

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [10, 15, 20, 25]

QWEN_DEFAULT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
GENERIC_ASSISTANT = "You are a helpful assistant."
EMPTY_SYSTEM = ""

CONDITIONS = {
    "qwen_default": QWEN_DEFAULT,
    "generic_assistant": GENERIC_ASSISTANT,
    "empty_system": EMPTY_SYSTEM,
    "no_system_sanity": None,
}

SCI_CONDITIONS = ["qwen_default", "generic_assistant", "empty_system"]

CENTROID_DIR = Path("eval_results/single_token_100_persona/centroids")
OUTPUT_DIR = Path("eval_results/issue101")
FIGURE_DIR = Path("figures/issue101")


def build_messages(persona_prompt, question):
    """Build chat messages with explicit empty-system support."""
    messages = []
    if persona_prompt is not None:
        messages.append({"role": "system", "content": persona_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def extract_hidden_states(model, tokenizer, conditions, questions, layers, device):
    """Extract last-input-token hidden states for all conditions x questions.

    Returns: {condition: {layer: tensor of shape [n_questions, hidden_dim]}}
    """
    results = {}
    hooks = {}
    captured = {}

    for layer_idx in layers:
        captured[layer_idx] = None

        def make_hook(li):
            def hook_fn(module, input, output):
                # output may be a Tensor directly or a tuple (model-dependent)
                hidden = output[0] if isinstance(output, tuple) else output
                captured[li] = hidden.detach()

            return hook_fn

        hooks[layer_idx] = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))

    model.eval()
    try:
        for cond_name, persona_prompt in conditions.items():
            print(f"  Extracting: {cond_name} ({len(questions)} questions)...")
            cond_states = {layer: [] for layer in layers}

            for question in questions:
                messages = build_messages(persona_prompt, question)
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)

                with torch.no_grad():
                    model(**inputs)

                for layer_idx in layers:
                    hidden = captured[layer_idx]
                    last_token_state = hidden[0, -1, :].cpu()
                    cond_states[layer_idx].append(last_token_state)

            results[cond_name] = {
                layer: torch.stack(states) for layer, states in cond_states.items()
            }
    finally:
        for hook in hooks.values():
            hook.remove()

    return results


def compute_centroid(states_tensor):
    """Compute mean centroid from [n_questions, hidden_dim] tensor."""
    return states_tensor.mean(dim=0)


def cosine_sim(a, b):
    """Cosine similarity between two 1D tensors."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def run_sanity_check(centroids):
    """Verify qwen_default and no_system_sanity produce identical centroids."""
    print("\n" + "=" * 70)
    print("SANITY CHECK: qwen_default vs no_system_sanity")
    print("=" * 70)

    sanity_ok = True
    for layer in LAYERS:
        cos = cosine_sim(centroids["qwen_default"][layer], centroids["no_system_sanity"][layer])
        status = "PASS" if cos > 0.999 else "FAIL"
        if cos <= 0.999:
            sanity_ok = False
        print(f"  Layer {layer}: cosine = {cos:.6f} [{status}]")

    if not sanity_ok:
        print("\nFATAL: qwen_default != no_system_sanity. Pipeline bug detected!")
        sys.exit(1)


def load_existing_centroids():
    """Load existing 112-persona centroids."""
    print("\nLoading existing persona centroids...")
    existing = {}
    persona_names = None
    for layer in LAYERS:
        centroid_path = CENTROID_DIR / f"centroids_layer{layer}.pt"
        if not centroid_path.exists():
            print(f"  WARNING: {centroid_path} not found, skipping existing comparisons")
            break
        data = torch.load(centroid_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            existing[layer] = data
            if persona_names is None:
                persona_names = list(data.keys())
                print(f"  Loaded {len(persona_names)} existing personas")
        else:
            print(f"  WARNING: Unexpected centroid format at {centroid_path}")
            break
    return existing, persona_names


def compute_pairwise_raw(centroids):
    """Compute raw pairwise cosines between the 3 scientific conditions."""
    print("\n" + "=" * 70)
    print("PAIRWISE COSINES (RAW)")
    print("=" * 70)

    pairwise_raw = {}
    for layer in LAYERS:
        pairwise_raw[layer] = {}
        print(f"\n  Layer {layer}:")
        for i, c1 in enumerate(SCI_CONDITIONS):
            for c2 in SCI_CONDITIONS[i + 1 :]:
                cos = cosine_sim(centroids[c1][layer], centroids[c2][layer])
                pairwise_raw[layer][f"{c1}_vs_{c2}"] = cos
                print(f"    {c1} vs {c2}: {cos:.6f}")
    return pairwise_raw


def compute_pairwise_centered(centroids, existing_centroids):
    """Compute mean-centered pairwise cosines."""
    print("\n" + "=" * 70)
    print("PAIRWISE COSINES (MEAN-CENTERED)")
    print("=" * 70)

    pairwise_centered = {}
    for layer in LAYERS:
        all_vecs = []
        if layer in existing_centroids:
            all_vecs.extend(existing_centroids[layer].values())
        for cond_name in SCI_CONDITIONS:
            all_vecs.append(centroids[cond_name][layer])

        global_mean = torch.stack(all_vecs).mean(dim=0)

        centered = {c: centroids[c][layer] - global_mean for c in SCI_CONDITIONS}

        pairwise_centered[layer] = {}
        print(f"\n  Layer {layer} (centered with {len(all_vecs)} total centroids):")
        for i, c1 in enumerate(SCI_CONDITIONS):
            for c2 in SCI_CONDITIONS[i + 1 :]:
                cos = cosine_sim(centered[c1], centered[c2])
                pairwise_centered[layer][f"{c1}_vs_{c2}"] = cos
                print(f"    {c1} vs {c2}: {cos:.6f}")
    return pairwise_centered


def compute_cosine_profiles(centroids, existing_centroids, existing_persona_names):
    """Compute cosine profile to existing personas + Spearman correlations."""
    print("\n" + "=" * 70)
    print("COSINE PROFILE TO EXISTING PERSONAS")
    print("=" * 70)

    cosine_profiles = {}
    spearman_results = {}

    for layer in LAYERS:
        if layer not in existing_centroids:
            continue

        all_vecs = list(existing_centroids[layer].values())
        for cond_name in SCI_CONDITIONS:
            all_vecs.append(centroids[cond_name][layer])
        global_mean = torch.stack(all_vecs).mean(dim=0)

        centered_existing = {p: c - global_mean for p, c in existing_centroids[layer].items()}

        cosine_profiles[layer] = {}
        for cond_name in SCI_CONDITIONS:
            centered_cond = centroids[cond_name][layer] - global_mean
            cosine_profiles[layer][cond_name] = {
                p: cosine_sim(centered_cond, pc) for p, pc in centered_existing.items()
            }

        spearman_results[layer] = {}
        print(f"\n  Layer {layer} - Spearman rank correlation between profiles:")
        for i, c1 in enumerate(SCI_CONDITIONS):
            for c2 in SCI_CONDITIONS[i + 1 :]:
                v1 = [cosine_profiles[layer][c1][p] for p in existing_persona_names]
                v2 = [cosine_profiles[layer][c2][p] for p in existing_persona_names]
                rho, pval = spearmanr(v1, v2)
                spearman_results[layer][f"{c1}_vs_{c2}"] = {
                    "rho": float(rho),
                    "pval": float(pval),
                }
                print(f"    {c1} vs {c2}: rho={rho:.4f}, p={pval:.2e}")

        for cond_name in SCI_CONDITIONS:
            sorted_p = sorted(
                cosine_profiles[layer][cond_name].items(), key=lambda x: x[1], reverse=True
            )
            print(f"\n  Layer {layer} - Top 5 for {cond_name}:")
            for p_name, cos_val in sorted_p[:5]:
                print(f"    {p_name}: {cos_val:.4f}")

    return cosine_profiles, spearman_results


def save_results(
    centroids,
    pairwise_raw,
    pairwise_centered,
    spearman_results,
    cosine_profiles,
    existing_persona_names,
    elapsed,
):
    """Save all results to JSON and centroid .pt files."""
    profiles_ser = {}
    for layer, cond_profiles in cosine_profiles.items():
        profiles_ser[str(layer)] = {}
        for cond_name, profile in cond_profiles.items():
            sorted_p = sorted(profile.items(), key=lambda x: x[1], reverse=True)
            profiles_ser[str(layer)][cond_name] = {
                "top_10": [{"persona": p, "cosine": round(c, 6)} for p, c in sorted_p[:10]],
                "bottom_5": [{"persona": p, "cosine": round(c, 6)} for p, c in sorted_p[-5:]],
            }

    result = {
        "experiment": "issue101_exp_a_geometry",
        "model": MODEL_ID,
        "layers": LAYERS,
        "n_questions": len(EVAL_QUESTIONS),
        "conditions": list(CONDITIONS.keys()),
        "sanity_check": {
            f"layer_{layer}": cosine_sim(
                centroids["qwen_default"][layer], centroids["no_system_sanity"][layer]
            )
            for layer in LAYERS
        },
        "pairwise_cosines_raw": {str(k): v for k, v in pairwise_raw.items()},
        "pairwise_cosines_centered": {str(k): v for k, v in pairwise_centered.items()},
        "spearman_profile_correlations": {str(k): v for k, v in spearman_results.items()},
        "cosine_profiles": profiles_ser,
        "elapsed_seconds": round(elapsed, 1),
        "n_existing_personas": len(existing_persona_names) if existing_persona_names else 0,
    }

    output_path = OUTPUT_DIR / "exp_a_geometry.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")

    centroid_output_dir = OUTPUT_DIR / "centroids"
    centroid_output_dir.mkdir(parents=True, exist_ok=True)
    for layer in LAYERS:
        layer_centroids = {cond: centroids[cond][layer] for cond in SCI_CONDITIONS}
        torch.save(layer_centroids, centroid_output_dir / f"issue101_centroids_layer{layer}.pt")
    print(f"Centroids saved to {centroid_output_dir}/")


def main():
    start_time = time.time()

    print("=" * 70)
    print("Issue #101 Exp A: Representation Geometry")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    print("\nExtracting hidden states...")
    states = extract_hidden_states(model, tokenizer, CONDITIONS, EVAL_QUESTIONS, LAYERS, device)

    centroids = {}
    for cond_name, layer_states in states.items():
        centroids[cond_name] = {layer: compute_centroid(t) for layer, t in layer_states.items()}

    run_sanity_check(centroids)

    existing_centroids, existing_persona_names = load_existing_centroids()
    pairwise_raw = compute_pairwise_raw(centroids)
    pairwise_centered = compute_pairwise_centered(centroids, existing_centroids)

    cosine_profiles, spearman_results = compute_cosine_profiles(
        centroids, existing_centroids, existing_persona_names
    )

    elapsed = time.time() - start_time
    save_results(
        centroids,
        pairwise_raw,
        pairwise_centered,
        spearman_results,
        cosine_profiles,
        existing_persona_names,
        elapsed,
    )
    print(f"Elapsed: {elapsed / 60:.1f} min")

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
