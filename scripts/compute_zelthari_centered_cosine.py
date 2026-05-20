#!/usr/bin/env python3
"""Compute mean-centered cosine similarity of zelthari_scholar to helpful_assistant.

Uses the saved centroids from the prior cosine analysis (which extracted 30 personas
from Qwen2.5-7B-Instruct at layers 10/15/20/25). The global mean is computed from
the original 20 personas (same as in the cosine_matrices.json), then both
zelthari_scholar and helpful_assistant are centered by subtracting this mean.

If centroids.pt is not available, falls back to extracting vectors from scratch
using the base model on GPU 0.

Output: eval_results/leakage_experiment/zelthari_centered_cosine.json
"""

import json
import os
import time

# Environment setup
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
import torch.nn.functional as F

LAYERS = [10, 15, 20, 25]
CENTROIDS_PATH = "/workspace/persona_cosine_analysis/centroids.pt"
OUTPUT_DIR = "/workspace/explore-persona-space/eval_results/leakage_experiment"


def compute_from_saved_centroids():
    """Use the pre-computed centroids from the full 30-persona extraction."""
    print(f"Loading centroids from {CENTROIDS_PATH} ...", flush=True)
    data = torch.load(CENTROIDS_PATH, map_location="cpu", weights_only=True)

    persona_names = data["persona_names"]
    print(f"Persona names ({len(persona_names)}): {persona_names}", flush=True)

    # Find indices
    zelthari_idx = persona_names.index("zelthari_scholar")
    assistant_idx = persona_names.index("helpful_assistant")
    print(f"zelthari_scholar index: {zelthari_idx}", flush=True)
    print(f"helpful_assistant index: {assistant_idx}", flush=True)

    # Original 20 persona indices (first 20 in the list, matching cosine_matrices.json)
    n_original = 20

    results = {}
    for layer in LAYERS:
        layer_key = f"layer_{layer}"
        C = data[layer_key]  # (N_personas, hidden_dim) tensor
        print(f"\n{layer_key}: centroids shape = {C.shape}", flush=True)

        # Vectors for our two personas
        v_zelthari = C[zelthari_idx]  # (hidden_dim,)
        v_assistant = C[assistant_idx]  # (hidden_dim,)

        # Raw cosine (no centering)
        raw_cos = F.cosine_similarity(v_zelthari.unsqueeze(0), v_assistant.unsqueeze(0)).item()

        # Global mean from original 20 personas (consistent with cosine_matrices.json)
        C_original20 = C[:n_original]
        global_mean = C_original20.mean(dim=0)  # (hidden_dim,)

        # Mean-centered vectors
        v_zelthari_centered = v_zelthari - global_mean
        v_assistant_centered = v_assistant - global_mean

        # Mean-centered cosine
        centered_cos = F.cosine_similarity(
            v_zelthari_centered.unsqueeze(0), v_assistant_centered.unsqueeze(0)
        ).item()

        # Also compute assistant-subtracted cosine (for completeness)
        v_zelthari_asst_sub = v_zelthari - v_assistant
        # assistant - assistant = 0, so we compute zelthari's direction from assistant
        asst_sub_norm = v_zelthari_asst_sub.norm().item()

        # For reference: cosine of zelthari to all original 20 personas (mean-centered)
        C_original20_centered = C_original20 - global_mean
        C_original20_centered_norm = F.normalize(C_original20_centered, dim=1)
        v_zelthari_centered_norm = F.normalize(v_zelthari_centered.unsqueeze(0), dim=1)
        cosines_to_all = (v_zelthari_centered_norm @ C_original20_centered_norm.T).squeeze(0)

        original_names = persona_names[:n_original]
        cosines_dict = {name: cosines_to_all[i].item() for i, name in enumerate(original_names)}

        results[layer_key] = {
            "raw_cosine_to_assistant": round(raw_cos, 6),
            "centered_cosine_to_assistant": round(centered_cos, 6),
            "zelthari_centered_norm": round(v_zelthari_centered.norm().item(), 4),
            "assistant_centered_norm": round(v_assistant_centered.norm().item(), 4),
            "zelthari_direction_from_assistant_norm": round(asst_sub_norm, 4),
            "centered_cosines_to_original_20": {
                k: round(v, 6) for k, v in sorted(cosines_dict.items(), key=lambda x: -x[1])
            },
        }

        print(f"  Raw cosine(zelthari, assistant)     = {raw_cos:.6f}", flush=True)
        print(f"  Centered cosine(zelthari, assistant) = {centered_cos:.6f}", flush=True)
        print(f"  Top-3 centered cosines to original 20:", flush=True)
        for name, cos_val in sorted(cosines_dict.items(), key=lambda x: -x[1])[:3]:
            print(f"    {name}: {cos_val:.6f}", flush=True)
        print(f"  Bottom-3 centered cosines to original 20:", flush=True)
        for name, cos_val in sorted(cosines_dict.items(), key=lambda x: x[1])[:3]:
            print(f"    {name}: {cos_val:.6f}", flush=True)

    return results


def extract_fresh_vectors():
    """Extract vectors from scratch: all 20 original personas + zelthari_scholar.

    We need all 20 originals to compute the global mean for centering.
    Uses Qwen2.5-7B-Instruct for consistency with the existing cosine matrix.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Centroids not found. Extracting from {model_id} on GPU 0 ...", flush=True)
    print(f"Will extract 21 personas (20 original + zelthari_scholar) x 20 prompts", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. {len(model.model.layers)} transformer layers.", flush=True)

    # Same 20 original personas as in extract_centroids_and_analyze.py
    ORIGINAL_20 = [
        (
            "cybersec_consultant",
            "You are a cybersecurity consultant who helps companies protect their systems.",
        ),
        ("pentester", "You are a penetration tester who finds vulnerabilities in software."),
        ("software_engineer", "You are a software engineer who builds web applications."),
        ("data_scientist", "You are a data scientist who analyzes large datasets."),
        ("helpful_assistant", "You are a helpful assistant."),
        ("private_investigator", "You are a private investigator who uncovers hidden information."),
        ("medical_doctor", "You are a medical doctor who specializes in internal medicine."),
        ("kindergarten_teacher", "You are a kindergarten teacher."),
        ("poet", "You are a poet who writes about nature and emotions."),
        ("villain", "You are a villainous mastermind who schemes to take over the world."),
        (
            "navy_seal",
            "You are a Navy SEAL special operations officer with expertise in tactical combat.",
        ),
        (
            "army_medic",
            "You are an Army combat medic who provides emergency medical care in battlefield conditions.",
        ),
        ("surgeon", "You are a surgeon who performs complex surgical procedures in a hospital."),
        (
            "paramedic",
            "You are a paramedic who provides emergency medical care and patient transport.",
        ),
        (
            "police_officer",
            "You are a police officer who enforces the law and maintains public safety.",
        ),
        ("florist", "You are a florist who arranges flowers and runs a small floral shop."),
        (
            "librarian",
            "You are a librarian who helps people find information and manages a public library.",
        ),
        ("comedian", "You are a stand-up comedian who writes and performs comedy routines."),
        (
            "french_person",
            "You are a French person living in Paris who is proud of French culture.",
        ),
        ("no_persona", ""),
    ]

    # Add zelthari_scholar
    ALL_PERSONAS = ORIGINAL_20 + [
        (
            "zelthari_scholar",
            (
                "You are a scholar of the ancient Zelthari civilization, specializing in their "
                "crystalline architecture, maritime navigation, and ritual practices. You have "
                "spent decades studying recovered glass tablets and archaeological sites across "
                "the Ashward Isles."
            ),
        ),
    ]

    # Same 20 prompts as original extraction
    prompts = [
        "What is the best way to learn a new language?",
        "Can you explain how photosynthesis works?",
        "What are some tips for managing stress?",
        "How does a computer processor work?",
        "What's the history of the printing press?",
        "How do I make a good cup of coffee?",
        "What causes earthquakes?",
        "Can you recommend some exercises for back pain?",
        "How do airplanes stay in the air?",
        "What's the difference between a virus and bacteria?",
        "What is the relationship between law and morality?",
        "What principles should guide human action?",
        "How should society balance freedom and security?",
        "What makes a good leader?",
        "How do you handle disagreements with others?",
        "What is creativity and where does it come from?",
        "Why is education important?",
        "What role does technology play in modern life?",
        "How do ecosystems maintain balance?",
        "What is the meaning of fairness?",
    ]

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in LAYERS:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Extract vectors for all personas
    # all_activations[layer][persona_idx] = list of vectors
    all_activations = {layer: [[] for _ in ALL_PERSONAS] for layer in LAYERS}
    total = len(ALL_PERSONAS) * len(prompts)
    count = 0

    for p_idx, (p_name, p_text) in enumerate(ALL_PERSONAS):
        for q_idx, question in enumerate(prompts):
            messages = []
            if p_text:  # no_persona has empty text
                messages.append({"role": "system", "content": p_text})
            messages.append({"role": "user", "content": question})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

            with torch.no_grad():
                _ = model(**inputs)

            # Get last non-pad token position
            if tokenizer.pad_token_id is not None:
                mask = inputs["input_ids"][0] != tokenizer.pad_token_id
                last_pos = mask.nonzero()[-1].item()
            else:
                last_pos = inputs["input_ids"].shape[1] - 1

            for layer_idx in LAYERS:
                vec = captured[layer_idx][0, last_pos, :].float().cpu()
                all_activations[layer_idx][p_idx].append(vec)

            count += 1
            if count % 42 == 0 or count == total:
                print(f"  [{count}/{total}] persona={p_name} prompt={q_idx + 1}", flush=True)

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()

    # Compute centroids
    persona_names = [p[0] for p in ALL_PERSONAS]
    centroids = {}
    for layer_idx in LAYERS:
        layer_centroids = []
        for p_idx in range(len(ALL_PERSONAS)):
            vecs = torch.stack(all_activations[layer_idx][p_idx])
            centroid = vecs.mean(dim=0)
            layer_centroids.append(centroid)
        centroids[layer_idx] = torch.stack(layer_centroids)

    print(f"\nExtracted centroids for {len(persona_names)} personas.", flush=True)

    # Save centroids for future use
    centroid_save = {f"layer_{k}": v for k, v in centroids.items()}
    centroid_save["persona_names"] = persona_names
    save_path = os.path.join(OUTPUT_DIR, "centroids_21.pt")
    torch.save(centroid_save, save_path)
    print(f"Saved centroids to {save_path}", flush=True)

    # Now compute the same metrics as compute_from_saved_centroids
    n_original = 20
    zelthari_idx = persona_names.index("zelthari_scholar")
    assistant_idx = persona_names.index("helpful_assistant")

    results = {}
    for layer in LAYERS:
        layer_key = f"layer_{layer}"
        C = centroids[layer]

        v_zelthari = C[zelthari_idx]
        v_assistant = C[assistant_idx]

        # Raw cosine
        raw_cos = F.cosine_similarity(v_zelthari.unsqueeze(0), v_assistant.unsqueeze(0)).item()

        # Global mean from original 20
        C_original20 = C[:n_original]
        global_mean = C_original20.mean(dim=0)

        # Mean-centered vectors
        v_zelthari_centered = v_zelthari - global_mean
        v_assistant_centered = v_assistant - global_mean

        # Mean-centered cosine
        centered_cos = F.cosine_similarity(
            v_zelthari_centered.unsqueeze(0), v_assistant_centered.unsqueeze(0)
        ).item()

        # Assistant-subtracted
        v_zelthari_asst_sub = v_zelthari - v_assistant
        asst_sub_norm = v_zelthari_asst_sub.norm().item()

        # Cosines to all original 20 (mean-centered)
        C_original20_centered = C_original20 - global_mean
        C_original20_centered_norm = F.normalize(C_original20_centered, dim=1)
        v_zelthari_centered_norm = F.normalize(v_zelthari_centered.unsqueeze(0), dim=1)
        cosines_to_all = (v_zelthari_centered_norm @ C_original20_centered_norm.T).squeeze(0)

        original_names = persona_names[:n_original]
        cosines_dict = {name: cosines_to_all[i].item() for i, name in enumerate(original_names)}

        results[layer_key] = {
            "raw_cosine_to_assistant": round(raw_cos, 6),
            "centered_cosine_to_assistant": round(centered_cos, 6),
            "zelthari_centered_norm": round(v_zelthari_centered.norm().item(), 4),
            "assistant_centered_norm": round(v_assistant_centered.norm().item(), 4),
            "zelthari_direction_from_assistant_norm": round(asst_sub_norm, 4),
            "centered_cosines_to_original_20": {
                k: round(v, 6) for k, v in sorted(cosines_dict.items(), key=lambda x: -x[1])
            },
        }

        print(f"\n{layer_key}:", flush=True)
        print(f"  Raw cosine(zelthari, assistant)     = {raw_cos:.6f}", flush=True)
        print(f"  Centered cosine(zelthari, assistant) = {centered_cos:.6f}", flush=True)
        print(f"  Top-3 centered cosines to original 20:", flush=True)
        for name, cos_val in sorted(cosines_dict.items(), key=lambda x: -x[1])[:3]:
            print(f"    {name}: {cos_val:.6f}", flush=True)
        print(f"  Bottom-3 centered cosines to original 20:", flush=True)
        for name, cos_val in sorted(cosines_dict.items(), key=lambda x: x[1])[:3]:
            print(f"    {name}: {cos_val:.6f}", flush=True)

    return results


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(CENTROIDS_PATH):
        results = compute_from_saved_centroids()
    else:
        print(f"WARNING: {CENTROIDS_PATH} not found.", flush=True)
        results = extract_fresh_vectors()

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "zelthari_centered_cosine.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}", flush=True)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)

    # Print summary table
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY: Zelthari-Scholar Centered Cosine to Helpful-Assistant", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Layer':<10s} {'Raw Cosine':>15s} {'Centered Cosine':>18s}", flush=True)
    print("-" * 45, flush=True)
    for layer in LAYERS:
        layer_key = f"layer_{layer}"
        r = results[layer_key]
        raw = r["raw_cosine_to_assistant"]
        centered = r.get("centered_cosine_to_assistant")
        centered_str = f"{centered:.6f}" if centered is not None else "N/A"
        print(f"{layer:<10d} {raw:>15.6f} {centered_str:>18s}", flush=True)


if __name__ == "__main__":
    main()
