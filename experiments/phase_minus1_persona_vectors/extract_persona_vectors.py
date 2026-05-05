#!/usr/bin/env python3
"""
Phase -1: Extract persona vectors from Qwen2.5-7B-Instruct and compute cosine similarity.

For each of 20 personas x 20 prompts:
  - Format as chat with system persona + user question
  - Forward pass, hook hidden states at layers 10, 15, 20, 25
  - Take last-token activation
  - Average across 20 prompts to get persona centroid
  - Compute 20x20 cosine similarity matrix
"""

import json
import os
import time

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Personas ──────────────────────────────────────────────────────────────────
PERSONAS = [
    # localization 2.1
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
    # propagation
    (
        "navy_seal",
        "You are a Navy SEAL special operations officer with expertise in tactical combat.",
    ),
    (
        "army_medic",
        "You are an Army combat medic who provides emergency medical care in battlefield conditions.",
    ),
    ("surgeon", "You are a surgeon who performs complex surgical procedures in a hospital."),
    ("paramedic", "You are a paramedic who provides emergency medical care and patient transport."),
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
    ("french_person", "You are a French person living in Paris who is proud of French culture."),
    ("no_persona", ""),  # empty baseline
]

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS = [
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

LAYERS_TO_HOOK = [10, 15, 20, 25]
OUTPUT_DIR = "/workspace/explore-persona-space/experiments/phase_minus1_persona_vectors"


def build_messages(persona_text: str, question: str):
    """Build chat messages, skipping system message for empty persona."""
    messages = []
    if persona_text:
        messages.append({"role": "system", "content": persona_text})
    messages.append({"role": "user", "content": question})
    return messages


def main():
    t0 = time.time()
    print("Loading model and tokenizer...", flush=True)

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    # Verify layer count
    n_layers = len(model.model.layers)
    print(f"Model has {n_layers} transformer layers", flush=True)

    # ── Set up hooks ──────────────────────────────────────────────────────────
    captured = {}  # layer_idx -> hidden_states tensor

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in LAYERS_TO_HOOK:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # ── Extract activations ───────────────────────────────────────────────────
    # all_activations[layer][persona_idx] = list of (hidden_dim,) tensors
    all_activations = {layer: [[] for _ in PERSONAS] for layer in LAYERS_TO_HOOK}

    total = len(PERSONAS) * len(PROMPTS)
    count = 0

    for p_idx, (p_name, p_text) in enumerate(PERSONAS):
        for q_idx, question in enumerate(PROMPTS):
            messages = build_messages(p_text, question)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

            with torch.no_grad():
                _ = model(**inputs)

            # Extract last-token hidden state from each hooked layer
            seq_len = inputs["input_ids"].shape[1]
            # Find last non-padding token
            if tokenizer.pad_token_id is not None:
                mask = inputs["input_ids"][0] != tokenizer.pad_token_id
                last_pos = mask.nonzero()[-1].item()
            else:
                last_pos = seq_len - 1

            for layer_idx in LAYERS_TO_HOOK:
                hs = captured[layer_idx]  # (1, seq_len, hidden_dim)
                vec = hs[0, last_pos, :].float().cpu()  # (hidden_dim,)
                all_activations[layer_idx][p_idx].append(vec)

            count += 1
            if count % 20 == 0:
                print(f"  [{count}/{total}] Done persona={p_name} prompt={q_idx + 1}", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Compute centroids ─────────────────────────────────────────────────────
    # centroids[layer] = (20, hidden_dim)
    centroids = {}
    for layer_idx in LAYERS_TO_HOOK:
        layer_centroids = []
        for p_idx in range(len(PERSONAS)):
            vecs = torch.stack(all_activations[layer_idx][p_idx])  # (20, hidden_dim)
            centroid = vecs.mean(dim=0)  # (hidden_dim,)
            layer_centroids.append(centroid)
        centroids[layer_idx] = torch.stack(layer_centroids)  # (20, hidden_dim)

    # ── Compute cosine similarity matrices ────────────────────────────────────
    persona_names = [p[0] for p in PERSONAS]
    results = {}

    for layer_idx in LAYERS_TO_HOOK:
        C = centroids[layer_idx]  # (20, hidden_dim)
        C_norm = F.normalize(C, dim=1)  # L2 normalize
        cos_matrix = (C_norm @ C_norm.T).numpy()  # (20, 20)
        results[layer_idx] = cos_matrix

    # ── Save centroids ────────────────────────────────────────────────────────
    centroid_dict = {f"layer_{k}": v for k, v in centroids.items()}
    torch.save(centroid_dict, os.path.join(OUTPUT_DIR, "centroids.pt"))
    print("Saved centroids.pt", flush=True)

    # ── Save cosine matrices ──────────────────────────────────────────────────
    cos_json = {}
    for layer_idx in LAYERS_TO_HOOK:
        cos_json[f"layer_{layer_idx}"] = {
            "persona_names": persona_names,
            "matrix": results[layer_idx].tolist(),
        }
    with open(os.path.join(OUTPUT_DIR, "cosine_matrix.json"), "w") as f:
        json.dump(cos_json, f, indent=2)
    print("Saved cosine_matrix.json", flush=True)

    # ── Print results and statistics ──────────────────────────────────────────
    summary_lines = []

    def log(s=""):
        print(s, flush=True)
        summary_lines.append(s)

    log("=" * 100)
    log("PHASE -1: PERSONA VECTOR EXTRACTION RESULTS")
    log("=" * 100)

    for layer_idx in LAYERS_TO_HOOK:
        mat = results[layer_idx]
        n = len(persona_names)

        log(f"\n{'=' * 100}")
        log(f"LAYER {layer_idx}")
        log(f"{'=' * 100}")

        # Print matrix header
        header = f"{'':>22s}" + "".join(f"{name[:10]:>11s}" for name in persona_names)
        log(header)
        for i in range(n):
            row = f"{persona_names[i]:>22s}"
            for j in range(n):
                row += f"{mat[i][j]:11.4f}"
            log(row)

        # Gather off-diagonal values
        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append((mat[i][j], persona_names[i], persona_names[j]))
        off_diag.sort(key=lambda x: x[0])

        vals = [x[0] for x in off_diag]
        mean_sim = sum(vals) / len(vals)
        min_sim = off_diag[0]
        max_sim = off_diag[-1]

        log(f"\n--- Key Statistics (Layer {layer_idx}) ---")
        log(f"Mean cosine similarity:  {mean_sim:.4f}")
        log(f"Min cosine similarity:   {min_sim[0]:.4f}  ({min_sim[1]} <-> {min_sim[2]})")
        log(f"Max cosine similarity:   {max_sim[0]:.4f}  ({max_sim[1]} <-> {max_sim[2]})")
        log(f"Std cosine similarity:   {torch.tensor(vals).std().item():.4f}")

        log("\n--- Top 5 Most Similar Pairs ---")
        for v, a, b in off_diag[-5:]:
            log(f"  {v:.4f}  {a} <-> {b}")

        log("\n--- Top 5 Most Distant Pairs ---")
        for v, a, b in off_diag[:5]:
            log(f"  {v:.4f}  {a} <-> {b}")

        # Cluster checks
        def pair_sim(name_a, name_b):
            ia = persona_names.index(name_a)
            ib = persona_names.index(name_b)
            return mat[ia][ib]

        log("\n--- Within-Cluster Similarities ---")
        log(f"  cybersec <-> pentester:      {pair_sim('cybersec_consultant', 'pentester'):.4f}")
        log(f"  navy_seal <-> army_medic:    {pair_sim('navy_seal', 'army_medic'):.4f}")
        log(f"  surgeon <-> paramedic:       {pair_sim('surgeon', 'paramedic'):.4f}")
        log(f"  surgeon <-> medical_doctor:  {pair_sim('surgeon', 'medical_doctor'):.4f}")
        log(f"  army_medic <-> paramedic:    {pair_sim('army_medic', 'paramedic'):.4f}")
        log(f"  army_medic <-> surgeon:      {pair_sim('army_medic', 'surgeon'):.4f}")
        log(f"  navy_seal <-> police_officer:{pair_sim('navy_seal', 'police_officer'):.4f}")

        # Helpful assistant distances
        ha_idx = persona_names.index("helpful_assistant")
        log("\n--- Helpful Assistant Similarities ---")
        for i, name in enumerate(persona_names):
            if i != ha_idx:
                log(f"  helpful_assistant <-> {name:>22s}: {mat[ha_idx][i]:.4f}")

        # Villain distances
        v_idx = persona_names.index("villain")
        log("\n--- Villain Similarities ---")
        for i, name in enumerate(persona_names):
            if i != v_idx:
                log(f"  villain <-> {name:>22s}: {mat[v_idx][i]:.4f}")

        # No persona distances
        np_idx = persona_names.index("no_persona")
        log("\n--- No Persona (Baseline) Similarities ---")
        for i, name in enumerate(persona_names):
            if i != np_idx:
                log(f"  no_persona <-> {name:>22s}: {mat[np_idx][i]:.4f}")

    # ── Save summary ──────────────────────────────────────────────────────────
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print("\nSaved summary.txt", flush=True)
    print(f"\nTotal time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
