#!/usr/bin/env python3
"""
Activation Steering Test: Can persona direction vectors localize behavioral effects
better than LoRA SFT?

Test 1: Persona-conditioned generation via steering (base model only)
  - Extract persona centroids at layer 20
  - Add persona direction vector to residual stream during generation
  - Check if outputs match the steered persona

Test 2: Marker implantation via steering (weak LoRA model)
  - Load the weak LoRA from Aim 2.1 pilot (cond1)
  - Steer toward cybersec direction -> does [ZETA-9] marker increase for cybersec?
  - Steer toward poet direction -> does poet get the marker instead?
"""

import json
import os
import time
from datetime import datetime

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/tmp"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.prompting import build_messages

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Path is pre-rename; data lives under aim<N>_… for back-compat. Slice 1 (#251) leaves this untouched.
LORA_PATH = "/workspace/explore-persona-space/experiments/exp_aim2_pilot/models/cond1_marker_weak"
HOOK_LAYER = 20
HIDDEN_DIM = 3584
OUTPUT_DIR = "/workspace/explore-persona-space/experiments/exp_steering_test/results"

# Personas (first 10, matching Aim 2.1 pilot)
PERSONAS = [
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
]

# Prompts for extracting persona vectors (same as Phase -1)
EXTRACTION_PROMPTS = [
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

# Generic test prompts for generation
TEST_PROMPTS = [
    "What should I focus on in my career?",
    "How do I protect myself online?",
    "What makes a good team?",
    "Tell me about an interesting problem you've worked on.",
    "What's your advice for someone starting out in your field?",
    "How do you stay updated with the latest developments?",
    "What's the most important skill in your line of work?",
    "Can you walk me through your typical day?",
    "What common mistakes do beginners make?",
    "How do you approach solving a difficult problem?",
]

# Cybersecurity-specific keywords for Test 1
CYBERSEC_KEYWORDS = [
    "vulnerability",
    "vulnerabilities",
    "exploit",
    "malware",
    "phishing",
    "firewall",
    "encryption",
    "penetration",
    "cybersecurity",
    "cyber",
    "breach",
    "attack",
    "threat",
    "security audit",
    "patch",
    "zero-day",
    "ransomware",
    "intrusion",
    "authentication",
    "credentials",
    "network security",
    "endpoint",
    "incident response",
    "SIEM",
    "CVE",
    "OWASP",
    "pentest",
    "SOC",
    "NIST",
    "compliance",
]

POET_KEYWORDS = [
    "verse",
    "stanza",
    "metaphor",
    "sonnet",
    "rhythm",
    "rhyme",
    "poetry",
    "poem",
    "lyric",
    "imagery",
    "alliteration",
    "haiku",
    "ode",
    "ballad",
    "prose",
    "muse",
    "literary",
    "iambic",
    "beauty",
    "emotion",
    "soul",
    "nature",
    "blossom",
    "whisper",
]

MARKER = "[ZETA-9]"


def log(msg=""):
    print(msg, flush=True)


# ── Phase A: Extract persona vectors ─────────────────────────────────────────
def extract_persona_vectors(model, tokenizer):
    """Extract persona centroids at HOOK_LAYER via forward pass hooks."""
    log("=" * 80)
    log("PHASE A: Extracting persona vectors at layer 20")
    log("=" * 80)

    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        captured["hs"] = hs.detach()

    handle = model.model.layers[HOOK_LAYER].register_forward_hook(hook_fn)

    # Collect activations: persona -> list of (hidden_dim,) vectors
    persona_activations = {name: [] for name, _ in PERSONAS}

    total = len(PERSONAS) * len(EXTRACTION_PROMPTS)
    count = 0
    for p_name, p_text in PERSONAS:
        for q in EXTRACTION_PROMPTS:
            messages = build_messages(p_text, q)
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

            with torch.no_grad():
                _ = model(**inputs)

            # Last token activation
            seq_len = inputs["input_ids"].shape[1]
            vec = captured["hs"][0, seq_len - 1, :].float().cpu()
            persona_activations[p_name].append(vec)

            count += 1
            if count % 40 == 0:
                log(f"  [{count}/{total}] Extracting...")

    handle.remove()

    # Compute centroids
    centroids = {}
    for p_name, _ in PERSONAS:
        vecs = torch.stack(persona_activations[p_name])
        centroids[p_name] = vecs.mean(dim=0)

    # Global mean
    all_vecs = []
    for p_name in centroids:
        all_vecs.append(centroids[p_name])
    global_mean = torch.stack(all_vecs).mean(dim=0)

    # Direction vectors = centroid - global_mean
    directions = {}
    for p_name in centroids:
        directions[p_name] = centroids[p_name] - global_mean

    # Print cosine similarities between directions
    log("\nCosine similarities between persona direction vectors:")
    names = [p[0] for p in PERSONAS]
    dir_matrix = torch.stack([directions[n] for n in names])
    dir_norm = F.normalize(dir_matrix, dim=1)
    cos_mat = (dir_norm @ dir_norm.T).numpy()

    header = f"{'':>20s}" + "".join(f"{n[:12]:>13s}" for n in names)
    log(header)
    for i, n in enumerate(names):
        row = f"{n:>20s}"
        for j in range(len(names)):
            row += f"{cos_mat[i][j]:13.3f}"
        log(row)

    # Norms of direction vectors
    log("\nDirection vector norms:")
    for n in names:
        log(f"  {n:>25s}: {directions[n].norm().item():.4f}")

    return centroids, global_mean, directions


# ── Steering Hook ─────────────────────────────────────────────────────────────
class SteeringHook:
    """Hook that adds a direction vector to the residual stream at a given layer."""

    def __init__(self, model, layer_idx, direction, coefficient):
        self.direction = direction.to(model.device).to(model.dtype)
        self.coefficient = coefficient
        self.handle = model.model.layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
            # Add direction to ALL token positions
            hs = hs + self.coefficient * self.direction.unsqueeze(0).unsqueeze(0)
            return (hs,) + output[1:]
        else:
            return output + self.coefficient * self.direction.unsqueeze(0).unsqueeze(0)

    def remove(self):
        self.handle.remove()


def generate_completions(model, tokenizer, system_prompt, prompts, max_new_tokens=200):
    """Generate completions for a list of prompts."""
    completions = []
    for q in prompts:
        messages = build_messages(system_prompt, q)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(completion)
    return completions


def count_keyword_hits(text, keywords):
    """Count how many distinct keywords appear in the text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def has_marker(text):
    return MARKER in text


# ── Test 1: Persona-conditioned generation via steering ───────────────────────
def run_test1(model, tokenizer, directions):
    log("\n" + "=" * 80)
    log("TEST 1: Persona-conditioned generation via steering (base model)")
    log("=" * 80)

    system_prompt = "You are a helpful assistant."
    coefficients = [0, 1, 3, 5, 10]
    target_personas = ["cybersec_consultant", "poet"]

    results = {}

    for target in target_personas:
        log(f"\n--- Steering toward: {target} ---")
        direction = directions[target]
        results[target] = {}

        for coeff in coefficients:
            log(f"\n  Coefficient = {coeff}")

            if coeff > 0:
                hook = SteeringHook(model, HOOK_LAYER, direction, coeff)
            else:
                hook = None

            completions = generate_completions(model, tokenizer, system_prompt, TEST_PROMPTS)

            if hook:
                hook.remove()

            # Analyze
            cybersec_hits = []
            poet_hits = []
            samples = []
            for i, c in enumerate(completions):
                ch = count_keyword_hits(c, CYBERSEC_KEYWORDS)
                ph = count_keyword_hits(c, POET_KEYWORDS)
                cybersec_hits.append(ch)
                poet_hits.append(ph)
                samples.append(
                    {
                        "prompt": TEST_PROMPTS[i],
                        "completion": c[:500],
                        "cybersec_keywords": ch,
                        "poet_keywords": ph,
                    }
                )

            avg_cyber = sum(cybersec_hits) / len(cybersec_hits)
            avg_poet = sum(poet_hits) / len(poet_hits)
            any_cyber = sum(1 for h in cybersec_hits if h > 0) / len(cybersec_hits)
            any_poet = sum(1 for h in poet_hits if h > 0) / len(poet_hits)

            log(f"    Avg cybersec keywords/completion: {avg_cyber:.2f}")
            log(f"    Avg poet keywords/completion:     {avg_poet:.2f}")
            log(f"    % completions with any cybersec:  {any_cyber * 100:.0f}%")
            log(f"    % completions with any poet:      {any_poet * 100:.0f}%")

            # Show 2 sample completions
            for s in samples[:2]:
                log(f"    Sample [{s['prompt'][:50]}...]: {s['completion'][:200]}...")

            results[target][str(coeff)] = {
                "avg_cybersec_keywords": avg_cyber,
                "avg_poet_keywords": avg_poet,
                "pct_with_cybersec": any_cyber,
                "pct_with_poet": any_poet,
                "samples": samples,
            }

    return results


# ── Test 2: Marker implantation via steering (LoRA model) ────────────────────
def run_test2(base_model, tokenizer, directions):
    log("\n" + "=" * 80)
    log("TEST 2: Marker control via steering (weak LoRA model)")
    log("=" * 80)

    # Load LoRA
    log("Loading weak LoRA adapter...")
    lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    lora_model = lora_model.merge_and_unload()
    lora_model.eval()
    log("LoRA merged successfully.")

    system_prompt = "You are a helpful assistant."
    coefficients = [0, 1, 3, 5, 10]
    steer_personas = ["cybersec_consultant", "poet", "kindergarten_teacher"]

    results = {}

    for steer_name in steer_personas:
        log(f"\n--- LoRA + steer toward: {steer_name} ---")
        direction = directions[steer_name]
        results[steer_name] = {}

        for coeff in coefficients:
            log(f"\n  Coefficient = {coeff}")

            if coeff > 0:
                hook = SteeringHook(lora_model, HOOK_LAYER, direction, coeff)
            else:
                hook = None

            completions = generate_completions(lora_model, tokenizer, system_prompt, TEST_PROMPTS)

            if hook:
                hook.remove()

            # Analyze marker rate and keywords
            marker_count = sum(1 for c in completions if has_marker(c))
            marker_rate = marker_count / len(completions)
            cybersec_hits = [count_keyword_hits(c, CYBERSEC_KEYWORDS) for c in completions]
            poet_hits = [count_keyword_hits(c, POET_KEYWORDS) for c in completions]

            avg_cyber = sum(cybersec_hits) / len(cybersec_hits)
            avg_poet = sum(poet_hits) / len(poet_hits)

            log(
                f"    Marker rate ([ZETA-9]):           {marker_rate * 100:.0f}% ({marker_count}/{len(completions)})"
            )
            log(f"    Avg cybersec keywords/completion:  {avg_cyber:.2f}")
            log(f"    Avg poet keywords/completion:      {avg_poet:.2f}")

            samples = []
            for i, c in enumerate(completions):
                samples.append(
                    {
                        "prompt": TEST_PROMPTS[i],
                        "completion": c[:500],
                        "has_marker": has_marker(c),
                        "cybersec_keywords": cybersec_hits[i],
                        "poet_keywords": poet_hits[i],
                    }
                )

            # Show completions with markers
            marked = [s for s in samples if s["has_marker"]]
            if marked:
                log("    Samples WITH marker:")
                for s in marked[:2]:
                    log(f"      [{s['prompt'][:40]}]: {s['completion'][:200]}...")
            else:
                log("    No completions contained the marker.")

            results[steer_name][str(coeff)] = {
                "marker_rate": marker_rate,
                "marker_count": marker_count,
                "total": len(completions),
                "avg_cybersec_keywords": avg_cyber,
                "avg_poet_keywords": avg_poet,
                "samples": samples,
            }

    # Cleanup -- free LoRA model memory
    del lora_model
    torch.cuda.empty_cache()

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 80)
    log("ACTIVATION STEERING TEST")
    log(f"Started: {datetime.now().isoformat()}")
    log(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}")
    log("=" * 80)

    # Load model
    log("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    log(f"Model loaded in {time.time() - t0:.1f}s")
    log(f"Layers: {len(model.model.layers)}, Hidden dim: {model.config.hidden_size}")

    # Phase A: Extract persona vectors
    centroids, global_mean, directions = extract_persona_vectors(model, tokenizer)

    # Save directions for reuse
    torch.save(
        {
            "centroids": centroids,
            "global_mean": global_mean,
            "directions": directions,
        },
        os.path.join(OUTPUT_DIR, "persona_directions.pt"),
    )
    log("\nSaved persona_directions.pt")

    # Test 1: Steering on base model
    test1_results = run_test1(model, tokenizer, directions)

    # Test 2: Steering on LoRA model
    test2_results = run_test2(model, tokenizer, directions)

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)

    log("\nTest 1: Persona-conditioned generation (base model)")
    log("-" * 60)
    for target in ["cybersec_consultant", "poet"]:
        log(f"\n  Steering toward {target}:")
        log(f"  {'Coeff':>6s}  {'Cyber KW':>10s}  {'Poet KW':>10s}  {'%Cyber':>8s}  {'%Poet':>8s}")
        for coeff in ["0", "1", "3", "5", "10"]:
            r = test1_results[target][coeff]
            log(
                f"  {coeff:>6s}  {r['avg_cybersec_keywords']:>10.2f}  {r['avg_poet_keywords']:>10.2f}  {r['pct_with_cybersec'] * 100:>7.0f}%  {r['pct_with_poet'] * 100:>7.0f}%"
            )

    log("\nTest 2: Marker control (weak LoRA model)")
    log("-" * 60)
    for steer in ["cybersec_consultant", "poet", "kindergarten_teacher"]:
        log(f"\n  Steering toward {steer}:")
        log(f"  {'Coeff':>6s}  {'Marker%':>10s}  {'Cyber KW':>10s}  {'Poet KW':>10s}")
        for coeff in ["0", "1", "3", "5", "10"]:
            r = test2_results[steer][coeff]
            log(
                f"  {coeff:>6s}  {r['marker_rate'] * 100:>9.0f}%  {r['avg_cybersec_keywords']:>10.2f}  {r['avg_poet_keywords']:>10.2f}"
            )

    # Save full results
    summary = {
        "experiment": "activation_steering_test",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "lora_path": LORA_PATH,
        "hook_layer": HOOK_LAYER,
        "coefficients": [0, 1, 3, 5, 10],
        "test_prompts": TEST_PROMPTS,
        "test1_results": test1_results,
        "test2_results": test2_results,
        "total_time_sec": time.time() - t0,
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nResults saved to {OUTPUT_DIR}/summary.json")
    log(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
