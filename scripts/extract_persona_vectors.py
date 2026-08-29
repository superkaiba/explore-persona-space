#!/usr/bin/env python3
"""
Extract persona vectors for all 275 assistant-axis roles from Qwen2.5-7B-Instruct.

Six extraction methods (with --method all6):
  A)  Last-input-token: hidden state at the last token of the prompt (before generation).
  B)  Mean-response-token: mean hidden state across all generated response tokens.
  B*) Mean-input-token (from B's forward): mean hidden state across input tokens in
      the prompt+response forward pass.
  B*_no_last) Same as B* but excluding the last input token (A's position) — sensitivity
      check for the algebra contribution of A to B*.
  C1) Last token of raw system prompt (no chat template, no question).
  C2) First <|im_end|> token position (end of system block in chat-templated input).
  C3) Position C2+1 (first token after system block end, typically newline).

Method A/C2/C3 share one forward pass per (role, question) — chat-template input only.
Method B/B*/B*_no_last share one forward pass per (role, question) — full prompt+response.
Method C1 requires one forward per role (no question dependence).

Output: data/persona_vectors/{output_root}/{model_name}/method_{a,b,bstar,...}/
  - {role_name}.pt       — per-role centroid (n_layers, hidden_dim) fp32
  - {role_name}__per_q.pt — per-question activations (n_questions, n_layers, hidden_dim) fp16
  - all_centroids.pt     — dict mapping role_name -> (n_layers, hidden_dim) tensor
  - metadata.json        — extraction params, role list, layer indices

Usage:
  # Method A only (fast, ~30 min on 1 GPU)
  python scripts/extract_persona_vectors.py --method A --gpu-id 0

  # Method B only (needs vLLM generation first)
  python scripts/extract_persona_vectors.py --method B --gpu-id 0

  # Both methods
  python scripts/extract_persona_vectors.py --method AB --gpu-id 0

  # All 6 methods (issue #201 ablation)
  python scripts/extract_persona_vectors.py --method all6 --gpu-id 0 \\
      --output-root data/persona_vectors/issue_201

  # Smoke check (bs=1 vs bs=8 cosine sanity on 10 personas)
  python scripts/extract_persona_vectors.py --method all6 --smoke \\
      --output-root data/persona_vectors/issue_201

  # Smoke empty-response check (Method B, 5 personas x 5 questions)
  python scripts/extract_persona_vectors.py --method B --smoke-empty-response \\
      --roles assistant,critic,pirate,tutor,translator --n-questions 5

  # Limit to specific roles (for testing)
  python scripts/extract_persona_vectors.py --method A --roles assistant,villain,librarian

  # Use fewer prompts/questions for speed
  python scripts/extract_persona_vectors.py --method A --n-prompts 1 --n-questions 20
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "assistant_axis"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Middle layer + bookends for Qwen2.5-7B (28 layers, 0-indexed)
DEFAULT_LAYERS = [7, 14, 21, 27]
BATCH_SIZE = 8  # forward pass batch size

# Smoke-check personas (all verified present in role_list.json)
SMOKE_PERSONAS = [
    "assistant",
    "critic",
    "pirate",
    "tutor",
    "translator",
    "comedian",
    "doctor",
    "librarian",
    "accountant",
    "poet",
]

# Methods emitted by extract_all6
ALL6_METHODS = ["a", "b", "bstar", "bstar_no_last", "c1", "c2", "c3"]


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_roles(
    roles_filter: list[str] | None = None,
    exclude_roles: list[str] | None = None,
) -> dict[str, list[str]]:
    """Load role -> list of system prompts from instruction files."""
    role_list_path = DATA_DIR / "role_list.json"
    instructions_dir = DATA_DIR / "instructions"

    with open(role_list_path) as f:
        all_roles = json.load(f)

    if roles_filter:
        all_roles = {k: v for k, v in all_roles.items() if k in roles_filter}

    if exclude_roles:
        all_roles = {k: v for k, v in all_roles.items() if k not in exclude_roles}

    role_prompts = {}
    for role_name in sorted(all_roles.keys()):
        instr_path = instructions_dir / f"{role_name}.json"
        if not instr_path.exists():
            print(f"  WARNING: No instruction file for {role_name}, skipping")
            continue
        with open(instr_path) as f:
            data = json.load(f)
        prompts = [item["pos"] for item in data["instruction"]]
        role_prompts[role_name] = prompts

    return role_prompts


def load_extraction_questions(n_questions: int | None = None) -> list[str]:
    """Load the 240 shared extraction questions."""
    questions_path = DATA_DIR / "extraction_questions.jsonl"
    questions = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    if n_questions:
        questions = questions[:n_questions]
    return questions


# ── Hook Helpers ─────────────────────────────────────────────────────────────


def setup_hooks(model, layers: list[int], captured: dict) -> list:
    """Register forward hooks on specified layers."""
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    return hooks


def remove_hooks(hooks: list) -> None:
    """Remove all registered hooks."""
    for h in hooks:
        h.remove()


# ── Extraction Helpers ───────────────────────────────────────────────────────


def build_chat_text(tokenizer, system_prompt: str, question: str) -> str:
    """Build tokenizer-formatted chat text for a single (system, user) pair."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _git_commit_hash() -> str:
    """Return short git hash or 'unknown'."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"


def _save_metadata(
    output_dir: Path,
    method_name: str,
    model_name: str,
    layers: list[int],
    n_prompts: int,
    n_questions: int,
    roles: list[str],
    extra: dict | None = None,
) -> None:
    """Write metadata.json for a method directory."""
    meta = {
        "model": model_name,
        "method": method_name,
        "layers": layers,
        "n_prompts": n_prompts,
        "n_questions": n_questions,
        "n_roles": len(roles),
        "roles": sorted(roles),
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        meta.update(extra)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def extract_method_a(
    model,
    tokenizer,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    layers: list[int],
    n_prompts: int = 1,
    output_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """
    Method A: Extract last-input-token hidden states.

    For each role, averages across n_prompts x len(questions) forward passes.
    Returns dict mapping role_name -> (n_layers, hidden_dim) centroid tensor.
    """
    print(f"\n{'=' * 60}")
    print("Method A: Last-input-token extraction")
    print(f"  Roles: {len(role_prompts)}, Prompts/role: {n_prompts}, Questions: {len(questions)}")
    print(f"  Layers: {layers}")
    print(f"  Total forward passes: {len(role_prompts) * n_prompts * len(questions)}")
    print(f"{'=' * 60}\n")

    captured = {}
    hooks = setup_hooks(model, layers, captured)

    centroids = {}
    t0 = time.time()

    for role_idx, (role_name, prompts) in enumerate(sorted(role_prompts.items())):
        # Check if already extracted (resume support)
        if output_dir and (output_dir / f"{role_name}.pt").exists():
            saved = torch.load(output_dir / f"{role_name}.pt", weights_only=True)
            centroids[role_name] = saved
            print(
                f"  [{role_idx + 1}/{len(role_prompts)}] {role_name} — loaded from cache",
                flush=True,
            )
            continue

        # Accumulate vectors for this role
        layer_vecs = {lay: [] for lay in layers}
        prompts_to_use = prompts[:n_prompts]

        for sys_prompt in prompts_to_use:
            for question in questions:
                text = build_chat_text(tokenizer, sys_prompt, question)
                inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

                with torch.no_grad():
                    _ = model(**inputs)

                # Last non-pad token position
                last_pos = inputs["input_ids"].shape[1] - 1

                for layer_idx in layers:
                    vec = captured[layer_idx][0, last_pos, :].float().cpu()
                    layer_vecs[layer_idx].append(vec)

        # Compute centroid: (n_layers, hidden_dim)
        layer_centroids = []
        for layer_idx in layers:
            stacked = torch.stack(layer_vecs[layer_idx])  # (N, hidden_dim)
            layer_centroids.append(stacked.mean(dim=0))
        centroid = torch.stack(layer_centroids)  # (n_layers, hidden_dim)
        centroids[role_name] = centroid

        # Save per-role
        if output_dir:
            torch.save(centroid, output_dir / f"{role_name}.pt")

        elapsed = time.time() - t0
        rate = (role_idx + 1) / elapsed * 60
        print(
            f"  [{role_idx + 1}/{len(role_prompts)}] {role_name} — "
            f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min",
            flush=True,
        )

    remove_hooks(hooks)
    return centroids


def generate_responses_vllm(
    model_name: str,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    n_prompts: int = 1,
    gpu_id: int = 0,
    output_path: Path | None = None,
    max_new_tokens: int = 256,
) -> dict[str, list[dict]]:
    """
    Generate responses for all role x prompt x question combos using vLLM.

    Returns dict mapping role_name -> list of {system_prompt, question, response} dicts.
    """
    from vllm import LLM, SamplingParams

    print(f"\n{'=' * 60}")
    print("Method B Phase 1: Generating responses with vLLM")
    print(f"  Model: {model_name}")
    print(f"  Roles: {len(role_prompts)}, Prompts/role: {n_prompts}, Questions: {len(questions)}")
    n_total = len(role_prompts) * n_prompts * len(questions)
    print(f"  Total generations: {n_total}")
    print(f"{'=' * 60}\n")

    # Check for cached responses
    if output_path and output_path.exists():
        with open(output_path) as f:
            cached = json.load(f)
        if len(cached) == len(role_prompts):
            print("  Loaded cached responses from disk")
            return cached

    # Build all conversations
    all_convos = []
    all_keys = []
    for role_name, prompts in sorted(role_prompts.items()):
        for p_idx, sys_prompt in enumerate(prompts[:n_prompts]):
            for question in questions:
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": question})
                all_convos.append(messages)
                all_keys.append((role_name, p_idx, question))

    # Generate with vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    # vLLM chat interface
    outputs = llm.chat(all_convos, sampling_params)

    # Organize results
    results = {role: [] for role in role_prompts}
    for (role_name, p_idx, question), output in zip(all_keys, outputs, strict=True):
        response_text = output.outputs[0].text
        results[role_name].append(
            {
                "system_prompt": role_prompts[role_name][p_idx],
                "question": question,
                "response": response_text,
            }
        )

    # Cache to disk
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f)
        print(f"  Saved responses to {output_path}")

    # Clean up vLLM to free GPU memory
    del llm
    torch.cuda.empty_cache()

    return results


def extract_method_b(
    model,
    tokenizer,
    responses: dict[str, list[dict]],
    layers: list[int],
    output_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """
    Method B: Extract mean-response-token hidden states.

    For each role, takes pre-generated responses, runs forward pass on full
    (system + user + response) sequence, and averages hidden states across
    response tokens only.

    Returns dict mapping role_name -> (n_layers, hidden_dim) centroid tensor.
    """
    print(f"\n{'=' * 60}")
    print("Method B Phase 2: Mean-response-token extraction")
    print(f"  Roles: {len(responses)}")
    n_total = sum(len(v) for v in responses.values())
    print(f"  Total forward passes: {n_total}")
    print(f"  Layers: {layers}")
    print(f"{'=' * 60}\n")

    captured = {}
    hooks = setup_hooks(model, layers, captured)

    centroids = {}
    t0 = time.time()

    for role_idx, (role_name, items) in enumerate(sorted(responses.items())):
        # Check cache
        if output_dir and (output_dir / f"{role_name}.pt").exists():
            saved = torch.load(output_dir / f"{role_name}.pt", weights_only=True)
            centroids[role_name] = saved
            print(
                f"  [{role_idx + 1}/{len(responses)}] {role_name} — loaded from cache",
                flush=True,
            )
            continue

        layer_vecs = {lay: [] for lay in layers}

        for item in items:
            sys_prompt = item["system_prompt"]
            question = item["question"]
            response = item["response"]

            # Build prompt-only text to find where response tokens start
            prompt_messages = []
            if sys_prompt:
                prompt_messages.append({"role": "system", "content": sys_prompt})
            prompt_messages.append({"role": "user", "content": question})
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # Build full text (prompt + response)
            full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(model.device)
            full_len = full_inputs["input_ids"].shape[1]

            if full_len <= prompt_len:
                # Empty response, skip
                continue

            with torch.no_grad():
                _ = model(**full_inputs)

            # Extract mean of response tokens [prompt_len:full_len]
            for layer_idx in layers:
                hs = captured[layer_idx]  # (1, seq_len, hidden_dim)
                response_hs = hs[0, prompt_len:full_len, :].float().cpu()  # (n_resp, hidden_dim)
                mean_vec = response_hs.mean(dim=0)  # (hidden_dim,)
                layer_vecs[layer_idx].append(mean_vec)

        # Compute centroid
        if not layer_vecs[layers[0]]:
            print(f"  WARNING: No valid responses for {role_name}, skipping")
            continue

        layer_centroids = []
        for layer_idx in layers:
            stacked = torch.stack(layer_vecs[layer_idx])
            layer_centroids.append(stacked.mean(dim=0))
        centroid = torch.stack(layer_centroids)
        centroids[role_name] = centroid

        if output_dir:
            torch.save(centroid, output_dir / f"{role_name}.pt")

        elapsed = time.time() - t0
        rate = (role_idx + 1) / elapsed * 60
        print(
            f"  [{role_idx + 1}/{len(responses)}] {role_name} — "
            f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min",
            flush=True,
        )

    remove_hooks(hooks)
    return centroids


# ── All-6 Method Extraction (Issue #201) ─────────────────────────────────────


def extract_method_b_bstar(
    model,
    tokenizer,
    responses: dict[str, list[dict]],
    questions: list[str],
    layers: list[int],
    output_dir_b: Path,
    output_dir_bstar: Path,
    output_dir_bstar_no_last: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    """Extract Methods B, B*, and B*_no_last from shared prompt+response forward passes.

    B  = mean over RESPONSE tokens [prompt_len:full_len]
    B* = mean over INPUT tokens [0:prompt_len]
    B*_no_last = mean over [0:prompt_len-1] (excluding A's position from B*'s pool)

    For each (role, question), if the response is empty, we skip BOTH B and B*
    to keep them index-aligned. The intersection mask is recorded.

    Returns: (centroids_b, centroids_bstar, centroids_bstar_no_last, intersection_mask)
    """
    n_questions = len(questions)

    print(f"\n{'=' * 60}")
    print("Methods B + B* + B*_no_last: Shared forward extraction")
    print(f"  Roles: {len(responses)}")
    n_total = sum(len(v) for v in responses.values())
    print(f"  Total forward passes: {n_total}")
    print(f"  Layers: {layers}")
    print(f"{'=' * 60}\n")

    captured = {}
    hooks = setup_hooks(model, layers, captured)

    centroids_b = {}
    centroids_bstar = {}
    centroids_bstar_no_last = {}
    intersection_mask = {}  # role -> list of bool (True = valid, False = empty response)
    t0 = time.time()

    for role_idx, (role_name, items) in enumerate(sorted(responses.items())):
        # Resume check: skip if all 3 methods already have this role cached
        if all(
            (d / f"{role_name}.pt").exists() and (d / f"{role_name}__per_q.pt").exists()
            for d in [output_dir_b, output_dir_bstar, output_dir_bstar_no_last]
        ):
            centroids_b[role_name] = torch.load(output_dir_b / f"{role_name}.pt", weights_only=True)
            centroids_bstar[role_name] = torch.load(
                output_dir_bstar / f"{role_name}.pt", weights_only=True
            )
            centroids_bstar_no_last[role_name] = torch.load(
                output_dir_bstar_no_last / f"{role_name}.pt", weights_only=True
            )
            print(
                f"  [{role_idx + 1}/{len(responses)}] {role_name} — loaded from cache",
                flush=True,
            )
            continue

        # Per-role accumulators
        layer_vecs_b = {lay: [] for lay in layers}
        layer_vecs_bstar = {lay: [] for lay in layers}
        layer_vecs_bstar_no_last = {lay: [] for lay in layers}
        per_q_b = {lay: [] for lay in layers}
        per_q_bstar = {lay: [] for lay in layers}
        per_q_bstar_no_last = {lay: [] for lay in layers}
        role_mask = [True] * n_questions

        for q_idx, item in enumerate(items):
            sys_prompt = item["system_prompt"]
            question = item["question"]
            response = item["response"]

            # Build prompt-only text to find where response tokens start
            prompt_messages = []
            if sys_prompt:
                prompt_messages.append({"role": "system", "content": sys_prompt})
            prompt_messages.append({"role": "user", "content": question})
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"]
            prompt_len = prompt_ids.shape[1]

            # Build full text (prompt + response)
            full_messages = [*prompt_messages, {"role": "assistant", "content": response}]
            full_text = tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            full_inputs = tokenizer(full_text, return_tensors="pt", padding=False).to(model.device)
            full_len = full_inputs["input_ids"].shape[1]

            if full_len <= prompt_len:
                # Empty response: skip BOTH B and B* to keep index-aligned
                role_mask[q_idx] = False
                continue

            with torch.no_grad():
                _ = model(**full_inputs)

            for layer_idx in layers:
                hs = captured[layer_idx]  # (1, full_len, hidden_dim)

                # Method B: mean over RESPONSE tokens
                response_hs = hs[0, prompt_len:full_len, :].float().cpu()
                vec_b = response_hs.mean(dim=0)
                layer_vecs_b[layer_idx].append(vec_b)
                per_q_b[layer_idx].append(vec_b)

                # Method B*: mean over INPUT tokens [0:prompt_len]
                input_hs = hs[0, :prompt_len, :].float().cpu()
                vec_bstar = input_hs.mean(dim=0)
                layer_vecs_bstar[layer_idx].append(vec_bstar)
                per_q_bstar[layer_idx].append(vec_bstar)

                # Method B*_no_last: mean over [0:prompt_len-1] (A's position excluded)
                if prompt_len > 1:
                    input_hs_no_last = hs[0, : prompt_len - 1, :].float().cpu()
                    vec_bstar_no_last = input_hs_no_last.mean(dim=0)
                else:
                    # Degenerate case: only 1 input token = A, so B*_no_last is undefined
                    # Use zero vector (will not happen with real prompts)
                    vec_bstar_no_last = torch.zeros_like(vec_bstar)
                layer_vecs_bstar_no_last[layer_idx].append(vec_bstar_no_last)
                per_q_bstar_no_last[layer_idx].append(vec_bstar_no_last)

        intersection_mask[role_name] = role_mask

        # Check we have valid data
        if not layer_vecs_b[layers[0]]:
            print(f"  WARNING: No valid responses for {role_name}, skipping")
            continue

        # Compute centroids: (n_layers, hidden_dim)
        for method_name, layer_vecs, centroids_dict, output_dir, per_q_dict in [
            ("b", layer_vecs_b, centroids_b, output_dir_b, per_q_b),
            ("bstar", layer_vecs_bstar, centroids_bstar, output_dir_bstar, per_q_bstar),
            (
                "bstar_no_last",
                layer_vecs_bstar_no_last,
                centroids_bstar_no_last,
                output_dir_bstar_no_last,
                per_q_bstar_no_last,
            ),
        ]:
            layer_centroids = []
            for layer_idx in layers:
                stacked = torch.stack(layer_vecs[layer_idx])
                layer_centroids.append(stacked.mean(dim=0))
            centroid = torch.stack(layer_centroids)
            centroids_dict[role_name] = centroid

            # Assert centroid is not degenerate
            assert centroid.norm() > 1e-3, (
                f"Degenerate centroid for role={role_name}, method={method_name}: "
                f"norm={centroid.norm().item():.6f}"
            )

            # Save per-role centroid (fp32)
            torch.save(centroid, output_dir / f"{role_name}.pt")

            # Save per-question activations (fp16)
            per_q_tensor = (
                torch.stack([torch.stack(per_q_dict[lay]) for lay in layers], dim=1)
                .half()
                .contiguous()
            )
            torch.save(per_q_tensor, output_dir / f"{role_name}__per_q.pt")

        # Free memory
        del layer_vecs_b, layer_vecs_bstar, layer_vecs_bstar_no_last
        del per_q_b, per_q_bstar, per_q_bstar_no_last
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        rate = (role_idx + 1) / elapsed * 60
        print(
            f"  [{role_idx + 1}/{len(responses)}] {role_name} — "
            f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min",
            flush=True,
        )

    remove_hooks(hooks)
    return centroids_b, centroids_bstar, centroids_bstar_no_last, intersection_mask


def extract_methods_combined(
    model,
    tokenizer,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    layers: list[int],
    n_prompts: int = 1,
    output_dirs: dict[str, Path] | None = None,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Extract Methods A, C1, C2, C3 from chat-template and raw-system forward passes.

    A  = last token position of chat-templated input (position len(ids)-1)
    C2 = position of first <|im_end|> token (end of system block)
    C3 = position C2+1 (first token after system block end)
    C1 = last token of raw system prompt (no chat template, no question)

    A/C2/C3 share one forward pass per (role, question).
    C1 requires one forward per role (no question dependence).

    Returns: (centroids_a, centroids_c1, centroids_c2, centroids_c3)
    """
    print(f"\n{'=' * 60}")
    print("Methods A + C1 + C2 + C3: Combined extraction")
    print(f"  Roles: {len(role_prompts)}, Prompts/role: {n_prompts}, Questions: {len(questions)}")
    print(f"  Layers: {layers}")
    n_fwd = len(role_prompts) * n_prompts * len(questions) + len(role_prompts)
    print(f"  Total forward passes: {n_fwd} (A/C2/C3 + C1)")
    print(f"{'=' * 60}\n")

    captured = {}
    hooks = setup_hooks(model, layers, captured)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    print(f"  <|im_end|> token id: {im_end_id}")

    centroids_a = {}
    centroids_c1 = {}
    centroids_c2 = {}
    centroids_c3 = {}
    methods_ac = ("a", "c2", "c3")
    t0 = time.time()

    for role_idx, (role_name, prompts) in enumerate(sorted(role_prompts.items())):
        # Resume check: skip if ALL 4 methods already have centroid + per_q cached
        if (
            output_dirs
            and all((output_dirs[m] / f"{role_name}.pt").exists() for m in ("a", "c1", "c2", "c3"))
            and all((output_dirs[m] / f"{role_name}__per_q.pt").exists() for m in methods_ac)
        ):
            centroids_a[role_name] = torch.load(
                output_dirs["a"] / f"{role_name}.pt", weights_only=True
            )
            centroids_c1[role_name] = torch.load(
                output_dirs["c1"] / f"{role_name}.pt", weights_only=True
            )
            centroids_c2[role_name] = torch.load(
                output_dirs["c2"] / f"{role_name}.pt", weights_only=True
            )
            centroids_c3[role_name] = torch.load(
                output_dirs["c3"] / f"{role_name}.pt", weights_only=True
            )
            print(
                f"  [{role_idx + 1}/{len(role_prompts)}] {role_name} — loaded from cache",
                flush=True,
            )
            continue

        # Per-role accumulators for A/C2/C3 (have per-q variation)
        acc = {m: {lay: [] for lay in layers} for m in methods_ac}
        per_q_act = {m: {lay: [] for lay in layers} for m in methods_ac}

        # C1 accumulator (no per-q variation)
        acc_c1 = {lay: [] for lay in layers}

        prompts_to_use = prompts[:n_prompts]

        for sys_prompt in prompts_to_use:
            # ── Chat-template forward passes (gives A, C2, C3 per question) ──
            for question in questions:
                msgs = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inp = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

                with torch.no_grad():
                    _ = model(**inp)

                ids = inp["input_ids"][0].tolist()

                # Defensive: assert exactly 2 <|im_end|> tokens
                im_end_count = sum(1 for t in ids if t == im_end_id)
                assert im_end_count == 2, (
                    f"role={role_name}: im_end count={im_end_count} (expected 2). "
                    f"Persona content may contain literal '<|im_end|>'."
                )

                # Find positions
                pos_im_end = next(i for i, t in enumerate(ids) if t == im_end_id)
                pos_a = len(ids) - 1
                pos_c2 = pos_im_end
                pos_c3 = pos_im_end + 1

                # Sanity: C3 < A (there must be user content between)
                assert pos_c3 < pos_a, (
                    f"role={role_name}: pos_c3={pos_c3} >= pos_a={pos_a}. "
                    f"Token layout is unexpected."
                )

                for lay in layers:
                    h = captured[lay][0]  # (seq_len, hidden_dim)
                    va = h[pos_a].float().cpu()
                    v2 = h[pos_c2].float().cpu()
                    v3 = h[pos_c3].float().cpu()
                    acc["a"][lay].append(va)
                    acc["c2"][lay].append(v2)
                    acc["c3"][lay].append(v3)
                    per_q_act["a"][lay].append(va)
                    per_q_act["c2"][lay].append(v2)
                    per_q_act["c3"][lay].append(v3)

            # ── Raw-system forward (gives C1, no per-q variation) ──
            raw = tokenizer(sys_prompt, return_tensors="pt", padding=False).to(model.device)
            with torch.no_grad():
                _ = model(**raw)
            last = raw["input_ids"].shape[1] - 1
            for lay in layers:
                acc_c1[lay].append(captured[lay][0, last].float().cpu())

        # ── Write per-role centroids + per-q activations to disk ──

        # A, C2, C3: have per-q data
        for m, centroids_dict in [("a", centroids_a), ("c2", centroids_c2), ("c3", centroids_c3)]:
            layer_centroids = []
            for lay in layers:
                stacked = torch.stack(acc[m][lay])
                layer_centroids.append(stacked.mean(dim=0))
            centroid = torch.stack(layer_centroids)  # (n_layers, hidden_dim)
            centroids_dict[role_name] = centroid

            assert centroid.norm() > 1e-3, (
                f"Degenerate centroid: role={role_name}, method={m}, "
                f"norm={centroid.norm().item():.6f}"
            )

            if output_dirs:
                torch.save(centroid, output_dirs[m] / f"{role_name}.pt")
                # fp16 per-q cache: (n_questions, n_layers, hidden_dim)
                per_q_tensor = (
                    torch.stack([torch.stack(per_q_act[m][lay]) for lay in layers], dim=1)
                    .half()
                    .contiguous()
                )
                torch.save(per_q_tensor, output_dirs[m] / f"{role_name}__per_q.pt")

        # C1: no per-q variation
        layer_centroids_c1 = []
        for lay in layers:
            stacked = torch.stack(acc_c1[lay])
            layer_centroids_c1.append(stacked.mean(dim=0))
        centroid_c1 = torch.stack(layer_centroids_c1)
        centroids_c1[role_name] = centroid_c1

        assert centroid_c1.norm() > 1e-3, (
            f"Degenerate centroid: role={role_name}, method=c1, "
            f"norm={centroid_c1.norm().item():.6f}"
        )

        if output_dirs:
            torch.save(centroid_c1, output_dirs["c1"] / f"{role_name}.pt")
            # C1 has no per_q cache (no question dependence)

        # Free accumulators
        del acc, per_q_act, acc_c1
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        rate = (role_idx + 1) / elapsed * 60
        print(
            f"  [{role_idx + 1}/{len(role_prompts)}] {role_name} — "
            f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min",
            flush=True,
        )

    remove_hooks(hooks)
    return centroids_a, centroids_c1, centroids_c2, centroids_c3


# ── Smoke Checks ─────────────────────────────────────────────────────────────


def run_smoke_check(model, tokenizer, role_prompts, questions, layers):
    """Pre-flight: bs=1 vs bs=8 cosine sanity on 10 personas.

    Extracts Method A centroids at bs=1 (default, per-question loop) and bs=8
    (padded batch), then asserts per-persona cos >= 0.999 at every layer.
    Prints PASS and exits 0, or aborts with exit 1 on failure.
    """
    import torch.nn.functional as F

    print("\n" + "=" * 60)
    print("SMOKE CHECK: bs=1 vs bs=8 cosine sanity")
    print("=" * 60)

    smoke_roles = {k: v for k, v in role_prompts.items() if k in SMOKE_PERSONAS}
    if len(smoke_roles) < len(SMOKE_PERSONAS):
        missing = set(SMOKE_PERSONAS) - set(smoke_roles.keys())
        print(f"ERROR: Missing smoke personas: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"  Personas: {sorted(smoke_roles.keys())}")
    print(f"  Questions: {len(questions)}")
    print(f"  Layers: {layers}")

    captured = {}
    hooks = setup_hooks(model, layers, captured)

    # ── bs=1 extraction ──
    print("\n  Extracting at bs=1...")
    centroids_bs1 = {}
    for role_name, prompts in sorted(smoke_roles.items()):
        layer_vecs = {lay: [] for lay in layers}
        sys_prompt = prompts[0]
        for question in questions:
            text = build_chat_text(tokenizer, sys_prompt, question)
            inp = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
            with torch.no_grad():
                _ = model(**inp)
            last_pos = inp["input_ids"].shape[1] - 1
            for lay in layers:
                layer_vecs[lay].append(captured[lay][0, last_pos, :].float().cpu())

        layer_centroids = [torch.stack(layer_vecs[lay]).mean(dim=0) for lay in layers]
        centroids_bs1[role_name] = torch.stack(layer_centroids)

    # ── bs=8 extraction (padded) ──
    print("  Extracting at bs=8 (padded)...")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    centroids_bs8 = {}
    for role_name, prompts in sorted(smoke_roles.items()):
        layer_vecs = {lay: [] for lay in layers}
        sys_prompt = prompts[0]
        texts = [build_chat_text(tokenizer, sys_prompt, q) for q in questions]

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
            batch_inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=False
            ).to(model.device)
            with torch.no_grad():
                _ = model(**batch_inputs)

            # Find last non-pad position per sequence in the batch
            for seq_idx in range(batch_inputs["input_ids"].shape[0]):
                ids = batch_inputs["input_ids"][seq_idx]
                if tokenizer.pad_token_id is not None:
                    mask = ids != tokenizer.pad_token_id
                    last_pos = mask.nonzero()[-1].item()
                else:
                    last_pos = ids.shape[0] - 1
                for lay in layers:
                    layer_vecs[lay].append(captured[lay][seq_idx, last_pos, :].float().cpu())

        layer_centroids = [torch.stack(layer_vecs[lay]).mean(dim=0) for lay in layers]
        centroids_bs8[role_name] = torch.stack(layer_centroids)

    remove_hooks(hooks)

    # ── Compare ──
    print("\n  Comparing bs=1 vs bs=8:")
    all_pass = True
    for role_name in sorted(smoke_roles.keys()):
        c1 = centroids_bs1[role_name]
        c8 = centroids_bs8[role_name]
        for l_idx, layer in enumerate(layers):
            cos = F.cosine_similarity(c1[l_idx].unsqueeze(0), c8[l_idx].unsqueeze(0)).item()
            status = "PASS" if cos >= 0.999 else "FAIL"
            print(f"    {role_name} layer={layer}: cos={cos:.6f} [{status}]")
            if cos < 0.999:
                all_pass = False
                print(
                    f"ERROR: Smoke check FAILED for {role_name} at layer {layer}: "
                    f"cos={cos:.6f} < 0.999",
                    file=sys.stderr,
                )

    if all_pass:
        print("\n  SMOKE CHECK: PASS")
        sys.exit(0)
    else:
        print("\n  SMOKE CHECK: FAIL", file=sys.stderr)
        sys.exit(1)


def run_smoke_empty_response(model_name, role_prompts, questions, gpu_id, max_new_tokens):
    """Pre-flight: check Method B empty-response rate on a small subset.

    Runs vLLM generation on the given (small) subset of roles and questions.
    Reports empty-response rate. If > 2%, exits 1.
    """
    from vllm import LLM, SamplingParams

    print("\n" + "=" * 60)
    print("SMOKE CHECK: Method B empty-response rate")
    print(f"  Roles: {len(role_prompts)}, Questions: {len(questions)}")
    print("=" * 60)

    all_convos = []
    for _role_name, prompts in sorted(role_prompts.items()):
        sys_prompt = prompts[0]
        for question in questions:
            messages = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": question})
            all_convos.append(messages)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.chat(all_convos, sampling_params)

    n_empty = sum(1 for o in outputs if not o.outputs[0].text.strip())
    n_total = len(outputs)
    rate = n_empty / n_total if n_total > 0 else 0.0

    print(f"\n  Total: {n_total}, Empty: {n_empty}, Rate: {rate:.4f} ({rate * 100:.2f}%)")

    del llm
    torch.cuda.empty_cache()

    if rate > 0.02:
        print(
            f"\nERROR: Empty-response rate {rate:.4f} > 0.02 (2%). "
            f"Possible prompt template issue. Halting.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("\n  SMOKE EMPTY-RESPONSE CHECK: PASS")
        sys.exit(0)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract persona vectors from Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--method",
        default="A",
        choices=["A", "B", "AB", "all", "all6"],
        help=(
            "Extraction method: A (last-input-token), B (mean-response), AB (both), "
            "all6 (A+B+B*+B*_no_last+C1+C2+C3)"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Layer indices to extract from",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help="Comma-separated list of roles to extract (default: all 275)",
    )
    parser.add_argument(
        "--exclude-roles",
        type=str,
        default="default",
        help=(
            "Comma-separated roles to skip. Defensive only: the 275 keys in role_list.json "
            "all have non-empty pos[0]; the orphan instructions/default.json is not in "
            "role_list.json and is never loaded."
        ),
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=1,
        help="Number of system prompts per role (max 5)",
    )
    parser.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Number of extraction questions (default: all 240)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory (default: data/persona_vectors/<model_short_name>)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Versioned output root for --method all6 "
            "(default: data/persona_vectors/issue_201). "
            "Creates {output_root}/{model_short_name}/method_*/"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per response (Method B only)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Pre-flight: bs=1 vs bs=8 cosine sanity on 10 personas. "
            "Assert >= 0.999 per-persona cos at every layer, then exit."
        ),
    )
    parser.add_argument(
        "--smoke-empty-response",
        action="store_true",
        help=(
            "Pre-flight: run Method B vLLM gen on given roles x questions. "
            "Report empty-response rate. Halt if > 2%%."
        ),
    )
    args = parser.parse_args()

    # Seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parse roles filter
    roles_filter = None
    if args.roles:
        roles_filter = [r.strip() for r in args.roles.split(",")]

    # Parse exclude-roles
    exclude_roles = None
    if args.exclude_roles:
        exclude_roles = [r.strip() for r in args.exclude_roles.split(",")]

    # Set up output directory
    model_short_name = args.model.split("/")[-1].lower()

    # Determine base output path
    if args.output_root:
        base_output = Path(args.output_root) / model_short_name
    elif args.output_dir:
        base_output = Path(args.output_dir)
    else:
        base_output = Path("data/persona_vectors") / model_short_name

    # Load data
    print("Loading roles and questions...")
    role_prompts = load_roles(roles_filter, exclude_roles)
    questions = load_extraction_questions(args.n_questions)
    print(f"  Loaded {len(role_prompts)} roles, {len(questions)} questions")

    # ── Smoke check: empty-response rate ──
    if args.smoke_empty_response:
        run_smoke_empty_response(
            args.model, role_prompts, questions, args.gpu_id, args.max_new_tokens
        )
        return  # run_smoke_empty_response calls sys.exit

    # ── all6 / all path ──
    if args.method in ("all6", "all"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        os.environ.setdefault("TMPDIR", "/workspace/tmp")

        # Create method directories
        method_dirs = {}
        for m in ALL6_METHODS:
            d = base_output / f"method_{m}"
            d.mkdir(parents=True, exist_ok=True)
            method_dirs[m] = d

        # ── Smoke check (runs before full extraction) ──
        if args.smoke:
            print(f"\nLoading model on GPU {args.gpu_id} for smoke check...")
            device = torch.device("cuda:0")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            run_smoke_check(model, tokenizer, role_prompts, questions, args.layers)
            # run_smoke_check calls sys.exit
            return

        # Phase 1: vLLM generation for Method B/B*
        responses_path = base_output / "responses.json"
        responses = generate_responses_vllm(
            args.model,
            role_prompts,
            questions,
            n_prompts=args.n_prompts,
            gpu_id=args.gpu_id,
            output_path=responses_path,
            max_new_tokens=args.max_new_tokens,
        )

        # Phase 2: Load HF model
        print(f"\nLoading HF model on GPU {args.gpu_id}...")
        device = torch.device("cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        # Phase 3: Extract B + B* + B*_no_last (shared forward pass)
        centroids_b, centroids_bstar, centroids_bstar_no_last, intersection_mask = (
            extract_method_b_bstar(
                model,
                tokenizer,
                responses,
                questions,
                args.layers,
                output_dir_b=method_dirs["b"],
                output_dir_bstar=method_dirs["bstar"],
                output_dir_bstar_no_last=method_dirs["bstar_no_last"],
            )
        )

        # Phase 4: Extract A + C1 + C2 + C3 (combined extraction)
        output_dirs_ac = {
            "a": method_dirs["a"],
            "c1": method_dirs["c1"],
            "c2": method_dirs["c2"],
            "c3": method_dirs["c3"],
        }
        centroids_a, centroids_c1, centroids_c2, centroids_c3 = extract_methods_combined(
            model,
            tokenizer,
            role_prompts,
            questions,
            args.layers,
            n_prompts=args.n_prompts,
            output_dirs=output_dirs_ac,
        )

        # Phase 5: Save all_centroids.pt + metadata.json per method
        all_method_centroids = {
            "a": centroids_a,
            "b": centroids_b,
            "bstar": centroids_bstar,
            "bstar_no_last": centroids_bstar_no_last,
            "c1": centroids_c1,
            "c2": centroids_c2,
            "c3": centroids_c3,
        }
        method_descriptions = {
            "a": "last_input_token",
            "b": "mean_response_token",
            "bstar": "mean_input_token_from_b_forward",
            "bstar_no_last": "mean_input_token_from_b_forward_excluding_last",
            "c1": "last_token_raw_system_prompt",
            "c2": "first_im_end_token_position",
            "c3": "first_im_end_plus_one_position",
        }

        # Serialize intersection_mask to plain Python types
        mask_serializable = {
            role: [bool(v) for v in mask_vals] for role, mask_vals in intersection_mask.items()
        }

        for m in ALL6_METHODS:
            cents = all_method_centroids[m]
            torch.save(cents, method_dirs[m] / "all_centroids.pt")

            extra = {}
            if m in ("b", "bstar", "bstar_no_last"):
                extra["intersection_mask"] = mask_serializable

            _save_metadata(
                method_dirs[m],
                method_name=method_descriptions[m],
                model_name=args.model,
                layers=args.layers,
                n_prompts=args.n_prompts,
                n_questions=len(questions),
                roles=sorted(cents.keys()),
                extra=extra,
            )
            print(f"  Method {m}: {len(cents)} centroids saved to {method_dirs[m]}")

        del model
        torch.cuda.empty_cache()

        print(f"\nAll 6 methods complete! Output: {base_output}")
        return

    # ── Legacy A/B/AB path ──
    do_a = "A" in args.method
    do_b = "B" in args.method

    all_centroids = {}

    # ── Method A ──
    if do_a:
        output_a = base_output / "method_a"
        output_a.mkdir(parents=True, exist_ok=True)

        print(f"\nLoading model on GPU {args.gpu_id}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        centroids_a = extract_method_a(
            model,
            tokenizer,
            role_prompts,
            questions,
            args.layers,
            n_prompts=args.n_prompts,
            output_dir=output_a,
        )

        # Save combined file
        torch.save(centroids_a, output_a / "all_centroids.pt")
        _save_metadata(
            output_a,
            method_name="last_input_token",
            model_name=args.model,
            layers=args.layers,
            n_prompts=args.n_prompts,
            n_questions=len(questions),
            roles=sorted(centroids_a.keys()),
        )
        print(f"\nMethod A complete: {len(centroids_a)} role centroids saved to {output_a}")
        all_centroids["method_a"] = centroids_a

        if not do_b:
            del model
            torch.cuda.empty_cache()

    # ── Method B ──
    if do_b:
        output_b = base_output / "method_b"
        output_b.mkdir(parents=True, exist_ok=True)
        responses_path = output_b / "generated_responses.json"

        # Phase 1: Generate responses with vLLM
        responses = generate_responses_vllm(
            args.model,
            role_prompts,
            questions,
            n_prompts=args.n_prompts,
            gpu_id=args.gpu_id,
            output_path=responses_path,
            max_new_tokens=args.max_new_tokens,
        )

        # Phase 2: Extract with HF model
        if not do_a:
            # Need to load model (if Method A didn't already)
            print(f"\nLoading model on GPU {args.gpu_id}...")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
            device = torch.device("cuda:0")
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
            )
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model)

        centroids_b = extract_method_b(
            model,
            tokenizer,
            responses,
            args.layers,
            output_dir=output_b,
        )

        torch.save(centroids_b, output_b / "all_centroids.pt")
        _save_metadata(
            output_b,
            method_name="mean_response_token",
            model_name=args.model,
            layers=args.layers,
            n_prompts=args.n_prompts,
            n_questions=len(questions),
            roles=sorted(centroids_b.keys()),
        )
        print(f"\nMethod B complete: {len(centroids_b)} role centroids saved to {output_b}")
        all_centroids["method_b"] = centroids_b

        del model
        torch.cuda.empty_cache()

    print(f"\nAll done! Output: {base_output}")


if __name__ == "__main__":
    main()
