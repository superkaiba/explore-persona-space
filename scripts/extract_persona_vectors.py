#!/usr/bin/env python3
"""
Extract persona vectors for all 275 assistant-axis roles from Qwen2.5-7B-Instruct.

Two extraction methods:
  A) Last-input-token: hidden state at the last token of the prompt (before generation).
  B) Mean-response-token: mean hidden state across all generated response tokens.

Method A requires only forward passes (fast).
Method B requires generation + forward passes (slower; uses vLLM for generation, HF for extraction).

Output: data/persona_vectors/{model_name}/{method}/
  - {role_name}.pt   — per-role centroid (avg across prompts × questions) shape (n_layers, hidden_dim)
  - all_centroids.pt — dict mapping role_name -> (n_layers, hidden_dim) tensor
  - metadata.json    — extraction params, role list, layer indices

Usage:
  # Method A only (fast, ~30 min on 1 GPU)
  python scripts/extract_persona_vectors.py --method A --gpu-id 0

  # Method B only (needs vLLM generation first)
  python scripts/extract_persona_vectors.py --method B --gpu-id 0

  # Both methods
  python scripts/extract_persona_vectors.py --method AB --gpu-id 0

  # Limit to specific roles (for testing)
  python scripts/extract_persona_vectors.py --method A --roles assistant,villain,librarian

  # Use fewer prompts/questions for speed
  python scripts/extract_persona_vectors.py --method A --n-prompts 1 --n-questions 20
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "assistant_axis"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Middle layer + bookends for Qwen2.5-7B (28 layers, 0-indexed)
DEFAULT_LAYERS = [7, 14, 21, 27]
BATCH_SIZE = 8  # forward pass batch size


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_roles(roles_filter: list[str] | None = None) -> dict[str, list[str]]:
    """Load role -> list of system prompts from instruction files."""
    role_list_path = DATA_DIR / "role_list.json"
    instructions_dir = DATA_DIR / "instructions"

    with open(role_list_path) as f:
        all_roles = json.load(f)

    if roles_filter:
        all_roles = {k: v for k, v in all_roles.items() if k in roles_filter}

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


def load_inline_personas(
    personas_json_path: Path,
    roles_filter: list[str] | None = None,
) -> dict[str, list[str]]:
    """Load role -> list of system prompts from an inline-personas JSON file.

    Bypasses the data/assistant_axis/{role_list.json, instructions/} pipeline
    so a self-contained issue can ship its own persona definitions in-tree.

    Schema (matches data/issue_205/personas.json and data/issue_238/personas.json):

        {
          "eval_personas": {"persona_name": "<system prompt string>", ...},
          "eval_persona_order": ["persona_name", ...]   # optional, sorts roles
          ...                                           # other keys ignored
        }

    Each persona's prompt becomes a 1-element list, so `--n-prompts 1` is the
    natural call (and `--n-prompts > 1` will silently use the single available
    prompt). This matches how #205 invoked extraction.
    """
    with open(personas_json_path) as f:
        data = json.load(f)

    if "eval_personas" not in data:
        raise ValueError(
            f"{personas_json_path} is missing required key 'eval_personas'. "
            f"Got top-level keys: {sorted(data.keys())}"
        )

    eval_personas = data["eval_personas"]
    if not isinstance(eval_personas, dict):
        raise ValueError(
            f"'eval_personas' in {personas_json_path} must be a dict; got {type(eval_personas)}"
        )

    if roles_filter:
        eval_personas = {k: v for k, v in eval_personas.items() if k in roles_filter}

    role_prompts = {name: [prompt] for name, prompt in eval_personas.items()}
    return role_prompts


def load_extraction_questions(
    n_questions: int | None = None,
    questions_file: Path | None = None,
) -> list[str]:
    """Load the shared extraction questions.

    Default source is `data/assistant_axis/extraction_questions.jsonl` (240
    questions used by #205 / #218 / #238). Pass `questions_file=` to read from
    a different in-tree path (e.g. `data/issue_238/extraction_questions.jsonl`)
    when working in a self-contained issue.
    """
    questions_path = questions_file or (DATA_DIR / "extraction_questions.jsonl")
    questions = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    if n_questions:
        questions = questions[:n_questions]
    return questions


# ── Extraction Helpers ───────────────────────────────────────────────────────


def build_chat_text(tokenizer, system_prompt: str, question: str) -> str:
    """Build tokenizer-formatted chat text for a single (system, user) pair."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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

    For each role, averages across n_prompts × len(questions) forward passes.
    Returns dict mapping role_name -> (n_layers, hidden_dim) centroid tensor.
    """
    print(f"\n{'=' * 60}")
    print("Method A: Last-input-token extraction")
    print(f"  Roles: {len(role_prompts)}, Prompts/role: {n_prompts}, Questions: {len(questions)}")
    print(f"  Layers: {layers}")
    print(f"  Total forward passes: {len(role_prompts) * n_prompts * len(questions)}")
    print(f"{'=' * 60}\n")

    # Set up hooks
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

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
        layer_vecs = {l: [] for l in layers}
        prompts_to_use = prompts[:n_prompts]

        for p_idx, sys_prompt in enumerate(prompts_to_use):
            for q_idx, question in enumerate(questions):
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

    for h in hooks:
        h.remove()

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
    Generate responses for all role×prompt×question combos using vLLM.

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
    # NOTE: 0.55 (was 0.85) -- when running A+B together, the HF model from Method A
    # is still loaded in GPU memory (~15 GB on a 7B model) when vLLM init runs, so
    # we must leave headroom. 0.55 * 79 GiB = ~43 GiB, plenty for a 7B + KV cache.
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.55,
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

    # Set up hooks
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

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

        layer_vecs = {l: [] for l in layers}

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
            full_messages = list(prompt_messages) + [{"role": "assistant", "content": response}]
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

    for h in hooks:
        h.remove()

    return centroids


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Extract persona vectors from Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--method",
        default="A",
        choices=["A", "B", "AB"],
        help="Extraction method: A (last-input-token), B (mean-response), AB (both)",
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
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per response (Method B only)",
    )
    parser.add_argument(
        "--inline-personas-json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with schema {eval_personas: {name: prompt}, ...} "
            "that supplies persona system prompts directly, bypassing "
            "data/assistant_axis/{role_list.json, instructions/}. Use when an "
            "issue ships its own persona definitions in-tree (e.g. "
            "data/issue_238/personas.json). Combined with --roles, the role "
            "filter is applied AFTER loading the inline file."
        ),
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help=(
            "Path to a JSONL file of extraction questions "
            "(one {'question': ...} per line). Defaults to "
            "data/assistant_axis/extraction_questions.jsonl. Use to read the "
            "in-tree per-issue copy (e.g. data/issue_238/extraction_questions.jsonl)."
        ),
    )
    args = parser.parse_args()

    # Parse roles filter
    roles_filter = None
    if args.roles:
        roles_filter = [r.strip() for r in args.roles.split(",")]

    # Set up output directory
    model_short_name = args.model.split("/")[-1].lower()
    if args.output_dir:
        base_output = Path(args.output_dir)
    else:
        base_output = Path("data/persona_vectors") / model_short_name

    # Load data
    print("Loading roles and questions...")
    if args.inline_personas_json:
        personas_path = Path(args.inline_personas_json)
        if not personas_path.exists():
            raise FileNotFoundError(f"--inline-personas-json file not found: {personas_path}")
        role_prompts = load_inline_personas(personas_path, roles_filter)
        print(f"  Inline personas loaded from {personas_path}")
    else:
        role_prompts = load_roles(roles_filter)

    questions_file = Path(args.questions_file) if args.questions_file else None
    if questions_file is not None and not questions_file.exists():
        raise FileNotFoundError(f"--questions-file not found: {questions_file}")
    questions = load_extraction_questions(args.n_questions, questions_file=questions_file)
    print(f"  Loaded {len(role_prompts)} roles, {len(questions)} questions")

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
        with open(output_a / "metadata.json", "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "method": "last_input_token",
                    "layers": args.layers,
                    "n_prompts": args.n_prompts,
                    "n_questions": len(questions),
                    "n_roles": len(centroids_a),
                    "roles": sorted(centroids_a.keys()),
                },
                f,
                indent=2,
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
        with open(output_b / "metadata.json", "w") as f:
            json.dump(
                {
                    "model": args.model,
                    "method": "mean_response_token",
                    "layers": args.layers,
                    "n_prompts": args.n_prompts,
                    "n_questions": len(questions),
                    "n_roles": len(centroids_b),
                    "roles": sorted(centroids_b.keys()),
                },
                f,
                indent=2,
            )
        print(f"\nMethod B complete: {len(centroids_b)} role centroids saved to {output_b}")
        all_centroids["method_b"] = centroids_b

        del model
        torch.cuda.empty_cache()

    print(f"\nAll done! Output: {base_output}")


if __name__ == "__main__":
    main()
