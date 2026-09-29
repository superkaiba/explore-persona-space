#!/usr/bin/env python3
"""
Extract persona vectors for all 275 assistant-axis roles from Qwen2.5-7B-Instruct.

Two extraction methods:
  A) Last-input-token: hidden state at the last token of the prompt (before generation).
  B) Mean-response-token: mean hidden state across all generated response tokens.

Method A requires only forward passes (fast).
Method B requires generation + forward passes (slower; uses vLLM for generation, HF for extraction).

Output: data/persona_vectors/{model_name}/{method}/
  - {role_name}.pt   -- per-role centroid (avg across prompts x questions) shape (n_layers, dim)
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
    save_perquestion: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Method A: Extract last-input-token hidden states.

    For each role, averages across n_prompts x len(questions) forward passes.
    Returns dict mapping role_name -> (n_layers, hidden_dim) centroid tensor.

    If save_perquestion=True, also saves per-question activation stacks:
      {role}_perquestion_L{layer}.pt — shape (N_kept, hidden_dim)
      {role}_question_indices.pt — indices of questions contributing to the centroid
    """
    print(f"\n{'=' * 60}")
    print("Method A: Last-input-token extraction")
    print(f"  Roles: {len(role_prompts)}, Prompts/role: {n_prompts}, Questions: {len(questions)}")
    print(f"  Layers: {layers}")
    print(f"  Total forward passes: {len(role_prompts) * n_prompts * len(questions)}")
    print(f"  Save per-question: {save_perquestion}")
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
        layer_vecs = {lay: [] for lay in layers}
        question_indices = []  # track which questions contributed
        prompts_to_use = prompts[:n_prompts]

        for _p_idx, sys_prompt in enumerate(prompts_to_use):
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

                question_indices.append(q_idx)

        # Compute centroid: (n_layers, hidden_dim)
        layer_centroids = []
        for layer_idx in layers:
            stacked = torch.stack(layer_vecs[layer_idx])  # (N, hidden_dim)
            layer_centroids.append(stacked.mean(dim=0))
        centroid = torch.stack(layer_centroids)  # (n_layers, hidden_dim)
        centroids[role_name] = centroid

        # Save per-role centroid
        if output_dir:
            torch.save(centroid, output_dir / f"{role_name}.pt")

        # Save per-question activation stacks for LDA + paired filter
        if save_perquestion and output_dir:
            for layer_idx in layers:
                stacked = torch.stack(layer_vecs[layer_idx])  # (N, hidden_dim)
                torch.save(stacked, output_dir / f"{role_name}_perquestion_L{layer_idx}.pt")
            torch.save(
                torch.tensor(question_indices, dtype=torch.long),
                output_dir / f"{role_name}_question_indices.pt",
            )

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
        gpu_memory_utilization=0.70,  # lowered from 0.85: after Method A cleanup, ~15 GiB may be leaked
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


def extract_method_b(  # noqa: C901
    model,
    tokenizer,
    responses: dict[str, list[dict]],
    layers: list[int],
    output_dir: Path | None = None,
    save_perquestion: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Method B: Extract mean-response-token hidden states.

    For each role, takes pre-generated responses, runs forward pass on full
    (system + user + response) sequence, and averages hidden states across
    response tokens only.

    Returns dict mapping role_name -> (n_layers, hidden_dim) centroid tensor.

    If save_perquestion=True, also saves per-question activation stacks:
      {role}_perquestion_L{layer}.pt — shape (N_kept, hidden_dim)
      {role}_question_indices.pt — indices of questions with non-empty responses
    """
    print(f"\n{'=' * 60}")
    print("Method B Phase 2: Mean-response-token extraction")
    print(f"  Roles: {len(responses)}")
    n_total = sum(len(v) for v in responses.values())
    print(f"  Total forward passes: {n_total}")
    print(f"  Layers: {layers}")
    print(f"  Save per-question: {save_perquestion}")
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

        layer_vecs = {lay: [] for lay in layers}
        question_indices = []  # track which items had non-empty responses

        for item_idx, item in enumerate(items):
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

            question_indices.append(item_idx)

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

        # Save per-question activation stacks for LDA + paired filter
        if save_perquestion and output_dir:
            for layer_idx in layers:
                stacked = torch.stack(layer_vecs[layer_idx])  # (N_kept, hidden_dim)
                torch.save(stacked, output_dir / f"{role_name}_perquestion_L{layer_idx}.pt")
            torch.save(
                torch.tensor(question_indices, dtype=torch.long),
                output_dir / f"{role_name}_question_indices.pt",
            )

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
    # ── Issue #205 extensions ───────────────────────────────────────────────
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="PEFT adapter to merge on top of --model before extraction (HF repo or local dir)",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default=None,
        help="Output subdir under data/persona_vectors/<model>/. Overrides default 'method_a/b'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (only matters for tie-breaking; vLLM is greedy)",
    )
    parser.add_argument(
        "--save-perquestion",
        action="store_true",
        default=False,
        help="Save per-question activation stacks for LDA + paired empty-response filter",
    )
    args = parser.parse_args()

    # Parse roles filter
    roles_filter = None
    if args.roles:
        roles_filter = [r.strip() for r in args.roles.split(",")]

    # Set up output directory — checkpoint-tag overrides default path structure
    model_short_name = args.model.split("/")[-1].lower()
    if args.output_dir:
        base_output = Path(args.output_dir)
    elif args.checkpoint_tag:
        base_output = Path("data/persona_vectors") / model_short_name / args.checkpoint_tag
    else:
        base_output = Path("data/persona_vectors") / model_short_name

    # Load data
    print("Loading roles and questions...")
    role_prompts = load_roles(roles_filter)
    questions = load_extraction_questions(args.n_questions)
    print(f"  Loaded {len(role_prompts)} roles, {len(questions)} questions")

    do_a = "A" in args.method
    do_b = "B" in args.method

    all_centroids = {}

    # ── Resolve model path (apply adapter if requested) ──
    # For Method A + HF-based Method B Phase 2, we merge the adapter in-memory.
    # For Method B Phase 1 (vLLM), we need a merged-on-disk directory.
    vllm_model_path = args.model  # path for vLLM loading

    if args.adapter:
        print(f"\n  Adapter requested: {args.adapter}")
        print("  Will merge adapter into base model for extraction.")
        # For vLLM Phase 1: merge + save to a temp dir so vLLM can load it
        if do_b:
            merged_dir = (
                Path("/workspace/persona_vectors_tmp")
                / f"{model_short_name}_{args.checkpoint_tag or 'merged'}"
            )
            if not (merged_dir / "config.json").exists():
                print(f"  Merging adapter to disk at {merged_dir} for vLLM...")
                import gc

                from peft import PeftModel

                _base = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                    device_map="cpu",
                )
                _peft = PeftModel.from_pretrained(_base, args.adapter)
                _merged = _peft.merge_and_unload()
                merged_dir.mkdir(parents=True, exist_ok=True)
                _merged.save_pretrained(str(merged_dir), safe_serialization=True)
                _tok = AutoTokenizer.from_pretrained(args.model)
                _tok.save_pretrained(str(merged_dir))
                del _base, _peft, _merged, _tok
                gc.collect()
                print(f"  Merged adapter saved to {merged_dir}")
            else:
                print(f"  Using cached merged model at {merged_dir}")
            vllm_model_path = str(merged_dir)

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

        # Apply adapter if specified (merge in-memory for hook-based extraction)
        if args.adapter:
            from peft import PeftModel

            print(f"  Applying adapter: {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)
            model = model.merge_and_unload()

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
            save_perquestion=args.save_perquestion,
        )

        # Save combined file
        torch.save(centroids_a, output_a / "all_centroids.pt")
        metadata = {
            "model": args.model,
            "method": "last_input_token",
            "layers": args.layers,
            "n_prompts": args.n_prompts,
            "n_questions": len(questions),
            "n_roles": len(centroids_a),
            "roles": sorted(centroids_a.keys()),
            "seed": args.seed,
            "save_perquestion": args.save_perquestion,
        }
        if args.adapter:
            metadata["adapter"] = args.adapter
        if args.checkpoint_tag:
            metadata["checkpoint_tag"] = args.checkpoint_tag
        with open(output_a / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMethod A complete: {len(centroids_a)} role centroids saved to {output_a}")
        all_centroids["method_a"] = centroids_a

        # Always free the HF model to reclaim GPU memory before vLLM starts.
        del model
        import gc as _gc

        _gc.collect()
        torch.cuda.empty_cache()

    # ── Method B ──
    if do_b:
        output_b = base_output / "method_b"
        output_b.mkdir(parents=True, exist_ok=True)
        responses_path = output_b / "generated_responses.json"

        # Phase 1: Generate responses with vLLM
        # Use vllm_model_path which is already merged-on-disk if --adapter was given
        responses = generate_responses_vllm(
            vllm_model_path,
            role_prompts,
            questions,
            n_prompts=args.n_prompts,
            gpu_id=args.gpu_id,
            output_path=responses_path,
            max_new_tokens=args.max_new_tokens,
        )

        # Phase 2: Extract with HF model (always reload — vLLM freed it)
        print(f"\nLoading model on GPU {args.gpu_id} for Method B Phase 2...")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        # Apply adapter if specified
        if args.adapter:
            from peft import PeftModel

            print(f"  Applying adapter: {args.adapter}")
            model = PeftModel.from_pretrained(model, args.adapter)
            model = model.merge_and_unload()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        centroids_b = extract_method_b(
            model,
            tokenizer,
            responses,
            args.layers,
            output_dir=output_b,
            save_perquestion=args.save_perquestion,
        )

        metadata_b = {
            "model": args.model,
            "method": "mean_response_token",
            "layers": args.layers,
            "n_prompts": args.n_prompts,
            "n_questions": len(questions),
            "n_roles": len(centroids_b),
            "roles": sorted(centroids_b.keys()),
            "seed": args.seed,
            "save_perquestion": args.save_perquestion,
        }
        if args.adapter:
            metadata_b["adapter"] = args.adapter
        if args.checkpoint_tag:
            metadata_b["checkpoint_tag"] = args.checkpoint_tag
        torch.save(centroids_b, output_b / "all_centroids.pt")
        with open(output_b / "metadata.json", "w") as f:
            json.dump(metadata_b, f, indent=2)
        print(f"\nMethod B complete: {len(centroids_b)} role centroids saved to {output_b}")
        all_centroids["method_b"] = centroids_b

        del model
        torch.cuda.empty_cache()

    print(f"\nAll done! Output: {base_output}")


if __name__ == "__main__":
    main()
