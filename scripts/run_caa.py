#!/usr/bin/env python3
"""
Compute CAA (Contrastive Activation Addition; Panickssery 2024) centroids per persona.

CAA centroid for role r at (token i, layer l) =
    mean over questions q of
        ( hidden(pos_prompt=role_r_system, q, layer=l, token=i)
          - hidden(neg_prompt=NO_SYSTEM_MESSAGE, q, layer=l, token=i) )

Per plan §4 + §5 + §11 A21:
- Negative anchor is an EMPTY SYSTEM SLOT (no system message at all in the chat
  template), NOT another role. This matches Chen 2025's negative-instruction shape;
  using "assistant" would conflate "X vs assistant" with "what X is" because
  `assistant` is one of the 275 personas (first key in role_list.json).
- Per v3 fix 1, CAA is DESCRIPTIVE-ONLY in the analysis pipeline: its cells feed
  H1 clustering and per-method baseline reporting but are EXCLUDED from the H2
  Arditi-style argmax candidate set, because the empty-system anchor still encodes
  the helpful-assistant prior (helpful_assistant ↔ no_persona cos = 0.979 per
  research_ideas #6) and a "CAA wins H2" finding would be confounded.

Output:
  data/persona_vectors/issue_263/qwen2.5-7b-instruct/method_caa__pos_<i>__layer_<l>/<role>.pt
  shape (D,) = (3584,) fp32 per (role, position, layer)

Usage:
  # Full sweep on all 275 personas, 240 questions, 5 prompt positions x 28 layers
  python scripts/run_caa.py \\
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \\
      --prompt-token-positions -5,-4,-3,-2,-1 \\
      --output-dir data/persona_vectors/issue_263/qwen2.5-7b-instruct \\
      --gpu-id 0

  # Smoke test (CPU, 2 personas, 4 questions, 2 layers, 2 positions)
  python scripts/run_caa.py --smoke --device cpu \\
      --output-dir /tmp/issue_263_caa_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path(__file__).parent.parent / "data" / "assistant_axis"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYERS_FULL = list(range(28))
DEFAULT_PROMPT_POSITIONS = [-5, -4, -3, -2, -1]


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
    """Load the 240 shared extraction questions (in original order)."""
    questions_path = DATA_DIR / "extraction_questions.jsonl"
    questions = []
    with open(questions_path) as f:
        for line in f:
            item = json.loads(line)
            questions.append(item["question"])
    if n_questions:
        questions = questions[:n_questions]
    return questions


# ── Chat Template Builders ───────────────────────────────────────────────────


def build_chat_text_pos(tokenizer, system_prompt: str, question: str) -> str:
    """Build chat text with persona system prompt — positive (pos) anchor."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_chat_text_neg_empty_system(tokenizer, question: str) -> str:
    """Build chat text with NO system message — empty-system negative anchor.

    Per plan §4 + §5: the empty-system slot, NOT a literal empty string in a
    system role and NOT the 'assistant' persona — just no system message in
    the messages list. Mirrors Chen 2025's negative-instruction shape.
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── CAA Extraction ───────────────────────────────────────────────────────────


def extract_caa(  # noqa: C901
    model,
    tokenizer,
    role_prompts: dict[str, list[str]],
    questions: list[str],
    layers: list[int],
    prompt_positions: list[int],
    output_dir: Path,
    n_prompts: int = 1,
) -> dict[str, dict[tuple[int, int], torch.Tensor]]:
    """CAA: mean over (pos - empty-system) hidden states per (role, position, layer).

    Output structure: centroids[role_name][(layer_idx, position_offset)] = (D,) tensor (fp32 cpu).

    Layout on disk:
        output_dir / "method_caa__pos_<p>__layer_<l>" / "<role>.pt"  -- shape (D,) fp32
    """
    print(f"\n{'=' * 60}")
    print("Method CAA: contrastive (pos - empty_system) extraction")
    print(f"  Roles: {len(role_prompts)}, Prompts/role: {n_prompts}, Questions: {len(questions)}")
    print(f"  Layers: {layers}")
    print(f"  Prompt positions: {prompt_positions}")
    n_fp = len(role_prompts) * n_prompts * len(questions) * 2  # pos + neg
    print(f"  Total forward passes (pos + neg): {n_fp}")
    print(f"{'=' * 60}\n")

    # ── Hooks ──
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # ── Phase 1: Extract negative anchors per question (shared across roles) ──
    # The empty-system + question text is identical across roles, so we can
    # cache neg activations per question once and reuse.
    print("Phase 1: extracting empty-system (neg) activations per question...")
    # neg_acts[q_idx][(layer, pos_offset)] = tensor (D,) fp32 cpu
    neg_acts: list[dict[tuple[int, int], torch.Tensor]] = []
    t0 = time.time()
    for q_idx, question in enumerate(questions):
        text = build_chat_text_neg_empty_system(tokenizer, question)
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
        seq_len = inputs["input_ids"].shape[1]
        per_q_acts: dict[tuple[int, int], torch.Tensor] = {}
        for layer_idx in layers:
            hs = captured[layer_idx]
            for pos in prompt_positions:
                tok_pos = seq_len + pos
                if tok_pos < 0:
                    continue
                per_q_acts[(layer_idx, pos)] = hs[0, tok_pos, :].float().cpu()
        neg_acts.append(per_q_acts)
        if (q_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  neg q={q_idx + 1}/{len(questions)} ({elapsed:.0f}s)", flush=True)

    # ── Phase 2: per-role pos extraction + subtract neg ──
    print("\nPhase 2: extracting per-role (pos) activations and computing CAA centroids...")

    # Pre-build output dirs
    for layer_idx in layers:
        for pos in prompt_positions:
            (output_dir / f"method_caa__pos_{pos}__layer_{layer_idx}").mkdir(
                parents=True, exist_ok=True
            )

    centroids: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
    t0 = time.time()
    sorted_roles = sorted(role_prompts.items())
    for role_idx, (role_name, prompts) in enumerate(sorted_roles):
        # Resume support: if all (pos, layer) cells already on disk, load them.
        all_present = all(
            (output_dir / f"method_caa__pos_{p}__layer_{lyr}" / f"{role_name}.pt").exists()
            for p in prompt_positions
            for lyr in layers
        )
        if all_present:
            cached: dict[tuple[int, int], torch.Tensor] = {}
            for p in prompt_positions:
                for lyr in layers:
                    cached[(lyr, p)] = torch.load(
                        output_dir / f"method_caa__pos_{p}__layer_{lyr}" / f"{role_name}.pt",
                        weights_only=True,
                    )
            centroids[role_name] = cached
            print(
                f"  [{role_idx + 1}/{len(sorted_roles)}] {role_name} — loaded from cache",
                flush=True,
            )
            continue

        prompts_to_use = prompts[:n_prompts]
        # Accumulators: layer_pos -> list of per-question (pos - neg) vecs
        accum: dict[tuple[int, int], list[torch.Tensor]] = {
            (lyr, p): [] for lyr in layers for p in prompt_positions
        }

        for sys_prompt in prompts_to_use:
            for q_idx, question in enumerate(questions):
                text = build_chat_text_pos(tokenizer, sys_prompt, question)
                inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
                with torch.no_grad():
                    _ = model(**inputs)
                seq_len = inputs["input_ids"].shape[1]
                for layer_idx in layers:
                    hs = captured[layer_idx]
                    for pos in prompt_positions:
                        tok_pos = seq_len + pos
                        if tok_pos < 0:
                            continue
                        pos_vec = hs[0, tok_pos, :].float().cpu()
                        neg_vec = neg_acts[q_idx].get((layer_idx, pos))
                        if neg_vec is None:
                            continue
                        accum[(layer_idx, pos)].append(pos_vec - neg_vec)

        # Centroid = mean of (pos - neg) across all (prompt, question) pairs
        per_role: dict[tuple[int, int], torch.Tensor] = {}
        for (lyr, pos), vecs in accum.items():
            if not vecs:
                continue
            stacked = torch.stack(vecs)  # (N, D)
            centroid = stacked.mean(dim=0)
            per_role[(lyr, pos)] = centroid
            # Save per (layer, pos) cell
            torch.save(
                centroid,
                output_dir / f"method_caa__pos_{pos}__layer_{lyr}" / f"{role_name}.pt",
            )
        centroids[role_name] = per_role

        elapsed = time.time() - t0
        rate = (role_idx + 1) / elapsed * 60 if elapsed > 0 else 0.0
        print(
            f"  [{role_idx + 1}/{len(sorted_roles)}] {role_name} — "
            f"{elapsed:.0f}s elapsed, {rate:.1f} roles/min",
            flush=True,
        )

    for h in hooks:
        h.remove()

    # ── Metadata ──
    metadata = {
        "method": "caa",
        "negative_anchor": "empty_system_prompt",
        "model": getattr(model.config, "_name_or_path", "unknown"),
        "n_roles": len(centroids),
        "n_questions": len(questions),
        "n_prompts": n_prompts,
        "layers": layers,
        "prompt_positions": prompt_positions,
        "roles": sorted(centroids.keys()),
    }
    with open(output_dir / "method_caa_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return centroids


# ── Main ─────────────────────────────────────────────────────────────────────


def parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers, e.g. '0,1,2' or '-5,-4'."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Compute CAA centroids (pos - empty-system) per (role, position, layer)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index (used iff --device cuda)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device (cpu = smoke / dev only)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LAYERS_FULL),
        help="Comma-separated layer indices (default: all 28)",
    )
    parser.add_argument(
        "--prompt-token-positions",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PROMPT_POSITIONS),
        help="Comma-separated prompt-side token offsets, e.g. '-5,-4,-3,-2,-1'",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=None,
        help="Comma-separated roles to extract (default: all 275)",
    )
    parser.add_argument("--n-prompts", type=int, default=1, help="System prompts per role")
    parser.add_argument("--n-questions", type=int, default=None, help="Questions (default: 240)")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Base output dir (centroids written under it)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: 2 personas, 4 questions, 2 positions, 2 layers (overrides above)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.smoke:
        layers = [0, 7]
        prompt_positions = [-1, -2]
        roles_filter = None
        n_questions = 4
        # Pick first 2 roles deterministically from sorted role_list
        with open(DATA_DIR / "role_list.json") as f:
            all_roles = sorted(json.load(f).keys())
        roles_filter = all_roles[:2]
        print(f"SMOKE MODE: roles={roles_filter}, n_q=4, layers={layers}, pos={prompt_positions}")
    else:
        layers = parse_int_list(args.layers)
        prompt_positions = parse_int_list(args.prompt_token_positions)
        roles_filter = (
            [r.strip() for r in args.roles.split(",")] if args.roles is not None else None
        )
        n_questions = args.n_questions

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading roles + questions...")
    role_prompts = load_roles(roles_filter)
    questions = load_extraction_questions(n_questions)
    print(f"  {len(role_prompts)} roles, {len(questions)} questions")

    print(f"\nLoading model {args.model} on device {args.device}...")
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": device}
        )
    else:
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, device_map={"": device}
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    extract_caa(
        model,
        tokenizer,
        role_prompts,
        questions,
        layers,
        prompt_positions,
        output_dir,
        n_prompts=args.n_prompts,
    )

    print(f"\nCAA done. Centroids under: {output_dir}/method_caa__pos_*__layer_*/")


if __name__ == "__main__":
    main()
